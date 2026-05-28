"""Neural crack segmenter — lightweight U-Net for fair JPEG baseline evaluation.

Architecture: U-Net with MobileNetV2 encoder (pretrained on ImageNet).
Trained on Crack500 train split. Used as the fixed "cloud detector" when
evaluating JPEG / ROI-JPEG baselines, replacing the classical segmenter.

Requires: segmentation_models_pytorch (pip install segmentation-models-pytorch)
          torch, torchvision
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports to avoid hard dependency at module load time
# ─────────────────────────────────────────────────────────────────────────────

def _get_smp():
    try:
        import segmentation_models_pytorch as smp
        return smp
    except ImportError:
        raise ImportError(
            "segmentation_models_pytorch is required.\n"
            "Install with:  pip install segmentation-models-pytorch"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    encoder: str = "mobilenet_v2",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
) -> torch.nn.Module:
    """
    Build a lightweight U-Net.

    MobileNetV2 encoder: ~4 M params, fast on CPU/Jetson.
    Can swap to resnet18/resnet34 for higher accuracy.
    """
    smp = _get_smp()
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,            # raw logits — sigmoid applied at inference
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (must match training)
# ─────────────────────────────────────────────────────────────────────────────

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = (448, 448)           # divisible by 32 for U-Net


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → (1, 3, H, W) float32 tensor, ImageNet normalised."""
    rgb = cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMG_MEAN) / IMG_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)   # (1,3,H,W)
    return x


def postprocess(logits: torch.Tensor, orig_hw: tuple[int, int],
                threshold: float = 0.5) -> np.ndarray:
    """(1,1,H,W) logit tensor → binary mask at original resolution."""
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()          # (H, W)
    h, w = orig_hw
    if prob.shape != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob >= threshold).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSegmenter:
    """Lightweight wrapper around a trained U-Net for crack segmentation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        encoder: str = "mobilenet_v2",
        device: str = "auto",
        threshold: float = 0.5,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = torch.device(device)
        self.threshold = threshold

        self.model = build_model(encoder=encoder, encoder_weights=None)
        ckpt = torch.load(str(checkpoint_path), map_location=self.device,
                          weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: BGR uint8 image of any size.
        Returns:
            Binary mask (0/1, uint8, same H×W as input).
        """
        orig_hw = image_bgr.shape[:2]
        x       = preprocess(image_bgr).to(self.device)
        logits  = self.model(x)
        return postprocess(logits, orig_hw, self.threshold)

    def eval_on_jpeg(
        self,
        image_bgr: np.ndarray,
        gt_mask:   np.ndarray,
        quality:   int,
        roi_only:  bool = False,
        pad:       int = 8,
    ) -> dict:
        """
        Full end-to-end evaluation: encode → decode → predict → compare to GT.

        Args:
            image_bgr: Original BGR image.
            gt_mask:   Ground-truth binary mask (0/1 or 0/255).
            quality:   JPEG quality [1–95].
            roi_only:  If True, encode only the GT-ROI crop (ROI-JPEG mode).
            pad:       Padding around ROI bounding box.

        Returns:
            dict with payload_bytes, dice, iou, precision, recall,
            balanced_accuracy.
        """
        gt = (gt_mask > 0).astype(np.uint8)
        H, W = image_bgr.shape[:2]

        if roi_only:
            ys, xs = np.where(gt > 0)
            if not len(xs):
                return {"payload_bytes": 0, "dice": 0.0, "iou": 0.0,
                        "precision": 0.0, "recall": 0.0,
                        "balanced_accuracy": 0.5}
            x0 = max(0,  int(xs.min()) - pad)
            y0 = max(0,  int(ys.min()) - pad)
            x1 = min(W,  int(xs.max()) + pad + 1)
            y1 = min(H,  int(ys.max()) + pad + 1)
            encode_img = image_bgr[y0:y1, x0:x1]
        else:
            encode_img = image_bgr
            x0 = y0 = 0
            x1, y1 = W, H

        _, buf = cv2.imencode(".jpg", encode_img.astype(np.uint8),
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
        payload_bytes = int(len(buf))
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        roi_pred = self.predict(decoded)

        full_pred = np.zeros((H, W), dtype=np.uint8)
        full_pred[y0:y1, x0:x1] = roi_pred[:y1-y0, :x1-x0]

        return {**_metrics(full_pred, gt), "payload_bytes": payload_bytes}


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric helper
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = (pred > 0).astype(np.float64)
    g = (gt   > 0).astype(np.float64)
    tp = (p * g).sum();  fp = (p * (1-g)).sum()
    fn = ((1-p)*g).sum(); tn = ((1-p)*(1-g)).sum()
    prec  = tp / (tp+fp)          if (tp+fp) > 0 else 0.0
    rec   = tp / (tp+fn)          if (tp+fn) > 0 else 0.0
    dice  = 2*tp / (2*tp+fp+fn)   if (2*tp+fp+fn) > 0 else 0.0
    iou   = tp / (tp+fp+fn)       if (tp+fp+fn) > 0 else 1.0
    spec  = tn / (tn+fp)          if (tn+fp) > 0 else 0.0
    return {"dice": float(dice), "iou": float(iou),
            "precision": float(prec), "recall": float(rec),
            "balanced_accuracy": float((rec+spec)/2)}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load or return None
# ─────────────────────────────────────────────────────────────────────────────

def load_segmenter_if_available(
    checkpoint_path: str | Path,
    encoder: str = "mobilenet_v2",
    device: str = "auto",
) -> Optional["NeuralSegmenter"]:
    """Return NeuralSegmenter if checkpoint exists, else None (falls back to classical)."""
    p = Path(checkpoint_path)
    if not p.exists():
        return None
    try:
        return NeuralSegmenter(p, encoder=encoder, device=device)
    except Exception as e:
        print(f"[WARN] Could not load neural segmenter: {e}")
        return None
