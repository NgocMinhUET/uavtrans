"""Classical crack segmenter for fair end-to-end evaluation.

Used as the fixed "cloud detector" when evaluating JPEG/ROI-JPEG baselines.
A classical (non-learned) detector is used to avoid train/test leakage and
ensure reproducibility without a pre-trained neural model.

Pipeline:
    1. Convert to grayscale
    2. CLAHE contrast enhancement (crack regions are often low contrast)
    3. Adaptive Gaussian thresholding
    4. Morphological cleanup
    5. Small-component pruning
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class SegmenterConfig:
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    adapt_block: int = 31       # must be odd
    adapt_c: float = 4.0        # subtracted constant
    morph_close_k: int = 3      # kernel for closing gaps
    morph_open_k: int = 2       # kernel for removing speckle
    min_component_px: int = 30  # prune tiny blobs
    use_roi_prior: bool = True  # restrict to GT bounding box if available


def _apply_clahe(gray: np.ndarray, cfg: SegmenterConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                             tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    return clahe.apply(gray)


def _prune_small_components(binary: np.ndarray, min_px: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[labels == i] = 1
    return out


def segment_image(
    image: np.ndarray,
    cfg: Optional[SegmenterConfig] = None,
    roi_bbox_xywh: Optional[tuple] = None
) -> np.ndarray:
    """
    Run classical crack segmenter on image.

    Args:
        image:          BGR or grayscale image (uint8).
        cfg:            SegmenterConfig (uses defaults if None).
        roi_bbox_xywh:  Optional (x,y,w,h) to restrict search region.
                        Pixels outside are forced to background.

    Returns:
        Binary mask (0/1, uint8, same spatial size as image).
    """
    if cfg is None:
        cfg = SegmenterConfig()

    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY) \
           if image.ndim == 3 else image.astype(np.uint8)

    enhanced = _apply_clahe(gray, cfg)

    thresh = cv2.adaptiveThreshold(
        enhanced, 1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adapt_block,
        cfg.adapt_c
    )

    if cfg.morph_close_k > 0:
        k_close = np.ones((cfg.morph_close_k, cfg.morph_close_k), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)

    if cfg.morph_open_k > 0:
        k_open = np.ones((cfg.morph_open_k, cfg.morph_open_k), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open)

    mask = _prune_small_components(thresh, cfg.min_component_px)

    if roi_bbox_xywh is not None and cfg.use_roi_prior:
        x, y, w, h = [int(v) for v in roi_bbox_xywh]
        roi_gate = np.zeros_like(mask)
        pad = 16
        H, W = mask.shape
        roi_gate[max(0, y-pad):min(H, y+h+pad),
                 max(0, x-pad):min(W, x+w+pad)] = 1
        mask = mask & roi_gate

    return mask.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# JPEG encode → decode → segment pipeline
# ─────────────────────────────────────────────────────────────────────────────

def jpeg_encode_decode(image: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through JPEG at given quality. Returns BGR uint8."""
    _, buf = cv2.imencode(".jpg", image.astype(np.uint8),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded


def eval_jpeg_segmentation(
    image: np.ndarray,
    gt_mask: np.ndarray,
    quality: int,
    seg_cfg: Optional[SegmenterConfig] = None,
    use_roi_prior: bool = True
) -> dict:
    """
    Measure end-to-end task accuracy when image is sent as full-image JPEG.

    Pipeline: image → JPEG(Q) → decode → segment → compare to GT mask.

    Args:
        image:          Original BGR image.
        gt_mask:        Ground-truth binary mask (0/1 or 0/255).
        quality:        JPEG quality [1–95].
        seg_cfg:        Segmenter config.
        use_roi_prior:  Restrict segmenter to GT ROI (models knowledge of
                        approximate crack location from header metadata).

    Returns:
        Dict with payload_bytes, dice, iou, precision, recall.
    """
    if seg_cfg is None:
        seg_cfg = SegmenterConfig(use_roi_prior=use_roi_prior)

    _, buf = cv2.imencode(".jpg", image.astype(np.uint8),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    payload_bytes = len(buf)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    gt = (gt_mask > 0).astype(np.uint8)
    roi_bbox = None
    if use_roi_prior:
        ys, xs = np.where(gt > 0)
        if len(xs):
            roi_bbox = (int(xs.min()), int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1))

    pred = segment_image(decoded, seg_cfg, roi_bbox_xywh=roi_bbox)
    metrics = _compute_metrics(pred, gt)
    metrics["payload_bytes"] = payload_bytes
    return metrics


def eval_roi_jpeg_segmentation(
    image: np.ndarray,
    gt_mask: np.ndarray,
    quality: int,
    seg_cfg: Optional[SegmenterConfig] = None,
    pad: int = 8
) -> dict:
    """
    Measure end-to-end task accuracy when only the ROI is sent as JPEG.

    Pipeline: crop ROI → JPEG(Q) → decode → segment → project back → compare.

    ROI is cropped using the GT bounding box (models receiver knowing
    approximate crack location from prior alert metadata).
    """
    if seg_cfg is None:
        seg_cfg = SegmenterConfig(use_roi_prior=False)

    gt = (gt_mask > 0).astype(np.uint8)
    H, W = image.shape[:2]

    ys, xs = np.where(gt > 0)
    if not len(xs):
        return {"payload_bytes": 0, "dice": 0.0, "iou": 0.0,
                "precision": 0.0, "recall": 0.0}

    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)
    y1 = min(H, int(ys.max()) + pad + 1)
    roi_img = image[y0:y1, x0:x1]

    _, buf = cv2.imencode(".jpg", roi_img.astype(np.uint8),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    payload_bytes = len(buf)
    decoded_roi = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    roi_pred = segment_image(decoded_roi, seg_cfg, roi_bbox_xywh=None)

    full_pred = np.zeros((H, W), dtype=np.uint8)
    full_pred[y0:y1, x0:x1] = roi_pred[:y1-y0, :x1-x0]

    metrics = _compute_metrics(full_pred, gt)
    metrics["payload_bytes"] = payload_bytes
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = (pred > 0).astype(np.float64)
    g = (gt > 0).astype(np.float64)

    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    tn = ((1 - p) * (1 - g)).sum()

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice       = 2 * tp / (2 * tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    union      = tp + fp + fn
    iou        = tp / union if union > 0 else 1.0
    spec       = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc    = (recall + spec) / 2.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_accuracy": float(bal_acc),
    }
