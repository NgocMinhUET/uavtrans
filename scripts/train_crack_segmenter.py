#!/usr/bin/env python
"""Train a lightweight U-Net crack segmenter on Crack500 train split.

This model is used as the "cloud detector" when evaluating JPEG/ROI-JPEG
baselines in the UDST experiment — replacing the classical segmenter that
had a hard accuracy ceiling of ~0.34 Dice.

Expected training time:
    GPU (RTX / Jetson AGX):  ~30-60 min for 50 epochs on Crack500 train
    CPU only:                 not recommended (hours)

Usage:
    python scripts/train_crack_segmenter.py \\
        --image-dir /path/to/Crack500_raw/train/images \\
        --mask-dir  /path/to/Crack500_raw/train/masks  \\
        --val-image-dir /path/to/Crack500_raw/val/images \\
        --val-mask-dir  /path/to/Crack500_raw/val/masks  \\
        --out checkpoints/ \\
        --epochs 50 \\
        --batch-size 8 \\
        --encoder mobilenet_v2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

from videc_uw.neural_segmenter import (
    build_model, preprocess, postprocess,
    INPUT_SIZE, _metrics
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CrackDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        augment: bool = True,
        input_size: tuple[int, int] = INPUT_SIZE,
    ):
        self.input_size = input_size
        self.augment    = augment
        exts = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in exts
        )
        self.mask_dir = mask_dir
        # filter to pairs that exist
        valid = []
        for ip in self.image_paths:
            mp = mask_dir / ip.name
            if not mp.exists():
                mp = mask_dir / (ip.stem + ".png")
            if mp.exists():
                valid.append((ip, mp))
        self.pairs = valid
        print(f"  Dataset: {len(self.pairs)} image-mask pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        img  = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            # return zeros on load failure
            img  = np.zeros((*self.input_size, 3), dtype=np.uint8)
            mask = np.zeros(self.input_size, dtype=np.uint8)

        # Resize
        img  = cv2.resize(img,  self.input_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalise image
        from videc_uw.neural_segmenter import IMG_MEAN, IMG_STD
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMG_MEAN) / IMG_STD
        img_t  = torch.from_numpy(rgb.transpose(2, 0, 1))      # (3,H,W)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)   # (1,H,W)
        return img_t, mask_t

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        # Horizontal flip
        if np.random.rand() > 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)
        # Vertical flip
        if np.random.rand() > 0.5:
            img  = cv2.flip(img,  0)
            mask = cv2.flip(mask, 0)
        # Random brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.7, 1.3)
            beta  = np.random.randint(-20, 20)
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# Loss: BCE + Dice combined
# ─────────────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_w  = bce_weight
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)
        prob      = torch.sigmoid(logits)
        inter     = (prob * targets).sum(dim=(2, 3))
        union     = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return self.bce_w * bce_loss + (1 - self.bce_w) * dice_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, optimizer, device, train: bool
) -> dict:
    model.train() if train else model.eval()
    total_loss = 0.0
    dice_sum   = 0.0
    n          = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss   = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # compute dice for monitoring
            preds = (torch.sigmoid(logits) >= 0.5).float()
            inter = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()
            dice_sum += (2 * inter + 1) / (union + 1)
            n += 1

    return {"loss": total_loss / max(n, 1), "dice": dice_sum / max(n, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation on val set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, image_dir, mask_dir, device, n_max=200) -> dict:
    """Compute mean Dice/IoU on val images (at original resolution)."""
    from videc_uw.evidence import load_image_mask
    model.eval()
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)[:n_max]

    dice_list, iou_list = [], []
    for ip in paths:
        mp = mask_dir / ip.name
        if not mp.exists():
            mp = mask_dir / (ip.stem + ".png")
            if not mp.exists():
                continue
        image, gt = load_image_mask(ip, mp)
        if gt.sum() == 0:
            continue
        x = preprocess(image).to(device)
        logits = model(x)
        pred = postprocess(logits, image.shape[:2])
        m = _metrics(pred, gt)
        dice_list.append(m["dice"])
        iou_list.append(m["iou"])

    return {
        "dice_mean": float(np.mean(dice_list)) if dice_list else 0.0,
        "iou_mean":  float(np.mean(iou_list))  if iou_list  else 0.0,
        "n": len(dice_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir",      type=Path, required=True)
    parser.add_argument("--mask-dir",       type=Path, required=True)
    parser.add_argument("--val-image-dir",  type=Path, default=None)
    parser.add_argument("--val-mask-dir",   type=Path, default=None)
    parser.add_argument("--out",            type=Path, default=Path("checkpoints"))
    parser.add_argument("--encoder",        type=str,  default="mobilenet_v2",
                        help="mobilenet_v2 | resnet18 | resnet34 | efficientnet-b0")
    parser.add_argument("--epochs",         type=int,  default=50)
    parser.add_argument("--batch-size",     type=int,  default=8)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--num-workers",    type=int,  default=4)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--patience",       type=int,  default=10,
                        help="Early stopping patience (epochs without val improvement)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")
    print(f"[TRAIN] Encoder: {args.encoder}")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("[TRAIN] Loading train dataset...")
    train_ds = CrackDataset(args.image_dir, args.mask_dir, augment=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          pin_memory=(device.type == "cuda"), drop_last=True)

    has_val = args.val_image_dir is not None and args.val_mask_dir is not None

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(encoder=args.encoder, encoder_weights="imagenet")
    model = model.to(device)

    criterion = BCEDiceLoss(bce_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                      eta_min=1e-6)

    # ── training loop ─────────────────────────────────────────────────────────
    history = []
    best_dice = 0.0
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
        scheduler.step()

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}

        if has_val and (epoch % 5 == 0 or epoch == args.epochs):
            val_stats = evaluate(model, args.val_image_dir, args.val_mask_dir, device)
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            val_dice = val_stats["dice_mean"]
            marker = ""

            if val_dice > best_dice:
                best_dice  = val_dice
                best_epoch = epoch
                no_improve = 0
                ckpt = {"epoch": epoch, "encoder": args.encoder,
                        "model_state_dict": model.state_dict(),
                        "val_dice": val_dice}
                torch.save(ckpt, args.out / "best_crack_segmenter.pt")
                marker = "  ← best"
            else:
                no_improve += 5

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_stats['loss']:.4f}  "
                  f"train_dice={train_stats['dice']:.4f}  "
                  f"val_dice={val_dice:.4f}  "
                  f"{elapsed:.0f}s{marker}")

            if no_improve >= args.patience and epoch > 20:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break
        else:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_stats['loss']:.4f}  "
                  f"train_dice={train_stats['dice']:.4f}  "
                  f"{elapsed:.0f}s")

        history.append(row)

    # ── save final checkpoint (even without val) ───────────────────────────────
    final_ckpt = {"epoch": args.epochs, "encoder": args.encoder,
                  "model_state_dict": model.state_dict()}
    torch.save(final_ckpt, args.out / "final_crack_segmenter.pt")

    best_path = args.out / "best_crack_segmenter.pt"
    if not best_path.exists():
        torch.save(final_ckpt, best_path)

    # ── save training log ─────────────────────────────────────────────────────
    import pandas as pd
    pd.DataFrame(history).to_csv(args.out / "training_log.csv", index=False)

    total = time.time() - t0
    print(f"\n[TRAIN] Done in {total:.0f}s")
    print(f"[TRAIN] Best val Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"[TRAIN] Checkpoint saved to: {best_path}")

    # ── save config for reproducibility ───────────────────────────────────────
    cfg = {
        "encoder": args.encoder,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "input_size": INPUT_SIZE,
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
    }
    (args.out / "segmenter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[TRAIN] Config saved to: {args.out / 'segmenter_config.json'}")


if __name__ == "__main__":
    main()
