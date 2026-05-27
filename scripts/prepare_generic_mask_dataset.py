#!/usr/bin/env python
"""Prepare a generic image/mask dataset for ViDEC-UW.

Expected input:
- images in --src-images
- binary masks in --src-masks with the same basename or the same filename

The script copies image/mask pairs into a clean layout:
    out/images/*.png
    out/masks/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-images", type=Path, required=True)
    parser.add_argument("--src-masks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out_img = args.out / "images"
    out_msk = args.out / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(args.src_images.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        mask_path = find_mask(args.src_masks, img_path.stem)
        if mask_path is None:
            print(f"[WARN] no mask for {img_path.name}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"[WARN] unreadable pair {img_path.name}")
            continue
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out_name = f"event_{count:05d}.png"
        cv2.imwrite(str(out_img / out_name), img)
        cv2.imwrite(str(out_msk / out_name), (mask > 127).astype("uint8") * 255)
        count += 1

    print(f"Prepared {count} image/mask pairs at {args.out}")


if __name__ == "__main__":
    main()
