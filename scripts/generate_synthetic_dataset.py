#!/usr/bin/env python
"""Generate a small synthetic crack dataset for end-to-end reproducibility.

This is not intended to replace real underwater inspection data. It allows the
pipeline, channel simulator, and plots to run immediately after cloning.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
import numpy as np


def draw_crack(mask: np.ndarray, rng: random.Random) -> None:
    h, w = mask.shape
    x = rng.randint(w // 8, w // 3)
    y = rng.randint(h // 4, 3 * h // 4)
    points = [(x, y)]
    for _ in range(rng.randint(4, 9)):
        x = int(np.clip(x + rng.randint(20, 70), 0, w - 1))
        y = int(np.clip(y + rng.randint(-40, 40), 0, h - 1))
        points.append((x, y))
    thickness = rng.randint(2, 6)
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(mask, p0, p1, 255, thickness=thickness)
    if rng.random() < 0.45:
        bx, by = points[rng.randint(1, len(points) - 2)]
        ex = int(np.clip(bx + rng.randint(20, 80), 0, w - 1))
        ey = int(np.clip(by + rng.randint(-60, 60), 0, h - 1))
        cv2.line(mask, (bx, by), (ex, ey), 255, thickness=max(1, thickness - 1))


def underwater_degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    img = img.astype(np.float32)
    # Blue/green cast, contrast loss, blur, and mild sensor noise.
    cast = np.array([rng.uniform(1.05, 1.35), rng.uniform(0.95, 1.15), rng.uniform(0.55, 0.80)], dtype=np.float32)
    img *= cast.reshape(1, 1, 3)
    img = img * rng.uniform(0.65, 0.90) + rng.uniform(15, 35)
    img = cv2.GaussianBlur(img, (5, 5), rng.uniform(0.6, 1.8))
    noise = np.random.normal(0, rng.uniform(3, 9), img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--num", type=int, default=120)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    image_dir = args.out / "images"
    mask_dir = args.out / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        base = np.full((args.height, args.width, 3), rng.randint(120, 180), dtype=np.uint8)
        texture = np.random.normal(0, 16, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + texture, 0, 255).astype(np.uint8)
        mask = np.zeros((args.height, args.width), dtype=np.uint8)
        draw_crack(mask, rng)
        img = base.copy()
        img[mask > 0] = np.clip(img[mask > 0].astype(np.int16) - rng.randint(60, 110), 0, 255)
        img = underwater_degrade(img, rng)
        name = f"event_{i:04d}.png"
        cv2.imwrite(str(image_dir / name), img)
        cv2.imwrite(str(mask_dir / name), mask)

    print(f"Generated {args.num} synthetic events under {args.out}")


if __name__ == "__main__":
    main()
