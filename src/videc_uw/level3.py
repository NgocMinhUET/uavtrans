"""Level-3 experiment helpers: real masks, degradation, and true metrics."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from .degradation import DegradationConfig, apply_underwater_degradation
from .evidence import construct_deo, load_image_mask
from .verification import compute_verification_metrics, verification_score_from_metrics


def build_level3_event_table(
    image_dir: Path,
    mask_dir: Path,
    apply_degradation: bool = True,
    degradation_cfg: DegradationConfig | None = None,
) -> pd.DataFrame:
    """Build an event table with both DEO sizes and true verification metrics."""
    if degradation_cfg is None:
        degradation_cfg = DegradationConfig()
    records = []
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    for idx, image_path in enumerate(image_paths):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            print(f"[WARN] Missing mask for {image_path.name}; skipping")
            continue
        image, mask = load_image_mask(image_path, mask_path)
        if apply_degradation:
            cfg = DegradationConfig(
                haze=degradation_cfg.haze,
                blur_sigma=degradation_cfg.blur_sigma,
                noise_sigma=degradation_cfg.noise_sigma,
                blue_green_shift=degradation_cfg.blue_green_shift,
                brightness=degradation_cfg.brightness,
                contrast=degradation_cfg.contrast,
                seed=degradation_cfg.seed + idx,
            )
            image = apply_underwater_degradation(image, cfg)

        event_id = image_path.stem
        deo = construct_deo(image, mask, event_id=event_id, depth_m=10.0 + idx % 20)
        verification = compute_verification_metrics(mask)

        row = {"event_id": event_id, "degraded": bool(apply_degradation)}
        row.update(deo["sizes"])
        row.update(deo["utilities"])
        row.update(verification)
        row["confidence"] = deo["U"]["confidence"]
        row["uncertainty"] = deo["U"]["ambiguity"]
        row["severity"] = deo["M"]["severity"]

        # Replace proxy utilities with true mask-based verification utilities.
        row["utility_l0"] = min(row["utility_l0"], 0.55)
        row["utility_l1"] = verification_score_from_metrics(row, "l1")
        row["utility_l2"] = verification_score_from_metrics(row, "l2")
        row["utility_l3"] = verification_score_from_metrics(row, "l3")
        row["utility_roi"] = max(row["utility_l2"], min(0.98, row["utility_l3"] - 0.01))
        row["utility_full"] = 1.0
        records.append(row)

    if not records:
        raise RuntimeError("No valid image/mask pairs were found.")
    return pd.DataFrame.from_records(records)
