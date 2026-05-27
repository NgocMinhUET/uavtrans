"""Lightweight runtime profiling helper for laptop/Jetson edge feasibility."""
from __future__ import annotations

from pathlib import Path
import platform
import time
from typing import Iterable

import pandas as pd

from .evidence import construct_deo, load_image_mask


def profile_dataset(image_dir: Path, mask_dir: Path, repeats: int = 1) -> pd.DataFrame:
    records = []
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        image, mask = load_image_mask(image_path, mask_path)
        for r in range(repeats):
            t0 = time.perf_counter()
            deo = construct_deo(image, mask, event_id=image_path.stem)
            elapsed = time.perf_counter() - t0
            records.append({
                "event_id": image_path.stem,
                "repeat": r,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "construction_time_ms": elapsed * 1000.0,
                "l1_bytes": deo["sizes"]["l1_bytes"],
                "l2_bytes": deo["sizes"]["l2_bytes"],
                "l3_bytes": deo["sizes"]["l3_bytes"],
            })
    return pd.DataFrame.from_records(records)
