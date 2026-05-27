#!/usr/bin/env python
"""End-to-end experiment runner for ViDEC-UW.

Example:
    python scripts/generate_synthetic_dataset.py --out data/synthetic --num 120
    python scripts/run_experiment.py --image-dir data/synthetic/images --mask-dir data/synthetic/masks --out results/exp_synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Allow running without installing as a package.
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from videc_uw.evidence import construct_deo, load_image_mask
from videc_uw.simulate import simulate_methods, summarise_results
from videc_uw.plots import plot_delay_by_method, plot_selected_levels, plot_utility_cost


def build_event_table(image_dir: Path, mask_dir: Path) -> pd.DataFrame:
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
        event_id = image_path.stem
        deo = construct_deo(image, mask, event_id=event_id, depth_m=10.0 + idx % 20)
        row = {"event_id": event_id}
        row.update(deo["sizes"])
        row.update(deo["utilities"])
        row["confidence"] = deo["U"]["confidence"]
        row["uncertainty"] = deo["U"]["ambiguity"]
        row["severity"] = deo["M"]["severity"]
        records.append(row)
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/exp"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    events = build_event_table(args.image_dir, args.mask_dir)
    events.to_csv(args.out / "evidence_events.csv", index=False)

    sim = simulate_methods(events)
    summary = summarise_results(sim)
    sim.to_csv(args.out / "channel_simulation_events.csv", index=False)
    summary.to_csv(args.out / "channel_simulation_summary.csv", index=False)

    plot_delay_by_method(summary, args.out)
    plot_utility_cost(sim, args.out)
    plot_selected_levels(sim, args.out)

    print("Saved:")
    for p in [
        args.out / "evidence_events.csv",
        args.out / "channel_simulation_events.csv",
        args.out / "channel_simulation_summary.csv",
        args.out / "fig_delay_by_method.png",
        args.out / "fig_utility_cost.png",
        args.out / "fig_selected_levels.png",
    ]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
