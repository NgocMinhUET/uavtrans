#!/usr/bin/env python
"""Run Level-3 ViDEC-UW experiment.

This runner adds three upgrades over the basic proof-of-concept:
1. optional underwater-like image degradation;
2. true mask-based verification metrics: IoU, Dice, length error, area error;
3. channel simulation using those verification metrics as utility.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from videc_uw.degradation import DegradationConfig
from videc_uw.level3 import build_level3_event_table
from videc_uw.simulate import simulate_methods, summarise_results
from videc_uw.plots import plot_delay_by_method, plot_selected_levels, plot_utility_cost


def save_improvement_ratios(summary: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    rows = []
    for channel in summary["channel"].unique():
        sub = summary[summary["channel"] == channel].set_index("method")
        if "proposed_channel_aware_deo" not in sub.index:
            continue
        prop = sub.loc["proposed_channel_aware_deo"]
        for base in ["roi_jpeg", "full_jpeg", "fixed_l2", "fixed_l3"]:
            if base not in sub.index:
                continue
            b = sub.loc[base]
            rows.append({
                "channel": channel,
                "baseline": base,
                "size_reduction_x": b["avg_size_bytes"] / prop["avg_size_bytes"],
                "delay_reduction_x": b["avg_delay_s"] / prop["avg_delay_s"],
                "utility_gap": b["avg_utility"] - prop["avg_utility"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def save_selected_level_distribution(sim: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    tab = (
        sim[sim["method"] == "proposed_channel_aware_deo"]
        .groupby(["channel", "selected_level"])
        .size()
        .reset_index(name="count")
    )
    tab["ratio"] = tab.groupby("channel")["count"].transform(lambda x: x / x.sum())
    tab.to_csv(out_csv, index=False)
    return tab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/level3"))
    parser.add_argument("--no-degradation", action="store_true")
    parser.add_argument("--haze", type=float, default=0.35)
    parser.add_argument("--blur-sigma", type=float, default=1.1)
    parser.add_argument("--noise-sigma", type=float, default=6.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = DegradationConfig(haze=args.haze, blur_sigma=args.blur_sigma, noise_sigma=args.noise_sigma)

    events = build_level3_event_table(
        args.image_dir,
        args.mask_dir,
        apply_degradation=not args.no_degradation,
        degradation_cfg=cfg,
    )
    events.to_csv(args.out / "level3_evidence_events.csv", index=False)

    sim = simulate_methods(events)
    summary = summarise_results(sim)
    sim.to_csv(args.out / "level3_channel_simulation_events.csv", index=False)
    summary.to_csv(args.out / "level3_channel_simulation_summary.csv", index=False)
    save_improvement_ratios(summary, args.out / "level3_improvement_ratios.csv")
    save_selected_level_distribution(sim, args.out / "level3_selected_level_distribution.csv")

    plot_delay_by_method(summary, args.out)
    plot_utility_cost(sim, args.out)
    plot_selected_levels(sim, args.out)

    print("Saved Level-3 outputs to", args.out)
    for p in sorted(args.out.iterdir()):
        print("  -", p)


if __name__ == "__main__":
    main()
