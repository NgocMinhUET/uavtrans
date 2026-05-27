"""Plotting utilities for ATC-ready figures."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_delay_by_method(summary: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    pivot = summary.pivot(index="method", columns="channel", values="avg_delay_s")
    ax = pivot.plot(kind="bar", figsize=(11, 5), logy=True)
    ax.set_ylabel("Average transmission delay (s, log scale)")
    ax.set_xlabel("Method")
    ax.set_title("Delay under underwater channel abstractions")
    ax.legend(title="Channel", fontsize=8)
    plt.tight_layout()
    path = out_dir / "fig_delay_by_method.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_utility_cost(sim_df: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    grouped = sim_df.groupby("method", as_index=False).agg(
        avg_size_bytes=("size_bytes", "mean"), avg_utility=("utility", "mean")
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(grouped["avg_size_bytes"], grouped["avg_utility"])
    for _, r in grouped.iterrows():
        ax.annotate(r["method"], (r["avg_size_bytes"], r["avg_utility"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Average transmitted size per event (bytes, log scale)")
    ax.set_ylabel("Verification utility")
    ax.set_title("Utility-cost trade-off")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig_utility_cost.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def plot_selected_levels(sim_df: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    proposed = sim_df[sim_df["method"] == "proposed_channel_aware_deo"]
    counts = proposed.groupby(["channel", "selected_level"]).size().unstack(fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    ax = fractions.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_ylabel("Fraction of events")
    ax.set_xlabel("Channel")
    ax.set_title("Evidence level selected by the channel-aware scheduler")
    ax.legend(title="Level")
    plt.tight_layout()
    path = out_dir / "fig_selected_levels.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return path
