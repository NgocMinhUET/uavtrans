#!/usr/bin/env python
"""Generate plots for UDST experiment results.

Creates:
1. Rate-Accuracy trade-off curve
2. Confidence class distribution
3. Payload breakdown by method
4. Uncertainty threshold sensitivity analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def set_paper_style():
    """Set matplotlib style for paper figures."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_rate_accuracy_curve(df: pd.DataFrame, out_dir: Path):
    """Plot rate vs accuracy trade-off curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    udst_methods = df[df["method"].str.startswith("UDST")]
    deo_methods = df[df["method"].str.startswith("DEO")]
    jpeg_methods = df[df["method"].str.contains("JPEG")]
    mask_methods = df[df["method"].str.startswith("Mask")]
    oracle = df[df["method"] == "Oracle"]
    
    colors = {
        "UDST": "#e41a1c",      # Red
        "DEO": "#377eb8",       # Blue  
        "JPEG": "#4daf4a",      # Green
        "Mask": "#984ea3",      # Purple
        "Oracle": "#ff7f00",    # Orange
    }
    
    for ax, metric, ylabel in [
        (axes[0], "dice", "Dice Score"),
        (axes[1], "iou", "IoU")
    ]:
        if len(udst_methods) > 0:
            ax.scatter(udst_methods["payload_kb"], udst_methods[metric],
                      c=colors["UDST"], marker='o', s=100, label="UDST (Proposed)",
                      edgecolors='black', linewidths=0.5, zorder=5)
            
            sorted_udst = udst_methods.sort_values("payload_kb")
            ax.plot(sorted_udst["payload_kb"], sorted_udst[metric],
                   c=colors["UDST"], linestyle='--', alpha=0.5, zorder=4)
        
        if len(deo_methods) > 0:
            ax.scatter(deo_methods["payload_kb"], deo_methods[metric],
                      c=colors["DEO"], marker='s', s=80, label="Fixed DEO",
                      edgecolors='black', linewidths=0.5, zorder=3)
        
        if len(jpeg_methods) > 0:
            roi_jpeg = jpeg_methods[jpeg_methods["method"].str.contains("ROI")]
            full_jpeg = jpeg_methods[jpeg_methods["method"].str.contains("Full")]
            
            if len(roi_jpeg) > 0:
                ax.scatter(roi_jpeg["payload_kb"], roi_jpeg[metric],
                          c=colors["JPEG"], marker='^', s=60, label="ROI-JPEG",
                          alpha=0.7, zorder=2)
            if len(full_jpeg) > 0:
                ax.scatter(full_jpeg["payload_kb"], full_jpeg[metric],
                          c=colors["JPEG"], marker='v', s=60, label="Full-JPEG",
                          alpha=0.5, zorder=2)
        
        if len(mask_methods) > 0:
            ax.scatter(mask_methods["payload_kb"], mask_methods[metric],
                      c=colors["Mask"], marker='D', s=70, label="Direct Mask",
                      edgecolors='black', linewidths=0.5, zorder=3)
        
        if len(oracle) > 0:
            ax.scatter(oracle["payload_kb"], oracle[metric],
                      c=colors["Oracle"], marker='*', s=150, label="Oracle",
                      edgecolors='black', linewidths=0.5, zorder=6)
        
        ax.set_xlabel("Payload Size (KB)")
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])
        
    fig.suptitle("Rate-Accuracy Trade-off: UDST vs Baselines", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_rate_accuracy.png")
    plt.savefig(out_dir / "fig_rate_accuracy.pdf")
    plt.close()
    print(f"Saved: {out_dir / 'fig_rate_accuracy.png'}")


def plot_confidence_distribution(df: pd.DataFrame, out_dir: Path):
    """Plot confidence class distribution for UDST methods."""
    udst = df[df["method"].str.startswith("UDST")]
    
    if len(udst) == 0:
        print("No UDST data found for confidence distribution plot")
        return
    
    if "high_conf_ratio" not in udst.columns:
        print("Missing confidence ratio columns")
        return
    
    summary = udst.groupby("method").agg({
        "high_conf_ratio": "mean",
        "medium_conf_ratio": "mean", 
        "low_conf_ratio": "mean",
        "payload_bytes": "mean",
        "dice": "mean"
    }).reset_index()
    
    summary = summary.sort_values("payload_bytes")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(summary))
    width = 0.6
    
    high = summary["high_conf_ratio"].values
    medium = summary["medium_conf_ratio"].values
    low = summary["low_conf_ratio"].values
    
    ax.bar(x, high, width, label="High Conf (geometry only)", color="#2ecc71")
    ax.bar(x, medium, width, bottom=high, label="Medium Conf (compressed mask)", color="#f39c12")
    ax.bar(x, low, width, bottom=high+medium, label="Low Conf (image patch)", color="#e74c3c")
    
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(i, 1.02, f"Dice={row['dice']:.3f}", ha='center', fontsize=8)
        ax.text(i, -0.08, f"{row['payload_bytes']/1024:.1f}KB", ha='center', fontsize=8)
    
    ax.set_xlabel("UDST Configuration")
    ax.set_ylabel("Fraction of Pixels")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("UDST_tau", "τ=") for m in summary["method"]], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 1.15])
    ax.set_title("Confidence Class Distribution by UDST Configuration")
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_confidence_distribution.png")
    plt.savefig(out_dir / "fig_confidence_distribution.pdf")
    plt.close()
    print(f"Saved: {out_dir / 'fig_confidence_distribution.png'}")


def plot_payload_breakdown(summary_df: pd.DataFrame, out_dir: Path):
    """Plot payload size comparison across methods."""
    methods_order = [
        "DEO_L0", "DEO_L1", "DEO_L2", "DEO_L3",
        "UDST_tau0.1_0.4", "UDST_tau0.2_0.6", "UDST_tau0.3_0.7", "UDST_tau0.4_0.8",
        "Mask_RLE", "Mask_PNG",
        "ROI_JPEG_Q35", "ROI_JPEG_Q75", "ROI_JPEG_Q95",
        "Full_JPEG_Q35", "Full_JPEG_Q75", "Full_JPEG_Q95",
    ]
    
    available = [m for m in methods_order if m in summary_df["method"].values]
    plot_df = summary_df[summary_df["method"].isin(available)].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=available, ordered=True)
    plot_df = plot_df.sort_values("method")
    
    if len(plot_df) == 0:
        print("No data for payload breakdown plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = []
    for m in plot_df["method"]:
        if "UDST" in m:
            colors.append("#e41a1c")
        elif "DEO" in m:
            colors.append("#377eb8")
        elif "JPEG" in m:
            colors.append("#4daf4a")
        else:
            colors.append("#984ea3")
    
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["payload_kb"], color=colors, edgecolor='black', linewidth=0.5)
    
    for i, (_, row) in enumerate(plot_df.iterrows()):
        dice_val = row.get("dice_mean", row.get("dice", 0))
        ax.text(i, row["payload_kb"] + 0.5, f"{dice_val:.2f}", ha='center', fontsize=8, rotation=90)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Payload Size (KB)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["method"], rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_title("Payload Size Comparison (numbers show Dice score)")
    
    legend_patches = [
        mpatches.Patch(color="#e41a1c", label="UDST (Proposed)"),
        mpatches.Patch(color="#377eb8", label="Fixed DEO"),
        mpatches.Patch(color="#4daf4a", label="JPEG"),
        mpatches.Patch(color="#984ea3", label="Direct Mask"),
    ]
    ax.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_payload_breakdown.png")
    plt.savefig(out_dir / "fig_payload_breakdown.pdf")
    plt.close()
    print(f"Saved: {out_dir / 'fig_payload_breakdown.png'}")


def plot_threshold_sensitivity(df: pd.DataFrame, out_dir: Path):
    """Plot sensitivity to uncertainty thresholds."""
    udst = df[df["method"].str.startswith("UDST")]
    
    if len(udst) == 0:
        return
    
    summary = udst.groupby("method").agg({
        "payload_bytes": "mean",
        "dice": ["mean", "std"],
        "iou": ["mean", "std"],
    }).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary["payload_kb"] = summary["payload_bytes_mean"] / 1024
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(
        summary["payload_kb"],
        summary["dice_mean"],
        yerr=summary["dice_std"],
        fmt='o-',
        capsize=5,
        capthick=2,
        markersize=10,
        color="#e41a1c",
        label="UDST"
    )
    
    for i, row in summary.iterrows():
        method = row["method_"]
        tau_str = method.replace("UDST_tau", "")
        ax.annotate(
            f"τ={tau_str}",
            (row["payload_kb"], row["dice_mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9
        )
    
    ax.set_xlabel("Payload Size (KB)")
    ax.set_ylabel("Dice Score")
    ax.set_title("UDST: Threshold Sensitivity Analysis")
    ax.set_ylim([0.5, 1.05])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_threshold_sensitivity.png")
    plt.savefig(out_dir / "fig_threshold_sensitivity.pdf")
    plt.close()
    print(f"Saved: {out_dir / 'fig_threshold_sensitivity.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with UDST results")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: same as results)")
    args = parser.parse_args()
    
    out_dir = args.out or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    set_paper_style()
    
    rate_acc_file = args.results_dir / "udst_rate_accuracy.csv"
    summary_file = args.results_dir / "udst_summary.csv"
    events_file = args.results_dir / "udst_all_events.csv"
    
    if rate_acc_file.exists():
        rate_acc = pd.read_csv(rate_acc_file)
        plot_rate_accuracy_curve(rate_acc, out_dir)
    
    if summary_file.exists():
        summary = pd.read_csv(summary_file)
        plot_payload_breakdown(summary, out_dir)
    
    if events_file.exists():
        events = pd.read_csv(events_file)
        plot_confidence_distribution(events, out_dir)
        plot_threshold_sensitivity(events, out_dir)
    
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
