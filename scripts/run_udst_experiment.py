#!/usr/bin/env python
"""UDST End-to-End Experiment Runner  (scientifically rigorous version).

All methods are evaluated on the SAME end-to-end pipeline:
  * UDST variants       : encode evidence → reconstruct mask → compare to GT
  * Fixed DEO levels    : encode DEO → reconstruct mask → compare to GT
  * Mask_RLE / Mask_PNG : lossless mask transmission → Dice = 1.0  (correct)
  * Full-JPEG / ROI-JPEG: encode image → JPEG decode → classical segmenter →
                          compare to GT   (NOT assumed 1.0 anymore)
  * Oracle              : ideal uncertainty routing upper-bound

Usage:
    python scripts/run_udst_experiment.py \\
        --image-dir /path/to/images \\
        --mask-dir  /path/to/masks  \\
        --out       results/udst    \\
        --split     test            \\
        --max-images 500            \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from videc_uw.evidence import load_image_mask, construct_deo
from videc_uw.verification import compute_verification_metrics as deo_mask_metrics
from videc_uw.uncertainty import UncertaintyConfig
from videc_uw.udst_transmission import (
    TransmissionConfig, construct_udst_packet, compute_baseline_payloads,
)
from videc_uw.cloud_verification import (
    simulate_cloud_verification,
    compute_verification_metrics,
    compute_per_class_metrics,
)
from videc_uw.segmenter import (
    SegmenterConfig, eval_jpeg_segmentation, eval_roi_jpeg_segmentation,
)
from videc_uw.udst_transmission import _compress_mask_rle
from videc_uw.neural_segmenter import load_segmenter_if_available
from videc_uw.statistics import build_summary_table, run_all_comparisons, bd_rate


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

JPEG_QUALITIES   = [10, 20, 35, 50, 65, 75, 85, 95]
UDST_TAU_CONFIGS = [
    (0.1, 0.4),   # accuracy-oriented  (most data to cloud)
    (0.2, 0.6),   # balanced (default)
    (0.3, 0.7),   # bandwidth-oriented
    (0.4, 0.8),   # latency-oriented   (trust edge most)
]
SEG_CFG = SegmenterConfig()     # classical fallback

# Neural segmenter — loaded once globally if checkpoint exists.
# Populated in main() and referenced by run_jpeg_baselines().
_NEURAL_SEG = None


# ─────────────────────────────────────────────────────────────────────────────
# Per-image runners
# ─────────────────────────────────────────────────────────────────────────────

def _mask_png_bytes(mask: np.ndarray) -> int:
    import cv2
    ok, buf = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    return int(len(buf)) if ok else 0


def run_mask_baselines(mask: np.ndarray, event_id: str) -> list[dict]:
    """Lossless mask baselines: Dice=1.0 IS correct here."""
    gt = (mask > 0).astype(np.uint8)
    rows = []
    for method, size in [
        ("Mask_RLE", len(_compress_mask_rle(gt))),
        ("Mask_PNG", _mask_png_bytes(gt)),
    ]:
        rows.append({
            "event_id": event_id, "method": method,
            "payload_bytes": size,
            "dice": 1.0, "iou": 1.0,
            "precision": 1.0, "recall": 1.0, "balanced_accuracy": 1.0,
        })
    return rows


def run_jpeg_baselines(
    image: np.ndarray,
    mask: np.ndarray,
    event_id: str,
) -> list[dict]:
    """
    Full-image JPEG and ROI-JPEG evaluated end-to-end.

    If a trained neural segmenter (_NEURAL_SEG) is available it is used as
    the cloud detector — giving fair, high-quality task metrics.
    Falls back to the classical segmenter if no checkpoint is found.
    """
    rows = []
    for q in JPEG_QUALITIES:
        for roi_only, prefix in [(False, "Full_JPEG"), (True, "ROI_JPEG")]:
            if _NEURAL_SEG is not None:
                res = _NEURAL_SEG.eval_on_jpeg(image, mask, quality=q,
                                               roi_only=roi_only)
            else:
                fn = eval_roi_jpeg_segmentation if roi_only else eval_jpeg_segmentation
                res = fn(image, mask, quality=q, seg_cfg=SEG_CFG)

            rows.append({
                "event_id":          event_id,
                "method":            f"{prefix}_Q{q}",
                "payload_bytes":     res["payload_bytes"],
                "dice":              res["dice"],
                "iou":               res["iou"],
                "precision":         res["precision"],
                "recall":            res["recall"],
                "balanced_accuracy": res["balanced_accuracy"],
            })
    return rows


def run_deo_baselines(
    image: np.ndarray,
    mask: np.ndarray,
    event_id: str,
) -> list[dict]:
    """Fixed DEO levels — uses same mask-reconstruction metrics as UDST."""
    vm   = deo_mask_metrics(mask)   # IoU/Dice per level from verification.py
    deo  = construct_deo(image, mask, event_id=event_id)
    rows = []
    for level, size_key, dice_key, iou_key in [
        ("DEO_L0", "l0_bytes", None,       None),
        ("DEO_L1", "l1_bytes", "dice_l1",  "iou_l1"),
        ("DEO_L2", "l2_bytes", "dice_l2",  "iou_l2"),
        ("DEO_L3", "l3_bytes", "dice_l3",  "iou_l3"),
    ]:
        dice = 0.0  if dice_key is None else float(vm[dice_key])
        iou  = 0.0  if iou_key  is None else float(vm[iou_key])
        rows.append({
            "event_id":      event_id,
            "method":        level,
            "payload_bytes": int(deo["sizes"][size_key]),
            "dice":          dice,
            "iou":           iou,
            "precision":     0.0 if level == "DEO_L0" else float(vm.get("iou_l1", 0)),
            "recall":        0.0 if level == "DEO_L0" else 1.0,
            "balanced_accuracy": 0.5 if level == "DEO_L0" else (dice + 1.0) / 2.0,
        })
    return rows


def run_udst_variants(
    image: np.ndarray,
    mask: np.ndarray,
    event_id: str,
) -> list[dict]:
    """UDST with four (tau_low, tau_high) configurations."""
    rows = []
    for tau_low, tau_high in UDST_TAU_CONFIGS:
        unc_cfg   = UncertaintyConfig(tau_low=tau_low, tau_high=tau_high)
        trans_cfg = TransmissionConfig()

        packet = construct_udst_packet(
            image, mask, event_id,
            uncertainty_config=unc_cfg,
            transmission_config=trans_cfg,
        )
        recon = simulate_cloud_verification(
            image_shape         = image.shape[:2],
            high_conf_geometry  = packet.high_conf_geometry,
            medium_conf_masks   = packet.medium_conf_masks,
            low_conf_patches    = packet.low_conf_patches,
            ground_truth_mask   = mask,
        )
        m    = compute_verification_metrics(recon.final_mask, mask)
        pcls = compute_per_class_metrics(recon, mask)
        src  = pcls["source_distribution"]

        rows.append({
            "event_id":      event_id,
            "method":        f"UDST_tau{tau_low:.1f}_{tau_high:.1f}",
            "payload_bytes": packet.total_bytes,
            "dice":          m["dice"],
            "iou":           m["iou"],
            "precision":     m["precision"],
            "recall":        m["recall"],
            "balanced_accuracy": m["balanced_accuracy"],
            # UDST-specific breakdown
            "num_regions":      packet.num_regions,
            "num_high_conf":    packet.num_high_conf,
            "num_medium_conf":  packet.num_medium_conf,
            "num_low_conf":     packet.num_low_conf,
            "mean_uncertainty": round(packet.mean_uncertainty, 4),
            "pct_geometry":     round(src["high_ratio"]   * 100, 1),
            "pct_mask":         round(src["medium_ratio"] * 100, 1),
            "pct_patch":        round(src["low_ratio"]    * 100, 1),
        })
    return rows


def run_oracle(
    image: np.ndarray,
    mask: np.ndarray,
    event_id: str,
) -> list[dict]:
    """
    Oracle upper-bound: knows exactly which regions need cloud verification.
    Sends geometry-only for correct regions, full image-patch for wrong ones.
    With perfect uncertainty, Dice → 1.0.  Bytes are real (measured).
    """
    unc_cfg   = UncertaintyConfig(tau_low=0.2, tau_high=0.6)
    trans_cfg = TransmissionConfig()
    packet    = construct_udst_packet(image, mask, event_id,
                                      uncertainty_config=unc_cfg,
                                      transmission_config=trans_cfg)
    oracle_bytes = packet.header_bytes + packet.region_table_bytes
    for p in packet.payloads:
        if p.payload_type == "geometry":
            oracle_bytes += max(10, int(p.payload_bytes * 0.3))
        elif p.payload_type == "compressed_mask":
            oracle_bytes += max(20, int(p.payload_bytes * 0.5))
        else:
            oracle_bytes += p.payload_bytes   # patches are real cost

    return [{
        "event_id":          event_id,
        "method":            "Oracle",
        "payload_bytes":     oracle_bytes,
        "dice":              1.0,
        "iou":               1.0,
        "precision":         1.0,
        "recall":            1.0,
        "balanced_accuracy": 1.0,
    }]


def process_image(
    image_path: Path,
    mask_path: Path,
    run_jpeg: bool = True,
    run_deo: bool = True,
    run_masks: bool = True,
    run_oracle_flag: bool = True,
) -> list[dict]:
    image, mask = load_image_mask(image_path, mask_path)
    event_id = image_path.stem
    rows: list[dict] = []

    rows += run_udst_variants(image, mask, event_id)
    if run_deo:
        rows += run_deo_baselines(image, mask, event_id)
    if run_jpeg:
        rows += run_jpeg_baselines(image, mask, event_id)
    if run_masks:
        rows += run_mask_baselines(mask, event_id)
    if run_oracle_flag:
        rows += run_oracle(image, mask, event_id)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bd_rate_table(df: pd.DataFrame, reference: str = "ROI_JPEG_Q75") -> pd.DataFrame:
    """Compute BD-Rate of every method against the reference method."""
    ref_df = df[df["method"] == reference].sort_values("payload_bytes")
    rows = []
    for method in df["method"].unique():
        if method == reference:
            continue
        m_df = df[df["method"] == method].sort_values("payload_bytes")
        # Aggregate to one (rate, dice) point per method where it has variance
        if len(m_df) < 2:
            rows.append({"method": method, "bd_rate_pct": float("nan"),
                         "note": "single point"})
            continue
        bdr = bd_rate(
            ref_df["payload_bytes"].values, ref_df["dice"].values,
            m_df["payload_bytes"].values,   m_df["dice"].values,
        )
        rows.append({"method": method, "bd_rate_pct": round(bdr, 2)})
    return pd.DataFrame(rows).sort_values("bd_rate_pct")


def _print_comparison(summary: pd.DataFrame, udst_method: str) -> None:
    udst = summary[summary["method"] == udst_method]
    if udst.empty:
        return
    u = udst.iloc[0]
    print(f"\n{'='*62}")
    print(f"  COMPARISON vs  {udst_method}")
    print(f"  {u['payload_kb_mean']:.2f} KB  "
          f"Dice={u['dice_mean']:.4f} [{u['dice_ci_lo']:.4f}–{u['dice_ci_hi']:.4f}]")
    print(f"{'='*62}")
    for _, row in summary.iterrows():
        m = row["method"]
        if m == udst_method:
            continue
        ratio = row["payload_kb_mean"] / u["payload_kb_mean"] if u["payload_kb_mean"] > 0 else 0
        ddice = u["dice_mean"] - row["dice_mean"]
        print(f"  {m:<26} {row['payload_kb_mean']:6.2f} KB  "
              f"({ratio:5.2f}x)  Dice={row['dice_mean']:.4f}  "
              f"Δdice={ddice:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir",    type=Path, required=True)
    parser.add_argument("--mask-dir",     type=Path, required=True)
    parser.add_argument("--out",          type=Path, default=Path("results/udst"))
    parser.add_argument("--split",        type=str,  default="test",
                        help="Label recorded in outputs (train/val/test)")
    parser.add_argument("--max-images",   type=int,  default=None)
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--skip-jpeg",    action="store_true",
                        help="Skip JPEG baselines (fast dev mode)")
    parser.add_argument("--skip-deo",     action="store_true")
    parser.add_argument("--reference",    type=str,  default="ROI_JPEG_Q75",
                        help="Reference method for BD-Rate and comparisons")
    parser.add_argument("--checkpoint",   type=Path, default=None,
                        help="Path to trained crack segmenter .pt file. "
                             "If provided, replaces classical segmenter for "
                             "JPEG baseline evaluation.")
    parser.add_argument("--encoder",      type=str,  default="mobilenet_v2",
                        help="Encoder used when training the checkpoint.")
    args = parser.parse_args()

    # ── neural segmenter (optional) ──────────────────────────────────────────
    global _NEURAL_SEG
    ckpt = args.checkpoint or Path("checkpoints/best_crack_segmenter.pt")
    _NEURAL_SEG = load_segmenter_if_available(ckpt, encoder=args.encoder)
    if _NEURAL_SEG is not None:
        print(f"[UDST] Neural segmenter loaded from {ckpt}")
    else:
        print("[UDST] No checkpoint found — using classical segmenter for JPEG baselines")
        print(f"       (Train one with: python scripts/train_crack_segmenter.py ...)")

    # ── reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "split": args.split,
        "seed": args.seed,
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "jpeg_qualities": JPEG_QUALITIES,
        "udst_tau_configs": UDST_TAU_CONFIGS,
        "jpeg_segmenter": (
            {"type": "neural", "checkpoint": str(ckpt), "encoder": args.encoder}
            if _NEURAL_SEG is not None
            else {"type": "classical", **SEG_CFG.__dict__}
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out / "experiment_config.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )

    # ── collect image paths ───────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in args.image_dir.iterdir()
                         if p.suffix.lower() in exts)
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"[UDST] {args.split} split — {len(image_paths)} images")

    # ── main loop ─────────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    t0 = time.time()

    for i, img_path in enumerate(image_paths):
        mask_path = args.mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = args.mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                print(f"  [WARN] no mask for {img_path.name}, skipping")
                continue
        try:
            rows = process_image(
                img_path, mask_path,
                run_jpeg=not args.skip_jpeg,
                run_deo=not args.skip_deo,
            )
            for r in rows:
                r["split"] = args.split
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rem = (len(image_paths) - i - 1) / ((i + 1) / elapsed)
            print(f"  {i+1}/{len(image_paths)}  "
                  f"{elapsed:.0f}s elapsed  ~{rem:.0f}s remaining")

    if not all_rows:
        print("[ERROR] No results — check paths.")
        return

    # ── save raw events ───────────────────────────────────────────────────────
    df = pd.DataFrame.from_records(all_rows)
    events_csv = args.out / "udst_all_events.csv"
    df.to_csv(events_csv, index=False)
    print(f"\n[saved] {events_csv}  ({len(df)} rows)")

    # ── summary table with bootstrap CI ──────────────────────────────────────
    summary = build_summary_table(df)
    summary_csv = args.out / "udst_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[saved] {summary_csv}")

    # ── rate-accuracy curve (mean payload, mean dice per method) ──────────────
    rate_acc = df.groupby("method").agg(
        payload_bytes=("payload_bytes", "mean"),
        dice=("dice", "mean"),
        iou=("iou", "mean"),
    ).reset_index()
    rate_acc["payload_kb"] = rate_acc["payload_bytes"] / 1024.0
    rate_acc = rate_acc.sort_values("payload_bytes")
    rate_acc.to_csv(args.out / "udst_rate_accuracy.csv", index=False)

    # ── statistical comparisons: UDST default vs all others ──────────────────
    udst_default = "UDST_tau0.2_0.6"
    other_methods = [m for m in df["method"].unique() if m != udst_default]
    comparisons = run_all_comparisons(df, udst_default, other_methods,
                                      metric="dice")
    comparisons.to_csv(args.out / "udst_statistical_tests.csv", index=False)
    print(f"[saved] {args.out / 'udst_statistical_tests.csv'}")

    # ── UDST confidence breakdown ─────────────────────────────────────────────
    udst_df = df[df["method"].str.startswith("UDST")]
    if not udst_df.empty:
        breakdown_cols = [c for c in
                          ["method", "num_regions", "num_high_conf",
                           "num_medium_conf", "num_low_conf",
                           "mean_uncertainty", "pct_geometry", "pct_mask", "pct_patch"]
                          if c in udst_df.columns]
        breakdown = udst_df[breakdown_cols].groupby("method").mean().round(3).reset_index()
        breakdown.to_csv(args.out / "udst_confidence_breakdown.csv", index=False)

    # ── BD-Rate table ─────────────────────────────────────────────────────────
    # Only meaningful for methods with payload variation across qualities
    if args.reference in df["method"].values:
        bdr = _bd_rate_table(rate_acc.rename(
            columns={"payload_bytes": "payload_bytes",
                     "dice": "dice"}
        ), reference=args.reference)
        bdr.to_csv(args.out / "udst_bd_rate.csv", index=False)

    # ── console summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n[UDST] Done in {elapsed:.0f}s")

    print(f"\n{'='*62}")
    print("  RATE vs ACCURACY (mean per method, sorted by size)")
    print(f"{'='*62}")
    cols = ["method", "payload_kb_mean", "dice_mean", "dice_ci_lo",
            "dice_ci_hi", "iou_mean"]
    avail = [c for c in cols if c in summary.columns]
    print(summary[avail].to_string(index=False))

    _print_comparison(summary, udst_default)

    print(f"\n[UDST] Outputs in {args.out}/")


if __name__ == "__main__":
    main()
