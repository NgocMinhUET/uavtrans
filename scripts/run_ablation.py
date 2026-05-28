#!/usr/bin/env python
"""Ablation study runner for UDST.

Experiments:
  A1 — uncertainty source   (entropy / edge-distance / combined)
  A2 — threshold grid       (tau_low x tau_high sweep)
  A3 — confidence classes   (2-class vs 3-class policy)
  A4 — region granularity   (connected-component vs fixed-tile regions)

Usage:
    python scripts/run_ablation.py \\
        --image-dir /path/to/images \\
        --mask-dir  /path/to/masks  \\
        --out       results/ablation \\
        --max-images 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from videc_uw.evidence import load_image_mask
from videc_uw.uncertainty import UncertaintyConfig
from videc_uw.udst_transmission import TransmissionConfig, construct_udst_packet
from videc_uw.cloud_verification import (
    simulate_cloud_verification,
    compute_verification_metrics,
)
from videc_uw.statistics import build_summary_table


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Uncertainty source
# ─────────────────────────────────────────────────────────────────────────────

def _eval_udst(image, mask, event_id, unc_cfg, label) -> dict:
    trans_cfg = TransmissionConfig()
    packet = construct_udst_packet(image, mask, event_id,
                                   uncertainty_config=unc_cfg,
                                   transmission_config=trans_cfg)
    recon = simulate_cloud_verification(
        image_shape        = image.shape[:2],
        high_conf_geometry = packet.high_conf_geometry,
        medium_conf_masks  = packet.medium_conf_masks,
        low_conf_patches   = packet.low_conf_patches,
        ground_truth_mask  = mask,
    )
    m = compute_verification_metrics(recon.final_mask, mask)
    return {
        "event_id":      event_id,
        "method":        label,
        "payload_bytes": packet.total_bytes,
        "dice":          m["dice"],
        "iou":           m["iou"],
        "precision":     m["precision"],
        "recall":        m["recall"],
        "balanced_accuracy": m["balanced_accuracy"],
        "num_high":      packet.num_high_conf,
        "num_medium":    packet.num_medium_conf,
        "num_low":       packet.num_low_conf,
    }


def ablation_uncertainty_source(image, mask, event_id) -> list[dict]:
    """A1: vary which uncertainty signal is used."""
    rows = []

    # Entropy-only  (edge_weight=0)
    rows.append(_eval_udst(image, mask, event_id,
                            UncertaintyConfig(entropy_weight=1.0, edge_weight=0.0),
                            "A1_entropy_only"))
    # Edge-only  (entropy_weight=0)
    rows.append(_eval_udst(image, mask, event_id,
                            UncertaintyConfig(entropy_weight=0.0, edge_weight=1.0),
                            "A1_edge_only"))
    # Combined (default)
    rows.append(_eval_udst(image, mask, event_id,
                            UncertaintyConfig(entropy_weight=0.6, edge_weight=0.4),
                            "A1_combined"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Threshold grid
# ─────────────────────────────────────────────────────────────────────────────

def ablation_threshold_grid(image, mask, event_id) -> list[dict]:
    """A2: sweep (tau_low, tau_high) on a grid."""
    rows = []
    for tau_low in [0.1, 0.2, 0.3, 0.4]:
        for tau_high in [0.4, 0.5, 0.6, 0.7, 0.8]:
            if tau_high <= tau_low:
                continue
            label = f"A2_tl{tau_low:.1f}_th{tau_high:.1f}"
            cfg = UncertaintyConfig(tau_low=tau_low, tau_high=tau_high)
            rows.append(_eval_udst(image, mask, event_id, cfg, label))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# A3 — 2-class vs 3-class policy
# ─────────────────────────────────────────────────────────────────────────────

def _eval_2class(image, mask, event_id, threshold: float) -> dict:
    """
    2-class policy: HIGH (geometry) vs LOW (image patch), no MEDIUM.
    Emulated by collapsing tau_low == tau_high.
    """
    cfg = UncertaintyConfig(tau_low=threshold, tau_high=threshold + 1e-6)
    return _eval_udst(image, mask, event_id, cfg,
                      f"A3_2class_t{threshold:.1f}")


def ablation_num_classes(image, mask, event_id) -> list[dict]:
    """A3: compare 2-class vs 3-class policy."""
    rows = []
    for t in [0.2, 0.3, 0.4]:
        rows.append(_eval_2class(image, mask, event_id, t))
    # 3-class (default UDST)
    for tl, th in [(0.2, 0.6), (0.3, 0.7)]:
        rows.append(_eval_udst(image, mask, event_id,
                               UncertaintyConfig(tau_low=tl, tau_high=th),
                               f"A3_3class_tl{tl:.1f}_th{th:.1f}"))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# A4 — Region granularity: connected components vs fixed tiles
# ─────────────────────────────────────────────────────────────────────────────

def _eval_tiled(image, mask, event_id, tile_size: int) -> dict:
    """
    Tile-based granularity: divide image into (tile_size x tile_size) tiles,
    treat each non-empty tile as a region, run UDST policy per tile.
    Returns equivalent payload + dice.
    """
    from videc_uw.uncertainty import (
        compute_image_uncertainty_summary, UncertaintyConfig,
        ConfidenceClass
    )
    from videc_uw.udst_transmission import (
        TransmissionConfig, compute_geometry_payload,
        compute_compressed_mask_payload, compute_image_patch_payload,
        _compress_mask_rle,
    )
    from videc_uw.cloud_verification import _compute_metrics  # type: ignore

    unc_cfg   = UncertaintyConfig()
    trans_cfg = TransmissionConfig()
    H, W = image.shape[:2]

    full_pred = np.zeros((H, W), dtype=np.uint8)
    total_bytes = trans_cfg.header_bytes

    gt = (mask > 0).astype(np.uint8)

    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            y1 = min(H, y0 + tile_size)
            x1 = min(W, x0 + tile_size)
            tile_mask = gt[y0:y1, x0:x1]
            if tile_mask.sum() == 0:
                continue   # background tile — skip entirely

            # mean uncertainty for tile: use combined uncertainty
            from videc_uw.uncertainty import combined_uncertainty, simulate_probability_from_mask
            prob = simulate_probability_from_mask(tile_mask)
            unc_map = combined_uncertainty(prob, tile_mask, unc_cfg)
            mean_unc = float(unc_map[tile_mask > 0].mean()) if tile_mask.sum() else 0.5

            # decide payload type
            if mean_unc < unc_cfg.tau_low:
                # geometry only — reconstruct as dilated bbox
                total_bytes += 60
                import cv2 as _cv2
                bm = np.zeros((H, W), dtype=np.uint8)
                bm[y0:y1, x0:x1] = tile_mask
                ys, xs = np.where(bm > 0)
                if len(xs):
                    rb = np.zeros((H, W), dtype=np.uint8)
                    rb[ys.min():ys.max()+1, xs.min():xs.max()+1] = 1
                    full_pred = np.maximum(full_pred, rb)
            elif mean_unc < unc_cfg.tau_high:
                # compressed mask
                small = _cv2.resize(tile_mask, (trans_cfg.compressed_mask_side,
                                                 trans_cfg.compressed_mask_side),
                                    interpolation=_cv2.INTER_AREA)
                small = (small > 0.25).astype(np.uint8)
                total_bytes += len(_compress_mask_rle(small))
                up = _cv2.resize(small, (x1-x0, y1-y0),
                                 interpolation=_cv2.INTER_NEAREST)
                full_pred[y0:y1, x0:x1] = np.maximum(full_pred[y0:y1, x0:x1], up)
            else:
                # image patch
                tile_img = image[y0:y1, x0:x1]
                _, buf = _cv2.imencode(".jpg", tile_img.astype(np.uint8),
                                       [_cv2.IMWRITE_JPEG_QUALITY, 65])
                total_bytes += len(buf)
                # oracle: cloud gets ground truth
                full_pred[y0:y1, x0:x1] = np.maximum(full_pred[y0:y1, x0:x1],
                                                       tile_mask)

    from videc_uw.cloud_verification import _compute_metrics as cm
    m = cm(full_pred, gt)
    m.update({
        "event_id":      event_id,
        "method":        f"A4_tile{tile_size}",
        "payload_bytes": total_bytes,
    })
    return m


def ablation_granularity(image, mask, event_id) -> list[dict]:
    """A4: connected-component UDST vs tile-based UDST."""
    rows = []
    # CC-based (default)
    rows.append(_eval_udst(image, mask, event_id,
                            UncertaintyConfig(),
                            "A4_connected_component"))
    # Tile-based
    for ts in [32, 64, 128]:
        try:
            rows.append(_eval_tiled(image, mask, event_id, ts))
        except Exception:
            pass
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir",  type=Path, required=True)
    parser.add_argument("--mask-dir",   type=Path, required=True)
    parser.add_argument("--out",        type=Path, default=Path("results/ablation"))
    parser.add_argument("--max-images", type=int,  default=200)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--ablations",  nargs="+",
                        default=["A1", "A2", "A3", "A4"],
                        help="Which ablations to run")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(p for p in args.image_dir.iterdir()
                         if p.suffix.lower() in exts)[:args.max_images]
    print(f"[ABLATION] {len(image_paths)} images")

    all_a1, all_a2, all_a3, all_a4 = [], [], [], []

    t0 = time.time()
    for i, img_path in enumerate(image_paths):
        mask_path = args.mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = args.mask_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                continue
        try:
            image, mask = load_image_mask(img_path, mask_path)
            eid = img_path.stem
            if "A1" in args.ablations:
                all_a1.extend(ablation_uncertainty_source(image, mask, eid))
            if "A2" in args.ablations:
                all_a2.extend(ablation_threshold_grid(image, mask, eid))
            if "A3" in args.ablations:
                all_a3.extend(ablation_num_classes(image, mask, eid))
            if "A4" in args.ablations:
                all_a4.extend(ablation_granularity(image, mask, eid))
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            continue

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(image_paths)}")

    elapsed = time.time() - t0
    print(f"\n[ABLATION] Completed in {elapsed:.0f}s")

    label_map = {"A1": all_a1, "A2": all_a2, "A3": all_a3, "A4": all_a4}
    for ablation, rows in label_map.items():
        if not rows:
            continue
        df = pd.DataFrame.from_records(rows)
        df.to_csv(args.out / f"{ablation.lower()}_events.csv", index=False)
        summary = build_summary_table(df, n_boot=1000)
        summary.to_csv(args.out / f"{ablation.lower()}_summary.csv", index=False)
        print(f"\n{'='*55}")
        print(f"  {ablation} SUMMARY")
        print(f"{'='*55}")
        cols = [c for c in ["method", "payload_kb_mean", "dice_mean",
                             "dice_ci_lo", "dice_ci_hi", "iou_mean"]
                if c in summary.columns]
        print(summary[cols].to_string(index=False))

    print(f"\n[ABLATION] Outputs in {args.out}/")


if __name__ == "__main__":
    main()
