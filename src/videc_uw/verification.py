"""Verification metrics for Level-3 ViDEC-UW experiments."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class VerificationMetrics:
    iou_l1: float
    iou_l2: float
    iou_l3: float
    dice_l1: float
    dice_l2: float
    dice_l3: float
    length_error_l1: float
    length_error_l2: float
    length_error_l3: float
    area_error_l1: float
    area_error_l2: float
    area_error_l3: float


def _bbox_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    out = np.zeros_like(mask, dtype=np.uint8)
    if len(xs) == 0:
        return out
    out[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1] = 1
    return out


def _low_res_reconstruct(mask: np.ndarray, side: int = 32) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    out = np.zeros_like(mask, dtype=np.uint8)
    if len(xs) == 0:
        return out
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    small = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    small = (small > 0.25).astype(np.uint8)
    rec = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    out[y0:y1, x0:x1] = rec
    return out


def _refined_reconstruct(mask: np.ndarray) -> np.ndarray:
    # Simulates a compact residual/refinement: mostly preserves the mask but
    # removes small jagged artifacts through morphology.
    m = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    rec = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    rec = cv2.morphologyEx(rec, cv2.MORPH_CLOSE, kernel)
    if rec.sum() == 0 and m.sum() > 0:
        rec = m
    return rec.astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom else 1.0


def _skeleton_length(mask: np.ndarray) -> float:
    return float(skeletonize(mask > 0).sum())


def _norm_err(est: float, ref: float) -> float:
    if ref <= 1e-9:
        return 0.0 if est <= 1e-9 else 1.0
    return float(min(abs(est - ref) / ref, 1.0))


def compute_verification_metrics(mask: np.ndarray) -> dict[str, Any]:
    """Compute true mask-based metrics for L1/L2/L3 reconstructions.

    L1 uses a bbox-only reconstruction, L2 uses compact low-resolution mask
    support, and L3 uses refined residual support. These are still simulated
    evidence reconstructions, but the metrics are computed against a ground
    truth mask rather than a hand-crafted utility proxy.
    """
    gt = (mask > 0).astype(np.uint8)
    rec_l1 = _bbox_mask(gt)
    rec_l2 = _low_res_reconstruct(gt)
    rec_l3 = _refined_reconstruct(gt)

    ref_len = _skeleton_length(gt)
    ref_area = float(gt.sum())
    metrics = VerificationMetrics(
        iou_l1=_iou(rec_l1, gt),
        iou_l2=_iou(rec_l2, gt),
        iou_l3=_iou(rec_l3, gt),
        dice_l1=_dice(rec_l1, gt),
        dice_l2=_dice(rec_l2, gt),
        dice_l3=_dice(rec_l3, gt),
        length_error_l1=_norm_err(_skeleton_length(rec_l1), ref_len),
        length_error_l2=_norm_err(_skeleton_length(rec_l2), ref_len),
        length_error_l3=_norm_err(_skeleton_length(rec_l3), ref_len),
        area_error_l1=_norm_err(float(rec_l1.sum()), ref_area),
        area_error_l2=_norm_err(float(rec_l2.sum()), ref_area),
        area_error_l3=_norm_err(float(rec_l3.sum()), ref_area),
    )
    return asdict(metrics)


def verification_score_from_metrics(row: dict[str, Any] | Any, level: str) -> float:
    """A paper-friendly verification score from true IoU/Dice/error metrics."""
    suffix = level.lower()
    iou = float(row[f"iou_{suffix}"])
    dice = float(row[f"dice_{suffix}"])
    le = float(row[f"length_error_{suffix}"])
    ae = float(row[f"area_error_{suffix}"])
    return float(max(0.0, min(1.0, 0.35 * iou + 0.35 * dice + 0.15 * (1 - le) + 0.15 * (1 - ae))))
