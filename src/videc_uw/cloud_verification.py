"""Cloud verification and mask reconstruction for UDST framework.

This module handles the receiver-side operations:
1. Reconstruct masks from UDST payloads
2. Fuse predictions from different confidence classes
3. Compute verification metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .uncertainty import ConfidenceClass


@dataclass
class ReconstructionResult:
    """Result of mask reconstruction from UDST packet."""
    final_mask: np.ndarray
    high_conf_mask: np.ndarray
    medium_conf_mask: np.ndarray
    low_conf_mask: np.ndarray
    reconstruction_source: np.ndarray  # 0=none, 1=high, 2=medium, 3=low


def reconstruct_from_geometry(
    image_shape: Tuple[int, int],
    geometry_data: dict
) -> np.ndarray:
    """
    Reconstruct mask from geometry-only payload (HIGH confidence).
    
    Uses skeleton points and bounding box to create approximate mask.
    This is a coarse reconstruction that trusts the edge prediction.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x, y, bw, bh = geometry_data["bbox_xywh"]
    skeleton_points = geometry_data.get("skeleton_points", [])
    
    if not skeleton_points:
        mask[y:y+bh, x:x+bw] = 1
        return mask
    
    for px, py in skeleton_points:
        cv2.circle(mask, (px, py), radius=3, color=1, thickness=-1)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    bbox_mask = np.zeros_like(mask)
    bbox_mask[y:y+bh, x:x+bw] = 1
    mask = mask & bbox_mask
    
    return mask


def reconstruct_from_compressed_mask(
    image_shape: Tuple[int, int],
    mask_data: dict,
    compressed_mask: np.ndarray
) -> np.ndarray:
    """
    Reconstruct mask from compressed mask payload (MEDIUM confidence).
    
    Upscales the downsampled mask back to original resolution.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x, y, bw, bh = mask_data["bbox_xywh"]
    orig_w, orig_h = mask_data["original_size"]
    
    if orig_w == 0 or orig_h == 0:
        return mask
    
    upscaled = cv2.resize(
        compressed_mask.astype(np.uint8), 
        (orig_w, orig_h), 
        interpolation=cv2.INTER_NEAREST
    )
    
    y_end = min(y + orig_h, h)
    x_end = min(x + orig_w, w)
    mask[y:y_end, x:x_end] = upscaled[:y_end-y, :x_end-x]
    
    return mask


def reconstruct_from_image_patch(
    image_shape: Tuple[int, int],
    patch_data: dict,
    ground_truth_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reconstruct mask from image patch payload (LOW confidence).
    
    In a real system, the cloud would run a segmentation model on the patch.
    For simulation, we use the ground truth mask for that region.
    
    Args:
        image_shape: Shape of full image
        patch_data: Metadata about the patch
        ground_truth_mask: Ground truth for simulation (optional)
    
    Returns:
        Reconstructed mask for this region
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x, y, bw, bh = patch_data["bbox_xywh"]
    
    if ground_truth_mask is not None:
        y_end = min(y + bh, h)
        x_end = min(x + bw, w)
        mask[y:y_end, x:x_end] = ground_truth_mask[y:y_end, x:x_end]
    else:
        mask[y:y+bh, x:x+bw] = 1
    
    return mask


def fuse_reconstructions(
    image_shape: Tuple[int, int],
    high_conf_masks: list[np.ndarray],
    medium_conf_masks: list[np.ndarray],
    low_conf_masks: list[np.ndarray]
) -> ReconstructionResult:
    """
    Fuse reconstructions from different confidence classes.
    
    Priority: LOW > MEDIUM > HIGH (more detailed overrides coarser)
    """
    h, w = image_shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    source_map = np.zeros((h, w), dtype=np.uint8)
    
    high_combined = np.zeros((h, w), dtype=np.uint8)
    medium_combined = np.zeros((h, w), dtype=np.uint8)
    low_combined = np.zeros((h, w), dtype=np.uint8)
    
    for m in high_conf_masks:
        high_combined = np.maximum(high_combined, m)
    
    for m in medium_conf_masks:
        medium_combined = np.maximum(medium_combined, m)
    
    for m in low_conf_masks:
        low_combined = np.maximum(low_combined, m)
    
    final_mask = high_combined.copy()
    source_map[high_combined > 0] = 1
    
    final_mask = np.where(medium_combined > 0, medium_combined, final_mask)
    source_map[medium_combined > 0] = 2
    
    final_mask = np.where(low_combined > 0, low_combined, final_mask)
    source_map[low_combined > 0] = 3
    
    return ReconstructionResult(
        final_mask=final_mask,
        high_conf_mask=high_combined,
        medium_conf_mask=medium_combined,
        low_conf_mask=low_combined,
        reconstruction_source=source_map
    )


def simulate_cloud_verification(
    image_shape: Tuple[int, int],
    high_conf_geometry: list[dict],
    medium_conf_masks: list[dict],
    low_conf_patches: list[dict],
    ground_truth_mask: np.ndarray,
    compressed_masks: Optional[list[np.ndarray]] = None
) -> ReconstructionResult:
    """
    Simulate complete cloud verification process.
    
    Args:
        image_shape: Shape of original image
        high_conf_geometry: Geometry data for HIGH confidence regions
        medium_conf_masks: Mask data for MEDIUM confidence regions
        low_conf_patches: Patch data for LOW confidence regions
        ground_truth_mask: Ground truth for simulation
        compressed_masks: Actual compressed masks (simulated if None)
    
    Returns:
        ReconstructionResult with fused mask
    """
    high_masks = []
    for geo in high_conf_geometry:
        mask = reconstruct_from_geometry(image_shape, geo)
        high_masks.append(mask)
    
    medium_masks = []
    for i, mask_data in enumerate(medium_conf_masks):
        x, y, bw, bh = mask_data["bbox_xywh"]
        mask_side = mask_data.get("mask_side", 32)
        
        gt_roi = ground_truth_mask[y:y+bh, x:x+bw]
        small = cv2.resize(gt_roi.astype(np.uint8), (mask_side, mask_side), 
                          interpolation=cv2.INTER_AREA)
        small = (small > 0.25).astype(np.uint8)
        
        mask = reconstruct_from_compressed_mask(image_shape, mask_data, small)
        medium_masks.append(mask)
    
    low_masks = []
    for patch_data in low_conf_patches:
        mask = reconstruct_from_image_patch(
            image_shape, patch_data, ground_truth_mask
        )
        low_masks.append(mask)
    
    return fuse_reconstructions(image_shape, high_masks, medium_masks, low_masks)


def compute_verification_metrics(
    reconstructed_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> dict:
    """
    Compute comprehensive verification metrics.
    
    Returns:
        Dictionary with all metrics
    """
    pred = (reconstructed_mask > 0).astype(np.float32)
    gt = (ground_truth_mask > 0).astype(np.float32)
    
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    tn = np.sum((1 - pred) * (1 - gt))
    
    total = tp + fp + fn + tn
    pixel_accuracy = (tp + tn) / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    dice = f1  # Same as F1 for binary
    
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / union if union > 0 else 1.0
    
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    def skeleton_length(m):
        return float(skeletonize(m > 0).sum())
    
    gt_length = skeleton_length(gt)
    pred_length = skeleton_length(pred)
    length_error = abs(pred_length - gt_length) / gt_length if gt_length > 0 else 0
    
    gt_area = float(gt.sum())
    pred_area = float(pred.sum())
    area_error = abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0
    
    return {
        "pixel_accuracy": float(pixel_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "specificity": float(specificity),
        "length_error": float(min(length_error, 1.0)),
        "area_error": float(min(area_error, 1.0))
    }


def compute_per_class_metrics(
    reconstruction_result: ReconstructionResult,
    ground_truth_mask: np.ndarray
) -> dict:
    """
    Compute metrics broken down by confidence class.
    
    This helps analyze where UDST gains/loses accuracy.
    """
    results = {}
    
    results["overall"] = compute_verification_metrics(
        reconstruction_result.final_mask, ground_truth_mask
    )
    
    for class_name, class_mask in [
        ("high_conf", reconstruction_result.high_conf_mask),
        ("medium_conf", reconstruction_result.medium_conf_mask),
        ("low_conf", reconstruction_result.low_conf_mask)
    ]:
        if class_mask.sum() > 0:
            gt_in_region = ground_truth_mask * (class_mask > 0)
            results[class_name] = compute_verification_metrics(class_mask, gt_in_region)
        else:
            results[class_name] = None
    
    source = reconstruction_result.reconstruction_source
    results["source_distribution"] = {
        "none_ratio": float((source == 0).sum() / source.size),
        "high_ratio": float((source == 1).sum() / source.size),
        "medium_ratio": float((source == 2).sum() / source.size),
        "low_ratio": float((source == 3).sum() / source.size),
    }
    
    return results
