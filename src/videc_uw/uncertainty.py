"""Uncertainty estimation for UDST framework.

This module provides multiple methods to estimate pixel-wise uncertainty
for crack segmentation predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import cv2
import numpy as np
from scipy import ndimage


class ConfidenceClass(Enum):
    HIGH = 0      # Trust edge prediction
    MEDIUM = 1    # Send compressed mask
    LOW = 2       # Need cloud verification (send image patch)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation."""
    tau_low: float = 0.2      # Below: HIGH confidence
    tau_high: float = 0.6     # Above: LOW confidence
    edge_band_width: int = 5  # Pixels around boundaries
    entropy_weight: float = 0.6
    edge_weight: float = 0.4
    min_region_area: int = 100


@dataclass 
class RegionInfo:
    """Information about a detected region with uncertainty."""
    region_id: int
    bbox_xywh: Tuple[int, int, int, int]
    mask: np.ndarray              # Binary mask for this region
    uncertainty_map: np.ndarray   # Pixel-wise uncertainty [0,1]
    mean_uncertainty: float
    max_uncertainty: float
    confidence_class: ConfidenceClass
    area_pixels: int


def entropy_uncertainty(prob_map: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute entropy-based uncertainty from probability map.
    
    For binary segmentation: H = -p*log(p) - (1-p)*log(1-p)
    Normalized to [0, 1] where 1 = maximum uncertainty (p=0.5)
    
    Args:
        prob_map: Probability of crack class, shape (H, W), values in [0, 1]
        eps: Small value to avoid log(0)
    
    Returns:
        Uncertainty map, shape (H, W), values in [0, 1]
    """
    p = np.clip(prob_map, eps, 1 - eps)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy  # Already in [0, 1] for binary case


def edge_based_uncertainty(
    mask: np.ndarray, 
    band_width: int = 5
) -> np.ndarray:
    """
    Compute uncertainty based on distance to segmentation boundary.
    
    Intuition: Pixels near boundaries are more uncertain than pixels
    far from boundaries.
    
    Args:
        mask: Binary mask, shape (H, W)
        band_width: Width of uncertainty band around edges
    
    Returns:
        Uncertainty map, shape (H, W), values in [0, 1]
    """
    binary = (mask > 0).astype(np.uint8)
    
    if binary.sum() == 0 or binary.sum() == binary.size:
        return np.zeros_like(mask, dtype=np.float32)
    
    edges = cv2.Canny(binary * 255, 50, 150)
    dist_to_edge = ndimage.distance_transform_edt(edges == 0)
    uncertainty = np.exp(-dist_to_edge / band_width)
    
    return uncertainty.astype(np.float32)


def prediction_variance_uncertainty(
    predictions: list[np.ndarray]
) -> np.ndarray:
    """
    Compute uncertainty from multiple predictions (ensemble/MC dropout).
    
    Args:
        predictions: List of binary masks from multiple forward passes
    
    Returns:
        Uncertainty map based on prediction variance
    """
    if len(predictions) < 2:
        raise ValueError("Need at least 2 predictions for variance")
    
    stacked = np.stack(predictions, axis=0).astype(np.float32)
    variance = np.var(stacked, axis=0)
    max_var = 0.25  # Max variance for binary (p=0.5)
    uncertainty = np.clip(variance / max_var, 0, 1)
    
    return uncertainty


def combined_uncertainty(
    prob_map: np.ndarray,
    mask: np.ndarray,
    config: UncertaintyConfig
) -> np.ndarray:
    """
    Combine multiple uncertainty sources.
    
    Args:
        prob_map: Probability map from model (or use mask as proxy)
        mask: Binary prediction mask
        config: Uncertainty configuration
    
    Returns:
        Combined uncertainty map
    """
    ent_unc = entropy_uncertainty(prob_map)
    edge_unc = edge_based_uncertainty(mask, config.edge_band_width)
    
    combined = (
        config.entropy_weight * ent_unc + 
        config.edge_weight * edge_unc
    )
    
    return np.clip(combined, 0, 1)


def simulate_probability_from_mask(
    mask: np.ndarray,
    noise_sigma: float = 0.15,
    edge_uncertainty_boost: float = 0.3
) -> np.ndarray:
    """
    Simulate a probability map from a binary mask.
    
    Since we're using ground-truth masks (Crack500), we simulate
    what a real detector's probability output might look like:
    - Core crack pixels: high probability (~0.9)
    - Edge pixels: medium probability (~0.6-0.8)
    - Background near crack: low probability (~0.1-0.3)
    - Far background: very low probability (~0.01)
    
    Args:
        mask: Binary ground-truth mask
        noise_sigma: Noise level for simulation
        edge_uncertainty_boost: Extra uncertainty at edges
    
    Returns:
        Simulated probability map
    """
    binary = (mask > 0).astype(np.float32)
    
    dist_to_crack = ndimage.distance_transform_edt(binary == 0)
    dist_to_bg = ndimage.distance_transform_edt(binary > 0)
    
    prob = np.zeros_like(mask, dtype=np.float32)
    
    prob[binary > 0] = 0.85 + 0.1 * np.random.randn(np.sum(binary > 0))
    
    near_crack = (dist_to_crack > 0) & (dist_to_crack < 10)
    prob[near_crack] = 0.2 * np.exp(-dist_to_crack[near_crack] / 3)
    
    edge_mask = edge_based_uncertainty(mask, band_width=3) > 0.5
    prob[edge_mask] = 0.5 + 0.2 * np.random.randn(np.sum(edge_mask))
    
    noise = noise_sigma * np.random.randn(*mask.shape)
    prob = prob + noise
    
    return np.clip(prob, 0.01, 0.99)


def classify_confidence(
    uncertainty: float, 
    config: UncertaintyConfig
) -> ConfidenceClass:
    """Classify region confidence based on uncertainty."""
    if uncertainty < config.tau_low:
        return ConfidenceClass.HIGH
    elif uncertainty < config.tau_high:
        return ConfidenceClass.MEDIUM
    else:
        return ConfidenceClass.LOW


def extract_regions_with_uncertainty(
    mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
    config: Optional[UncertaintyConfig] = None
) -> list[RegionInfo]:
    """
    Extract connected regions and compute uncertainty for each.
    
    Args:
        mask: Binary segmentation mask
        prob_map: Probability map (optional, will simulate if None)
        config: Uncertainty configuration
    
    Returns:
        List of RegionInfo with uncertainty estimates
    """
    if config is None:
        config = UncertaintyConfig()
    
    if prob_map is None:
        prob_map = simulate_probability_from_mask(mask)
    
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    uncertainty_map = combined_uncertainty(prob_map, mask, config)
    
    regions = []
    for i in range(1, num_labels):  # Skip background (0)
        x, y, w, h, area = stats[i]
        
        if area < config.min_region_area:
            continue
        
        region_mask = (labels == i).astype(np.uint8)
        region_uncertainty = uncertainty_map * region_mask
        
        valid_uncertainty = region_uncertainty[region_mask > 0]
        if len(valid_uncertainty) == 0:
            continue
            
        mean_unc = float(np.mean(valid_uncertainty))
        max_unc = float(np.max(valid_uncertainty))
        
        conf_class = classify_confidence(mean_unc, config)
        
        regions.append(RegionInfo(
            region_id=i,
            bbox_xywh=(x, y, w, h),
            mask=region_mask,
            uncertainty_map=region_uncertainty,
            mean_uncertainty=mean_unc,
            max_uncertainty=max_unc,
            confidence_class=conf_class,
            area_pixels=area
        ))
    
    return regions


def compute_image_uncertainty_summary(
    mask: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
    config: Optional[UncertaintyConfig] = None
) -> dict:
    """
    Compute overall uncertainty summary for an image.
    
    Returns:
        Dictionary with uncertainty statistics
    """
    if config is None:
        config = UncertaintyConfig()
    
    regions = extract_regions_with_uncertainty(mask, prob_map, config)
    
    if not regions:
        return {
            "num_regions": 0,
            "num_high_conf": 0,
            "num_medium_conf": 0,
            "num_low_conf": 0,
            "mean_uncertainty": 0.0,
            "max_uncertainty": 0.0,
            "high_conf_ratio": 1.0,
            "regions": []
        }
    
    high_conf = sum(1 for r in regions if r.confidence_class == ConfidenceClass.HIGH)
    medium_conf = sum(1 for r in regions if r.confidence_class == ConfidenceClass.MEDIUM)
    low_conf = sum(1 for r in regions if r.confidence_class == ConfidenceClass.LOW)
    
    mean_unc = np.mean([r.mean_uncertainty for r in regions])
    max_unc = max(r.max_uncertainty for r in regions)
    
    return {
        "num_regions": len(regions),
        "num_high_conf": high_conf,
        "num_medium_conf": medium_conf,
        "num_low_conf": low_conf,
        "mean_uncertainty": float(mean_unc),
        "max_uncertainty": float(max_unc),
        "high_conf_ratio": high_conf / len(regions) if regions else 1.0,
        "regions": regions
    }
