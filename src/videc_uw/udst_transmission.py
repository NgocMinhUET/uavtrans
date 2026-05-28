"""Uncertainty-Driven Selective Transmission (UDST) module.

This module implements adaptive bit allocation and payload construction
based on region-wise uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Any, Optional, Tuple
import json
import zlib

import cv2
import numpy as np
from PIL import Image

from .uncertainty import (
    ConfidenceClass, 
    UncertaintyConfig, 
    RegionInfo,
    extract_regions_with_uncertainty,
    compute_image_uncertainty_summary
)


@dataclass
class TransmissionConfig:
    """Configuration for UDST transmission."""
    geometry_base_bytes: int = 50
    skeleton_points_per_region: int = 16
    compressed_mask_quality: int = 50
    compressed_mask_side: int = 32
    image_patch_quality: int = 65
    image_patch_max_side: int = 128
    header_bytes: int = 32
    region_table_bytes_per_region: int = 16


@dataclass
class PayloadInfo:
    """Information about a region's payload."""
    region_id: int
    confidence_class: ConfidenceClass
    payload_type: str  # "geometry", "compressed_mask", "image_patch"
    payload_bytes: int
    uncertainty: float


@dataclass
class UDSTPacket:
    """Complete UDST transmission packet."""
    event_id: str
    image_shape: Tuple[int, int]
    num_regions: int
    total_bytes: int
    header_bytes: int
    region_table_bytes: int
    payload_bytes: int
    payloads: list[PayloadInfo]
    
    # Aggregated statistics
    num_high_conf: int
    num_medium_conf: int
    num_low_conf: int
    mean_uncertainty: float
    
    # For reconstruction
    high_conf_geometry: list[dict]
    medium_conf_masks: list[dict]
    low_conf_patches: list[dict]


def _encode_jpeg_bytes(image: np.ndarray, quality: int = 75) -> bytes:
    """Encode image to JPEG bytes."""
    if image.ndim == 2:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def _extract_skeleton_points(mask: np.ndarray, max_points: int = 16) -> list[Tuple[int, int]]:
    """Extract skeleton points from binary mask."""
    from skimage.morphology import skeletonize
    
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    
    skel = skeletonize(binary > 0)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return []
    
    coords = np.column_stack([xs, ys])
    if len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points).astype(int)
        coords = coords[idx]
    
    return [(int(x), int(y)) for x, y in coords]


def _compress_mask_rle(mask: np.ndarray) -> bytes:
    """Compress binary mask using run-length encoding."""
    flat = mask.flatten().astype(np.uint8)
    
    runs = []
    current_val = flat[0]
    count = 1
    
    for val in flat[1:]:
        if val == current_val and count < 255:
            count += 1
        else:
            runs.append((current_val, count))
            current_val = val
            count = 1
    runs.append((current_val, count))
    
    rle_bytes = bytes([v for val, cnt in runs for v in (val, cnt)])
    compressed = zlib.compress(rle_bytes, level=9)
    
    return compressed


def _downsample_mask(mask: np.ndarray, side: int = 32) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Downsample mask to fixed side length."""
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros((side, side), dtype=np.uint8), (w, h)
    
    small = cv2.resize(mask.astype(np.uint8), (side, side), interpolation=cv2.INTER_AREA)
    small = (small > 0.25).astype(np.uint8)
    
    return small, (w, h)


def compute_geometry_payload(
    region: RegionInfo,
    config: TransmissionConfig
) -> Tuple[dict, int]:
    """
    Compute geometry-only payload for HIGH confidence regions.
    
    Returns:
        (geometry_dict, payload_bytes)
    """
    skeleton = _extract_skeleton_points(region.mask, config.skeleton_points_per_region)
    
    geometry = {
        "region_id": region.region_id,
        "bbox_xywh": region.bbox_xywh,
        "skeleton_points": skeleton,
        "area_pixels": region.area_pixels,
        "uncertainty": round(region.mean_uncertainty, 3)
    }
    
    json_str = json.dumps(geometry, separators=(",", ":"))
    payload_bytes = len(zlib.compress(json_str.encode(), level=9))
    
    return geometry, payload_bytes


def compute_compressed_mask_payload(
    region: RegionInfo,
    config: TransmissionConfig
) -> Tuple[dict, int]:
    """
    Compute compressed mask payload for MEDIUM confidence regions.
    
    Returns:
        (mask_dict, payload_bytes)
    """
    x, y, w, h = region.bbox_xywh
    roi_mask = region.mask[y:y+h, x:x+w]
    
    small_mask, orig_size = _downsample_mask(roi_mask, config.compressed_mask_side)
    rle_bytes = _compress_mask_rle(small_mask)
    
    mask_data = {
        "region_id": region.region_id,
        "bbox_xywh": region.bbox_xywh,
        "mask_side": config.compressed_mask_side,
        "original_size": orig_size,
        "uncertainty": round(region.mean_uncertainty, 3),
        "rle_length": len(rle_bytes)
    }
    
    header_bytes = len(json.dumps(mask_data, separators=(",", ":")))
    payload_bytes = header_bytes + len(rle_bytes)
    
    return mask_data, payload_bytes


def compute_image_patch_payload(
    region: RegionInfo,
    image: np.ndarray,
    config: TransmissionConfig
) -> Tuple[dict, int]:
    """
    Compute image patch payload for LOW confidence regions.
    
    Returns:
        (patch_dict, payload_bytes)
    """
    x, y, w, h = region.bbox_xywh
    pad = 8
    h_img, w_img = image.shape[:2]
    
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    
    patch = image[y0:y1, x0:x1]
    
    ph, pw = patch.shape[:2]
    max_side = config.image_patch_max_side
    if max(ph, pw) > max_side:
        scale = max_side / max(ph, pw)
        new_w = max(1, int(pw * scale))
        new_h = max(1, int(ph * scale))
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    jpeg_bytes = _encode_jpeg_bytes(patch, config.image_patch_quality)
    
    patch_data = {
        "region_id": region.region_id,
        "bbox_xywh": region.bbox_xywh,
        "patch_bbox": (x0, y0, x1 - x0, y1 - y0),
        "patch_shape": patch.shape[:2],
        "uncertainty": round(region.mean_uncertainty, 3),
        "jpeg_length": len(jpeg_bytes)
    }
    
    header_bytes = len(json.dumps(patch_data, separators=(",", ":")))
    payload_bytes = header_bytes + len(jpeg_bytes)
    
    return patch_data, payload_bytes


def construct_udst_packet(
    image: np.ndarray,
    mask: np.ndarray,
    event_id: str,
    prob_map: Optional[np.ndarray] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    transmission_config: Optional[TransmissionConfig] = None
) -> UDSTPacket:
    """
    Construct a complete UDST packet with adaptive bit allocation.
    
    Args:
        image: Input image
        mask: Segmentation mask
        event_id: Unique event identifier
        prob_map: Optional probability map (simulated if None)
        uncertainty_config: Uncertainty estimation config
        transmission_config: Transmission config
    
    Returns:
        UDSTPacket with all payload information
    """
    if uncertainty_config is None:
        uncertainty_config = UncertaintyConfig()
    if transmission_config is None:
        transmission_config = TransmissionConfig()
    
    summary = compute_image_uncertainty_summary(mask, prob_map, uncertainty_config)
    regions = summary["regions"]
    
    payloads = []
    high_conf_geometry = []
    medium_conf_masks = []
    low_conf_patches = []
    
    total_payload_bytes = 0
    
    for region in regions:
        if region.confidence_class == ConfidenceClass.HIGH:
            geo_data, geo_bytes = compute_geometry_payload(region, transmission_config)
            payloads.append(PayloadInfo(
                region_id=region.region_id,
                confidence_class=region.confidence_class,
                payload_type="geometry",
                payload_bytes=geo_bytes,
                uncertainty=region.mean_uncertainty
            ))
            high_conf_geometry.append(geo_data)
            total_payload_bytes += geo_bytes
            
        elif region.confidence_class == ConfidenceClass.MEDIUM:
            mask_data, mask_bytes = compute_compressed_mask_payload(region, transmission_config)
            payloads.append(PayloadInfo(
                region_id=region.region_id,
                confidence_class=region.confidence_class,
                payload_type="compressed_mask",
                payload_bytes=mask_bytes,
                uncertainty=region.mean_uncertainty
            ))
            medium_conf_masks.append(mask_data)
            total_payload_bytes += mask_bytes
            
        else:  # LOW confidence
            patch_data, patch_bytes = compute_image_patch_payload(
                region, image, transmission_config
            )
            payloads.append(PayloadInfo(
                region_id=region.region_id,
                confidence_class=region.confidence_class,
                payload_type="image_patch",
                payload_bytes=patch_bytes,
                uncertainty=region.mean_uncertainty
            ))
            low_conf_patches.append(patch_data)
            total_payload_bytes += patch_bytes
    
    header_bytes = transmission_config.header_bytes
    region_table_bytes = len(regions) * transmission_config.region_table_bytes_per_region
    total_bytes = header_bytes + region_table_bytes + total_payload_bytes
    
    return UDSTPacket(
        event_id=event_id,
        image_shape=image.shape[:2],
        num_regions=len(regions),
        total_bytes=total_bytes,
        header_bytes=header_bytes,
        region_table_bytes=region_table_bytes,
        payload_bytes=total_payload_bytes,
        payloads=payloads,
        num_high_conf=summary["num_high_conf"],
        num_medium_conf=summary["num_medium_conf"],
        num_low_conf=summary["num_low_conf"],
        mean_uncertainty=summary["mean_uncertainty"],
        high_conf_geometry=high_conf_geometry,
        medium_conf_masks=medium_conf_masks,
        low_conf_patches=low_conf_patches
    )


def compute_baseline_payloads(
    image: np.ndarray,
    mask: np.ndarray,
    jpeg_quality: int = 75
) -> dict:
    """
    Compute baseline transmission payloads for comparison.
    
    Returns:
        Dictionary with payload sizes for various baselines
    """
    full_jpeg_bytes = len(_encode_jpeg_bytes(image, jpeg_quality))
    
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad = 8
        h, w = image.shape[:2]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)
        roi = image[y0:y1, x0:x1]
        roi_jpeg_bytes = len(_encode_jpeg_bytes(roi, jpeg_quality))
    else:
        roi_jpeg_bytes = full_jpeg_bytes
    
    mask_png_bytes = len(cv2.imencode('.png', mask.astype(np.uint8) * 255)[1])
    mask_rle_bytes = len(_compress_mask_rle(mask))
    
    return {
        "full_jpeg_bytes": full_jpeg_bytes,
        "roi_jpeg_bytes": roi_jpeg_bytes,
        "mask_png_bytes": mask_png_bytes,
        "mask_rle_bytes": mask_rle_bytes
    }


def udst_packet_to_dict(packet: UDSTPacket) -> dict:
    """Convert UDST packet to dictionary for serialization."""
    return {
        "event_id": packet.event_id,
        "image_shape": packet.image_shape,
        "num_regions": packet.num_regions,
        "total_bytes": packet.total_bytes,
        "header_bytes": packet.header_bytes,
        "region_table_bytes": packet.region_table_bytes,
        "payload_bytes": packet.payload_bytes,
        "num_high_conf": packet.num_high_conf,
        "num_medium_conf": packet.num_medium_conf,
        "num_low_conf": packet.num_low_conf,
        "mean_uncertainty": packet.mean_uncertainty,
        "payloads": [
            {
                "region_id": p.region_id,
                "confidence_class": p.confidence_class.name,
                "payload_type": p.payload_type,
                "payload_bytes": p.payload_bytes,
                "uncertainty": p.uncertainty
            }
            for p in packet.payloads
        ]
    }
