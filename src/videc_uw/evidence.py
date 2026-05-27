"""Evidence object construction for ViDEC-UW.

The module converts an image and a defect mask into a structured Defect
Evidence Object (DEO): E={H,G,M,V,U,R}.  The implementation is intentionally
lightweight so the same code can run on a laptop or an edge device.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import time
import zlib

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


Point = Tuple[int, int]


@dataclass
class EvidenceSizes:
    l0_bytes: int
    l1_bytes: int
    l2_bytes: int
    l3_bytes: int
    roi_jpeg_bytes: int
    full_jpeg_bytes: int


@dataclass
class EvidenceUtilities:
    utility_l0: float
    utility_l1: float
    utility_l2: float
    utility_l3: float
    utility_roi: float
    utility_full: float


def _json_size(obj: Dict[str, Any], compressed: bool = True) -> int:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return len(zlib.compress(payload, level=9)) if compressed else len(payload)


def _encode_jpeg_bytes(image: np.ndarray, quality: int = 75) -> bytes:
    if image.ndim == 2:
        pil_img = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def _safe_crop(image: np.ndarray, bbox_xywh: Tuple[int, int, int, int], pad: int = 8) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox_xywh
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return image
    return crop


def _resized_jpeg_size(image: np.ndarray, side: int = 128, quality: int = 50) -> Tuple[int, Tuple[int, int]]:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return 0, (0, 0)
    scale = side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return len(_encode_jpeg_bytes(resized, quality=quality)), (new_w, new_h)


def _mask_to_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    lab = label(binary)
    if lab.max() == 0:
        return binary
    props = regionprops(lab)
    largest = max(props, key=lambda r: r.area)
    return (lab == largest.label).astype(np.uint8)


def _contour_points(mask: np.ndarray, max_points: int = 64) -> List[Point]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) > max_points:
        idx = np.linspace(0, len(contour) - 1, max_points).astype(int)
        contour = contour[idx]
    return [(int(x), int(y)) for x, y in contour]


def _skeleton_points(mask: np.ndarray, max_points: int = 64) -> List[Point]:
    skel = skeletonize(mask > 0)
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return []
    coords = np.column_stack([xs, ys])
    if len(coords) > max_points:
        idx = np.linspace(0, len(coords) - 1, max_points).astype(int)
        coords = coords[idx]
    return [(int(x), int(y)) for x, y in coords]


def _endpoints_and_branches(mask: np.ndarray, max_points: int = 24) -> Tuple[List[Point], List[Point]]:
    skel = skeletonize(mask > 0).astype(np.uint8)
    if skel.sum() == 0:
        return [], []
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skel, -1, kernel)
    endpoints_yx = np.argwhere(conv == 11)
    branches_yx = np.argwhere(conv >= 13)
    endpoints = [(int(x), int(y)) for y, x in endpoints_yx[:max_points]]
    branches = [(int(x), int(y)) for y, x in branches_yx[:max_points]]
    return endpoints, branches


def _estimate_width(mask: np.ndarray) -> Tuple[float, float]:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0, 0.0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    skel = skeletonize(binary > 0)
    width_values = 2.0 * dist[skel]
    if width_values.size == 0:
        return 0.0, 0.0
    return float(np.mean(width_values)), float(np.max(width_values))


def _normalised_error(est: float, ref: float) -> float:
    if ref <= 1e-9:
        return 0.0
    return float(min(abs(est - ref) / ref, 1.0))


def construct_deo(
    image: np.ndarray,
    mask: np.ndarray,
    event_id: str,
    sensor: str = "camera",
    depth_m: Optional[float] = None,
    pose: Optional[Iterable[float]] = None,
    confidence: Optional[float] = None,
    pixel_scale_mm: Optional[float] = None,
    jpeg_quality: int = 75,
    thumbnail_side: int = 128,
) -> Dict[str, Any]:
    """Build a structured DEO from an image and a binary defect mask."""
    mask_largest = _mask_to_largest_component(mask)
    ys, xs = np.where(mask_largest > 0)
    h, w = image.shape[:2]
    if len(xs) == 0:
        bbox = (0, 0, w, h)
        area = 0
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        area = int(mask_largest.sum())

    contour = _contour_points(mask_largest)
    skeleton = _skeleton_points(mask_largest)
    endpoints, branches = _endpoints_and_branches(mask_largest)
    mean_width, max_width = _estimate_width(mask_largest)
    length_px = float(len(np.where(skeletonize(mask_largest > 0))[0]))
    conf = float(confidence) if confidence is not None else min(0.98, max(0.05, area / max(1, h * w) * 20.0 + 0.55))
    ambiguity = float(1.0 - conf)

    roi = _safe_crop(image, bbox)
    thumbnail_bytes, thumbnail_shape = _resized_jpeg_size(roi, side=thumbnail_side, quality=50)
    full_jpeg_bytes = len(_encode_jpeg_bytes(image, quality=jpeg_quality))
    roi_jpeg_bytes = len(_encode_jpeg_bytes(roi, quality=jpeg_quality))

    header = {
        "event_id": event_id,
        "timestamp": time.time(),
        "sensor": sensor,
        "image_shape": [int(h), int(w)],
        "depth_m": depth_m,
        "pose": list(pose) if pose is not None else None,
        "pixel_scale_mm": pixel_scale_mm,
    }
    geometry = {
        "bbox_xywh": [int(v) for v in bbox],
        "contour": contour,
        "skeleton": skeleton,
        "endpoints": endpoints,
        "branch_points": branches,
    }
    metrology = {
        "length_px": length_px,
        "mean_width_px": mean_width,
        "max_width_px": max_width,
        "area_px": area,
        "length_mm": length_px * pixel_scale_mm if pixel_scale_mm else None,
        "severity": "severe" if area > 0.02 * h * w else "moderate" if area > 0.005 * h * w else "minor",
    }
    visual_support = {
        "type": "roi_thumbnail_jpeg",
        "thumbnail_side": thumbnail_side,
        "encoded_shape": list(thumbnail_shape),
        "bytes": thumbnail_bytes,
    }
    uncertainty = {
        "confidence": conf,
        "ambiguity": ambiguity,
        "source": "calibrated_detector_or_proxy",
    }
    residual = {
        "type": "selective_mask_residual",
        "description": "compressed unresolved/uncertain mask support",
        "bytes": int(max(64, 0.25 * roi_jpeg_bytes * ambiguity)),
    }

    l0_obj = {"H": {"event_id": event_id, "sensor": sensor}, "alert": True}
    l1_obj = {"H": header, "G": geometry, "M": metrology, "U": uncertainty}
    l0 = _json_size(l0_obj)
    l1 = _json_size(l1_obj)
    l2 = l1 + thumbnail_bytes
    l3 = l2 + residual["bytes"]

    # Utility proxy: enough for channel experiments; can be replaced by measured
    # receiver verification metrics when labels are available.
    area_norm = min(1.0, area / max(1, 0.03 * h * w))
    shape_bonus = 0.15 if len(skeleton) > 3 else 0.0
    utility_l0 = min(0.55, 0.35 + 0.20 * conf)
    utility_l1 = min(0.82, 0.45 + 0.25 * conf + 0.10 * area_norm + shape_bonus)
    utility_l2 = min(0.93, utility_l1 + 0.10 + 0.12 * ambiguity)
    utility_l3 = min(0.98, utility_l2 + 0.05 + 0.10 * ambiguity)

    return {
        "H": header,
        "G": geometry,
        "M": metrology,
        "V": visual_support,
        "U": uncertainty,
        "R": residual,
        "sizes": asdict(EvidenceSizes(l0, l1, l2, l3, roi_jpeg_bytes, full_jpeg_bytes)),
        "utilities": asdict(EvidenceUtilities(utility_l0, utility_l1, utility_l2, utility_l3, min(0.97, utility_l2 + 0.05), 1.0)),
    }


def load_image_mask(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    mask = (mask > 127).astype(np.uint8)
    return image, mask
