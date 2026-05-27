"""Underwater-like image degradation utilities.

This module does not claim to be a full optical underwater image formation
model. It provides a reproducible controlled degradation layer so that the
communication experiment can be evaluated on real structural defect images
under underwater-like visibility conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class DegradationConfig:
    haze: float = 0.35
    blur_sigma: float = 1.1
    noise_sigma: float = 6.0
    blue_green_shift: float = 0.18
    brightness: float = 0.88
    contrast: float = 0.82
    seed: int = 123


def apply_underwater_degradation(image_bgr: np.ndarray, cfg: DegradationConfig) -> np.ndarray:
    """Apply a deterministic underwater-like degradation to a BGR image.

    The degradation includes contrast reduction, blue/green color shift,
    depth-like haze, mild blur, and additive noise. It is designed for
    controlled experiments rather than photorealistic rendering.
    """
    rng = np.random.default_rng(cfg.seed)
    img = image_bgr.astype(np.float32) / 255.0

    # Lower contrast and brightness.
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * cfg.contrast + mean
    img = img * cfg.brightness

    # BGR color bias: suppress red, emphasize blue/green.
    img[..., 0] = np.clip(img[..., 0] * (1.0 + cfg.blue_green_shift), 0, 1)
    img[..., 1] = np.clip(img[..., 1] * (1.0 + 0.5 * cfg.blue_green_shift), 0, 1)
    img[..., 2] = np.clip(img[..., 2] * (1.0 - cfg.blue_green_shift), 0, 1)

    # Haze veiling light, stronger at lower image rows as a simple proxy.
    h, w = img.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    veiling = np.array([0.78, 0.90, 0.72], dtype=np.float32)[None, None, :]
    haze_map = cfg.haze * (0.4 + 0.6 * y)
    img = img * (1.0 - haze_map) + veiling * haze_map

    # Blur and sensor noise.
    if cfg.blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), cfg.blur_sigma)
    if cfg.noise_sigma > 0:
        noise = rng.normal(0, cfg.noise_sigma / 255.0, img.shape).astype(np.float32)
        img = img + noise

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)
