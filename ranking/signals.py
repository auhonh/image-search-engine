"""Cheap visual signals used to re-score ANN candidates.

These deliberately capture things a global CNN embedding underweights:
exact color palette, framing/aspect, and overall exposure.
"""

from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=4096)
def color_histogram(image_path: str, bins: int = 32) -> np.ndarray:
    """Hue-Saturation histogram, L1-normalized, flattened to (bins*bins,)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    # Smooth so near-identical colors in adjacent bins still overlap
    # (hard binning makes intersection brittle, esp. for flat color regions)
    hist = cv2.GaussianBlur(hist, (5, 5), 0)
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


def color_similarity(path_a: str, path_b: str, bins: int = 32) -> float:
    """Histogram intersection in [0, 1]. 1 = identical palettes."""
    ha, hb = color_histogram(path_a, bins), color_histogram(path_b, bins)
    return float(np.minimum(ha, hb).sum())


@lru_cache(maxsize=4096)
def _shape_stats(image_path: str) -> tuple[float, float]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    aspect = w / h
    brightness = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) / 255.0
    return aspect, brightness


def aspect_similarity(path_a: str, path_b: str) -> float:
    """1 when aspect ratios match, decaying toward 0 as they diverge."""
    a, _ = _shape_stats(path_a)
    b, _ = _shape_stats(path_b)
    return float(min(a, b) / max(a, b))


def brightness_similarity(path_a: str, path_b: str) -> float:
    _, a = _shape_stats(path_a)
    _, b = _shape_stats(path_b)
    return float(1.0 - abs(a - b))
