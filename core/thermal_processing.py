"""Pure-functional thermal frame processing helpers.

The project now uses an internal 14-bit grayscale representation (`raw14`) for
all thermal calculations. Display-oriented steps can map raw14 to uint8.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


RAW14_MAX = 16383


@dataclass
class ProcessConfig:
    auto_normalize: bool = True
    clahe_enabled: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    denoise_enabled: bool = False
    denoise_ksize: int = 3
    sharpen_enabled: bool = False
    sharpen_amount: float = 0.5


_clahe_cache: dict[tuple[float, int], cv2.CLAHE] = {}


def to_raw14(frame: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary frame to uint16 raw14 range [0, 16383]."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if gray.dtype == np.uint16:
        # Thermal Y16 streams may use values above 14-bit; squeeze robustly.
        max_v = int(gray.max()) if gray.size else 0
        if max_v <= RAW14_MAX:
            return gray.copy()
        scale = RAW14_MAX / float(max(max_v, 1))
        return np.clip(gray.astype(np.float32) * scale, 0, RAW14_MAX).astype(np.uint16)

    if gray.dtype == np.uint8:
        return (gray.astype(np.uint16) << 6)  # 8-bit -> 14-bit linear lift

    g = cv2.normalize(gray, None, 0, RAW14_MAX, cv2.NORM_MINMAX)
    return g.astype(np.uint16)


def raw14_to_u8(raw14: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Map raw14 to uint8 with percentile clipping (display-oriented AGC)."""
    if raw14.ndim == 3:
        raw14 = cv2.cvtColor(raw14, cv2.COLOR_BGR2GRAY)
    if raw14.dtype != np.uint16:
        raw14 = to_raw14(raw14)

    lo = float(np.percentile(raw14, p_low))
    hi = float(np.percentile(raw14, p_high))
    if hi <= lo + 1.0:
        lo = float(raw14.min())
        hi = float(raw14.max())
    if hi <= lo:
        return np.zeros_like(raw14, dtype=np.uint8)
    stretched = (raw14.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Legacy helper: return uint8 grayscale from arbitrary input."""
    return raw14_to_u8(to_raw14(frame))


def normalize_minmax(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """Stretch contrast to fill 0..255 and report raw stats from the stretched image."""
    g_min = int(gray.min())
    g_max = int(gray.max())
    h, w = gray.shape
    center_val = int(gray[h // 2, w // 2])
    if g_max > g_min:
        out = ((gray.astype(np.int32) - g_min) * 255 // (g_max - g_min)).astype(np.uint8)
    else:
        out = gray.copy()
    hotspot_val = int(out.max())
    return out, {
        "gray_min": g_min,
        "gray_max": g_max,
        "gray_center": center_val,
        "gray_hotspot": hotspot_val,
    }


def _get_clahe(clip: float, tile: int) -> cv2.CLAHE:
    key = (round(clip, 2), int(tile))
    obj = _clahe_cache.get(key)
    if obj is None:
        obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        _clahe_cache[key] = obj
    return obj


def apply_clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    return _get_clahe(clip, tile).apply(gray)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = max(1, ksize | 1)  # ensure odd
    return cv2.GaussianBlur(gray, (k, k), 0)


def sharpen(gray: np.ndarray, amount: float = 0.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return sharp


def run_pipeline(frame_or_gray: np.ndarray, cfg: ProcessConfig) -> tuple[np.ndarray, dict]:
    """Full enhancement chain. Returns (processed_gray, stats_dict)."""
    gray = to_gray(frame_or_gray)
    raw_stats = {
        "gray_min": int(gray.min()),
        "gray_max": int(gray.max()),
        "gray_center": int(gray[gray.shape[0] // 2, gray.shape[1] // 2]),
        "gray_hotspot": int(gray.max()),
    }
    if cfg.auto_normalize:
        gray, _ = normalize_minmax(gray)
    if cfg.denoise_enabled:
        gray = denoise(gray, cfg.denoise_ksize)
    if cfg.clahe_enabled:
        gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_tile)
    if cfg.sharpen_enabled:
        gray = sharpen(gray, cfg.sharpen_amount)
    return gray, raw_stats
