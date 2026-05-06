"""HOLOSUN-style outline rendering pipeline on 14-bit thermal data.

Pipeline:
  raw14 -> denoise -> Sobel5x5 + Scharr mixed gradients -> magnitude/direction
  -> non-maximum suppression -> hysteresis edge linking -> short-gap bridging
  -> pure high-frequency outline image (low-frequency forced to zero)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import thermal_processing


@dataclass
class OutlineConfig:
    level: int = 3
    gaussian_ksize: int = 5
    sobel_weight: float = 0.58
    scharr_weight: float = 0.42
    high_percentile: float = 92.0
    low_ratio: float = 0.44
    bridge_strength_ratio: float = 0.55
    bridge_max_gap: int = 2
    glow_gain: float = 0.22
    glow_sigma: float = 0.9


def render_outline(raw14: np.ndarray, cfg: OutlineConfig) -> np.ndarray:
    """Return uint8 pure-outline frame: low-frequency zero + enhanced edges."""
    raw14 = thermal_processing.to_raw14(raw14)
    img = raw14.astype(np.float32)

    k = max(3, int(cfg.gaussian_ksize) | 1)
    denoised = cv2.GaussianBlur(img, (k, k), 0)

    sobel_gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=5)
    sobel_gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=5)
    scharr_gx = cv2.Scharr(denoised, cv2.CV_32F, 1, 0)
    scharr_gy = cv2.Scharr(denoised, cv2.CV_32F, 0, 1)

    sobel_gx, sobel_gy = _normalize_grad_pair(sobel_gx, sobel_gy)
    scharr_gx, scharr_gy = _normalize_grad_pair(scharr_gx, scharr_gy)

    level = int(np.clip(cfg.level, 1, 5))
    high_pct = cfg.high_percentile + (level - 3) * 1.2
    low_ratio = np.clip(cfg.low_ratio - (level - 3) * 0.03, 0.28, 0.60)
    glow_gain = np.clip(cfg.glow_gain + (level - 3) * 0.03, 0.08, 0.35)

    gx = cfg.sobel_weight * sobel_gx + cfg.scharr_weight * scharr_gx
    gy = cfg.sobel_weight * sobel_gy + cfg.scharr_weight * scharr_gy
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)

    nms = _non_max_suppression(mag, theta)
    edges, high_thr = _hysteresis_connect(nms, high_pct=high_pct, low_ratio=low_ratio)
    edges = _bridge_small_gaps(
        edges, theta, nms, high_thr, strength_ratio=cfg.bridge_strength_ratio, max_gap=cfg.bridge_max_gap
    )
    return _to_outline_image(edges, glow_gain=glow_gain, glow_sigma=cfg.glow_sigma)


def _normalize_grad_pair(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.hypot(gx, gy)
    scale = float(np.percentile(mag, 99.5)) + 1e-6
    return gx / scale, gy / scale


def _quantize_dirs(theta: np.ndarray) -> np.ndarray:
    ang = (np.rad2deg(theta) + 180.0) % 180.0
    dirs = np.zeros_like(ang, dtype=np.uint8)
    dirs[((ang >= 22.5) & (ang < 67.5))] = 1     # 45
    dirs[((ang >= 67.5) & (ang < 112.5))] = 2    # 90
    dirs[((ang >= 112.5) & (ang < 157.5))] = 3   # 135
    return dirs


def _non_max_suppression(mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    dirs = _quantize_dirs(theta)
    left = np.roll(mag, 1, axis=1)
    right = np.roll(mag, -1, axis=1)
    up = np.roll(mag, -1, axis=0)
    down = np.roll(mag, 1, axis=0)
    up_right = np.roll(up, -1, axis=1)
    down_left = np.roll(down, 1, axis=1)
    up_left = np.roll(up, 1, axis=1)
    down_right = np.roll(down, -1, axis=1)

    keep = np.zeros_like(mag, dtype=bool)
    keep |= (dirs == 0) & (mag >= left) & (mag >= right)
    keep |= (dirs == 1) & (mag >= up_right) & (mag >= down_left)
    keep |= (dirs == 2) & (mag >= up) & (mag >= down)
    keep |= (dirs == 3) & (mag >= up_left) & (mag >= down_right)

    out = np.where(keep, mag, 0.0).astype(np.float32)
    out[0, :] = 0.0
    out[-1, :] = 0.0
    out[:, 0] = 0.0
    out[:, -1] = 0.0
    return out


def _hysteresis_connect(nms: np.ndarray, high_pct: float, low_ratio: float) -> tuple[np.ndarray, float]:
    pos = nms[nms > 0]
    if pos.size == 0:
        return np.zeros_like(nms, dtype=np.uint8), 0.0

    high_thr = float(np.percentile(pos, np.clip(high_pct, 75.0, 99.8)))
    low_thr = float(high_thr * np.clip(low_ratio, 0.2, 0.8))
    strong = nms >= high_thr
    weak = nms >= low_thr

    num, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if num <= 1:
        return strong.astype(np.uint8), high_thr

    keep_labels = np.unique(labels[strong])
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(nms, dtype=np.uint8), high_thr
    edges = np.isin(labels, keep_labels)
    return edges.astype(np.uint8), high_thr


def _bridge_small_gaps(
    edges: np.ndarray,
    theta: np.ndarray,
    strength: np.ndarray,
    high_thr: float,
    strength_ratio: float,
    max_gap: int = 2,
) -> np.ndarray:
    if max_gap < 1:
        return edges

    h, w = edges.shape[:2]
    dirs = _quantize_dirs(theta)
    # gradient direction -> edge tangent direction
    tangent = {
        0: (0, 1),
        1: (1, 1),
        2: (1, 0),
        3: (1, -1),
    }
    out = edges.copy().astype(np.uint8)
    min_strength = float(high_thr * np.clip(strength_ratio, 0.2, 1.0))

    ys, xs = np.where(out > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if strength[y, x] < min_strength:
            continue
        tdir = int(dirs[y, x])
        dy, dx = tangent[tdir]
        for sign in (-1, 1):
            sy, sx = dy * sign, dx * sign
            for gap in range(1, max_gap + 1):
                y2 = y + sy * (gap + 1)
                x2 = x + sx * (gap + 1)
                if y2 < 0 or x2 < 0 or y2 >= h or x2 >= w:
                    break
                if out[y2, x2] == 0:
                    continue
                if strength[y2, x2] < min_strength:
                    continue
                if abs(int(dirs[y2, x2]) - tdir) > 1 and abs(int(dirs[y2, x2]) - tdir) < 3:
                    continue
                clear = True
                for s in range(1, gap + 1):
                    yi = y + sy * s
                    xi = x + sx * s
                    if out[yi, xi] > 0:
                        clear = False
                        break
                if not clear:
                    continue
                for s in range(1, gap + 1):
                    yi = y + sy * s
                    xi = x + sx * s
                    out[yi, xi] = 1
                break
    return out


def _to_outline_image(edges: np.ndarray, glow_gain: float, glow_sigma: float) -> np.ndarray:
    core = (edges > 0).astype(np.uint8) * 255
    if glow_gain <= 0.0:
        return core
    blur = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sigmaX=max(glow_sigma, 0.1))
    glow = blur * float(glow_gain)
    out = np.maximum(core.astype(np.float32), glow)
    return np.clip(out, 0, 255).astype(np.uint8)

