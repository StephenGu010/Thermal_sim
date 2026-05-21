"""Dual-profile outline rendering for thermal and visible-light inputs.

Thermal profile:
  raw14 -> bad-pixel suppression -> temporal EMA -> light denoise
  -> warm-target gate -> Sobel/Scharr gradients -> NMS -> hysteresis
  -> density cap -> strength-weighted outline.

Visible profile:
  BGR/gray -> bilateral denoise -> mild contrast stretch -> temporal EMA
  -> Sobel/Scharr + Canny-like linking -> component cleanup -> density cap.

The visible profile is only a black-background edge demo. It must not be
described as thermal enhancement because it has no temperature contrast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import thermal_processing


PROFILE_THERMAL = "thermal"
PROFILE_VISIBLE = "visible"
ALL_PROFILES = (PROFILE_THERMAL, PROFILE_VISIBLE)


@dataclass
class OutlineConfig:
    level: int = 3
    profile: str = PROFILE_THERMAL
    gaussian_ksize: int = 5
    sobel_weight: float = 0.58
    scharr_weight: float = 0.42
    high_percentile: float = 92.0
    low_ratio: float = 0.44
    bridge_strength_ratio: float = 0.55
    bridge_max_gap: int = 2
    glow_gain: float = 0.22
    glow_sigma: float = 0.9
    temporal_alpha: float = 0.58
    thermal_gate_percentile: float = 74.0
    thermal_gate_dilate: int = 7
    thermal_edge_density: float = 0.045
    visible_edge_density: float = 0.024
    visible_min_component_area: int = 8
    visible_bilateral_d: int = 5


class OutlineProcessor:
    """Stateful outline processor that owns temporal smoothing buffers."""

    def __init__(self) -> None:
        self._ema: Optional[np.ndarray] = None
        self._ema_profile: Optional[str] = None

    def reset(self) -> None:
        self._ema = None
        self._ema_profile = None

    def render(
        self,
        raw14: np.ndarray,
        cfg: OutlineConfig,
        visible_frame: Optional[np.ndarray] = None,
        update_temporal: bool = True,
    ) -> np.ndarray:
        profile = cfg.profile if cfg.profile in ALL_PROFILES else PROFILE_THERMAL
        if profile == PROFILE_VISIBLE:
            return self._render_visible(raw14, cfg, visible_frame, update_temporal)
        return self._render_thermal(raw14, cfg, update_temporal)

    def _render_thermal(
        self,
        raw14: np.ndarray,
        cfg: OutlineConfig,
        update_temporal: bool,
    ) -> np.ndarray:
        raw14 = thermal_processing.to_raw14(raw14)
        cleaned = _suppress_bad_pixels(raw14).astype(np.float32)
        smoothed = self._apply_temporal(cleaned, PROFILE_THERMAL, cfg.temporal_alpha, update_temporal)

        k = max(3, int(cfg.gaussian_ksize) | 1)
        denoised = cv2.GaussianBlur(smoothed, (k, k), 0)
        gate = _thermal_target_gate(smoothed, cfg)

        nms, theta = _gradient_nms(denoised, cfg)
        # Background edges are not removed completely; they are attenuated so
        # a very strong thermal contrast can still survive outside the gate.
        nms = nms * np.where(gate > 0, 1.0, 0.18).astype(np.float32)

        level = _level(cfg.level)
        high_pct = np.clip(cfg.high_percentile - (level - 3) * 1.4, 78.0, 98.5)
        low_ratio = np.clip(cfg.low_ratio - (level - 3) * 0.03, 0.28, 0.62)
        glow_gain = np.clip(cfg.glow_gain + (level - 3) * 0.03, 0.08, 0.36)

        edges, high_thr = _hysteresis_connect(nms, high_pct=high_pct, low_ratio=low_ratio)
        if high_thr > 0.0:
            edges = np.where(gate > 0, edges, edges & (nms >= high_thr * 1.35)).astype(np.uint8)
        edges = _bridge_small_gaps(
            edges,
            theta,
            nms,
            high_thr,
            strength_ratio=cfg.bridge_strength_ratio,
            max_gap=cfg.bridge_max_gap,
        )
        edges = _cap_edge_density(edges, nms, _density_for_level(cfg.thermal_edge_density, level))
        return _to_strength_outline(edges, nms, high_thr, glow_gain=glow_gain, glow_sigma=cfg.glow_sigma)

    def _render_visible(
        self,
        raw14: np.ndarray,
        cfg: OutlineConfig,
        visible_frame: Optional[np.ndarray],
        update_temporal: bool,
    ) -> np.ndarray:
        gray = _visible_gray(raw14, visible_frame)
        gray = _contrast_stretch_u8(gray, p_low=1.0, p_high=99.0)

        d = max(3, int(cfg.visible_bilateral_d) | 1)
        denoised = cv2.bilateralFilter(gray, d=d, sigmaColor=34, sigmaSpace=5)
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        smoothed = self._apply_temporal(
            denoised.astype(np.float32),
            PROFILE_VISIBLE,
            np.clip(cfg.temporal_alpha + 0.10, 0.50, 0.86),
            update_temporal,
        )

        nms, theta = _gradient_nms(smoothed, cfg)
        level = _level(cfg.level)
        high_pct = np.clip(93.0 - (level - 3) * 1.1, 86.0, 98.5)
        low_ratio = np.clip(0.50 - (level - 3) * 0.03, 0.34, 0.64)
        edges, high_thr = _hysteresis_connect(nms, high_pct=high_pct, low_ratio=low_ratio)

        canny = _auto_canny(smoothed.astype(np.uint8), high_pct=high_pct, low_ratio=low_ratio)
        if canny.any():
            edges = ((edges > 0) & ((canny > 0) | (nms >= high_thr * 1.15))).astype(np.uint8)

        edges = _clean_small_components(edges, min_area=max(1, int(cfg.visible_min_component_area)))
        edges = _bridge_small_gaps(
            edges,
            theta,
            nms,
            high_thr,
            strength_ratio=0.70,
            max_gap=max(1, cfg.bridge_max_gap),
        )
        edges = _cap_edge_density(edges, nms, _density_for_level(cfg.visible_edge_density, level))
        glow_gain = np.clip(0.16 + (level - 3) * 0.025, 0.08, 0.26)
        return _to_strength_outline(edges, nms, high_thr, glow_gain=glow_gain, glow_sigma=0.75)

    def _apply_temporal(
        self,
        img: np.ndarray,
        profile: str,
        alpha: float,
        update_temporal: bool,
    ) -> np.ndarray:
        img = img.astype(np.float32, copy=False)
        if (
            self._ema is None
            or self._ema_profile != profile
            or self._ema.shape != img.shape
            or not update_temporal
        ):
            if update_temporal:
                self._ema = img.copy()
                self._ema_profile = profile
            return img

        a = float(np.clip(alpha, 0.05, 0.95))
        self._ema = cv2.addWeighted(img, a, self._ema, 1.0 - a, 0.0)
        return self._ema.copy()


_DEFAULT_PROCESSOR = OutlineProcessor()


def render_outline(
    raw14: np.ndarray,
    cfg: OutlineConfig,
    visible_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compatibility wrapper around a module-level stateful processor."""
    return _DEFAULT_PROCESSOR.render(raw14, cfg, visible_frame=visible_frame)


def reset_temporal_state() -> None:
    _DEFAULT_PROCESSOR.reset()


def _level(level: int) -> int:
    return int(np.clip(level, 1, 5))


def _density_for_level(base: float, level: int) -> float:
    # ENH 1 should be sparse and stable; ENH 5 intentionally exposes more weak
    # texture for tuning and demos.
    return float(np.clip(base * (0.66 + 0.18 * (level - 1)), 0.004, 0.085))


def _visible_gray(raw14: np.ndarray, visible_frame: Optional[np.ndarray]) -> np.ndarray:
    if visible_frame is None:
        return thermal_processing.raw14_to_u8(raw14, p_low=1.0, p_high=99.0)
    frame = visible_frame
    if frame.ndim == 3:
        if frame.dtype == np.uint8 and frame.shape[2] == 2:
            h, w = frame.shape[:2]
            packed = frame.view(np.uint16).reshape(h, w)
            return thermal_processing.raw14_to_u8(packed, p_low=1.0, p_high=99.0)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.dtype == np.uint8:
        return frame.copy()
    if frame.dtype == np.uint16:
        return thermal_processing.raw14_to_u8(frame, p_low=1.0, p_high=99.0)
    return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _suppress_bad_pixels(raw14: np.ndarray) -> np.ndarray:
    if raw14.size == 0:
        return raw14
    med = cv2.medianBlur(raw14, 3)
    diff = np.abs(raw14.astype(np.int32) - med.astype(np.int32))
    threshold = max(96.0, float(np.percentile(diff, 99.6)))
    return np.where(diff > threshold, med, raw14).astype(np.uint16)


def _contrast_stretch_u8(gray: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    lo = float(np.percentile(gray, p_low))
    hi = float(np.percentile(gray, p_high))
    if hi <= lo + 1.0:
        return gray.astype(np.uint8, copy=True)
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _thermal_target_gate(img: np.ndarray, cfg: OutlineConfig) -> np.ndarray:
    raw = np.clip(img, 0, thermal_processing.RAW14_MAX).astype(np.uint16)
    u8 = thermal_processing.raw14_to_u8(raw, p_low=3.0, p_high=99.2)
    level = _level(cfg.level)
    gate_pct = np.clip(cfg.thermal_gate_percentile - (level - 3) * 2.0, 60.0, 88.0)
    thr = float(np.percentile(u8, gate_pct))
    mask = (u8 >= thr).astype(np.uint8)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = _clean_small_components(mask, min_area=max(8, int(mask.size * 0.0015)))

    k = max(3, int(cfg.thermal_gate_dilate) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    if not mask.any():
        return np.ones_like(u8, dtype=np.uint8)
    return mask.astype(np.uint8)


def _gradient_nms(img: np.ndarray, cfg: OutlineConfig) -> tuple[np.ndarray, np.ndarray]:
    img = img.astype(np.float32, copy=False)
    sobel_gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    sobel_gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    scharr_gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    scharr_gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)

    sobel_gx, sobel_gy = _normalize_grad_pair(sobel_gx, sobel_gy)
    scharr_gx, scharr_gy = _normalize_grad_pair(scharr_gx, scharr_gy)

    total = max(1e-6, float(cfg.sobel_weight + cfg.scharr_weight))
    sw = float(cfg.sobel_weight) / total
    cw = float(cfg.scharr_weight) / total
    gx = sw * sobel_gx + cw * scharr_gx
    gy = sw * sobel_gy + cw * scharr_gy
    mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    return _non_max_suppression(mag, theta), theta


def _normalize_grad_pair(gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.hypot(gx, gy)
    pos = mag[mag > 0]
    if pos.size == 0:
        return gx * 0.0, gy * 0.0
    scale = float(np.percentile(pos, 99.3)) + 1e-6
    return gx / scale, gy / scale


def _quantize_dirs(theta: np.ndarray) -> np.ndarray:
    ang = (np.rad2deg(theta) + 180.0) % 180.0
    dirs = np.zeros_like(ang, dtype=np.uint8)
    dirs[((ang >= 22.5) & (ang < 67.5))] = 1
    dirs[((ang >= 67.5) & (ang < 112.5))] = 2
    dirs[((ang >= 112.5) & (ang < 157.5))] = 3
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


def _auto_canny(gray: np.ndarray, high_pct: float, low_ratio: float) -> np.ndarray:
    high = float(np.percentile(gray, np.clip(high_pct, 70.0, 99.5)))
    low = high * float(np.clip(low_ratio, 0.2, 0.8))
    high_i = int(np.clip(high, 16, 255))
    low_i = int(np.clip(low, 4, max(4, high_i - 1)))
    return cv2.Canny(gray, low_i, high_i, L2gradient=True)


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
                if out[y2, x2] == 0 or strength[y2, x2] < min_strength:
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


def _clean_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (mask > 0).astype(np.uint8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == i] = 1
    return out


def _cap_edge_density(edges: np.ndarray, strength: np.ndarray, max_density: float) -> np.ndarray:
    edges = (edges > 0)
    count = int(edges.sum())
    max_count = int(max(1, round(float(max_density) * edges.size)))
    if count <= max_count:
        return edges.astype(np.uint8)
    vals = strength[edges]
    if vals.size == 0:
        return np.zeros_like(strength, dtype=np.uint8)
    kth = max(0, vals.size - max_count)
    thr = float(np.partition(vals, kth)[kth])
    return (edges & (strength >= thr)).astype(np.uint8)


def _to_strength_outline(
    edges: np.ndarray,
    strength: np.ndarray,
    high_thr: float,
    glow_gain: float,
    glow_sigma: float,
) -> np.ndarray:
    edges_bool = edges > 0
    if not edges_bool.any():
        return np.zeros_like(edges, dtype=np.uint8)
    if high_thr <= 1e-6:
        vals = strength[edges_bool]
        high_thr = float(np.percentile(vals, 75.0)) + 1e-6
    rel = np.clip(strength / float(high_thr), 0.0, 1.65)
    brightness = 68.0 + 187.0 * np.clip((rel - 0.34) / 1.12, 0.0, 1.0)
    core = np.where(edges_bool, brightness, 0.0).astype(np.float32)
    if glow_gain > 0.0:
        blur = cv2.GaussianBlur(core, (0, 0), sigmaX=max(float(glow_sigma), 0.1))
        core = np.maximum(core, blur * float(glow_gain))
    return np.clip(core, 0, 255).astype(np.uint8)
