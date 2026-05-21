"""Render a thermal-scope screen from WHOT/BHOT or pure-outline inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import target_classifier as tc
from .target_classifier import TargetClassification


PALETTE_WHITEHOT = "whitehot"
PALETTE_BLACKHOT = "blackhot"


@dataclass
class ScopeRenderState:
    menu_open: bool = False
    menu_index: int = 0
    enhancement_level: int = 3
    zoom: int = 1
    palette: str = PALETTE_WHITEHOT
    outline_enabled: bool = False
    frozen: bool = False
    frame_id: int = 0
    fps: float = 0.0
    processing_profile: str = "thermal"
    resolution_label: str = "256x192"
    outline_processing_scale: float = 1.0
    outline_scale_x: float = 1.0
    outline_scale_y: float = 1.0


SCREEN_W = 960
SCREEN_H = 540
HUD = (220, 255, 255)
HUD_DIM = (96, 130, 130)
PERSON_COLOR = (235, 255, 220)
PERSON_GLOW = (170, 255, 255)
OBJECT_COLOR = (150, 170, 170)
HOTSPOT_COLOR = (190, 255, 255)


_U8_VALUES = np.arange(256, dtype=np.float32)


def _make_lut(multiplier: float) -> np.ndarray:
    return np.clip(_U8_VALUES * multiplier, 0, 255).astype(np.uint8)


_WHITEHOT_LUTS = (
    _make_lut(1.08),
    _make_lut(1.02),
    _make_lut(0.90),
)
_BLACKHOT_LUTS = (
    _make_lut(0.92),
    _make_lut(0.96),
    _make_lut(1.00),
)
_OUTLINE_LUTS = (
    _make_lut(0.98),
    _make_lut(1.00),
    _make_lut(0.86),
)


def _make_vignette(strength: float, floor: float) -> np.ndarray:
    yy, xx = np.mgrid[0:SCREEN_H, 0:SCREEN_W].astype(np.float32)
    rr = ((xx - SCREEN_W / 2) / (SCREEN_W / 2)) ** 2 + ((yy - SCREEN_H / 2) / (SCREEN_H / 2)) ** 2
    return np.clip(1.0 - strength * rr, floor, 1.0).astype(np.float32)


_THERMAL_VIGNETTE = _make_vignette(0.38, 0.58)
_OUTLINE_VIGNETTE = _make_vignette(0.32, 0.62)


def render_scope_frame(
    enhanced_gray: np.ndarray,
    outline_gray: np.ndarray,
    mask: Optional[np.ndarray],
    classifications: list[TargetClassification],
    hotspot: Optional[tuple[int, int, int]],
    state: ScopeRenderState,
) -> np.ndarray:
    """Return a 960x540 BGR image that looks like the scope display itself."""
    base_gray = outline_gray if state.outline_enabled else enhanced_gray
    view_gray, crop = _crop_zoom_16x9(base_gray, state.zoom)
    interp = cv2.INTER_NEAREST if state.outline_enabled else cv2.INTER_LINEAR
    view_gray = cv2.resize(view_gray, (SCREEN_W, SCREEN_H), interpolation=interp)
    if not state.outline_enabled and state.palette == PALETTE_BLACKHOT:
        view_gray = 255 - view_gray
    bgr = _outline_to_bgr(view_gray) if state.outline_enabled else _whitehot_to_bgr(view_gray, state.palette)

    sx, sy = _scale_from_crop(crop)
    if not state.outline_enabled:
        _draw_target_outlines(bgr, classifications, crop, sx, sy)
    if hotspot is not None:
        draw_hotspot = hotspot
        if state.outline_enabled:
            hx, hy, hv = hotspot
            draw_hotspot = (
                int(round(hx * max(state.outline_scale_x, 0.001))),
                int(round(hy * max(state.outline_scale_y, 0.001))),
                hv,
            )
        _draw_hotspot(bgr, draw_hotspot, crop, sx, sy)
    _draw_reticle(bgr)
    _draw_hud(bgr, state)
    if state.menu_open:
        _draw_menu(bgr, state)
    return bgr


def _crop_zoom_16x9(gray: np.ndarray, zoom: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = gray.shape[:2]
    target_ratio = 16.0 / 9.0
    crop_w = w
    crop_h = int(round(crop_w / target_ratio))
    if crop_h > h:
        crop_h = h
        crop_w = int(round(crop_h * target_ratio))

    zoom = max(1, int(zoom))
    crop_w = max(16, crop_w // zoom)
    crop_h = max(9, crop_h // zoom)
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return gray[y0:y0 + crop_h, x0:x0 + crop_w], (x0, y0, crop_w, crop_h)


def _scale_from_crop(crop: tuple[int, int, int, int]) -> tuple[float, float]:
    _, _, cw, ch = crop
    return SCREEN_W / float(cw), SCREEN_H / float(ch)


def _whitehot_to_bgr(gray: np.ndarray, palette: str) -> np.ndarray:
    luts = _BLACKHOT_LUTS if palette == PALETTE_BLACKHOT else _WHITEHOT_LUTS
    out = cv2.merge([cv2.LUT(gray, lut) for lut in luts])
    # Dark vignette helps the view read like an optic rather than a flat image.
    return (out.astype(np.float32) * _THERMAL_VIGNETTE[..., None]).astype(np.uint8)


def _outline_to_bgr(gray: np.ndarray) -> np.ndarray:
    out = cv2.merge([cv2.LUT(gray, lut) for lut in _OUTLINE_LUTS])
    return (out.astype(np.float32) * _OUTLINE_VIGNETTE[..., None]).astype(np.uint8)


def _map_pt(x: float, y: float, crop: tuple[int, int, int, int], sx: float, sy: float) -> tuple[int, int]:
    x0, y0, _, _ = crop
    return int(round((x - x0) * sx)), int(round((y - y0) * sy))


def _contour_visible(cnt: np.ndarray, crop: tuple[int, int, int, int]) -> bool:
    x0, y0, cw, ch = crop
    x, y, w, h = cv2.boundingRect(cnt)
    return not (x + w < x0 or y + h < y0 or x > x0 + cw or y > y0 + ch)


def _draw_target_outlines(
    img: np.ndarray,
    classifications: list[TargetClassification],
    crop: tuple[int, int, int, int],
    sx: float,
    sy: float,
) -> None:
    for cls in classifications:
        if cls.target_type == tc.HOTSPOT_NOISE:
            continue
        cnt = cls.feature.contour
        if not _contour_visible(cnt, crop):
            continue
        x0, y0, _, _ = crop
        pts = cnt.astype(np.float32).copy()
        pts[:, 0, 0] = (pts[:, 0, 0] - x0) * sx
        pts[:, 0, 1] = (pts[:, 0, 1] - y0) * sy
        pts_i = pts.astype(np.int32)
        if cls.target_type == tc.PERSON:
            cv2.drawContours(img, [pts_i], -1, PERSON_GLOW, 8, cv2.LINE_AA)
            cv2.drawContours(img, [pts_i], -1, PERSON_COLOR, 2, cv2.LINE_AA)
        elif cls.target_type == tc.OBJECT:
            cv2.drawContours(img, [pts_i], -1, OBJECT_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.drawContours(img, [pts_i], -1, HUD_DIM, 1, cv2.LINE_AA)


def _draw_hotspot(
    img: np.ndarray,
    hotspot: tuple[int, int, int],
    crop: tuple[int, int, int, int],
    sx: float,
    sy: float,
) -> None:
    hx, hy, _ = hotspot
    x0, y0, cw, ch = crop
    if hx < x0 or hy < y0 or hx >= x0 + cw or hy >= y0 + ch:
        return
    x, y = _map_pt(hx, hy, crop, sx, sy)
    cv2.circle(img, (x, y), 6, HOTSPOT_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (x - 14, y), (x - 5, y), HOTSPOT_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (x + 5, y), (x + 14, y), HOTSPOT_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - 14), (x, y - 5), HOTSPOT_COLOR, 1, cv2.LINE_AA)
    cv2.line(img, (x, y + 5), (x, y + 14), HOTSPOT_COLOR, 1, cv2.LINE_AA)


def _draw_reticle(img: np.ndarray) -> None:
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    color = (205, 245, 245)
    cv2.line(img, (cx - 44, cy), (cx - 10, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 10, cy), (cx + 44, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 44), (cx, cy - 10), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 10), (cx, cy + 44), color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)
    for off in (-70, -50, 50, 70):
        cv2.line(img, (cx + off, cy - 3), (cx + off, cy + 3), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - 3, cy + off), (cx + 3, cy + off), color, 1, cv2.LINE_AA)


def _draw_hud(img: np.ndarray, state: ScopeRenderState) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "MRAD", (28, 42), font, 0.75, HUD, 2, cv2.LINE_AA)
    cv2.rectangle(img, (22, 18), (108, 50), HUD_DIM, 1, cv2.LINE_AA)

    start_x = SCREEN_W // 2 - 86
    for i in range(5):
        x = start_x + i * 34
        rect = (x, 25, 24, 24)
        if i < state.enhancement_level:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), HUD, -1)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), HUD, 1)

    cv2.putText(img, "ZOOM", (SCREEN_W - 86, 185), font, 0.55, HUD, 1, cv2.LINE_AA)
    cv2.putText(img, f"{state.zoom}x", (SCREEN_W - 78, 230), font, 0.72, HUD, 2, cv2.LINE_AA)
    visible = state.processing_profile == "visible"
    if visible:
        mode = "VOUT" if state.outline_enabled else "VIS"
    else:
        mode = "OUT" if state.outline_enabled else ("BHOT" if state.palette == PALETTE_BLACKHOT else "WHOT")
    freeze = " HOLD" if state.frozen else ""
    cv2.putText(img, f"{mode}{freeze}", (SCREEN_W - 132, SCREEN_H - 24), font, 0.48, HUD_DIM, 1, cv2.LINE_AA)
    scale = f" x{state.outline_processing_scale:.1f}".replace(".0", "") \
        if state.outline_processing_scale > 1.01 else ""
    profile = ("VISIBLE DEMO" if visible else "THERMAL RAW") + f" {state.resolution_label}{scale}"
    cv2.putText(img, profile, (28, SCREEN_H - 24), font, 0.46, HUD_DIM, 1, cv2.LINE_AA)


def _draw_menu(img: np.ndarray, state: ScopeRenderState) -> None:
    items = [
        f"1 ENH {state.enhancement_level}",
        f"2 ZOOM {state.zoom}x",
        f"3 {'BHOT' if state.palette == PALETTE_BLACKHOT else 'WHOT'}",
        f"4 OUTLINE {'ON' if state.outline_enabled else 'OFF'}",
        f"5 FREEZE {'ON' if state.frozen else 'OFF'}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 28, 115
    for i, text in enumerate(items):
        top = y + i * 42
        if i == state.menu_index:
            cv2.rectangle(img, (x - 8, top - 24), (x + 176, top + 8), HUD, -1)
            color = (0, 20, 20)
            thick = 2
        else:
            cv2.rectangle(img, (x - 8, top - 24), (x + 176, top + 8), HUD_DIM, 1)
            color = HUD
            thick = 1
        cv2.putText(img, text, (x, top), font, 0.55, color, thick, cv2.LINE_AA)
