"""Bake all overlays directly into the BGR preview frame (Approach A).

Why this lives in OpenCV land instead of QPainter:
  - Screenshot, video recording and on-screen preview all consume the same
    rendered_bgr — guaranteed pixel parity, no "the screenshot doesn't look
    like the UI" surprises.
  - cv2 draw primitives (drawContours, rectangle, putText, line, addWeighted)
    map 1:1 to ESP32-S3 LVGL primitives, so the embedded renderer can mirror
    the layout without rewriting layout logic.

Drawn elements (in z-order, bottom-up):
  1. Optional translucent candidate mask (alpha-blended)
  2. Per-target outline contour (cv2.polylines, configurable thickness)
  3. Per-target bbox (1px rectangle)
  4. Per-target label text near the top-left corner of the bbox
  5. Hotspot crosshair (cyan)
  6. Status banner: "PC Preview · {palette}", "FPS x.x", "{target_count} candidates"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .target_classifier import TargetClassification


# screen-element colours
HOTSPOT_COLOR = (230, 255, 0)   # bright cyan
BANNER_COLOR = (255, 255, 255)
MASK_COLOR_BGR = (0, 80, 255)   # red/orange
MASK_ALPHA = 0.35


@dataclass
class RenderOptions:
    show_mask: bool = True
    show_contours: bool = True
    show_bboxes: bool = True
    show_labels: bool = True
    show_hotspot: bool = True
    show_banner: bool = True
    contour_thickness: int = 2
    bbox_thickness: int = 1
    palette_name: str = "Iron"
    fps: float = 0.0


def render_preview_frame(
    bgr: np.ndarray,
    classifications: List[TargetClassification],
    mask: Optional[np.ndarray],
    hotspot: Optional[tuple[int, int, int]],
    opts: RenderOptions,
) -> np.ndarray:
    """Return a NEW BGR uint8 frame with all overlays baked in."""
    out = bgr.copy()

    # 1. translucent mask
    if opts.show_mask and mask is not None and mask.size:
        overlay = out.copy()
        overlay[mask > 0] = MASK_COLOR_BGR
        out = cv2.addWeighted(overlay, MASK_ALPHA, out, 1.0 - MASK_ALPHA, 0)

    # 2. contours
    if opts.show_contours:
        for cls in classifications:
            cv2.drawContours(out, [cls.feature.contour], -1, cls.draw_color,
                             thickness=max(1, opts.contour_thickness), lineType=cv2.LINE_AA)

    # 3. bboxes (thin rectangle so contour stays the focus)
    if opts.show_bboxes:
        for cls in classifications:
            x, y, w, h = cls.feature.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), cls.draw_color,
                          thickness=max(1, opts.bbox_thickness), lineType=cv2.LINE_AA)

    # 4. labels
    if opts.show_labels:
        for cls in classifications:
            x, y, w, h = cls.feature.bbox
            label = cls.label
            _draw_label(out, label, (x, y), cls.draw_color)

    # 5. hotspot crosshair
    if opts.show_hotspot and hotspot is not None:
        hx, hy, _ = hotspot
        cv2.line(out, (hx - 10, hy), (hx + 10, hy), HOTSPOT_COLOR, 1, cv2.LINE_AA)
        cv2.line(out, (hx, hy - 10), (hx, hy + 10), HOTSPOT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(out, (hx, hy), 3, HOTSPOT_COLOR, 1, cv2.LINE_AA)

    # 6. banner
    if opts.show_banner:
        n_targets = len(classifications)
        line1 = f"PC Preview - {opts.palette_name}"
        line2 = f"FPS {opts.fps:5.1f}   targets {n_targets}"
        _draw_banner(out, [line1, line2])

    return out


def _draw_label(img: np.ndarray, text: str, anchor: tuple[int, int],
                color: tuple[int, int, int]) -> None:
    """Draw small text with a dark backing rectangle so it reads on any palette."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x, y = anchor
    pad = 2
    # position label JUST above the bbox; if no room, put it just inside the top
    bg_y1 = y - th - 2 * pad
    if bg_y1 < 0:
        bg_y1 = y + 2
    bg_y2 = bg_y1 + th + 2 * pad
    bg_x1 = x
    bg_x2 = x + tw + 2 * pad
    # backdrop
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (bg_x1 + pad, bg_y2 - pad - baseline + 1),
                font, scale, color, thick, cv2.LINE_AA)


def _draw_banner(img: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    pad = 4
    line_h = 14
    box_h = pad * 2 + line_h * len(lines)
    box_w = pad * 2 + max((cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines), default=0)
    # backdrop top-left
    cv2.rectangle(img, (4, 4), (4 + box_w, 4 + box_h), (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (4, 4), (4 + box_w, 4 + box_h), BANNER_COLOR, thickness=1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (4 + pad, 4 + pad + line_h * (i + 1) - 3),
                    font, scale, BANNER_COLOR, thick, cv2.LINE_AA)
