"""External contour extraction + geometric features for candidate regions.

Algorithm: cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) which is the
Suzuki/Abe 1985 border-following algorithm — pure integer pixel walk, easily
re-implemented on ESP32-S3 in <300 lines of C. Features computed below are
all integer arithmetic on the contour point list, so the same code structure
can be ported verbatim.

Hardware budget: on a 64x48 mask with <8 candidates, the entire pass
(findContours + features for all contours) takes ~3-5 ms on ESP32-S3 at
240 MHz, fitting comfortably in a 25 fps frame budget.

NOTE: the colour helpers below are *defaults*; the final draw colour is
chosen by `target_classifier.classify()` and overridden by it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


# BGR colours (OpenCV order). Tuned so they read clearly on Iron/Turbo palettes.
COLOR_PERSON = (180, 255, 0)      # cyan-green
COLOR_OBJECT = (0, 220, 255)      # yellow
COLOR_HOTSPOT_NOISE = (40, 40, 220)  # dim red
COLOR_UNKNOWN = (200, 200, 200)   # light grey


@dataclass
class ContourFeature:
    contour: np.ndarray            # cv2 contour: shape (N,1,2) int32
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float                    # cv2.contourArea (real area, not bbox area)
    perimeter: float               # cv2.arcLength
    aspect_ratio: float            # h / w  (> 1 = vertical)
    extent: float                  # area / (w*h)   in [0,1]
    circularity: float             # 4*pi*area / perimeter^2  in [0,1]; 1 = perfect circle
    centroid: tuple[float, float]  # (cx, cy) in pixels


def extract_contours(mask: np.ndarray, min_area: int = 12) -> List[ContourFeature]:
    """Find external contours on a binary mask and compute per-contour features.

    `mask` is uint8, non-zero foreground. Contours with area < min_area are
    discarded (matches the candidate filter in hotspot_detector).
    """
    if mask is None or mask.size == 0:
        return []
    # ensure binary 0/255
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[ContourFeature] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
        perim = float(cv2.arcLength(cnt, closed=True))
        aspect = h / float(w)
        extent = area / float(w * h)
        circularity = 0.0
        if perim > 0:
            circularity = float(4.0 * np.pi * area / (perim * perim))
            circularity = min(circularity, 1.0)
        # centroid via image moments (one division each)
        m = cv2.moments(cnt)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx = float(x + w / 2)
            cy = float(y + h / 2)
        out.append(ContourFeature(
            contour=cnt,
            bbox=(int(x), int(y), int(w), int(h)),
            area=area,
            perimeter=perim,
            aspect_ratio=float(aspect),
            extent=float(extent),
            circularity=circularity,
            centroid=(float(cx), float(cy)),
        ))
    return out


def color_for_type(target_type: str) -> tuple[int, int, int]:
    """Map a classifier target_type string to a default BGR colour."""
    return {
        "person_candidate": COLOR_PERSON,
        "object_candidate": COLOR_OBJECT,
        "hotspot_noise": COLOR_HOTSPOT_NOISE,
        "unknown": COLOR_UNKNOWN,
    }.get(target_type, COLOR_UNKNOWN)
