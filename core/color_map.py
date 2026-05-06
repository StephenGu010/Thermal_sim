"""Software pseudo-colour palettes for grayscale thermal frames.

OpenCV ships several colormaps but lacks a classical "iron" palette which
thermal optics traditionally use, so it is provided here as a 256-entry BGR
look-up table built from key colour stops via linear interpolation.

Switching the palette must NOT alter the underlying grayscale data — all
statistics (hotspot, normalize, candidate detection) operate on the original
gray, the palette only affects the final BGR shown to the user.
"""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class Palette(str, Enum):
    GRAY = "Gray"
    IRON = "Iron"
    INFERNO = "Inferno"
    JET = "Jet"
    TURBO = "Turbo"
    HOT = "Hot"
    BONE = "Bone"


# OpenCV palette dispatch (BGR LUT applied internally).
_OPENCV_MAP = {
    Palette.INFERNO: cv2.COLORMAP_INFERNO,
    Palette.JET: cv2.COLORMAP_JET,
    Palette.TURBO: cv2.COLORMAP_TURBO,
    Palette.HOT: cv2.COLORMAP_HOT,
    Palette.BONE: cv2.COLORMAP_BONE,
}


def _build_iron_lut() -> np.ndarray:
    """Classic black -> purple -> red -> orange -> yellow -> white iron ramp.

    Stops are given as RGB; we convert to BGR for OpenCV compatibility and
    return shape (256, 1, 3) uint8 ready for cv2.applyColorMap-style LUT use.
    """
    stops = [
        (0.00, (0,   0,   0)),
        (0.20, (60,  0,   80)),
        (0.40, (200, 30,  60)),
        (0.60, (255, 110, 20)),
        (0.80, (255, 220, 60)),
        (1.00, (255, 255, 255)),
    ]
    lut = np.zeros((256, 3), dtype=np.float32)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=np.float32)  # RGB
    xs = np.linspace(0.0, 1.0, 256)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, pos, cols[:, ch])
    rgb = np.clip(lut, 0, 255).astype(np.uint8)
    bgr = rgb[:, ::-1]  # RGB -> BGR
    return bgr.reshape(256, 1, 3)


_IRON_LUT = _build_iron_lut()


def apply(gray: np.ndarray, palette: Palette) -> np.ndarray:
    """Map a uint8 grayscale frame to a uint8 BGR image."""
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if palette == Palette.GRAY:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if palette == Palette.IRON:
        return cv2.applyColorMap(gray, _IRON_LUT)
    return cv2.applyColorMap(gray, _OPENCV_MAP[palette])
