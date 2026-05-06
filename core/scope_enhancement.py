"""Scope-style thermal enhancement for the PC simulation.

This module intentionally implements an AGC/DDE-like display enhancement, not a
Sobel edge detector. The goal is to mimic a thermal scope display: dark
background, compressed dynamic range, and boosted local detail around warm
targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import thermal_processing


@dataclass
class ScopeEnhanceConfig:
    level: int = 3
    p_low: float = 2.0
    p_high: float = 98.0
    base_sigma: float = 3.0
    denoise_ksize: int = 3


def _clip_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _robust_agc(raw14: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    return thermal_processing.raw14_to_u8(raw14, p_low=p_low, p_high=p_high)


def enhance_scope_whitehot(raw14: np.ndarray, config: ScopeEnhanceConfig) -> np.ndarray:
    """Return a uint8 white-hot display frame using robust AGC + detail boost.

    Level 1..5 maps to increasing detail gain and stronger black background
    compression. This is designed for display simulation and screenshot
    generation; it is not a calibrated temperature transform.
    """
    raw14 = thermal_processing.to_raw14(raw14)

    level = int(np.clip(config.level, 1, 5))
    detail_gain = [0.35, 0.55, 0.78, 1.05, 1.35][level - 1]
    gamma = [1.35, 1.25, 1.15, 1.05, 0.95][level - 1]
    black_clip = [12.0, 18.0, 24.0, 30.0, 36.0][level - 1]

    agc = _robust_agc(raw14, config.p_low, config.p_high).astype(np.float32)
    base = cv2.GaussianBlur(agc, (0, 0), sigmaX=config.base_sigma)
    detail = agc - base

    enhanced = base + detail_gain * detail
    enhanced = np.maximum(enhanced - black_clip, 0.0)
    enhanced = 255.0 * np.power(np.clip(enhanced / 255.0, 0.0, 1.0), gamma)

    out = _clip_u8(enhanced)
    k = max(1, int(config.denoise_ksize) | 1)
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out
