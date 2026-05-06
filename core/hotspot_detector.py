"""Hotspot and candidate-region detection on grayscale thermal frames.

The hotspot is simply argmax of the (processed) grayscale image. Candidate
regions are derived by percentile thresholding + morphological cleanup +
connected-component labelling, mirroring the FPGA-side `candidate_mask_gen`
behaviour in spirit so the PC viewer's overlays match the look of the eventual
embedded pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class CandidateRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area: int


@dataclass
class DetectConfig:
    percentile: float = 95.0
    min_area: int = 12
    morph_ksize: int = 3


def find_hotspot(gray: np.ndarray) -> tuple[int, int, int]:
    _, max_val, _, max_loc = cv2.minMaxLoc(gray)
    return int(max_loc[0]), int(max_loc[1]), int(max_val)


def find_candidates(
    gray: np.ndarray, cfg: DetectConfig
) -> tuple[np.ndarray, List[CandidateRegion]]:
    thr = float(np.percentile(gray, cfg.percentile))
    mask = (gray >= thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_ksize, cfg.morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: List[CandidateRegion] = []
    for i in range(1, num):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < cfg.min_area:
            continue
        cx, cy = centroids[i]
        regions.append(CandidateRegion(
            bbox=(int(x), int(y), int(w), int(h)),
            centroid=(float(cx), float(cy)),
            area=int(area),
        ))
    return mask, regions


def summarize(regions: List[CandidateRegion]) -> tuple[int, float, float]:
    """Area-weighted overall centroid, matching candidate_count/cx/cy in the FPGA packet."""
    if not regions:
        return 0, 0.0, 0.0
    total_area = sum(r.area for r in regions)
    cx = sum(r.centroid[0] * r.area for r in regions) / total_area
    cy = sum(r.centroid[1] * r.area for r in regions) / total_area
    return len(regions), float(cx), float(cy)
