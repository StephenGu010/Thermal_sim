"""Rule-based target classification for thermal candidate regions.

THIS IS NOT A NEURAL NETWORK. THIS IS NOT A TRAINED CLASSIFIER.
It is a hand-written geometric heuristic that scores each candidate against
a small set of rules and returns the best-matching label. The "confidence"
score is the soft-rule similarity (0..1), NOT a probability of being correct.
The labels intentionally end in `_candidate` to remind callers that this is
suggestion-grade information.

Why this approach (not HOG/CNN/Haar):
  The downstream embedded plan is FPGA(Tang Nano 9K) -> ESP32-S3. Neither
  platform has the FLOPs/RAM budget to run real person detection at 25 fps
  on raw thermal frames. The rules below operate on the candidate mask the
  FPGA already produces (`candidate_mask_gen` in rtl/) and the connected-
  component statistics that Suzuki contour extraction yields. Every value
  used is integer-friendly and computable from a single pass over the bbox
  region.

Rules (all thresholds in `ClassifyConfig` are exposed to the UI):

  PERSON_CANDIDATE
    - aspect_ratio (h/w) in [aspect_min, aspect_max], typically 1.4..4.0
    - area >= person_min_area_frac * frame_area
    - extent in [0.30, 0.85]   (filled silhouette but with limb gaps)
    - vertical projection profile shows a head/shoulder/torso shape:
        top 20% of bbox has narrower mean width than middle 50%
    - circularity < 0.5 (humans aren't blobs)
    Score = product of soft membership of each rule, clipped.

  OBJECT_CANDIDATE
    - aspect_ratio in [0.4, 1.4] (square-ish or horizontal)
    - extent >= 0.55  (filled rectangles, cups)
    - area smaller than typical person area but above noise

  HOTSPOT_NOISE
    - area < noise_max_area
    - circularity > 0.7 (point sources)

  UNKNOWN
    - default if nothing fits
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .contour_overlay import (
    COLOR_HOTSPOT_NOISE,
    COLOR_OBJECT,
    COLOR_PERSON,
    COLOR_UNKNOWN,
    ContourFeature,
)


PERSON = "person_candidate"
OBJECT = "object_candidate"
HOTSPOT_NOISE = "hotspot_noise"
UNKNOWN = "unknown"


@dataclass
class ClassifyConfig:
    # area thresholds expressed as fraction of frame area
    person_min_area_frac: float = 0.012
    object_min_area_frac: float = 0.002
    noise_max_area: int = 60
    # person aspect ratio (h/w) range
    person_aspect_min: float = 1.30
    person_aspect_max: float = 4.50
    # extent (filled fraction of bbox) range for a valid person silhouette
    person_extent_min: float = 0.28
    person_extent_max: float = 0.88
    # head/torso ratio: top_band_mean_width / mid_band_mean_width
    head_torso_ratio_max: float = 0.85
    # object rules
    object_aspect_min: float = 0.40
    object_aspect_max: float = 1.40
    object_extent_min: float = 0.55
    # noise rules
    noise_circularity_min: float = 0.70


@dataclass
class TargetClassification:
    target_type: str          # one of PERSON/OBJECT/HOTSPOT_NOISE/UNKNOWN
    confidence: float         # soft rule score in [0,1]; NOT a probability
    reason: str               # short human-readable diagnostic, e.g. "aspect=2.1 head/torso=0.32"
    draw_color: tuple[int, int, int]
    label: str                # short uppercase label for on-screen text
    feature: ContourFeature   # back-reference for downstream draw


def _soft_band(value: float, lo: float, hi: float, slack: float = 0.15) -> float:
    """Smooth membership function: 1 inside [lo, hi], drops linearly to 0 over slack on each side."""
    if value < lo - slack or value > hi + slack:
        return 0.0
    if value < lo:
        return (value - (lo - slack)) / slack
    if value > hi:
        return ((hi + slack) - value) / slack
    return 1.0


def _head_torso_ratio(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Return mean_width(top 20% band) / mean_width(middle 50% band) inside bbox.

    Mean width per row = number of foreground pixels in that row of the mask.
    A standing person typically has ratio < 0.85 because the head is narrower
    than the shoulders/torso. A symmetric blob (cup, square heater) is ~1.0.

    All operations are sums over rows — directly portable to ESP32-S3.
    """
    x, y, w, h = bbox
    if h < 5 or w < 3:
        return 1.0  # too small to tell, treat as non-person
    sub = mask[y:y + h, x:x + w]
    if sub.size == 0:
        return 1.0
    fg = (sub > 0).astype(np.int32)
    row_widths = fg.sum(axis=1)  # (h,)
    top_h = max(1, int(round(h * 0.20)))
    mid_lo = int(round(h * 0.25))
    mid_hi = int(round(h * 0.75))
    if mid_hi <= mid_lo:
        return 1.0
    top_mean = float(row_widths[:top_h].mean())
    mid_mean = float(row_widths[mid_lo:mid_hi].mean())
    if mid_mean <= 0:
        return 1.0
    return top_mean / mid_mean


def classify(
    feature: ContourFeature,
    processed_gray: np.ndarray,
    mask: np.ndarray,
    cfg: ClassifyConfig,
) -> TargetClassification:
    h_img, w_img = processed_gray.shape[:2]
    frame_area = float(h_img * w_img)
    area = feature.area

    # ---- 1. hotspot noise (small bright dot) -------------------------------
    if area <= cfg.noise_max_area and feature.circularity >= cfg.noise_circularity_min:
        return TargetClassification(
            target_type=HOTSPOT_NOISE,
            confidence=min(1.0, feature.circularity),
            reason=f"area={int(area)} circ={feature.circularity:.2f}",
            draw_color=COLOR_HOTSPOT_NOISE,
            label="SMALL HOTSPOT",
            feature=feature,
        )

    # ---- 2. score against PERSON rule --------------------------------------
    person_score_parts = []
    person_score_parts.append(_soft_band(
        feature.aspect_ratio, cfg.person_aspect_min, cfg.person_aspect_max, slack=0.30))
    person_score_parts.append(_soft_band(
        feature.extent, cfg.person_extent_min, cfg.person_extent_max, slack=0.10))
    person_score_parts.append(1.0 if area >= cfg.person_min_area_frac * frame_area else 0.0)
    person_score_parts.append(0.0 if feature.circularity > 0.55 else 1.0 - feature.circularity)
    head_torso = _head_torso_ratio(mask, feature.bbox)
    person_score_parts.append(_soft_band(head_torso, 0.0, cfg.head_torso_ratio_max, slack=0.15))
    person_score = float(np.prod(person_score_parts))

    # ---- 3. score against OBJECT rule --------------------------------------
    obj_score_parts = [
        _soft_band(feature.aspect_ratio, cfg.object_aspect_min, cfg.object_aspect_max, slack=0.20),
        1.0 if feature.extent >= cfg.object_extent_min else feature.extent / cfg.object_extent_min,
        1.0 if area >= cfg.object_min_area_frac * frame_area else 0.0,
    ]
    obj_score = float(np.prod(obj_score_parts))

    # ---- 4. pick the winner ------------------------------------------------
    # apply minimum thresholds so we don't label something with score=0.05 as a person
    person_pass = person_score >= 0.35
    object_pass = obj_score >= 0.35
    reason_common = (f"asp={feature.aspect_ratio:.2f} ext={feature.extent:.2f} "
                     f"circ={feature.circularity:.2f} ht={head_torso:.2f}")

    if person_pass and person_score >= obj_score:
        return TargetClassification(
            target_type=PERSON,
            confidence=person_score,
            reason=reason_common,
            draw_color=COLOR_PERSON,
            label=f"PERSON CANDIDATE {person_score:.2f}",
            feature=feature,
        )
    if object_pass:
        return TargetClassification(
            target_type=OBJECT,
            confidence=obj_score,
            reason=reason_common,
            draw_color=COLOR_OBJECT,
            label=f"HOT OBJECT {obj_score:.2f}",
            feature=feature,
        )
    return TargetClassification(
        target_type=UNKNOWN,
        confidence=max(person_score, obj_score),
        reason=reason_common,
        draw_color=COLOR_UNKNOWN,
        label="UNKNOWN",
        feature=feature,
    )


def classify_all(
    features: list[ContourFeature],
    processed_gray: np.ndarray,
    mask: np.ndarray,
    cfg: ClassifyConfig,
) -> list[TargetClassification]:
    return [classify(f, processed_gray, mask, cfg) for f in features]


def summarize(classifications: list[TargetClassification]) -> dict:
    """Roll-up stats for the right-side data panel."""
    person = [c for c in classifications if c.target_type == PERSON]
    obj = [c for c in classifications if c.target_type == OBJECT]
    main = max(classifications, key=lambda c: (
        2 if c.target_type == PERSON else 1 if c.target_type == OBJECT else 0,
        c.confidence,
    ), default=None)
    return {
        "target_count": len(classifications),
        "person_candidate_count": len(person),
        "object_candidate_count": len(obj),
        "main_target_type": main.target_type if main else "-",
        "main_target_score": f"{main.confidence:.2f}" if main else "-",
        "main_target_reason": main.reason if main else "-",
    }
