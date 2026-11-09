"""Skin analysis using free OpenCV heuristics (no paid vision API).

Qoves uses a trained vision model for skin; the free substitute here samples
skin regions from the face mesh (forehead, cheeks, nose, chin) and computes
classical image statistics:

  * redness   -> mean a* channel in CIELAB (higher = more red/irritation)
  * texture   -> variance of the Laplacian (higher = rougher/more detail)
  * evenness  -> inverse of lightness spread across regions (even tone)
  * clarity   -> inverse of detected blemish-blob density

These are coarse, lighting-sensitive proxies, not a dermatological diagnosis.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .landmarks import SKIN_REGIONS, FaceLandmarks


def _region_mask(face: FaceLandmarks, indices: List[int]) -> np.ndarray:
    import cv2

    poly = np.array([face.xy(i) for i in indices], dtype=np.int32)
    mask = np.zeros((face.height, face.width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _region_pixels(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image_bgr[mask > 0]


def _blemish_density(gray_region: np.ndarray, mask: np.ndarray) -> float:
    """Estimate blemish density via local contrast blob detection."""
    import cv2

    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    # Difference-of-Gaussians highlights small dark/light spots.
    blur_small = cv2.GaussianBlur(gray_region, (0, 0), 1.0)
    blur_large = cv2.GaussianBlur(gray_region, (0, 0), 4.0)
    dog = cv2.absdiff(blur_small, blur_large)
    dog = cv2.bitwise_and(dog, dog, mask=mask)
    spots = int(np.count_nonzero(dog > 12))
    return spots / area


def analyze(image_bgr: np.ndarray, face: FaceLandmarks) -> Dict:
    """Compute per-region and overall skin metrics."""
    import cv2

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    region_stats: Dict[str, Dict] = {}
    region_lightness: List[float] = []
    redness_vals: List[float] = []
    texture_vals: List[float] = []
    blemish_vals: List[float] = []

    for name, indices in SKIN_REGIONS.items():
        mask = _region_mask(face, indices)
        if np.count_nonzero(mask) < 50:
            continue

        lab_px = lab[mask > 0].astype(np.float64)
        # OpenCV LAB: a* is centred at 128; subtract to get signed redness.
        redness = float(np.mean(lab_px[:, 1]) - 128.0)
        lightness = float(np.mean(lab_px[:, 0]))

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        texture = float(np.var(lap[mask > 0]))

        blemish = _blemish_density(gray, mask)

        region_stats[name] = {
            "redness": round(redness, 2),
            "texture_variance": round(texture, 1),
            "blemish_density": round(blemish, 4),
            "lightness": round(lightness, 1),
        }
        region_lightness.append(lightness)
        redness_vals.append(redness)
        texture_vals.append(texture)
        blemish_vals.append(blemish)

    if not region_stats:
        return {"regions": {}, "scores": {}, "overall_skin": 0.0}

    mean_redness = float(np.mean(redness_vals))
    mean_texture = float(np.mean(texture_vals))
    mean_blemish = float(np.mean(blemish_vals))
    tone_spread = float(np.std(region_lightness))

    # Map raw stats to 0-100 "good skin" scores (higher = better).
    # Thresholds are heuristic and tuned for typical selfie lighting.
    def inv_score(value: float, zero_at: float) -> float:
        return max(0.0, round((1.0 - value / zero_at) * 100, 1))

    scores = {
        "redness": inv_score(max(0.0, mean_redness), zero_at=25.0),
        "smoothness": inv_score(mean_texture, zero_at=800.0),
        "evenness": inv_score(tone_spread, zero_at=25.0),
        "clarity": inv_score(mean_blemish, zero_at=0.15),
    }
    overall = round(float(np.mean(list(scores.values()))), 1)

    return {
        "regions": region_stats,
        "scores": scores,
        "overall_skin": overall,
    }
