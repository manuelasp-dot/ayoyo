"""Skin analysis using free OpenCV image statistics (no paid vision API).

The pipeline is designed to measure the *skin*, not the *photo*:

1. Scale normalisation: the face is cropped and resized to a fixed width, so
   texture is measured in face-relative units regardless of image resolution.
2. Skin-only sampling: within each landmark region we keep pixels whose colour
   is close to the face's own median skin colour. That drops beard, hair,
   shadows and specular glare; small holes (blemishes) are closed back in.
3. Lighting removal: a quadratic brightness surface is fitted to each region
   and subtracted, removing lighting gradients and broad 3D shading. Texture
   is then measured with band-pass filters as % of the fitted illumination,
   so exposure changes cancel out. Blurs are mask-normalised so excluded
   pixels do not leak in.
4. Tone-independent redness: redness is measured relative to the person's own
   median a* (CIELAB), so naturally warm or dark skin is not penalised; only
   patches redder than the person's baseline count.

Raw metrics (lower = better):
  * redness    -> mean a* excess above the face's own median (CIELAB units)
  * smoothness -> robust std of fine-scale contrast (% of local brightness)
  * evenness   -> robust std of mid-scale contrast, i.e. blotchiness (%)
  * clarity    -> % of skin area covered by dark or red spots

These are photo-based estimates, not a dermatological diagnosis.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .landmarks import SKIN_REGIONS, FaceLandmarks, region_hull

CANON_FACE_WIDTH = 256   # px; faces are resized to this width before measuring
LOW_RES_FACE_WIDTH = 180  # px in the source image; below this, results are unreliable
MIN_SKIN_PIXELS = 150
# Regions where less than this fraction is skin-coloured are mostly covered by
# hair, beard or deep shadow; the few pixels that pass are unreliable (e.g.
# red hair strands reading as "redness"), so they are left out of the overall.
MIN_REGION_COVERAGE = 0.5
CLIP_LEVEL = 250          # 8-bit value treated as overexposed (no information)
OVEREXPOSED_FRACTION = 0.15  # flag the photo if this much of the skin is clipped

DARK_SPOT_PCT = 7.0  # a spot is >=7% darker than its surroundings...
RED_SPOT_A = 5.0     # ...or >=5 a* units redder than its surroundings

# Score anchors per metric: (raw value scoring 100, raw value scoring 0).
# Calibrated on 14 frontal photos of 5 people (varied skin tone, lighting,
# resolution, beards): clean studio portraits land ~85-100, typical photos in
# the 60s-80s, and only clearly problematic photos approach 0.
ANCHORS = {
    "redness": (0.4, 4.0),     # observed 0.5-2.2
    "smoothness": (0.8, 4.0),  # observed 0.8-2.5
    "evenness": (0.6, 4.0),    # observed 0.8-3.0
    "clarity": (0.1, 6.0),     # observed 0.04-4.7
}


def _mblur(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of ``x`` using only pixels where mask ``m`` is 1."""
    num = cv2.GaussianBlur(x * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def _robust_std(v: np.ndarray) -> float:
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def _illumination(L: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit a quadratic brightness surface to each labelled region.

    The surface captures lighting gradients and broad 3D shading within the
    region; what remains after subtracting it is skin texture and tone.
    """
    illum = np.zeros_like(L)
    for lab_id in np.unique(labels[labels > 0]):
        yy, xx = np.nonzero(labels == lab_id)
        x = (xx - xx.mean()) / (xx.std() + 1e-6)
        y = (yy - yy.mean()) / (yy.std() + 1e-6)
        design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
        coef, *_ = np.linalg.lstsq(design, L[yy, xx], rcond=None)
        illum[yy, xx] = design @ coef
    return illum


def _maps(lab: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-pixel band-pass maps used by every metric.

    ``labels`` is a region label image (0 = not skin, 1..k = regions).
    """
    m = (labels > 0).astype(np.float32)
    L, A = lab[..., 0], lab[..., 1]

    illum = _illumination(L, labels)
    resid = (L - illum) * m
    base = np.maximum(illum, 1.0)
    R1, R15, R4, R6, R16 = (_mblur(resid, m, s) for s in (1.0, 1.5, 4.0, 6.0, 16.0))
    A15, A6 = _mblur(A, m, 1.5), _mblur(A, m, 6.0)

    return {
        "fine": (R1 - R4) / base * 100,
        "mid": (R4 - R16) / base * 100,
        "spot": ((R15 - R6) / base * 100 < -DARK_SPOT_PCT) | (A15 - A6 > RED_SPOT_A),
        "a": _mblur(A, m, 2.0),
    }


def _raw(maps: Dict[str, np.ndarray], sel: np.ndarray, baseline_a: float) -> Dict[str, float]:
    spots = cv2.morphologyEx(
        (maps["spot"] & sel).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    return {
        "redness": float(np.mean(np.maximum(0.0, maps["a"][sel] - baseline_a))),
        "smoothness": _robust_std(maps["fine"][sel]),
        "evenness": _robust_std(maps["mid"][sel]),
        "clarity": float(np.count_nonzero(spots) / np.count_nonzero(sel) * 100),
    }


def measure(lab: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Raw skin metrics for a float32 CIELAB image.

    ``labels`` is a region label image (0 = not skin); a plain 0/1 mask is
    treated as a single region.
    """
    sel = labels > 0
    maps = _maps(lab, labels)
    return _raw(maps, sel, float(np.median(maps["a"][sel])))


def score(raw: Dict[str, float]) -> Dict[str, float]:
    """Map raw metrics to 0-100 scores (higher = better) via ANCHORS."""
    out = {}
    for key, (good, bad) in ANCHORS.items():
        frac = (raw[key] - good) / (bad - good)
        out[key] = round(float(np.clip(1.0 - frac, 0.0, 1.0)) * 100, 1)
    return out


def _normalise(image_bgr: np.ndarray, face: FaceLandmarks):
    """Crop to the face and resize to CANON_FACE_WIDTH.

    Returns (lab, points_xy, clipped) where ``clipped`` marks overexposed
    pixels (any channel near 255), used to flag unreliable lighting.
    """
    xs, ys = face.points[:, 0], face.points[:, 1]
    pad = 0.05 * (xs.max() - xs.min())
    x0, y0 = max(0, int(xs.min() - pad)), max(0, int(ys.min() - pad))
    x1, y1 = min(face.width, int(xs.max() + pad)), min(face.height, int(ys.max() + pad))
    s = CANON_FACE_WIDTH / max(1, x1 - x0)
    crop = cv2.resize(
        image_bgr[y0:y1, x0:x1],
        (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))),
        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
    )
    clipped = cv2.dilate((crop.max(axis=2) >= CLIP_LEVEL).astype(np.uint8), np.ones((3, 3), np.uint8))
    lab = cv2.cvtColor(crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    pts = (face.points[:, :2] - [x0, y0]) * s
    return lab, pts, clipped


def _skin_mask(lab: np.ndarray, region_masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Pixels inside the regions whose colour is close to the face's median skin."""
    union = np.zeros(lab.shape[:2], np.uint8)
    for m in region_masks.values():
        union |= m
    inside = union > 0
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    med_l, med_a, med_b = (float(np.median(c[inside])) for c in (L, A, B))

    close_l = np.abs(L - med_l) <= max(10.0, 0.35 * med_l)
    close_c = np.hypot(A - med_a, B - med_b) <= 15.0
    skin = (close_l & close_c & inside).astype(np.uint8)
    # Re-admit small holes (blemishes, red spots) but not large areas (beard).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    return skin & union


def analyze(image_bgr: np.ndarray, face: FaceLandmarks) -> Dict:
    """Compute per-region and overall skin metrics."""
    lab, pts, clipped = _normalise(image_bgr, face)
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    region_masks = {}
    for name in SKIN_REGIONS:
        m = np.zeros(lab.shape[:2], np.uint8)
        cv2.fillPoly(m, [region_hull(pts, name)], 1)
        region_masks[name] = cv2.erode(m, erode)

    skin = _skin_mask(lab, region_masks)
    # Clipped pixels are flagged rather than excluded: removing them from
    # partially blown-out skin leaves a speckled mask of the darker pixels,
    # which inflates texture far more than the clipping itself does.
    clipped_fraction = np.count_nonzero(skin & clipped) / max(1, np.count_nonzero(skin))
    face_width_px = int(face.points[:, 0].max() - face.points[:, 0].min())
    result = {
        "regions": {},
        "face_width_px": face_width_px,
        "low_resolution": face_width_px < LOW_RES_FACE_WIDTH,
        "overexposed": clipped_fraction > OVEREXPOSED_FRACTION,
    }
    if np.count_nonzero(skin) < MIN_SKIN_PIXELS:
        return {**result, "raw": {}, "scores": {}, "overall_skin": None}

    labels = np.zeros(lab.shape[:2], np.int32)
    for i, rmask in enumerate(region_masks.values(), start=1):
        labels[(rmask > 0) & (skin > 0)] = i
    maps = _maps(lab, labels)
    sel = labels > 0
    baseline_a = float(np.median(maps["a"][sel]))

    reliable_sel = np.zeros_like(sel)
    for name, rmask in region_masks.items():
        rsel = sel & (rmask > 0)
        if np.count_nonzero(rsel) < MIN_SKIN_PIXELS:
            continue
        stats = {k: round(v, 2) for k, v in _raw(maps, rsel, baseline_a).items()}
        coverage = np.count_nonzero(rsel) / max(1, np.count_nonzero(rmask))
        stats["skin_coverage"] = round(coverage, 2)
        stats["reliable"] = bool(coverage >= MIN_REGION_COVERAGE)
        if stats["reliable"]:
            reliable_sel |= rsel
        result["regions"][name] = stats
    if np.count_nonzero(reliable_sel) < MIN_SKIN_PIXELS:
        reliable_sel = sel  # nothing reliable: fall back to all skin

    # Overall metrics pool skin pixels from reliable regions (area-weighted).
    # A per-region median was tried and rejected: small regions are noisy and
    # the nose carries 3D shading, so it roughly doubled drift.
    raw = _raw(maps, reliable_sel, baseline_a)
    scores = score(raw)
    return {
        **result,
        "raw": {k: round(v, 2) for k, v in raw.items()},
        "scores": scores,
        "overall_skin": round(float(np.mean(list(scores.values()))), 1),
    }
