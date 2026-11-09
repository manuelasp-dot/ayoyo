"""Facial harmony metrics: symmetry and classical proportions.

All measurements here are geometric and deterministic — derived directly from
landmark coordinates, with no trained "beauty" model involved. Each metric
reports a raw measured value, a reference ("ideal") value drawn from classical
facial-proportion canons, and a 0-100 score expressing how close the measured
value is to the reference.

These canons (facial thirds, facial fifths, the golden ratio) are aesthetic
conventions, not objective truth. Scores are a guide, not a verdict.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .landmarks import SYMMETRIC_PAIRS, FaceLandmarks

PHI = 1.6180339887  # golden ratio

# MediaPipe's top landmark (idx 10) sits at mid-forehead, not the hairline
# (trichion). We extrapolate upward along the brow->landmark-10 direction to
# estimate the true hairline. 1.6 is calibrated so a typical frontal selfie
# yields realistic facial-thirds ratios instead of a truncated upper third.
HAIRLINE_SCALE = 1.6


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _brow_midpoint(face: FaceLandmarks) -> np.ndarray:
    return (face.named("left_brow") + face.named("right_brow")) / 2


def estimate_hairline(face: FaceLandmarks) -> np.ndarray:
    """Estimate the trichion (hairline) point from the mesh.

    Landmark 10 is roughly mid-forehead, so we extend the brow->10 vector
    beyond landmark 10 by ``HAIRLINE_SCALE`` to approximate the hairline. This
    is an estimate: true hairline position is not recoverable from the mesh.
    """
    top = face.named("face_top")  # idx 10, mid-forehead
    brow = _brow_midpoint(face)
    return brow + (top - brow) * HAIRLINE_SCALE


def _closeness_score(measured: float, ideal: float, tolerance: float) -> float:
    """Map a measured/ideal pair to 0-100.

    ``tolerance`` is the relative deviation (fraction of ``ideal``) at which the
    score reaches 0. A measured value equal to ``ideal`` scores 100.
    """
    if ideal == 0:
        return 0.0
    rel = abs(measured - ideal) / abs(ideal)
    score = max(0.0, 1.0 - rel / tolerance)
    return round(score * 100, 1)


def symmetry(face: FaceLandmarks) -> Dict:
    """Score left/right symmetry about the facial midline.

    The midline is the best-fit axis through the forehead top, nose bridge,
    nose tip and chin. For each symmetric landmark pair we reflect the left
    point across the midline and measure how far it lands from the right point,
    normalised by face width.
    """
    top = face.named("face_top")
    chin = face.named("chin")

    # Midline as a unit vector from forehead to chin, plus a point on it.
    axis = chin - top
    axis_len = np.linalg.norm(axis)
    if axis_len == 0:
        return {"score": 0.0, "mean_offset_pct": None}
    axis_unit = axis / axis_len

    face_width = _dist(face.named("face_left"), face.named("face_right"))
    if face_width == 0:
        return {"score": 0.0, "mean_offset_pct": None}

    def reflect(p: np.ndarray) -> np.ndarray:
        # Reflect point p across the line through `top` with direction axis_unit.
        v = p - top
        proj = np.dot(v, axis_unit) * axis_unit
        perp = v - proj
        return top + proj - perp

    offsets = []
    for left_idx, right_idx in SYMMETRIC_PAIRS:
        left = face.xy(left_idx)
        right = face.xy(right_idx)
        reflected = reflect(left)
        offsets.append(_dist(reflected, right))

    mean_offset = float(np.mean(offsets))
    offset_pct = mean_offset / face_width  # 0 = perfect symmetry
    # 8% mean offset (relative to face width) maps to a score of 0.
    score = max(0.0, 1.0 - offset_pct / 0.08)
    return {
        "score": round(score * 100, 1),
        "mean_offset_pct": round(offset_pct * 100, 2),
    }


def proportions(face: FaceLandmarks) -> Dict[str, Dict]:
    """Measure classical facial proportions against reference canons."""
    out: Dict[str, Dict] = {}

    hairline = estimate_hairline(face)
    brow = _brow_midpoint(face)
    nose_bottom = face.named("nose_bottom")
    chin = face.named("chin")

    # --- Facial thirds: upper / middle / lower should be roughly equal (1:1:1).
    # Upper third runs from the *estimated* hairline, not landmark 10.
    upper = _dist(hairline, brow)
    middle = _dist(brow, nose_bottom)
    lower = _dist(nose_bottom, chin)
    total = upper + middle + lower
    if total > 0:
        thirds = [upper / total, middle / total, lower / total]
        # Deviation from the ideal 1/3 each.
        dev = float(np.mean([abs(t - 1 / 3) for t in thirds]))
        out["facial_thirds"] = {
            "measured": [round(t, 3) for t in thirds],
            "ideal": [0.333, 0.333, 0.333],
            # Mean deviation of 0.18 per third maps to 0 (calibrated on real faces).
            "score": max(0.0, round((1.0 - dev / 0.18) * 100, 1)),
            "note": "upper third uses an estimated hairline",
        }

    # --- Facial fifths: face width should be ~5 eye-widths.
    face_width = _dist(face.named("face_left"), face.named("face_right"))
    eye_width = _dist(face.named("left_eye_outer"), face.named("left_eye_inner"))
    if eye_width > 0:
        fifths = face_width / eye_width
        out["facial_fifths"] = {
            "measured": round(fifths, 2),
            "ideal": 5.0,
            "score": _closeness_score(fifths, 5.0, 0.25),
        }

    # --- Interocular spacing: gap between eyes should equal one eye-width.
    inter = _dist(face.named("left_eye_inner"), face.named("right_eye_inner"))
    if eye_width > 0:
        ratio = inter / eye_width
        out["eye_spacing"] = {
            "measured": round(ratio, 2),
            "ideal": 1.0,
            # Real faces commonly range ~0.9-1.3 eye-widths; 0.55 keeps
            # normal variation out of the "needs attention" band.
            "score": _closeness_score(ratio, 1.0, 0.55),
        }

    # --- Nose-to-mouth width: mouth width ~= 1.618 x nose width (golden ratio).
    nose_w = _dist(face.named("nose_left"), face.named("nose_right"))
    mouth_w = _dist(face.named("mouth_left"), face.named("mouth_right"))
    if nose_w > 0:
        ratio = mouth_w / nose_w
        out["mouth_nose_ratio"] = {
            "measured": round(ratio, 2),
            "ideal": round(PHI, 2),
            "score": _closeness_score(ratio, PHI, 0.3),
        }

    return out


def analyze(face: FaceLandmarks) -> Dict:
    """Run the full harmony analysis and return a structured report."""
    sym = symmetry(face)
    props = proportions(face)

    prop_scores = [v["score"] for v in props.values() if v.get("score") is not None]
    components = [sym["score"]] + prop_scores
    overall = round(float(np.mean(components)), 1) if components else 0.0

    return {
        "symmetry": sym,
        "proportions": props,
        "overall_harmony": overall,
    }
