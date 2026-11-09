"""Assemble the full facial-analysis report from the component modules."""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import harmony, overlay, skin
from .landmarks import FaceLandmarks, detect


def _grade(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "fair"
    return "needs attention"


def _suggestions(harmony_report: Dict, skin_report: Dict) -> list[str]:
    """Turn the lowest-scoring metrics into plain-language, kind suggestions."""
    tips: list[str] = []

    sym = harmony_report["symmetry"]["score"]
    if sym < 70:
        tips.append(
            "Facial symmetry reads a little uneven — often this is just head "
            "tilt or camera angle. Try a straight-on, eye-level photo."
        )

    skin_scores = skin_report.get("scores", {})
    if skin_scores.get("redness", 100) < 60:
        tips.append("Some redness detected — even, soft lighting reduces this in photos.")
    if skin_scores.get("smoothness", 100) < 60:
        tips.append("Skin texture reads high — this is very sensitive to harsh lighting.")
    if skin_scores.get("evenness", 100) < 60:
        tips.append("Uneven tone across regions — often a lighting artifact rather than skin.")

    if not tips:
        tips.append("All metrics look balanced. Remember these are geometric guides, not judgments.")
    return tips


def build(image_bgr: np.ndarray) -> Dict:
    """Run landmarks -> harmony -> skin and return the combined report.

    Raises ``landmarks.NoFaceDetected`` if no face is present.
    """
    face: FaceLandmarks = detect(image_bgr)

    harmony_report = harmony.analyze(face)
    skin_report = skin.analyze(image_bgr, face)

    overall = round(
        float(np.mean([harmony_report["overall_harmony"], skin_report["overall_skin"]])),
        1,
    )

    return {
        "image": {"width": face.width, "height": face.height},
        "landmark_count": int(face.points.shape[0]),
        "harmony": harmony_report,
        "skin": skin_report,
        "overall_score": overall,
        "grade": _grade(overall),
        "suggestions": _suggestions(harmony_report, skin_report),
        "overlay_jpeg_base64": overlay.render(image_bgr, face),
        "disclaimer": (
            "Ayoyo reports geometric and image-based metrics only. Facial-proportion "
            "canons are aesthetic conventions, not measures of worth or health, and "
            "skin metrics are lighting-sensitive estimates, not medical advice."
        ),
    }
