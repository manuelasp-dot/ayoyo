"""Assemble the full facial-analysis report from the component modules."""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import harmony, overlay, skin
from .landmarks import FaceLandmarks, detect


# "left"/"right" follow the photo (MediaPipe's image-space sides), not the person's.
REGION_NAMES = {
    "forehead": "forehead",
    "left_cheek": "cheek on the left of the photo",
    "right_cheek": "cheek on the right of the photo",
    "nose": "nose",
    "chin": "chin",
}


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
    regions = {n: r for n, r in skin_report.get("regions", {}).items() if r.get("reliable")}

    def worst(metric: str) -> str:
        if not regions:
            return "face"
        name = max(regions, key=lambda n: regions[n][metric])
        return REGION_NAMES.get(name, name)

    skin_tips = {
        "redness": "Some areas look redder than the rest of your skin, most on the {}.",
        "smoothness": "Fine texture is most visible on the {}.",
        "evenness": "Skin tone is least even on the {}.",
        "clarity": "Spots or marks are most noticeable on the {}.",
    }
    for metric, text in skin_tips.items():
        if skin_scores.get(metric, 100) < 60:
            tips.append(text.format(worst(metric)))

    if skin_report.get("low_resolution"):
        tips.append("The face is small in this photo, so skin results are less reliable — try a closer shot.")
    if skin_report.get("overexposed"):
        tips.append("Parts of the skin are blown out by bright light, so skin results are less reliable — try softer lighting.")
    if skin_report.get("overall_skin") is None:
        tips.append("Not enough clear skin was visible to analyze — try a photo without hair or glare covering the face.")

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

    parts = [harmony_report["overall_harmony"], skin_report["overall_skin"]]
    overall = round(float(np.mean([p for p in parts if p is not None])), 1)

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
