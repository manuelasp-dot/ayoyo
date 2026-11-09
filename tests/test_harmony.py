"""Unit tests for the harmony geometry — pure numpy, no MediaPipe needed.

We build a synthetic, near-ideal face (vertical midline, equal facial thirds,
facial fifths = 5, interocular = one eye-width, mouth:nose = golden ratio) and
assert that the metrics score it highly, then perturb it and assert the scores
drop.
"""

import numpy as np

from app import harmony
from app.landmarks import FaceLandmarks

# (x, y) placements that satisfy the classical canons about a midline at x=100.
IDEAL = {
    10: (100, 10),      # face_top
    152: (100, 210),    # chin
    168: (100, 60),     # nose_bridge
    1: (100, 130),      # nose_tip
    2: (100, 143.33),   # nose_bottom
    48: (90, 140), 278: (110, 140),          # nostrils (nose width 20)
    234: (20, 110), 454: (180, 110),         # face edges (width 160)
    33: (52, 76), 133: (84, 76),             # left eye (width 32)
    362: (116, 76), 263: (148, 76),          # right eye
    61: (83.82, 170), 291: (116.18, 170),    # mouth (width 32.36 = 20*phi)
    13: (100, 165), 14: (100, 175),          # mouth top/bottom
    105: (68, 76.67), 334: (132, 76.67),     # brows
    205: (50, 130), 425: (150, 130),         # cheeks
    58: (40, 170), 288: (160, 170),          # jaw
}


def _make_face(overrides=None) -> FaceLandmarks:
    pts = np.zeros((478, 3), dtype=np.float64)
    coords = dict(IDEAL)
    if overrides:
        coords.update(overrides)
    for idx, (x, y) in coords.items():
        pts[idx] = (x, y, 0.0)
    return FaceLandmarks(points=pts, width=200, height=220)


def test_ideal_face_scores_high():
    report = harmony.analyze(_make_face())
    assert report["symmetry"]["score"] >= 95
    assert report["proportions"]["facial_fifths"]["score"] >= 95
    assert report["proportions"]["eye_spacing"]["score"] >= 95
    assert report["proportions"]["mouth_nose_ratio"]["score"] >= 90
    assert report["overall_harmony"] >= 85


def test_asymmetry_lowers_symmetry_score():
    # Shove the left eye and mouth corner inward to break the midline mirror.
    skewed = _make_face({33: (70, 76), 61: (95, 170)})
    report = harmony.analyze(skewed)
    assert report["symmetry"]["score"] < 95


def test_closeness_score_bounds():
    assert harmony._closeness_score(5.0, 5.0, 0.25) == 100.0
    assert harmony._closeness_score(10.0, 5.0, 0.25) == 0.0
    mid = harmony._closeness_score(5.625, 5.0, 0.25)  # 12.5% off, half tolerance
    assert 45 <= mid <= 55
