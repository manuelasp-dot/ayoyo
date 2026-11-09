"""Facial landmark detection using MediaPipe Face Mesh.

MediaPipe Face Mesh is Apache-2.0 licensed and free for commercial use. With
``refine_landmarks=True`` it returns 478 3D landmarks (the base 468 plus 10
iris points), matching the mesh density used by modern selfie-analysis apps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Named landmark indices into the MediaPipe Face Mesh (478-point) topology.
# These are stable, well-documented indices used throughout the pipeline.
IDX = {
    "face_top": 10,        # top of forehead region (not hairline)
    "chin": 152,           # bottom of chin
    "nose_bridge": 168,    # between the eyes
    "nose_tip": 1,
    "nose_bottom": 2,
    "nose_left": 48,       # left alar (nostril) edge
    "nose_right": 278,     # right alar (nostril) edge
    "face_left": 234,      # left edge of face oval
    "face_right": 454,     # right edge of face oval
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "left_brow": 105,
    "right_brow": 334,
}

# Symmetric landmark pairs (left, right) used for the symmetry metric.
SYMMETRIC_PAIRS = [
    (33, 263),    # eye outer corners
    (133, 362),   # eye inner corners
    (61, 291),    # mouth corners
    (48, 278),    # nostril edges
    (234, 454),   # face edges
    (105, 334),   # brows
    (205, 425),   # cheeks
    (58, 288),    # jaw
]

# Skin-sampling regions, as landmark index sets. Each region is the convex hull
# of its points, so point order does not matter. Regions avoid eyes, brows,
# nostrils and lips; masks are eroded further before measurement.
SKIN_REGIONS = {
    "forehead": [10, 109, 67, 103, 104, 69, 108, 151, 337, 299, 333, 332, 297, 338, 9],
    "left_cheek": [50, 101, 118, 117, 111, 116, 123, 147, 187, 205, 207],
    "right_cheek": [280, 330, 347, 346, 340, 345, 352, 376, 411, 425, 427],
    "nose": [168, 6, 197, 195, 5, 4, 1, 45, 275],
    "chin": [200, 199, 175, 152, 148, 377, 176, 400, 208, 428, 201, 421],
}


def region_hull(points_xy: np.ndarray, name: str) -> np.ndarray:
    """Convex hull (int32 polygon) of a skin region, from ``(N, 2)`` points."""
    import cv2

    pts = np.round(points_xy[SKIN_REGIONS[name]]).astype(np.int32)
    return cv2.convexHull(pts)


class NoFaceDetected(Exception):
    """Raised when no face can be located in the supplied image."""


@dataclass
class FaceLandmarks:
    """Detected landmarks for a single face.

    Attributes
    ----------
    points:
        ``(N, 3)`` float array of landmark coordinates in *pixel* units
        (x, y) with a relative z-depth in the third column.
    width, height:
        Source image dimensions in pixels.
    """

    points: np.ndarray
    width: int
    height: int

    def xy(self, index: int) -> np.ndarray:
        """Return the (x, y) pixel coordinate of a single landmark."""
        return self.points[index, :2]

    def named(self, name: str) -> np.ndarray:
        """Return the (x, y) pixel coordinate of a named landmark."""
        return self.xy(IDX[name])


_FACE_MESH = None


def _get_face_mesh():
    """Lazily construct a reusable MediaPipe FaceMesh instance."""
    global _FACE_MESH
    if _FACE_MESH is None:
        import mediapipe as mp

        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _FACE_MESH


def detect(image_bgr: np.ndarray) -> FaceLandmarks:
    """Detect 478 facial landmarks in a BGR image.

    Parameters
    ----------
    image_bgr:
        An OpenCV-style BGR image array.

    Raises
    ------
    NoFaceDetected
        If MediaPipe cannot find a face.
    """
    import cv2

    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = _get_face_mesh().process(rgb)
    if not results.multi_face_landmarks:
        raise NoFaceDetected("No face found in the image.")

    landmarks = results.multi_face_landmarks[0].landmark
    pts = np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
        dtype=np.float64,
    )
    return FaceLandmarks(points=pts, width=width, height=height)
