"""Render the face mesh and measurement guides onto the image.

Draws, on a copy of the input image:
  * all 478 MediaPipe landmarks
  * tinted skin-sampling regions
  * the facial midline
  * facial-thirds guides (estimated hairline, brow, nose base, chin)
  * eye-width / interocular and nose / mouth width guides

Returns a base64-encoded JPEG suitable for embedding as a data URI.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .harmony import _brow_midpoint, estimate_hairline
from .landmarks import SKIN_REGIONS, FaceLandmarks, region_hull

# BGR colours
REGION_COLOURS = {
    "forehead": (255, 180, 90),
    "left_cheek": (120, 220, 120),
    "right_cheek": (120, 220, 120),
    "nose": (90, 160, 255),
    "chin": (220, 120, 220),
}
DOT = (0, 220, 255)
MIDLINE = (255, 255, 255)
THIRDS = (255, 220, 0)
EYES = (80, 255, 80)
NOSE_MOUTH = (255, 80, 255)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _hline(img, y: float, x0: float, x1: float, colour, thickness: int, label: str | None = None) -> int:
    """Draw a horizontal guide; return the x pixel where its label ends."""
    cv2.line(img, (int(x0), int(y)), (int(x1), int(y)), colour, thickness, cv2.LINE_AA)
    if not label:
        return int(x1)
    scale, weight = 0.4 * thickness, max(1, thickness - 1)
    cv2.putText(img, label, (int(x1) + 4, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, weight, cv2.LINE_AA)
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    return int(x1) + 4 + text_w


def render(image_bgr: np.ndarray, face: FaceLandmarks, max_width: int = 720) -> str:
    """Return a base64 JPEG of the image with the mesh and guides drawn on."""
    img = image_bgr.copy()
    # Scale line/dot sizes with image resolution so overlays stay legible.
    t = max(1, int(round(face.width / 500)))

    # Tinted skin regions.
    tint = img.copy()
    for name in SKIN_REGIONS:
        poly = region_hull(face.points[:, :2], name)
        cv2.fillPoly(tint, [poly], REGION_COLOURS.get(name, (200, 200, 200)))
    cv2.addWeighted(tint, 0.22, img, 0.78, 0, img)

    # All landmarks.
    for x, y, _ in face.points:
        cv2.circle(img, (int(x), int(y)), t, DOT, -1, cv2.LINE_AA)

    # Midline (estimated hairline -> chin).
    hairline = estimate_hairline(face)
    chin = face.named("chin")
    cv2.line(img, _pt(hairline), _pt(chin), MIDLINE, t, cv2.LINE_AA)

    # Facial-thirds guides.
    x0 = face.named("face_left")[0]
    x1 = face.named("face_right")[0]
    brow = _brow_midpoint(face)
    nose_bottom = face.named("nose_bottom")
    label_right = max(
        _hline(img, y, x0, x1, THIRDS, t, label)
        for y, label in (
            (hairline[1], "hairline (est.)"),
            (brow[1], "brow"),
            (nose_bottom[1], "nose base"),
            (chin[1], "chin"),
        )
    )

    # Eye width + interocular.
    for a, b in (("left_eye_outer", "left_eye_inner"),
                 ("left_eye_inner", "right_eye_inner"),
                 ("right_eye_inner", "right_eye_outer")):
        cv2.line(img, _pt(face.named(a)), _pt(face.named(b)), EYES, t + 1, cv2.LINE_AA)

    # Nose and mouth widths.
    for a, b in (("nose_left", "nose_right"), ("mouth_left", "mouth_right")):
        cv2.line(img, _pt(face.named(a)), _pt(face.named(b)), NOSE_MOUTH, t + 1, cv2.LINE_AA)

    # Crop to the face (plus the estimated hairline) with padding for labels,
    # so the mesh stays readable even in wide or full-body shots.
    xs = np.append(face.points[:, 0], hairline[0])
    ys = np.append(face.points[:, 1], hairline[1])
    fw, fh = xs.max() - xs.min(), ys.max() - ys.min()
    left = int(max(0, xs.min() - 0.35 * fw))
    right = int(min(face.width, max(xs.max() + 0.35 * fw, label_right + 8)))
    top = int(max(0, ys.min() - 0.15 * fh))
    bottom = int(min(face.height, ys.max() + 0.15 * fh))
    img = img[top:bottom, left:right]

    # Downscale for transport.
    h, w = img.shape[:2]
    if w > max_width:
        s = max_width / w
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
