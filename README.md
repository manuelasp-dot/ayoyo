# Ayoyo

Free, open-source facial analysis — a Qoves-style pipeline built entirely from
permissively licensed, commercial-use-friendly components. Runs **locally and
offline**; no paid cloud vision API.

## What it does

Upload a selfie and get a structured report with three layers:

| Layer | How | Output |
|-------|-----|--------|
| **Landmarks** | [MediaPipe Face Mesh](https://github.com/google-ai-edge/mediapipe) (Apache-2.0) | 478 3D points |
| **Harmony** | Our own geometry on the landmarks | Symmetry + classical proportion scores (facial thirds, fifths, eye spacing, golden-ratio mouth/nose) |
| **Skin** | [OpenCV](https://opencv.org/) color/texture heuristics (BSD) | Redness, smoothness, evenness, clarity |

Each metric reports a measured value and a 0–100 score vs. a reference canon.

The report also includes `overlay_jpeg_base64`: a face-cropped image showing the
478-point mesh, midline, facial-thirds guides, eye/nose/mouth widths and the
tinted skin regions.

### Calibration notes

- **Hairline is estimated.** MediaPipe's top landmark sits mid-forehead, so the
  upper facial third uses an extrapolated hairline (`HAIRLINE_SCALE` in
  `app/harmony.py`). Hair covering the forehead will skew this metric.
- **Tolerances** (the deviation at which a score hits 0) are tuned so normal
  facial variation scores in the fair–good range rather than near 0. They are
  constants in `app/harmony.py` and easy to adjust.

## Licensing

Every dependency is free **and** commercial-use friendly (Apache-2.0 / BSD /
MIT). No SCUT-FBP5500 "beauty score" model is used, so there is no
non-commercial-dataset restriction and less demographic bias baggage.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

## API

```bash
curl -F "image=@selfie.jpg" http://127.0.0.1:8000/api/analyze
```

Returns JSON:

```jsonc
{
  "overall_score": 78.4,
  "grade": "good",
  "harmony": { "symmetry": {...}, "proportions": {...}, "overall_harmony": 80.1 },
  "skin":    { "scores": {...}, "regions": {...}, "overall_skin": 76.7 },
  "suggestions": ["..."],
  "disclaimer": "..."
}
```

## Project layout

```
app/
  landmarks.py   MediaPipe Face Mesh wrapper (478 pts, named indices)
  harmony.py     symmetry + proportion geometry
  skin.py        OpenCV skin metrics per face region
  overlay.py     draws the mesh + measurement guides onto the image
  report.py      assembles the combined report + suggestions
  main.py        FastAPI app (/, /health, /api/analyze)
static/
  index.html     minimal browser upload UI
tests/
  test_harmony.py  geometry unit tests (no MediaPipe required)
```

## Tests

```bash
pip install pytest
pytest
```

## Ethics

Facial-proportion canons are aesthetic conventions, not measures of worth or
health. Skin metrics are lighting-sensitive estimates, not medical advice.
Ayoyo deliberately omits a trained "attractiveness score."
