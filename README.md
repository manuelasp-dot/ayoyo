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
| **Skin** | [OpenCV](https://opencv.org/) image statistics (BSD) | Redness, smoothness, evenness, clarity, plus low-resolution / overexposure flags |

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

### How skin is measured

Skin metrics are designed to measure the *skin*, not the *photo*:

1. **Scale:** the face is resized to a fixed width, so texture is measured in
   face-relative units regardless of image resolution.
2. **Skin only:** within each region, only pixels close to the person's own
   skin color are used (beard, hair, glare excluded). Regions that are less
   than half skin are reported but left out of the overall score.
3. **Lighting removed:** a smooth brightness surface is fitted per region and
   subtracted, so side lighting, shading and exposure largely cancel out.
4. **Tone-independent redness:** redness is measured against the person's own
   baseline, so a naturally warmer or darker skin tone is not penalized —
   only patches redder than the rest of the face count.

Calibrated on 14 frontal photos of 5 people (varied skin tones, lighting,
resolution, beards). Re-scoring each photo after a resize, darkening or heavy
JPEG compression moves scores by ~1–5 points on average. Score anchors are
`ANCHORS` in `app/skin.py`.

### Known limitations

- **Overexposed photos:** blown-out highlights destroy skin detail. These are
  flagged (`overexposed`) and their skin scores are unreliable.
- **Small faces:** below ~180 px face width, results are flagged
  (`low_resolution`).
- **Beards and stubble** close to skin color can still pass the skin filter on
  the chin and read as texture or spots.
- Skin metrics are photo-based estimates, not a dermatological diagnosis.

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
  test_skin.py     skin-metric property tests on synthetic skin
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
