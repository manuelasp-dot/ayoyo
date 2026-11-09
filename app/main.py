"""Ayoyo FastAPI application.

Endpoints
---------
GET  /            -> minimal browser upload UI
GET  /health      -> liveness probe
POST /api/analyze -> multipart image upload, returns the JSON analysis report
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__, report
from .landmarks import NoFaceDetected

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Ayoyo Facial Analysis",
    version=__version__,
    description="Free, open-source facial harmony and skin analysis.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")

    buffer = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        result = report.build(image_bgr)
    except NoFaceDetected:
        raise HTTPException(status_code=422, detail="No face detected in the image.")

    return JSONResponse(result)
