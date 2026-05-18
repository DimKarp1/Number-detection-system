from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from app.recognizer import (
    DEFAULT_DETECT_CONF,
    DEFAULT_DIGIT_CONF,
    DEFAULT_SEGMENT_CONF,
    DETECT_MODEL_PATH,
    DIGIT_MODEL_PATH,
    SEGMENT_MODEL_PATH,
    ML_DEVICE,
    process_image,
)


TMP_DIR = Path("tmp")
RUNS_DIR = Path("runs/api_debug")

TMP_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Competition Number Recognition ML Service")

detect_model: YOLO | None = None
segment_model: YOLO | None = None
digit_model: YOLO | None = None


@app.on_event("startup")
def load_models() -> None:
    global detect_model, segment_model, digit_model

    if not DETECT_MODEL_PATH.exists():
        raise RuntimeError(f"Detect model not found: {DETECT_MODEL_PATH}")

    if not SEGMENT_MODEL_PATH.exists():
        raise RuntimeError(f"Segment model not found: {SEGMENT_MODEL_PATH}")

    if not DIGIT_MODEL_PATH.exists():
        raise RuntimeError(f"Digit model not found: {DIGIT_MODEL_PATH}")

    print("Loading YOLO models...")

    detect_model = YOLO(str(DETECT_MODEL_PATH))
    segment_model = YOLO(str(SEGMENT_MODEL_PATH))
    digit_model = YOLO(str(DIGIT_MODEL_PATH))

    print("Models loaded.")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": {
            "number_detect": str(DETECT_MODEL_PATH),
            "number_segment": str(SEGMENT_MODEL_PATH),
            "digit_detect": str(DIGIT_MODEL_PATH),
        },
        "device": ML_DEVICE,
    }


@app.post("/recognize/image")
async def recognize_image(
    file: UploadFile = File(...),
) -> JSONResponse:
    if detect_model is None or segment_model is None or digit_model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded")

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type",
        )

    request_id = uuid.uuid4().hex

    input_path = TMP_DIR / f"{request_id}{suffix}"
    output_dir = RUNS_DIR / request_id

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_image(
            image_path=input_path,
            out_dir=output_dir,
            detect_model=detect_model,
            segment_model=segment_model,
            digit_model=digit_model,
            detect_conf=DEFAULT_DETECT_CONF,
            segment_conf=DEFAULT_SEGMENT_CONF,
            digit_conf=DEFAULT_DIGIT_CONF,
            max_candidates=8,
        )

        response = {
            "request_id": request_id,
            "recognized_numbers": result.get("recognized_numbers", []),
        }

        return JSONResponse(response)

    finally:
        if input_path.exists():
            input_path.unlink()
