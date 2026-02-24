"""
Egyptian ID OCR — FastAPI Service
===================================
Endpoints:
    GET  /health          → Model status
    POST /ocr/full-card   → Full card → YOLO detect → OCR all fields
    POST /ocr/single-field → Single cropped field → OCR
    POST /ocr/batch       → Batch processing (max 20)
    GET  /metrics         → Latency/confidence stats
"""

import sys, time, json, logging
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── Resolve imports ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference import EgyptianIDOCR

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "ocr_service.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("egyptian_id_ocr")

# ── Model paths ───────────────────────────────────────────────
DET_ONNX = str(ROOT / "model" / "field_detector.onnx")
REC_ONNX = str(ROOT / "onnx" / "rec_sim.onnx")
DICT_PATH = str(ROOT / "arabic_dict.txt")


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading OCR model...")
    app.state.ocr = EgyptianIDOCR(
        det_onnx=DET_ONNX,
        rec_onnx=REC_ONNX,
        dict_path=DICT_PATH,
        use_gpu=False,
    )
    logger.info("✅ Model ready")
    yield
    logger.info("🛑 Shutting down")


app = FastAPI(
    title="Egyptian ID OCR API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────
def decode_image(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img


def log_request(endpoint: str, latency_ms: float,
                n_fields: int, avg_conf: float):
    logger.info(json.dumps({
        "endpoint": endpoint,
        "latency_ms": round(latency_ms, 1),
        "fields": n_fields,
        "avg_conf": round(avg_conf, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Check API and model status."""
    return {
        "status": "healthy",
        "model": "EgyptianIDOCR v1.0",
        "device": "CPU",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ocr/full-card")
async def ocr_full_card(file: UploadFile = File(...)):
    """
    Extract all fields from a full ID card image.
    Uses YOLO for detection then PaddleOCR for recognition.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(415, "Only JPEG/PNG supported")

    t0 = time.perf_counter()
    img = decode_image(await file.read())

    try:
        results = app.state.ocr.extract(crops=None, id_card_path="")
        # Need to pass image directly
        fields = app.state.ocr.detector.detect(img)
        if not fields:
            raise HTTPException(422, "No fields detected in image")

        crops = {}
        for f in fields:
            x1, y1, x2, y2 = f["bbox"]
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops[f["class_name"]] = crop

        results = app.state.ocr.extract("", crops=crops)
    except ValueError as e:
        raise HTTPException(422, str(e))

    elapsed = (time.perf_counter() - t0) * 1000

    response = {}
    confs = []
    for field_name, result in results.items():
        response[field_name] = {
            "text": result.text,
            "confidence": result.confidence,
            "valid": result.valid,
        }
        confs.append(result.confidence)

    avg_conf = sum(confs) / len(confs) if confs else 0.0
    log_request("/ocr/full-card", elapsed, len(results), avg_conf)

    return {
        "status": "success",
        "latency_ms": round(elapsed, 1),
        "fields": response,
        "avg_confidence": round(avg_conf, 3),
    }


@app.post("/ocr/single-field")
async def ocr_single_field(
    file: UploadFile = File(...),
    field: str = Query("name", description="Field name"),
):
    """Extract text from a single cropped field image."""
    t0 = time.perf_counter()
    img = decode_image(await file.read())
    text, conf = app.state.ocr.recognize(img)
    valid = app.state.ocr._validate(field, text)
    elapsed = (time.perf_counter() - t0) * 1000

    log_request("/ocr/single-field", elapsed, 1, conf)

    return {
        "field": field,
        "text": text,
        "confidence": round(conf, 3),
        "valid": valid,
        "latency_ms": round(elapsed, 1),
    }


@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...)):
    """Process multiple cropped field images in one request."""
    if len(files) > 20:
        raise HTTPException(400, "Max 20 images per batch")

    t0 = time.perf_counter()
    results = []

    for f in files:
        try:
            img = decode_image(await f.read())
            text, conf = app.state.ocr.recognize(img)
            results.append({
                "filename": f.filename,
                "text": text,
                "confidence": round(conf, 3),
                "status": "success",
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "status": f"error: {e}",
            })

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "total": len(files),
        "success": sum(1 for r in results if r["status"] == "success"),
        "latency_ms": round(elapsed, 1),
        "results": results,
    }


@app.get("/metrics")
def metrics():
    """Usage statistics from the log."""
    try:
        log_file = ROOT / "ocr_service.log"
        if not log_file.exists():
            return {"message": "No requests yet"}

        with open(log_file) as f:
            lines = [l for l in f if '"endpoint"' in l]

        logs = []
        for l in lines[-1000:]:
            try:
                payload = l.split(" | INFO | ")[-1]
                logs.append(json.loads(payload))
            except (json.JSONDecodeError, IndexError):
                continue

        if not logs:
            return {"message": "No requests yet"}

        import statistics
        latencies = [l["latency_ms"] for l in logs]
        return {
            "total_requests": len(logs),
            "avg_latency_ms": round(statistics.mean(latencies), 1),
            "p95_latency_ms": round(
                sorted(latencies)[int(len(latencies) * 0.95)], 1
            ),
            "avg_confidence": round(
                sum(l.get("avg_conf", 0) for l in logs) / len(logs), 3
            ),
        }
    except Exception as e:
        return {"error": str(e)}
