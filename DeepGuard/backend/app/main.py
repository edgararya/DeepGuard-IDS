import base64
import binascii
import io
import os
import time
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .detector import run_inference, get_model_status, FAKE_THRESHOLD
from .face_utils import detect_and_crop_face
from .scoring import compute_ela_score, compute_blur_score
from .history import history_store
from .schemas import HealthResponse, DetectResponse, HistoryResponse

APP_VERSION = "0.1.0"
ALERT_THRESHOLD = 0.75
LIVE_SOURCE_DEFAULT = "desktop_screen"
SCAN_INTERVAL_MS = 1000

MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTS = {".mp4"}

ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png"}
ALLOWED_VIDEO_MIMES = {"video/mp4", "video/x-m4v"}
GENERIC_MIMES = {"", "application/octet-stream", "binary/octet-stream"}

IMAGE_SNIFF_FORMATS = {"JPEG", "MPO", "PNG"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    port = os.environ.get("PORT", "8000")
    print(f"[Xeptocore] Backend ready on port {port}", flush=True)
    yield


app = FastAPI(
    title="Xeptocore API",
    description="Backend API for Xeptocore deepfake detection",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _declared_mime_ok(content_type: Optional[str], allowed: set) -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    return ct in GENERIC_MIMES or ct in allowed


def _magic_image_format(contents: bytes) -> Optional[str]:
    for magic, name in (
        (b"\xff\xd8\xff", "JPEG"),
        (b"\x89PNG\r\n\x1a\n", "PNG"),
    ):
        if contents.startswith(magic):
            return name
    return None


def _identify_image(contents: bytes) -> Optional[str]:
    try:
        with Image.open(io.BytesIO(contents)) as im:
            return im.format
    except Exception:
        return _magic_image_format(contents)


def _sniff_video_container(contents: bytes) -> Optional[str]:
    if len(contents) < 12:
        return None
    head = contents[:12]
    if head[4:8] == b"ftyp":
        return "isobmff"
    return None


def _err(error_code: str, message: str, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "message": message, "detail": detail},
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_status, _, _ = get_model_status()
    return HealthResponse(
        status=model_status,
        models_loaded=(model_status != "error"),
        version=APP_VERSION,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    start_time = time.time()
    model_status, _, _ = get_model_status()

    try:
        contents = await file.read()
    except Exception:
        return _err("INVALID_FILE", "File tidak bisa dibaca", "Gagal membaca isi file upload")

    if not contents:
        return _err("INVALID_FILE", "File korup atau kosong", "Ukuran file 0 byte")

    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())

    if ext in ALLOWED_IMAGE_EXTS:
        media_type = "image"
        max_size = MAX_IMAGE_SIZE
        allowed_mimes = ALLOWED_IMAGE_MIMES
    elif ext in ALLOWED_VIDEO_EXTS:
        media_type = "video"
        max_size = MAX_VIDEO_SIZE
        allowed_mimes = ALLOWED_VIDEO_MIMES
    else:
        return _err(
            "UNSUPPORTED_FORMAT",
            "Format file tidak didukung",
            f"Hanya menerima jpg, png, atau mp4 (diterima: '{ext or 'tanpa ekstensi'}')",
        )

    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > max_size:
        return _err("INVALID_FILE", "File terlalu besar",
                    f"Ukuran file melebihi batas maksimal {max_size // (1024 * 1024)}MB")
    if len(contents) > max_size:
        return _err("INVALID_FILE", "File terlalu besar",
                    f"Ukuran file melebihi batas maksimal {max_size // (1024 * 1024)}MB")

    if not _declared_mime_ok(file.content_type, allowed_mimes):
        return _err("UNSUPPORTED_FORMAT", "Format file tidak didukung",
                    f"Tipe MIME '{file.content_type}' tidak sesuai untuk {media_type}")

    if media_type == "image":
        if _identify_image(contents) not in IMAGE_SNIFF_FORMATS:
            return _err("UNSUPPORTED_FORMAT", "Format file tidak didukung",
                        "Isi file bukan gambar yang valid (format tidak dikenali)")
    else:
        if _sniff_video_container(contents) is None:
            return _err("UNSUPPORTED_FORMAT", "Format file tidak didukung",
                        "Isi file bukan video mp4 yang valid (container tidak dikenali)")

    if model_status == "error":
        return _err("MODEL_LOAD_FAILED", "Model gagal dimuat",
                    "Tidak ada model yang berhasil dimuat saat startup", status.HTTP_503_SERVICE_UNAVAILABLE)

    if media_type == "image":
        result = _detect_image(contents, start_time)
    else:
        result = _detect_video(contents, ext, start_time)

    if isinstance(result, JSONResponse):
        return result

    history_store.add("file_upload", result["label"], result["confidence"])
    return result


def _detect_image(contents: bytes, start_time: float):
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            return _err("INVALID_FILE", "File tidak bisa dibaca", "Gambar gagal didecode")
    except Exception:
        return _err("INVALID_FILE", "File tidak bisa dibaca", "Gambar gagal didecode")

    face_crop, _ = detect_and_crop_face(img_bgr)
    if face_crop is None:
        return _err("INVALID_FILE", "Wajah tidak terdeteksi",
                    "Tidak ada wajah yang terdeteksi pada gambar")

    try:
        inf = run_inference(face_crop)
    except RuntimeError:
        return _err("MODEL_LOAD_FAILED", "Model gagal dimuat",
                    "Semua model gagal saat inferensi", status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception as e:
        return _err("INTERNAL_ERROR", "Error tak terduga",
                    f"Kesalahan saat inferensi: {e}", status.HTTP_500_INTERNAL_SERVER_ERROR)

    return {
        "label": inf["label"],
        "confidence": inf["confidence"],
        "per_model_scores": inf["per_model_scores"],
        "ela_score": compute_ela_score(face_crop),
        "blur_score": compute_blur_score(face_crop),
        "media_type": "image",
        "frames_analyzed": None,
        "processing_time_ms": int((time.time() - start_time) * 1000),
    }


def _detect_video(contents: bytes, ext: str, start_time: float):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_video.name
    try:
        temp_video.write(contents)
        temp_video.close()

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return _err("INVALID_FILE", "File tidak bisa dibaca", "Video gagal didecode")

        SAMPLE_INTERVAL = 10
        frame_idx = 0
        confidences = []
        ela_values = []
        blur_values = []
        per_model_accum = {"mesonet_1": [], "mesonet_2": [], "mesonet_3": [], "xception": []}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % SAMPLE_INTERVAL == 0:
                face_crop, _ = detect_and_crop_face(frame)
                if face_crop is not None:
                    try:
                        inf = run_inference(face_crop)
                        confidences.append(inf["confidence"])
                        for k in per_model_accum:
                            v = inf["per_model_scores"].get(k)
                            if v is not None:
                                per_model_accum[k].append(v)
                        ela_values.append(compute_ela_score(face_crop))
                        blur_values.append(compute_blur_score(face_crop))
                    except Exception as e:
                        print(f"[ERR] Frame {frame_idx} inference error: {e}")
            frame_idx += 1

        cap.release()

        analyzed = len(confidences)
        if analyzed == 0:
            return _err("INVALID_FILE", "Wajah tidak terdeteksi",
                        "Tidak ada wajah terdeteksi di sepanjang video")

        confidence = round(float(np.mean(confidences)), 4)
        label = "fake" if confidence > FAKE_THRESHOLD else "real"
        per_model_scores = {
            k: (round(float(np.mean(v)), 4) if v else None)
            for k, v in per_model_accum.items()
        }

        return {
            "label": label,
            "confidence": confidence,
            "per_model_scores": per_model_scores,
            "ela_score": round(float(np.mean(ela_values)), 4),
            "blur_score": round(float(np.mean(blur_values)), 4),
            "media_type": "video",
            "frames_analyzed": analyzed,
            "processing_time_ms": int((time.time() - start_time) * 1000),
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.websocket("/ws/live-scan")
async def live_scan(websocket: WebSocket, source: str = LIVE_SOURCE_DEFAULT):
    await websocket.accept()
    session_id = uuid.uuid4().hex[:6]
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "scan_interval_ms": SCAN_INTERVAL_MS,
    })

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "frame":
                continue
            for message in _process_live_frame(data, source):
                await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ERR] live-scan error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


def _process_live_frame(data: dict, source: str) -> list:
    sequence = data.get("sequence")
    timestamp = data.get("timestamp") or _now_ms()

    empty_scores = {"mesonet_1": None, "mesonet_2": None, "mesonet_3": None, "xception": None}
    neutral = {
        "type": "result",
        "sequence": sequence,
        "label": "real",
        "confidence": 0.0,
        "alert": False,
        "per_model_scores": empty_scores,
        "timestamp": timestamp,
    }

    image_b64 = data.get("image")
    if not image_b64:
        return [neutral]

    try:
        img_bytes = base64.b64decode(image_b64)
    except (binascii.Error, ValueError):
        return [neutral]

    if not img_bytes:
        return [neutral]

    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        return [neutral]

    face_crop, _ = detect_and_crop_face(img_bgr)
    if face_crop is None:
        return [neutral]

    try:
        inf = run_inference(face_crop)
    except Exception:
        return [neutral]

    alert = inf["confidence"] > ALERT_THRESHOLD
    result = {
        "type": "result",
        "sequence": sequence,
        "label": inf["label"],
        "confidence": inf["confidence"],
        "alert": alert,
        "per_model_scores": inf["per_model_scores"],
        "timestamp": timestamp,
    }

    messages = [result]
    if alert:
        entry = history_store.add(source, inf["label"], inf["confidence"], timestamp)
        messages.append({"type": "history_update", "entry": entry})
    return messages


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    return {"entries": history_store.list()}


@app.delete("/history/{entry_id}")
async def delete_history(entry_id: str):
    if history_store.delete(entry_id):
        return {"deleted": True, "id": entry_id}
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"deleted": False, "detail": f"Tidak ada entri dengan id '{entry_id}'"},
    )
