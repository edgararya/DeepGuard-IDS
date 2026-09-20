import io
import os
import time
import tempfile
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .detector import run_inference, get_model_status
from .face_utils import detect_and_crop_face
from .schemas import HealthResponse, ImageAnalysisResponse, VideoAnalysisResponse

app = FastAPI(
    title="DeepGuard API",
    description="Backend API for DeepGuard Deepfake Detection System",
    version="2.0.0"
)

# Enable CORS for all origins (Electron / browser).
# allow_credentials must stay False: browsers reject a wildcard origin combined
# with credentials, so that pair is invalid per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/jpg", "image/pjpeg", "image/png",
    "image/webp", "image/bmp", "image/x-ms-bmp",
}
ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/x-m4v", "video/quicktime", "video/x-msvideo",
    "video/avi", "video/msvideo", "video/x-matroska", "video/webm",
}
# Postman / Electron uploads frequently omit a specific content type.
GENERIC_MIMES = {"", "application/octet-stream", "binary/octet-stream"}

# Pillow formats accepted as "this really is an image".
IMAGE_SNIFF_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP", "DIB"}


def _declared_mime_ok(content_type: Optional[str], allowed: set) -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    return ct in GENERIC_MIMES or ct in allowed


def _magic_image_format(contents: bytes) -> Optional[str]:
    """Signature-only identification, used when Pillow cannot parse the header."""
    for magic, name in (
        (b"\xff\xd8\xff", "JPEG"),
        (b"\x89PNG\r\n\x1a\n", "PNG"),
        (b"GIF87a", "GIF"),
        (b"GIF89a", "GIF"),
        (b"II*\x00", "TIFF"),
        (b"MM\x00*", "TIFF"),
        (b"BM", "BMP"),
    ):
        if contents.startswith(magic):
            return name
    if contents[:4] == b"RIFF" and contents[8:12] == b"WEBP":
        return "WEBP"
    return None


def _identify_image(contents: bytes) -> Optional[str]:
    """Identify the real image format.

    Truncation-tolerant by design: this answers 'is this really an image?',
    not 'is it intact?'. A recognizable container with a damaged payload must
    reach the decode step (CORRUPT_FILE) rather than be rejected here.
    """
    try:
        with Image.open(io.BytesIO(contents)) as im:
            return im.format
    except Exception:
        return _magic_image_format(contents)


def _sniff_video_container(contents: bytes) -> Optional[str]:
    """Magic-byte sniff for the container formats we accept."""
    if len(contents) < 12:
        return None
    head = contents[:12]
    if head[4:8] == b"ftyp":                      # ISO-BMFF: mp4 / mov / m4v
        return "isobmff"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if head[:4] == b"\x1a\x45\xdf\xa3":           # EBML: mkv / webm
        return "ebml"
    return None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    start_time = time.time()
    model_status, warnings, models_loaded = get_model_status()
    proc_time = int((time.time() - start_time) * 1000)

    msg = "Server is running"
    if model_status == "partial":
        msg = "Server is running with partial models"
    elif model_status == "error":
        msg = "Server is running but all models failed to load"

    return HealthResponse(
        success=True,
        message=msg,
        processing_time_ms=proc_time,
        status=model_status,
        warnings=warnings,
        models_loaded=models_loaded
    )

@app.post("/analyze/image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    start_time = time.time()
    model_status, model_warnings, _ = get_model_status()

    def err_res(msg: str, err_code: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        proc_time = int((time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "message": msg,
                "processing_time_ms": proc_time,
                "warnings": model_warnings,
                "result": None,
                "error": err_code
            }
        )

    # 1. Size check
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > MAX_IMAGE_SIZE:
        return err_res(
            f"Ukuran file melebihi batas maksimal 10MB ({round(declared_size/(1024*1024), 2)}MB)",
            "FILE_TOO_LARGE"
        )

    try:
        contents = await file.read()
    except Exception:
        return err_res("Gagal membaca file upload", "CORRUPT_FILE")

    if len(contents) > MAX_IMAGE_SIZE:
        return err_res(
            f"Ukuran file melebihi batas maksimal 10MB ({round(len(contents)/(1024*1024), 2)}MB)",
            "FILE_TOO_LARGE"
        )

    # 2. Extension + MIME check
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_IMAGE_EXTS:
        return err_res(
            f"Ekstensi file '{ext}' tidak didukung. Format didukung: {', '.join(sorted(ALLOWED_IMAGE_EXTS))}",
            "INVALID_FILE_TYPE"
        )

    if not _declared_mime_ok(file.content_type, ALLOWED_IMAGE_MIMES):
        return err_res(
            f"Tipe MIME '{file.content_type}' tidak didukung untuk gambar",
            "INVALID_FILE_TYPE"
        )

    if not contents:
        return err_res("File gambar kosong atau rusak", "CORRUPT_FILE")

    if _identify_image(contents) not in IMAGE_SNIFF_FORMATS:
        return err_res(
            "Isi file bukan gambar yang valid (format tidak dikenali)",
            "INVALID_FILE_TYPE"
        )

    # 3. Decode check
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            return err_res("File gambar rusak atau tidak dapat di-decode", "CORRUPT_FILE")
    except Exception:
        return err_res("Gagal mendekode gambar", "CORRUPT_FILE")

    # 4. Model availability check
    if model_status == "error":
        return err_res("Sistem tidak dapat melakukan inferensi, semua model gagal diload", "ALL_MODELS_FAILED", status.HTTP_503_SERVICE_UNAVAILABLE)

    # 5. Face detection check
    face_crop, face_box = detect_and_crop_face(img_bgr)
    if face_crop is None:
        return err_res("Wajah tidak terdeteksi pada gambar", "NO_FACE_DETECTED")

    # 6. Run Inference
    try:
        inf_result = run_inference(face_crop)
    except RuntimeError as re:
        if str(re) == "ALL_MODELS_FAILED":
            return err_res("Sistem tidak dapat melakukan inferensi", "ALL_MODELS_FAILED", status.HTTP_503_SERVICE_UNAVAILABLE)
        return err_res(f"Kesalahan inferensi: {re}", "CORRUPT_FILE")
    except Exception as e:
        return err_res(f"Terjadi kesalahan saat inferensi: {e}", "CORRUPT_FILE")

    proc_time = int((time.time() - start_time) * 1000)
    return {
        "success": True,
        "message": "Analisis gambar berhasil",
        "processing_time_ms": proc_time,
        "warnings": model_warnings,
        "result": {
            "has_face": True,
            "fake_prob": inf_result["fake_prob"],
            "verdict": inf_result["verdict"],
            "scores": inf_result["scores"],
            "face_box": face_box
        },
        "error": None
    }

@app.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    start_time = time.time()
    model_status, model_warnings, _ = get_model_status()

    def err_res(msg: str, err_code: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        proc_time = int((time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "message": msg,
                "processing_time_ms": proc_time,
                "warnings": model_warnings,
                "result": None,
                "error": err_code
            }
        )

    # 1. Size check
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > MAX_VIDEO_SIZE:
        return err_res(
            f"Ukuran file melebihi batas maksimal 100MB ({round(declared_size/(1024*1024), 2)}MB)",
            "FILE_TOO_LARGE"
        )

    try:
        contents = await file.read()
    except Exception:
        return err_res("Gagal membaca file upload", "CORRUPT_FILE")

    if len(contents) > MAX_VIDEO_SIZE:
        return err_res(
            f"Ukuran file melebihi batas maksimal 100MB ({round(len(contents)/(1024*1024), 2)}MB)",
            "FILE_TOO_LARGE"
        )

    # 2. Extension + MIME check
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_VIDEO_EXTS:
        return err_res(
            f"Ekstensi file '{ext}' tidak didukung. Format didukung: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}",
            "INVALID_FILE_TYPE"
        )

    if not _declared_mime_ok(file.content_type, ALLOWED_VIDEO_MIMES):
        return err_res(
            f"Tipe MIME '{file.content_type}' tidak didukung untuk video",
            "INVALID_FILE_TYPE"
        )

    if _sniff_video_container(contents) is None:
        return err_res(
            "Isi file bukan video yang valid (container tidak dikenali)",
            "INVALID_FILE_TYPE"
        )

    # 3. Model availability check
    if model_status == "error":
        return err_res("Sistem tidak dapat melakukan inferensi, semua model gagal diload", "ALL_MODELS_FAILED", status.HTTP_503_SERVICE_UNAVAILABLE)

    # 4. Save to temp and open via cv2.VideoCapture
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_video.name
    try:
        temp_video.write(contents)
        temp_video.close()

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return err_res("File tidak dapat dibaca atau rusak", "CORRUPT_FILE")

        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = round(float(fps_val), 1) if (fps_val and fps_val > 0 and not np.isnan(fps_val)) else 30.0
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frame_count <= 0:
            duration = 0.0
        else:
            duration = round(total_frame_count / fps, 1)

        fake_frame_count = 0
        real_frame_count = 0
        no_face_count = 0
        fake_probs = []

        frame_idx = 0
        SAMPLE_INTERVAL = 10

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % SAMPLE_INTERVAL == 0:
                face_crop, _ = detect_and_crop_face(frame)
                if face_crop is None:
                    no_face_count += 1
                else:
                    try:
                        inf = run_inference(face_crop)
                        fp = inf["fake_prob"]
                        fake_probs.append(fp)
                        if inf["verdict"] == "FAKE":
                            fake_frame_count += 1
                        else:
                            real_frame_count += 1
                    except Exception as e:
                        print(f"[ERR] Frame {frame_idx} inference error: {e}")

            frame_idx += 1

        cap.release()

        total_analyzed = fake_frame_count + real_frame_count
        if total_analyzed == 0:
            return err_res("Tidak ada wajah terdeteksi di sepanjang video", "NO_FACE_DETECTED_IN_VIDEO")

        avg_fake = round(float(np.mean(fake_probs)), 1)
        max_fake = round(float(np.max(fake_probs)), 1)
        verdict = "FAKE" if fake_frame_count > real_frame_count else "REAL"

        proc_time = int((time.time() - start_time) * 1000)
        return {
            "success": True,
            "message": "Analisis video berhasil",
            "processing_time_ms": proc_time,
            "warnings": model_warnings,
            "result": {
                "fps": fps,
                "duration_seconds": duration,
                "total_frames_analyzed": total_analyzed,
                "fake_frame_count": fake_frame_count,
                "real_frame_count": real_frame_count,
                "no_face_count": no_face_count,
                "avg_fake_prob": avg_fake,
                "max_fake_prob": max_fake,
                "verdict": verdict
            },
            "error": None
        }

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

