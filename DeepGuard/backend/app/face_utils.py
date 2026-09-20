import cv2
import numpy as np
from typing import Tuple, Optional, List

# Load Haar cascade once
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_and_crop_face(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    Detect largest face from frame_bgr, apply CLAHE contrast enhancement,
    and return face_crop with 35% margin along with [x, y, w, h] bounding box.
    Returns (None, None) if no face detected or face is too small (<20px).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None, None

    ih, iw = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    faces = _face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.03,
        minNeighbors=1,
        minSize=(15, 15),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None, None

    # Pick largest face
    best_face = max(faces, key=lambda r: r[2] * r[3])
    fx, fy, fw, fh = best_face
    cx = fx + fw // 2
    cy = fy + fh // 2
    side = max(fw, fh)
    half_side = int((side // 2) * 1.35)  # 35% margin

    top = max(0, cy - half_side)
    bottom = min(ih, cy + half_side)
    left = max(0, cx - half_side)
    right = min(iw, cx + half_side)

    face_crop = frame_bgr[top:bottom, left:right]
    if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
        return None, None

    face_box = [int(left), int(top), int(right - left), int(bottom - top)]
    return face_crop, face_box
