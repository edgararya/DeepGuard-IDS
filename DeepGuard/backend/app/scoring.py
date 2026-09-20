import cv2
import numpy as np

_ELA_JPEG_QUALITY = 90
_ELA_AMPLIFY = 15.0
_BLUR_VAR_REFERENCE = 200.0


def compute_ela_score(image_bgr: np.ndarray) -> float:
    """Error Level Analysis score in [0, 1].

    Re-saves the image as JPEG and measures the average recompression error,
    amplified for visibility. Higher values indicate more manipulation.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, _ELA_JPEG_QUALITY])
    if not ok or buf is None:
        return 0.0
    recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if recompressed is None:
        return 0.0
    diff = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
    ela = np.clip(diff * _ELA_AMPLIFY, 0.0, 255.0).astype(np.uint8)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    score = float(ela_gray.mean()) / 255.0
    return round(min(1.0, max(0.0, score)), 4)


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """Blurriness score in [0, 1] via variance of the Laplacian.

    0 means sharp, 1 means heavily blurred. Deepfake faces often score higher.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 1.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = 1.0 - min(1.0, float(variance) / _BLUR_VAR_REFERENCE)
    return round(min(1.0, max(0.0, score)), 4)
