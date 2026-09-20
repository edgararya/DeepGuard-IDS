import os
import sys
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as tf_layers

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

from .classifiers import Meso4, MesoInception4

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_APP_DIR)
_WEIGHTS_DIR = os.path.join(_BACKEND_DIR, 'weights')

MESO_SIZE = 256
XCP_SIZE = 299
FAKE_THRESHOLD = 0.44

def _build_xception_model():
    """Build Xception single-frame model matching trained architecture."""
    base = keras.applications.Xception(weights=None, include_top=False, input_shape=(XCP_SIZE, XCP_SIZE, 3))
    x = base.output
    x = tf_layers.GlobalAveragePooling2D()(x)
    x = tf_layers.Dropout(0.5)(x)
    out = tf_layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=base.input, outputs=out)

clf_meso_df = None
clf_meso_f2f = None
clf_inception = None
clf_xception = None

# 1. Load Meso4_DF
try:
    p = os.path.join(_WEIGHTS_DIR, 'Meso4_DF.h5')
    if os.path.exists(p):
        clf_meso_df = Meso4()
        clf_meso_df.load(p)
        print("[OK] Meso4_DF loaded")
    else:
        print(f"[WARN] {p} not found")
except Exception as e:
    print(f"[WARN] Failed to load Meso4_DF: {e}")
    clf_meso_df = None

# 2. Load Meso4_F2F
try:
    p = os.path.join(_WEIGHTS_DIR, 'Meso4_F2F.h5')
    if os.path.exists(p):
        clf_meso_f2f = Meso4()
        clf_meso_f2f.load(p)
        print("[OK] Meso4_F2F loaded")
    else:
        print(f"[WARN] {p} not found")
except Exception as e:
    print(f"[WARN] Failed to load Meso4_F2F: {e}")
    clf_meso_f2f = None

# 3. Load MesoInception_DF
try:
    p = os.path.join(_WEIGHTS_DIR, 'MesoInception_DF.h5')
    if os.path.exists(p):
        clf_inception = MesoInception4()
        clf_inception.load(p)
        print("[OK] MesoInception_DF loaded")
    else:
        print(f"[WARN] {p} not found")
except Exception as e:
    print(f"[WARN] Failed to load MesoInception_DF: {e}")
    clf_inception = None

# 4. Load XceptionNet
try:
    p = os.path.join(_WEIGHTS_DIR, 'xception_finetuned.h5')
    if os.path.exists(p):
        clf_xception = _build_xception_model()
        clf_xception.load_weights(p)
        print("[OK] XceptionNet loaded")
    else:
        print(f"[WARN] {p} not found")
except Exception as e:
    print(f"[WARN] Failed to load XceptionNet: {e}")
    clf_xception = None

def get_model_status():
    loaded = {
        "mesonet_1": clf_meso_df is not None,
        "mesonet_2": clf_meso_f2f is not None,
        "mesonet_3": clf_inception is not None,
        "xception": clf_xception is not None,
    }
    warnings = []
    if not loaded["mesonet_1"]:
        warnings.append("Meso4_DF gagal diload")
    if not loaded["mesonet_2"]:
        warnings.append("Meso4_F2F gagal diload")
    if not loaded["mesonet_3"]:
        warnings.append("MesoInception_DF gagal diload")
    if not loaded["xception"]:
        warnings.append("XceptionNet gagal diload, pembobotan disesuaikan")

    active_count = sum(loaded.values())
    if active_count == 4:
        status = "ok"
    elif active_count > 0:
        status = "degraded"
    else:
        status = "error"
    return status, warnings, loaded

def run_inference(face_crop_bgr: np.ndarray) -> dict:
    """
    Run 4-model ensemble prediction on a face crop.
    Dynamically weights models based on availability.
    Returns dict:
      {
        "label": "real" | "fake",
        "confidence": float (0..1 fake probability),
        "per_model_scores": {
          "mesonet_1": float | None,   # Meso4_DF
          "mesonet_2": float | None,   # Meso4_F2F
          "mesonet_3": float | None,   # MesoInception_DF
          "xception": float | None     # XceptionNet
        },
        "warnings": list
      }
    """
    per_model_scores = {
        "mesonet_1": None,
        "mesonet_2": None,
        "mesonet_3": None,
        "xception": None
    }
    weighted_real_sum = 0.0
    total_weight = 0.0

    # MesoNet predictions (input 256x256, normalized 0..1)
    face_meso = cv2.resize(face_crop_bgr, (MESO_SIZE, MESO_SIZE))
    img_meso = face_meso.astype(np.float32) / 255.0
    img_meso = np.expand_dims(img_meso, axis=0)

    # 1. Meso4_DF -> mesonet_1 (weight 1.0)
    if clf_meso_df is not None:
        try:
            r = clf_meso_df.predict(img_meso)
            if r is not None and len(r) > 0:
                real_score = float(r[0][0])
                per_model_scores["mesonet_1"] = round(1.0 - real_score, 4)
                weighted_real_sum += real_score * 1.0
                total_weight += 1.0
        except Exception as e:
            print(f"[ERR] Inference Meso4_DF: {e}")

    # 2. Meso4_F2F -> mesonet_2 (weight 1.0)
    if clf_meso_f2f is not None:
        try:
            r = clf_meso_f2f.predict(img_meso)
            if r is not None and len(r) > 0:
                real_score = float(r[0][0])
                per_model_scores["mesonet_2"] = round(1.0 - real_score, 4)
                weighted_real_sum += real_score * 1.0
                total_weight += 1.0
        except Exception as e:
            print(f"[ERR] Inference Meso4_F2F: {e}")

    # 3. MesoInception_DF -> mesonet_3 (weight 2.0)
    if clf_inception is not None:
        try:
            r = clf_inception.predict(img_meso)
            if r is not None and len(r) > 0:
                real_score = float(r[0][0])
                per_model_scores["mesonet_3"] = round(1.0 - real_score, 4)
                weighted_real_sum += real_score * 2.0
                total_weight += 2.0
        except Exception as e:
            print(f"[ERR] Inference MesoInception_DF: {e}")

    # 4. XceptionNet -> xception (weight 3.0)
    if clf_xception is not None:
        try:
            face_xcp = cv2.resize(face_crop_bgr, (XCP_SIZE, XCP_SIZE))
            img_xcp = face_xcp.astype(np.float32) / 255.0
            img_xcp = np.expand_dims(img_xcp, axis=0)
            pred = clf_xception.predict(img_xcp, verbose=0)
            if pred is not None and len(pred) > 0:
                # Xception training: 0 = REAL, 1 = FAKE
                fake_raw = float(pred[0][0])
                per_model_scores["xception"] = round(fake_raw, 4)
                weighted_real_sum += (1.0 - fake_raw) * 3.0
                total_weight += 3.0
        except Exception as e:
            print(f"[ERR] Inference XceptionNet: {e}")

    if total_weight == 0:
        raise RuntimeError("ALL_MODELS_FAILED")

    avg_real = weighted_real_sum / total_weight
    confidence = round(1.0 - avg_real, 4)
    label = "fake" if confidence > FAKE_THRESHOLD else "real"

    _, warnings, _ = get_model_status()

    return {
        "label": label,
        "confidence": confidence,
        "per_model_scores": per_model_scores,
        "warnings": warnings
    }
