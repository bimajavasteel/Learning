# hand_occlusion.py
# Modul deteksi tangan untuk membantu occlusion detection pada face swap.
# Bisa memakai model ONNX (handseg.onnx) jika tersedia,
# atau fallback skin-color detection jika tidak ada model.

import cv2
import numpy as np
import os
import onnxruntime as ort
import roop.globals
from roop.utilities import resolve_relative_path

HAND_SESSION = None
HAND_INPUT_NAME = None
HAND_INPUT_SIZE = (256, 256)   # boleh diganti jika model berbeda


# ==========================================================
#  LOAD MODEL HAND SEGMENTATION
# ==========================================================
def _load_handseg_model():
    global HAND_SESSION, HAND_INPUT_NAME
    if HAND_SESSION is not None:
        return HAND_SESSION

    # path default: /kaggle/working/Learning/models/handseg.onnx
    model_rel = getattr(roop.globals, "handseg_model_path", "../models/handseg.onnx")
    model_path = resolve_relative_path(model_rel)

    if not os.path.exists(model_path):
        return None

    try:
        HAND_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        HAND_INPUT_NAME = HAND_SESSION.get_inputs()[0].name
        print(f"✅ [hand_occlusion] Loaded handseg.onnx: {model_path}")
        return HAND_SESSION
    except Exception as e:
        print(f"[hand_occlusion] Failed load handseg model: {e}")
        HAND_SESSION = None
        HAND_INPUT_NAME = None
        return None


# ==========================================================
#  ONNX HAND SEGMENTATION INFERENCE
# ==========================================================
def _onnx_hand_mask(crop: np.ndarray) -> np.ndarray:
    """
    Menghasilkan mask tangan (float 0..1) dari crop wajah.
    Jika ONNX tidak ada → return zero-mask.
    """
    session = _load_handseg_model()
    if session is None:
        return np.zeros((crop.shape[0], crop.shape[1]), dtype=np.float32)

    try:
        inp = cv2.resize(crop, HAND_INPUT_SIZE)
        inp = inp.astype("float32") / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]

        outs = session.run(None, {HAND_INPUT_NAME: inp})
        pred = outs[0]

        # Output bisa (1,1,H,W) atau (1,H,W)
        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred[0]

        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))
        mask = np.clip((mask - mask.min()) / (mask.max() - mask.min() + 1e-8), 0, 1)
        return mask.astype(np.float32)

    except Exception:
        return np.zeros((crop.shape[0], crop.shape[1]), dtype=np.float32)


# ==========================================================
#  SKIN-COLOR FALLBACK (dipakai saat ONNX tidak ada)
# ==========================================================
def _skin_color_mask(crop: np.ndarray) -> np.ndarray:
    """
    Deteksi area kulit menggunakan warna (fallback jika ONNX tidak tersedia).
    Cukup baik untuk mendeteksi tangan menutup wajah, tapi tidak seakurat model ONNX.
    """
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # threshold default – bisa kamu tuning di roop.globals
    cr_min = getattr(roop.globals, "skin_cr_min", 135)
    cr_max = getattr(roop.globals, "skin_cr_max", 180)
    cb_min = getattr(roop.globals, "skin_cb_min", 85)
    cb_max = getattr(roop.globals, "skin_cb_max", 135)

    mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))
    mask = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = mask.astype(np.float32) / 255.0
    return mask


# ==========================================================
#  REFINEMENT (hilangkan wajah yang ikut terbaca sebagai kulit)
# ==========================================================
def _refine_mask(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Kurangi false-positive pada area wajah yang memiliki edge kuat.
    (Wajah punya edge halus, tangan biasanya edge rendah pada area pipi)
    """
    try:
        edges = cv2.Canny(crop, 50, 150)
        edges_mask = (edges > 0).astype(np.float32)

        # kurangi mask pada area ber-edge tinggi
        refined = np.clip(mask - 0.25 * edges_mask, 0.0, 1.0)
        return refined.astype(np.float32)
    except Exception:
        return mask


# ==========================================================
#  FINAL API: get_hand_mask()
# ==========================================================
def get_hand_mask(crop: np.ndarray) -> np.ndarray:
    """
    API utama:
    - coba pakai handseg.onnx
    - fallback skin-color
    - refine mask supaya tidak salah deteksi wajah
    """
    if crop is None or crop.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    # coba ONNX dulu
    mask = _onnx_hand_mask(crop)

    # jika ONNX tidak ada / keluaran kosong → fallback skin color
    if mask is None or mask.size == 0 or np.mean(mask) < 0.001:
        mask = _skin_color_mask(crop)

    # refine mask
    mask = _refine_mask(crop, mask)

    return mask
