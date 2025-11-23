from typing import Tuple, Any
import threading

import numpy as np
import cv2
import onnxruntime as ort  # pastikan: pip install onnxruntime

from roop.utilities import conditional_download, resolve_relative_path

# ==========================
# Konfigurasi BiseNet
# ==========================

BISENET_SESSION: Any = None
BISENET_LOCK = threading.Lock()

# URL yang kamu kasih
BISENET_URL = (
    "https://huggingface.co/qualcomm/BiseNet/resolve/"
    "aeb57eda69d58721c5c186eb65b612dfa43faeab/BiseNet.onnx"
)
BISENET_FILENAME = "BiseNet.onnx"

# threshold: seberapa “campur aduk” label di dalam bbox
# semakin tinggi → semakin sensitif deteksi occlusion
OCCLUSION_MIX_THRESHOLD = 0.30  # 0.3 = cukup agresif


# ==========================
# Download & Load Model
# ==========================

def pre_check_bisenet() -> bool:
    """
    Download model BiseNet sekali saja kalau belum ada.
    Bisa dipanggil dari pre_check() face_swapper / enhancer.
    """
    download_dir = resolve_relative_path("../models")
    conditional_download(download_dir, [BISENET_URL])
    return True


def get_bisenet_session() -> ort.InferenceSession:
    """
    Lazy init onnxruntime session BiseNet, thread-safe.
    """
    global BISENET_SESSION

    with BISENET_LOCK:
        if BISENET_SESSION is None:
            model_path = resolve_relative_path(f"../models/{BISENET_FILENAME}")
            # CPUExecutionProvider aman untuk semua device
            BISENET_SESSION = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            print("✅ [BiseNet] ONNX session loaded")

    return BISENET_SESSION


# ==========================
# Util
# ==========================

def _preprocess_frame_for_bisenet(frame: np.ndarray, input_shape) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess:
    - BGR -> RGB
    - resize ke ukuran input model (kalau fixed)
    - normalisasi 0–1
    - HWC -> CHW
    """
    h_in = input_shape[2]
    w_in = input_shape[3]

    # Kalau shape dinamis (None / 'None') -> pakai 512x512 default
    if not isinstance(h_in, int) or not isinstance(w_in, int):
        h_in, w_in = 512, 512

    orig_h, orig_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # -> (1, 3, H, W)

    return img, orig_h, orig_w


def _run_bisenet(frame: np.ndarray) -> np.ndarray:
    """
    Jalankan BiseNet dan kembalikan peta label 2D (Hseg x Wseg, int32).
    """
    session = get_bisenet_session()
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape  # [1, 3, H, W] atau [1, 3, None, None]

    inp, _, _ = _preprocess_frame_for_bisenet(frame, input_shape)
    outputs = session.run(None, {input_name: inp})
    seg = outputs[0]

    # kemungkinan bentuk (1, C, H, W) atau (1, H, W)
    if seg.ndim == 4:
        # ambil argmax di channel C
        seg = np.argmax(seg, axis=1)[0]
    elif seg.ndim == 3:
        seg = seg[0]
    else:
        raise RuntimeError(f"[BiseNet] Unexpected output shape: {seg.shape}")

    return seg.astype(np.int32)


def _map_bbox_to_seg_coords(
    bbox, frame_shape: Tuple[int, int], seg_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Map bbox dari koordinat frame → koordinat map segmentasi.
    """
    fh, fw = frame_shape
    sh, sw = seg_shape

    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), fw - 1.0))
    y1 = max(0.0, min(float(y1), fh - 1.0))
    x2 = max(0.0, min(float(x2), fw - 1.0))
    y2 = max(0.0, min(float(y2), fh - 1.0))

    sx1 = int(x1 / fw * sw)
    sy1 = int(y1 / fh * sh)
    sx2 = int(x2 / fw * sw)
    sy2 = int(y2 / fh * sh)

    sx1 = max(0, min(sx1, sw - 1))
    sy1 = max(0, min(sy1, sh - 1))
    sx2 = max(0, min(sx2, sw))
    sy2 = max(0, min(sy2, sh))

    if sx2 <= sx1:
        sx2 = min(sx1 + 1, sw)
    if sy2 <= sy1:
        sy2 = min(sy1 + 1, sh)

    return sx1, sy1, sx2, sy2


def compute_occlusion_mix_ratio(frame: np.ndarray, bbox) -> float:
    """
    Mengukur 'kecampuran' kelas di dalam bbox wajah.
    Ide:
    - Kalau wajah bersih (tidak tertutup), pixel di area itu didominasi 1 label → mix kecil
    - Kalau tertutup tangan/objek, label jadi campur → mix besar
    Return: 0.0–1.0 (0 = bersih, 1 = campur banget)
    """
    seg = _run_bisenet(frame)
    fh, fw = frame.shape[:2]
    sh, sw = seg.shape[:2]

    sx1, sy1, sx2, sy2 = _map_bbox_to_seg_coords(bbox, (fh, fw), (sh, sw))
    region = seg[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return 0.0

    labels = region.reshape(-1)
    counts = np.bincount(labels)
    if counts.size == 0:
        return 0.0

    dominant = counts.max()
    total = labels.size
    dominant_ratio = float(dominant) / float(total)

    # makin kecil dominant_ratio → makin campur → occlusion tinggi
    mix_ratio = 1.0 - dominant_ratio
    return mix_ratio


# ==========================
# API untuk dipakai di face_swapper / enhancer
# ==========================

def is_occluded_bisenet(
    frame: np.ndarray,
    face,
    threshold: float = OCCLUSION_MIX_THRESHOLD
) -> bool:
    """
    Cek occlusion berdasarkan segmentasi BiseNet:
    - frame: BGR (OpenCV)
    - face: object insightface (punya .bbox) atau dict['bbox']
    """
    bbox = getattr(face, "bbox", None)
    if bbox is None and isinstance(face, dict):
        bbox = face.get("bbox", None)

    if bbox is None:
        return False

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except Exception:
        return False

    mix_ratio = 0.0
    try:
        mix_ratio = compute_occlusion_mix_ratio(frame, (x1, y1, x2, y2))
    except Exception as e:
        print(f"[BiseNet] Occlusion check failed: {e}")
        return False

    return mix_ratio >= threshold
