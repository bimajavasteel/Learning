import threading
from typing import Optional, List, Tuple

import numpy as np
import cv2
import onnxruntime as ort

from roop.utilities import conditional_download, resolve_relative_path

# ============================================
# Konfigurasi model
# ============================================

_FACE_PARSING_SESSION: Optional[ort.InferenceSession] = None
_FACE_PARSING_LOCK = threading.Lock()

# ONNX yang kamu kasih
FACE_PARSING_URL = (
    "https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/"
    "releases/download/ckpts_and_onnx/79999_iter.onnx"
)

# Nama file lokal di folder ../models
FACE_PARSING_FILENAME = "79999_iter.onnx"

# Label yang dianggap area wajah (CelebAMaskHQ, 19 kelas)
FACE_LABELS: List[int] = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # skin, brow, eye, nose, lip


# ============================================
# Download + load ONNX
# ============================================

def pre_check_face_parsing() -> bool:
    """
    Download model parsing jika belum ada.
    """
    download_dir = resolve_relative_path("../models")
    # conditional_download pakai nama file dari URL (79999_iter.onnx)
    conditional_download(download_dir, [FACE_PARSING_URL])
    return True


def get_face_parsing_session() -> ort.InferenceSession:
    """
    Lazy-init ONNXRuntime session dengan CUDA kalau tersedia.
    """
    global _FACE_PARSING_SESSION

    with _FACE_PARSING_LOCK:
        if _FACE_PARSING_SESSION is None:
            model_path = resolve_relative_path(f"../models/{FACE_PARSING_FILENAME}")
            _FACE_PARSING_SESSION = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print("✅ [FaceParsing] CelebAMaskHQ ONNX loaded")
    return _FACE_PARSING_SESSION


# ============================================
# Util preprocess & koordinat
# ============================================

def _preprocess_frame(frame: np.ndarray, input_shape) -> np.ndarray:
    """
    BGR -> RGB, resize ke input model, normalisasi, CHW, add batch.
    """
    h_in = input_shape[2]
    w_in = input_shape[3]

    # Kalau dynamic shape, pakai 512x512 default
    if not isinstance(h_in, int) or not isinstance(w_in, int):
        h_in, w_in = 512, 512

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # -> (1, 3, H, W)

    return img


def _map_bbox_to_seg_coords(
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int],
    seg_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Map bbox (koordinat frame) ke koordinat seg_map.
    """
    fh, fw = frame_shape
    sh, sw = seg_shape

    x1, y1, x2, y2 = bbox

    # clamp ke frame
    x1 = max(0.0, min(float(x1), fw - 1.0))
    y1 = max(0.0, min(float(y1), fh - 1.0))
    x2 = max(0.0, min(float(x2), fw - 1.0))
    y2 = max(0.0, min(float(y2), fh - 1.0))

    # skala ke seg_map
    sx1 = int(x1 / fw * sw)
    sy1 = int(y1 / fh * sh)
    sx2 = int(x2 / fw * sw)
    sy2 = int(y2 / fh * sh)

    # clamp ke seg_map
    sx1 = max(0, min(sx1, sw - 1))
    sy1 = max(0, min(sy1, sh - 1))
    sx2 = max(0, min(sx2, sw))
    sy2 = max(0, min(sy2, sh))

    if sx2 <= sx1:
        sx2 = min(sx1 + 1, sw)
    if sy2 <= sy1:
        sy2 = min(sy1 + 1, sh)

    return sx1, sy1, sx2, sy2


# ============================================
# Inference face parsing
# ============================================

def run_face_parsing(frame: np.ndarray) -> np.ndarray:
    """
    Jalankan face parsing, hasil: seg_map (H x W) int32.
    """
    session = get_face_parsing_session()
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape

    inp = _preprocess_frame(frame, input_shape)
    out = session.run(None, {input_meta.name: inp})[0]

    # output shape: (1, C, H, W)
    if out.ndim == 4:
        seg = np.argmax(out, axis=1)[0]
    elif out.ndim == 3:
        seg = out[0]
    else:
        raise RuntimeError(f"[FaceParsing] Unexpected output shape: {out.shape}")

    return seg.astype(np.int32)


# ============================================
# Generate mask wajah
# ============================================

def get_face_mask(
    frame: np.ndarray,
    face,
    dilate_iter: int = 2
) -> Optional[np.ndarray]:
    """
    Menghasilkan mask wajah 0/1 ukuran sama dengan frame.
    - frame: BGR (H, W, 3)
    - face: Face object (punya .bbox) atau dict['bbox']
    """
    h, w = frame.shape[:2]

    bbox = getattr(face, "bbox", None)
    if bbox is None and isinstance(face, dict):
        bbox = face.get("bbox", None)
    if bbox is None:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except Exception:
        return None

    seg = run_face_parsing(frame)
    sh, sw = seg.shape[:2]

    sx1, sy1, sx2, sy2 = _map_bbox_to_seg_coords(
        (x1, y1, x2, y2),
        (h, w),
        (sh, sw)
    )

    region = seg[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return None

    # mask wajah di dalam region bbox
    face_region_mask = np.isin(region, np.array(FACE_LABELS, dtype=np.int32))
    if not face_region_mask.any():
        return None

    seg_mask = np.zeros_like(seg, dtype=np.uint8)
    seg_mask[sy1:sy2, sx1:sx2] = face_region_mask.astype(np.uint8)

    # resize ke ukuran frame
    mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # sedikit dilation biar tidak "ketat" banget
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    mask = (mask > 0).astype(np.uint8)
    return mask
