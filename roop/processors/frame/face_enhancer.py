from typing import Any, List, Callable, Optional, Dict
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ============================================================
# GLOBALS
# ============================================================

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

# Cache menyimpan: { face_index_or_id: (center_point, enhanced_image) }
PREV_FACE_CACHE: Dict[int, Any] = {}


# ============================================================
# MODEL INIT
# ============================================================

def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=1,
                device=get_device()
            )
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER, PREV_FACE_CACHE
    FACE_ENHANCER = None
    PREV_FACE_CACHE.clear()


# ============================================================
# TEMPORAL ENHANCER CORE
# ============================================================

def _adaptive_alpha(det_score: float) -> float:
    """Confidence-gated alpha"""
    # Gunakan default 0.5 jika global belum di-set
    base = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 0.65

    if det_score < 0.4:
        return base * 0.5
    if det_score < 0.6:
        return base * 0.7
    return base


def _temporal_luma_ema(curr: np.ndarray, prev: np.ndarray, alpha: float) -> np.ndarray:
    """EMA hanya di channel luminance (Y) untuk menjaga detail tapi stabilkan cahaya"""
    try:
        curr_ycc = cv2.cvtColor(curr, cv2.COLOR_BGR2YCrCb)
        prev_ycc = cv2.cvtColor(prev, cv2.COLOR_BGR2YCrCb)

        curr_y = curr_ycc[:, :, 0].astype(np.float32)
        prev_y = prev_ycc[:, :, 0].astype(np.float32)

        blended_y = alpha * curr_y + (1.0 - alpha) * prev_y
        curr_ycc[:, :, 0] = np.clip(blended_y, 0, 255).astype(np.uint8)

        return cv2.cvtColor(curr_ycc, cv2.COLOR_YCrCb2BGR)
    except Exception:
        return curr


def _find_closest_cache(center_current, threshold=50):
    """
    Mencari wajah di cache yang posisinya paling dekat dengan wajah sekarang.
    Ini menggantikan id() yang tidak reliable antar frame.
    """
    best_id = None
    min_dist = float('inf')

    for fid, data in PREV_FACE_CACHE.items():
        center_prev = data['center']
        dist = np.linalg.norm(np.array(center_current) - np.array(center_prev))
        
        if dist < min_dist and dist < threshold:
            min_dist = dist
            best_id = fid
            
    return best_id


def _temporal_enhance(target_center: tuple,
                      enhanced: np.ndarray,
                      det_score: float) -> np.ndarray:
    
    # 1. Cari cache yang cocok berdasarkan posisi
    face_id = _find_closest_cache(target_center)
    
    # 2. Jika tidak ada di cache, buat ID baru
    if face_id is None:
        face_id = len(PREV_FACE_CACHE) + 1
        PREV_FACE_CACHE[face_id] = {
            'image': enhanced.copy(),
            'center': target_center
        }
        return enhanced

    # 3. Ambil data lama
    prev_data = PREV_FACE_CACHE[face_id]
    prev_img = prev_data['image']

    # Pastikan dimensi sama (penting jika ukuran crop berubah sedikit)
    if prev_img.shape != enhanced.shape:
        prev_img = cv2.resize(prev_img, (enhanced.shape[1], enhanced.shape[0]))

    # 4. Lakukan Blending
    alpha = _adaptive_alpha(det_score)
    out = _temporal_luma_ema(enhanced, prev_img, alpha)

    # 5. Update Cache
    PREV_FACE_CACHE[face_id] = {
        'image': out.copy(),
        'center': target_center
    }
    
    # Bersihkan cache jika terlalu besar (opsional garbage collection sederhana)
    if len(PREV_FACE_CACHE) > 20:
        PREV_FACE_CACHE.clear()

    return out


# ============================================================
# FACE ENHANCE PIPELINE
# ============================================================

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    # PERBAIKAN 1: Akses atribut menggunakan dot notation, bukan dictionary key
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
    except AttributeError:
        # Fallback jika ternyata objectnya dictionary (jarang terjadi di Roop standard)
        x1, y1, x2, y2 = map(int, target_face['bbox'])

    pad_x = int((x2 - x1) * 0.2)
    pad_y = int((y2 - y1) * 0.2)

    h, w = temp_frame.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = temp_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return temp_frame

    with THREAD_SEMAPHORE:
        _, _, enhanced = get_face_enhancer().enhance(crop, paste_back=False)

    # PERBAIKAN 2: Gunakan center point untuk matching temporal
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    det_score = float(getattr(target_face, 'det_score', 1.0))

    enhanced = _temporal_enhance((center_x, center_y), enhanced, det_score)

    # color consistency (simple mean match)
    mean_src = np.mean(crop, axis=(0, 1))
    mean_dst = np.mean(enhanced, axis=(0, 1))
    enhanced = np.clip(enhanced + (mean_src - mean_dst), 0, 255).astype(np.uint8)

    # soft mask
    mask = np.zeros(crop.shape[:2], dtype=np.float32)
    cv2.ellipse(
        mask,
        ((x2 - x1) // 2, (y2 - y1) // 2),
        (int((x2 - x1) * 0.45), int((y2 - y1) * 0.45)),
        0, 0, 360, 1.0, -1
    )
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = np.dstack([mask] * 3)

    result = (enhanced * mask + crop * (1.0 - mask)).astype(np.uint8)
    temp_frame[y1:y2, x1:x2] = result
    return temp_frame


# ============================================================
# PROCESSORS
# ============================================================

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    faces = get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    for face in faces:
        temp_frame = enhance_face(face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for frame_path in temp_frame_paths:
        frame = cv2.imread(frame_path)
        result = process_frame(None, None, frame)
        cv2.imwrite(frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    frame = cv2.imread(target_path)
    result = process_frame(None, None, frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(
        None,
        temp_frame_paths,
        process_frames
    )
