from typing import Any, List, Callable
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

# 🔥 EMA TEMPORAL CACHE (GLOBAL, PER PROCESS)
PREV_ENHANCED_FACE: np.ndarray | None = None

# EMA alpha (0.6 – 0.7 recommended)
EMA_ALPHA = 0.65


# ============================================================
# INIT MODEL
# ============================================================

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


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER, PREV_ENHANCED_FACE
    FACE_ENHANCER = None
    PREV_ENHANCED_FACE = None


# ============================================================
# PRE CHECK
# ============================================================

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(
        download_directory_path,
        ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth']
    )
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


# ============================================================
# EMA TEMPORAL BLEND (CORE)
# ============================================================

def temporal_ema_blend(current: np.ndarray) -> np.ndarray:
    """
    Exponential Moving Average temporal blending
    current_t = α * current + (1 - α) * previous
    """
    global PREV_ENHANCED_FACE

    if PREV_ENHANCED_FACE is None:
        PREV_ENHANCED_FACE = current.copy()
        return current

    # safety: shape mismatch (scene cut / face lost)
    if PREV_ENHANCED_FACE.shape != current.shape:
        PREV_ENHANCED_FACE = current.copy()
        return current

    blended = cv2.addWeighted(
        current, EMA_ALPHA,
        PREV_ENHANCED_FACE, 1.0 - EMA_ALPHA,
        0
    )

    PREV_ENHANCED_FACE = blended.copy()
    return blended


# ============================================================
# FACE ENHANCE CORE
# ============================================================

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    x1, y1, x2, y2 = map(int, target_face['bbox'])

    padding_x = int((x2 - x1) * 0.2)
    padding_y = int((y2 - y1) * 0.2)

    h_frame, w_frame = temp_frame.shape[:2]
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w_frame, x2 + padding_x)
    y2 = min(h_frame, y2 + padding_y)

    face_crop = temp_frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return temp_frame

    with THREAD_SEMAPHORE:
        _, _, enhanced = get_face_enhancer().enhance(
            face_crop,
            paste_back=False
        )

    if enhanced is None or enhanced.size == 0:
        return temp_frame

    # resize safety
    if enhanced.shape[:2] != face_crop.shape[:2]:
        enhanced = cv2.resize(enhanced, (face_crop.shape[1], face_crop.shape[0]))

    # 🔥 EMA TEMPORAL BLEND (DI SINI INTINYA)
    enhanced = temporal_ema_blend(enhanced)

    temp_frame[y1:y2, x1:x2] = enhanced
    return temp_frame


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    faces = get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    for face in faces:
        temp_frame = enhance_face(face, temp_frame)

    return temp_frame


# ============================================================
# PROCESS FRAMES
# ============================================================

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for frame_path in temp_frame_paths:
        temp_frame = cv2.imread(frame_path)
        result = process_frame(None, None, temp_frame)
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
