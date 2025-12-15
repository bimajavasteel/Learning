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

# ================= TEMPORAL ENHANCER =================
TEMPORAL_ENHANCER_CACHE = {}
BASE_TEMPORAL_ALPHA = 0.7  # default EMA strength (0.65–0.8 optimal)

# ============================================================
# DEVICE & MODEL
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
    global FACE_ENHANCER
    FACE_ENHANCER = None

# ============================================================
# TEMPORAL EMA (ANTI-FLICKER)
# ============================================================

def temporal_smooth_enhanced(
    face_id: int,
    enhanced_crop: np.ndarray,
    motion: float = 0.0
) -> np.ndarray:
    """
    Temporal Exponential Moving Average (EMA)
    + Adaptive Alpha berbasis motion.
    """

    prev = TEMPORAL_ENHANCER_CACHE.get(face_id)

    # adaptive alpha
    if motion < 4.0:
        alpha = 0.80
    elif motion < 10.0:
        alpha = BASE_TEMPORAL_ALPHA
    else:
        alpha = 0.55  # gerakan cepat → kurang smoothing

    if prev is None or prev.shape != enhanced_crop.shape:
        TEMPORAL_ENHANCER_CACHE[face_id] = enhanced_crop
        return enhanced_crop

    smoothed = (
        alpha * enhanced_crop.astype(np.float32) +
        (1.0 - alpha) * prev.astype(np.float32)
    )

    smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    TEMPORAL_ENHANCER_CACHE[face_id] = smoothed
    return smoothed

# ============================================================
# BLENDING & COLOR MATCH (ANTI-FLICKER SPATIAL)
# ============================================================

def apply_blend_and_color_match(
    enhanced_crop: np.ndarray,
    original_crop: np.ndarray,
    fidelity: float
) -> np.ndarray:
    try:
        h, w = original_crop.shape[:2]
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))

        # Color mean matching
        orig_mean = np.mean(original_crop, axis=(0, 1))
        enh_mean = np.mean(enhanced_crop, axis=(0, 1))
        corrected = enhanced_crop.astype(np.float32) + (orig_mean - enh_mean)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # Fidelity blend
        blended = cv2.addWeighted(
            corrected, fidelity,
            original_crop, 1.0 - fidelity,
            0
        )

        # Soft elliptical mask
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        blur = int(min(w, h) * 0.1)
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask_3ch = np.dstack([mask] * 3)

        result = (
            blended * mask_3ch +
            original_crop * (1.0 - mask_3ch)
        ).astype(np.uint8)

        return result

    except Exception as e:
        update_status(f"[Enhancer] Blend error: {e}", NAME)
        return original_crop

# ============================================================
# FACE ENHANCE CORE
# ============================================================

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    try:
        x1, y1, x2, y2 = map(int, target_face['bbox'])

        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)

        h, w = temp_frame.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        face_crop = temp_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return temp_frame

        with THREAD_SEMAPHORE:
            _, _, enhanced = get_face_enhancer().enhance(
                face_crop,
                paste_back=True
            )

        blend_amount = (
            roop.globals.face_enhancer_blend
            if roop.globals.face_enhancer_blend is not None
            else 0.6
        )

        result_face = apply_blend_and_color_match(
            enhanced,
            face_crop,
            fidelity=blend_amount
        )

        # ================= TEMPORAL EMA =================
        face_id = id(target_face)
        motion = getattr(target_face, "motion", 0.0)
        result_face = temporal_smooth_enhanced(
            face_id,
            result_face,
            motion
        )

        temp_frame[y1:y2, x1:x2] = result_face
        return temp_frame

    except Exception as e:
        update_status(f"[Enhancer] Face error: {e}", NAME)
        return temp_frame

# ============================================================
# FRAME PROCESSING
# ============================================================

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    faces = get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    for face in faces:
        temp_frame = enhance_face(face, temp_frame)
    return temp_frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
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

# ============================================================
# LIFECYCLE
# ============================================================

def pre_check() -> bool:
    download_dir = resolve_relative_path('../models')
    conditional_download(
        download_dir,
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
    TEMPORAL_ENHANCER_CACHE.clear()
