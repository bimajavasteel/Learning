from typing import Any, List, Callable
import cv2
import threading
import numpy as np
import os
import torch
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ============================================================
# FORCE GPU 1 FOR ENHANCER
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)  # GPU 1 becomes cuda:0 in this context

FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()
NAME = 'ROOP.FACE-ENHANCER'

TEMPORAL_ENHANCER_CACHE = {}
BASE_TEMPORAL_ALPHA = 0.7


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=1,
                device='cuda'
            )
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def temporal_smooth_enhanced(face_id: int, enhanced: np.ndarray) -> np.ndarray:
    prev = TEMPORAL_ENHANCER_CACHE.get(face_id)
    if prev is None or prev.shape != enhanced.shape:
        TEMPORAL_ENHANCER_CACHE[face_id] = enhanced
        return enhanced

    alpha = BASE_TEMPORAL_ALPHA
    out = alpha * enhanced.astype(np.float32) + (1 - alpha) * prev.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    TEMPORAL_ENHANCER_CACHE[face_id] = out
    return out


def enhance_face(target_face: Face, frame: Frame) -> Frame:
    try:
        x1, y1, x2, y2 = map(int, target_face['bbox'])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return frame

        with THREAD_SEMAPHORE:
            _, _, enhanced = get_face_enhancer().enhance(face_crop, paste_back=True)

        enhanced = temporal_smooth_enhanced(id(target_face), enhanced)
        frame[y1:y2, x1:x2] = enhanced
        return frame

    except Exception as e:
        update_status(f"[Enhancer] error: {e}", NAME)
        return frame


def process_frame(source_face: Face, reference_face: Face, frame: Frame) -> Frame:
    faces = get_many_faces(frame)
    if not faces:
        return frame

    for face in faces:
        frame = enhance_face(face, frame)
    return frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for path in temp_frame_paths:
        frame = cv2.imread(path)
        result = process_frame(None, None, frame)
        cv2.imwrite(path, result)
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


def pre_check() -> bool:
    download_dir = resolve_relative_path('../models')
    conditional_download(
        download_dir,
        ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth']
    )
    return True


def pre_start() -> bool:
    return True


def post_process() -> None:
    clear_face_enhancer()
    TEMPORAL_ENHANCER_CACHE.clear()
