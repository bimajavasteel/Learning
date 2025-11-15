from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def is_face_suitable_for_swap(source_face: Face, target_face: Face) -> bool:
    # Validasi skala
    s_area = (source_face.bbox[2] - source_face.bbox[0]) * (source_face.bbox[3] - source_face.bbox[1])
    t_area = (target_face.bbox[2] - target_face.bbox[0]) * (target_face.bbox[3] - target_face.bbox[1])
    if t_area == 0 or s_area == 0:
        return False
    ratio = t_area / s_area
    if not (0.25 < ratio < 4.0):
        return False
    # Validasi pose
    if hasattr(source_face, 'normed_embedding') and hasattr(target_face, 'normed_embedding'):
        dist = np.sum(np.square(source_face.normed_embedding - target_face.normed_embedding))
        if dist > 1.3:
            return False
    return True


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    except Exception as e:
        print(f"[FaceSwapper] Swap failed: {e}")
        return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if is_face_suitable_for_swap(source_face, target_face):
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face and is_face_suitable_for_swap(source_face, target_face):
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


# ... (fungsi lain seperti pre_check, process_image, dll tetap sama)
# Tidak perlu ubah karena sudah optimal
