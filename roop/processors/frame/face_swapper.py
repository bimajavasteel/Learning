#face-swapper full stable version (temporal ready)

from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
    smooth_bbox_for_face  # temporal smoothing
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


# ============================================================
# INIT MODEL
# ============================================================

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_dir = resolve_relative_path('../models')
    conditional_download(download_dir, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False

    src = cv2.imread(roop.globals.source_path)
    if not get_one_face(src):
        update_status('No face in source path detected.', NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ============================================================
# POSE AWARE BBOX SYSTEM
# ============================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]

    x1, y1, x2, y2 = np.array(face.bbox, dtype=np.float32)
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    # yaw adjustment
    if abs(yaw) > 25.0:
        extra = min((abs(yaw) - 25.0) * 0.02, 0.12)  # max 12%
        pad_x = w * extra

    # pitch adjustment
    if pitch < -15.0:
        extra = min((abs(pitch) - 15.0) * 0.02, 0.20)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = min((pitch - 20.0) * 0.015, 0.15)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 <= nx1 or ny2 <= ny1:
        return

    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
# ============================================================
# TEMPORAL BBOX SMOOTHING (AFTER POSE ADJUST)
# ============================================================
# ============================================================
# SWAP FACE CORE
# ============================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame

    # pose adjust
    adapt_bbox_for_pose(target_face, temp_frame.shape)

    # temporal smoothing tahap 2
    smooth_bbox_for_swapper(target_face)

    # swap
    return get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )


def _select_best_target_by_embedding(faces: List[Face], reference_face: Face):
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_dist = float('inf')

    threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            dist = np.sum((f.normed_embedding - ref_emb) ** 2)
        except:
            continue

        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_face = f

    return best_face


# ============================================================
# PROCESS SINGLE FRAME
# ============================================================

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:

    # MODE: many faces = swap semua
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            if detect_occlusion(target_face, temp_frame):
                continue
            temp_frame = swap_face(source_face, target_face, temp_frame)

        return temp_frame

    # MODE: single face (pakai reference)
    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)

    if not tracked:
        return temp_frame

    valid = [f for f in tracked if not detect_occlusion(f, temp_frame)]
    if not valid:
        return temp_frame

    best_face = None
    if reference_face is not None:
        best_face = _select_best_target_by_embedding(valid, reference_face)

    if best_face is None:
        best_face = valid[0]

    temp_frame = swap_face(source_face, best_face, temp_frame)
    return temp_frame


# ============================================================
# PROCESS LIST OF FRAMES
# ============================================================

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:

    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, frame_path in enumerate(temp_frame_paths):

        temp_frame = cv2.imread(frame_path)

        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx
        )

        cv2.imwrite(frame_path, result)

        if update:
            update()


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_frame,
            roop.globals.reference_face_position
        )

    result = process_frame(
        source_face,
        reference_face,
        target_frame,
        frame_number=0
    )
    cv2.imwrite(output_path, result)


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:

    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            ref_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(
                ref_frame,
                roop.globals.reference_face_position
            )
            set_face_reference(reference_face)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
