from typing import Any, List, Callable, Optional
import cv2
import insightface
import threading
import numpy as np
import os

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ============================================================
# FORCE GPU 0 FOR SWAPPER + ANALYSER
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            providers = [('CUDAExecutionProvider', {'device_id': 0})]
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=providers
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


def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]

    x1, y1, x2, y2 = np.array(face.bbox, dtype=np.float32)
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    if abs(yaw) > 25.0:
        pad_x = w * min((abs(yaw) - 25.0) * 0.02, 0.12)

    if pitch < -15.0:
        pad_y_top = h * min((abs(pitch) - 15.0) * 0.02, 0.20)
    elif pitch > 20.0:
        pad_y_bottom = h * min((pitch - 20.0) * 0.015, 0.15)

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape)

    return get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        return temp_frame

    for target_face in faces:
        if detect_occlusion(target_face, temp_frame):
            continue
        temp_frame = swap_face(source_face, target_face, temp_frame)

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    for idx, frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(frame_path)
        result = process_frame(source_face, None, temp_frame, idx)
        cv2.imwrite(frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)
    result = process_frame(source_face, None, target_frame, 0)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
