# ============================================================
#  FACE SWAPPER + WRINKLE ENHANCER V2 (FINAL VERSION)
# ============================================================

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
    smart_face_tracking,
    detect_occlusion,
    get_face_pose
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# === IMPORT WRINKLE ENHANCER V2 ===
from wrinkle_enhancer_v2 import enhance_wrinkles_after_gfpgan

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"


# ============================================================
#  INIT SWAPPER MODEL
# ============================================================
def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path("../models/inswapper_128.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
            print("✔ Loaded inswapper_128")
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


# ============================================================
#  PRE CHECK
# ============================================================
def pre_check() -> bool:
    download_directory_path = resolve_relative_path("../models")
    conditional_download(download_directory_path, [
        "https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx"
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    src = cv2.imread(roop.globals.source_path)
    if not get_one_face(src):
        update_status("No face in source path detected.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ============================================================
#  BBOX ADJUST BASED ON POSE
# ============================================================
def adapt_bbox_for_pose(face: Face, frame_shape):
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]

    bbox = np.array(face.bbox, dtype=float)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = pad_y_top = pad_y_bottom = 0

    if abs(yaw) > 25:
        extra = min((abs(yaw) - 25) * 0.02, 0.20)
        pad_x = w * extra

    if pitch < -15:
        extra = min((abs(pitch) - 15) * 0.02, 0.25)
        pad_y_top = h * extra
    elif pitch > 20:
        extra = min((pitch - 20) * 0.015, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=float)


# ============================================================
#  SWAP + WRINKLE ENHANCER V2 
# ============================================================
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:

    if source_face is None or target_face is None:
        return temp_frame

    # Adjust BBOX berdasarkan pose
    adapt_bbox_for_pose(target_face, temp_frame.shape)

    # Hasil swap dari INSWAPPER
    swapped = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )

    # =============================================
    #  APPLY WRINKLE + DARK CIRCLE (V2)
    # =============================================
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)

        # safety clamp
        h, w = swapped.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = swapped[y1:y2, x1:x2].copy()

        enhanced = enhance_wrinkles_after_gfpgan(crop, target_face)

        swapped[y1:y2, x1:x2] = enhanced

    except Exception as e:
        print(f"[WrinkleEnhancer] Error: {e}")

    return swapped


# ============================================================
#  FRAME PROCESSING
# ============================================================
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number=0) -> Frame:

    if source_face is None:
        return temp_frame

    # MULTI FACE
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        for f in faces:
            if not detect_occlusion(f, temp_frame):
                temp_frame = swap_face(source_face, f, temp_frame)

        return temp_frame

    # SINGLE FACE
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        faces = get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    valid_faces = [f for f in faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    target = valid_faces[0]

    return swap_face(source_face, target, temp_frame)


# ============================================================
#  PROCESS FOR VIDEO
# ============================================================
def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable):
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, path in enumerate(temp_frame_paths):
        frame = cv2.imread(path)
        result = process_frame(source_face, reference_face, frame, idx)
        cv2.imwrite(path, result)

        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str):
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    target = cv2.imread(target_path)
    reference_face = None

    result = process_frame(source_face, reference_face, target, 0)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]):

    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            ref_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(ref_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
