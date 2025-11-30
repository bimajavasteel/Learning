#face-swpper support new
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
    get_face_pose
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ===========================================
#  IMPORT WRINKLE ENHANCER
# ===========================================
from wrinkle_enhancer_v2 import enhance_wrinkles_after_gfpgan as wrinkle_pass_final
from wrinkle_enhancer_v2 import enhance_wrinkles_after_gfpgan as wrinkle_pass_pre


FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


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
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
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
#  POSE AWARE BBOX FIX
# ============================================================
def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    pad_x = 0
    pad_y_top = 0
    pad_y_bottom = 0

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
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# ============================================================
#  ULTRA-REALISM PIPELINE
# ============================================================
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape)

    # -----------------------------------------------------------
    # STEP 1 — FACE SWAP
    # -----------------------------------------------------------
    swapped = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )

    # -----------------------------------------------------------
    # STEP 2 — WRINKLE PASS BEFORE GFPGAN
    # (supaya GFPGAN ikut menyeimbangkan tekstur)
    # -----------------------------------------------------------
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        crop = swapped[y1:y2, x1:x2]
        if crop.size > 0:
            pre_wrinkle = wrinkle_pass_pre(crop, target_face)
            swapped[y1:y2, x1:x2] = pre_wrinkle
    except:
        pass

    # -----------------------------------------------------------
    # STEP 3 — GFPGAN AKAN DIJALANKAN DI ENHANCER-FINAL
    # (di luar face_swapper, pipeline resmi roop)
    # -----------------------------------------------------------

    # swapped = swapped

    # -----------------------------------------------------------
    # STEP 4 — WRINKLE PASS AFTER GFPGAN
    # (mengembalikan micro details yang hilang)
    # -----------------------------------------------------------
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        crop2 = swapped[y1:y2, x1:x2]
        if crop2.size > 0:
            post_wrinkle = wrinkle_pass_final(crop2, target_face)
            swapped[y1:y2, x1:x2] = post_wrinkle
    except:
        pass

    return swapped


# ============================================================
#  FRAME PROCESS
# ============================================================
def _select_best_target_by_embedding(faces: List[Face], reference_face: Face):
    if not faces or reference_face is None:
        return None
    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue
        try:
            dist = np.sum(np.square(f.normed_embedding - ref_emb))
        except:
            continue
        if dist < similar_threshold and dist < best_distance:
            best_distance = dist
            best_face = f

    return best_face


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:

    if source_face is None:
        return temp_frame

    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame

        for tf in faces:
            if detect_occlusion(tf, temp_frame):
                continue
            temp_frame = swap_face(source_face, tf, temp_frame)

        return temp_frame

    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)
    if not tracked:
        return temp_frame

    valid = [f for f in tracked if not detect_occlusion(f, temp_frame)]
    if not valid:
        return temp_frame

    best = None
    if reference_face is not None:
        best = _select_best_target_by_embedding(valid, reference_face)
    if best is None:
        best = valid[0]

    return swap_face(source_face, best, temp_frame)


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, p in enumerate(temp_frame_paths):
        f = cv2.imread(p)
        r = process_frame(source_face, reference_face, f, frame_number=idx)
        cv2.imwrite(p, r)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    s = cv2.imread(source_path)
    t = cv2.imread(target_path)
    sf = get_one_face(s)

    ref = None
    if not roop.globals.many_faces:
        ref = get_one_face(t, roop.globals.reference_face_position)

    result = process_frame(sf, ref, t, frame_number=0)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            idx = roop.globals.reference_frame_number
            rf = cv2.imread(temp_frame_paths[idx])
            ref_face = get_one_face(rf, roop.globals.reference_face_position)
            set_face_reference(ref_face)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
