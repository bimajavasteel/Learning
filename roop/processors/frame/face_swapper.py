# ============================================================
#  FACE SWAPPER – UNIFACE_256 + AUTO DOWNLOAD FIXED
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
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"

UNIFACE_URL = [
    "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/uniface_256.onnx"
]


# ============================================================
# AUTO DOWNLOAD UNIFACE 256 — FIXED
# ============================================================

def ensure_uniface_model() -> str:
    """
    Auto-download uniface_256.onnx menggunakan conditional_download()
    sesuai versi project ini (tanpa file_names).
    """
    model_dir = resolve_relative_path("../models")
    conditional_download(model_dir, UNIFACE_URL)

    model_path = f"{model_dir}/uniface_256.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Gagal download uniface_256.onnx ke {model_path}"
        )
    return model_path


# ============================================================
#  LOAD SWAPPER MODEL
# ============================================================

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:

            model_path = ensure_uniface_model()

            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers,
            )

            print("✅ Loaded uniface_256 (auto-download OK)")

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


# ============================================================
# PRE-CHECK / PRE-START
# ============================================================

def pre_check() -> bool:
    try:
        ensure_uniface_model()
        return True
    except Exception as e:
        update_status(f"Gagal download UNIFACE 256: {e}", NAME)
        return False


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    src = cv2.imread(roop.globals.source_path)
    if not get_one_face(src):
        update_status("No face found in source.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target.", NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ============================================================
#  POSE AWARE BBOX
# ============================================================

def adapt_bbox_for_pose(face: Face, frame_shape):
    pitch, yaw, roll = get_face_pose(face)
    h, w = frame_shape[:2]

    x1, y1, x2, y2 = map(float, face.bbox)
    bw, bh = x2 - x1, y2 - y1

    pad_x = pad_y_top = pad_y_bottom = 0.0

    if abs(yaw) > 25:
        pad_x = bw * min(0.02 * (abs(yaw) - 25), 0.20)

    if pitch < -15:
        pad_y_top = bh * min(0.02 * (abs(pitch) - 15), 0.25)
    elif pitch > 20:
        pad_y_bottom = bh * min(0.015 * (pitch - 20), 0.18)

    nx1 = max(0, int(x1 - pad_x))
    nx2 = min(w - 1, int(x2 + pad_x))
    ny1 = max(0, int(y1 - pad_y_top))
    ny2 = min(h - 1, int(y2 + pad_y_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# ============================================================
# SWAP
# ============================================================

def swap_face(source_face: Face, target_face: Face, frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return frame

    adapt_bbox_for_pose(target_face, frame.shape)

    return get_face_swapper().get(
        frame,
        target_face,
        source_face,
        paste_back=True,
    )


# ============================================================
# FRAME LOGIC
# ============================================================

def _select_best_target_by_embedding(faces, ref_face):
    if not faces or ref_face is None:
        return None
    if not hasattr(ref_face, "normed_embedding"):
        return None

    best = None
    best_dist = float("inf")
    th = getattr(roop.globals, "similar_face_distance", 1.0)

    ref_emb = ref_face.normed_embedding
    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            dist = np.sum((f.normed_embedding - ref_emb) ** 2)
            if dist < th and dist < best_dist:
                best_dist, best = dist, f
        except:
            continue

    return best


def process_frame(source_face, reference_face, frame, frame_number=0):
    if source_face is None:
        return frame

    faces = smart_face_tracking(frame, frame_number)
    if not faces:
        faces = get_many_faces(frame)
    if not faces:
        return frame

    valid = [f for f in faces if not detect_occlusion(f, frame)]
    if not valid:
        return frame

    if roop.globals.many_faces:
        for f in valid:
            frame = swap_face(source_face, f, frame)
        return frame

    best = None
    if reference_face:
        best = _select_best_target_by_embedding(valid, reference_face)
    if best is None:
        best = valid[0]

    return swap_face(source_face, best, frame)


# ============================================================
# VIDEO PROCESS LOOP
# ============================================================

def process_frames(src_path, temp_frame_paths, update):
    src = cv2.imread(src_path)
    source_face = get_one_face(src)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for i, p in enumerate(temp_frame_paths):
        frame = cv2.imread(p)
        out = process_frame(source_face, reference_face, frame, i)
        cv2.imwrite(p, out)
        if update:
            update()


def process_image(src_path, tgt_path, out_path):
    src = cv2.imread(src_path)
    tgt = cv2.imread(tgt_path)

    src_face = get_one_face(src)

    ref_face = None
    if not roop.globals.many_faces:
        ref_face = get_one_face(tgt, roop.globals.reference_face_position)

    out = process_frame(src_face, ref_face, tgt)
    cv2.imwrite(out_path, out)


def process_video(src_path, temp_frame_paths):
    if not roop.globals.many_faces and not get_face_reference():
        try:
            idx = roop.globals.reference_frame_number
            f = cv2.imread(temp_frame_paths[idx])
            ref = get_one_face(f, roop.globals.reference_face_position)
            set_face_reference(ref)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        src_path,
        temp_frame_paths,
        process_frames,
    )
