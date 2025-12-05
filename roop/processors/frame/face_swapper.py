# face_swapper.py (FINAL FIXED COMPATIBLE WITH core.py)
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
    get_face_pose,
    FACE_TRACKING   # ← ini cara benar (ambil reference, jangan override)
)
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference
)
from roop.typing import Frame, Face
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video
)

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"


# ============================================================
# MODEL LOADING
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
    return FACE_SWAPPER


def clear_face_swapper():
    global FACE_SWAPPER
    FACE_SWAPPER = None


# ============================================================
# PRE CHECK / PRE START
# ============================================================

def pre_check() -> bool:
    conditional_download(resolve_relative_path("../models"), [
        "https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx"
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    img = cv2.imread(roop.globals.source_path)
    if get_one_face(img) is None:
        update_status("No face in source image.", NAME)
        return False

    if not (is_image(roop.globals.target_path) or is_video(roop.globals.target_path)):
        update_status("Target is not an image/video.", NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ============================================================
# POSE AWARE BBOX
# ============================================================

def adapt_bbox_for_pose(face: Face, frame_shape):
    pitch, yaw, roll = get_face_pose(face)

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(float, face.bbox)

    bw = x2 - x1
    bh = y2 - y1

    pad_x = 0
    pad_top = 0
    pad_bottom = 0

    if abs(yaw) > 25:
        pad_x = bw * min((abs(yaw) - 25) * 0.02, 0.2)

    if pitch < -15:
        pad_top = bh * min((abs(pitch) - 15) * 0.02, 0.25)
    elif pitch > 20:
        pad_bottom = bh * min((pitch - 20) * 0.015, 0.18)

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_top))
    ny2 = int(min(h - 1, y2 + pad_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# ============================================================
# SWAP OPERATION
# ============================================================

def swap_face(src_face, tgt_face, frame):
    adapt_bbox_for_pose(tgt_face, frame.shape)
    try:
        return get_face_swapper().get(
            frame,
            tgt_face,
            src_face,
            paste_back=True
        )
    except:
        return frame


# ============================================================
# EMBEDDING MATCH
# ============================================================

def _select_best_face(faces: List[Face], reference_face: Face):
    if reference_face is None:
        return None

    ref_emb = getattr(reference_face, "normed_embedding", None)
    if ref_emb is None:
        return None

    thr = roop.globals.similar_face_distance
    best = None
    best_dist = 999

    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            continue
        dist = np.sum((emb - ref_emb) ** 2)
        if dist < thr and dist < best_dist:
            best = f
            best_dist = dist

    return best


# ============================================================
# FRAME PROCESSOR (THE MOST IMPORTANT PART)
# ============================================================

def process_frame(source_face, reference_face, frame, frame_number):

    if frame is None:
        return frame

    # MULTI-FACE MODE
    if roop.globals.many_faces:

        faces = smart_face_tracking(frame, frame_number)
        if not faces:
            faces = get_many_faces(frame)

        if not faces:
            return frame

        for idx, face in enumerate(faces):

            setattr(face, "track_id", idx)

            prev_face = None
            if idx in FACE_TRACKING:
                prev_face = FACE_TRACKING[idx]["last_face"]

            if detect_occlusion(face, frame, prev_face):
                continue

            frame = swap_face(source_face, face, frame)

        return frame

    # SINGLE-FACE MODE
    faces = smart_face_tracking(frame, frame_number)
    if not faces:
        faces = get_many_faces(frame)

    if not faces:
        return frame

    valid_faces = []

    for idx, f in enumerate(faces):
        setattr(f, "track_id", idx)

        prev_face = FACE_TRACKING[idx]["last_face"] if idx in FACE_TRACKING else None

        if not detect_occlusion(f, frame, prev_face):
            valid_faces.append(f)

    if not valid_faces:
        return frame

    best_face = _select_best_face(valid_faces, reference_face)
    if best_face is None:
        best_face = valid_faces[0]

    return swap_face(source_face, best_face, frame)


# ============================================================
# IMAGE MODE
# ============================================================

def process_image(src_path, tgt_path, output_path):
    src = cv2.imread(src_path)
    tgt = cv2.imread(tgt_path)

    src_face = get_one_face(src)
    ref_face = None

    if not roop.globals.many_faces:
        ref_face = get_one_face(tgt, roop.globals.reference_face_position)

    out = process_frame(src_face, ref_face, tgt, 0)
    cv2.imwrite(output_path, out)


# ============================================================
# VIDEO MODE
# ============================================================

def process_video(src_path, frame_paths):

    # reference face
    if not roop.globals.many_faces and not get_face_reference():
        try:
            idx = roop.globals.reference_frame_number
            ref = cv2.imread(frame_paths[idx])
            set_face_reference(get_one_face(ref, roop.globals.reference_face_position))
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        src_path,
        frame_paths,
        process_frames
    )


def process_frames(src_path, frame_paths, update):

    src_img = cv2.imread(src_path)
    src_face = get_one_face(src_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for i, path in enumerate(frame_paths):

        frame = cv2.imread(path)
        out = process_frame(src_face, reference_face, frame, i)

        cv2.imwrite(path, out)

        if callable(update):
            update()
