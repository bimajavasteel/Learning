# (FILE 3) – face_swapper_cscs256.py
# ----------------------------------

from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np

from roop.face_analyser_3D import (
    get_one_face,
    smart_face_tracking,
    get_many_faces,
    headpose_from_landmarks,
    occlusion_mask,
)
from occlusion_handler import apply_occlusion_mask
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
import roop.globals
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"


# ----------------------------------------------
# LOAD CSCS-256
# ----------------------------------------------
def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            path = resolve_relative_path("../models/cscs_256.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                path,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER


def clear_face_swapper():
    global FACE_SWAPPER
    FACE_SWAPPER = None


# ----------------------------------------------
# PRE CHECK (download model)
# ----------------------------------------------
def pre_check():
    download_directory_path = resolve_relative_path("../models")
    conditional_download(download_directory_path, [
        "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/cscs_256.onnx"
    ])
    return True


# ----------------------------------------------
# PRE START
# ----------------------------------------------
def pre_start():
    if not is_image(roop.globals.source_path):
        return False

    src = cv2.imread(roop.globals.source_path)
    if not get_one_face(src):
        return False

    return True


def post_process():
    clear_face_swapper()
    clear_face_reference()


# ----------------------------------------------
# SWAP FUNCTION (pose-aware + occlusion-aware)
# ----------------------------------------------
def swap_face(source_face, target_face, frame):
    swapper = get_face_swapper()
    if source_face is None or target_face is None:
        return frame

    # Extract target bbox
    x1, y1, x2, y2 = map(int, target_face.bbox)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return frame

    # ------------------------------------------
    # HEADPOSE → dynamic padding + scale
    # ------------------------------------------
    lm = target_face.landmark_3d_68
    yaw, pitch = headpose_from_landmarks(lm)

    # dynamic padding
    pad_v = int(abs(pitch) * 0.8)
    pad_h = int(abs(yaw) * 0.6)

    x1p = max(0, x1 - pad_h)
    y1p = max(0, y1 - pad_v)
    x2p = min(frame.shape[1], x2 + pad_h)
    y2p = min(frame.shape[0], y2 + pad_v)

    crop2 = frame[y1p:y2p, x1p:x2p].copy()

    # ------------------------------------------
    # OCCLUSION HANDLING
    # ------------------------------------------
    occ = occlusion_mask(crop2)
    crop2, visible_mask = apply_occlusion_mask(crop2, occ)

    # ------------------------------------------
    # CALL SWAPPER
    # ------------------------------------------
    out = swapper.get(frame, target_face, source_face, paste_back=False)

    # ------------------------------------------
    # FINAL BLEND USING VISIBLE MASK
    # ------------------------------------------
    out_crop = out[y1p:y2p, x1p:x2p]

    blend = (out_crop * visible_mask + crop2 * (1-visible_mask)).astype("uint8")

    frame[y1p:y2p, x1p:x2p] = blend
    return frame


# ----------------------------------------------
# PIPELINE
# ----------------------------------------------
def process_frame(src_face, ref_face, frame, idx):
    faces = smart_face_tracking(frame, idx)
    if not faces:
        return frame

    for t in faces:
        frame = swap_face(src_face, t, frame)

    return frame


def process_frames(src_path, frames, update):
    src = cv2.imread(src_path)
    src_face = get_one_face(src)
    ref_face = get_face_reference()

    for i, fp in enumerate(frames):
        img = cv2.imread(fp)
        out = process_frame(src_face, ref_face, img, i)
        cv2.imwrite(fp, out)
        if update:
            update()


def process_image(src, tgt, out):
    s = cv2.imread(src)
    t = cv2.imread(tgt)
    sf = get_one_face(s)
    tf = get_one_face(tgt)
    result = swap_face(sf, tf, t)
    cv2.imwrite(out, result)


def process_video(src, frames):
    roop.processors.frame.core.process_video(src, frames, process_frames)
