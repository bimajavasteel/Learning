from typing import Any, List, Callable
import cv2
import numpy as np
import onnxruntime as ort
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    detect_occlusion,
)
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference,
)
from roop.typing import Frame, Face
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video,
)

# ==================================================================
# GLOBALS
# ==================================================================

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

CSCS_URL = (
    "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/cscs_256.onnx"
)
CSCS_FILENAME = "cscs_256.onnx"

SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.log_severity_level = 3

NAME = "ROOP.FACE-SWAPPER"


# ==================================================================
# LOAD CSCS_256 MODEL
# ==================================================================

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(f"../models/{CSCS_FILENAME}")

            providers = []
            if "CUDAExecutionProvider" in roop.globals.execution_providers:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            FACE_SWAPPER = ort.InferenceSession(
                model_path,
                sess_options=SESSION_OPTIONS,
                providers=providers,
            )

            print("✅ [face_swapper] CSCS_256 ONNXRuntime loaded")

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_dir = resolve_relative_path("../models")
    conditional_download(download_dir, [CSCS_URL])
    return True


# ==================================================================
# VALIDATION
# ==================================================================

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    src = cv2.imread(roop.globals.source_path)
    if not get_one_face(src):
        update_status("No face detected in source.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ==================================================================
# KEY FIX: MANUAL FACE CROP FROM BBOX
# ==================================================================

def crop_from_bbox(frame: Frame, face: Face, size=256):
    bbox = face.bbox
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (size, size))
    return crop


# ==================================================================
# POSE-AWARE BBOX EXPANSION
# ==================================================================

def adapt_bbox_for_pose(face: Face, frame: Frame) -> Face:
    bbox = face.bbox
    pose = getattr(face, "pose", None)
    if pose is None:
        return face

    pitch, yaw, roll = pose[:3]
    mag = min(60, float(np.linalg.norm([pitch, yaw, roll])))

    scale = 1.0 + 0.25 * (mag / 60)

    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1, ny1 = cx - bw / 2, cy - bh / 2
    nx2, ny2 = cx + bw / 2, cy + bh / 2

    h, w = frame.shape[:2]

    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    nx2 = max(1, min(w, nx2))
    ny2 = max(1, min(h, ny2))

    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    return face


# ==================================================================
# MAIN SWAP FUNCTION
# ==================================================================

def swap_face(source_face: Face, target_face: Face, frame: Frame) -> Frame:

    session = get_face_swapper()

    # manual bbox-based crop (FIX)
    src_crop = crop_from_bbox(frame, source_face)
    tgt_crop = crop_from_bbox(frame, target_face)

    if src_crop is None or tgt_crop is None:
        return frame

    inp_src = src_crop[..., ::-1].astype(np.float32) / 255.0
    inp_tgt = tgt_crop[..., ::-1].astype(np.float32) / 255.0

    inp_src = np.transpose(inp_src, (2, 0, 1))[None, :]
    inp_tgt = np.transpose(inp_tgt, (2, 0, 1))[None, :]

    in0, in1 = [i.name for i in session.get_inputs()]

    out = session.run(None, {in0: inp_src, in1: inp_tgt})[0][0]

    swapped = (np.transpose(out, (1, 2, 0)) * 255).astype("uint8")
    swapped = swapped[..., ::-1]

    x1, y1, x2, y2 = map(int, target_face.bbox)
    swapped = cv2.resize(swapped, (x2 - x1, y2 - y1))

    frame[y1:y2, x1:x2] = swapped
    return frame


# ==================================================================
# FRAME PIPELINE
# ==================================================================

def process_frame(source_face, reference_face, frame, frame_number=0):

    faces = smart_face_tracking(frame, frame_number)
    if not faces:
        faces = get_many_faces(frame)
    if not faces:
        return frame

    valid = [f for f in faces if not detect_occlusion(f, frame)]
    if not valid:
        return frame

    # many-faces: swap semua
    if roop.globals.many_faces:
        for f in valid:
            f = adapt_bbox_for_pose(f, frame)
            frame = swap_face(source_face, f, frame)
        return frame

    # single-face: swap wajah pertama
    f = adapt_bbox_for_pose(valid[0], frame)
    return swap_face(source_face, f, frame)


# ==================================================================
# VIDEO / IMAGE PROCESSING
# ==================================================================

def process_frames(source_path, temp_frame_paths, update):
    src = cv2.imread(source_path)
    source_face = get_one_face(src)

    for idx, p in enumerate(temp_frame_paths):
        frame = cv2.imread(p)

        frame = process_frame(
            source_face,
            None,
            frame,
            idx,
        )

        cv2.imwrite(p, frame)
        if update:
            update()


def process_image(source_path, target_path, output_path):
    src = cv2.imread(source_path)
    tgt = cv2.imread(target_path)
    face = get_one_face(src)
    result = process_frame(face, None, tgt)
    cv2.imwrite(output_path, result)


def process_video(source_path, temp_frame_paths):
    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames,
    )
