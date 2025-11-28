#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import threading
import insightface

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
from roop.utilities import resolve_relative_path, is_image, is_video, conditional_download

# =====================================================================
# GLOBALS
# =====================================================================

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


# =====================================================================
# AUTO-DOWNLOAD MODEL (UNIFACE_256)
# =====================================================================

UNIFACE_URL = (
    "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/uniface_256.onnx"
)


def download_uniface_if_missing():
    """
    Auto-download uniface_256.onnx dari HuggingFace bila belum ada.
    Menggunakan conditional_download bawaan project.
    """
    model_dir = resolve_relative_path("../models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "uniface_256.onnx")

    conditional_download(
        model_dir,
        [UNIFACE_URL],  # hanya download file ini
        file_names=["uniface_256.onnx"]
    )

    return model_path


# =====================================================================
# MODEL LOADER
# =====================================================================

def get_face_swapper() -> any:
    """
    Loader khusus uniface_256.onnx.
    Tidak ada fallback ke inswapper_128.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:

            model_path = download_uniface_if_missing()

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"[FACE-SWAPPER] Gagal menemukan atau mendownload uniface_256.onnx "
                    f"di path: {model_path}"
                )

            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )

            print("✅ [FACE-SWAPPER] Loaded uniface_256 (auto-downloaded).")

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


# =====================================================================
# PRE-START CHECKS
# =====================================================================

def pre_check() -> bool:
    """
    Pastikan model siap (auto-download jika hilang).
    """
    try:
        download_uniface_if_missing()
    except Exception as e:
        update_status(f"Gagal download model: {e}", NAME)
        return False

    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status("No face detected in source path.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# =====================================================================
#  POSE-AWARE BBOX ADJUSTMENT
# =====================================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)

    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = pad_y_top = pad_y_bottom = 0.0

    if abs(yaw) > 25.0:
        extra = (abs(yaw) - 25.0) * 0.02
        extra = min(extra, 0.20)
        pad_x = w * extra

    if pitch < -15.0:
        extra = (abs(pitch) - 15.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = (pitch - 20.0) * 0.015
        extra = min(extra, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 <= nx1 or ny2 <= ny1:
        return

    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# =====================================================================
#  CORE SWAP
# =====================================================================

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


# =====================================================================
# FRAME PROCESSING
# =====================================================================

def _select_best_target_by_embedding(faces, reference_face):
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_dist = float('inf')

    threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue
        try:
            dist = np.sum(np.square(f.normed_embedding - ref_emb))
        except:
            continue

        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_face = f

    return best_face


def process_frame(source_face, reference_face, temp_frame, frame_number=0):
    if source_face is None:
        return temp_frame

    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for face in faces:
            if detect_occlusion(face, temp_frame):
                continue
            temp_frame = swap_face(source_face, face, temp_frame)

        return temp_frame

    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        faces = get_many_faces(temp_frame)

    if not faces:
        return temp_frame

    valid_faces = [f for f in faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)
    if best_target is None:
        best_target = valid_faces[0]

    return swap_face(source_face, best_target, temp_frame)


# =====================================================================
# VIDEO ENGINE
# =====================================================================

def process_frames(source_path, temp_frame_paths, update):
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for i, p in enumerate(temp_frame_paths):
        frame = cv2.imread(p)
        result = process_frame(
            source_face,
            reference_face,
            frame,
            frame_number=i
        )
        cv2.imwrite(p, result)

        if update:
            update()


def process_image(source_path, target_path, output_path):
    src = cv2.imread(source_path)
    tgt = cv2.imread(target_path)

    src_face = get_one_face(src)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(tgt, roop.globals.reference_face_position)

    result = process_frame(src_face, reference_face, tgt)
    cv2.imwrite(output_path, result)


def process_video(source_path, temp_frame_paths):
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            frame = cv2.imread(temp_frame_paths[ref_idx])
            ref_face = get_one_face(frame, roop.globals.reference_face_position)
            set_face_reference(ref_face)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
