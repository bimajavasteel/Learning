# face_swapper.py
# Versi final patch: occlusion-aware bbox adapt, occlusion_ratio passed ke swap,
# dan flow many/single face diperketat terhadap occlusion.
#
# Letakkan di Learning/roop/processors/frame/  (overwrite)
# Referensi implementasi asal: face-swapper original. :contentReference[oaicite:3]{index=3}

from typing import Any, List, Callable, Optional
import cv2
import numpy as np
import threading

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
from roop.utilities import resolve_relative_path, is_image, is_video

# model (inswapper) lazy init
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = __import__('insightface').model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    # conditional_download kept in original pipeline if needed
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', 'ROOP.FACE-SWAPPER')
        return False
    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status('No face in source path detected.', 'ROOP.FACE-SWAPPER')
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', 'ROOP.FACE-SWAPPER')
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

# -------------------- pose-aware bbox adapt (occlusion-aware) --------------------
def adapt_bbox_for_pose(face, frame_shape, occluded_ratio: float = 0.0) -> None:
    """
    Sesuaikan bbox dgn padding, namun kurangi padding saat occluded_ratio tinggi
    occluded_ratio: 0..1
    """
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    # jika occlusion parsial terdeteksi, kurangi faktor padding
    if occluded_ratio > 0.10:
        yaw_limit_factor = 0.35
        pitch_limit_factor = 0.35
    else:
        yaw_limit_factor = 1.0
        pitch_limit_factor = 1.0

    if abs(yaw) > 25.0:
        extra = (abs(yaw) - 25.0) * 0.02 * yaw_limit_factor
        extra = min(extra, 0.20)
        pad_x = w * extra

    if pitch < -15.0:
        extra = (abs(pitch) - 15.0) * 0.02 * pitch_limit_factor
        extra = min(extra, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = (pitch - 20.0) * 0.015 * pitch_limit_factor
        extra = min(extra, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 <= nx1 or ny2 <= ny1:
        return
    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

# -------------------- core swap helpers --------------------
def swap_face(source_face, target_face, temp_frame, occluded_ratio: float = 0.0):
    """
    Panggil inswapper dengan bbox yg sudah diadaptasi.
    occluded_ratio: dipakai untuk adapt_bbox_for_pose agar tidak memperlebar bbox saat occlusion parsial
    """
    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape, occluded_ratio)

    try:
        return get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )
    except Exception:
        # bila inswapper error, kembalikan frame tanpa crash
        return temp_frame

def _select_best_target_by_embedding(faces: List, reference_face) -> Optional[Any]:
    if not faces or reference_face is None:
        return None
    if not hasattr(reference_face, 'normed_embedding'):
        return None
    ref_emb = reference_face.normed_embedding
    best = None
    best_distance = float('inf')
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)
    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue
        try:
            d = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue
        if d < similar_threshold and d < best_distance:
            best_distance = d
            best = f
    return best

# -------------------- process frame (occlusion-aware) --------------------
def process_frame(source_face, reference_face, temp_frame, frame_number: int = 0):
    """
    Alur:
    - ambil faces via smart_face_tracking (atau get_many_faces)
    - hitung occlusion_ratio per face via detect_occlusion (mengembalikan float)
    - jika occlusion_ratio >= globals threshold -> skip
    - pass occlusion_ratio ke swap_face untuk adapt bbox
    """
    if source_face is None:
        return temp_frame

    threshold = getattr(roop.globals, "occluder_threshold", 0.15)

    # mode many faces
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame

        for tgt in faces:
            try:
                occl_ratio = detect_occlusion(tgt, temp_frame)
            except Exception:
                occl_ratio = 0.0

            if occl_ratio >= threshold:
                # skip face yang ter-occlusion cukup besar
                continue

            temp_frame = swap_face(source_face, tgt, temp_frame, occluded_ratio=occl_ratio)
        return temp_frame

    # single / fokus mode
    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)
    if not tracked:
        return temp_frame

    # hitung occlusion dan filter
    valid = []
    occl_map = {}
    for f in tracked:
        try:
            r = detect_occlusion(f, temp_frame)
        except Exception:
            r = 0.0
        occl_map[id(f)] = r
        if r < threshold:
            valid.append(f)

    if not valid:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid, reference_face)
    if best_target is None:
        best_target = valid[0]

    occl_ratio = occl_map.get(id(best_target), 0.0)
    temp_frame = swap_face(source_face, best_target, temp_frame, occluded_ratio=occl_ratio)
    return temp_frame

# -------------------- frame loop connectors --------------------
def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp, frame_number=idx)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    source_face = get_one_face(source_img)
    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame, frame_number=0)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
