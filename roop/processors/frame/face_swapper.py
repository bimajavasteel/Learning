# face_swap_patched.py
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

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0
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

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame
    adapt_bbox_for_pose(target_face, temp_frame.shape)
    try:
        return get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )
    except Exception:
        # fail-safe: kalau swap gagal, kembalikan frame asli
        return temp_frame

def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Face | None:
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
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue
        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f
    return best_face

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame
    # many_faces mode
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame
        # For each face, use detect_occlusion with prev_face from tracking store if available
        for target_face in faces:
            # retrieve prev_face if tracked
            prev_face = None
            try:
                # simple: if face has attribute track_id store it earlier; else skip
                track_id = getattr(target_face, "track_id", None)
                if track_id is not None:
                    prev_face = None  # track history retrieval omitted here — rely on face_analyser tracking state
            except Exception:
                prev_face = None
            if detect_occlusion(target_face, temp_frame, prev_face):
                # skip swap for occluded face
                continue
            temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame

    # single mode
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)
    if not tracked_faces:
        return temp_frame

    # pass prev_face via simple heuristic: find by closest bbox center (optional)
    # Filter occlusion
    valid_faces = []
    for f in tracked_faces:
        # attempt to find prev_face from face tracking store (if available)
        prev_face = None
        try:
            # read last_seen mapping from roop.face_analyser (not imported here to avoid circular)
            prev_face = None
        except Exception:
            prev_face = None
        if not detect_occlusion(f, temp_frame, prev_face):
            valid_faces.append(f)

    if not valid_faces:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)
    if best_target is None:
        best_target = valid_faces[0]
    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=temp_frame, frame_number=idx)
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
    result = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=target_frame, frame_number=0)
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
