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
    detect_occlusion
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    """
    Inisialisasi model ReaSwapper 256.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # ✅ UPDATED: Gunakan ReaSwapper 256
            model_path = resolve_relative_path('../models/reaSwapper_256.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers,
                # ✅ Optimasi khusus untuk ReaSwapper 256
                session_options=roop.globals.session_options
            )
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    Pastikan model ReaSwapper 256 sudah ke-download.
    """
    download_directory_path = resolve_relative_path('../models')
    # ✅ UPDATED: Download link untuk ReaSwapper 256
    conditional_download(download_directory_path, [
        'https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/reswapper_256.onnx'
    ])
    return True


def pre_start() -> bool:
    """
    Validasi dengan optimasi untuk ReaSwapper 256.
    """
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    source_face = get_one_face(source_img)
    
    if not source_face:
        update_status('No face in source path detected.', NAME)
        return False

    # ✅ ENHANCED: Validasi kualitas source face untuk ReaSwapper 256
    if hasattr(source_face, 'det_score') and source_face.det_score < 0.5:
        update_status('Source face quality too low for ReaSwapper 256.', NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False

    return True


def post_process() -> None:
    """
    Bersihkan model & reference setelah selesai.
    """
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dengan optimasi ReaSwapper 256.
    """
    if source_face is None or target_face is None:
        return temp_frame

    try:
        # ✅ ENHANCED: Optimasi parameter untuk ReaSwapper 256
        result = get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )
        
        # ✅ ENHANCED: Post-processing untuk hasil lebih natural
        if result is not None and roop.globals.face_enhancer:
            result = enhance_swapped_face(result, target_face)
            
        return result
    except Exception as e:
        print(f"[ReaSwapper] Swap failed: {e}")
        return temp_frame


def enhance_swapped_face(swapped_frame: Frame, original_face: Face) -> Frame:
    """
    Enhancement khusus untuk hasil ReaSwapper 256.
    """
    try:
        # Soft blending untuk hasil lebih natural
        alpha = 0.95  # Blending factor
        if hasattr(original_face, 'bbox'):
            bbox = original_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(swapped_frame.shape[1], x2), min(swapped_frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # Extract face region
                face_region = swapped_frame[y1:y2, x1:x2]
                if face_region.size > 0:
                    # Apply subtle Gaussian blur for blending
                    blended = cv2.GaussianBlur(face_region, (3, 3), 0)
                    swapped_frame[y1:y2, x1:x2] = cv2.addWeighted(
                        face_region, alpha, blended, 1 - alpha, 0
                    )
    except Exception:
        pass
    
    return swapped_frame


def _select_best_target_by_embedding(
    faces: List[Face],
    reference_face: Face
) -> Face | None:
    """
    Pilih wajah target terbaik dengan threshold optimal untuk ReaSwapper 256.
    """
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')

    # ✅ UPDATED: Threshold lebih ketat untuk ReaSwapper 256
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 0.6)  # Lebih ketat

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue

        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue

        # ✅ ENHANCED: Tambah filter kualitas wajah
        face_quality = getattr(f, 'det_score', 1.0)
        if distance < similar_threshold and distance < best_distance and face_quality > 0.4:
            best_distance = distance
            best_face = f

    return best_face


def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Proses frame dengan optimasi ReaSwapper 256.
    """
    if source_face is None:
        return temp_frame

    # MODE: banyak wajah → swap semua yang valid
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            # ✅ ENHANCED: Filter lebih ketat untuk ReaSwapper 256
            if detect_occlusion(target_face) or getattr(target_face, 'det_score', 0) < 0.4:
                continue

            temp_frame = swap_face(source_face, target_face, temp_frame)

        return temp_frame

    # MODE: single / fokus 1 wajah
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    # ✅ ENHANCED: Filter kualitas untuk ReaSwapper 256
    valid_faces = [
        f for f in tracked_faces 
        if not detect_occlusion(f) and getattr(f, 'det_score', 0) >= 0.4
    ]
    
    if not valid_faces:
        return temp_frame

    best_target = None

    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame


# ✅ FUNGSI BARU: Pre-process frame untuk ReaSwapper 256
def preprocess_frame(frame: Frame) -> Frame:
    """
    Pre-processing frame untuk optimasi ReaSwapper 256.
    """
    try:
        # Normalize brightness/contrast
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
        return frame
    except Exception:
        return frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    """
    Proses frames dengan optimasi ReaSwapper 256.
    """
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        
        # ✅ ENHANCED: Pre-process frame
        temp_frame = preprocess_frame(temp_frame)
        
        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx
        )
        cv2.imwrite(temp_frame_path, result)

        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Proses image dengan optimasi ReaSwapper 256.
    """
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    
    # ✅ ENHANCED: Pre-process target frame
    target_frame = preprocess_frame(target_frame)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_frame,
            roop.globals.reference_face_position
        )

    result = process_frame(
        source_face=source_face,
        reference_face=reference_face,
        temp_frame=target_frame,
        frame_number=0
    )
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Entry point untuk video dengan ReaSwapper 256.
    """
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_frame = preprocess_frame(reference_frame)  # ✅ ENHANCED
            reference_face = get_one_face(
                reference_frame,
                roop.globals.reference_face_position
            )
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
