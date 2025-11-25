import copy
import threading
from typing import Any, List, Callable, Optional

import cv2
import insightface
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
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    """
    Inisialisasi model inswapper.
    """
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


# =====================================================================
#  ENHANCED POSE-AWARE BBOX ADJUSTMENT
# =====================================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    """
    Enhanced pose-aware bbox adjustment dengan ekspansi lebih agresif untuk sudut ekstrim.
    """
    pitch, yaw, roll = get_face_pose(face)

    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # BASE PADDING + POSE-AWARE EXPANSION
    base_pad = 0.20  # Increased base padding
    pad_x = base_pad
    pad_y_top = base_pad  
    pad_y_bottom = base_pad

    # EXTREME ANGLE COMPENSATION
    yaw_abs = abs(yaw)
    if yaw_abs > 30.0:
        # More aggressive expansion for extreme angles
        extra = (yaw_abs - 30.0) * 0.025  # Increased from 0.02
        extra = min(extra, 0.35)  # Increased max from 0.20
        pad_x += extra

    # Enhanced pitch compensation
    if pitch < -20.0:  # Looking up
        extra = (abs(pitch) - 20.0) * 0.025
        extra = min(extra, 0.35)
        pad_y_top += extra
    elif pitch > 25.0:  # Looking down  
        extra = (pitch - 25.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_bottom += extra

    # Roll compensation (untuk kepala miring)
    roll_abs = abs(roll)
    if roll_abs > 25.0:
        roll_factor = min(roll_abs * 0.01, 0.15)
        pad_x += roll_factor
        pad_y_top += roll_factor * 0.5
        pad_y_bottom += roll_factor * 0.5

    # Apply padding
    pad_x_pixels = int(w * pad_x)
    pad_y_top_pixels = int(h * pad_y_top)
    pad_y_bottom_pixels = int(h * pad_y_bottom)

    nx1 = max(0, x1 - pad_x_pixels)
    nx2 = min(w_frame, x2 + pad_x_pixels)
    ny1 = max(0, y1 - pad_y_top_pixels)
    ny2 = min(h_frame, y2 + pad_y_bottom_pixels)

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# =====================================================================
#  ROTATION-AWARE BLENDING SYSTEM
# =====================================================================

def create_rotated_ellipse_mask(shape, center, axes, roll_angle):
    """
    Buat elliptical mask yang dirotasi sesuai roll wajah.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Ellipse dengan rotasi berdasarkan roll
    cv2.ellipse(mask, center, axes, roll_angle, 0, 360, 1.0, -1)
    return mask

def enhanced_face_blending(swapped_frame: Frame, original_frame: Frame, face: Face) -> Frame:
    """
    Enhanced blending dengan rotated elliptical mask.
    """
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        swapped_region = swapped_frame[y1:y2, x1:x2]
        original_region = original_frame[y1:y2, x1:x2]
        
        if swapped_region.size == 0 or original_region.size == 0:
            return swapped_frame
            
        h, w = swapped_region.shape[:2]
        
        # Dapatkan pose wajah untuk rotasi
        pitch, yaw, roll = get_face_pose(face)
        
        # Buat mask dengan rotasi sesuai roll wajah
        center = (w // 2, h // 2)
        ellipse_w = int(w * 0.38)
        ellipse_h = int(h * 0.38)
        
        # Mask utama dengan rotasi
        mask = create_rotated_ellipse_mask((h, w), center, (ellipse_w, ellipse_h), roll)
        
        # Feathering dengan mempertimbangkan orientasi wajah
        feather_size = int(min(w, h) * 0.18)
        if feather_size % 2 == 0:
            feather_size += 1
            
        # Multiple blur passes
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        
        # Adaptive dilation berdasarkan yaw (untuk wajah samping)
        if abs(yaw) > 30.0:
            dilation_factor = min(abs(yaw) * 0.01, 0.3)
            kernel_size = max(3, int(min(w, h) * dilation_factor))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel)
            mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        
        mask_3ch = np.dstack([mask] * 3)
        
        # Apply blending
        blended_region = (swapped_region * mask_3ch + 
                         original_region * (1.0 - mask_3ch)).astype(np.uint8)
        
        # Border smoothing yang juga mengikuti rotasi
        border_size = int(min(w, h) * 0.08)
        if border_size > 1:
            border_mask = np.zeros((h, w), dtype=np.float32)
            # Inner ellipse (rotated)
            cv2.ellipse(border_mask, center, 
                       (ellipse_w - border_size, ellipse_h - border_size), 
                       roll, 0, 360, 1, -1)
            # Outer ellipse (rotated)  
            cv2.ellipse(border_mask, center, (ellipse_w, ellipse_h), 
                       roll, 0, 360, 0, -1)
            
            border_blur = border_size * 2 + 1
            border_mask = cv2.GaussianBlur(border_mask, (border_blur, border_blur), 0)
            border_mask_3ch = np.dstack([border_mask] * 3)
            
            blended_region = (blended_region * (1 - border_mask_3ch) + 
                            original_region * border_mask_3ch).astype(np.uint8)
        
        result_frame = swapped_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_region
        
        return result_frame
        
    except Exception as e:
        print(f"Enhanced blending error: {e}")
        return swapped_frame


def standard_face_blending(swapped_frame: Frame, original_frame: Frame, face: Face) -> Frame:
    """
    Standard blending dengan rotasi dasar.
    """
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        swapped_region = swapped_frame[y1:y2, x1:x2]
        original_region = original_frame[y1:y2, x1:x2]
        
        if swapped_region.size == 0:
            return swapped_frame
            
        h, w = swapped_region.shape[:2]
        
        # Dapatkan roll untuk rotasi
        pitch, yaw, roll = get_face_pose(face)
        
        center = (w // 2, h // 2)
        ellipse_w = int(w * 0.42)
        ellipse_h = int(h * 0.42)
        
        # Rotated ellipse mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, (ellipse_w, ellipse_h), roll, 0, 360, 1.0, -1)
        
        feather_size = int(min(w, h) * 0.12)
        if feather_size % 2 == 0:
            feather_size += 1
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        
        mask_3ch = np.dstack([mask] * 3)
        
        blended_region = (swapped_region * mask_3ch + 
                         original_region * (1.0 - mask_3ch)).astype(np.uint8)
        
        result_frame = swapped_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_region
        
        return result_frame
        
    except Exception as e:
        print(f"Standard blending error: {e}")
        return swapped_frame


# =====================================================================
#  CORE SWAP WITH ENHANCED BLENDING (FIXED COPY ISSUE)
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhanced swap dengan advanced blending system.
    """
    if source_face is None or target_face is None:
        return temp_frame

    # Deteksi angle untuk menentukan blending strategy
    pitch, yaw, roll = get_face_pose(target_face)
    is_extreme_angle = abs(yaw) > 45.0 or abs(pitch) > 35.0

    # Simpan frame original untuk blending
    original_frame = temp_frame.copy()
    
    # Simpan bbox asli untuk restore nanti
    original_bbox = target_face.bbox.copy() if hasattr(target_face, 'bbox') else None
    
    # Enhanced BBOX adjustment
    adapt_bbox_for_pose(target_face, temp_frame.shape)

    try:
        # Perform swap
        swapped_frame = get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )

        # Pilih blending strategy berdasarkan angle
        if is_extreme_angle:
            return enhanced_face_blending(swapped_frame, original_frame, target_face)
        else:
            return standard_face_blending(swapped_frame, original_frame, target_face)
    
    except Exception as e:
        print(f"Swap face error: {e}")
        return temp_frame
    finally:
        # Restore original bbox untuk menjaga konsistensi tracking
        if original_bbox is not None:
            target_face.bbox = original_bbox


def _select_best_target_by_embedding(
    faces: List[Face],
    reference_face: Face
) -> Optional[Face]:
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


def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    if source_face is None:
        return temp_frame

    # MODE: Many Faces
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            if detect_occlusion(target_face, temp_frame):
                continue
            temp_frame = swap_face(source_face, target_face, temp_frame)

        return temp_frame

    # MODE: Single Face
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
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
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
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
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
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
