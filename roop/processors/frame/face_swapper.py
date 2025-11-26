from typing import Any, List, Callable, Optional
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
    get_face_pose,
    smart_face_tracking,
    get_occlusion_mask,
    detect_occlusion
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video
from roop.blending import apply_blend_and_color_match

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
    if source_img is None or source_img.size == 0:
        update_status('Failed to read source image.', NAME)
        return False
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
#  POSE-AWARE BBOX ADJUSTMENT (ENHANCED)
# =====================================================================
def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    """
    Enhanced bbox adjustment untuk pose ekstrem
    """
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Base padding factors
    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0
    
    # Handle yaw (side poses)
    if yaw > 30:  # Looking right → expand left side untuk rambut
        pad_x = w * 0.3
        pad_y_top = h * 0.1
    elif yaw < -30:  # Looking left → expand right side
        pad_x = w * 0.3
        pad_y_top = h * 0.1
    
    # Handle pitch (up/down)
    if pitch < -20:  # Looking up → lebih banyak dahi
        pad_y_top = h * 0.3
    elif pitch > 25:  # Looking down → lebih banyak dagu
        pad_y_bottom = h * 0.25
    
    # Calculate new bbox
    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))
    
    # Safety check
    if nx2 <= nx1 or ny2 <= ny1 or nx1 < 0 or ny1 < 0 or nx2 >= w_frame or ny2 >= h_frame:
        return
    
    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

# =====================================================================
#  CUSTOM BLENDING SWAP
# =====================================================================
def swap_face_with_blending(
    source_face: Face, 
    target_face: Face, 
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Enhanced swap dengan custom blending dan occlusion awareness
    """
    if source_face is None or target_face is None or temp_frame is None or temp_frame.size == 0:
        return temp_frame
    
    # Simpan crop asli sebelum modifikasi
    bbox = np.array(target_face.bbox, dtype=np.int32)
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = temp_frame.shape[:2]
    
    # Validasi bbox
    x1 = max(0, min(x1, w_frame - 1))
    x2 = max(0, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame - 1))
    y2 = max(0, min(y2, h_frame))
    
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        return temp_frame
    
    # Ambil crop asli dengan validasi
    try:
        original_crop = temp_frame[y1:y2, x1:x2].copy()
        if original_crop.size == 0:
            return temp_frame
    except Exception as e:
        print(f"Error extracting original crop: {e}")
        return temp_frame
    
    # Dapatkan occlusion mask
    try:
        occlusion_mask = get_occlusion_mask(target_face, temp_frame)
    except Exception as e:
        print(f"Error getting occlusion mask: {e}")
        occlusion_mask = None
    
    # Enhanced bbox untuk pose
    original_bbox = target_face.bbox.copy()
    adapt_bbox_for_pose(target_face, temp_frame.shape)
    
    # Dapatkan frame hasil swap (tanpa paste_back)
    try:
        swapped_result = get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=False
        )
    except Exception as e:
        print(f"Swap error: {e}")
        target_face.bbox = original_bbox  # Restore original bbox
        return temp_frame
    
    # Handle return value dari inswapper
    if isinstance(swapped_result, tuple):
        swapped_frame = swapped_result[0]
    else:
        swapped_frame = swapped_result
    
    if swapped_frame is None or swapped_frame.size == 0:
        target_face.bbox = original_bbox
        return temp_frame
    
    # Ekstrak crop hasil swap menggunakan bbox yang sudah disesuaikan
    bbox_new = np.array(target_face.bbox, dtype=np.int32)
    x1_new, y1_new, x2_new, y2_new = bbox_new
    
    # Validasi bbox baru
    x1_new = max(0, min(x1_new, w_frame - 1))
    x2_new = max(0, min(x2_new, w_frame))
    y1_new = max(0, min(y1_new, h_frame - 1))
    y2_new = max(0, min(y2_new, h_frame))
    
    if x2_new <= x1_new or y2_new <= y1_new or (x2_new - x1_new) < 5 or (y2_new - y1_new) < 5:
        target_face.bbox = original_bbox
        return temp_frame
    
    # Ambil crop hasil swap dengan validasi
    try:
        swapped_crop = swapped_frame[y1_new:y2_new, x1_new:x2_new]
        if swapped_crop.size == 0:
            target_face.bbox = original_bbox
            return temp_frame
    except Exception as e:
        print(f"Error extracting swapped crop: {e}")
        target_face.bbox = original_bbox
        return temp_frame
    
    # Blending dengan occlusion awareness
    try:
        blended_crop = apply_blend_and_color_match(
            enhanced_crop=swapped_crop,
            original_crop=original_crop,
            occlusion_mask=occlusion_mask,
            fidelity=0.7
        )
    except Exception as e:
        print(f"Blending error: {e}")
        blended_crop = swapped_crop
    
    # Tempel kembali ke frame menggunakan bbox asli (bukan yang sudah disesuaikan)
    try:
        if blended_crop.shape[:2] != (y2-y1, x2-x1):
            blended_crop = cv2.resize(blended_crop, (x2-x1, y2-y1), interpolation=cv2.INTER_LANCZOS4)
        temp_frame[y1:y2, x1:x2] = blended_crop
    except Exception as e:
        print(f"Error pasting blended crop: {e}")
    
    return temp_frame

# =====================================================================
#  CORE PROCESSING
# =====================================================================
def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    if source_face is None or temp_frame is None or temp_frame.size == 0:
        return temp_frame
    
    # Dapatkan wajah dengan tracking
    if roop.globals.many_faces:
        try:
            faces = smart_face_tracking(temp_frame, frame_number)
            if not faces:
                faces = get_many_faces(temp_frame)
            if not faces:
                return temp_frame
            
            for target_face in faces:
                # Skip wajah yang ter-occlusion berat
                if detect_occlusion(target_face, temp_frame):
                    continue
                temp_frame = swap_face_with_blending(source_face, target_face, temp_frame, frame_number)
            return temp_frame
        except Exception as e:
            print(f"Error processing many faces: {e}")
            return temp_frame
    
    # Mode single face
    try:
        tracked_faces = smart_face_tracking(temp_frame, frame_number)
        if not tracked_faces:
            tracked_faces = get_many_faces(temp_frame)
        if not tracked_faces:
            return temp_frame
        
        # Filter wajah yang ter-occlusion
        valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
        if not valid_faces:
            return temp_frame
        
        # Pilih wajah terbaik berdasarkan reference atau wajah pertama yang valid
        best_target = None
        if reference_face is not None and hasattr(reference_face, 'normed_embedding'):
            best_similarity = -1
            for face in valid_faces:
                if hasattr(face, 'normed_embedding'):
                    similarity = 1.0 - np.dot(face.normed_embedding, reference_face.normed_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = face
        
        if best_target is None:
            best_target = valid_faces[0]
        
        return swap_face_with_blending(source_face, best_target, temp_frame, frame_number)
    except Exception as e:
        print(f"Error in single face processing: {e}")
        return temp_frame

def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None or source_img.size == 0:
        print(f"Failed to read source image: {source_path}")
        return
    
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        # Cek apakah file ada sebelum membaca
        import os
        if not os.path.exists(temp_frame_path):
            print(f"Frame file not found: {temp_frame_path}")
            continue
        
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None or temp_frame.size == 0:
            print(f"Failed to read frame: {temp_frame_path}")
            continue
        
        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx
        )
        
        # Validasi sebelum menulis
        if result is not None and result.size > 0:
            cv2.imwrite(temp_frame_path, result)
        else:
            print(f"Skipping empty result for frame: {temp_frame_path}")
        
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None or source_img.size == 0:
        print(f"Failed to read source image: {source_path}")
        return
    
    target_frame = cv2.imread(target_path)
    if target_frame is None or target_frame.size == 0:
        print(f"Failed to read target image: {target_path}")
        return
    
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
    
    if result is not None and result.size > 0:
        cv2.imwrite(output_path, result)
    else:
        print(f"Failed to generate result for image")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            if 0 <= ref_idx < len(temp_frame_paths):
                reference_frame_path = temp_frame_paths[ref_idx]
                reference_frame = cv2.imread(reference_frame_path)
                if reference_frame is not None and reference_frame.size > 0:
                    reference_face = get_one_face(
                        reference_frame,
                        roop.globals.reference_face_position
                    )
                    set_face_reference(reference_face)
        except Exception as e:
            print(f"Error setting face reference: {e}")
            set_face_reference(None)
    
    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
