from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
from scipy import ndimage
import os
import time

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
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
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128_fp16.onnx'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def safe_get_landmarks(face: Face) -> Optional[np.ndarray]:
    """Safely get landmarks from face object with comprehensive checks"""
    if face is None:
        return None
    
    # Try different attribute names for landmarks
    landmark_attrs = ['landmark_2d_106', 'landmark_2d', 'kps', 'landmarks']
    
    for attr in landmark_attrs:
        if hasattr(face, attr):
            landmarks = getattr(face, attr)
            if landmarks is not None and len(landmarks) > 0:
                return landmarks
    
    return None

def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray]:
    """Robust face alignment dengan comprehensive error handling"""
    try:
        # Check if faces have landmarks
        source_landmarks = safe_get_landmarks(source_face)
        target_landmarks = safe_get_landmarks(target_face)
        
        if source_landmarks is None or target_landmarks is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)
        
        # Verify landmark dimensions
        if len(source_landmarks) < 5 or len(target_landmarks) < 5:
            return temp_frame, np.eye(2, 3, dtype=np.float32)
        
        # Use key facial points yang universally available
        key_points = []
        landmark_indices = [0, 1, 2, 3, 4]  # Simple points: eyes, nose
        
        for idx in landmark_indices:
            if idx < len(source_landmarks) and idx < len(target_landmarks):
                key_points.append(idx)
        
        if len(key_points) < 3:  # Minimum points for affine transformation
            return temp_frame, np.eye(2, 3, dtype=np.float32)
        
        src_points = np.array([source_landmarks[i] for i in key_points], dtype=np.float32)
        dst_points = np.array([target_landmarks[i] for i in key_points], dtype=np.float32)
        
        # Calculate transformation matrix
        transform_matrix = cv2.estimateAffinePartial2D(
            src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0
        )[0]
        
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
            return aligned_frame, transform_matrix
        
        return temp_frame, np.eye(2, 3, dtype=np.float32)
        
    except Exception as e:
        # print(f"Robust face alignment error: {e}")  # Skip logging untuk performance
        return temp_frame, np.eye(2, 3, dtype=np.float32)

def fast_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Fast color correction dengan error handling"""
    try:
        if target_face is None or swapped_face is None:
            return swapped_face
        
        # Extract target face region
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        target_region = target_frame[y1:y2, x1:x2]
        
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        
        # Resize jika diperlukan
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        
        # Simple LAB color correction
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        
        # Calculate mean and std
        swapped_mean = np.mean(swapped_lab, axis=(0, 1))
        swapped_std = np.std(swapped_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # Avoid division by zero
        swapped_std = np.where(swapped_std == 0, 1, swapped_std)
        target_std = np.where(target_std == 0, 1, target_std)
        
        # Color correction
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Gentle blending
        blend_ratio = 0.6
        result_face = cv2.addWeighted(swapped_face, 0.4, corrected_face, 0.6, 0)
        
        return result_face
        
    except Exception as e:
        # print(f"Color correction error: {e}")
        return swapped_face

def create_simple_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Create simple mask dengan error handling"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Create elliptical mask
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Create ellipse
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 1.0, -1)
        
        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        
        return np.clip(mask, 0, 1)
        
    except Exception as e:
        # print(f"Mask creation error: {e}")
        # Fallback to simple rectangular mask
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

def seamless_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Seamless blending dengan error handling"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Ensure swapped face has correct size
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Create mask
        mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
        
        # Get center point for blending
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Use seamless clone
        result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        
        return result
        
    except Exception as e:
        # print(f"Seamless blending error: {e}")
        return simple_face_blending(swapped_face, target_frame, target_face)

def simple_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Simple blending fallback"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Ensure swapped face has correct size
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Create smooth mask
        mask = create_simple_mask(target_face, target_frame.shape)
        mask_region = mask[y1:y2, x1:x2]
        
        # Ensure mask has correct dimensions
        if mask_region.shape != swapped_face.shape[:2]:
            mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
        
        # Create 3-channel mask
        mask_3d = np.stack([mask_region] * 3, axis=-1)
        
        # Blend
        result = target_frame.copy()
        face_region = result[y1:y2, x1:x2]
        blended_face = (swapped_face * mask_3d + face_region * (1 - mask_3d)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended_face
        
        return result
        
    except Exception as e:
        # print(f"Simple blending error: {e}")
        return target_frame

def enhance_face_quality(face: Frame) -> Frame:
    """Simple face quality enhancement"""
    try:
        if face is None:
            return face
            
        # Mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.15
        
        sharpened = cv2.filter2D(face, -1, kernel)
        
        # Mild bilateral filter
        denoised = cv2.bilateralFilter(sharpened, 5, 15, 15)
        
        return denoised
        
    except Exception as e:
        # print(f"Face enhancement error: {e}")
        return face

def ensure_frame_format(frame: Any) -> Optional[Frame]:
    """Ensure the frame is in correct numpy array format"""
    if frame is None:
        return None
    
    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
        return frame
    
    if isinstance(frame, tuple):
        try:
            frame_array = np.array(frame)
            if frame_array.size > 0:
                return frame_array
        except:
            pass
    
    return None

def swap_face_optimized(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping dengan minimal error handling"""
    try:
        # Apply robust face alignment
        aligned_frame, _ = robust_face_alignment(source_face, target_face, temp_frame)
        
        # Get basic face swap
        swapped_result = get_face_swapper().get(aligned_frame, target_face, source_face, paste_back=False)
        
        # Ensure proper format
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply color correction
        swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)
        
        # Enhance face quality
        swapped_frame = enhance_face_quality(swapped_frame)
        
        # Apply blending
        result_frame = seamless_face_blending(swapped_frame, temp_frame, target_face)
        
        return result_frame
        
    except Exception as e:
        # print(f"Face swap error: {e}")
        # Fallback to original face swapper
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process single frame dengan optimized face swapping"""
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        # print(f"Process frame error: {e}")
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        
        for temp_frame_path in temp_frame_paths:
            try:
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is not None:
                    result = process_frame(source_face, reference_face, temp_frame)
                    cv2.imwrite(temp_frame_path, result)
                if update:
                    update()
            except Exception as e:
                # print(f"Error processing frame {temp_frame_path}: {e}")
                continue
    except Exception as e:
        print(f"Process frames error: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video"""
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Process video error: {e}")
