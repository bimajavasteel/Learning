from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
from scipy import ndimage
import os

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
    conditional_download(download_directory_path, ['https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'])
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


def ensure_frame_format(frame: Any) -> Optional[Frame]:
    """Ensure the frame is in correct numpy array format"""
    if frame is None:
        return None
    
    # If it's already a numpy array with correct shape
    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
        return frame
    
    # If it's a tuple (likely from face swapper), convert to numpy array
    if isinstance(frame, tuple):
        try:
            # Try to convert tuple to numpy array
            frame_array = np.array(frame)
            if frame_array.size > 0:
                return frame_array
        except:
            pass
    
    return None


def simple_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Simple and robust color correction"""
    try:
        if target_face is None:
            return swapped_face
        
        # Extract target face region for color reference
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        target_region = target_frame[y1:y2, x1:x2]
        
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        
        # Resize swapped face to match target region if needed
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        
        # Convert to LAB color space for better color matching
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        
        # Calculate mean and standard deviation for each channel
        swapped_mean, swapped_std = np.mean(swapped_lab, axis=(0,1)), np.std(swapped_lab, axis=(0,1))
        target_mean, target_std = np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))
        
        # Avoid division by zero
        swapped_std = np.where(swapped_std == 0, 1, swapped_std)
        target_std = np.where(target_std == 0, 1, target_std)
        
        # Color correction
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Blend with original for natural look
        blend_ratio = 0.7
        result_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, corrected_face, blend_ratio, 0)
        
        return result_face
        
    except Exception as e:
        print(f"Simple color correction error: {e}")
        return swapped_face


def create_smooth_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Create smooth mask for blending"""
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
        print(f"Mask creation error: {e}")
        # Fallback to simple rectangular mask
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask


def enhance_face_quality(face: Frame) -> Frame:
    """Simple face quality enhancement"""
    try:
        if face is None:
            return face
            
        # Ensure it's a numpy array
        face_array = ensure_frame_format(face)
        if face_array is None:
            return face
            
        # Mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.2
        
        sharpened = cv2.filter2D(face_array, -1, kernel)
        
        # Mild bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(sharpened, 5, 25, 25)
        
        return denoised
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return face


def seamless_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Use OpenCV's seamlessClone for better blending"""
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
        
        # Use seamless clone for natural blending
        result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        
        return result
        
    except Exception as e:
        print(f"Seamless blending error: {e}")
        # Fallback to simple blending
        return simple_blending(swapped_face, target_frame, target_face)


def simple_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Simple alpha blending fallback"""
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
        mask = create_smooth_mask(target_face, target_frame.shape)
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
        print(f"Simple blending error: {e}")
        return target_frame


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Robust face swapping with error handling"""
    try:
        # Get basic face swap
        swapped_result = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        
        # Ensure proper format
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            # Fallback to original method
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply color correction
        swapped_frame = simple_color_correction(swapped_frame, temp_frame, target_face)
        
        # Enhance face quality
        swapped_frame = enhance_face_quality(swapped_frame)
        
        # Apply blending
        result_frame = seamless_blending(swapped_frame, temp_frame, target_face)
        
        return result_frame
        
    except Exception as e:
        print(f"Face swap error: {e}")
        # Fallback to original face swapper
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process single frame with enhanced face swapping"""
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames with enhanced face swapping"""
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
                print(f"Error processing frame {temp_frame_path}: {e}")
                continue
    except Exception as e:
        print(f"Process frames error: {e}")


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image with enhanced face swapping"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video with enhanced face swapping"""
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Process video error: {e}")
