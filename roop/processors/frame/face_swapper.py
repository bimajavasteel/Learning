from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np

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

# Configuration
DETECTION_THRESHOLD = 0.3  # Lowered threshold for better face detection
BLEND_EXPAND_PIXELS = 25   # Expanded blending area
GAUSSIAN_BLUR_SIZE = (45, 45)  # Larger blur for smoother blending

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            # Try to load 256 model first, fallback to 128
            try:
                model_path_256 = resolve_relative_path('../models/inswapper_256.onnx')
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path_256, providers=roop.globals.execution_providers)
                print("Loaded inswapper_256.onnx model")
            except:
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
                print("Loaded inswapper_128.onnx model (fallback)")
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx',
        'https://huggingface.co/ezioruan/inswapper_128_fp16/resolve/main/inswapper_128_fp16.onnx'
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    
    # Preprocess source image for better quality
    source_image = preprocess_source_image(cv2.imread(roop.globals.source_path))
    source_face = get_one_face(source_image)
    
    if not source_face:
        update_status('No face in source path detected.', NAME)
        return False
    
    if hasattr(source_face, 'det_score') and source_face.det_score < 0.5:  # Slightly lowered threshold
        update_status('Source face detection score low. Use a clearer source image.', NAME)
        return False
        
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def preprocess_source_image(source_image: Frame) -> Frame:
    """
    Enhance source image quality for better face swapping
    """
    try:
        # Apply gentle enhancement
        enhanced = cv2.detailEnhance(source_image, sigma_s=8, sigma_r=0.12)
        # Reduce noise while preserving edges
        denoised = cv2.medianBlur(enhanced, 3)
        # Sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return sharpened
    except Exception as e:
        print(f"Source image preprocessing failed: {e}")
        return source_image

def apply_color_correction(result_roi: Frame, original_roi: Frame) -> Frame:
    """
    Apply color matching between swapped face and original face
    """
    try:
        if result_roi.size == 0 or original_roi.size == 0:
            return result_roi
        
        # Ensure same dimensions
        if result_roi.shape != original_roi.shape:
            result_roi = cv2.resize(result_roi, (original_roi.shape[1], original_roi.shape[0]))
        
        # Convert to float for calculations
        result_float = result_roi.astype(np.float32)
        original_float = original_roi.astype(np.float32)
        
        # Calculate mean and standard deviation
        result_mean = np.mean(result_float, axis=(0, 1))
        result_std = np.std(result_float, axis=(0, 1))
        original_mean = np.mean(original_float, axis=(0, 1))
        original_std = np.std(original_float, axis=(0, 1))
        
        # Avoid division by zero
        result_std[result_std < 1] = 1
        original_std[original_std < 1] = 1
        
        # Color correction formula
        corrected = (result_float - result_mean) * (original_std / result_std) + original_mean
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
        
    except Exception as e:
        print(f"Color correction failed: {e}")
        return result_roi

def enhance_face_blending(result_frame: Frame, original_frame: Frame, target_face: Face) -> Frame:
    """
    Improved blending between swapped face and original frame
    """
    try:
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Expand blending area
        h, w = result_frame.shape[:2]
        x1 = max(0, x1 - BLEND_EXPAND_PIXELS)
        y1 = max(0, y1 - BLEND_EXPAND_PIXELS)
        x2 = min(w, x2 + BLEND_EXPAND_PIXELS)
        y2 = min(h, y2 + BLEND_EXPAND_PIXELS)
        
        # Check valid dimensions
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return result_frame
        
        result_face_roi = result_frame[y1:y2, x1:x2]
        original_face_roi = original_frame[y1:y2, x1:x2]
        
        if result_face_roi.size == 0 or original_face_roi.size == 0:
            return result_frame
        
        # Resize if dimensions don't match
        if result_face_roi.shape != original_face_roi.shape:
            result_face_roi = cv2.resize(result_face_roi, (original_face_roi.shape[1], original_face_roi.shape[0]))
        
        # Create elliptical mask for natural blending
        mask = np.zeros(result_face_roi.shape[:2], dtype=np.float32)
        center_x, center_y = result_face_roi.shape[1] // 2, result_face_roi.shape[0] // 2
        radius_x = result_face_roi.shape[1] // 2 - 8
        radius_y = result_face_roi.shape[0] // 2 - 8
        
        # Draw filled ellipse
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1, -1)
        
        # Apply Gaussian blur for smooth transition
        mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR_SIZE, 0)
        
        # Apply color correction before blending
        color_corrected_roi = apply_color_correction(result_face_roi, original_face_roi)
        
        # Blend using the mask
        mask_3d = np.stack([mask] * 3, axis=-1)
        blended_roi = (color_corrected_roi * mask_3d + original_face_roi * (1 - mask_3d)).astype(np.uint8)
        
        result_frame[y1:y2, x1:x2] = blended_roi
        
    except Exception as e:
        print(f"Blending enhancement failed: {e}")
    
    return result_frame

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Main face swapping function with enhanced processing
    """
    original_frame = temp_frame.copy()
    
    try:
        # Perform face swap
        result_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply enhanced blending
        result_frame = enhance_face_blending(result_frame, original_frame, target_face)
        
    except Exception as e:
        print(f"Face swap failed: {e}")
        return original_frame
    
    return result_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Process individual frame with improved face detection
    """
    if roop.globals.many_faces:
        # Process multiple faces
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                det_score = getattr(target_face, 'det_score', 1.0)
                if det_score > DETECTION_THRESHOLD:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        # Process single face
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            det_score = getattr(target_face, 'det_score', 1.0)
            if det_score > DETECTION_THRESHOLD:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Process all frames in sequence
    """
    # Preprocess source image for better quality
    source_image = preprocess_source_image(cv2.imread(source_path))
    source_face = get_one_face(source_image)
    
    if not source_face:
        print("No source face found!")
        return
        
    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    for i, temp_frame_path in enumerate(temp_frame_paths):
        try:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                continue
                
            result = process_frame(source_face, reference_face, temp_frame)
            cv2.imwrite(temp_frame_path, result)
            
            if update and i % 10 == 0:  # Update every 10 frames
                update()
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Process single image
    """
    # Preprocess source image
    source_image = preprocess_source_image(cv2.imread(source_path))
    source_face = get_one_face(source_image)
    
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Process video frames
    """
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        if reference_face:
            set_face_reference(reference_face)
        else:
            print("No reference face found in video!")
            return
            
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
