from typing import Any, List, Callable
import cv2
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, is_image, is_video

NAME = 'ROOP.FACE-ENHANCER'

def pre_check() -> bool:
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    pass

def enhance_face_simple(face_image: Frame) -> Frame:
    """Very simple face enhancement - just basic improvements"""
    try:
        if face_image is None or face_image.size == 0:
            return face_image
            
        h, w = face_image.shape[:2]
        if h < 20 or w < 20:
            return face_image
        
        # 1. Basic sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(face_image, -1, kernel)
        
        # 2. Simple contrast enhancement
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
        
    except Exception as e:
        print(f"Simple enhancement error: {e}")
        return face_image

def create_blend_mask(shape: tuple) -> np.ndarray:
    """Create elliptical blend mask"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    center_y, center_x = h // 2, w // 2
    axis_x, axis_y = w // 2, h // 2
    
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return np.clip(mask, 0, 1)

def blend_images(enhanced: Frame, original: Frame, mask: np.ndarray) -> Frame:
    """Blend enhanced and original images"""
    mask_3d = np.stack([mask] * 3, axis=-1)
    blended = (enhanced * mask_3d + original * (1 - mask_3d)).astype(np.uint8)
    return blended

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face processing"""
    try:
        frame_height, frame_width = temp_frame.shape[:2]
        start_x, start_y, end_x, end_y = map(int, target_face.bbox)
        
        # Calculate face size
        face_w, face_h = end_x - start_x, end_y - start_y
        if face_w <= 15 or face_h <= 15:
            return temp_frame

        # Add padding
        padding_x = int(face_w * 0.2)
        padding_y = int(face_h * 0.2)
        
        # Ensure within frame bounds
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(frame_width, end_x + padding_x)
        end_y = min(frame_height, end_y + padding_y)
        
        # Extract face region
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        if temp_face.size == 0:
            return temp_frame
            
        if temp_face.shape[0] < 10 or temp_face.shape[1] < 10:
            return temp_frame

        # Apply enhancement
        enhanced_face = enhance_face_simple(temp_face)
        
        if enhanced_face is not None and enhanced_face.size > 0:
            # Ensure same dimensions
            if enhanced_face.shape != temp_face.shape:
                enhanced_face = cv2.resize(enhanced_face, (temp_face.shape[1], temp_face.shape[0]))
            
            # Create smooth blending mask
            blend_mask = create_blend_mask(temp_face.shape)
            
            # Blend enhanced face with original
            blended_face = blend_images(enhanced_face, temp_face, blend_mask)
            temp_frame[start_y:end_y, start_x:end_x] = blended_face
                
    except Exception as e:
        print(f"Face enhancement error: {e}")
    
    return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process all faces in frame"""
    try:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = enhance_face(target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames"""
    for temp_frame_path in temp_frame_paths:
        try:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is not None:
                result = process_frame(None, None, temp_frame)
                cv2.imwrite(temp_frame_path, result)
            if update:
                update()
        except Exception as e:
            print(f"Process frame {temp_frame_path} error: {e}")
            continue

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image"""
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is not None:
            result = process_frame(None, None, target_frame)
            cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video frames"""
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
