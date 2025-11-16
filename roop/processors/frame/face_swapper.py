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

# Enhanced configuration
ENHANCE_CONFIG = {
    'blend_ratio': 0.7,
    'color_correction_strength': 0.8,
    'sharpness_enhance': 1.2,
    'smooth_edges': True,
    'adaptive_lighting': True
}


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


def adaptive_color_correction(source_face: Face, target_face: Face, swapped_face: Frame, target_frame: Frame) -> Frame:
    """
    Advanced color correction to match lighting conditions between source and target
    """
    try:
        # Extract face regions for color analysis
        source_region = extract_face_region(source_face, cv2.imread(roop.globals.source_path))
        target_region = extract_face_region(target_face, target_frame)
        
        if source_region is not None and target_region is not None:
            # Calculate mean colors for simple color matching
            source_mean = np.mean(source_region, axis=(0, 1))
            target_mean = np.mean(target_region, axis=(0, 1))
            swapped_mean = np.mean(swapped_face, axis=(0, 1))
            
            # Calculate color adjustment ratios
            color_ratio = target_mean / (source_mean + 1e-6)
            color_ratio = np.clip(color_ratio, 0.8, 1.2)
            
            # Apply color correction
            corrected_face = np.clip(swapped_face.astype(np.float32) * color_ratio, 0, 255).astype(np.uint8)
            
            # Blend with original for natural look
            blend_ratio = 0.6
            swapped_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, corrected_face, blend_ratio, 0)
        
        # Additional brightness and contrast matching
        swapped_face = match_brightness_contrast(swapped_face, target_region)
        
    except Exception as e:
        print(f"Color correction error: {e}")
        # Return original if correction fails
        pass
    
    return swapped_face


def extract_face_region(face: Face, frame: Frame) -> Optional[Frame]:
    """Extract the face region from frame"""
    if face is None or frame is None:
        return None
    
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        # Expand the region slightly for better color analysis
        margin = 5
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face_region = frame[y1:y2, x1:x2]
        
        # Check if region is valid
        if face_region.size == 0 or face_region.shape[0] == 0 or face_region.shape[1] == 0:
            return None
            
        return face_region
    except Exception as e:
        print(f"Error extracting face region: {e}")
        return None


def match_brightness_contrast(source_face: Frame, target_region: Optional[Frame]) -> Frame:
    """Match brightness and contrast between source and target"""
    if target_region is None or target_region.size == 0:
        return source_face
    
    try:
        # Convert to LAB color space for better brightness matching
        source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        
        # Calculate mean and std for L channel (lightness)
        source_l_mean, source_l_std = np.mean(source_lab[:,:,0]), np.std(source_lab[:,:,0])
        target_l_mean, target_l_std = np.mean(target_lab[:,:,0]), np.std(target_lab[:,:,0])
        
        # Adjust brightness and contrast
        source_lab[:,:,0] = np.clip(
            (source_lab[:,:,0] - source_l_mean) * (target_l_std / (source_l_std + 1e-6)) + target_l_mean,
            0, 255
        )
        
        # Convert back to BGR
        result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
        return result
        
    except Exception as e:
        print(f"Brightness matching error: {e}")
        return source_face


def create_enhanced_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Create enhanced mask with smooth edges and proper feathering"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    
    try:
        if hasattr(face, 'kps') and face.kps is not None:
            # Use facial landmarks for precise masking
            landmarks = face.kps.astype(np.int32)
            
            # Create convex hull from landmarks
            hull = cv2.convexHull(landmarks)
            cv2.fillConvexPoly(mask, hull, 1.0)
            
            # Apply Gaussian blur for smooth edges
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            # Enhance mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Final smoothing
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
        else:
            # Fallback to bbox-based mask
            x1, y1, x2, y2 = map(int, face.bbox)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min(x2 - x1, y2 - y1) // 2
            
            cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (25, 25), 0)
    
    except Exception as e:
        print(f"Mask creation error: {e}")
        # Fallback: create simple rectangular mask
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    return np.clip(mask, 0, 1)


def pyramid_blending(src: Frame, dst: Frame, mask: np.ndarray, levels: int = 4) -> Frame:
    """Multi-resolution pyramid blending for seamless integration"""
    try:
        # Ensure all inputs have same dimensions
        if src.shape != dst.shape:
            src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
        if mask.shape != src.shape[:2]:
            mask = cv2.resize(mask, (src.shape[1], src.shape[0]))
        
        # Convert to float for processing
        src = src.astype(np.float32)
        dst = dst.astype(np.float32)
        mask = mask.astype(np.float32)
        
        # Generate Gaussian pyramid for mask
        G_mask = mask.copy()
        gp_mask = [G_mask]
        for i in range(levels):
            G_mask = cv2.pyrDown(G_mask)
            gp_mask.append(G_mask)
        
        # Generate Laplacian pyramid for source
        G_src = src.copy()
        gp_src = [G_src]
        for i in range(levels):
            G_src = cv2.pyrDown(G_src)
            gp_src.append(G_src)
        
        lp_src = [gp_src[levels]]
        for i in range(levels, 0, -1):
            GE = cv2.pyrUp(gp_src[i])
            GE = cv2.resize(GE, (gp_src[i-1].shape[1], gp_src[i-1].shape[0]))
            L = cv2.subtract(gp_src[i-1], GE)
            lp_src.append(L)
        
        # Generate Laplacian pyramid for destination
        G_dst = dst.copy()
        gp_dst = [G_dst]
        for i in range(levels):
            G_dst = cv2.pyrDown(G_dst)
            gp_dst.append(G_dst)
        
        lp_dst = [gp_dst[levels]]
        for i in range(levels, 0, -1):
            GE = cv2.pyrUp(gp_dst[i])
            GE = cv2.resize(GE, (gp_dst[i-1].shape[1], gp_dst[i-1].shape[0]))
            L = cv2.subtract(gp_dst[i-1], GE)
            lp_dst.append(L)
        
        # Blend pyramids
        LS = []
        for la, lb, gm in zip(lp_src, lp_dst, gp_mask[::-1]):
            # Resize mask to match pyramid level
            gm_resized = cv2.resize(gm, (la.shape[1], la.shape[0]))
            gm_3d = np.stack([gm_resized] * 3, axis=-1) if len(la.shape) == 3 else gm_resized
            ls = la * gm_3d + lb * (1.0 - gm_3d)
            LS.append(ls)
        
        # Reconstruct
        ls_ = LS[0]
        for i in range(1, len(LS)):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
            ls_ = cv2.add(ls_, LS[i])
        
        return np.clip(ls_, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Pyramid blending error: {e}")
        # Fallback to simple blending
        mask_3d = np.stack([mask] * 3, axis=-1)
        return (src * mask_3d + dst * (1 - mask_3d)).astype(np.uint8)


def enhance_face_quality(face: Frame) -> Frame:
    """Enhance face quality with sharpening and noise reduction"""
    try:
        # Ensure input is valid
        if face is None or face.size == 0:
            return face
            
        # Convert to float for processing
        face_float = face.astype(np.float32)
        
        # Mild sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.3
        
        # Apply sharpening
        sharpened = cv2.filter2D(face_float, -1, kernel)
        
        # Blend with original
        alpha = 0.3
        enhanced_face = cv2.addWeighted(face_float, 1 - alpha, sharpened, alpha, 0)
        
        # Mild bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(enhanced_face.astype(np.uint8), 3, 25, 25)
        
        return denoised
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return face


def advanced_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """
    Advanced blending with edge-aware processing and texture preservation
    """
    if target_face is None:
        return swapped_face
    
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        
        # Ensure coordinates are within frame bounds
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Create improved mask
        mask = create_enhanced_mask(target_face, target_frame.shape)
        
        # Extract regions
        result = target_frame.copy()
        face_region = result[y1:y2, x1:x2]
        
        # Resize swapped face to match target region
        if face_region.shape[:2] != swapped_face.shape[:2]:
            swapped_face_resized = cv2.resize(swapped_face, (face_region.shape[1], face_region.shape[0]))
        else:
            swapped_face_resized = swapped_face
        
        # Extract mask region
        mask_region = mask[y1:y2, x1:x2]
        
        # Apply pyramid blending
        blended_face = pyramid_blending(swapped_face_resized, face_region, mask_region)
        
        # Apply the blended result
        result[y1:y2, x1:x2] = blended_face
        
        return result
        
    except Exception as e:
        print(f"Advanced blending error: {e}")
        # Fallback to simple replacement
        return target_frame


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face swapping with improved blending and color correction"""
    try:
        # Get basic face swap
        swapped_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        
        # Apply advanced color correction
        swapped_frame = adaptive_color_correction(source_face, target_face, swapped_frame, temp_frame)
        
        # Enhance face quality
        swapped_frame = enhance_face_quality(swapped_frame)
        
        # Apply advanced blending
        result_frame = advanced_face_blending(swapped_frame, temp_frame, target_face)
        
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
