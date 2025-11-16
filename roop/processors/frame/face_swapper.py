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

# Import configuration
from config import get_config, set_config

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

def safe_face_extraction(face: Face, frame: Frame, margin: int = 10) -> Optional[Frame]:
    """Safely extract face region with error handling"""
    if face is None or frame is None:
        return None
    
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        
        # Apply margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Validate coordinates
        if x2 <= x1 or y2 <= y1:
            return None
            
        region = frame[y1:y2, x1:x2]
        return region if region.size > 0 else None
        
    except Exception as e:
        if get_config().debug_mode:
            print(f"❌ Face extraction error: {e}")
        return None

def calculate_adaptive_histogram(region: Frame) -> List[np.ndarray]:
    """Calculate adaptive histogram for color matching"""
    if region is None or region.size == 0:
        return [np.zeros(256) for _ in range(3)]
    
    try:
        hists = []
        for channel in range(3):
            hist = cv2.calcHist([region], [channel], None, [256], [0, 256])
            hist = cv2.GaussianBlur(hist, (5, 5), 0)
            hists.append(hist.flatten())
        return hists
    except Exception:
        return [np.zeros(256) for _ in range(3)]

def histogram_matching(source_channel: np.ndarray, source_hist: np.ndarray, target_hist: np.ndarray) -> np.ndarray:
    """Apply histogram matching between source and target"""
    try:
        source_cdf = source_hist.cumsum()
        source_cdf = source_cdf / (source_cdf[-1] + 1e-8)
        
        target_cdf = target_hist.cumsum()
        target_cdf = target_cdf / (target_cdf[-1] + 1e-8)
        
        mapping = np.interp(source_cdf, target_cdf, np.arange(256))
        matched_channel = np.interp(source_channel.flatten(), np.arange(256), mapping)
        return matched_channel.reshape(source_channel.shape).astype(np.uint8)
    except Exception:
        return source_channel

def adaptive_color_correction(source_face: Face, target_face: Face, swapped_face: Frame, target_frame: Frame) -> Frame:
    """Advanced color correction with configurable strength"""
    config = get_config()
    
    if config.color_correction_strength <= 0 or source_face is None or target_face is None:
        return swapped_face
        
    try:
        # Load source image
        source_image = cv2.imread(roop.globals.source_path)
        if source_image is None:
            return swapped_face
            
        # Extract face regions for color analysis
        source_region = safe_face_extraction(source_face, source_image)
        target_region = safe_face_extraction(target_face, target_frame)
        
        # Apply histogram matching if enabled
        if source_region is not None and target_region is not None and config.histogram_matching:
            source_hist = calculate_adaptive_histogram(source_region)
            target_hist = calculate_adaptive_histogram(target_region)
            
            for channel in range(3):
                swapped_face[:, :, channel] = histogram_matching(
                    swapped_face[:, :, channel], 
                    source_hist[channel], 
                    target_hist[channel]
                )
        
        # Apply adaptive lighting correction
        if config.adaptive_lighting:
            swapped_face = apply_adaptive_lighting(swapped_face, target_frame, target_face)
        
        # Apply configurable strength with blending
        if config.color_correction_strength < 1.0:
            original_swapped = get_face_swapper().get(target_frame, target_face, source_face, paste_back=False)
            if original_swapped is not None:
                swapped_face = cv2.addWeighted(
                    swapped_face, 
                    config.color_correction_strength, 
                    original_swapped, 
                    1 - config.color_correction_strength, 
                    0
                )
        
        if config.debug_mode:
            print("✅ Color correction applied successfully")
            
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Color correction error: {e}")
    
    return swapped_face

def apply_adaptive_lighting(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Apply adaptive lighting based on surrounding environment"""
    config = get_config()
    
    if target_face is None or config.color_balance_strength <= 0:
        return swapped_face
    
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        margin = 25
        h, w = target_frame.shape[:2]
        
        # Define surrounding region
        surround_x1 = max(0, x1 - margin)
        surround_y1 = max(0, y1 - margin)
        surround_x2 = min(w, x2 + margin)
        surround_y2 = min(h, y2 + margin)
        
        # Validate surrounding region
        if surround_x2 <= surround_x1 or surround_y2 <= surround_y1:
            return swapped_face
            
        surround_region = target_frame[surround_y1:surround_y2, surround_x1:surround_x2]
        
        if surround_region.size > 0:
            # Calculate color statistics
            surround_mean = np.mean(surround_region, axis=(0, 1))
            face_mean = np.mean(swapped_face, axis=(0, 1))
            
            # Calculate color ratio with safety bounds
            color_ratio = surround_mean / (face_mean + 1e-8)
            color_ratio = np.clip(color_ratio, 0.5, 2.0)  # More conservative bounds
            
            # Apply configurable strength
            adjusted_ratio = 1 + (color_ratio - 1) * config.color_balance_strength
            swapped_face = np.clip(swapped_face * adjusted_ratio, 0, 255).astype(np.uint8)
            
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Adaptive lighting error: {e}")
    
    return swapped_face

def create_enhanced_mask(face: Face, frame_shape: Tuple[int, int], config) -> np.ndarray:
    """Create enhanced mask with configurable parameters"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    
    try:
        if hasattr(face, 'kps') and face.kps is not None:
            # Use facial landmarks for precise masking
            landmarks = face.kps.astype(np.int32)
            hull = cv2.convexHull(landmarks)
            cv2.fillConvexPoly(mask, hull, 1.0)
            
            # Apply smoothing
            mask = cv2.GaussianBlur(mask, (config.edge_feather, config.edge_feather), 0)
            
            # Morphological operations for better mask quality
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.gaussian_kernel, config.gaussian_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Final smoothing
            mask = cv2.GaussianBlur(mask, (config.mask_smoothness, config.mask_smoothness), 0)
        else:
            # Fallback to bbox-based mask
            x1, y1, x2, y2 = map(int, face.bbox)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = min(x2 - x1, y2 - y1) // 2
            
            cv2.circle(mask, center, radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (25, 25), 0)
            
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Mask creation error: {e}")
        # Simple fallback mask
        x1, y1, x2, y2 = map(int, face.bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = min(x2 - x1, y2 - y1) // 2
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
    
    return np.clip(mask, 0, 1)

def pyramid_blending(src: Frame, dst: Frame, mask: np.ndarray, config) -> Frame:
    """Multi-resolution pyramid blending with configurable levels"""
    try:
        levels = min(config.blend_levels, 5)  # Limit to 5 levels for performance
        
        # Generate Gaussian pyramid for source
        G_src = src.astype(np.float32)
        gp_src = [G_src]
        for i in range(levels):
            G_src = cv2.pyrDown(G_src)
            gp_src.append(G_src)
        
        # Generate Gaussian pyramid for destination
        G_dst = dst.astype(np.float32)
        gp_dst = [G_dst]
        for i in range(levels):
            G_dst = cv2.pyrDown(G_dst)
            gp_dst.append(G_dst)
        
        # Generate Laplacian pyramid for source
        lp_src = [gp_src[levels-1]]
        for i in range(levels-1, 0, -1):
            GE = cv2.pyrUp(gp_src[i])
            L = cv2.subtract(gp_src[i-1], GE)
            lp_src.append(L)
        
        # Generate Laplacian pyramid for destination
        lp_dst = [gp_dst[levels-1]]
        for i in range(levels-1, 0, -1):
            GE = cv2.pyrUp(gp_dst[i])
            L = cv2.subtract(gp_dst[i-1], GE)
            lp_dst.append(L)
        
        # Blend pyramids using mask
        blended_pyramid = []
        for la, lb in zip(lp_src, lp_dst):
            rows, cols, dpt = la.shape
            mask_resized = cv2.resize(mask, (cols, rows))
            mask_3d = np.stack([mask_resized] * 3, axis=-1)
            ls = la * mask_3d + lb * (1.0 - mask_3d)
            blended_pyramid.append(ls)
        
        # Reconstruct the final image
        result = blended_pyramid[0]
        for i in range(1, levels):
            result = cv2.pyrUp(result)
            result = cv2.add(result, blended_pyramid[i])
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Pyramid blending error: {e}")
        # Fallback to simple blending
        mask_3d = np.stack([mask] * 3, axis=-1)
        return (src * mask_3d + dst * (1.0 - mask_3d)).astype(np.uint8)

def enhance_face_quality(face: Frame) -> Frame:
    """Enhance face quality with configurable parameters"""
    config = get_config()
    
    if not config.quality_enhance or face is None:
        return face
    
    try:
        # Validate input
        if not isinstance(face, np.ndarray) or face.size == 0:
            return face
        
        enhanced_face = face.copy()
        
        # Adaptive sharpening
        if config.sharpness_enhance > 1.0:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]], dtype=np.float32) * 0.15
            
            sharpened = cv2.filter2D(enhanced_face.astype(np.float32), -1, kernel)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            # Blend with original based on strength
            alpha = min(0.3, (config.sharpness_enhance - 1.0) * 0.5)
            enhanced_face = cv2.addWeighted(enhanced_face, 1 - alpha, sharpened, alpha, 0)
        
        # Denoising
        if config.denoise_strength > 0 and enhanced_face.shape[0] > 10 and enhanced_face.shape[1] > 10:
            enhanced_face = cv2.bilateralFilter(
                enhanced_face, 
                config.bilateral_filter_d, 
                config.bilateral_filter_sigma, 
                config.bilateral_filter_sigma
            )
        
        if config.debug_mode:
            print("✅ Face quality enhancement applied")
            
        return enhanced_face
        
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Quality enhancement error: {e}")
        return face

def advanced_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Advanced blending with configurable parameters"""
    config = get_config()
    
    if target_face is None:
        return target_frame
    
    try:
        # Get face coordinates
        x1, y1, x2, y2 = map(int, target_face.bbox)
        
        # Create enhanced mask
        mask = create_enhanced_mask(target_face, target_frame.shape, config)
        
        # Prepare result frame
        result = target_frame.copy()
        face_region = result[y1:y2, x1:x2]
        
        # Resize swapped face to match target region
        if face_region.shape[:2] != swapped_face.shape[:2]:
            swapped_face = cv2.resize(swapped_face, (face_region.shape[1], face_region.shape[0]))
        
        # Apply blending based on configuration
        if config.blend_levels > 1 and face_region.shape[0] > 32 and face_region.shape[1] > 32:
            # Use pyramid blending for high quality
            mask_region = mask[y1:y2, x1:x2]
            blended_face = pyramid_blending(swapped_face, face_region, mask_region, config)
        else:
            # Use simple blending for performance
            mask_region = mask[y1:y2, x1:x2]
            if mask_region.shape[:2] == face_region.shape[:2]:
                mask_3d = np.stack([mask_region] * 3, axis=-1)
                blended_face = (swapped_face * mask_3d + face_region * (1 - mask_3d)).astype(np.uint8)
            else:
                blended_face = swapped_face  # Fallback
        
        # Apply blended result
        result[y1:y2, x1:x2] = blended_face
        
        if config.debug_mode:
            print("✅ Advanced blending applied successfully")
            
        return result
        
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Blending error: {e}")
        return target_frame

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face swapping with configurable parameters"""
    config = get_config()
    
    try:
        # Get basic face swap from model
        swapped_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        
        if swapped_frame is None:
            if config.debug_mode:
                print("❌ Face swapper returned None")
            return temp_frame
        
        # Apply color correction
        if config.color_correction_strength > 0:
            swapped_frame = adaptive_color_correction(source_face, target_face, swapped_frame, temp_frame)
        
        # Apply quality enhancement
        if config.quality_enhance:
            swapped_frame = enhance_face_quality(swapped_frame)
        
        # Apply advanced blending
        result_frame = advanced_face_blending(swapped_frame, temp_frame, target_face)
        
        if config.debug_mode:
            print("✅ Face swap completed successfully")
            
        return result_frame
        
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Face swap error: {e}")
        return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process single frame with enhanced face swapping"""
    config = get_config()
    
    try:
        if roop.globals.many_faces:
            # Process multiple faces
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
                    if config.debug_mode:
                        print(f"✅ Processed {len(many_faces)} faces")
        else:
            # Process single face
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
                if config.debug_mode:
                    print("✅ Processed single face")
        
        return temp_frame
        
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Frame processing error: {e}")
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames with enhanced face swapping"""
    config = get_config()
    
    try:
        # Load source face
        source_image = cv2.imread(source_path)
        if source_image is None:
            if config.debug_mode:
                print("❌ Cannot load source image")
            return
            
        source_face = get_one_face(source_image)
        if source_face is None:
            if config.debug_mode:
                print("❌ No face found in source image")
            return
        
        # Get reference face
        reference_face = None if roop.globals.many_faces else get_face_reference()
        
        if config.debug_mode:
            print(f"🚀 Starting to process {len(temp_frame_paths)} frames...")
        
        # Process each frame
        for i, temp_frame_path in enumerate(temp_frame_paths):
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is not None:
                result = process_frame(source_face, reference_face, temp_frame)
                cv2.imwrite(temp_frame_path, result)
                
                if config.debug_mode and i % 50 == 0:
                    print(f"📊 Processed {i}/{len(temp_frame_paths)} frames")
            
            if update:
                update()
                
        if config.debug_mode:
            print("✅ All frames processed successfully")
            
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Frames processing error: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image with enhanced face swapping"""
    config = get_config()
    
    try:
        if config.debug_mode:
            print("🖼️ Starting image processing...")
        
        # Load source face
        source_face = get_one_face(cv2.imread(source_path))
        if source_face is None:
            if config.debug_mode:
                print("❌ No face found in source image")
            return
        
        # Load target image
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            if config.debug_mode:
                print("❌ Cannot load target image")
            return
        
        # Get reference face
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        
        # Process image
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
        
        if config.debug_mode:
            print("✅ Image processing completed successfully")
            
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Image processing error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video with enhanced face swapping"""
    config = get_config()
    
    try:
        if config.debug_mode:
            print("🎥 Starting video processing...")
        
        # Set reference face if needed
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            if reference_frame is not None:
                reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
                if reference_face is not None:
                    set_face_reference(reference_face)
                    if config.debug_mode:
                        print("✅ Reference face set")
        
        # Process video frames
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
        
        if config.debug_mode:
            print("✅ Video processing completed successfully")
            
    except Exception as e:
        if config.debug_mode:
            print(f"❌ Video processing error: {e}")
