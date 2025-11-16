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

def adaptive_color_correction(source_face: Face, target_face: Face, swapped_face: Frame, target_frame: Frame) -> Frame:
    """
    Advanced color correction with configurable strength
    """
    config = get_config()
    
    if config.color_correction_strength <= 0:
        return swapped_face
        
    try:
        source_region = extract_face_region(source_face, cv2.imread(roop.globals.source_path))
        target_region = extract_face_region(target_face, target_frame)
        
        if source_region is not None and target_region is not None and config.histogram_matching:
            source_hist = calculate_adaptive_histogram(source_region)
            target_hist = calculate_adaptive_histogram(target_region)
            
            for channel in range(3):
                swapped_face[:, :, channel] = histogram_matching(
                    swapped_face[:, :, channel], 
                    source_hist[channel], 
                    target_hist[channel]
                )
        
        if config.adaptive_lighting:
            swapped_face = automatic_color_balance(swapped_face, target_frame, target_face)
        
        # Apply configurable strength
        if config.color_correction_strength < 1.0:
            original_face = get_face_swapper().get(target_frame, target_face, source_face, paste_back=False)
            swapped_face = cv2.addWeighted(
                swapped_face, 
                config.color_correction_strength, 
                original_face, 
                1 - config.color_correction_strength, 
                0
            )
        
    except Exception as e:
        if config.debug_mode:
            print(f"Color correction error: {e}")
    
    return swapped_face

def extract_face_region(face: Face, frame: Frame) -> Optional[Frame]:
    """Extract the face region from frame"""
    if face is None:
        return None
    
    x1, y1, x2, y2 = map(int, face.bbox)
    margin = 10
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    return frame[y1:y2, x1:x2]

def calculate_adaptive_histogram(region: Frame) -> List[np.ndarray]:
    """Calculate adaptive histogram for color matching"""
    if region is None or region.size == 0:
        return [np.zeros(256) for _ in range(3)]
    
    hists = []
    for channel in range(3):
        hist = cv2.calcHist([region], [channel], None, [256], [0, 256])
        hist = cv2.GaussianBlur(hist, (5, 5), 0)
        hists.append(hist.flatten())
    
    return hists

def histogram_matching(source_channel: np.ndarray, source_hist: np.ndarray, target_hist: np.ndarray) -> np.ndarray:
    """Apply histogram matching between source and target"""
    source_cdf = source_hist.cumsum()
    source_cdf = source_cdf / source_cdf[-1]
    
    target_cdf = target_hist.cumsum()
    target_cdf = target_cdf / target_cdf[-1]
    
    mapping = np.interp(source_cdf, target_cdf, np.arange(256))
    matched_channel = np.interp(source_channel.flatten(), np.arange(256), mapping)
    return matched_channel.reshape(source_channel.shape).astype(np.uint8)

def automatic_color_balance(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Automatic color balance based on surrounding environment"""
    config = get_config()
    
    if target_face is None or config.color_balance_strength <= 0:
        return swapped_face
    
    x1, y1, x2, y2 = map(int, target_face.bbox)
    margin = 20
    h, w = target_frame.shape[:2]
    
    surround_x1 = max(0, x1 - margin)
    surround_y1 = max(0, y1 - margin)
    surround_x2 = min(w, x2 + margin)
    surround_y2 = min(h, y2 + margin)
    
    surround_region = target_frame[surround_y1:surround_y2, surround_x1:surround_x2]
    
    if surround_region.size > 0:
        surround_mean = np.mean(surround_region, axis=(0, 1))
        face_mean = np.mean(swapped_face, axis=(0, 1))
        
        color_ratio = surround_mean / (face_mean + 1e-6)
        color_ratio = np.clip(color_ratio, 0.7, 1.3)
        
        # Apply configurable strength
        adjusted_ratio = 1 + (color_ratio - 1) * config.color_balance_strength
        swapped_face = (swapped_face * adjusted_ratio).astype(np.uint8)
    
    return swapped_face

def advanced_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """
    Advanced blending with configurable parameters
    """
    config = get_config()
    
    if target_face is None:
        return swapped_face
    
    x1, y1, x2, y2 = map(int, target_face.bbox)
    mask = create_enhanced_mask(target_face, target_frame.shape, config)
    
    result = target_frame.copy()
    face_region = result[y1:y2, x1:x2]
    
    if face_region.shape != swapped_face.shape:
        swapped_face = cv2.resize(swapped_face, (face_region.shape[1], face_region.shape[0]))
    
    if config.blend_levels > 1:
        blended_face = pyramid_blending(swapped_face, face_region, mask[y1:y2, x1:x2], config)
    else:
        # Simple blending for performance
        mask_region = mask[y1:y2, x1:x2]
        mask_3d = np.stack([mask_region] * 3, axis=-1)
        blended_face = (swapped_face * mask_3d + face_region * (1 - mask_3d)).astype(np.uint8)
    
    result[y1:y2, x1:x2] = blended_face
    return result

def create_enhanced_mask(face: Face, frame_shape: Tuple[int, int], config) -> np.ndarray:
    """Create enhanced mask with configurable parameters"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    
    if hasattr(face, 'kps') and face.kps is not None:
        landmarks = face.kps.astype(np.int32)
        hull = cv2.convexHull(landmarks)
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        mask = cv2.GaussianBlur(mask, (config.edge_feather, config.edge_feather), 0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.gaussian_kernel, config.gaussian_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        mask = cv2.GaussianBlur(mask, (config.mask_smoothness, config.mask_smoothness), 0)
    else:
        x1, y1, x2, y2 = map(int, face.bbox)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = min(x2 - x1, y2 - y1) // 2
        
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
    
    return np.clip(mask, 0, 1)

def pyramid_blending(src: Frame, dst: Frame, mask: np.ndarray, config) -> Frame:
    """Multi-resolution pyramid blending with configurable levels"""
    levels = config.blend_levels
    
    # Generate Gaussian pyramid
    G = src.astype(np.float32)
    gpA = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gpA.append(G)
    
    G = dst.astype(np.float32)
    gpB = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gpB.append(G)
    
    # Generate Laplacian pyramid
    lpA = [gpA[levels-1]]
    for i in range(levels-1, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)
    
    lpB = [gpB[levels-1]]
    for i in range(levels-1, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)
    
    # Blend pyramids
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        mask_resized = cv2.resize(mask, (cols, rows))
        mask_3d = np.stack([mask_resized] * 3, axis=-1)
        ls = la * mask_3d + lb * (1.0 - mask_3d)
        LS.append(ls)
    
    # Reconstruct
    ls_ = LS[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    
    return np.clip(ls_, 0, 255).astype(np.uint8)

def enhance_face_quality(face: Frame) -> Frame:
    """Enhance face quality with configurable parameters"""
    config = get_config()
    
    if not config.quality_enhance:
        return face
    
    # Sharpening
    if config.sharpness_enhance > 1.0:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.3
        sharpened = cv2.filter2D(face, -1, kernel)
        face = cv2.addWeighted(face, 1 - 0.3, sharpened, 0.3, 0)
    
    # Denoising
    if config.denoise_strength > 0:
        face = cv2.bilateralFilter(face, config.bilateral_filter_d, 
                                 config.bilateral_filter_sigma, 
                                 config.bilateral_filter_sigma)
    
    return face

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face swapping with configurable parameters"""
    config = get_config()
    
    # Get basic face swap
    swapped_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
    
    # Apply enhancements based on config
    if config.color_correction_strength > 0:
        swapped_frame = adaptive_color_correction(source_face, target_face, swapped_frame, temp_frame)
    
    if config.quality_enhance:
        swapped_frame = enhance_face_quality(swapped_frame)
    
    # Apply blending
    result_frame = advanced_face_blending(swapped_frame, temp_frame, target_face)
    
    return result_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process single frame with enhanced face swapping"""
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

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames with enhanced face swapping"""
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image with enhanced face swapping"""
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video with enhanced face swapping"""
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
