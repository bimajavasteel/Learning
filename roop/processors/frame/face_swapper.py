from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
from scipy import ndimage
import os
import torch

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

# GPU optimization settings
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'
TORCH_DTYPE = torch.float16 if CUDA_AVAILABLE else torch.float32

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            
            # Optimized providers for GPU T4
            if CUDA_AVAILABLE:
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                providers = roop.globals.execution_providers
            
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
            
            # Warm up the model
            if FACE_SWAPPER is not None:
                dummy_input = np.random.rand(128, 128, 3).astype(np.float32)
                try:
                    FACE_SWAPPER.get(dummy_input, dummy_input)
                except:
                    pass
                    
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
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

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

def optimized_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Advanced color correction similar to FaceFusion"""
    try:
        if target_face is None:
            return swapped_face
        
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        target_region = target_frame[y1:y2, x1:x2]
        
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        
        # Resize swapped face to match target region
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        
        # Convert to different color spaces for better matching
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        
        # Calculate statistics with GPU acceleration if available
        swapped_mean = np.mean(swapped_lab, axis=(0,1))
        swapped_std = np.std(swapped_lab, axis=(0,1))
        target_mean = np.mean(target_lab, axis=(0,1))
        target_std = np.std(target_lab, axis=(0,1))
        
        # Avoid division by zero
        swapped_std = np.maximum(swapped_std, 1.0)
        target_std = np.maximum(target_std, 1.0)
        
        # Advanced color matching with smooth transition
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Multi-level blending for natural look
        blend_ratio = 0.8
        result_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, corrected_face, blend_ratio, 0)
        
        # Final color balance adjustment
        result_ycrcb = cv2.cvtColor(result_face, cv2.COLOR_BGR2YCrCb)
        result_ycrcb[:, :, 0] = cv2.equalizeHist(result_ycrcb[:, :, 0])
        result_face = cv2.cvtColor(result_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return result_face
        
    except Exception as e:
        print(f"Optimized color correction error: {e}")
        return swapped_face

def create_advanced_mask(face: Face, frame_shape: Tuple[int, int], feather_amount: int = 25) -> np.ndarray:
    """Create advanced mask with better edge handling"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Create more accurate face mask using landmarks if available
        if hasattr(face, 'kps'):
            landmarks = face.kps.astype(np.int32)
            
            # Create convex hull from landmarks
            hull = cv2.convexHull(landmarks)
            cv2.fillConvexPoly(mask, hull, 1.0)
        else:
            # Fallback to elliptical mask
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            
            cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 1.0, -1)
        
        # Advanced feathering with multiple blur passes
        mask = cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0)
        mask = cv2.GaussianBlur(mask, (feather_amount//2, feather_amount//2), 0)
        
        # Enhance mask edges
        mask = np.clip(mask * 1.2, 0, 1)
        
        return mask
        
    except Exception as e:
        print(f"Advanced mask creation error: {e}")
        # Fallback
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

def enhance_face_quality_gpu(face: Frame) -> Frame:
    """GPU-accelerated face quality enhancement"""
    try:
        if face is None:
            return face
            
        face_array = ensure_frame_format(face)
        if face_array is None:
            return face
        
        # Convert to tensor for GPU processing if available
        if CUDA_AVAILABLE:
            face_tensor = torch.from_numpy(face_array).float().to(DEVICE).permute(2, 0, 1) / 255.0
            
            # Mild sharpening with GPU
            kernel = torch.tensor([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]], dtype=TORCH_DTYPE, device=DEVICE).view(1, 1, 3, 3) * 0.2
            
            sharpened = torch.nn.functional.conv2d(
                face_tensor.unsqueeze(0), 
                kernel.repeat(3, 1, 1, 1), 
                padding=1, 
                groups=3
            ).squeeze(0)
            
            # Mild bilateral filter approximation
            denoised = torch.clamp(sharpened * 0.8 + face_tensor * 0.2, 0, 1)
            
            # Convert back to numpy
            result = (denoised.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            # CPU fallback
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * 0.2
            
            sharpened = cv2.filter2D(face_array, -1, kernel)
            denoised = cv2.bilateralFilter(sharpened, 5, 25, 25)
            result = denoised
        
        return result
        
    except Exception as e:
        print(f"GPU face enhancement error: {e}")
        return face

def advanced_seamless_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Advanced blending with multiple techniques"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Try multiple blending methods
        try:
            # Method 1: Advanced seamless clone
            mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
            
            # Method 2: Additional alpha blending for smoothness
            advanced_mask = create_advanced_mask(target_face, target_frame.shape)
            mask_region = advanced_mask[y1:y2, x1:x2]
            
            if mask_region.shape != swapped_face.shape[:2]:
                mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
            
            mask_3d = np.stack([mask_region] * 3, axis=-1)
            
            # Final blend with original for natural look
            final_blend = (result[y1:y2, x1:x2] * (1 - mask_3d) + swapped_face * mask_3d).astype(np.uint8)
            result[y1:y2, x1:x2] = final_blend
            
            return result
            
        except Exception as e:
            print(f"Advanced blending failed, using fallback: {e}")
            return simple_blending(swapped_face, target_frame, target_face)
        
    except Exception as e:
        print(f"Advanced blending error: {e}")
        return simple_blending(swapped_face, target_frame, target_face)

def optimized_swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping with GPU acceleration"""
    try:
        # Batch processing optimization
        start_time = cv2.getTickCount()
        
        # Get basic face swap
        swapped_result = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        
        # Ensure proper format
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply optimized color correction
        swapped_frame = optimized_color_correction(swapped_frame, temp_frame, target_face)
        
        # GPU-accelerated face enhancement
        swapped_frame = enhance_face_quality_gpu(swapped_frame)
        
        # Advanced blending
        result_frame = advanced_seamless_blending(swapped_frame, temp_frame, target_face)
        
        # Performance monitoring
        if roop.globals.log_level == 'debug':
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            print(f"Face swap FPS: {fps:.2f}")
        
        return result_frame
        
    except Exception as e:
        print(f"Optimized face swap error: {e}")
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Optimized frame processing"""
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = optimized_swap_face(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = optimized_swap_face(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Optimized batch frame processing"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        
        # Batch processing optimization
        batch_size = 4 if CUDA_AVAILABLE else 1
        
        for i in range(0, len(temp_frame_paths), batch_size):
            batch_paths = temp_frame_paths[i:i + batch_size]
            
            for temp_frame_path in batch_paths:
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
            
            # Clear GPU cache between batches
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Process frames error: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Optimized image processing"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Optimized video processing"""
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Process video error: {e}")

# Additional optimization function
def optimize_memory_usage():
    """Optimize memory usage for GPU"""
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
