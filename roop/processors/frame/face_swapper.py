from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
from scipy import ndimage
import os
import gc

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

# Try import onnxruntime dengan error handling
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    print("ONNX Runtime not available, using default providers")
    ONNXRUNTIME_AVAILABLE = False
    ort = None

# Try import torch dengan error handling  
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available, skipping torch optimizations")
    TORCH_AVAILABLE = False
    torch = None


def clear_gpu_memory():
    """Clear GPU memory secara eksplisit"""
    try:
        # Clear CUDA memory jika torch tersedia
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear general garbage collection
        gc.collect()
            
    except Exception as e:
        print(f"GPU memory clear warning: {e}")


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            
            # Cek jika file model ada
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Optimasi provider settings untuk GPU dengan error handling
            providers = []
            
            if ONNXRUNTIME_AVAILABLE:
                try:
                    available_providers = ort.get_available_providers()
                    
                    if 'CUDAExecutionProvider' in available_providers:
                        providers = [
                            ('CUDAExecutionProvider', {
                                'device_id': 0,
                                'arena_extend_strategy': 'kSameAsRequested',
                                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit
                                'cudnn_conv_algo_search': 'HEURISTIC',
                            })
                        ]
                        print("Using CUDAExecutionProvider for GPU acceleration")
                    else:
                        providers = ['CPUExecutionProvider']
                        print("Using CPUExecutionProvider (CUDA not available)")
                        
                except Exception as e:
                    print(f"Error configuring providers: {e}, using default")
                    providers = roop.globals.execution_providers
            else:
                # Fallback ke providers dari globals
                providers = roop.globals.execution_providers
                print(f"Using providers from globals: {providers}")
                
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
                print("Face swapper model loaded successfully")
            except Exception as e:
                print(f"Error loading face swapper model: {e}")
                # Fallback ke CPU saja
                try:
                    providers = ['CPUExecutionProvider']
                    FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
                    print("Face swapper model loaded with CPU fallback")
                except Exception as fallback_error:
                    print(f"Critical error loading model: {fallback_error}")
                    raise fallback_error
                    
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None
    clear_gpu_memory()


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
    clear_gpu_memory()


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


def optimized_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Color correction yang lebih efisien untuk GPU"""
    try:
        if target_face is None:
            return swapped_face
        
        # Extract target face region untuk color reference
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Cek jika region valid
        if x2 <= x1 or y2 <= y1:
            return swapped_face
            
        target_region = target_frame[y1:y2, x1:x2]
        
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        
        # Resize swapped face untuk match target region jika diperlukan
        if swapped_face.shape[:2] != target_region.shape[:2]:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        
        # Simple color matching yang robust
        swapped_face_float = swapped_face.astype(np.float32)
        target_region_float = target_region.astype(np.float32)
        
        # Match color statistics per channel
        for channel in range(3):
            swapped_mean = np.mean(swapped_face_float[:,:,channel])
            target_mean = np.mean(target_region_float[:,:,channel])
            swapped_std = np.std(swapped_face_float[:,:,channel])
            target_std = np.std(target_region_float[:,:,channel])
            
            # Avoid division by zero
            if swapped_std > 1.0 and target_std > 1.0:
                swapped_face_float[:,:,channel] = (
                    (swapped_face_float[:,:,channel] - swapped_mean) * 
                    (target_std / swapped_std) + target_mean
                )
            else:
                # Simple mean matching
                swapped_face_float[:,:,channel] += (target_mean - swapped_mean) * 0.3
        
        result_face = np.clip(swapped_face_float, 0, 255).astype(np.uint8)
        
        # Mild blending dengan original
        blend_ratio = 0.3
        result_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, result_face, blend_ratio, 0)
        
        return result_face
        
    except Exception as e:
        print(f"Optimized color correction error: {e}")
        return swapped_face


def create_optimized_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Create optimized mask untuk blending"""
    try:
        mask = np.zeros(frame_shape[:2], dtype=np.float32)
        
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Pastikan coordinates dalam bounds
        h, w = frame_shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return mask
            
        # Create elliptical mask
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Create ellipse dengan parameter yang optimal
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 1.0, -1)
        
        # Apply Gaussian blur untuk smooth edges
        kernel_size = max(15, min(width, height) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 51)  # Batasi maximum size
        
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return np.clip(mask, 0, 1)
        
    except Exception as e:
        print(f"Optimized mask creation error: {e}")
        # Fallback ke simple rectangular mask
        try:
            mask = np.zeros(frame_shape[:2], dtype=np.float32)
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_shape[1], x2), min(frame_shape[0], y2)
            mask[y1:y2, x1:x2] = 1.0
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            return mask
        except:
            return np.zeros(frame_shape[:2], dtype=np.float32)


def enhance_face_quality(face: Frame) -> Frame:
    """Simple face quality enhancement yang dioptimalkan"""
    try:
        if face is None:
            return face
            
        # Ensure it's a numpy array
        face_array = ensure_frame_format(face)
        if face_array is None:
            return face
            
        # Mild sharpening
        kernel = np.array([[0, -0.25, 0],
                          [-0.25,  2, -0.25],
                          [0, -0.25, 0]])
        
        sharpened = cv2.filter2D(face_array, -1, kernel)
        
        # Mild bilateral filter untuk noise reduction
        denoised = cv2.bilateralFilter(sharpened, 3, 15, 15)
        
        return denoised
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return face


def optimized_seamless_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Optimized seamless blending dengan fallback"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return target_frame
        
        # Ensure swapped face has correct size
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Coba seamless clone terlebih dahulu
        try:
            mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
            return result
        except Exception as seamless_error:
            print(f"Seamless clone failed, using alpha blending: {seamless_error}")
            return optimized_alpha_blending(swapped_face, target_frame, target_face)
        
    except Exception as e:
        print(f"Optimized seamless blending error: {e}")
        return optimized_alpha_blending(swapped_face, target_frame, target_face)


def optimized_alpha_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Optimized alpha blending dengan memory efficiency"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return target_frame
        
        # Ensure swapped face has correct size
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Create optimized mask
        mask = create_optimized_mask(target_face, target_frame.shape)
        mask_region = mask[y1:y2, x1:x2]
        
        # Ensure mask has correct dimensions
        if mask_region.shape != swapped_face.shape[:2]:
            mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
        
        # Optimized blending dengan operasi numpy
        result = target_frame.copy()
        face_region = result[y1:y2, x1:x2]
        
        # Create 3-channel mask
        mask_3d = np.stack([mask_region] * 3, axis=-1)
        
        # Blend operation
        blended_face = (swapped_face.astype(np.float32) * mask_3d + 
                       face_region.astype(np.float32) * (1 - mask_3d))
        
        result[y1:y2, x1:x2] = np.clip(blended_face, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"Optimized alpha blending error: {e}")
        return target_frame


def swap_face_optimized(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping dengan GPU efficiency"""
    try:
        # Get face swapper instance
        face_swapper = get_face_swapper()
        
        # Get basic face swap
        swapped_result = face_swapper.get(temp_frame, target_face, source_face, paste_back=False)
        
        # Ensure proper format
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            # Fallback ke original method
            return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply optimized color correction
        swapped_frame = optimized_color_correction(swapped_frame, temp_frame, target_face)
        
        # Enhance face quality
        swapped_frame = enhance_face_quality(swapped_frame)
        
        # Apply optimized blending
        result_frame = optimized_seamless_blending(swapped_frame, temp_frame, target_face)
        
        return result_frame
        
    except Exception as e:
        print(f"Optimized face swap error: {e}")
        # Fallback ke original face swapper
        try:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        except Exception as fallback_error:
            print(f"Critical face swap error: {fallback_error}")
            return temp_frame


def process_frame_batch(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process frame dengan batch optimization untuk multiple faces"""
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                # Process semua wajah
                for target_face in many_faces:
                    temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame batch error: {e}")
        return temp_frame


def process_frames_optimized(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames dengan memory management yang lebih baik"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        if source_face is None:
            print("No source face detected")
            return
            
        reference_face = None if roop.globals.many_faces else get_face_reference()
        
        total_frames = len(temp_frame_paths)
        print(f"Processing {total_frames} frames with optimized GPU pipeline")
        
        # Dynamic batch size
        if total_frames > 100:
            batch_size = 10
            memory_clear_interval = 25
        elif total_frames > 50:
            batch_size = 8
            memory_clear_interval = 20
        else:
            batch_size = 5
            memory_clear_interval = 10
        
        processed_count = 0
        
        for i in range(0, total_frames, batch_size):
            batch_paths = temp_frame_paths[i:i + batch_size]
            
            try:
                for temp_frame_path in batch_paths:
                    if not os.path.exists(temp_frame_path):
                        print(f"Frame not found: {temp_frame_path}")
                        continue
                        
                    temp_frame = cv2.imread(temp_frame_path)
                    if temp_frame is not None:
                        result = process_frame_batch(source_face, reference_face, temp_frame)
                        cv2.imwrite(temp_frame_path, result)
                        processed_count += 1
                    
                    if update:
                        update()
                
                # Clear memory secara berkala
                if processed_count % memory_clear_interval == 0:
                    clear_gpu_memory()
                    print(f"Processed {processed_count}/{total_frames} frames - Memory cleared")
                    
            except Exception as e:
                print(f"Error processing batch starting at {i}: {e}")
                continue
                
        print(f"Completed processing {processed_count}/{total_frames} frames")
                
    except Exception as e:
        print(f"Process frames optimized error: {e}")
    finally:
        clear_gpu_memory()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image dengan optimized face swapping"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        if source_face is None:
            print("No face found in source image")
            return
            
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            print("Cannot read target image")
            return
            
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = process_frame_batch(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
        print(f"Image processed and saved to: {output_path}")
    except Exception as e:
        print(f"Process image error: {e}")


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video dengan optimasi GPU"""
    try:
        if not roop.globals.many_faces and not get_face_reference():
            if temp_frame_paths and roop.globals.reference_frame_number < len(temp_frame_paths):
                reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
                if reference_frame is not None:
                    reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
                    if reference_face is not None:
                        set_face_reference(reference_face)
                    else:
                        print("No reference face found in reference frame")
                else:
                    print("Cannot read reference frame")
        
        # Gunakan fungsi optimized
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames_optimized)
    except Exception as e:
        print(f"Process video error: {e}")
    finally:
        clear_gpu_memory()


# Backward compatibility functions
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Alias untuk compatibility"""
    return process_frame_batch(source_face, reference_face, temp_frame)


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Alias untuk compatibility"""
    process_frames_optimized(source_path, temp_frame_paths, update)
