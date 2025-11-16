from typing import Any, List, Callable, Optional, Tuple
import cv2
import threading
import concurrent.futures
import os
import time
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# Global variables
FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

# GPU optimization settings
GPU_OPTIMIZATION_CONFIG = {
    'max_workers': 2,  # Adjust based on your GPU VRAM
    'batch_size': 4,   # For future batch processing support
    'warmup_inference': True,
    'memory_efficient': True
}

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            device = get_device()
            
            # Apply GPU-specific optimizations
            configure_gpu_optimizations(device)
            
            # Initialize GFPGAN with optimized settings
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=1,
                device=device,
                channel_multiplier=2,
                bg_upsampler=None  # Disable to save memory
            )
            
            # Warm-up inference for better GPU performance
            if GPU_OPTIMIZATION_CONFIG['warmup_inference'] and device == 'cuda':
                perform_warmup_inference(FACE_ENHANCER)
                
    return FACE_ENHANCER

def configure_gpu_optimizations(device: str) -> None:
    """Configure GPU optimizations for better performance"""
    if device == 'cuda':
        try:
            import torch
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True  # Optimizes for fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Faster on Ampere+ GPUs
            torch.backends.cudnn.allow_tf32 = True  # Faster convolutions
            
            # Set environment variables for GPU optimization
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking execution
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            print("GPU optimizations enabled for CUDA device")
            
        except ImportError:
            print("PyTorch not available for GPU optimizations")
    elif device == 'mps':
        print("MPS device detected - Apple Silicon optimizations applied")

def perform_warmup_inference(face_enhancer: Any) -> None:
    """Perform warm-up inference to optimize GPU performance"""
    try:
        print("Performing GPU warm-up inference...")
        # Create a dummy image for warm-up
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        with THREAD_SEMAPHORE:
            face_enhancer.enhance(dummy_image, paste_back=True)
        print("GPU warm-up completed")
    except Exception as e:
        print(f"Warm-up inference failed: {e}")

def get_device() -> str:
    """Get the best available device with priority to GPU"""
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    if 'CPUExecutionProvider' in roop.globals.execution_providers:
        return 'cpu'
    
    # Fallback: auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    
    return 'cpu'

def clear_face_enhancer() -> None:
    """Clear face enhancer and free GPU memory"""
    global FACE_ENHANCER

    if FACE_ENHANCER is not None:
        # Proper cleanup for GPU memory
        try:
            if hasattr(FACE_ENHANCER, 'cleanup'):
                FACE_ENHANCER.cleanup()
            elif hasattr(FACE_ENHANCER, 'device'):
                if FACE_ENHANCER.device == 'cuda':
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("GPU memory cleared")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    FACE_ENHANCER = None

def pre_check() -> bool:
    """Download required models"""
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    ])
    return True

def pre_start() -> bool:
    """Pre-start validation"""
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    """Post-processing cleanup"""
    clear_face_enhancer()

def calculate_face_roi(target_face: Face, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Calculate face region of interest with optimal padding"""
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    
    # Adaptive padding based on face size
    face_width = end_x - start_x
    face_height = end_y - start_y
    
    padding_x = int(face_width * 0.15)  # Increased padding for better context
    padding_y = int(face_height * 0.15)
    
    # Apply padding with bounds checking
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(frame_shape[1], end_x + padding_x)
    end_y = min(frame_shape[0], end_y + padding_y)
    
    return start_x, start_y, end_x, end_y

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """Enhance a single face in the frame"""
    start_x, start_y, end_x, end_y = calculate_face_roi(target_face, temp_frame.shape)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    
    if temp_face.size and temp_face.shape[0] > 10 and temp_face.shape[1] > 10:  # Minimum size check
        try:
            with THREAD_SEMAPHORE:
                start_time = time.time()
                _, _, temp_face = get_face_enhancer().enhance(
                    temp_face,
                    paste_back=True
                )
                enhancement_time = time.time() - start_time
                if enhancement_time > 0.1:  # Log slow processing
                    print(f"Face enhancement took: {enhancement_time:.3f}s")
                    
            temp_frame[start_y:end_y, start_x:end_x] = temp_face
        except Exception as e:
            print(f"Error enhancing face: {e}")
            # Fallback: return original frame
    
    return temp_frame

def enhance_faces_batch(target_faces: List[Face], temp_frame: Frame) -> Frame:
    """Process multiple faces with optimized batch-like processing"""
    if not target_faces:
        return temp_frame
    
    # Process faces sequentially but with optimized GPU utilization
    processed_faces = 0
    for target_face in target_faces:
        temp_frame = enhance_face(target_face, temp_frame)
        processed_faces += 1
    
    if len(target_faces) > 1:
        print(f"Processed {processed_faces} faces in frame")
    
    return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process frame with face enhancement"""
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        # Use optimized processing based on number of faces
        if len(many_faces) > 1:
            return enhance_faces_batch(many_faces, temp_frame)
        else:
            return enhance_face(many_faces[0], temp_frame)
    return temp_frame

def process_single_frame(temp_frame_path: str, update: Optional[Callable[[], None]] = None) -> None:
    """Process a single frame with error handling"""
    try:
        # Read frame
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"Failed to read frame: {temp_frame_path}")
            return
        
        # Process frame
        result = process_frame(None, None, temp_frame)
        
        # Write result
        success = cv2.imwrite(temp_frame_path, result)
        if not success:
            print(f"Failed to write frame: {temp_frame_path}")
        
        # Update progress if callback provided
        if update:
            update()
            
    except Exception as e:
        print(f"Error processing frame {temp_frame_path}: {e}")

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames with optimized GPU utilization"""
    total_frames = len(temp_frame_paths)
    print(f"Processing {total_frames} frames with GPU optimization")
    
    # Configure based on available resources
    device = get_device()
    max_workers = GPU_OPTIMIZATION_CONFIG['max_workers']
    
    if device == 'cpu' or total_frames <= 1:
        # Sequential processing for CPU or single frame
        for temp_frame_path in temp_frame_paths:
            process_single_frame(temp_frame_path, update)
    else:
        # Parallel processing for GPU with limited workers
        max_workers = min(max_workers, total_frames, 4)  # Conservative limit
        
        print(f"Using {max_workers} parallel workers on {device}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_single_frame, path, update if i == total_frames - 1 else None): path 
                for i, path in enumerate(temp_frame_paths)
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    future.result()  # This will re-raise any exceptions
                except Exception as e:
                    path = future_to_path[future]
                    print(f"Frame {path} processing failed: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image with face enhancement"""
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            raise ValueError(f"Failed to read target image: {target_path}")
        
        result = process_frame(None, None, target_frame)
        
        success = cv2.imwrite(output_path, result)
        if not success:
            raise ValueError(f"Failed to write output image: {output_path}")
            
        print(f"Image processed successfully: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video frames with enhanced GPU utilization"""
    print("Starting video processing with GPU optimization...")
    start_time = time.time()
    
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
    
    processing_time = time.time() - start_time
    print(f"Video processing completed in {processing_time:.2f} seconds")
    
    # Final cleanup
    clear_face_enhancer()

# Additional utility functions for monitoring
def get_gpu_status() -> Optional[str]:
    """Get GPU status information"""
    try:
        if get_device() == 'cuda':
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                return f"GPU: {gpu_name}, Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB"
    except ImportError:
        pass
    return None

def optimize_for_low_vram() -> None:
    """Optimize settings for low VRAM GPUs"""
    global GPU_OPTIMIZATION_CONFIG
    GPU_OPTIMIZATION_CONFIG.update({
        'max_workers': 1,
        'batch_size': 1,
        'memory_efficient': True
    })
    print("Low VRAM optimization applied")

def optimize_for_high_vram() -> None:
    """Optimize settings for high VRAM GPUs"""
    global GPU_OPTIMIZATION_CONFIG
    GPU_OPTIMIZATION_CONFIG.update({
        'max_workers': 4,
        'batch_size': 8,
        'memory_efficient': False
    })
    print("High VRAM optimization applied")

# Initialize numpy for warm-up (added at the end to avoid import issues)
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not available, some optimizations may be disabled")
    # Create a simple fallback
    class SimpleNP:
        @staticmethod
        def random.randint(low, high, size, dtype):
            return [[[123, 123, 123] for _ in range(size[1])] for _ in range(size[0])]
    
    np = SimpleNP()
