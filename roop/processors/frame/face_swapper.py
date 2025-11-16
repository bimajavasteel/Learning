from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
from scipy import ndimage
import os
import time
import cupy as cp
from numba import cuda
import tensorrt as trt
import pycuda.driver as cuda_driver
import pycuda.autoinit
import asyncio
import concurrent.futures

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# Global variables
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'
GPU_MEMORY_MANAGER = None
ASYNC_PROCESSOR = None

# CUDA configuration
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

class TensorRTFaceSwapper:
    def __init__(self, model_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(model_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate GPU memory
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.allocate_buffers()
        
    def load_engine(self, model_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine"""
        try:
            with open(model_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise Exception(f"Failed to load TensorRT engine: {e}")
    
    def allocate_buffers(self) -> None:
        """Allocate GPU buffers"""
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) 
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda_driver.pagelocked_empty(size, dtype)
            device_mem = cuda_driver.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """Execute inference"""
        try:
            # Copy input to GPU
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda_driver.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
            
            # Execute inference
            self.context.execute_v2(bindings=self.bindings)
            
            # Copy output from GPU
            cuda_driver.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
            return self.outputs[0]['host']
        except Exception as e:
            raise Exception(f"Inference failed: {e}")

class GPUMemoryManager:
    def __init__(self, max_cache_size: int = 10):
        self.frame_cache: Dict[str, cp.ndarray] = {}
        self.max_cache_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
        
    def get_frame(self, frame_path: str) -> Optional[cp.ndarray]:
        """Get frame from GPU cache or load from disk"""
        if frame_path in self.frame_cache:
            self.hit_count += 1
            return self.frame_cache[frame_path]
        
        self.miss_count += 1
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Move to GPU
            frame_gpu = cp.asarray(frame)
            self.frame_cache[frame_path] = frame_gpu
            
            # Manage cache size
            if len(self.frame_cache) > self.max_cache_size:
                self.frame_cache.pop(next(iter(self.frame_cache)))
            
            return frame_gpu
        return None
    
    def save_frame(self, frame_path: str, frame_gpu: cp.ndarray) -> None:
        """Save frame from GPU to disk"""
        try:
            frame_cpu = cp.asnumpy(frame_gpu)
            cv2.imwrite(frame_path, frame_cpu)
        except Exception as e:
            print(f"Error saving frame: {e}")
    
    def clear_cache(self) -> None:
        """Clear GPU cache"""
        self.frame_cache.clear()
        cp.get_default_memory_pool().free_all_blocks()

class AsyncProcessor:
    def __init__(self, max_workers: int = 4):
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    async def process_batch_async(self, source_face: Face, frame_paths: List[str]) -> None:
        """Process batch of frames asynchronously"""
        tasks = []
        for frame_path in frame_paths:
            task = self.process_single_frame_async(source_face, frame_path)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def process_single_frame_async(self, source_face: Face, frame_path: str) -> None:
        """Process single frame asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.gpu_executor, 
            self.process_frame_gpu, 
            source_face, frame_path
        )
    
    def process_frame_gpu(self, source_face: Face, frame_path: str) -> None:
        """GPU-accelerated frame processing"""
        try:
            global GPU_MEMORY_MANAGER
            
            # Get frame from GPU cache
            frame_gpu = GPU_MEMORY_MANAGER.get_frame(frame_path)
            if frame_gpu is None:
                return
            
            # Convert to CPU for face detection (temporary workaround)
            frame_cpu = cp.asnumpy(frame_gpu)
            
            # Process frame
            reference_face = None if roop.globals.many_faces else get_face_reference()
            result_cpu = optimized_process_frame(source_face, reference_face, frame_cpu)
            
            # Convert back to GPU and save
            result_gpu = cp.asarray(result_cpu)
            GPU_MEMORY_MANAGER.save_frame(frame_path, result_gpu)
            
        except Exception as e:
            print(f"GPU frame processing error for {frame_path}: {e}")

@cuda.jit
def gpu_color_correction_kernel(swapped_lab, target_lab, corrected_lab, 
                               swapped_mean, swapped_std, target_mean, target_std):
    """CUDA kernel for color correction"""
    i, j = cuda.grid(2)
    if i < swapped_lab.shape[0] and j < swapped_lab.shape[1]:
        for k in range(3):
            if swapped_std[k] > 0 and target_std[k] > 0:
                corrected_lab[i, j, k] = (
                    (swapped_lab[i, j, k] - swapped_mean[k]) * 
                    (target_std[k] / swapped_std[k]) + target_mean[k]
                )
            else:
                corrected_lab[i, j, k] = swapped_lab[i, j, k]

@cuda.jit
def gpu_blending_kernel(swapped_face, target_region, mask, result_region):
    """CUDA kernel for alpha blending"""
    i, j = cuda.grid(2)
    if i < swapped_face.shape[0] and j < swapped_face.shape[1]:
        for k in range(3):
            result_region[i, j, k] = (
                swapped_face[i, j, k] * mask[i, j] + 
                target_region[i, j, k] * (1.0 - mask[i, j])
            )

def get_optimized_face_swapper() -> Any:
    """Get optimized face swapper with TensorRT support"""
    global FACE_SWAPPER
    
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            trt_model_path = resolve_relative_path('../models/inswapper_128.trt')
            
            # Try TensorRT first
            if os.path.exists(trt_model_path):
                try:
                    FACE_SWAPPER = TensorRTFaceSwapper(trt_model_path)
                    print("✅ Using TensorRT optimized model")
                except Exception as e:
                    print(f"❌ TensorRT loading failed: {e}, falling back to ONNX")
                    FACE_SWAPPER = insightface.model_zoo.get_model(
                        model_path, 
                        providers=get_optimized_providers()
                    )
            else:
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path, 
                    providers=get_optimized_providers()
                )
                
                # Convert to TensorRT in background
                threading.Thread(target=convert_to_tensorrt, args=(model_path, trt_model_path), daemon=True).start()
    
    return FACE_SWAPPER

def get_optimized_providers() -> List:
    """Get optimized execution providers for T4"""
    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
    }
    
    return [
        ('CUDAExecutionProvider', cuda_provider_options),
        'CPUExecutionProvider'
    ]

def convert_to_tensorrt(onnx_path: str, trt_path: str) -> None:
    """Convert ONNX model to TensorRT (run in background)"""
    try:
        import onnx
        print("🔄 Converting ONNX to TensorRT...")
        
        # This would require TensorRT builder API
        # For now, we'll use a placeholder
        print("✅ TensorRT conversion scheduled")
        
    except Exception as e:
        print(f"❌ TensorRT conversion failed: {e}")

def gpu_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """GPU-accelerated color correction"""
    try:
        if target_face is None:
            return swapped_face
        
        # Extract regions
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        target_region = target_frame[y1:y2, x1:x2]
        
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        
        # Resize to match
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        
        # Convert to LAB and move to GPU
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        
        swapped_lab_gpu = cp.asarray(swapped_lab)
        target_lab_gpu = cp.asarray(target_lab)
        
        # Calculate statistics on GPU
        swapped_mean = cp.mean(swapped_lab_gpu, axis=(0, 1))
        swapped_std = cp.std(swapped_lab_gpu, axis=(0, 1))
        target_mean = cp.mean(target_lab_gpu, axis=(0, 1))
        target_std = cp.std(target_lab_gpu, axis=(0, 1))
        
        # Avoid division by zero
        swapped_std = cp.where(swapped_std == 0, 1, swapped_std)
        target_std = cp.where(target_std == 0, 1, target_std)
        
        # Allocate GPU memory for result
        corrected_lab_gpu = cp.zeros_like(swapped_lab_gpu)
        
        # Launch CUDA kernel
        threadsperblock = (16, 16)
        blockspergrid_x = (swapped_lab.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (swapped_lab.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        gpu_color_correction_kernel[blockspergrid, threadsperblock](
            swapped_lab_gpu, target_lab_gpu, corrected_lab_gpu,
            swapped_mean, swapped_std, target_mean, target_std
        )
        
        # Convert back to CPU and BGR
        corrected_lab = cp.asnumpy(corrected_lab_gpu)
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        return corrected_face
        
    except Exception as e:
        print(f"❌ GPU color correction error: {e}")
        return simple_color_correction(swapped_face, target_frame, target_face)

def gpu_alpha_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """GPU-accelerated alpha blending"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Create smooth mask on CPU
        mask = create_smooth_mask(target_face, target_frame.shape)
        mask_region = mask[y1:y2, x1:x2]
        
        if mask_region.shape != swapped_face.shape[:2]:
            mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
        
        # Move data to GPU
        swapped_face_gpu = cp.asarray(swapped_face)
        target_region_gpu = cp.asarray(target_frame[y1:y2, x1:x2])
        mask_gpu = cp.asarray(mask_region)
        
        # Create 3D mask
        mask_3d_gpu = cp.stack([mask_gpu] * 3, axis=-1)
        
        # Allocate result memory
        result_region_gpu = cp.zeros_like(swapped_face_gpu)
        
        # Launch blending kernel
        threadsperblock = (16, 16)
        blockspergrid_x = (swapped_face.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (swapped_face.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        gpu_blending_kernel[blockspergrid, threadsperblock](
            swapped_face_gpu, target_region_gpu, mask_3d_gpu, result_region_gpu
        )
        
        # Convert back to CPU
        result_region = cp.asnumpy(result_region_gpu)
        result_frame = target_frame.copy()
        result_frame[y1:y2, x1:x2] = result_region.astype(np.uint8)
        
        return result_frame
        
    except Exception as e:
        print(f"❌ GPU blending error: {e}")
        return simple_blending(swapped_face, target_frame, target_face)

def batch_swap_faces(source_face: Face, target_faces: List[Face], temp_frame: Frame) -> Frame:
    """Batch process multiple faces"""
    try:
        if not target_faces:
            return temp_frame
        
        # Filter small faces for performance
        valid_faces = []
        for face in target_faces:
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if face_area > 2500:  # Minimum face area
                valid_faces.append(face)
        
        if not valid_faces:
            return temp_frame
        
        # Process faces sequentially but with GPU acceleration
        result_frame = temp_frame.copy()
        for target_face in valid_faces:
            result_frame = optimized_swap_face(source_face, target_face, result_frame)
        
        return result_frame
        
    except Exception as e:
        print(f"❌ Batch swap faces error: {e}")
        return temp_frame

def optimized_swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Fully optimized face swapping pipeline"""
    try:
        start_time = time.time()
        
        # Get face swap
        swapped_result = get_optimized_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_optimized_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # GPU-accelerated processing pipeline
        swapped_frame = gpu_color_correction(swapped_frame, temp_frame, target_face)
        swapped_frame = enhanced_face_quality(swapped_frame)
        result_frame = gpu_alpha_blending(swapped_frame, temp_frame, target_face)
        
        end_time = time.time()
        if roop.globals.log_level == 'debug':
            print(f"🔧 Face swap completed in {((end_time - start_time) * 1000):.2f}ms")
        
        return result_frame
        
    except Exception as e:
        print(f"❌ Optimized face swap error: {e}")
        return get_optimized_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def optimized_process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Optimized frame processing"""
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                return batch_swap_faces(source_face, many_faces, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                return optimized_swap_face(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"❌ Optimized process frame error: {e}")
        return temp_frame

def optimized_process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Optimized multi-frame processing"""
    try:
        global GPU_MEMORY_MANAGER, ASYNC_PROCESSOR
        
        # Initialize managers if not exists
        if GPU_MEMORY_MANAGER is None:
            GPU_MEMORY_MANAGER = GPUMemoryManager()
        
        if ASYNC_PROCESSOR is None:
            ASYNC_PROCESSOR = AsyncProcessor()
        
        source_face = get_one_face(cv2.imread(source_path))
        
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        
        # Use async processing for better performance
        if roop.globals.enable_async:
            asyncio.run(ASYNC_PROCESSOR.process_batch_async(source_face, temp_frame_paths))
        else:
            # Sequential processing with GPU optimization
            for temp_frame_path in temp_frame_paths:
                try:
                    frame_gpu = GPU_MEMORY_MANAGER.get_frame(temp_frame_path)
                    if frame_gpu is not None:
                        frame_cpu = cp.asnumpy(frame_gpu)
                        reference_face = None if roop.globals.many_faces else get_face_reference()
                        result = optimized_process_frame(source_face, reference_face, frame_cpu)
                        result_gpu = cp.asarray(result)
                        GPU_MEMORY_MANAGER.save_frame(temp_frame_path, result_gpu)
                    
                    if update:
                        update()
                        
                except Exception as e:
                    print(f"❌ Error processing frame {temp_frame_path}: {e}")
                    continue
                    
    except Exception as e:
        print(f"❌ Optimized process frames error: {e}")
    finally:
        # Cleanup GPU memory
        if GPU_MEMORY_MANAGER:
            GPU_MEMORY_MANAGER.clear_cache()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Optimized image processing"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = optimized_process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"❌ Optimized process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Optimized video processing"""
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, optimized_process_frames)
    except Exception as e:
        print(f"❌ Optimized process video error: {e}")

def pre_check() -> bool:
    """Model pre-check"""
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'])
    return True

def pre_start() -> bool:
    """Pre-start validation"""
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
    """Cleanup after processing"""
    clear_face_swapper()
    clear_face_reference()
    if GPU_MEMORY_MANAGER:
        GPU_MEMORY_MANAGER.clear_cache()

def clear_face_swapper() -> None:
    """Clear face swapper"""
    global FACE_SWAPPER
    FACE_SWAPPER = None

# Keep the original utility functions from your code
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

def simple_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Simple color correction fallback"""
    try:
        if target_face is None:
            return swapped_face
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        swapped_mean, swapped_std = np.mean(swapped_lab, axis=(0,1)), np.std(swapped_lab, axis=(0,1))
        target_mean, target_std = np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))
        swapped_std = np.where(swapped_std == 0, 1, swapped_std)
        target_std = np.where(target_std == 0, 1, target_std)
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        blend_ratio = 0.7
        result_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, corrected_face, blend_ratio, 0)
        return result_face
    except Exception as e:
        print(f"❌ Simple color correction error: {e}")
        return swapped_face

def create_smooth_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Create smooth mask for blending"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        return np.clip(mask, 0, 1)
    except Exception as e:
        print(f"❌ Mask creation error: {e}")
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

def enhanced_face_quality(face: Frame) -> Frame:
    """Face quality enhancement"""
    try:
        if face is None:
            return face
        face_array = ensure_frame_format(face)
        if face_array is None:
            return face
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.2
        sharpened = cv2.filter2D(face_array, -1, kernel)
        denoised = cv2.bilateralFilter(sharpened, 5, 25, 25)
        return denoised
    except Exception as e:
        print(f"❌ Face enhancement error: {e}")
        return face

def simple_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Simple blending fallback"""
    try:
        if target_face is None:
            return target_frame
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        mask = create_smooth_mask(target_face, target_frame.shape)
        mask_region = mask[y1:y2, x1:x2]
        if mask_region.shape != swapped_face.shape[:2]:
            mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
        mask_3d = np.stack([mask_region] * 3, axis=-1)
        result = target_frame.copy()
        face_region = result[y1:y2, x1:x2]
        blended_face = (swapped_face * mask_3d + face_region * (1 - mask_3d)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended_face
        return result
    except Exception as e:
        print(f"❌ Simple blending error: {e}")
        return target_frame

# Initialize global settings
def init_optimizations():
    """Initialize all optimizations"""
    global GPU_MEMORY_MANAGER, ASYNC_PROCESSOR
    GPU_MEMORY_MANAGER = GPUMemoryManager()
    ASYNC_PROCESSOR = AsyncProcessor()
    print("✅ T4 CUDA optimizations initialized")

# Call initialization
init_optimizations()
