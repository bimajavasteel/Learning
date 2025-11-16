# hybrid-swapper-optimized.py
from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
import os
import gc
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional libs
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# ROOP integration
import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# Global
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER-HYBRID'

# --- Config tunable (ubah sesuai GPU/keperluan) ---
GPU_CONFIG = {
    'prefer_onnx_cuda': True,
    'gpu_mem_limit_bytes': 6 * 1024**3,   # 6GB default (sesuaikan)
    'max_workers_high_vram': 4,
    'max_workers_low_vram': 1,
    'batch_size_high_vram': 8,
    'batch_size_low_vram': 4,
    'memory_clear_interval': 25,
    'enable_tf32': True,    # gunakan jika PyTorch tersedia & Ampere+
    'warmup': True,
}

# --- Utility: safe GPU memory cleanup ---
def clear_gpu_memory():
    """Clear GPU memory and force GC - safe to call periodically."""
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # synchronize can help ensure memory is freed
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        gc.collect()
    except Exception as e:
        print(f"[clear_gpu_memory] warning: {e}")

# --- Device/Providers detection and model loader ---
def detect_providers():
    """Return a list of providers for onnxruntime / insightface model load."""
    if ONNXRUNTIME_AVAILABLE and GPU_CONFIG.get('prefer_onnx_cuda', True):
        try:
            provs = ort.get_available_providers()
            if 'CUDAExecutionProvider' in provs:
                cuda_opts = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': GPU_CONFIG['gpu_mem_limit_bytes'],
                    'cudnn_conv_algo_search': 'DEFAULT',
                }
                return [('CUDAExecutionProvider', cuda_opts)]
            # fallback to CPU provider if no CUDA
            return ['CPUExecutionProvider']
        except Exception as e:
            print(f"[detect_providers] ONNX provider detection error: {e}")
    # Fallback to whatever roop provides (compatibility)
    return roop.globals.execution_providers

def get_face_swapper() -> Any:
    """Load face swapper model with robust provider handling and fallbacks."""
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            providers = detect_providers()
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=providers)
                print("[get_face_swapper] loaded with providers:", providers)
            except Exception as e:
                print(f"[get_face_swapper] primary load failed: {e}, trying CPU fallback")
                try:
                    FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
                    print("[get_face_swapper] loaded with CPU fallback")
                except Exception as e2:
                    print(f"[get_face_swapper] CPU fallback failed: {e2}")
                    raise e2
            # Optional warm-up
            if GPU_CONFIG.get('warmup', True):
                try:
                    warmup_input = np.zeros((128,128,3), dtype=np.uint8)
                    # call minimal get to init kernels (paste_back False to save mem)
                    FACE_SWAPPER.get(warmup_input, None, None, paste_back=False)
                    print("[get_face_swapper] warmup done")
                except Exception as e:
                    print(f"[get_face_swapper] warmup failed: {e}")
    return FACE_SWAPPER

def clear_face_swapper():
    global FACE_SWAPPER
    FACE_SWAPPER = None
    clear_gpu_memory()

# --- Frame helpers ---
def ensure_frame_format(frame: Any) -> Optional[Frame]:
    if frame is None:
        return None
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        return frame
    if isinstance(frame, tuple) or isinstance(frame, list):
        try:
            arr = np.array(frame)
            if arr.size > 0 and arr.ndim >= 3:
                return arr
        except Exception:
            pass
    return None

# --- Color correction (hybrid LAB + per-channel fallback) ---
def color_correction_hybrid(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Combine LAB stats matching with per-channel fallback for robust mapping."""
    try:
        if target_face is None:
            return swapped_face
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return swapped_face
        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        # Resize swapped face to match target region
        if swapped_face.shape[:2] != target_region.shape[:2]:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        # Try LAB-based correction
        try:
            swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)
            s_mean, s_std = swapped_lab.mean(axis=(0,1)), swapped_lab.std(axis=(0,1))
            t_mean, t_std = target_lab.mean(axis=(0,1)), target_lab.std(axis=(0,1))
            s_std = np.where(s_std == 0, 1.0, s_std)
            t_std = np.where(t_std == 0, 1.0, t_std)
            corrected = (swapped_lab - s_mean) * (t_std / s_std) + t_mean
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            corrected_face = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
            # Blend to keep naturalness
            blend_ratio = 0.5
            result_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, corrected_face, blend_ratio, 0)
            return result_face
        except Exception as e:
            # Fallback: per-channel mean/std correction (RGB)
            sf = swapped_face.astype(np.float32)
            tf = target_region.astype(np.float32)
            for c in range(3):
                s_mean, s_std = sf[:,:,c].mean(), sf[:,:,c].std()
                t_mean, t_std = tf[:,:,c].mean(), tf[:,:,c].std()
                if s_std > 1 and t_std > 1:
                    sf[:,:,c] = (sf[:,:,c] - s_mean) * (t_std / max(s_std,1e-6)) + t_mean
                else:
                    sf[:,:,c] += (t_mean - s_mean) * 0.3
            result_face = np.clip(sf, 0, 255).astype(np.uint8)
            return result_face
    except Exception as e:
        print(f"[color_correction_hybrid] error: {e}")
        return swapped_face

# --- Mask creation (robust & adaptive) ---
def create_adaptive_mask(face: Face, frame_shape: Tuple[int,int]) -> np.ndarray:
    """Create an elliptical mask with dynamic kernel for smooth blending."""
    try:
        mask = np.zeros(frame_shape[:2], dtype=np.float32)
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame_shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return mask
        cx, cy = (x1+x2)//2, (y1+y2)//2
        width, height = x2-x1, y2-y1
        # ellipse
        cv2.ellipse(mask, (cx, cy), (max(1,width//2), max(1,height//2)), 0, 0, 360, 1.0, -1)
        # gaussian blur kernel adaptive
        k = max(15, min(width, height)//8)
        if k % 2 == 0:
            k += 1
        k = min(k, 81)
        mask = cv2.GaussianBlur(mask, (k,k), 0)
        return np.clip(mask, 0.0, 1.0)
    except Exception as e:
        print(f"[create_adaptive_mask] error: {e}")
        # fallback simple rectangle
        mask = np.zeros(frame_shape[:2], dtype=np.float32)
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            mask[y1:y2, x1:x2] = 1.0
            mask = cv2.GaussianBlur(mask, (51,51), 0)
            return np.clip(mask,0,1)
        except:
            return np.zeros(frame_shape[:2], dtype=np.float32)

# --- Face quality enhancement (light & fast) ---
def enhance_face_quality_fast(face: Frame) -> Frame:
    try:
        if face is None:
            return face
        arr = ensure_frame_format(face)
        if arr is None:
            return face
        # mild unsharp mask (fast)
        blurred = cv2.GaussianBlur(arr, (0,0), 1.0)
        sharpened = cv2.addWeighted(arr, 1.4, blurred, -0.4, 0)
        # fast bilateral denoise with small params
        denoised = cv2.bilateralFilter(sharpened, 3, 20, 20)
        return denoised
    except Exception as e:
        print(f"[enhance_face_quality_fast] {e}")
        return face

# --- Blending: optimized alpha blending with seamless fallback ---
def optimized_blend(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        if target_face is None:
            return target_frame
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(w,x2), min(h,y2)
        if x2 <= x1 or y2 <= y1:
            return target_frame
        face_h, face_w = y2-y1, x2-x1
        if swapped_face.shape[:2] != (face_h, face_w):
            swapped_face = cv2.resize(swapped_face, (face_w, face_h))
        # try seamlessClone first (usually best)
        try:
            mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
            center = ((x1+x2)//2, (y1+y2)//2)
            return cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            # fallback to alpha blending with adaptive mask
            mask = create_adaptive_mask(target_face, target_frame.shape)
            mask_region = mask[y1:y2, x1:x2]
            if mask_region.shape != swapped_face.shape[:2]:
                mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
            mask3 = np.stack([mask_region]*3, axis=-1)
            region = target_frame[y1:y2, x1:x2].astype(np.float32)
            blended = swapped_face.astype(np.float32) * mask3 + region * (1 - mask3)
            out = target_frame.copy()
            out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            return out
    except Exception as e:
        print(f"[optimized_blend] error: {e}")
        return target_frame

# --- Single-frame swap pipeline ---
def swap_face_pipeline(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        face_swapper = get_face_swapper()
        # get swapped face (paste_back False to get face image and let us blend)
        swapped_result = face_swapper.get(temp_frame, target_face, source_face, paste_back=False)
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            # fallback to paste_back True single call
            return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
        # color correct
        swapped_frame = color_correction_hybrid(swapped_frame, temp_frame, target_face)
        # enhance
        swapped_frame = enhance_face_quality_fast(swapped_frame)
        # blend
        result = optimized_blend(swapped_frame, temp_frame, target_face)
        return result
    except Exception as e:
        print(f"[swap_face_pipeline] error: {e}")
        try:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        except Exception as e2:
            print(f"[swap_face_pipeline] fallback error: {e2}")
            return temp_frame

# --- Batch / frames processing (dynamic batch size + memory management) ---
def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Main entrypoint: processes list of frame file paths with hybrid optimizations."""
    try:
        # get source face once
        src_img = cv2.imread(source_path)
        source_face = get_one_face(src_img)
        if source_face is None:
            print("[process_frames] no source face detected")
            return
        total = len(temp_frame_paths)
        if total == 0:
            return
        # choose parallelism based on GPU presence
        gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        max_workers = GPU_CONFIG['max_workers_high_vram'] if gpu_available else GPU_CONFIG['max_workers_low_vram']
        batch_size = GPU_CONFIG['batch_size_high_vram'] if gpu_available else GPU_CONFIG['batch_size_low_vram']
        memory_clear_interval = GPU_CONFIG['memory_clear_interval']

        print(f"[process_frames] total={total}, gpu={gpu_available}, max_workers={max_workers}, batch_size={batch_size}")

        processed = 0
        for i in range(0, total, batch_size):
            batch = temp_frame_paths[i:i+batch_size]
            # Use threaded workers but limit concurrency to avoid OOM
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(_process_single_path, p, source_face): p for p in batch}
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[process_frames] error processing {p}: {e}")
                    processed += 1
                    if update:
                        update()
            # periodic memory clear
            if processed % memory_clear_interval == 0:
                clear_gpu_memory()
                print(f"[process_frames] processed {processed}/{total} - memory cleared")
        print(f"[process_frames] completed {processed}/{total}")
    except Exception as e:
        print(f"[process_frames] critical error: {e}")
    finally:
        clear_gpu_memory()

def _process_single_path(path: str, source_face: Face) -> None:
    try:
        if not os.path.exists(path):
            print(f"[worker] missing frame: {path}")
            return
        frame = cv2.imread(path)
        if frame is None:
            print(f"[worker] cannot read: {path}")
            return
        if roop.globals.many_faces:
            many = get_many_faces(frame)
            if many:
                for tface in many:
                    frame = swap_face_pipeline(source_face, tface, frame)
        else:
            ref = get_face_reference()
            if ref is None:
                # find similar face on-the-fly
                target_face = find_similar_face(frame, None)
            else:
                target_face = find_similar_face(frame, ref)
            if target_face:
                frame = swap_face_pipeline(source_face, target_face, frame)
        cv2.imwrite(path, frame)
    except Exception as e:
        print(f"[_process_single_path] error: {e}")

# --- Image / Video helpers for compatibility ---
def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        src_face = get_one_face(cv2.imread(source_path))
        if src_face is None:
            print("[process_image] no source face")
            return
        frame = cv2.imread(target_path)
        if frame is None:
            print("[process_image] cannot read target")
            return
        if roop.globals.many_faces:
            many = get_many_faces(frame)
            if many:
                for t in many:
                    frame = swap_face_pipeline(src_face, t, frame)
        else:
            ref = get_face_reference()
            target_face = find_similar_face(frame, ref)
            if target_face:
                frame = swap_face_pipeline(src_face, target_face, frame)
        cv2.imwrite(output_path, frame)
        print(f"[process_image] saved {output_path}")
    except Exception as e:
        print(f"[process_image] error: {e}")
    finally:
        clear_gpu_memory()

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    try:
        if not roop.globals.many_faces and not get_face_reference():
            if temp_frame_paths and roop.globals.reference_frame_number < len(temp_frame_paths):
                reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
                if reference_frame is not None:
                    reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
                    if reference_face is not None:
                        set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"[process_video] error: {e}")
    finally:
        clear_gpu_memory()

# --- Pre/post checks ---
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
