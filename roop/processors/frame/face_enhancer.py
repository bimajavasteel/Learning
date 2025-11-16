# face_enhancer_gpu_opt.py
from typing import Any, List, Callable, Optional, Tuple
import os
import time
import threading
import traceback

import cv2
import numpy as np

# GFPGAN
from gfpgan.utils import GFPGANer

# ROOP integration
import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# Optional: Torch (for GPU tuning & VRAM detection)
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# Global state
FACE_ENHANCER: Optional[Any] = None
NAME = 'ROOP.FACE-ENHANCER-GPU-OPT'
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()  # will be adjusted on init

# --- TUNING ---
GPU_TUNING = {
    # Target VRAM usage: adjust to your GPU (bytes). Default tuned for ~11GB usage on T4.
    'target_vram_bytes': 11 * 1024**3,
    'max_workers_high_vram': 4,
    'max_workers_low_vram': 1,
    'warmup_enabled': True,
    'warmup_iters': 2,
    'enable_tf32': True,     # use on Ampere+ GPUs when torch available
    'cudnn_benchmark': True,
    'torch_alloc_conf': 'max_split_size_mb:128'
}

# --- UTILS ---
def get_device() -> str:
    """Pilih device prioritas: cuda -> mps -> cpu berdasarkan roop globals & torch."""
    # Prefer execution providers from roop.globals first
    try:
        eps = roop.globals.execution_providers
        if 'CUDAExecutionProvider' in eps:
            return 'cuda'
        if 'CoreMLExecutionProvider' in eps:
            return 'mps'
        if 'CPUExecutionProvider' in eps:
            return 'cpu'
    except Exception:
        pass

    # Fallback to torch detection
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass

    return 'cpu'

def configure_torch_for_gpu(device: str):
    """Konfigurasi PyTorch untuk performa GPU maksimal (jika tersedia)."""
    if not TORCH_AVAILABLE:
        print("[GPU OPT] PyTorch not available - skipping torch GPU optimizations")
        return

    try:
        if device == 'cuda':
            # TF32 settings (Ampere+)
            if GPU_TUNING.get('enable_tf32', False):
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
            # cudnn benchmark
            if GPU_TUNING.get('cudnn_benchmark', True):
                try:
                    torch.backends.cudnn.benchmark = True
                except Exception:
                    pass
            # PyTorch allocator config for fragmentation
            try:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = GPU_TUNING.get('torch_alloc_conf', 'max_split_size_mb:128')
            except Exception:
                pass

            print("[GPU OPT] Torch GPU optimizations applied")
        elif device == 'mps':
            print("[GPU OPT] MPS device detected (Apple Silicon) - limited tuning applied")
    except Exception as e:
        print(f"[GPU OPT] configure error: {e}")

def estimate_gpu_vram() -> int:
    """Estimate total VRAM (bytes) using torch if available; fallback 8GB."""
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return int(prop.total_memory)
    except Exception:
        pass
    # fallback guess
    return 8 * 1024**3

def choose_worker_counts() -> Tuple[int, int]:
    """Return (max_workers, batch_size) based on detected VRAM."""
    total_vram = estimate_gpu_vram()
    target = GPU_TUNING['target_vram_bytes']
    if total_vram >= target:
        return GPU_TUNING['max_workers_high_vram'], max(1, int(min(12, target // (64 * 1024**2))))
    else:
        return GPU_TUNING['max_workers_low_vram'], 1

def clear_gpu_cache():
    """Clear GPU memory caches if torch is available."""
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # optional: collect ipc handles (if available)
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as e:
        print(f"[GPU OPT] clear cache error: {e}")

# --- GFPGAN loader & warmup ---
def get_face_enhancer() -> Any:
    """Lazy-load GFPGANer with GPU optimizations and warmup."""
    global FACE_ENHANCER, THREAD_SEMAPHORE

    device = get_device()
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # ensure models present
            if not os.path.exists(model_path):
                # pre_check handles download, but safeguard:
                print(f"[{NAME}] Model not found at {model_path}, call pre_check() first.")
            # configure torch for GPU
            configure_torch_for_gpu(device)

            # adjust semaphore initial value based on VRAM
            try:
                max_workers, _ = choose_worker_counts()
                THREAD_SEMAPHORE = threading.Semaphore(max_workers)
            except Exception:
                THREAD_SEMAPHORE = threading.Semaphore(1)

            # create GFPGANer
            try:
                FACE_ENHANCER = GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    device=device,
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print(f"[{NAME}] GFPGAN loaded on {device}")
            except Exception as e:
                print(f"[{NAME}] Error loading GFPGAN: {e}")
                raise

            # warmup model (run a couple dummy inferences to initialize CUDA kernels)
            if GPU_TUNING.get('warmup_enabled', True) and device == 'cuda':
                try:
                    perform_warmup_inference(FACE_ENHANCER, iters=GPU_TUNING.get('warmup_iters', 2))
                except Exception as e:
                    print(f"[{NAME}] Warmup failed: {e}")

    return FACE_ENHANCER

def perform_warmup_inference(face_enhancer: Any, iters: int = 2):
    """Perform warmup inferences with a dummy image to reduce first-frame lag."""
    try:
        print(f"[{NAME}] Warm-up: running {iters} dummy inferences...")
        for i in range(iters):
            dummy = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            # Use semaphore to ensure concurrency respects limits
            with THREAD_SEMAPHORE:
                face_enhancer.enhance(dummy, paste_back=False)
        print(f"[{NAME}] Warm-up completed")
    except Exception as e:
        print(f"[{NAME}] Warm-up exception: {e}")

# --- cleanup ---
def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    try:
        FACE_ENHANCER = None
        clear_gpu_cache()
        print(f"[{NAME}] enhancer cleared and GPU cache emptied")
    except Exception as e:
        print(f"[{NAME}] clear error: {e}")

# --- pre/post checks (ROOP API) ---
def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_enhancer()

# --- face enhancement logic (optimized) ---
def calculate_roi_with_padding(face: Face, frame_shape: Tuple[int, int], min_size: int = 24) -> Tuple[int,int,int,int]:
    """Calculate ROI with adaptive padding and bounds checking."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, face['bbox'])
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)

    # adaptive padding: smaller faces get larger relative padding
    pad_ratio = max(0.12, min(0.3, 120.0 / max(face_w, face_h)))
    px = int(face_w * pad_ratio)
    py = int(face_h * pad_ratio)

    sx = max(0, x1 - px)
    sy = max(0, y1 - py)
    ex = min(w, x2 + px)
    ey = min(h, y2 + py)

    # ensure minimal size to avoid tiny crops
    if (ex - sx) < min_size:
        center_x = (sx + ex) // 2
        sx = max(0, center_x - min_size//2)
        ex = min(w, center_x + min_size//2)
    if (ey - sy) < min_size:
        center_y = (sy + ey) // 2
        sy = max(0, center_y - min_size//2)
        ey = min(h, center_y + min_size//2)

    return sx, sy, ex, ey

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """Enhance a single face region using GFPGAN, with semaphore to limit concurrency."""
    try:
        sx, sy, ex, ey = calculate_roi_with_padding(target_face, temp_frame.shape)
        crop = temp_frame[sy:ey, sx:ex]
        if crop.size == 0:
            return temp_frame

        # Acquire semaphore (limits parallel enhancement on GPU)
        with THREAD_SEMAPHORE:
            try:
                # prefer paste_back=True because we want in-place paste and speed
                _, _, enhanced = get_face_enhancer().enhance(crop, paste_back=True)
                # sometimes enhancer changes size; check and paste intelligently
                if enhanced is not None and enhanced.size != 0:
                    eh, ew = enhanced.shape[:2]
                    # ensure target region same size; if not, resize
                    if (eh, ew) != (ey - sy, ex - sx):
                        try:
                            enhanced_resized = cv2.resize(enhanced, (ex - sx, ey - sy), interpolation=cv2.INTER_LINEAR)
                            temp_frame[sy:ey, sx:ex] = enhanced_resized
                        except Exception:
                            # as last resort, only paste center
                            temp_frame[sy:sy+min(eh, ey-sy), sx:sx+min(ew, ex-sx)] = enhanced[:min(eh, ey-sy), :min(ew, ex-sx)]
                    else:
                        temp_frame[sy:ey, sx:ex] = enhanced
            except Exception as e:
                print(f"[{NAME}] enhancer error: {e}")
                # fallback: leave original crop
    except Exception as e:
        print(f"[{NAME}] enhance_face unexpected: {e}\n{traceback.format_exc()}")
    return temp_frame

# --- frame processors (ROOP API) ---
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Single-frame processing: detect faces and enhance each."""
    try:
        faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame
        for f in faces:
            temp_frame = enhance_face(f, temp_frame)
    except Exception as e:
        print(f"[{NAME}] process_frame error: {e}")
    return temp_frame

def _process_single_path(path: str) -> None:
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"[{NAME}] cannot read {path}")
            return
        res = process_frame(None, None, img)
        cv2.imwrite(path, res)
    except Exception as e:
        print(f"[{NAME}] _process_single_path error: {e}")

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames with ThreadPoolExecutor limited by VRAM-based worker count."""
    try:
        max_workers, batch_size = choose_worker_counts()
        # limit concurrency sensibly
        max_workers = max(1, min(max_workers, len(temp_frame_paths)))
        print(f"[{NAME}] processing {len(temp_frame_paths)} frames with max_workers={max_workers}, batch_size={batch_size}")
        # Process sequentially by batches to control memory spikes
        for i in range(0, len(temp_frame_paths), batch_size):
            batch = temp_frame_paths[i:i+batch_size]
            # Use ThreadPoolExecutor; semaphore limits actual GFPGAN concurrency
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(_process_single_path, p): p for p in batch}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[{NAME}] frame job failed: {e}")
                    if update:
                        update()
            # periodic cleanup to avoid fragmentation
            clear_gpu_cache()
    except Exception as e:
        print(f"[{NAME}] process_frames error: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        frame = cv2.imread(target_path)
        if frame is None:
            print(f"[{NAME}] cannot read {target_path}")
            return
        res = process_frame(None, None, frame)
        cv2.imwrite(output_path, res)
    except Exception as e:
        print(f"[{NAME}] process_image error: {e}")
    finally:
        clear_gpu_cache()

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    try:
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"[{NAME}] process_video error: {e}")
    finally:
        clear_gpu_cache()
