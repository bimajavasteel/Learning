# roop/processors/frame/realesrgan_upscaler.py
"""
RealESRGAN_x2plus upscaler module (auto-download, no fallback) +
temporal stabilizer (EMA anti-flicker).

Cara pakai oleh pipeline:
- pre_check() dipanggil sebelum frame processors berjalan
- process_frames() dijalankan setelah face_enhancer
"""

from typing import Any, List, Callable, Optional
import os
import cv2
import numpy as np
import threading

import roop.globals
import roop.core as core
from roop.utilities import conditional_download, resolve_relative_path

NAME = "ROOP.REALESRGAN-X2PLUS"
THREAD_LOCK = threading.Lock()
MODEL_PATH: Optional[str] = None

# Model URL (tanpa fallback — sesuai permintaan)
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

# Check library availability
REALSRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer  # type: ignore
    REALSRGAN_AVAILABLE = True
except Exception:
    REALSRGAN_AVAILABLE = False


# ---------------------------------------------------------------
# PRE-CHECK: download & verify model
# ---------------------------------------------------------------
def pre_check() -> bool:
    global MODEL_PATH
    models_dir = resolve_relative_path("../models")
    os.makedirs(models_dir, exist_ok=True)

    target_path = os.path.join(models_dir, "RealESRGAN_x2plus.pth")

    # auto-download (no fallback)
    conditional_download(models_dir, [MODEL_URL])

    if not os.path.exists(target_path):
        core.update_status(
            f"{NAME}: model file not found after download attempt: {target_path}",
            NAME
        )
        return False

    MODEL_PATH = target_path
    return True


# ---------------------------------------------------------------
# UPSCALER INITIALIZER
# ---------------------------------------------------------------
def get_upscaler(device: str = "cuda") -> Any:
    global MODEL_PATH, REALSRGAN_AVAILABLE
    with THREAD_LOCK:
        if not REALSRGAN_AVAILABLE:
            raise RuntimeError(f"{NAME}: Python package 'realesrgan' not installed.")
        if MODEL_PATH is None:
            raise RuntimeError(f"{NAME}: Model path not set. Run pre_check().")

        upscaler = RealESRGANer(
            model_path=MODEL_PATH,
            scale=2,
            device=device
        )
        return upscaler


# ---------------------------------------------------------------
# TEMPORAL STABILIZER (EMA)
# ---------------------------------------------------------------
def _temporal_stabilize(frames: List[np.ndarray], alpha: float = 0.6) -> List[np.ndarray]:
    if not frames:
        return frames

    stabilized = []
    prev = frames[0].astype(np.float32)
    stabilized.append(prev.astype(np.uint8))

    for i in range(1, len(frames)):
        cur = frames[i].astype(np.float32)
        s = alpha * cur + (1.0 - alpha) * prev
        out = np.clip(s, 0, 255).astype(np.uint8)
        stabilized.append(out)
        prev = s

    return stabilized


# ---------------------------------------------------------------
# UPSCALE SINGLE FRAME
# ---------------------------------------------------------------
def _upscale_frame_with_realesrgan(upscaler: Any, frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        result = upscaler.enhance(rgb, outscale=2)
        sr = result[0] if isinstance(result, (tuple, list)) else result

        if sr.dtype != np.uint8:
            sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

        return cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

    except Exception as e:
        raise RuntimeError(f"{NAME}: Upscale failed: {str(e)}")


# ---------------------------------------------------------------
# PROCESS VIDEO FRAMES
# ---------------------------------------------------------------
def process_frames(source_path: Optional[str], temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    if not temp_frame_paths:
        return

    # Determine device
    device = "cuda" if "CUDAExecutionProvider" in roop.globals.execution_providers else "cpu"
    upscaler = get_upscaler(device=device)

    loaded_frames = []
    for p in temp_frame_paths:
        img = cv2.imread(p)
        if img is None:
            core.update_status(f"{NAME}: gagal baca frame {p}", NAME)
            raise RuntimeError(f"{NAME}: Failed to read frame {p}")
        loaded_frames.append(img)

    # Upscale each frame
    upscaled = []
    for frame in loaded_frames:
        sr = _upscale_frame_with_realesrgan(upscaler, frame)
        upscaled.append(sr)
        if update:
            update()

    # Temporal Stabilizer
    alpha = getattr(roop.globals, "realesrgan_temporal_alpha", 0.6)
    stabilized = _temporal_stabilize(upscaled, alpha=alpha)

    # Overwrite frames
    for path, img in zip(temp_frame_paths, stabilized):
        cv2.imwrite(path, img)
        if update:
            update()


# ---------------------------------------------------------------
# PROCESS SINGLE IMAGE
# ---------------------------------------------------------------
def process_image(source_path: str, target_path: str, output_path: str) -> None:
    device = "cuda" if "CUDAExecutionProvider" in roop.globals.execution_providers else "cpu"
    upscaler = get_upscaler(device=device)

    img = cv2.imread(target_path)
    if img is None:
        raise RuntimeError(f"{NAME}: gagal baca image {target_path}")

    sr = _upscale_frame_with_realesrgan(upscaler, img)
    cv2.imwrite(output_path, sr)


# ---------------------------------------------------------------
# STARTUP VALIDATION
# ---------------------------------------------------------------
def pre_start() -> bool:
    if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
        core.update_status(f"{NAME}: Model belum tersedia. Jalankan pre_check().", NAME)
        return False
    return True


def post_process() -> None:
    return None
