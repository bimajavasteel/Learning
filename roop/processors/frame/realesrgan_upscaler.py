# roop/processors/frame/realesrgan_upscaler.py
"""
RealESRGAN_x2plus ONNX version (no RealESRGANer required)
- Loads fp16 ONNX from HuggingFace
- Runs inference through onnxruntime (CUDA or CPU)
- Temporal Stabilizer included
"""

from typing import Any, List, Callable, Optional
import os
import cv2
import numpy as np
import threading
import onnxruntime as ort

import roop.globals
import roop.core as core
from roop.utilities import conditional_download, resolve_relative_path

NAME = "ROOP.REALESRGAN-X2PLUS"
THREAD_LOCK = threading.Lock()
MODEL_PATH: Optional[str] = None

MODEL_URL = "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/RealESRGAN_x2plus.fp16.onnx"


# ---------------------------------------------------------------
# DOWNLOAD MODEL
# ---------------------------------------------------------------
def pre_check() -> bool:
    global MODEL_PATH
    models_dir = resolve_relative_path("../models")
    os.makedirs(models_dir, exist_ok=True)

    target_path = os.path.join(models_dir, "RealESRGAN_x2plus.onnx")

    conditional_download(models_dir, [MODEL_URL])

    # rename downloaded file if needed
    if os.path.exists(os.path.join(models_dir, "RealESRGAN_x2plus.fp16.onnx")):
        os.rename(
            os.path.join(models_dir, "RealESRGAN_x2plus.fp16.onnx"),
            target_path
        )

    if not os.path.exists(target_path):
        core.update_status(f"{NAME}: model not found after download.", NAME)
        return False

    MODEL_PATH = target_path
    return True


# ---------------------------------------------------------------
# LOAD ONNX SESSION
# ---------------------------------------------------------------
def get_onnx_session() -> ort.InferenceSession:
    global MODEL_PATH

    providers = []
    if "CUDAExecutionProvider" in roop.globals.execution_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    return ort.InferenceSession(
        MODEL_PATH,
        providers=providers
    )


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
# ONNX UPSCALING LOGIC
# ---------------------------------------------------------------
def upscale_onnx(session: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    # Convert BGR→RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0,1]
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]  # NCHW

    inp_tensor = {session.get_inputs()[0].name: inp}

    out = session.run(None, inp_tensor)[0]

    # Convert back
    out = np.squeeze(out, axis=0)
    out = np.transpose(out, (1, 2, 0))
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------
# MAIN PROCESS FUNCTION
# ---------------------------------------------------------------
def process_frames(source_path: Optional[str], temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    if not temp_frame_paths:
        return

    session = get_onnx_session()

    frames = []
    for p in temp_frame_paths:
        img = cv2.imread(p)
        if img is None:
            core.update_status(f"{NAME}: Failed to read frame {p}", NAME)
            raise RuntimeError(f"{NAME}: Cannot load frame {p}")
        frames.append(img)

    # Upscale
    upscaled = []
    for f in frames:
        out = upscale_onnx(session, f)
        upscaled.append(out)
        if update:
            update()

    # Temporal smoothing
    alpha = getattr(roop.globals, "realesrgan_temporal_alpha", 0.6)
    stabilized = _temporal_stabilize(upscaled, alpha)

    # Write back
    for p, img in zip(temp_frame_paths, stabilized):
        cv2.imwrite(p, img)
        if update:
            update()


# ---------------------------------------------------------------
# SINGLE IMAGE MODE
# ---------------------------------------------------------------
def process_image(source_path: str, target_path: str, output_path: str) -> None:
    session = get_onnx_session()
    img = cv2.imread(target_path)
    if img is None:
        raise RuntimeError(f"{NAME}: Cannot load image {target_path}")

    out = upscale_onnx(session, img)
    cv2.imwrite(output_path, out)


def pre_start() -> bool:
    if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
        core.update_status(f"{NAME}: model not ready, run pre_check().", NAME)
        return False
    return True


def post_process() -> None:
    return None
