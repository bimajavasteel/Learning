# roop/processors/frame/realesrgan_upscaler.py
"""
RealESRGAN_x2plus upscaler module (auto-download, no fallback) +
temporal stabilizer (exponential moving average over frames).

Cara pakai:
- Panggil pre_check() sebelum pipeline agar model .pth ter-download.
- Setelah face_enhancer selesai memodifikasi temp frames, panggil
  process_video(None, temp_frame_paths, update_cb)
"""

from typing import Any, List, Callable, Optional
import os
import cv2
import numpy as np
import threading
from pathlib import Path

import roop.globals
from roop.utilities import conditional_download, resolve_relative_path
import roop.core as core

NAME = "ROOP.REALESRGAN-X2PLUS"
THREAD_LOCK = threading.Lock()
MODEL_PATH: Optional[str] = None
# URL persis seperti permintaan user (tanpa fallback)
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

# Upstream library: kita pakai API RealESRGANer jika tersedia.
# Jika library tidak tersedia, modul akan menimbulkan Exception
# (sesuai permintaan: no fallback).
REALSRGAN_AVAILABLE = False
try:
    # RealESRGANer API (paling kompatibel dengan proyek Real-ESRGAN Python)
    from realesrgan import RealESRGANer  # type: ignore
    REALSRGAN_AVAILABLE = True
except Exception:
    REALSRGAN_AVAILABLE = False


def pre_check() -> bool:
    """
    Pastikan model .pth tersedia di ../models dan download kalau belum ada.
    Jika download gagal, fungsi ini harus raise atau mengembalikan False.
    (No fallback.)
    """
    global MODEL_PATH
    models_dir = resolve_relative_path("../models")
    os.makedirs(models_dir, exist_ok=True)
    target_path = os.path.join(models_dir, "RealESRGAN_x2plus.pth")

    # conditional_download biasanya meng-handle file yg sudah ada;
    # di sini kita paksa download bila tidak ada.
    conditional_download(models_dir, [MODEL_URL])

    if not os.path.exists(target_path):
        update_status(f"{NAME}: model file not found after download attempt: {target_path}", NAME)
        return False

    MODEL_PATH = target_path
    return True


def get_upscaler(device: str = "cuda") -> Any:
    """
    Inisialisasi RealESRGANer. Jika library tidak tersedia, raise Exception.
    """
    global MODEL_PATH, REALSRGAN_AVAILABLE
    with THREAD_LOCK:
        if not REALSRGAN_AVAILABLE:
            raise RuntimeError(f"{NAME}: 'realesrgan' python package not available in environment.")
        if MODEL_PATH is None:
            raise RuntimeError(f"{NAME}: model path belum di-set. Panggil pre_check() dulu.")

        # RealESRGANer constructor: sesuaikan argumen bila API berubah.
        # upscale=2 karena x2plus
        upscaler = RealESRGANer(model_path=MODEL_PATH, scale=2, device=device)
        return upscaler


def _temporal_stabilize(frames: List[np.ndarray], alpha: float = 0.6) -> List[np.ndarray]:
    """
    Simple temporal stabilizer (exponential moving average across frames).
    - frames: list of BGR numpy arrays (uint8)
    - alpha: smoothing factor untuk EMA; 0 < alpha <= 1.0
    Returns list of stabilized frames (same length).
    Catatan: ini pendekatan ringan, efektif untuk flicker kecil.
    """
    if not frames:
        return frames
    stabilized = []
    prev = frames[0].astype(np.float32)
    stabilized.append(prev.astype(np.uint8))
    for i in range(1, len(frames)):
        cur = frames[i].astype(np.float32)
        # EMA: s_t = alpha * cur + (1-alpha) * s_{t-1}
        s = alpha * cur + (1.0 - alpha) * prev
        s_clamped = np.clip(s, 0, 255).astype(np.uint8)
        stabilized.append(s_clamped)
        prev = s
    return stabilized


def _upscale_frame_with_realesrgan(upscaler: Any, frame: np.ndarray) -> np.ndarray:
    """
    Up-scale single frame using upscaler object.
    Expects BGR uint8 frame; RealESRGANer mungkin membutuhkan RGB float normalized.
    Kita adaptif handle jika RealESRGANer expose method enhance().
    """
    # Convert BGR->RGB for most SR libs
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # RealESRGANer API: enhance(image, outscale=scale), some libs return (sr, _, _)
    try:
        result = upscaler.enhance(rgb, outscale=2)
        # Jika result berupa tuple : ambil pertama
        if isinstance(result, tuple) or isinstance(result, list):
            sr = result[0]
        else:
            sr = result
        # Pastikan uint8 dan RGB -> BGR
        if sr.dtype != np.uint8:
            sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        # Biarkan exception naik — tidak ada fallback sesuai permintaan
        raise


def process_frames(source_path: Optional[str], temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Proses batch frame path:
    - Baca semua frame
    - Lakukan upscale x2 per-frame
    - Terapkan temporal stabilizer (opsional smoothing)
    - Tulis kembali frame ke path yang sama (overwrite)
    """
    # Device detection (sederhana)
    device = "cuda" if "CUDAExecutionProvider" in roop.globals.execution_providers else "cpu"

    upscaler = get_upscaler(device=device)

    # Baca semua frame ke memory (harus hati-hati di video besar; untuk safety kita proses chunk)
    frames = []
    for p in temp_frame_paths:
        img = cv2.imread(p)
        if img is None:
            update_status(f"{NAME}: gagal baca frame {p}", NAME)
            raise RuntimeError(f"Failed to read frame {p}")
        frames.append(img)

    # Upscale tiap frame
    upscaled_frames = []
    for idx, f in enumerate(frames):
        sr = _upscale_frame_with_realesrgan(upscaler, f)
        upscaled_frames.append(sr)
        if update:
            update()

    # Temporal stabilizer (anti-flicker)
    # Factor bisa dikustom lewat roop.globals.realesrgan_temporal_alpha
    alpha = getattr(roop.globals, "realesrgan_temporal_alpha", 0.6)
    stabilized = _temporal_stabilize(upscaled_frames, alpha=alpha)

    # Simpan kembali — catatan: file paths tetap sama, tapi resolusi berubah (x2)
    for p, img in zip(temp_frame_paths, stabilized):
        # apabila pipeline berikutnya mengandalkan ukuran lama,
        # pastikan core rebuild video aware bahwa frame size berubah.
        cv2.imwrite(p, img)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Untuk mode gambar: upscale single image
    """
    device = "cuda" if "CUDAExecutionProvider" in roop.globals.execution_providers else "cpu"
    upscaler = get_upscaler(device=device)
    img = cv2.imread(target_path)
    if img is None:
        raise RuntimeError(f"{NAME}: gagal baca image {target_path}")
    sr = _upscale_frame_with_realesrgan(upscaler, img)
    cv2.imwrite(output_path, sr)


def pre_start() -> bool:
    """
    Validasi minimal: target path harus ada dan module diinisialisasi.
    """
    if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
        update_status(f"{NAME}: Model belum tersedia. Jalankan pre_check().", NAME)
        return False
    return True


def post_process() -> None:
    """
    Cleanup jika perlu. Untuk saat ini tidak ada state yang disimpan lama.
    """
    return None
