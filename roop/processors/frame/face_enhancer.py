# enhancer_skintexture.py
# Hybrid GFPGAN enhancer + Skin Texture Restorer
# Integrates: GFPGAN enhancement (existing pipeline) + high-frequency restoration,
# micro-detail sharpening, and micro-grain injection.
# Designed to be a drop-in replacement for existing face_enhancer module.

from typing import Any, List, Callable
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER-SKINTEXTURE'

# -------------------------
# Helper: GFPGAN loader
# -------------------------

def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()

# -------------------------
# Skin / texture helpers
# -------------------------

def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def restore_skin_texture(enhanced: np.ndarray, original: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    High-frequency restoration: ambil high-frequency dari original dan blending ke enhanced.
    strength: proporsi HF yang ditambahkan ke enhanced (0..1)
    """
    try:
        # gunakan Gaussian blur besar untuk low-frequency
        k = 21
        if k % 2 == 0:
            k += 1
        blur_org = cv2.GaussianBlur(original, (k, k), 0)
        hf = cv2.subtract(original, blur_org)
        restored = cv2.addWeighted(enhanced.astype(np.float32), 1.0, hf.astype(np.float32), strength, 0)
        return _to_uint8(restored)
    except Exception:
        return enhanced


def microdetail_sharpen(face: np.ndarray, amount: float = 0.15) -> np.ndarray:
    """Unsharp mask kecil untuk menonjolkan microtexture tanpa membuat oversharpen."""
    try:
        blur = cv2.GaussianBlur(face, (5, 5), 0)
        sharp = cv2.addWeighted(face.astype(np.float32), 1.0 + amount, blur.astype(np.float32), -amount, 0)
        return _to_uint8(sharp)
    except Exception:
        return face


def add_micrograin(face: np.ndarray, level: float = 4.0) -> np.ndarray:
    """Add subtle sensor-like grain. level = stddev of gaussian noise"""
    try:
        h, w = face.shape[:2]
        noise = np.random.normal(0.0, level, (h, w, 1)).astype(np.float32)
        noise = np.repeat(noise, 3, axis=2)
        out = face.astype(np.float32) + noise
        return _to_uint8(out)
    except Exception:
        return face


def skin_mask_from_rgb(img: np.ndarray) -> np.ndarray:
    """
    Simple skin mask in YCrCb + HSV combination to localize processing to skin areas.
    Returns single-channel float mask 0..1.
    """
    try:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # thresholds tuned to include wide range skin tones
        mask1 = (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        mask2 = (s > 15) & (s < 200)

        mask = (mask1 | mask2).astype(np.uint8)
        # smooth the mask and convert to float
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        mask = np.clip(mask, 0.0, 1.0)
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        return mask
    except Exception:
        h, w = img.shape[:2]
        return np.ones((h, w, 3), dtype=np.float32)

# -------------------------
# Main enhancement flow
# -------------------------

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhancer utama, membuat per-face enhancement termasuk skin texture restoration.
    Target_face dapat berupa dict atau object yang memiliki ['bbox'] / .bbox
    """
    try:
        # baca bbox
        if isinstance(target_face, dict):
            bx = target_face.get('bbox')
        else:
            bx = getattr(target_face, 'bbox', None)
        if bx is None:
            return temp_frame

        start_x, start_y, end_x, end_y = map(int, bx)
        if start_x >= end_x or start_y >= end_y:
            return temp_frame

        frame_h, frame_w = temp_frame.shape[:2]

        # padding adaptif (inspirasi dari optimizer)
        face_w = end_x - start_x
        face_h = end_y - start_y
        pad_ratio = max(0.10, min(0.30, 100.0 / max(face_w, face_h)))
        px = int(face_w * pad_ratio)
        py = int(face_h * pad_ratio)

        sx = max(0, start_x - px)
        sy = max(0, start_y - py)
        ex = min(frame_w, end_x + px)
        ey = min(frame_h, end_y + py)

        crop = temp_frame[sy:ey, sx:ex]
        if crop.size == 0:
            return temp_frame

        with THREAD_SEMAPHORE:
            try:
                _, _, enhanced_face = get_face_enhancer().enhance(crop, paste_back=False)
            except Exception:
                try:
                    _, _, enhanced_face = get_face_enhancer().enhance(crop, paste_back=True)
                except Exception:
                    return temp_frame

        if enhanced_face is None:
            return temp_frame

        # resize back kalau GFPGAN mengganti ukuran
        if enhanced_face.shape[:2] != crop.shape[:2]:
            try:
                enhanced_face = cv2.resize(enhanced_face, (crop.shape[1], crop.shape[0]))
            except Exception:
                return temp_frame

        # parameter dari globals (tweakable)
        texture_strength = getattr(roop.globals, 'face_texture_strength', 0.35)
        micro_amount = getattr(roop.globals, 'face_micro_sharpen', 0.15)
        grain_level = getattr(roop.globals, 'face_micro_grain', 4.0)
        blend_amount = getattr(roop.globals, 'face_enhancer_blend', 0.6)

        # buat skin mask berdasarkan crop (agar tidak menimpa mata/bibir)
        mask = skin_mask_from_rgb(crop)

        # 1) Restore high-frequency dari original crop
        restored = restore_skin_texture(enhanced_face, crop, strength=texture_strength)

        # 2) Micro-sharpen
        restored = microdetail_sharpen(restored, amount=micro_amount)

        # 3) Micro-grain
        restored = add_micrograin(restored, level=grain_level)

        # 4) Composite hanya di skin area (mask) untuk menjaga mata & bibir
        composite = (restored.astype(np.float32) * mask + crop.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

        # 5) Color-match + fidelity blend (ambil dari enhancer-final style)
        # simple mean color alignment
        orig_mean = np.mean(crop, axis=(0, 1))
        comp_mean = np.mean(composite, axis=(0, 1))
        color_diff = orig_mean - comp_mean
        corrected = composite.astype(np.float32) + color_diff
        corrected = _to_uint8(corrected)

        final = cv2.addWeighted(corrected, blend_amount, crop, 1.0 - blend_amount, 0)

        # paste back
        temp_frame[sy:ey, sx:ex] = final

        return temp_frame

    except Exception as e:
        update_status(f"Enhancer error: {e}", NAME)
        return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            # jika occlusion besar, skip enhancing (agar tidak mempertegas noise)
            try:
                if getattr(target_face, 'det_score', 1.0) < 0.15:
                    continue
            except Exception:
                pass
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
