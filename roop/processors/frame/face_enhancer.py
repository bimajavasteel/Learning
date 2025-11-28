# enhancer-hybrid.py
# Hybrid GFPGAN enhancer: fast mode + quality mode (blend + anti-flicker)
# Integrates: adaptive padding & safe-paste (from enhancer-optimize),
# and color-match + fidelity blending + ellipse mask (from enhancer-final).
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
NAME = 'ROOP.FACE-ENHANCER-HYBRID'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('./models/GFPGANv1.4.pth')
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
    download_directory_path = resolve_relative_path('./models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


# ---------------------------
# Utilities: color match, blend, mask
# ---------------------------
def color_match_and_correct(enhanced: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Simple mean color alignment (anti-flicker helper)."""
    try:
        if enhanced.shape[:2] != original.shape[:2]:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
        orig_mean = np.mean(original, axis=(0, 1))
        enh_mean = np.mean(enhanced, axis=(0, 1))
        diff = orig_mean - enh_mean
        corrected = enhanced.astype(np.float32) + diff
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected
    except Exception:
        return enhanced


def create_ellipse_mask(h: int, w: int, blur_scale: float = 0.1) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.45))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    br = int(min(w, h) * blur_scale)
    if br % 2 == 0:
        br += 1
    if br < 1:
        br = 1
    mask = cv2.GaussianBlur(mask, (br, br), 0)
    return np.dstack([mask] * 3)


def apply_fidelity_blend(enhanced_crop: np.ndarray, original_crop: np.ndarray, fidelity: float) -> np.ndarray:
    """Combine corrected enhanced with original using ellipse mask and fidelity weight."""
    try:
        corrected = color_match_and_correct(enhanced_crop, original_crop)
        blended = cv2.addWeighted(corrected, fidelity, original_crop, 1.0 - fidelity, 0)
        h, w = original_crop.shape[:2]
        mask_3ch = create_ellipse_mask(h, w)
        final = (blended * mask_3ch + original_crop * (1.0 - mask_3ch)).astype(np.uint8)
        return final
    except Exception:
        return original_crop


# ---------------------------
# Adaptive padding (fast-safe)
# ---------------------------
def compute_adaptive_padding(box, frame_shape, min_ratio=0.10, max_ratio=0.30, clamp_base=100):
    x1, y1, x2, y2 = map(int, box)
    face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
    pad_ratio = max(min_ratio, min(max_ratio, clamp_base / max(face_w, face_h)))
    pad_x = int(face_w * pad_ratio)
    pad_y = int(face_h * pad_ratio)
    h, w = frame_shape[:2]
    sx = max(0, x1 - pad_x)
    sy = max(0, y1 - pad_y)
    ex = min(w, x2 + pad_x)
    ey = min(h, y2 + pad_y)
    return sx, sy, ex, ey


# ---------------------------
# Main enhancer logic
# ---------------------------
def enhance_face_hybrid(target_face: Face, temp_frame: Frame) -> Frame:
    # get bbox and adaptive pad
    try:
        frame_h, frame_w = temp_frame.shape[:2]
        # support both face dicts and objects with bbox attr
        if isinstance(target_face, dict):
            box = target_face.get('bbox')
        else:
            box = getattr(target_face, 'bbox', None)
        if box is None:
            return temp_frame

        sx, sy, ex, ey = compute_adaptive_padding(box, temp_frame.shape)
        crop = temp_frame[sy:ey, sx:ex]
        if crop.size == 0:
            return temp_frame

        # mode selection: fast if global flag set, else quality
        fast_mode = getattr(roop.globals, 'face_enhancer_fast', False)

        with THREAD_SEMAPHORE:
            try:
                _, _, enhanced = get_face_enhancer().enhance(crop, paste_back=False)
            except Exception as e:
                # fallback safe call (paste_back True) to avoid shape mismatch causing crash
                try:
                    _, _, enhanced = get_face_enhancer().enhance(crop, paste_back=True)
                except Exception:
                    # if GFPGAN fails, silently return original crop
                    return temp_frame

        # if GFPGAN changed size, resize back to crop shape (safe-paste behavior)
        if enhanced is None:
            return temp_frame
        if enhanced.shape[:2] != crop.shape[:2]:
            try:
                enhanced = cv2.resize(enhanced, (crop.shape[1], crop.shape[0]))
            except Exception:
                return temp_frame

        if fast_mode:
            # fast path: simple safe replace (less flicker handling)
            temp_frame[sy:ey, sx:ex] = enhanced
            return temp_frame

        # quality path: color-match + fidelity blend using global blend param
        blend_amount = getattr(roop.globals, 'face_enhancer_blend', 0.6)
        final_crop = apply_fidelity_blend(enhanced, crop, fidelity=blend_amount)
        temp_frame[sy:ey, sx:ex] = final_crop
        return temp_frame

    except Exception:
        return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for f in many_faces:
            temp_frame = enhance_face_hybrid(f, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for p in temp_frame_paths:
        temp_frame = cv2.imread(p)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(p, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
