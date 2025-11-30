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
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video
)

# ==========================
#  SINGLE WRINKLE MODULE
# ==========================
from roop.processors.frame.smart_wrinkle_map import (
    detect_expression,
    compute_wrinkle_strength,
    full_face_wrinkle,
    apply_expression_wrinkle,
    apply_smart_wrinkle_map
)

# ==========================
#  PERLIN MICRO NOISE
# ==========================
from roop.processors.frame.perlin_skin_noise import add_subtle_skin_noise


FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-ENHANCER"


# ============================================================
# DEVICE
# ============================================================
def get_device() -> str:
    if "CUDAExecutionProvider" in roop.globals.execution_providers:
        return "cuda"
    if "CoreMLExecutionProvider" in roop.globals.execution_providers:
        return "mps"
    return "cpu"


# ============================================================
# LOAD GFPGAN
# ============================================================
def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path("../models/GFPGANv1.4.pth")
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=1,
                device=get_device()
            )
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path("../models")
    conditional_download(download_directory_path, [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


# ============================================================
# COLOR-MATCH + ELLIPSE MASK BLENDING
# ============================================================
def apply_blend_and_color_match(
    enhanced_crop: np.ndarray,
    original_crop: np.ndarray,
    fidelity: float
) -> np.ndarray:

    try:
        h, w = original_crop.shape[:2]

        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))

        # Color correction
        orig_mean = np.mean(original_crop, axis=(0, 1))
        enh_mean = np.mean(enhanced_crop, axis=(0, 1))
        diff = orig_mean - enh_mean

        corrected = enhanced_crop.astype(np.float32) + diff
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        # Blend
        blended = cv2.addWeighted(corrected, fidelity, original_crop, 1 - fidelity, 0)

        # Anti-box mask
        mask = np.zeros((h, w), np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        br = int(min(w, h) * 0.1)
        if br % 2 == 0:
            br += 1
        mask = cv2.GaussianBlur(mask, (br, br), 0)
        mask3 = np.dstack([mask] * 3)

        out = blended * mask3 + original_crop * (1 - mask3)
        return out.astype(np.uint8)

    except Exception:
        return original_crop


# ============================================================
# FACE ENHANCER CORE
# ============================================================
def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:

    try:
        # BBOX + padding
        x1, y1, x2, y2 = map(int, target_face["bbox"])

        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)

        H, W = temp_frame.shape[:2]
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)

        crop = temp_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return temp_frame

        # ----------------------------------------------------
        # 1) GFPGAN
        # ----------------------------------------------------
        with THREAD_SEMAPHORE:
            try:
                _, _, enhanced = get_face_enhancer().enhance(crop, paste_back=True)
            except Exception:
                enhanced = crop.copy()

        # ----------------------------------------------------
        # 2) BLENDING
        # ----------------------------------------------------
        fidelity = getattr(roop.globals, "face_enhancer_blend", 0.6)
        try:
            fidelity = float(fidelity)
        except:
            fidelity = 0.6

        result = apply_blend_and_color_match(enhanced, crop, fidelity)

        # ----------------------------------------------------
        # 3) WRINKLE BASELINE (AGE)
        # ----------------------------------------------------
        age = getattr(target_face, "age", 30)
        try:
            age_val = float(age)
        except:
            age_val = 30.0

        wrinkle_strength = compute_wrinkle_strength(age_val)

        if wrinkle_strength > 0:
            result = full_face_wrinkle(result, target_face, wrinkle_strength)

        # ----------------------------------------------------
        # 4) WRINKLE BOOSTER (EXPRESSION)
        # ----------------------------------------------------
        expression = detect_expression(target_face)
        result = apply_expression_wrinkle(
            result, target_face, expression, wrinkle_strength
        )

        # ----------------------------------------------------
        # 5) SMART WRINKLE MAP (crow’s feet, nasolabial, forehead, glabella)
        # ----------------------------------------------------
        result = apply_smart_wrinkle_map(
            result, target_face, expression, wrinkle_strength
        )

        # ----------------------------------------------------
        # 6) PERLIN MICRO NOISE
        # ----------------------------------------------------
        perlin_strength = getattr(roop.globals, "perlin_noise_strength", 0.07)
        try:
            perlin_strength = float(perlin_strength)
        except:
            perlin_strength = 0.07

        result = add_subtle_skin_noise(result, strength=perlin_strength)

        # paste
        temp_frame[y1:y2, x1:x2] = result
        return temp_frame

    except Exception as e:
        update_status(f"[Enhancer Error] {e}", NAME)
        return temp_frame


# ============================================================
# FRAME PROCESSORS
# ============================================================
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    faces = get_many_faces(temp_frame)
    if faces:
        for face in faces:
            temp_frame = enhance_face(face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for fp in temp_frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        out = process_frame(None, None, frame)
        cv2.imwrite(fp, out)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    frame = cv2.imread(target_path)
    out = process_frame(None, None, frame)
    cv2.imwrite(output_path, out)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(
        None, temp_frame_paths, process_frames
    )
