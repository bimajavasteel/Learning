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
NAME = 'ROOP.FACE-ENHANCER'


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
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


# =====================================================================
#  PRE-UPSCALE HOOK (UNTUK REAL-ESRGAN / DLL)
# =====================================================================

def apply_pre_upscale(crop: np.ndarray) -> np.ndarray:
    """
    🔧 OPTIMASI H:
    Hook opsional untuk upscaler eksternal (misalnya RealESRGAN) SEBELUM GFPGAN.
    Di Kaggle, kamu bisa set:
        roop.globals.pre_upscale_fn = lambda img: <fungsi upscaling kamu>
    Kalau tidak diset / error → crop dikembalikan apa adanya.
    """
    pre_fn = getattr(roop.globals, "pre_upscale_fn", None)
    if callable(pre_fn):
        try:
            return pre_fn(crop)
        except Exception as e:
            update_status(f"Pre-upscale error: {e}", NAME)
            return crop
    return crop


# =====================================================================
#  BLEND + COLOR MATCH + DETAIL MASK
# =====================================================================

def apply_blend_and_color_match(enhanced_crop: np.ndarray, original_crop: np.ndarray, fidelity: float) -> np.ndarray:
    """
    Menggabungkan hasil enhance dengan frame asli menggunakan:
    - Fidelity Blending
    - Color Matching (Anti-Flicker)
    - Masking (Anti-Box/Occlusion)
    - 🔧 OPTIMASI G: Sharpen Guard (anti over-sharp GFPGAN)
    - 🔧 OPTIMASI I: Detail Core Mask (bagian tengah wajah lebih dipertahankan)
    """
    try:
        # 1. Validasi Dimensi
        h, w = original_crop.shape[:2]
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))

        # 2. Color Matching (Anti-Flicker)
        original_mean = np.mean(original_crop, axis=(0, 1))
        enhanced_mean = np.mean(enhanced_crop, axis=(0, 1))
        color_diff = original_mean - enhanced_mean

        corrected_crop = enhanced_crop.astype(np.float32) + color_diff
        corrected_crop = np.clip(corrected_crop, 0, 255).astype(np.uint8)

        # 3. 🔧 OPTIMASI G: Sharpen Guard (GFPGAN kadang terlalu tajam)
        lap = cv2.Laplacian(corrected_crop, cv2.CV_32F)
        sharpness = float(lap.var())
        max_sharp = getattr(roop.globals, "face_enhancer_max_sharpness", 1500.0)
        if sharpness > max_sharp:
            corrected_crop = cv2.GaussianBlur(corrected_crop, (3, 3), 0)

        # 4. Fidelity Blending (Menjaga Mimik Wajah)
        fidelity = float(fidelity)
        fidelity = max(0.0, min(1.0, fidelity))
        blended_expression = cv2.addWeighted(corrected_crop, fidelity, original_crop, 1.0 - fidelity, 0)

        # 5. Masking (Occlusion & Box Removal) + Detail Core Mask
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Inner core mask untuk fitur detail (mata, hidung, mulut) – OPTIMASI I
        inner_mask = np.zeros((h, w), dtype=np.float32)
        inner_axes = (int(w * 0.25), int(h * 0.25))
        cv2.ellipse(inner_mask, center, inner_axes, 0, 0, 360, 1.0, -1)

        blur_radius = int(min(w, h) * 0.10)
        if blur_radius % 2 == 0:
            blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        inner_blur = int(min(w, h) * 0.06)
        if inner_blur % 2 == 0:
            inner_blur += 1
        inner_mask = cv2.GaussianBlur(inner_mask, (inner_blur, inner_blur), 0)

        # gabungkan: core lebih dominan (detail), pinggir lebih halus
        combined_mask = np.clip(0.7 * mask + 0.3 * inner_mask, 0.0, 1.0)
        mask_3ch = np.dstack([combined_mask] * 3)

        # 6. Final Compositing
        final_result = (blended_expression.astype(np.float32) * mask_3ch +
                        original_crop.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

        return final_result

    except Exception as e:
        update_status(f"Error in blending: {e}", NAME)
        return original_crop


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhance 1 wajah + padding.
    Support:
    - target_face dari InsightFace (Face) maupun dict dengan key 'bbox'
    """
    # Support kedua tipe: Face object dan dict
    if hasattr(target_face, "bbox"):
        bbox = target_face.bbox
    else:
        bbox = target_face['bbox']

    start_x, start_y, end_x, end_y = map(int, bbox)
    padding_x = int((end_x - start_x) * 0.2)
    padding_y = int((end_y - start_y) * 0.2)

    h_frame, w_frame = temp_frame.shape[:2]
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(w_frame, end_x + padding_x)
    end_y = min(h_frame, end_y + padding_y)

    temp_face = temp_frame[start_y:end_y, start_x:end_x]

    if temp_face.size:
        # 🔧 OPTIMASI H: pre-upscale hook (misalnya RealESRGAN)
        temp_face_for_enhance = apply_pre_upscale(temp_face)

        with THREAD_SEMAPHORE:
            _, _, enhanced_face = get_face_enhancer().enhance(
                temp_face_for_enhance,
                paste_back=True
            )

        # 📌 AMBIL NILAI BLEND DARI GLOBAL (0.6 default jika CLI tidak diisi)
        blend_amount = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 0.6

        result_face = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount)

        temp_frame[start_y:end_y, start_x:end_x] = result_face

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Proses semua frame secara berurutan.
    (Parallelism frame-level biasanya diatur di roop.processors.frame.core)
    """
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
