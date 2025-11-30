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
from roop.processors.frame.wrinkle_enhancer_v2 import enhance_wrinkles_after_gfpgan

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


def apply_blend_and_color_match(enhanced_crop: np.ndarray, original_crop: np.ndarray, fidelity: float) -> np.ndarray:
    """
    Menggabungkan hasil enhance dengan frame asli menggunakan Fidelity Blending,
    Color Matching (Anti-Flicker), dan Masking (Anti-Box/Occlusion).
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

        # 3. Fidelity Blending (Menjaga Mimik Wajah)
        blended_expression = cv2.addWeighted(corrected_crop, fidelity, original_crop, 1.0 - fidelity, 0)

        # 4. Masking (Occlusion & Box Removal)
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.45)) 
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        blur_radius = int(min(w, h) * 0.1) 
        if blur_radius % 2 == 0: blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        mask_3ch = np.dstack([mask] * 3)

        # 5. Final Compositing
        final_result = (blended_expression * mask_3ch + original_crop * (1.0 - mask_3ch)).astype(np.uint8)
        
        return final_result
        
    except Exception as e:
        update_status(f"Error in blending: {e}", NAME)
        return original_crop


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.2)
    padding_y = int((end_y - start_y) * 0.2)
    
    h_frame, w_frame = temp_frame.shape[:2]
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(w_frame, end_x + padding_x)
    end_y = min(h_frame, end_y + padding_y)
    
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, enhanced_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        
        # 📌 AMBIL NILAI BLEND DARI GLOBAL (0.6 default jika CLI tidak diisi)
        blend_amount = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 0.6
        
        result_face = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount)
        from roop.processors.frame.wrinkle_enhancer_v2 import enhance_wrinkles_after_gfpgan

result_face = enhance_wrinkles_after_gfpgan(result_face, target_face)

        
        temp_frame[start_y:end_y, start_x:end_x] = result_face
        
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
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
