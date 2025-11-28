from typing import Any, List, Callable, Optional
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
NAME = 'ROOP.FACE-ENHANCER-ROBUST'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            # Pastikan path model benar
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # Initialize GFPGAN
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
    Hybrid Blending: Menggabungkan ketajaman GFPGAN dengan tekstur asli 
    untuk menghindari wajah terlihat seperti 'kartun/plastik'.
    """
    try:
        if enhanced_crop is None or original_crop is None:
            return original_crop

        h, w = original_crop.shape[:2]
        # Resize jika dimensi output GFPGAN beda sedikit
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))

        # 1. Color Matching (Anti-Belang)
        # Menyamakan tone warna enhanced dengan original
        enhanced_crop = enhanced_crop.astype(np.float32)
        original_crop = original_crop.astype(np.float32)

        original_mean = np.mean(original_crop, axis=(0, 1))
        enhanced_mean = np.mean(enhanced_crop, axis=(0, 1))
        color_diff = original_mean - enhanced_mean
        
        corrected_crop = enhanced_crop + color_diff
        corrected_crop = np.clip(corrected_crop, 0, 255).astype(np.uint8)
        original_crop = original_crop.astype(np.uint8)

        # 2. Fidelity Blending
        # fidelity 0.0 = full original, 1.0 = full enhanced
        # Mengembalikan tipe data ke uint8 untuk blending
        blended_expression = cv2.addWeighted(corrected_crop, fidelity, original_crop, 1.0 - fidelity, 0)

        # 3. Soft Masking (Anti-Kotak)
        # Membuat mask oval agar pinggiran kotak tidak terlihat tajam
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.45)) 
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Blur mask agar transisi halus
        blur_radius = int(min(w, h) * 0.1) 
        if blur_radius % 2 == 0: blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        mask_3ch = np.dstack([mask] * 3)

        # Compositing akhir
        final_result = (blended_expression * mask_3ch + original_crop * (1.0 - mask_3ch)).astype(np.uint8)
        
        return final_result
        
    except Exception as e:
        # Jika blending gagal, kembalikan original saja daripada crash
        return original_crop if original_crop is not None else enhanced_crop


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    try:
        start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
        
        # Padding sedikit agar GFPGAN menangkap konteks wajah
        padding_x = int((end_x - start_x) * 0.2)
        padding_y = int((end_y - start_y) * 0.2)
        
        h_frame, w_frame = temp_frame.shape[:2]
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(w_frame, end_x + padding_x)
        end_y = min(h_frame, end_y + padding_y)
        
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        
        if temp_face.size > 0:
            with THREAD_SEMAPHORE:
                # Panggil Model GFPGAN
                _, _, enhanced_face = get_face_enhancer().enhance(
                    temp_face,
                    paste_back=True
                )
            
            # Ambil nilai blend dari globals atau default 0.6 (60% enhanced, 40% original texture)
            blend_amount = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 0.6
            
            # Proses blending hybrid
            result_face = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount)
            
            # Tempel kembali ke frame utama
            temp_frame[start_y:end_y, start_x:end_x] = result_face
            
    except Exception:
        pass # Jika gagal enhance 1 wajah, biarkan wajah itu blur (jangan crash video)
        
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Optional[Frame]:
    # Safety Check
    if temp_frame is None or temp_frame.size == 0:
        return None

    try:
        # Deteksi wajah di frame ini
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = enhance_face(target_face, temp_frame)
        return temp_frame
    except Exception:
        return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        try:
            # 1. Baca Frame dengan aman
            temp_frame = cv2.imread(temp_frame_path)
            
            # 2. KRUSIAL: Cek apakah frame corrupt/None (Penyebab libpng error & crash sebelumnya)
            if temp_frame is None:
                # Jika file rusak, skip saja. Jangan diproses, jangan ditulis ulang.
                if update: update()
                continue

            # 3. Proses Enhance
            result = process_frame(None, None, temp_frame)

            # 4. Tulis Ulang HANYA jika hasil valid
            if result is not None and result.size > 0:
                cv2.imwrite(temp_frame_path, result)
            
        except Exception as e:
            # Catch-all agar satu frame error tidak mematikan seluruh render
            print(f"Skipping bad frame {temp_frame_path}: {e}")
            pass

        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is not None:
        result = process_frame(None, None, target_frame)
        if result is not None:
            cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
