# face-enhancer-v2.py
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
            # upscale=1 artinya kita hanya enhance kualitas wajah, bukan resize seluruh gambar
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

# --- OPTIMASI 2: Better Color Matching (LAB Space) ---
def match_color_lab(target_img, source_img):
    """
    Menyamakan tone warna enhanced_face (source) dengan original_face (target)
    menggunakan LAB color space agar lighting tetap natural.
    """
    try:
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        source_l, source_a, source_b = cv2.split(source_lab)
        target_l, target_a, target_b = cv2.split(target_lab)

        # Transfer statistik L (Lightness)
        source_l = (source_l - source_l.mean()) * (target_l.std() / (source_l.std() + 1e-5)) + target_l.mean()
        # Transfer statistik A & B (Color/Chrominance)
        source_a = (source_a - source_a.mean()) * (target_a.std() / (source_a.std() + 1e-5)) + target_a.mean()
        source_b = (source_b - source_b.mean()) * (target_b.std() / (source_b.std() + 1e-5)) + target_b.mean()

        result_lab = cv2.merge([source_l, source_a, source_b])
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return source_img

# --- OPTIMASI 3: Soft Blending (No More Ellipse) ---
def apply_advanced_blending(enhanced_crop: np.ndarray, original_crop: np.ndarray, blend_ratio: float) -> np.ndarray:
    """
    Mengganti 'apply_blend_and_color_match' lama.
    HAPUS cv2.ellipse agar rambut tidak terpotong.
    Gunakan 'Soft Box Mask' dan Lab Color Match.
    """
    h, w = original_crop.shape[:2]
    
    # 1. Resize jika perlu
    if enhanced_crop.shape[:2] != (h, w):
        enhanced_crop = cv2.resize(enhanced_crop, (w, h))

    # 2. Advanced Color Matching
    enhanced_crop_matched = match_color_lab(original_crop, enhanced_crop)

    # 3. Buat Mask "Soft Box" (Bukan Ellipse)
    # Mask ini putih di tengah, memudar lembut ke hitam di tepian (kotak).
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Margin 5-10% dari tepi agar transisi halus
    m_x = int(w * 0.05)
    m_y = int(h * 0.05)
    
    # Isi kotak tengah dengan putih
    mask[m_y:h-m_y, m_x:w-m_x] = 1.0
    
    # Blur mask dengan radius besar agar tidak ada garis kotak terlihat
    # Blur disesuaikan dengan ukuran wajah
    k_size = int(max(w, h) * 0.2) | 1 # Ganjil
    mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
    
    # Expand dimensi mask ke 3 channel
    mask_3ch = np.dstack([mask] * 3)

    # 4. Blending Akhir
    # Gabungkan enhanced dan original berdasarkan Mask DAN user blend_ratio
    # blend_ratio (fidelity): 1.0 = full original, 0.0 = full enhanced
    # Kita balik logikanya: user biasa pakai 0.0 - 1.0. 
    # Anggap blend_ratio adalah seberapa banyak 'enhanced' yang diinginkan.
    # Tapi code lama: blended = enhanced * fidelity + orig * (1-fidelity).
    
    # Mari kita blend enhanced yang sudah di-color-match dengan original
    # Menggunakan mask sebagai alpha channel
    final_face = (enhanced_crop_matched * mask_3ch + original_crop * (1.0 - mask_3ch))
    
    # Lalu blend global opacity (fidelity control)
    # Jika user ingin wajah lebih asli, kurangi impact final_face
    # Default blend biasanya 1.0 (full result) di logic ini
    return final_face.astype(np.uint8)

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    
    # Padding standar
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
        
        # Ambil blend amount (default 1.0 = full enhanced result with soft mask)
        # Di code lama, fidelity terbalik-balik. Kita set blending mask otomatis di atas.
        # User blend opsional:
        blend_amount = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 1.0
        
        result_face = apply_advanced_blending(enhanced_face, temp_face, blend_amount)
        
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
