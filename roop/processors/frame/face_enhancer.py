from typing import Any, List, Callable, Optional
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
# Import face_analyser untuk akses occlusion mask
import roop.face_analyser 
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
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
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


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhance wajah dengan Soft Blending & Occlusion Masking.
    """
    # 1. Hitung Koordinat Crop dengan Padding
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    
    # Simpan koordinat asli (raw) untuk referensi mask occlusion
    raw_x1, raw_y1, raw_x2, raw_y2 = start_x, start_y, end_x, end_y

    padding_x = int((end_x - start_x) * 0.2)
    padding_y = int((end_y - start_y) * 0.2)

    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)

    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size == 0:
        return temp_frame

    # 2. Jalankan GFPGAN
    # paste_back=False : Kita butuh raw result untuk manual blending
    with THREAD_SEMAPHORE:
        try:
            _, _, restored_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=False
            )
        except Exception:
            # Fallback jika model gagal
            return temp_frame

    # 3. Manual Blending (Soft Edge + Anti-Occlusion)
    if restored_face is not None:
        h_crop, w_crop = temp_face.shape[:2]
        
        # Resize hasil restore ke ukuran crop asli
        restored_face = cv2.resize(restored_face, (w_crop, h_crop))

        # A. Buat Base Mask (Lingkaran/Elips Halus)
        # Ini mengatasi masalah "Kotak Samar" di pinggiran
        base_mask = np.zeros((h_crop, w_crop), dtype=np.float32)
        center = (w_crop // 2, h_crop // 2)
        axes = (int(w_crop * 0.45), int(h_crop * 0.45))
        angle = 0
        
        cv2.ellipse(base_mask, center, axes, angle, 0, 360, 1.0, -1)
        base_mask = cv2.GaussianBlur(base_mask, (51, 51), 0)

        # B. Ambil Mask Occlusion (Tangan/Objek)
        final_mask = base_mask
        
        # Cek apakah modul face_analyser punya fungsi yang kita buat
        if hasattr(roop.face_analyser, 'get_occlusion_mask'):
            raw_occ_mask = roop.face_analyser.get_occlusion_mask(target_face, temp_frame)
            
            if raw_occ_mask is not None:
                # raw_occ_mask seukuran bbox asli (raw_x1...raw_x2)
                # Kita perlu menaruhnya di dalam crop yang sudah di-padding (start_x...end_x)
                
                full_crop_occ = np.zeros((h_crop, w_crop), dtype=np.float32)
                
                # Hitung offset posisi bbox asli di dalam crop padding
                off_x = raw_x1 - start_x
                off_y = raw_y1 - start_y
                
                h_occ, w_occ = raw_occ_mask.shape[:2]
                
                # Pastikan tidak out of bounds
                y1_paste = max(0, off_y)
                y2_paste = min(h_crop, off_y + h_occ)
                x1_paste = max(0, off_x)
                x2_paste = min(w_crop, off_x + w_occ)
                
                # Hitung area source yang sesuai
                sy1 = y1_paste - off_y
                sy2 = sy1 + (y2_paste - y1_paste)
                sx1 = x1_paste - off_x
                sx2 = sx1 + (x2_paste - x1_paste)

                if (y2_paste > y1_paste) and (x2_paste > x1_paste):
                    paste_area = raw_occ_mask[sy1:sy2, sx1:sx2]
                    # Resize kecil jika ada selisih pembulatan 1-2 pixel
                    target_h = y2_paste - y1_paste
                    target_w = x2_paste - x1_paste
                    
                    if paste_area.shape[:2] != (target_h, target_w):
                         paste_area = cv2.resize(paste_area, (target_w, target_h))

                    full_crop_occ[y1_paste:y2_paste, x1_paste:x2_paste] = paste_area
                
                # Blur mask tangan sedikit
                full_crop_occ = cv2.GaussianBlur(full_crop_occ, (15, 15), 0)
                
                # LOGIC: Mask Final = Base Mask * (1 - Mask Tangan)
                # Artinya: Enhance area lingkaran KECUALI yang ada tangannya
                final_mask = base_mask * (1.0 - full_crop_occ)

        # C. Apply Blending
        final_mask = np.clip(final_mask, 0.0, 1.0)
        final_mask = final_mask[:, :, np.newaxis] # HxWx1 channel

        # Blend: (Enhanced * Mask) + (Original * (1-Mask))
        temp_face = (restored_face * final_mask + temp_face * (1.0 - final_mask)).astype(np.uint8)

    # Kembalikan ke frame utama
    temp_frame[start_y:end_y, start_x:end_x] = temp_face
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
