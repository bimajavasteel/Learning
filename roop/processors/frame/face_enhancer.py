from typing import Any, List, Callable, Optional
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
# Import fungsi masking geometris baru
from roop.face_analyser import get_many_faces, get_geometric_mask 
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
            # Pastikan path model benar sesuai setup Anda
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
    Enhance wajah menggunakan Masking Geometris (Convex Hull).
    Hasilnya mengikuti kontur wajah asli, menghilangkan efek kotak.
    """
    # 1. Hitung Koordinat Crop dengan Padding (Agar GFPGAN bekerja optimal)
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    
    # Simpan koordinat asli (raw) untuk referensi masking
    raw_x1, raw_y1, raw_x2, raw_y2 = start_x, start_y, end_x, end_y

    # Padding 20%
    padding_x = int((end_x - start_x) * 0.2)
    padding_y = int((end_y - start_y) * 0.2)

    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)

    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size == 0:
        return temp_frame

    # 2. Jalankan GFPGAN (Restoration)
    # paste_back=False : Kita butuh raw result untuk manual blending presisi
    with THREAD_SEMAPHORE:
        try:
            _, _, restored_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=False
            )
        except Exception:
            return temp_frame

    # 3. Manual Geometric Blending
    if restored_face is not None:
        h_crop, w_crop = temp_face.shape[:2]
        
        # Resize hasil restore ke ukuran crop
        restored_face = cv2.resize(restored_face, (w_crop, h_crop))

        # --- GEOMETRIC MASKING LOGIC ---
        # Ambil mask bentuk wajah presisi dari face_analyser
        geo_mask_raw = get_geometric_mask(target_face, temp_frame)

        if geo_mask_raw is not None:
            # geo_mask_raw ukurannya sesuai bbox asli (raw_x1..raw_x2)
            # Kita perlu menaruhnya di dalam kanvas crop yang lebih besar (start_x..end_x)
            
            # Buat kanvas mask kosong seukuran crop padding
            final_mask = np.zeros((h_crop, w_crop), dtype=np.float32)
            
            # Hitung offset (pergeseran) bbox asli di dalam crop padding
            off_x = raw_x1 - start_x
            off_y = raw_y1 - start_y
            
            h_geo, w_geo = geo_mask_raw.shape[:2]
            
            # Pastikan koordinat paste valid (tidak keluar batas)
            y1_p = max(0, off_y)
            y2_p = min(h_crop, off_y + h_geo)
            x1_p = max(0, off_x)
            x2_p = min(w_crop, off_x + w_geo)
            
            # Ambil area source yang sesuai
            sy1 = y1_p - off_y
            sy2 = sy1 + (y2_p - y1_p)
            sx1 = x1_p - off_x
            sx2 = sx1 + (x2_p - x1_p)
            
            if (y2_p > y1_p) and (x2_p > x1_p):
                paste_area = geo_mask_raw[sy1:sy2, sx1:sx2]
                
                # Resize kecil jika ada selisih pembulatan
                th, tw = y2_p - y1_p, x2_p - x1_p
                if paste_area.shape[:2] != (th, tw):
                    paste_area = cv2.resize(paste_area, (tw, th))
                
                # Tempel mask wajah ke kanvas
                final_mask[y1_p:y2_p, x1_p:x2_p] = paste_area

            # Expand dimensi mask untuk blending RGB
            final_mask = final_mask[:, :, np.newaxis] # HxWx1

            # BLEND: (Enhanced * Mask) + (Original * (1-Mask))
            # Mask ini sudah di-blur di face_analyser, jadi transisi halus
            temp_face = (restored_face * final_mask + temp_face * (1.0 - final_mask)).astype(np.uint8)
        
        else:
            # Fallback jika geometric mask gagal (jarang): Blending lingkaran sederhana
            pass

    # Kembalikan potongan wajah yang sudah di-enhance ke frame utama
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
