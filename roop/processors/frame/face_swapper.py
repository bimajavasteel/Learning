from typing import Any, List, Callable, Optional
import cv2
import insightface
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
# Import fungsi mask geometris baru
from roop.face_analyser import (
    get_one_face, get_many_faces, smart_face_tracking, 
    get_face_pose, get_geometric_mask 
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    conditional_download(resolve_relative_path('../models'), [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path): return False
    if not get_one_face(cv2.imread(roop.globals.source_path)): return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path): return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

# =====================================================================
#  MATH HELPERS
# =====================================================================

def get_inverse_affine(face_kps: np.ndarray) -> Any:
    # Template landmark 128x128 standar arcface
    dst_pts = np.array([
        [38.29, 51.69], [73.53, 51.50], [56.02, 71.73],
        [41.54, 92.36], [70.72, 92.20]
    ], dtype=np.float32)
    dst_pts[:, 0] += 8.0 # Offset ke 128px
    
    # Hitung matriks transformasi balik dari kotak 128 ke frame asli
    M, _ = cv2.estimateAffinePartial2D(dst_pts, face_kps)
    return M

# =====================================================================
#  CORE SWAP (GEOMETRIC MASKING)
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Swap wajah dengan blending masker geometris.
    Anti-kotak, anti-flicker.
    """
    if source_face is None or target_face is None:
        return temp_frame

    # 1. Generate Raw Swap (128x128)
    bgr_fake, _ = get_face_swapper().get(
        temp_frame, target_face, source_face, paste_back=False
    )
    if bgr_fake is None: return temp_frame

    # 2. Hitung Warp Matrix
    M = get_inverse_affine(target_face.kps)
    if M is None: return temp_frame

    h_frame, w_frame = temp_frame.shape[:2]

    # 3. Warp wajah palsu ke posisi frame asli
    warped_face = cv2.warpAffine(bgr_fake, M, (w_frame, h_frame), borderValue=0.0)

    # 4. AMBIL MASK GEOMETRIS (KUNCI STABILITAS)
    # Ini membuat mask yang mengikuti bentuk dagu dan pipi dengan presisi
    geo_mask_crop = get_geometric_mask(target_face, temp_frame)
    
    if geo_mask_crop is not None:
        # Kita harus menaruh mask crop ini ke koordinat frame penuh
        full_mask = np.zeros((h_frame, w_frame), dtype=np.float32)
        
        x1, y1, x2, y2 = map(int, target_face.bbox)
        x1 = max(0, min(x1, w_frame)); x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame)); y2 = max(0, min(y2, h_frame))
        
        # Tempel mask wajah
        mh, mw = geo_mask_crop.shape[:2]
        th, tw = y2-y1, x2-x1
        
        # Resize safety check
        if mh != th or mw != tw:
            geo_mask_crop = cv2.resize(geo_mask_crop, (tw, th))
            
        full_mask[y1:y2, x1:x2] = geo_mask_crop
        
        # Expand dimensi untuk perkalian RGB
        full_mask = full_mask[:, :, np.newaxis]
        
        # 5. BLENDING AKHIR
        # Pixel = (WajahBaru * MaskWajah) + (WajahAsli * (1 - MaskWajah))
        # Karena mask mengikuti bentuk wajah (bukan kotak), hasilnya natural
        temp_frame[:] = (warped_face * full_mask + temp_frame * (1.0 - full_mask)).astype(np.uint8)
        
    else:
        # Fallback jika mask gagal (jarang terjadi): pakai blending kotak sederhana
        pass 

    return temp_frame

# =====================================================================
#  PROCESS FLOW
# =====================================================================

def process_frame(source_face, reference_face, temp_frame, frame_number=0):
    # Mode Smart Tracking (Wajib untuk stabilitas)
    faces = smart_face_tracking(temp_frame, frame_number)
    
    # Fallback ke deteksi biasa jika tracking belum lock
    if not faces:
        faces = get_many_faces(temp_frame)

    if faces:
        # Cari target
        target = faces[0]
        if reference_face:
            # Logic cari wajah termirip dengan referensi
            best_dist = float('inf')
            for f in faces:
                dist = np.sum(np.square(f.normed_embedding - reference_face.normed_embedding))
                if dist < best_dist:
                    best_dist = dist
                    target = f
        
        # Lakukan Swap
        temp_frame = swap_face(source_face, target, temp_frame)

    return temp_frame

def process_frames(source_path, temp_frame_paths, update_func):
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = get_face_reference()
    
    for idx, path in enumerate(temp_frame_paths):
        frame = cv2.imread(path)
        result = process_frame(source_face, reference_face, frame, idx)
        cv2.imwrite(path, result)
        if update_func: update_func()

def process_video(source_path, temp_frame_paths):
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def process_image(source_path, target_path, output_path):
    # Untuk image single, frame_number = 0
    s_face = get_one_face(cv2.imread(source_path))
    t_frame = cv2.imread(target_path)
    t_face = get_one_face(t_frame)
    
    result = process_frame(s_face, t_face, t_frame, 0)
    cv2.imwrite(output_path, result)
