import cv2
import numpy as np
import threading
import os
import onnxruntime as ort
from typing import List, Any

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.utilities import resolve_relative_path, is_image, is_video, conditional_download

# ---------------------------------------------------------------------
# 1. DEFINISI MODUL (WAJIB ADA)
# ---------------------------------------------------------------------
NAME = 'ROOP.FACE-ENHANCER'
MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' # Fallback URL (meski kita pakai ONNX)
# Ganti nama file model sesuai yang ada di folder models Anda
MODEL_NAME = 'GFPGANv1.4.onnx' 

THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore(2) # Limit GPU threads
ENHANCER_SESSION = None

# Konfigurasi Resolusi
MODEL_SIZE = 512
TEMPLATE_STD = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)
RATIO = MODEL_SIZE / 112.0
TEMPLATE_512 = TEMPLATE_STD * RATIO

# ---------------------------------------------------------------------
# 2. FUNGSI WAJIB (INTERFACE ROOP)
# Tanpa fungsi-fungsi ini, error "Not Implemented" muncul.
# ---------------------------------------------------------------------

def get_face_enhancer():
    global ENHANCER_SESSION
    with THREAD_LOCK:
        if ENHANCER_SESSION is None:
            model_path = resolve_relative_path(f'../models/{MODEL_NAME}')
            
            # Cek fallback ke FP16 jika ada
            fp16_path = resolve_relative_path('../models/GFPGANv1.4-fp16.onnx')
            if os.path.exists(fp16_path):
                model_path = fp16_path

            if not os.path.exists(model_path):
                print(f"Error: Model {MODEL_NAME} tidak ditemukan di folder models!")
                return None

            try:
                ENHANCER_SESSION = ort.InferenceSession(
                    model_path, 
                    providers=roop.globals.execution_providers
                )
            except Exception as e:
                print(f"Error loading ONNX: {e}")
    return ENHANCER_SESSION

def pre_check() -> bool:
    # Fungsi ini dipanggil saat start untuk cek apakah model ada
    download_directory_path = resolve_relative_path('../models')
    model_path = resolve_relative_path(f'../models/{MODEL_NAME}')
    fp16_path = resolve_relative_path('../models/GFPGANv1.4-fp16.onnx')

    if not os.path.exists(model_path) and not os.path.exists(fp16_path):
        print(f"Model {MODEL_NAME} hilang. Harap letakkan file ONNX di folder models.")
        return False
    return True

def pre_start() -> bool:
    # Fungsi ini dipanggil sebelum proses dimulai
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Pilih gambar atau video target.', NAME)
        return False
    return True

# ---------------------------------------------------------------------
# 3. LOGIKA UTAMA (CORE LOGIC)
# ---------------------------------------------------------------------

def create_soft_mask(size=512):
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask, (size // 2, size // 2), int(size // 2 * 0.9), (1.0), -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    return cv2.merge([mask, mask, mask])

SOFT_MASK = create_soft_mask(MODEL_SIZE)

def warp_face(frame, kps):
    try:
        M = cv2.estimateAffinePartial2D(np.array(kps), TEMPLATE_512, method=cv2.LMEDS)[0]
        warped = cv2.warpAffine(frame, M, (MODEL_SIZE, MODEL_SIZE), borderValue=0)
        return warped, M
    except:
        return None, None

def paste_back(frame, enhanced_face, M):
    try:
        IM = cv2.invertAffineTransform(M)
        pasted = cv2.warpAffine(enhanced_face, IM, (frame.shape[1], frame.shape[0]))
        mask_warped = cv2.warpAffine(SOFT_MASK, IM, (frame.shape[1], frame.shape[0]))
        return pasted, mask_warped
    except:
        return None, None

def match_color(target, source):
    target = target.astype(np.float32)
    source = source.astype(np.float32)
    matched = source.copy()
    for i in range(3):
        t_mean, t_std = np.mean(target[..., i]), np.std(target[..., i])
        s_mean, s_std = np.mean(source[..., i]), np.std(source[..., i])
        if s_std <= 1e-6: s_std = 1e-6
        matched[..., i] = (source[..., i] - s_mean) * (t_std / s_std) + t_mean
    return np.clip(matched, 0, 255).astype(np.uint8)

def normalize(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return np.transpose(img, (2, 0, 1))[None]

def denormalize(img):
    img = np.transpose(img[0], (1, 2, 0))
    img = (img + 1) * 0.5
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def enhance_one_face(frame, face):
    # 1. Warp
    warped_face, M = warp_face(frame, face['kps'])
    if warped_face is None: return frame
    
    # 2. Inference
    session = get_face_enhancer()
    if session is None: return frame

    inp = normalize(warped_face)
    with THREAD_SEMAPHORE:
        try:
            output = session.run(None, {"input": inp})[0]
        except Exception:
            return frame

    # 3. Post-Process
    enhanced_face = denormalize(output)
    enhanced_face = match_color(warped_face, enhanced_face)
    
    # 4. Internal Blending
    blended_face = (enhanced_face * SOFT_MASK + warped_face * (1 - SOFT_MASK)).astype(np.uint8)

    # 5. Paste Back
    pasted_face, mask_warped = paste_back(frame, blended_face, M)
    
    if pasted_face is not None and mask_warped is not None:
        frame = frame.astype(np.float32)
        pasted_face = pasted_face.astype(np.float32)
        frame = pasted_face * mask_warped + frame * (1.0 - mask_warped)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame

# ---------------------------------------------------------------------
# 4. FUNGSI EKSEKUSI UTAMA (PROCESS_FRAME)
# ---------------------------------------------------------------------

def process_frame(source_face, reference_face, frame):
    # Ini adalah fungsi yang dipanggil oleh Roop untuk setiap frame
    faces = get_many_faces(frame)
    if faces:
        for face in faces:
            frame = enhance_one_face(frame, face)
    return frame

# ---------------------------------------------------------------------
# 5. API VIDEO & IMAGE (WAJIB ADA)
# ---------------------------------------------------------------------

def process_frames(source_path: str, temp_frame_paths: List[str], update: Any) -> None:
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
