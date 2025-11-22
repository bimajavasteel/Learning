import cv2
import numpy as np
import threading
import os
import onnxruntime as ort
from typing import List, Any

import roop.globals
from roop.face_analyser import get_many_faces
from roop.utilities import resolve_relative_path, is_video, is_image

# ---------------------------------------------------------------------
# KONFIGURASI & KONSTANTA
# ---------------------------------------------------------------------
NAME = "ROOP.FACE-ENHANCER-PRO"
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore(2) # Batasi max 2 thread GPU agar tidak OOM
ENHANCER_SESSION = None

# Template Landmark ArcFace standar (untuk resolusi 112x112)
# Kita akan menskalakannya secara dinamis nanti.
TEMPLATE_STD = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Target Resolusi Model (GFPGAN v1.4 standar adalah 512px)
MODEL_SIZE = 512 

# Kalkulasi Template untuk 512px
# Ratio = 512 / 112 = 4.5714...
RATIO = MODEL_SIZE / 112.0
TEMPLATE_512 = TEMPLATE_STD * RATIO

# ---------------------------------------------------------------------
# 1. MASKER & BLENDING (Mencegah Kotak)
# ---------------------------------------------------------------------
def create_soft_mask(size=512):
    """Membuat masker lingkaran dengan pinggiran blur (feathered)"""
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    # Radius sedikit lebih kecil dari frame agar pinggiran aman
    radius = int(size // 2 * 0.9) 
    
    cv2.circle(mask, center, radius, (1.0), -1)
    
    # Blur yang kuat agar transisi halus
    mask = cv2.GaussianBlur(mask, (61, 61), 0)
    
    # Expand dimemsi jadi (512, 512, 3)
    return cv2.merge([mask, mask, mask])

# Inisialisasi Masker Global
SOFT_MASK = create_soft_mask(MODEL_SIZE)

# ---------------------------------------------------------------------
# 2. ALIGNMENT & WARPING (Kunci Ketajaman)
# ---------------------------------------------------------------------
def warp_face(frame, kps):
    """Memutar & Meluruskan wajah ke 512x512"""
    try:
        # Hitung Matriks Affine dari titik wajah ke Template 512
        M = cv2.estimateAffinePartial2D(np.array(kps), TEMPLATE_512, method=cv2.LMEDS)[0]
        
        # Warp (Potong & Putar)
        warped = cv2.warpAffine(frame, M, (MODEL_SIZE, MODEL_SIZE), borderValue=0)
        return warped, M
    except Exception:
        return None, None

def paste_back(frame, enhanced_face, M):
    """Mengembalikan wajah 512 ke posisi asli di frame"""
    try:
        # Balikkan matriks (Inverse)
        IM = cv2.invertAffineTransform(M)
        
        # Tempel balik ke resolusi asli frame
        pasted = cv2.warpAffine(enhanced_face, IM, (frame.shape[1], frame.shape[0]))
        
        # Kita juga perlu masker invers untuk blending di frame utama
        # (Opsional: bisa pakai seamlessClone, tapi warp mask lebih cepat)
        mask_warped = cv2.warpAffine(SOFT_MASK, IM, (frame.shape[1], frame.shape[0]))
        
        return pasted, mask_warped
    except:
        return None, None

# ---------------------------------------------------------------------
# 3. COLOR CORRECTION (Agar warna kulit nyatu)
# ---------------------------------------------------------------------
def match_color(target, source):
    """Menyamakan tone warna hasil AI (source) ke wajah asli (target)"""
    target = target.astype(np.float32)
    source = source.astype(np.float32)
    
    matched = source.copy()
    for i in range(3): # Loop RGB/BGR
        t_mean = np.mean(target[..., i])
        t_std  = np.std(target[..., i])
        s_mean = np.mean(source[..., i])
        s_std  = np.std(source[..., i])
        
        # Rumus transfer warna statistik
        if s_std <= 1e-6: s_std = 1e-6 # Cegah bagi nol
        matched[..., i] = (source[..., i] - s_mean) * (t_std / s_std) + t_mean
        
    return np.clip(matched, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------
# 4. CORE AI ENGINE (ONNX)
# ---------------------------------------------------------------------
def get_enhancer_session():
    global ENHANCER_SESSION
    with THREAD_LOCK:
        if ENHANCER_SESSION is None:
            base_dir = resolve_relative_path("../models")
            # Prioritaskan model FP16 (lebih cepat)
            model_path = os.path.join(base_dir, "GFPGANv1.4.onnx") 
            
            if not os.path.exists(model_path):
                print(f"[{NAME}] ERROR: Model tidak ditemukan di {model_path}")
                return None
                
            print(f"[{NAME}] Memuat Model: {model_path}")
            ENHANCER_SESSION = ort.InferenceSession(
                model_path, 
                providers=roop.globals.execution_providers
            )
    return ENHANCER_SESSION

def normalize(img):
    # 0-255 (HWC) -> 0-1 (CHW)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5 # GFPGAN biasanya butuh range -1 s/d 1
    return np.transpose(img, (2, 0, 1))[None]

def denormalize(img):
    # -1 s/d 1 -> 0-255
    img = np.transpose(img[0], (1, 2, 0))
    img = (img + 1) * 0.5
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------
# 5. FUNGSI UTAMA
# ---------------------------------------------------------------------
def enhance_one_face(frame, face):
    # A. Warp Wajah
    warped_face, M = warp_face(frame, face['kps'])
    if warped_face is None: 
        return frame
    
    # B. Inference AI
    session = get_enhancer_session()
    if session is None: 
        return frame

    inp = normalize(warped_face)
    
    with THREAD_SEMAPHORE:
        try:
            # Jalankan model
            output = session.run(None, {"input": inp})[0]
        except Exception as e:
            print(f"[{NAME}] Gagal proses AI: {e}")
            return frame

    # C. Post-Processing
    enhanced_face = denormalize(output)
    
    # D. Samakan Warna (PENTING)
    enhanced_face = match_color(warped_face, enhanced_face)
    
    # E. Blending Internal (Wajah HD vs Wajah Blur)
    # Mencampur hasil enhanced dengan warped asli pakai soft mask
    # agar pinggiran wajah tetap natural
    blended_face = (enhanced_face * SOFT_MASK + warped_face * (1 - SOFT_MASK)).astype(np.uint8)

    # F. Tempel Balik
    pasted_face, mask_warped = paste_back(frame, blended_face, M)
    
    if pasted_face is not None and mask_warped is not None:
        # Blending final ke frame utama
        # Hanya timpa pixel jika mask > 0
        # Rumus: Frame = (Pasted * Mask) + (FrameAsli * (1-Mask))
        mask_alpha = mask_warped # Range 0.0 - 1.0
        
        # Optimasi numpy agar cepat
        frame = frame.astype(np.float32)
        pasted_face = pasted_face.astype(np.float32)
        
        frame = pasted_face * mask_alpha + frame * (1.0 - mask_alpha)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame

def process_frame(source_face, reference_face, frame):
    # 1. Cari semua wajah di frame
    faces = get_many_faces(frame)
    
    if faces:
        for face in faces:
            # 2. Enhance setiap wajah satu per satu
            frame = enhance_one_face(frame, face)
            
    return frame

# ---------------------------------------------------------------------
# API STANDARD ROOP
# ---------------------------------------------------------------------
def process_image(source_path, target_path, output_path):
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path, temp_frame_paths):
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames_video)

def process_frames_video(source_path, temp_frame_paths, update):
    for path in temp_frame_paths:
        frame = cv2.imread(path)
        frame = process_frame(None, None, frame)
        cv2.imwrite(path, frame)
        if update: update()
