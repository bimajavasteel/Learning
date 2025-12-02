from typing import List, Optional
import cv2
import numpy as np
import random
from roop.typing import Face, Frame
import roop.globals

def create_wrinkle_texture(size: tuple, intensity: float = 0.3) -> np.ndarray:
    """
    Membuat tekstur kerutan menggunakan noise Perlin sederhana
    """
    h, w = size
    wrinkles = np.zeros((h, w), dtype=np.float32)
    
    # Multi-layer noise untuk kerutan alami
    for scale in [0.02, 0.05, 0.1, 0.2]:
        scale_h = int(h * scale)
        scale_w = int(w * scale)
        if scale_h < 2 or scale_w < 2:
            continue
            
        # Generate noise
        noise = np.random.randn(scale_h, scale_w).astype(np.float32)
        noise = cv2.resize(noise, (w, h))
        
        # Apply directional filters untuk kerutan
        kernel_size = max(1, int(min(h, w) * 0.01))
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), 
            4.0,  # sigma
            random.uniform(0, np.pi),  # theta
            10.0,  # lambda
            0.5,  # gamma
            0,  # psi
            ktype=cv2.CV_32F
        )
        
        filtered = cv2.filter2D(noise, -1, kernel)
        wrinkles += filtered * intensity * scale
    
    # Normalize
    wrinkles = (wrinkles - wrinkles.min()) / (wrinkles.max() - wrinkles.min() + 1e-7)
    return wrinkles

def add_wrinkles_to_face(
    face: Face,
    frame: Frame,
    intensity: float = 0.15,
    age_pattern: str = 'moderate'  # 'light', 'moderate', 'heavy'
) -> Frame:
    """
    Menambahkan kerutan pada area tertentu wajah
    """
    if intensity <= 0:
        return frame
    
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = map(int, face.bbox)
    
    # Pastikan bbox valid
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # Crop area wajah
    face_crop = frame[y1:y2, x1:x2].copy()
    h_crop, w_crop = face_crop.shape[:2]
    
    if h_crop < 10 or w_crop < 10:
        return frame
    
    # Pattern berdasarkan usia
    pattern_multipliers = {
        'light': 0.7,
        'moderate': 1.0,
        'heavy': 1.5
    }
    multiplier = pattern_multipliers.get(age_pattern, 1.0)
    
    # Mask untuk area kerutan
    wrinkle_mask = np.zeros((h_crop, w_crop), dtype=np.float32)
    
    # 1. Dahi (forehead) - kerutan horizontal
    forehead_h = int(h_crop * 0.3)
    forehead_y = int(h_crop * 0.1)
    
    forehead_wrinkles = create_wrinkle_texture((forehead_h, w_crop), intensity * 0.8 * multiplier)
    
    # Arah horizontal untuk dahi
    for i in range(forehead_h):
        shift = int(np.sin(i / max(1, forehead_h) * np.pi) * 2)
        wrinkle_mask[forehead_y + i, max(0, shift):min(w_crop, w_crop + shift)] += \
            forehead_wrinkles[i, max(0, -shift):min(w_crop, w_crop - shift)]
    
    # 2. Mata (crow's feet) - area sudut mata
    eye_corners = [
        (int(w_crop * 0.25), int(h_crop * 0.4)),  # kiri
        (int(w_crop * 0.75), int(h_crop * 0.4))   # kanan
    ]
    
    for cx, cy in eye_corners:
        radius = int(min(w_crop, h_crop) * 0.15)
        y_start = max(0, cy - radius)
        y_end = min(h_crop, cy + radius)
        x_start = max(0, cx - radius)
        x_end = min(w_crop, cx + radius)
        
        if y_end > y_start and x_end > x_start:
            size = (y_end - y_start, x_end - x_start)
            eye_wrinkles = create_wrinkle_texture(size, intensity * 1.2 * multiplier)
            
            # Pattern radial untuk crow's feet
            Y, X = np.ogrid[y_start:y_end, x_start:x_end]
            dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
            radial_mask = np.clip(1 - dist_from_center / radius, 0, 1)
            
            wrinkle_mask[y_start:y_end, x_start:x_end] += eye_wrinkles * radial_mask
    
    # 3. Bibir (smile lines) - nasolabial folds
    lip_start_y = int(h_crop * 0.7)
    lip_h = int(h_crop * 0.15)
    
    if lip_h > 0 and lip_start_y + lip_h <= h_crop:
        lip_wrinkles = create_wrinkle_texture((lip_h, w_crop), intensity * 0.6 * multiplier)
        
        # Pattern vertikal di samping hidung
        for col in range(w_crop):
            if col < w_crop * 0.4 or col > w_crop * 0.6:
                curve = np.sin(col / w_crop * np.pi * 2) * 3
                row_start = min(h_crop - 1, max(0, lip_start_y + int(curve)))
                row_end = min(h_crop, row_start + lip_h)
                
                if row_end > row_start:
                    wrinkle_mask[row_start:row_end, col] += lip_wrinkles[:row_end-row_start, col]
    
    # Normalize dan apply mask
    wrinkle_mask = np.clip(wrinkle_mask, 0, 1)
    
    # Warna kerutan (lebih gelap)
    wrinkle_color = np.array([-10, -5, 0], dtype=np.float32)  # BGR: sedikit lebih merah/kuning
    
    # Apply ke face crop
    for c in range(3):
        face_crop[:, :, c] = np.clip(
            face_crop[:, :, c].astype(np.float32) + 
            wrinkle_mask * wrinkle_color[c] * intensity * 50,
            0, 255
        ).astype(np.uint8)
    
    # Tambahkan texture detail
    texture = (wrinkle_mask * 20).astype(np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
    
    # Blend dengan original
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        frame[y1:y2, x1:x2], 0.7,
        face_crop, 0.3,
        0
    )
    
    return frame

def add_dark_circles(
    face: Face,
    frame: Frame,
    intensity: float = 0.3,
    color: tuple = (40, 30, 80)  # BGR: biru-kecoklatan
) -> Frame:
    """
    Menambahkan dark circles (lingkaran hitam) di bawah mata
    """
    if intensity <= 0:
        return frame
    
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = map(int, face.bbox)
    
    # Pastikan bbox valid
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    h_crop, w_crop = y2 - y1, x2 - x1
    
    # Mask untuk dark circles
    dark_mask = np.zeros((h_crop, w_crop), dtype=np.float32)
    
    # Posisi mata (estimasi)
    eye_y = int(h_crop * 0.35)  # posisi vertikal mata
    eye_height = int(h_crop * 0.1)
    
    # Mata kiri
    left_eye_x = int(w_crop * 0.3)
    left_eye_width = int(w_crop * 0.15)
    
    # Mata kanan
    right_eye_x = int(w_crop * 0.55)
    right_eye_width = int(w_crop * 0.15)
    
    eye_positions = [
        (left_eye_x, left_eye_width),
        (right_eye_x, right_eye_width)
    ]
    
    for eye_center_x, eye_width in eye_positions:
        # Area di bawah mata
        under_eye_y = eye_y + eye_height
        under_eye_height = int(h_crop * 0.1)
        
        # Buat elips untuk dark circle
        center = (eye_center_x, under_eye_y + under_eye_height // 2)
        axes = (eye_width // 2, under_eye_height // 2)
        
        # Gambar elips pada mask
        cv2.ellipse(
            dark_mask,
            center,
            axes,
            0, 0, 360,
            1.0,
            -1
        )
        
        # Soft edges dengan Gaussian blur
        blur_size = max(1, int(min(axes) * 0.5))
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Extract region untuk blur
        y_start = max(0, center[1] - axes[1] - blur_size)
        y_end = min(h_crop, center[1] + axes[1] + blur_size)
        x_start = max(0, center[0] - axes[0] - blur_size)
        x_end = min(w_crop, center[0] + axes[0] + blur_size)
        
        if y_end > y_start and x_end > x_start:
            region = dark_mask[y_start:y_end, x_start:x_end]
            region_blurred = cv2.GaussianBlur(region, (blur_size, blur_size), axes[0] * 0.3)
            dark_mask[y_start:y_end, x_start:x_end] = region_blurred
    
    # Normalize mask
    dark_mask = np.clip(dark_mask, 0, 1) * intensity
    
    # Warna dark circles (BGR)
    dark_color = np.array(color, dtype=np.float32)
    
    # Extract face region
    face_region = frame[y1:y2, x1:x2].copy()
    
    # Apply dark circles dengan blending
    for c in range(3):
        face_region[:, :, c] = np.clip(
            face_region[:, :, c].astype(np.float32) * (1 - dark_mask * 0.3) +
            dark_color[c] * dark_mask * 0.7,
            0, 255
        ).astype(np.uint8)
    
    # Blend kembali ke frame
    frame[y1:y2, x1:x2] = cv2.addWeighted(
        frame[y1:y2, x1:x2], 0.5,
        face_region, 0.5,
        0
    )
    
    return frame

def apply_aging_effects(
    face: Face,
    frame: Frame,
    wrinkles_intensity: float = 0.0,
    dark_circles_intensity: float = 0.0,
    age_pattern: str = 'moderate'
) -> Frame:
    """
    Fungsi utama untuk apply efek penuaan
    """
    result = frame.copy()
    
    # Apply wrinkles jika intensity > 0
    if wrinkles_intensity > 0:
        result = add_wrinkles_to_face(face, result, wrinkles_intensity, age_pattern)
    
    # Apply dark circles jika intensity > 0
    if dark_circles_intensity > 0:
        result = add_dark_circles(face, result, dark_circles_intensity)
    
    return result
