from typing import Any, List, Callable
import cv2
import numpy as np
import threading
import random

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces, get_one_face
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_AGER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.AGING-EFFECTS'


def get_face_ager() -> Any:
    """
    Inisialisasi aging effects handler.
    Tidak butuh model, hanya inisialisasi state.
    """
    global FACE_AGER
    with THREAD_LOCK:
        if FACE_AGER is None:
            FACE_AGER = AgingEffects()
    return FACE_AGER


def clear_face_ager() -> None:
    """
    Reset aging effects handler.
    """
    global FACE_AGER
    FACE_AGER = None


class AgingEffects:
    """
    Class untuk mengelola efek penuaan.
    """
    def __init__(self):
        self.params = {
            'wrinkle_intensity': roop.globals.wrinkle_intensity,
            'dark_circle_intensity': roop.globals.dark_circle_intensity,
            'apply_to_all_faces': roop.globals.apply_aging_to_all_faces,
            'wrinkle_color': (80, 80, 100),  # Warna kerutan (BGR)
            'dark_circle_color': (40, 40, 80),  # Warna dark circle (BGR)
        }


def pre_check() -> bool:
    """
    Validasi pre-check.
    """
    return True


def pre_start() -> bool:
    """
    Validasi sebelum mulai.
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    """
    Bersihkan setelah selesai.
    """
    clear_face_ager()


def create_wrinkle_texture(size, intensity=0.3):
    """
    Buat tekstur kerutan procedural.
    """
    h, w = size
    wrinkles = np.zeros((h, w), dtype=np.float32)
    
    # Tambahkan noise Perlin sederhana untuk tekstur kerutan
    for scale in [4, 8, 16, 32]:
        grid_h = h // scale + 1
        grid_w = w // scale + 1
        
        # Generate random grid
        random_grid = np.random.randn(grid_h, grid_w).astype(np.float32)
        
        # Upsample grid menggunakan OpenCV resize
        upsampled = cv2.resize(random_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Scale down with frequency
        wrinkles += upsampled * (1.0 / scale)
    
    # Normalize dan apply intensity
    wrinkles = (wrinkles - wrinkles.min()) / (wrinkles.max() - wrinkles.min() + 1e-7)
    wrinkles = np.clip(wrinkles * intensity, 0, 1)
    
    return wrinkles


def add_dark_circles(face_region, landmarks, intensity=0.4, color=(40, 40, 80)):
    """
    Tambahkan dark circles di bawah mata.
    """
    if face_region is None or face_region.size == 0:
        return face_region
    
    h, w = face_region.shape[:2]
    
    # Buat mask untuk dark circles
    dark_circle_mask = np.zeros((h, w), dtype=np.float32)
    
    if landmarks is not None and len(landmarks) >= 68:
        # Indeks untuk area bawah mata (dalam 68-point landmarks)
        # Kiri: [36, 37, 38, 39, 40, 41]
        # Kanan: [42, 43, 44, 45, 46, 47]
        
        # Untuk mata kiri
        left_eye_points = []
        for idx in [36, 37, 38, 39, 40, 41]:
            if idx < len(landmarks):
                x = int((landmarks[idx][0] - landmarks[0][0]) * w / 
                       (landmarks[16][0] - landmarks[0][0]))
                y = int((landmarks[idx][1] - landmarks[19][1]) * h / 
                       (landmarks[8][1] - landmarks[19][1]))
                left_eye_points.append((x, y))
        
        if len(left_eye_points) >= 4:
            # Buat ellipse untuk area bawah mata kiri
            pts = np.array(left_eye_points, np.int32)
            x, y, w_rect, h_rect = cv2.boundingRect(pts)
            center = (x + w_rect // 2, y + h_rect // 2 + h_rect // 3)
            axes = (int(w_rect * 0.6), int(h_rect * 0.3))
            cv2.ellipse(dark_circle_mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Untuk mata kanan
        right_eye_points = []
        for idx in [42, 43, 44, 45, 46, 47]:
            if idx < len(landmarks):
                x = int((landmarks[idx][0] - landmarks[0][0]) * w / 
                       (landmarks[16][0] - landmarks[0][0]))
                y = int((landmarks[idx][1] - landmarks[19][1]) * h / 
                       (landmarks[8][1] - landmarks[19][1]))
                right_eye_points.append((x, y))
        
        if len(right_eye_points) >= 4:
            # Buat ellipse untuk area bawah mata kanan
            pts = np.array(right_eye_points, np.int32)
            x, y, w_rect, h_rect = cv2.boundingRect(pts)
            center = (x + w_rect // 2, y + h_rect // 2 + h_rect // 3)
            axes = (int(w_rect * 0.6), int(h_rect * 0.3))
            cv2.ellipse(dark_circle_mask, center, axes, 0, 0, 360, 1.0, -1)
    
    else:
        # Fallback jika landmarks tidak tersedia
        # Mata kiri (area 25-35% dari lebar, 20-30% dari tinggi)
        left_center = (int(w * 0.3), int(h * 0.25))
        left_axes = (int(w * 0.15), int(h * 0.08))
        cv2.ellipse(dark_circle_mask, left_center, left_axes, 0, 0, 360, 1.0, -1)
        
        # Mata kanan (area 65-75% dari lebar, 20-30% dari tinggi)
        right_center = (int(w * 0.7), int(h * 0.25))
        right_axes = (int(w * 0.15), int(h * 0.08))
        cv2.ellipse(dark_circle_mask, right_center, right_axes, 0, 0, 360, 1.0, -1)
    
    # Blur mask untuk transisi halus
    blur_size = max(1, int(min(h, w) * 0.03))
    if blur_size % 2 == 0:
        blur_size += 1
    dark_circle_mask = cv2.GaussianBlur(dark_circle_mask, (blur_size, blur_size), 0)
    
    # Terapkan dark circles
    dark_color = np.array(color, dtype=np.float32) / 255.0
    result = face_region.astype(np.float32) / 255.0
    
    # Blend dark circles dengan intensitas
    for i in range(3):
        result[:, :, i] = result[:, :, i] * (1 - dark_circle_mask * intensity) + \
                         dark_color[i] * dark_circle_mask * intensity
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def add_wrinkles(face_region, landmarks, intensity=0.3, color=(80, 80, 100)):
    """
    Tambahkan kerutan di wajah.
    """
    if face_region is None or face_region.size == 0:
        return face_region
    
    h, w = face_region.shape[:2]
    
    # Buat tekstur kerutan
    wrinkle_texture = create_wrinkle_texture((h, w), intensity)
    
    # Buat mask untuk area kerutan
    wrinkle_mask = np.zeros((h, w), dtype=np.float32)
    
    if landmarks is not None and len(landmarks) >= 68:
        # Area dahi
        forehead_points = []
        for idx in [18, 19, 20, 21, 22, 23, 24, 25, 26]:
            if idx < len(landmarks):
                x = int((landmarks[idx][0] - landmarks[0][0]) * w / 
                       (landmarks[16][0] - landmarks[0][0]))
                y = int((landmarks[idx][1] - landmarks[19][1]) * h / 
                       (landmarks[8][1] - landmarks[19][1]))
                forehead_points.append((x, y))
        
        if len(forehead_points) >= 4:
            pts = np.array(forehead_points, np.int32)
            cv2.fillPoly(wrinkle_mask, [pts], 0.8)
        
        # Garis senyum
        smile_lines = []
        for idx in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]:
            if idx < len(landmarks):
                x = int((landmarks[idx][0] - landmarks[0][0]) * w / 
                       (landmarks[16][0] - landmarks[0][0]))
                y = int((landmarks[idx][1] - landmarks[19][1]) * h / 
                       (landmarks[8][1] - landmarks[19][1]))
                smile_lines.append((x, y))
        
        if len(smile_lines) >= 4:
            pts = np.array(smile_lines, np.int32)
            for i in range(len(pts) - 1):
                cv2.line(wrinkle_mask, tuple(pts[i]), tuple(pts[i+1]), 0.6, 
                        max(1, int(min(h, w) * 0.01)))
    
    else:
        # Fallback: tambahkan kerutan di area umum
        forehead_y1, forehead_y2 = int(h * 0.2), int(h * 0.4)
        cv2.rectangle(wrinkle_mask, (0, forehead_y1), (w, forehead_y2), 0.7, -1)
        
        # Garis dari hidung ke mulut
        center_x = w // 2
        nose_y = int(h * 0.4)
        mouth_y = int(h * 0.6)
        cv2.line(wrinkle_mask, (center_x - w//4, nose_y), 
                (center_x - w//6, mouth_y), 0.5, max(1, w//50))
        cv2.line(wrinkle_mask, (center_x + w//4, nose_y), 
                (center_x + w//6, mouth_y), 0.5, max(1, w//50))
    
    # Blur mask
    blur_size = max(1, int(min(h, w) * 0.02))
    if blur_size % 2 == 0:
        blur_size += 1
    wrinkle_mask = cv2.GaussianBlur(wrinkle_mask, (blur_size, blur_size), 0)
    
    # Kombinasikan tekstur dengan mask
    wrinkle_effect = wrinkle_texture * wrinkle_mask
    
    # Terapkan efek kerutan
    wrinkle_color = np.array(color, dtype=np.float32) / 255.0
    result = face_region.astype(np.float32) / 255.0
    
    # Blend efek kerutan
    for i in range(3):
        channel_multiplier = 1.0 - wrinkle_effect * 0.5 * wrinkle_color[i]
        result[:, :, i] = result[:, :, i] * channel_multiplier
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_aging_effects(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Terapkan efek penuaan ke wajah target.
    """
    ager = get_face_ager()
    params = ager.params
    
    # Extract face region
    x1, y1, x2, y2 = map(int, target_face.bbox)
    
    # Tambah padding
    padding_x = int((x2 - x1) * 0.2)
    padding_y = int((y2 - y1) * 0.2)
    
    h_frame, w_frame = temp_frame.shape[:2]
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w_frame, x2 + padding_x)
    y2 = min(h_frame, y2 + padding_y)
    
    face_region = temp_frame[y1:y2, x1:x2]
    
    if face_region.size == 0:
        return temp_frame
    
    # Dapatkan landmarks wajah
    landmarks = None
    if hasattr(target_face, 'landmark_2d_106'):
        landmarks = target_face.landmark_2d_106
    elif hasattr(target_face, 'kps'):
        landmarks = target_face.kps
    
    # Terapkan dark circles
    if params['dark_circle_intensity'] > 0:
        face_region = add_dark_circles(
            face_region, 
            landmarks, 
            params['dark_circle_intensity'],
            params['dark_circle_color']
        )
    
    # Terapkan kerutan
    if params['wrinkle_intensity'] > 0:
        face_region = add_wrinkles(
            face_region,
            landmarks,
            params['wrinkle_intensity'],
            params['wrinkle_color']
        )
    
    # Kembalikan ke frame
    temp_frame[y1:y2, x1:x2] = face_region
    
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Proses satu frame dengan efek penuaan.
    """
    ager = get_face_ager()
    params = ager.params
    
    if not params.get('apply_to_all_faces', True):
        # Hanya terapkan ke wajah utama
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = apply_aging_effects(target_face, temp_frame)
    else:
        # Terapkan ke semua wajah
        many_faces = get_many_faces(temp_frame)
        if many_faces:
