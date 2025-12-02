from typing import Any, List, Callable
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer
import scipy.ndimage

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


def apply_wrinkle_effect(target_face: np.ndarray, wrinkle_map: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Menerapkan peta kerutan ke wajah target dengan kontrol strength.
    
    Args:
        target_face: Wajah target (BGR)
        wrinkle_map: Peta kerutan dari source (grayscale, 0-255)
        strength: Kekuatan efek (0.0-2.0)
    
    Returns:
        Wajah dengan efek kerutan
    """
    if strength <= 0 or wrinkle_map is None:
        return target_face
    
    # Normalisasi wrinkle_map
    wrinkle_normalized = wrinkle_map.astype(np.float32) / 255.0
    
    # Terapkan efek kerutan dengan adjustable strength
    result = target_face.copy().astype(np.float32)
    
    # Untuk area gelap (kerutan), tambahkan kontras
    wrinkle_mask = wrinkle_normalized * strength
    
    # Aplikasikan efek depth untuk kerutan
    # Area gelap (kerutan) dibuat lebih gelap, area terang tetap
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - wrinkle_mask * 0.3)  # Menggelapkan area kerutan
    
    # Tambahkan tekstur halus
    texture_strength = 0.1 * strength
    wrinkle_texture = wrinkle_normalized * texture_strength
    
    # Blend dengan original
    result = result * (1 - wrinkle_texture[:, :, None]) + target_face * wrinkle_texture[:, :, None]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_dark_circles(target_face: np.ndarray, dark_circle_mask: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    Menerapkan efek dark circles ke area bawah mata.
    
    Args:
        target_face: Wajah target (BGR)
        dark_circle_mask: Mask untuk area dark circles
        intensity: Intensitas efek (0.0-2.0)
    
    Returns:
        Wajah dengan efek dark circles
    """
    if intensity <= 0 or dark_circle_mask is None:
        return target_face
    
    result = target_face.copy().astype(np.float32)
    
    # Normalisasi mask
    mask_normalized = dark_circle_mask.astype(np.float32) / 255.0 * intensity
    
    # Warna untuk dark circles (warna kecoklatan/keunguan)
    dark_color = np.array([30, 20, 40], dtype=np.float32)  # BGR: sedikit keunguan
    
    # Terapkan warna gelap berdasarkan mask
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - mask_normalized * 0.7) + dark_color[c] * mask_normalized * 0.7
    
    # Soften edges
    kernel_size = max(1, int(min(target_face.shape[:2]) * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred_mask = cv2.GaussianBlur(mask_normalized, (kernel_size, kernel_size), 0)
    
    # Final blend
    final_result = target_face.copy().astype(np.float32)
    mask_3ch = blurred_mask[:, :, None]
    
    result = final_result * (1 - mask_3ch) + result * mask_3ch
    
    return np.clip(result, 0, 255).astype(np.uint8)


def extract_wrinkle_features(face_image: np.ndarray) -> np.ndarray:
    """
    Ekstrak fitur kerutan dari wajah.
    
    Args:
        face_image: Gambar wajah (BGR)
    
    Returns:
        Peta kerutan (grayscale)
    """
    # Konversi ke grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Enhance kerutan dengan filter high-pass
    kernel_size = max(3, int(min(face_image.shape[:2]) * 0.03))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Smoothing untuk mengurangi noise
    smoothed = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # High-pass filter untuk deteksi kerutan
    high_pass = gray.astype(np.float32) - smoothed.astype(np.float32)
    high_pass = np.clip(high_pass + 128, 0, 255).astype(np.uint8)
    
    # Threshold adaptif untuk ekstraksi kerutan
    wrinkle_map = cv2.adaptiveThreshold(
        high_pass, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations untuk membersihkan noise
    kernel = np.ones((2, 2), np.uint8)
    wrinkle_map = cv2.morphologyEx(wrinkle_map, cv2.MORPH_CLOSE, kernel)
    wrinkle_map = cv2.morphologyEx(wrinkle_map, cv2.MORPH_OPEN, kernel)
    
    return wrinkle_map


def extract_dark_circle_mask(face_image: np.ndarray) -> np.ndarray:
    """
    Ekstrak mask untuk area dark circles di bawah mata.
    
    Args:
        face_image: Gambar wajah (BGR)
    
    Returns:
        Mask dark circles
    """
    h, w = face_image.shape[:2]
    
    # Buat mask untuk area bawah mata
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Area bawah mata (sekitar 20-40% dari tinggi wajah)
    eye_y_start = int(h * 0.25)
    eye_y_end = int(h * 0.45)
    eye_x_mid = w // 2
    
    # Buat elips untuk area dark circles
    left_center = (int(w * 0.35), int(h * 0.35))
    right_center = (int(w * 0.65), int(h * 0.35))
    
    axes = (int(w * 0.15), int(h * 0.1))
    
    # Gambar elips untuk kedua mata
    cv2.ellipse(mask, left_center, axes, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, right_center, axes, 0, 0, 360, 255, -1)
    
    # Blur mask untuk edges yang smooth
    kernel_size = max(3, int(min(h, w) * 0.05))
    if kernel_size % 2 == 0:
        kernel_size += 1
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    return mask


def apply_age_texture_transfer(source_face: np.ndarray, target_face: np.ndarray, 
                              wrinkle_preservation: float = 1.0,
                              dark_circle_intensity: float = 1.0,
                              preserve_age_texture: bool = True) -> np.ndarray:
    """
    Transfer tekstur usia dari source ke target.
    
    Args:
        source_face: Wajah source
        target_face: Wajah target
        wrinkle_preservation: Strength preservasi kerutan (0.0-2.0)
        dark_circle_intensity: Intensitas dark circles (0.0-2.0)
        preserve_age_texture: Flag untuk preservasi tekstur usia
    
    Returns:
        Wajah target dengan tekstur usia
    """
    if not preserve_age_texture:
        return target_face
    
    result = target_face.copy()
    
    # Ekstrak fitur dari source
    source_wrinkles = extract_wrinkle_features(source_face)
    source_dark_circle_mask = extract_dark_circle_mask(source_face)
    
    # Resize fitur source ke ukuran target
    h_target, w_target = target_face.shape[:2]
    h_source, w_source = source_face.shape[:2]
    
    if (h_source, w_source) != (h_target, w_target):
        source_wrinkles = cv2.resize(source_wrinkles, (w_target, h_target))
        source_dark_circle_mask = cv2.resize(source_dark_circle_mask, (w_target, h_target))
    
    # Terapkan efek kerutan
    if wrinkle_preservation > 0:
        result = apply_wrinkle_effect(result, source_wrinkles, wrinkle_preservation)
    
    # Terapkan dark circles
    if dark_circle_intensity > 0:
        # Ambil dark circle mask dari source dan terapkan ke target
        result = apply_dark_circles(result, source_dark_circle_mask, dark_circle_intensity)
    
    # Final blending dengan original untuk kontrol
    blend_ratio = 0.7  # 70% efek, 30% original
    result = cv2.addWeighted(result, blend_ratio, target_face, 1 - blend_ratio, 0)
    
    return result


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


def enhance_face(target_face: Face, temp_frame: Frame, source_face_crop: np.ndarray = None) -> Frame:
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
        
        # 📌 TERAPKAN TEKSTUR USIA JIKA ADA SOURCE FACE
        if source_face_crop is not None and roop.globals.preserve_age_texture:
            wrinkle_strength = getattr(roop.globals, 'wrinkle_preservation', 1.0)
            dark_circle_strength = getattr(roop.globals, 'dark_circle_intensity', 1.0)
            
            enhanced_face = apply_age_texture_transfer(
                source_face=source_face_crop,
                target_face=enhanced_face,
                wrinkle_preservation=wrinkle_strength,
                dark_circle_intensity=dark_circle_strength,
                preserve_age_texture=roop.globals.preserve_age_texture
            )
        
        result_face = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount)
        
        temp_frame[start_y:end_y, start_x:end_x] = result_face
        
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            # Dapatkan crop wajah source jika tersedia
            source_face_crop = None
            if source_face is not None and hasattr(source_face, 'bbox'):
                s_x1, s_y1, s_x2, s_y2 = map(int, source_face.bbox)
                if hasattr(source_face, '_frame'):
                    source_frame = source_face._frame
                    source_face_crop = source_frame[s_y1:s_y2, s_x1:s_x2]
            
            temp_frame = enhance_face(target_face, temp_frame, source_face_crop)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    # Baca source image untuk ekstraksi tekstur
    source_img = cv2.imread(source_path) if source_path else None
    source_face_crop = None
    
    if source_img is not None:
        source_faces = get_many_faces(source_img)
        if source_faces:
            # Ambil wajah pertama dari source
            s_face = source_faces[0]
            s_x1, s_y1, s_x2, s_y2 = map(int, s_face.bbox)
            source_face_crop = source_img[s_y1:s_y2, s_x1:s_x2]
    
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
