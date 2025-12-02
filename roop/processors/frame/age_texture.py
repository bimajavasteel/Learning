import cv2
import numpy as np
from typing import Tuple, Optional


class AgeTexturePreserver:
    """Kelas untuk menangani preservasi tekstur usia."""
    
    def __init__(self):
        self.wrinkle_kernel = None
        self.dark_circle_kernel = None
    
    def detect_wrinkles(self, image: np.ndarray) -> np.ndarray:
        """Deteksi kerutan pada gambar wajah."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast untuk deteksi kerutan
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge detection untuk kerutan
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Morphological operations untuk membersihkan
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        return edges
    
    def detect_dark_circles(self, image: np.ndarray) -> np.ndarray:
        """Deteksi area dark circles di bawah mata."""
        # Convert to Lab color space untuk deteksi pigmentasi
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Fokus pada channel b (blue-yellow) untuk dark circles
        b_normalized = cv2.normalize(b_channel, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold untuk area gelap
        _, dark_mask = cv2.threshold(b_normalized, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Buat mask untuk area bawah mata
        h, w = image.shape[:2]
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Area bawah mata (estimated)
        y_start = int(h * 0.25)
        y_end = int(h * 0.45)
        x_center = w // 2
        
        cv2.ellipse(eye_mask, (int(w * 0.35), int(h * 0.35)), 
                   (int(w * 0.15), int(h * 0.08)), 0, 0, 360, 255, -1)
        cv2.ellipse(eye_mask, (int(w * 0.65), int(h * 0.35)), 
                   (int(w * 0.15), int(h * 0.08)), 0, 0, 360, 255, -1)
        
        # Combine masks
        result = cv2.bitwise_and(dark_mask, eye_mask)
        
        # Blur untuk soft edges
        result = cv2.GaussianBlur(result, (5, 5), 0)
        
        return result
    
    def transfer_age_features(self, source: np.ndarray, target: np.ndarray,
                             wrinkle_strength: float = 1.0,
                             dark_circle_strength: float = 1.0) -> np.ndarray:
        """Transfer fitur usia dari source ke target."""
        if source.shape != target.shape:
            source = cv2.resize(source, (target.shape[1], target.shape[0]))
        
        # Deteksi fitur di source
        source_wrinkles = self.detect_wrinkles(source)
        source_dark_circles = self.detect_dark_circles(source)
        
        # Terapkan ke target
        result = target.copy().astype(np.float32)
        
        # Terapkan kerutan
        if wrinkle_strength > 0:
            wrinkle_mask = source_wrinkles.astype(np.float32) / 255.0 * wrinkle_strength
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - wrinkle_mask * 0.2)
        
        # Terapkan dark circles
        if dark_circle_strength > 0:
            dark_mask = source_dark_circles.astype(np.float32) / 255.0 * dark_circle_strength
            dark_color = np.array([40, 30, 50], dtype=np.float32)  # BGR: ungu-kecoklatan
            
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - dark_mask * 0.5) + \
                                 dark_color[c] * dark_mask * 0.5
        
        return np.clip(result, 0, 255).astype(np.uint8)


# Singleton instance
_age_preserver = None

def get_age_preserver() -> AgeTexturePreserver:
    global _age_preserver
    if _age_preserver is None:
        _age_preserver = AgeTexturePreserver()
    return _age_preserver
