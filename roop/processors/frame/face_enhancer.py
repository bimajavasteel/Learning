from typing import Any, List, Callable
import cv2
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, is_image, is_video

NAME = 'ROOP.FACE-ENHANCER'

class FaceEnhancer:
    def __init__(self):
        self.name = "Simple Face Enhancer"
    
    def enhance_face(self, face_image: Frame) -> Frame:
        """Enhanced face processing with multiple techniques"""
        try:
            if face_image is None or face_image.size == 0:
                return face_image
                
            h, w = face_image.shape[:2]
            if h < 20 or w < 20:  # Too small to enhance meaningfully
                return face_image
            
            # 1. Color correction and white balance
            balanced = self.white_balance(face_image)
            
            # 2. Smart sharpening based on image size
            sharpened = self.adaptive_sharpen(balanced)
            
            # 3. Contrast enhancement in LAB space
            contrast_enhanced = self.enhance_contrast(sharpened)
            
            # 4. Noise reduction while preserving details
            denoised = self.smart_denoise(contrast_enhanced)
            
            # 5. Skin smoothing and texture improvement
            smoothed = self.smooth_skin(denoised)
            
            return smoothed
            
        except Exception as e:
            print(f"Enhancement error: {e}")
            return face_image
    
    def white_balance(self, img: Frame) -> Frame:
        """Simple white balance using gray world assumption"""
        try:
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        except:
            return img
    
    def adaptive_sharpen(self, img: Frame) -> Frame:
        """Adaptive sharpening based on image size"""
        try:
            h, w = img.shape[:2]
            
            if max(h, w) > 200:
                # Strong sharpening for large faces
                kernel = np.array([[-1, -1, -1, -1, -1],
                                  [-1,  2,  2,  2, -1],
                                  [-1,  2,  8,  2, -1],
                                  [-1,  2,  2,  2, -1],
                                  [-1, -1, -1, -1, -1]]) / 8.0
            elif max(h, w) > 100:
                # Medium sharpening
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]]) * 0.5
            else:
                # Light sharpening for small faces
                kernel = np.array([[0, -0.25, 0],
                                  [-0.25, 2, -0.25],
                                  [0, -0.25, 0]])
            
            return cv2.filter2D(img, -1, kernel)
        except:
            return img
    
    def enhance_contrast(self, img: Frame) -> Frame:
        """Enhance contrast using CLAHE in LAB color space"""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            
            # Merge back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        except:
            return img
    
    def smart_denoise(self, img: Frame) -> Frame:
        """Adaptive denoising based on image characteristics"""
        try:
            # Calculate noise level (variance of Laplacian)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if variance < 100:  # Low detail, likely noisy
                # Stronger denoising for noisy images
                denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            else:
                # Lighter denoising for detailed images
                denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
            
            return denoised
        except:
            return img
    
    def smooth_skin(self, img: Frame) -> Frame:
        """Skin smoothing while preserving details"""
        try:
            # Bilateral filter for edge-preserving smoothing
            smoothed = cv2.bilateralFilter(img, 5, 25, 25)
            
            # Blend with original to preserve some texture
            return cv2.addWeighted(smoothed, 0.7, img, 0.3, 0)
        except:
            return img

# Global enhancer instance
ENHANCER = FaceEnhancer()

def pre_check() -> bool:
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    pass

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face processing with better padding and blending"""
    try:
        frame_height, frame_width = temp_frame.shape[:2]
        start_x, start_y, end_x, end_y = map(int, target_face.bbox)
        
        # Calculate face size
        face_w, face_h = end_x - start_x, end_y - start_y
        if face_w <= 15 or face_h <= 15:
            return temp_frame

        # Smart padding
        pad_ratio = 0.25
        padding_x = int(face_w * pad_ratio)
        padding_y = int(face_h * pad_ratio)
        
        # Ensure within frame bounds
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = min(frame_width, end_x + padding_x)
        end_y = min(frame_height, end_y + padding_y)
        
        # Extract face region
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        if temp_face.size == 0:
            return temp_frame
            
        if temp_face.shape[0] < 10 or temp_face.shape[1] < 10:
            return temp_frame

        # Apply enhancement
        enhanced_face = ENHANCER.enhance_face(temp_face)
        
        if enhanced_face is not None and enhanced_face.size > 0:
            # Ensure same dimensions
            if enhanced_face.shape != temp_face.shape:
                enhanced_face = cv2.resize(enhanced_face, (temp_face.shape[1], temp_face.shape[0]))
            
            # Create smooth blending mask
            blend_mask = self.create_blend_mask(temp_face.shape)
            
            # Blend enhanced face with original
            blended_face = self.blend_images(enhanced_face, temp_face, blend_mask)
            temp_frame[start_y:end_y, start_x:end_x] = blended_face
                
    except Exception as e:
        print(f"Face enhancement error: {e}")
    
    return temp_frame

def create_blend_mask(shape: tuple) -> np.ndarray:
    """Create elliptical blend mask for smooth transitions"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    center_y, center_x = h // 2, w // 2
    axis_x, axis_y = w // 2, h // 2
    
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    return np.clip(mask, 0, 1)

def blend_images(enhanced: Frame, original: Frame, mask: np.ndarray) -> Frame:
    """Blend enhanced and original images using mask"""
    mask_3d = np.stack([mask] * 3, axis=-1)
    blended = (enhanced * mask_3d + original * (1 - mask_3d)).astype(np.uint8)
    return blended

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Process all faces in frame"""
    try:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = enhance_face(target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames"""
    for temp_frame_path in temp_frame_paths:
        try:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is not None:
                result = process_frame(None, None, temp_frame)
                cv2.imwrite(temp_frame_path, result)
            if update:
                update()
        except Exception as e:
            print(f"Process frame {temp_frame_path} error: {e}")
            continue

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image"""
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is not None:
            result = process_frame(None, None, target_frame)
            cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video frames"""
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
