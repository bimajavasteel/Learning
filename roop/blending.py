import numpy as np
import cv2
from typing import Optional

def apply_blend_and_color_match(
    enhanced_crop: np.ndarray, 
    original_crop: np.ndarray, 
    occlusion_mask: Optional[np.ndarray] = None,
    fidelity: float = 0.6
) -> np.ndarray:
    """
    Advanced blending dengan validasi input yang lebih baik
    """
    try:
        # Validasi input: pastikan gambar tidak kosong
        if enhanced_crop is None or original_crop is None or enhanced_crop.size == 0 or original_crop.size == 0:
            print("Blending error: Empty input crop detected")
            return original_crop.copy() if original_crop is not None else np.zeros((128, 128, 3), dtype=np.uint8)
        
        h, w = original_crop.shape[:2]
        if h == 0 or w == 0:
            print("Blending error: Original crop has zero dimensions")
            return enhanced_crop.copy() if enhanced_crop is not None else np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Resize enhanced_crop jika dimensi berbeda
        if enhanced_crop.shape[:2] != (h, w):
            # Tambahkan validasi sebelum resize
            if enhanced_crop.shape[0] == 0 or enhanced_crop.shape[1] == 0:
                print("Blending error: Enhanced crop has zero dimensions")
                return original_crop.copy()
            enhanced_crop = cv2.resize(enhanced_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Color matching (fix flickering)
        if original_crop.shape[2] == 3 and enhanced_crop.shape[2] == 3:
            original_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB)
            enhanced_lab = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2LAB)
            
            original_mean, original_std = cv2.meanStdDev(original_lab)
            enhanced_mean, enhanced_std = cv2.meanStdDev(enhanced_lab)
            
            # Handle division by zero
            enhanced_std = np.where(enhanced_std == 0, 1, enhanced_std)
            ratio = original_std / enhanced_std
            
            # Normalize and color match
            enhanced_lab = enhanced_lab.astype(np.float32)
            enhanced_lab = ((enhanced_lab - enhanced_mean) * ratio) + original_mean
            enhanced_lab = np.clip(enhanced_lab, 0, 255).astype(np.uint8)
            color_matched = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            color_matched = enhanced_crop

        # Fidelity blend (preserve original expressions)
        blended = cv2.addWeighted(color_matched, fidelity, original_crop, 1.0 - fidelity, 0)

        # Create base elliptical mask
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.42), int(h * 0.48))  # Oval shape untuk cakupan wajah natural
        if axes[0] > 0 and axes[1] > 0:  # Pastikan axes valid
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Feather the edges
        blur_radius = max(3, int(min(w, h) * 0.1))
        if blur_radius % 2 == 0:
            blur_radius += 1
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        # Integrate occlusion mask if available
        if occlusion_mask is not None and occlusion_mask.size > 0:
            # Pastikan dimensi cocok
            if occlusion_mask.shape[:2] != (h, w):
                occlusion_mask = cv2.resize(occlusion_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            occlusion_mask = np.clip(occlusion_mask, 0, 1)
            
            # Invert: kita ingin bobot lebih tinggi di area TIDAK terhalang
            visibility_mask = 1.0 - occlusion_mask
            
            # Combine dengan elliptical mask
            combined_mask = mask * visibility_mask
            
            # Additional smoothing untuk menghindari tepi keras
            combined_mask = cv2.GaussianBlur(combined_mask, (blur_radius, blur_radius), 0)
        else:
            combined_mask = mask

        # Apply hair-aware feathering untuk pose ekstrem
        combined_mask = cv2.dilate(combined_mask, np.ones((3, 3), np.uint8), iterations=1)
        combined_mask = cv2.GaussianBlur(combined_mask, (blur_radius + 2, blur_radius + 2), 0)
        combined_mask = np.clip(combined_mask, 0, 1)

        # Convert ke 3-channel
        mask_3ch = np.dstack([combined_mask] * 3)
        
        # Final compositing dengan validasi
        result = np.zeros_like(original_crop)
        valid_mask = (mask_3ch > 0) & (mask_3ch <= 1)
        result[valid_mask] = (blended * mask_3ch + original_crop * (1.0 - mask_3ch))[valid_mask]
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    except Exception as e:
        print(f"Blending error: {str(e)}")
        # Fallback: return original crop jika terjadi error
        return original_crop.copy() if original_crop is not None else np.zeros((128, 128, 3), dtype=np.uint8)
