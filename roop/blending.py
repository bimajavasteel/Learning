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
    Advanced blending dengan validasi input yang lebih baik dan perbaikan color matching
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

        # Color matching (fix flickering) - PERBAIKAN UTAMA
        if original_crop.shape[2] == 3 and enhanced_crop.shape[2] == 3:
            # Gunakan LAB color space untuk matching yang lebih natural
            original_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB)
            enhanced_lab = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2LAB)
            
            # Hitung mean dan std untuk setiap channel secara terpisah
            original_mean = np.mean(original_lab, axis=(0, 1))
            original_std = np.std(original_lab, axis=(0, 1)) + 1e-6  # tambahkan epsilon untuk hindari div by zero
            enhanced_mean = np.mean(enhanced_lab, axis=(0, 1))
            enhanced_std = np.std(enhanced_lab, axis=(0, 1)) + 1e-6
            
            # Normalisasi dan color match - PERBAIKAN DIMENSI
            enhanced_lab_normalized = enhanced_lab.astype(np.float32)
            enhanced_lab_normalized = (enhanced_lab_normalized - enhanced_mean) / enhanced_std
            enhanced_lab_normalized = enhanced_lab_normalized * original_std + original_mean
            enhanced_lab_normalized = np.clip(enhanced_lab_normalized, 0, 255).astype(np.uint8)
            color_matched = cv2.cvtColor(enhanced_lab_normalized, cv2.COLOR_LAB2BGR)
        else:
            color_matched = enhanced_crop.copy()  # Jangan lupa copy

        # Fidelity blend (preserve original expressions)
        if color_matched.shape != original_crop.shape:
            color_matched = cv2.resize(color_matched, (original_crop.shape[1], original_crop.shape[0]))
        blended = cv2.addWeighted(color_matched, fidelity, original_crop, 1.0 - fidelity, 0)

        # Create base elliptical mask
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.42), int(h * 0.48))  # Oval shape untuk cakupan wajah natural
        if axes[0] > 0 and axes[1] > 0 and center[0] < w and center[1] < h:  # Pastikan axes valid
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Feather the edges
        blur_radius = max(3, int(min(w, h) * 0.1))
        if blur_radius % 2 == 0:
            blur_radius += 1
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        # Integrate occlusion mask if available
        if occlusion_mask is not None and occlusion_mask.size > 0 and occlusion_mask.shape[:2] == (h, w):
            occlusion_mask = np.clip(occlusion_mask, 0, 1)
            # Invert: kita ingin bobot lebih tinggi di area TIDAK terhalang
            visibility_mask = 1.0 - occlusion_mask
            # Combine dengan elliptical mask
            combined_mask = mask * visibility_mask
            # Additional smoothing untuk menghindari tepi keras
            combined_mask = cv2.GaussianBlur(combined_mask, (blur_radius, blur_radius), 0)
        else:
            combined_mask = mask

        # Convert ke 3-channel
        mask_3ch = np.dstack([combined_mask] * 3)
        
        # Final compositing dengan validasi
        if blended.shape != original_crop.shape:
            blended = cv2.resize(blended, (original_crop.shape[1], original_crop.shape[0]))
        
        result = blended * mask_3ch + original_crop * (1.0 - mask_3ch)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    except Exception as e:
        print(f"Blending error: {str(e)}")
        # Fallback: return original crop jika terjadi error
        return original_crop.copy() if original_crop is not None else np.zeros((128, 128, 3), dtype=np.uint8)
