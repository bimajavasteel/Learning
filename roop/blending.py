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
    Advanced blending with occlusion awareness, color matching, and fidelity control.
    Handles hair and extreme poses by integrating occlusion mask.
    """
    try:
        h, w = original_crop.shape[:2]
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))

        # Color matching (fix flickering)
        original_lab = cv2.cvtColor(original_crop, cv2.COLOR_BGR2LAB)
        enhanced_lab = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2LAB)
        
        original_mean, original_std = cv2.meanStdDev(original_lab)
        enhanced_mean, enhanced_std = cv2.meanStdDev(enhanced_lab)
        
        # Avoid division by zero
        enhanced_std = np.where(enhanced_std == 0, 1, enhanced_std)
        ratio = original_std / enhanced_std
        
        # Normalize and color match
        enhanced_lab = enhanced_lab.astype(np.float32)
        enhanced_lab = ((enhanced_lab - enhanced_mean) * ratio) + original_mean
        enhanced_lab = np.clip(enhanced_lab, 0, 255).astype(np.uint8)
        color_matched = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Fidelity blend (preserve original expressions)
        blended = cv2.addWeighted(color_matched, fidelity, original_crop, 1.0 - fidelity, 0)

        # Create base elliptical mask
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.42), int(h * 0.48))  # Oval shape for natural face coverage
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        
        # Feather the edges
        blur_radius = max(5, int(min(w, h) * 0.1))
        if blur_radius % 2 == 0:
            blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        # Integrate occlusion mask if available
        if occlusion_mask is not None:
            # Normalize occlusion mask (0.0 = visible, 1.0 = fully occluded)
            occlusion_mask = cv2.resize(occlusion_mask, (w, h))
            occlusion_mask = np.clip(occlusion_mask, 0, 1)
            
            # Invert: we want higher weight where NOT occluded
            visibility_mask = 1.0 - occlusion_mask
            
            # Combine with elliptical mask (multiply for precision)
            combined_mask = mask * visibility_mask
            
            # Additional smoothing to prevent hard edges
            combined_mask = cv2.GaussianBlur(combined_mask, (blur_radius, blur_radius), 0)
        else:
            combined_mask = mask

        # Apply hair-aware feathering for extreme poses
        combined_mask = cv2.dilate(combined_mask, np.ones((3, 3), np.uint8), iterations=1)
        combined_mask = cv2.GaussianBlur(combined_mask, (blur_radius + 2, blur_radius + 2), 0)

        # Convert to 3-channel
        mask_3ch = np.dstack([combined_mask] * 3)
        
        # Final compositing
        result = (blended * mask_3ch + original_crop * (1.0 - mask_3ch)).astype(np.uint8)
        return result

    except Exception as e:
        print(f"Blending error: {e}")
        return original_crop
