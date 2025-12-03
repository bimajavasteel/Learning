# ================================================================
#  video_sharpener.py (UPGRADED VERSION)
#  - Adaptive Unsharp Mask
#  - Edge-Aware Sharpening
#  - Smart Noise Gate
#  - High-Frequency Boost (Controlled)
# ================================================================

import cv2
import numpy as np

# ------------------------------------------------
# Utility: Variance of Laplacian (sharpness metric)
# ------------------------------------------------
def _lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ------------------------------------------------
# Adaptive Unsharp Mask
# ------------------------------------------------
def adaptive_unsharp(frame, base_strength=1.0):
    try:
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        detail = cv2.subtract(frame, blur)

        # Buat mask adaptif berdasarkan local variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_32F).var()

        # normalize 0–1
        adaptive_gain = np.clip(var / 150.0, 0.2, 1.5)
        strength = base_strength * adaptive_gain

        sharpen = cv2.addWeighted(frame, 1 + strength, blur, -strength, 0)
        return sharpen
    except Exception as e:
        print(f"[adaptive_unsharp] Error: {e}")
        return frame


# ------------------------------------------------
# Edge-aware sharpening (menggunakan Sobel mask)
# ------------------------------------------------
def edge_aware_sharpen(frame, intensity=0.25):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(sobelx, sobely)

        # normalisasi 0–1
        edge_norm = cv2.normalize(edge, None, 0.0, 1.0, cv2.NORM_MINMAX)
        edge_norm = edge_norm[..., None]

        # unsharp kernel
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        high = cv2.subtract(frame, blur)

        adaptive = frame + high * (intensity * edge_norm)
        return np.clip(adaptive, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[edge_aware_sharpen] Error: {e}")
        return frame


# ------------------------------------------------
# Smart Noise Gate
# ------------------------------------------------
def smart_noise_gate(frame, noise_threshold=1.5):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        var = _lap_var(gray)

        # jika noise tinggi → kurangi sharpening
        if var > 180:
            return 0.5  # reduce sharpening 50%
        if var > 250:
            return 0.2  # reduce heavy

        return 1.0  # normal
    except:
        return 1.0


# ------------------------------------------------
# High Frequency Boost upgrade
# ------------------------------------------------
def controlled_hf_boost(frame, boost=0.3):
    try:
        low = cv2.bilateralFilter(frame, 9, 50, 50)
        high = cv2.subtract(frame, low)

        # Clarity-like mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        mask = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX)
        mask = mask[..., None]

        enhanced = frame + high * boost * mask
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"[HF_Boost] Error: {e}")
        return frame


# ------------------------------------------------
# Main pipeline
# ------------------------------------------------
def sharpen_frame_pipeline(frame):
    try:
        # smart noise scaling
        noise_scale = smart_noise_gate(frame)

        # step 1: adaptive unsharp
        frame = adaptive_unsharp(frame, base_strength=0.8 * noise_scale)

        # step 2: edge-aware enhancement
        frame = edge_aware_sharpen(frame, intensity=0.25 * noise_scale)

        # step 3: controlled HF boost
        frame = controlled_hf_boost(frame, boost=0.25 * noise_scale)

        return frame
    except Exception as e:
        print(f"[video_sharpener pipeline] Error: {e}")
        return frame


# ------------------------------------------------
# ROOP Hook
# ------------------------------------------------
def process_frame(frame, faces=None, **kwargs):
    return sharpen_frame_pipeline(frame)
