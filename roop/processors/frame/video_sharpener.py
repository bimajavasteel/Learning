# ================================================================
#  video_sharpener.py (UPGRADED + ROOP NOTIFICATION)
# ================================================================

import cv2
import numpy as np
from roop.core import update_status

# ------------------------------------------------
# Utility: Variance Laplacian
# ------------------------------------------------
def _lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ------------------------------------------------
# Adaptive Unsharp
# ------------------------------------------------
def adaptive_unsharp(frame, base_strength=1.0):
    try:
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        detail = cv2.subtract(frame, blur)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_32F).var()

        adaptive_gain = np.clip(var / 150.0, 0.2, 1.5)
        strength = base_strength * adaptive_gain

        sharpen = cv2.addWeighted(frame, 1 + strength, blur, -strength, 0)
        return sharpen
    except Exception as e:
        print(f"[adaptive_unsharp] Error: {e}")
        return frame


# ------------------------------------------------
# Edge-aware Sharpen
# ------------------------------------------------
def edge_aware_sharpen(frame, intensity=0.25):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(sobelx, sobely)

        edge_norm = cv2.normalize(edge, None, 0.0, 1.0, cv2.NORM_MINMAX)
        edge_norm = edge_norm[..., None]

        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        high = cv2.subtract(frame, blur)

        enhanced = frame + high * (intensity * edge_norm)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[edge_aware_sharpen] Error: {e}")
        return frame


# ------------------------------------------------
# Noise detection → dynamic scaling
# ------------------------------------------------
def smart_noise_gate(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v = _lap_var(gray)

        if v > 180:
            return 0.5
        if v > 260:
            return 0.2
        return 1.0
    except:
        return 1.0


# ------------------------------------------------
# High Frequency Boost
# ------------------------------------------------
def controlled_hf_boost(frame, boost=0.25):
    try:
        low = cv2.bilateralFilter(frame, 9, 50, 50)
        high = cv2.subtract(frame, low)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        mask = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX)
        mask = mask[..., None]

        return np.clip(frame + high * boost * mask, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[HF_Boost] Error: {e}")
        return frame


# ------------------------------------------------
# Main pipeline
# ------------------------------------------------
def sharpen_frame_pipeline(frame):
    noise_scale = smart_noise_gate(frame)

    frame = adaptive_unsharp(frame, base_strength=0.8 * noise_scale)
    frame = edge_aware_sharpen(frame, intensity=0.25 * noise_scale)
    frame = controlled_hf_boost(frame, boost=0.25 * noise_scale)

    return frame


# ------------------------------------------------
# ROOP Hook + Notification
# ------------------------------------------------
frame_count = 0
total_frames = None

def process_frame(frame, faces=None, **kwargs):
    global frame_count, total_frames

    # Set total frames only once
    if total_frames is None:
        total_frames = kwargs.get("total_frames", None)

    # Progress notification
    if total_frames:
        pct = (frame_count / total_frames) * 100
        update_status(f"[ROOP.VIDEO-SHARPENER] Processing... {pct:.1f}%")

    frame_count += 1

    # Main process
    return sharpen_frame_pipeline(frame)
