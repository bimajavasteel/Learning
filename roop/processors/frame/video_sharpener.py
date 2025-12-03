# ================================================================
#  video_sharpener.py (SUPER SHARP EDITION 100%)
#  - Multi-Stage Unsharp Mask
#  - Directional Edge Sharpening
#  - Frequency Split Sharpening
#  - Detail Booster (Texture)
#  - Adaptive Sharpening
#  - Smart Noise Gate
#  - HF Boost v2
# ================================================================

import cv2
import numpy as np

# ------------------------------------------------
# Utility: Variance of Laplacian
# ------------------------------------------------
def _lap_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ------------------------------------------------
# 1. Multi-Stage Unsharp Mask
# ------------------------------------------------
def multi_stage_unsharp(frame, strength=1.0):
    try:
        blur_small = cv2.GaussianBlur(frame, (3, 3), 0)
        blur_mid = cv2.GaussianBlur(frame, (7, 7), 0)
        blur_large = cv2.GaussianBlur(frame, (13, 13), 0)

        usm1 = cv2.addWeighted(frame, 1 + 0.6 * strength, blur_small, -0.6 * strength, 0)
        usm2 = cv2.addWeighted(usm1, 1 + 0.4 * strength, blur_mid, -0.4 * strength, 0)
        usm3 = cv2.addWeighted(usm2, 1 + 0.3 * strength, blur_large, -0.3 * strength, 0)

        return usm3
    except:
        return frame


# ------------------------------------------------
# 2. Directional Edge Sharpening (DES)
# ------------------------------------------------
def directional_edge_sharpen(frame, intensity=0.25):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # normalize mask
        mask = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
        mask = mask[..., None]

        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        high = cv2.subtract(frame, blur)

        enhanced = frame + high * mask * intensity
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    except:
        return frame


# ------------------------------------------------
# 3. Frequency Split Sharpening
# ------------------------------------------------
def frequency_split_sharpen(frame, amount=1.0):
    try:
        low = cv2.GaussianBlur(frame, (9, 9), 0)
        high = cv2.subtract(frame, low)

        boosted = frame + high * (0.8 * amount)
        return np.clip(boosted, 0, 255).astype(np.uint8)
    except:
        return frame


# ------------------------------------------------
# 4. Detail Booster (micro-contrast)
# ------------------------------------------------
def detail_booster(frame, strength=0.3):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)

        mask = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)[..., None]

        boosted = frame * (1 + mask * strength)
        return np.clip(boosted, 0, 255).astype(np.uint8)
    except:
        return frame


# ------------------------------------------------
# 5. Smart Noise Gate (adaptive sharpening)
# ------------------------------------------------
def smart_noise_gate(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v = _lap_var(gray)

        if v > 200:
            return 0.6     # reduce sharpening
        if v > 300:
            return 0.3     # heavy noise reduction
        return 1.0
    except:
        return 1.0


# ------------------------------------------------
# 6. High Frequency Boost v2
# ------------------------------------------------
def hf_boost_v2(frame, boost=0.45):
    try:
        low = cv2.bilateralFilter(frame, 9, 40, 40)
        high = cv2.subtract(frame, low)

        # boost + texture mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        mask = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX)[..., None]

        enhanced = frame + high * mask * boost
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    except:
        return frame


# ------------------------------------------------
# FINAL PIPELINE (100% SHARP)
# ------------------------------------------------
def sharpen_frame_pipeline(frame):
    try:
        noise_scale = smart_noise_gate(frame)

        # 1) multi-stage unsharp (big clarity boost)
        frame = multi_stage_unsharp(frame, strength=1.2 * noise_scale)

        # 2) directional edge sharpening
        frame = directional_edge_sharpen(frame, intensity=0.35 * noise_scale)

        # 3) frequency split sharpen
        frame = frequency_split_sharpen(frame, amount=0.8 * noise_scale)

        # 4) detail booster
        frame = detail_booster(frame, strength=0.3 * noise_scale)

        # 5) HF boost v2
        frame = hf_boost_v2(frame, boost=0.35 * noise_scale)

        return frame

    except Exception as e:
        print(f"[video_sharpener pipeline] Error: {e}")
        return frame


# ------------------------------------------------
# ROOP Hook
# ------------------------------------------------
def process_frame(frame, faces=None, **kwargs):
    return sharpen_frame_pipeline(frame)
