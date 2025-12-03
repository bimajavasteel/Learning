# ================================================================
#  video_sharpener.py
#  - Melakukan Unsharp Mask + High Frequency Boost
#  - Dipanggil setelah face_swapper, sebelum face_enhancer
# ================================================================

import cv2
import numpy as np

# ------------------------------------------------
# 1. Unsharp Mask (stabil, aman untuk wajah)
# ------------------------------------------------
def apply_unsharp_mask(frame, strength=1.1, blur_size=3):
    try:
        blur = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        sharpened = cv2.addWeighted(frame, 1 + strength, blur, -strength, 0)
        return sharpened
    except Exception as e:
        print(f"[video_sharpener] Unsharp Mask error: {e}")
        return frame


# ------------------------------------------------
# 2. High Frequency Boost (detail kecil + pori)
# ------------------------------------------------
def apply_high_freq_boost(frame, boost=0.35):
    try:
        # bilateral menjaga struktur wajah
        low = cv2.bilateralFilter(frame, 9, 75, 75)
        high = cv2.subtract(frame, low)
        merged = cv2.add(frame, high * boost)
        return np.clip(merged, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[video_sharpener] HighFreq error: {e}")
        return frame


# ------------------------------------------------
# 3. Proses utama untuk setiap frame
# ------------------------------------------------
def sharpen_frame_pipeline(frame):
    try:
        # Step 1: Unsharp Mask
        frame = apply_unsharp_mask(frame, strength=1.1, blur_size=3)

        # Step 2: High Frequency Boost
        frame = apply_high_freq_boost(frame, boost=0.35)

        return frame

    except Exception as e:
        print(f"[video_sharpener] Pipeline error: {e}")
        return frame


# ------------------------------------------------
# 4. Hook untuk ROOP (wajib agar bisa dipanggil processor)
# ------------------------------------------------
def process_frame(frame, faces=None, **kwargs):
    """
    Dipanggil setelah face_swapper dan sebelum face_enhancer.
    Tidak memakai data wajah karena sharpening berlaku global.
    """
    return sharpen_frame_pipeline(frame)
