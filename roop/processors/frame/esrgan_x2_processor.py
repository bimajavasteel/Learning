# ================================================================
#  esrgan_x2_sharpener.py  (RealESRGAN_x2plus VERSION)
#
#  Pipeline: face_swapper → ESRGAN_x2 → face_enhancer
#
#  Fitur:
#  - Auto-download model RealESRGAN_x2plus.pth
#  - Auto-load sekali saja (global)
#  - ESRGAN scale X2 → lalu downscale kembali (sharp natural)
#  - Error handling lengkap
# ================================================================

import os
import cv2
import torch
import numpy as np

from roop.utilities import conditional_download, resolve_relative_path


# ================================================================
# 1. Auto Download Model
# ================================================================
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
MODEL_NAME = "RealESRGAN_x2plus.pth"
MODEL_PATH = f"/kaggle/working/{MODEL_NAME}"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"[ESRGAN_X2] Downloading model... {MODEL_NAME}")
        conditional_download(MODEL_URL, MODEL_PATH)
    else:
        print(f"[ESRGAN_X2] Model already exists.")


# ================================================================
# 2. Load Real-ESRGAN Model
# ================================================================
RELOADED = False
upsampler = None

def load_esrgan_x2():
    global RELOADED, upsampler

    if RELOADED:
        return upsampler

    from realesrgan import RealESRGANer

    download_model_if_needed()

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        upsampler = RealESRGANer(
            scale=2,
            model_path=MODEL_PATH,
            dni_weight=None,
            device=device,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if torch.cuda.is_available() else False
        )
        RELOADED = True
        print("[ESRGAN_X2] Model loaded successfully.")
        return upsampler

    except Exception as e:
        print(f"[ESRGAN_X2] Failed loading ESRGAN: {e}")
        return None


# ================================================================
# 3. ESRGAN X2 Processing per-frame
# ================================================================
def esrgan_process_frame(frame):
    global upsampler

    if upsampler is None:
        upsampler = load_esrgan_x2()
        if upsampler is None:
            print("[ESRGAN_X2] ERROR: ESRGAN not loaded, skipping sharpener.")
            return frame

    try:
        # Convert BGR→RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Super-resolution inference (scale=2)
        output_rgb, _ = upsampler.enhance(img_rgb, outscale=2)

        # Downscale back to original size (sharpening effect)
        h, w = frame.shape[:2]
        output_rgb = cv2.resize(output_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # Convert back to BGR
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        return output_bgr

    except Exception as e:
        print(f"[ESRGAN_X2] Enhance error: {e}")
        return frame


# ================================================================
# 4. ROOP Hook
# ================================================================
def process_frame(frame, faces=None, **kwargs):
    return esrgan_process_frame(frame)
