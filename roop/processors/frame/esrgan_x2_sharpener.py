# ================================================================
#  esrgan_x2_sharpener.py  (RealESRGAN_x2plus — NO DOWNSCALE)
#
#  Pipeline:
#     face_swapper → ESRGAN_x2 (native upscale 2×) → face_enhancer
#
#  Fitur:
#  - Auto-download model
#  - Auto-load model sekali saja
#  - Output final 2× dari resolusi asli
#  - Tidak ada downscale → detail paling tinggi
# ================================================================

import os
import cv2
import torch
import numpy as np

from roop.utilities import conditional_download


# ================================================================
# 1. Auto-download model
# ================================================================
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
MODEL_NAME = "RealESRGAN_x2plus.pth"
MODEL_PATH = f"/kaggle/working/{MODEL_NAME}"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"[ESRGAN_X2] Downloading model... {MODEL_NAME}")
        conditional_download(MODEL_URL, MODEL_PATH)
    else:
        print("[ESRGAN_X2] Model already exists.")


# ================================================================
# 2. Load ESRGAN model
# ================================================================
upsampler = None
RELOADED = False

def load_esrgan_x2():
    global upsampler, RELOADED

    if RELOADED:
        return upsampler

    from realesrgan import RealESRGANer

    download_model_if_needed()

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
        print(f"[ESRGAN_X2] Failed to load ESRGAN model: {e}")
        return None


# ================================================================
# 3. Enhance frame (no downscale)
# ================================================================
def esrgan_process_frame(frame):
    global upsampler

    if upsampler is None:
        upsampler = load_esrgan_x2()
        if upsampler is None:
            print("[ESRGAN_X2] ERROR: Cannot load ESRGAN. Skipping.")
            return frame

    try:
        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Native 2× super-resolution (no downscale)
        output_rgb, _ = upsampler.enhance(img_rgb, outscale=2)

        # Convert back to BGR
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        return output_bgr

    except Exception as e:
        print(f"[ESRGAN_X2] Enhance error: {e}")
        return frame


# ================================================================
# 4. ROOP hook
# ================================================================
def process_frame(frame, faces=None, **kwargs):
    return esrgan_process_frame(frame)
