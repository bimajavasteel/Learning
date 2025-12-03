# ================================================================
#  realesrgan_x1_lite.py
#  Menggunakan RealESRGAN x1 model (super fast & stable)
#  Pipeline: face_swapper → realesrgan_x1_lite → face_enhancer
# ================================================================

import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------
# Import RealESRGAN
# ------------------------------------------------
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except Exception as e:
    raise RuntimeError(f"[RealESRGAN_x1_lite] ERROR import: {e}")

# ------------------------------------------------
# Load model once (global)
# ------------------------------------------------
MODEL = None

def load_realesrgan_x1():
    global MODEL

    if MODEL is not None:
        return MODEL

    try:
        # RRDBNet untuk model x1 (tanpa upscale)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            nf=32,
            nb=4,
            gc=16
        )

        MODEL = RealESRGANer(
            scale=1,                                   # x1 model
            model_path="/kaggle/working/Learning/models/RealESRGAN_x1_lite.pth",
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True                                   # FAST on GPU
        )

        print("[RealESRGAN_x1_lite] Model loaded successfully.")
        return MODEL

    except Exception as e:
        print(f"[RealESRGAN_x1_lite] Failed loading model: {e}")
        raise e


# ------------------------------------------------
# Process one frame with RealESRGAN x1
# ------------------------------------------------
def realesrgan_x1_process(frame):
    try:
        model = load_realesrgan_x1()
        output, _ = model.enhance(frame, outscale=1)

        return output
    except Exception as e:
        print(f"[RealESRGAN_x1_lite] Enhance error: {e}")
        return frame


# ------------------------------------------------
# ROOP Processor Hook
# ------------------------------------------------
def process_frame(frame, faces=None, **kwargs):
    """
    Dipanggil di frame pipeline:
    face_swapper → realesrgan_x1_lite → face_enhancer
    """
    return realesrgan_x1_process(frame)
