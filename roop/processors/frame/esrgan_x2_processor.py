# ================================================================
#   esrgan_x2_processor.py
#   RealESRGAN_x2plus video enhancer (2X Super Resolution)
#   Optimized for Kaggle GPU (CUDA Execution Provider)
# ================================================================

import cv2
import numpy as np
import os
import onnxruntime as ort

from roop.core import update_status

# ================================================================
#   LOAD ESRGAN X2 MODEL (ONNX)
# ================================================================
MODEL_NAME = "RealESRGAN_x2plus.onnx"
MODEL_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x2plus.onnx"

MODEL_PATH = f"/kaggle/working/{MODEL_NAME}"

# Auto-download model if missing
if not os.path.exists(MODEL_PATH):
    import urllib.request
    print(f"[ESRGAN] Downloading {MODEL_NAME} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[ESRGAN] Download completed.")

# Create ONNX runtime session
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# ================================================================
#   ESRGAN INFERRING
# ================================================================
def esrgan_x2(frame):
    """
    RealESRGAN_x2plus inference
    2X upscaling → downscale back to original size
    """
    try:
        h, w = frame.shape[:2]

        # ----- preprocess -----
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))              # CHW
        img = np.expand_dims(img, 0)                   # NCHW

        # ONNX Inference
        output = session.run([output_name], {input_name: img})[0]

        # ----- postprocess -----
        out = output[0]
        out = np.clip(out, 0, 1)
        out = (out * 255.0).astype(np.uint8)
        out = np.transpose(out, (1, 2, 0))             # HWC

        # Downscale back to original resolution
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        return out

    except Exception as e:
        print(f"[ESRGAN_X2] Error: {e}")
        return frame


# ================================================================
#   ROOP PIPELINE HOOK
# ================================================================
frame_count = 0
total_frames = None

def process_frame(frame, faces=None, **kwargs):
    global frame_count, total_frames

    if total_frames is None:
        total_frames = kwargs.get("total_frames", None)

    if total_frames:
        pct = (frame_count / total_frames) * 100
        update_status(f"[ROOP.ESRGAN-X2] Enhancing... {pct:.1f}%")

    frame_count += 1

    # Run ESRGAN X2
    return esrgan_x2(frame)
