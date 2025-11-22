import cv2
import numpy as np
import threading
import os
import onnxruntime as ort
from collections import deque

import roop.globals
from roop.face_analyser import get_many_faces
from roop.utilities import resolve_relative_path, is_video

NAME = "ROOP.FACE-ENHANCER"

# ---------------------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------------------
ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore(2)
FRAME_SMOOTH = deque(maxlen=5)

# Template landmark 5-point
TEMPLATE_5PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Soft mask untuk blending
def build_soft_mask(size=112):
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask, (size//2, size//2), size//2 - 8, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return cv2.merge([mask, mask, mask])

SOFT_MASK = build_soft_mask()


# ---------------------------------------------------------------------
# ALIGNMENT WAJAH
# ---------------------------------------------------------------------
def align_face(frame, face):
    try:
        kps = np.array(face["kps"], dtype=np.float32)
        M = cv2.estimateAffinePartial2D(kps, TEMPLATE_5PT, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0)
        return aligned, M
    except:
        return None, None


def paste_back(frame, enhanced, M):
    try:
        inv = cv2.invertAffineTransform(M)
        restored = cv2.warpAffine(enhanced, inv, (frame.shape[1], frame.shape[0]))
        return restored
    except:
        return frame


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------
def normalize_input(img):
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None]


def denormalize_output(out):
    out = np.transpose(out[0], (1, 2, 0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# COLOR CORRECTION
# ---------------------------------------------------------------------
def color_correction(src, dst):
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    for i in range(3):
        s_mean, s_std = src[..., i].mean(), src[..., i].std()
        d_mean, d_std = dst[..., i].mean(), dst[..., i].std()
        dst[..., i] = (dst[..., i] - d_mean) * (s_std / (d_std + 1e-6)) + s_mean
    return np.clip(dst, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# LOAD ENHANCER (FP16 → ONNX → ERROR)
# ---------------------------------------------------------------------
def get_enhancer():
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:

            base_dir = resolve_relative_path("../models")

            fp16_path = os.path.join(base_dir, "GFPGANv1.4-fp16.onnx")
            full_path = os.path.join(base_dir, "GFPGANv1.4.onnx")

            if os.path.exists(fp16_path):
                model_path = fp16_path
                print(f"[FACE-ENHANCER] Menggunakan FP16 ONNX: {model_path}")
            elif os.path.exists(full_path):
                model_path = full_path
                print(f"[FACE-ENHANCER] FP16 tidak ada, memakai ONNX FULL: {model_path}")
            else:
                raise FileNotFoundError(
                    "Tidak menemukan GFPGANv1.4.onnx atau GFPGANv1.4-fp16.onnx!"
                )

            providers = roop.globals.execution_providers
            ENHANCER = ort.InferenceSession(model_path, providers=providers)

    return ENHANCER


# ---------------------------------------------------------------------
# ENHANCE PER WAJAH
# ---------------------------------------------------------------------
def enhance_face(frame, face):
    aligned, M = align_face(frame, face)
    if aligned is None:
        return frame

    inp = normalize_input(aligned)
    enhancer = get_enhancer()

    with THREAD_SEMAPHORE:
        try:
            out = enhancer.run(None, {"input": inp})[0]
        except Exception as e:
            print(f"[ERROR] Inference ONNX gagal: {e}")
            return frame

    enhanced = denormalize_output(out)
    enhanced = color_correction(aligned, enhanced)

    blended = enhanced * SOFT_MASK + aligned * (1 - SOFT_MASK)
    blended = blended.astype(np.uint8)

    restored = paste_back(frame, blended, M)
    return restored


# ---------------------------------------------------------------------
# TEMPORAL SMOOTHING
# ---------------------------------------------------------------------
def smooth_frame(frame):
    FRAME_SMOOTH.append(frame)
    return np.mean(FRAME_SMOOTH, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------
# FRAME PROCESSING
# ---------------------------------------------------------------------
def process_frame(source_face, reference_face, frame):
    faces = get_many_faces(frame)
    if not faces:
        return frame

    for f in faces:
        try:
            frame = enhance_face(frame, f)
        except Exception as e:
            print(f"[WARNING] Enhance error: {e}")

    if is_video(roop.globals.target_path):
        frame = smooth_frame(frame)

    return frame


# ---------------------------------------------------------------------
# IMAGE / VIDEO API
# ---------------------------------------------------------------------
def process_image(source_path, target_path, output_path):
    img = cv2.imread(target_path)
    out = process_frame(None, None, img)
    cv2.imwrite(output_path, out)


def process_frames(source_path, temp_frame_paths, update):
    for p in temp_frame_paths:
        try:
            frame = cv2.imread(p)
            result = process_frame(None, None, frame)
            cv2.imwrite(p, result)
        except Exception as e:
            print(f"[FRAME ERROR] {e}")
        if update:
            update()


def process_video(source_path, temp_frame_paths):
    from roop.processors.frame.core import process_video
    process_video(None, temp_frame_paths, process_frames)
