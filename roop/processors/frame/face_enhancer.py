import cv2
import numpy as np
import threading
import os
import onnxruntime as ort
from collections import deque

import roop.globals
from roop.face_analyser import get_many_faces
from roop.utilities import is_video
from roop.core import update_status

NAME = "ROOP.FACE-ENHANCER"

# ------------------------------------------------------------
# GLOBAL
# ------------------------------------------------------------
ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore(2)
FRAME_SMOOTH = deque(maxlen=5)


# ------------------------------------------------------------
# TEMPLATE ALIGN 5-POINT
# ------------------------------------------------------------
TEMPLATE_5PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


# ------------------------------------------------------------
# SOFT MASK
# ------------------------------------------------------------
def build_soft_mask(size=112):
    m = np.zeros((size, size), dtype=np.float32)
    cv2.circle(m, (size//2, size//2), size//2 - 8, 1.0, -1)
    m = cv2.GaussianBlur(m, (41, 41), 0)
    return cv2.merge([m, m, m])

SOFT_MASK = build_soft_mask()


# ------------------------------------------------------------
# ALIGNMENT
# ------------------------------------------------------------
def align_face(frame, face):
    try:
        kps = np.array(face.get("kps", []), dtype=np.float32)
        if kps.shape != (5, 2):
            return None, None

        M = cv2.estimateAffinePartial2D(kps, TEMPLATE_5PT, method=cv2.LMEDS)[0]
        if M is None:
            return None, None

        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0)
        return aligned, M
    except:
        return None, None


def paste_back(frame, enhanced, M):
    try:
        inv = cv2.invertAffineTransform(M)
        return cv2.warpAffine(enhanced, inv, (frame.shape[1], frame.shape[0]))
    except:
        return frame


# ------------------------------------------------------------
# NORMALISASI
# ------------------------------------------------------------
def normalize_input(img):
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None]


def denormalize_output(out):
    out = np.transpose(out[0], (1, 2, 0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# COLOR CORRECTION
# ------------------------------------------------------------
def color_correction(src, dst):
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    for i in range(3):
        s_mean, s_std = src[..., i].mean(), src[..., i].std()
        d_mean, d_std = dst[..., i].mean(), dst[..., i].std()
        dst[..., i] = (dst[..., i] - d_mean) * (s_std / (d_std + 1e-6)) + s_mean
    return np.clip(dst, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# MODEL LOADER — PRIORITAS: [project]/roop/models
# ------------------------------------------------------------
def get_model_paths():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

    return [
        base,                                                  # PRIORITAS
        "/kaggle/working/Learning/roop/models",
        "/kaggle/working/Learning/models",
    ]


def get_enhancer():
    global ENHANCER

    with THREAD_LOCK:
        if ENHANCER is None:

            model_path = None
            dirs = get_model_paths()

            for d in dirs:
                try:
                    files = os.listdir(d)
                except:
                    continue

                fp16 = os.path.join(d, "GFPGANv1.4-fp16.onnx")
                full = os.path.join(d, "GFPGANv1.4.onnx")

                if os.path.exists(fp16):
                    model_path = fp16
                    update_status(f"[{NAME}] Memakai FP16 ONNX: {model_path}", NAME)
                    break

                if os.path.exists(full):
                    model_path = full
                    update_status(f"[{NAME}] Memakai ONNX FULL: {model_path}", NAME)
                    break

            if model_path is None:
                raise FileNotFoundError(
                    f"Tidak menemukan GFPGANv1.4 ONNX di:\n" +
                    "\n".join(dirs)
                )

            providers = roop.globals.execution_providers
            ENHANCER = ort.InferenceSession(model_path, providers=providers)

    return ENHANCER


# ------------------------------------------------------------
# ENHANCE
# ------------------------------------------------------------
def enhance_face(frame, face):
    aligned, M = align_face(frame, face)
    if aligned is None:
        return frame

    enhancer = get_enhancer()
    inp = normalize_input(aligned)

    with THREAD_SEMAPHORE:
        try:
            # FP16 MODE
            inp16 = inp.astype(np.float16)
            out = enhancer.run(None, {"input": inp16})[0]
        except:
            try:
                # FALLBACK FLOAT32
                inp32 = inp.astype(np.float32)
                out = enhancer.run(None, {"input": inp32})[0]
            except Exception as e:
                update_status(f"[{NAME}] Inference gagal: {e}", NAME)
                return frame

    enhanced = denormalize_output(out)
    enhanced = color_correction(aligned, enhanced)

    blended = (enhanced.astype(np.float32) * SOFT_MASK +
               aligned.astype(np.float32) * (1 - SOFT_MASK)).astype(np.uint8)

    restored = paste_back(frame, blended, M)
    return restored


# ------------------------------------------------------------
# SMOOTHING
# ------------------------------------------------------------
def smooth_frame(frame):
    FRAME_SMOOTH.append(frame)
    return np.mean(FRAME_SMOOTH, axis=0).astype(np.uint8)


# ------------------------------------------------------------
# MAIN PER-FRAME
# ------------------------------------------------------------
def process_frame(source_face, reference_face, frame):
    try:
        faces = get_many_faces(frame)
    except Exception as e:
        update_status(f"[{NAME}] Face detect error: {e}", NAME)
        return frame

    if not faces:
        return frame

    for f in faces:
        frame = enhance_face(frame, f)

    if is_video(roop.globals.target_path):
        frame = smooth_frame(frame)

    return frame


# ------------------------------------------------------------
# PIPELINE HOOKS
# ------------------------------------------------------------
def pre_check():
    try:
        get_enhancer()
        return True
    except:
        return False


def pre_start():
    if not os.path.exists(roop.globals.target_path):
        update_status(f"[{NAME}] Target tidak ditemukan.", NAME)
        return False
    return True


def post_process():
    global ENHANCER
    with THREAD_LOCK:
        ENHANCER = None
    FRAME_SMOOTH.clear()
    update_status(f"[{NAME}] post_process selesai.", NAME)


# ------------------------------------------------------------
# PROCESS IMAGE
# ------------------------------------------------------------
def process_image(source_path, target_path, output_path):
    frame = cv2.imread(target_path)
    if frame is None:
        update_status(f"[{NAME}] Gagal membaca image.", NAME)
        return
    out = process_frame(None, None, frame)
    cv2.imwrite(output_path, out)


# ------------------------------------------------------------
# PROCESS FRAMES
# ------------------------------------------------------------
def process_frames(source_path, temp_frame_paths, update=None):
    total = len(temp_frame_paths)

    for idx, p in enumerate(temp_frame_paths):
        try:
            frame = cv2.imread(p)
            if frame is None:
                continue
            result = process_frame(None, None, frame)
            cv2.imwrite(p, result)
        except Exception as e:
            update_status(f"[{NAME}] Error frame {e}", NAME)

        if update:
            try:
                update(idx, total)
            except:
                try:
                    update()
                except:
                    pass


# ------------------------------------------------------------
# PROCESS VIDEO
# ------------------------------------------------------------
def process_video(source_path, temp_frame_paths):
    from roop.processors.frame import core as frame_core
    frame_core.process_video(source_path, temp_frame_paths, process_frames)
