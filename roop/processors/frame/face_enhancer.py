import cv2
import numpy as np
import threading
import os
import onnxruntime as ort
from collections import deque

import roop.globals
from roop.face_analyser import get_many_faces
from roop.utilities import resolve_relative_path, is_video
from roop.core import update_status

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
        kps = np.array(face.get("kps", []), dtype=np.float32)
        if kps.shape != (5, 2):
            return None, None
        M = cv2.estimateAffinePartial2D(kps, TEMPLATE_5PT, method=cv2.LMEDS)[0]
        if M is None:
            return None, None
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0)
        return aligned, M
    except Exception:
        return None, None


def paste_back(frame, enhanced, M):
    try:
        inv = cv2.invertAffineTransform(M)
        restored = cv2.warpAffine(enhanced, inv, (frame.shape[1], frame.shape[0]))
        return restored
    except Exception:
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
# MODEL LOADING (ONNX FP16 AUTODETECT)
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
                update_status(f"[{NAME}] Menggunakan FP16 ONNX: {os.path.basename(model_path)}", NAME)
            elif os.path.exists(full_path):
                model_path = full_path
                update_status(f"[{NAME}] Menggunakan ONNX Full: {os.path.basename(model_path)}", NAME)
            else:
                msg = "GFPGAN onnx tidak ditemukan. Jalankan konversi atau tempatkan file ONNX di folder models."
                update_status(msg, NAME)
                raise FileNotFoundError(msg)

            providers = roop.globals.execution_providers
            try:
                ENHANCER = ort.InferenceSession(model_path, providers=providers)
            except Exception as e:
                update_status(f"[{NAME}] Gagal load ONNX: {e}", NAME)
                raise

    return ENHANCER


# ---------------------------------------------------------------------
# ENHANCE PER WAJAH
# ---------------------------------------------------------------------
def enhance_face(frame, face):
    aligned, M = align_face(frame, face)
    if aligned is None or M is None:
        return frame

    inp = normalize_input(aligned)
    enhancer = None
    try:
        enhancer = get_enhancer()
    except FileNotFoundError:
        return frame
    except Exception as e:
        update_status(f"[{NAME}] get_enhancer error: {e}", NAME)
        return frame

    with THREAD_SEMAPHORE:
        try:
            out = enhancer.run(None, {"input": inp})[0]
        except Exception as e:
            update_status(f"[{NAME}] Inference ONNX gagal: {e}", NAME)
            return frame

    enhanced = denormalize_output(out)
    enhanced = color_correction(aligned, enhanced)
    blended = (enhanced.astype(np.float32) * SOFT_MASK + aligned.astype(np.float32) * (1 - SOFT_MASK)).astype(np.uint8)
    restored = paste_back(frame, blended, M)
    return restored


# ---------------------------------------------------------------------
# TEMPORAL SMOOTHING
# ---------------------------------------------------------------------
def smooth_frame(frame):
    FRAME_SMOOTH.append(frame)
    if len(FRAME_SMOOTH) == 0:
        return frame
    return np.mean(list(FRAME_SMOOTH), axis=0).astype(np.uint8)


# ---------------------------------------------------------------------
# FRAME PROCESSING CORE
# ---------------------------------------------------------------------
def process_frame(source_face, reference_face, frame):
    """
    Dipanggil untuk setiap frame oleh process_frames.
    source_face / reference_face disediakan oleh pipeline (boleh None).
    """
    try:
        faces = get_many_faces(frame)
    except Exception as e:
        update_status(f"[{NAME}] face detection error: {e}", NAME)
        return frame

    if not faces:
        return frame

    for f in faces:
        try:
            frame = enhance_face(frame, f)
        except Exception as e:
            # catat warning tapi jangan crash pipeline
            update_status(f"[{NAME}] enhance warning: {e}", NAME)

    # smoothing hanya untuk video
    if is_video(roop.globals.target_path):
        frame = smooth_frame(frame)

    return frame


# ---------------------------------------------------------------------
# API YANG HARUS ADA UNTUK FRAME PROCESSOR (CORE)
# ---------------------------------------------------------------------
def pre_check() -> bool:
    """
    Dipanggil sebelum pipeline dimulai untuk memastikan model tersedia.
    """
    base_dir = resolve_relative_path("../models")
    fp16_path = os.path.join(base_dir, "GFPGANv1.4-fp16.onnx")
    full_path = os.path.join(base_dir, "GFPGANv1.4.onnx")
    if os.path.exists(fp16_path) or os.path.exists(full_path):
        return True
    update_status(f"[{NAME}] Model ONNX tidak ditemukan di folder models.", NAME)
    return False


def pre_start() -> bool:
    """
    Validasi target path (image/video) sebelum start.
    """
    if not (os.path.exists(roop.globals.target_path)):
        update_status(f"[{NAME}] Target path tidak ada: {roop.globals.target_path}", NAME)
        return False
    return True


def post_process() -> None:
    """
    Bersihkan resource setelah selesai.
    """
    global ENHANCER
    with THREAD_LOCK:
        ENHANCER = None
    FRAME_SMOOTH.clear()
    update_status(f"[{NAME}] post_process complete.", NAME)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Dipanggil saat pipeline memproses single image.
    """
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            update_status(f"[{NAME}] Gagal baca image: {target_path}", NAME)
            return
        result = process_frame(None, None, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        update_status(f"[{NAME}] process_image error: {e}", NAME)


def process_frames(source_path: str, temp_frame_paths: list, update: callable = None) -> None:
    """
    Dipanggil oleh core processor untuk memproses list frame sementara.
    - source_path: path sumber asli (bisa None)
    - temp_frame_paths: list path file frame sementara (ordered)
    - update: callable untuk melaporkan progress (boleh None)
    """
    for idx, temp_path in enumerate(temp_frame_paths):
        try:
            frame = cv2.imread(temp_path)
            if frame is None:
                update_status(f"[{NAME}] Gagal baca frame: {temp_path}", NAME)
                continue
            result = process_frame(None, None, frame)
            cv2.imwrite(temp_path, result)
        except Exception as e:
            update_status(f"[{NAME}] process_frames error ({temp_path}): {e}", NAME)
        finally:
            if update:
                try:
                    update(idx, len(temp_frame_paths))
                except TypeError:
                    # beberapa core pass update() tanpa args
                    try:
                        update()
                    except Exception:
                        pass


def process_video(source_path: str, temp_frame_paths: list) -> None:
    """
    Integrasi dengan roop.processors.frame.core.process_video
    Pastikan memanggil core.process_video dengan signature (source_path, temp_frame_paths, processor_callback)
    """
    # import local reference to avoid shadowing function name
    from roop.processors.frame import core as frame_core
    try:
        frame_core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        update_status(f"[{NAME}] process_video error: {e}", NAME)
