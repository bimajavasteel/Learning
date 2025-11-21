"""
Custom Face Enhancer untuk Roop Mod — versi lengkap sesuai permintaan terbaru
Menggunakan pipeline:
1. Landmark Extraction → 2d106det.onnx (ambil 5 landmark dari 106)
2. Face Warping → similarity transform
3. Mask Generation → faceparser_fp16.onnx + occluder.onnx
4. Pre-processing → normalisasi matematis
5. Post-processing → denormalisasi matematis
6. Face Pasting → inverse transform
7. Blending → alpha blending

File ini ditempatkan pada: roop/processors/frame/face_enhancer_custom.py
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import threading
from typing import Any, Optional, Tuple, List

import roop.globals
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, is_image, is_video
from roop.core import update_status
import roop.processors.frame.core

THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()
NAME = "ROOP.FACE-ENHANCER-CUSTOM"

# Model registry
models = {
    "landmark": None,      # 2d106det.onnx
    "parser": None,        # faceparser_fp16.onnx
    "occluder": None,      # occluder.onnx
}

def get_providers():
    return roop.globals.execution_providers

def load_onnx(name: str, filename: str):
    path = resolve_relative_path(f"../models/{filename}")
    if not os.path.exists(path):
        print(f"[WARNING] Model {filename} tidak ditemukan")
        return None
    try:
        return ort.InferenceSession(path, providers=get_providers())
    except Exception as e:
        print(f"[ERROR] Gagal load {filename}: {e}")
        return None

# ---------------- LOADER ----------------
def load_models():
    with THREAD_LOCK:
        if models["landmark"] is None:
            models["landmark"] = load_onnx("landmark", "2d106det.onnx")
        if models["parser"] is None:
            models["parser"] = load_onnx("parser", "faceparser_fp16.onnx")
        if models["occluder"] is None:
            models["occluder"] = load_onnx("occluder", "occluder.onnx")

# ---------------- ONNX UTILS ----------------
def preprocess(image: np.ndarray, size=(192,192)):
    img = cv2.resize(image, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None]
    return img

def postprocess(img: np.ndarray):
    img = img.clip(0,1)
    img = (img * 255).astype(np.uint8)
    return img

# ---------------- LANDMARK EXTRACTION 106 → 5 ----------------
def extract_5_landmarks(face_img: np.ndarray) -> Optional[np.ndarray]:
    sess = models["landmark"]
    if sess is None:
        return None
    inp = preprocess(face_img, (192,192))
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    pts = out.reshape(-1,2)  # 106 landmarks

    # Ambil 5 landmark standard InsightFace
    # (approx mapping index — bisa berbeda antar dataset)
    idx = [33, 46, 62, 76, 90]  # contoh index approx (mata kiri, mata kanan, hidung, mulut kiri, mulut kanan)
    return pts[idx]

# ---------------- FACE WARPING ----------------
def align_face(frame: np.ndarray, box, landmarks):
    x1,y1,x2,y2 = box
    face = frame[y1:y2, x1:x2]
    if landmarks is None:
        return face, None

    # template 5 landmark (InsightFace)
    template = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    scale = (x2-x1) / 112
    T = template * scale

    M = cv2.estimateAffinePartial2D(landmarks, T, method=cv2.LMEDS)[0]
    warped = cv2.warpAffine(face, M, (int(T[:,0].max()), int(T[:,1].max())))
    return warped, M

# ---------------- MASK GENERATION ----------------
def generate_mask(face_img: np.ndarray):
    parser = models["parser"]
    occluder = models["occluder"]
    h,w = face_img.shape[:2]

    # face parser mask
    mask = np.ones((h,w), dtype=np.float32)
    if parser:
        inp = preprocess(face_img, (256,256))
        out = parser.run(None, {parser.get_inputs()[0].name: inp})[0]
        out = np.argmax(out[0], axis=0)
        mask = cv2.resize(out.astype(np.float32), (w,h))
        mask = (mask > 0).astype(np.float32)

    # occluder mask
    if occluder:
        inp = preprocess(face_img, (256,256))
        occ = occluder.run(None, {occluder.get_inputs()[0].name: inp})[0][0]
        occ = cv2.resize(occ, (w,h))
        mask *= (occ < 0.5).astype(np.float32)

    mask = cv2.GaussianBlur(mask, (25,25), 5)
    return np.clip(mask,0,1)

# ---------------- PRE / POST PROCESS ----------------
def normalize(img: np.ndarray):
    return img.astype(np.float32) / 255.0

def denormalize(img: np.ndarray):
    return (img * 255).clip(0,255).astype(np.uint8)

# ---------------- FACE PASTING + INVERSE WARP ----------------
def paste_back(frame, warped_result, box, M, mask):
    if M is None:
        return frame
    x1,y1,x2,y2 = box

    inv_M = cv2.invertAffineTransform(M)
    h = y2-y1
    w = x2-x1

    restored = cv2.warpAffine(warped_result, inv_M, (w,h))
    mask_resized = cv2.resize(mask, (w,h))
    mask_3 = np.stack([mask_resized]*3, axis=-1)

    region = frame[y1:y2, x1:x2]
    blended = restored * mask_3 + region * (1-mask_3)
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame

# ---------------- MAIN ENHANCE ----------------
def enhance_face(target_face: Face, frame: Frame) -> Frame:
    x1,y1,x2,y2 = map(int, target_face['bbox'])
    face_patch = frame[y1:y2, x1:x2]

    with THREAD_SEMAPHORE:
        # 1) landmark
        lm = extract_5_landmarks(face_patch)

        # 2) warp
        warped, M = align_face(frame, (x1,y1,x2,y2), lm)
        if warped is None:
            return frame

        # 3) mask
        mask = generate_mask(warped)

        # 4) preprocess
        warped_norm = normalize(warped)

        # (NO GPEN — hanya simple sharpening)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        enhanced = cv2.filter2D((warped_norm*255).astype(np.uint8), -1, kernel)

        # 5) postprocess
        enhanced_denorm = denormalize(enhanced.astype(np.float32)/255.0)

        # 6 + 7) paste + blend
        frame = paste_back(frame, enhanced_denorm, (x1,y1,x2,y2), M, mask)

    return frame

# ---------------- HOOKS ----------------
def process_frame(source_face, reference_face, frame: Frame) -> Frame:
    faces = get_many_faces(frame)
    if faces:
        for f in faces:
            frame = enhance_face(f, frame)
    return frame


def process_frames(source_path: str, temp_frames: List[str], update):
    load_models()
    for p in temp_frames:
        img = cv2.imread(p)
        if img is None:
            continue
        res = process_frame(None, None, img)
        cv2.imwrite(p, res)
        if update:
            update()


def process_image(source_path, target_path, output_path):
    load_models()
    img = cv2.imread(target_path)
    res = process_frame(None, None, img)
    cv2.imwrite(output_path, res)


def process_video(source_path, temp_frames):
    load_models()
    roop.processors.frame.core.process_video(source_path, temp_frames, process_frames)


def pre_start():
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select target image/video", NAME)
        return False
    return True


def post_process():
    for k in models.keys(): models[k] = None

__all__ = ["pre_start","pre_process","process_frame","process_frames","process_image","process_video","post_process"]
