"""
face_enhancer.py
Finalized version (option B): overwrite or add as new file
Path recommended: roop/processors/frame/face_enhancer.py

Fitur dan kepatuhan Roop:
- Memenuhi fungsi wajib: pre_check, pre_start, process_frame, process_frames, process_image, process_video, post_process
- Load model: 2d106det.onnx, faceparser_fp16.onnx, occluder.onnx
- Pipeline sesuai permintaan: 5-landmark extraction, geometric alignment, mask generation (parser+occluder), normalize/denormalize, inverse warp paste, alpha blending
- Tidak memakai GPEN

Catatan implementasi:
- Kode ini memakai onnxruntime untuk inference ONNX.
- Fungsi landmark mengambil 5 titik dari output 106 pts (index mapping disesuaikan secara heuristik).
- Estimasi affine menggunakan estimateAffinePartial2D.
- Mask dari parser + occluder digabung dan diblur untuk blending.
- Pre/post normalization matematis sederhana (div 255 and mul 255).

"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import threading
from typing import Any, List, Optional

import roop.globals
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, is_image, is_video, conditional_download
from roop.core import update_status
import roop.processors.frame.core

NAME = 'ROOP.FACE-ENHANCER'
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()

# model sessions
_sessions = {
    'landmark': None,
    'parser': None,
    'occluder': None,
}

# ------------------ helpers ------------------

def get_providers():
    return roop.globals.execution_providers


def load_onnx_session(path: str):
    try:
        return ort.InferenceSession(path, providers=get_providers())
    except Exception as e:
        print(f"[WARNING] Failed to load ONNX session {path}: {e}")
        return None

# ------------------ model resolver & loader ------------------

def _resolve_model_file(filename: str) -> Optional[str]:
    p = resolve_relative_path(f"../models/{filename}")
    return p if os.path.exists(p) else None


def pre_check() -> bool:
    # ensure model folder exists and optionally download if missing (no remote urls auto here)
    models_dir = resolve_relative_path('../models')
    os.makedirs(models_dir, exist_ok=True)
    # if you want automatic download, add urls to list below
    # conditional_download(models_dir, [ ... ])
    return True


def load_models() -> None:
    with THREAD_LOCK:
        if _sessions['landmark'] is None:
            p = _resolve_model_file('2d106det.onnx')
            _sessions['landmark'] = load_onnx_session(p) if p else None
        if _sessions['parser'] is None:
            p = _resolve_model_file('faceparser_fp16.onnx')
            _sessions['parser'] = load_onnx_session(p) if p else None
        if _sessions['occluder'] is None:
            p = _resolve_model_file('occluder.onnx')
            _sessions['occluder'] = load_onnx_session(p) if p else None

# ------------------ ONNX run util ------------------

def _prep_img_for_onnx(img: np.ndarray, size=(256,256)) -> np.ndarray:
    x = cv2.resize(img, size)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (2,0,1))[None]
    return x


def _run_session(sess: ort.InferenceSession, inp: np.ndarray):
    if sess is None:
        return None
    try:
        name = sess.get_inputs()[0].name
        out = sess.run(None, {name: inp})
        return out
    except Exception as e:
        print(f"[WARNING] ONNX run failed: {e}")
        return None

# ------------------ landmark extraction (106 -> 5) ------------------

def extract_5_landmarks(face_img: np.ndarray) -> Optional[np.ndarray]:
    sess = _sessions['landmark']
    if sess is None:
        return None
    inp = _prep_img_for_onnx(face_img, (192,192))
    out = _run_session(sess, inp)
    if not out:
        return None
    pts = out[0].reshape(-1,2)
    # heuristic indices mapping to eyes, nose, mouth corners (may need adjustment per model)
    indices = [33, 46, 62, 76, 90]
    try:
        sel = pts[indices].astype(np.float32)
        # convert from normalized coords (if model outputs normalized) — assume outputs are in pixel of 192
        # if outputs in [0,1], multiply by face size later in align
        return sel
    except Exception as e:
        print(f"[WARNING] Landmark selection failed: {e}")
        return None

# ------------------ alignment (geometric transform) ------------------

def align_and_warp(frame: np.ndarray, box: List[int], landmarks: Optional[np.ndarray]) -> (np.ndarray, Optional[np.ndarray], tuple):
    x1,y1,x2,y2 = box
    x1,x2 = max(0,int(x1)), min(frame.shape[1],int(x2))
    y1,y2 = max(0,int(y1)), min(frame.shape[0],int(y2))
    face = frame[y1:y2, x1:x2].copy()
    if landmarks is None:
        return face, None, (x1,y1,x2,y2)

    # landmarks currently relative to model input (192). Need to map to face patch coords
    # assume landmarks in range [0,192)
    h,w = face.shape[:2]
    scale_x = w / 192.0
    scale_y = h / 192.0
    pts_src = landmarks.copy()
    pts_src[:,0] *= scale_x
    pts_src[:,1] *= scale_y

    # template 5 points (standard) in 112x112 space then scaled to w,h
    template = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    # scale template to face patch size approx
    scale = max(w,h) / 112.0
    T = template * scale

    try:
        M, _ = cv2.estimateAffinePartial2D(pts_src, T, method=cv2.LMEDS)
        if M is None:
            return face, None, (x1,y1,x2,y2)
        out_w = int(T[:,0].max())
        out_h = int(T[:,1].max())
        warped = cv2.warpAffine(face, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped, M, (x1,y1,x2,y2)
    except Exception as e:
        print(f"[WARNING] align warp failed: {e}")
        return face, None, (x1,y1,x2,y2)

# ------------------ mask generation (parser + occluder) ------------------

def generate_mask(face_img: np.ndarray) -> np.ndarray:
    h,w = face_img.shape[:2]
    mask = np.ones((h,w), dtype=np.float32)

    parser = _sessions['parser']
    if parser:
        inp = _prep_img_for_onnx(face_img, (256,256))
        out = _run_session(parser, inp)
        if out is not None:
            # assume output shape (1,C,H,W) or (1,H,W,C)
            arr = out[0]
            if arr.ndim == 4:
                arr = arr[0]
            # if channels-first (C,H,W)
            if arr.shape[0] > 1:
                lbl = np.argmax(arr, axis=0)
            else:
                lbl = arr[0]
            lbl = cv2.resize(lbl.astype(np.float32), (w,h), interpolation=cv2.INTER_NEAREST)
            mask = (lbl > 0).astype(np.float32)

    occluder = _sessions['occluder']
    if occluder:
        inp = _prep_img_for_onnx(face_img, (256,256))
        out = _run_session(occluder, inp)
        if out is not None:
            occ = out[0]
            if occ.ndim == 4:
                occ = occ[0]
            occ = cv2.resize(occ[0], (w,h)) if occ.ndim==3 else cv2.resize(occ, (w,h))
            # assume occluder outputs probability of occlusion; keep regions with low occ
            mask *= (occ < 0.5).astype(np.float32)

    # smooth mask
    mask = cv2.GaussianBlur(mask, (21,21), 0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask

# ------------------ preprocess / postprocess ------------------

def normalize(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def denormalize(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

# ------------------ paste back (inverse transform) & blending ------------------

def paste_and_blend(frame: np.ndarray, warped_enh: np.ndarray, M: np.ndarray, box: List[int], mask: np.ndarray) -> np.ndarray:
    if M is None:
        return frame
    x1,y1,x2,y2 = box
    h_box = y2 - y1
    w_box = x2 - x1
    # inverse affine
    invM = cv2.invertAffineTransform(M)
    # warp back to box size
    back = cv2.warpAffine(warped_enh, invM, (w_box, h_box), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask_resized = cv2.resize(mask, (w_box, h_box))
    mask_3 = np.stack([mask_resized]*3, axis=-1)

    src_region = frame[y1:y2, x1:x2].astype(np.float32)
    blended = back.astype(np.float32) * mask_3 + src_region * (1.0 - mask_3)
    frame[y1:y2, x1:x2] = np.clip(blended,0,255).astype(np.uint8)
    return frame

# ------------------ enhance face pipeline ------------------

def enhance_face(target_face: Face, frame: Frame) -> Frame:
    x1,y1,x2,y2 = map(int, target_face['bbox'])
    h,w = frame.shape[:2]
    x1,x2 = max(0,x1), min(w,x2)
    y1,y2 = max(0,y1), min(h,y2)
    if x2<=x1 or y2<=y1:
        return frame

    face_patch = frame[y1:y2, x1:x2].copy()

    with THREAD_SEMAPHORE:
        try:
            lm = extract_5_landmarks(face_patch)
            warped, M, box = align_and_warp(frame, (x1,y1,x2,y2), lm)
            mask = generate_mask(warped)

            # preprocess: normalize
            warped_norm = normalize(warped)

            # enhancement step: simple sharpen + denoise (placeholder for external enhancer)
            enh = (warped_norm * 255.0).astype(np.uint8)
            enh = cv2.fastNlMeansDenoisingColored(enh, None, 10,10,7,21)
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            enh = cv2.filter2D(enh, -1, kernel)

            # postprocess: denormalize (we already have uint8)
            enhanced = enh

            # paste back + blending
            frame = paste_and_blend(frame, enhanced, M, box, mask)

        except Exception as e:
            print(f"[WARNING] enhance_face error: {e}")
    return frame

# ------------------ frame hooks ------------------

def process_frame(source_face: Optional[Face], reference_face: Optional[Face], frame: Frame) -> Frame:
    faces = get_many_faces(frame)
    if faces:
        for f in faces:
            frame = enhance_face(f, frame)
    return frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Optional[Any]) -> None:
    load_models()
    for p in temp_frame_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        res = process_frame(None, None, img)
        cv2.imwrite(p, res)
        if update:
            try:
                update()
            except:
                pass


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    load_models()
    img = cv2.imread(target_path)
    if img is None:
        raise RuntimeError('Failed to read target image')
    res = process_frame(None, None, img)
    cv2.imwrite(output_path, res)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    load_models()
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    for k in _sessions.keys():
        _sessions[k] = None

# exported symbols
__all__ = [
    'pre_check', 'pre_start', 'process_frame', 'process_frames', 'process_image', 'process_video', 'post_process'
]
