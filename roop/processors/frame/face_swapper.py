# ============================
#  FACE SWAPPER — PRO VERSION
# ============================

from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np
import math

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


# ===========================================================
#                  MODEL HANDLER
# ===========================================================
def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER


def clear_face_swapper():
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status("No face detected in source path.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process():
    clear_face_swapper()
    clear_face_reference()


# ===========================================================
#     DETAIL ANALYSIS — WRINKLE / DARK CIRCLE PRO ENGINE
# ===========================================================

# ----------- Perlin Noise PRO -----------

def _perlin_noise(h, w, scale=45.0, seed=0):
    np.random.seed(seed)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    nx = grid_x / scale
    ny = grid_y / scale
    noise = np.sin(nx * 2*np.pi) * np.cos(ny * 2*np.pi)
    noise = cv2.GaussianBlur(noise, (0,0), 3)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    return noise


# ---------- High-Frequency Wrinkle Map ----------
def _hf_wrinkle(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(lap)

    # Contrast normalization
    norm = cv2.normalize(abs_lap, None, 0, 1, cv2.NORM_MINMAX)

    # Remove noise
    norm = cv2.GaussianBlur(norm, (5,5), 0)

    return norm


# ---------- Edge Contrast Wrinkle Map ----------
def _edge_wrinkle(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(sobelx**2 + sobely**2)

    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    mag = cv2.GaussianBlur(mag, (7,7), 1.5)

    return mag


# ---------- Combine Wrinkle Map ----------
def _combine_wrinkle(gray):
    hf = _hf_wrinkle(gray)
    ed = _edge_wrinkle(gray)

    comb = (hf * 0.6) + (ed * 0.4)

    return np.clip(comb, 0, 1)


# ---------- Region Mask (Mata, Nasolabial, Pipi) ----------
def _region_mask(h, w):
    mask = np.zeros((h, w), dtype=np.float32)

    # Eye region
    y1, y2 = int(h*0.28), int(h*0.55)
    x1, x2 = int(w*0.18), int(w*0.82)
    mask[y1:y2, x1:x2] = 1.0

    # Cheek region
    y1, y2 = int(h*0.45), int(h*0.85)
    x1, x2 = int(w*0.10), int(w*0.90)
    mask[y1:y2, x1:x2] += 0.5

    mask = np.clip(mask, 0, 1)
    mask = cv2.GaussianBlur(mask, (21,21), 11)

    return mask


# ---------- Dark Circle PRO ----------
def _dark_circle_mask(h, w):
    mask = np.zeros((h, w), dtype=np.float32)

    cy = int(h * 0.45)
    cx = int(w * 0.5)

    for y in range(h):
        for x in range(w):
            dy = (y - cy) / (h * 0.20)
            dx = (x - cx) / (w * 0.50)
            d = math.sqrt(dx*dx + dy*dy)
            mask[y, x] = math.exp(-(d**2) * 3.5)

    mask = cv2.GaussianBlur(mask, (19,19), 8)
    mask = mask / mask.max()

    return mask


# ---------- Age → Strength Mapping ----------
def _age_strength(age):
    if age is None:
        return 0.0
    if age < 20:
        return 0.05
    if age < 30:
        return 0.15
    if age < 40:
        return 0.25
    if age < 50:
        return 0.35
    return 0.45


# ---------- MAIN WRINKLE ENGINE PRO ----------
def _apply_wrinkle_pro(source_face, target_face, swapped):
    try:
        age = getattr(source_face, "age", None)
        strength = _age_strength(age)

        if strength <= 0:
            return swapped

        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = swapped.shape[:2]

        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))

        crop = swapped[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        wrinkle = _combine_wrinkle(gray)
        region = _region_mask(ch, cw)

        wrinkle *= region

        perlin = _perlin_noise(ch, cw, scale=33.0, seed=age or 0)
        perlin = (perlin - 0.5) * 0.35
        wrinkle = wrinkle * 0.85 + perlin * 0.15

        dark_circle = _dark_circle_mask(ch, cw)
        dark_strength = strength * 35

        enhanced = crop.astype(np.float32)

        enhanced -= wrinkle[...,None] * (45 * strength)
        enhanced -= dark_circle[...,None] * dark_strength

        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(crop, 1.0, enhanced, strength, 0)

        swapped[y1:y2, x1:x2] = blended
        return swapped

    except Exception as e:
        print("WRINKLE PRO ERROR:", e)
        return swapped


# ===========================================================
#              POSE AWARE BBOX ADAPTATION
# ===========================================================

def adapt_bbox_for_pose(face: Face, frame_shape):

    pitch, yaw, roll = get_face_pose(face)

    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    if abs(yaw) > 25:
        extra = min((abs(yaw)-25) * 0.02, 0.22)
        pad_x = w * extra

    if pitch < -15:
        extra = min((abs(pitch)-15) * 0.02, 0.25)
        pad_y_top = h * extra
    elif pitch > 20:
        extra = min((pitch-20)*0.015, 0.20)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)



# ===========================================================
#                    CORE SWAP PROCESS
# ===========================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:

    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape)

    swapped = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )

    # ======= APPLY WRINKLE PRO ========
    if getattr(roop.globals, "preserve_wrinkle", True):
        swapped = _apply_wrinkle_pro(source_face, target_face, swapped)

    return swapped



# ===========================================================
#         FRAME PROCESSING ENGINE
# ===========================================================

def _select_best_target_by_embedding(faces, reference_face):
    if not faces or reference_face is None:
        return None

    ref_emb = getattr(reference_face, "normed_embedding", None)
    if ref_emb is None:
        return None

    best_face = None
    best_dist = float("inf")
    thr = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            d = np.sum(np.square(f.normed_embedding - ref_emb))
        except:
            continue

        if d < best_dist and d < thr:
            best_dist = d
            best_face = f

    return best_face


def process_frame(source_face, reference_face, temp_frame, frame_number=0):

    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for f in faces:
            if detect_occlusion(f, temp_frame):
                continue
            temp_frame = swap_face(source_face, f, temp_frame)

        return temp_frame

    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)

    tracked = [f for f in tracked if not detect_occlusion(f, temp_frame)]

    if not tracked:
        return temp_frame

    best = None
    if reference_face:
        best = _select_best_target_by_embedding(tracked, reference_face)

    if best is None:
        best = tracked[0]

    return swap_face(source_face, best, temp_frame)


def process_frames(source_path, temp_frame_paths, update):

    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for i, fpath in enumerate(temp_frame_paths):
        frame = cv2.imread(fpath)
        out = process_frame(source_face, reference_face, frame, i)
        cv2.imwrite(fpath, out)
        if update: update()


def process_image(source_path, target_path, output_path):

    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)
    reference_face = None

    if not roop.globals.many_faces:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)

    result = process_frame(source_face, reference_face, target_frame, 0)
    cv2.imwrite(output_path, result)


def process_video(source_path, temp_frame_paths):

    if not roop.globals.many_faces and not get_face_reference():
        try:
            idx = roop.globals.reference_frame_number
            rf = cv2.imread(temp_frame_paths[idx])
            ref = get_one_face(rf, roop.globals.reference_face_position)
            set_face_reference(ref)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
