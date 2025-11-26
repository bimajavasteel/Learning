# ============================================================
#   FACE SWAPPER — FINAL VERSION (with last_swap_mask integration)
#   - Menyimpan roop.globals.last_swap_mask (full-frame uint8 mask)
#   - Aman untuk multiprocessing/threading (pakai lock)
# ============================================================

from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np
import os

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

# lock untuk update last_swap_mask agar thread-safe
SWAP_MASK_LOCK = threading.Lock()


# ============================================================
#   MODEL LOADING (INSWAPPER)
# ============================================================

DEFAULT_INSWAPPER = getattr(
    roop.globals,
    "inswapper_model",
    "../models/inswapper_128.onnx"
)

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(DEFAULT_INSWAPPER)
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
            print(f"✅ [face_swapper] Loaded model: {model_path}")
    return FACE_SWAPPER


def clear_face_swapper() -> None:
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
        update_status('Select an image for source path.', NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status('No face detected in source image.', NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video target.', NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()



# ============================================================
#   HAIR MASK BUILDER
# ============================================================

def landmarks_to_hair_mask(frame_shape, landmarks, scale=1.15):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if landmarks is None or len(landmarks) == 0:
        return mask

    lm = np.array(landmarks, dtype=np.int32)

    try:
        if lm.shape[0] >= 68:
            jaw = lm[0:17]
            brow1 = lm[17:22]
            brow2 = lm[22:27]
            poly = np.vstack([jaw, brow1, brow2[::-1]])
        else:
            poly = lm
    except:
        poly = lm

    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tmp, [poly], 255)

    x, y, ww, hh = cv2.boundingRect(poly)
    cx, cy = x + ww // 2, y + hh // 2
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    mask_scaled = cv2.warpAffine(tmp, M, (w, h))

    blur = int(max(7, (h + w) * 0.015))
    if blur % 2 == 0:
        blur += 1

    mask_blur = cv2.GaussianBlur(mask_scaled.astype(np.float32)/255.0, (blur, blur), 0)
    return (mask_blur * 255).astype(np.uint8)



# ============================================================
#   BOUNDING BOX ADJUSTMENT (POSE-AWARE)
# ============================================================

def adapt_bbox_for_pose(face: Face, frame_shape):
    pitch, yaw, roll = get_face_pose(face)
    H, W = frame_shape[:2]

    x1, y1, x2, y2 = map(float, face.bbox)
    w = x2 - x1
    h = y2 - y1

    pad_x = 0
    pad_y_top = 0
    pad_y_bottom = 0

    # YAW besar -> tambah padding horizontal
    if abs(yaw) > 15:
        extra = (abs(yaw) - 15) * 0.03
        extra = min(extra, 0.45)
        pad_x = w * extra

    # PITCH -> lihat atas/bawah
    if pitch < -10:
        extra = (abs(pitch) - 10) * 0.03
        pad_y_top = min(extra, 0.45) * h
    elif pitch > 12:
        extra = (pitch - 12) * 0.02
        pad_y_bottom = min(extra, 0.25) * h

    # Selalu tambah area rambut
    pad_y_top += h * 0.18

    # Perbesar crop keseluruhan
    s = 0.08
    pad_x += w * s
    pad_y_top += h * s
    pad_y_bottom += h * s

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(W - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(H - 1, y2 + pad_y_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)



# ============================================================
#   SWAP FACE (CORE FUNCTION)
#   - Juga meng-update roop.globals.last_swap_mask (full-frame uint8)
# ============================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:

    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape)

    swapper = get_face_swapper()

    swapped = swapper.get(
        temp_frame.copy(),
        target_face,
        source_face,
        paste_back=True
    )

    # ------------------------------------------------------------
    # SAFE LANDMARK EXTRACTION
    # ------------------------------------------------------------
    landmarks = None
    for key in ["landmark_2d_106", "landmark_2d_68", "landmark"]:
        lm = getattr(target_face, key, None)
        if lm is not None and hasattr(lm, "__len__") and len(lm) > 0:
            landmarks = lm
            break


    # ------------------------------------------------------------
    # BUILD MASK (alpha: single-channel [0..1])
    # ------------------------------------------------------------
    if landmarks is not None:
        lms = np.array(landmarks)
        if lms.max() <= 1.1:  # relative coords
            H, W = temp_frame.shape[:2]
            lms = (lms * np.array([W, H])).astype(np.int32)

        mask_u8 = landmarks_to_hair_mask(temp_frame.shape, lms, scale=1.15)
        alpha = mask_u8.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (41, 41), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    else:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        mask = np.zeros(temp_frame.shape[:2], dtype=np.float32)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        axes = (int((x2-x1)*0.55), int((y2-y1)*0.65))
        cv2.ellipse(mask, (cx,cy), axes, 0, 0, 360, 1.0, -1)
        alpha = cv2.GaussianBlur(mask, (31,31), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    # ------------------------------------------------------------
    # OCCLUSION-AWARE BLENDING (intensity scalar)
    # ------------------------------------------------------------
    occluded = detect_occlusion(target_face, temp_frame)
    intensity = 1.0
    if occluded:
        intensity = 0.45
    else:
        det = getattr(target_face, "det_score", 1.0)
        if det < 0.55:
            intensity = 0.85

    # alpha_single adalah mask single-channel final (0..1) untuk area ini
    alpha_single = np.clip(alpha * intensity, 0.0, 1.0)
    alpha_final = np.expand_dims(alpha_single, axis=2)  # untuk compositing 3-channel

    # ------------------------------------------------------------
    # FINAL COMPOSITING
    # ------------------------------------------------------------
    result = (
        swapped.astype(np.float32) * alpha_final +
        temp_frame.astype(np.float32) * (1.0 - alpha_final)
    ).astype(np.uint8)

    # ------------------------------------------------------------
    # UPDATE roop.globals.last_swap_mask (uint8 full-frame)
    # - gunakan lock supaya aman multi-thread
    # - akumulasi dengan np.maximum sehingga beberapa wajah ter-cover
    # ------------------------------------------------------------
    try:
        mask_uint8 = (alpha_single * 255.0).astype(np.uint8)
        with SWAP_MASK_LOCK:
            existing = getattr(roop.globals, "last_swap_mask", None)
            if existing is None or existing.shape != mask_uint8.shape:
                # inisialisasi ulang kalau belum ada atau ukuran beda
                roop.globals.last_swap_mask = np.zeros_like(mask_uint8, dtype=np.uint8)
                existing = roop.globals.last_swap_mask
            # akumulasi area (ambil nilai maksimum per pixel)
            roop.globals.last_swap_mask = np.maximum(existing, mask_uint8)
    except Exception:
        # jangan ganggu pipeline kalau update mask gagal
        pass

    return result



# ============================================================
#   SELECT BEST FACE (SINGLE MODE)
# ============================================================

def _select_best_target_by_embedding(faces: List[Face], reference_face: Face):
    if not faces or reference_face is None:
        return None
    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref = reference_face.normed_embedding
    best = None
    best_dist = float("inf")
    thr = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            d = np.sum((f.normed_embedding - ref)**2)
        except:
            continue

        if d < thr and d < best_dist:
            best_dist = d
            best = f

    return best



# ============================================================
#   PROCESS EACH FRAME
#   - init roop.globals.last_swap_mask per frame
# ============================================================

def process_frame(source_face, reference_face, temp_frame, frame_number=0):

    if source_face is None:
        return temp_frame

    # inisialisasi last_swap_mask full-frame untuk frame ini (uint8)
    try:
        h, w = temp_frame.shape[:2]
        with SWAP_MASK_LOCK:
            roop.globals.last_swap_mask = np.zeros((h, w), dtype=np.uint8)
    except Exception:
        # ignore failure to init mask (pipeline tetap jalan)
        pass

    # MULTI FACE MODE
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for f in faces:
            temp_frame = swap_face(source_face, f, temp_frame)

        return temp_frame

    # SINGLE FACE MODE
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        faces = get_many_faces(temp_frame)

    if not faces:
        return temp_frame

    valid = [f for f in faces if not detect_occlusion(f, temp_frame)]
    if not valid:
        valid = faces

    best = None
    if reference_face:
        best = _select_best_target_by_embedding(valid, reference_face)
    if best is None:
        best = valid[0]

    temp_frame = swap_face(source_face, best, temp_frame)
    return temp_frame



# ============================================================
#   PROCESS FRAME LIST (VIDEO MODE)
# ============================================================

def process_frames(source_path, temp_frame_paths, update):
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, path in enumerate(temp_frame_paths):
        frame = cv2.imread(path)
        out = process_frame(source_face, reference_face, frame, idx)
        cv2.imwrite(path, out)
        if update:
            update()



# ============================================================
#   PROCESS IMAGE
# ============================================================

def process_image(source_path, target_path, output_path):
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    source_face = get_one_face(source_img)
    reference_face = None

    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_img,
            roop.globals.reference_face_position
        )

    result = process_frame(source_face, reference_face, target_img)
    cv2.imwrite(output_path, result)



# ============================================================
#   PROCESS VIDEO
# ============================================================

def process_video(source_path, temp_frame_paths):

    if not roop.globals.many_faces and not get_face_reference():
        try:
            idx = roop.globals.reference_frame_number
            ref_frame = cv2.imread(temp_frame_paths[idx])
            ref_face = get_one_face(
                ref_frame,
                roop.globals.reference_face_position
            )
            set_face_reference(ref_face)
        except:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
