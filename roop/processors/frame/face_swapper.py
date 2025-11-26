# face-swppe-new.py
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

# Config: ganti ke inswapper_256 jika tersedia untuk kualitas lebih baik
DEFAULT_INSWAPPER = getattr(roop.globals, "inswapper_model", "../models/inswapper_128.onnx")

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(getattr(roop.globals, "inswapper_model", DEFAULT_INSWAPPER))
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
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
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

# ---------------------------
# Utilities: hair mask + feather
# ---------------------------
def landmarks_to_hair_mask(frame_shape, landmarks, scale=1.15):
    """
    Buat soft mask berdasarkan landmarks 2D (landmarks 106/68).
    - scale: perbesar untuk menangkap rambut atas kepala.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks is None or len(landmarks) == 0:
        return mask
    lm = np.array(landmarks, dtype=np.int32)
    # gunakan polygon dari jawline + brow + bagian atas kepala perkiraan
    # ambil jawline (0..16 for 68) dan brow (17..26) jika tersedia
    try:
        # fallback jika 106 panjang: gunakan subset
        if lm.shape[0] >= 68:
            jaw = lm[0:17]
            left_brow = lm[17:22]
            right_brow = lm[22:27]
            eyes = lm[36:48] if lm.shape[0] >= 48 else []
        else:
            jaw = lm
            left_brow = []
            right_brow = []
            eyes = []
        poly = np.vstack([jaw, left_brow, right_brow[::-1]])
    except Exception:
        poly = lm

    # compute centroid and expand
    r = cv2.boundingRect(poly)
    x, y, ww, hh = r
    cx, cy = x + ww // 2, y + hh // 2
    # expand polygon outward by scale (simple dilation via resize of bounding mask)
    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tmp, [poly], 255)
    # dilate then resize to simulate scale
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    mask_scaled = cv2.warpAffine(tmp, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # feather
    blur = int(max(7, min(h, w) * 0.02))
    if blur % 2 == 0: blur += 1
    mask_blur = cv2.GaussianBlur(mask_scaled.astype(np.float32)/255.0, (blur, blur), 0)
    return (mask_blur * 255).astype(np.uint8)

def feather_mask(alpha, ksize=31):
    if ksize % 2 == 0: ksize += 1
    return cv2.GaussianBlur(alpha, (ksize, ksize), 0)

# ---------------------------
# Adapt bbox for pose (lebih agresif & tambahkan area rambut)
# ---------------------------
def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1; h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    # yaw besar -> tambah horizontal padding signifikan
    if abs(yaw) > 15.0:
        extra = (abs(yaw) - 15.0) * 0.03  # 3% per deg di atas 15
        extra = min(extra, 0.45)          # max +45% lebar
        pad_x = w * extra

    # pitch: lihat ke atas -> tambah dahi & rambut
    if pitch < -10.0:
        extra = (abs(pitch) - 10.0) * 0.03
        extra = min(extra, 0.45)
        pad_y_top = h * extra
    elif pitch > 12.0:
        extra = (pitch - 12.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_bottom = h * extra

    # tambahkan area atas kepala default supaya rambut tidak terpotong
    hair_extra_top = h * 0.18
    pad_y_top += hair_extra_top

    # sedikit perbesaran keseluruhan agar ada kepala & telinga
    overall_scale = 0.08
    pad_x += w * overall_scale
    pad_y_top += h * overall_scale
    pad_y_bottom += h * overall_scale

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 <= nx1 or ny2 <= ny1:
        return
    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

# ---------------------------
# Core swap: sekarang menghasilkan soft alpha mask dan blend hair-aware
# ---------------------------
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape)

    # panggil inswapper untuk memperoleh swapped result + mask (paste_back True dipakai,
    # namun kita juga akan buat alpha blending tambahan berdasarkan landmarks)
    swapper = get_face_swapper()
    # langsung jalankan get -> biasanya paste_back mengembalikan frame dengan hasil temp_patch
    swapped = swapper.get(temp_frame.copy(), target_face, source_face, paste_back=True)

    # Buat soft mask berdasarkan landmarks kalau ada
    # Ambil landmarks dengan aman tanpa boolean OR
landmarks = None
for key in ["landmark_2d_106", "landmark_2d_68", "landmark"]:
    lm = getattr(target_face, key, None)
    if lm is not None and hasattr(lm, "__len__") and len(lm) > 0:
        landmarks = lm
        break

    if landmarks is None:
        # fallback: gunakan bounding box with ellipse
        x1, y1, x2, y2 = map(int, target_face.bbox)
        mask = np.zeros(temp_frame.shape[:2], dtype=np.float32)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        axes = (int((x2-x1)*0.55), int((y2-y1)*0.60))
        cv2.ellipse(mask, (cx,cy), axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31,31), 0)
        alpha = np.clip(mask, 0.0, 1.0)
    else:
        # landmarks relatif ke frame coordinate (beberapa model mengembalikan relatif)
        try:
            lms = np.array(landmarks)
            if lms.max() <= 1.1:
                # relatif 0..1 -> skalakan ke ukuran frame
                h, w = temp_frame.shape[:2]
                lms = (lms * np.array([w, h])).astype(np.int32)
            mask_u8 = landmarks_to_hair_mask(temp_frame.shape, lms, scale=1.15)
            alpha = (mask_u8.astype(np.float32)/255.0)
            # feather lebih besar untuk hairline
            alpha = cv2.GaussianBlur(alpha, (41,41), 0)
            alpha = np.clip(alpha, 0.0, 1.0)
        except Exception:
            x1, y1, x2, y2 = map(int, target_face.bbox)
            mask = np.zeros(temp_frame.shape[:2], dtype=np.float32)
            cv2.rectangle(mask, (x1,y1),(x2,y2), 1, -1)
            alpha = cv2.GaussianBlur(mask, (31,31), 0)

    # occlusion-aware intensity reduction: kalau ada occlusion tipis (rambut) -> turunkan intensity
    occluded = detect_occlusion(target_face, temp_frame)
    intensity = 1.0
    # jika occluded -> jangan langsung skip tapi kurangi intensitas / blend lebih lembut
    if occluded:
        # kalau occlusion model memberi informasi, kita reduce lebih besar
        intensity = 0.45
    else:
        # periksa jika det_score rendah tapi > threshold
        det_score = getattr(target_face, "det_score", 1.0)
        if det_score < 0.55:
            intensity = 0.85

    # final compositing: soft alpha * intensity + original*(1-alpha*intensity)
    alpha_final = np.expand_dims(alpha * intensity, axis=2)
    result = (swapped.astype(np.float32) * alpha_final + temp_frame.astype(np.float32) * (1.0 - alpha_final)).astype(np.uint8)
    return result

# ---------------------------
# Selection helper (tetap kompatibel)
# ---------------------------
def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Face | None:
    if not faces or reference_face is None:
        return None
    if not hasattr(reference_face, 'normed_embedding'):
        return None
    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)
    for f in faces:
        if not hasattr(f, 'normed_embedding'): continue
        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue
        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f
    return best_face

# ---------------------------
# Frame processing (compatible dengan pipeline lamamu)
# ---------------------------
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame

    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame
        for target_face in faces:
            if detect_occlusion(target_face, temp_frame):
                # kurangi intensitas saat occluded (diproses dalam swap_face)
                pass
            temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame

    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)
    if not tracked_faces:
        return temp_frame

    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        # jika semua occluded, fallback: pilih non-occluded tapi kurangi intensity => process_frame akan men-skip
        valid_faces = tracked_faces

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)
    if best_target is None:
        best_target = valid_faces[0]
    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=temp_frame, frame_number=idx)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    source_face = get_one_face(source_img)
    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=target_frame, frame_number=0)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
