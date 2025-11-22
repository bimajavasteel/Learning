# roop/face_analyser.py

import threading
from typing import Any, Optional, List

import numpy as np
import insightface

import roop.globals
from roop.typing import Frame, Face

from roop.face_segmentation import get_face_segmenter  # ← tambahkan import ini

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

# =========================
#   ANALYSER SINGLETON
# =========================

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=providers
            )
            FACE_ANALYSER.prepare(
                ctx_id=roop.globals.gpu_id if roop.globals.gpu_id is not None else 0,
                det_size=(640, 640)
            )
        return FACE_ANALYSER


# =========================
#   FUNGSI BANTU OCCLUSION
# =========================

def _distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def landmarks_ok(lm) -> bool:
    """
    lm: 5 landmark, format [(x,y), ...] atau np.array shape (5,2)
    """
    lm = np.array(lm)

    # Jarak antar mata cukup besar
    eye_dist = _distance(lm[0], lm[1])
    if eye_dist < 20:
        return False

    # Mata harus di atas hidung
    if lm[0][1] > lm[2][1] or lm[1][1] > lm[2][1]:
        return False

    # Hidung harus di atas mulut
    if lm[2][1] > lm[3][1] or lm[2][1] > lm[4][1]:
        return False

    return True


def aspect_ok(bbox) -> bool:
    """
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    ratio = h / float(w)

    # Wajah normal tidak terlalu tinggi/pipih
    if ratio > 2.0 or ratio < 0.5:
        return False

    return True


def segmentation_ok(frame: Frame, faces: List[Face], min_ratio: float = 0.40) -> List[bool]:
    """
    Mengembalikan list boolean: apakah face_i lolos segmentation (tidak terlalu ter-occlude).
    min_ratio: minimal rasio area wajah (0–1). Semakin tinggi, semakin ketat.
    """
    if len(faces) == 0:
        return []

    h, w, _ = frame.shape
    crops = []
    boxes = []

    for face in faces:
        x1, y1, x2, y2 = face.bbox
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w - 1)
        y2 = min(int(y2), h - 1)

        if x2 <= x1 or y2 <= y1:
            crops.append(None)
            boxes.append(None)
            continue

        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
        boxes.append((x1, y1, x2, y2))

    # Filter crop valid
    valid_idx = [i for i, c in enumerate(crops) if c is not None]
    if not valid_idx:
        return [False] * len(faces)

    valid_crops = [crops[i] for i in valid_idx]

    segmenter = get_face_segmenter()
    visibility_scores = segmenter.face_visibility_scores(valid_crops)

    # Rekonstruksi ke list full
    results = [False] * len(faces)
    for i, idx in enumerate(valid_idx):
        score = visibility_scores[i]
        results[idx] = (score >= min_ratio)

    return results


# =========================
#   FILTER FACES
# =========================

def filter_faces(frame: Frame, faces: List[Face]) -> List[Face]:
    if not faces:
        return []

    # 1. Filter basic (confidence, landmark, aspect ratio)
    basic_valid_flags = []
    basic_valid_faces = []

    for face in faces:
        # Confidence minimal
        if getattr(face, "det_score", 0.0) < 0.60:
            basic_valid_flags.append(False)
            continue

        # Landmark check
        if not hasattr(face, "landmark_5"):
            basic_valid_flags.append(False)
            continue

        if not landmarks_ok(face.landmark_5):
            basic_valid_flags.append(False)
            continue

        # Aspect ratio check
        if not aspect_ok(face.bbox):
            basic_valid_flags.append(False)
            continue

        basic_valid_flags.append(True)
        basic_valid_faces.append(face)

    if not basic_valid_faces:
        return []

    # 2. Segmentation check (BiSeNet) – hanya untuk yang basic valid
    # mapping index
    mapping = [i for i, flag in enumerate(basic_valid_flags) if flag]
    seg_flags_full = [False] * len(faces)

    seg_flags_basic = segmentation_ok(frame, basic_valid_faces, min_ratio=0.40)

    for j, orig_idx in enumerate(mapping):
        seg_flags_full[orig_idx] = seg_flags_basic[j]

    # 3. Gabungkan hasil
    filtered = []
    for i, face in enumerate(faces):
        if i >= len(basic_valid_flags):
            continue
        if not basic_valid_flags[i]:
            continue
        if not seg_flags_full[i]:
            continue
        filtered.append(face)

    return filtered


# =========================
#   API UTAMA ROOP
# =========================

def get_many_faces(frame: Frame) -> List[Face]:
    analyser = get_face_analyser()
    faces: List[Face] = analyser.get(frame)
    faces = filter_faces(frame, faces)
    return faces


def get_one_face(frame: Frame) -> Optional[Face]:
    faces = get_many_faces(frame)
    if not faces:
        return None

    # Bisa pilih face terbesar / skor tertinggi
    faces = sorted(faces, key=lambda f: (f.det_score, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])), reverse=True)
    return faces[0]
