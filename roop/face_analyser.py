import threading
from typing import Any, Optional, List
import numpy as np
import insightface

import roop.globals
from roop.typing import Frame, Face
from roop.face_segmentation import get_face_segmenter

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()


# ======================================================
#   LOAD INSIGHTFACE MODEL
# ======================================================

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=providers
            )
            ctx_id = roop.globals.gpu_id if getattr(roop.globals, "gpu_id", None) is not None else 0
            FACE_ANALYSER.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return FACE_ANALYSER


# ======================================================
#   TOLERANT FILTERS
# ======================================================

def _distance(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def landmarks_ok(lm) -> bool:
    """
    SUPER-TOLERANT LANDMARK CHECK:
    - cukup periksa struktur dan sedikit jarak antar mata
    - tidak cek anatomi & posisi
    """
    if lm is None:
        return False

    lm = np.array(lm)

    if lm.ndim != 2:
        return False
    if lm.shape != (5, 2):
        return False
    if not np.isfinite(lm).all():
        return False

    # mata minimal sedikit berjarak
    eye_dist = _distance(lm[0], lm[1])
    if eye_dist < 3:  
        return False

    return True


def segmentation_ok(frame: Frame, faces: List[Face], min_ratio: float = 0.03) -> List[bool]:
    """
    SUPER-TOLERANT SEGMENTATION:
    - ratio minimal 3% (sangat rendah)
    - tetap berguna agar objek lain tidak dianggap wajah
    """
    if frame is None or len(faces) == 0:
        return [False] * len(faces)

    h, w = frame.shape[:2]
    crops = []
    valid_idx = []

    for i, face in enumerate(faces):
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append(crop)
        valid_idx.append(i)

    if not crops:
        return [False] * len(faces)

    segmenter = get_face_segmenter()
    scores = segmenter.face_visibility_scores(crops)

    results = [False] * len(faces)
    for idx, score in zip(valid_idx, scores):
        results[idx] = (score >= min_ratio)

    return results


# ======================================================
#   MAIN FILTER PIPELINE (SUPER TOLERAN)
# ======================================================

def filter_faces(frame: Frame, faces: List[Face]) -> List[Face]:
    """
    Pipeline:
    1. det_score minimal 0.15
    2. landmark minimal valid struktur
    3. aspect ratio DINONAKAN (wajib untuk cutout)
    4. segmentation minimal 3%
    """
    if not faces:
        return []

    tolerant_valid = []

    # BASIC FILTERS
    for face in faces:
        score = float(getattr(face, "det_score", 0.0))
        if score < 0.15:  
            continue

        if not hasattr(face, "landmark_5"):
            continue
        if not landmarks_ok(face.landmark_5):
            continue

        # aspect ratio CHECK DISABLED (untuk gambar cutout)
        # if not aspect_ok(face.bbox): continue

        tolerant_valid.append(face)

    if not tolerant_valid:
        return []

    # OCCLUSION SEGMENTATION SUPER-TOLERANT
    seg_ok = segmentation_ok(frame, tolerant_valid, min_ratio=0.03)

    final = []
    for f, ok in zip(tolerant_valid, seg_ok):
        if ok:
            final.append(f)

    return final


# ======================================================
#   PUBLIC API
# ======================================================

def get_many_faces(frame: Frame) -> List[Face]:
    if frame is None:
        return []
    analyser = get_face_analyser()
    faces = analyser.get(frame)
    return filter_faces(frame, faces)


def _face_sort_key(face: Face):
    score = float(getattr(face, "det_score", 0.0))
    bbox = getattr(face, "bbox", None)
    area = 0.0
    if bbox is not None:
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return (score, area)


def get_one_face(frame: Frame) -> Optional[Face]:
    faces = get_many_faces(frame)
    if not faces:
        return None
    return sorted(faces, key=_face_sort_key, reverse=True)[0]
