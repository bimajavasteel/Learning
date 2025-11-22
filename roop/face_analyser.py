import threading
from typing import Any, Optional, List

import numpy as np
import insightface

import roop.globals
from roop.typing import Frame, Face
from roop.face_segmentation import get_face_segmenter  # pastikan file ini sudah ada

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()


# ======================================================
#   SINGLETON FACE ANALYSER (InsightFace)
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
#   FUNGSI BANTU UNTUK FILTER WAJAH
# ======================================================

def _distance(p1, p2) -> float:
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))


def landmarks_ok(lm) -> bool:
    """
    Validasi bentuk & anatomi landmark.
    Harus array (5,2), nilai finite, dan posisi wajar.
    """
    if lm is None:
        return False

    lm = np.array(lm)

    # Harus 2 dimensi dan shape (5,2)
    if lm.ndim != 2:
        return False
    if lm.shape != (5, 2):
        return False

    if not np.isfinite(lm).all():
        return False

    # urutan InsightFace: [left_eye, right_eye, nose, left_mouth, right_mouth]
    eye_dist = _distance(lm[0], lm[1])
    if eye_dist < 10:  # terlalu kecil → noise
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
    Filter aspect ratio bounding box supaya tidak aneh.
    bbox: [x1, y1, x2, y2]
    """
    if bbox is None:
        return False

    bbox = np.array(bbox, dtype=np.float32)
    if bbox.size < 4:
        return False

    x1, y1, x2, y2 = bbox.tolist()
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
    Cek visibility wajah dengan BiSeNet segmentation.
    Return: list bool dengan panjang = len(faces).
    True  → wajah cukup kelihatan (tidak terlalu ter-occlude)
    False → wajah terlalu tertutup (tangan, jari, rambut, dll)
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

        x1, y1, x2, y2 = np.array(bbox, dtype=np.float32).tolist()
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w - 1)
        y2 = min(int(y2), h - 1)

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
    visibility_scores = segmenter.face_visibility_scores(crops)

    results = [False] * len(faces)
    for idx, score in zip(valid_idx, visibility_scores):
        results[idx] = (score >= min_ratio)

    return results


# ======================================================
#   FILTER UTAMA FACES
# ======================================================

def filter_faces(frame: Frame, faces: List[Face]) -> List[Face]:
    """
    Tahap filter:
    1. Confidence threshold (det_score)
    2. Validasi landmark (struktur + anatomi)
    3. Filter aspect ratio bbox
    4. Cek occlusion via segmentation BiSeNet (face visibility ratio)
    """
    if not faces:
        return []

    basic_valid_faces: List[Face] = []

    # 1–3: BASIC FILTER
    for face in faces:
        det_score = getattr(face, "det_score", 0.0)
        if det_score < 0.60:
            continue

        if not hasattr(face, "landmark_5"):
            continue
        if not landmarks_ok(face.landmark_5):
            continue

        if not hasattr(face, "bbox"):
            continue
        if not aspect_ok(face.bbox):
            continue

        basic_valid_faces.append(face)

    if not basic_valid_faces:
        return []

    # 4: SEGMENTATION OCCLUSION CHECK
    seg_flags = segmentation_ok(frame, basic_valid_faces, min_ratio=0.40)

    filtered: List[Face] = []
    for face, ok in zip(basic_valid_faces, seg_flags):
        if ok:
            filtered.append(face)

    return filtered


# ======================================================
#   API UTAMA YANG DIPAKAI FRAME PROCESSOR
# ======================================================

def get_many_faces(frame: Frame) -> List[Face]:
    """
    Analisa banyak wajah dalam 1 frame + occlusion filtering.
    Dipakai oleh frame processors.
    """
    if frame is None:
        return []

    analyser = get_face_analyser()
    faces: List[Face] = analyser.get(frame)
    faces = filter_faces(frame, faces)
    return faces


def _face_sort_key(face: Face):
    det_score = float(getattr(face, "det_score", 0.0))
    bbox = getattr(face, "bbox", None)
    area = 0.0
    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = np.array(bbox, dtype=np.float32).tolist()
        area = max(0.0, (x2 - x1) * (y2 - y1))
    return (det_score, area)


def get_one_face(frame: Frame) -> Optional[Face]:
    """
    Ambil 1 wajah terbaik:
    - dipakai saat source image (pre_start)
    - jika tidak ada wajah valid → return None
    """
    faces = get_many_faces(frame)
    if not faces:
        return None

    faces_sorted = sorted(faces, key=_face_sort_key, reverse=True)
    return faces_sorted[0]
