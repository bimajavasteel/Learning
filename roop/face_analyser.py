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
#   LOAD INSIGHTFACE (BUFFALO_L)
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
#   HELPER FUNGI
# ======================================================

def _distance(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) -
                                np.array(p2, dtype=np.float32)))


def _landmarks_struct_ok(lm) -> bool:
    """
    Cek struktur landmark secara longgar.
    Dipakai baik untuk SOURCE maupun TARGET (beda di threshold lain).
    """
    if lm is None:
        return False

    lm = np.array(lm)
    if lm.ndim != 2 or lm.shape != (5, 2):
        return False
    if not np.isfinite(lm).all():
        return False

    # Mata harus punya jarak minimal
    eye_dist = _distance(lm[0], lm[1])
    if eye_dist < 3.0:
        return False

    return True


def _aspect_ok(bbox) -> bool:
    """
    Aspect ratio wajar untuk TARGET (tidak dipakai untuk SOURCE).
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

    # cukup longgar tapi masih masuk akal
    if ratio > 3.0 or ratio < 0.3:
        return False

    return True


def _segmentation_flags(frame: Frame,
                        faces: List[Face],
                        min_ratio: float = 0.35) -> List[bool]:
    """
    Hitung visibility wajah memakai BiSeNet.
    Dipakai HANYA untuk TARGET.
    """
    if frame is None or len(faces) == 0:
        return [False] * len(faces)

    h, w = frame.shape[:2]
    crops = []
    idx_map = []

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
        idx_map.append(i)

    if not crops:
        return [False] * len(faces)

    segmenter = get_face_segmenter()
    scores = segmenter.face_visibility_scores(crops)

    flags = [False] * len(faces)
    for i, idx in enumerate(idx_map):
        flags[idx] = (scores[i] >= min_ratio)

    return flags


def _face_sort_key(face: Face):
    score = float(getattr(face, "det_score", 0.0))
    bbox = getattr(face, "bbox", None)
    area = 0.0
    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox
        area = max(0.0, (x2 - x1) * (y2 - y1))
    return (score, area)


# ======================================================
#   FILTER UNTUK SOURCE IMAGE (SUPER TOLERAN)
#   → digunakan untuk validasi source_path
# ======================================================

def _filter_faces_source(frame: Frame, faces: List[Face]) -> List[Face]:
    """
    Sangat toleran:
    - det_score minimal 0.15
    - landmark hanya cek struktur
    - TIDAK pakai aspect ratio
    - TIDAK pakai segmentation
    """
    if not faces:
        return []

    valid: List[Face] = []

    for face in faces:
        score = float(getattr(face, "det_score", 0.0))
        if score < 0.15:
            continue

        lm = getattr(face, "landmark_5", None)
        if not _landmarks_struct_ok(lm):
            continue

        valid.append(face)

    return valid


# ======================================================
#   FILTER UNTUK TARGET FRAME (KETAT + ANTI OCCLUSION)
#   → digunakan saat proses swap di video
# ======================================================

def _filter_faces_target(frame: Frame, faces: List[Face]) -> List[Face]:
    """
    Lebih ketat:
    - det_score minimal 0.50
    - landmark struktur ok
    - aspect ratio wajar
    - segmentation (BiSeNet) untuk cek occlusion
    """
    if not faces:
        return []

    basic_valid: List[Face] = []

    # Basic filters: score, landmarks, aspect
    for face in faces:
        score = float(getattr(face, "det_score", 0.0))
        if score < 0.50:
            continue

        lm = getattr(face, "landmark_5", None)
        if not _landmarks_struct_ok(lm):
            continue

        if not _aspect_ok(getattr(face, "bbox", None)):
            continue

        basic_valid.append(face)

    if not basic_valid:
        return []

    # Segmentation-based occlusion check
    seg_flags = _segmentation_flags(frame, basic_valid, min_ratio=0.35)

    final: List[Face] = []
    for face, ok in zip(basic_valid, seg_flags):
        if ok:
            final.append(face)

    return final


# ======================================================
#   PUBLIC API: SOURCE
#   (dipakai Roop untuk cv2.imread(source_path))
# ======================================================

def get_many_faces_source(frame: Frame) -> List[Face]:
    if frame is None:
        return []
    analyser = get_face_analyser()
    faces: List[Face] = analyser.get(frame)
    return _filter_faces_source(frame, faces)


def get_one_face_source(frame: Frame) -> Optional[Face]:
    faces = get_many_faces_source(frame)
    if not faces:
        return None
    faces_sorted = sorted(faces, key=_face_sort_key, reverse=True)
    return faces_sorted[0]


# ======================================================
#   PUBLIC API: TARGET
#   (dipakai frame processor untuk video)
# ======================================================

def get_many_faces_target(frame: Frame) -> List[Face]:
    if frame is None:
        return []
    analyser = get_face_analyser()
    faces: List[Face] = analyser.get(frame)
    return _filter_faces_target(frame, faces)


def get_one_face_target(frame: Frame) -> Optional[Face]:
    faces = get_many_faces_target(frame)
    if not faces:
        return None
    faces_sorted = sorted(faces, key=_face_sort_key, reverse=True)
    return faces_sorted[0]


# ======================================================
#   KOMPATIBILITAS DENGAN ROOP ASLI
# ======================================================

def get_many_faces(frame: Frame) -> List[Face]:
    """
    DEFAULT: anggap dipakai untuk TARGET (video frame).
    Semua pemanggilan lama ke get_many_faces tetap jalan.
    """
    return get_many_faces_target(frame)


def get_one_face(frame: Frame) -> Optional[Face]:
    """
    DEFAULT: anggap dipakai untuk SOURCE.
    Roop asli memanggil ini di pre_start() untuk source_path.
    """
    return get_one_face_source(frame)
