import threading
from typing import Any, Optional, List
import cv2
import insightface
import numpy as np

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

# ==========================
# Konfigurasi model & kualitas
# ==========================
PRIMARY_PACK = "antelopev2"     # target utama: ArcFace R100 + SCRFD
FALLBACK_PACK = "buffalo_l"     # fallback aman kalau antelopev2 gagal
MIN_FACE_SIZE = 64              # minimal lebar/tinggi wajah (px)
MIN_DET_SCORE = 0.5             # minimal confidence deteksi wajah
USE_CENTER_PRIORITY = True      # prioritas wajah yang paling besar & paling di tengah
MAX_FACES = 5                   # batasi jumlah wajah yang dikembalikan
DET_MAX_SIZE = 720              # deteksi dilakukan max di resolusi ini (lebih kecil = lebih cepat)


def _init_face_analysis(pack_name: str) -> Any:
    """
    Load model pack InsightFace.
    Pastikan module 'detection' tersedia.
    """
    analyser = insightface.app.FaceAnalysis(
        name=pack_name,
        root="/root/.insightface",
        providers=roop.globals.execution_providers,
        allowed_modules=['detection', 'recognition']
    )

    analyser.prepare(ctx_id=0, det_size=(640, 640))

    # Cek apakah detection benar-benar termuat
    if not hasattr(analyser, "models") or "detection" not in analyser.models:
        raise AssertionError(f"Pack '{pack_name}' tidak memiliki model detection.")

    return analyser


def get_face_analyser() -> Any:
    """
    Urutan load:
    1. Coba antelopev2
    2. Kalau gagal → buffalo_l
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is not None:
            return FACE_ANALYSER

        # Coba antelopev2 dulu
        try:
            FACE_ANALYSER = _init_face_analysis(PRIMARY_PACK)
            print(f"[FaceAnalyser] Menggunakan pack utama: {PRIMARY_PACK}")
            return FACE_ANALYSER
        except Exception as e:
            print(f"[FaceAnalyser] Gagal load '{PRIMARY_PACK}'. Fallback ke '{FALLBACK_PACK}'. Error: {e}")

        # Fallback ke buffalo_l
        try:
            FACE_ANALYSER = _init_face_analysis(FALLBACK_PACK)
            print(f"[FaceAnalyser] Menggunakan pack fallback: {FALLBACK_PACK}")
            return FACE_ANALYSER
        except Exception as e:
            print(f"[FaceAnalyser] Tidak bisa load fallback '{FALLBACK_PACK}'. Error: {e}")
            raise RuntimeError("Gagal memuat antelopev2 dan buffalo_l.")


def clear_face_analyser() -> None:
    global FACE_ANALYSER
    FACE_ANALYSER = None


# ==========================
# Helper fungsi speed-up detection
# ==========================

def _resize_for_detection(frame: Frame):
    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side <= DET_MAX_SIZE:
        return frame, 1.0

    scale = DET_MAX_SIZE / float(max_side)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _rescale_faces_to_original(faces: List[Face], scale: float) -> None:
    if scale == 1.0 or not faces:
        return

    inv = 1.0 / scale
    for face in faces:
        if hasattr(face, "bbox"):
            face.bbox *= inv
        if hasattr(face, "kps") and face.kps is not None:
            face.kps *= inv
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            face.landmark_2d_106 *= inv


def _face_area_and_center_score(face: Face, frame: Frame) -> float:
    h, w = frame.shape[:2]
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return 0.0

    x1, y1, x2, y2 = bbox
    fw = max(0.0, x2 - x1)
    fh = max(0.0, y2 - y1)
    area = fw * fh
    if area <= 0:
        return 0.0

    if not USE_CENTER_PRIORITY:
        return area

    cx_face = (x1 + x2) / 2
    cy_face = (y1 + y2) / 2
    cx_frame = w / 2
    cy_frame = h / 2

    dx = (cx_face - cx_frame) / w
    dy = (cy_face - cy_frame) / h
    penalty = (dx * dx + dy * dy)

    return area * (1.0 - min(0.9, penalty))


def _filter_and_sort_faces(faces: List[Face], frame: Frame) -> List[Face]:
    if not faces:
        return []

    filt = []
    for face in faces:
        bbox = face.bbox
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        fw = x2 - x1
        fh = y2 - y1

        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            continue

        det_score = getattr(face, "det_score", None)
        if det_score is not None and det_score < MIN_DET_SCORE:
            continue

        filt.append(face)

    if not filt:
        return []

    filt.sort(key=lambda f: _face_area_and_center_score(f, frame), reverse=True)

    return filt[:MAX_FACES]


# ==========================
# API utama untuk Roop
# ==========================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    if frame is None or frame.size == 0:
        return None

    try:
        det_frame, scale = _resize_for_detection(frame)
        faces = get_face_analyser().get(det_frame)
        _rescale_faces_to_original(faces, scale)
        faces = _filter_and_sort_faces(faces, frame)
        return faces or None
    except Exception as e:
        print(f"[FaceAnalyser] Error membaca wajah: {e}")
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    faces = get_many_faces(frame)
    if faces:
        return faces[position] if position < len(faces) else faces[-1]
    return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    faces = get_many_faces(frame)
    if not faces:
        return None

    ref_emb = getattr(reference_face, "normed_embedding", None)
    if ref_emb is None:
        return None

    best_face = None
    best_dist = None

    for face in faces:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            continue

        dist = float(np.sum((np.array(emb) - np.array(ref_emb)) ** 2))

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_face = face

    if best_dist is not None and best_dist < roop.globals.similar_face_distance:
        return best_face

    return None
