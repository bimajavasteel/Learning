import threading
from typing import Any, Optional, List
import cv2
import insightface

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

# ==========================
# Konfigurasi kualitas + speed (mode FaceFusion-like)
# ==========================
MIN_FACE_SIZE = 64            # minimal lebar/tinggi wajah (px)
MIN_DET_SCORE = 0.5           # minimal confidence deteksi wajah
USE_CENTER_PRIORITY = True    # prioritas wajah yang paling besar & paling di tengah
MAX_FACES = 5                 # batasi jumlah wajah yang dikembalikan
DET_MAX_SIZE = 720            # deteksi dilakukan max di resolusi ini (lebih kecil = lebih cepat)


def get_face_analyser() -> Any:
    """
    Menggunakan pack 'antelopev2' -> SCRFD + ArcFace IResNet100.
    execution_providers diambil dari roop.globals.execution_providers.
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='antelopev2',  # ArcFace IResNet100
                providers=roop.globals.execution_providers,
                allowed_modules=['detection', 'recognition']
            )
            # ctx_id = 0 -> GPU pertama; pastikan CUDAExecutionProvider ada di globals
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    global FACE_ANALYSER
    FACE_ANALYSER = None


# ==========================
# Helper internal
# ==========================

def _resize_for_detection(frame: Frame):
    """
    Downscale frame untuk deteksi agar lebih cepat.
    Return: resized_frame, scale_factor (float)
    scale_factor = berapa kali resized_frame lebih kecil dari frame asli.
    """
    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side <= DET_MAX_SIZE:
        return frame, 1.0

    scale = DET_MAX_SIZE / float(max_side)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _rescale_faces_to_original(faces: List[Face], scale: float) -> None:
    """
    Mengembalikan koordinat bbox & landmark ke skala frame asli.
    Dilakukan in-place.
    """
    if scale == 1.0 or not faces:
        return

    inv = 1.0 / scale
    for face in faces:
        # bbox
        if hasattr(face, "bbox") and face.bbox is not None:
            face.bbox *= inv

        # landmark
        if hasattr(face, "kps") and face.kps is not None:
            face.kps *= inv


def _face_area_and_center_score(face: Face, frame: Frame) -> float:
    """
    Skor prioritas wajah:
    - area lebih besar -> skor naik
    - lebih dekat tengah frame -> skor naik
    """
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

    # jarak dari tengah frame (semakin dekat semakin bagus)
    cx_face = (x1 + x2) / 2.0
    cy_face = (y1 + y2) / 2.0
    cx_frame = w / 2.0
    cy_frame = h / 2.0

    dx = (cx_face - cx_frame) / w
    dy = (cy_face - cy_frame) / h
    center_penalty = (dx * dx + dy * dy)

    score = area * (1.0 - min(center_penalty, 0.9))
    return score


def _filter_and_sort_faces(faces: List[Face], frame: Frame) -> List[Face]:
    """
    - Buang wajah kecil / confidence rendah
    - Urutkan wajah berdasarkan prioritas (besar + tengah)
    - Batasi maksimal MAX_FACES
    """
    if not faces:
        return []

    filtered = []
    for face in faces:
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        fw = max(0.0, x2 - x1)
        fh = max(0.0, y2 - y1)

        # filter ukuran wajah
        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            continue

        # filter score deteksi
        det_score = getattr(face, "det_score", None)
        if det_score is not None and det_score < MIN_DET_SCORE:
            continue

        filtered.append(face)

    if not filtered:
        return []

    # urutkan wajah: skor tertinggi (besar + dekat tengah) di depan
    filtered.sort(
        key=lambda f: _face_area_and_center_score(f, frame),
        reverse=True
    )

    if len(filtered) > MAX_FACES:
        filtered = filtered[:MAX_FACES]

    return filtered


# ==========================
# API utama (kompatibel dengan versi kamu)
# ==========================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    if frame is None or frame.size == 0:
        return None

    try:
        # deteksi lebih cepat: resize dulu
        det_frame, scale = _resize_for_detection(frame)
        faces = get_face_analyser().get(det_frame)

        # kembalikan koordinat bbox & landmark ke skala frame asli
        _rescale_faces_to_original(faces, scale)

        faces = _filter_and_sort_faces(faces, frame)
        return faces or None
    except (ValueError, RuntimeError) as e:
        print(f"[FaceAnalyser] Skipped invalid frame: {e}")
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    """
    Cari wajah yang embedding ArcFace-nya paling dekat dengan reference_face.
    Menggunakan normed_embedding (cosine ~ L2 pada vektor ternormalisasi).
    """
    many_faces = get_many_faces(frame)
    if not many_faces:
        return None

    ref_emb = getattr(reference_face, "normed_embedding", None)
    if ref_emb is None:
        return None

    best_face = None
    best_distance = None

    for face in many_faces:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            continue

        # L2 distance di ruang embedding ternormalisasi
        dist = sum((float(a) - float(b)) ** 2 for a, b in zip(emb, ref_emb))

        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_face = face

    if best_face is not None and best_distance is not None:
        if best_distance < roop.globals.similar_face_distance:
            return best_face

    return None
