from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os
import onnxruntime as ort

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking (penting untuk multi-thread)

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Threshold / hyper-parameter default (boleh kamu tuning)
MIN_DET_SCORE = 0.30        # min score agar wajah dianggap valid (untuk get_many_faces)
OCCLUSION_THRESHOLD = 0.40  # det_score < ini dianggap occluded (fallback basic)
MAX_TRACK_GAP = 10          # frame: kalau lebih lama dari ini → track di-skip saat matching
MAX_TRACK_AGE = 15          # frame: track dihapus bila tidak terlihat selama ini
MIN_EMBED_SIMILARITY = 0.70 # cosine similarity minimal untuk dianggap match (0–1)

# Occluder model (ONNX) untuk deteksi occlusion yang lebih akurat
OCCLUDER_URL = "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/occluder.onnx"
OCCLUDER_FILENAME = "occluder.onnx"
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_LOCK = threading.Lock()
OCCLUDER_SCORE_THRESHOLD = 0.5  # threshold mean score mask → occluded


# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_face_analyser() -> Any:
    """
    Lazy init insightface FaceAnalysis.
    Sekali saja per proses, thread-safe.
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            print("✅ [face_analyser] Using buffalo_l with optimized settings")
    return FACE_ANALYSER


def get_occluder_session() -> Optional[ort.InferenceSession]:
    """
    Lazy init occluder.onnx via onnxruntime.
    Auto-download ke folder ../models.
    """
    global OCCLUDER_SESSION

    with OCCLUDER_LOCK:
        if OCCLUDER_SESSION is None:
            models_dir = resolve_relative_path('../models')
            # auto-download file occluder.onnx
            conditional_download(models_dir, [OCCLUDER_URL])
            model_path = resolve_relative_path(f'../models/{OCCLUDER_FILENAME}')

            providers = roop.globals.execution_providers
            # Mapping sederhana: kalau ada CUDAExecutionProvider, pakai GPU
            ort_providers = []
            if 'CUDAExecutionProvider' in providers:
                ort_providers.append('CUDAExecutionProvider')
            ort_providers.append('CPUExecutionProvider')

            OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=ort_providers)
            print("✅ [face_analyser] occluder.onnx loaded")
    return OCCLUDER_SESSION


def clear_face_analyser() -> None:
    """
    Reset analyser & tracking state.
    Dipanggil saat post_process / cleanup.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY, OCCLUDER_SESSION

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None

    with OCCLUDER_LOCK:
        OCCLUDER_SESSION = None


# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Deteksi banyak wajah di satu frame.
    - Pakai buffalo_l
    - Filter berdasarkan det_score minimal (untuk video dance / gerak cepat)
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []

        # filter berdasarkan confidence
        faces = [face for face in faces if getattr(face, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces
    except ValueError:
        return None
    except Exception:
        # kalau ada error aneh dari insightface, jangan matikan pipeline
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    Ambil 1 wajah dari frame:
    - default: index 0
    - kalau index out-of-range → pakai wajah terakhir
    """
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


# =====================================================================
#  MOTION & TRACKING
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    """
    Hitung pergerakan (jarak Euclidean) antara dua bbox wajah berturutan.
    Dipakai untuk informasi tambahan tracking (walau saat ini lebih fokus ke embedding).
    """
    if prev_face is None or current_face is None:
        return 0.0

    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox

    # hitung titik tengah
    prev_center = np.array([
        (prev_bbox[0] + prev_bbox[2]) / 2,
        (prev_bbox[1] + prev_bbox[3]) / 2
    ])
    current_center = np.array([
        (current_bbox[0] + current_bbox[2]) / 2,
        (current_bbox[1] + current_bbox[3]) / 2
    ])

    motion = np.linalg.norm(current_center - prev_center)
    return float(motion)


def _compute_embedding_similarity(current_embedding: np.ndarray,
                                  track_embedding: np.ndarray) -> float:
    """
    Hitung similarity embedding (cosine-based).
    Return 0 kalau terjadi error.
    """
    try:
        # cosine() dari scipy.spatial.distance mengembalikan *distance*
        # kita ubah jadi similarity: 1 - distance
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """
    Smart tracking:
    - gunakan embedding similarity + sedikit motion
    - jaga agar ID wajah konsisten antar frame
    - smoothing bbox pakai TRACKING_HISTORY
    - thread-safe: di-protect oleh TRACK_LOCK
    """
    global FACE_TRACKING, TRACKING_HISTORY

    current_faces = get_many_faces(frame)
    if not current_faces:
        return None

    tracked_faces: List[Face] = []

    with TRACK_LOCK:
        for face in current_faces:
            face_id = None
            max_similarity = MIN_EMBED_SIMILARITY
            best_match_id = None

            # embedding wajah sekarang
            current_embedding = getattr(face, "normed_embedding", None)
            if current_embedding is None or len(current_embedding) == 0:
                current_embedding = np.array([])

            # cari track yang paling cocok (snapshot list() → aman dari perubahan size)
            for track_id, track_data in list(FACE_TRACKING.items()):
                # lupakan track yang terlalu lama tidak terlihat
                if frame_number - track_data.get('last_seen', -9999) > MAX_TRACK_GAP:
                    continue

                last_face = track_data.get('last_face', None)
                if last_face is None:
                    continue

                track_embedding = getattr(last_face, "normed_embedding", None)
                if track_embedding is None:
                    continue

                embedding_similarity = _compute_embedding_similarity(
                    current_embedding, track_embedding
                )

                if embedding_similarity > max_similarity:
                    max_similarity = embedding_similarity
                    best_match_id = track_id

            if best_match_id is not None:
                # update track yang ada
                face_id = best_match_id
                prev_face = FACE_TRACKING[face_id]['last_face']
                motion = calculate_motion_vector(prev_face, face)

                FACE_TRACKING[face_id].update({
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': motion
                })
            else:
                # buat track baru
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': 0.0
                }

            # smoothing bbox sederhana dengan history 2 frame terakhir
            if len(TRACKING_HISTORY) >= 2:
                recent_faces = list(TRACKING_HISTORY)[-2:]
                if all('bbox' in f for f in recent_faces):
                    smoothed_bbox = np.mean([f['bbox'] for f in recent_faces], axis=0)
                    face.bbox = smoothed_bbox

            # simpan ke history
            face_data = {
                'bbox': np.array(face.bbox, dtype=np.float32).copy()
            }
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)

        # bersihkan track yang sudah terlalu tua
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }

    return tracked_faces


# =====================================================================
#  OCCLUDER & OCCLUSION
# =====================================================================

def _run_occluder_on_face(frame: Frame, face: Face) -> Optional[float]:
    """
    Jalankan occluder.onnx pada crop wajah.
    Output diinterpretasikan sebagai score [0..1] (mean dari output tensor).
    """
    session = get_occluder_session()
    if session is None:
        return None

    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # preprocess generic: 224x224 RGB, [0,1], NCHW
    inp = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    inp = inp.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))  # CHW
    inp = np.expand_dims(inp, 0)        # NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    if not outputs:
        return None

    score = float(np.mean(outputs[0]))
    return score


def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    """
    Deteksi wajah yang ter-occlusion.
    Kombinasi:
    - det_score rendah → langsung dianggap occluded (fallback aman)
    - jika frame tersedia & occluder siap → pakai occluder.onnx

    NOTE:
    - Tidak fallback ke model swap lain (sesuai permintaan), hanya heuristik internal.
    """
    # Basic filter by det_score
    base_score = getattr(face, "det_score", 1.0)
    if base_score < OCCLUSION_THRESHOLD:
        return True

    # Kalau tidak ada frame, tidak bisa pakai occluder → anggap non-occluded.
    if frame is None:
        return False

    try:
        occ_score = _run_occluder_on_face(frame, face)
    except Exception:
        occ_score = None

    if occ_score is None:
        # tidak bisa hitung occluder → kembali ke basic
        return False

    return occ_score >= OCCLUDER_SCORE_THRESHOLD


# =====================================================================
#  SIMILAR FACE
# =====================================================================

def find_similar_face(frame: Frame,
                      reference_face: Face,
                      use_tracking: bool = True) -> Optional[Face]:
    """
    Cari wajah paling mirip di frame terhadap reference_face.
    - Bisa pakai smart tracking (use_tracking=True)
    - Atau fallback ke get_many_faces biasa
    - Menggunakan embedding distance seperti di mod kamu sebelumnya
    """
    if reference_face is None:
        return None

    if use_tracking:
        many_faces = smart_face_tracking(frame, frame_number=0)
    else:
        many_faces = get_many_faces(frame)

    if not many_faces:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')

    # threshold diambil dari globals kalau ada, else fallback
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for face in many_faces:
        if not hasattr(face, "normed_embedding"):
            continue

        try:
            distance = np.sum(np.square(face.normed_embedding - ref_emb))
        except Exception:
            continue

        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = face

    return best_face
