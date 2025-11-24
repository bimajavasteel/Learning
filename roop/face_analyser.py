from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import os
import cv2
import numpy as np
import insightface
import onnxruntime as ort

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, conditional_download

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking

# Tracking
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Threshold / hyper-parameter (boleh kamu tuning)
MIN_DET_SCORE = 0.30         # min score untuk get_many_faces
MAX_TRACK_GAP = 10           # frame: gap max untuk match track lama
MAX_TRACK_AGE = 15           # frame: hapus track jika terlalu lama tak terlihat
MIN_EMBED_SIMILARITY = 0.70  # cosine similarity minimal match embedding (0–1)

# =====================================================================
#  OCCLUDER MODEL (ONNX)
# =====================================================================

OCCLUDER_URL = (
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/"
    "d783e61585b3d83a85c91ca8a3b299e8ade94d72/occluder.onnx"
)

OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None
OCCLUDER_LOCK = threading.Lock()


def get_occluder_session() -> tuple[ort.InferenceSession, str]:
    """
    Lazy init occluder.onnx.
    - Auto download ke folder:  resolve_relative_path('models')
    - Tanpa fallback ke metode lain.
    """
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME

    with OCCLUDER_LOCK:
        if OCCLUDER_SESSION is None:
            models_dir = resolve_relative_path('models')
            conditional_download(models_dir, [OCCLUDER_URL])

            model_path = os.path.join(models_dir, 'occluder.onnx')

            providers = getattr(roop.globals, "execution_providers", None)
            if providers:
                OCCLUDER_SESSION = ort.InferenceSession(
                    model_path,
                    providers=providers
                )
            else:
                OCCLUDER_SESSION = ort.InferenceSession(model_path)

            OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name

    return OCCLUDER_SESSION, OCCLUDER_INPUT_NAME  # type: ignore[return-value]


# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_face_analyser() -> Any:
    """
    Lazy init insightface FaceAnalysis (buffalo_l).
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
            print("✅ [face_analyser] Using buffalo_l (pose + embedding + landmark siap)")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    """
    Reset analyser & tracking state.
    Bisa dipanggil saat post_process / cleanup.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None

    with OCCLUDER_LOCK:
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None


# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Deteksi banyak wajah di satu frame.
    - Pakai buffalo_l
    - Filter berdasarkan det_score minimal (untuk noise deteksi kecil)
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []

        faces = [
            face for face in faces
            if getattr(face, "det_score", 0.0) >= MIN_DET_SCORE
        ]
        return faces
    except Exception:
        # Kalau insightface error, jangan matikan pipeline
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    Ambil 1 wajah dari frame:
    - default: index 0
    - kalau index out-of-range → pakai wajah terakhir.
    """
    many_faces = get_many_faces(frame)
    if not many_faces:
        return None

    try:
        return many_faces[position]
    except IndexError:
        return many_faces[-1]


# =====================================================================
#  MOTION & TRACKING
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    """
    Hitung pergerakan (jarak Euclidean) antara dua bbox wajah berturutan.
    Dipakai sebagai informasi tambahan tracking.
    """
    if prev_face is None or current_face is None:
        return 0.0

    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox

    prev_center = np.array([
        (prev_bbox[0] + prev_bbox[2]) / 2.0,
        (prev_bbox[1] + prev_bbox[3]) / 2.0
    ])
    current_center = np.array([
        (current_bbox[0] + current_bbox[2]) / 2.0,
        (current_bbox[1] + current_bbox[3]) / 2.0
    ])

    motion = np.linalg.norm(current_center - prev_center)
    return float(motion)


def _compute_embedding_similarity(
    current_embedding: np.ndarray,
    track_embedding: np.ndarray
) -> float:
    """
    Hitung similarity embedding (cosine-based).
    Return 0 kalau error.
    """
    try:
        # cosine() mengembalikan distance → ubah jadi similarity: 1 - distance
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


def smart_face_tracking(frame: Frame, frame_number: int) -> List[Face]:
    """
    Tracking wajah sederhana tapi stabil:
    - deteksi wajah per frame,
    - match ke track lama via embedding similarity + jarak bbox,
    - smoothing bbox via history.
    """
    global FACE_TRACKING, TRACKING_HISTORY
    many_faces = get_many_faces(frame)
    if not many_faces:
        return []

    tracked_faces: List[Face] = []

    with TRACK_LOCK:
        for face in many_faces:
            # cari track terbaik untuk face ini
            best_track_id = None
            best_sim = -1.0

            current_emb = getattr(face, "normed_embedding", None)
            if current_emb is None:
                # kalau tidak ada embedding, anggap track baru saja
                current_emb = None

            for track_id, track_info in list(FACE_TRACKING.items()):
                last_face = track_info.get('last_face')
                last_seen = track_info.get('last_seen', -9999)

                if last_face is None:
                    continue

                # skip track terlalu lama tidak terlihat
                if frame_number - last_seen > MAX_TRACK_GAP:
                    continue

                if current_emb is not None and hasattr(last_face, "normed_embedding"):
                    sim = _compute_embedding_similarity(
                        current_emb,
                        last_face.normed_embedding
                    )
                else:
                    sim = 0.0

                if sim > best_sim and sim >= MIN_EMBED_SIMILARITY:
                    best_sim = sim
                    best_track_id = track_id

            # update / buat track baru
            if best_track_id is not None:
                prev_face = FACE_TRACKING[best_track_id]['last_face']
                motion = calculate_motion_vector(prev_face, face)
                FACE_TRACKING[best_track_id].update({
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': motion
                })
            else:
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': 0.0
                }

            # smoothing bbox dengan 2 history terakhir
            if len(TRACKING_HISTORY) >= 2:
                recent_faces = list(TRACKING_HISTORY)[-2:]
                if all('bbox' in f for f in recent_faces):
                    smoothed_bbox = np.mean(
                        [f['bbox'] for f in recent_faces],
                        axis=0
                    )
                    face.bbox = smoothed_bbox

            # simpan ke history
            face_data = {
                'bbox': np.array(face.bbox, dtype=np.float32).copy()
            }
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)

        # buang track yang terlalu tua
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }

    return tracked_faces


# =====================================================================
#  OCCLUSION (PAKAI occluder.onnx, TANPA FALLBACK)
# =====================================================================

def detect_occlusion(face: Face, frame: Frame) -> bool:
    """
    Deteksi occlusion berat (tangan, rambut, benda) memakai occluder.onnx saja.
    - Crop area bbox wajah dari frame,
    - Resize & normalisasi,
    - Run occluder,
    - Hitung rasio area occluded,
    - Bandingkan dengan threshold.
    Tidak ada fallback ke det_score atau metode lain.
    """
    try:
        session, input_name = get_occluder_session()

        bbox = np.array(face.bbox, dtype=int)
        x1, y1, x2, y2 = bbox.tolist()

        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            # bbox tidak valid → anggap occluded (fail-closed, tanpa fallback)
            return True

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return True

     target_size = 256
crop_resized = cv2.resize(
    crop,
    (target_size, target_size),
    interpolation=cv2.INTER_LINEAR
)

        # asumsikan model pakai RGB, range 0–1
        crop_resized = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        inp = crop_resized.astype('float32') / 255.0
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]  # 1x3xHxW

        outputs = session.run(None, {input_name: inp})
        mask = np.asarray(outputs[0])
        mask = np.squeeze(mask)

        if mask.ndim == 3:
            mask = mask[0]

        occluded_ratio = float(np.mean(mask > 0.5))

        # threshold bisa diatur dari roop.globals.occlusion_threshold kalau ada
        threshold = getattr(roop.globals, 'occlusion_threshold', 0.15)
        return occluded_ratio >= threshold

    except Exception as e:
        # Tidak ada fallback; jika occluder gagal, anggap occluded
        print(f"[OCCLUDER] Error during occlusion detection: {e}")
        return True


# =====================================================================
#  SIMILAR FACE (EMBEDDING MATCHING)
# =====================================================================

def find_similar_face(
    frame: Frame,
    reference_face: Face,
    use_tracking: bool = True
) -> Optional[Face]:
    """
    Cari wajah paling mirip di frame terhadap reference_face.
    - Bisa pakai smart tracking (use_tracking=True)
    - Atau pakai get_many_faces biasa
    - Pakai embedding distance (normed_embedding buffalo_l)
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
