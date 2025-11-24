# roop/face_analyser.py
"""
Face analyser: wrapper InsightFace untuk deteksi, pose, landmark, embedding, dan tracking ringan.
Modular, thread-safe, dan kompatibel dengan face_swapper ReSwapper backend.

Pastikan dependency terpasang (Kaggle):
pip install insightface==0.7.4 opencv-python-headless scipy onnxruntime
"""
from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os

# Optional occluder (ONNX)
try:
    import onnxruntime as ort
except Exception:
    ort = None

# =====================================================================
#  GLOBALS & HYPERPARAMS
# =====================================================================
FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking (penting untuk multi-thread)

FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Threshold / hyper-parameter default (boleh disesuaikan via roop.globals)
MIN_DET_SCORE = 0.30        # min score agar wajah dianggap valid (untuk get_many_faces)
OCCLUSION_THRESHOLD = 0.40  # det_score < ini dianggap occluded (fallback)
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None


# =====================================================================
#  MODEL HANDLING
# =====================================================================
def get_face_analyser() -> Any:
    """
    Lazy init insightface FaceAnalysis (buffalo_l recommended).
    Thread-safe. Memakai CPU fallback bila CUDA tidak tersedia.
    """
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            providers = ['CPUExecutionProvider']
            # allow CUDA if available in insightface build
            try:
                # insightface may support 'CUDAExecutionProvider' if onnxruntime-gpu present
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except Exception:
                providers = ['CPUExecutionProvider']

            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            # prepare dengan ctx_id=0 (jika tidak tersedia, FaceAnalysis akan fallback)
            try:
                FACE_ANALYSER.prepare(ctx_id=0)
            except Exception:
                # fallback tanpa ctx_id
                FACE_ANALYSER.prepare()
            print("✅ [face_analyser] Using buffalo_l (pose + 2d106 + embedding)")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    """
    Reset analyser & tracking state.
    Dipanggil saat post_process / cleanup.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
    with THREAD_LOCK:
        FACE_ANALYSER = None


# =====================================================================
#  OCCLUDER ONNX (opsional)
# =====================================================================
def _get_occluder_session(model_path: Optional[str] = None) -> Optional[ort.InferenceSession]:
    """
    Lazy init occluder.onnx.
    Kalau file tidak ada / gagal load → return None dan sistem fallback ke det_score.
    """
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if ort is None:
        return None
    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION

    if model_path is None:
        return None
    if not os.path.exists(model_path):
        return None

    try:
        OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception:
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None
    return OCCLUDER_SESSION


def _run_occluder_onnx(crop: np.ndarray) -> float:
    """
    Jalankan occluder.onnx di atas crop wajah.
    Return: occlusion score 0–1 (semakin besar artinya semakin tertutup).
    Kalau model tidak tersedia / error → return 0.0 (anggap tidak occluded).
    """
    if crop is None or crop.size == 0 or ort is None:
        return 0.0
    session = OCCLUDER_SESSION
    if session is None:
        return 0.0
    try:
        inp = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW
        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0]
        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred[0]
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))
        occl_ratio = float(np.mean(mask > 0.5))
        return occl_ratio
    except Exception:
        return 0.0


# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================
def get_many_faces(frame: np.ndarray) -> Optional[List[Any]]:
    """
    Deteksi banyak wajah di satu frame.
    - Pakai buffalo_l
    - Filter berdasarkan det_score minimal (untuk video dance / gerak cepat)
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []
        faces = [face for face in faces if getattr(face, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces
    except ValueError:
        return None
    except Exception:
        return None


def get_one_face(frame: np.ndarray, position: int = 0) -> Optional[Any]:
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


def get_face_pose(face: Any) -> tuple[float, float, float]:
    """
    Ambil pose dari Face (pitch, yaw, roll) dalam derajat.
    InsightFace menyimpan di face.pose dengan urutan (pitch, yaw, roll).
    """
    pose = getattr(face, "pose", None)
    if pose is None:
        return 0.0, 0.0, 0.0
    try:
        pitch = float(pose[0]); yaw = float(pose[1]); roll = float(pose[2])
        return pitch, yaw, roll
    except Exception:
        return 0.0, 0.0, 0.0


# =====================================================================
#  MOTION & TRACKING
# =====================================================================
def calculate_motion_vector(prev_face: Any, current_face: Any) -> float:
    if prev_face is None or current_face is None:
        return 0.0
    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox
    prev_center = np.array([ (prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2 ])
    current_center = np.array([ (current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2 ])
    return float(np.linalg.norm(current_center - prev_center))


def _compute_embedding_similarity(current_embedding: np.ndarray, track_embedding: np.ndarray) -> float:
    try:
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


def smart_face_tracking(frame: np.ndarray, frame_number: int) -> Optional[List[Any]]:
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
    tracked_faces: List[Any] = []
    with TRACK_LOCK:
        for face in current_faces:
            face_id = None
            max_similarity = MIN_EMBED_SIMILARITY
            best_match_id = None
            current_embedding = getattr(face, "normed_embedding", None)
            if current_embedding is None or len(current_embedding) == 0:
                current_embedding = np.array([])
            for track_id, track_data in list(FACE_TRACKING.items()):
                if frame_number - track_data.get('last_seen', -9999) > MAX_TRACK_GAP:
                    continue
                last_face = track_data.get('last_face', None)
                if last_face is None:
                    continue
                track_embedding = getattr(last_face, "normed_embedding", None)
                if track_embedding is None:
                    continue
                embedding_similarity = _compute_embedding_similarity(current_embedding, track_embedding)
                if embedding_similarity > max_similarity:
                    max_similarity = embedding_similarity
                    best_match_id = track_id
            if best_match_id is not None:
                face_id = best_match_id
                prev_face = FACE_TRACKING[face_id]['last_face']
                motion = calculate_motion_vector(prev_face, face)
                FACE_TRACKING[face_id].update({'last_face': face, 'last_seen': frame_number, 'motion': motion})
            else:
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {'last_face': face, 'last_seen': frame_number, 'motion': 0.0}
            # smoothing bbox sederhana dengan history 2 frame terakhir
            if len(TRACKING_HISTORY) >= 2:
                recent_faces = list(TRACKING_HISTORY)[-2:]
                if all('bbox' in f for f in recent_faces):
                    smoothed_bbox = np.mean([f['bbox'] for f in recent_faces], axis=0)
                    face.bbox = smoothed_bbox
            face_data = { 'bbox': np.array(face.bbox, dtype=np.float32).copy() }
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)
        # bersihkan track yang sudah terlalu tua
        FACE_TRACKING = { k: v for k, v in list(FACE_TRACKING.items()) if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE }
    return tracked_faces


# =====================================================================
#  OCCLUSION & SIMILAR FACE
# =====================================================================
def detect_occlusion(face: Any, frame: Optional[np.ndarray] = None, occluder_path: Optional[str] = None) -> bool:
    """
    Deteksi wajah yang ter-occlusion (tertutup tangan, rambut, dsb).
    Prioritas:
    1. Kalau occluder.onnx tersedia & frame disediakan: pakai occlusion model
    2. Kalau tidak: fallback ke det_score < OCCLUSION_THRESHOLD
    """
    base_flag = getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD
    if frame is None:
        return base_flag
    # lazy init occluder jika path diberikan
    if occluder_path:
        _get_occluder_session(occluder_path)
    if OCCLUDER_SESSION is None:
        return base_flag
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return base_flag
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return base_flag
        occl_score = _run_occluder_onnx(crop)
        threshold = 0.20
        return occl_score > threshold
    except Exception:
        return base_flag


def find_similar_face(frame: np.ndarray, reference_face: Any, use_tracking: bool = True) -> Optional[Any]:
    """
    Cari wajah paling mirip di frame terhadap reference_face.
    - Bisa pakai smart tracking (use_tracking=True)
    - Atau fallback ke get_many_faces biasa
    - Menggunakan embedding distance seperti di mod sebelumnya
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
    similar_threshold = 1.0
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
