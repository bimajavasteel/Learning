# roop/face_analyser.py
"""
Face analyser: wrapper InsightFace untuk deteksi, pose, landmark, embedding, dan tracking ringan.
Modular, thread-safe, dan kompatibel dengan face_swapper yang dibuat untuk ReSwapper.
Dasar kode diambil dan direfaktor dari file user (face-anlyser2.txt).
"""
from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os

# Optional occluder
try:
    import onnxruntime as ort
except Exception:
    ort = None

# Globals (konfigurasi default — boleh diubah via roop.globals)
FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

MIN_DET_SCORE = 0.30
OCCLUSION_THRESHOLD = 0.40
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

OCCLUDER_SESSION = None
OCCLUDER_INPUT_NAME = None


def get_face_analyser() -> Any:
    """
    Lazy init InsightFace FaceAnalysis (buffalo_l recommended).
    Thread-safe.
    """
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            print("✅ [face_analyser] Using buffalo_l (pose + landmark + embedding).")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
    with THREAD_LOCK:
        FACE_ANALYSER = None


# ----------------- occluder (opsional) -----------------
def _get_occluder_session(model_path: Optional[str] = None):
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if ort is None:
        return None
    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION
    if model_path is None:
        return None
    if not os.path.exists(model_path):
        print(f"[face_analyser] occluder model tidak ditemukan: {model_path}")
        return None
    try:
        OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder: {model_path}")
    except Exception as e:
        print(f"[face_analyser] Gagal load occluder: {e}")
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None
    return OCCLUDER_SESSION


def _run_occluder_onnx(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0 or ort is None:
        return 0.0
    session = OCCLUDER_SESSION
    if session is None:
        return 0.0
    try:
        inp = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]
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


# ----------------- accessors -----------------
def get_many_faces(frame: np.ndarray) -> Optional[List[Any]]:
    """
    Kembalikan list face objects dari InsightFace yang confidence >= MIN_DET_SCORE.
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []
        faces = [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces
    except Exception:
        return None


def get_one_face(frame: np.ndarray, position: int = 0) -> Optional[Any]:
    many = get_many_faces(frame)
    if many:
        try:
            return many[position]
        except IndexError:
            return many[-1]
    return None


def get_face_pose(face: Any) -> tuple[float, float, float]:
    pose = getattr(face, "pose", None)
    if pose is None:
        return 0.0, 0.0, 0.0
    try:
        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
        return pitch, yaw, roll
    except Exception:
        return 0.0, 0.0, 0.0


# ----------------- tracking -----------------
def calculate_motion_vector(prev_face: Any, current_face: Any) -> float:
    if prev_face is None or current_face is None:
        return 0.0
    prev_bbox = prev_face.bbox
    cur_bbox = current_face.bbox
    prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
    cur_center = np.array([(cur_bbox[0] + cur_bbox[2]) / 2, (cur_bbox[1] + cur_bbox[3]) / 2])
    return float(np.linalg.norm(cur_center - prev_center))


def _compute_embedding_similarity(current_embedding: np.ndarray, track_embedding: np.ndarray) -> float:
    try:
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


def smart_face_tracking(frame: np.ndarray, frame_number: int) -> Optional[List[Any]]:
    """
    Tracking ringan berbasis embedding similarity & motion.
    """
    global FACE_TRACKING, TRACKING_HISTORY
    current_faces = get_many_faces(frame)
    if not current_faces:
        return None
    tracked_faces = []
    with TRACK_LOCK:
        for face in current_faces:
            face_id = None
            max_similarity = MIN_EMBED_SIMILARITY
            best_match_id = None
            current_embedding = getattr(face, "normed_embedding", None)
            if current_embedding is None:
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
                sim = _compute_embedding_similarity(current_embedding, track_embedding)
                if sim > max_similarity:
                    max_similarity = sim
                    best_match_id = track_id
            if best_match_id is not None:
                face_id = best_match_id
                prev_face = FACE_TRACKING[face_id]['last_face']
                motion = calculate_motion_vector(prev_face, face)
                FACE_TRACKING[face_id].update({'last_face': face, 'last_seen': frame_number, 'motion': motion})
            else:
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {'last_face': face, 'last_seen': frame_number, 'motion': 0.0}
            # smoothing bbox menggunakan TRACKING_HISTORY
            if len(TRACKING_HISTORY) >= 2:
                recent = list(TRACKING_HISTORY)[-2:]
                if all('bbox' in f for f in recent):
                    smoothed_bbox = np.mean([f['bbox'] for f in recent], axis=0)
                    face.bbox = smoothed_bbox
            TRACKING_HISTORY.append({'bbox': np.array(face.bbox, dtype=np.float32).copy()})
            tracked_faces.append(face)
        # bersihkan track lama
        FACE_TRACKING = {k: v for k, v in list(FACE_TRACKING.items()) if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE}
    return tracked_faces


# ----------------- occlusion & similar -----------------
def detect_occlusion(face: Any, frame: Optional[np.ndarray] = None, occluder_path: Optional[str] = None) -> bool:
    """
    Deteksi occlusion. Prioritas: occluder.onnx bila tersedia, else fallback det_score.
    """
    base_flag = getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD
    if frame is None:
        return base_flag
    if occluder_path:
        _get_occluder_session(occluder_path)
    if OCCLUDER_SESSION is None:
        return base_flag
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w)); y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
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
    Cari wajah paling mirip di frame berdasarkan embedding L2.
    """
    if reference_face is None:
        return None
    if use_tracking:
        many = smart_face_tracking(frame, frame_number=0)
    else:
        many = get_many_faces(frame)
    if not many:
        return None
    if not hasattr(reference_face, "normed_embedding"):
        return None
    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')
    similar_threshold = 1.0
    for f in many:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue
        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f
    return best_face
