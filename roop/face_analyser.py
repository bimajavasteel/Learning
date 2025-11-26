# face_anlyser2.py
from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os
import time

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path

# optional: kalau kamu punya occluder.onnx
import onnxruntime as ort

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking (penting untuk multi-thread)

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=60)  # lebih besar untuk smoothing temporal

# Threshold / hyper-parameter default (boleh kamu tuning)
MIN_DET_SCORE = 0.50        # naikkan ke 0.5 untuk mengurangi noisy detections
OCCLUSION_THRESHOLD = 0.40  # det_score < ini dianggap occluded (fallback)
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 20
MIN_EMBED_SIMILARITY = 0.70

# Occluder ONNX (opsional)
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None

# simple EMA smoothing factor default
EMA_ALPHA = 0.65

# ===============================================
#  OneEuroFilter (sederhana, cukup untuk smoothing)
# ===============================================
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.x_prev = None
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if self.x_prev is None:
            self.x_prev = np.array(x, dtype=float)
            self.t_prev = t if t is not None else time.time()
            return self.x_prev
        t = t if t is not None else time.time()
        dt = t - self.t_prev if t - self.t_prev > 1e-6 else 1.0 / self.freq
        self.freq = 1.0 / dt
        d = np.linalg.norm(np.array(x, dtype=float) - self.x_prev)
        cutoff = self.min_cutoff + self.beta * d
        a = self.alpha(cutoff)
        self.x_prev = a * np.array(x, dtype=float) + (1 - a) * self.x_prev
        self.t_prev = t
        return self.x_prev

# per-track filter storage
_TRACK_FILTERS: dict[int, dict[str, Any]] = {}

# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            print("✅ [face_analyser] Using buffalo_l (pose + 2d106 + 3d68)")
    return FACE_ANALYSER

def clear_face_analyser() -> None:
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY, _TRACK_FILTERS
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
        _TRACK_FILTERS.clear()
    with THREAD_LOCK:
        FACE_ANALYSER = None

# =====================================================================
#  OCCLUDER ONNX (opsional)
# =====================================================================

def _get_occluder_session() -> Optional[ort.InferenceSession]:
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION
    model_rel = getattr(roop.globals, "occluder_model_path", "../models/occluder.onnx")
    model_path = resolve_relative_path(model_rel)
    if not os.path.exists(model_path):
        # fallback
        return None
    try:
        OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=roop.globals.execution_providers)
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception as e:
        print(f"[face_analyser] Failed load occluder model: {e}")
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None
    return OCCLUDER_SESSION

def _run_occluder_onnx(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    session = _get_occluder_session()
    if session is None:
        return 0.0
    try:
        h, w = crop.shape[:2]
        inp = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]
        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0]
        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred[0]
        mask = cv2.resize(mask, (w, h))
        occl_ratio = float(np.mean(mask > 0.5))
        return occl_ratio
    except Exception:
        return 0.0

# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []
        faces = [face for face in faces if getattr(face, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces
    except Exception:
        return None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_face_pose(face: Face) -> tuple[float, float, float]:
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

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    if prev_face is None or current_face is None:
        return 0.0
    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox
    prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
    current_center = np.array([(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2])
    motion = np.linalg.norm(current_center - prev_center)
    return float(motion)

def _compute_embedding_similarity(current_embedding: np.ndarray, track_embedding: np.ndarray) -> float:
    try:
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0

def _apply_bbox_smoothing(track_id: int, face: Face):
    """
    Smoothing per-track untuk bbox & landmarks menggunakan OneEuroFilter sederhana.
    """
    if track_id not in _TRACK_FILTERS:
        _TRACK_FILTERS[track_id] = {
            "bbox_filter": OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.001),
            "landmark_filter": OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.002)
        }
    f = _TRACK_FILTERS[track_id]
    try:
        sm_bbox = f["bbox_filter"].filter(np.array(face.bbox, dtype=float))
        face.bbox = np.array(sm_bbox, dtype=np.float32)
    except Exception:
        pass

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
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
                # create new track id
                # reuse smallest missing positive integer for stability
                new_id = 1
                while new_id in FACE_TRACKING:
                    new_id += 1
                face_id = new_id
                FACE_TRACKING[face_id] = {'last_face': face, 'last_seen': frame_number, 'motion': 0.0}

            # apply smoothing per-track
            _apply_bbox_smoothing(face_id, face)

            # save to history
            face_data = {'bbox': np.array(face.bbox, dtype=np.float32).copy()}
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)

        # cleanup old tracks
        FACE_TRACKING = {k: v for k, v in list(FACE_TRACKING.items()) if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE}

    return tracked_faces

# =====================================================================
#  OCCLUSION & SIMILAR FACE
# =====================================================================

def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    base_flag = getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD
    if frame is None:
        return base_flag
    occl_session = _get_occluder_session()
    if occl_session is None:
        return base_flag
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return base_flag
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return base_flag
        occl_score = _run_occluder_onnx(crop)
        threshold = getattr(roop.globals, "occluder_threshold", 0.20)
        return occl_score > threshold
    except Exception:
        return base_flag

def find_similar_face(frame: Frame, reference_face: Face, use_tracking: bool = True) -> Optional[Face]:
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
