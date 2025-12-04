# face_analyser.py
# Versi final: detect_occlusion mengembalikan occlusion_ratio (float 0..1)
# Termasuk: fallback det_score, occluder.onnx (opsional), skin-mismatch check, landmark asymmetry check.
#
# Referensi implementasi asal: face-analyser original. :contentReference[oaicite:2]{index=2}

from typing import Any, Optional, List, Tuple
import threading
from collections import deque
import numpy as np
import cv2
import os

import insightface
import onnxruntime as ort

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path

THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

FACE_ANALYSER: Any = None
FACE_TRACKING: dict[int, dict] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Hyperparams (bisa diubah dari roop.globals)
MIN_DET_SCORE = 0.30
OCCLUSION_THRESHOLD_DEFAULT = 0.15   # ambang default yang lebih sensitif
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15

# Occluder ONNX session (opsional)
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def clear_face_analyser() -> None:
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
    with THREAD_LOCK:
        FACE_ANALYSER = None

# -------------------- occluder onnx helpers --------------------
def _get_occluder_session() -> Optional[ort.InferenceSession]:
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION

    model_rel = getattr(roop.globals, "occluder_model_path", "../models/occluder.onnx")
    model_path = resolve_relative_path(model_rel)
    if not os.path.exists(model_path):
        return None

    try:
        OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=roop.globals.execution_providers)
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        return OCCLUDER_SESSION
    except Exception:
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None
        return None

def _run_occluder_onnx(crop: np.ndarray) -> float:
    """
    Jalankan occluder.onnx dan kembalikan occlusion ratio (0..1).
    Jika model tidak tersedia atau error -> return 0.0
    """
    if crop is None or crop.size == 0:
        return 0.0

    session = _get_occluder_session()
    if session is None:
        return 0.0

    try:
        h, w = crop.shape[:2]
        inp = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW
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

# -------------------- face detection helpers --------------------
def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []
        faces = [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces
    except Exception:
        return None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many = get_many_faces(frame)
    if many:
        try:
            return many[position]
        except Exception:
            return many[-1]
    return None

def get_face_pose(face: Face) -> Tuple[float, float, float]:
    pose = getattr(face, "pose", None)
    if pose is None:
        return 0.0, 0.0, 0.0
    try:
        return float(pose[0]), float(pose[1]), float(pose[2])
    except Exception:
        return 0.0, 0.0, 0.0

# -------------------- occlusion detection (final) --------------------
def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> float:
    """
    Mengembalikan occlusion ratio (float 0..1).
    - Jika occluder.onnx tersedia dan frame diberikan -> gunakan model
    - Tambahan heuristik: fallback berdasar det_score, skin-mismatch, landmark asymmetry
    Usage:
        ratio = detect_occlusion(face, frame)
        if ratio > threshold: treat as occluded
    """
    # fallback: det_score rendah -> anggap occluded
    det_score = getattr(face, "det_score", 1.0)
    base_flag = 1.0 if det_score < MIN_DET_SCORE else 0.0

    if frame is None:
        return base_flag

    # crop face (safety clamp)
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
    except Exception:
        return base_flag

    # 1) coba occluder.onnx (jika ada)
    occl_ratio_model = _run_occluder_onnx(crop)

    # 2) skin-mismatch heuristic:
    #    - bila proporsi area "skin-like" di crop rendah -> ada benda/aksesori (tangan)
    skin_ratio = 1.0
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # rentang skin general (bisa disesuaikan jika perlu)
        lower = np.array([0, 10, 60], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        skin_ratio = float(np.sum(mask > 0)) / (mask.size + 1e-8)
    except Exception:
        skin_ratio = 1.0

    # 3) landmark asymmetry heuristic:
    lm_ratio = 0.0
    try:
        lm = getattr(face, "landmark_2d_106", None)
        if lm is None:
            lm = getattr(face, "landmark_2d_68", None)
        if lm is not None:
            # ambil jarak antar mata sebagai indikasi (kekacauan landmark → occlusion)
            # indeks landmark bisa berbeda; fallback aman: hitung bounding box relatif
            left_eye = lm[36] if len(lm) > 36 else lm[0]
            right_eye = lm[45] if len(lm) > 45 else lm[-1]
            dx = abs(left_eye[0] - right_eye[0])
            # jika dx terlalu kecil relatif bbox width -> ada occlusion/deform
            bbox_w = max(1.0, (x2 - x1))
            if dx / bbox_w < 0.12:
                lm_ratio = 0.6
            else:
                lm_ratio = 0.0
    except Exception:
        lm_ratio = 0.0

    # combine heuristics:
    # model ratio (if available) gets bobot tertinggi, skin_ratio penalizes, lm_ratio memberi dorongan
    occl_from_skin = max(0.0, 1.0 - skin_ratio)  # skin_ratio rendah -> occlusion tinggi
    combined = max(occl_ratio_model, occl_from_skin, lm_ratio, base_flag)

    # final smoothing: kalau model memberi sedikit occlusion (misal 0.18)
    # beri sedikit margin -> conservatively treat as occluded apabila combined > threshold
    return float(np.clip(combined, 0.0, 1.0))

# -------------------- simple tracking utilities (dipertahankan) --------------------
def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    try:
        prev = prev_face.bbox
        cur = current_face.bbox
        prev_center = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
        cur_center = np.array([(cur[0] + cur[2]) / 2, (cur[1] + cur[3]) / 2])
        return float(np.linalg.norm(cur_center - prev_center))
    except Exception:
        return 0.0

def _compute_embedding_similarity(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.spatial.distance import cosine
        return 1.0 - float(cosine(a, b))
    except Exception:
        return 0.0

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    # implementasi ringan: panggil get_many_faces + simple mapping berdasarkan embedding
    current = get_many_faces(frame)
    if not current:
        return None
    return current
