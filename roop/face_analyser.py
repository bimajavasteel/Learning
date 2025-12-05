# face_analyser.py
from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine
from roop.hand_occlusion import get_hand_mask

import insightface
import numpy as np
import cv2
import os

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path

# optional occluder / hand segmentation ONNX
import onnxruntime as ort

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Threshold / hyper-parameter default (boleh kamu tuning)
MIN_DET_SCORE = 0.30        # untuk filter deteksi kasar
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

# Occluder ONNX (opsional)
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None
OCCLUDER_INPUT_SIZE = (224, 224)   # ubah jika model beda

# Hand segmentation ONNX (opsional)
HANDSEG_SESSION: Optional[ort.InferenceSession] = None
HANDSEG_INPUT_NAME: Optional[str] = None
HANDSEG_INPUT_SIZE = (256, 256)

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
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
    with THREAD_LOCK:
        FACE_ANALYSER = None

# =====================================================================
#  ONNX loaders
# =====================================================================

def _get_occluder_session() -> Optional[ort.InferenceSession]:
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION

    model_rel = getattr(roop.globals, "occluder_model_path", "../models/occluder.onnx")
    model_path = resolve_relative_path(model_rel)

    if not os.path.exists(model_path):
        return None

    try:
        OCCLUDER_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        # Try to read expected input size if possible (optional)
        try:
            shape = OCCLUDER_SESSION.get_inputs()[0].shape
            if shape and len(shape) >= 4:
                OCCLUDER_INPUT_SIZE = (int(shape[-1]), int(shape[-2]))
        except Exception:
            pass
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception as e:
        print(f"[face_analyser] Failed load occluder model: {e}")
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None

    return OCCLUDER_SESSION


def _get_handseg_session() -> Optional[ort.InferenceSession]:
    global HANDSEG_SESSION, HANDSEG_INPUT_NAME
    if HANDSEG_SESSION is not None:
        return HANDSEG_SESSION

    model_rel = getattr(roop.globals, "handseg_model_path", "../models/handseg.onnx")
    model_path = resolve_relative_path(model_rel)

    if not os.path.exists(model_path):
        return None

    try:
        HANDSEG_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        HANDSEG_INPUT_NAME = HANDSEG_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded handseg model: {model_path}")
    except Exception as e:
        print(f"[face_analyser] Failed load handseg model: {e}")
        HANDSEG_SESSION = None
        HANDSEG_INPUT_NAME = None

    return HANDSEG_SESSION

# =====================================================================
#  RUN ONNX helpers
# =====================================================================

def _run_occluder_onnx(crop: np.ndarray) -> float:
    """
    Run occluder.onnx on crop. Returns occlusion score [0..1].
    If model not available or error -> return 0.0
    """
    if crop is None or crop.size == 0:
        return 0.0

    session = _get_occluder_session()
    if session is None:
        return 0.0

    try:
        h, w = crop.shape[:2]
        target_w, target_h = OCCLUDER_INPUT_SIZE
        inp = cv2.resize(crop, (target_w, target_h))
        inp = inp.astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW

        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0]

        # handle: output could be [1,1,H,W] (mask) or [1,H,W] or [1,N]
        if pred.ndim == 4:
            mask = pred[0, 0]
        elif pred.ndim == 3:
            mask = pred[0]
        else:
            # if single scalar or vector, try to interpret average
            flat = np.array(pred).flatten()
            if flat.size == 0:
                return 0.0
            return float(np.mean(flat))

        # resize mask back to crop size
        mask = cv2.resize(mask, (w, h))
        # use average intensity (lebih sensitif untuk partial occlusion)
        occl_score = float(np.mean(mask))
        # clamp
        occl_score = max(0.0, min(1.0, occl_score))
        return occl_score
    except Exception:
        return 0.0


def _run_handseg_onnx(crop: np.ndarray) -> np.ndarray:
    """
    Run hand segmentation model. Return mask float32 [0..1] same HxW as crop.
    If model not available or error -> return zero-mask.
    """
    if crop is None or crop.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    session = _get_handseg_session()
    if session is None:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        target_w, target_h = HANDSEG_INPUT_SIZE
        inp = cv2.resize(crop, (target_w, target_h))
        inp = inp.astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]
        outputs = session.run(None, {HANDSEG_INPUT_NAME: inp})
        pred = outputs[0]
        # Normalize to 0..1 and resize back
        if pred.ndim == 4:
            mask = pred[0, 0]
        elif pred.ndim == 3:
            mask = pred[0]
        else:
            mask = np.array(pred).squeeze()
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.astype(np.float32)
    except Exception:
        return np.zeros((crop.shape[0], crop.shape[1]), dtype=np.float32)

# =====================================================================
#  BASIC FACE ACCESSORS (sama seperti sebelumnya)
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
        pitch = float(pose[0])
        yaw = float(pose[1])
        roll = float(pose[2])
        return pitch, yaw, roll
    except Exception:
        return 0.0, 0.0, 0.0

# =====================================================================
#  MOTION & TRACKING (sama seperti sebelumnya)
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
                FACE_TRACKING[face_id].update({
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
            # smoothing bbox sederhana
            if len(TRACKING_HISTORY) >= 2:
                recent_faces = list(TRACKING_HISTORY)[-2:]
                if all('bbox' in f for f in recent_faces):
                    smoothed_bbox = np.mean([f['bbox'] for f in recent_faces], axis=0)
                    face.bbox = smoothed_bbox
            face_data = {'bbox': np.array(face.bbox, dtype=np.float32).copy()}
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }
    return tracked_faces

# =====================================================================
#  OCCLUSION DETECTION: gabungan landmark, onnx occluder, handseg overlap, motion
# =====================================================================

def detect_occlusion(face: Face, frame: Optional[Frame] = None, prev_face: Optional[Face] = None) -> bool:
    """
    Returns True if face is considered occluded (partially or heavily).
    Strategy:
    1) Landmark visibility heuristic (mata/mulut)
    2) ONNX occluder average-score (sensitive)
    3) Hand segmentation overlap ratio (jika model handseg ada)
    4) Motion + bbox-change heuristic (jika prev_face disediakan)
    """
    # 1) Landmark visibility
    try:
        lm = getattr(face, "landmark_2d_106", None)
        if lm is not None and len(lm) >= 90:
            # indices depend on buffalo_l landmarks mapping; adapt jika berbeda
            # gunakan area Y-variance sebagai proxy visibility: tangan menutup cenderung meratakan intensitas
            left_eye = np.array(lm[60:68])
            right_eye = np.array(lm[68:76])
            mouth = np.array(lm[76:90])
            # jika landmark vertikal sangat rendah variance -> kemungkinan tertutup / tidak terdeteksi
            if np.std(left_eye[:,1]) < 1.5 or np.std(right_eye[:,1]) < 1.5:
                return True
            if np.std(mouth[:,1]) < 1.2:
                return True
    except Exception:
        pass

    # prepare bbox-crop if frame available
    occl_score = 0.0
    hand_overlap = 0.0
    if frame is not None:
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    # 2) occluder model
                    occl_score = _run_occluder_onnx(crop)
                    # 3) hand segmentation
                    hand_mask = _run_handseg_onnx(crop)
                    if hand_mask.size > 0:
                        # compute hand overlap ratio wrt face bbox area
                        hand_overlap = float(np.mean(hand_mask > 0.3))
        except Exception:
            pass

    # thresholds (boleh di roop.globals)
    occl_threshold = getattr(roop.globals, "occluder_threshold", 0.08)
    hand_overlap_threshold = getattr(roop.globals, "hand_occlusion_threshold", 0.06)

    if occl_score >= occl_threshold:
        return True
    if hand_overlap >= hand_overlap_threshold:
        return True

    # 4) motion heuristic: jika prev_face ada dan motion tiba-tiba besar saat ada occlusion proxy -> occluded
    try:
        if prev_face is not None:
            motion = calculate_motion_vector(prev_face, face)
            w_box = abs(face.bbox[2] - face.bbox[0])
            if motion > max(2.0, 0.35 * w_box) and (occl_score > 0.02 or hand_overlap > 0.01):
                return True
    except Exception:
        pass

    # default: not occluded
    return False

# =====================================================================
#  find_similar_face (sama logic tapi memakai smart tracking)
# =====================================================================

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
