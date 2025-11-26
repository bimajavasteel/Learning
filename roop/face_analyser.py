from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine
import insightface
import numpy as np
import cv2
import os
import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path
import onnxruntime as ort
# =====================================================================
#  GLOBALS
# =====================================================================
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
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None
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
#  OCCLUDER ONNX (enhanced)
# =====================================================================
def _get_occluder_session() -> Optional[ort.InferenceSession]:
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION
    model_rel = getattr(roop.globals, "occluder_model_path", "../models/occluder.onnx")
    model_path = resolve_relative_path(model_rel)
    if not os.path.exists(model_path):
        print(f"[face_analyser] occluder model not found at {model_path}, fallback ke det_score.")
        return None
    try:
        OCCLUDER_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception as e:
        print(f"[face_analyser] Failed load occluder model: {e}")
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None
    return OCCLUDER_SESSION
def get_occlusion_mask(face: Face, frame: Frame) -> np.ndarray:
    """
    Return detailed occlusion mask (0.0 = visible, 1.0 = fully occluded)
    """
    if frame is None or face is None:
        return np.zeros((128, 128), dtype=np.float32)
    
    x1, y1, x2, y2 = map(int, face.bbox)
    h, w = frame.shape[:2]
    
    # Safeguard bounding box
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((128, 128), dtype=np.float32)
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((128, 128), dtype=np.float32)
    
    session = _get_occluder_session()
    if session is None:
        # Fallback to simple gradient-based occlusion estimation
        return _estimate_occlusion_fallback(crop)
    
    try:
        h_crop, w_crop = crop.shape[:2]
        inp = cv2.resize(crop, (224, 224))
        inp = inp.astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]
        
        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0]
        
        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred[0]
        
        mask = cv2.resize(mask, (w_crop, h_crop))
        mask = np.clip(mask, 0, 1)
        return mask
    except Exception as e:
        print(f"Occlusion mask error: {e}")
        return _estimate_occlusion_fallback(crop)
def _estimate_occlusion_fallback(crop: np.ndarray) -> np.ndarray:
    """
    Fallback occlusion estimation using edge detection and gradients
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Create gradient-based occlusion estimation
    h, w = gray.shape
    y_coords = np.linspace(0, 1, h)[:, np.newaxis]
    x_coords = np.linspace(0, 1, w)[np.newaxis, :]
    
    # Center bias (faces are usually centered)
    center_y = 0.5
    center_x = 0.5
    distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    visibility = 1 - np.clip(distance * 1.5, 0, 1)
    
    # Combine with edge information
    edge_density = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0) / 255.0
    occlusion = np.clip(edge_density * 0.7 + (1 - visibility) * 0.3, 0, 1)
    
    return occlusion
def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    """
    Original function for backward compatibility
    """
    if frame is None:
        return getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD
    
    occlusion_mask = get_occlusion_mask(face, frame)
    occl_ratio = float(np.mean(occlusion_mask > 0.5))
    threshold = getattr(roop.globals, "occluder_threshold", 0.25)
    return occl_ratio > threshold
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
    except ValueError:
        return None
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
#  MOTION & TRACKING
# =====================================================================
def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    if prev_face is None or current_face is None:
        return 0.0
    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox
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
                embedding_similarity = _compute_embedding_similarity(
                    current_embedding, track_embedding
                )
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
            if len(TRACKING_HISTORY) >= 2:
                recent_faces = list(TRACKING_HISTORY)[-2:]
                if all(hasattr(f, 'bbox') for f in recent_faces):
                    smoothed_bbox = np.mean([f.bbox for f in recent_faces], axis=0)
                    face.bbox = smoothed_bbox
            face_data = {
                'bbox': np.array(face.bbox, dtype=np.float32).copy()
            }
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }
    return tracked_faces
# =====================================================================
#  SIMILAR FACE
# =====================================================================
def find_similar_face(frame: Frame,
                      reference_face: Face,
                      use_tracking: bool = True) -> Optional[Face]:
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
