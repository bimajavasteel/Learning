from typing import Any, Optional, List, Tuple
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

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

# Tracking Globals
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Config
MIN_DET_SCORE = 0.30
OCCLUSION_THRESHOLD = 0.40
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

# Occluder
OCCLUDER_SESSION: Any = None
OCCLUDER_INPUT_NAME: Optional[str] = None

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def clear_face_analyser() -> None:
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
    with THREAD_LOCK:
        FACE_ANALYSER = None

# --- Occluder Logic ---
def _get_occluder_session() -> Any:
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME
    if OCCLUDER_SESSION: return OCCLUDER_SESSION
    
    import onnxruntime as ort
    model_path = resolve_relative_path('../models/occluder.onnx')
    if not os.path.exists(model_path): return None
    
    try:
        OCCLUDER_SESSION = ort.InferenceSession(model_path, providers=roop.globals.execution_providers)
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
    except: OCCLUDER_SESSION = None
    return OCCLUDER_SESSION

def _run_occluder_onnx(crop: np.ndarray) -> float:
    if crop.size == 0: return 0.0
    sess = _get_occluder_session()
    if not sess: return 0.0
    try:
        inp = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]
        outputs = sess.run(None, {OCCLUDER_INPUT_NAME: inp})
        mask = outputs[0][0, 0] if outputs[0].ndim == 4 else outputs[0][0]
        return float(np.mean(mask > 0.5))
    except: return 0.0

def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    base_flag = getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD
    if frame is None or _get_occluder_session() is None: return base_flag
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return _run_occluder_onnx(crop) > getattr(roop.globals, "occluder_threshold", 0.20)
    except: return base_flag

# --- Face Accessors ---
def get_many_faces(frame: Frame) -> List[Face]:
    try:
        faces = get_face_analyser().get(frame)
        return [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE] if faces else []
    except: return []

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    faces = get_many_faces(frame)
    if faces:
        try: return faces[position]
        except IndexError: return faces[-1]
    return None

def get_face_pose(face: Face) -> Tuple[float, float, float]:
    pose = getattr(face, "pose", None)
    if pose is None: return 0.0, 0.0, 0.0
    return float(pose[0]), float(pose[1]), float(pose[2])

# --- Tracking Logic ---
def smart_face_tracking(frame: Frame, frame_number: int) -> List[Face]:
    global FACE_TRACKING, TRACKING_HISTORY
    current_faces = get_many_faces(frame)
    if not current_faces: return []
    
    tracked_faces = []
    with TRACK_LOCK:
        for face in current_faces:
            best_match_id = None
            max_sim = MIN_EMBED_SIMILARITY
            curr_emb = getattr(face, "normed_embedding", np.array([]))
            
            for tid, data in list(FACE_TRACKING.items()):
                if frame_number - data.get('last_seen', -9999) > MAX_TRACK_GAP: continue
                last_emb = getattr(data['last_face'], "normed_embedding", None)
                if last_emb is not None:
                    sim = 1.0 - float(cosine(curr_emb, last_emb))
                    if sim > max_sim:
                        max_sim = sim
                        best_match_id = tid
            
            face_id = best_match_id if best_match_id is not None else len(FACE_TRACKING) + 1
            FACE_TRACKING[face_id] = {'last_face': face, 'last_seen': frame_number}
            
            # Attach ID for swapper
            setattr(face, 'face_id', face_id)
            tracked_faces.append(face)
            
            # Simple bbox smoothing
            if len(TRACKING_HISTORY) >= 2:
                recent = list(TRACKING_HISTORY)[-2:]
                if all('bbox' in f for f in recent):
                    face.bbox = np.mean([f['bbox'] for f in recent] + [face.bbox], axis=0)
            TRACKING_HISTORY.append({'bbox': face.bbox})

        # Cleanup old tracks
        FACE_TRACKING = {k: v for k, v in list(FACE_TRACKING.items()) 
                         if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE}
                         
    return tracked_faces
