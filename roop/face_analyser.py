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

# optional: kalau kamu punya occluder.onnx
import onnxruntime as ort

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Threshold / hyper-parameter
MIN_DET_SCORE = 0.30        # min score agar wajah dianggap valid
OCCLUSION_THRESHOLD = 0.40  # det_score < ini dianggap occluded (fallback)

MAX_TRACK_GAP = 10          # frame: kalau lebih lama dari ini -> track di-skip
MAX_TRACK_AGE = 15          # frame: track dihapus bila tidak terlihat
MIN_EMBED_SIMILARITY = 0.70 # cosine similarity minimal

# [MOD] EMA Factor untuk smoothing bbox (0.0 - 1.0)
# Semakin kecil (misal 0.2) = Sangat halus tapi agak delay (bagus untuk video lambat)
# Semakin besar (misal 0.7) = Responsif tapi agak jitter
# 0.5 adalah keseimbangan yang baik.
EMA_ALPHA = 0.5 

# Occluder ONNX (opsional)
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
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None


# =====================================================================
#  OCCLUDER ONNX (Logic Anti-Tangan)
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
    except Exception as e:
        print(f"[face_analyser] Failed load occluder model: {e}")
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None

    return OCCLUDER_SESSION


def _run_occluder_onnx(crop: np.ndarray) -> float:
    """Mengembalikan skor rata-rata occlusion (0.0 - 1.0)."""
    if crop is None or crop.size == 0:
        return 0.0

    session = _get_occluder_session()
    if session is None:
        return 0.0

    try:
        h, w = crop.shape[:2]
        inp = cv2.resize(crop, (224, 224))
        inp = inp.astype('float32') / 255.0
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

# [MOD] FUNGSI BARU: Expose Mask Occlusion untuk Swapper/Enhancer
def get_occlusion_mask(face: Face, frame: Frame) -> Optional[np.ndarray]:
    """
    Mengembalikan MASK occlusion (bukan cuma skor).
    Output: Grayscale float32 image (0.0=bersih, 1.0=tangan/objek), ukuran sesuai bbox wajah.
    """
    session = _get_occluder_session()
    if session is None or frame is None:
        return None

    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h_frame, w_frame = frame.shape[:2]

        # Clamp coordinates agar tidak error crop
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Preprocessing standar occluder.onnx
        h_crop, w_crop = crop.shape[:2]
        inp = cv2.resize(crop, (224, 224))
        inp = inp.astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...] # NCHW

        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0] # output shape biasanya [1, 1, 224, 224]

        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred[0]

        # Resize mask kembali ke ukuran crop asli agar mapping pixel pas
        mask = cv2.resize(mask, (w_crop, h_crop))
        
        # Mask ini akan dipakai oleh swapper/enhancer untuk blending
        return mask

    except Exception as e:
        print(f"Error getting occlusion mask: {e}")
        return None


# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []
        # filter det_score
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
        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
        return pitch, yaw, roll
    except Exception:
        return 0.0, 0.0, 0.0


# =====================================================================
#  MOTION & TRACKING (Modified for Anti-Flicker)
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    if prev_face is None or current_face is None:
        return 0.0
    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox
    prev_center = np.array([(prev_bbox[0]+prev_bbox[2])/2, (prev_bbox[1]+prev_bbox[3])/2])
    current_center = np.array([(current_bbox[0]+current_bbox[2])/2, (current_bbox[1]+current_bbox[3])/2])
    return float(np.linalg.norm(current_center - prev_center))


def _compute_embedding_similarity(current_embedding: np.ndarray, track_embedding: np.ndarray) -> float:
    try:
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """
    [MODIFIED] Menggunakan EMA (Exponential Moving Average) untuk smoothing bbox.
    Ini mengatasi flicker/jitter jauh lebih baik daripada rata-rata biasa.
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

            current_embedding = getattr(face, "normed_embedding", np.array([]))

            # Matching Logic (Embedding based)
            for track_id, track_data in list(FACE_TRACKING.items()):
                if frame_number - track_data.get('last_seen', -9999) > MAX_TRACK_GAP:
                    continue

                last_face = track_data.get('last_face', None)
                if last_face is None: 
                    continue

                track_embedding = getattr(last_face, "normed_embedding", None)
                if track_embedding is None:
                    continue

                similarity = _compute_embedding_similarity(current_embedding, track_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = track_id

            # [MOD] EMA SMOOTHING LOGIC
            current_bbox = np.array(face.bbox, dtype=np.float32)

            if best_match_id is not None:
                # Update track yang ada
                face_id = best_match_id
                prev_track = FACE_TRACKING[face_id]
                prev_face = prev_track['last_face']
                
                # Ambil bbox yang sudah di-smooth dari frame sebelumnya (jika ada)
                prev_smooth_bbox = prev_track.get('smoothed_bbox', np.array(prev_face.bbox, dtype=np.float32))

                # RUMUS EMA: Smooth = (alpha * Current) + ((1-alpha) * Previous)
                smoothed_bbox = (EMA_ALPHA * current_bbox) + ((1.0 - EMA_ALPHA) * prev_smooth_bbox)

                # Update face object saat ini dengan koordinat stabil
                face.bbox = smoothed_bbox

                motion = calculate_motion_vector(prev_face, face)
                
                FACE_TRACKING[face_id].update({
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': motion,
                    'smoothed_bbox': smoothed_bbox # Simpan untuk referensi next frame
                })

            else:
                # Buat track baru
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': 0.0,
                    'smoothed_bbox': current_bbox # Init pertama
                }

            # Simpan history mentah (untuk debug/cadangan)
            face_data = {'bbox': current_bbox.copy()}
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)

        # Cleanup old tracks
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }

    return tracked_faces


# =====================================================================
#  OCCLUSION & SIMILAR FACE
# =====================================================================

def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    """Deteksi simple: apakah tertutup? (Returns True/False)"""
    base_flag = getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD
    if frame is None:
        return base_flag

    occl_session = _get_occluder_session()
    if occl_session is None:
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
