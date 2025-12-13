from typing import Any, Optional, List, Dict
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

# Optional: Untuk deteksi oklusi yang lebih canggih
import onnxruntime as ort

# =====================================================================
#  GLOBALS & CONSTANTS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()         # Lock untuk inisialisasi model
TRACK_LOCK = threading.Lock()          # Lock untuk akses dictionary tracking

# Tracking State
# Key: ID (int), Value: Dict berisi 'kalman', 'last_face', 'last_seen'
FACE_TRACKING: Dict[int, Dict[str, Any]] = {}

# Parameter Tuning
MIN_DET_SCORE = 0.30         # Skor minimal agar wajah dianggap valid
MIN_EMBED_SIMILARITY = 0.70  # Batas kemiripan wajah (Cosine Sim) untuk tracking ID yang sama
MAX_TRACK_GAP = 10           # Maksimal frame wajah hilang sebelum tracking di-pause
MAX_TRACK_AGE = 15           # Maksimal frame wajah hilang sebelum track dihapus permanen
OCCLUSION_THRESHOLD = 0.40   # Ambang batas deteksi oklusi (berdasarkan det_score atau model)

# Global Occluder Session
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None


# =====================================================================
#  KALMAN FILTER CORE (8-STATE ZOOM SUPPORT)
# =====================================================================

def _init_kalman(bbox: np.ndarray) -> cv2.KalmanFilter:
    """
    Inisialisasi Kalman Filter dengan 8 state variabel.
    State: [x, y, w, h, vx, vy, vw, vh]
    - x, y : Posisi pojok kiri atas
    - w, h : Lebar dan Tinggi
    - v... : Kecepatan perubahan (velocity)
    
    Penambahan vw dan vh penting agar bbox responsif saat kamera
    melakukan zoom in/out atau subjek maju-mundur.
    """
    # 8 state variables (pos + vel), 4 measurement variables (pos)
    kf = cv2.KalmanFilter(8, 4)
    
    # Transition Matrix (F)
    # Memetakan state sebelumnya ke state sekarang berdasarkan fisika (pos' = pos + vel)
    kf.transitionMatrix = np.array([
        [1,0,0,0, 1,0,0,0], # x' = x + vx
        [0,1,0,0, 0,1,0,0], # y' = y + vy
        [0,0,1,0, 0,0,1,0], # w' = w + vw
        [0,0,0,1, 0,0,0,1], # h' = h + vh
        [0,0,0,0, 1,0,0,0], # vx' = vx (asumsi kecepatan konstan)
        [0,0,0,0, 0,1,0,0], # vy' = vy
        [0,0,0,0, 0,0,1,0], # vw' = vw
        [0,0,0,0, 0,0,0,1], # vh' = vh
    ], np.float32)

    # Measurement Matrix (H)
    # Kita hanya bisa mengukur [x, y, w, h] dari deteksi wajah
    kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
    
    # Process Noise Covariance (Q)
    # Seberapa besar kita mentolerir penyimpangan model fisika.
    # Nilai 1e-2 = cukup halus (smooth). Jika ingin lebih responsif, naikkan ke 1e-1.
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    
    # Measurement Noise Covariance (R)
    # Seberapa besar noise pada deteksi InsightFace.
    # Nilai 1e-1 = kita cukup percaya pada deteksi AI.
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

    # Inisialisasi state awal
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # State awal: posisi sesuai deteksi, kecepatan 0
    kf.statePost = np.array([[x1], [y1], [w], [h], [0], [0], [0], [0]], np.float32)
    return kf


def _kalman_bbox(kf: cv2.KalmanFilter) -> np.ndarray:
    """
    Mengambil prediksi bbox dari state Kalman Filter.
    """
    s = kf.statePost.flatten()
    
    x, y = s[0], s[1]
    # Pastikan lebar dan tinggi tidak negatif atau nol (bisa terjadi saat overshoot)
    w = max(1.0, s[2])
    h = max(1.0, s[3])
    
    return np.array([x, y, x + w, y + h], dtype=np.float32)


# =====================================================================
#  MODEL HANDLING (INSIGHTFACE & OCCLUDER)
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
    global FACE_ANALYSER, FACE_TRACKING

    with TRACK_LOCK:
        FACE_TRACKING.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None


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
    except Exception:
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
        # Resize standard input size (biasanya 224x224 untuk klasifikasi/seg)
        inp = cv2.resize(crop, (224, 224)).astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]

        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0]

        # Logika sederhana: rata-rata heatmap atau nilai klasifikasi
        if pred.ndim == 4:
            mask = pred[0, 0]
            mask = cv2.resize(mask, (w, h))
            return float(np.mean(mask > 0.5))
        else:
            return float(pred[0])
    except Exception:
        return 0.0


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    """
    Mengembalikan True jika wajah tertutup (occluded).
    Jika model ONNX ada, gunakan model. Jika tidak, pakai threshold det_score.
    """
    # 1. Cek Basic Score
    score = getattr(face, "det_score", 1.0)
    if score < OCCLUSION_THRESHOLD:
        return True

    # 2. Cek Model ONNX (Jika frame disediakan)
    if frame is not None:
        sess = _get_occluder_session()
        if sess is not None:
            try:
                x1, y1, x2, y2 = map(int, face.bbox)
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                crop = frame[y1:y2, x1:x2]
                occl_val = _run_occluder_onnx(crop)
                
                # Threshold sensitivitas occluder (bisa diatur)
                return occl_val > 0.25 
            except:
                pass

    return False


def _embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Menghitung Cosine Similarity (0.0 - 1.0).
    """
    try:
        return 1.0 - float(cosine(emb1, emb2))
    except:
        return 0.0


def get_face_pose(face: Face) -> tuple[float, float, float]:
    pose = getattr(face, "pose", None)
    if pose is not None:
        return float(pose[0]), float(pose[1]), float(pose[2])
    return 0.0, 0.0, 0.0


# =====================================================================
#  FACE GETTERS
# =====================================================================

def get_many_faces(frame: Frame) -> List[Face]:
    try:
        faces = get_face_analyser().get(frame)
        # Filter awal hanya untuk membuang noise deteksi yang sangat rendah
        valid_faces = [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]
        return valid_faces
    except Exception:
        return []


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    faces = get_many_faces(frame)
    if faces:
        try:
            return faces[position]
        except IndexError:
            return faces[-1]
    return None


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    """
    Mencari wajah yang paling mirip dengan referensi menggunakan Smart Tracking.
    """
    if reference_face is None:
        return None
        
    # Gunakan smart tracking agar mendapatkan posisi yang stabil
    faces = smart_face_tracking(frame, frame_number=0) 
    if not faces:
        return None

    ref_emb = getattr(reference_face, "normed_embedding", None)
    if ref_emb is None:
        return None

    best_face = None
    best_dist = float('inf')
    
    # Disini kita pakai Euclidean distance (sesuai standar Face Swap umumnya)
    # atau bisa pakai Cosine. Roop standar pakai Euclidean sum square.
    threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for face in faces:
        curr_emb = getattr(face, "normed_embedding", None)
        if curr_emb is None: 
            continue
            
        dist = np.sum(np.square(curr_emb - ref_emb))
        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_face = face

    return best_face


# =====================================================================
#  SMART FACE TRACKING + KALMAN (MAIN LOGIC)
# =====================================================================

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """
    Fungsi utama tracking.
    1. Mengambil deteksi mentah.
    2. Mencocokkan dengan ID sebelumnya (Cosine Sim).
    3. Memperbarui posisi menggunakan Kalman Filter (Predict -> Correct).
    4. Membersihkan track yang sudah kedaluwarsa.
    """
    global FACE_TRACKING

    # 1. Ambil deteksi mentah
    raw_faces = get_many_faces(frame)
    
    tracked_results: List[Face] = []

    with TRACK_LOCK:
        for face in raw_faces:
            best_id = None
            best_sim = MIN_EMBED_SIMILARITY
            
            curr_emb = getattr(face, "normed_embedding", None)

            # 2. Matching dengan track yang ada
            if curr_emb is not None:
                for tid, data in FACE_TRACKING.items():
                    # Skip jika track sudah hilang terlalu lama (gap check)
                    if frame_number - data["last_seen"] > MAX_TRACK_GAP:
                        continue
                    
                    last_face = data["last_face"]
                    last_emb = getattr(last_face, "normed_embedding", None)
                    
                    if last_emb is not None:
                        sim = _embedding_similarity(curr_emb, last_emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_id = tid

            # 3. Inisialisasi atau Update Track
            if best_id is None:
                # NEW TRACK
                new_id = len(FACE_TRACKING) + 1
                kf = _init_kalman(np.array(face.bbox, dtype=np.float32))
                
                FACE_TRACKING[new_id] = {
                    "last_face": face,
                    "last_seen": frame_number,
                    "kalman": kf
                }
                current_kf = kf
                track_data = FACE_TRACKING[new_id]
            else:
                # EXISTING TRACK
                track_data = FACE_TRACKING[best_id]
                current_kf = track_data["kalman"]

            # 4. Kalman Predict (Langkah Wajib: Prediksi posisi berdasarkan kecepatan sebelumnya)
            current_kf.predict()

            # 5. Kalman Correct (Langkah Kondisional: Koreksi jika data valid)
            is_occluded = detect_occlusion(face, frame)
            
            if not is_occluded:
                # Jika wajah jelas, kita percaya pada deteksi saat ini -> Update Kalman
                x1, y1, x2, y2 = face.bbox
                w = x2 - x1
                h = y2 - y1
                
                # Measurement z [x, y, w, h]
                z = np.array([[x1], [y1], [w], [h]], np.float32)
                current_kf.correct(z)
                
                # Update data referensi wajah
                track_data["last_face"] = face
                track_data["last_seen"] = frame_number
            else:
                # Jika occluded, kita JANGAN panggil correct().
                # Kita biarkan Kalman menggunakan hasil predict() sebelumnya (inersia).
                # Ini mencegah bbox mengecil/hilang saat wajah tertutup tangan.
                pass

            # 6. Output Stabilization
            # Timpa bbox asli dengan hasil perhitungan Kalman yang mulus
            face.bbox = _kalman_bbox(current_kf)
            tracked_results.append(face)

        # 7. Cleanup Old Tracks (Safe Method)
        # Hapus track yang sudah tidak terlihat lebih lama dari MAX_TRACK_AGE
        # Gunakan dict comprehension agar tidak mereset global dict
        FACE_TRACKING = {
            tid: data 
            for tid, data in FACE_TRACKING.items() 
            if (frame_number - data["last_seen"]) <= MAX_TRACK_AGE
        }

    # Jika tidak ada wajah terdeteksi sama sekali, return None
    if not tracked_results:
        return None

    return tracked_results
