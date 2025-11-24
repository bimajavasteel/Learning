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

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Settings
MIN_DET_SCORE = 0.40        # Naikkan sedikit agar deteksi lebih stabil
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.75 # Diperketat agar tidak salah swap orang
EMA_ALPHA = 0.3             # 0.3 = Lebih smooth (mengurangi getar), 0.7 = Lebih responsif

# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # buffalo_l memiliki deteksi 2d106 (106 titik wajah) yang presisi
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
#  GEOMETRIC MASKING (PENGGANTI OCCLUDER ONNX)
# =====================================================================

def get_geometric_mask(face: Face, frame: Frame) -> Optional[np.ndarray]:
    """
    Membuat mask presisi berdasarkan 106 landmark wajah (Convex Hull).
    Tanpa AI berat, murni Geometri. Sangat stabil (anti-flicker).
    
    Output: Mask Grayscale (0-255) seukuran bbox wajah.
            Putih (255) = Area Wajah Aman.
            Hitam (0) = Background/Rambut Luar.
    """
    try:
        # Ambil landmark presisi 106 titik (jika ada), kalau tidak fallback ke 5 kps
        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is None:
            landmarks = face.kps

        # Konversi ke integer
        landmarks = np.round(landmarks).astype(np.int32)

        # Ambil koordinat bbox
        x1, y1, x2, y2 = map(int, face.bbox)
        h_frame, w_frame = frame.shape[:2]
        
        # Clamp bbox
        x1 = max(0, min(x1, w_frame)); x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame)); y2 = max(0, min(y2, h_frame))
        
        w_crop = x2 - x1
        h_crop = y2 - y1
        
        if w_crop <= 0 or h_crop <= 0:
            return None

        # Buat kanvas mask hitam seukuran crop
        mask = np.zeros((h_crop, w_crop), dtype=np.uint8)

        # Geser landmark agar relatif terhadap crop (bukan frame global)
        local_landmarks = landmarks.copy()
        local_landmarks[:, 0] -= x1
        local_landmarks[:, 1] -= y1

        # TEKNIK MATEMATIS: Convex Hull
        # Menghubungkan titik-titik terluar wajah menjadi poligon tertutup
        hull = cv2.convexHull(local_landmarks)
        
        # Gambar poligon wajah berwarna putih solid
        cv2.fillConvexPoly(mask, hull, 255)

        # Erosi (Kikis pinggiran) sedikit agar tidak ada sisa background yang masuk
        kernel_erode = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=2)

        # Gaussian Blur untuk Soft Edge (agar tidak terlihat tempelan stiker)
        # Kernel ganjil, sigma menyesuaikan ukuran wajah
        blur_radius = max(1, int(w_crop * 0.05)) | 1 
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

        # Kembalikan mask float 0.0 - 1.0
        return mask.astype(np.float32) / 255.0

    except Exception as e:
        print(f"GeoMask Error: {e}")
        return None

# =====================================================================
#  BASIC ACCESSORS
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
    return float(pose[0]), float(pose[1]), float(pose[2])

# =====================================================================
#  SMART TRACKING (EMA STABILIZER)
# =====================================================================

def _compute_similarity(emb1, emb2):
    try:
        return 1.0 - float(cosine(emb1, emb2))
    except:
        return 0.0

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    global FACE_TRACKING, TRACKING_HISTORY
    
    current_faces = get_many_faces(frame)
    if not current_faces:
        return None

    tracked_faces: List[Face] = []

    with TRACK_LOCK:
        for face in current_faces:
            best_match_id = None
            max_sim = MIN_EMBED_SIMILARITY
            curr_emb = getattr(face, "normed_embedding", None)

            # Cari match di history
            if curr_emb is not None:
                for tid, tdata in FACE_TRACKING.items():
                    if frame_number - tdata.get('last_seen', -999) > MAX_TRACK_GAP:
                        continue
                    
                    last_emb = getattr(tdata['last_face'], "normed_embedding", None)
                    if last_emb is None: continue
                    
                    sim = _compute_similarity(curr_emb, last_emb)
                    if sim > max_sim:
                        max_sim = sim
                        best_match_id = tid

            # EMA Smoothing Logic
            curr_bbox = np.array(face.bbox, dtype=np.float32)
            
            if best_match_id is not None:
                # Update existing track
                face_id = best_match_id
                prev_bbox = FACE_TRACKING[face_id].get('smooth_bbox', curr_bbox)
                
                # Rumus EMA: Stabilizer Matematis
                # smooth = (alpha * current) + ((1-alpha) * previous)
                smooth_bbox = (EMA_ALPHA * curr_bbox) + ((1.0 - EMA_ALPHA) * prev_bbox)
                
                face.bbox = smooth_bbox
                FACE_TRACKING[face_id].update({
                    'last_face': face,
                    'last_seen': frame_number,
                    'smooth_bbox': smooth_bbox
                })
            else:
                # New track
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'smooth_bbox': curr_bbox
                }
            
            tracked_faces.append(face)

        # Cleanup
        del_keys = [k for k,v in FACE_TRACKING.items() if frame_number - v['last_seen'] > MAX_TRACK_AGE]
        for k in del_keys: del FACE_TRACKING[k]

    return tracked_faces

# Placeholder fungsi lama agar code lain tidak error (return False)
def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    return False
