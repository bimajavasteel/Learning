from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine
import numpy as np
import cv2
import os

import insightface
import roop.globals
from roop.typing import Frame, Face

# =====================================================================
#  GLOBALS & CONFIGURATION
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

# Tracking variables untuk video processing
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# ✅ OPTIMIZED THRESHOLDS UNTUK ReaSwapper 256
MIN_DET_SCORE = 0.40        # Lebih ketat untuk kualitas tinggi
OCCLUSION_THRESHOLD = 0.35  # Lebih sensitif untuk ReaSwapper
MAX_TRACK_GAP = 8           # Lebih responsif
MAX_TRACK_AGE = 12          # Cleanup lebih cepat
MIN_EMBED_SIMILARITY = 0.75 # Similarity lebih tinggi untuk akurasi

# ✅ QUALITY PRESETS UNTUK ReaSwapper 256
QUALITY_PRESETS = {
    'high': {'det_size': (640, 640), 'min_score': 0.5},
    'balanced': {'det_size': (512, 512), 'min_score': 0.4},
    'fast': {'det_size': (384, 384), 'min_score': 0.3}
}

# =====================================================================
#  MODEL HANDLING - OPTIMIZED UNTUK ReaSwapper
# =====================================================================

def get_face_analyser() -> Any:
    """
    Inisialisasi FaceAnalysis yang dioptimalkan untuk ReaSwapper 256.
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # ✅ Gunakan quality preset berdasarkan config
            quality_preset = getattr(roop.globals, 'face_analysis_quality', 'balanced')
            preset = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS['balanced'])
            
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers,
                allowed_modules=['detection', 'recognition']  # ✅ Hanya modul diperlukan
            )
            FACE_ANALYSER.prepare(
                ctx_id=0, 
                det_size=preset['det_size']  # ✅ Size optimal
            )
            print(f"✅ [FaceAnalyser] Loaded with {quality_preset} preset (det_size: {preset['det_size']})")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    """
    Reset analyser & tracking state.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None


# =====================================================================
#  FRAME PRE-PROCESSING - OPTIMIZED UNTUK ReaSwapper
# =====================================================================

def preprocess_frame(frame: Frame) -> Frame:
    """
    Pre-processing frame untuk meningkatkan akurasi deteksi.
    Dioptimalkan untuk ReaSwapper 256.
    """
    if frame is None or frame.size == 0:
        return frame

    try:
        # ✅ Normalize ukuran frame untuk konsistensi
        h, w = frame.shape[:2]
        if h > 1080 or w > 1920:
            # Scale down frame besar untuk performa
            scale = min(1080/h, 1920/w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # ✅ Enhance contrast untuk deteksi lebih baik
        if len(frame.shape) == 3:  # Color image
            # Convert to YUV dan enhance luminance
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return frame
    except Exception as e:
        print(f"[FaceAnalyser] Pre-processing failed: {e}")
        return frame


# =====================================================================
#  BASIC FACE DETECTION - OPTIMIZED UNTUK ReaSwapper
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Deteksi banyak wajah dengan quality filtering untuk ReaSwapper 256.
    """
    if frame is None or frame.size == 0:
        return None

    try:
        # ✅ Pre-process frame terlebih dahulu
        processed_frame = preprocess_frame(frame)
        
        # ✅ Deteksi wajah
        faces = get_face_analyser().get(processed_frame)
        if not faces:
            return []

        # ✅ Filter ketat untuk ReaSwapper 256
        quality_faces = []
        for face in faces:
            score = getattr(face, "det_score", 0.0)
            # Filter berdasarkan score dan size wajah
            bbox = face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            min_face_size = min(face_width, face_height)
            
            if score >= MIN_DET_SCORE and min_face_size >= 40:  # Minimal 40px
                quality_faces.append(face)

        return quality_faces
        
    except ValueError as e:
        print(f"[FaceAnalyser] Detection error: {e}")
        return None
    except Exception as e:
        print(f"[FaceAnalyser] Unexpected error: {e}")
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    Ambil 1 wajah terbaik dengan prioritas kualitas untuk ReaSwapper 256.
    """
    many_faces = get_many_faces(frame)
    if not many_faces:
        return None

    try:
        # ✅ Prioritaskan wajah dengan score tertinggi
        sorted_faces = sorted(many_faces, 
                            key=lambda x: getattr(x, "det_score", 0), 
                            reverse=True)
        
        return sorted_faces[position]
    except IndexError:
        return sorted_faces[-1] if sorted_faces else None


# =====================================================================
#  SMART TRACKING - ENHANCED UNTUK ReaSwapper
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    """
    Hitung pergerakan wajah untuk tracking stability.
    """
    if prev_face is None or current_face is None:
        return 0.0

    try:
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
    except Exception:
        return 0.0


def _compute_embedding_similarity(current_embedding: np.ndarray, 
                                  track_embedding: np.ndarray) -> float:
    """
    Hitung similarity embedding dengan cosine distance.
    """
    try:
        similarity = 1.0 - float(cosine(current_embedding, track_embedding))
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    except Exception:
        return 0.0


def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """
    Smart tracking yang dioptimalkan untuk ReaSwapper 256.
    """
    global FACE_TRACKING, TRACKING_HISTORY

    current_faces = get_many_faces(frame)
    if not current_faces:
        return None

    tracked_faces: List[Face] = []

    with TRACK_LOCK:
        # ✅ Process setiap wajah yang terdeteksi
        for face in current_faces:
            face_id = None
            max_similarity = MIN_EMBED_SIMILARITY
            best_match_id = None

            current_embedding = getattr(face, "normed_embedding", None)
            if current_embedding is None or len(current_embedding) == 0:
                continue  # Skip wajah tanpa embedding

            # ✅ Cari track yang cocok dari existing tracks
            for track_id, track_data in list(FACE_TRACKING.items()):
                # Skip track yang terlalu lama tidak terlihat
                if frame_number - track_data.get('last_seen', -9999) > MAX_TRACK_GAP:
                    continue

                last_face = track_data.get('last_face', None)
                if last_face is None:
                    continue

                track_embedding = getattr(last_face, "normed_embedding", None)
                if track_embedding is None:
                    continue

                # ✅ Hitung similarity
                embedding_similarity = _compute_embedding_similarity(
                    current_embedding, track_embedding
                )

                # ✅ Tambah penalty untuk motion yang terlalu cepat
                motion_penalty = 0.0
                if 'motion' in track_data and track_data['motion'] > 50:  # Motion threshold
                    motion_penalty = 0.2  # Reduce similarity untuk motion cepat

                adjusted_similarity = embedding_similarity - motion_penalty

                if adjusted_similarity > max_similarity:
                    max_similarity = adjusted_similarity
                    best_match_id = track_id

            if best_match_id is not None:
                # ✅ Update existing track
                face_id = best_match_id
                prev_face = FACE_TRACKING[face_id]['last_face']
                motion = calculate_motion_vector(prev_face, face)

                FACE_TRACKING[face_id].update({
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': motion,
                    'similarity': max_similarity
                })
            else:
                # ✅ Buat new track
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': 0.0,
                    'similarity': 1.0,
                    'created_at': frame_number
                }

            # ✅ Smoothing bbox dengan weighted average
            if len(TRACKING_HISTORY) >= 2:
                recent_faces = list(TRACKING_HISTORY)[-2:]
                valid_faces = [f for f in recent_faces if 'bbox' in f]
                if valid_faces:
                    weights = [0.7, 0.3]  # Weight untuk frame terbaru lebih tinggi
                    if len(valid_faces) == 1:
                        weights = [1.0]
                    
                    weighted_bbox = np.average(
                        [f['bbox'] for f in valid_faces[:len(weights)]], 
                        axis=0, 
                        weights=weights[:len(valid_faces)]
                    )
                    face.bbox = weighted_bbox

            # ✅ Simpan ke history
            face_data = {
                'bbox': np.array(face.bbox, dtype=np.float32).copy(),
                'score': getattr(face, "det_score", 0.0),
                'frame_num': frame_number
            }
            TRACKING_HISTORY.append(face_data)
            
            # ✅ Attach tracking info ke face object
            setattr(face, 'track_id', face_id)
            setattr(face, 'track_similarity', max_similarity)
            
            tracked_faces.append(face)

        # ✅ Cleanup old tracks
        current_track_ids = list(FACE_TRACKING.keys())
        for track_id in current_track_ids:
            if frame_number - FACE_TRACKING[track_id].get('last_seen', -9999) > MAX_TRACK_AGE:
                del FACE_TRACKING[track_id]

    return tracked_faces if tracked_faces else None


# =====================================================================
#  ADVANCED FEATURES - OPTIMIZED UNTUK ReaSwapper
# =====================================================================

def detect_occlusion(face: Face) -> bool:
    """
    Deteksi occlusion yang dioptimalkan untuk ReaSwapper 256.
    """
    score = getattr(face, "det_score", 1.0)
    
    # ✅ Additional checks untuk ReaSwapper
    bbox = face.bbox
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    aspect_ratio = face_width / face_height if face_height > 0 else 1.0
    
    # ✅ Check untuk aspect ratio tidak normal (indicative of partial face)
    if aspect_ratio < 0.6 or aspect_ratio > 1.8:
        return True
    
    # ✅ Check size terlalu kecil
    if min(face_width, face_height) < 30:
        return True
        
    return score < OCCLUSION_THRESHOLD


def find_similar_face(frame: Frame, reference_face: Face, 
                     use_tracking: bool = True) -> Optional[Face]:
    """
    Cari wajah paling mirip dengan optimasi untuk ReaSwapper 256.
    """
    if reference_face is None:
        return None

    # ✅ Gunakan tracking jika available
    if use_tracking and hasattr(roop.globals, 'process_video') and roop.globals.process_video:
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

    # ✅ Threshold yang dioptimalkan untuk ReaSwapper
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 0.6)

    for face in many_faces:
        if not hasattr(face, "normed_embedding"):
            continue

        # ✅ Additional quality check
        face_score = getattr(face, "det_score", 0.0)
        if face_score < MIN_DET_SCORE:
            continue

        try:
            # ✅ Euclidean distance untuk embedding
            distance = np.sum(np.square(face.normed_embedding - ref_emb))
            
            # ✅ Adjust threshold berdasarkan quality score
            quality_adjustment = (1.0 - face_score) * 0.2  # Poor quality faces get penalty
            adjusted_threshold = similar_threshold + quality_adjustment
            
            if distance < adjusted_threshold and distance < best_distance:
                best_distance = distance
                best_face = face
                
        except Exception:
            continue

    return best_face


def get_face_quality(face: Face) -> float:
    """
    Return quality score (0-1) untuk wajah.
    Useful untuk prioritization dalam ReaSwapper 256.
    """
    if face is None:
        return 0.0
    
    base_score = getattr(face, "det_score", 0.0)
    
    # ✅ Additional quality metrics
    bbox = face.bbox
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    
    # Size quality (normalize to 0-1)
    size_quality = min(1.0, min(face_width, face_height) / 200.0)
    
    # Aspect ratio quality (penalty for extreme ratios)
    aspect_ratio = face_width / face_height if face_height > 0 else 1.0
    aspect_quality = 1.0 - abs(1.0 - aspect_ratio) * 0.5  # Penalty up to 0.5
    
    # Final quality score
    quality = base_score * 0.6 + size_quality * 0.2 + aspect_quality * 0.2
    return max(0.0, min(1.0, quality))


# =====================================================================
#  UTILITY FUNCTIONS
# =====================================================================

def get_face_landmarks(face: Face) -> Optional[np.ndarray]:
    """
    Extract landmarks dari face object jika available.
    """
    return getattr(face, "kps", None)


def get_face_pose(face: Face) -> dict:
    """
    Estimate face pose dari landmarks.
    """
    landmarks = get_face_landmarks(face)
    if landmarks is None or len(landmarks) < 5:
        return {'pitch': 0, 'yaw': 0, 'roll': 0}
    
    # Simple pose estimation (placeholder implementation)
    # Untuk implementasi lengkap, butuh complex 3D pose estimation
    return {
        'pitch': 0.0,  # Placeholder
        'yaw': 0.0,    # Placeholder  
        'roll': 0.0    # Placeholder
    }
