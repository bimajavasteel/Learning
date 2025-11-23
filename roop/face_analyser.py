import threading
from typing import Any, Optional, List
import insightface
import numpy as np
import cv2
import os
from scipy.spatial.distance import cosine
from collections import deque

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

# Tracking variables
FACE_TRACKING = {}
TRACKING_HISTORY = deque(maxlen=30)

def resolve_relative_path(path: str) -> str:
    """Resolve relative path untuk model loading"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            try:
                # 🔥 COBA ANTELOPEV2 dengan path yang benar
                antelope_path = resolve_relative_path('../models/antelopev2')
                print(f"🔍 Mencari AntelopeV2 di: {antelope_path}")
                
                # Cek apakah model AntelopeV2 sudah ada
                if os.path.exists(antelope_path):
                    FACE_ANALYSER = insightface.app.FaceAnalysis(
                        name='antelopev2', 
                        providers=roop.globals.execution_providers,
                        root=resolve_relative_path('../models')
                    )
                    FACE_ANALYSER.prepare(
                        ctx_id=0, 
                        det_thresh=0.2,
                        det_size=(640, 640)
                    )
                    print("✅ Menggunakan AntelopeV2 dengan SCRFD 10G KPS")
                else:
                    raise FileNotFoundError("Model AntelopeV2 tidak ditemukan")
                    
            except Exception as e:
                print(f"❌ AntelopeV2 gagal: {e}")
                print("🔄 Fallback ke buffalo_l...")
                # Fallback ke buffalo_l - biarkan insightface handle download
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l', 
                    providers=roop.globals.execution_providers
                )
                FACE_ANALYSER.prepare(ctx_id=0)
                print("✅ Menggunakan buffalo_l (fallback)")
    return FACE_ANALYSER

def clear_face_analyser() -> Any:
    global FACE_ANALYSER
    FACE_TRACKING.clear()
    TRACKING_HISTORY.clear()
    FACE_ANALYSER = None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        faces = [face for face in faces if face.det_score > 0.3]
        return faces
    except ValueError:
        return None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    if not prev_face or not current_face:
        return 0.0
    
    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox
    
    prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
    current_center = np.array([(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2])
    
    motion = np.linalg.norm(current_center - prev_center)
    return motion

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    global FACE_TRACKING, TRACKING_HISTORY
    
    current_faces = get_many_faces(frame)
    if not current_faces:
        return None
    
    tracked_faces = []
    
    for face in current_faces:
        face_id = None
        max_similarity = 0.7
        best_match_id = None
        
        partial_embedding = face.normed_embedding if hasattr(face, 'normed_embedding') else np.array([])
        
        for track_id, track_data in FACE_TRACKING.items():
            if frame_number - track_data['last_seen'] > 10:
                continue
                
            if not hasattr(track_data['last_face'], 'normed_embedding'):
                continue
                
            # Simple similarity calculation
            try:
                embedding_similarity = 1 - cosine(partial_embedding, track_data['last_face'].normed_embedding)
            except:
                embedding_similarity = 0
                
            if embedding_similarity > max_similarity:
                max_similarity = embedding_similarity
                best_match_id = track_id
        
        if best_match_id:
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
        
        # Simple smoothing
        if len(TRACKING_HISTORY) >= 2:
            recent_faces = list(TRACKING_HISTORY)[-2:]
            if all('bbox' in f for f in recent_faces):
                smoothed_bbox = np.mean([f['bbox'] for f in recent_faces], axis=0)
                face.bbox = smoothed_bbox
        
        face_data = {'bbox': face.bbox.copy()}
        TRACKING_HISTORY.append(face_data)
        tracked_faces.append(face)
    
    FACE_TRACKING = {k: v for k, v in FACE_TRACKING.items() 
                    if frame_number - v['last_seen'] <= 15}
    
    return tracked_faces

def detect_occlusion(face: Face) -> bool:
    return face.det_score < 0.4

def find_similar_face(frame: Frame, reference_face: Face, use_tracking: bool = False) -> Optional[Face]:
    if use_tracking:
        many_faces = smart_face_tracking(frame, 0)
    else:
        many_faces = get_many_faces(frame)
        
    if many_faces and hasattr(reference_face, 'normed_embedding'):
        best_face = None
        best_distance = float('inf')
        
        for face in many_faces:
            if hasattr(face, 'normed_embedding'):
                distance = np.sum(np.square(face.normed_embedding - reference_face.normed_embedding))
                
                if distance < roop.globals.similar_face_distance and distance < best_distance:
                    best_distance = distance
                    best_face = face
        
        return best_face
    
    return None
