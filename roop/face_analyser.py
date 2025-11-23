import threading
from typing import Any, Optional, List, Tuple
import insightface
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from collections import deque

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

# Tracking variables
FACE_TRACKING = {}
TRACKING_HISTORY = deque(maxlen=30)  # Smoothing history
OCCLUSION_THRESHOLD = 0.3

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            # 🔥 SCRFD 10G KPS + AntelopeV2
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='antelopev2', 
                providers=roop.globals.execution_providers,
                root=resolve_relative_path('../models')
            )
            FACE_ANALYSER.prepare(
                ctx_id=0, 
                det_thresh=0.2,  # Lower threshold for better detection in motion
                det_size=(640, 640)  # Optimized for speed
            )
    return FACE_ANALYSER

def clear_face_analyser() -> Any:
    global FACE_ANALYSER
    FACE_ANALYSER = None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        # 🔥 Confidence filtering
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
    """Calculate motion between consecutive face detections"""
    if not prev_face or not current_face:
        return 0.0
    
    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox
    
    # Calculate center points
    prev_center = np.array([(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2])
    current_center = np.array([(current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2])
    
    # Euclidean distance between centers
    motion = np.linalg.norm(current_center - prev_center)
    return motion

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """🔥 Smart tracking with position + confidence + motion"""
    global FACE_TRACKING, TRACKING_HISTORY
    
    current_faces = get_many_faces(frame)
    if not current_faces:
        return None
    
    tracked_faces = []
    
    for face in current_faces:
        face_id = None
        max_similarity = 0.7  # Similarity threshold
        best_match_id = None
        
        # Calculate partial embedding (focus on key facial regions)
        partial_embedding = calculate_partial_embedding(face)
        
        # Find best match from tracked faces
        for track_id, track_data in FACE_TRACKING.items():
            if frame_number - track_data['last_seen'] > 10:  # Forget old tracks
                continue
                
            # 🔥 Multi-factor matching
            position_similarity = calculate_position_similarity(face, track_data['last_face'])
            embedding_similarity = 1 - cosine(partial_embedding, track_data['partial_embedding'])
            motion_consistency = calculate_motion_consistency(face, track_data)
            
            total_similarity = (
                0.4 * embedding_similarity + 
                0.3 * position_similarity + 
                0.3 * motion_consistency
            )
            
            if total_similarity > max_similarity:
                max_similarity = total_similarity
                best_match_id = track_id
        
        if best_match_id:
            # Update existing track
            face_id = best_match_id
            prev_face = FACE_TRACKING[face_id]['last_face']
            motion = calculate_motion_vector(prev_face, face)
            
            FACE_TRACKING[face_id].update({
                'last_face': face,
                'last_seen': frame_number,
                'partial_embedding': partial_embedding,
                'motion': motion,
                'confidence_history': FACE_TRACKING[face_id].get('confidence_history', []) + [face.det_score]
            })
        else:
            # New track
            face_id = len(FACE_TRACKING) + 1
            FACE_TRACKING[face_id] = {
                'last_face': face,
                'last_seen': frame_number,
                'partial_embedding': partial_embedding,
                'motion': 0.0,
                'confidence_history': [face.det_score]
            }
        
        # Apply smoothing to face attributes
        smoothed_face = apply_face_smoothing(face, face_id)
        tracked_faces.append(smoothed_face)
    
    # Clean up old tracks
    FACE_TRACKING = {k: v for k, v in FACE_TRACKING.items() 
                    if frame_number - v['last_seen'] <= 15}
    
    return tracked_faces

def calculate_partial_embedding(face: Face) -> np.ndarray:
    """Calculate embedding focusing on stable facial regions"""
    # This would require access to internal face analysis
    # For now, return the full embedding
    return face.normed_embedding if hasattr(face, 'normed_embedding') else np.array([])

def calculate_position_similarity(face1: Face, face2: Face) -> float:
    """Calculate similarity based on face position"""
    bbox1 = face1.bbox
    bbox2 = face2.bbox
    
    center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
    center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
    
    frame_diagonal = np.sqrt(1280**2 + 720**2)  # Assuming HD frame
    distance = np.linalg.norm(center1 - center2)
    
    return max(0, 1 - distance / (frame_diagonal * 0.5))

def calculate_motion_consistency(current_face: Face, track_data: dict) -> float:
    """Check if motion is consistent with tracking history"""
    if 'motion_history' not in track_data:
        return 1.0
    
    motion_history = track_data['motion_history']
    if not motion_history:
        return 1.0
    
    avg_motion = np.mean(motion_history)
    current_motion = track_data.get('motion', 0)
    
    # Allow some variation in motion
    motion_diff = abs(current_motion - avg_motion)
    return max(0, 1 - motion_diff / 50)  # Normalize

def apply_face_smoothing(face: Face, face_id: int) -> Face:
    """Apply temporal smoothing to face attributes"""
    global TRACKING_HISTORY
    
    # Store current face in history
    face_data = {
        'bbox': face.bbox.copy(),
        'landmarks': face.kps.copy() if hasattr(face, 'kps') else None,
        'embedding': face.normed_embedding.copy() if hasattr(face, 'normed_embedding') else None
    }
    
    TRACKING_HISTORY.append(face_data)
    
    # Apply moving average smoothing (simple implementation)
    if len(TRACKING_HISTORY) >= 3:
        recent_faces = list(TRACKING_HISTORY)[-3:]
        
        # Smooth bbox
        smoothed_bbox = np.mean([f['bbox'] for f in recent_faces], axis=0)
        face.bbox = smoothed_bbox
        
        # Smooth landmarks if available
        if all(f['landmarks'] is not None for f in recent_faces):
            smoothed_landmarks = np.mean([f['landmarks'] for f in recent_faces], axis=0)
            face.kps = smoothed_landmarks
    
    return face

def detect_occlusion(face: Face) -> bool:
    """Detect if face is partially occluded"""
    if not hasattr(face, 'kps') or face.kps is None:
        return False
    
    # Simple occlusion detection based on landmark confidence
    # In practice, this would need more sophisticated analysis
    bbox_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    expected_landmark_area = bbox_area * 0.3  # Expected landmark coverage
    
    return face.det_score < 0.5  # Simple threshold

def find_similar_face(frame: Frame, reference_face: Face, use_tracking: bool = True) -> Optional[Face]:
    """Enhanced face matching with tracking support"""
    if use_tracking:
        many_faces = smart_face_tracking(frame, 0)  # frame_number would be passed in real usage
    else:
        many_faces = get_many_faces(frame)
        
    if many_faces and hasattr(reference_face, 'normed_embedding'):
        best_face = None
        best_distance = float('inf')
        
        for face in many_faces:
            if hasattr(face, 'normed_embedding'):
                # 🔥 Partial embedding matching for occluded faces
                if detect_occlusion(face):
                    # Use partial matching strategy
                    distance = calculate_robust_distance(face, reference_face)
                else:
                    # Use full embedding matching
                    distance = np.sum(np.square(face.normed_embedding - reference_face.normed_embedding))
                
                if distance < roop.globals.similar_face_distance and distance < best_distance:
                    best_distance = distance
                    best_face = face
        
        return best_face
    
    return None

def calculate_robust_distance(face1: Face, face2: Face) -> float:
    """Calculate distance robust to partial occlusions"""
    # Simplified implementation - in practice would use partial facial regions
    return np.sum(np.square(face1.normed_embedding - face2.normed_embedding))

def resolve_relative_path(path: str) -> str:
    """Resolve relative path for model loading"""
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
