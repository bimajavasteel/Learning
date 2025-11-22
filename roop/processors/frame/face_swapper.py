from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
import time
from scipy.spatial.distance import cdist
from collections import deque

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# =====================
# Enhanced OneEuroFilter dengan Adaptive Parameters
# =====================
class OneEuroFilter:
    """Enhanced One Euro Filter dengan adaptive parameters berdasarkan motion speed"""
    
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.freq = float(max(freq, 1e-6))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        self.motion_history = deque(maxlen=5)  # Store recent motion magnitudes

    def alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def get_adaptive_beta(self, current_speed: float) -> float:
        """Adaptive beta berdasarkan motion speed"""
        base_beta = self.beta
        
        if current_speed > 15.0:  # Fast motion
            return min(base_beta * 2.5, 0.03)
        elif current_speed > 8.0:  # Medium motion
            return min(base_beta * 1.5, 0.02)
        elif current_speed < 2.0:  # Very slow motion
            return base_beta * 0.7
        else:  # Normal motion
            return base_beta

    def filter(self, x: np.ndarray, t: Optional[float] = None, detection_confidence: float = 1.0) -> np.ndarray:
        """Enhanced filter dengan adaptive parameters dan confidence awareness"""
        if t is not None and self.t_prev is not None:
            dt = max(t - self.t_prev, 1e-6)
            self.freq = 1.0 / dt
        self.t_prev = t if t is not None else time.time()

        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # Calculate current motion
        dx = (x - self.x_prev) * self.freq
        current_speed = np.mean(np.linalg.norm(dx, axis=1))
        self.motion_history.append(current_speed)
        
        # Adaptive parameters based on motion speed
        adaptive_beta = self.get_adaptive_beta(current_speed)
        
        # Confidence-based adjustment
        confidence_factor = min(detection_confidence / 0.7, 1.0)  # Normalize confidence
        adaptive_min_cutoff = self.min_cutoff * (2.0 - confidence_factor)  # More smoothing for low confidence

        # Filter derivative
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = adaptive_min_cutoff + adaptive_beta * np.abs(dx_hat)

        # Filter signal
        alpha_c = self.alpha(cutoff)
        x_hat = alpha_c * x + (1 - alpha_c) * self.x_prev

        # Update
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

# =====================
# Optical Flow Tracker
# =====================
class OpticalFlowTracker:
    """Optical flow-based face tracking untuk backup tracking"""
    
    def __init__(self, max_frames_to_keep: int = 10):
        self.max_frames_to_keep = max_frames_to_keep
        self.prev_frame = None
        self.prev_gray = None
        self.prev_landmarks = {}
        self.track_history = {}
        self.next_track_id = 0

    def update(self, current_frame: Frame, current_faces: List[Face], frame_count: int) -> Dict[int, Face]:
        """Update tracking dengan optical flow"""
        if current_frame is None:
            return {}
            
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        tracked_faces = {}

        if self.prev_gray is not None and self.prev_frame is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Track existing faces
            for track_id, face_data in self.track_history.items():
                if frame_count - face_data['last_seen'] > 5:  # Skip if too old
                    continue
                    
                prev_landmarks = face_data['landmarks']
                if prev_landmarks is not None:
                    # Predict new landmark positions using optical flow
                    new_landmarks = []
                    for lm in prev_landmarks:
                        x, y = int(lm[0]), int(lm[1])
                        if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
                            flow_x = flow[y, x, 0]
                            flow_y = flow[y, x, 1]
                            new_landmarks.append([lm[0] + flow_x, lm[1] + flow_y])
                        else:
                            new_landmarks.append(lm.copy())
                    
                    # Find best matching face based on landmark distance
                    best_face_idx = self._find_best_matching_face(new_landmarks, current_faces)
                    if best_face_idx is not None:
                        tracked_faces[track_id] = current_faces[best_face_idx]
                        self.track_history[track_id].update({
                            'landmarks': safe_get_landmarks(current_faces[best_face_idx]),
                            'last_seen': frame_count,
                            'bbox': current_faces[best_face_idx].bbox
                        })

        # Assign new IDs to untracked faces
        for i, face in enumerate(current_faces):
            if not any(face is tracked_face for tracked_face in tracked_faces.values()):
                track_id = self.next_track_id
                tracked_faces[track_id] = face
                self.track_history[track_id] = {
                    'landmarks': safe_get_landmarks(face),
                    'last_seen': frame_count,
                    'bbox': face.bbox
                }
                self.next_track_id += 1

        # Cleanup old tracks
        self._cleanup_old_tracks(frame_count)

        self.prev_frame = current_frame.copy()
        self.prev_gray = current_gray.copy()
        
        return tracked_faces

    def _find_best_matching_face(self, predicted_landmarks: np.ndarray, current_faces: List[Face]) -> Optional[int]:
        """Find best matching face based on landmark distance"""
        if not current_faces or predicted_landmarks is None:
            return None
            
        min_distance = float('inf')
        best_idx = None
        
        for i, face in enumerate(current_faces):
            face_landmarks = safe_get_landmarks(face)
            if face_landmarks is not None and len(face_landmarks) == len(predicted_landmarks):
                distance = np.mean(np.linalg.norm(face_landmarks - predicted_landmarks, axis=1))
                if distance < min_distance and distance < 20.0:  # Reasonable threshold
                    min_distance = distance
                    best_idx = i
                    
        return best_idx

    def _cleanup_old_tracks(self, current_frame: int):
        """Remove tracks that haven't been seen recently"""
        tracks_to_remove = []
        for track_id, data in self.track_history.items():
            if current_frame - data['last_seen'] > self.max_frames_to_keep:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.track_history[track_id]

# =====================
# Enhanced Module Variables
# =====================
FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER-ENHANCED'

# Enhanced tracking system
LANDMARK_FILTERS: Dict[str, Any] = {}
OPTICAL_FLOW_TRACKER = OpticalFlowTracker()
FRAME_COUNT = 0

# Adaptive configuration
ONE_EURO_CONFIG = {
    'freq': 30.0,
    'min_cutoff': 1.0,
    'beta': 0.007,
    'd_cutoff': 1.0
}

# =====================
# Enhanced Helper Functions
# =====================

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()
    clear_landmark_filters()
    global FRAME_COUNT, OPTICAL_FLOW_TRACKER
    FRAME_COUNT = 0
    OPTICAL_FLOW_TRACKER = OpticalFlowTracker()

def safe_get_landmarks(face: Face) -> Optional[np.ndarray]:
    if face is None:
        return None
    landmark_attrs = ['landmark_2d_106', 'landmark_2d', 'kps', 'landmarks']
    for attr in landmark_attrs:
        if hasattr(face, attr):
            landmarks = getattr(face, attr)
            if landmarks is not None and len(landmarks) > 0:
                return np.asarray(landmarks, dtype=float)
    return None

def get_face_detection_confidence(face: Face) -> float:
    """Extract face detection confidence score"""
    if hasattr(face, 'det_score'):
        return float(face.det_score)
    elif hasattr(face, 'score'):
        return float(face.score)
    return 0.8  # Default medium confidence

def get_filter_for_key(key: str, detection_confidence: float = 0.8) -> OneEuroFilter:
    """Enhanced filter getter dengan confidence awareness"""
    if key not in LANDMARK_FILTERS:
        cfg = ONE_EURO_CONFIG.copy()
        
        # Confidence-based initial configuration
        if detection_confidence < 0.6:
            cfg['min_cutoff'] *= 0.6  # More smoothing for low confidence
        elif detection_confidence > 0.9:
            cfg['min_cutoff'] *= 1.2  # Less smoothing for high confidence
            
        LANDMARK_FILTERS[key] = OneEuroFilter(
            freq=cfg['freq'], 
            min_cutoff=cfg['min_cutoff'], 
            beta=cfg['beta'], 
            d_cutoff=cfg['d_cutoff']
        )
    return LANDMARK_FILTERS[key]

def clear_landmark_filters() -> None:
    LANDMARK_FILTERS.clear()

def smooth_landmarks(landmarks: np.ndarray, key: str, timestamp: Optional[float] = None, detection_confidence: float = 0.8) -> np.ndarray:
    """Enhanced landmark smoothing dengan confidence awareness"""
    try:
        if landmarks is None:
            return landmarks
            
        f = get_filter_for_key(key, detection_confidence)
        smoothed = f.filter(landmarks, t=timestamp, detection_confidence=detection_confidence)
        return smoothed
    except Exception:
        return landmarks

# =====================
# Enhanced Face Processing dengan Better Tracking
# =====================

def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray]:
    """Enhanced face alignment dengan confidence-aware smoothing"""
    try:
        source_landmarks = safe_get_landmarks(source_face)
        target_landmarks = safe_get_landmarks(target_face)

        if source_landmarks is None or target_landmarks is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        # Get detection confidence untuk adaptive smoothing
        detection_confidence = get_face_detection_confidence(target_face)
        timestamp = time.time()
        
        # Enhanced tracking key generation
        if roop.globals.many_faces:
            track_id = getattr(target_face, 'track_id', getattr(target_face, 'face_index', None))
            key_str = f"face_{track_id}" if track_id is not None else 'many_unknown'
        else:
            key_str = 'reference'

        # Confidence-aware landmark smoothing
        sm_target = smooth_landmarks(target_landmarks, key_str, timestamp, detection_confidence)
        sm_source = smooth_landmarks(source_landmarks, 'source', timestamp, 1.0)  # Source high confidence

        if sm_source is None or sm_target is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        if len(sm_source) < 3 or len(sm_target) < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        # Choose optimal landmark points untuk alignment
        landmark_indices = list(range(min(5, len(sm_source))))
        key_points = [i for i in landmark_indices if i < len(sm_source) and i < len(sm_target)]

        if len(key_points) < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        src_points = np.array([sm_source[i] for i in key_points], dtype=np.float32)
        dst_points = np.array([sm_target[i] for i in key_points], dtype=np.float32)

        transform_matrix = cv2.estimateAffinePartial2D(
            src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0
        )[0]
        
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
            return aligned_frame, transform_matrix

        return temp_frame, np.eye(2, 3, dtype=np.float32)
    except Exception:
        return temp_frame, np.eye(2, 3, dtype=np.float32)

# =====================
# Enhanced Process Frame dengan Optical Flow Tracking
# =====================

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced process frame dengan better multi-face tracking"""
    global FRAME_COUNT
    
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                # Update optical flow tracker
                tracked_faces = OPTICAL_FLOW_TRACKER.update(temp_frame, many_faces, FRAME_COUNT)
                
                # Process tracked faces
                for track_id, target_face in tracked_faces.items():
                    # Assign track ID untuk consistent smoothing
                    setattr(target_face, 'track_id', track_id)
                    temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
                    
                FRAME_COUNT += 1
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
                
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame

# =====================
# Tetap pertahankan fungsi-fungsi berikut (tidak berubah)
# =====================

def fast_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    # [Implementasi tetap sama seperti sebelumnya]
    try:
        if target_face is None or swapped_face is None:
            return swapped_face
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        swapped_mean = np.mean(swapped_lab, axis=(0,1))
        swapped_std = np.std(swapped_lab, axis=(0,1))
        target_mean = np.mean(target_lab, axis=(0,1))
        target_std = np.std(target_lab, axis=(0,1))
        swapped_std = np.where(swapped_std == 0, 1, swapped_std)
        target_std = np.where(target_std == 0, 1, target_std)
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        result_face = cv2.addWeighted(swapped_face, 0.4, corrected_face, 0.6, 0)
        return result_face
    except Exception:
        return swapped_face

def create_simple_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    # [Implementasi tetap sama]
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        return np.clip(mask, 0, 1)
    except Exception:
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

def seamless_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    # [Implementasi tetap sama]
    try:
        if target_face is None:
            return target_frame
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        return result
    except Exception:
        return simple_face_blending(swapped_face, target_frame, target_face)

def simple_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    # [Implementasi tetap sama]
    try:
        if target_face is None:
            return target_frame
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        mask = create_simple_mask(target_face, target_frame.shape)
        mask_region = mask[y1:y2, x1:x2]
        if mask_region.shape != swapped_face.shape[:2]:
            mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
        mask_3d = np.stack([mask_region] * 3, axis=-1)
        result = target_frame.copy()
        face_region = result[y1:y2, x1:x2]
        blended_face = (swapped_face * mask_3d + face_region * (1 - mask_3d)).astype(np.uint8)
        result[y1:y2, x1:x2] = blended_face
        return result
    except Exception:
        return target_frame

def enhance_face_quality(face: Frame) -> Frame:
    # [Implementasi tetap sama]
    try:
        if face is None:
            return face
        kernel = np.array([[-1, -1, -1],[-1,  9, -1],[-1, -1, -1]]) * 0.15
        sharpened = cv2.filter2D(face, -1, kernel)
        denoised = cv2.bilateralFilter(sharpened, 5, 15, 15)
        return denoised
    except Exception:
        return face

def ensure_frame_format(frame: Any) -> Optional[Frame]:
    # [Implementasi tetap sama]
    if frame is None:
        return None
    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
        return frame
    if isinstance(frame, tuple):
        try:
            frame_array = np.array(frame)
            if frame_array.size > 0:
                return frame_array
        except:
            pass
    return None

def swap_face_optimized(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # [Implementasi tetap sama]
    try:
        aligned_frame, _ = robust_face_alignment(source_face, target_face, temp_frame)
        swapped_result = get_face_swapper().get(aligned_frame, target_face, source_face, paste_back=False)
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)
        swapped_frame = enhance_face_quality(swapped_frame)
        result_frame = seamless_face_blending(swapped_frame, temp_frame, target_face)
        return result_frame
    except Exception:
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

# =====================
# Tetap pertahankan fungsi processing lainnya
# =====================

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        for temp_frame_path in temp_frame_paths:
            try:
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is not None:
                    result = process_frame(source_face, reference_face, temp_frame)
                    cv2.imwrite(temp_frame_path, result)
                if update:
                    update()
            except Exception:
                continue
    except Exception as e:
        print(f"Process frames error: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Process video error: {e}")
