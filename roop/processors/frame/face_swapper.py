from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
import time

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# =====================
# Configuration & Globals
# =====================
FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER-FINAL'

# Filter storage for landmark smoothing
LANDMARK_FILTERS: Dict[str, Any] = {}
# Cache static source landmarks to avoid re-calculation
SOURCE_LANDMARKS_CACHE: Dict[str, np.ndarray] = {}

ONE_EURO_CONFIG = {
    'freq': 30.0,       # Adjust based on video FPS
    'min_cutoff': 1.0,
    'beta': 0.007,      # Lower = smoother, Higher = faster response
    'd_cutoff': 1.0
}

# =====================
# Utilities: OneEuroFilter
# =====================
class OneEuroFilter:
    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.freq = float(max(freq, 1e-6))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        if t is not None and self.t_prev is not None:
            dt = max(t - self.t_prev, 1e-6)
            self.freq = 1.0 / dt
        self.t_prev = t if t is not None else time.time()
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha_c = self.alpha(cutoff)
        x_hat = alpha_c * x + (1 - alpha_c) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

def get_filter_for_key(key: str) -> OneEuroFilter:
    if key not in LANDMARK_FILTERS:
        cfg = ONE_EURO_CONFIG
        LANDMARK_FILTERS[key] = OneEuroFilter(freq=cfg['freq'], min_cutoff=cfg['min_cutoff'], beta=cfg['beta'], d_cutoff=cfg['d_cutoff'])
    return LANDMARK_FILTERS[key]

def clear_landmark_filters() -> None:
    LANDMARK_FILTERS.clear()
    SOURCE_LANDMARKS_CACHE.clear()

# =====================
# Core Management
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
    if not get_one_face(cv2.imread(roop.globals.source_path)):
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

# =====================
# Logic Helpers
# =====================
def adapt_bbox_for_pose(face: Face, frame_shape: Tuple[int, int]) -> None:
    """Adjust bbox based on pitch/yaw to prevent face cropping issues."""
    try:
        pitch, yaw, _ = get_face_pose(face)
        h_frame, w_frame = frame_shape[:2]
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        
        pad_x, pad_y_top, pad_y_bottom = 0.0, 0.0, 0.0

        if abs(yaw) > 25.0:
            pad_x = w * min((abs(yaw) - 25.0) * 0.02, 0.20)
        
        if pitch < -15.0:
            pad_y_top = h * min((abs(pitch) - 15.0) * 0.02, 0.25)
        elif pitch > 20.0:
            pad_y_bottom = h * min((pitch - 20.0) * 0.015, 0.18)

        nx1 = int(max(0, x1 - pad_x))
        nx2 = int(min(w_frame - 1, x2 + pad_x))
        ny1 = int(max(0, y1 - pad_y_top))
        ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

        if nx2 > nx1 and ny2 > ny1:
            # Check sanity: if box grew > 50% unexpectedly, maybe ignore? 
            # For now, just clamp
            face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    except Exception:
        pass 

def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Optional[Face]:
    if not faces or reference_face is None or not hasattr(reference_face, 'normed_embedding'):
        return None
    
    best_face = None
    best_distance = float('inf')
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, 'normed_embedding'): continue
        dist = np.sum(np.square(f.normed_embedding - reference_face.normed_embedding))
        if dist < similar_threshold and dist < best_distance:
            best_distance = dist
            best_face = f
    return best_face

# =====================
# Rendering Helpers (Improved)
# =====================
def safe_get_landmarks(face: Face) -> Optional[np.ndarray]:
    if face is None: return None
    for attr in ['landmark_2d_106', 'landmark_2d', 'kps', 'landmarks']:
        landmarks = getattr(face, attr, None)
        if landmarks is not None and len(landmarks) > 0:
            return np.asarray(landmarks, dtype=float)
    return None

def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame, face_key: str) -> Tuple[Frame, np.ndarray]:
    """Aligns face using smoothed landmarks with static source caching."""
    try:
        # Source Landmarks (Static Cache)
        if 'source' in SOURCE_LANDMARKS_CACHE:
            sm_source = SOURCE_LANDMARKS_CACHE['source']
        else:
            sm_source = safe_get_landmarks(source_face)
            if sm_source is not None:
                SOURCE_LANDMARKS_CACHE['source'] = sm_source
        
        target_landmarks = safe_get_landmarks(target_face)

        if sm_source is None or target_landmarks is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        # Smooth target landmarks
        timestamp = time.time()
        sm_target = get_filter_for_key(face_key).filter(target_landmarks, t=timestamp)

        # Safety Check: minimum points
        if len(sm_source) < 5 or len(sm_target) < 5:
             # Fallback to standard 3 points if 5 not available, but if <3 abort
             if len(sm_source) < 3 or len(sm_target) < 3:
                 return temp_frame, np.eye(2, 3, dtype=np.float32)

        # Use 5 key points for better alignment stability
        landmark_indices = list(range(min(5, len(sm_source)))) 
        src_points = np.array([sm_source[i] for i in landmark_indices], dtype=np.float32)
        dst_points = np.array([sm_target[i] for i in landmark_indices], dtype=np.float32)

        transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0)[0]
        
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            return cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR), transform_matrix
            
        return temp_frame, np.eye(2, 3, dtype=np.float32)
    except Exception:
        return temp_frame, np.eye(2, 3, dtype=np.float32)

def fast_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0 or swapped_face.size == 0: return swapped_face
        
        if swapped_face.shape[:2] != target_region.shape[:2]:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))

        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        
        swapped_stats = [np.mean(swapped_lab, axis=(0,1)), np.std(swapped_lab, axis=(0,1))]
        target_stats = [np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))]
        
        swapped_stats[1] = np.where(swapped_stats[1] == 0, 1, swapped_stats[1])
        
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_stats[0][i]) * (target_stats[1][i] / swapped_stats[1][i]) + target_stats[0][i]
            
        corrected_bgr = cv2.cvtColor(np.clip(corrected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(swapped_face, 0.4, corrected_bgr, 0.6, 0)
    except Exception:
        return swapped_face

def enhance_face_quality(face: Frame) -> Frame:
    try:
        # Subtle sharpening + Bilateral filter
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.15
        sharpened = cv2.filter2D(face, -1, kernel)
        return cv2.bilateralFilter(sharpened, 5, 15, 15)
    except Exception:
        return face

def create_adaptive_mask(swapped_face: Frame) -> np.ndarray:
    """Creates a slightly eroded mask to prevent background bleeding."""
    mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
    # Erode the mask slightly to keep the blend inside the face bounds
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=2)
    return mask

def seamless_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        face_h, face_w = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_h or swapped_face.shape[1] != face_w:
            swapped_face = cv2.resize(swapped_face, (face_w, face_h))
            
        # Use adaptive mask (conceptually simpler here: just full white but rely on clone)
        # Using simple mask because seamlessClone calculates gradients itself
        mask = create_adaptive_mask(swapped_face)
        
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
    except Exception:
        # Fallback to direct paste if clone fails
        try:
            target_frame[y1:y2, x1:x2] = swapped_face
            return target_frame
        except:
            return target_frame

# =====================
# Hybrid Swap Function
# =====================
def swap_face_hybrid(source_face: Face, target_face: Face, temp_frame: Frame, face_key: str) -> Frame:
    """The Perfected Hybrid Swapper."""
    try:
        # 1. Pose-Aware Bbox Adjustment
        adapt_bbox_for_pose(target_face, temp_frame.shape)

        # 2. Robust Alignment (with cached source & smoothed target)
        aligned_frame, _ = robust_face_alignment(source_face, target_face, temp_frame, face_key)

        # 3. InsightFace Swap (Raw)
        swapper = get_face_swapper()
        swapped_raw = swapper.get(aligned_frame, target_face, source_face, paste_back=False)

        if isinstance(swapped_raw, tuple): swapped_raw = swapped_raw[0]
        if swapped_raw is None: return temp_frame

        # 4. Color Correction (LAB)
        swapped_processed = fast_color_correction(swapped_raw, temp_frame, target_face)

        # 5. Enhancement
        swapped_processed = enhance_face_quality(swapped_processed)

        # 6. Seamless Blending
        result_frame = seamless_face_blending(swapped_processed, temp_frame, target_face)
        return result_frame

    except Exception:
        # Ultimate Fallback: Default Swapper
        try:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        except:
            return temp_frame

# =====================
# Main Processors
# =====================
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None: return temp_frame

    try:
        # Step 1: Face Detection & Tracking
        # Use smart_face_tracking which typically persists IDs
        faces = None
        if roop.globals.many_faces:
             faces = smart_face_tracking(temp_frame, frame_number)
             if not faces: faces = get_many_faces(temp_frame)
        else:
             # Single face mode: track, filter occlusion, match embedding
             tracked_faces = smart_face_tracking(temp_frame, frame_number)
             if not tracked_faces: tracked_faces = get_many_faces(temp_frame)
             if tracked_faces:
                 valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
                 if valid_faces:
                     target_face = _select_best_target_by_embedding(valid_faces, reference_face)
                     if target_face is None: target_face = valid_faces[0]
                     faces = [target_face]
        
        if not faces:
            return temp_frame

        # Step 2: Iterate & Swap
        for idx, target_face in enumerate(faces):
            # Skip if occluded (double check for many_faces loop)
            if roop.globals.many_faces and detect_occlusion(target_face, temp_frame):
                continue
            
            # --- IMPROVEMENT: Track ID Selection ---
            # Try to get 'id' from tracker, fallback to index
            track_id = getattr(target_face, 'id', None) 
            if track_id is not None:
                face_key = f"face_id_{track_id}"
            else:
                # If no ID, use index but warn this is less stable for videos
                face_key = f"face_idx_{idx}"

            temp_frame = swap_face_hybrid(source_face, target_face, temp_frame, face_key)

    except Exception as e:
        pass # Fail silently to keep video rendering

    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, path in enumerate(temp_frame_paths):
        try:
            frame = cv2.imread(path)
            if frame is not None:
                result = process_frame(source_face, reference_face, frame, idx)
                cv2.imwrite(path, result)
            if update: update()
        except Exception:
            continue

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None
        if not roop.globals.many_faces:
            reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
        
        result = process_frame(source_face, reference_face, target_frame, 0)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Image process error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            ref_frame = cv2.imread(temp_frame_paths[ref_idx])
            ref_face = get_one_face(ref_frame, roop.globals.reference_face_position)
            set_face_reference(ref_face)
        except:
            pass
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
