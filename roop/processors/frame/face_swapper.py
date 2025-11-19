from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
import time

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# =====================
# OneEuroFilter Utility (tetap sama)
# =====================
class OneEuroFilter:
    """Simple One Euro Filter implementation for smoothing landmark coordinates."""
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

        # derivative
        dx = (x - self.x_prev) * self.freq

        # filter derivative
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        # cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # filter signal
        alpha_c = self.alpha(cutoff)
        x_hat = alpha_c * x + (1 - alpha_c) * self.x_prev

        # update
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

# =====================
# Module variables & config
# =====================
FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER-ADAFACE'

# Landmark filters storage
LANDMARK_FILTERS: Dict[str, Any] = {}
ONE_EURO_CONFIG = {
    'freq': 30.0,
    'min_cutoff': 1.0,
    'beta': 0.007,
    'd_cutoff': 1.0
}

# NEW: AdaFace quality metrics
ADAFACE_QUALITY_THRESHOLD = 0.3  # Threshold untuk kualitas matching

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
    
    # NEW: Download AdaFace model jika digunakan
    if getattr(roop.globals, 'use_adaface', False):
        try:
            import gdown
            adaface_model_url = 'https://drive.google.com/uc?id=1nQqaPQ1CPmRX7XZaaHr1Xy4tqK7KALst'
            adaface_model_path = resolve_relative_path('../models/adaface_ir101_webface12m.pt')
            conditional_download(resolve_relative_path('../models'), [adaface_model_url])
            print("[AdaFace] Model download prepared")
        except Exception as e:
            print(f"[AdaFace] Model download setup failed: {e}")
    
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

# safe_get_landmarks (tetap sama)
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

# Landmark smoothing wrappers (tetap sama)
def get_filter_for_key(key: str) -> OneEuroFilter:
    if key not in LANDMARK_FILTERS:
        cfg = ONE_EURO_CONFIG
        LANDMARK_FILTERS[key] = OneEuroFilter(freq=cfg['freq'], min_cutoff=cfg['min_cutoff'], beta=cfg['beta'], d_cutoff=cfg['d_cutoff'])
    return LANDMARK_FILTERS[key]

def clear_landmark_filters() -> None:
    LANDMARK_FILTERS.clear()

def smooth_landmarks(landmarks: np.ndarray, key: str, timestamp: Optional[float] = None) -> np.ndarray:
    try:
        if landmarks is None:
            return landmarks
        f = get_filter_for_key(key)
        sm = f.filter(landmarks, t=timestamp)
        return sm
    except Exception:
        return landmarks

# NEW: Enhanced face matching dengan AdaFace confidence
def enhanced_face_matching(source_face: Face, target_face: Face) -> float:
    """
    Return confidence score antara 0-1 untuk kualitas matching
    """
    try:
        use_adaface = getattr(roop.globals, 'use_adaface', False)
        
        if use_adaface:
            # AdaFace matching
            if (hasattr(source_face, 'adaface_embedding') and hasattr(target_face, 'adaface_embedding') and
                source_face.adaface_embedding is not None and target_face.adaface_embedding is not None):
                
                src_emb = source_face.adaface_embedding
                tgt_emb = target_face.adaface_embedding
                
                # Cosine similarity
                similarity = np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb))
                confidence = (similarity + 1) / 2  # Convert to 0-1 range
                return float(confidence)
        
        # Fallback: InsightFace matching
        if (hasattr(source_face, 'normed_embedding') and hasattr(target_face, 'normed_embedding')):
            distance = np.sum((source_face.normed_embedding - target_face.normed_embedding) ** 2)
            max_distance = 2.0  # Empirical max distance untuk InsightFace
            confidence = 1.0 - min(distance / max_distance, 1.0)
            return confidence
        
        return 0.5  # Default medium confidence
    except Exception:
        return 0.5

# MODIFIED: robust_face_alignment dengan confidence checking
def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray, float]:
    """
    Return: (aligned_frame, transform_matrix, confidence_score)
    """
    try:
        # Check matching confidence
        confidence = enhanced_face_matching(source_face, target_face)
        
        # Skip jika confidence terlalu rendah
        if confidence < ADAFACE_QUALITY_THRESHOLD:
            return temp_frame, np.eye(2, 3, dtype=np.float32), confidence

        source_landmarks = safe_get_landmarks(source_face)
        target_landmarks = safe_get_landmarks(target_face)

        if source_landmarks is None or target_landmarks is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32), confidence

        timestamp = time.time()
        if roop.globals.many_faces:
            key = getattr(target_face, 'face_index', None)
            key_str = f"face_{key}" if key is not None else 'many_unknown'
        else:
            key_str = 'reference'

        # Smooth target landmarks
        sm_target = smooth_landmarks(target_landmarks, key_str, timestamp)
        sm_source = smooth_landmarks(source_landmarks, 'source', timestamp)

        if sm_source is None or sm_target is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32), confidence

        if len(sm_source) < 3 or len(sm_target) < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32), confidence

        landmark_indices = list(range(min(5, len(sm_source))))
        key_points = [i for i in landmark_indices if i < len(sm_source) and i < len(sm_target)]

        if len(key_points) < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32), confidence

        src_points = np.array([sm_source[i] for i in key_points], dtype=np.float32)
        dst_points = np.array([sm_target[i] for i in key_points], dtype=np.float32)

        transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0)[0]
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
            return aligned_frame, transform_matrix, confidence

        return temp_frame, np.eye(2, 3, dtype=np.float32), confidence
    except Exception:
        return temp_frame, np.eye(2, 3, dtype=np.float32), 0.3

# MODIFIED: swap_face_optimized dengan confidence-based processing
def swap_face_optimized(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        # Apply robust face alignment dengan confidence
        aligned_frame, _, confidence = robust_face_alignment(source_face, target_face, temp_frame)
        
        # Skip swap jika confidence rendah
        if confidence < ADAFACE_QUALITY_THRESHOLD:
            print(f"[AdaFace] Skipping swap - low confidence: {confidence:.3f}")
            return temp_frame

        # Get basic face swap
        swapped_result = get_face_swapper().get(aligned_frame, target_face, source_face, paste_back=False)
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

        # Adjust processing berdasarkan confidence
        if confidence > 0.7:  # High confidence - full processing
            swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)
            swapped_frame = enhance_face_quality(swapped_frame)
            result_frame = seamless_face_blending(swapped_frame, temp_frame, target_face)
        else:  # Medium confidence - simple blending
            swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)
            result_frame = simple_face_blending(swapped_frame, temp_frame, target_face)

        return result_frame
    except Exception:
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

# MODIFIED: process_frame dengan enhanced logging
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for idx, target_face in enumerate(many_faces):
                    setattr(target_face, 'face_index', idx)
                    
                    # NEW: Log matching confidence
                    if hasattr(roop.globals, 'use_adaface') and roop.globals.use_adaface:
                        confidence = enhanced_face_matching(source_face, target_face)
                        if confidence > 0.5:  # Only log good matches
                            print(f"[AdaFace] Face {idx} confidence: {confidence:.3f}")
                    
                    temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        return temp_frame
    except Exception:
        return temp_frame

# Fungsi-fungsi berikut TETAP SAMA (tidak perlu modifikasi):
# - fast_color_correction
# - create_simple_mask  
# - seamless_face_blending
# - simple_face_blending
# - enhance_face_quality
# - ensure_frame_format
# - process_frames
# - process_image
# - process_video

# [Fungsi-fungsi yang tidak disebutkan di atas tetap sama seperti original]
