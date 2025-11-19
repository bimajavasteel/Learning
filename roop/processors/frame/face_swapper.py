from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
import time

import roop.globals
import roop.processors.frame.core
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# =====================
# OneEuroFilter Utility
# =====================
class OneEuroFilter:
    """Simple One Euro Filter implementation for smoothing landmark coordinates.
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """
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
        """x: numpy array shape (N,2 or 3..)
           t: timestamp (optional), used to adapt freq if provided
        """
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
NAME = 'ROOP.FACE-SWAPPER-ONEEURO'

# Landmark filters storage
# For non-many_faces mode we use key 'reference'
# For many_faces mode we keep a list and index by enumerate order per frame.
LANDMARK_FILTERS: Dict[str, Any] = {}
ONE_EURO_CONFIG = {
    'freq': 30.0,        # assumed FPS -- adjust to actual FPS of your video
    'min_cutoff': 1.0,
    'beta': 0.007,       # low beta = smoother; increase to follow faster motion
    'd_cutoff': 1.0
}

# =====================
# Helpers (original ultimate code adapted)
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
    from roop.core import update_status   # moved to inside function

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

# safe_get_landmarks from ultimate

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

# Landmark smoothing wrappers

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

# robust_face_alignment reused from ultimate but uses smoothed landmarks if provided

def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray]:
    try:
        source_landmarks = safe_get_landmarks(source_face)
        target_landmarks = safe_get_landmarks(target_face)

        if source_landmarks is None or target_landmarks is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        # If many_faces, assume ordering stable and key by frame index provided by caller
        # Otherwise use 'reference' key
        timestamp = time.time()
        if roop.globals.many_faces:
            # caller must set a temporary attribute 'face_index' on target_face (we do that in process_frame)
            key = getattr(target_face, 'face_index', None)
            key_str = f"face_{key}" if key is not None else 'many_unknown'
        else:
            key_str = 'reference'

        # Smooth target landmarks
        sm_target = smooth_landmarks(target_landmarks, key_str, timestamp)

        # For source (static) we can smooth with 'source' key once
        sm_source = smooth_landmarks(source_landmarks, 'source', timestamp)

        # Continue similar to original but using smoothed landmarks
        if sm_source is None or sm_target is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        if len(sm_source) < 3 or len(sm_target) < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        # choose a subset of keypoints (eyes + nose + mouth corners) if available
        landmark_indices = list(range(min(5, len(sm_source))))
        key_points = [i for i in landmark_indices if i < len(sm_source) and i < len(sm_target)]

        if len(key_points) < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32)

        src_points = np.array([sm_source[i] for i in key_points], dtype=np.float32)
        dst_points = np.array([sm_target[i] for i in key_points], dtype=np.float32)

        transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0)[0]
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
            return aligned_frame, transform_matrix

        return temp_frame, np.eye(2, 3, dtype=np.float32)
    except Exception:
        return temp_frame, np.eye(2, 3, dtype=np.float32)

# color correction + blending + enhancement (kept compact, re-used logic)

def fast_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
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

# swap_face_optimized that uses OneEuro smoothed landmarks

def swap_face_optimized(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        # Apply robust face alignment (which now uses smoothed landmarks)
        aligned_frame, _ = robust_face_alignment(source_face, target_face, temp_frame)

        # Get basic face swap from InsightFace
        swapped_result = get_face_swapper().get(aligned_frame, target_face, source_face, paste_back=False)
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

        # Color correction
        swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)

        # Enhance
        swapped_frame = enhance_face_quality(swapped_frame)

        # Blend
        result_frame = seamless_face_blending(swapped_frame, temp_frame, target_face)
        return result_frame
    except Exception:
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

# process_frame adapted to assign face indices for many_faces mode

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for idx, target_face in enumerate(many_faces):
                    # attach a stable index per frame to help landmark smoothing list
                    setattr(target_face, 'face_index', idx)
                    temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                # use 'reference' key for smoothing
                temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        return temp_frame
    except Exception:
        return temp_frame

# frames processing (same as original)

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
