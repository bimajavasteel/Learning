from typing import Any, List, Callable, Tuple, Optional, Dict
import cv2
import insightface
import threading
import numpy as np
import os
import time
from scipy import ndimage

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# ----------------------------
# Enhanced swapper with improvements:
# - Pre-enhance source face (GFPGAN)
# - Temporal smoothing (OneEuroFilter) for bbox & landmarks
# - Landmark-based mild rotation alignment
# - Robust color correction + seamless cloning
# - Thread-safe singletons
# ----------------------------

FACE_SWAPPER = None
FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()
NAME = 'ROOP.SWAPPER-ENHANCED'

# per-face smoothing state
SMOOTHERS: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# OneEuroFilter (simple implementation)
# ----------------------------
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if self.t_prev is None and t is None:
            self.t_prev = time.time()
        if t is None:
            t = time.time()
        dt = t - self.t_prev if self.t_prev is not None else 1.0 / self.freq
        if dt <= 0:
            dt = 1.0 / self.freq
        self.freq = 1.0 / dt

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dx = (x - self.x_prev) * self.freq
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self.alpha(cutoff)

        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

# ----------------------------
# Utility helpers
# ----------------------------

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def get_face_enhancer() -> Any:
    """Lazy-load GFPGANer if available. Returns None if not present."""
    global FACE_ENHANCER
    try:
        from gfpgan.utils import GFPGANer
    except Exception:
        return None

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            device = 'cuda' if 'CUDAExecutionProvider' in roop.globals.execution_providers else ('mps' if 'CoreMLExecutionProvider' in roop.globals.execution_providers else 'cpu')
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=device)
    return FACE_ENHANCER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'])
    # GFPGAN optional
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])
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
    clear_face_enhancer()
    SMOOTHERS.clear()

# ----------------------------
# Smoothing helpers
# ----------------------------

def _smoother_key_for_face(face: Face) -> str:
    # create a deterministic key from bbox rounded
    try:
        bbox = getattr(face, 'bbox', None) or (face['bbox'] if isinstance(face, dict) and 'bbox' in face else None)
    except Exception:
        bbox = None
    if bbox is None:
        return str(id(face))
    x1, y1, x2, y2 = map(int, bbox)
    return f"{x1}-{y1}-{x2}-{y2}"


def _ensure_smoothers(key: str, n_landmarks: int = 5):
    if key not in SMOOTHERS:
        SMOOTHERS[key] = {
            'bbox_filter': OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.01),
            'landmark_filters': [OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.01) for _ in range(n_landmarks)]
        }
    return SMOOTHERS[key]


def smooth_face_props(face: Face) -> Face:
    """Return a shallow copy of face with smoothed bbox & landmarks where available."""
    try:
        key = _smoother_key_for_face(face)
        smoother = _ensure_smoothers(key)

        # handle bbox
        bbox = None
        if hasattr(face, 'bbox'):
            bbox = np.array(list(map(float, face.bbox)))
        elif isinstance(face, dict) and 'bbox' in face:
            bbox = np.array(list(map(float, face['bbox'])))

        if bbox is not None:
            bbox_sm = smoother['bbox_filter'].filter(bbox)
            if hasattr(face, 'bbox'):
                face.bbox = tuple(map(float, bbox_sm))
            elif isinstance(face, dict):
                face['bbox'] = tuple(map(float, bbox_sm))

        # handle landmarks (try several common keys)
        landmarks = None
        if hasattr(face, 'landmark') and face.landmark is not None:
            landmarks = np.array(face.landmark)
        elif isinstance(face, dict) and 'kps' in face:
            landmarks = np.array(face['kps'])
        elif isinstance(face, dict) and 'landmarks' in face:
            landmarks = np.array(face['landmarks'])

        if landmarks is not None and len(landmarks) >= len(smoother['landmark_filters']):
            sm_ld = []
            for i, lf in enumerate(smoother['landmark_filters']):
                sm = lf.filter(np.array(landmarks[i]))
                sm_ld.append(tuple(sm.tolist()))
            # write back
            if hasattr(face, 'landmark'):
                face.landmark = sm_ld
            elif isinstance(face, dict):
                if 'kps' in face:
                    face['kps'] = sm_ld
                else:
                    face['landmarks'] = sm_ld

    except Exception as e:
        print(f"[WARNING] smooth_face_props error: {e}")
    return face

# ----------------------------
# Image utilities
# ----------------------------

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


def simple_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        if target_face is None:
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
        swapped_mean, swapped_std = np.mean(swapped_lab, axis=(0,1)), np.std(swapped_lab, axis=(0,1))
        target_mean, target_std = np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))
        swapped_std = np.where(swapped_std == 0, 1, swapped_std)
        target_std = np.where(target_std == 0, 1, target_std)
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        blend_ratio = 0.7
        result_face = cv2.addWeighted(swapped_face, 1 - blend_ratio, corrected_face, blend_ratio, 0)
        return result_face
    except Exception as e:
        print(f"Simple color correction error: {e}")
        return swapped_face


def create_smooth_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        cv2.ellipse(mask, (center_x, center_y), (max(1,width//2), max(1,height//2)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        return np.clip(mask, 0, 1)
    except Exception as e:
        print(f"Mask creation error: {e}")
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            mask[y1:y2, x1:x2] = 1.0
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            return mask
        except Exception:
            return mask

# ----------------------------
# Enhancement utilities
# ----------------------------

def pre_enhance_source(source_path: str) -> str:
    """Enhance source face and write to temp file, return new path. If enhancer not available -> original."""
    try:
        enhancer = get_face_enhancer()
        if enhancer is None:
            return source_path
        img = cv2.imread(source_path)
        if img is None:
            return source_path
        face = get_one_face(img)
        if not face:
            return source_path
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return source_path
        with THREAD_SEMAPHORE:
            _, _, enhanced = enhancer.enhance(crop, paste_back=False)
        if enhanced is not None and enhanced.shape[0] > 0:
            out = img.copy()
            # resize enhanced to original crop size
            if enhanced.shape[:2] != crop.shape[:2]:
                enhanced = cv2.resize(enhanced, (crop.shape[1], crop.shape[0]))
            out[y1:y2, x1:x2] = enhanced
            tmp = os.path.join('/tmp', f'enhanced_source_{int(time.time()*1000)}.png')
            cv2.imwrite(tmp, out)
            return tmp
    except Exception as e:
        print(f"pre_enhance_source error: {e}")
    return source_path

# ----------------------------
# Rotation estimation from landmarks (simple)
# ----------------------------

def estimate_rotation_angle_from_landmarks(landmarks: List[Tuple[float, float]]) -> float:
    # use eyes: assume landmarks[0]=left eye, [1]=right eye (common conventions)
    try:
        if len(landmarks) >= 2:
            (x1, y1), (x2, y2) = landmarks[0], landmarks[1]
            dx = x2 - x1
            dy = y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            return angle
    except Exception:
        pass
    return 0.0

# ----------------------------
# Blending helpers
# ----------------------------

def seamless_blending(swapped_face: Frame, target_frame: Frame, target_face: Face, angle: float = 0.0) -> Frame:
    try:
        if target_face is None:
            return target_frame
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_h, face_w = y2 - y1, x2 - x1
        if face_h <= 0 or face_w <= 0:
            return target_frame
        # resize
        if swapped_face.shape[0] != face_h or swapped_face.shape[1] != face_w:
            swapped_face = cv2.resize(swapped_face, (face_w, face_h))
        # rotate swapped_face by -angle to align
        if abs(angle) > 1.0:
            M = cv2.getRotationMatrix2D((face_w/2, face_h/2), -angle, 1.0)
            swapped_face = cv2.warpAffine(swapped_face, M, (face_w, face_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        return result
    except Exception as e:
        print(f"Seamless blending error: {e}")
        return target_frame

# ----------------------------
# Core swap logic with smoothing & enhancements
# ----------------------------

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        # Smooth target properties to reduce jitter
        target_face = smooth_face_props(target_face)

        # Attempt to estimate rotation from landmarks (if present)
        landmarks = None
        if hasattr(target_face, 'landmark') and target_face.landmark is not None:
            landmarks = target_face.landmark
        elif isinstance(target_face, dict) and 'kps' in target_face:
            landmarks = target_face['kps']
        angle = estimate_rotation_angle_from_landmarks(landmarks) if landmarks is not None else 0.0

        # Call swapper
        swapped_result = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=False)
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            # fallback
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

        # Color correction
        swapped_frame = simple_color_correction(swapped_frame, temp_frame, target_face)

        # Mild enhancement of swapped face (sharpen + denoise)
        try:
            kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]) * 0.15
            swapped_frame = cv2.filter2D(swapped_frame, -1, kernel)
            swapped_frame = cv2.bilateralFilter(swapped_frame, 5, 25, 25)
        except Exception:
            pass

        # Blending with rotation compensation
        result = seamless_blending(swapped_frame, temp_frame, target_face, angle=angle)
        return result
    except Exception as e:
        print(f"Face swap error: {e}")
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    try:
        # Get many faces
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    try:
        # Pre-enhance source (write temp file)
        enhanced_source = pre_enhance_source(source_path)
        source_face = get_one_face(cv2.imread(enhanced_source))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        for temp_frame_path in temp_frame_paths:
            try:
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is not None:
                    result = process_frame(source_face, reference_face, temp_frame)
                    cv2.imwrite(temp_frame_path, result)
                if update:
                    update()
            except Exception as e:
                print(f"Error processing frame {temp_frame_path}: {e}")
                continue
    except Exception as e:
        print(f"Process frames error: {e}")


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        enhanced_source = pre_enhance_source(source_path)
        source_face = get_one_face(cv2.imread(enhanced_source))
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
