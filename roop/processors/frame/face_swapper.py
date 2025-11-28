from typing import Any, List, Callable, Optional, Tuple, Dict
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
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# Hybrid face-swapper: combine smart tracking/pose/occlusion (face-swpper support new)
# with OneEuro smoothing, robust alignment, color correction and blending (from swapper OneEuro)

NAME = 'ROOP.FACE-SWAPPER-HYBRID'
FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
LANDMARK_FILTERS: Dict[str, Any] = {}

ONE_EURO_CONFIG = {
    'freq': 30.0,
    'min_cutoff': 1.0,
    'beta': 0.007,
    'd_cutoff': 1.0
}

# ------------------- OneEuroFilter -------------------
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

# ------------------- face swapper model -------------------

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

# ------------------- utilities adapted -------------------

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

# robust alignment uses smoothed landmarks

def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray]:
    try:
        src_land = safe_get_landmarks(source_face)
        dst_land = safe_get_landmarks(target_face)
        if src_land is None or dst_land is None:
            return temp_frame, np.eye(2,3,dtype=np.float32)
        timestamp = time.time()
        # use stable key per-target when many_faces, else 'reference'
        if roop.globals.many_faces:
            key = getattr(target_face, 'face_index', None)
            key_str = f"face_{key}" if key is not None else 'many_unknown'
        else:
            key_str = 'reference'
        sm_dst = get_filter_for_key(key_str).filter(dst_land, t=timestamp)
        sm_src = get_filter_for_key('source').filter(src_land, t=timestamp)
        if sm_src is None or sm_dst is None:
            return temp_frame, np.eye(2,3,dtype=np.float32)
        # pick up to 5 stable keypoints
        landmark_indices = list(range(min(5, len(sm_src))))
        key_points = [i for i in landmark_indices if i < len(sm_src) and i < len(sm_dst)]
        if len(key_points) < 3:
            return temp_frame, np.eye(2,3,dtype=np.float32)
        src_points = np.array([sm_src[i] for i in key_points], dtype=np.float32)
        dst_points = np.array([sm_dst[i] for i in key_points], dtype=np.float32)
        transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0)[0]
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w,h), flags=cv2.INTER_LINEAR)
            return aligned_frame, transform_matrix
        return temp_frame, np.eye(2,3,dtype=np.float32)
    except Exception:
        return temp_frame, np.eye(2,3,dtype=np.float32)

# color correction from old swapper

def fast_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        if target_face is None or swapped_face is None:
            return swapped_face
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0,x1), max(0,y1)
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

# simple mask & blending

def create_simple_mask(face: Face, frame_shape: Tuple[int,int]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        x1,y1,x2,y2 = map(int, face.bbox)
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        w = x2-x1
        h = y2-y1
        cv2.ellipse(mask, (cx,cy), (w//2, h//2), 0,0,360,1.0,-1)
        mask = cv2.GaussianBlur(mask, (25,25), 0)
        return np.clip(mask,0,1)
    except Exception:
        x1,y1,x2,y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51,51), 0)
        return mask


def seamless_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        if target_face is None:
            return target_frame
        x1,y1,x2,y2 = map(int, target_face.bbox)
        h,w = target_frame.shape[:2]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        fh,fw = y2-y1, x2-x1
        if swapped_face.shape[0] != fh or swapped_face.shape[1] != fw:
            swapped_face = cv2.resize(swapped_face, (fw, fh))
        mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
        center = ((x1+x2)//2, (y1+y2)//2)
        result = cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        return result
    except Exception:
        # fallback to simple blending
        try:
            mask = create_simple_mask(target_face, target_frame.shape)
            mask_reg = mask[y1:y2, x1:x2]
            if mask_reg.shape != swapped_face.shape[:2]:
                mask_reg = cv2.resize(mask_reg, (swapped_face.shape[1], swapped_face.shape[0]))
            mask_3 = np.stack([mask_reg]*3, axis=-1)
            res = target_frame.copy()
            face_region = res[y1:y2, x1:x2]
            blended = (swapped_face * mask_3 + face_region * (1-mask_3)).astype(np.uint8)
            res[y1:y2, x1:x2] = blended
            return res
        except Exception:
            return target_frame

# enhancement hook (call GFPGAN pipeline if available in your pipeline)

def enhance_face_quality(face: Frame) -> Frame:
    try:
        if face is None:
            return face
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) * 0.15
        sharpened = cv2.filter2D(face, -1, kernel)
        denoised = cv2.bilateralFilter(sharpened, 5, 15, 15)
        return denoised
    except Exception:
        return face

# ------------------- face enhancer wrapper (kept for compatibility)

def face_enhancer(frame: Frame) -> Frame:
    try:
        # Placeholder: integrate your GFPGAN/CodeFormer enhancer-final here
        # This ensures the processor name 'face_enhancer' still exists and avoids CLI errors.
        return frame
    except Exception:
        return frame

# ------------------- core hybrid swapping -------------------

def swap_face_hybrid(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # 1) pose-aware bbox already applied by caller
    # 2) alignment using smoothed landmarks
    aligned_frame, _ = robust_face_alignment(source_face, target_face, temp_frame)
    # 3) call inswapper on aligned frame, requesting paste_back=False to get swapped crop
    try:
        swapped_result = get_face_swapper().get(aligned_frame, target_face, source_face, paste_back=False)
        swapped_frame = swapped_result if isinstance(swapped_result, np.ndarray) else None
    except Exception:
        swapped_frame = None
    if swapped_frame is None:
        # fallback: direct paste_back
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    # 4) color correction
    swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)
    # 5) enhance
    swapped_frame = enhance_face_quality(swapped_frame)
    # 6) blend
    result = seamless_face_blending(swapped_frame, temp_frame, target_face)
    return result

# ------------------- process_frame with smart tracking -------------------

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    # minimal re-implementation: use get_face_pose
    try:
        pitch, yaw, roll = get_face_pose(face)
        h_frame, w_frame = frame_shape[:2]
        x1,y1,x2,y2 = map(int, face.bbox)
        w = x2-x1
        h = y2-y1
        pad_x = 0.0
        pad_y_top = 0.0
        pad_y_bottom = 0.0
        if abs(yaw) > 25.0:
            extra = (abs(yaw) - 25.0) * 0.02
            extra = min(extra, 0.20)
            pad_x = w * extra
        if pitch < -15.0:
            extra = (abs(pitch) - 15.0) * 0.02
            extra = min(extra, 0.25)
            pad_y_top = h * extra
        elif pitch > 20.0:
            extra = (pitch - 20.0) * 0.015
            extra = min(extra, 0.18)
            pad_y_bottom = h * extra
        nx1 = int(max(0, x1 - pad_x))
        nx2 = int(min(w_frame - 1, x2 + pad_x))
        ny1 = int(max(0, y1 - pad_y_top))
        ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))
        if nx2 <= nx1 or ny2 <= ny1:
            return
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    except Exception:
        return


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame
    # many faces mode: swap semua yang valid
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame
        for idx, target_face in enumerate(faces):
            setattr(target_face, 'face_index', idx)
            # occlusion
            if detect_occlusion(target_face, temp_frame):
                continue
            adapt_bbox_for_pose(target_face, temp_frame.shape)
            temp_frame = swap_face_hybrid(source_face, target_face, temp_frame)
        return temp_frame
    # single focus mode
    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)
    if not tracked:
        return temp_frame
    valid = [f for f in tracked if not detect_occlusion(f, temp_frame)]
    if not valid:
        return temp_frame
    best_target = None
    if reference_face is not None:
        # embedding-based selection (fallback to find_similar_face logic)
        best_target = find_similar_face(temp_frame, reference_face, use_tracking=True)
    if best_target is None:
        best_target = valid[0]
    adapt_bbox_for_pose(best_target, temp_frame.shape)
    temp_frame = swap_face_hybrid(source_face, best_target, temp_frame)
    return temp_frame

# ------------------- drivers -------------------

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame, frame_number=idx)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame, frame_number=0)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
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
