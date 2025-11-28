# face_swapper.py (hybrid v3)
# Hybrid v3: improvements over v2
# - profile/side-pose handling (jawline weighting + horizontal scaling)
# - explicit mouth-aware local warp before swap
# - adaptive mask that shifts based on yaw/pitch
# - GFPGAN fidelity auto-scheduler based on pose/expression
# - noise re-introduction to avoid "waxy" skin

from typing import Any, List, Callable, Optional, Tuple, Dict
import threading
import time
import cv2
import numpy as np

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

# optional enhancer import
try:
    from roop.processors.frame.face_enhancer import get_face_enhancer
except Exception:
    get_face_enhancer = None

# ----------------- OneEuroFilter -----------------
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

# ----------------- module state & defaults -----------------
NAME = 'ROOP.FACE-SWAPPER'
FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
LANDMARK_FILTERS: Dict[str, Any] = {}
ONE_EURO_CONFIG = getattr(roop.globals, 'one_euro_config', {
    'freq': 30.0, 'min_cutoff': 1.0, 'beta': 0.012, 'd_cutoff': 1.0
})

# GFPGAN defaults
GFPGAN_BASE = float(getattr(roop.globals, 'face_enhancer_blend', 0.8))

# ----------------- model init -----------------
def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = __import__('insightface').model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx',
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx',
    ])
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
    LANDMARK_FILTERS.clear()

# ----------------- smoothing helpers -----------------
def get_filter_for_key(key: str) -> OneEuroFilter:
    if key not in LANDMARK_FILTERS:
        cfg = ONE_EURO_CONFIG
        LANDMARK_FILTERS[key] = OneEuroFilter(freq=cfg['freq'], min_cutoff=cfg['min_cutoff'], beta=cfg['beta'], d_cutoff=cfg['d_cutoff'])
    return LANDMARK_FILTERS[key]

def smooth_landmarks(landmarks: np.ndarray, key: str, timestamp: Optional[float] = None) -> np.ndarray:
    try:
        if landmarks is None:
            return landmarks
        f = get_filter_for_key(key)
        return f.filter(landmarks, t=timestamp)
    except Exception:
        return landmarks

def safe_get_landmarks(face: Face) -> Optional[np.ndarray]:
    if face is None:
        return None
    for attr in ['landmark_2d_106', 'landmark_2d', 'kps', 'landmarks']:
        if hasattr(face, attr):
            lm = getattr(face, attr)
            if lm is not None and len(lm) > 0:
                return np.asarray(lm, dtype=float)
    return None

# ----------------- pose helpers -----------------
def compute_pose_severity(face: Face) -> float:
    pitch, yaw, roll = get_face_pose(face)
    # severity 0..1 based on yaw magnitude
    return min(1.0, abs(yaw) / 60.0)

# ----------------- pose-aware bbox (softer) -----------------
def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    try:
        pitch, yaw, roll = get_face_pose(face)
        h_frame, w_frame = frame_shape[:2]
        x1, y1, x2, y2 = map(float, face.bbox)
        w = x2 - x1
        h = y2 - y1
        pad_x = 0.0
        pad_y_top = 0.0
        pad_y_bottom = 0.0
        if abs(yaw) > 20.0:
            extra = (abs(yaw) - 20.0) * 0.02
            extra = min(extra, 0.28)
            pad_x = w * extra
        if pitch < -12.0:
            extra = (abs(pitch) - 12.0) * 0.02
            extra = min(extra, 0.30)
            pad_y_top = h * extra
        elif pitch > 18.0:
            extra = (pitch - 18.0) * 0.015
            extra = min(extra, 0.18)
            pad_y_bottom = h * extra
        # soften but scale with pose severity
        severity = compute_pose_severity(face)
        soften = 0.75
        pad_x *= soften * (1.0 + severity*0.25)
        pad_y_top *= soften
        pad_y_bottom *= soften
        nx1 = int(max(0, x1 - pad_x))
        nx2 = int(min(w_frame - 1, x2 + pad_x))
        ny1 = int(max(0, y1 - pad_y_top))
        ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))
        if nx2 <= nx1 or ny2 <= ny1:
            return
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    except Exception:
        pass

# ----------------- mouth-aware local warp -----------------
def mouth_open_ratio(landmarks: np.ndarray) -> float:
    try:
        # indices approximate for 68/106 mapping; guard bounds
        if len(landmarks) > 66:
            top = landmarks[62]
            bottom = landmarks[66]
        elif len(landmarks) > 48:
            top = landmarks[51]
            bottom = landmarks[57]
        else:
            return 0.0
        vert = np.linalg.norm(bottom - top)
        eye_dist = np.linalg.norm(landmarks[36] - landmarks[45]) if len(landmarks) > 45 else 1.0
        return float(vert / max(1e-6, eye_dist))
    except Exception:
        return 0.0

def local_mouth_warp(frame: Frame, landmarks: np.ndarray, strength: float = 0.6) -> Frame:
    try:
        ratio = mouth_open_ratio(landmarks)
        if ratio < 0.18:
            return frame
        # compute mouth bbox (use inner lip indices if available)
        if len(landmarks) > 66:
            pts = landmarks[60:68]
        elif len(landmarks) > 57:
            pts = landmarks[48:60]
        else:
            return frame
        pts = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        pad = int(max(4, 0.25 * max(w, h)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return frame
        # apply a subtle vertical scaling to match mouth opening
        scale = 1.0 + (ratio - 0.18) * strength
        ch, cw = crop.shape[:2]
        new_h = max(1, int(ch * scale))
        resized = cv2.resize(crop, (cw, new_h), interpolation=cv2.INTER_LINEAR)
        # place center-aligned
        y_off = y1 - (new_h - ch)//2
        y_off = max(0, min(frame.shape[0]-new_h, y_off))
        # blend resized into frame using feathered mask
        mask = np.ones((new_h, cw), dtype=np.float32)
        k = int(max(3, min(cw, new_h) * 0.1))
        if k % 2 == 0: k += 1
        mask = cv2.GaussianBlur(mask, (k,k), 0)
        mask_3 = np.dstack([mask]*3)
        res = frame.copy()
        tgt_region = res[y_off:y_off+new_h, x1:x1+cw]
        if tgt_region.shape[:2] != resized.shape[:2]:
            resized = cv2.resize(resized, (tgt_region.shape[1], tgt_region.shape[0]))
            mask_3 = cv2.resize(mask_3, (tgt_region.shape[1], tgt_region.shape[0]))
        blended = (resized.astype(np.float32)*mask_3 + tgt_region.astype(np.float32)*(1-mask_3)).astype(np.uint8)
        res[y_off:y_off+blended.shape[0], x1:x1+blended.shape[1]] = blended
        return res
    except Exception:
        return frame

# ----------------- alignment (68-focused + jawline weighting + side scaling) -----------------
def robust_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray]:
    try:
        source_landmarks = safe_get_landmarks(source_face)
        target_landmarks = safe_get_landmarks(target_face)
        if source_landmarks is None or target_landmarks is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)
        timestamp = time.time()
        key_str = 'reference' if not getattr(roop.globals, 'many_faces', False) else f"face_{getattr(target_face,'face_index',0)}"
        sm_target = smooth_landmarks(target_landmarks, key_str, timestamp)
        sm_source = smooth_landmarks(source_landmarks, 'source', timestamp)
        if sm_source is None or sm_target is None:
            return temp_frame, np.eye(2, 3, dtype=np.float32)
        max_k = min(68, len(sm_source), len(sm_target))
        if max_k < 3:
            return temp_frame, np.eye(2, 3, dtype=np.float32)
        # prioritize: eyes(36-45), nose(27-35), mouth(48-67), jaw(0-16)
        indices = []
        # eyes
        for i in range(36,46):
            if i < max_k: indices.append(i)
        # nose
        for i in range(27,36):
            if i < max_k: indices.append(i)
        # mouth
        for i in range(48,68):
            if i < max_k: indices.append(i)
        # jaw (only a few)
        for i in range(0,9):
            if i < max_k: indices.append(i)
        if len(indices) < 3:
            indices = list(range(max_k))
        src_points = np.array([sm_source[i] for i in indices], dtype=np.float32)
        dst_points = np.array([sm_target[i] for i in indices], dtype=np.float32)
        # side-pose scaling: if yaw large, apply horizontal scale to source points to compensate
        pitch, yaw, roll = get_face_pose(target_face)
        yaw_abs = abs(yaw)
        if yaw_abs > 30 and src_points.shape[0] > 3:
            # scale X of source toward center to reduce profile collapse
            cx = np.mean(dst_points[:,0])
            scale = 1.0 + (yaw_abs - 30)/70.0  # modest scaling
            src_points[:,0] = cx + (src_points[:,0] - cx) * (1.0/scale)
        # mouth duplication for better mouth alignment
        try:
            if len(sm_target) >= 67:
                mouth_open = mouth_open_ratio(sm_target) > 0.2
                if mouth_open:
                    extra_idx = [60,62,64,66]
                    extra_src = np.array([sm_source[i] for i in extra_idx if i < len(sm_source)], dtype=np.float32)
                    extra_dst = np.array([sm_target[i] for i in extra_idx if i < len(sm_target)], dtype=np.float32)
                    if extra_src.size and extra_dst.size:
                        src_points = np.vstack([src_points, extra_src])
                        dst_points = np.vstack([dst_points, extra_dst])
        except Exception:
            pass
        transform_matrix = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS, ransacReprojThreshold=5.0)[0]
        if transform_matrix is not None:
            h, w = temp_frame.shape[:2]
            aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
            return aligned_frame, transform_matrix
        return temp_frame, np.eye(2, 3, dtype=np.float32)
    except Exception:
        return temp_frame, np.eye(2, 3, dtype=np.float32)

# ----------------- masks & blending -----------------
def create_adaptive_mask(face: Face, frame_shape: Tuple[int,int]) -> np.ndarray:
    try:
        pitch, yaw, roll = get_face_pose(face)
        x1, y1, x2, y2 = map(int, face.bbox)
        w = x2 - x1; h = y2 - y1
        mask = np.zeros(frame_shape[:2], dtype=np.float32)
        # shift ellipse center slightly toward camera-facing side
        yaw_sign = -1 if yaw < 0 else 1
        shift = int(w * 0.08 * np.sign(yaw_sign))
        center_x = int((x1 + x2)//2 + shift*(abs(yaw)/60.0))
        center_y = int((y1 + y2)//2)
        axes = (int(w*0.55), int(h*0.6))
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 1.0, -1)
        k = int(max(11, min(w,h) * 0.12))
        if k % 2 == 0: k += 1
        mask = cv2.GaussianBlur(mask, (k,k), 0)
        return np.clip(mask, 0, 1)
    except Exception:
        mask = np.zeros(frame_shape[:2], dtype=np.float32)
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        return cv2.GaussianBlur(mask, (51,51), 0)

def seamless_face_blending(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    try:
        if target_face is None:
            return target_frame
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_h, face_w = y2 - y1, x2 - x1
        if swapped_face.shape[:2] != (face_h, face_w):
            swapped_face = cv2.resize(swapped_face, (face_w, face_h))
        mask = 255 * np.ones(swapped_face.shape, swapped_face.dtype)
        center = ((x1+x2)//2, (y1+y2)//2)
        try:
            return cv2.seamlessClone(swapped_face, target_frame, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            maskf = create_adaptive_mask(target_face, target_frame.shape)
            mask_region = maskf[y1:y2, x1:x2]
            if mask_region.shape != swapped_face.shape[:2]:
                mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
            mask_3d = np.stack([mask_region]*3, axis=-1)
            res = target_frame.copy()
            face_region = res[y1:y2, x1:x2]
            blended = (swapped_face * mask_3d + face_region * (1 - mask_3d)).astype(np.uint8)
            res[y1:y2, x1:x2] = blended
            return res
    except Exception:
        return target_frame

# ----------------- color correction & enhance -----------------
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
        if swapped_face.shape[:2] != target_region.shape[:2]:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
        swapped_mean = np.mean(swapped_lab, axis=(0,1)); swapped_std = np.std(swapped_lab, axis=(0,1))
        target_mean = np.mean(target_lab, axis=(0,1)); target_std = np.std(target_lab, axis=(0,1))
        swapped_std = np.where(swapped_std == 0, 1, swapped_std)
        target_std = np.where(target_std == 0, 1, target_std)
        corrected_lab = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected_lab[:,:,i] = (swapped_lab[:,:,i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        result_face = cv2.addWeighted(swapped_face, 0.33, corrected_face, 0.67, 0)
        return result_face
    except Exception:
        return swapped_face

def enhance_face_quality(face: Frame) -> Frame:
    try:
        kernel = np.array([[-1,-1,-1],[-1,11,-1],[-1,-1,-1]]) * 0.12
        sharpened = cv2.filter2D(face, -1, kernel)
        denoised = cv2.bilateralFilter(sharpened, 5, 12, 12)
        return denoised
    except Exception:
        return face

# ----------------- noise reintroduction -----------------
def reintroduce_noise(face: Frame, strength: float = 0.02) -> Frame:
    try:
        if strength <= 0:
            return face
        noise = np.random.normal(loc=0.0, scale=25.0*strength, size=face.shape).astype(np.float32)
        res = face.astype(np.float32) + noise
        res = np.clip(res, 0, 255).astype(np.uint8)
        return res
    except Exception:
        return face

# ----------------- GFPGAN fidelity scheduler -----------------
def compute_enhancer_fidelity(face: Face, landmarks: Optional[np.ndarray]) -> float:
    try:
        base = GFPGAN_BASE
        severity = compute_pose_severity(face)
        mouth_r = mouth_open_ratio(landmarks) if landmarks is not None else 0.0
        # heuristics: more mouth -> higher fidelity, more profile -> lower fidelity
        fidelity = base
        fidelity += min(0.18, mouth_r * 0.8)
        fidelity -= min(0.25, severity * 0.35)
        fidelity = float(max(0.45, min(0.95, fidelity)))
        return fidelity
    except Exception:
        return float(base)

# ----------------- swap core (v3) -----------------
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame
    adapt_bbox_for_pose(target_face, temp_frame.shape)
    # mouth-aware prewarp
    target_lm = safe_get_landmarks(target_face)
    if target_lm is not None:
        temp_frame = local_mouth_warp(temp_frame, target_lm, strength=0.8)
    aligned_frame, _ = robust_face_alignment(source_face, target_face, temp_frame)
    try:
        inswapper = get_face_swapper()
        swapped = inswapper.get(aligned_frame, target_face, source_face, paste_back=False)
        swapped_frame = swapped if isinstance(swapped, np.ndarray) else np.array(swapped)
    except Exception:
        try:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        except Exception:
            return temp_frame
    if swapped_frame is None or swapped_frame.size == 0:
        return temp_frame
    swapped_frame = fast_color_correction(swapped_frame, temp_frame, target_face)
    swapped_frame = enhance_face_quality(swapped_frame)
    # slight noise to avoid waxy look
    swapped_frame = reintroduce_noise(swapped_frame, strength=0.015)
    result = seamless_face_blending(swapped_frame, temp_frame, target_face)
    # post-enhancer (GFPGAN) with fidelity scheduler
    try:
        if get_face_enhancer is not None:
            enhancer = get_face_enhancer()
            x1, y1, x2, y2 = map(int, target_face.bbox)
            h, w = result.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = result[y1:y2, x1:x2]
            if crop.size:
                with threading.Semaphore():
                    try:
                        _, _, enhanced = enhancer.enhance(crop, paste_back=False)
                        fidelity = compute_enhancer_fidelity(target_face, target_lm)
                        if enhanced is not None and enhanced.size:
                            if enhanced.shape[:2] != crop.shape[:2]:
                                enhanced = cv2.resize(enhanced, (crop.shape[1], crop.shape[0]))
                            blended = (enhanced.astype(np.float32)*fidelity + crop.astype(np.float32)*(1.0-fidelity)).astype(np.uint8)
                            blended = reintroduce_noise(blended, strength=0.008)
                            result[y1:y2, x1:x2] = blended
                    except Exception:
                        pass
    except Exception:
        pass
    return result

# ----------------- frame processing glue -----------------
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame
    if getattr(roop.globals, 'many_faces', False):
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame
        for idx, f in enumerate(faces):
            setattr(f, 'face_index', idx)
            if detect_occlusion(f, temp_frame):
                continue
            temp_frame = swap_face(source_face, f, temp_frame)
        return temp_frame
    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)
    if not tracked:
        return temp_frame
    valid = [f for f in tracked if not detect_occlusion(f, temp_frame)]
    if not valid:
        return temp_frame
    best = None
    if reference_face is not None:
        best = find_similar_face(temp_frame, reference_face, use_tracking=True)
    if best is None:
        best = valid[0]
    temp_frame = swap_face(source_face, best, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if getattr(roop.globals, 'many_faces', False) else get_face_reference()
    for idx, temp_path in enumerate(temp_frame_paths):
        temp = cv2.imread(temp_path)
        res = process_frame(source_face, reference_face, temp, frame_number=idx)
        cv2.imwrite(temp_path, res)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if getattr(roop.globals, 'many_faces', False) else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame, frame_number=0)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not getattr(roop.globals, 'many_faces', False) and not get_face_reference():
        try:
            ref_idx = int(getattr(roop.globals, 'reference_frame_number', 0))
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
