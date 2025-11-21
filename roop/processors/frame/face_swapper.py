"""
face_swapper_ultimate.py

Face Swapper Ultimate (FaceFusion-like implementation) for Roop
- Uses InsightFace InSwapper (inswapper_128.onnx)
- Adds: tracking (IoU + embedding), OneEuro landmark smoothing, robust alignment
  optical-flow-assisted temporal stabilization of swapped patches, temporal color
  stabilization, dynamic masks, multi-face safe processing, and graceful fallbacks.
- Compatible with roop processors (process_frames/process_image/process_video)

NOTE: this file assumes the presence of roop modules (face_analyser, face_reference, processors.frame.core, etc.)
and that Face objects expose landmarks and optional embeddings similar to Roop/InsightFace.
"""

from typing import Any, List, Callable, Tuple, Optional, Dict
import threading
import time
import math
import cv2
import numpy as np
import insightface

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# --------------------
# OneEuroFilter (landmark smoothing)
# --------------------
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
        tau = 1.0 / (2 * math.pi * cutoff)
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

# --------------------
# Globals & configs
# --------------------
FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER-ULTIMATE'

# OneEuro per track storage
LANDMARK_FILTERS: Dict[str, Any] = {}
ONE_EURO_CONFIG = {
    'freq': 30.0,
    'min_cutoff': 1.0,
    'beta': 0.007,
    'd_cutoff': 1.0
}

# Tracking state
class TrackState:
    def __init__(self, track_id: int, bbox: Tuple[int,int,int,int], embedding: Optional[np.ndarray]=None):
        self.track_id = track_id
        self.bbox = np.array(bbox, dtype=float)
        self.embedding = embedding
        self.last_seen = 0
        self.last_swapped_patch: Optional[np.ndarray] = None
        self.last_frame_gray: Optional[np.ndarray] = None
        self.color_mean: Optional[np.ndarray] = None
        self.color_std: Optional[np.ndarray] = None

TRACKS: Dict[int, TrackState] = {}
NEXT_TRACK_ID = 0
FRAME_INDEX = 0

# Parameters
IOU_MATCH_THRESHOLD = 0.12
IOU_RETIRE_FRAMES = 60
FLOW_MAG_CLIP = 20.0
OPTICAL_FLOW_BLEND_MAX = 0.85
MIN_FACE_SIZE_FOR_PATCH = 32

# --------------------
# InsightFace InSwapper loader
# --------------------

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
    TRACKS.clear()

# --------------------
# Landmark helpers
# --------------------

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
        return f.filter(landmarks, t=timestamp)
    except Exception:
        return landmarks

# --------------------
# Track matching (IoU + embedding)
# --------------------

def bbox_array(face: Face) -> np.ndarray:
    return np.array(face.bbox, dtype=float)


def iou(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def get_face_embedding(face: Face) -> Optional[np.ndarray]:
    try:
        if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
            emb = np.asarray(face.normed_embedding, dtype=np.float32)
        elif hasattr(face, 'embedding') and face.embedding is not None:
            emb = np.asarray(face.embedding, dtype=np.float32)
        else:
            return None
        if np.linalg.norm(emb) > 0:
            return emb / np.linalg.norm(emb)
    except Exception:
        return None
    return None


def match_tracks(detected_faces: List[Face]) -> Dict[int, int]:
    """Return mapping index_face -> track_id"""
    global NEXT_TRACK_ID, FRAME_INDEX
    mapping: Dict[int, int] = {}
    used = set()

    det_bboxes = [bbox_array(f) for f in detected_faces]
    det_embs = [get_face_embedding(f) for f in detected_faces]

    for idx, (box, emb) in enumerate(zip(det_bboxes, det_embs)):
        best_tid = None
        best_score = -1.0
        for tid, t in TRACKS.items():
            if tid in used:
                continue
            s_iou = iou(box, t.bbox)
            score = 0.7 * s_iou
            if emb is not None and t.embedding is not None:
                cos = float(np.dot(emb, t.embedding))
                score += 0.3 * max(cos, 0.0)
            if score > best_score:
                best_score = score
                best_tid = tid

        if best_tid is None or best_score < IOU_MATCH_THRESHOLD:
            tid = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1
            TRACKS[tid] = TrackState(tid, tuple(box.tolist()), det_embs[idx])
            TRACKS[tid].last_seen = FRAME_INDEX
            mapping[idx] = tid
            used.add(tid)
        else:
            t = TRACKS[best_tid]
            t.bbox = box
            t.last_seen = FRAME_INDEX
            if emb is not None:
                t.embedding = emb
            mapping[idx] = best_tid
            used.add(best_tid)

    # cleanup old tracks
    to_del = [tid for tid, t in TRACKS.items() if FRAME_INDEX - t.last_seen > IOU_RETIRE_FRAMES]
    for tid in to_del:
        TRACKS.pop(tid, None)

    return mapping

# --------------------
# Optical flow helpers
# --------------------

def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow
    except Exception:
        return None


def warp_with_flow(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)
    warped = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

# --------------------
# Color correction, mask & blending
# --------------------

def fast_color_correction(swapped_face: np.ndarray, target_frame: np.ndarray, bbox: Tuple[int,int,int,int], track: Optional[TrackState]=None) -> np.ndarray:
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))

        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

        s_mean = swapped_lab.mean(axis=(0,1))
        s_std = swapped_lab.std(axis=(0,1))
        t_mean = target_lab.mean(axis=(0,1))
        t_std = target_lab.std(axis=(0,1))
        s_std = np.where(s_std==0, 1.0, s_std)
        t_std = np.where(t_std==0, 1.0, t_std)

        if track is not None and track.color_mean is not None:
            t_mean = 0.7*t_mean + 0.3*track.color_mean
            t_std = 0.7*t_std + 0.3*track.color_std

        corrected = np.zeros_like(swapped_lab)
        for i in range(3):
            corrected[:,:,i] = (swapped_lab[:,:,i] - s_mean[i]) * (t_std[i]/s_std[i]) + t_mean[i]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)

        # update track stats
        if track is not None:
            track.color_mean = t_mean
            track.color_std = t_std
        return corrected_bgr
    except Exception:
        return swapped_face


def create_mask_for_bbox(bbox: Tuple[int,int,int,int], frame_shape: Tuple[int,int]) -> np.ndarray:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2)//2
    cy = (y1 + y2)//2
    fw = max(1, x2-x1)
    fh = max(1, y2-y1)
    ax = int(fw*0.5)
    ay = int(fh*0.6)
    mask = np.zeros((h,w), dtype=np.float32)
    cv2.ellipse(mask, (cx,cy), (ax,ay), 0, 0, 360, 1.0, -1)
    k = int(max(15, min(51, int(max(fw,fh)*0.22))))
    if k % 2 == 0:
        k += 1
    mask = cv2.GaussianBlur(mask, (k,k), 0)
    return np.clip(mask, 0.0, 1.0)


def simple_blend_patch(frame: np.ndarray, patch: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x1,y1,x2,y2 = map(int, bbox)
    sx,sy,ex,ey = x1,y1,x2,y2
    if patch.shape[:2] != (ey-sy, ex-sx):
        patch = cv2.resize(patch, (ex-sx, ey-sy), interpolation=cv2.INTER_LINEAR)
    mask = create_mask_for_bbox((sx,sy,ex,ey), frame.shape)
    mask_region = mask[sy:ey, sx:ex]
    mask_3 = np.stack([mask_region]*3, axis=-1)
    roi = frame[sy:ey, sx:ex].astype(np.float32)
    blended = (patch.astype(np.float32)*mask_3 + roi*(1.0-mask_3)).astype(np.uint8)
    frame[sy:ey, sx:ex] = blended
    return frame

# --------------------
# Alignment + robust swap
# --------------------

def safe_get_keypoints(face: Face) -> Optional[np.ndarray]:
    return safe_landmarks = safe_get_landmarks(face)


def robust_alignment_and_swap(source_face: Face, target_face: Face, frame: Frame, track: Optional[TrackState]=None) -> Tuple[Frame, np.ndarray]:
    """Return (swapped_face_patch, transform_matrix) where swapped_face_patch is the swapped face image patch (not yet pasted)"""
    try:
        src_land = safe_get_landmarks(source_face)
        tgt_land = safe_get_landmarks(target_face)
        if src_land is None or tgt_land is None:
            # fallback to direct get with paste_back True
            out = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
            return out, np.eye(2,3,dtype=np.float32)

        # smooth landmarks
        timestamp = time.time()
        key = getattr(target_face, 'track_id', 'reference')
        key_str = f"track_{key}"
        sm_tgt = smooth_landmarks(tgt_land, key_str, timestamp)
        sm_src = smooth_landmarks(src_land, 'source', timestamp)

        if sm_src is None or sm_tgt is None:
            out = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
            return out, np.eye(2,3,dtype=np.float32)

        if len(sm_src) < 3 or len(sm_tgt) < 3:
            out = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
            return out, np.eye(2,3,dtype=np.float32)

        # choose subset indices (eyes, nose, mouth corners) conservative approach
        k = min(8, len(sm_src), len(sm_tgt))
        src_pts = np.asarray([sm_src[i] for i in range(k)], dtype=np.float32)
        dst_pts = np.asarray([sm_tgt[i] for i in range(k)], dtype=np.float32)

        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS, ransacReprojThreshold=4.0)
        if M is None:
            out = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
            return out, np.eye(2,3,dtype=np.float32)

        h,w = frame.shape[:2]
        aligned = cv2.warpAffine(frame, M, (w,h), flags=cv2.INTER_LINEAR)

        # get swap result on aligned frame but don't paste back yet
        swapped = get_face_swapper().get(aligned, target_face, source_face, paste_back=False)
        swapped_img = swapped if isinstance(swapped, np.ndarray) else swapped
        swapped_patch = ensure_frame_format(swapped_img)
        if swapped_patch is None:
            out = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
            return out, M

        # swapped_patch usually is same dimension as aligned frame; we will crop to bbox later
        return swapped_patch, M
    except Exception:
        out = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
        return out, np.eye(2,3,dtype=np.float32)


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
        except Exception:
            pass
    return None

# --------------------
# Swap flow with temporal stabilization
# --------------------

def swap_face_with_stabilization(source_face: Face, target_face: Face, frame: Frame) -> Frame:
    """High level: align+swap -> color correct patch -> warp previous patch with optical flow -> blend"""
    try:
        # get track if exists
        track = None
        tid = getattr(target_face, 'track_id', None)
        if tid is not None and tid in TRACKS:
            track = TRACKS[tid]

        swapped_full, M = robust_alignment_and_swap(source_face, target_face, frame, track)
        if swapped_full is None:
            return frame

        # compute bbox in aligned coords (use target_face.bbox)
        x1,y1,x2,y2 = map(int, target_face.bbox)
        # if swapper returned full aligned frame, crop corresponding region after inverse transform
        # To simplify: assume swapped_full is same size as frame (aligned frame) so crop directly
        try:
            patch = swapped_full[y1:y2, x1:x2].copy()
        except Exception:
            # fallback: use get_face_swapper paste_back True
            res = get_face_swapper().get(frame, target_face, source_face, paste_back=True)
            return res

        if patch is None or patch.size == 0:
            return frame

        # color correct using target region
        corrected = fast_color_correction(patch, frame, (x1,y1,x2,y2), track)

        # optical flow warp of previous patch
        prev_warped = None
        if track is not None and track.last_swapped_patch is not None and track.last_frame_gray is not None:
            try:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = compute_optical_flow(track.last_frame_gray, curr_gray)
                if flow is not None:
                    # crop flow to bbox
                    h_flow, w_flow = flow.shape[:2]
                    hx1 = max(0, x1); hy1 = max(0, y1); hx2 = min(w_flow, x2); hy2 = min(h_flow, y2)
                    if hx2 > hx1 and hy2 > hy1:
                        flow_crop = flow[hy1:hy2, hx1:hx2]
                        prev = track.last_swapped_patch
                        if prev.shape[:2] != flow_crop.shape[:2]:
                            prev = cv2.resize(prev, (flow_crop.shape[1], flow_crop.shape[0]))
                        prev_warped = warp_with_flow(prev, flow_crop)
            except Exception:
                prev_warped = None

        final_patch = corrected
        if prev_warped is not None:
            mag = np.linalg.norm(prev_warped.astype(np.float32) - corrected.astype(np.float32), axis=2).mean()
            mag = float(np.clip(mag, 0.0, FLOW_MAG_CLIP))
            blend_factor = np.clip((mag / FLOW_MAG_CLIP), 0.05, 0.95)
            blend_factor = max(1.0 - OPTICAL_FLOW_BLEND_MAX, blend_factor)
            final_patch = (blend_factor * corrected.astype(np.float32) + (1.0 - blend_factor) * prev_warped.astype(np.float32)).astype(np.uint8)

        # refine edges (simple) and blend
        final_patch = cv2.bilateralFilter(final_patch, 5, 15, 15)
        frame = simple_blend_patch(frame, final_patch, (x1,y1,x2,y2))

        # update track state
        if track is not None:
            track.last_swapped_patch = final_patch.copy()
            try:
                track.last_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                track.last_frame_gray = None

        return frame
    except Exception:
        # fallback to simple swap
        try:
            return get_face_swapper().get(frame, target_face, source_face, paste_back=True)
        except Exception:
            return frame

# --------------------
# Processors (compatible with Roop)
# --------------------

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    global FRAME_INDEX
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                mapping = match_tracks(many_faces)
                for idx, target_face in enumerate(many_faces):
                    tid = mapping.get(idx)
                    if tid is not None:
                        try:
                            setattr(target_face, 'track_id', tid)
                        except Exception:
                            pass
                    temp_frame = swap_face_with_stabilization(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                setattr(target_face, 'track_id', 0)
                if 0 not in TRACKS:
                    TRACKS[0] = TrackState(0, tuple(target_face.bbox), get_face_embedding(target_face))
                    TRACKS[0].last_seen = FRAME_INDEX
                temp_frame = swap_face_with_stabilization(source_face, target_face, temp_frame)

        FRAME_INDEX += 1
        return temp_frame
    except Exception:
        FRAME_INDEX += 1
        return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        for temp_frame_path in temp_frame_paths:
            try:
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is None:
                    continue
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
