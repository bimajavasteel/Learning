# face-swapper (bbox fix pro)
# Drop-in replacement for roop/processors/frame/face-swapper.py
# - Tight bbox from landmarks (if available)
# - Conservative pose padding (no overshoot)
# - Simple bbox smoothing
# - Seamless/elliptical paste to avoid rectangle edges
# - Defensive error handling (Kaggle-friendly)

from typing import Any, List, Callable, Optional
import cv2
import insightface
import threading
import numpy as np
import math
import traceback

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

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

# --- Tracking smoothing (local)
_SMOOTH_HISTORY = {}
_SM_HISTORY_LEN = 3  # number of boxes to average for smoothing


# -------------------------
#  Model initialization
# -------------------------
def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
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
    _SMOOTH_HISTORY.clear()


# -------------------------
#  Bounding-box helpers
# -------------------------
def _landmarks_candidates(face: Face) -> Optional[np.ndarray]:
    """
    Try several common attribute names for landmarks (defensive).
    Returns Nx2 float numpy array or None.
    """
    try:
        for attr in ('kps', 'kps_5', 'landmark_2d_106', 'landmark_3d_68', 'landmark'):
            pts = getattr(face, attr, None)
            if pts is None:
                continue
            arr = np.array(pts, dtype=np.float32)
            if arr.ndim == 1 and arr.size == 10:
                arr = arr.reshape(5, 2)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
    except Exception:
        return None
    return None


def _bbox_from_landmarks(landmarks: np.ndarray, margin: float = 0.18) -> np.ndarray:
    """
    Compute tight bbox from landmarks with relative margin (fraction of max dimension).
    Returns [x1,y1,x2,y2] in float.
    """
    x_min = float(np.min(landmarks[:, 0]))
    x_max = float(np.max(landmarks[:, 0]))
    y_min = float(np.min(landmarks[:, 1]))
    y_max = float(np.max(landmarks[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    pad = margin * max(w, h)

    nx1 = x_min - pad
    ny1 = y_min - pad
    nx2 = x_max + pad
    ny2 = y_max + pad

    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def _clamp_bbox(bbox: np.ndarray, frame_shape) -> np.ndarray:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        # fallback to full frame center box small
        cx, cy = w // 2, h // 2
        sx, sy = int(w * 0.2), int(h * 0.3)
        return np.array([cx - sx, cy - sy, cx + sx, cy + sy], dtype=np.float32)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _constrain_padding(original_bbox: np.ndarray, new_bbox: np.ndarray, max_rel: float = 0.35) -> np.ndarray:
    """
    Prevent new_bbox from growing too large relative to original_bbox.
    max_rel: maximum additional half-pad as fraction of original max(size)
    """
    ox1, oy1, ox2, oy2 = original_bbox
    ow = max(1.0, ox2 - ox1)
    oh = max(1.0, oy2 - oy1)
    ocenter = np.array([(ox1 + ox2)/2.0, (oy1 + oy2)/2.0])

    nx1, ny1, nx2, ny2 = new_bbox
    ncenter = np.array([(nx1 + nx2)/2.0, (ny1 + ny2)/2.0])
    nw = max(1.0, nx2 - nx1)
    nh = max(1.0, ny2 - ny1)

    max_w = ow * (1.0 + max_rel)
    max_h = oh * (1.0 + max_rel)

    # clamp width/height
    final_w = min(nw, max_w)
    final_h = min(nh, max_h)

    # center keep near original center but allow some shift
    center = (0.65 * ocenter) + (0.35 * ncenter)

    nx1f = center[0] - final_w / 2.0
    nx2f = center[0] + final_w / 2.0
    ny1f = center[1] - final_h / 2.0
    ny2f = center[1] + final_h / 2.0

    return np.array([nx1f, ny1f, nx2f, ny2f], dtype=np.float32)


def _smooth_bbox(track_id: int, bbox: np.ndarray) -> np.ndarray:
    """
    Simple history average smoothing per track id.
    """
    if track_id not in _SMOOTH_HISTORY:
        _SMOOTH_HISTORY[track_id] = []
    hist = _SMOOTH_HISTORY[track_id]
    hist.append(bbox.astype(np.float32))
    if len(hist) > _SM_HISTORY_LEN:
        hist.pop(0)
    arr = np.array(hist, dtype=np.float32)
    avg = np.mean(arr, axis=0)
    return avg.astype(np.float32)


def _compute_safe_bbox(face: Face, frame_shape, track_id: Optional[int] = None) -> np.ndarray:
    """
    Compute a conservative, safe bbox to pass to inswapper:
    - try landmarks -> tight bbox w/ small margin
    - else use face.bbox but clamp padding from pose-aware adjustment
    - apply clamp to frame and constrain max expansion
    - apply smoothing by track_id
    """
    try:
        orig_bbox = np.array(face.bbox, dtype=np.float32)
    except Exception:
        # fallback to full frame small box
        h, w = frame_shape[:2]
        return np.array([w*0.25, h*0.25, w*0.75, h*0.75], dtype=np.float32)

    landmarks = _landmarks_candidates(face)
    if landmarks is not None and landmarks.shape[0] >= 5:
        try:
            lb = _bbox_from_landmarks(landmarks, margin=0.18)
            lb = _clamp_bbox(lb, frame_shape)
            lb = _constrain_padding(orig_bbox, lb, max_rel=0.35)
            if track_id is not None:
                lb = _smooth_bbox(track_id, lb)
            return lb
        except Exception:
            pass

    # fallback: use original bbox but shrink excessive padding from pose adapt
    # compute conservative shrink: remove extremely large padding
    ox1, oy1, ox2, oy2 = orig_bbox
    ow = max(1.0, ox2 - ox1)
    oh = max(1.0, oy2 - oy1)
    # ensure bbox not smaller than 0.6 * original in either dim
    min_w = ow * 0.6
    min_h = oh * 0.6
    cx = (ox1 + ox2) / 2.0
    cy = (oy1 + oy2) / 2.0
    nx1 = cx - min_w / 2.0
    ny1 = cy - min_h / 2.0
    nx2 = cx + min_w / 2.0
    ny2 = cy + min_h / 2.0
    nb = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    nb = _clamp_bbox(nb, frame_shape)
    if track_id is not None:
        nb = _smooth_bbox(track_id, nb)
    return nb


# -------------------------
#  Mask & paste helpers
# -------------------------
def _mask_from_landmarks_or_ellipse(bbox: np.ndarray, landmarks: Optional[np.ndarray], shape) -> np.ndarray:
    """
    Return 3-channel mask in bbox coordinates (0..1 float) to blend pasted face.
    Prefer convex hull of landmarks if available, else elliptical mask.
    """
    bh = int(round(bbox[3] - bbox[1]))
    bw = int(round(bbox[2] - bbox[0]))
    if bh <= 0 or bw <= 0:
        return np.ones((shape[0], shape[1], 3), dtype=np.float32)

    mask_small = np.zeros((bh, bw), dtype=np.uint8)
    if landmarks is not None and landmarks.shape[0] >= 5:
        # shift landmarks to bbox coord
        shift_x = bbox[0]
        shift_y = bbox[1]
        pts = np.round(landmarks - np.array([shift_x, shift_y])).astype(np.int32)
        # limit pts in box
        pts[:, 0] = np.clip(pts[:, 0], 0, bw - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, bh - 1)
        try:
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask_small, hull, 255)
        except Exception:
            mask_small = np.zeros((bh, bw), dtype=np.uint8)
    if mask_small.sum() == 0:
        # fallback ellipse
        center = (bw//2, bh//2)
        axes = (int(bw*0.48), int(bh*0.48))
        cv2.ellipse(mask_small, center, axes, 0, 0, 360, 255, -1)

    # feather mask
    blur_k = int(max(3, min(bw, bh) * 0.08))
    if blur_k % 2 == 0:
        blur_k += 1
    mask_small = cv2.GaussianBlur(mask_small.astype(np.float32)/255.0, (blur_k, blur_k), 0)
    mask3 = np.dstack([mask_small]*3).astype(np.float32)
    # place into full frame mask shape
    full_mask = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    x1, y1, x2, y2 = map(int, bbox)
    x2 = min(full_mask.shape[1], x2)
    y2 = min(full_mask.shape[0], y2)
    full_mask[y1:y2, x1:x2] = mask3[:(y2-y1), :(x2-x1)]
    return full_mask


def _blend_paste(swapped_frame: np.ndarray, original_frame: np.ndarray, bbox: np.ndarray, landmarks: Optional[np.ndarray]) -> np.ndarray:
    """
    Blend swapped_frame onto original_frame using soft mask in bbox area.
    swapped_frame is expected to be full-frame (inswapper may paste already) — we extract bbox region.
    """
    out = original_frame.copy()
    x1, y1, x2, y2 = map(int, bbox)
    x2 = min(x2, swapped_frame.shape[1])
    y2 = min(y2, swapped_frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return swapped_frame

    region_src = swapped_frame[y1:y2, x1:x2].astype(np.float32)
    region_dst = original_frame[y1:y2, x1:x2].astype(np.float32)
    if region_src.size == 0 or region_dst.size == 0:
        return swapped_frame

    mask = _mask_from_landmarks_or_ellipse(bbox, landmarks, swapped_frame.shape)
    mask_region = mask[y1:y2, x1:x2]
    # weighted blend
    blended = (region_src * mask_region + region_dst * (1.0 - mask_region)).astype(np.uint8)
    out[y1:y2, x1:x2] = blended
    return out


# -------------------------
#  CORE SWAP
# -------------------------
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame, track_id: Optional[int] = None) -> Frame:
    """
    Swap using safe bbox computed from landmarks or original bbox.
    After inswapper.get we perform soft mask blending to avoid visible rectangle edges.
    """
    if source_face is None or target_face is None:
        return temp_frame

    frame_shape = temp_frame.shape

    # Compute a safe bbox to hand to inswapper (do NOT over-expand)
    try:
        safe_bbox = _compute_safe_bbox(target_face, frame_shape, track_id)
        # assign back to target_face in-place (inswapper reads face.bbox)
        target_face.bbox = safe_bbox.tolist()
    except Exception:
        # fallback: leave face.bbox unchanged
        try:
            target_face.bbox = np.array(target_face.bbox, dtype=np.float32).tolist()
        except Exception:
            pass

    # call inswapper
    try:
        swapped_full = get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )
    except Exception:
        # if inswapper fails, return original
        traceback.print_exc()
        return temp_frame

    # Now blend swapped area softly using mask (use landmarks if available)
    try:
        landmarks = _landmarks_candidates(target_face)
        result = _blend_paste(swapped_full, temp_frame, _clamp_bbox(np.array(target_face.bbox, dtype=np.float32), frame_shape), landmarks)
        return result
    except Exception:
        return swapped_full


# -------------------------
#  Selection helper
# -------------------------
def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Face | None:
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')

    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue
        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue
        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f

    return best_face


# -------------------------
#  Frame processing
# -------------------------
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame

    # many faces mode
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame

        for idx, tgt in enumerate(faces):
            # track id fallback: use idx
            track_id = getattr(tgt, 'track_id', idx)
            if detect_occlusion(tgt, temp_frame):
                continue
            temp_frame = swap_face(source_face, tgt, temp_frame, track_id)
        return temp_frame

    # single face mode
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)
    if not tracked_faces:
        return temp_frame

    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)
    if best_target is None:
        best_target = valid_faces[0]

    # create a stable track_id if possible
    track_id = getattr(best_target, 'track_id', 0)
    out = swap_face(source_face, best_target, temp_frame, track_id)
    return out


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=temp_frame, frame_number=idx)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)
    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)

    result = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=target_frame, frame_number=0)
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

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
