import cv2
import threading
import numpy as np
import onnxruntime as ort
from typing import Any, List, Callable, Tuple, Optional

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# -----------------------
# Stable GPEN Face Enhancer
# - GPEN ONNX (auto-download)
# - bbox smoothing per-identity (via embedding if available)
# - color match GPEN output to original crop
# - feathered mask + cv2.MIXED_CLONE for stable blending
# - small padding to reduce halo
# -----------------------

FACE_ENHANCER: Optional[ort.InferenceSession] = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'
MODEL_URL = 'https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GPEN-BFR-512.onnx'
MODEL_NAME = 'GPEN-BFR-512.onnx'

# simple bbox cache keyed by embedding signature or center coordinate
BBOX_CACHE: dict[str, Tuple[float, float, float, float]] = {}
SMOOTH_ALPHA = 0.65  # smoothing factor (higher = more inertia)


def _sign_from_face(face: Face) -> str:
    """Create a stable-ish signature for a face.
    Prefer normalized embedding if available, else use rounded center coords.
    """
    try:
        emb = getattr(face, 'normed_embedding', None)
        if emb is not None and len(emb) >= 8:
            # use first 8 dims rounded to 3 decimals
            key = ','.join([f'{float(x):.3f}' for x in emb[:8]])
            return f'emb:{key}'
    except Exception:
        pass

    try:
        bbox = face['bbox'] if isinstance(face, dict) else getattr(face, 'bbox', None)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return f'center:{cx}:{cy}'
    except Exception:
        pass

    return 'unknown'


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def get_face_enhancer() -> ort.InferenceSession:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_dir = resolve_relative_path('../models')
            model_path = resolve_relative_path(f"../models/{MODEL_NAME}")
            conditional_download(model_dir, [MODEL_URL])
            FACE_ENHANCER = ort.InferenceSession(model_path, providers=roop.globals.execution_providers)
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [MODEL_URL])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def _prepare_input(img: np.ndarray) -> np.ndarray:
    inp = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    inp = inp.astype(np.float32) / 127.5 - 1.0
    inp = inp.transpose(2, 0, 1)[None, ...]
    return inp


def _postprocess_output(out: np.ndarray, target_size: tuple) -> np.ndarray:
    out = np.clip(out, -1.0, 1.0)
    out = (out + 1.0) * 127.5
    out = out.transpose(1, 2, 0).astype(np.uint8)
    out = cv2.resize(out, target_size, interpolation=cv2.INTER_LINEAR)
    return out


def _color_match(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Match mean brightness of dst (GPEN out) to src (original crop).
    Uses simple gain on V channel in HSV for stable results.
    """
    try:
        src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV).astype(np.float32)
        dst_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV).astype(np.float32)
        mean_src = np.mean(src_hsv[..., 2])
        mean_dst = np.mean(dst_hsv[..., 2])
        if mean_dst > 1e-3:
            gain = mean_src / mean_dst
            dst_hsv[..., 2] = np.clip(dst_hsv[..., 2] * gain, 0, 255)
            dst = cv2.cvtColor(dst_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    except Exception:
        pass
    return dst


def _make_feathered_mask(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # circle mask centered -> good for faces
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, 255, -1)
    # large gaussian blur to make feather
    k = max(31, (min(w, h) // 8) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _smooth_bbox(sig: str, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if sig in BBOX_CACHE:
        px1, py1, px2, py2 = BBOX_CACHE[sig]
        nx1 = int(SMOOTH_ALPHA * px1 + (1 - SMOOTH_ALPHA) * x1)
        ny1 = int(SMOOTH_ALPHA * py1 + (1 - SMOOTH_ALPHA) * y1)
        nx2 = int(SMOOTH_ALPHA * px2 + (1 - SMOOTH_ALPHA) * x2)
        ny2 = int(SMOOTH_ALPHA * py2 + (1 - SMOOTH_ALPHA) * y2)
    else:
        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    BBOX_CACHE[sig] = (nx1, ny1, nx2, ny2)
    return nx1, ny1, nx2, ny2


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    try:
        bbox = target_face['bbox'] if isinstance(target_face, dict) else getattr(target_face, 'bbox', None)
    except Exception:
        bbox = None
    if bbox is None:
        return temp_frame

    # raw bbox
    x1, y1, x2, y2 = map(int, bbox)

    # small padding to avoid large color bleed
    padding_x = max(1, int((x2 - x1) * 0.06))
    padding_y = max(1, int((y2 - y1) * 0.06))

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(temp_frame.shape[1], x2 + padding_x)
    y2 = min(temp_frame.shape[0], y2 + padding_y)

    # smoothing by signature
    sig = _sign_from_face(target_face)
    x1, y1, x2, y2 = _smooth_bbox(sig, (x1, y1, x2, y2))

    crop = temp_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return temp_frame

    inp = _prepare_input(crop)
    session = get_face_enhancer()
    input_name = session.get_inputs()[0].name

    with THREAD_SEMAPHORE:
        outputs = session.run(None, {input_name: inp})

    out = outputs[0][0]
    target_w = x2 - x1
    target_h = y2 - y1
    out = _postprocess_output(out, (target_w, target_h))

    # color match to original crop
    out = _color_match(crop, out)

    # feathered mask
    mask = _make_feathered_mask((out.shape[0], out.shape[1]))

    # blending with MIXED_CLONE (more stable than NORMAL_CLONE for videos)
    try:
        center = (int(x1 + out.shape[1] // 2), int(y1 + out.shape[0] // 2))
        temp_frame = cv2.seamlessClone(out, temp_frame, mask, center, cv2.MIXED_CLONE)
    except Exception:
        # fallback to alpha blend with feather mask
        try:
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            inv = 1.0 - alpha
            temp_frame[y1:y2, x1:x2] = (alpha * out + inv * temp_frame[y1:y2, x1:x2]).astype(np.uint8)
        except Exception:
            temp_frame[y1:y2, x1:x2] = out

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
