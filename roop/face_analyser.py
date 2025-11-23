import os
import threading
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import insightface
import onnxruntime as ort

import roop.globals
from roop.typing import Frame, Face

T4_GPU_MEM_LIMIT = int(os.getenv('ROOP_GPU_MEM_LIMIT', 12 * 1024**3))
BATCH_SIZE = int(os.getenv('ROOP_BATCH_SIZE', 8))
DET_SIZE = (1280, 1280)
PARTIAL_EMB_MASK = np.concatenate([np.ones(256, dtype=np.float32), np.zeros(256, dtype=np.float32)])
SIMILAR_DISTANCE_THRESHOLD = roop.globals.similar_face_distance

_FACE_ANALYSER: Optional[Any] = None
_THREAD_LOCK = threading.Lock()


def _build_providers():
    try:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'gpu_mem_limit': T4_GPU_MEM_LIMIT,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'do_copy_in_default_stream': True
            }),
            'CPUExecutionProvider'
        ]
    except Exception:
        providers = ['CPUExecutionProvider']
    return providers


def get_face_analyser() -> Any:
    global _FACE_ANALYSER
    if _FACE_ANALYSER is not None:
        return _FACE_ANALYSER

    with _THREAD_LOCK:
        if _FACE_ANALYSER is None:
            providers = _build_providers()

            _FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='antelopev2',
                providers=providers,
                allowed_modules=['detection', 'recognition'],
                root="/kaggle/working/Learning/roop/models"
            )

            _FACE_ANALYSER.prepare(
                ctx_id=0,
                det_size=DET_SIZE
            )

            try:
                if hasattr(_FACE_ANALYSER.app, 'landmark_model'):
                    _FACE_ANALYSER.app.landmark_model.use_smoothing = True
            except Exception:
                pass

    return _FACE_ANALYSER


@lru_cache(maxsize=1024)
def _cached_get_faces(frame_bytes: bytes) -> Tuple:
    analyser = get_face_analyser()
    return tuple(analyser.get(np.frombuffer(frame_bytes, dtype=np.uint8)))


def _frame_key(frame: Frame) -> bytes:
    try:
        h = cv2.resize(frame, (32, 32)).tobytes()
    except Exception:
        h = frame.tobytes()
    return h


def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    if frame is None or getattr(frame, 'size', 0) == 0:
        return None

    try:
        key = _frame_key(frame)
        faces = _cached_get_faces(key)
        return list(faces)
    except Exception:
        try:
            return get_face_analyser().get(frame)
        except Exception:
            return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    faces = get_many_faces(frame)
    if not faces:
        return None
    if position < len(faces):
        return faces[position]
    return faces[-1]


_prev_embeddings = {}


def _smooth_embedding(face_id: int, emb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    prev = _prev_embeddings.get(face_id)
    if prev is None:
        _prev_embeddings[face_id] = emb.copy()
        return emb
    sm = alpha * emb + (1 - alpha) * prev
    _prev_embeddings[face_id] = sm
    return sm


def partial_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(((a - b) * PARTIAL_EMB_MASK) ** 2))


def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    if reference_face is None:
        return None

    faces = get_many_faces(frame)
    if not faces:
        return None

    ref_emb = reference_face.normed_embedding
    best = None
    best_dist = float('inf')

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue
        emb = f.normed_embedding
        face_id = int(np.sum(f.bbox))
        emb = _smooth_embedding(face_id, emb)
        dist = partial_distance(emb, ref_emb)
        if dist < best_dist:
            best_dist = dist
            best = f

    if best_dist < SIMILAR_DISTANCE_THRESHOLD:
        return best
    return None


def clear_face_analyser() -> None:
    global _FACE_ANALYSER
    _FACE_ANALYSER = None
    _cached_get_faces.cache_clear()
