import cv2
import numpy as np
import onnxruntime as ort
import insightface

import roop.globals
from roop.occlusion_utils import (
    run_occluder_session,
    update_history_and_decide
)

# ============================================================
# MODEL INITIALIZATION
# ============================================================

_face_app = None
_occluder_session = None


def get_face_app():
    global _face_app
    if _face_app is None:
        _face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _face_app.prepare(ctx_id=0)
    return _face_app


def _get_occluder_session():
    global _occluder_session
    if _occluder_session is not None:
        return _occluder_session

    try:
        model_path = getattr(roop.globals, "occluder_onnx_path", None)
        if model_path is None:
            return None

        _occluder_session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    except Exception:
        _occluder_session = None

    return _occluder_session


# ============================================================
# FACE ANALYSIS BASIC FUNCTIONS (REQUIRED BY ROOP)
# ============================================================

def get_one_face(frame):
    """
    Returns a single face with highest detection score.
    """
    faces = get_face_app().get(frame)
    if not faces:
        return None
    faces = sorted(faces, key=lambda f: getattr(f, "det_score", 0), reverse=True)
    return faces[0]


def get_many_faces(frame):
    """
    Returns all detected faces.
    """
    return get_face_app().get(frame)


def find_similar_face(face, faces, distance_threshold=1.0):
    """
    Used by multi-face tracking. Compares embeddings.
    """
    if face is None or not faces:
        return None

    base_emb = face.embedding
    if base_emb is None:
        return None

    best = None
    best_dist = 999

    for f in faces:
        if f.embedding is None:
            continue
        dist = np.linalg.norm(base_emb - f.embedding)
        if dist < best_dist and dist < distance_threshold:
            best_dist = dist
            best = f

    return best


# ============================================================
# OCCLUSION DETECTION (FULL VERSION)
# ============================================================

def detect_occlusion(face, frame, track_id=None):
    """
    Returns:
        is_occluded: bool
        mask: HxW float32 mask (1 = occluded), None if no mask
    """

    det_score = getattr(face, "det_score", 1.0)
    base_flag = det_score < getattr(roop.globals, "occluder_det_threshold", 0.35)

    if frame is None:
        if track_id is not None:
            occ, _ = update_history_and_decide(track_id, None, 0.0, det_score)
            return occ, None
        return base_flag, None

    x1, y1, x2, y2 = map(int, face.bbox)

    H, W = frame.shape[:2]
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H))

    if x2 <= x1 or y2 <= y1:
        if track_id:
            occ, _ = update_history_and_decide(track_id, None, 0.0, det_score)
            return occ, None
        return base_flag, None

    crop = frame[y1:y2, x1:x2]

    sess = _get_occluder_session()
    if sess is None:
        return base_flag, None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    occl_ratio, mask = run_occluder_session(sess, crop_rgb)

    if mask is None:
        if track_id:
            occ, _ = update_history_and_decide(track_id, None, occl_ratio, det_score)
            return occ, None
        return base_flag, None

    full_mask = np.zeros((H, W), dtype=np.float32)
    full_mask[y1:y2, x1:x2] = mask

    if track_id is None:
        threshold = getattr(roop.globals, "occluder_ratio_threshold", 0.20)
        is_occ = (occl_ratio > threshold) or base_flag
        return is_occ, full_mask

    is_occ, smooth_mask = update_history_and_decide(
        track_id,
        full_mask,
        occl_ratio,
        det_score,
    )

    if smooth_mask is not None:
        return is_occ, smooth_mask

    return is_occ, full_mask
