import cv2
import numpy as np
import insightface
import onnxruntime as ort
import roop.globals

from roop.occlusion_utils import (
    run_occluder_session,
    update_history_and_decide
)

# =========================================================
# INITIALIZATION
# =========================================================

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


# =========================================================
# SMART FACE TRACKING (WAJIB ADA)
# =========================================================

_track_memory = {}   # track_id -> face object

def smart_face_tracking(frame, frame_number=0):
    """
    Signature HARUS seperti ini untuk kompatibel
    dengan face_swapper kamu.
    """
    faces = get_face_app().get(frame)
    if not faces:
        return []

    global _track_memory
    updated = []

    for f in faces:
        emb = getattr(f, "embedding", None)
        if emb is None:
            f.track_id = id(f)
            updated.append(f)
            continue

        best_id = None
        best_dist = 999

        for tid, tf in _track_memory.items():
            temb = getattr(tf, "embedding", None)
            if temb is None:
                continue
            d = np.linalg.norm(emb - temb)
            if d < best_dist and d < 1.0:
                best_dist = d
                best_id = tid

        if best_id is None:
            best_id = id(f)

        f.track_id = best_id
        _track_memory[best_id] = f
        updated.append(f)

    return updated


# =========================================================
# BASIC FACE FUNCTIONS (WAJIB ADA)
# =========================================================

def get_one_face(frame):
    faces = get_face_app().get(frame)
    if not faces:
        return None
    faces = sorted(faces, key=lambda f: getattr(f, "det_score", 0), reverse=True)
    return faces[0]


def get_many_faces(frame):
    return get_face_app().get(frame)


def find_similar_face(face, faces, threshold=1.0):
    if face is None or not faces:
        return None
    emb = getattr(face, "embedding", None)
    if emb is None:
        return None

    best = None
    best_dist = 999

    for f in faces:
        e = getattr(f, "embedding", None)
        if e is None:
            continue
        d = np.linalg.norm(emb - e)
        if d < best_dist and d < threshold:
            best = f
            best_dist = d

    return best


# =========================================================
# OCCLUDER SESSION
# =========================================================

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
        return _occluder_session

    except:
        _occluder_session = None
        return None


# =========================================================
# DETECT OCCLUSION (MASK READY)
# =========================================================

def detect_occlusion(face, frame, track_id=None):
    det_score = getattr(face, "det_score", 1.0)
    base_flag = det_score < getattr(roop.globals, "occluder_det_threshold", 0.35)

    if frame is None:
        if track_id:
            occ, _ = update_history_and_decide(track_id, None, 0.0, det_score)
            return occ, None
        return base_flag, None

    x1, y1, x2, y2 = map(int, face.bbox)
    H, W = frame.shape[:2]
    x1 = max(0, min(x1, W-1))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H-1))
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
        return (occl_ratio > threshold) or base_flag, full_mask

    occ, smooth_mask = update_history_and_decide(
        track_id,
        full_mask,
        occl_ratio,
        det_score,
    )

    if smooth_mask is not None:
        return occ, smooth_mask

    return occ, full_mask
