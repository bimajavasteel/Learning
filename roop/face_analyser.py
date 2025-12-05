import cv2
import numpy as np
import onnxruntime as ort

import roop.globals
from roop.occlusion_utils import (
    run_occluder_session,
    update_history_and_decide
)

_occluder_session = None


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

    full_mask = np.zeros((H, W), dtype="float32")
    full_mask[y1:y2, x1:x2] = mask

    if track_id is None:
        threshold = getattr(roop.globals, "occluder_ratio_threshold", 0.20)
        return (occl_ratio > threshold) or base_flag, full_mask

    occ, smooth_mask = update_history_and_decide(
        track_id,
        full_mask,
        occl_ratio,
        det_score
    )

    if smooth_mask is None:
        return occ, full_mask

    return occ, smooth_mask
