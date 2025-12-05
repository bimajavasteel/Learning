import cv2
import numpy as np
import collections

TEMPORAL_WINDOW = 5
TEMPORAL_REQUIRED = 2
PIXEL_THRESHOLD = 0.5
FEATHER_KERNEL = (21, 21)

_occlusion_history = {}
_occlusion_last_mask = {}

def _ensure_history(track_id):
    if track_id not in _occlusion_history:
        _occlusion_history[track_id] = collections.deque(maxlen=TEMPORAL_WINDOW)
    return _occlusion_history[track_id]


def run_occluder_session(session, crop_rgb, input_name="input.1"):
    try:
        h, w = crop_rgb.shape[:2]
        inp = cv2.resize(crop_rgb, (224, 224))
        inp = inp.astype("float32") / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW

        outputs = session.run(None, {input_name: inp})
        pred = outputs[0]

        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred.squeeze()

        mask = cv2.resize(mask, (w, h))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        occl_ratio = float((mask > PIXEL_THRESHOLD).mean())

        return occl_ratio, mask.astype("float32")
    except:
        return 0.0, None


def update_history_and_decide(track_id, mask, occl_ratio, det_score=None, det_threshold=0.35):
    base_flag = (det_score is not None and det_score < det_threshold)

    if mask is None:
        hq = _ensure_history(track_id)
        hq.append(base_flag)
        decision = (sum(hq) >= TEMPORAL_REQUIRED)
        return decision, None

    bin_mask = (mask >= PIXEL_THRESHOLD).astype("uint8")
    pixel_ratio = float(bin_mask.mean())

    frame_flag = (pixel_ratio > 0.02) or base_flag

    hq = _ensure_history(track_id)
    hq.append(frame_flag)

    final = (sum(hq) >= TEMPORAL_REQUIRED)

    blur = cv2.GaussianBlur(mask, FEATHER_KERNEL, 0)
    blur = np.clip(blur, 0, 1).astype("float32")

    _occlusion_last_mask[track_id] = blur

    return final, blur


def composite_with_mask(original, swapped, mask):
    if mask is None:
        return swapped

    m3 = np.repeat(mask[:, :, None], 3, axis=2).astype("float32")

    out = original.astype("float32") * m3 + swapped.astype("float32") * (1 - m3)
    return np.clip(out, 0, 255).astype("uint8")
