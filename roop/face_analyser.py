# (FILE 1) – face_analyser_3D.py (AUTO DOWNLOAD VERSION)
# ------------------------------------------------------

from typing import Any, Optional, List
import threading
from scipy.spatial.distance import cosine
from collections import deque
import numpy as np
import cv2
import onnxruntime as ort

import insightface
import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, conditional_download

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

FACE_TRACKING = {}
TRACKING_HISTORY = deque(maxlen=30)

MIN_DET_SCORE = 0.30
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIM = 0.70

OCCLUDER = None


# ------------------------------------------------------
# AUTO DOWNLOAD OCCLUDER
# ------------------------------------------------------
def pre_check_occluder():
    """
    Auto download occluder.onnx jika belum ada.
    """
    download_directory = resolve_relative_path("../models")
    conditional_download(download_directory, [
        "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/occluder.onnx"
    ])
    return True


# ------------------------------------------------------
# LOAD OCCLUDER.ONNX
# ------------------------------------------------------
def load_occluder():
    global OCCLUDER
    if OCCLUDER is None:
        pre_check_occluder()
        path = resolve_relative_path("../models/occluder.onnx")
        OCCLUDER = ort.InferenceSession(
            path,
            providers=roop.globals.execution_providers
        )
    return OCCLUDER


def occlusion_mask(face_crop: np.ndarray):
    occl = load_occluder()
    img = cv2.resize(face_crop, (256,256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None]

    out = occl.run(None, {"input": img})[0]
    mask = out.squeeze()
    mask = (mask > 0.3).astype(np.uint8)
    return cv2.resize(mask, (face_crop.shape[1], face_crop.shape[0]))


# ------------------------------------------------------
# HEADPOSE FROM LANDMARKS (Yaw / Pitch)
# ------------------------------------------------------
def headpose_from_landmarks(lm):
    left_eye = lm[36]
    right_eye = lm[45]
    nose = lm[30]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    yaw = np.degrees(np.arctan2(dy, dx))

    nose_up = nose[1] - ((left_eye[1] + right_eye[1]) / 2)
    pitch = np.degrees(np.arctan2(nose_up, abs(dx)))

    return yaw, pitch


# ------------------------------------------------------
# INIT FACE ANALYSER (BUFFALO_L)
# ------------------------------------------------------
def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            pre_check_occluder()  # auto download occluder
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640,640))
    return FACE_ANALYSER


# ------------------------------------------------------
# FACE DETECTION
# ------------------------------------------------------
def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []
        return [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]
    except:
        return None


def get_one_face(frame: Frame, pos=0):
    f = get_many_faces(frame)
    if f:
        return f[pos] if pos < len(f) else f[-1]
    return None


# ------------------------------------------------------
# SIMILARITY
# ------------------------------------------------------
def _sim(e1, e2):
    try:
        return 1.0 - float(cosine(e1, e2))
    except:
        return 0.0


# ------------------------------------------------------
# SMART FACE TRACKING
# ------------------------------------------------------
def smart_face_tracking(frame: Frame, idx: int):
    global FACE_TRACKING, TRACKING_HISTORY

    faces = get_many_faces(frame)
    if not faces:
        return None

    output = []
    with TRACK_LOCK:
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue

            best = None
            best_sim = MIN_EMBED_SIM

            for tid, t in list(FACE_TRACKING.items()):
                if idx - t["last"] > MAX_TRACK_GAP:
                    continue

                tface = t["face"]
                tsim = _sim(emb, tface.normed_embedding)
                if tsim > best_sim:
                    best_sim = tsim
                    best = tid

            if best is None:
                new_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[new_id] = {"face": f, "last": idx}
            else:
                FACE_TRACKING[best] = {"face": f, "last": idx}

            output.append(f)

        FACE_TRACKING = {
            tid: t for tid, t in FACE_TRACKING.items()
            if idx - t["last"] <= MAX_TRACK_AGE
        }

    return output
