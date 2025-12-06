# ================================================================
#   FACE ANALYSER — FINAL FULL VERSION (FIXED)
#   - Face Parsing ResNet34 (CelebAMaskHQ)
#   - No occluder.onnx
#   - FaceFusion-style occlusion
#   - Compatible with your face_swapper
# ================================================================

from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os
import onnxruntime as ort

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, conditional_download

# ================================================================
#   GLOBALS
# ================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

MIN_DET_SCORE = 0.30
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

# ===========================
#   FACE PARSING GLOBALS
# ===========================

PARSING_SESSION = None
PARSING_INPUT = None


# ================================================================
#   LOAD FACE PARSING MODEL (FIXED)
# ================================================================

def get_face_parsing_session():
    """
    Load ResNet34 Face Parsing model.
    Auto-download if missing.
    FIXED version (no rename_map).
    """
    global PARSING_SESSION, PARSING_INPUT
    if PARSING_SESSION is not None:
        return PARSING_SESSION

    model_dir = resolve_relative_path("../models")
    downloaded_path = os.path.join(model_dir, "resnet34.onnx")
    final_path = os.path.join(model_dir, "face_parsing_resnet34.onnx")

    # Step 1 — Download original name
    conditional_download(
        model_dir,
        [
            "https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.onnx"
        ]
    )

    # Step 2 — Rename manually
    if os.path.exists(downloaded_path) and not os.path.exists(final_path):
        os.rename(downloaded_path, final_path)

    # Step 3 — Load ONNX
    PARSING_SESSION = ort.InferenceSession(
        final_path,
        providers=roop.globals.execution_providers
    )
    PARSING_INPUT = PARSING_SESSION.get_inputs()[0].name

    print("✅ [face_parsing] Loaded ResNet34 parsing model")
    return PARSING_SESSION


# ================================================================
#   RUN FACE PARSING
# ================================================================

def run_face_parsing(crop: np.ndarray) -> np.ndarray:
    """
    Run face parsing on crop.
    Output is mask HxW (19 classes).
    """
    session = get_face_parsing_session()

    inp = cv2.resize(crop, (256, 256))
    inp = inp[:, :, ::-1] / 255.0
    inp = inp.astype(np.float32).transpose(2, 0, 1)[None]

    out = session.run(None, {PARSING_INPUT: inp})[0]  # (1,19,512,512)
    mask = np.argmax(out[0], axis=0).astype(np.uint8)

    # resize back to original crop size
    mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


# ================================================================
#   FACE ANALYSER INIT
# ================================================================

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            print("✅ [face_analyser] Using buffalo_l (pose + 106 landmarks)")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY
    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None


# ================================================================
#   FACE ACCESS
# ================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []

        faces = [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces

    except Exception:
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many = get_many_faces(frame)
    if many:
        try:
            return many[position]
        except IndexError:
            return many[-1]
    return None


def get_face_pose(face: Face):
    pose = getattr(face, "pose", None)
    if pose is None:
        return (0.0, 0.0, 0.0)
    return float(pose[0]), float(pose[1]), float(pose[2])


# ================================================================
#   TRACKING
# ================================================================

def calculate_motion_vector(prev_face, current_face):
    if prev_face is None or current_face is None:
        return 0.0

    p = prev_face.bbox
    c = current_face.bbox

    prev_center = np.array([(p[0]+p[2])/2, (p[1]+p[3])/2])
    curr_center = np.array([(c[0]+c[2])/2, (c[1]+c[3])/2])

    return float(np.linalg.norm(curr_center - prev_center))


def _compute_embedding_similarity(a, b):
    try:
        return 1.0 - float(cosine(a, b))
    except Exception:
        return 0.0


def smart_face_tracking(frame: Frame, frame_number: int):
    global FACE_TRACKING, TRACKING_HISTORY

    faces = get_many_faces(frame)
    if not faces:
        return None

    tracked = []

    with TRACK_LOCK:
        for face in faces:
            best_id = None
            max_sim = MIN_EMBED_SIMILARITY

            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = np.array([])

            # search best track
            for tid, data in list(FACE_TRACKING.items()):
                if frame_number - data["last_seen"] > MAX_TRACK_GAP:
                    continue

                last_f = data["last_face"]
                prev_emb = getattr(last_f, "normed_embedding", None)
                if prev_emb is None:
                    continue

                sim = _compute_embedding_similarity(emb, prev_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_id = tid

            # update / new track
            if best_id is not None:
                prev_f = FACE_TRACKING[best_id]["last_face"]
                motion = calculate_motion_vector(prev_f, face)

                FACE_TRACKING[best_id].update({
                    "last_face": face,
                    "last_seen": frame_number,
                    "motion": motion
                })
            else:
                best_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[best_id] = {
                    "last_face": face,
                    "last_seen": frame_number,
                    "motion": 0.0
                }

            # bbox smoothing
            if len(TRACKING_HISTORY) >= 2:
                prev = list(TRACKING_HISTORY)[-2:]
                smoothed = np.mean([f["bbox"] for f in prev], axis=0)
                face.bbox = smoothed

            TRACKING_HISTORY.append({"bbox": np.array(face.bbox, dtype=np.float32)})
            tracked.append(face)

        # cleanup
        FACE_TRACKING = {
            k: v for k, v in FACE_TRACKING.items()
            if frame_number - v["last_seen"] <= MAX_TRACK_AGE
        }

    return tracked


# ================================================================
#   OCCLUSION — FACE PARSING VISIBILITY
# ================================================================

def detect_occlusion(face: Face, frame: Frame) -> bool:
    """
    FaceFusion-style occlusion detection using parsing:
    - compute visibility ratio (skin + eyes + nose + mouth)
    - low visibility → occluded
    """
    if face is None or frame is None:
        return True

    x1, y1, x2, y2 = map(int, face.bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return True

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return True

    mask = run_face_parsing(crop)

    # core facial classes
    FACE_CLASSES = {1,2,3,4,5,6,7,8}

    total = crop.size // 3
    visible = np.count_nonzero(np.isin(mask, list(FACE_CLASSES)))
    visibility = visible / total

    VIS_THRESHOLD = 0.45  # recommended value

    return visibility < VIS_THRESHOLD


# ================================================================
#   FIND SIMILAR FACE
# ================================================================

def find_similar_face(frame: Frame, reference_face: Face, use_tracking=True):
    if reference_face is None:
        return None

    faces = smart_face_tracking(frame, 0) if use_tracking else get_many_faces(frame)
    if not faces:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_dist = float("inf")
    threshold = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue

        try:
            dist = np.sum(np.square(f.normed_embedding - ref_emb))
        except:
            continue

        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_face = f

    return best_face
