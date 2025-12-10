from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os

import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path

import onnxruntime as ort


# ============================================================
# GLOBALS
# ============================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)
TEMPORAL_BUFFER: deque = deque(maxlen=5)

MIN_DET_SCORE = 0.30
OCCLUSION_THRESHOLD = 0.40
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

OCCLUDER_SESSION = None
OCCLUDER_INPUT_NAME = None


# ============================================================
# RAFT LARGE - SKHT_K_V2
# ============================================================

RAFT_MODEL = None
RAFT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAFT_PREV_FRAME_T = None
RAFT_LAST_FLOW = None
RAFT_LAST_FRAME_IDX = -1
RAFT_ALPHA = 0.7  # blending strength


def _get_raft_model():
    """Lazy init RAFT Large SKHT_K_V2."""
    global RAFT_MODEL

    if RAFT_MODEL is not None:
        return RAFT_MODEL

    try:
        weights = Raft_Large_Weights.C_T_SKHT_K_V2
        model = raft_large(weights=weights, progress=False)
        model = model.to(RAFT_DEVICE)
        model.eval()
        RAFT_MODEL = model
        print(f"✅ [face_analyser] RAFT Large (C_T_SKHT_K_V2) loaded on {RAFT_DEVICE}")
    except Exception as e:
        print(f"[face_analyser][RAFT] gagal load RAFT Large:", e)
        RAFT_MODEL = None

    return RAFT_MODEL


def _frame_to_raft_tensor(frame: Frame):
    """Convert BGR uint8 → [1,3,H,W] RGB float32 (0–1)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(RAFT_DEVICE)


@torch.no_grad()
def _get_raft_flow(frame: Frame, frame_idx: int):
    """
    Compute flow using RAFT Large (torchvision).
    torchvision RAFT returns ONLY ONE tensor.
    """
    global RAFT_PREV_FRAME_T, RAFT_LAST_FLOW, RAFT_LAST_FRAME_IDX

    model = _get_raft_model()
    if model is None:
        return None

    cur_t = _frame_to_raft_tensor(frame)

    if RAFT_PREV_FRAME_T is None:
        RAFT_PREV_FRAME_T = cur_t
        RAFT_LAST_FLOW = None
        RAFT_LAST_FRAME_IDX = frame_idx
        return None

    if RAFT_LAST_FRAME_IDX == frame_idx and RAFT_LAST_FLOW is not None:
        return RAFT_LAST_FLOW

    try:
        # ONLY ONE OUTPUT
        flow_up = model(RAFT_PREV_FRAME_T, cur_t)        # [1,2,Hf,Wf]
        flow_up = flow_up[0].detach().cpu()              # [2,Hf,Wf]
        flow_np = flow_up.permute(1, 2, 0).numpy()

        h, w = frame.shape[:2]
        if flow_np.shape[:2] != (h, w):
            fx = cv2.resize(flow_np[..., 0], (w, h))
            fy = cv2.resize(flow_np[..., 1], (w, h))
            flow_np = np.stack([fx, fy], axis=-1).astype(np.float32)

        RAFT_PREV_FRAME_T = cur_t
        RAFT_LAST_FLOW = flow_np
        RAFT_LAST_FRAME_IDX = frame_idx

        return flow_np

    except Exception as e:
        print("[face_analyser][RAFT] error inference:", e)
        RAFT_PREV_FRAME_T = cur_t
        RAFT_LAST_FLOW = None
        RAFT_LAST_FRAME_IDX = frame_idx
        return None


def raft_stabilize_bbox(face: Face, flow):
    if flow is None:
        return

    x1, y1, x2, y2 = map(float, face.bbox)
    h, w, _ = flow.shape
    cx = int(min(max((x1 + x2) / 2, 0), w - 1))
    cy = int(min(max((y1 + y2) / 2, 0), h - 1))

    dx, dy = flow[cy, cx]
    raft_bbox = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
    cur_bbox = np.array([x1, y1, x2, y2])

    face.bbox = (RAFT_ALPHA * raft_bbox + (1 - RAFT_ALPHA) * cur_bbox).astype(np.float32)


# ============================================================
# FACE ANALYSER INIT
# ============================================================

def get_face_analyser():
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            print("✅ [face_analyser] Using buffalo_l (pose + 2d106 + 3d68)")

    return FACE_ANALYSER


def clear_face_analyser():
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY, TEMPORAL_BUFFER
    global RAFT_MODEL, RAFT_PREV_FRAME_T, RAFT_LAST_FLOW, RAFT_LAST_FRAME_IDX

    FACE_ANALYSER = None
    FACE_TRACKING.clear()
    TRACKING_HISTORY.clear()
    TEMPORAL_BUFFER.clear()

    RAFT_MODEL = None
    RAFT_PREV_FRAME_T = None
    RAFT_LAST_FLOW = None
    RAFT_LAST_FRAME_IDX = -1


# ============================================================
# OCCLUSION DETECTION
# ============================================================

def _get_occluder_session():
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME

    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION

    model_path = resolve_relative_path("../models/occluder.onnx")
    if not os.path.exists(model_path):
        print("[face_analyser] Occluder model not found.")
        return None

    try:
        OCCLUDER_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception:
        OCCLUDER_SESSION = None

    return OCCLUDER_SESSION


def detect_occlusion(face: Face, frame=None):
    if getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD:
        return True

    if frame is None:
        return False

    session = _get_occluder_session()
    if session is None:
        return False

    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        crop = cv2.resize(crop, (224, 224)).astype("float32") / 255
        crop = crop.transpose(2, 0, 1)[None]

        out = session.run(None, {OCCLUDER_INPUT_NAME: crop})
        mask = out[0][0, 0]
        score = np.mean(mask > 0.5)

        return score > getattr(roop.globals, "occluder_threshold", 0.20)

    except:
        return False


# ============================================================
# FACE DETECTION / BASIC ACCESSORS
# ============================================================

def get_many_faces(frame):
    try:
        faces = get_face_analyser().get(frame)
        if faces:
            return [f for f in faces if getattr(f, "det_score", 0) >= MIN_DET_SCORE]
        return []
    except:
        return []


def get_one_face(frame, idx=0):
    faces = get_many_faces(frame)
    if not faces:
        return None
    return faces[idx] if idx < len(faces) else faces[-1]


# ============================================================
# TEMPORAL BUFFER + SMOOTHING
# ============================================================

def push_temporal_frame(faces: List[Face], frame_number: int):
    snapshot = []
    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        pose = getattr(f, "pose", None)
        snapshot.append({
            "bbox": np.array(f.bbox, np.float32),
            "embedding": emb.copy() if isinstance(emb, np.ndarray) else emb,
            "pose": pose,
        })
    TEMPORAL_BUFFER.append({"frame": frame_number, "faces": snapshot})


def smooth_bbox_for_face(face: Face):
    if not TEMPORAL_BUFFER:
        return

    x1, y1, x2, y2 = face.bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bboxes = []
    dists = []

    for entry in TEMPORAL_BUFFER:
        for f in entry["faces"]:
            bx1, by1, bx2, by2 = f["bbox"]
            fcx = (bx1 + bx2) / 2
            fcy = (by1 + by2) / 2
            d = np.hypot(fcx - cx, fcy - cy)
            dists.append(d)
            bboxes.append(f["bbox"])

    if not bboxes:
        return

    idx = np.argsort(np.array(dists))[:3]
    avg = np.mean([bboxes[i] for i in idx], axis=0)
    face.bbox = avg.astype(np.float32)


# ============================================================
# SMART TRACKING
# ============================================================

def smart_face_tracking(frame, frame_number):
    global FACE_TRACKING

    faces = get_many_faces(frame)
    if not faces:
        return []

    tracked = []

    with TRACK_LOCK:
        for face in faces:
            best_id = None
            best_sim = MIN_EMBED_SIMILARITY

            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = np.array([])

            for tid, tdata in FACE_TRACKING.items():
                if frame_number - tdata["last_seen"] > MAX_TRACK_GAP:
                    continue

                last_face = tdata["last_face"]
                last_emb = getattr(last_face, "normed_embedding", None)
                if last_emb is None:
                    continue

                sim = 1 - cosine(emb, last_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = tid

            if best_id:
                FACE_TRACKING[best_id]["last_face"] = face
                FACE_TRACKING[best_id]["last_seen"] = frame_number
            else:
                new_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[new_id] = {"last_face": face, "last_seen": frame_number}

            tracked.append(face)

        FACE_TRACKING = {
            k: v for k, v in FACE_TRACKING.items()
            if frame_number - v["last_seen"] <= MAX_TRACK_AGE
        }

        # temporal buffer
        push_temporal_frame(tracked, frame_number)

        # smooth by buffer
        for f in tracked:
            smooth_bbox_for_face(f)

        # RAFT stabilization
        flow = _get_raft_flow(frame, frame_number)
        if flow is not None:
            for f in tracked:
                raft_stabilize_bbox(f, flow)

    return tracked


# ============================================================
# SIMILAR FACE FOR REFERENCE
# ============================================================

def find_similar_face(frame, ref_face, use_tracking=True):
    if ref_face is None:
        return None

    faces = smart_face_tracking(frame, 0) if use_tracking else get_many_faces(frame)
    if not faces:
        return None

    ref_emb = ref_face.normed_embedding
    best = None
    best_dist = float("inf")
    th = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        dist = np.sum((f.normed_embedding - ref_emb) ** 2)
        if dist < best_dist and dist < th:
            best = f
            best_dist = dist

    return best
