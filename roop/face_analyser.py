from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os

import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as VF

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path

# optional: kalau kamu punya occluder.onnx
import onnxruntime as ort

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking (penting untuk multi-thread)

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# 🔥 Temporal Frame Buffer: simpan beberapa frame terakhir
TEMPORAL_BUFFER: deque = deque(maxlen=5)  # bisa ubah jadi 3–7 sesuai kebutuhan

# Threshold / hyper-parameter default (boleh kamu tuning)
MIN_DET_SCORE = 0.30        # min score agar wajah dianggap valid (untuk get_many_faces)

# fallback occlusion kalau occluder.onnx tidak ada
OCCLUSION_THRESHOLD = 0.40  # det_score < ini dianggap occluded

MAX_TRACK_GAP = 10          # frame: kalau lebih lama dari ini → track di-skip saat matching
MAX_TRACK_AGE = 15          # frame: track dihapus bila tidak terlihat selama ini
MIN_EMBED_SIMILARITY = 0.70 # cosine similarity minimal untuk dianggap match (0–1)

# Occluder ONNX (opsional)
OCCLUSION_SESSION: Optional[ort.InferenceSession] = None
OCCLUSION_INPUT_NAME: Optional[str] = None

# =====================================================================
#  RAFT OPTICAL FLOW (GPU / CUDA)
# =====================================================================

RAFT_MODEL: Optional[torch.nn.Module] = None
RAFT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
RAFT_LOCK = threading.Lock()

RAFT_PREV_FRAME_T: Optional[torch.Tensor] = None  # [1,3,H,W] RGB, dinormalisasi
RAFT_LAST_FLOW: Optional[np.ndarray] = None       # (H,W,2) numpy
RAFT_LAST_FRAME_IDX: int = -1

# seberapa kuat pengaruh flow vs bbox biasa
RAFT_ALPHA: float = 0.7   # 0 = matiin RAFT, 1 = full RAFT


def _get_raft_model() -> Optional[torch.nn.Module]:
    """
    Lazy init RAFT (torchvision.models.optical_flow.raft_small)
    Jalan di CUDA kalau tersedia.
    """
    global RAFT_MODEL
    with RAFT_LOCK:
        if RAFT_MODEL is not None:
            return RAFT_MODEL

        try:
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)
            model = model.to(RAFT_DEVICE)
            model.eval()
            RAFT_MODEL = model
            print(f"✅ [face_analyser] RAFT Small loaded on {RAFT_DEVICE}")
        except Exception as e:
            print(f"[face_analyser][RAFT] gagal init RAFT: {e}")
            RAFT_MODEL = None
        return RAFT_MODEL


def _frame_to_raft_tensor(frame: Frame) -> torch.Tensor:
    """
    BGR uint8 -> Tensor [1,3,H,W] RGB float (0–1) untuk RAFT.
    """
    # ke RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
    return t.unsqueeze(0).to(RAFT_DEVICE)


@torch.no_grad()
def _get_raft_flow(frame: Frame, frame_idx: int) -> Optional[np.ndarray]:
    """
    Hitung optical flow dense pakai RAFT antara frame sebelumnya -> frame ini.
    Output: numpy (H,W,2), diresample ke resolusi asli frame.
    """
    global RAFT_PREV_FRAME_T, RAFT_LAST_FLOW, RAFT_LAST_FRAME_IDX

    model = _get_raft_model()
    if model is None:
        return None
    if frame is None:
        return None

    cur_t = _frame_to_raft_tensor(frame)  # [1,3,H,W]
    if RAFT_PREV_FRAME_T is None:
        RAFT_PREV_FRAME_T = cur_t
        RAFT_LAST_FLOW = None
        RAFT_LAST_FRAME_IDX = frame_idx
        return None

    # kalau sudah dihitung untuk frame ini, pakai cache
    if RAFT_LAST_FRAME_IDX == frame_idx and RAFT_LAST_FLOW is not None:
        return RAFT_LAST_FLOW

    # RAFT lebih stabil kalau input punya resolusi tetap (misal 256x256 / 320x480),
    # tapi di sini kita langsung pakai ukuran asli (torchvision RAFT sudah handle).
    try:
        # model mengembalikan list of flows; ambil hasil terakhir (resolusi tertinggi)
        flow_list = model(RAFT_PREV_FRAME_T, cur_t)
        # torchvision RAFT: output tuple (flow_low, flow_up) atau list; kita ambil yang terakhir.
        if isinstance(flow_list, (list, tuple)):
            flow = flow_list[-1]  # [1,2,Hf,Wf]
        else:
            flow = flow_list

        flow = flow[0].detach().cpu()  # [2,Hf,Wf]
        # [2,H,W] -> [H,W,2]
        flow_np = flow.permute(1, 2, 0).numpy().astype(np.float32)

        # kalau resolusi RAFT != resolusi frame, resize pakai bilinear
        h, w = frame.shape[:2]
        if flow_np.shape[0] != h or flow_np.shape[1] != w:
            # cv2.resize butuh (W,H)
            fx = cv2.resize(flow_np[..., 0], (w, h), interpolation=cv2.INTER_LINEAR)
            fy = cv2.resize(flow_np[..., 1], (w, h), interpolation=cv2.INTER_LINEAR)
            flow_np = np.stack([fx, fy], axis=-1).astype(np.float32)

        RAFT_PREV_FRAME_T = cur_t
        RAFT_LAST_FLOW = flow_np
        RAFT_LAST_FRAME_IDX = frame_idx
        return flow_np
    except Exception as e:
        print(f"[face_analyser][RAFT] error inference: {e}")
        RAFT_PREV_FRAME_T = cur_t
        RAFT_LAST_FLOW = None
        RAFT_LAST_FRAME_IDX = frame_idx
        return None


def raft_stabilize_bbox(face: Face, flow: np.ndarray) -> None:
    """
    Geser bbox mengikuti vektor flow di titik tengah,
    lalu blend dengan bbox sebelumnya (RAFT_ALPHA).
    """
    if flow is None:
        return
    try:
        x1, y1, x2, y2 = map(float, face.bbox)
    except Exception:
        return

    h, w, _ = flow.shape
    cx = int(max(0, min((x1 + x2) / 2.0, w - 1)))
    cy = int(max(0, min((y1 + y2) / 2.0, h - 1)))

    try:
        dx, dy = flow[cy, cx]  # (fx, fy)
    except Exception:
        return

    flow_bbox = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy], dtype=np.float32)
    cur_bbox = np.array(face.bbox, dtype=np.float32)

    alpha = float(RAFT_ALPHA)
    alpha = max(0.0, min(1.0, alpha))

    blended = alpha * flow_bbox + (1.0 - alpha) * cur_bbox
    face.bbox = blended.astype(np.float32)


# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_face_analyser() -> Any:
    """
    Lazy init insightface FaceAnalysis (buffalo_l).
    Sekali saja per proses, thread-safe.
    """
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


def clear_face_analyser() -> None:
    """
    Reset analyser & tracking state.
    Dipanggil saat post_process / cleanup.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY, TEMPORAL_BUFFER
    global RAFT_PREV_FRAME_T, RAFT_LAST_FLOW, RAFT_LAST_FRAME_IDX, RAFT_MODEL

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
        TEMPORAL_BUFFER.clear()

    RAFT_PREV_FRAME_T = None
    RAFT_LAST_FLOW = None
    RAFT_LAST_FRAME_IDX = -1
    RAFT_MODEL = None

    with THREAD_LOCK:
        FACE_ANALYSER = None


# =====================================================================
#  OCCLUDER ONNX (opsional)
# =====================================================================

def _get_occluder_session() -> Optional[ort.InferenceSession]:
    """
    Lazy init occluder.onnx.
    Kalau file tidak ada / gagal load → return None dan sistem fallback ke det_score.
    """
    global OCCLUSION_SESSION, OCCLUSION_INPUT_NAME

    if OCCLUSION_SESSION is not None:
        return OCCLUSION_SESSION

    model_rel = getattr(roop.globals, "occluder_model_path", "../models/occluder.onnx")
    model_path = resolve_relative_path(model_rel)

    if not os.path.exists(model_path):
        print(f"[face_analyser] occluder model not found at {model_path}, fallback ke det_score.")
        return None

    try:
        OCCLUSION_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        OCCLUSION_INPUT_NAME = OCCLUSION_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception as e:
        print(f"[face_analyser] Failed load occluder model: {e}")
        OCCLUSION_SESSION = None
        OCCLUSION_INPUT_NAME = None

    return OCCLUSION_SESSION


def _run_occluder_onnx(crop: np.ndarray) -> float:
    """
    Jalankan occluder.onnx di atas crop wajah.
    Return: occlusion score 0–1 (semakin besar artinya semakin tertutup).
    Kalau model tidak tersedia / error → return 0.0 (anggap tidak occluded).
    """
    if crop is None or crop.size == 0:
        return 0.0

    session = _get_occluder_session()
    if session is None:
        return 0.0

    try:
        h, w = crop.shape[:2]
        inp = cv2.resize(crop, (224, 224))
        inp = inp.astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW

        outputs = session.run(None, {OCCLUSION_INPUT_NAME: inp})
        pred = outputs[0]

        if pred.ndim == 4:
            mask = pred[0, 0]
        else:
            mask = pred[0]

        mask = cv2.resize(mask, (w, h))
        occl_ratio = float(np.mean(mask > 0.5))
        return occl_ratio
    except Exception:
        return 0.0


# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Deteksi banyak wajah di satu frame.
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []

        faces = [face for face in faces if getattr(face, "det_score", 0.0) >= MIN_DET_SCORE]
        return faces
    except ValueError:
        return None
    except Exception:
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_face_pose(face: Face) -> tuple[float, float, float]:
    pose = getattr(face, "pose", None)
    if pose is None:
        return 0.0, 0.0, 0.0

    try:
        pitch = float(pose[0])
        yaw = float(pose[1])
        roll = float(pose[2])
        return pitch, yaw, roll
    except Exception:
        return 0.0, 0.0, 0.0


# =====================================================================
#  MOTION & TRACKING
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    if prev_face is None or current_face is None:
        return 0.0

    prev_bbox = prev_face.bbox
    current_bbox = current_face.bbox

    prev_center = np.array([
        (prev_bbox[0] + prev_bbox[2]) / 2,
        (prev_bbox[1] + prev_bbox[3]) / 2
    ])
    current_center = np.array([
        (current_bbox[0] + current_bbox[2]) / 2,
        (current_bbox[1] + current_bbox[3]) / 2
    ])

    motion = np.linalg.norm(current_center - prev_center)
    return float(motion)


def _compute_embedding_similarity(current_embedding: np.ndarray,
                                  track_embedding: np.ndarray) -> float:
    try:
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


# =====================================================================
#  TEMPORAL FRAME BUFFER
# =====================================================================

def push_temporal_frame(faces: List[Face], frame_number: int) -> None:
    if not faces:
        return

    snapshot = []
    for f in faces:
        emb = getattr(f, "normed_embedding", None)
        pose = getattr(f, "pose", None)

        snapshot.append({
            "bbox": np.array(f.bbox, dtype=np.float32).copy(),
            "embedding": emb.copy() if isinstance(emb, np.ndarray) else emb,
            "pose": np.array(pose, dtype=np.float32).copy() if pose is not None else None
        })

    TEMPORAL_BUFFER.append({
        "frame": frame_number,
        "faces": snapshot
    })


def smooth_bbox_for_face(face: Face) -> None:
    if not TEMPORAL_BUFFER:
        return

    try:
        cx = float((face.bbox[0] + face.bbox[2]) / 2.0)
        cy = float((face.bbox[1] + face.bbox[3]) / 2.0)
    except Exception:
        return

    centers: List[float] = []
    bboxes: List[np.ndarray] = []

    for entry in TEMPORAL_BUFFER:
        for f in entry["faces"]:
            bbox = f.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            fx1, fy1, fx2, fy2 = bbox
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0
            dist = float(np.hypot(fcx - cx, fcy - cy))
            centers.append(dist)
            bboxes.append(bbox)

    if not bboxes:
        return

    idx_order = np.argsort(np.array(centers))
    k = min(3, len(idx_order))
    selected = [bboxes[i] for i in idx_order[:k]]

    avg_bbox = np.mean(selected, axis=0)
    face.bbox = avg_bbox.astype(np.float32)


# =====================================================================
#  SMART TRACKING (EMBEDDING + TEMPORAL + RAFT)
# =====================================================================

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    global FACE_TRACKING, TRACKING_HISTORY, TEMPORAL_BUFFER

    current_faces = get_many_faces(frame)
    if not current_faces:
        return None

    tracked_faces: List[Face] = []

    with TRACK_LOCK:
        for face in current_faces:
            face_id = None
            max_similarity = MIN_EMBED_SIMILARITY
            best_match_id = None

            current_embedding = getattr(face, "normed_embedding", None)
            if current_embedding is None or len(current_embedding) == 0:
                current_embedding = np.array([])

            for track_id, track_data in list(FACE_TRACKING.items()):
                if frame_number - track_data.get('last_seen', -9999) > MAX_TRACK_GAP:
                    continue

                last_face = track_data.get('last_face', None)
                if last_face is None:
                    continue

                track_embedding = getattr(last_face, "normed_embedding", None)
                if track_embedding is None:
                    continue

                embedding_similarity = _compute_embedding_similarity(
                    current_embedding, track_embedding
                )

                if embedding_similarity > max_similarity:
                    max_similarity = embedding_similarity
                    best_match_id = track_id

            if best_match_id is not None:
                face_id = best_match_id
                prev_face = FACE_TRACKING[face_id]['last_face']
                motion = calculate_motion_vector(prev_face, face)

                FACE_TRACKING[face_id].update({
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': motion
                })
            else:
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': 0.0
                }

            face_data = {
                'bbox': np.array(face.bbox, dtype=np.float32).copy()
            }
            TRACKING_HISTORY.append(face_data)
            tracked_faces.append(face)

        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }

        if tracked_faces:
            push_temporal_frame(tracked_faces, frame_number)

            for f in tracked_faces:
                smooth_bbox_for_face(f)

            # 🔥 RAFT flow di GPU (kalau tersedia)
            flow = _get_raft_flow(frame, frame_number)
            if flow is not None and RAFT_ALPHA > 0.0:
                for f in tracked_faces:
                    raft_stabilize_bbox(f, flow)

    return tracked_faces


# =====================================================================
#  OCCLUSION & SIMILAR FACE
# =====================================================================

def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    base_flag = getattr(face, "det_score", 1.0) < OCCLUSION_THRESHOLD

    if frame is None:
        return base_flag

    occl_session = _get_occluder_session()
    if occl_session is None:
        return base_flag

    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return base_flag

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return base_flag

        occl_score = _run_occluder_onnx(crop)
        threshold = getattr(roop.globals, "occluder_threshold", 0.20)
        return occl_score > threshold
    except Exception:
        return base_flag


def find_similar_face(frame: Frame,
                      reference_face: Face,
                      use_tracking: bool = True) -> Optional[Face]:
    if reference_face is None:
        return None

    if use_tracking:
        many_faces = smart_face_tracking(frame, frame_number=0)
    else:
        many_faces = get_many_faces(frame)

    if not many_faces:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')

    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for face in many_faces:
        if not hasattr(face, "normed_embedding"):
            continue

        try:
            distance = np.sum(np.square(face.normed_embedding - ref_emb))
        except Exception:
            continue

        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = face

    return best_face
