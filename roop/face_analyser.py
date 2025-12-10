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

import onnxruntime as ort  # optional occluder

# =====================================================================
# GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()
TRACK_LOCK = threading.Lock()

# Tracking
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Temporal buffer (untuk smoothing multi-frame)
TEMPORAL_BUFFER: deque = deque(maxlen=5)

# Threshold
MIN_DET_SCORE = 0.30
OCCLUSION_THRESHOLD = 0.40
MAX_TRACK_GAP = 10
MAX_TRACK_AGE = 15
MIN_EMBED_SIMILARITY = 0.70

# Occluder
OCCLUDER_SESSION: Optional[ort.InferenceSession] = None
OCCLUDER_INPUT_NAME: Optional[str] = None

# =====================================================================
# RAFT LARGE (C_T_SKHT_K_V2) – OPTICAL FLOW CUDA
# =====================================================================

RAFT_MODEL: Optional[torch.nn.Module] = None
RAFT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
RAFT_LOCK = threading.Lock()

# kita simpan frame sebelumnya dalam bentuk raw BGR (numpy)
RAFT_PREV_FRAME: Optional[np.ndarray] = None
RAFT_LAST_FLOW: Optional[np.ndarray] = None
RAFT_LAST_FRAME_IDX: int = -1

# blending RAFT vs bbox asli
RAFT_ALPHA: float = 0.7

# transforms dari weights (normalisasi ke [-1, 1], resize, dll)
RAFT_TRANSFORMS = None


def _get_raft_model() -> Optional[torch.nn.Module]:
    """
    Lazy init RAFT Large C_T_SKHT_K_V2 dari torchvision.
    """
    global RAFT_MODEL, RAFT_TRANSFORMS

    with RAFT_LOCK:
        if RAFT_MODEL is not None:
            return RAFT_MODEL

        try:
            weights = Raft_Large_Weights.C_T_SKHT_K_V2
            RAFT_TRANSFORMS = weights.transforms()
            model = raft_large(weights=weights, progress=False)
            model = model.to(RAFT_DEVICE)
            model.eval()
            RAFT_MODEL = model
            print(f"✅ [face_analyser][RAFT] RAFT Large (C_T_SKHT_K_V2) loaded on {RAFT_DEVICE}")
        except Exception as e:
            print(f"❌ [face_analyser][RAFT] gagal init RAFT Large: {e}")
            RAFT_MODEL = None
            RAFT_TRANSFORMS = None

        return RAFT_MODEL


def _preprocess_for_raft(prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    BGR uint8 → batch [1,3,H,W] float untuk RAFT, gunakan transforms bawaan weights.
    """
    # BGR → RGB
    prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    cur_rgb = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2RGB)

    # [H,W,3] → [3,H,W]
    prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).float() / 255.0
    cur_t = torch.from_numpy(cur_rgb).permute(2, 0, 1).float() / 255.0

    # buat batch
    prev_t = prev_t.unsqueeze(0)  # [1,3,H,W]
    cur_t = cur_t.unsqueeze(0)

    if RAFT_TRANSFORMS is not None:
        try:
            prev_t, cur_t = RAFT_TRANSFORMS(prev_t, cur_t)
        except Exception as e:
            # kalau transforms error, minimal tetap jalan tanpa transforms
            print(f"[face_analyser][RAFT] warning transforms error: {e}")
    return prev_t.to(RAFT_DEVICE), cur_t.to(RAFT_DEVICE)


@torch.no_grad()
def _get_raft_flow(frame: Frame, frame_idx: int) -> Optional[np.ndarray]:
    """
    Hitung optical flow dense pakai RAFT Large.
    Output: numpy (H,W,2) dalam koordinat piksel (dx,dy).
    """
    global RAFT_PREV_FRAME, RAFT_LAST_FLOW, RAFT_LAST_FRAME_IDX

    model = _get_raft_model()
    if model is None or frame is None:
        return None

    # frame pertama: belum bisa hitung flow
    if RAFT_PREV_FRAME is None:
        RAFT_PREV_FRAME = frame.copy()
        RAFT_LAST_FLOW = None
        RAFT_LAST_FRAME_IDX = frame_idx
        print(f"[face_analyser][RAFT] first frame #{frame_idx}, belum ada flow.")
        return None

    # cache kalau sudah dihitung di frame ini
    if RAFT_LAST_FRAME_IDX == frame_idx and RAFT_LAST_FLOW is not None:
        return RAFT_LAST_FLOW

    try:
        img1_t, img2_t = _preprocess_for_raft(RAFT_PREV_FRAME, frame)

        # torchvision RAFT: output = list of flows (iterasi)
        list_of_flows = model(img1_t, img2_t)

        if isinstance(list_of_flows, (list, tuple)) and len(list_of_flows) > 0:
            flow_tensor = list_of_flows[-1][0]  # ambil iterasi terakhir, batch idx 0 → [2,Hf,Wf]
        else:
            # fallback kalau versi lain: asumsikan [N,2,H,W]
            flow_tensor = list_of_flows[0]

        flow_tensor = flow_tensor.detach().cpu()  # [2,Hf,Wf]
        flow_np = flow_tensor.permute(1, 2, 0).numpy().astype(np.float32)  # [Hf,Wf,2]

        # resize ke ukuran frame asli kalau perlu
        h, w = frame.shape[:2]
        if flow_np.shape[0] != h or flow_np.shape[1] != w:
            fx = cv2.resize(flow_np[..., 0], (w, h), interpolation=cv2.INTER_LINEAR)
            fy = cv2.resize(flow_np[..., 1], (w, h), interpolation=cv2.INTER_LINEAR)
            flow_np = np.stack([fx, fy], axis=-1).astype(np.float32)

        RAFT_PREV_FRAME = frame.copy()
        RAFT_LAST_FLOW = flow_np
        RAFT_LAST_FRAME_IDX = frame_idx

        # debug ringan
        # print(f"[face_analyser][RAFT] flow computed for frame {frame_idx}, shape={flow_np.shape}")
        return flow_np

    except Exception as e:
        print(f"[face_analyser][RAFT] error inference: {e}")
        RAFT_PREV_FRAME = frame.copy()
        RAFT_LAST_FLOW = None
        RAFT_LAST_FRAME_IDX = frame_idx
        return None


def raft_stabilize_bbox(face: Face, flow: np.ndarray) -> None:
    """
    Geser bbox mengikuti vektor flow di titik tengah lalu di-blend dengan bbox asli.
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
        dx, dy = flow[cy, cx]
    except Exception:
        return

    flow_bbox = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy], dtype=np.float32)
    cur_bbox = np.array(face.bbox, dtype=np.float32)

    alpha = float(RAFT_ALPHA)
    alpha = max(0.0, min(1.0, alpha))

    blended = alpha * flow_bbox + (1.0 - alpha) * cur_bbox
    face.bbox = blended.astype(np.float32)
# =====================================================================
# FACE ANALYSER (INSIGHTFACE)
# =====================================================================

def get_face_analyser() -> Any:
    """
    Lazy init insightface FaceAnalysis (buffalo_l).
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            print("[face_analyser] init insightface buffalo_l...")
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            print("✅ [face_analyser] Using buffalo_l (pose + 2d106 + 3d68)")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    """
    Reset analyser & seluruh state tracking + RAFT.
    Panggil di post_process.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY, TEMPORAL_BUFFER
    global RAFT_PREV_FRAME, RAFT_LAST_FLOW, RAFT_LAST_FRAME_IDX, RAFT_MODEL, RAFT_TRANSFORMS

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()
        TEMPORAL_BUFFER.clear()

    RAFT_PREV_FRAME = None
    RAFT_LAST_FLOW = None
    RAFT_LAST_FRAME_IDX = -1
    RAFT_MODEL = None
    RAFT_TRANSFORMS = None

    with THREAD_LOCK:
        FACE_ANALYSER = None

    print("[face_analyser] analyser + tracking + RAFT cleared.")


# =====================================================================
# OCCLUDER ONNX (opsional)
# =====================================================================

def _get_occluder_session() -> Optional[ort.InferenceSession]:
    """
    Lazy init occluder.onnx.
    Jika tidak ada → akan fallback ke det_score.
    """
    global OCCLUDER_SESSION, OCCLUDER_INPUT_NAME

    if OCCLUDER_SESSION is not None:
        return OCCLUDER_SESSION

    model_rel = getattr(roop.globals, "occluder_model_path", "../models/occluder.onnx")
    model_path = resolve_relative_path(model_rel)

    if not os.path.exists(model_path):
        print(f"[face_analyser] occluder model not found at {model_path}, fallback ke det_score.")
        return None

    try:
        OCCLUDER_SESSION = ort.InferenceSession(
            model_path,
            providers=roop.globals.execution_providers
        )
        OCCLUDER_INPUT_NAME = OCCLUDER_SESSION.get_inputs()[0].name
        print(f"✅ [face_analyser] Loaded occluder model: {model_path}")
    except Exception as e:
        print(f"❌ [face_analyser] Failed load occluder model: {e}")
        OCCLUDER_SESSION = None
        OCCLUDER_INPUT_NAME = None

    return OCCLUDER_SESSION


def _run_occluder_onnx(crop: np.ndarray) -> float:
    """
    Jalankan occluder.onnx di atas crop wajah.
    Return: rasio area ter-occlude (0–1).
    Jika error / tidak ada model → return 0.0 (anggap aman).
    """
    if crop is None or crop.size == 0:
        return 0.0

    session = _get_occluder_session()
    if session is None:
        return 0.0

    try:
        h, w = crop.shape[:2]
        inp = cv2.resize(crop, (256, 256))
        inp = inp.astype('float32') / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW

        outputs = session.run(None, {OCCLUDER_INPUT_NAME: inp})
        pred = outputs[0]

        # asumsi output [1,1,H,W] atau [1,H,W]
        if pred.ndim == 4:
            mask = pred[0, 0]
        elif pred.ndim == 3:
            mask = pred[0]
        else:
            mask = pred

        mask = cv2.resize(mask, (w, h))
        occl_ratio = float(np.mean(mask > 0.5))
        return occl_ratio
    except Exception as e:
        print(f"[face_analyser] occluder error: {e}")
        return 0.0


# =====================================================================
# BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Deteksi banyak wajah, filter berdasarkan det_score.
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []

        faces = [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]
        # debug ringan
        # print(f"[face_analyser] detected {len(faces)} faces (after threshold).")
        return faces
    except ValueError:
        return None
    except Exception as e:
        print(f"[face_analyser] error get_many_faces: {e}")
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    Ambil 1 wajah dari frame.
    position=0 default, kalau out-of-range pakai wajah terakhir.
    """
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_face_pose(face: Face) -> tuple[float, float, float]:
    """
    Ambil (pitch, yaw, roll) dari objek Face insightface.
    """
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
# MOTION & TRACKING
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    """
    Hitung jarak perpindahan center bbox (Euclidean).
    """
    if prev_face is None or current_face is None:
        return 0.0

    prev_bbox = prev_face.bbox
    cur_bbox = current_face.bbox

    prev_center = np.array([
        (prev_bbox[0] + prev_bbox[2]) / 2.0,
        (prev_bbox[1] + prev_bbox[3]) / 2.0
    ])
    cur_center = np.array([
        (cur_bbox[0] + cur_bbox[2]) / 2.0,
        (cur_bbox[1] + cur_bbox[3]) / 2.0
    ])

    return float(np.linalg.norm(cur_center - prev_center))


def _compute_embedding_similarity(current_embedding: np.ndarray,
                                  track_embedding: np.ndarray) -> float:
    """
    Cosine similarity antara dua embedding (0–1).
    """
    try:
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


# =====================================================================
# TEMPORAL BUFFER (STABILITAS TEMPORAL)
# =====================================================================

def push_temporal_frame(faces: List[Face], frame_number: int) -> None:
    """
    Simpan snapshot bbox + embedding + pose ke buffer.
    """
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
    """
    Smoothing bbox dengan cara:
    - cari beberapa bbox di buffer yang centernya paling dekat
    - rata-ratakan
    """
    if not TEMPORAL_BUFFER:
        return

    try:
        cx = float((face.bbox[0] + face.bbox[2]) / 2.0)
        cy = float((face.bbox[1] + face.bbox[3]) / 2.0)
    except Exception:
        return

    centers = []
    bboxes = []

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
# SMART TRACKING (EMBEDDING + TEMPORAL + RAFT)
# =====================================================================

def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """
    Tracking wajah antar frame:
    - matching pakai embedding similarity
    - jaga ID konsisten
    - smoothing temporal buffer
    - RAFT optical flow untuk pergerakan halus
    """
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

                similarity = _compute_embedding_similarity(current_embedding, track_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
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
                # print(f"[tracking] frame {frame_number}: update track {face_id}, sim={max_similarity:.3f}")
            else:
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    'last_face': face,
                    'last_seen': frame_number,
                    'motion': 0.0
                }
                # print(f"[tracking] frame {frame_number}: new track {face_id}")

            TRACKING_HISTORY.append({
                'bbox': np.array(face.bbox, dtype=np.float32).copy()
            })
            tracked_faces.append(face)

        # buang track yang terlalu lama tidak muncul
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get('last_seen', -9999) <= MAX_TRACK_AGE
        }

        if tracked_faces:
            # simpan ke buffer
            push_temporal_frame(tracked_faces, frame_number)

            # smoothing bbox spatial
            for f in tracked_faces:
                smooth_bbox_for_face(f)

            # RAFT optical flow (temporal motion)
            flow = _get_raft_flow(frame, frame_number)
            if flow is not None and RAFT_ALPHA > 0.0:
                for f in tracked_faces:
                    raft_stabilize_bbox(f, flow)

    return tracked_faces
# =====================================================================
# OCCLUSION & SIMILAR FACE
# =====================================================================

def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    """
    Deteksi wajah terhalang:
    - kalau occluder.onnx ada → pakai mask model
    - kalau tidak → fallback ke det_score
    """
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

        is_occ = occl_score > threshold
        # print(f"[occlusion] score={occl_score:.3f}, thr={threshold:.3f}, occ={is_occ}")
        return is_occ
    except Exception as e:
        print(f"[face_analyser] detect_occlusion error: {e}")
        return base_flag


def find_similar_face(frame: Frame,
                      reference_face: Face,
                      use_tracking: bool = True) -> Optional[Face]:
    """
    Cari wajah paling mirip dengan reference_face pada frame.
    - kalau use_tracking=True → pakai smart_face_tracking
    - kalau False → pakai get_many_faces biasa
    """
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
            dist = np.sum(np.square(face.normed_embedding - ref_emb))
        except Exception:
            continue

        if dist < similar_threshold and dist < best_distance:
            best_distance = dist
            best_face = face

    # if best_face is not None:
    #     print(f"[similar_face] best dist={best_distance:.4f}")
    return best_face
