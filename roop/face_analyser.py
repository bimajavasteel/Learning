from typing import Any, Optional, List
import threading
from collections import deque
from scipy.spatial.distance import cosine

import insightface
import numpy as np
import cv2
import os

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, conditional_download

# onnxruntime untuk occluder.onnx
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

# =====================================================================
#  GLOBALS
# =====================================================================

FACE_ANALYSER: Any = None
THREAD_LOCK = threading.Lock()        # lock untuk init model
TRACK_LOCK = threading.Lock()         # lock khusus tracking (penting untuk multi-thread)

# Tracking variables
FACE_TRACKING: dict[int, dict[str, Any]] = {}
TRACKING_HISTORY: deque = deque(maxlen=30)

# Threshold / hyper-parameter default (boleh kamu tuning)
MIN_DET_SCORE = 0.30        # min score agar wajah dianggap valid (untuk get_many_faces)
OCCLUSION_THRESHOLD = 0.40  # det_score < ini dianggap occluded (fallback)
MAX_TRACK_GAP = 10          # frame: kalau lebih lama dari ini → track di-skip saat matching
MAX_TRACK_AGE = 15          # frame: track dihapus bila tidak terlihat selama ini
MIN_EMBED_SIMILARITY = 0.70 # cosine similarity minimal untuk dianggap match (0–1)

# Occluder ONNX
OCCLUDER_MODEL_URL = "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/occluder.onnx"
OCCLUDER_SESSION: Any = None
OCCLUDER_LOCK = threading.Lock()
OCCLUDER_OCCLUDED_RATIO = 0.30  # jika > 30% area wajah tertutup → dianggap occluded


# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_face_analyser() -> Any:
    """
    Lazy init insightface FaceAnalysis.
    Sekali saja per proses, thread-safe.
    """
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=roop.globals.execution_providers
            )
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ [face_analyser] Using buffalo_l with optimized settings")
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    """
    Reset analyser & tracking state.
    Dipanggil saat post_process / cleanup.
    """
    global FACE_ANALYSER, FACE_TRACKING, TRACKING_HISTORY, OCCLUDER_SESSION

    with TRACK_LOCK:
        FACE_TRACKING.clear()
        TRACKING_HISTORY.clear()

    with THREAD_LOCK:
        FACE_ANALYSER = None

    with OCCLUDER_LOCK:
        OCCLUDER_SESSION = None


def _update_bbox_from_landmarks(face: Face) -> None:
    """
    Recompute bbox dari landmark (kalau ada)
    supaya bounding box lebih mengikuti kontur wajah (anti 'masker' kaku).
    """
    pts = None

    # buffalo_l biasanya punya 'landmark_2d_106' atau 'kps'
    if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        pts = np.array(face.landmark_2d_106, dtype=np.float32)
    elif hasattr(face, "kps") and face.kps is not None:
        pts = np.array(face.kps, dtype=np.float32)

    if pts is None or pts.size == 0:
        return

    xs = pts[:, 0]
    ys = pts[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    # margin kecil di kiri-kanan, atas-bawah
    margin_x = (max_x - min_x) * 0.12
    margin_y = (max_y - min_y) * 0.18

    x1 = min_x - margin_x
    y1 = min_y - margin_y
    x2 = max_x + margin_x
    y2 = max_y + margin_y

    face.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)


def get_occluder_session() -> Optional[Any]:
    """
    Lazy init occluder.onnx dengan onnxruntime.
    Auto-download ke ../models/occluder.onnx
    """
    global OCCLUDER_SESSION
    if ort is None:
        return None

    with OCCLUDER_LOCK:
        if OCCLUDER_SESSION is not None:
            return OCCLUDER_SESSION

        try:
            models_dir = resolve_relative_path("../models")
            # auto-download file jika belum ada
            conditional_download(models_dir, [OCCLUDER_MODEL_URL])
            occluder_path = os.path.join(models_dir, "occluder.onnx")
            if not os.path.exists(occluder_path):
                occluder_path = resolve_relative_path("../models/occluder.onnx")

            OCCLUDER_SESSION = ort.InferenceSession(
                occluder_path,
                providers=roop.globals.execution_providers
            )
            print("✅ [face_analyser] occluder.onnx loaded")
        except Exception as e:
            print(f"⚠️ [face_analyser] Failed to init occluder.onnx: {e}")
            OCCLUDER_SESSION = None

    return OCCLUDER_SESSION


# =====================================================================
#  BASIC FACE ACCESSORS
# =====================================================================

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    """
    Deteksi banyak wajah di satu frame.
    - Pakai buffalo_l
    - Filter berdasarkan det_score minimal
    - Update bbox berdasarkan landmark (kalau ada)
    """
    try:
        faces = get_face_analyser().get(frame)
        if not faces:
            return []

        faces = [f for f in faces if getattr(f, "det_score", 0.0) >= MIN_DET_SCORE]

        for f in faces:
            _update_bbox_from_landmarks(f)

        return faces
    except ValueError:
        return None
    except Exception:
        return None


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    """
    Ambil 1 wajah dari frame:
    - default: index 0
    - kalau index out-of-range → pakai wajah terakhir
    """
    many = get_many_faces(frame)
    if many:
        try:
            return many[position]
        except IndexError:
            return many[-1]
    return None


# =====================================================================
#  MOTION & TRACKING
# =====================================================================

def calculate_motion_vector(prev_face: Face, current_face: Face) -> float:
    """
    Hitung pergerakan (jarak Euclidean) antara dua bbox wajah berturutan.
    """
    if prev_face is None or current_face is None:
        return 0.0

    p = prev_face.bbox
    c = current_face.bbox

    prev_center = np.array([
        (p[0] + p[2]) / 2,
        (p[1] + p[3]) / 2
    ])
    curr_center = np.array([
        (c[0] + c[2]) / 2,
        (c[1] + c[3]) / 2
    ])

    return float(np.linalg.norm(curr_center - prev_center))


def _compute_embedding_similarity(current_embedding: np.ndarray,
                                  track_embedding: np.ndarray) -> float:
    """
    Hitung similarity embedding (cosine-based).
    """
    try:
        # cosine() mengembalikan distance → similarity = 1 - distance
        return 1.0 - float(cosine(current_embedding, track_embedding))
    except Exception:
        return 0.0


def smart_face_tracking(frame: Frame, frame_number: int) -> Optional[List[Face]]:
    """
    Smart tracking:
    - gunakan embedding similarity + sedikit motion
    - jaga agar ID wajah konsisten antar frame
    - smoothing bbox pakai TRACKING_HISTORY
    """
    global FACE_TRACKING, TRACKING_HISTORY

    current_faces = get_many_faces(frame)
    if not current_faces:
        return None

    tracked_faces: List[Face] = []

    with TRACK_LOCK:
        for face in current_faces:
            face_id = None
            max_sim = MIN_EMBED_SIMILARITY
            best_match_id = None

            curr_emb = getattr(face, "normed_embedding", None)
            if curr_emb is None or len(curr_emb) == 0:
                curr_emb = np.array([])

            for tid, tdata in list(FACE_TRACKING.items()):
                if frame_number - tdata.get("last_seen", -9999) > MAX_TRACK_GAP:
                    continue

                last_face = tdata.get("last_face", None)
                if last_face is None:
                    continue

                track_emb = getattr(last_face, "normed_embedding", None)
                if track_emb is None:
                    continue

                sim = _compute_embedding_similarity(curr_emb, track_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_match_id = tid

            if best_match_id is not None:
                face_id = best_match_id
                prev_face = FACE_TRACKING[face_id]["last_face"]
                motion = calculate_motion_vector(prev_face, face)

                FACE_TRACKING[face_id].update({
                    "last_face": face,
                    "last_seen": frame_number,
                    "motion": motion
                })
            else:
                face_id = len(FACE_TRACKING) + 1
                FACE_TRACKING[face_id] = {
                    "last_face": face,
                    "last_seen": frame_number,
                    "motion": 0.0
                }

            # bbox smoothing sederhana
            if len(TRACKING_HISTORY) >= 2:
                recent = list(TRACKING_HISTORY)[-2:]
                if all("bbox" in f for f in recent):
                    smoothed = np.mean([f["bbox"] for f in recent], axis=0)
                    face.bbox = smoothed

            TRACKING_HISTORY.append({
                "bbox": np.array(face.bbox, dtype=np.float32).copy()
            })
            tracked_faces.append(face)

        # buang track yang terlalu lama
        FACE_TRACKING = {
            k: v for k, v in list(FACE_TRACKING.items())
            if frame_number - v.get("last_seen", -9999) <= MAX_TRACK_AGE
        }

    return tracked_faces


# =====================================================================
#  OCCLUSION & SIMILAR FACE
# =====================================================================

def _detect_occlusion_with_model(face: Face, frame: Frame) -> Optional[bool]:
    """
    Occlusion detect pakai occluder.onnx (kalau tersedia).
    Return:
    - True  → occluded
    - False → tidak occluded
    - None  → gagal / fallback
    """
    sess = get_occluder_session()
    if sess is None:
        return None

    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        inp_size = 256
        img = cv2.resize(crop, (inp_size, inp_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC->CHW
        img = np.expand_dims(img, 0)        # ->NCHW

        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: img})[0]

        m = np.squeeze(out)
        occluded_ratio = float((m > 0.5).mean())

        return occluded_ratio >= OCCLUDER_OCCLUDED_RATIO
    except Exception as e:
        print(f"⚠️ [face_analyser] occluder inference failed: {e}")
        return None


def detect_occlusion(face: Face, frame: Optional[Frame] = None) -> bool:
    """
    Deteksi occlusion:
    - Jika frame & occluder.onnx tersedia → pakai model
    - Jika tidak → fallback ke det_score threshold
    """
    score = getattr(face, "det_score", 1.0)
    det_based = score < OCCLUSION_THRESHOLD

    if frame is None:
        return det_based

    model_res = _detect_occlusion_with_model(face, frame)
    if model_res is None:
        return det_based

    return model_res


def find_similar_face(frame: Frame,
                      reference_face: Face,
                      use_tracking: bool = True) -> Optional[Face]:
    """
    Cari wajah paling mirip di frame terhadap reference_face.
    - Bisa pakai smart tracking (use_tracking=True)
    - Atau fallback ke get_many_faces biasa
    """
    if reference_face is None:
        return None

    if use_tracking:
        many = smart_face_tracking(frame, frame_number=0)
    else:
        many = get_many_faces(frame)

    if not many:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_dist = float("inf")

    similar_threshold = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in many:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            dist = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue

        if dist < similar_threshold and dist < best_dist:
            best_dist = dist
            best_face = f

    return best_face
