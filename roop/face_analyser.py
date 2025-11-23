import os
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
from typing import List, Optional

from roop.typing import Face
import roop.globals


# ===========================================================
# CONFIG MODEL PATHS
# ===========================================================

MODEL_DIR = "/kaggle/working/Learning/models"
BUFFALO_DIR = os.path.join(MODEL_DIR, "buffalo_l")
OCCLUDER_PATH = os.path.join(MODEL_DIR, "occluder.onnx")

# Buffalo_L expected files
BUFFALO_FACE = os.path.join(BUFFALO_DIR, "det_10g.onnx")
BUFFALO_LAND = os.path.join(BUFFALO_DIR, "2d106det.onnx")
BUFFALO_REC = os.path.join(BUFFALO_DIR, "w600k_r50.onnx")


# ===========================================================
# LOAD ONNX MODELS
# ===========================================================

def _onnx_session(path: str) -> ort.InferenceSession:
    EP = roop.globals.execution_providers
    return ort.InferenceSession(path, providers=EP)


face_detector = None
landmark_model = None
recognition_model = None
occluder_model = None

TRACKING_HISTORY = deque(maxlen=32)


def load_models():
    global face_detector, landmark_model, recognition_model, occluder_model

    if face_detector is None:
        face_detector = _onnx_session(BUFFALO_FACE)
    if landmark_model is None:
        landmark_model = _onnx_session(BUFFALO_LAND)
    if recognition_model is None:
        recognition_model = _onnx_session(BUFFALO_REC)
    if occluder_model is None:
        occluder_model = _onnx_session(OCCLUDER_PATH)


# ===========================================================
# UTILITY FUNCTIONS
# ===========================================================

def detect_faces(img):
    """Return list of (bbox, score)"""
    h, w = img.shape[:2]
    blob = cv2.resize(img, (640, 640))
    blob = blob[:, :, ::-1].astype(np.float32)
    blob = np.expand_dims(blob.transpose(2, 0, 1), 0)

    outputs = face_detector.run(None, {face_detector.get_inputs()[0].name: blob})[0]

    faces = []
    for det in outputs:
        score = det[4]
        if score < 0.5:
            continue
        x1 = max(0, int(det[0] * w / 640))
        y1 = max(0, int(det[1] * h / 640))
        x2 = min(w, int(det[2] * w / 640))
        y2 = min(h, int(det[3] * h / 640))
        faces.append((np.array([x1, y1, x2, y2]), float(score)))

    return faces


def detect_landmarks(img, bbox):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (192, 192))
    crop = crop[:, :, ::-1].astype(np.float32) / 255.0
    crop = np.expand_dims(crop.transpose(2, 0, 1), 0)

    out = landmark_model.run(None, {landmark_model.get_inputs()[0].name: crop})[0][0]
    pts = out.reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * (x2 - x1) + x1
    pts[:, 1] = pts[:, 1] * (y2 - y1) + y1
    return pts


def get_embedding(img, pts):
    if pts is None:
        return None
    src = _five_points_template()
    M, _ = cv2.estimateAffinePartial2D(pts[:5], src, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, M, (112, 112))
    blob = aligned[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.expand_dims(blob.transpose(2, 0, 1), 0)

    emb = recognition_model.run(None, {recognition_model.get_inputs()[0].name: blob})[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb


def _five_points_template():
    return np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ])


def check_occlusion(img, bbox):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0  # highly occluded
    crop = cv2.resize(crop, (128, 128))
    crop = crop[:, :, ::-1].astype(np.float32) / 255.0
    crop = np.expand_dims(crop.transpose(2, 0, 1), 0)

    pred = occluder_model.run(None, {occluder_model.get_inputs()[0].name: crop})[0][0]
    # model output: [not_occluded_prob]
    occlusion_score = 1 - pred
    return occlusion_score
# ===========================================================
# FACE OBJECT BUILDER (Return always Face object)
# ===========================================================

def build_face_object(bbox, landmarks, embedding, occlusion):
    # Convert bbox to float32
    bbox = np.array(bbox).astype(np.float32)

    # If no landmarks, create placeholder
    if landmarks is None:
        landmarks = np.zeros((106, 2), dtype=np.float32)

    # Construct Roop Face object
    return Face(
        bbox=bbox,
        kps=landmarks,
        embedding=embedding,
        occlusion=float(occlusion)
    )


# ===========================================================
# SIMILARITY LOGIC
# ===========================================================

def cosine_sim(a, b):
    if a is None or b is None:
        return -1
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_best_match(faces: List[Face], source_face: Face) -> Optional[Face]:
    if source_face is None:
        return None
    best = None
    best_sim = -999
    for f in faces:
        s = cosine_sim(f.embedding, source_face.embedding)
        if s > best_sim:
            best_sim = s
            best = f
    return best


# ===========================================================
# SMART TRACKING (Stable video face keeping)
# ===========================================================

def track_face(current_faces: List[Face], previous_face: Optional[Face]) -> Optional[Face]:
    if previous_face is None:
        return None
    if not current_faces:
        return None

    best = None
    best_dist = 999999

    prev_bbox = previous_face.bbox
    px = (prev_bbox[0] + prev_bbox[2]) / 2
    py = (prev_bbox[1] + prev_bbox[3]) / 2

    for f in current_faces:
        bx = (f.bbox[0] + f.bbox[2]) / 2
        by = (f.bbox[1] + f.bbox[3]) / 2
        dist = (px - bx) ** 2 + (py - by) ** 2
        if dist < best_dist:
            best = f
            best_dist = dist
    return best


# ===========================================================
# MAIN ANALYSIS FUNCTIONS
# ===========================================================

def analyse_frame(frame):
    """Full detection → landmarks → embedding → occlusion → Face objects list"""
    faces = detect_faces(frame)
    out = []

    for (bbox, score) in faces:
        if score < 0.5:
            continue

        lmk = detect_landmarks(frame, bbox)
        emb = get_embedding(frame, lmk)
        occ = check_occlusion(frame, bbox)

        # Skip if heavily occluded
        if occ > 0.65:
            continue

        face = build_face_object(bbox, lmk, emb, occ)
        out.append(face)

    return out


# ===========================================================
# PUBLIC API — REQUIRED BY ROOP
# ===========================================================

_last_tracked = None


def get_one_face(frame):
    """Return the best face (used for source image)."""

    load_models()
    faces = analyse_frame(frame)

    if not faces:
        return None

    # Pick largest face
    areas = [(f, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) for f in faces]
    faces_sorted = sorted(areas, key=lambda x: x[1], reverse=True)
    return faces_sorted[0][0]


def get_many_faces(frame):
    """Return all valid faces (target frame)."""
    load_models()
    return analyse_frame(frame)


def find_similar_face(source_face: Face, faces: List[Face]):
    if source_face is None:
        return None
    if not faces:
        return None
    return find_best_match(faces, source_face)


def smart_face_tracking(source_face: Face, current_faces: List[Face]):
    global _last_tracked

    if not current_faces:
        return None

    if _last_tracked is None:
        best = find_best_match(current_faces, source_face)
        _last_tracked = best
        return best

    # Step 1: tracking position
    tracked = track_face(current_faces, _last_tracked)

    # Step 2: verify similarity from source
    sim = cosine_sim(tracked.embedding, source_face.embedding)
    if sim < 0.1:
        best = find_best_match(current_faces, source_face)
        _last_tracked = best
        return best

    _last_tracked = tracked
    return tracked


def detect_occlusion(face: Face):
    """Return occlusion score (0 = clean, 1 = blocked)"""
    return face.occlusion
# ===========================================================
# RESET + CLEANUP
# ===========================================================

def reset_face_tracking():
    """Reset tracking history (dipakai Roop saat video berganti)."""
    global _last_tracked
    _last_tracked = None
    TRACKING_HISTORY.clear()


# ===========================================================
# OPTIONAL FORCE-SELECTOR FOR SOURCE
# (If you want manual face selection in future)
# ===========================================================

def set_face_reference(ref_face: Face):
    """Store source face if needed for similarity."""
    global _last_tracked
    _last_tracked = ref_face


# ===========================================================
# COMPATIBILITY WRAPPERS — REQUIRED BY ROOP ORIGINAL
# ===========================================================

def get_face_reference():
    """Return last tracked face but usually unused."""
    global _last_tracked
    return _last_tracked


def analyse_image(frame):
    """Alias required by older Roop code."""
    return get_many_faces(frame)


def analyse_video_frame(frame):
    """Alias for consistency."""
    return get_many_faces(frame)


# ===========================================================
# EXPORT SYMBOLS
# ===========================================================

__all__ = [
    "get_one_face",
    "get_many_faces",
    "find_similar_face",
    "smart_face_tracking",
    "detect_occlusion",
    "set_face_reference",
    "get_face_reference",
    "reset_face_tracking"
]
