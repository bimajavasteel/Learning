from typing import Any, List, Callable, Optional, Tuple, Dict
import cv2
import insightface
import threading
import numpy as np
import time
import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, smart_face_tracking, detect_occlusion, get_face_pose
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# OneEuro Filter Implementation
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = None, None, None

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if t and self.t_prev: self.freq = 1.0 / max(t - self.t_prev, 1e-6)
        self.t_prev = t if t else time.time()
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        dx_hat = self.alpha(self.d_cutoff) * dx + (1 - self.alpha(self.d_cutoff)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        x_hat = self.alpha(cutoff) * x + (1 - self.alpha(cutoff)) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER-HYBRID'
LANDMARK_FILTERS: Dict[str, Any] = {}

def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None
    LANDMARK_FILTERS.clear()

def pre_check() -> bool:
    conditional_download(resolve_relative_path('../models'), ['https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path): return False
    if not get_one_face(cv2.imread(roop.globals.source_path)): return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def adapt_bbox_for_pose(face: Face, frame_shape: Tuple[int, ...]) -> None:
    pitch, yaw, _ = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    pad_x = w * min((abs(yaw) - 25.0) * 0.02, 0.20) if abs(yaw) > 25.0 else 0
    pad_y_top = h * min((abs(pitch) - 15.0) * 0.02, 0.25) if pitch < -15.0 else 0
    pad_y_bot = h * min((pitch - 20.0) * 0.015, 0.18) if pitch > 20.0 else 0
    
    nx1, nx2 = max(0, x1 - pad_x), min(w_frame, x2 + pad_x)
    ny1, ny2 = max(0, y1 - pad_y_top), min(h_frame, y2 + pad_y_bot)
    if nx2 > nx1 and ny2 > ny1: face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def smooth_landmarks(landmarks, key):
    if key not in LANDMARK_FILTERS: LANDMARK_FILTERS[key] = OneEuroFilter()
    return LANDMARK_FILTERS[key].filter(landmarks, t=time.time())

def robust_alignment(source_face, target_face, frame):
    try:
        slm, tlm = source_face.kps, target_face.kps
        if slm is None or tlm is None: return frame, None
        
        # Use ID from tracking if avail, else fallback
        key = getattr(target_face, 'face_id', 'unknown')
        stlm = smooth_landmarks(tlm, f"face_{key}")
        
        M = cv2.estimateAffinePartial2D(slm, stlm, method=cv2.LMEDS)[0]
        if M is None: return frame, None
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0])), M
    except: return frame, None

def seamless_blend(swapped, target, face):
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = target.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        fw, fh = x2 - x1, y2 - y1
        if swapped.shape[:2] != (fh, fw): swapped = cv2.resize(swapped, (fw, fh))
        
        mask = 255 * np.ones(swapped.shape, swapped.dtype)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return cv2.seamlessClone(swapped, target, mask, center, cv2.NORMAL_CLONE)
    except:
        target[y1:y2, x1:x2] = swapped
        return target

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if not source_face: return temp_frame
    
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces: faces = get_many_faces(temp_frame)
    
    if not roop.globals.many_faces and reference_face:
        ref_emb = reference_face.normed_embedding
        faces = sorted(faces, key=lambda f: np.sum(np.square(f.normed_embedding - ref_emb)))[:1]
    
    for target_face in faces:
        if detect_occlusion(target_face, temp_frame): continue
        
        adapt_bbox_for_pose(target_face, temp_frame.shape)
        aligned, _ = robust_alignment(source_face, target_face, temp_frame)
        
        swap = get_face_swapper().get(aligned, target_face, source_face, paste_back=False)
        if swap is None: continue
        
        # Enhancing swap result (Sharpen/Color)
        kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]) * 0.15
        swap = cv2.bilateralFilter(cv2.filter2D(swap, -1, kernel), 5, 15, 15)
        
        temp_frame = seamless_blend(swap, temp_frame, target_face)
        
    return temp_frame

def process_frames(source_path, temp_frame_paths, update):
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for idx, path in enumerate(temp_frame_paths):
        frame = cv2.imread(path)
        if frame is not None:
            cv2.imwrite(path, process_frame(source_face, reference_face, frame, idx))
        if update: update()

def process_image(source_path, target_path, output_path):
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    cv2.imwrite(output_path, process_frame(source_face, reference_face, target_frame))

def process_video(source_path, temp_frame_paths):
    if not roop.globals.many_faces and not get_face_reference():
        ref = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        set_face_reference(get_one_face(ref, roop.globals.reference_face_position))
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
