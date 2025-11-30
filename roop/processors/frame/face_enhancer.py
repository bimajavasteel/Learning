# roop/processors/frame/face_enhancer.py
from typing import Any, List, Callable
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

NAME = "ROOP.FACE-ENHANCER"
FACE_ENHANCER: Any = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()

# ---------------------------
# Helper: device selection
# ---------------------------
def get_device() -> str:
    if "CUDAExecutionProvider" in roop.globals.execution_providers:
        return "cuda"
    if "CoreMLExecutionProvider" in roop.globals.execution_providers:
        return "mps"
    return "cpu"

# ---------------------------
# Load GFPGAN
# ---------------------------
def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path("../models/GFPGANv1.4.pth")
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER

def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path("../models")
    conditional_download(download_directory_path, [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True

def post_process() -> None:
    clear_face_enhancer()

# ---------------------------
# Perlin-like subtle noise util (fast)
# ---------------------------
DEFAULT_PERLIN_STRENGTH = 0.07
DEFAULT_PERLIN_SCALE = 48

def generate_perlin_noise(h: int, w: int, scale: int = DEFAULT_PERLIN_SCALE) -> np.ndarray:
    try:
        grid_h = max(2, h // scale + 2)
        grid_w = max(2, w // scale + 2)
        grid = np.random.rand(grid_h, grid_w).astype(np.float32)
        noise = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise
    except Exception:
        return np.zeros((h, w), dtype=np.float32)

def add_subtle_skin_noise(img: np.ndarray, strength: float = DEFAULT_PERLIN_STRENGTH, scale: int = DEFAULT_PERLIN_SCALE) -> np.ndarray:
    if img is None or img.size == 0 or strength <= 0:
        return img
    h, w = img.shape[:2]
    noise = generate_perlin_noise(h, w, scale=scale)
    noise3 = np.dstack([noise]*3)
    img_f = img.astype(np.float32)
    # subtle modulation: multiply luminance channel slightly
    luminance = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    lum3 = np.dstack([luminance]*3)
    mod = 1.0 + (noise3 - 0.5) * (strength * 1.8)
    result = img_f * mod
    # preserve overall brightness using luminance scaling
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# ---------------------------
# Wrinkle helpers (combined smart_wrinkle_map + expression + age)
# ---------------------------
def detect_expression(face: Face) -> str:
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return "neutral"
    lm = np.array(lm)
    # simple heuristics (empirical)
    mouth_top = lm[52]; mouth_bottom = lm[58]; mouth_left = lm[48]; mouth_right = lm[54]
    brow_left = lm[19]; brow_center = lm[21]
    mouth_open = abs(mouth_bottom[1] - mouth_top[1])
    mouth_width = abs(mouth_right[0] - mouth_left[0])
    brow_drop = abs(brow_center[1] - brow_left[1])
    if mouth_open > 6:
        return "open_mouth"
    if mouth_width > 40:
        return "smile"
    if brow_drop > 5:
        return "frown"
    return "neutral"

def compute_wrinkle_strength(age: float) -> float:
    try:
        age = float(age)
    except Exception:
        return 0.0
    if age >= 40:
        return 0.0
    elif age >= 30:
        return 0.25
    elif age >= 20:
        return 0.35
    elif age >= 13:
        return 0.55
    return 0.0

def full_face_wrinkle(frame: np.ndarray, face: Face, strength: float) -> np.ndarray:
    if strength <= 0 or frame is None:
        return frame
    x1, y1, x2, y2 = map(int, face.bbox)
    H, W = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    base = crop.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0,0), sigmaX=3)
    high = base - blur
    enhanced = base + high * (strength * 1.8)
    result = cv2.addWeighted(base, 1 - strength, enhanced, strength, 0)
    frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
    return frame

# Smart wrinkle map helpers
def _poly_mask(shape, pts, dilation=3):
    mask = np.zeros(shape[:2], dtype=np.float32)
    hull = cv2.convexHull(np.array(pts).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)
    if dilation > 0:
        k = dilation*2 + 1
        mask = cv2.GaussianBlur(mask, (k,k), 0)
    return mask

def build_wrinkle_map(face: Face, shape, expression: str) -> np.ndarray:
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return np.zeros(shape[:2], dtype=np.float32)
    lm = np.array(lm)
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    try:
        left_crow = lm[[94,95,96,97]]
        right_crow = lm[[101,102,103,104]]
        under_left = lm[[94,95,96,97,98]]
        under_right = lm[[101,102,103,104,105]]
        naso_left = lm[[46,47,58,67,68]]
        naso_right = lm[[53,54,56,65,66]]
        smile_left = lm[[48,49,59,60]]
        smile_right = lm[[53,54,64,65]]
        forehead = lm[[17,18,19,20,21,22,23,24]] + np.array([0,-25])
        glabella = lm[[21,22,27]] + np.array([0,-10])
        chin = lm[[57,58,59,60]] + np.array([0,18])
    except Exception:
        return np.zeros((H, W), dtype=np.float32)

    if expression == "smile":
        mask += _poly_mask((H,W), left_crow)
        mask += _poly_mask((H,W), right_crow)
        mask += _poly_mask((H,W), naso_left)
        mask += _poly_mask((H,W), naso_right)
        mask += _poly_mask((H,W), smile_left)
        mask += _poly_mask((H,W), smile_right)
    elif expression == "frown":
        mask += _poly_mask((H,W), glabella)
        mask += _poly_mask((H,W), forehead)
    elif expression == "open_mouth":
        mask += _poly_mask((H,W), chin)
        mask += _poly_mask((H,W), naso_left)
        mask += _poly_mask((H,W), naso_right)
    else:
        mask += _poly_mask((H,W), under_left, dilation=2)
        mask += _poly_mask((H,W), under_right, dilation=2)

    mask = np.clip(mask, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (49,49), 0)
    return mask

def apply_smart_wrinkle_map(frame: np.ndarray, face: Face, expression: str, strength: float) -> np.ndarray:
    if strength <= 0 or frame is None:
        return frame
    x1, y1, x2, y2 = map(int, face.bbox)
    H, W = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame
    mask = build_wrinkle_map(face, crop.shape, expression)
    mask3 = np.dstack([mask]*3)
    base = crop.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0,0), 3)
    high = base - blur
    detail = base + high * (strength * 1.8)
    blended = base * (1 - mask3) + detail * (mask3 * strength * 1.2)
    frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame

# Expression-based wrinkle booster
def apply_expression_wrinkle(frame: np.ndarray, face: Face, expression: str, base_strength: float) -> np.ndarray:
    if base_strength <= 0 or frame is None:
        return frame
    if expression == "smile":
        strength = base_strength * 1.30
    elif expression == "open_mouth":
        strength = base_strength * 1.15
    elif expression == "frown":
        strength = base_strength * 1.40
    else:
        strength = base_strength
    return full_face_wrinkle(frame, face, strength)

# ---------------------------
# Color-match + ellipse mask blending (anti-flicker)
# ---------------------------
def apply_blend_and_color_match(enhanced_crop: np.ndarray, original_crop: np.ndarray, fidelity: float) -> np.ndarray:
    try:
        h, w = original_crop.shape[:2]
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        orig_mean = np.mean(original_crop, axis=(0,1))
        enh_mean = np.mean(enhanced_crop, axis=(0,1))
        diff = orig_mean - enh_mean
        corrected = enhanced_crop.astype(np.float32) + diff
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        blended = cv2.addWeighted(corrected, fidelity, original_crop, 1 - fidelity, 0)
        mask = np.zeros((h,w), dtype=np.float32)
        center = (w//2, h//2)
        axes = (max(1,int(w*0.45)), max(1,int(h*0.45)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        br = max(1, int(min(w,h)*0.1))
        if br % 2 == 0:
            br += 1
        mask = cv2.GaussianBlur(mask, (br,br), 0)
        mask3 = np.dstack([mask]*3)
        final = blended * mask3 + original_crop.astype(np.float32) * (1 - mask3)
        return np.clip(final, 0, 255).astype(np.uint8)
    except Exception:
        return original_crop

# ---------------------------
# Main enhance_face
# ---------------------------
def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    try:
        x1, y1, x2, y2 = map(int, target_face["bbox"])
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        H, W = temp_frame.shape[:2]
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)
        crop = temp_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return temp_frame

        with THREAD_SEMAPHORE:
            try:
                _, _, enhanced = get_face_enhancer().enhance(crop, paste_back=True)
            except Exception:
                enhanced = crop.copy()

        # blending params
        fidelity = getattr(roop.globals, "face_enhancer_blend", 0.6)
        try:
            fidelity = float(fidelity)
        except Exception:
            fidelity = 0.6

        result = apply_blend_and_color_match(enhanced, crop, fidelity)

        # age-based baseline wrinkle
        age = getattr(target_face, "age", 30)
        try:
            age_val = float(age)
        except Exception:
            age_val = 30.0
        wrinkle_strength = compute_wrinkle_strength(age_val)
        if wrinkle_strength > 0:
            result = full_face_wrinkle(result, target_face, wrinkle_strength)

        # expression-based booster
        expression = detect_expression(target_face)
        result = apply_expression_wrinkle(result, target_face, expression, wrinkle_strength)

        # smart region-aware wrinkle
        result = apply_smart_wrinkle_map(result, target_face, expression, wrinkle_strength)

        # perlin micro-noise (strength can be overridden via roop.globals.perlin_noise_strength)
        perlin_strength = getattr(roop.globals, "perlin_noise_strength", DEFAULT_PERLIN_STRENGTH)
        try:
            perlin_strength = float(perlin_strength)
        except Exception:
            perlin_strength = DEFAULT_PERLIN_STRENGTH
        perlin_scale = getattr(roop.globals, "perlin_noise_scale", DEFAULT_PERLIN_SCALE)
        try:
            perlin_scale = int(perlin_scale)
        except Exception:
            perlin_scale = DEFAULT_PERLIN_SCALE
        result = add_subtle_skin_noise(result, strength=perlin_strength, scale=perlin_scale)

        temp_frame[y1:y2, x1:x2] = result
        return temp_frame

    except Exception as e:
        update_status(f"[Enhancer Error] {e}", NAME)
        return temp_frame

# ---------------------------
# Frame loop
# ---------------------------
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    faces = get_many_faces(temp_frame)
    if faces:
        for f in faces:
            temp_frame = enhance_face(f, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for fp in temp_frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        out = process_frame(None, None, frame)
        cv2.imwrite(fp, out)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    frame = cv2.imread(target_path)
    out = process_frame(None, None, frame)
    cv2.imwrite(output_path, out)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
