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


# ============================================================
#  GLOBALS
# ============================================================
NAME = "ROOP.FACE-ENHANCER"
FACE_ENHANCER: Any = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()


# ============================================================
#  DEVICE SELECTION
# ============================================================
def get_device() -> str:
    if "CUDAExecutionProvider" in roop.globals.execution_providers:
        return "cuda"
    if "CoreMLExecutionProvider" in roop.globals.execution_providers:
        return "mps"
    return "cpu"


# ============================================================
#  GFPGAN LOADER
# ============================================================
def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path("../models/GFPGANv1.4.pth")
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=1,
                device=get_device()
            )
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


# ============================================================
#  PERLIN NOISE (SUBTLE SKIN TEXTURE)
# ============================================================
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
    except:
        return np.zeros((h, w), dtype=np.float32)


def add_subtle_skin_noise(img: np.ndarray, strength: float, scale: int) -> np.ndarray:
    if img is None or img.size == 0 or strength <= 0:
        return img

    h, w = img.shape[:2]
    noise = generate_perlin_noise(h, w, scale=scale)
    noise3 = np.dstack([noise] * 3)

    img_f = img.astype(np.float32)
    mod = 1.0 + (noise3 - 0.5) * (strength * 2.0)
    out = img_f * mod
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
#  WRINKLE BOOSTER (AGE + EXPRESSION)
# ============================================================
def detect_expression(face: Face) -> str:
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return "neutral"
    lm = np.array(lm)

    mouth_top = lm[52]; mouth_bottom = lm[58]
    mouth_left = lm[48]; mouth_right = lm[54]
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
    except:
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


def apply_wrinkle(frame: np.ndarray, face: Face, strength: float) -> np.ndarray:
    if strength <= 0:
        return frame

    x1, y1, x2, y2 = map(int, face.bbox)
    H, W = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame

    base = crop.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), sigmaX=3)
    high = base - blur
    enhanced = base + high * (strength * 2.0)
    result = np.clip(enhanced, 0, 255).astype(np.uint8)

    frame[y1:y2, x1:x2] = result
    return frame


# ============================================================
#  BLENDING (ANTI-FLICKER + COLOR MATCH)
# ============================================================
def apply_blend_and_color_match(enh: np.ndarray, orig: np.ndarray, fidelity: float) -> np.ndarray:
    try:
        h, w = orig.shape[:2]
        if enh.shape[:2] != (h, w):
            enh = cv2.resize(enh, (w, h), interpolation=cv2.INTER_CUBIC)

        orig_mean = np.mean(orig, axis=(0, 1))
        enh_mean = np.mean(enh, axis=(0, 1))
        diff = orig_mean - enh_mean

        corrected = enh.astype(np.float32) + diff
        blended = cv2.addWeighted(corrected.astype(np.uint8), fidelity, orig, 1 - fidelity, 0)

        mask = np.zeros((h, w), dtype=np.float32)
        ctr = (w // 2, h // 2)
        axs = (int(w * 0.45), int(h * 0.45))
        cv2.ellipse(mask, ctr, axs, 0, 0, 360, 1.0, -1)

        br = max(3, int(min(w, h) * 0.12))
        if br % 2 == 0:
            br += 1
        mask = cv2.GaussianBlur(mask, (br, br), 0)

        mask3 = np.dstack([mask] * 3)
        final = blended * mask3 + orig.astype(np.float32) * (1 - mask3)
        return np.clip(final, 0, 255).astype(np.uint8)
    except:
        return orig


# ============================================================
#  MAIN ENHANCE
# ============================================================
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

        # GFPGAN
        with THREAD_SEMAPHORE:
            try:
                _, _, enh = get_face_enhancer().enhance(crop, paste_back=True)
            except:
                enh = crop

        # COLOR + SMOOTH BLEND
        fidelity = getattr(roop.globals, "face_enhancer_blend", 0.6)
        try:
            fidelity = float(fidelity)
        except:
            fidelity = 0.6

        result = apply_blend_and_color_match(enh, crop, fidelity)

        # WRINKLES
        age = getattr(target_face, "age", 30)
        strength = compute_wrinkle_strength(age)

        # Ekspresi → modifikasi strength
        expression = detect_expression(target_face)
        if expression == "smile":
            strength *= 1.25
        elif expression == "frown":
            strength *= 1.45
        elif expression == "open_mouth":
            strength *= 1.15

        result = apply_wrinkle(result, target_face, strength)

        # PERLIN MICRO NOISE
        perlin_strength = getattr(roop.globals, "perlin_noise_strength", DEFAULT_PERLIN_STRENGTH)
        perlin_scale = getattr(roop.globals, "perlin_noise_scale", DEFAULT_PERLIN_SCALE)
        result = add_subtle_skin_noise(result, float(perlin_strength), int(perlin_scale))

        temp_frame[y1:y2, x1:x2] = result
        return temp_frame

    except Exception as e:
        update_status(f"[Enhancer Error] {e}", NAME)
        return temp_frame


# ============================================================
#  FRAME LOOP
# ============================================================
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
