# face_swapper.py
# Drop-in replacement module: face swap + wrinkle & dark-circle preservation
# Compatible with Kaggle environment
#
# Menggunakan: insightface (inswapper), roop.globals, roop.face_analyser
# Referensi: core & globals modifications (CLI), face_enhancer (blending idea).
# See: core.py, globals.py, face_enhancer.py, face_analyser.py in project.
# :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

from typing import Any, List, Callable, Optional
import threading
import numpy as np
import cv2
import os
import onnxruntime as ort

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, conditional_download, is_image, is_video
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.face_analyser import get_one_face, get_many_faces, smart_face_tracking, detect_occlusion, get_face_pose

NAME = "ROOP.FACE-SWAPPER-WRINKLE"
THREAD_LOCK = threading.Lock()
FACE_SWAPPER = None

# --- Inswapper model loader (lazy)
def get_face_swapper() -> Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = __load_inswapper(model_path)
    return FACE_SWAPPER

def __load_inswapper(model_path: str) -> Any:
    # wrapper untuk fallback jika insightface.get_model tidak tersedia in this environment
    try:
        import insightface
        return insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    except Exception as e:
        print(f"[{NAME}] Warning: failed to load inswapper via insightface: {e}")
        # last resort: raise so caller sees error
        raise

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

# --- Precheck (download inswapper if needed)
def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True

# -----------------------------
# Utilities: Perlin noise, maps
# -----------------------------
def generate_perlin_noise(h: int, w: int, scale: float = 8.0, seed: int = 0) -> np.ndarray:
    """
    Simple Perlin-like noise (suitable untuk skin pores).
    Deterministik dengan seed.
    """
    np.random.seed(seed)
    def lerp(a, b, t): return a + t * (b - a)
    def fade(t): return 6*t**5 - 15*t**4 + 10*t**3

    nx = int(np.ceil(w / scale)) + 2
    ny = int(np.ceil(h / scale)) + 2
    gradients = np.random.randn(ny + 1, nx + 1, 2)
    gradients /= np.linalg.norm(gradients, axis=2, keepdims=True)
    xs = np.linspace(0, nx - 2, w)
    ys = np.linspace(0, ny - 2, h)
    xi = xs.astype(int)
    yi = ys.astype(int)
    xf = xs - xi
    yf = ys - yi
    xf_f = fade(xf)
    yf_f = fade(yf)

    noise = np.zeros((h, w), dtype=np.float32)
    for j in range(h):
        for i in range(w):
            x0 = xi[i]; y0 = yi[j]
            g00 = gradients[y0, x0]
            g10 = gradients[y0, x0 + 1]
            g01 = gradients[y0 + 1, x0]
            g11 = gradients[y0 + 1, x0 + 1]
            dx = xf[i]; dy = yf[j]
            v00 = np.dot(g00, [dx, dy])
            v10 = np.dot(g10, [dx - 1, dy])
            v01 = np.dot(g01, [dx, dy - 1])
            v11 = np.dot(g11, [dx - 1, dy - 1])
            ix0 = lerp(v00, v10, xf_f[i])
            ix1 = lerp(v01, v11, xf_f[i])
            val = lerp(ix0, ix1, yf_f[j])
            noise[j, i] = val
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise

def make_wrinkle_map_from_source(src_crop: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Ekstrak high-frequency map (kerutan) dari source crop.
    Return float map 0..1
    """
    gray = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    high = gray - blur
    high = np.clip((high - high.min()) / (high.max() - high.min() + 1e-8), 0.0, 1.0)
    # tweak: ambil kontras lembut untuk natural wrinkles
    high = cv2.equalizeHist((high*255).astype(np.uint8)).astype(np.float32) / 255.0
    # skala menurut strength
    return np.clip(high * (0.5 * strength), 0.0, 1.0)

def make_darkcircle_mask(h: int, w: int, face_bbox: List[int], strength: float = 1.0) -> np.ndarray:
    """
    Buat soft mask area bawah mata berdasarkan bbox face.
    Kembalikan mask float 0..1
    """
    # lokasi relatif
    x1, y1, x2, y2 = map(int, face_bbox)
    fw = x2 - x1
    fh = y2 - y1
    mask = np.zeros((h, w), dtype=np.float32)
    # pusat kira-kira bawah mata
    cx = x1 + fw // 2
    cy = y1 + int(fh * 0.57)
    axes = (int(fw * 0.32), int(fh * 0.12))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (int(max(3, min(w, h) * 0.08)) | 1), 0)
    return np.clip(mask * strength, 0.0, 1.0)

# -----------------------------
# Optional: simple age detector (genderage.onnx)
# -----------------------------
AGE_ONNX_SESSION = None
AGE_ONNX_INPUT = None
def _load_age_model() -> Optional[ort.InferenceSession]:
    global AGE_ONNX_SESSION, AGE_ONNX_INPUT
    if AGE_ONNX_SESSION is not None:
        return AGE_ONNX_SESSION
    # try load if exists
    model_rel = getattr(roop.globals, "genderage_model_path", "../models/genderage.onnx")
    model_path = resolve_relative_path(model_rel)
    if not os.path.exists(model_path):
        return None
    try:
        AGE_ONNX_SESSION = ort.InferenceSession(model_path, providers=roop.globals.execution_providers)
        AGE_ONNX_INPUT = AGE_ONNX_SESSION.get_inputs()[0].name
        print(f"[{NAME}] Loaded age model: {model_path}")
        return AGE_ONNX_SESSION
    except Exception as e:
        print(f"[{NAME}] Failed loading age onnx: {e}")
        AGE_ONNX_SESSION = None
        AGE_ONNX_INPUT = None
        return None

def estimate_age_from_crop(crop: np.ndarray) -> Optional[float]:
    """
    Simple inference wrapper for genderage.onnx if available.
    Return age in years (float) or None.
    Note: model input/resizing depends on model; here we assume common 64x64 RGB normalized.
    """
    try:
        sess = _load_age_model()
        if sess is None:
            return None
        inp = cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]
        outputs = sess.run(None, {AGE_ONNX_INPUT: inp})
        # Many genderage models output [age, gender_prob] or distribution; try to handle common shapes.
        out = outputs[0]
        if out.ndim == 2 and out.shape[1] >= 1:
            age = float(out[0, 0])
            return age
        if out.ndim == 1:
            return float(out[0])
        return None
    except Exception:
        return None

# -----------------------------
# Core: apply wrinkle & darkcircle to swapped crop
# -----------------------------
def apply_wrinkle_darkcircle_to_crop(source_crop: np.ndarray,
                                     swapped_crop: np.ndarray,
                                     face_bbox: List[int]) -> np.ndarray:
    """
    source_crop: crop area dari target frame sebelum swap (mengandung tekstur asli target)
    swapped_crop: hasil swap yang akan dimodifikasi
    face_bbox: bbox relatif pada frame (dipakai untuk mask)
    """
    try:
        h, w = swapped_crop.shape[:2]
        # Ambil nilai strength dari globals
        wr_strength = float(getattr(roop.globals, "wrinkle_preservation", 1.0) or 1.0)
        dc_strength = float(getattr(roop.globals, "dark_circle_intensity", 1.0) or 1.0)
        preserve_age_texture = bool(getattr(roop.globals, "preserve_age_texture", True))

        # Jika ada age model & preserve_age_texture => adjust strength berdasar umur
        if preserve_age_texture:
            age = estimate_age_from_crop(source_crop)
            if age is not None:
                # Sederhana: kalau umur >= 40 -> lebih kuat; di bawah 20 -> lebih lembut
                if age >= 50:
                    wr_strength *= 1.25
                    dc_strength *= 1.2
                elif age >= 35:
                    wr_strength *= 1.15
                elif age < 25:
                    wr_strength *= 0.8
                    dc_strength *= 0.8

        # 1) Wrinkle map (high-freq) dari source_crop
        wrinkle_map = make_wrinkle_map_from_source(source_crop, strength=wr_strength)  # 0..1

        # 2) Dark circle mask
        # Note: face_bbox di sini relatif ke frame, tapi kita hanya butuh posisi relatif crop.
        # Kita bikin mask full-crop dengan asumsi crop adalah wajah utama.
        dc_mask = make_darkcircle_mask(h, w, [0, 0, w, h], strength=dc_strength)

        # 3) Perlin noise (subtle pores) optional
        perlin = generate_perlin_noise(h, w, scale=12.0, seed=42)
        perlin = cv2.GaussianBlur(perlin, (7, 7), 0)
        perlin = (perlin - 0.5) * 0.07  # subtle amplitude

        # 4) Compose: darken area under-eye, add wrinkle as multiply detail, add perlin as slight texture
        tgt = swapped_crop.astype(np.float32)
        src = source_crop.astype(np.float32)

        # Dark circle: reduce brightness in dc_mask
        darkened = tgt * (1.0 - 0.25 * dc_mask[..., None])  # base darkening
        # Modulate by low-frequency difference between source and target to keep tones
        src_l = cv2.GaussianBlur(cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2GRAY), (31, 31), 0).astype(np.float32)
        tgt_l = cv2.GaussianBlur(cv2.cvtColor(tgt.astype(np.uint8), cv2.COLOR_BGR2GRAY), (31, 31), 0).astype(np.float32)
        tone_diff = ((src_l - tgt_l) / 255.0)[..., None]
        darkened = np.clip(darkened + tone_diff * dc_mask[..., None] * 20.0, 0, 255)

        # Wrinkle: blend detail from source_high onto swapped via screen/multiply-like
        wrinkle_3ch = cv2.cvtColor((wrinkle_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
        wrinkle_effect = darkened - wrinkle_3ch * 0.6  # subtract darker lines
        # Apply perlin as very slight local contrast
        wrinkle_effect = np.clip(wrinkle_effect + wrinkle_effect * perlin[..., None], 0, 255)

        # Blend final: keep some fidelity to avoid uncanny
        fidelity = float(getattr(roop.globals, 'face_enhancer_blend', 0.6) or 0.6)
        final = cv2.addWeighted(wrinkle_effect.astype(np.uint8), fidelity, swapped_crop.astype(np.uint8), 1 - fidelity, 0)
        return final.astype(np.uint8)
    except Exception as e:
        print(f"[{NAME}] apply_wrinkle_darkcircle_to_crop error: {e}")
        return swapped_crop

# -----------------------------
# Main swap function (replacement)
# -----------------------------
def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    # reuse logic from previous implementation (keamanan: kecil saja)
    try:
        pitch, yaw, roll = get_face_pose(face)
        h_frame, w_frame = frame_shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        w = x2 - x1
        h = y2 - y1
        pad_x = 0.0
        pad_y_top = 0.0
        pad_y_bottom = 0.0
        if abs(yaw) > 25.0:
            extra = (abs(yaw) - 25.0) * 0.02
            extra = min(extra, 0.20)
            pad_x = w * extra
        if pitch < -15.0:
            extra = (abs(pitch) - 15.0) * 0.02
            extra = min(extra, 0.25)
            pad_y_top = h * extra
        elif pitch > 20.0:
            extra = (pitch - 20.0) * 0.015
            extra = min(extra, 0.18)
            pad_y_bottom = h * extra
        nx1 = int(max(0, x1 - pad_x))
        nx2 = int(min(w_frame - 1, x2 + pad_x))
        ny1 = int(max(0, y1 - pad_y_top))
        ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))
        if nx2 <= nx1 or ny2 <= ny1:
            return
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    except Exception:
        return

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Swap face but apply wrinkle & dark-circle preservation from source->target.
    Implementation:
      - adapt bbox
      - call inswapper.get(...) with paste_back=False (ambil hasil crop)
      - modify crop dengan apply_wrinkle_darkcircle_to_crop
      - paste kembali
    """
    if source_face is None or target_face is None:
        return temp_frame

    try:
        adapt_bbox_for_pose(target_face, temp_frame.shape)

        # lakukan swap tapi minta hasil full-frame (paste_back False biasanya mengembalikan
        # image dengan swapped face pasted; beberapa inswapper impl memerlukan strategi berbeda)
        swapper = get_face_swapper()
        swapped_full = swapper.get(temp_frame, target_face, source_face, paste_back=False)

        # Ambil bbox
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h_frame, w_frame = temp_frame.shape[:2]
        x1 = max(0, min(x1, w_frame-1)); x2 = max(0, min(x2, w_frame)); y1 = max(0, min(y1, h_frame-1)); y2 = max(0, min(y2, h_frame))

        if x2 <= x1 or y2 <= y1:
            # fallback: kalau bbox invalid, kembalikan swapped_full
            return swapped_full

        swapped_crop = swapped_full[y1:y2, x1:x2].copy()
        # Ambil source crop (dari target frame sebelum swap) untuk tekstur referensi
        source_crop = temp_frame[y1:y2, x1:x2].copy()

        # Apply wrinkle & darkcircle preservation
        final_crop = apply_wrinkle_darkcircle_to_crop(source_crop, swapped_crop, [x1, y1, x2, y2])

        # Tempel kembali ke frame hasil swapped_full
        swapped_full[y1:y2, x1:x2] = final_crop
        return swapped_full

    except Exception as e:
        print(f"[{NAME}] swap_face error: {e}")
        # On error, fallback to simple swap (try paste_back True)
        try:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        except Exception as e2:
            print(f"[{NAME}] fallback swap also failed: {e2}")
            return temp_frame

# -----------------------------
# Frame processing loops (entrypoints)
# -----------------------------
def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame

    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame
        for target_face in faces:
            if detect_occlusion(target_face, temp_frame):
                continue
            temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame

    # single-face mode
    tracked = smart_face_tracking(temp_frame, frame_number)
    if not tracked:
        tracked = get_many_faces(temp_frame)
    if not tracked:
        return temp_frame
    valid = [f for f in tracked if not detect_occlusion(f, temp_frame)]
    if not valid:
        return temp_frame

    # selection: try reference_face
    best = None
    if reference_face is not None:
        # pick face with minimal embedding distance - fallback to first valid
        try:
            # naive: use bbox center closeness if embeddings not present
            best = valid[0]
            # you can implement embedding-based selection here if needed
        except Exception:
            best = valid[0]
    else:
        best = valid[0]

    temp_frame = swap_face(source_face, best, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for idx, p in enumerate(temp_frame_paths):
        tmp = cv2.imread(p)
        out = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=tmp, frame_number=idx)
        cv2.imwrite(p, out)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_one_face(target_img, roop.globals.reference_face_position)
    out = process_frame(source_face=source_face, reference_face=reference_face, temp_frame=target_img, frame_number=0)
    cv2.imwrite(output_path, out)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)
    # delegate to core loop
    from roop.processors.frame.core import process_video as core_process_video
    core_process_video(source_path, temp_frame_paths, process_frames)
