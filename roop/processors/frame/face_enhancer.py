# enhancer-final.py
"""
Enhanced Face Enhancer:
- GFPGAN enhancement + color-matching + fidelity blending
- Adaptive ellipse mask based on face size / landmarks
- Unsharp mask (sharpness control)
- Age-based under-eye wrinkle + dark circle injection
Compatible with existing roop pipeline (get_many_faces, GFPGANer, roop.globals)
"""

from typing import Any, List, Callable, Optional, Tuple
import cv2
import threading
import numpy as np
import math
import os
import traceback

from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

NAME = 'ROOP.FACE-ENHANCER-ADV'
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
FACE_ENHANCER: Optional[Any] = None

# ------- DEFAULT configurable parameters (fallback jika tidak di-set di roop.globals)
DEFAULT_BLEND = 0.6
DEFAULT_SHARPNESS = 0.15     # 0.0 = no change, 0.2 = modest sharpening
DEFAULT_WRINKLE_ENABLED = True
DEFAULT_DARKCIRCLE_ENABLED = True
DEFAULT_WRINKLE_STRENGTH_OVERRIDE = None  # None => compute from age
DEFAULT_WRINKLE_NOISE_SEED = 1234

# ===================================================================
#  Model init / helpers
# ===================================================================

def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'

def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
            print(f"✅ [{NAME}] GFPGANer inited on device {get_device()}")
    return FACE_ENHANCER

def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_enhancer()

# ===================================================================
#  Image utilities
# ===================================================================

def apply_blend_and_color_match(enhanced_crop: np.ndarray, original_crop: np.ndarray, fidelity: float) -> np.ndarray:
    """
    Color matching + fidelity blending + soft elliptical mask.
    """
    try:
        h, w = original_crop.shape[:2]
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))

        # Color match (per-channel mean)
        original_mean = np.mean(original_crop, axis=(0, 1))
        enhanced_mean = np.mean(enhanced_crop, axis=(0, 1))
        color_diff = original_mean - enhanced_mean
        corrected_crop = enhanced_crop.astype(np.float32) + color_diff
        corrected_crop = np.clip(corrected_crop, 0, 255).astype(np.uint8)

        # Fidelity blending to preserve expression
        blended_expression = cv2.addWeighted(corrected_crop, fidelity, original_crop, 1.0 - fidelity, 0)

        # Soft elliptical mask (centered) - geometry will be adapted later at final composition
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)
        axes = (int(w * 0.45), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        blur_radius = int(min(w, h) * 0.1)
        if blur_radius % 2 == 0:
            blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        mask_3ch = np.dstack([mask] * 3)

        final_result = (blended_expression * mask_3ch + original_crop * (1.0 - mask_3ch)).astype(np.uint8)
        return final_result
    except Exception as e:
        update_status(f"Error in blending: {e}", NAME)
        traceback.print_exc()
        return original_crop

def apply_unsharp_mask(img: np.ndarray, amount: float = 0.15) -> np.ndarray:
    """
    Unsharp mask sharpening. Amount typically 0.05-0.35 for faces.
    """
    try:
        if amount is None or amount <= 0.0:
            return img
        # Choose sigma relative to image size
        h, w = img.shape[:2]
        sigma = max(1.0, min(h, w) / 200.0)  # heuristic
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharp = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        return sharp
    except Exception:
        return img

# ===================================================================
#  Wrinkle & dark circle generators
# ===================================================================

def generate_wrinkle_mask(shape: Tuple[int, int], strength: float = 0.2, seed: int = 0) -> np.ndarray:
    """
    Buat peta kerutan halus: memakai multi-scale high-frequency noise + directional blur.
    Hasil: grayscale 0..1
    """
    h, w = shape
    rng = np.random.RandomState(seed)
    base = np.zeros((h, w), dtype=np.float32)

    # multiple octaves of line-like noise
    for scale, amp in [(1, 0.6), (2, 0.25), (4, 0.15)]:
        # create random thin line patterns
        noise = rng.randn(h // scale + 1, w // scale + 1).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        # emphasize thin ridges by Laplacian and threshold
        ridge = cv2.Laplacian(noise, cv2.CV_32F)
        ridge = np.abs(ridge)
        ridge = ridge / (ridge.max() + 1e-9)
        base += ridge * amp

    # directional smoothing to imitate wrinkle lines (horizontal-ish)
    ksize = int(max(3, min(w, h) * 0.01))
    if ksize % 2 == 0: ksize += 1
    base = cv2.GaussianBlur(base, (ksize, ksize), 0)
    # emphasize contrast
    base = np.clip((base - base.mean()) * 2.5 + 0.5, 0.0, 1.0)
    # scale by strength
    mask = np.clip(base * strength, 0.0, 1.0)
    return mask

def apply_under_eye_wrinkles_and_darkness(face: Face, crop: np.ndarray, strength_override: Optional[float] = None, enable_wrinkle: bool = True, enable_dark: bool = True) -> np.ndarray:
    """
    Tambahkan kerutan & dark circle pada crop wajah berdasarkan usia & posisi mata.
    - face.kps (5 keypoints) digunakan untuk lokasi mata
    """
    try:
        h, w = crop.shape[:2]
        kps = getattr(face, "kps", None)  # expected (5,2): left_eye, right_eye, nose, left_mouth, right_mouth
        age = getattr(face, "age", None)

        # fallback: jika tidak ada kps, kembalikan crop utuh
        if kps is None or len(kps) < 2:
            return crop

        le = np.array(kps[0], dtype=np.float32)
        re = np.array(kps[1], dtype=np.float32)

        # hitung face scale
        eye_dist = np.linalg.norm(le - re)
        face_scale = max(1.0, eye_dist)

        # tentukan strength berdasarkan age (jika tidak override)
        if strength_override is not None:
            strength = float(strength_override)
        else:
            if age is None:
                strength = 0.10
            else:
                try:
                    age = float(age)
                except Exception:
                    age = None
                if age is None:
                    strength = 0.10
                elif age >= 50:
                    strength = 0.40
                elif age >= 40:
                    strength = 0.30
                elif age >= 30:
                    strength = 0.20
                elif age >= 20:
                    strength = 0.15
                else:
                    strength = 0.08

        # normalize strength a bit for stability
        strength = np.clip(strength, 0.02, 0.6)

        # create mask area for under-eye: ellipse under each eye position
        mask_total = np.zeros((h, w), dtype=np.float32)

        # compute eye positions relative to crop: face.kps are in absolute frame coordinates;
        # convert to crop coordinates by estimating face.bbox
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            # fallback: place mask by relative positions
            left_eye_px = (int(w * 0.33), int(h * 0.4))
            right_eye_px = (int(w * 0.66), int(h * 0.4))
        else:
            x1, y1, x2, y2 = map(int, bbox)
            # convert absolute kps to crop coords
            crop_origin = np.array([x1, y1], dtype=np.int32)
            left_eye_px = tuple(np.clip((le - crop_origin).astype(int), [0,0], [w-1,h-1]))
            right_eye_px = tuple(np.clip((re - crop_origin).astype(int), [0,0], [w-1,h-1]))

        # define ellipse sizes based on eye_dist (scale to crop)
        axes_x = max(6, int(eye_dist * 0.6))
        axes_y = max(3, int(eye_dist * 0.25))

        # tweak axes relative to crop size
        axes_x = int(np.clip(axes_x, 6, w * 0.45))
        axes_y = int(np.clip(axes_y, 3, h * 0.25))

        # draw elliptical regions below each eye
        for ex, ey in [left_eye_px, right_eye_px]:
            # move center slightly downwards to target under-eye
            center = (int(ex), int(min(h-1, ey + axes_y * 0.8)))
            axes = (axes_x, axes_y)
            angle = 0
            cv2.ellipse(mask_total, center, axes, angle, 0, 360, 1.0, -1)

        # blur mask for smoothness
        blur_radius = int(max(3, min(w, h) * 0.02))
        if blur_radius % 2 == 0: blur_radius += 1
        mask_total = cv2.GaussianBlur(mask_total, (blur_radius, blur_radius), 0)

        result = crop.copy().astype(np.float32)

        # apply darkening for dark circles
        if enable_dark:
            dark_intensity = strength * 0.65  # darkening amplitude
            dark = np.zeros_like(result, dtype=np.float32)
            dark[:, :] = -80.0  # uniform darken value before scaling
            # blend dark with mask
            for c in range(3):
                result[:, :, c] = np.clip(result[:, :, c] + dark[:, :, c] * (mask_total * dark_intensity), 0, 255)

        # apply wrinkle texture
        if enable_wrinkle:
            noise_seed = getattr(roop.globals, "wrinkle_noise_seed", DEFAULT_WRINKLE_NOISE_SEED)
            wrinkle_mask = generate_wrinkle_mask((h, w), strength=strength, seed=int(noise_seed))
            # blend fine wrinkle as darker thin lines + slight sharpen contrast
            # create grayscale wrinkle image from crop luminance
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            wrinkle_map = (gray * 0.0) + wrinkle_mask  # we can scale by local luminance if needed
            wrinkle_map = np.clip(wrinkle_map * mask_total, 0.0, 1.0)

            # darken along wrinkle_map (subtle)
            for c in range(3):
                result[:, :, c] = np.clip(result[:, :, c] - (wrinkle_map * strength * 80.0), 0, 255)

            # optionally enhance micro-contrast along wrinkle ridges
            micro = (wrinkle_map * 255.0).astype(np.uint8)
            micro = cv2.cvtColor(micro, cv2.COLOR_GRAY2BGR).astype(np.float32)
            result = np.clip(result - (micro * 0.05), 0, 255)

        return result.astype(np.uint8)

    except Exception:
        traceback.print_exc()
        return crop

# ===================================================================
#  Adaptive ellipse mask builder
# ===================================================================

def build_adaptive_ellipse_mask(h: int, w: int, face: Face, padding: float = 0.0) -> np.ndarray:
    """
    Buat masker ellipse yang adaptif berdasarkan ukuran wajah & landmarks.
    Return mask float 0..1 (3 channels)
    """
    mask = np.zeros((h, w), dtype=np.float32)
    try:
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            # fallback to center ellipse
            cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.45)), 0, 0, 360, 1.0, -1)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            # clip
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # prefer ellipse axes from bbox but scaled to cover cheeks/temples
            axes_x = int(bw * (0.5 + padding))
            axes_y = int(bh * (0.55 + padding))

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # if face has kps (5 points), adjust center slightly upward to include forehead
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 2:
                le = np.array(kps[0], dtype=np.float32)
                re = np.array(kps[1], dtype=np.float32)
                eye_center_y = int((le[1] + re[1]) / 2)
                # place ellipse center slightly above bbox center (to better cover face)
                center_y = int(center_y - max(0, (center_y - eye_center_y) * 0.15))

            axes_x = int(np.clip(axes_x, 10, w//1))
            axes_y = int(np.clip(axes_y, 10, h//1))
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 1.0, -1)

        # blur mask
        blur_radius = int(max(3, min(h, w) * 0.06))
        if blur_radius % 2 == 0: blur_radius += 1
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        mask_3ch = np.dstack([mask] * 3)
        return mask_3ch
    except Exception:
        traceback.print_exc()
        return np.dstack([mask] * 3)

# ===================================================================
#  Main face enhancement worker
# ===================================================================

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Proses enhancement per wajah:
    - ambil crop + padding
    - GFPGAN enhance
    - color match + fidelity blend
    - unsharp mask (sharpness)
    - age-based wrinkles & dark circles
    - adaptive mask compositing ke frame
    """
    try:
        bbox = target_face['bbox'] if isinstance(target_face, dict) else getattr(target_face, 'bbox', None)
        if bbox is None:
            return temp_frame

        x1, y1, x2, y2 = map(int, bbox)
        padding_x = int((x2 - x1) * 0.2)
        padding_y = int((y2 - y1) * 0.2)

        h_frame, w_frame = temp_frame.shape[:2]
        sx = max(0, x1 - padding_x)
        sy = max(0, y1 - padding_y)
        ex = min(w_frame, x2 + padding_x)
        ey = min(h_frame, y2 + padding_y)

        if sx >= ex or sy >= ey:
            return temp_frame

        temp_face = temp_frame[sy:ey, sx:ex]
        if temp_face.size == 0:
            return temp_frame

        with THREAD_SEMAPHORE:
            _, _, enhanced_face = get_face_enhancer().enhance(temp_face, paste_back=True)

        # blend amount from globals
        blend_amount = roop.globals.face_enhancer_blend if getattr(roop.globals, "face_enhancer_blend", None) is not None else DEFAULT_BLEND
        blended = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount)

        # apply sharpness
        sharpness_amount = roop.globals.sharpness if getattr(roop.globals, "sharpness", None) is not None else DEFAULT_SHARPNESS
        if sharpness_amount is None:
            sharpness_amount = DEFAULT_SHARPNESS
        blended = apply_unsharp_mask(blended, amount=float(sharpness_amount))

        # age-based wrinkle & dark circle
        wrinkle_enabled = getattr(roop.globals, "enable_wrinkle", DEFAULT_WRINKLE_ENABLED)
        dark_enabled = getattr(roop.globals, "enable_darkcircle", DEFAULT_DARKCIRCLE_ENABLED)
        strength_override = getattr(roop.globals, "wrinkle_strength", DEFAULT_WRINKLE_STRENGTH_OVERRIDE)

        # face object conformity: ensure we pass the correct face object (roop face often is class-like)
        face_obj = target_face if not isinstance(target_face, dict) else None
        if face_obj is None:
            # try to retrieve face in frame matching bbox (best-effort)
            # fallback: create a simple proxy with kps/bbox/age if possible
            class Proxy:
                pass
            p = Proxy()
            p.bbox = bbox
            # try to find kps & age from get_many_faces nearby - expensive, skip for speed; let face param be None
            p.kps = getattr(target_face, 'kps', None)
            p.age = getattr(target_face, 'age', None)
            face_obj = p

        processed = apply_under_eye_wrinkles_and_darkness(face_obj, blended, strength_override=strength_override, enable_wrinkle=wrinkle_enabled, enable_dark=dark_enabled)

        # final composite with adaptive mask
        mask = build_adaptive_ellipse_mask(h_frame, w_frame, face_obj, padding=0.05)
        # Need to paste processed back into temp_frame area only
        # build localized mask for the crop region:
        global_mask = np.zeros((h_frame, w_frame, 3), dtype=np.float32)
        global_mask[sy:ey, sx:ex] = mask[sy:ey, sx:ex]
        local_mask = global_mask[sy:ey, sx:ex]
        if local_mask.sum() <= 0:
            # fallback to full paste
            temp_frame[sy:ey, sx:ex] = processed
        else:
            # composite
            local_mask_3ch = local_mask
            src = processed.astype(np.float32)
            dst = temp_face.astype(np.float32)
            comp = (src * local_mask_3ch + dst * (1.0 - local_mask_3ch)).astype(np.uint8)
            temp_frame[sy:ey, sx:ex] = comp

        return temp_frame

    except Exception as e:
        update_status(f"Error in enhance_face: {e}", NAME)
        traceback.print_exc()
        return temp_frame

# ===================================================================
#  Frame processing API (used by roop pipeline)
# ===================================================================

def process_frame(_: Face, __: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            continue
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        update_status("Failed to read target image.", NAME)
        return
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)

# Entry for module test
if __name__ == "__main__":
    print(f"{NAME} loaded. This module is meant to be used by roop pipeline.")
