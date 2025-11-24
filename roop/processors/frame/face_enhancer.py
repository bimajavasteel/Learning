from typing import Any, List, Callable, Optional
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces, detect_occlusion
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


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


# -----------------------------
#  Helper utilities
# -----------------------------

def make_face_mask_from_landmarks(face: Face, frame_shape) -> np.ndarray:
    """
    Buat mask binary (0/255) dari landmark atau bbox face.
    Mengembalikan mask berukuran sama dengan frame.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    kps = None
    if hasattr(face, "kps"):
        try:
            kps = np.array(face.kps).reshape(-1, 2)
        except Exception:
            kps = None
    elif hasattr(face, "kps_2d"):
        try:
            kps = np.array(face.kps_2d).reshape(-1, 2)
        except Exception:
            kps = None

    if kps is not None and kps.size:
        try:
            hull = cv2.convexHull(kps.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        except Exception:
            kps = None

    if kps is None:
        # fallback ke bbox
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        except Exception:
            pass

    return mask


def feather_mask(mask: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Feather atau soft-edge mask. Mengembalikan float mask 0..1."""
    if ksize % 2 == 0:
        ksize += 1
    # normalisasi
    mask_f = mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    return np.clip(blurred, 0.0, 1.0)


def paste_with_seamless_clone(src_face_img: np.ndarray, dst_frame: np.ndarray, center: tuple) -> np.ndarray:
    """
    Gunakan Poisson blending via cv2.seamlessClone untuk hasil paling halus.
    src_face_img: BGR image yang akan dipaste (ukuran sama dengan area yang ingin di-paste)
    center: (x,y) center lokasi pada dst_frame
    """
    gray = cv2.cvtColor(src_face_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    # pastikan mask single channel
    mask = mask.astype(np.uint8)
    try:
        output = cv2.seamlessClone(src_face_img, dst_frame, mask, center, cv2.NORMAL_CLONE)
        return output
    except Exception:
        # fallback ke direct paste
        return dst_frame


def upscale_background(frame: np.ndarray, faces: List[Face], scale: float = 1.06, use_real_esrgan: bool = False) -> np.ndarray:
    """
    Upscale area background (non-face) sedikit dan blend kembali.
    scale kecil (1.03..1.10) direkomendasikan.

    Jika real esrgan tersedia, kamu bisa panggilnya dengan ganti resize.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for f in faces:
        try:
            m = make_face_mask_from_landmarks(f, frame.shape)
            mask = np.maximum(mask, m)
        except Exception:
            continue

    inv_mask = (mask == 0).astype(np.uint8) * 255

    ys, xs = np.where(inv_mask)
    if len(xs) == 0 or len(ys) == 0:
        return frame

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # jaga padding kecil supaya tidak memperbesar area terlalu jauh
    bg = frame[y1:y2+1, x1:x2+1]
    if bg.size == 0:
        return frame

    new_w = max(1, int(bg.shape[1] * scale))
    new_h = max(1, int(bg.shape[0] * scale))
    up_bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # sedikit unsharp mask
    gauss = cv2.GaussianBlur(up_bg, (0, 0), 3)
    up_bg = cv2.addWeighted(up_bg, 1.3, gauss, -0.3, 0)

    # center crop kembali ke ukuran semula
    start_x = max(0, (up_bg.shape[1] - bg.shape[1]) // 2)
    start_y = max(0, (up_bg.shape[0] - bg.shape[0]) // 2)
    up_back = up_bg[start_y:start_y + bg.shape[0], start_x:start_x + bg.shape[1]]

    inv_mask_crop = inv_mask[y1:y2+1, x1:x2+1]
    inv_mask_3 = inv_mask_crop[:, :, None] / 255.0

    frame[y1:y2+1, x1:x2+1] = (frame[y1:y2+1, x1:x2+1].astype(np.float32) * (1 - inv_mask_3) + up_back.astype(np.float32) * inv_mask_3).astype(np.uint8)
    return frame


# -----------------------------
#  Main enhancer hooks
# -----------------------------

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    # baca bbox
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.18)  # sedikit kurangi padding default
    padding_y = int((end_y - start_y) * 0.18)
    sx = max(0, start_x - padding_x)
    sy = max(0, start_y - padding_y)
    ex = min(temp_frame.shape[1], end_x + padding_x)
    ey = min(temp_frame.shape[0], end_y + padding_y)

    # cek occlusion dulu
    try:
        occluded = detect_occlusion(target_face, temp_frame)
    except Exception:
        occluded = False

    # kalau ter-occluded => jangan paksa GFPGAN, cukup lakukan mild denoise/sharpen
    if occluded:
        face_region = temp_frame[sy:ey, sx:ex]
        try:
            face_region = cv2.bilateralFilter(face_region, d=5, sigmaColor=75, sigmaSpace=75)
            temp_frame[sy:ey, sx:ex] = face_region
        except Exception:
            pass
        return temp_frame

    temp_face = temp_frame[sy:ey, sx:ex].copy()
    if temp_face.size:
        with THREAD_SEMAPHORE:
            # ambil hasil saja (paste_back=False)
            try:
                restored_face, _, _ = get_face_enhancer().enhance(
    temp_face,
    paste_back=False
)

# --- FIX: NORMALISASI OUTPUT GFPGAN ---
# GFPGAN kadang mengembalikan list berisi ndarray
if isinstance(restored_face, list):
    if len(restored_face) > 0:
        restored_face = restored_face[0]
    else:
        restored_face = temp_face.copy()

# Pastikan final output adalah numpy array 3D
restored_face = np.asarray(restored_face)

# Jika GFPGAN memberi channel aneh (1 channel dst), paksa jadi BGR 3-channel
if restored_face.ndim == 2:
    restored_face = cv2.cvtColor(restored_face, cv2.COLOR_GRAY2BGR)
elif restored_face.ndim == 3 and restored_face.shape[2] == 1:
    restored_face = cv2.cvtColor(restored_face, cv2.COLOR_GRAY2BGR)

# Jika ukuran tidak match, resize
if restored_face.shape[:2] != temp_face.shape[:2]:
    restored_face = cv2.resize(restored_face, (temp_face.shape[1], temp_face.shape[0]))

            except Exception:
                # fallback: coba sekali lagi pakai paste_back True untuk safety
                try:
                    _, _, restored_face = get_face_enhancer().enhance(temp_face, paste_back=True)
                except Exception:
                    return temp_frame

        # buat mask full-frame -> crop
        mask_full = make_face_mask_from_landmarks(target_face, temp_frame.shape)
        mask_crop = mask_full[sy:ey, sx:ex]
        # feather mask (ksize bisa kamu tuning)
        alpha = feather_mask(mask_crop, ksize=41)
        alpha_3 = alpha[:, :, None]

        # blend sederhana dulu
        try:
            blended = restored_face.astype(np.float32) * alpha_3 + temp_face.astype(np.float32) * (1 - alpha_3)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            temp_frame[sy:ey, sx:ex] = blended
        except Exception:
            # fallback paste langsung
            temp_frame[sy:ey, sx:ex] = restored_face

        # jika masih terlihat seam, opsi gunakan seamlessClone pada area tersebut
        # deteksi seam sederhana: bandingkan statistik tepi mask
        try:
            edge = cv2.Canny((mask_crop > 0).astype(np.uint8) * 255, 50, 150)
            edge_percent = (edge > 0).sum() / float(edge.size)
            # jika area edge relatif kecil tapi ada seam, pakai seamlessClone
            if edge_percent > 0.0005:
                center = ((sx + ex) // 2, (sy + ey) // 2)
                temp_frame = paste_with_seamless_clone(restored_face, temp_frame, center)
        except Exception:
            pass

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)

        # opsi: sedikit upscale background setelah enhancement
        try:
            bg_scale = getattr(roop.globals, 'background_upscale', 1.06)
            if bg_scale is not None and bg_scale > 1.0 and bg_scale <= 1.12:
                temp_frame = upscale_background(temp_frame, many_faces, scale=bg_scale)
        except Exception:
            pass

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
