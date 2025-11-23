from typing import Any, List, Callable
import cv2
import threading
import numpy as np

from CodeFormer.basicsr.archs.codeformer_arch import CodeFormer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces, smart_face_tracking, detect_occlusion
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER: Any = None
THREAD_SEMAPHORE = threading.Semaphore(1)   # batasi concurrent inference
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


# =====================================================================
#  MODEL HANDLING
# =====================================================================

def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def get_face_enhancer() -> Any:
    """
    Lazy init CodeFormer, thread-safe.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/codeformer.pth')
            device = get_device()
            # Pastikan argumen sesuai dengan wrapper CodeFormer di environment kamu.
            FACE_ENHANCER = CodeFormer(
                model_path=model_path,
                upscale=1,
                device=device
            )
            print(f"✅ [face_enhancer] Using CodeFormer on {device}")
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    """
    Bersihkan model dari memori.
    """
    global FACE_ENHANCER
    with THREAD_LOCK:
        FACE_ENHANCER = None


# =====================================================================
#  PRE/POST CHECK
# =====================================================================

def pre_check() -> bool:
    """
    Download model CodeFormer jika belum ada.
    """
    download_directory_path = resolve_relative_path('../models')
    conditional_download(
        download_directory_path,
        ['https://github.com/sczhou/CodeFormer/releases/download/v1.0/codeformer.pth']
    )
    return True


def pre_start() -> bool:
    """
    Validasi target_path: harus image atau video.
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    """
    Cleanup setelah selesai proses.
    """
    clear_face_enhancer()


# =====================================================================
#  CORE ENHANCE LOGIC
# =====================================================================

def _safe_crop_with_padding(frame: Frame, bbox: np.ndarray, pad_ratio: float = 0.20):
    """
    Crop wajah dari frame berdasarkan bbox + padding, dijepit supaya tidak keluar frame.
    Return:
      - cropped_face (H,W,3)
      - (start_x, start_y, end_x, end_y) koordinat di frame asli
    """
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    start_x = max(0, int(x1 - pad_x))
    start_y = max(0, int(y1 - pad_y))
    end_x   = min(w, int(x2 + pad_x))
    end_y   = min(h, int(y2 + pad_y))

    if start_x >= end_x or start_y >= end_y:
        return None, (0, 0, 0, 0)

    cropped = frame[start_y:end_y, start_x:end_x]
    if cropped.size == 0:
        return None, (0, 0, 0, 0)

    return cropped, (start_x, start_y, end_x, end_y)


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhance satu wajah di frame dengan CodeFormer.
    - Pakai bbox dari Face.bbox (bukan dict)
    - Ada padding & clamping ke ukuran frame
    - Thread-safe via semaphore
    """
    # Face dari insightface punya attribute .bbox
    bbox = getattr(target_face, "bbox", None)
    if bbox is None:
        return temp_frame

    bbox_arr = np.array(bbox, dtype=np.float32)
    temp_face, (sx, sy, ex, ey) = _safe_crop_with_padding(temp_frame, bbox_arr, pad_ratio=0.20)

    if temp_face is None or temp_face.size == 0:
        return temp_frame

    # Inference CodeFormer
    with THREAD_SEMAPHORE:
        enhancer = get_face_enhancer()
        try:
            # fidelity_weight bisa kamu tuning : 0.5–0.9
            # lebih kecil → lebih halus / natural, lebih besar → makin tajam (kadang terasa "AI look")
            enhanced = enhancer.inference(temp_face, fidelity_weight=0.7)
        except Exception:
            # kalau ada error di enhancer, jangan stop pipeline
            return temp_frame

    # Pastikan ukuran cocok sebelum ditempel balik
    if enhanced is None or enhanced.size == 0:
        return temp_frame

    eh, ew = enhanced.shape[:2]
    th, tw = temp_frame.shape[:2]

    # Clamp lagi kalau ada mismatch setelah enhancement
    ex_clamped = min(ex, tw)
    ey_clamped = min(ey, th)
    sx_clamped = max(0, sx)
    sy_clamped = max(0, sy)

    crop_w = ex_clamped - sx_clamped
    crop_h = ey_clamped - sy_clamped

    if crop_w <= 0 or crop_h <= 0:
        return temp_frame

    # Resize hasil enhanced supaya pas ke slot crop
    enhanced_resized = cv2.resize(enhanced, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    temp_frame[sy_clamped:ey_clamped, sx_clamped:ex_clamped] = enhanced_resized
    return temp_frame


# =====================================================================
#  FRAME PROCESSING
# =====================================================================

def process_frame(
    source_face: Face | None,
    reference_face: Face | None,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Proses 1 frame:
    - Pakai smart_face_tracking untuk ID wajah yang stabil (kalau tersedia)
    - Fallback ke get_many_faces
    - Skip wajah yang occluded (pakai detect_occlusion dari face_analyser)
    - Enhance tiap wajah valid dengan CodeFormer
    """
    # Coba gunakan tracking pintar (sinkron dengan face_swapper & face_analyser super)
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        faces = get_many_faces(temp_frame)

    if not faces:
        return temp_frame

    for target_face in faces:
        # Skip wajah yang dianggap occluded (tangan, rambut, dsb)
        if detect_occlusion(target_face):
            continue

        temp_frame = enhance_face(target_face, temp_frame)

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Dipanggil oleh core.process_video, bertugas memproses list frame.
    Di sini kita yang assign frame_number (index dalam segment).
    """
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            continue

        result = process_frame(
            source_face=None,
            reference_face=None,
            temp_frame=temp_frame,
            frame_number=idx
        )

        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Mode image → image.
    Tracking tidak terlalu penting di sini, jadi frame_number = 0 saja.
    """
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        return

    result = process_frame(
        source_face=None,
        reference_face=None,
        temp_frame=target_frame,
        frame_number=0
    )
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Entry point untuk mode video.
    - source_path tidak dipakai di enhancer → bisa None
    - core.process_video akan handle multi-thread/multi-process
    """
    roop.processors.frame.core.process_video(
        None,
        temp_frame_paths,
        process_frames
    )
