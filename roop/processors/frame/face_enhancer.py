from typing import Any, List, Callable
import cv2
import threading
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_many_faces,
    smart_face_tracking,
    detect_occlusion
)
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()  # batasi concurrent enhance (hindari OOM)
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


# ==========================
#  MODEL HANDLING
# ==========================

def get_face_enhancer() -> Any:
    """
    Lazy init GFPGAN, thread-safe.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=1,
                device=get_device()
            )
    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def clear_face_enhancer() -> None:
    """
    Reset enhancer agar bisa re-init kalau perlu.
    """
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    """
    Pastikan model GFPGAN sudah di-download.
    """
    download_directory_path = resolve_relative_path('../models')
    conditional_download(
        download_directory_path,
        ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth']
    )
    return True


def pre_start() -> bool:
    """
    Validasi target path (image / video).
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    """
    Cleanup setelah selesai enhance.
    """
    clear_face_enhancer()


# ==========================
#  INTI ENHANCE
# ==========================

def _extract_bbox_from_face(target_face: Face, frame_shape) -> tuple[int, int, int, int] | None:
    """
    Helper untuk ambil dan clamp bbox dari objek Face (bukan dict).
    Menggunakan bbox hasil smoothing dari face_analyser.
    """
    frame_height, frame_width = frame_shape[:2]

    bbox = getattr(target_face, "bbox", None)
    if bbox is None:
        return None

    # bbox bisa float → convert ke int
    x1, y1, x2, y2 = map(int, bbox)

    # pastikan urutan benar (kadang bisa kebalik kalau ada bug upstream)
    start_x, end_x = sorted([x1, x2])
    start_y, end_y = sorted([y1, y2])

    # clamp ke dalam frame
    start_x = max(0, min(start_x, frame_width - 1))
    end_x   = max(0, min(end_x,   frame_width))
    start_y = max(0, min(start_y, frame_height - 1))
    end_y   = max(0, min(end_y,   frame_height))

    if end_x <= start_x or end_y <= start_y:
        return None

    return start_x, start_y, end_x, end_y


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhance satu wajah pada frame menggunakan GFPGAN:
    - pakai bbox dari Face (bukan dict)
    - padding adaptif agar konteks cukup tanpa terlalu lebar
    - aman terhadap error enhancer
    """
    frame_height, frame_width = temp_frame.shape[:2]

    bbox = _extract_bbox_from_face(target_face, temp_frame.shape)
    if bbox is None:
        return temp_frame

    start_x, start_y, end_x, end_y = bbox

    face_w, face_h = end_x - start_x, end_y - start_y
    if face_w <= 0 or face_h <= 0:
        return temp_frame

    # Padding adaptif:
    # - minimal 10% bbox
    # - maksimal 30% bbox
    # - kalau wajah kecil → padding relatif lebih besar biar cukup konteks
    pad_ratio = max(0.10, min(0.30, 100 / max(face_w, face_h)))
    padding_x = int(face_w * pad_ratio)
    padding_y = int(face_h * pad_ratio)

    # terapkan padding & clamp
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x   = min(frame_width,  end_x + padding_x)
    end_y   = min(frame_height, end_y + padding_y)

    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size == 0:
        return temp_frame

    # GFPGAN kadang makan banyak VRAM → batasi dengan semaphore
    with THREAD_SEMAPHORE:
        try:
            # enhance:
            # return: cropped_faces, restored_faces, restored_img
            _, _, enhanced_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )

            # Pastikan ukuran match sebelum dipaste kembali
            if enhanced_face is not None and enhanced_face.shape == temp_face.shape:
                temp_frame[start_y:end_y, start_x:end_x] = enhanced_face
        except Exception as e:
            print(f"[WARNING] Enhance face failed: {e}")

    return temp_frame


# ==========================
#  FRAME PROCESSING
# ==========================

def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Proses 1 frame:
    - Pakai smart_face_tracking kalau perlu konsistensi ID
    - Skip wajah occluded (tangan, rambut, objek)
    - Enhance hanya wajah valid
    """
    # Mode banyak wajah → gunakan tracking agar stabil
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
    else:
        # Mode single-face, cukup deteksi biasa
        faces = get_many_faces(temp_frame)

    if not faces:
        return temp_frame

    for face in faces:
        # Skip wajah yang ter-occlusion (det_score rendah)
        if detect_occlusion(face):
            continue

        temp_frame = enhance_face(face, temp_frame)

    return temp_frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    """
    Dipanggil oleh core.process_video untuk batch frame (per-thread).
    Kita kasih frame_number lokal (index loop) ke process_frame untuk tracking.
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
    Enhance image tunggal.
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
    Entry point mode video.
    - source_path tidak dipakai untuk enhancer (hanya target yang di-enhance)
    - core.process_video yang akan mengatur multi-thread / progres
    """
    roop.processors.frame.core.process_video(
        None,
        temp_frame_paths,
        process_frames
    )
