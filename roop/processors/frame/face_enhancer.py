from typing import Any, List, Callable
import cv2
import threading
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces, detect_occlusion  # ✅ pakai occlusion dari face_analyser
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER: Any = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

# ==========================
# Hyper-parameter enhancer
# ==========================
MIN_FACE_SIZE = 32          # lebar/tinggi minimal wajah yang akan di-enhance
BASE_PAD_RATIO = 0.15       # padding dasar 15%
MIN_PAD_RATIO = 0.08        # batas bawah padding
MAX_PAD_RATIO = 0.30        # batas atas padding


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
            print("✅ [face_enhancer] Using GFPGANv1.4")
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
    """
    Download model GFPGAN bila belum ada.
    """
    download_directory_path = resolve_relative_path('../models')
    conditional_download(
        download_directory_path,
        ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth']
    )
    return True


def pre_start() -> bool:
    """
    Pastikan target_path valid (image / video).
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def _get_bbox_from_face(target_face: Face):
    """
    Support dua gaya:
    - Face object dengan atribut .bbox (insightface)
    - dict dengan key 'bbox'
    """
    bbox = getattr(target_face, "bbox", None)
    if bbox is None and isinstance(target_face, dict):
        bbox = target_face.get("bbox", None)
    return bbox


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhance satu wajah dengan:
    - bbox + padding adaptif
    - aman terhadap out-of-bound
    - resize kalau output GFPGAN beda ukuran
    """
    frame_height, frame_width = temp_frame.shape[:2]

    bbox = _get_bbox_from_face(target_face)
    if bbox is None:
        return temp_frame

    try:
        start_x, start_y, end_x, end_y = map(int, bbox)
    except Exception:
        return temp_frame

    face_w, face_h = end_x - start_x, end_y - start_y
    if face_w <= 0 or face_h <= 0:
        return temp_frame

    # Skip wajah terlalu kecil (biasanya noise / jauh)
    if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
        return temp_frame

    # Padding adaptif:
    # - makin kecil wajah → padding relatif lebih besar
    # - tetap dibatasi MIN_PAD_RATIO–MAX_PAD_RATIO
    max_side = max(face_w, face_h)
    dynamic_ratio = BASE_PAD_RATIO + (80.0 / max(max_side, 1_000))  # kecil → naik sedikit
    pad_ratio = max(MIN_PAD_RATIO, min(MAX_PAD_RATIO, dynamic_ratio))

    padding_x = int(face_w * pad_ratio)
    padding_y = int(face_h * pad_ratio)

    # Clamp ke dalam frame
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(frame_width, end_x + padding_x)
    end_y = min(frame_height, end_y + padding_y)

    if end_x <= start_x or end_y <= start_y:
        return temp_frame

    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size == 0:
        return temp_frame

    with THREAD_SEMAPHORE:
        try:
            # GFPGANer.enhance() → (cropped_faces, restored_faces, restored_img)
            _, _, enhanced_face = get_face_enhancer().enhance(
                temp_face,
                has_aligned=False,
                only_center_face=True,
                paste_back=False
            )

            if enhanced_face is None:
                return temp_frame

            # Pastikan ukuran cocok (resize kalau perlu)
            if enhanced_face.shape[:2] != temp_face.shape[:2]:
                enhanced_face = cv2.resize(
                    enhanced_face,
                    (temp_face.shape[1], temp_face.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

            temp_frame[start_y:end_y, start_x:end_x] = enhanced_face

        except Exception as e:
            print(f"[WARNING] Enhance face failed: {e}")

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhance semua wajah di frame:
    - Deteksi wajah via get_many_faces (buffalo_l)
    - Skip wajah occluded via detect_occlusion()
    """
    many_faces = get_many_faces(temp_frame)
    if not many_faces:
        return temp_frame

    for target_face in many_faces:
        # ✅ Occlusion-aware: skip wajah dengan det_score rendah / tertutup
        try:
            if detect_occlusion(target_face):
                continue
        except Exception:
            # Kalau detect_occlusion error, lanjut saja (fallback)
            pass

        temp_frame = enhance_face(target_face, temp_frame)

    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Proses batch frame (dipanggil core.process_video).
    """
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            continue

        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)

        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Mode image ke image.
    """
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Mode video:
    - core.process_video akan handle multi-thread frame
    - enhancer hanya fokus ke per-frame processing
    """
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
