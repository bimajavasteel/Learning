from typing import Any, List, Callable
import cv2
import threading
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

# ------------------------
# Konfigurasi gaya FaceFusion
# ------------------------
GFPGAN_UPSCALE = 1          # Sama seperti FaceFusion default (1x, fokus restorasi wajah)
GFPGAN_STRENGTH = 0.8       # Bobot restorasi (mendekati karakter FaceFusion)
MIN_FACE_SIZE = 64          # Minimal ukuran wajah (px) agar diproses


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(
                model_path=model_path,
                upscale=GFPGAN_UPSCALE,
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
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(
        download_directory_path,
        ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth']
    )
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def _has_valid_face(faces: List[Face], frame: Frame) -> bool:
    if not faces:
        return False

    h, w = frame.shape[:2]

    for face in faces:
        bbox = face.get('bbox')
        if bbox is None:
            continue

        # pastikan bbox selalu list 4 angka
        bbox_list = list(bbox)
        if len(bbox_list) != 4:
            continue

        start_x, start_y, end_x, end_y = map(int, bbox_list)

        # clamp agar tidak keluar frame
        start_x = max(0, min(w, start_x))
        end_x   = max(0, min(w, end_x))
        start_y = max(0, min(h, start_y))
        end_y   = max(0, min(h, end_y))

        fw = max(0, end_x - start_x)
        fh = max(0, end_y - start_y)

        if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE:
            return True

    return False



def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    # Tetap disediakan untuk kompatibilitas, namun pipeline utama
    # kini memakai enhance satu frame penuh (gaya FaceFusion).
    h, w = temp_frame.shape[:2]
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])

    padding_x = int((end_x - start_x) * 0.25)
    padding_y = int((end_y - start_y) * 0.25)

    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(w, end_x + padding_x)
    end_y = min(h, end_y + padding_y)

    if start_x >= end_x or start_y >= end_y:
        return temp_frame

    temp_face = temp_frame[start_y:end_y, start_x:end_x]

    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, restored = get_face_enhancer().enhance(
                temp_face,
                has_aligned=False,
                only_center_face=True,
                paste_back=True,
                weight=GFPGAN_STRENGTH
            )
        temp_frame[start_y:end_y, start_x:end_x] = restored

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)

    # Jika tidak ada wajah yang layak, langsung skip
    if not _has_valid_face(many_faces, temp_frame):
        return temp_frame

    # Mode utama: GFPGAN memproses 1 frame penuh sekali,
    # mirip pipeline FaceFusion (multi-face otomatis).
    with THREAD_SEMAPHORE:
        _, _, restored_frame = get_face_enhancer().enhance(
            temp_frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=GFPGAN_STRENGTH
        )

    return restored_frame


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
        return
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
