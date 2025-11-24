import cv2
import threading
import numpy as np
import onnxruntime as ort
from typing import Any, List, Callable

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
MODEL_URL = 'https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GPEN-BFR-512.onnx'
MODEL_NAME = 'GPEN-BFR-512.onnx'


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def get_face_enhancer() -> Any:
    """Lazy-load ONNX GPEN model (auto-download jika belum ada).

    Menggunakan roop.globals.execution_providers untuk memilih provider ONNXRuntime.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_dir = resolve_relative_path('../models')
            model_path = resolve_relative_path(f"../models/{MODEL_NAME}")

            # pastikan folder models ada dan file ter-download
            conditional_download(model_dir, [MODEL_URL])

            # buat session ONNXRuntime dengan provider yang sama seperti roop
            FACE_ENHANCER = ort.InferenceSession(
                model_path,
                providers=roop.globals.execution_providers
            )
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [MODEL_URL])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def _prepare_input(img: np.ndarray) -> np.ndarray:
    """Resize ke 512x512 dan normalisasi sesuai ekspektasi GPEN (-1..1), NCHW."""
    inp = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    inp = inp.astype(np.float32) / 127.5 - 1.0
    inp = inp.transpose(2, 0, 1)[None, ...]
    return inp


def _postprocess_output(out: np.ndarray, target_size: tuple) -> np.ndarray:
    """Konversi output model (-1..1) -> uint8 HxWxC sesuai target_size."""
    out = np.clip(out, -1.0, 1.0)
    out = (out + 1.0) * 127.5
    out = out.transpose(1, 2, 0).astype(np.uint8)
    out = cv2.resize(out, target_size, interpolation=cv2.INTER_LINEAR)
    return out


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    # support both dict-like bbox or attribute
    try:
        bbox = target_face['bbox']
    except Exception:
        bbox = getattr(target_face, 'bbox', None)

    if bbox is None:
        return temp_frame

    start_x, start_y, end_x, end_y = map(int, bbox)

    # padding kecil agar transisi natural
    padding_x = int((end_x - start_x) * 0.20)
    padding_y = int((end_y - start_y) * 0.20)

    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(temp_frame.shape[1], end_x + padding_x)
    end_y = min(temp_frame.shape[0], end_y + padding_y)

    crop = temp_frame[start_y:end_y, start_x:end_x]
    if crop.size == 0:
        return temp_frame

    # prepare input
    inp = _prepare_input(crop)

    session = get_face_enhancer()
    input_name = session.get_inputs()[0].name

    with THREAD_SEMAPHORE:
        outputs = session.run(None, {input_name: inp})

    # asumsi output pertama adalah image dengan shape (1, C, H, W)
    out = outputs[0][0]

    # postprocess ke ukuran crop asli
    target_w = end_x - start_x
    target_h = end_y - start_y
    out = _postprocess_output(out, (target_w, target_h))

    # === Seamless cloning (Poisson blending) ===
    try:
        # mask harus single channel uint8
        mask = 255 * np.ones((out.shape[0], out.shape[1]), dtype=np.uint8)

        center_x = int(start_x + out.shape[1] // 2)
        center_y = int(start_y + out.shape[0] // 2)

        temp_frame = cv2.seamlessClone(
            out,
            temp_frame,
            mask,
            (center_x, center_y),
            cv2.NORMAL_CLONE
        )
    except Exception:
        # fallback: paste biasa kalau seamlessClone gagal
        temp_frame[start_y:end_y, start_x:end_x] = out

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
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
