from typing import Any, List, Callable, Optional
import cv2
import insightface
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    detect_occlusion,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

from roop.face_parsing_celebamask import (
    pre_check_face_parsing,
    get_face_mask,
)

FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


# ============================================================
#  MODEL HANDLING
# ============================================================

def get_face_swapper() -> Any:
    """
    Inisialisasi model inswapper.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
            print("✅ [face_swapper] Using inswapper_128")
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    Pastikan inswapper & model parsing tersedia.
    """
    download_directory_path = resolve_relative_path('../models')

    # inswapper
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])

    # face parsing (CelebAMaskHQ ONNX)
    pre_check_face_parsing()

    return True


def pre_start() -> bool:
    """
    Validasi path source & target.
    """
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status('No face in source path detected.', NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ============================================================
#  INTI MASKED SWAP
# ============================================================

def _is_face_occluded_basic(face: Face) -> bool:
    """
    Occlusion basic: pakai det_score dari face_analyser (cepat).
    """
    try:
        return detect_occlusion(face)
    except Exception:
        return False


def _swap_face_masked(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Swap dengan masking:
    - Jalankan inswapper → swapped full-frame
    - Hitung mask wajah (CelebAMaskHQ)
    - Blend hanya area wajah → tangan/bahu tetap asli
    """
    if source_face is None or target_face is None:
        return temp_frame

    original = temp_frame
    frame_for_swap = temp_frame.copy()

    swapped = get_face_swapper().get(
        frame_for_swap,
        target_face,
        source_face,
        paste_back=True
    )

    if swapped is None:
        return original

    # mask wajah
    try:
        mask = get_face_mask(swapped, target_face, dilate_iter=2)
    except Exception as e:
        print(f"[face_swapper] get_face_mask failed: {e}")
        mask = None

    if mask is None or mask.sum() == 0:
        # fallback: pakai swapped full
        return swapped

    h, w = original.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    mask_3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
    inv_mask_3 = 1.0 - mask_3

    blended = (original.astype(np.float32) * inv_mask_3 +
               swapped.astype(np.float32) * mask_3)

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# ============================================================
#  FRAME PROCESSING
# ============================================================

def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Proses 1 frame:
    - many_faces = True  → swap semua wajah valid, tapi hanya area wajah (masked)
    - many_faces = False → fokus 1 wajah (tracking) + masked
    """
    if source_face is None:
        return temp_frame

    # =========================
    # MODE: banyak wajah
    # =========================
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for tgt in faces:
            if _is_face_occluded_basic(tgt):
                continue
            temp_frame = _swap_face_masked(source_face, tgt, temp_frame)

        return temp_frame

    # =========================
    # MODE: single / fokus 1 wajah
    # =========================
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        faces = get_many_faces(temp_frame)

    if not faces:
        return temp_frame

    best_target: Optional[Face] = None
    for f in faces:
        if _is_face_occluded_basic(f):
            continue
        best_target = f
        break

    if best_target is None:
        return temp_frame

    temp_frame = _swap_face_masked(source_face, best_target, temp_frame)
    return temp_frame


# ============================================================
#  BATCH PROCESSING
# ============================================================

def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            continue

        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx
        )
        cv2.imwrite(temp_frame_path, result)

        if update:
            update()


# ============================================================
#  IMAGE & VIDEO ENTRY POINT
# ============================================================

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_frame,
            roop.globals.reference_face_position
        )

    result = process_frame(
        source_face=source_face,
        reference_face=reference_face,
        temp_frame=target_frame,
        frame_number=0
    )
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(
                reference_frame,
                roop.globals.reference_face_position
            )
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
