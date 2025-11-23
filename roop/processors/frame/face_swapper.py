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
    detect_occlusion
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# BiseNet: occlusion + mask wajah
from roop.face_parsing_bisenet import (
    pre_check_bisenet,
    is_occluded_bisenet,
    get_face_mask_for_face,
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
    Kalau nanti mau upgrade ke inswapper_256, cukup ganti path di sini.
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
    Pastikan model-model yang dibutuhkan sudah ke-download sebelum mulai:
    - inswapper_128
    - BiseNet (untuk mask wajah & occlusion PRO)
    """
    download_directory_path = resolve_relative_path('../models')

    # Model face swapper
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])

    # Model BiseNet untuk occlusion PRO + face mask
    pre_check_bisenet()

    return True


def pre_start() -> bool:
    """
    Validasi path source & target sebelum proses.
    Sekaligus pastikan source punya wajah yang bisa dianalisis.
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
    """
    Bersihkan model & reference setelah selesai.
    """
    clear_face_swapper()
    clear_face_reference()


# ============================================================
#  MASKED SWAP CORE
# ============================================================

def _is_face_occluded_pro(frame: Frame, face: Face) -> bool:
    """
    Cek occlusion gabungan:
    - Basic: detect_occlusion (berdasarkan det_score dari detector)
    - PRO: is_occluded_bisenet (berdasarkan segmentasi BiseNet)
    """
    try:
        if detect_occlusion(face):
            return True
    except Exception:
        pass

    try:
        if is_occluded_bisenet(frame, face):
            return True
    except Exception as e:
        print(f"[face_swapper] BiseNet occlusion check failed: {e}")

    return False


def _swap_face_masked(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Swap dengan MASK:
    - Jalankan inswapper → hasil full-frame (swapped_frame)
    - Hitung mask wajah dengan BiseNet
    - Hanya area wajah (mask=1) yang diambil dari swapped_frame
      → tangan & bahu diambil dari frame asli (anti ikut swap)
    """
    if source_face is None or target_face is None:
        return temp_frame

    # Simpan frame original & buat salinan untuk di-swap
    original = temp_frame
    frame_for_swap = temp_frame.copy()

    # Jalankan inswapper dengan paste_back=True (hasil full image)
    swapped = get_face_swapper().get(
        frame_for_swap,
        target_face,
        source_face,
        paste_back=True
    )

    if swapped is None:
        return original

    # Dapatkan mask wajah (H x W, 0/1) dari BiseNet
    mask = None
    try:
        mask = get_face_mask_for_face(swapped, target_face, dilate_iter=2)
    except Exception as e:
        print(f"[face_swapper] get_face_mask_for_face failed: {e}")
        mask = None

    # Kalau mask gagal → fallback: pakai swapped apa adanya
    if mask is None or mask.sum() == 0:
        return swapped

    # Pastikan ukuran mask cocok
    h, w = original.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    # Bikin 3 channel mask untuk blending
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
    inv_mask_3 = 1.0 - mask_3

    # Blend: hanya area wajah dari swapped, sisanya dari original
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
    Proses 1 frame dengan strategi:
    - many_faces = True  → swap ke semua wajah valid, tapi hanya area wajah (masked)
    - many_faces = False → fokus 1 wajah (tracking + embedding) + masked
    """
    if source_face is None:
        return temp_frame

    # =====================================================
    # MODE: banyak wajah → swap semua wajah yang valid
    # =====================================================
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            if _is_face_occluded_pro(temp_frame, target_face):
                continue

            temp_frame = _swap_face_masked(source_face, target_face, temp_frame)

        return temp_frame

    # =====================================================
    # MODE: single / fokus 1 wajah
    # =====================================================

    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    valid_faces: List[Face] = []
    for f in tracked_faces:
        if _is_face_occluded_pro(temp_frame, f):
            continue
        valid_faces.append(f)

    if not valid_faces:
        return temp_frame

    # Pilih wajah utama (di sini kita pakai wajah pertama valid)
    # Kalau mau pakai reference embedding seperti versi sebelumnya, bisa ditambah lagi.
    best_target: Optional[Face] = valid_faces[0]

    temp_frame = _swap_face_masked(source_face, best_target, temp_frame)
    return temp_frame


# ============================================================
#  BATCH PROCESSING (untuk video)
# ============================================================

def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    """
    Dipanggil oleh core.process_video untuk memproses semua frame.
    """
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    # Single-face mode → pakai reference_face global kalau kamu mau extend
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
#  IMAGE MODE
# ============================================================

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Proses mode gambar ke gambar.
    """
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


# ============================================================
#  VIDEO MODE
# ============================================================

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Entry point untuk mode video.
    Sekarang referensi dipakai minimal (opsional).
    """
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
