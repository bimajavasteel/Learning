from typing import Any, List, Callable
import cv2
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    find_similar_face,
    detect_occlusion
)
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference
)
from roop.typing import Frame, Face
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video
)

# ============================================================
# GLOBALS
# ============================================================

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"

# URL CSCS Swapper
CSCS256_URL = "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/cscs_256.onnx"


# ============================================================
# MODEL LOADER — FIXED FOR CSCS_256
# ============================================================

def get_face_swapper() -> Any:
    """
    Load CSCS_256 using Swapper class.
    GUARANTEED: Tidak akan memuat RetinaFace, hanya Swapper.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:

            model_path = resolve_relative_path("../models/cscs_256.onnx")

            # Wajib pakai Swapper() — get_model() TIDAK boleh dipakai!
            from insightface.model_zoo import Swapper

            FACE_SWAPPER = Swapper(
                model_file=model_path,
                session_options=None,
                providers=roop.globals.execution_providers
            )

            print("✅ [face_swapper] CSCS_256 loaded via Swapper()")

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    """
    Reset swapper model
    """
    global FACE_SWAPPER
    FACE_SWAPPER = None


# ============================================================
# INITIAL CHECKS
# ============================================================

def pre_check() -> bool:
    """
    Pastikan CSCS_256 tersedia (auto-download).
    """
    download_dir = resolve_relative_path("../models")
    conditional_download(download_dir, [CSCS256_URL])
    return True


def pre_start() -> bool:
    """
    Validasi source image + target image/video
    """
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if get_one_face(source_img) is None:
        update_status("No face in source path detected.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process() -> None:
    """
    Cleanup setelah selesai
    """
    clear_face_swapper()
    clear_face_reference()


# ============================================================
# FACE SWAP EXECUTION
# ============================================================

def swap_face(source_face: Face, target_face: Face, frame: Frame) -> Frame:
    """
    Panggil CSCS_256.get() untuk swap.
    Pastikan kedua face tidak None.
    """
    if source_face is None or target_face is None:
        return frame

    try:
        return get_face_swapper().get(
            frame,
            target_face,
            source_face,
            paste_back=True
        )
    except Exception as e:
        print(f"⚠️ [swap_face] Swap failed: {e}")
        return frame


# ============================================================
# MULTI-FACE & SINGLE-FACE LOGIC
# ============================================================

def _select_best_target(faces: List[Face], reference_face: Face) -> Face | None:
    """
    Pilih wajah target berdasarkan embedding similarity.
    """
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_dist = float("inf")

    threshold = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            dist = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue

        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_face = f

    return best_face


# ============================================================
# FRAME PROCESSING CORE
# ============================================================

def process_frame(
    source_face: Face,
    reference_face: Face,
    frame: Frame,
    frame_number: int = 0
) -> Frame:

    if source_face is None:
        return frame

    # ==========================#
    # MODE: MANY FACES (swap semua)
    # ==========================#
    if roop.globals.many_faces:

        faces = smart_face_tracking(frame, frame_number)
        if not faces:
            faces = get_many_faces(frame)

        if not faces:
            return frame

        for tf in faces:

            # skip wajah occluded (pakai occluder.onnx dari face_analyser)
            if detect_occlusion(tf, frame):
                continue

            frame = swap_face(source_face, tf, frame)

        return frame

    # ==========================#
    # MODE: SINGLE-FACE (pilih 1 terbaik)
    # ==========================#
    faces = smart_face_tracking(frame, frame_number)
    if not faces:
        faces = get_many_faces(frame)

    if not faces:
        return frame

    valid_faces = [f for f in faces if not detect_occlusion(f, frame)]
    if not valid_faces:
        return frame

    best_target = None

    # bila ada reference face (dari reference frame)
    if reference_face is not None:
        best_target = _select_best_target(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

    frame = swap_face(source_face, best_target, frame)
    return frame


# ============================================================
# MULTI-FRAME PROCESS HANDLER
# ============================================================

def process_frames(source_path: str, frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Dipanggil Roop saat proses video.
    """
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_face_reference()

    for idx, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        out = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            frame=frame,
            frame_number=idx
        )
        cv2.imwrite(fp, out)

        if update:
            update()


# ============================================================
# IMAGE MODE
# ============================================================

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_img,
            roop.globals.reference_face_position
        )

    out = process_frame(
        source_face=source_face,
        reference_face=reference_face,
        frame=target_img,
        frame_number=0
    )
    cv2.imwrite(output_path, out)


# ============================================================
# VIDEO MODE ENTRY
# ============================================================

def process_video(source_path: str, frame_paths: List[str]) -> None:
    """
    Entry untuk mode video.
    """
    # Ambil reference face sekali di awal (single-face mode)
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            ref_frame = cv2.imread(frame_paths[ref_idx])
            ref_face = get_one_face(ref_frame, roop.globals.reference_face_position)
            set_face_reference(ref_face)
        except Exception:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        frame_paths,
        process_frames
    )
