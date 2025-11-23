from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np
import os
import shutil

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    find_similar_face,
    detect_occlusion,
)
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference,
)
from roop.typing import Frame, Face
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video,
)

FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"

# URL model CSCS256
CSCS256_URL = "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/cscs_256.onnx"


def _ensure_inswapper_alias(download_dir: str) -> None:
    """
    Pastikan file cscs_256.onnx yang ter-download
    juga tersedia sebagai inswapper_128.onnx
    supaya insightface.model_zoo.get_model()
    memuatnya sebagai model swapper, bukan RetinaFace.
    """
    cscs_path = os.path.join(download_dir, "cscs_256.onnx")
    inswapper_path = os.path.join(download_dir, "inswapper_128.onnx")

    if os.path.exists(cscs_path):
        try:
            # selalu override inswapper_128 dengan CSCS,
            # karena memang kamu ingin pakai CSCS sebagai pengganti
            shutil.copy2(cscs_path, inswapper_path)
            print("✅ [face_swapper] cscs_256.onnx -> inswapper_128.onnx alias created")
        except Exception as e:
            print(f"⚠️ [face_swapper] Failed to alias cscs_256 to inswapper_128: {e}")
    else:
        print("⚠️ [face_swapper] cscs_256.onnx not found after download")


def get_face_swapper() -> Any:
    """
    Load model swapper via insightface.model_zoo.get_model("inswapper_128.onnx").
    File inswapper_128.onnx sudah di-alias ke CSCS_256 di pre_check().
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path("../models/inswapper_128.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers,
            )
            print("✅ [face_swapper] Loaded swapper from inswapper_128.onnx (CSCS aliased)")
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    1. Download CSCS_256 ke ../models/cscs_256.onnx
    2. Copy jadi ../models/inswapper_128.onnx (alias)
    3. Supaya get_model('../models/inswapper_128.onnx') mengembalikan swapper, bukan RetinaFace.
    """
    download_dir = resolve_relative_path("../models")
    # download file cscs_256.onnx
    conditional_download(download_dir, [CSCS256_URL])
    # buat alias inswapper_128.onnx
    _ensure_inswapper_alias(download_dir)
    return True


def pre_start() -> bool:
    """
    Validasi path source & target.
    Pastikan source mengandung wajah.
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
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, frame: Frame) -> Frame:
    """
    Panggil model swapper (CSCS aliased sebagai inswapper_128).
    """
    if source_face is None or target_face is None:
        return frame

    try:
        return get_face_swapper().get(
            frame,
            target_face,
            source_face,
            paste_back=True,
        )
    except Exception as e:
        print(f"⚠️ [swap_face] Swap failed: {e}")
        return frame


def _select_best_target_by_embedding(
    faces: List[Face],
    reference_face: Face,
) -> Face | None:
    """
    Pilih wajah target terbaik berdasarkan embedding similarity.
    """
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, "normed_embedding"):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float("inf")

    similar_threshold = getattr(roop.globals, "similar_face_distance", 1.0)

    for f in faces:
        if not hasattr(f, "normed_embedding"):
            continue
        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue

        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f

    return best_face


def process_frame(
    source_face: Face,
    reference_face: Face,
    frame: Frame,
    frame_number: int = 0,
) -> Frame:
    """
    Proses 1 frame:
    - many_faces=True  → swap ke semua wajah valid & tidak occluded
    - many_faces=False → fokus 1 wajah (pakai reference + embedding)
    """
    if source_face is None:
        return frame

    # ======================
    # MODE: many faces
    # ======================
    if roop.globals.many_faces:
        faces = smart_face_tracking(frame, frame_number)
        if not faces:
            faces = get_many_faces(frame)

        if not faces:
            return frame

        for target_face in faces:
            # occlusion-aware (pakai occluder dari face_analyser)
            if detect_occlusion(target_face, frame):
                continue

            frame = swap_face(source_face, target_face, frame)

        return frame

    # ======================
    # MODE: single face
    # ======================
    faces = smart_face_tracking(frame, frame_number)
    if not faces:
        faces = get_many_faces(frame)

    if not faces:
        return frame

    # filter occluded faces
    valid_faces = [f for f in faces if not detect_occlusion(f, frame)]
    if not valid_faces:
        return frame

    best_target: Face | None = None

    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

    frame = swap_face(source_face, best_target, frame)
    return frame


def process_frames(
    source_path: str,
    frame_paths: List[str],
    update: Callable[[], None],
) -> None:
    """
    Dipanggil core.process_video.
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
            frame_number=idx,
        )
        cv2.imwrite(fp, out)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Mode gambar ke gambar.
    """
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_img,
            roop.globals.reference_face_position,
        )

    out = process_frame(
        source_face=source_face,
        reference_face=reference_face,
        frame=target_img,
        frame_number=0,
    )
    cv2.imwrite(output_path, out)


def process_video(source_path: str, frame_paths: List[str]) -> None:
    """
    Entry point mode video.
    """
    # siapkan reference_face sekali di awal (single-face mode)
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
        process_frames,
    )
