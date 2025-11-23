from typing import Any, List, Callable
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
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"

# URL model CSCS 256
CSCS_256_URL = "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/cscs_256.onnx"
CSCS_256_FILENAME = "cscs_256.onnx"


def get_face_swapper() -> Any:
    """
    Inisialisasi model CSCS_256 (ONLY).
    Tidak ada fallback ke inswapper_128 atau model lain.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(f"../models/{CSCS_256_FILENAME}")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers,
            )
            print("✅ [face_swapper] Using CSCS_256 ONNX model")
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    Pastikan model CSCS_256 sudah ke-download sebelum mulai.
    """
    download_directory_path = resolve_relative_path("../models")
    # auto-download CSCS_256
    conditional_download(download_directory_path, [CSCS_256_URL])
    return True


def pre_start() -> bool:
    """
    Validasi path source & target sebelum proses.
    Sekaligus pastikan source punya wajah yang bisa dianalisis.
    """
    if not is_image(roop.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status("No face in source path detected.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process() -> None:
    """
    Bersihkan model & reference setelah selesai.
    """
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dasar (panggil CSCS_256).
    Dipisah supaya mudah di-mod / patch kalau mau upgrade model.
    """
    if source_face is None or target_face is None:
        return temp_frame

    return get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True,
    )


# =====================================================================
#  POSE-AWARE BBOX ADAPTATION
# =====================================================================

def _adapt_bbox_for_pose(target_face: Face, frame: Frame) -> Face:
    """
    Sesuaikan bbox target_face berdasarkan pose (yaw/pitch/roll) supaya:
    - Saat menengok atas / samping → area swap sedikit diperbesar.
    - Mengurangi efek “masker terlepas” & “wajah mengecil”.

    Implementasi:
    - Ambil vektor pose (3 dimensi, apapun urutannya).
    - Hitung magnitude (deviasi dari frontal).
    - Besarkan bbox hingga +25% pada pose ekstrem.
    """
    bbox = getattr(target_face, "bbox", None)
    if bbox is None:
        return target_face

    pose = getattr(target_face, "pose", None)
    if pose is None:
        return target_face

    pose_vec = np.array(pose, dtype=np.float32).reshape(-1)
    if pose_vec.size < 3:
        return target_face

    # magnitude pose (derajat) → 0 .. ~90
    pose_mag = float(np.clip(np.linalg.norm(pose_vec), 0.0, 90.0))

    # skala: 0% (frontal) → ~25% (pose ekstrem)
    scale = 1.0 + 0.25 * (pose_mag / 60.0)
    scale = float(np.clip(scale, 1.0, 1.25))

    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)

    new_bw = bw * scale
    new_bh = bh * scale

    nx1 = cx - new_bw / 2.0
    ny1 = cy - new_bh / 2.0
    nx2 = cx + new_bw / 2.0
    ny2 = cy + new_bh / 2.0

    h, w = frame.shape[:2]
    nx1 = max(0.0, min(w - 1.0, nx1))
    ny1 = max(0.0, min(h - 1.0, ny1))
    nx2 = max(0.0, min(w * 1.0, nx2))
    ny2 = max(0.0, min(h * 1.0, ny2))

    target_face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    return target_face


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
    temp_frame: Frame,
    frame_number: int = 0,
) -> Frame:
    """
    Proses 1 frame dengan strategi:
    - many_faces = True  → swap ke semua wajah yang lolos filter & tidak occluded
    - many_faces = False → cari wajah paling mirip + stabil (tracking + embedding)
    """
    if source_face is None:
        return temp_frame

    # MODE: banyak wajah → swap semua yang valid
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            # occlusion pakai occluder + det_score
            if detect_occlusion(target_face, temp_frame):
                continue

            target_face = _adapt_bbox_for_pose(target_face, temp_frame)
            temp_frame = swap_face(source_face, target_face, temp_frame)

        return temp_frame

    # MODE: single / fokus 1 wajah → pakai reference + embedding matching
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    # Filter occlusion dulu
    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    best_target = None

    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

    best_target = _adapt_bbox_for_pose(best_target, temp_frame)
    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None],
) -> None:
    """
    Dipanggil oleh core.process_video untuk memproses semua frame.
    Di sini kita pegang:
    - source_face: konstan
    - reference_face: diambil dari face_reference (single-mode)
    - frame_number: index frame → dipakai di smart_face_tracking
    """
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx,
        )
        cv2.imwrite(temp_frame_path, result)

        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Proses mode gambar ke gambar.
    Di sini tidak butuh tracking frame_number kompleks → pakai 0 saja.
    """
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_frame,
            roop.globals.reference_face_position,
        )

    result = process_frame(
        source_face=source_face,
        reference_face=reference_face,
        temp_frame=target_frame,
        frame_number=0,
    )
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Entry point untuk mode video.
    - Set face_reference sekali di awal (single-face)
    - Lalu serahkan looping frame ke core.process_video
    """
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(
                reference_frame,
                roop.globals.reference_face_position,
            )
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames,
    )
