from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np  # ✅ untuk hitung jarak embedding

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    find_similar_face,   # masih disediakan kalau mau fallback
    smart_face_tracking, # ✅ tracking pintar
    detect_occlusion     # ✅ deteksi occlusion
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


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
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    Pastikan model sudah ke-download sebelum mulai.
    """
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True


def pre_start() -> bool:
    """
    Validasi path source & target sebelum proses.
    Sekaligus pastikan source punya wajah yang bisa dianalisis.
    """
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False

    # ✅ pakai get_one_face dari face_analyser (sudah pakai buffalo_l + filter det_score)
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


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dasar (panggil inswapper).
    Dipisah supaya mudah di-mod / patch kalau mau upgrade model.
    """
    if source_face is None or target_face is None:
        return temp_frame

    return get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )


def _select_best_target_by_embedding(
    faces: List[Face],
    reference_face: Face
) -> Face | None:
    """
    Pilih wajah target terbaik berdasarkan embedding similarity
    (mengikuti logika di find_similar_face, tapi dengan kontrol lebih besar).
    """
    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')

    # gunakan threshold dari roop.globals bila tersedia, else fallback
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
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
    frame_number: int = 0
) -> Frame:
    """
    Proses 1 frame dengan strategi:
    - many_faces = True  → swap ke semua wajah yang lolos filter & tidak occluded
    - many_faces = False → cari wajah paling mirip + stabil (tracking + embedding)
    """
    if source_face is None:
        # Safety guard: sudah dicek di pre_start, tapi buat jaga-jaga.
        return temp_frame

    # MODE: banyak wajah → swap semua yang valid
    if roop.globals.many_faces:
        # ✅ pakai smart_face_tracking agar ID wajah konsisten antar frame
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            # ✅ skip wajah yang ter-occlusion berat (tangan, rambut, dsb)
            if detect_occlusion(target_face):
                continue

            temp_frame = swap_face(source_face, target_face, temp_frame)

        return temp_frame

    # MODE: single / fokus 1 wajah → pakai reference + embedding matching
    # tracking pintar untuk target
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    # Filter occlusion dulu
    valid_faces = [f for f in tracked_faces if not detect_occlusion(f)]
    if not valid_faces:
        return temp_frame

    best_target = None

    # Kalau ada reference_face (dari reference frame) → pakai embedding-based selection
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    # Kalau belum ketemu, fallback ke wajah pertama yang valid
    if best_target is None:
        best_target = valid_faces[0]

    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame


def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
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

    # Single-face mode → pakai reference_face global yang sudah diset di process_video
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx
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

    # reference_face hanya dipakai kalau many_faces = False
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
    """
    Entry point untuk mode video.
    - Set face_reference sekali di awal (single-face)
    - Lalu serahkan looping frame ke core.process_video
    """
    # Untuk mode fokus 1 wajah, ambil reference_face dari frame & posisi pilihan user
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
            # Kalau gagal ambil reference, biarkan None (fallback ke first valid face per frame)
            set_face_reference(None)

    # core.process_video akan memanggil process_frames di atas
    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
