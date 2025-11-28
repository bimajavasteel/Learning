#!/usr/bin/env python3

# face-swapper (modifikasi: age-based wrinkle augmentation)
# Basis & banyak fungsi asli berasal dari implementasi ROOP yang kamu miliki.
# Referensi asli: face_swapper (buffalo_l pipeline). :contentReference[oaicite:3]{index=3}

from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np
import os

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    find_similar_face,   # masih disediakan kalau mau fallback
    smart_face_tracking, # tracking pintar
    detect_occlusion,    # kini pakai frame untuk occluder
    get_face_pose,       # pose (pitch, yaw, roll)
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# tambahan untuk age model
import onnxruntime as ort

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

# =========================
# AGE-BASED WRINKLE MODULE
# =========================

AGE_SESSION: Any = None
AGE_INPUT_NAME: str | None = None

def _load_age_model() -> Any:
    """
    Lazy load genderage.onnx (model harus diletakkan di ../models/genderage.onnx).
    Tidak memodifikasi face_analyser; model dipanggil terpisah.
    """
    global AGE_SESSION, AGE_INPUT_NAME
    if AGE_SESSION is not None:
        return AGE_SESSION

    model_rel = getattr(roop.globals, "genderage_model_path", "../models/genderage.onnx")
    model_path = resolve_relative_path(model_rel)

    if not os.path.exists(model_path):
        # Tidak fatal — hanya log warning dan lanjut tanpa age-estimation.
        print(f"[face-swapper] genderage model not found at {model_path}, age estimation disabled.")
        return None

    try:
        AGE_SESSION = ort.InferenceSession(model_path, providers=roop.globals.execution_providers)
        AGE_INPUT_NAME = AGE_SESSION.get_inputs()[0].name
        print(f"✅ [face-swapper] Loaded age model: {model_path}")
    except Exception as e:
        print(f"[face-swapper] Failed to load age model: {e}")
        AGE_SESSION = None
        AGE_INPUT_NAME = None

    return AGE_SESSION


def estimate_age_from_face(face: Face, frame: np.ndarray) -> int:
    """
    Kembalikan estimasi umur (int). Jika gagal atau model tidak tersedia -> -1.
    Asumsi model mengembalikan array [age, ...] atau single value pada index 0.
    Sesuaikan jika genderage.onnx output berbeda.
    """
    try:
        session = _load_age_model()
        if session is None or AGE_INPUT_NAME is None:
            return -1

        x1, y1, x2, y2 = map(int, face.bbox)
        h_frame, w_frame = frame.shape[:2]
        # safety clamp
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame))

        if x2 <= x1 or y2 <= y1:
            return -1

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return -1

        # resize & normalize sesuai kebanyakan age models (112x112 / 0-1)
        inp = cv2.resize(crop, (112, 112))
        # convert BGR -> RGB
        inp = inp[:, :, ::-1].astype("float32") / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]  # NCHW

        outputs = session.run(None, {AGE_INPUT_NAME: inp})
        pred = outputs[0]

        # interpretasi output: jika scalar -> ambil pertama, jika vector -> ambil index 0
        if pred is None:
            return -1

        # pred bisa shape (1,1) atau (1,) atau (1,N). Ambil elemen pertama yang valid.
        try:
            age_val = float(pred.flatten()[0])
            age_int = int(round(age_val))
            return age_int
        except Exception:
            return -1
    except Exception:
        return -1


def apply_wrinkle_boost(crop: np.ndarray, percent: float) -> np.ndarray:
    """
    Terapkan peningkatan detail (wrinkle) ringan menggunakan filter high-pass.
    percent: fraksi (0.05 = 5%, 0.10 = 10%).
    Kembalikan crop yang diubah (dtype uint8).
    """
    try:
        if crop is None or crop.size == 0 or percent <= 0:
            return crop

        # Gaussian blur sebagai low-frequency content
        # sigma relatif ke ukuran: gunakan fixed sigma untuk stabilitas
        sigma = 3.0
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # high-pass-ish: tambah detail dengan addWeighted
        # menjaga nilai tetap dalam 0..255
        alpha = 1.0 + percent   # e.g., 1.05 atau 1.10
        beta = -percent         # e.g., -0.05 or -0.10
        enhanced = cv2.addWeighted(crop.astype("float32"), alpha, blurred.astype("float32"), beta, 0.0)
        enhanced = np.clip(enhanced, 0, 255).astype("uint8")

        # optional: sedikit unsharp masking sharpening untuk accentuate wrinkles
        # gunakan kernel kecil agar tidak menghasilkan artefak
        # kita blend sangat ringan dengan original untuk menjaga natural look
        sharpen = cv2.addWeighted(enhanced, 0.7, crop, 0.3, 0)
        return sharpen
    except Exception:
        return crop


# =========================
# ORIGINAL FACE SWAPPER
# =========================

def get_face_swapper() -> Any:
    """
    Inisialisasi model inswapper.
    Kalau nanti mau upgrade ke inswapper_256 / CSCS_256, cukup ganti path di sini.
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

    # pakai get_one_face dari face_analyser (sudah pakai buffalo_l + filter det_score)
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


# =====================================================================
#  POSE-AWARE BBOX ADJUSTMENT (ANTI MASKER / ANTI KECIL)
# =====================================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    """
    Sesuaikan bbox target berdasarkan pose:
    - yaw besar → tambah padding kiri/kanan supaya wajah tidak mengecil
    - pitch ke atas → tambah padding ke atas (dahi ikut, anti topeng lepas)
    - pitch ke bawah → tambah padding ke bawah sedikit

    face.bbox dimodifikasi in-place.
    """
    pitch, yaw, roll = get_face_pose(face)

    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    # yaw: menoleh ke samping → wajah cenderung terlihat kecil
    # tambahkan padding horizontal bertahap setelah |yaw| > 25°
    if abs(yaw) > 25.0:
        extra = (abs(yaw) - 25.0) * 0.02   # 2% per derajat di atas 25
        extra = min(extra, 0.20)          # max +20% lebar
        pad_x = w * extra

    # pitch: negatif = lihat ke atas, positif = lihat ke bawah (per definisi InsightFace)
    if pitch < -15.0:
        # melihat ke atas → tambah padding dahi
        extra = (abs(pitch) - 15.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0:
        # melihat ke bawah → dagu sedikit keluar, tambahkan bawah
        extra = (pitch - 20.0) * 0.015
        extra = min(extra, 0.18)
        pad_y_bottom = h * extra

    # hitung bbox baru
    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    # safety: jangan sampai invalid
    if nx2 <= nx1 or ny2 <= ny1:
        return

    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# =====================================================================
#  CORE SWAP
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dasar (panggil inswapper).
    Dipisah supaya mudah di-mod / patch kalau mau upgrade model.
    """
    if source_face is None or target_face is None:
        return temp_frame

    # pose-aware bbox adjust (anti masker / anti wajah kecil)
    adapt_bbox_for_pose(target_face, temp_frame.shape)

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


def _maybe_apply_age_wrinkle(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Cek umur target_face; jika memenuhi kriteria, aplikasikan wrinkle boost pada crop.
    Dipanggil sebelum swap atau sebelum enhancer (post-swap/pre-enhance lebih direkomendasikan).
    """
    try:
        age = estimate_age_from_face(target_face, temp_frame)
        if age < 0:
            return temp_frame

        wrinkle_factor = 0.0
        if 40 <= age < 50:
            wrinkle_factor = 0.05
        elif age >= 50:
            wrinkle_factor = 0.10

        if wrinkle_factor <= 0:
            return temp_frame

        x1, y1, x2, y2 = map(int, target_face.bbox)
        # clamp bbox
        h_frame, w_frame = temp_frame.shape[:2]
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame))

        if x2 <= x1 or y2 <= y1:
            return temp_frame

        crop = temp_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return temp_frame

        boosted = apply_wrinkle_boost(crop, wrinkle_factor)
        temp_frame[y1:y2, x1:x2] = boosted
        return temp_frame
    except Exception:
        return temp_frame


def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Proses 1 frame dengan strategi:
    - many_faces = True  → swap ke semua wajah yang lolos filter & tidak occluded
    - many_faces = False → cari wajah paling mirip + stabil (tracking + embedding) + pose-aware

    Modifikasi: sebelum swap/enhance kita coba apply age-based wrinkle augmentation.
    """
    if source_face is None:
        # Safety guard: sudah dicek di pre_start, tapi buat jaga-jaga.
        return temp_frame

    # MODE: banyak wajah → swap semua yang valid
    if roop.globals.many_faces:
        # pakai smart_face_tracking agar ID wajah konsisten antar frame
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            # skip wajah yang ter-occlusion berat (tangan, rambut, dsb)
            if detect_occlusion(target_face, temp_frame):
                continue

            # APPLY wrinkle berdasarkan usia sebelum swap (opsional, bisa dipindah ke pre-enhance)
            temp_frame = _maybe_apply_age_wrinkle(target_face, temp_frame)

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

    # Kalau ada reference_face (dari reference frame) → pakai embedding-based selection
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    # Kalau belum ketemu, fallback ke wajah pertama yang valid
    if best_target is None:
        best_target = valid_faces[0]

    # APPLY wrinkle berdasarkan usia sebelum swap (untuk single mode)
    temp_frame = _maybe_apply_age_wrinkle(best_target, temp_frame)

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
