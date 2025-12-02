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
    find_similar_face,   # masih disediakan kalau mau fallback
    smart_face_tracking, # tracking pintar
    detect_occlusion,    # kini pakai frame untuk occluder
    get_face_pose,       # pose (pitch, yaw, roll)
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


def extract_age_features(source_face_img: np.ndarray, target_face_img: np.ndarray) -> tuple:
    """
    Ekstrak fitur usia dari source dan target.
    """
    # Deteksi perbedaan usia berdasarkan tekstur kulit
    def analyze_skin_texture(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Hitung tekstur menggunakan GLCM-like features
        # Simple approach: variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        # Detect wrinkles (high frequency components)
        kernel_size = max(3, int(min(image.shape[:2]) * 0.05))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        high_freq = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
        wrinkle_strength = np.mean(high_freq)
        
        return texture_variance, wrinkle_strength
    
    source_texture, source_wrinkles = analyze_skin_texture(source_face_img)
    target_texture, target_wrinkles = analyze_skin_texture(target_face_img)
    
    # Rasio preservasi berdasarkan perbedaan usia
    age_preservation_ratio = source_wrinkles / max(target_wrinkles, 1.0)
    age_preservation_ratio = np.clip(age_preservation_ratio, 0.5, 2.0)
    
    return age_preservation_ratio


def apply_age_preserved_swap(source_face: Face, target_face: Face, temp_frame: Frame, 
                            source_face_img: np.ndarray = None) -> Frame:
    """
    Swap wajah dengan preservasi fitur usia.
    """
    if source_face is None or target_face is None:
        return temp_frame
    
    # Dapatkan crop wajah target sebelum swap
    x1, y1, x2, y2 = map(int, target_face.bbox)
    target_crop_before = temp_frame[y1:y2, x1:x2].copy()
    
    # Lakukan swap normal
    swapped_frame = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )
    
    # Dapatkan crop setelah swap
    swapped_crop = swapped_frame[y1:y2, x1:x2]
    
    # Terapkan preservasi usia jika diperlukan
    if (roop.globals.preserve_age_texture and 
        source_face_img is not None and 
        target_crop_before.size > 0):
        
        # Import fungsi dari enhancer
        from roop.processors.frame.face_enhancer import apply_age_texture_transfer
        
        # Terapkan transfer tekstur usia
        age_adjusted_crop = apply_age_texture_transfer(
            source_face=source_face_img,
            target_face=swapped_crop,
            wrinkle_preservation=roop.globals.wrinkle_preservation,
            dark_circle_intensity=roop.globals.dark_circle_intensity,
            preserve_age_texture=roop.globals.preserve_age_texture
        )
        
        # Masukkan kembali ke frame
        swapped_frame[y1:y2, x1:x2] = age_adjusted_crop
    
    return swapped_frame


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
    """
    if source_face is None:
        # Safety guard: sudah dicek di pre_start, tapi buat jaga-jaga.
        return temp_frame

    # Dapatkan crop wajah source untuk preservasi usia
    source_face_crop = None
    if hasattr(source_face, '_frame'):
        s_x1, s_y1, s_x2, s_y2 = map(int, source_face.bbox)
        source_frame = source_face._frame
        source_face_crop = source_frame[s_y1:s_y2, s_x1:s_x2]

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

            # Gunakan age-preserved swap jika diaktifkan
            if roop.globals.preserve_age_texture and source_face_crop is not None:
                temp_frame = apply_age_preserved_swap(
                    source_face, 
                    target_face, 
                    temp_frame,
                    source_face_crop
                )
            else:
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

    # Gunakan age-preserved swap jika diaktifkan
    if roop.globals.preserve_age_texture and source_face_crop is not None:
        temp_frame = apply_age_preserved_swap(
            source_face, 
            best_target, 
            temp_frame,
            source_face_crop
        )
    else:
        temp_frame = swap_face(source_face, best_target, temp_frame)
    
    return temp_frame


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
    
    # Dapatkan crop wajah source untuk preservasi usia
    source_face_crop = None
    if source_face is not None:
        s_x1, s_y1, s_x2, s_y2 = map(int, source_face.bbox)
        source_face_crop = source_img[s_y1:s_y2, s_x1:s_x2]

    # Single-face mode → pakai reference_face global yang sudah diset di process_video
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        
        # Proses dengan atau tanpa preservasi usia
        if roop.globals.preserve_age_texture and source_face_crop is not None:
            # Update source face dengan crop untuk preservasi usia
            if not hasattr(source_face, '_frame'):
                source_face._frame = source_img
            if not hasattr(source_face, '_crop'):
                source_face._crop = source_face_crop
        
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
