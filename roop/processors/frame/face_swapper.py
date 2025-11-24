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
    get_face_pose,
    get_occlusion_mask # Penting: Pastikan ini ada di face_analyser Anda
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
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
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True


def pre_start() -> bool:
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


# =====================================================================
#  HELPER: WARPING MATRIX
# =====================================================================

def get_inverse_affine(face_kps: np.ndarray, size: int = 128) -> Any:
    """
    Menghitung matriks transformasi untuk mengembalikan wajah 128x128
    ke posisi aslinya di frame video.
    """
    # Standar 5 point landmark untuk ArcFace (112x112 base)
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    # Sesuaikan ke 128x128 (tambah offset 8 pixel karena 112+16=128)
    if size == 128:
        dst_pts[:, 0] += 8.0 

    src_pts = np.array(face_kps, dtype=np.float32)

    # Estimate affine transform (Similarity) dari DST (template) ke SRC (wajah di frame)
    M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    return M


# =====================================================================
#  POSE ADJUSTMENT (Legacy Support)
# =====================================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    """
    Modifikasi bbox in-place agar mencakup dahi/dagu saat pose ekstrem.
    """
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    x1, y1, x2, y2 = face.bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    if abs(yaw) > 25.0:
        extra = (abs(yaw) - 25.0) * 0.02
        extra = min(extra, 0.20)
        pad_x = w * extra

    if pitch < -15.0: # Lihat atas
        extra = (abs(pitch) - 15.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0: # Lihat bawah
        extra = (pitch - 20.0) * 0.015
        extra = min(extra, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# =====================================================================
#  CORE SWAP LOGIC (MODIFIED FOR ANTI-HAND & SOFT BLEND)
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Melakukan swap dengan teknik Manual Warping + Masking.
    Mengatasi:
    1. Kotak samar (Hard edges) -> via Soft Masking
    2. Tangan/Objek tertimpa wajah (Hallucination) -> via Occlusion Masking
    """
    if source_face is None or target_face is None:
        return temp_frame

    # Optional: Adjust bbox agar crop model lebih enak (tidak terlalu ketat)
    adapt_bbox_for_pose(target_face, temp_frame.shape)

    # 1. Dapatkan Raw Swapped Face (128x128 pixel)
    # paste_back=False agar kita dapat raw image, bukan langsung ditempel
    bgr_fake, _ = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=False 
    )

    if bgr_fake is None:
        return temp_frame

    # 2. Hitung Matrix Transformasi
    # Matrix untuk memindahkan wajah 128x128 kembali ke posisi wajah di frame besar
    M = get_inverse_affine(target_face.kps, size=128)
    if M is None:
        return temp_frame

    h_frame, w_frame = temp_frame.shape[:2]

    # 3. Warp Wajah Baru ke Ukuran Frame
    # Hasil: Frame hitam transparan dengan wajah baru melayang di posisi yang benar
    warped_face = cv2.warpAffine(bgr_fake, M, (w_frame, h_frame), borderValue=0.0)

    # 4. Buat Mask Wajah (Soft Edge)
    # Mask putih bulat di kanvas 128x128
    face_mask = np.zeros((128, 128), dtype=np.float32)
    # Radius 60 (dari total 64) memberi sedikit margin agar tidak kotak
    cv2.circle(face_mask, (64, 64), 60, (1.0,), -1)
    # Blur pinggiran agar menyatu dengan kulit
    face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)

    # Warp mask wajah ke Ukuran Frame
    warped_mask = cv2.warpAffine(face_mask, M, (w_frame, h_frame), borderValue=0.0)

    # 5. INTEGRASI ANTI-TANGAN (Occlusion Masking)
    # Cek apakah ada tangan/objek menutupi wajah
    # Mengambil mask dari face_analyser yang sudah dimodif
    raw_occ_mask = get_occlusion_mask(target_face, temp_frame)

    if raw_occ_mask is not None:
        # raw_occ_mask adalah crop seukuran bbox. Kita perlu menaruhnya di frame penuh.
        full_occ_mask = np.zeros((h_frame, w_frame), dtype=np.float32)
        
        ox1, oy1, ox2, oy2 = map(int, target_face.bbox)
        
        # Validasi ukuran dan posisi paste
        ox1 = max(0, ox1); ox2 = min(w_frame, ox2)
        oy1 = max(0, oy1); oy2 = min(h_frame, oy2)
        
        th, tw = oy2 - oy1, ox2 - ox1
        
        if th > 0 and tw > 0:
            # Resize mask occlusion ke ukuran bbox saat ini
            resized_occ = cv2.resize(raw_occ_mask, (tw, th))
            
            # Tempel ke canvas full frame
            full_occ_mask[oy1:oy2, ox1:ox2] = resized_occ
            
            # Blur mask tangan sedikit agar transisi tidak kasar
            full_occ_mask = cv2.GaussianBlur(full_occ_mask, (15, 15), 0)
            
            # LOGIC UTAMA: Mask Akhir = Mask Wajah - Mask Tangan
            # Jika full_occ_mask bernilai 1 (tangan), hasilnya jadi 0 (transparan)
            warped_mask = warped_mask * (1.0 - full_occ_mask)

    # 6. Blending Akhir
    # Pastikan range 0.0 - 1.0
    warped_mask = np.clip(warped_mask, 0.0, 1.0)
    warped_mask = warped_mask[:, :, np.newaxis] # Tambah channel agar bisa dikali RGB

    # Formula: Pixel = (WajahBaru * Mask) + (WajahAsli * (1 - Mask))
    temp_frame[:] = (warped_face * warped_mask + temp_frame * (1.0 - warped_mask)).astype(np.uint8)

    return temp_frame


# =====================================================================
#  PROCESSORS (Sama seperti sebelumnya, memanggil swap_face baru)
# =====================================================================

def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Face | None:
    if not faces or reference_face is None:
        return None
    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')
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


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None:
        return temp_frame

    # Mode: Many Faces
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        
        if faces:
            for target_face in faces:
                # Kita tidak skip occlusion di sini, tapi membiarkan swap_face 
                # menangani masking-nya agar lebih natural (hanya bagian tangan yg tidak di-swap)
                temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame

    # Mode: Single Face
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(tracked_faces, reference_face)
    
    if best_target is None:
        best_target = tracked_faces[0]

    temp_frame = swap_face(source_face, best_target, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
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
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)

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
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)

    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
