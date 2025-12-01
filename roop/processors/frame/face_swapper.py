#face-swpper support new
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


# =====================================================================
#  WRINKLE & AGE PRESERVATION FUNCTIONS
# =====================================================================

def extract_age_from_source(source_face: Face) -> float:
    """
    Ekstrak usia dari source face untuk menentukan intensitas wrinkle preservation
    """
    age = getattr(source_face, "age", None)
    
    if age is None:
        # Jika age tidak ada, coba prediksi dari gender/features
        gender = getattr(source_face, "gender", None)
        if gender is not None:
            # Estimasi kasar berdasarkan gender (jika ada)
            return 35.0 if gender == 1 else 30.0  # 1=Male, 0=Female di InsightFace
        return 30.0  # Default middle age
    
    return float(age)


def calculate_wrinkle_strength_from_age(age: float) -> float:
    """
    Hitung strength berdasarkan usia sumber dengan kurva non-linear
    """
    if age >= 60:
        return 1.2  # Sangat kuat untuk usia lanjut
    elif age >= 50:
        return 1.0
    elif age >= 40:
        return 0.8
    elif age >= 30:
        return 0.6
    elif age >= 25:
        return 0.4
    elif age >= 20:
        return 0.3
    elif age >= 15:
        return 0.15
    else:
        return 0.05  # Hampir tidak ada untuk anak-anak


def create_age_appropriate_mask(age: float, width: int, height: int) -> np.ndarray:
    """
    Buat mask kerutan yang sesuai dengan usia
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Area tengah wajah (selalu dapat tekstur)
    center_x, center_y = width // 2, height // 2
    face_radius = min(width, height) // 3
    
    # Base face ellipse
    cv2.ellipse(mask, (center_x, center_y), 
                (face_radius, face_radius), 0, 0, 360, 100, -1)
    
    # Tambahan area berdasarkan usia
    if age >= 40:
        # Dahi (top third)
        cv2.ellipse(mask, (center_x, center_y - face_radius//2),
                   (face_radius//2, face_radius//4), 0, 0, 360, 150, -1)
        # Garis senyum (bottom sides)
        left_smile = (center_x - face_radius//2, center_y + face_radius//3)
        right_smile = (center_x + face_radius//2, center_y + face_radius//3)
        cv2.circle(mask, left_smile, face_radius//4, 120, -1)
        cv2.circle(mask, right_smile, face_radius//4, 120, -1)
    
    if age >= 30:
        # Area bawah mata
        left_eye = (center_x - face_radius//3, center_y - face_radius//6)
        right_eye = (center_x + face_radius//3, center_y - face_radius//6)
        cv2.ellipse(mask, left_eye, (face_radius//5, face_radius//8), 
                   0, 0, 360, 180, -1)
        cv2.ellipse(mask, right_eye, (face_radius//5, face_radius//8), 
                   0, 0, 360, 180, -1)
    
    # Blur mask untuk transisi smooth
    blur_size = max(3, min(width, height) // 20)
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_size//3)
    
    return mask


def create_under_eye_mask(width: int, height: int, age: float) -> np.ndarray:
    """
    Buat mask untuk dark circle berdasarkan usia
    """
    mask = np.zeros((height, width), dtype=np.float32)
    
    center_x, center_y = width // 2, height // 2
    eye_spacing = min(width, height) // 4
    
    # Posisi mata
    left_eye_center = (center_x - eye_spacing//2, center_y - height//10)
    right_eye_center = (center_x + eye_spacing//2, center_y - height//10)
    
    # Size based on age (semakin tua, semakin besar area)
    if age >= 50:
        eye_width, eye_height = width//6, height//8
    elif age >= 40:
        eye_width, eye_height = width//7, height//9
    elif age >= 30:
        eye_width, eye_height = width//8, height//10
    else:
        eye_width, eye_height = width//10, height//12
    
    # Buat elliptical mask untuk bawah mata
    for eye_center in [left_eye_center, right_eye_center]:
        # Under-eye area (lower half of ellipse)
        cv2.ellipse(mask, eye_center, (eye_width, eye_height), 
                   0, 180, 360, 1.0, -1)
        
        # Tambahkan gradien untuk natural look
        y_start = eye_center[1]
        for y in range(y_start, min(height, y_start + eye_height)):
            # Fade out ke bawah
            fade = 1.0 - (y - y_start) / eye_height
            if fade > 0:
                x_start = max(0, eye_center[0] - eye_width)
                x_end = min(width, eye_center[0] + eye_width)
                if x_end > x_start:
                    row = mask[y, x_start:x_end]
                    mask[y, x_start:x_end] = np.maximum(row, fade * 0.7)
    
    # Blur mask
    blur_size = max(3, min(width, height) // 25)
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), blur_size//4)
    
    return mask


def preserve_wrinkle_texture(source_face: Face, target_bbox, frame: Frame, 
                            strength: float, source_age: float) -> Frame:
    """
    Preservasi tekstur kerutan dari sumber ke hasil swap
    """
    try:
        x1, y1, x2, y2 = map(int, target_bbox)
        h_frame, w_frame = frame.shape[:2]
        
        # Validasi bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame-1, x2), min(h_frame-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Crop area wajah hasil swap
        swapped_crop = frame[y1:y2, x1:x2]
        if swapped_crop.size == 0:
            return frame
        
        h_crop, w_crop = swapped_crop.shape[:2]
        
        # 1. Extract high-frequency details (texture/wrinkles)
        gray = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale detail extraction
        blurred_small = cv2.GaussianBlur(gray, (0, 0), 1.0)  # Fine details
        blurred_medium = cv2.GaussianBlur(gray, (0, 0), 2.5)  # Medium details
        blurred_large = cv2.GaussianBlur(gray, (0, 0), 4.0)  # Coarse details
        
        # Combine details dengan bobot berdasarkan usia
        if source_age >= 50:
            # Usia lanjut: lebih banyak coarse details (kerutan dalam)
            details = (gray - blurred_large) * 0.7 + (gray - blurred_medium) * 0.3
        elif source_age >= 30:
            # Dewasa: mix medium dan fine details
            details = (gray - blurred_medium) * 0.6 + (gray - blurred_small) * 0.4
        else:
            # Muda: hanya fine details
            details = (gray - blurred_small) * 0.8
        
        # Amplify details berdasarkan strength
        details_amplified = details * (1.0 + strength * 3.0)
        
        # 2. Create wrinkle mask berdasarkan area wajah
        wrinkle_mask = create_age_appropriate_mask(source_age, w_crop, h_crop)
        
        # 3. Apply dark circles jika usia > 25
        if source_age > 25:
            under_eye_mask = create_under_eye_mask(w_crop, h_crop, source_age)
            # Darken under-eye area
            darken_factor = min(strength * 0.4, 0.3)
            for c in range(3):
                swapped_crop[:,:,c] = swapped_crop[:,:,c].astype(float) * \
                                     (1.0 - under_eye_mask * darken_factor)
        
        # 4. Apply wrinkle details dengan mask
        details_3ch = cv2.cvtColor(details_amplified, cv2.COLOR_GRAY2BGR)
        mask_3ch = cv2.cvtColor(wrinkle_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Enhanced result
        enhanced = swapped_crop.astype(float) + details_3ch * mask_3ch * strength
        
        # 5. Local contrast enhancement untuk area kerutan
        if source_age > 30:
            # CLAHE untuk meningkatkan kontras lokal
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_yuv = cv2.cvtColor(np.clip(enhanced, 0, 255).astype(np.uint8), 
                                       cv2.COLOR_BGR2YUV)
            enhanced_yuv[:,:,0] = clahe.apply(enhanced_yuv[:,:,0])
            enhanced = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR).astype(float)
        
        # 6. Apply back to frame
        frame[y1:y2, x1:x2] = np.clip(enhanced, 0, 255).astype(np.uint8)
        
    except Exception as e:
        # Jika error, return frame asli tanpa modifikasi
        print(f"Wrinkle preservation error: {e}")
    
    return frame


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
#  CORE SWAP WITH WRINKLE PRESERVATION
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dasar dengan preservasi tekstur usia sumber
    """
    if source_face is None or target_face is None:
        return temp_frame

    # 1. Dapatkan usia SUMBER
    source_age = extract_age_from_source(source_face)
    
    # 2. Hitung strength berdasarkan usia sumber
    wrinkle_strength = calculate_wrinkle_strength_from_age(source_age)
    
    # 3. Pose-aware bbox adjust
    adapt_bbox_for_pose(target_face, temp_frame.shape)
    
    # 4. Lakukan swap normal
    swapped_frame = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )
    
    # 5. Apply wrinkle preservation/transfer jika usia > 20
    if source_age > 20 and wrinkle_strength > 0.1:
        # Apply global wrinkle preservation strength multiplier
        global_multiplier = getattr(roop.globals, 'wrinkle_preservation', 1.0)
        final_strength = wrinkle_strength * global_multiplier
        
        swapped_frame = preserve_wrinkle_texture(
            source_face=source_face,
            target_bbox=target_face.bbox,
            frame=swapped_frame,
            strength=final_strength,
            source_age=source_age
        )
    
    return swapped_frame


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
