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
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
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
#  POSE-AWARE BBOX ADJUSTMENT
# =====================================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)

    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    # Logic adaptasi padding (diperkuat sedikit untuk memberi ruang morphing)
    if abs(yaw) > 25.0:
        extra = (abs(yaw) - 25.0) * 0.04  
        extra = min(extra, 0.40)          
        pad_x = w * extra

    if pitch < -15.0:
        extra = (abs(pitch) - 15.0) * 0.03
        extra = min(extra, 0.35)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = (pitch - 20.0) * 0.025
        extra = min(extra, 0.25)
        pad_y_bottom = h * extra

    # Padding statis minimal
    pad_x += w * 0.05
    pad_y_top += h * 0.05
    pad_y_bottom += h * 0.05

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 <= nx1 or ny2 <= ny1:
        return

    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# =====================================================================
#  SHAPE MORPHING MODULE (BARU: Untuk Mengecilkan Hidung)
# =====================================================================

def apply_nose_morph(img: np.ndarray, face_landmarks: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """
    Melakukan warping lokal (pinch effect) pada area hidung.
    strength: 0.0 - 1.0 (Semakin tinggi, hidung semakin kecil/ramping).
    """
    try:
        if face_landmarks is None or len(face_landmarks) < 5:
            return img

        h, w = img.shape[:2]
        
        # Landmark index 2 adalah ujung hidung
        nose_x, nose_y = face_landmarks[2]
        
        # Radius efek berdasarkan jarak mata
        eye_dist = np.linalg.norm(face_landmarks[0] - face_landmarks[1])
        radius = eye_dist * 0.9 
        
        # Buat grid map
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Jarak pixel ke pusat hidung
        dx = map_x - nose_x
        dy = map_y - nose_y
        dist = np.sqrt(dx*dx + dy*dy)
        dist = np.maximum(dist, 1.0) # hindari div by zero

        # Masking area hidung
        mask = np.exp(-(dist**2) / (2 * (radius**2)))
        
        # Hitung pergeseran pixel (menjauh dari pusat -> pinch effect)
        warp_amount = mask * strength * (radius * 0.6)
        
        map_x += (dx / dist) * warp_amount
        map_y += (dy / dist) * warp_amount
        
        # Remap
        warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return warped_img
    except Exception as e:
        print(f"[Morph Error] {e}")
        return img


# =====================================================================
#  CORE SWAP (DIMODIFIKASI)
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Melakukan swap wajah, kemudian menerapkan shape morphing pada hasilnya.
    """
    if source_face is None or target_face is None:
        return temp_frame

    # 1. Adaptasi BBox (Anti Masker/Topeng)
    adapt_bbox_for_pose(target_face, temp_frame.shape)

    # 2. Lakukan Swap Standard
    swapped_frame = get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )

    # 3. Post-Process: Nose Morphing (Merampingkan hidung hasil swap)
    try:
        # Ambil bbox target untuk memproses area wajah saja
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        h_frm, w_frm = swapped_frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frm, x2)
        y2 = min(h_frm, y2)
        
        face_crop = swapped_frame[y1:y2, x1:x2]
        
        if face_crop.size > 0:
            # Konversi landmark global ke lokal crop
            if hasattr(target_face, 'kps'):
                local_kps = target_face.kps.copy()
                local_kps[:, 0] -= x1
                local_kps[:, 1] -= y1
                
                # Terapkan Morphing
                # Strength 0.5 cukup kuat untuk merampingkan hidung mancung
                morphed_crop = apply_nose_morph(face_crop, local_kps, strength=0.5)
                
                # Tempel kembali
                swapped_frame[y1:y2, x1:x2] = morphed_crop
                
    except Exception as e:
        # Jika error, lanjut saja dengan hasil swap biasa
        pass

    return swapped_frame


def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Optional[Face]:
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

    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            if detect_occlusion(target_face, temp_frame):
                continue
            temp_frame = swap_face(source_face, target_face, temp_frame)

        return temp_frame

    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    best_target = None
    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

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
