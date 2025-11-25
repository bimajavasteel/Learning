import threading
from typing import Any, List, Callable, Optional
import copy

import cv2
import insightface
import numpy as np
from insightface.app.common import Face as InsightFaceObject 

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'
MASK_TEMPLATE = None

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
#  COLOR TRANSFER & MASKING
# =====================================================================

def apply_color_transfer(source_img, target_img):
    """
    [BARU] Mencocokkan warna wajah hasil swap (source) 
    agar sama dengan wajah asli di video (target).
    Menggunakan teknik Mean/Std deviasi di color space LAB.
    """
    try:
        # Convert ke LAB color space (L=Lightness, A/B=Color channels)
        s_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Hitung rata-rata (mean) dan sebaran (std) warna
        s_mean, s_std = cv2.meanStdDev(s_lab)
        t_mean, t_std = cv2.meanStdDev(t_lab)

        # Reshape agar bisa dikalikan matrix
        s_mean = s_mean.reshape((1, 1, 3))
        s_std = s_std.reshape((1, 1, 3))
        t_mean = t_mean.reshape((1, 1, 3))
        t_std = t_std.reshape((1, 1, 3))

        # Rumus Color Transfer: (Source - S_Mean) * (T_Std / S_Std) + T_Mean
        # Menyesuaikan kontras dan brightness source ke target
        res_lab = (s_lab - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
        
        # Clip nilai agar tidak error saat convert balik (0-255)
        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        # Jika gagal, kembalikan gambar asli
        return source_img

def get_soft_mask_template(size=128):
    global MASK_TEMPLATE
    if MASK_TEMPLATE is not None:
        return MASK_TEMPLATE
    
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    # Sedikit diperkecil radiusnya agar blending lebih seamless
    radius = (size // 2) - 10 
    cv2.circle(mask, center, radius, (1.0), -1)
    
    # Blur yang cukup kuat
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    
    MASK_TEMPLATE = mask
    return MASK_TEMPLATE

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

    if abs(yaw) > 20.0:
        extra = (abs(yaw) - 20.0) * 0.02
        extra = min(extra, 0.25)
        pad_x = w * extra

    if pitch < -15.0:
        extra = (abs(pitch) - 15.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = (pitch - 20.0) * 0.015
        extra = min(extra, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))
    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def adapt_kps_for_pose(face: Face) -> None:
    pitch, yaw, roll = get_face_pose(face)
    if abs(yaw) < 20.0:
        return
    strength = (abs(yaw) - 20.0) * 0.005
    strength = min(strength, 0.20)
    if strength <= 0: return

    kps = face.kps
    center = np.mean(kps, axis=0)
    new_kps = kps + (kps - center) * strength
    face.kps = new_kps.astype(np.float32)

# =====================================================================
#  CORE SWAP
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame

    # 1. Pose Adjustments
    adapt_bbox_for_pose(target_face, temp_frame.shape)
    
    try:
        target_face_adj = InsightFaceObject(
            bbox=target_face.bbox.copy(),
            kps=target_face.kps.copy(),
            det_score=target_face.det_score,
            embedding=target_face.embedding
        )
        if hasattr(target_face, 'landmark_2d_106'):
            target_face_adj.landmark_2d_106 = target_face.landmark_2d_106
        if hasattr(target_face, 'pose'):
            target_face_adj.pose = target_face.pose
    except:
        target_face_adj = target_face

    adapt_kps_for_pose(target_face_adj)

    # 2. Inference (Get 128x128 face)
    # paste_back=False wajib
    res = get_face_swapper().get(temp_frame, target_face_adj, source_face, paste_back=False)
    
    if isinstance(res, tuple):
        bgr_fake, M = res
    else:
        # Fallback jika library versi lama
        return get_face_swapper().get(temp_frame, target_face_adj, source_face, paste_back=True)

    # 3. [BARU] Color Matching
    # Kita ambil wajah asli (target) dengan ukuran 128x128 menggunakan Matrix M yang sama
    # Ini memberikan kita 'apa yang ada di belakang' wajah palsu
    target_128 = cv2.warpAffine(temp_frame, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
    
    # Terapkan color transfer: Ubah warna bgr_fake mengikuti target_128
    bgr_fake_corrected = apply_color_transfer(bgr_fake, target_128)

    # 4. Warping & Blending
    IM = cv2.invertAffineTransform(M)
    h_frame, w_frame = temp_frame.shape[:2]

    # Warp wajah palsu yang SUDAH dikoreksi warnanya
    warped_face = cv2.warpAffine(
        bgr_fake_corrected, IM, (w_frame, h_frame), borderMode=cv2.BORDER_TRANSPARENT
    )

    mask_template = get_soft_mask_template()
    warped_mask = cv2.warpAffine(
        mask_template, IM, (w_frame, h_frame), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    warped_mask = np.expand_dims(warped_mask, axis=-1)

    # 5. Composition
    temp_frame_float = temp_frame.astype(np.float32)
    warped_face_float = warped_face.astype(np.float32)

    # Blending
    output = temp_frame_float * (1.0 - warped_mask) + warped_face_float * warped_mask
    
    return output.astype(np.uint8)


def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Optional[Face]:
    if not faces or reference_face is None: return None
    if not hasattr(reference_face, 'normed_embedding'): return None
    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')
    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)
    for f in faces:
        if not hasattr(f, 'normed_embedding'): continue
        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except: continue
        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f
    return best_face

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None: return temp_frame
    
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces: faces = get_many_faces(temp_frame)
        if not faces: return temp_frame
        for target_face in faces:
            if detect_occlusion(target_face, temp_frame): continue
            temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame

    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces: tracked_faces = get_many_faces(temp_frame)
    if not tracked_faces: return temp_frame
    
    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces: return temp_frame

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
        result = process_frame(source_face, reference_face, temp_frame, idx)
        cv2.imwrite(temp_frame_path, result)
        if update: update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    source_face = get_one_face(source_img)
    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame, 0)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except:
            set_face_reference(None)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
