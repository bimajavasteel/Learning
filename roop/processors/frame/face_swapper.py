import threading
from typing import Any, List, Callable, Optional
import cv2
import insightface
import numpy as np
import onnxruntime
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

# =====================================================================
#  GLOBALS
# =====================================================================
FACE_SWAPPER: Any = None
FACE_PARSER: Any = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'
MASK_TEMPLATE_CACHE = {} # Cache untuk berbagai ukuran mask

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

def get_face_parser() -> Any:
    global FACE_PARSER
    with THREAD_LOCK:
        if FACE_PARSER is None:
            model_path = resolve_relative_path('../models/resnet34.onnx')
            try:
                FACE_PARSER = onnxruntime.InferenceSession(
                    model_path,
                    providers=roop.globals.execution_providers
                )
            except Exception as e:
                print(f"Warning: Gagal load resnet34.onnx ({e}).")
                FACE_PARSER = None
    return FACE_PARSER

def clear_face_swapper() -> None:
    global FACE_SWAPPER, FACE_PARSER
    FACE_SWAPPER = None
    FACE_PARSER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    conditional_download(download_directory_path, [
        'https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet34.onnx'
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    if not get_one_face(cv2.imread(roop.globals.source_path)):
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
#  HELPER FUNCTIONS
# =====================================================================

def apply_color_transfer(source_img, target_img):
    try:
        s_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        s_mean, s_std = cv2.meanStdDev(s_lab)
        t_mean, t_std = cv2.meanStdDev(t_lab)
        
        res_lab = (s_lab - s_mean.reshape((1,1,3))) * (t_std.reshape((1,1,3)) / (s_std.reshape((1,1,3)) + 1e-6)) + t_mean.reshape((1,1,3))
        res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
    except:
        return source_img

def get_simple_mask(size=128):
    # Cache mask lingkaran standar untuk performa
    if size in MASK_TEMPLATE_CACHE:
        return MASK_TEMPLATE_CACHE[size]
        
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    # Radius sedikit lebih besar agar mencakup pipi saat menoleh
    radius = (size // 2) - 4 
    cv2.circle(mask, center, radius, (1.0), -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    MASK_TEMPLATE_CACHE[size] = mask
    return mask

def create_parsing_mask(crop_frame, session):
    """
    Generate mask pintar (ResNet34). Bagus untuk poni, tapi buruk untuk pipi samping.
    """
    try:
        inp = cv2.resize(crop_frame, (512, 512))
        inp = inp.astype(np.float32) / 127.5 - 1.0
        inp = inp.transpose(2, 0, 1)[None, ...]

        inputs = {session.get_inputs()[0].name: inp}
        out = session.run(None, inputs)[0]
        parsing_map = out[0].argmax(0).astype(np.uint8)

        # Labels: 1=Skin, 2-13=Face features. Exclude hair (17).
        mask = np.zeros_like(parsing_map, dtype=np.float32)
        face_parts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        
        for part_idx in face_parts:
            mask[parsing_map == part_idx] = 1.0
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = cv2.resize(mask, (crop_frame.shape[1], crop_frame.shape[0]))
        return mask
    except Exception:
        return get_simple_mask(crop_frame.shape[0])

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)
    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1; h = y2 - y1
    pad_x = 0; pad_y_top = 0; pad_y_bottom = 0

    if abs(yaw) > 20.0:
        extra = min((abs(yaw) - 20.0) * 0.02, 0.25)
        pad_x = w * extra
    if pitch < -15.0:
        extra = min((abs(pitch) - 15.0) * 0.02, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = min((pitch - 20.0) * 0.015, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))
    if nx2 > nx1 and ny2 > ny1:
        face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def adapt_kps_for_pose(face: Face) -> None:
    pitch, yaw, roll = get_face_pose(face)
    if abs(yaw) < 20.0: return
    strength = min((abs(yaw) - 20.0) * 0.005, 0.20)
    if strength <= 0: return
    kps = face.kps; center = np.mean(kps, axis=0)
    face.kps = (kps + (kps - center) * strength).astype(np.float32)

# =====================================================================
#  CORE SWAP WITH DYNAMIC MASK MIXING
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None: return temp_frame

    # Ambil pose info
    pitch, yaw, roll = get_face_pose(target_face)
    abs_yaw = abs(yaw)

    # 1. Adjust BBox & KPS
    adapt_bbox_for_pose(target_face, temp_frame.shape)
    try:
        target_face_adj = InsightFaceObject(
            bbox=target_face.bbox.copy(), kps=target_face.kps.copy(),
            det_score=target_face.det_score, embedding=target_face.embedding
        )
        if hasattr(target_face, 'pose'): target_face_adj.pose = target_face.pose
    except:
        target_face_adj = target_face
    adapt_kps_for_pose(target_face_adj)

    # 2. Inference Inswapper
    res = get_face_swapper().get(temp_frame, target_face_adj, source_face, paste_back=False)
    if isinstance(res, tuple): bgr_fake, M = res
    else: return temp_frame 

    # 3. Color Transfer
    IM = cv2.invertAffineTransform(M)
    target_128 = cv2.warpAffine(temp_frame, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
    bgr_fake_corrected = apply_color_transfer(bgr_fake, target_128)

    # 4. [LOGIKA BARU] Dynamic Mask Selection based on Yaw
    # Tujuannya: Hindari glitch 'pipi bolong' saat menoleh
    
    parser_session = get_face_parser()
    
    # Ambil Simple Mask (Aman untuk pipi, tapi jelek untuk poni)
    mask_simple = get_simple_mask(128)
    
    # Ambil Parsing Mask (Bagus untuk poni, tapi bolong pipinya kalau miring)
    mask_parsing = mask_simple # Default fallback
    if parser_session:
        mask_parsing = create_parsing_mask(target_128, parser_session)

    # Hitung bobot mixing
    # Jika yaw < 25 derajat (depan): Prioritaskan Parsing (100%)
    # Jika yaw > 40 derajat (samping): Prioritaskan Simple (100%)
    # Di antaranya: Blend
    
    if abs_yaw < 25.0:
        final_mask_128 = mask_parsing
    elif abs_yaw > 40.0:
        final_mask_128 = mask_simple
    else:
        # Interpolasi Linear (25 -> 40)
        ratio = (abs_yaw - 25.0) / (40.0 - 25.0) # 0.0 s/d 1.0
        # Ratio 1.0 berarti Full Simple
        final_mask_128 = (mask_parsing * (1.0 - ratio)) + (mask_simple * ratio)

    # 5. Warping Final
    h_frame, w_frame = temp_frame.shape[:2]
    
    warped_face = cv2.warpAffine(bgr_fake_corrected, IM, (w_frame, h_frame), borderMode=cv2.BORDER_TRANSPARENT)
    warped_mask = cv2.warpAffine(final_mask_128, IM, (w_frame, h_frame), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    
    warped_mask = np.expand_dims(warped_mask, axis=-1)

    # 6. Blending
    temp_frame_float = temp_frame.astype(np.float32)
    warped_face_float = warped_face.astype(np.float32)

    output = temp_frame_float * (1.0 - warped_mask) + warped_face_float * warped_mask
    return output.astype(np.uint8)

def _select_best_target_by_embedding(faces: List[Face], reference_face: Face) -> Optional[Face]:
    if not faces or reference_face is None: return None
    if not hasattr(reference_face, 'normed_embedding'): return None
    ref_emb = reference_face.normed_embedding
    best_face = None; best_distance = float('inf')
    for f in faces:
        if not hasattr(f, 'normed_embedding'): continue
        dist = np.sum(np.square(f.normed_embedding - ref_emb))
        if dist < 1.0 and dist < best_distance:
            best_distance = dist; best_face = f
    return best_face

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if source_face is None: return temp_frame
    
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces: faces = get_many_faces(temp_frame)
    if not faces: return temp_frame
    
    valid_faces = [f for f in faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces: return temp_frame

    if roop.globals.many_faces:
        for target_face in valid_faces:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face) if reference_face else valid_faces[0]
        if best_target:
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
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame, 0)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        except: set_face_reference(None)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
