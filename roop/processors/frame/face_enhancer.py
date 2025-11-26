# enhancer-final.py
from typing import Any, List, Callable
import cv2
import threading
import numpy as np
from gfpgan.utils import GFPGANer
import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'

def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER

def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_enhancer()

# improved local color transfer (ROI based) untuk mengurangi flicker
def local_color_transfer(src, dst):
    """
    Transfer warna dari dst->src pada area ROI kecil:
    - matching mean & std per channel agar tidak merubah portrait global
    """
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    src_mean, src_std = cv2.meanStdDev(src)
    dst_mean, dst_std = cv2.meanStdDev(dst)
    dst_std[dst_std < 1.0] = 1.0
    result = (src - src_mean.reshape((1,1,3))) * (dst_std.reshape((1,1,3)) / src_std.reshape((1,1,3))) + dst_mean.reshape((1,1,3))
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def apply_blend_and_color_match(enhanced_crop: np.ndarray, original_crop: np.ndarray, fidelity: float, swap_mask: np.ndarray) -> np.ndarray:
    try:
        h, w = original_crop.shape[:2]
        if enhanced_crop.shape[:2] != (h, w):
            enhanced_crop = cv2.resize(enhanced_crop, (w, h))
        # color transfer local: gunakan original_crop sebagai reference
        corrected = local_color_transfer(enhanced_crop, original_crop)
        # fidelity blending di area swap mask only
        mask_float = swap_mask.astype(np.float32) / 255.0
        if mask_float.ndim == 2:
            mask_float = np.expand_dims(mask_float, 2)
        # soften mask
        k = int(max(7, min(h,w) * 0.02))
        if k % 2 == 0: k += 1
        mask_float = cv2.GaussianBlur(mask_float, (k,k), 0)
        blended_expression = corrected * fidelity + original_crop * (1.0 - fidelity)
        final_result = (blended_expression * mask_float + original_crop * (1.0 - mask_float)).astype(np.uint8)
        return final_result
    except Exception as e:
        update_status(f"Error in blending: {e}", NAME)
        return original_crop

def enhance_face(target_face: Face, temp_frame: Frame, swap_mask: np.ndarray = None) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.2)
    padding_y = int((end_y - start_y) * 0.2)
    h_frame, w_frame = temp_frame.shape[:2]
    sx = max(0, start_x - padding_x); sy = max(0, start_y - padding_y)
    ex = min(w_frame, end_x + padding_x); ey = min(h_frame, end_y + padding_y)
    temp_face = temp_frame[sy:ey, sx:ex]
    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, enhanced_face = get_face_enhancer().enhance(temp_face, paste_back=True)
        blend_amount = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 0.6
        # buat swap_mask lokal kalau tidak di-pass
        if swap_mask is None:
            # default: gunakan ellipse pada crop
            mh, mw = temp_face.shape[:2]
            mask_local = np.zeros((mh, mw), dtype=np.uint8)
            cv2.ellipse(mask_local, (mw//2, mh//2), (int(mw*0.45), int(mh*0.45)), 0, 0, 360, 255, -1)
            mask_local = cv2.GaussianBlur(mask_local, (31,31), 0)
        else:
            # potong swap_mask sesuai crop
            mask_local = swap_mask[sy:ey, sx:ex]
            if mask_local is None or mask_local.size == 0:
                mh, mw = temp_face.shape[:2]
                mask_local = np.zeros((mh, mw), dtype=np.uint8)
                cv2.ellipse(mask_local, (mw//2, mh//2), (int(mw*0.45), int(mh*0.45)), 0, 0, 360, 255, -1)
                mask_local = cv2.GaussianBlur(mask_local, (31,31), 0)
        result_face = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount, swap_mask=mask_local)
        temp_frame[sy:ey, sx:ex] = result_face
    return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        # buat global swap_mask dari face-swapping stage:
        # jika pipeline tidak melewatkan mask, kita fallback ke bbox-based.
        # Asumsi: roop.globals dapat menyimpan last_swap_mask per frame (opsional)
        global_swap_mask = getattr(roop.globals, "last_swap_mask", None)
        for idx, target_face in enumerate(many_faces):
            swap_mask = None
            if global_swap_mask is not None:
                swap_mask = global_swap_mask
            temp_frame = enhance_face(target_face, temp_frame, swap_mask=swap_mask)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
