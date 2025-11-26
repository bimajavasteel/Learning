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
from roop.blending import apply_blend_and_color_match  # Import dari file baru

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'

def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER

def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'

def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_enhancer()

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    if 'bbox' not in target_face or target_face.bbox is None:
        return temp_frame
        
    start_x, start_y, end_x, end_y = map(int, target_face.bbox)
    padding_x = int((end_x - start_x) * 0.2)
    padding_y = int((end_y - start_y) * 0.2)
    h_frame, w_frame = temp_frame.shape[:2]
    
    # Validasi bbox
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(w_frame, end_x + padding_x)
    end_y = min(h_frame, end_y + padding_y)
    
    if end_x <= start_x or end_y <= start_y:
        return temp_frame
    
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face is None or temp_face.size == 0:
        return temp_frame
    
    try:
        with THREAD_SEMAPHORE:
            _, _, enhanced_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        
        # Validasi enhanced_face
        if enhanced_face is None or enhanced_face.size == 0:
            print("Enhancement returned empty result")
            return temp_frame
        
        # 📌 AMBIL NILAI BLEND DARI GLOBAL (0.6 default jika CLI tidak diisi)
        blend_amount = roop.globals.face_enhancer_blend if roop.globals.face_enhancer_blend is not None else 0.6
        
        # Gunakan fungsi blending yang sudah diperbaiki
        result_face = apply_blend_and_color_match(enhanced_face, temp_face, fidelity=blend_amount)
        
        # Pastikan dimensi cocok sebelum menempel kembali
        if result_face.shape[:2] != (end_y - start_y, end_x - start_x):
            result_face = cv2.resize(result_face, (end_x - start_x, end_y - start_y))
        
        temp_frame[start_y:end_y, start_x:end_x] = result_face
    except Exception as e:
        print(f"Enhancement error: {str(e)}")
    
    return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None or temp_frame.size == 0:
            print(f"Failed to read frame: {temp_frame_path}")
            continue
            
        result = process_frame(None, None, temp_frame)
        if result is not None and result.size > 0:
            cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None or target_frame.size == 0:
        print(f"Failed to read target image: {target_path}")
        return
        
    result = process_frame(None, None, target_frame)
    if result is not None and result.size > 0:
        cv2.imwrite(output_path, result)
    else:
        print("Failed to enhance image")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
