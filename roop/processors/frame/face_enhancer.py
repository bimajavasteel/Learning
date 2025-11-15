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


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
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
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


def calculate_smart_padding(face_bbox: List[int], frame_shape: tuple) -> tuple:
    """
    Calculate smart padding to prevent forehead cropping and maintain proportions
    """
    start_x, start_y, end_x, end_y = face_bbox
    frame_height, frame_width = frame_shape[:2]
    
    face_width = end_x - start_x
    face_height = end_y - start_y
    
    # ASYMMETRIC PADDING - lebih banyak ke atas untuk jidat/rambut
    padding_top = int(face_height * 0.45)    # 45% ke atas (jidat/rambut)
    padding_bottom = int(face_height * 0.25) # 25% ke bawah (dagu/leher)
    padding_sides = int(face_width * 0.3)    # 30% ke samping (telinga/rambut samping)
    
    # Calculate available space in frame
    available_top = start_y
    available_bottom = frame_height - end_y
    available_left = start_x
    available_right = frame_width - end_x
    
    # Adjust padding if near frame edges (prevent overflow)
    padding_top = min(padding_top, int(available_top * 0.9)) if available_top < padding_top else padding_top
    padding_bottom = min(padding_bottom, int(available_bottom * 0.9)) if available_bottom < padding_bottom else padding_bottom
    padding_left = min(padding_sides, int(available_left * 0.9)) if available_left < padding_sides else padding_sides
    padding_right = min(padding_sides, int(available_right * 0.9)) if available_right < padding_sides else padding_sides
    
    return padding_left, padding_top, padding_right, padding_bottom


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Enhanced face enhancement with smart padding to prevent forehead cropping
    """
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    
    # SMART PADDING CALCULATION
    padding_left, padding_top, padding_right, padding_bottom = calculate_smart_padding(
        [start_x, start_y, end_x, end_y], 
        temp_frame.shape
    )
    
    # APPLY SMART PADDING WITH FRAME BOUNDARY CHECK
    start_x = max(0, start_x - padding_left)
    start_y = max(0, start_y - padding_top)
    end_x = min(temp_frame.shape[1], end_x + padding_right)   # jangan lewat batas kanan frame
    end_y = min(temp_frame.shape[0], end_y + padding_bottom)  # jangan lewat batas bawah frame
    
    # Ensure minimum face size for enhancement
    if (end_y - start_y) < 50 or (end_x - start_x) < 50:
        return temp_frame  # Skip enhancement for very small faces
    
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    
    if temp_face.size and temp_face.shape[0] > 0 and temp_face.shape[1] > 0:
        try:
            with THREAD_SEMAPHORE:
                # Enhanced face with better blending
                _, _, temp_face = get_face_enhancer().enhance(
                    temp_face,
                    paste_back=True
                )
            
            # Smooth blending back to original frame
            if temp_face is not None and temp_face.size > 0:
                temp_frame[start_y:end_y, start_x:end_x] = temp_face
                
        except Exception as e:
            print(f"Face enhancement error: {e}")
            # Return original frame if enhancement fails
    
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Process frame with enhanced face detection and handling
    """
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            # Only enhance faces with sufficient size and confidence
            if target_face.get('score', 1) > 0.5:  # Minimum confidence threshold
                temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            result = process_frame(None, None, temp_frame)
            cv2.imwrite(temp_frame_path, result)
            if update:
                update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is not None:
        result = process_frame(None, None, target_frame)
        cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
