from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face, smart_face_tracking, detect_occlusion
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

# Occlusion handling
LAST_GOOD_SWAP = None

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
        'https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'
    ])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def handle_occlusion_fallback(temp_frame: Frame, swapped_frame: Frame, target_face: Face) -> Frame:
    global LAST_GOOD_SWAP
    
    if detect_occlusion(target_face):
        if LAST_GOOD_SWAP is not None:
            # Blend dengan frame sebelumnya untuk transisi smooth
            alpha = 0.6
            blended_frame = cv2.addWeighted(swapped_frame, alpha, LAST_GOOD_SWAP, 1-alpha, 0)
            return blended_frame
        else:
            return temp_frame
    
    LAST_GOOD_SWAP = swapped_frame.copy()
    return swapped_frame

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        swapped_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Simple occlusion handling
        swapped_frame = handle_occlusion_fallback(temp_frame, swapped_frame, target_face)
        
        return swapped_frame
    except Exception as e:
        print(f"Face swap error: {e}")
        return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    if roop.globals.many_faces:
        many_faces = smart_face_tracking(temp_frame, frame_number)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face, use_tracking=True)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    for frame_number, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame, frame_number)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def resolve_relative_path(path: str) -> str:
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
