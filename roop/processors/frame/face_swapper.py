from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
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
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    
    source_image = cv2.imread(roop.globals.source_path)
    source_face = get_one_face(source_image)
    
    if not source_face:
        update_status('No face in source path detected.', NAME)
        return False
    
    if hasattr(source_face, 'det_score') and source_face.det_score < 0.6:
        update_status('Source face detection score low. Use a clearer source image.', NAME)
        return False
        
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def enhance_face_blending(result_frame: Frame, original_frame: Frame, target_face: Face) -> Frame:
    """
    Improve blending between swapped face and original frame
    """
    try:
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        h, w = result_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return result_frame
        
        result_face_roi = result_frame[y1:y2, x1:x2]
        original_face_roi = original_frame[y1:y2, x1:x2]
        
        if result_face_roi.size > 0 and original_face_roi.size > 0:
            if result_face_roi.shape != original_face_roi.shape:
                result_face_roi = cv2.resize(result_face_roi, (original_face_roi.shape[1], original_face_roi.shape[0]))
            
            mask = np.ones(result_face_roi.shape[:2], dtype=np.float32)
            border = 15
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0
            mask = cv2.GaussianBlur(mask, (25, 25), 0)
            
            for c in range(3):
                result_face_roi[:, :, c] = (
                    result_face_roi[:, :, c] * mask + 
                    original_face_roi[:, :, c] * (1 - mask)
                )
            
            result_frame[y1:y2, x1:x2] = result_face_roi
            
    except Exception as e:
        print(f"Blending enhancement failed: {e}")
    
    return result_frame

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    original_frame = temp_frame.copy()
    result_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    result_frame = enhance_face_blending(result_frame, original_frame, target_face)
    return result_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if hasattr(target_face, 'det_score') and target_face.det_score > 0.5:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            if hasattr(target_face, 'det_score') and target_face.det_score > 0.5:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
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
