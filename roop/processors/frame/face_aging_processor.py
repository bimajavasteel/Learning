from typing import Any, List, Callable
import cv2
import threading
import numpy as np

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces, get_one_face
from roop.typing import Face, Frame
from roop.utilities import is_image, is_video

# Import modul aging
from .face_aging import apply_aging_effects

NAME = 'ROOP.FACE-AGING'

def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:
    """
    Proses aging efek untuk satu frame
    """
    # Ambil parameter dari globals atau gunakan default
    wrinkles_intensity = getattr(roop.globals, 'wrinkles_intensity', 0.0)
    dark_circles_intensity = getattr(roop.globals, 'dark_circles_intensity', 0.0)
    age_pattern = getattr(roop.globals, 'age_pattern', 'moderate')
    
    # Jika tidak ada efek yang diaktifkan, return asli
    if wrinkles_intensity <= 0 and dark_circles_intensity <= 0:
        return temp_frame
    
    # Deteksi wajah
    many_faces = get_many_faces(temp_frame)
    if not many_faces:
        return temp_frame
    
    result = temp_frame.copy()
    
    # Apply efek ke setiap wajah
    for face in many_faces:
        result = apply_aging_effects(
            face=face,
            frame=result,
            wrinkles_intensity=wrinkles_intensity,
            dark_circles_intensity=dark_circles_intensity,
            age_pattern=age_pattern
        )
    
    return result

def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:
    """
    Proses aging untuk semua frames (video mode)
    """
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame, idx)
        cv2.imwrite(temp_frame_path, result)
        
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Proses aging untuk single image
    """
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame, 0)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Entry point untuk video processing
    """
    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )

def pre_start() -> bool:
    """
    Validasi sebelum memulai
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def pre_check() -> bool:
    """
    Tidak perlu download model tambahan
    """
    return True

def post_process() -> None:
    """
    Cleanup setelah selesai
    """
    pass
