from typing import Any, List, Callable
import cv2
import threading
import numpy as np
import os
import urllib.request

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

def download_codeformer_model():
    """Download CodeFormer model manually"""
    model_path = resolve_relative_path('../models/codeformer.pth')
    model_dir = os.path.dirname(model_path)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print("Downloading CodeFormer model...")
        url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        try:
            urllib.request.urlretrieve(url, model_path)
            print("CodeFormer model downloaded successfully")
        except Exception as e:
            print(f"Error downloading CodeFormer model: {e}")
    
    return model_path

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            try:
                # Download model first
                model_path = download_codeformer_model()
                
                # Import CodeFormer
                import sys
                codeformer_path = '/opt/conda/lib/python3.10/site-packages/CodeFormer'
                if os.path.exists(codeformer_path):
                    sys.path.append(codeformer_path)
                
                from CodeFormer.basicsr.utils.download_util import load_file_from_url
                from CodeFormer.codeformer import CodeFormer
                
                # Initialize CodeFormer
                FACE_ENHANCER = CodeFormer(
                    model_path=model_path,
                    upscale=1,
                    bg_upsampler=None,
                    face_upsample=True,
                    device=get_device()
                )
                print("CodeFormer enhancer loaded successfully")
                
            except Exception as e:
                print(f"Error loading CodeFormer: {e}")
                # Fallback to simple enhancement
                FACE_ENHANCER = "simple"
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
    return True  # Download handled in get_face_enhancer

def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_enhancer()

def simple_enhance_face(face_image: Frame) -> Frame:
    """Simple enhancement fallback"""
    try:
        # Basic sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(face_image, -1, kernel)
        
        # Denoising
        denoised = cv2.bilateralFilter(sharpened, 5, 15, 15)
        
        # Contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    except Exception:
        return face_image

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    frame_height, frame_width = temp_frame.shape[:2]
    start_x, start_y, end_x, end_y = map(int, target_face.bbox)
    
    # Calculate face size
    face_w, face_h = end_x - start_x, end_y - start_y
    if face_w <= 0 or face_h <= 0:
        return temp_frame

    # Adaptive padding
    pad_ratio = max(0.1, min(0.3, 100 / max(face_w, face_h)))
    padding_x = int(face_w * pad_ratio)
    padding_y = int(face_h * pad_ratio)
    
    # Ensure within frame bounds
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(frame_width, end_x + padding_x)
    end_y = min(frame_height, end_y + padding_y)
    
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size == 0:
        return temp_frame

    with THREAD_SEMAPHORE:
        try:
            enhancer = get_face_enhancer()
            
            if enhancer != "simple" and hasattr(enhancer, 'enhance'):
                # Use CodeFormer enhancement
                try:
                    enhanced_face, _, _ = enhancer.enhance(
                        temp_face,
                        has_aligned=False,
                        only_center_face=False,
                        draw_box=False,
                        fidelity_weight=0.7
                    )
                    
                    if enhanced_face is not None and enhanced_face.size > 0:
                        if enhanced_face.shape != temp_face.shape:
                            enhanced_face = cv2.resize(enhanced_face, (temp_face.shape[1], temp_face.shape[0]))
                        temp_frame[start_y:end_y, start_x:end_x] = enhanced_face
                        
                except Exception as e:
                    print(f"CodeFormer enhancement failed, using fallback: {e}")
                    enhanced_face = simple_enhance_face(temp_face)
                    temp_frame[start_y:end_y, start_x:end_x] = enhanced_face
            else:
                # Use simple enhancement
                enhanced_face = simple_enhance_face(temp_face)
                temp_frame[start_y:end_y, start_x:end_x] = enhanced_face
                
        except Exception as e:
            print(f"[WARNING] Face enhancement failed: {e}")
    
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
