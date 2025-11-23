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

# Enhanced tracking and occlusion handling
LAST_GOOD_SWAP = None
OCCLUSION_FALLBACK_ENABLED = True
TRACKING_CACHE = {}

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, 
                providers=roop.globals.execution_providers
            )
            print("✅ Face swapper model loaded successfully")
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER, LAST_GOOD_SWAP, TRACKING_CACHE
    FACE_SWAPPER = None
    LAST_GOOD_SWAP = None
    TRACKING_CACHE.clear()

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

def enhanced_occlusion_detection(face: Face, frame: Frame) -> bool:
    """Enhanced occlusion detection with multiple checks"""
    if not face or face.det_score < 0.4:
        return True
    
    # Check face area for reasonable size
    bbox = face.bbox
    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    frame_area = frame.shape[0] * frame.shape[1]
    
    if face_area < frame_area * 0.01:  # Face too small
        return True
        
    # Check if face is near frame edges (potential partial face)
    h, w = frame.shape[:2]
    if (bbox[0] < 10 or bbox[1] < 10 or 
        bbox[2] > w - 10 or bbox[3] > h - 10):
        return True
        
    return False

def handle_occlusion_fallback(temp_frame: Frame, swapped_frame: Frame, target_face: Face, frame_hash: str = None) -> Frame:
    """Advanced occlusion handling with frame caching"""
    global LAST_GOOD_SWAP, TRACKING_CACHE
    
    if enhanced_occlusion_detection(target_face, temp_frame):
        # Try to find cached good frame for this position
        if frame_hash and frame_hash in TRACKING_CACHE:
            cached_frame = TRACKING_CACHE[frame_hash]
            alpha = 0.8  # Strong blend with cached frame
            blended_frame = cv2.addWeighted(swapped_frame, alpha, cached_frame, 1-alpha, 0)
            print("🔄 Using cached frame for occlusion")
            return blended_frame
        elif LAST_GOOD_SWAP is not None and OCCLUSION_FALLBACK_ENABLED:
            # Blend with last good swap for smooth transition
            alpha = 0.7
            blended_frame = cv2.addWeighted(swapped_frame, alpha, LAST_GOOD_SWAP, 1-alpha, 0)
            print("🔄 Using last good swap for occlusion")
            return blended_frame
        else:
            # Return original frame if no fallback available
            print("⚠️ Occlusion detected, no fallback available")
            return temp_frame
    
    # Update last good swap and cache
    LAST_GOOD_SWAP = swapped_frame.copy()
    if frame_hash:
        TRACKING_CACHE[frame_hash] = swapped_frame.copy()
        # Keep cache size manageable
        if len(TRACKING_CACHE) > 50:
            oldest_key = next(iter(TRACKING_CACHE))
            del TRACKING_CACHE[oldest_key]
    
    return swapped_frame

def calculate_frame_hash(face: Face, frame_shape: tuple) -> str:
    """Calculate simple hash for frame position-based caching"""
    if not face:
        return "unknown"
    
    bbox = face.bbox
    # Create hash based on face position and size
    position_hash = f"{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2]-bbox[0])}_{int(bbox[3]-bbox[1])}"
    return position_hash

def optimized_swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping with enhanced error handling"""
    try:
        # Pre-check face quality
        if enhanced_occlusion_detection(target_face, temp_frame):
            print("⚠️ Poor face quality detected, attempting optimized swap")
            
        swapped_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Calculate frame hash for caching
        frame_hash = calculate_frame_hash(target_face, temp_frame.shape)
        
        # Enhanced occlusion handling
        swapped_frame = handle_occlusion_fallback(temp_frame, swapped_frame, target_face, frame_hash)
        
        return swapped_frame
        
    except Exception as e:
        print(f"❌ Face swap error: {e}")
        # Return original frame on error
        return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, frame_number: int = 0) -> Frame:
    """Process frame with enhanced tracking and occlusion handling for fast dance movements"""
    
    # Use smart tracking for better performance with fast movements
    if roop.globals.many_faces:
        many_faces = smart_face_tracking(temp_frame, frame_number)
        if many_faces:
            print(f"🎭 Processing {len(many_faces)} faces in frame {frame_number}")
            for target_face in many_faces:
                temp_frame = optimized_swap_face(source_face, target_face, temp_frame)
    else:
        # Enhanced single face matching with tracking
        target_face = find_similar_face(temp_frame, reference_face, use_tracking=True)
        if target_face:
            temp_frame = optimized_swap_face(source_face, target_face, temp_frame)
        else:
            print(f"🔍 No matching face found in frame {frame_number}")
    
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    if not source_face:
        update_status('No source face detected.', NAME)
        return
        
    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    total_frames = len(temp_frame_paths)
    print(f"🚀 Starting face swap on {total_frames} frames")
    
    for frame_number, temp_frame_path in enumerate(temp_frame_paths):
        try:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                print(f"⚠️ Could not read frame {temp_frame_path}, skipping")
                continue
                
            result = process_frame(source_face, reference_face, temp_frame, frame_number)
            cv2.imwrite(temp_frame_path, result)
            
            # Progress update
            if update and frame_number % 10 == 0:
                progress = (frame_number + 1) / total_frames * 100
                print(f"📊 Progress: {progress:.1f}% ({frame_number + 1}/{total_frames})")
                update()
                
        except Exception as e:
            print(f"❌ Error processing frame {frame_number}: {e}")
            continue
    
    print("✅ Face swap completed!")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    if not source_face:
        update_status('No source face detected.', NAME)
        return
        
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)
    print(f"✅ Image face swap saved to: {output_path}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        if reference_face:
            set_face_reference(reference_face)
            print("✅ Reference face set for video processing")
        else:
            print("⚠️ No reference face found, using first detected face")
    
    print(f"🎬 Processing video with {len(temp_frame_paths)} frames")
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def resolve_relative_path(path: str) -> str:
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
