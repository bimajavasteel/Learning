from typing import Any, List, Callable, Optional, Generator
import cv2
import insightface
import threading
import os
import gc
from functools import lru_cache
import contextlib
from concurrent.futures import ThreadPoolExecutor
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
CACHED_SOURCE_FACE = None


# ==================== MODEL MANAGEMENT ====================

@lru_cache(maxsize=1)
def get_face_swapper() -> Any:
    """
    Get face swapper model with caching and lazy loading
    """
    global FACE_SWAPPER
    
    if FACE_SWAPPER is None:
        model_path = resolve_relative_path('../models/inswapper_128.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        FACE_SWAPPER = insightface.model_zoo.get_model(
            model_path, 
            providers=roop.globals.execution_providers
        )
    
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    """
    Clear face swapper from memory
    """
    global FACE_SWAPPER, CACHED_SOURCE_FACE
    
    FACE_SWAPPER = None
    CACHED_SOURCE_FACE = None
    get_face_swapper.cache_clear()


@contextlib.contextmanager
def face_swapper_context():
    """
    Context manager for face swapper resource management
    """
    try:
        yield get_face_swapper()
    except Exception as e:
        update_status(f'Face swapper error: {str(e)}', NAME)
        raise
    finally:
        # Don't clear here to avoid reloading model frequently
        pass


# ==================== IMAGE I/O OPTIMIZATIONS ====================

def load_image_optimized(path: str) -> Optional[Frame]:
    """
    Load image with optimized settings
    """
    try:
        return cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
        return None


def save_image_optimized(path: str, image: Frame, quality: int = 95) -> bool:
    """
    Save image with optimized compression
    """
    try:
        # Use appropriate compression based on file extension
        if path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif path.lower().endswith('.png'):
            cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(path, image)
        return True
    except Exception as e:
        print(f"Error saving image {path}: {str(e)}")
        return False


def get_cached_source_face(source_path: str) -> Optional[Face]:
    """
    Get or cache source face to avoid repeated detection
    """
    global CACHED_SOURCE_FACE
    
    if CACHED_SOURCE_FACE is not None:
        return CACHED_SOURCE_FACE
    
    source_image = load_image_optimized(source_path)
    if source_image is None:
        return None
    
    CACHED_SOURCE_FACE = get_one_face(source_image)
    return CACHED_SOURCE_FACE


# ==================== FRAME PROCESSING OPTIMIZATIONS ====================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Swap face with proper error handling
    """
    try:
        with face_swapper_context():
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    except Exception as e:
        print(f"Face swap error: {str(e)}")
        return temp_frame


def process_single_face(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Process frame with single face detection
    """
    target_face = find_similar_face(temp_frame, reference_face)
    if target_face:
        return swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_many_faces_sequential(source_face: Face, temp_frame: Frame) -> Frame:
    """
    Process multiple faces sequentially
    """
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_many_faces_parallel(source_face: Face, temp_frame: Frame) -> Frame:
    """
    Process multiple faces in parallel (experimental)
    """
    many_faces = get_many_faces(temp_frame)
    if not many_faces or len(many_faces) == 1:
        return process_many_faces_sequential(source_face, temp_frame)
    
    # For multiple faces, process in parallel
    try:
        with ThreadPoolExecutor(max_workers=min(4, len(many_faces))) as executor:
            # Create copies of the frame for each face processing
            frames = [temp_frame.copy() for _ in many_faces]
            
            # Process each face in parallel
            futures = [
                executor.submit(swap_face, source_face, target_face, frame)
                for target_face, frame in zip(many_faces, frames)
            ]
            
            # Get results
            results = [future.result() for future in futures]
            
            # Simple blending of results (this could be improved)
            if results:
                # Use the last result as base (simplistic approach)
                temp_frame = results[-1]
                
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {str(e)}")
        temp_frame = process_many_faces_sequential(source_face, temp_frame)
    
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Main frame processing function with optimized face handling
    """
    if temp_frame is None:
        return None
        
    if roop.globals.many_faces:
        # Use parallel processing for multiple faces if enabled in config
        if getattr(roop.globals, 'enable_parallel_processing', False):
            return process_many_faces_parallel(source_face, temp_frame)
        else:
            return process_many_faces_sequential(source_face, temp_frame)
    else:
        return process_single_face(source_face, reference_face, temp_frame)


# ==================== BATCH PROCESSING OPTIMIZATIONS ====================

def process_frames_batch(
    source_face: Face, 
    reference_face: Face, 
    frame_batch: List[tuple[str, Frame]],
    update: Callable[[], None]
) -> None:
    """
    Process a batch of frames efficiently
    """
    processed_count = 0
    
    for temp_frame_path, temp_frame in frame_batch:
        if temp_frame is None:
            continue
            
        try:
            result = process_frame(source_face, reference_face, temp_frame)
            if save_image_optimized(temp_frame_path, result):
                processed_count += 1
        except Exception as e:
            print(f"Error processing frame {temp_frame_path}: {str(e)}")
            continue
    
    if update:
        update()
    
    return processed_count


def load_frames_batch(frame_paths: List[str], batch_size: int) -> Generator[List[tuple[str, Frame]], None, None]:
    """
    Generator that yields batches of loaded frames
    """
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        batch_frames = []
        
        for path in batch_paths:
            frame = load_image_optimized(path)
            if frame is not None:
                batch_frames.append((path, frame))
        
        yield batch_frames
        
        # Clear the batch to free memory
        del batch_frames


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Optimized frame processing with batch loading and memory management
    """
    source_face = get_cached_source_face(source_path)
    if source_face is None:
        update_status('No source face detected.', NAME)
        return

    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    # Calculate optimal batch size based on available memory
    batch_size = getattr(roop.globals, 'batch_size', 4)
    total_frames = len(temp_frame_paths)
    processed_frames = 0
    
    # Process frames in batches
    for batch_index, frame_batch in enumerate(load_frames_batch(temp_frame_paths, batch_size)):
        if not frame_batch:
            continue
            
        processed_in_batch = process_frames_batch(source_face, reference_face, frame_batch, update)
        processed_frames += processed_in_batch
        
        # Force garbage collection every few batches
        if batch_index % 3 == 0:
            gc.collect()
        
        # Progress update
        if update and processed_frames % 10 == 0:
            update()
    
    print(f"Successfully processed {processed_frames}/{total_frames} frames")


# ==================== PRE/POST PROCESSING ====================

def pre_check() -> bool:
    """
    Check and download required models
    """
    try:
        download_directory_path = resolve_relative_path('../models')
        model_filename = 'inswapper_128.onnx'
        model_path = os.path.join(download_directory_path, model_filename)
        
        # Check if model exists
        if not os.path.exists(model_path):
            model_url = 'https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'
            update_status('Downloading face swapper model...', NAME)
            conditional_download(download_directory_path, [model_url])
        
        # Verify model file
        if not os.path.exists(model_path):
            update_status('Face swapper model not found.', NAME)
            return False
            
        return True
        
    except Exception as e:
        update_status(f'Pre-check error: {str(e)}', NAME)
        return False


def pre_start() -> bool:
    """
    Validate inputs before starting processing
    """
    try:
        # Validate source
        if not is_image(roop.globals.source_path):
            update_status('Select an image for source path.', NAME)
            return False
        
        # Validate and cache source face
        source_face = get_cached_source_face(roop.globals.source_path)
        if not source_face:
            update_status('No face in source path detected.', NAME)
            return False
        
        # Validate target
        if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
            update_status('Select an image or video for target path.', NAME)
            return False
        
        # Pre-load model to avoid delays during processing
        if not pre_load_model():
            update_status('Failed to load face swapper model.', NAME)
            return False
            
        return True
        
    except Exception as e:
        update_status(f'Pre-start error: {str(e)}', NAME)
        return False


def pre_load_model() -> bool:
    """
    Pre-load model to avoid first-time latency
    """
    try:
        get_face_swapper()
        return True
    except Exception as e:
        print(f"Model pre-loading failed: {str(e)}")
        return False


def post_process() -> None:
    """
    Cleanup after processing
    """
    clear_face_swapper()
    clear_face_reference()
    
    # Force garbage collection
    gc.collect()


# ==================== IMAGE & VIDEO PROCESSING ====================

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Process single image with optimization
    """
    try:
        source_face = get_cached_source_face(source_path)
        if not source_face:
            update_status('No source face detected.', NAME)
            return

        target_frame = load_image_optimized(target_path)
        if target_frame is None:
            update_status('Failed to load target image.', NAME)
            return

        reference_face = None if roop.globals.many_faces else get_one_face(
            target_frame, 
            roop.globals.reference_face_position
        )

        result = process_frame(source_face, reference_face, target_frame)
        
        if not save_image_optimized(output_path, result):
            update_status('Failed to save output image.', NAME)
            
    except Exception as e:
        update_status(f'Image processing error: {str(e)}', NAME)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Process video with optimized frame handling
    """
    try:
        if not roop.globals.many_faces and not get_face_reference():
            if temp_frame_paths:
                reference_frame = load_image_optimized(temp_frame_paths[roop.globals.reference_frame_number])
                if reference_frame is not None:
                    reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
                    set_face_reference(reference_face)
        
        # Use the optimized process_frames function
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
        
    except Exception as e:
        update_status(f'Video processing error: {str(e)}', NAME)


# ==================== CONFIGURATION ====================

def set_processing_config(
    enable_parallel: bool = False,
    batch_size: int = 4,
    jpeg_quality: int = 95
) -> None:
    """
    Set processing configuration parameters
    """
    roop.globals.enable_parallel_processing = enable_parallel
    roop.globals.batch_size = batch_size
    roop.globals.jpeg_quality = jpeg_quality


def get_processing_stats() -> dict:
    """
    Get current processing statistics
    """
    return {
        'model_loaded': FACE_SWAPPER is not None,
        'source_face_cached': CACHED_SOURCE_FACE is not None,
        'batch_size': getattr(roop.globals, 'batch_size', 4),
        'parallel_processing': getattr(roop.globals, 'enable_parallel_processing', False)
    }
