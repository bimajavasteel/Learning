# ================================================================
#   frame/core.py — FINAL (Anti Circular Import)
# ================================================================

import cv2
import numpy as np
import traceback
from typing import List, Callable

import roop.globals
from roop.typing import Frame, Face
from roop.face_analyser import get_many_faces
from roop.utilities import is_image


# ================================================================
#   IMPORT PROCESSORS
# ================================================================
from roop.processors.frame.face_swapper import process_frame as face_swapper_process
from roop.processors.frame.video_sharpener import process_frame as sharpener_process
from roop.processors.frame.face_enhancer import process_frame as face_enhancer_process


# ================================================================
#   Processor Order
# ================================================================
PROCESSORS: List[Callable] = [
    face_swapper_process,
    sharpener_process,
    face_enhancer_process
]

PROCESSOR_NAMES = [
    "ROOP.FACE-SWAPPER",
    "ROOP.VIDEO-SHARPENER",
    "ROOP.FACE-ENHANCER"
]


# ================================================================
#   Lazy import update_status (anti circular import)
# ================================================================
def _update_status(message: str):
    from roop.core import update_status   # ← SAFE
    update_status(message)


# ================================================================
#   Safe processor call
# ================================================================
def apply_processor(frame: Frame, faces: List[Face], processor: Callable, processor_name: str,
                    frame_index: int, total_frames: int) -> Frame:
    try:
        percent = (frame_index / total_frames) * 100
        _update_status(f"[{processor_name}] Progressing... {percent:.1f}%")

        return processor(frame, faces=faces, total_frames=total_frames)

    except Exception as e:
        print(f"[{processor_name}] ERROR pada frame {frame_index}: {e}")
        traceback.print_exc()
        return frame


# ================================================================
#   Process per frame
# ================================================================
def process_frame(frame: Frame, frame_index: int, total_frames: int) -> Frame:
    if frame is None:
        return frame

    # Face Detection
    try:
        faces = get_many_faces(frame)
    except Exception as e:
        print(f"[FACE-ANALYSER] Error pada frame {frame_index}: {e}")
        faces = []

    # Processor pipeline
    for processor, name in zip(PROCESSORS, PROCESSOR_NAMES):
        frame = apply_processor(frame, faces, processor, name, frame_index, total_frames)

    return frame


# ================================================================
#   Batch processing (dipanggil dari core.run)
# ================================================================
def run_frame_processors(frames: List[Frame]) -> List[Frame]:
    total = len(frames)
    processed_frames = []

    _update_status("⏳ [ROOP.FRAME-PIPELINE] Starting...")

    for idx, frame in enumerate(frames):
        out = process_frame(frame, idx, total)
        processed_frames.append(out)

    _update_status("✅ [ROOP.FRAME-PIPELINE] Completed.")

    return processed_frames
