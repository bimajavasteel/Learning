# ================================================================
#   frame/core.py  —  FINAL VERSION
#   Dengan integrasi video_sharpener & progress notification penuh
# ================================================================

import cv2
import numpy as np
import traceback
from typing import List, Callable

import roop.globals
from roop.core import update_status
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
#   Processor Order (JANGAN UBAH)
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
#   Helper: Safe processor call
# ================================================================
def apply_processor(frame: Frame, faces: List[Face], processor: Callable, processor_name: str,
                    frame_index: int, total_frames: int) -> Frame:
    """
    Menjalankan processor dengan error-handling aman.
    Menyediakan informasi progress untuk video_sharpener.
    """
    try:
        # Update status
        percent = (frame_index / total_frames) * 100
        update_status(f"[{processor_name}] Progressing... {percent:.1f}%")

        # Jalankan processor (semua processor menerima argumen yg sama)
        return processor(frame, faces=faces, total_frames=total_frames)

    except Exception as e:
        print(f"[{processor_name}] ERROR pada frame {frame_index}: {e}")
        traceback.print_exc()
        return frame  # gagal → tetap lanjut tanpa crash


# ================================================================
#   Process a single frame
# ================================================================
def process_frame(frame: Frame, frame_index: int, total_frames: int) -> Frame:
    """
    Pipeline final:
      1) Face-analyse → detect faces
      2) Face-swapper
      3) Video-sharpener (enhanced)
      4) Face-enhancer
    """

    if frame is None:
        return frame

    # ------------------------------------------------------------
    # Analisa wajah
    # ------------------------------------------------------------
    try:
        faces = get_many_faces(frame)
    except Exception as e:
        print(f"[FACE-ANALYSER] Error pada frame {frame_index}: {e}")
        faces = []

    # ------------------------------------------------------------
    # Jalankan semua processor berurutan
    # ------------------------------------------------------------
    for processor, name in zip(PROCESSORS, PROCESSOR_NAMES):
        frame = apply_processor(frame, faces, processor, name, frame_index, total_frames)

    return frame


# ================================================================
#   Batch / Video Processing Entry
# ================================================================
def run_frame_processors(frames: List[Frame]) -> List[Frame]:
    """
    Menjalankan pipeline per frame.
    Dipanggil oleh core.run()
    """

    total = len(frames)
    processed_frames = []

    update_status("⏳ [ROOP.FRAME-PIPELINE] Starting...")

    for idx, frame in enumerate(frames):
        processed = process_frame(frame, idx, total)
        processed_frames.append(processed)

    update_status("✅ [ROOP.FRAME-PIPELINE] Completed.")

    return processed_frames
