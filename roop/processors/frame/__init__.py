# roop/processors/frame/__init__.py

from .face_swapper import process_frames as process_face_swapper
from .face_enhancer import process_frames as process_face_enhancer

FRAME_PROCESSORS = {
    "face_swapper": process_face_swapper,
    "face_enhancer": process_face_enhancer,
}


