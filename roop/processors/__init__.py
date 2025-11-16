# roop/processors/__init__.py
from .frame import processors as frame_processors

PROCESSORS = {
    'face_swapper': frame_processors['face_swapper'],
    # tambahkan processor lainnya jika ada
}

def get_processor(name):
    return PROCESSORS.get(name)
