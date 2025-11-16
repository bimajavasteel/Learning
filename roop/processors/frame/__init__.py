# roop/processors/frame/__init__.py
from .core import get_frame_processors

processors = get_frame_processors()

__all__ = ['processors']
