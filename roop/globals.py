from typing import List, Optional

# Core parameters
source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = None
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: Optional[int] = None
reference_frame_number: Optional[int] = None
similar_face_distance: Optional[float] = None
temp_frame_format: Optional[str] = None
temp_frame_quality: Optional[int] = None
output_video_encoder: Optional[str] = None
output_video_quality: Optional[int] = None
max_memory: Optional[int] = None
execution_providers: List[str] = []
execution_threads: Optional[int] = None

# Face enhancer parameters
face_enhancer_blend: Optional[float] = None

# Aging effects parameters - PERBAIKAN NAMA VARIABEL
wrinkles_intensity: float = 0.0  # 0.0 to 1.0
dark_circles_intensity: float = 0.0  # 0.0 to 1.0
age_pattern: str = 'moderate'  # 'light', 'moderate', 'heavy'

# Log level
log_level: str = 'error'
