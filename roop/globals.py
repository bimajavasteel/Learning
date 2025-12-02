from typing import List, Optional

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
# Tambahkan ini di roop/globals.py
face_enhancer_blend: float = None
log_level: str = 'error'
# Tambahkan di roop/globals.py (di bagian variabel global yang sudah ada)
wrinkle_preservation: float = 1.0  # 0.0 (none) to 2.0 (strong)
dark_circle_intensity: float = 1.0  # 0.0 to 2.0
preserve_age_texture: bool = True  # Preserve age-appropriate textures
