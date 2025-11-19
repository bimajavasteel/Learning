from typing import List, Optional
import queue

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
execution_threads: Optional[int] = None
log_level: str = 'error'

# ============================================
# MULTI GPU CONFIG
# ============================================
use_multi_gpu: bool = True
gpu_device_ids: List[int] = [0, 1]

# GPU job queues
gpu_job_queue = {
    0: queue.Queue(),
    1: queue.Queue()
}

# notify when worker done
gpu_result_queue = queue.Queue()

active_gpu_workers = {}

# provider builder (face-swapper & enhancer respecting device_id)
def build_execution_providers():
    providers = []
    if use_multi_gpu:
        for gpu in gpu_device_ids:
            providers.append(("CUDAExecutionProvider", {"device_id": gpu}))
    else:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers

execution_providers = build_execution_providers()
