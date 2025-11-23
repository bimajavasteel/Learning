from typing import Any, List, Callable
import cv2
import numpy as np
import onnxruntime as ort
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    smart_face_tracking,
    detect_occlusion
)
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference
)
from roop.typing import Frame, Face
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video,
)

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"

CSCS_URL = (
    "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/cscs_256.onnx"
)
CSCS_FILENAME = "cscs_256.onnx"

# ONNX silent logs
SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.log_severity_level = 3


# ===========================================================
# AUTO-DOWNLOAD + LOAD CSCS-256 USING ONNXRUNTIME
# ===========================================================
def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(f"../models/{CSCS_FILENAME}")

            # provider mapping
            providers = []
            if "CUDAExecutionProvider" in roop.globals.execution_providers:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            FACE_SWAPPER = ort.InferenceSession(
                model_path,
                sess_options=SESSION_OPTIONS,
                providers=providers,
            )
            print("✅ [face_swapper] Loaded CSCS_256 via ONNXRuntime")

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_dir = resolve_relative_path("../models")
    conditional_download(download_dir, [CSCS_URL])
    return True


# ===========================================================
# BASIC VALIDATION BEFORE PROCESS
# ===========================================================
def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Select image for source path.", NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status("No face in source path detected.", NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# ===========================================================
# POSE-AWARE BBOX EXPANSION
# ===========================================================
def adapt_bbox_for_pose(target_face: Face, frame: Frame) -> Face:
    bbox = getattr(target_face, "bbox", None)
    if bbox is None:
        return target_face

    pose = getattr(target_face, "pose", None)
    if pose is None:
        return target_face

    pose_vec = np.array(pose, dtype=np.float32).flatten()
    if pose_vec.size < 3:
        return target_face

    pitch, yaw, roll = pose_vec[:3]

    pose_mag = float(np.linalg.norm([pitch, yaw, roll]))
    pose_mag = np.clip(pose_mag, 0, 60)

    scale = 1.0 + (pose_mag / 60) * 0.25
    scale = np.clip(scale, 1.0, 1.25)

    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1, ny1 = cx - bw / 2, cy - bh / 2
    nx2, ny2 = cx + bw / 2, cy + bh / 2

    h, w = frame.shape[:2]
    nx1 = max(0, min(w - 1, nx1))
    ny1 = max(0, min(h - 1, ny1))
    nx2 = max(0, min(w, nx2))
    ny2 = max(0, min(h, ny2))

    target_face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)
    return target_face


# ===========================================================
# MANUAL ONNX INFERENCE FOR CSCS-256
# ===========================================================
def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if source_face is None or target_face is None:
        return temp_frame

    session = get_face_swapper()

    src_crop = source_face.face
    tgt_crop = target_face.face

    if src_crop is None or tgt_crop is None:
        return temp_frame

    src = cv2.resize(src_crop, (256, 256))
    tgt = cv2.resize(tgt_crop, (256, 256))

    inp_src = src[..., ::-1].astype(np.float32) / 255.0
    inp_tgt = tgt[..., ::-1].astype(np.float32) / 255.0

    inp_src = np.transpose(inp_src, (2, 0, 1))[None, :]
    inp_tgt = np.transpose(inp_tgt, (2, 0, 1))[None, :]

    input_names = [i.name for i in session.get_inputs()]

    out = session.run(
        None,
        {
            input_names[0]: inp_src,
            input_names[1]: inp_tgt,
        },
    )[0][0]

    swapped = (np.transpose(out, (1, 2, 0)) * 255).astype("uint8")
    swapped = swapped[..., ::-1]

    x1, y1, x2, y2 = map(int, target_face.bbox)
    swp = cv2.resize(swapped, (x2 - x1, y2 - y1))

    temp_frame[y1:y2, x1:x2] = swp
    return temp_frame


# ===========================================================
# MAIN FRAME PROCESSING
# ===========================================================
def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0,
) -> Frame:
    if source_face is None:
        return temp_frame

    # MANY FACES MODE
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)
        if not faces:
            return temp_frame

        for tface in faces:
            if detect_occlusion(tface, temp_frame):
                continue
            tface = adapt_bbox_for_pose(tface, temp_frame)
            temp_frame = swap_face(source_face, tface, temp_frame)

        return temp_frame

    # SINGLE TARGET MODE
    faces = smart_face_tracking(temp_frame, frame_number)
    if not faces:
        faces = get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    valid = [f for f in faces if not detect_occlusion(f, temp_frame)]
    if not valid:
        return temp_frame

    target_face = valid[0]

    target_face = adapt_bbox_for_pose(target_face, temp_frame)
    temp_frame = swap_face(source_face, target_face, temp_frame)

    return temp_frame


# ===========================================================
# PROCESS MULTI-FRAME (VIDEO)
# ===========================================================
def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None],
) -> None:

    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, p in enumerate(temp_frame_paths):
        frame = cv2.imread(p)

        frame = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=frame,
            frame_number=idx,
        )

        cv2.imwrite(p, frame)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    target = cv2.imread(target_path)
    result = process_frame(source_face, None, target, 0)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames,
    )
