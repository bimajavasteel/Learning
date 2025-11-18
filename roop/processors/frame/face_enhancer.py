from typing import Any, List, Callable
import cv2
import threading
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.codeformer_arch import CodeFormer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.CODEFORMER-TENSORCORE'


# =============================
#  GPU OPTIMIZATION DETECTOR
# =============================

def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


def gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return name
    return "CPU"


# =============================
#  LOAD CODEFORMER ON TENSORCORE
# =============================

def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/codeformer.pth')
            device = get_device()

            net = CodeFormer()
            ckpt = torch.load(model_path, map_location=device)
            net.load_state_dict(ckpt['params_ema'], strict=True)

            # FP16 → SUPER FAST untuk T4 / L4 / A100
            net = net.half().to(device)

            # A100 bisa compile → boost speed 20–40%
            if "A100" in gpu_info():
                net = torch.compile(net, mode="reduce-overhead")

            net.eval()
            FACE_ENHANCER = net

    return FACE_ENHANCER


# =============================
#  PREP & UTILITY
# =============================

def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check() -> bool:
    conditional_download(
        resolve_relative_path('../models'),
        ["https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"]
    )
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


# =============================
#  FACE ENHANCEMENT CORE
# =============================

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    h, w = temp_frame.shape[:2]
    x1, y1, x2, y2 = map(int, target_face['bbox'])

    fw, fh = x2 - x1, y2 - y1
    if fw <= 0 or fh <= 0:
        return temp_frame

    pad = max(0.12, min(0.32, 100 / max(fw, fh)))
    px, py = int(fw * pad), int(fh * pad)

    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)

    crop = temp_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return temp_frame

    device = get_device()
    net = get_face_enhancer()

    try:
        with THREAD_SEMAPHORE:

            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            img = img.unsqueeze(0).to(device)

            # FP16 autocast → super fast on T4/L4/A100
            img = img.half()

            with torch.cuda.amp.autocast(enabled=True):

                out = net(img, w=0.65)[0]
                out = out.clamp(0, 1)

            out_np = (out.squeeze().permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
            out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

            if out_np.shape == crop.shape:
                temp_frame[y1:y2, x1:x2] = out_np

    except Exception as e:
        print("[CodeFormer ERROR]", e)

    return temp_frame


# =============================
#  MAIN PIPELINE
# =============================

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    for face in get_many_faces(temp_frame):
        temp_frame = enhance_face(face, temp_frame)
    return temp_frame


def process_frames(source_path: str, paths: List[str], update: Callable[[], None]) -> None:
    for p in paths:
        frame = cv2.imread(p)
        frame = process_frame(None, None, frame)
        cv2.imwrite(p, frame)
        if update:
            update()


def process_image(source_path: str, target_path: str, output: str) -> None:
    img = cv2.imread(target_path)
    img = process_frame(None, None, img)
    cv2.imwrite(output, img)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
