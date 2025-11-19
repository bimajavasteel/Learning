from typing import Any, List, Callable
import cv2
import threading
import torch
import numpy as np

from codeformer.codeformer_arch import CodeFormer
from basicsr.utils import img2tensor, tensor2img

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
THREAD_SEMAPHORE = threading.Semaphore()
NAME = "ROOP.CODEFORMER"


def get_device():
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return "cuda"
    return "cpu"


def get_face_enhancer():
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path("../models/codeformer.pth")
            device = get_device()

            net = CodeFormer(dim_emb=512, codebook_size=1024)
            ckpt = torch.load(model_path, map_location=device)
            net.load_state_dict(ckpt["params_ema"])
            net.to(device)
            net.eval()

            FACE_ENHANCER = net
    return FACE_ENHANCER


def clear_face_enhancer():
    global FACE_ENHANCER
    FACE_ENHANCER = None


def pre_check():
    conditional_download(
        resolve_relative_path("../models"),
        ["https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"]
    )
    return True


def pre_start():
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def enhance_face(target_face: Face, temp_frame: Frame):
    h, w = temp_frame.shape[:2]
    x1, y1, x2, y2 = map(int, target_face["bbox"])

    fw, fh = x2 - x1, y2 - y1
    if fw <= 0 or fh <= 0:
        return temp_frame

    pad = max(0.10, min(0.30, 100 / max(fw, fh)))
    px, py = int(fw * pad), int(fh * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    crop = temp_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return temp_frame

    device = get_device()

    try:
        with THREAD_SEMAPHORE:
            net = get_face_enhancer()

            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = img2tensor(img / 255.0).float().unsqueeze(0).to(device)

            with torch.no_grad():
                output = net(tensor, w=0.65)[0]

            result = tensor2img(output.cpu().clamp(0, 1))
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            if result.shape == crop.shape:
                temp_frame[y1:y2, x1:x2] = result

    except Exception as e:
        print("[CodeFormer ERROR]", e)

    return temp_frame


def process_frame(source_face, reference_face, frame):
    for face in get_many_faces(frame):
        frame = enhance_face(face, frame)
    return frame


def process_frames(src, paths, update):
    for p in paths:
        frame = cv2.imread(p)
        frame = process_frame(None, None, frame)
        cv2.imwrite(p, frame)
        if update:
            update()


def process_image(src, target, out):
    frame = cv2.imread(target)
    frame = process_frame(None, None, frame)
    cv2.imwrite(out, frame)


def process_video(src, frame_paths):
    roop.processors.frame.core.process_video(src, frame_paths, process_frames)
