from typing import Any, List, Callable
import cv2
import threading
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces
from roop.typing import Frame, Face
from roop.utilities import resolve_relative_path, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-ENHANCER"

CODEFORMER_MODEL_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)
CODEFORMER_FACE_SIZE = 512


# ---------------------------------------------------------------
# DEVICE SELECTION
# ---------------------------------------------------------------
def get_device() -> str:
    if "CUDAExecutionProvider" in roop.globals.execution_providers and torch.cuda.is_available():
        return "cuda"
    if "CoreMLExecutionProvider" in roop.globals.execution_providers:
        return "mps"
    return "cpu"


# ---------------------------------------------------------------
# LOAD CODEFORMER
# ---------------------------------------------------------------
def _load_codeformer_model(device: str) -> Any:
    model_dir = resolve_relative_path("../models/codeformer")

    ckpt_path = load_file_from_url(
        url=CODEFORMER_MODEL_URL,
        model_dir=model_dir,
        progress=True,
    )

    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )

    checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
    net.load_state_dict(checkpoint, strict=True)

    net.to(device)
    net.eval()
    return net


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            device = get_device()
            FACE_ENHANCER = {
                "net": _load_codeformer_model(device),
                "device": device,
            }
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    global FACE_ENHANCER
    FACE_ENHANCER = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------
# PRE CHECK
# ---------------------------------------------------------------
def pre_check() -> bool:
    try:
        model_dir = resolve_relative_path("../models/codeformer")
        load_file_from_url(CODEFORMER_MODEL_URL, model_dir, progress=True)
        return True
    except Exception as e:
        update_status(f"Failed preparing CodeFormer: {e}", NAME)
        return False


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Select image or video for target path.", NAME)
        return False
    return True


def post_process() -> None:
    clear_face_enhancer()


# ---------------------------------------------------------------
# UTIL BBOX
# ---------------------------------------------------------------
def _get_bbox(face: Face):
    b = getattr(face, "bbox", None)
    if b is None:
        b = face["bbox"]
    return list(map(int, b))


# ---------------------------------------------------------------
# COLOR MATCHING (LAB)
# ---------------------------------------------------------------
def color_match(src, ref):
    """Match warna restorasi agar nyatu dengan frame asli."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)

    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)

    # mean & std
    for (s, r) in [(src_l, ref_l), (src_a, ref_a), (src_b, ref_b)]:
        s_mean, s_std = s.mean(), s.std()
        r_mean, r_std = r.mean(), r.std()
        if s_std > 1:
            s[:] = (s - s_mean) * (r_std / s_std) + r_mean

    merged = cv2.merge([src_l, src_a, src_b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------
# GAMMA MATCHING
# ---------------------------------------------------------------
def gamma_match(img, ref):
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).mean()
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()

    if gray2 < 1:
        return img

    gamma = gray / gray2
    gamma = np.clip(gamma, 0.6, 1.4)

    look = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, look)


# ---------------------------------------------------------------
# MAIN RESTORATION
# ---------------------------------------------------------------
def enhance_face(face: Face, frame: Frame) -> Frame:
    try:
        x1, y1, x2, y2 = _get_bbox(face)
    except:
        return frame

    # padding lebih besar → hasil lebih natural
    hbox = (x2 - x1)
    vbox = (y2 - y1)
    pad_x = int(hbox * 0.28)
    pad_y = int(vbox * 0.28)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame.shape[1], x2 + pad_x)
    y2 = min(frame.shape[0], y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return frame

    face_orig = frame[y1:y2, x1:x2]
    if face_orig.size == 0:
        return frame

    enhancer = get_face_enhancer()
    net = enhancer["net"]
    device = enhancer["device"]

    H, W = face_orig.shape[:2]
    resized = cv2.resize(face_orig, (CODEFORMER_FACE_SIZE, CODEFORMER_FACE_SIZE))
    ten = img2tensor(resized / 255.0, bgr2rgb=True, float32=True)
    normalize(ten, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ten = ten.unsqueeze(0).to(device)

    w = getattr(roop.globals, "codeformer_fidelity", 0.15)

    with THREAD_SEMAPHORE:
        try:
            with torch.no_grad():
                out = net(ten, w=w, adain=True)[0]
                restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
        except:
            restored = resized.copy()

    restored = cv2.resize(restored, (W, H))

    # -------------------------------
    # COLOR MATCH + GAMMA
    # -------------------------------
    restored = color_match(restored, face_orig)
    restored = gamma_match(restored, face_orig)

    # -------------------------------
    # OVAL MASK BLENDING (ANTI KOTAK)
    # -------------------------------
    mask = np.zeros((H, W), np.float32)
    center = (W // 2, H // 2)
    axes = (int(W * 0.42), int(H * 0.50))  # proporsi wajah natural
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=35, sigmaY=35)
    mask = mask[..., None]

    blend = (restored * mask + face_orig * (1 - mask)).astype("uint8")

    frame[y1:y2, x1:x2] = blend
    return frame


# ---------------------------------------------------------------
# PROCESSORS
# ---------------------------------------------------------------
def process_frame(source_face, reference_face, frame):
    faces = get_many_faces(frame)
    if faces:
        for f in faces:
            frame = enhance_face(f, frame)
    return frame


def process_frames(source_path, paths, update):
    for p in paths:
        fr = cv2.imread(p)
        out = process_frame(None, None, fr)
        cv2.imwrite(p, out)
        if update:
            update()


def process_image(src, tgt, out):
    fr = cv2.imread(tgt)
    fr = process_frame(None, None, fr)
    cv2.imwrite(out, fr)


def process_video(src, paths):
    roop.processors.frame.core.process_video(None, paths, process_frames)
