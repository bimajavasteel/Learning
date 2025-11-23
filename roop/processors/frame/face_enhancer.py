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
NAME = 'ROOP.FACE-ENHANCER'

# URL resmi pretrain CodeFormer
CODEFORMER_MODEL_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
CODEFORMER_FACE_SIZE = 512  # ukuran face crop untuk CodeFormer (seperti di script resmi)


def get_device() -> str:
    """
    Tentukan device untuk CodeFormer berdasarkan execution_providers Roop.
    """
    if 'CUDAExecutionProvider' in roop.globals.execution_providers and torch.cuda.is_available():
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        # di Mac dengan MPS
        return 'mps'
    return 'cpu'


def _load_codeformer_model(device: str) -> Any:
    """
    Load network CodeFormer dari checkpoint.
    Mengikuti konfigurasi dari script official inference_codeformer.py.
    """
    # Direktori lokal untuk menyimpan model
    model_dir = resolve_relative_path('../models/codeformer')

    # Download ckpt jika belum ada
    ckpt_path = load_file_from_url(
        url=CODEFORMER_MODEL_URL,
        model_dir=model_dir,
        progress=True
    )

    # Inisialisasi arsitektur CodeFormer via registry
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256']
    )

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)['params_ema']
    net.load_state_dict(checkpoint, strict=True)

    net.to(device)
    net.eval()
    return net


def get_face_enhancer() -> Any:
    """
    Lazy init CodeFormer (sekali saja).
    Return dict: {'net': model, 'device': device}
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            device = get_device()
            net = _load_codeformer_model(device)
            FACE_ENHANCER = {
                'net': net,
                'device': device
            }
    return FACE_ENHANCER


def clear_face_enhancer() -> None:
    """
    Reset enhancer (misalnya saat cleanup).
    """
    global FACE_ENHANCER
    FACE_ENHANCER = None
    # optional: bersihkan cache CUDA
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def pre_check() -> bool:
    """
    Pastikan model CodeFormer sudah ke-download sebelum mulai.
    (Dipanggil Roop sebelum proses.)
    """
    try:
        model_dir = resolve_relative_path('../models/codeformer')
        load_file_from_url(
            url=CODEFORMER_MODEL_URL,
            model_dir=model_dir,
            progress=True
        )
        return True
    except Exception as e:
        update_status(f'Failed to prepare CodeFormer model: {e}', NAME)
        return False


def pre_start() -> bool:
    """
    Validasi path target (image / video).
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    """
    Cleanup setelah proses selesai.
    """
    clear_face_enhancer()


def _get_bbox_from_face(target_face: Face):
    """
    Mendukung dua bentuk:
    - object dengan atribut .bbox (insightface Face)
    - dict dengan key 'bbox' (kalau masih ada kode lama)
    """
    bbox = getattr(target_face, 'bbox', None)
    if bbox is None:
        bbox = target_face['bbox']  # type: ignore[index]
    return bbox


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Restorasi 1 wajah di dalam frame menggunakan CodeFormer
    dengan anti-kotak (feathered blending).

    Langkah:
    - ambil bbox dari Face,
    - crop + padding tipis,
    - resize ke 512x512,
    - kirim ke CodeFormer,
    - hasilnya di-resize kembali ke ukuran crop,
    - blend halus dengan wajah asli (anti edge box),
    - paste balik ke frame.
    """
    try:
        bbox = _get_bbox_from_face(target_face)
        start_x, start_y, end_x, end_y = map(int, bbox)
    except Exception:
        # kalau bbox tidak valid, skip
        return temp_frame

    # padding sedikit di sekitar wajah biar natural, tapi tidak kebesaran
    padding_x = int((end_x - start_x) * 0.08)
    padding_y = int((end_y - start_y) * 0.08)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = min(temp_frame.shape[1], end_x + padding_x)
    end_y = min(temp_frame.shape[0], end_y + padding_y)

    if start_x >= end_x or start_y >= end_y:
        return temp_frame

    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size == 0:
        return temp_frame

    # simpan original crop untuk blending anti kotak
    temp_face_original = temp_face.copy()

    enhancer = get_face_enhancer()
    net = enhancer['net']
    device = enhancer['device']

    # Resize ke ukuran yang diharapkan CodeFormer
    face_h, face_w = temp_face.shape[:2]
    face_resized = cv2.resize(
        temp_face,
        (CODEFORMER_FACE_SIZE, CODEFORMER_FACE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # Konversi BGR numpy -> tensor
    face_tensor = img2tensor(face_resized / 255.0, bgr2rgb=True, float32=True)
    normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Fidelity weight: ambil dari globals kalau ada, default 0.5
    w = getattr(roop.globals, 'codeformer_fidelity', 0.5)

    restored_face = None
    with THREAD_SEMAPHORE:
        try:
            with torch.no_grad():
                # mengikuti inference_codeformer.py:
                # output = net(cropped_face_t, w=w, adain=True)[0]
                output = net(face_tensor, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        except Exception as error:
            # fallback: kalau gagal, pakai inputnya saja
            print(f'[CodeFormer] Failed inference: {error}')
            restored_face = tensor2img(face_tensor, rgb2bgr=True, min_max=(-1, 1))

    restored_face = restored_face.astype('uint8')

    # Resize balik ke ukuran crop asli
    restored_face = cv2.resize(
        restored_face,
        (face_w, face_h),
        interpolation=cv2.INTER_LINEAR
    )

    # ========= ANTI-KOTAK: FEATHERED BLENDING =========
    # mask lingkaran di tengah wajah, tepi di-blur supaya kotak hilang
    mask = np.zeros((face_h, face_w), dtype=np.float32)

    # radius sedikit lebih kecil dari min(w,h)/2 supaya tepi halus
    radius = int(min(face_h, face_w) * 0.48)
    cv2.circle(mask, (face_w // 2, face_h // 2), radius, 1.0, -1)

    # blur mask agar transisi lembut (pakai sigma, kernel otomatis)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)
    mask = np.clip(mask, 0.0, 1.0)
    mask = mask[..., None]  # (H, W, 1)

    # convert ke float untuk blending
    restored_f = restored_face.astype(np.float32)
    original_f = temp_face_original.astype(np.float32)

    blended = (restored_f * mask + original_f * (1.0 - mask)).astype('uint8')

    # paste balik hasil blend ke frame
    temp_frame[start_y:end_y, start_x:end_x] = blended
    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Proses 1 frame:
    - deteksi semua wajah,
    - apply CodeFormer + anti-kotak ke tiap wajah.
    """
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Dipanggil oleh core.process_video untuk memproses semua frame video.
    """
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Mode gambar ke gambar: restorasi semua wajah di 1 gambar.
    """
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Entry point mode video (dipanggil dari core).
    """
    roop.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
