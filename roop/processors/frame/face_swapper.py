# roop/processors/frame/face_swapper.py
"""
Face swapper module (ReSwapper backend).
- Menggunakan insightface untuk crop/landmark
- Menggunakan ReSwapper model (.pth) untuk swap pada aligned 256x256
- Hasil dipaste-back dengan blending (seamlessClone)
"""
from typing import Any, List, Callable, Optional, Tuple
import os
import cv2
import numpy as np
import threading

import torch

# coba import reswapper (user harus memastikan repo/packagenya ada)
try:
    # beberapa repo reswapper menyediakan class/func berbeda
    # common pattern: from reswapper import Reswapper
    from reswapper import ReSwapper as _ReSwapper
    _HAS_RESWAPPER = True
except Exception:
    _ReSwapper = None
    _HAS_RESWAPPER = False

# import face_analyser (lokal)
from roop import face_analyser
from roop.typing import Face, Frame  # jika typing lokal ada, else boleh ignore

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'
MODEL_PATH_HINT = '/kaggle/input/reswapper/reswapper_256-1567500.pth'


def get_face_swapper(model_path: Optional[str] = None, device: Optional[str] = None) -> Any:
    """
    Inisialisasi wrapper ReSwapper.
    - model_path: path ke .pth
    - device: 'cuda' atau 'cpu' (jika None, otomatis pilih cuda bila tersedia)
    User harus memastikan package reswapper ada, atau meletakkan implementasi swap function pada obj yang di-return.
    """
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is not None:
            return FACE_SWAPPER

        # device auto
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not _HAS_RESWAPPER:
            raise RuntimeError(
                "ReSwapper package tidak ditemukan. Silakan letakkan repo ReSwapper di PYTHONPATH "
                "atau install package 'reswapper'. Contoh (Kaggle): upload repo ke working dir dan "
                "importable sebagai 'reswapper'.\n"
                f"Model path yang diharapkan: {MODEL_PATH_HINT}"
            )

        if model_path is None:
            model_path = MODEL_PATH_HINT
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model ReSwapper tidak ditemukan: {model_path}")

        # inisialisasi model — sesuaikan konstruktor sesuai implementasi repo ReSwapper
        try:
            # asumsi: _ReSwapper(model_path, device=device) atau _ReSwapper.load_from_checkpoint(...)
            try:
                FACE_SWAPPER = _ReSwapper(model_path, device=device)
            except TypeError:
                # coba alternatif konstruktor
                FACE_SWAPPER = _ReSwapper()
                # load state dict jika ada method
                if hasattr(FACE_SWAPPER, 'load_state_dict'):
                    sd = torch.load(model_path, map_location=device)
                    if 'state_dict' in sd:
                        FACE_SWAPPER.load_state_dict(sd['state_dict'])
                    else:
                        FACE_SWAPPER.load_state_dict(sd)
                    FACE_SWAPPER.to(device)
            print(f"✅ [face_swapper] ReSwapper loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Gagal inisialisasi ReSwapper: {e}")

    return FACE_SWAPPER


# ----------------- util: align / transform -----------------
def _get_5pts_from_face(face) -> Optional[np.ndarray]:
    """
    Ambil 5-point landmark dari object face (InsightFace),
    return shape (5,2) or None.
    """
    lm = getattr(face, 'landmark_2d_106', None)
    if lm is not None:
        # fallback: ambil 5 titik utama (kiri mata, kanan mata, hidung, kiri mulut, kanan mulut)
        try:
            # indices for buffallo_l 106 landmarks common positions:
            left_eye = lm[36]   # may vary; jaga fallback
            right_eye = lm[45]
            nose = lm[30]
            left_mouth = lm[48]
            right_mouth = lm[54]
            pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
            return pts
        except Exception:
            pass
    # try 5-point if available
    lm5 = getattr(face, 'kps', None)
    if lm5 is not None:
        return np.array(lm5, dtype=np.float32)
    return None


def _estimate_norm_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transform (2x3) from src_pts -> dst_pts (both Nx2).
    """
    assert src_pts.shape == dst_pts.shape
    tfm = cv2.estimateAffinePartial2D(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2))[0]
    return tfm


def align_face(img: np.ndarray, face, size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop & align face → menghasilkan square aligned image (size x size).
    Return: aligned_img, transform_matrix(2x3), inv_transform(2x3)
    """
    pts = _get_5pts_from_face(face)
    if pts is None:
        # fallback: use bbox center
        x1, y1, x2, y2 = map(int, face.bbox)
        w = x2 - x1; h = y2 - y1
        center = np.array([x1 + w/2, y1 + h/2], dtype=np.float32)
        # create canonical points centered
        dst_pts = np.array([
            [size*0.3, size*0.35],
            [size*0.7, size*0.35],
            [size*0.5, size*0.55],
            [size*0.33, size*0.78],
            [size*0.67, size*0.78]
        ], dtype=np.float32)
        src_pts = np.array([
            [center[0]-w*0.2, center[1]-h*0.2],
            [center[0]+w*0.2, center[1]-h*0.2],
            [center[0], center[1]],
            [center[0]-w*0.2, center[1]+h*0.3],
            [center[0]+w*0.2, center[1]+h*0.3],
        ], dtype=np.float32)
    else:
        # dst canonical 5 points for aligned 256x256
        dst_pts = np.array([
            [size * 0.30, size * 0.35],
            [size * 0.70, size * 0.35],
            [size * 0.50, size * 0.54],
            [size * 0.33, size * 0.78],
            [size * 0.67, size * 0.78]
        ], dtype=np.float32)
        src_pts = pts.astype(np.float32)

    tfm = _estimate_norm_transform(src_pts, dst_pts)
    if tfm is None:
        # fallback crop center bbox
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = img[y1:y2, x1:x2]
        aligned = cv2.resize(crop, (size, size))
        inv = None
        return aligned, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), inv

    aligned = cv2.warpAffine(img, tfm, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # compute inverse transform
    inv_tfm = cv2.invertAffineTransform(tfm)
    return aligned, tfm, inv_tfm


def paste_back(orig_img: np.ndarray, swapped_img: np.ndarray, inv_tfm: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Paste swapped_img (aligned) kembali ke orig_img menggunakan inv_tfm.
    Blending memakai seamlessClone untuk hasil lebih natural.
    """
    h, w = orig_img.shape[:2]
    size = swapped_img.shape[0]
    # warp swapped back to image space
    try:
        warped = cv2.warpAffine(swapped_img, inv_tfm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    except Exception:
        # fallback simple paste at bbox center
        if bbox is None:
            return orig_img
        x1, y1, x2, y2 = bbox
        th, tw = y2 - y1, x2 - x1
        resized = cv2.resize(swapped_img, (tw, th))
        out = orig_img.copy()
        out[y1:y2, x1:x2] = resized
        return out

    # create mask from swapped alpha or face area
    gray = cv2.cvtColor(swapped_img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype('uint8') * 255
    mask_warped = cv2.warpAffine(mask, inv_tfm, (w, h), flags=cv2.INTER_LINEAR)
    mask_warped = cv2.GaussianBlur(mask_warped, (15, 15), 0)
    center = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    else:
        # fallback center of nonzero mask
        ys, xs = np.where(mask_warped > 10)
        if len(xs) == 0:
            return orig_img
        center = (int(np.mean(xs)), int(np.mean(ys)))

    try:
        output = cv2.seamlessClone(warped, orig_img, mask_warped, center, cv2.NORMAL_CLONE)
        return output
    except Exception:
        # fallback linear blend
        alpha = (mask_warped.astype(np.float32) / 255.0)[:, :, None]
        out = (warped.astype(np.float32) * alpha + orig_img.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return out


# ----------------- core swap function -----------------
def swap_face(source_face: Any, target_face: Any, temp_frame: np.ndarray, model_path: Optional[str] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Alur:
    1. ambil aligned source (256)
    2. ambil aligned target (256)
    3. panggil swapper.swap(aligned_src, aligned_tgt) -> swapped_aligned
    4. paste_back swapped_aligned ke temp_frame
    """
    if source_face is None or target_face is None:
        return temp_frame

    if model_path is None:
        model_path = MODEL_PATH_HINT

    swapper = get_face_swapper(model_path=model_path, device=device)

    # source image harus tersedia via source_face.array jika ada; fallback: require external source_img
    # Untuk integrasi ke pipeline, asumsikan source_face berasal dari image yang sama seperti di process_images
    # Jadi caller harus memberi source image path (handled di process_image / process_frames)
    # Untuk safety: ambil source image dari attribute jika ada
    source_img = getattr(source_face, 'src_img', None)
    if source_img is None:
        # kalau tidak ada, assume source_face._image tersedia atau caller menyediakan source image path
        raise RuntimeError("source_face tidak mengandung source image. Gunakan process_image/process_frames yang men-set source image pada source_face.src_img")

    aligned_src, _, _ = align_face(source_img, source_face, size=256)
    aligned_tgt, tfm, inv_tfm = align_face(temp_frame, target_face, size=256)

    # convert to model input (BGR->RGB, float, normalize) sesuai implementasi ReSwapper repo
    input_src = cv2.cvtColor(aligned_src, cv2.COLOR_BGR2RGB)
    input_tgt = cv2.cvtColor(aligned_tgt, cv2.COLOR_BGR2RGB)

    # swap via swapper API — coba beberapa method umum
    try:
        if hasattr(swapper, 'swap'):
            swapped = swapper.swap(input_src, input_tgt)  # asumsi output RGB uint8
        elif hasattr(swapper, 'forward'):
            with torch.no_grad():
                # adapt input to torch tensor if needed
                t_src = torch.from_numpy(input_src.transpose(2, 0, 1)).unsqueeze(0).float().to(next(swapper.parameters()).device) / 255.0
                t_tgt = torch.from_numpy(input_tgt.transpose(2, 0, 1)).unsqueeze(0).float().to(next(swapper.parameters()).device) / 255.0
                out = swapper.forward(t_src, t_tgt)
                # asumsi out -> [B, C, H, W] 0..1
                out_img = (out[0].cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
                swapped = out_img
        else:
            raise RuntimeError("Swapper object tidak memiliki method 'swap' atau 'forward'. Sesuaikan wrapper.")
    except Exception as e:
        raise RuntimeError(f"Gagal melakukan swap: {e}")

    # pastikan swapped dalam BGR
    if swapped.ndim == 3 and swapped.shape[2] == 3:
        if swapped.dtype != np.uint8:
            swapped = swapped.astype(np.uint8)
        try:
            swapped_bgr = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
        except Exception:
            swapped_bgr = swapped
    else:
        swapped_bgr = swapped

    # paste back
    out = paste_back(temp_frame, swapped_bgr, inv_tfm, bbox=tuple(map(int, target_face.bbox)))
    return out


# ----------------- convenience: process_image / process_frames -----------------
def process_image(source_path: str, target_path: str, output_path: str, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    if source_img is None or target_img is None:
        raise FileNotFoundError("source atau target image tidak ditemukan.")

    # buat objek face dari face_analyser
    source_face = face_analyser.get_one_face(source_img)
    if source_face is None:
        raise RuntimeError("Tidak ada wajah di source image.")
    # agar swap_face bisa akses source_img
    setattr(source_face, 'src_img', source_img)

    # pilih reference face di target (index 0)
    target_face = face_analyser.get_one_face(target_img)
    if target_face is None:
        raise RuntimeError("Tidak ada wajah di target image.")

    result = swap_face(source_face, target_face, target_img, model_path=model_path, device=device)
    cv2.imwrite(output_path, result)


def process_frames(source_path: str, temp_frame_paths: List[str], model_path: Optional[str] = None, device: Optional[str] = None, update: Optional[Callable[[], None]] = None) -> None:
    """
    Dipakai saat memproses video: source_path (image), list of frame paths yang sudah diekstrak.
    Sama pola dengan roop sebelumnya.
    """
    source_img = cv2.imread(source_path)
    source_face = face_analyser.get_one_face(source_img)
    if source_face is None:
        raise RuntimeError("Tidak ada wajah di source image.")
    setattr(source_face, 'src_img', source_img)

    for idx, frame_path in enumerate(temp_frame_paths):
        frame = cv2.imread(frame_path)
        # gunakan tracking untuk dapat banyak wajah
        faces = face_analyser.smart_face_tracking(frame, frame_number=idx)
        if not faces:
            faces = face_analyser.get_many_faces(frame)
        if not faces:
            # tulis file tetap
            cv2.imwrite(frame_path, frame)
            if update:
                update()
            continue

        # simple: swap ke semua face yang valid
        out = frame
        for tgt in faces:
            if face_analyser.detect_occlusion(tgt, frame):
                continue
            out = swap_face(source_face, tgt, out, model_path=model_path, device=device)
        cv2.imwrite(frame_path, out)
        if update:
            update()
