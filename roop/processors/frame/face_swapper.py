# roop/processors/frame/face_swapper.py
"""
ROOP frame processor: Face swapper backend menggunakan ReSwapper (.pth).
Implementasi ini mengikuti interface processor ROOP:
- NAME
- pre_check()
- pre_start()
- process_image(...)
- process_frames(...)
- post_process()

Catatan:
- Menggunakan roop.face_analyser untuk deteksi/landmark/tracking/occlusion.
- Mengharapkan model ReSwapper berada di path yang dapat di-resolve (lihat MODEL_PATH_DEFAULT).
- Jika package 'reswapper' tidak tersedia, pesan error jelas akan muncul.
"""
from typing import Any, List, Callable, Optional
import os
import cv2
import numpy as np
import threading
import torch

# ROOP internals (asumsi tersedia di project)
import roop.globals
from roop.core import update_status
from roop.utilities import resolve_relative_path, conditional_download, is_image, is_video
from roop.typing import Frame, Face

# gunakan face_analyser yang sudah ada (kita refactor sebelumnya)
from roop import face_analyser

# coba import reswapper (user harus meletakkan repo/packagenya di PYTHONPATH)
try:
    from reswapper import ReSwapper as _ReSwapper
    _HAS_RESWAPPER = True
except Exception:
    _ReSwapper = None
    _HAS_RESWAPPER = False

# Globals processor
NAME = "ROOP.FACE-SWAPPER"
THREAD_LOCK = threading.Lock()
SWAPPER: Any = None

# Default model path hint (disarankan taruh di dataset Kaggle)
MODEL_PATH_DEFAULT = resolve_relative_path(getattr(roop.globals, "reswapper_model_path", "../models/reswapper_256-1567500.pth"))

# Small helper untuk pesan status
def _set_status(msg: str) -> None:
    try:
        update_status(msg, NAME)
    except Exception:
        print(f"[{NAME}] {msg}")


# -------------------- Swapper init --------------------
def get_face_swapper(model_path: Optional[str] = None, device: Optional[str] = None) -> Any:
    """
    Lazy init ReSwapper backend.
    model_path: path ke .pth checkpoint
    device: 'cuda' atau 'cpu' (otomatis jika None)
    """
    global SWAPPER
    with THREAD_LOCK:
        if SWAPPER is not None:
            return SWAPPER

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not _HAS_RESWAPPER:
            raise RuntimeError(
                "Package 'reswapper' tidak ditemukan. Upload repo ReSwapper ke working dir atau tambahkan ke PYTHONPATH.\n"
                f"Contoh letakkan checkpoint di: {MODEL_PATH_DEFAULT}"
            )

        if model_path is None:
            model_path = MODEL_PATH_DEFAULT

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model ReSwapper tidak ditemukan di path: {model_path}")

        # coba inisialisasi model menggunakan beberapa pola umum
        try:
            try:
                SWAPPER = _ReSwapper(model_path=model_path, device=device)
            except TypeError:
                # beberapa implementasi mungkin memerlukan load_state_dict atau method load
                SWAPPER = _ReSwapper()
                sd = torch.load(model_path, map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                if hasattr(SWAPPER, "load_state_dict"):
                    SWAPPER.load_state_dict(sd)
                if hasattr(SWAPPER, "to"):
                    SWAPPER.to(device)
            _set_status(f"ReSwapper loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Gagal inisialisasi ReSwapper: {e}")

    return SWAPPER


# -------------------- Align / paste helpers (menggunakan landmark InsightFace) --------------------
def _get_5pts_from_face(face: Face):
    # Prefer 5-point kps, fallback ke beberapa indeks landmark 106
    lm = getattr(face, "kps", None)
    if lm is not None:
        return np.array(lm, dtype=np.float32)
    lm106 = getattr(face, "landmark_2d_106", None)
    if lm106 is not None:
        try:
            # Indeks konservatif: mata kiri(36), mata kanan(45), hidung(30), mulut kiri(48), mulut kanan(54)
            pts = np.array([lm106[36], lm106[45], lm106[30], lm106[48], lm106[54]], dtype=np.float32)
            return pts
        except Exception:
            pass
    return None


def _estimate_transform(src_pts: np.ndarray, dst_pts: np.ndarray):
    if src_pts is None or dst_pts is None:
        return None
    tfm = cv2.estimateAffinePartial2D(src_pts.reshape(-1, 1, 2), dst_pts.reshape(-1, 1, 2))[0]
    return tfm


def align_face(img: Frame, face: Face, size: int = 256):
    pts = _get_5pts_from_face(face)
    dst = np.array([
        [size * 0.30, size * 0.35],
        [size * 0.70, size * 0.35],
        [size * 0.50, size * 0.54],
        [size * 0.33, size * 0.78],
        [size * 0.67, size * 0.78],
    ], dtype=np.float32)
    if pts is None:
        # fallback gunakan bbox center
        x1, y1, x2, y2 = map(int, face.bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        src = np.array([
            [cx - w * 0.2, cy - h * 0.25],
            [cx + w * 0.2, cy - h * 0.25],
            [cx, cy],
            [cx - w * 0.25, cy + h * 0.35],
            [cx + w * 0.25, cy + h * 0.35]
        ], dtype=np.float32)
    else:
        src = pts.astype(np.float32)

    tfm = _estimate_transform(src, dst)
    if tfm is None:
        # fallback crop bbox
        x1, y1, x2, y2 = map(int, face.bbox)
        crop = img[y1:y2, x1:x2]
        aligned = cv2.resize(crop, (size, size))
        inv = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return aligned, inv
    aligned = cv2.warpAffine(img, tfm, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inv = cv2.invertAffineTransform(tfm)
    return aligned, inv


def paste_back(orig_img: Frame, swapped_aligned: Frame, inv_tfm, bbox):
    h, w = orig_img.shape[:2]
    try:
        warped = cv2.warpAffine(swapped_aligned, inv_tfm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    except Exception:
        # fallback simpel paste ke bbox
        x1, y1, x2, y2 = bbox
        th, tw = y2 - y1, x2 - x1
        resized = cv2.resize(swapped_aligned, (tw, th))
        out = orig_img.copy()
        out[y1:y2, x1:x2] = resized
        return out

    gray = cv2.cvtColor(swapped_aligned, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype('uint8') * 255
    mask_warp = cv2.warpAffine(mask, inv_tfm, (w, h), flags=cv2.INTER_LINEAR)
    mask_warp = cv2.GaussianBlur(mask_warp, (15, 15), 0)

    ys, xs = np.where(mask_warp > 10)
    if len(xs) == 0:
        return orig_img
    center = (int(np.mean(xs)), int(np.mean(ys)))
    try:
        out = cv2.seamlessClone(warped, orig_img, mask_warp, center, cv2.NORMAL_CLONE)
        return out
    except Exception:
        alpha = (mask_warp.astype(np.float32) / 255.0)[:, :, None]
        out = (warped.astype(np.float32) * alpha + orig_img.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        return out


# -------------------- core swapper wrapper --------------------
def _do_swap(swapper, src_img, src_face: Face, tgt_img, tgt_face: Face, model_path: Optional[str] = None, device: Optional[str] = None):
    # align both
    aligned_src, inv_src = align_face(src_img, src_face, size=256)
    aligned_tgt, inv_tgt = align_face(tgt_img, tgt_face, size=256)

    # ensure RGB input to model if model expects it
    inp_src = cv2.cvtColor(aligned_src, cv2.COLOR_BGR2RGB)
    inp_tgt = cv2.cvtColor(aligned_tgt, cv2.COLOR_BGR2RGB)

    # call common api patterns
    try:
        if hasattr(swapper, "swap"):
            swapped = swapper.swap(inp_src, inp_tgt)  # expect RGB uint8
        elif hasattr(swapper, "forward"):
            with torch.no_grad():
                t_src = torch.from_numpy(inp_src.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                t_tgt = torch.from_numpy(inp_tgt.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                out = swapper.forward(t_src, t_tgt)
                out_img = (out[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                swapped = out_img
        else:
            raise RuntimeError("Swapper object tidak mendukung method 'swap' atau 'forward'.")
    except Exception as e:
        raise RuntimeError(f"Failed to run swapper: {e}")

    # convert to BGR for paste_back
    if swapped.ndim == 3 and swapped.shape[2] == 3:
        try:
            swapped_bgr = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
        except Exception:
            swapped_bgr = swapped
    else:
        swapped_bgr = swapped

    out = paste_back(tgt_img, swapped_bgr, inv_tgt, tuple(map(int, tgt_face.bbox)))
    return out


# -------------------- Processor interface --------------------
def pre_check() -> bool:
    """
    Pastikan model tersedia (download jika perlu).
    """
    # coba download sample hint (tidak otomatis download ReSwapper .pth karena lisensi)
    models_dir = resolve_relative_path("../models")
    os.makedirs(models_dir, exist_ok=True)
    # jika user ingin automated download, bisa tambahkan URL di roop.globals
    _set_status("Pre-check: pastikan model ReSwapper (.pth) tersedia di folder models.")
    return True


def pre_start() -> bool:
    """
    Validasi source/target path & ketersediaan wajah source.
    """
    if not is_image(roop.globals.source_path):
        _set_status("Select an image for source path.")
        return False

    src_img = cv2.imread(roop.globals.source_path)
    if src_img is None:
        _set_status("Cannot read source image.")
        return False

    if not face_analyser.get_one_face(src_img):
        _set_status("No face detected in source image.")
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        _set_status("Select an image or video for target path.")
        return False

    return True


def post_process() -> None:
    """
    Cleanup jika perlu.
    """
    global SWAPPER
    SWAPPER = None
    face_analyser.clear_face_analyser()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Proses mode gambar ke gambar sesuai interface ROOP.
    """
    src = cv2.imread(source_path)
    tgt = cv2.imread(target_path)
    if src is None or tgt is None:
        raise FileNotFoundError("Source/Target not found.")

    src_face = face_analyser.get_one_face(src)
    if src_face is None:
        raise RuntimeError("No face in source image.")

    tgt_face = face_analyser.get_one_face(tgt)
    if tgt_face is None:
        raise RuntimeError("No face in target image.")

    # attach source image ke object supaya _do_swap dapat akses
    setattr(src_face, "src_img", src)

    swapper = get_face_swapper()  # akan raise jika tidak ada
    out = _do_swap(swapper, src, src_face, tgt, tgt_face)
    cv2.imwrite(output_path, out)


def process_frames(source_path: str, temp_frame_paths: List[str], update: Optional[Callable[[], None]] = None) -> None:
    """
    Dipanggil oleh core.process_video untuk memproses semua frame.
    Struktur dan nama fungsi sesuai ekspektasi ROOP.
    """
    src = cv2.imread(source_path)
    if src is None:
        raise RuntimeError("Source image cannot be read.")

    src_face = face_analyser.get_one_face(src)
    if src_face is None:
        raise RuntimeError("No face in source image.")
    setattr(src_face, "src_img", src)

    swapper = get_face_swapper()

    for idx, fp in enumerate(temp_frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            continue

        # tracking + many_faces logic sesuai roop.globals
        faces = face_analyser.smart_face_tracking(frame, frame_number=idx)
        if not faces:
            faces = face_analyser.get_many_faces(frame)
        if not faces:
            cv2.imwrite(fp, frame)
            if update:
                update()
            continue

        out = frame
        # many_faces true -> swap ke semua wajah yang valid
        if getattr(roop.globals, "many_faces", False):
            for f in faces:
                if face_analyser.detect_occlusion(f, frame):
                    continue
                out = _do_swap(swapper, src, src_face, out, f)
        else:
            # single-face mode: pilih berdasarkan reference atau embedding
            ref = None
            if not getattr(roop.globals, "many_faces", False):
                try:
                    ref_idx = getattr(roop.globals, "reference_frame_number", 0)
                    ref_frame = cv2.imread(temp_frame_paths[ref_idx])
                    ref = face_analyser.get_one_face(ref_frame, getattr(roop.globals, "reference_face_position", 0))
                except Exception:
                    ref = None
            # pilih best match
            best = None
            if ref is not None:
                best = face_analyser.find_similar_face(frame, ref, use_tracking=True)
            if best is None:
                # fallback ke face pertama valid
                valid = [f for f in faces if not face_analyser.detect_occlusion(f, frame)]
                if not valid:
                    cv2.imwrite(fp, frame)
                    if update:
                        update()
                    continue
                best = valid[0]
            out = _do_swap(swapper, src, src_face, out, best)

        cv2.imwrite(fp, out)
        if update:
            update()
