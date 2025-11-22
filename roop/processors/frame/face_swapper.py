from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
import onnxruntime as ort

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
FACE_PARSER_SESSION = None
THREAD_LOCK = threading.Lock()
PARSER_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

# -----------------------------
# Konfigurasi
# -----------------------------
MIN_FACE_SIZE = 64          # px
MIN_DET_SCORE = 0.45        # minimal det_score agar di-swap
TEMPORAL_ALPHA = 0.7        # smoothing factor bbox (0.0 = no smoothing, 1.0 = no change)
PYRAMID_LEVELS = 3          # level Gaussian Pyramid untuk blending

# state untuk temporal smoothing (single-face mode)
TEMPORAL_STATE = {
    "bbox": None  # np.array([x1, y1, x2, y2], dtype=float32)
}
TEMPORAL_LOCK = threading.Lock()


# =========================
# MODEL LOADER
# =========================

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128_fp16.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER


def get_face_parser() -> Optional[ort.InferenceSession]:
    """
    Face Parsing (BiSeNet) loader.
    Kalau gagal load → None (otomatis fallback ke ellipse mask).
    """
    global FACE_PARSER_SESSION

    with PARSER_LOCK:
        if FACE_PARSER_SESSION is not None:
            return FACE_PARSER_SESSION

        try:
            model_path = resolve_relative_path('../models/BiseNet.onnx')
            FACE_PARSER_SESSION = ort.InferenceSession(
                model_path,
                providers=roop.globals.execution_providers
            )
            print("[Face Swapper] Face parsing ONNX loaded.")
        except Exception as e:
            print(f"[Face Swapper] Face parsing model not available, fallback to ellipse mask. Error: {e}")
            FACE_PARSER_SESSION = None

        return FACE_PARSER_SESSION


def clear_face_swapper() -> None:
    global FACE_SWAPPER, FACE_PARSER_SESSION
    FACE_SWAPPER = None
    FACE_PARSER_SESSION = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')

    # INSwapper 128 FP16
    conditional_download(
        download_directory_path,
        [
            'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128_fp16.onnx'
        ]
    )

    # BiSeNet ONNX untuk face parsing
    conditional_download(
        download_directory_path,
        [
            'https://huggingface.co/qualcomm/BiseNet/resolve/aeb57eda69d58721c5c186eb65b612dfa43faeab/BiseNet.onnx'
        ]
    )

    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


# =========================
# HELPER UTAMA
# =========================

def safe_get_landmarks(face: Face) -> Optional[np.ndarray]:
    if face is None:
        return None
    for attr in ['landmark_2d_106', 'landmark_2d', 'kps', 'landmarks']:
        if hasattr(face, attr):
            landmarks = getattr(face, attr)
            if landmarks is not None and len(landmarks) > 0:
                return landmarks
    return None


def is_face_reliable(face: Face) -> bool:
    """Filter sederhana untuk occlusion & jitter: ukuran + score."""
    if face is None or not hasattr(face, "bbox") or face.bbox is None:
        return False

    x1, y1, x2, y2 = face.bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    det_score = getattr(face, "det_score", None)
    if det_score is not None and det_score < MIN_DET_SCORE:
        return False

    return True


def temporal_smooth_bbox(face: Face) -> None:
    """
    Temporal smoothing bbox (anti jitter) untuk single-face mode.
    Hanya dipakai ketika roop.globals.many_faces = False.
    """
    if face is None or not hasattr(face, "bbox") or face.bbox is None:
        return

    with TEMPORAL_LOCK:
        prev = TEMPORAL_STATE["bbox"]
        current = np.array(face.bbox, dtype=np.float32)

        if prev is None:
            TEMPORAL_STATE["bbox"] = current
            return

        smoothed = TEMPORAL_ALPHA * prev + (1.0 - TEMPORAL_ALPHA) * current
        TEMPORAL_STATE["bbox"] = smoothed
        face.bbox = smoothed


# =========================
# ADVANCED COLOR CORRECTION
# =========================

def match_histogram_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Histogram matching satu channel (src → ref).
    src, ref = 2D array (H, W)
    """
    src_flat = src.ravel()
    ref_flat = ref.ravel()

    src_hist, _ = np.histogram(src_flat, 256, [0, 256])
    ref_hist, _ = np.histogram(ref_flat, 256, [0, 256])

    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)

    if src_cdf[-1] != 0:
        src_cdf /= src_cdf[-1]
    if ref_cdf[-1] != 0:
        ref_cdf /= ref_cdf[-1]

    mapping = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_idx in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
            ref_idx += 1
        mapping[src_idx] = ref_idx

    matched = mapping[src_flat].reshape(src.shape)
    return matched


def advanced_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """
    Advanced color correction dengan histogram matching (BGR).
    """
    try:
        if target_face is None or swapped_face is None:
            return swapped_face

        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        target_region = target_frame[y1:y2, x1:x2]
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face

        if swapped_face.shape[:2] != target_region.shape[:2]:
            swapped_face = cv2.resize(
                swapped_face,
                (target_region.shape[1], target_region.shape[0])
            )

        src = swapped_face.astype(np.uint8)
        ref = target_region.astype(np.uint8)

        matched = np.zeros_like(src)
        for c in range(3):
            matched[:, :, c] = match_histogram_channel(src[:, :, c], ref[:, :, c])

        out = cv2.addWeighted(src, 0.3, matched, 0.7, 0)
        return out

    except Exception:
        return swapped_face


# =========================
# FACE PARSING MASK (BiSeNet)
# =========================

def parse_face_mask(frame: Frame, face: Face) -> Optional[np.ndarray]:
    """
    Face parsing pakai BiSeNet ONNX (jika tersedia).
    Return: mask float32 [H, W] (0..1) pada full-frame.
    Kalau parser tidak tersedia → return None.
    """
    session = get_face_parser()
    if session is None or frame is None or face is None:
        return None

    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # Asumsi input BiSeNet: BGR, 512x512, float32, [0,1], NCHW
        input_size = (512, 512)
        resized = cv2.resize(face_crop, input_size, interpolation=cv2.INTER_LINEAR)
        inp = resized.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # NCHW

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: inp})
        seg_logits = outputs[0]  # [N, C, H, W]

        seg = np.argmax(seg_logits, axis=1)[0].astype(np.uint8)  # [H, W]

        # Generic: semua kelas > 0 dianggap bagian wajah
        mask_local = np.where(seg > 0, 1.0, 0.0).astype(np.float32)

        # resize balik ke ukuran bbox
        mask_local = cv2.resize(mask_local, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

        full_mask = np.zeros((h, w), dtype=np.float32)
        full_mask[y1:y2, x1:x2] = mask_local

        full_mask = cv2.GaussianBlur(full_mask, (25, 25), 0)
        full_mask = np.clip(full_mask, 0.0, 1.0)

        return full_mask

    except Exception as e:
        print(f"[Face Swapper] Face parsing failed, fallback to ellipse mask. Error: {e}")
        return None


def create_ellipse_mask(face: Face, frame_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        cv2.ellipse(
            mask,
            (center_x, center_y),
            (width // 2, height // 2),
            0, 0, 360,
            1.0, -1
        )
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        mask = np.clip(mask, 0.0, 1.0)
    except Exception:
        pass
    return mask


# =========================
# GAUSSIAN PYRAMID BLENDING
# =========================

def gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    gp = [img.astype(np.float32)]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        gp.append(img.astype(np.float32))
    return gp


def laplacian_pyramid(gp: List[np.ndarray]) -> List[np.ndarray]:
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        ge = cv2.pyrUp(gp[i], dstsize=size)
        lf = cv2.subtract(gp[i - 1], ge)
        lp.append(lf)
    lp.reverse()
    return lp


def pyramid_blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, levels: int = PYRAMID_LEVELS) -> np.ndarray:
    """
    Multi-scale blending:
    - src: wajah swap
    - dst: frame region
    - mask: 3-channel [0..1]
    """
    src = src.astype(np.float32)
    dst = dst.astype(np.float32)
    mask = mask.astype(np.float32)

    gp_mask = gaussian_pyramid(mask, levels)
    gp_src = gaussian_pyramid(src, levels)
    gp_dst = gaussian_pyramid(dst, levels)

    lp_src = laplacian_pyramid(gp_src)
    lp_dst = laplacian_pyramid(gp_dst)

    blended_pyr = []
    for l_src, l_dst, g_m in zip(lp_src, lp_dst, gp_mask):
        blended = l_src * g_m + l_dst * (1.0 - g_m)
        blended_pyr.append(blended)

    img = blended_pyr[-1]
    for i in range(len(blended_pyr) - 2, -1, -1):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, blended_pyr[i])

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# =========================
# BLENDING FINAL
# =========================

def advanced_blending(swapped_face: Frame, target_frame: Frame, target_face: Face, parsing_mask: Optional[np.ndarray]) -> Frame:
    """
    Blending:
    - Face parsing (kalau ada) + ellipse mask.
    - Gaussian pyramid blending pada region wajah.
    """
    try:
        h, w = target_frame.shape[:2]
        x1, y1, x2, y2 = map(int, target_face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_h, face_w = y2 - y1, x2 - x1
        if face_h <= 0 or face_w <= 0:
            return target_frame

        if swapped_face.shape[:2] != (face_h, face_w):
            swapped_face = cv2.resize(swapped_face, (face_w, face_h))

        face_region = target_frame[y1:y2, x1:x2]

        # full-frame mask kombinasi parsing + ellipse
        if parsing_mask is not None:
            full_mask = parsing_mask
        else:
            full_mask = np.zeros((h, w), dtype=np.float32)

        ellipse = create_ellipse_mask(target_face, target_frame.shape)

        if parsing_mask is None:
            full_mask = ellipse
        else:
            full_mask = np.clip(parsing_mask * 0.7 + ellipse * 0.3, 0.0, 1.0)

        local_mask = full_mask[y1:y2, x1:x2]
        if local_mask.shape != swapped_face.shape[:2]:
            local_mask = cv2.resize(local_mask, (face_w, face_h), interpolation=cv2.INTER_LINEAR)

        mask_3d = np.stack([local_mask] * 3, axis=-1)

        blended_face = pyramid_blend(swapped_face, face_region, mask_3d, levels=PYRAMID_LEVELS)

        result = target_frame.copy()
        result[y1:y2, x1:x2] = blended_face
        return result

    except Exception as e:
        print(f"[Face Swapper] Advanced blending failed, fallback. Error: {e}")
        return target_frame


# =========================
# FACE ENHANCEMENT
# =========================

def enhance_face_quality(face: Frame) -> Frame:
    try:
        if face is None:
            return face

        kernel = np.array(
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]],
            dtype=np.float32
        ) * 0.10

        sharpened = cv2.filter2D(face, -1, kernel)
        denoised = cv2.bilateralFilter(sharpened, 5, 15, 15)
        return denoised
    except Exception:
        return face


def ensure_frame_format(frame: Any) -> Optional[Frame]:
    if frame is None:
        return None
    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
        return frame
    if isinstance(frame, tuple):
        try:
            arr = np.array(frame)
            if arr.size > 0:
                return arr
        except Exception:
            pass
    return None


# =========================
# PIPELINE SWAP PER FRAME
# =========================

def swap_face_optimized(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Pipeline gabungan:
    - Filter occlusion & jitter (size + det_score).
    - Temporal smoothing bbox (single-face).
    - INSwapper (tanpa paste_back).
    - Advanced color correction (histogram matching).
    - Enhancement.
    - Face parsing (jika ada).
    - Gaussian pyramid blending dengan mask (parsing + ellipse).
    """
    try:
        if not is_face_reliable(target_face):
            return temp_frame

        if not roop.globals.many_faces:
            temporal_smooth_bbox(target_face)

        swapped_result = get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=False
        )

        swapped_face = ensure_frame_format(swapped_result)
        if swapped_face is None:
            return get_face_swapper().get(
                temp_frame,
                target_face,
                source_face,
                paste_back=True
            )

        swapped_face = advanced_color_correction(swapped_face, temp_frame, target_face)
        swapped_face = enhance_face_quality(swapped_face)

        parsing_mask = None
        try:
            parsing_mask = parse_face_mask(temp_frame, target_face)
        except Exception:
            parsing_mask = None

        result_frame = advanced_blending(swapped_face, temp_frame, target_face, parsing_mask)

        return result_frame

    except Exception as e:
        print(f"[Face Swapper] swap_face_optimized error, fallback to basic paste_back. Error: {e}")
        return get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face_optimized(source_face, target_face, temp_frame)
        return temp_frame
    except Exception as e:
        print(f"[Face Swapper] process_frame error: {e}")
        return temp_frame


# =========================
# BATCH PROCESS
# =========================

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()

        for temp_frame_path in temp_frame_paths:
            try:
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is not None:
                    result = process_frame(source_face, reference_face, temp_frame)
                    cv2.imwrite(temp_frame_path, result)
                if update:
                    update()
            except Exception as e:
                print(f"[Face Swapper] Error processing frame {temp_frame_path}: {e}")
                continue
    except Exception as e:
        print(f"[Face Swapper] process_frames error: {e}")


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(
            target_frame,
            roop.globals.reference_face_position
        )
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"[Face Swapper] process_image error: {e}")


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"[Face Swapper] process_video error: {e}")
