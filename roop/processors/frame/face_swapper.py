# face_swapper.py (FP16-compatible)
# Versi: patched untuk menangani model inswapper FP16 dengan monkey-patch session.run
# Tempatkan di path yang sama seperti sebelumnya, ganti model jika perlu.

from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np
import os
import sys
import traceback

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    find_similar_face,
    smart_face_tracking,
    detect_occlusion,
    get_face_pose,
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

# wrinkle enhancer (jika ada)
try:
    from roop.processors.frame.wrinkle_enhancer import enhance_under_eye_wrinkles
except Exception:
    # jika modul tidak ada, buat stub pass-through
    def enhance_under_eye_wrinkles(frame, face):
        return frame

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def _safe_get_session(obj) -> Any:
    """
    Ambil attribute session dari model object insightface (beberapa versi menaruh attribute berbeda).
    """
    s = None
    for attr in ('session', '_sess', '_session'):
        s = getattr(obj, attr, None)
        if s is not None:
            return s
    return None


def _make_session_run_wrapper(session):
    """
    Kembalikan fungsi wrapper untuk session.run yang otomatis casting input numpy arrays
    ke dtype yang diharapkan oleh model (mis. float16 jika input defined sebagai tensor(float16)).
    """

    orig_run = getattr(session, 'run', None)
    if orig_run is None:
        return None

    # cache input metadata: name -> expected_type (string seperti 'tensor(float16)')
    try:
        input_meta = {i.name: getattr(i, 'type', '') for i in session.get_inputs()}
    except Exception:
        input_meta = {}

    def run_wrapper(output_names, input_feed, run_options=None):
        # buat salinan input_feed agar tidak merusak original
        new_feed = {}
        try:
            for name, arr in input_feed.items():
                expected = input_meta.get(name, '')
                # jika adalah numpy array, lakukan casting sesuai expected
                if isinstance(arr, np.ndarray):
                    if 'float16' in expected and arr.dtype != np.float16:
                        try:
                            new_feed[name] = arr.astype(np.float16)
                        except Exception:
                            # fallback ke arr original jika cast gagal
                            new_feed[name] = arr
                    elif 'float' in expected and 'float16' not in expected and arr.dtype != np.float32:
                        # model expects float32
                        try:
                            new_feed[name] = arr.astype(np.float32)
                        except Exception:
                            new_feed[name] = arr
                    else:
                        new_feed[name] = arr
                else:
                    # bukan numpy array: forward as-is
                    new_feed[name] = arr
        except Exception:
            # jangan biarkan wrapper menyebabkan crash; gunakan original input_feed
            new_feed = input_feed

        # panggil original run
        if run_options is None:
            return orig_run(output_names, new_feed)
        else:
            return orig_run(output_names, new_feed, run_options)

    return run_wrapper


def _patch_model_for_fp16(model, model_path: str):
    """
    Jika model file mengandung 'fp16' atau session input tipe float16,
    patch session.run untuk otomatis cast input ke float16.
    """
    try:
        sess = _safe_get_session(model)
        if sess is None:
            print("[face_swapper] Tidak menemukan session di model, skip patch FP16.")
            return

        # cek apakah filename mengindikasikan FP16
        is_fp16_by_name = False
        try:
            if model_path and 'fp16' in os.path.basename(model_path).lower():
                is_fp16_by_name = True
        except Exception:
            is_fp16_by_name = False

        # cek apakah salah satu input meta mengharapkan float16
        expects_fp16 = False
        try:
            for i in sess.get_inputs():
                t = getattr(i, 'type', '')
                if 'float16' in t:
                    expects_fp16 = True
                    break
        except Exception:
            expects_fp16 = False

        if not (is_fp16_by_name or expects_fp16):
            # tidak perlu patch
            return

        run_wrapper = _make_session_run_wrapper(sess)
        if run_wrapper is None:
            print("[face_swapper] Gagal membuat run wrapper.")
            return

        # simpan original run jika ingin restore (opsional)
        if not hasattr(sess, '__orig_run'):
            sess.__orig_run = sess.run

        sess.run = run_wrapper
        print(f"✅ [face_swapper] Session patched for FP16 support (model: {os.path.basename(model_path)})")
    except Exception as e:
        print(f"[face_swapper] Patch FP16 gagal: {e}")
        traceback.print_exc()


def get_face_swapper() -> Any:
    """
    Inisialisasi model inswapper.
    Jika ada model FP16 (nama file mengandung 'fp16' atau input expect float16),
    patch session.run supaya otomatis casting ke float16.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # default: cari file di ../models/ (sebagaimana repo kamu)
            # jika ingin model lain, ubah model_name
            model_name = os.environ.get('ROOP_INSWAPPER_MODEL', '../models/inswapper_128.onnx')
            model_path = resolve_relative_path(model_name)

            # jika model tidak ada, fallback ke download default inswapper_128.onnx
            if not os.path.exists(model_path):
                download_directory_path = resolve_relative_path('../models')
                conditional_download(download_directory_path, [
                    'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
                ])
                model_path = resolve_relative_path('../models/inswapper_128.onnx')

            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )

            # Patch session jika model FP16
            try:
                _patch_model_for_fp16(FACE_SWAPPER, model_path)
            except Exception:
                # jangan crash jika patch gagal
                pass

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    Pastikan model sudah ke-download sebelum mulai.
    Jika kamu ingin memakai model FP16, letakkan file fp16 di ../models/
    dan set env ROOP_INSWAPPER_MODEL jika perlu.
    """
    download_directory_path = resolve_relative_path('../models')
    # default tetap download inswapper_128 (float32) bila tidak ada file lain
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx'
    ])
    return True


def pre_start() -> bool:
    """
    Validasi path source & target sebelum proses.
    """
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False

    source_img = cv2.imread(roop.globals.source_path)
    if not get_one_face(source_img):
        update_status('No face in source path detected.', NAME)
        return False

    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False

    return True


def post_process() -> None:
    """
    Bersihkan model & reference setelah selesai.
    """
    clear_face_swapper()
    clear_face_reference()


# =====================================================================
#  POSE-AWARE BBOX ADJUSTMENT (sama seperti versi asli)
# =====================================================================

def adapt_bbox_for_pose(face: Face, frame_shape) -> None:
    pitch, yaw, roll = get_face_pose(face)

    h_frame, w_frame = frame_shape[:2]
    bbox = np.array(face.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    pad_x = 0.0
    pad_y_top = 0.0
    pad_y_bottom = 0.0

    if abs(yaw) > 25.0:
        extra = (abs(yaw) - 25.0) * 0.02
        extra = min(extra, 0.20)
        pad_x = w * extra

    if pitch < -15.0:
        extra = (abs(pitch) - 15.0) * 0.02
        extra = min(extra, 0.25)
        pad_y_top = h * extra
    elif pitch > 20.0:
        extra = (pitch - 20.0) * 0.015
        extra = min(extra, 0.18)
        pad_y_bottom = h * extra

    nx1 = int(max(0, x1 - pad_x))
    nx2 = int(min(w_frame - 1, x2 + pad_x))
    ny1 = int(max(0, y1 - pad_y_top))
    ny2 = int(min(h_frame - 1, y2 + pad_y_bottom))

    if nx2 <= nx1 or ny2 <= ny1:
        return

    face.bbox = np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


# =====================================================================
#  CORE SWAP (memakai model.get seperti biasa)
# =====================================================================

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dasar (panggil inswapper).
    """
    if source_face is None or target_face is None:
        return temp_frame

    adapt_bbox_for_pose(target_face, temp_frame.shape)

    try:
        return get_face_swapper().get(
            temp_frame,
            target_face,
            source_face,
            paste_back=True
        )
    except Exception as e:
        # jika terjadi error, jangan crash pipeline; kembalikan frame asli
        print(f"[face_swapper] swap_face gagal: {e}")
        traceback.print_exc()
        return temp_frame


# =====================================================================
#  Pemilihan target terbaik & loop frame (identik dengan versi asli)
# =====================================================================

def _select_best_target_by_embedding(
    faces: List[Face],
    reference_face: Face
) -> Face | None:

    if not faces or reference_face is None:
        return None

    if not hasattr(reference_face, 'normed_embedding'):
        return None

    ref_emb = reference_face.normed_embedding
    best_face = None
    best_distance = float('inf')

    similar_threshold = getattr(roop.globals, 'similar_face_distance', 1.0)

    for f in faces:
        if not hasattr(f, 'normed_embedding'):
            continue

        try:
            distance = np.sum(np.square(f.normed_embedding - ref_emb))
        except Exception:
            continue

        if distance < similar_threshold and distance < best_distance:
            best_distance = distance
            best_face = f

    return best_face


def process_frame(
    source_face: Face,
    reference_face: Face,
    temp_frame: Frame,
    frame_number: int = 0
) -> Frame:

    if source_face is None:
        return temp_frame

    # MANY FACES
    if roop.globals.many_faces:
        faces = smart_face_tracking(temp_frame, frame_number)
        if not faces:
            faces = get_many_faces(temp_frame)

        if not faces:
            return temp_frame

        for target_face in faces:
            if detect_occlusion(target_face, temp_frame):
                continue

            temp_frame = swap_face(source_face, target_face, temp_frame)

            # Apply wrinkle after swap
            temp_frame = enhance_under_eye_wrinkles(temp_frame, target_face)

        return temp_frame

    # SINGLE FACE
    tracked_faces = smart_face_tracking(temp_frame, frame_number)
    if not tracked_faces:
        tracked_faces = get_many_faces(temp_frame)

    if not tracked_faces:
        return temp_frame

    valid_faces = [f for f in tracked_faces if not detect_occlusion(f, temp_frame)]
    if not valid_faces:
        return temp_frame

    best_target = None

    if reference_face is not None:
        best_target = _select_best_target_by_embedding(valid_faces, reference_face)

    if best_target is None:
        best_target = valid_faces[0]

    temp_frame = swap_face(source_face, best_target, temp_frame)

    # Wrinkle AFTER swap
    temp_frame = enhance_under_eye_wrinkles(temp_frame, best_target)

    return temp_frame


# =====================================================================
#  FRAME LOOP & UTIL
# =====================================================================

def process_frames(
    source_path: str,
    temp_frame_paths: List[str],
    update: Callable[[], None]
) -> None:

    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)

    reference_face = None if roop.globals.many_faces else get_face_reference()

    for idx, temp_frame_path in enumerate(temp_frame_paths):
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(
            source_face=source_face,
            reference_face=reference_face,
            temp_frame=temp_frame,
            frame_number=idx
        )
        cv2.imwrite(temp_frame_path, result)

        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)

    source_face = get_one_face(source_img)

    reference_face = None
    if not roop.globals.many_faces:
        reference_face = get_one_face(
            target_frame,
            roop.globals.reference_face_position
        )

    result = process_frame(
        source_face=source_face,
        reference_face=reference_face,
        temp_frame=target_frame,
        frame_number=0
    )
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:

    if not roop.globals.many_faces and not get_face_reference():
        try:
            ref_idx = roop.globals.reference_frame_number
            reference_frame = cv2.imread(temp_frame_paths[ref_idx])
            reference_face = get_one_face(
                reference_frame,
                roop.globals.reference_face_position
            )
            set_face_reference(reference_face)
        except Exception:
            set_face_reference(None)

    roop.processors.frame.core.process_video(
        source_path,
        temp_frame_paths,
        process_frames
    )
