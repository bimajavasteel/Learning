import threading
import numpy as np
import cv2
import onnxruntime as ort

from roop.utilities import conditional_download, resolve_relative_path

# ============================================
# Model & Config
# ============================================

FACE_PARSING_SESSION = None
FACE_PARSING_LOCK = threading.Lock()

FACE_PARSING_URL = (
    "https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/79999_iter.onnx"
)
FACE_PARSING_FILENAME = "face_parsing.onnx"

# Label wajah menurut CelebAMaskHQ (19 kelas)
FACE_LABELS = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # skin, brow, eye, nose, lip


# ============================================
# Download + Load ONNX
# ============================================

def pre_check_face_parsing():
    """
    Auto-download ONNX model jika belum ada.
    """
    download_dir = resolve_relative_path("../models")
    conditional_download(download_dir, [FACE_PARSING_URL])
    return True


def get_face_parsing_session():
    global FACE_PARSING_SESSION
    with FACE_PARSING_LOCK:
        if FACE_PARSING_SESSION is None:
            model_path = resolve_relative_path(f"../models/{FACE_PARSING_FILENAME}")
            FACE_PARSING_SESSION = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            print("✅ [FaceParsing] CelebAMaskHQ ONNX loaded (GPU enabled)")
    return FACE_PARSING_SESSION


# ============================================
# Preprocess
# ============================================

def _preprocess(frame, input_shape):
    h_in = input_shape[2]
    w_in = input_shape[3]

    if not isinstance(h_in, int) or not isinstance(w_in, int):
        h_in, w_in = 512, 512

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w_in, h_in), interpolation=cv2.INTER_LINEAR)

    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return img


# ============================================
# Run parsing → return HxW labels
# ============================================

def run_face_parsing(frame):
    session = get_face_parsing_session()
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape

    inp = _preprocess(frame, input_shape)
    out = session.run(None, {input_meta.name: inp})[0]

    # ONNX output shape: (1, 19, H, W)
    seg = np.argmax(out, axis=1)[0]
    return seg


# ============================================
# Mask wajah (anti tangan/bahu)
# ============================================

def get_face_mask(frame, face, dilate_iter=2):
    """
    Hasilkan mask wajah 0/1 ukuran frame.
    """
    h, w = frame.shape[:2]

    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None

    x1, y1, x2, y2 = map(int, bbox)

    seg = run_face_parsing(frame)
    sh, sw = seg.shape

    # Map bbox ke seg map
    sx1 = int(x1 / w * sw)
    sy1 = int(y1 / h * sh)
    sx2 = int(x2 / w * sw)
    sy2 = int(y2 / h * sh)

    region = seg[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return None

    face_region_mask = np.isin(region, FACE_LABELS).astype(np.uint8)
    full_mask = np.zeros_like(seg, np.uint8)
    full_mask[sy1:sy2, sx1:sx2] = face_region_mask

    mask = cv2.resize(full_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, dilate_iter)

    return (mask > 0).astype(np.uint8)
