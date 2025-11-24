"""
ReSwapper wrapper for ROOP.
File: roop/processors/frame/reswapper.py
"""

from typing import Any, Optional
import os
import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

# kemungkinan nama class yang digunakan di repo ReSwapper
_possible_imports = [
    ("reswapper.model", "ReSwapperModel"),
    ("reswapper.models", "ReSwapper"),
    ("reswapper", "ReSwapper"),
    ("reswapper.net", "Reswapper"),
]


def _find_model_class():
    """Cari class model yang cocok dari repo ReSwapper."""
    for module_name, class_name in _possible_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
        except Exception:
            continue
    return None


class ReSwapperWrapper:
    def __init__(self, model_path: str):
        if torch is None:
            raise RuntimeError("PyTorch tidak ditemukan. Install torch terlebih dahulu.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model ReSwapper tidak ditemukan: {model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ModelClass = _find_model_class()
        if ModelClass is None:
            raise RuntimeError(
                "Model class ReSwapper tidak ditemukan.\n"
                "Silakan clone + install repo: https://github.com/somanchiu/ReSwapper\n"
                "Contoh: git clone dan pip install -e ReSwapper"
            )

        # load model
        self.model = ModelClass()
        ckpt = torch.load(model_path, map_location=self.device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

    def _crop_face(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def get(self, frame, target_face, source_face, paste_back=True):
        crop, (x1, y1, x2, y2) = self._crop_face(frame, target_face.bbox)
        if crop.size == 0:
            return frame

        # ambil image source
        src_img = None
        try:
            if hasattr(source_face, "img"):
                src_img = source_face.img
            elif hasattr(source_face, "image"):
                src_img = source_face.image
        except:
            src_img = None

        if src_img is None:
            if isinstance(source_face, np.ndarray):
                src_img = source_face
            else:
                src_img = crop.copy()

        # convert ke RGB
        tgt_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        # resize
        tgt_in = cv2.resize(tgt_rgb, (256, 256))
        src_in = cv2.resize(src_rgb, (256, 256))

        import torch
        tgt_tensor = torch.from_numpy(tgt_in).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        src_tensor = torch.from_numpy(src_in).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        tgt_tensor = tgt_tensor.to(self.device)
        src_tensor = src_tensor.to(self.device)

        # panggil model (3 kemungkinan API)
        with torch.no_grad():
            out = None
            try:
                out = self.model.swap(src_tensor, tgt_tensor)
            except:
                try:
                    out = self.model(src_tensor, tgt_tensor)
                except:
                    try:
                        out = self.model.infer(src_tensor, tgt_tensor)
                    except Exception as e:
                        raise RuntimeError("API ReSwapper tidak sesuai. Lihat class model aslinya.") from e

        if isinstance(out, (list, tuple)):
            out = out[0]

        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype("uint8")
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(out_bgr, (x2 - x1, y2 - y1))

        if paste_back:
            frame[y1:y2, x1:x2] = out_bgr
            return frame

        return out_bgr
