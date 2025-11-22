# roop/face_segmentation.py
import os
import urllib.request
from typing import List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

# =====================================================
#   KONFIGURASI
# =====================================================

MODEL_DIR = "models"
BISENET_WEIGHTS_PATH = os.path.join(MODEL_DIR, "face_parsing_bisenet.pth")
BISENET_WEIGHTS_URL = "https://huggingface.co/qualcomm/BiseNet/resolve/aeb57eda69d58721c5c186eb65b612dfa43faeab/BiseNet.onnx"
BISENET_INPUT_SIZE = 512

# Label wajah menurut CelebAMask-HQ
FACE_LABELS = {1, 2, 3, 4, 5, 6, 7, 9}


# =====================================================
#   AUTO DOWNLOAD
# =====================================================

def download_bisenet_weights():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(BISENET_WEIGHTS_PATH):
        print("[BiSeNet] Weights OK.")
        return

    print("[BiSeNet] Mengunduh weights BiSeNet (±80MB)...")

    try:
        urllib.request.urlretrieve(BISENET_WEIGHTS_URL, BISENET_WEIGHTS_PATH)
        print("[BiSeNet] Download selesai.")
    except Exception as e:
        print(f"[BiSeNet] Gagal download otomatis: {e}")
        print(f"Silakan download manual: {BISENET_WEIGHTS_URL}")


# =====================================================
#   MODEL BISENET (Ringkas tapi kompatibel)
# =====================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetSimple(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, 7, 2, 3)
        self.conv2 = ConvBNReLU(64, 128, 3, 2, 1)
        self.conv3 = ConvBNReLU(128, 256, 3, 2, 1)
        self.conv4 = ConvBNReLU(256, 512, 3, 2, 1)
        self.conv_out = nn.Conv2d(512, n_classes, 1)

    def forward(self, x):
        x = self.conv1(x)   # /2
        x = self.conv2(x)   # /4
        x = self.conv3(x)   # /8
        x = self.conv4(x)   # /16
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)
        return x


# =====================================================
#   WRAPPER SEGMENTER
# =====================================================

class BiSeNetFaceSegmenter:
    def __init__(self):
        download_bisenet_weights()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = BiSeNetSimple().to(self.device)
        self.model.eval()

        state = torch.load(BISENET_WEIGHTS_PATH, map_location=self.device)
        if "state_dict" in state:
            state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
        self.model.load_state_dict(state, strict=False)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        self.use_amp = (self.device.type == "cuda")

    def preprocess(self, crops: List[np.ndarray]):
        tensors = []
        for img in crops:
            img = cv2.resize(img, (BISENET_INPUT_SIZE, BISENET_INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        return batch

    @torch.inference_mode()
    def segment_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        if not crops:
            return []

        batch = self.preprocess(crops)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(batch)
        else:
            logits = self.model(batch)

        pred = torch.argmax(logits, dim=1).cpu().numpy()
        return list(pred)

    def face_visibility_scores(self, crops: List[np.ndarray]) -> List[float]:
        maps = self.segment_batch(crops)
        scores = []

        for lab in maps:
            h, w = lab.shape
            area_total = h * w
            mask = np.isin(lab, list(FACE_LABELS))
            area_face = mask.sum()
            scores.append(area_face / (area_total + 1e-6))

        return scores


# =====================================================
#   SINGLETON
# =====================================================

_SEGMENTER = None

def get_face_segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = BiSeNetFaceSegmenter()
    return _SEGMENTER
