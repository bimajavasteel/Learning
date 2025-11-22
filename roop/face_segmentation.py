# roop/face_segmentation.py

import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

# ======================================================
#   KONFIG
# ======================================================

# Sesuaikan path ini dengan lokasi weight BiSeNet kamu
BISENET_WEIGHTS_PATH = "models/face_parsing_bisenet.pth"
BISENET_INPUT_SIZE = 512

# Label-class untuk area wajah (skin + facial parts) tergantung dataset
# Ini asumsi umum CelebAMask-HQ / face-parsing:
# 1: skin, 2: left brow, 3: right brow, 4: left eye, 5: right eye,
# 6: nose, 7: upper lip, 9: lower lip, dll.
FACE_LIKE_LABELS = {1, 2, 3, 4, 5, 6, 7, 9}


# ======================================================
#   MODEL Bisenet (SIMPLE VERSION)
#   (arsitektur dipersingkat, fokus ke integrasi)
# ======================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetSimple(nn.Module):
    """
    Versi ringkas BiSeNet untuk face parsing.
    Di sini fokus ke: input -> feature -> output logits kelas.
    Pastikan weight .pth yang kamu pakai kompatibel.
    """

    def __init__(self, n_classes: int = 19):
        super().__init__()
        # Sangat dipersingkat; asumsi weight kompatibel.
        # Kalau weight kamu pakai arsitektur lain, sesuaikan sendiri.
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
        x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)
        return x


# ======================================================
#   WRAPPER SEGMENTER
# ======================================================

class BiSeNetFaceSegmenter:
    def __init__(self, weights_path: str = BISENET_WEIGHTS_PATH, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = BiSeNetSimple(n_classes=19).to(self.device)
        self.model.eval()

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"BiSeNet weights tidak ditemukan: {weights_path}. "
                "Download/taruh weight face-parsing di path tersebut."
            )

        state = torch.load(weights_path, map_location=self.device)
        # Jika state_dict nest / ada key 'state_dict'
        if "state_dict" in state:
            state = {k.replace("module.", "").replace("model.", ""): v for k, v in state["state_dict"].items()}
        else:
            state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

        # Transform gambar
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Gunakan half precision kalau GPU support
        self.use_amp = (self.device.type == "cuda")

    def _preprocess(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        crops: list gambar RGB (np.uint8) bentuk (H, W, 3)
        """
        tensors = []
        for img in crops:
            img_resized = cv2.resize(img, (BISENET_INPUT_SIZE, BISENET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            tensor = self.transform(img_resized)
            tensors.append(tensor)

        batch = torch.stack(tensors, dim=0)
        return batch.to(self.device)

    @torch.inference_mode()
    def segment_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Return: list of label maps (H, W) dengan nilai kelas (int)
        """
        if len(crops) == 0:
            return []

        x = self._preprocess(crops)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(x)
        else:
            logits = self.model(x)

        # logits: (B, C, H, W) -> pred: (B, H, W)
        preds = torch.argmax(logits, dim=1)
        preds = preds.detach().cpu().numpy()

        # Resize ke ukuran input BiSeNet (kalau mau)
        # Di sini sudah sama dengan BISENET_INPUT_SIZE
        return list(preds)

    def face_visibility_scores(
        self,
        crops: List[np.ndarray]
    ) -> List[float]:
        """
        Hitung rasio area wajah (skin + facial parts) terhadap area crop.
        Semakin kecil → kemungkinan occlusion besar.
        Return: list float [0..1]
        """
        label_maps = self.segment_batch(crops)
        scores = []

        for lab in label_maps:
            h, w = lab.shape
            area_total = h * w

            face_mask = np.isin(lab, list(FACE_LIKE_LABELS))
            area_face = int(face_mask.sum())
            ratio = area_face / float(area_total + 1e-6)
            scores.append(ratio)

        return scores


# ======================================================
#   HELPER GLOBAL (SINGLETON)
# ======================================================

_SEGMENTER: BiSeNetFaceSegmenter = None


def get_face_segmenter() -> BiSeNetFaceSegmenter:
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = BiSeNetFaceSegmenter(BISENET_WEIGHTS_PATH)
    return _SEGMENTER
