from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import (
    get_one_face,
    get_many_faces,
    find_similar_face,
    smart_face_tracking,
    detect_occlusion
)
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

# Reswapper specific configuration
RESWAPPER_INPUT_SIZE = 256
RESWAPPER_MEAN = [0.5, 0.5, 0.5]
RESWAPPER_STD = [0.5, 0.5, 0.5]

class ReswapperWrapper:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_reswapper_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=RESWAPPER_MEAN, std=RESWAPPER_STD)
        ])
        
    def load_reswapper_model(self, model_path: str) -> Any:
        """Load Reswapper model from .pth file"""
        try:
            # Load model architecture and weights
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model_weights = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    model_weights = checkpoint['model']
                else:
                    model_weights = checkpoint
            else:
                model_weights = checkpoint
                
            # Initialize model (you might need to adjust based on Reswapper architecture)
            # This is a placeholder - you'll need the actual Reswapper model class
            model = self.create_reswapper_model()
            
            # Load weights
            model.load_state_dict(model_weights, strict=False)
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Reswapper 256 model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading Reswapper model: {e}")
            raise
    
    def create_reswapper_model(self) -> nn.Module:
        """
        Create Reswapper model architecture.
        NOTE: You'll need to replace this with the actual Reswapper model class
        """
        # Placeholder - you need to import the actual Reswapper model architecture
        class SimpleReswapper(nn.Module):
            def __init__(self):
                super().__init__()
                # This should be replaced with actual Reswapper architecture
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 3, 3, 1, 1),
                    nn.Tanh()
                )
                
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        return SimpleReswapper()
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face for Reswapper"""
        # Resize to model input size
        face_resized = cv2.resize(face_image, (RESWAPPER_INPUT_SIZE, RESWAPPER_INPUT_SIZE))
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # Apply transforms
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return tensor
    
    def postprocess_face(self, tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
        """Convert model output back to image"""
        with torch.no_grad():
            output = tensor.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            # Denormalize
            output = (output * 0.5) + 0.5
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            # Convert RGB to BGR
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            # Resize back to original size
            output = cv2.resize(output, original_size)
            return output
    
    def get(self, frame: Frame, target_face: Face, source_face: Face, paste_back: bool = True) -> Frame:
        """
        Main face swapping function compatible with InsightFace interface
        """
        try:
            # Extract face region from target
            bbox = target_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return frame
            
            original_size = (face_region.shape[1], face_region.shape[0])
            
            # Preprocess target face
            target_tensor = self.preprocess_face(face_region)
            
            # TODO: Implement actual face swapping logic with Reswapper
            # This is a placeholder - you need to implement the actual swapping
            swapped_tensor = self.model(target_tensor)
            
            # Postprocess
            swapped_face = self.postprocess_face(swapped_tensor, original_size)
            
            if paste_back:
                # Paste swapped face back to frame
                frame[y1:y2, x1:x2] = swapped_face
            
            return frame
            
        except Exception as e:
            print(f"❌ Error in Reswapper face swap: {e}")
            return frame


def get_face_swapper() -> Any:
    """
    Inisialisasi model Reswapper 256.
    """
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/reswapper_256-1567500.pth')
            FACE_SWAPPER = ReswapperWrapper(model_path)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """
    Pastikan model Reswapper sudah ke-download sebelum mulai.
    """
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/somanchiu/reswapper/resolve/main/reswapper_256-1567500.pth'
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


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """
    Fungsi swap dasar menggunakan Reswapper.
    """
    if source_face is None or target_face is None:
        return temp_frame

    return get_face_swapper().get(
        temp_frame,
        target_face,
        source_face,
        paste_back=True
    )


# ... (sisanya tetap sama - process_frame, process_frames, dll)
