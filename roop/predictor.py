import threading
import numpy
from PIL import Image
from keras import Model

from roop.typing import Frame

# --- NSFW DETECTION DIBUANG ---
# Semua fungsi dan variabel NSFW dihapus
# Tidak ada lagi ketergantungan pada opennsfw2

# Fungsi placeholder: selalu return False (tidak pernah blokir frame)
def predict_frame(target_frame: Frame) -> bool:
    return False  # Selalu aman, tidak ada filter NSFW

def predict_image(target_path: str) -> bool:
    return False  # Selalu aman

def predict_video(target_path: str) -> bool:
    return False  # Selalu aman

# Jika ada fungsi lain yang butuh predictor, kita buang saja
def get_predictor():
    raise NotImplementedError("NSFW detection has been removed.")

def clear_predictor():
    pass  # Tidak perlu clear apa-apa
