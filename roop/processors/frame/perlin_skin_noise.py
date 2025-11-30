import numpy as np
import cv2
import math

# ============================================================
#  PERLIN NOISE GENERATOR (fast, simple, DRY)
# ============================================================

def generate_perlin_noise(height, width, scale=32):
    """
    Generate Perlin-like noise (simplified for speed on Kaggle GPUs)

    scale: semakin besar → noise semakin halus (32–64 recommended)
    """
    try:
        # base random grid (low-res)
        grid_h = height // scale + 2
        grid_w = width // scale + 2

        grid = np.random.rand(grid_h, grid_w)

        # resize ke full-res (bicubic → halus, mirip Perlin)
        noise = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        noise = noise.astype(np.float32)

        # normalisasi 0–1
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        return noise
    except Exception:
        return np.zeros((height, width), np.float32)


# ============================================================
#  APPLY NOISE TO FACE REGION
# ============================================================

def add_subtle_skin_noise(face_img, strength=0.20):
    """
    Tambah efek tekstur kulit:
    - Perlin noise halus
    - Adaptive, tetap natural
    """

    try:
        h, w = face_img.shape[:2]

        # generate noise
        noise = generate_perlin_noise(h, w, scale=48)

        # konversi ke 3 channel
        noise3 = np.dstack([noise, noise, noise])

        # strength 0.05–0.12 recommended
        result = face_img.astype(np.float32)
        result = result * (1 - strength) + (result * (1 + (noise3 - 0.5) * strength * 2))

        return np.clip(result, 0, 255).astype(np.uint8)

    except Exception:
        return face_img
