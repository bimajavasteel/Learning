import cv2
import numpy as np

def enhance_under_eye_wrinkles(frame, face):
    """
    Tambah kerutan di bawah mata berdasarkan usia (age dari buffalo_l).
    Fokus hanya pada area bawah mata.
    """

    # Ambil umur dari hasil buffalo_l
    age = getattr(face, "age", None)
    if age is None:
        return frame

    # ==============================
    #  ATURAN BARU SESUAI PERMINTAAN
    # ==============================
    if age >= 40:
        strength = 0.0
    elif age >= 30:
        strength = 0.25
    elif age >= 20:
        strength = 0.35
    elif age >= 13:
        strength = 0.55
    else:
        strength = 0.0

    if strength <= 0:
        return frame

    # Landmark 106 buffalo_l
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return frame

    lm = np.array(lm)

    # titik bawah mata
    under_eye_idx = [94, 95, 96, 97, 98, 101, 102, 103, 104, 105]

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)

    # Buat area mask di bawah mata
    for idx in under_eye_idx:
        x, y = lm[idx]
        x = int(x)
        y = int(y)
        cv2.circle(mask, (x, y + 5), 9, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    # kernel "sharpen" untuk menambah tekstur seperti kerutan
    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    wrinkled = cv2.filter2D(frame, -1, sharp_kernel)

    # Blend hasil sharpening dengan frame asli
    mask3 = np.dstack([mask] * 3)
    final = frame * (1 - mask3 * strength) + wrinkled * (mask3 * strength)
    final = np.clip(final, 0, 255).astype(np.uint8)

    return final
