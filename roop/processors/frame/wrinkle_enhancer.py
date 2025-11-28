import cv2
import numpy as np

def enhance_under_eye_wrinkles(frame, face):
    """
    Tambah kerutan di bawah mata berdasarkan usia (age dari buffalo_l).
    Fokus hanya pada area bawah mata.
    """

    age = getattr(face, "age", None)
    if age is None:
        return frame  # tidak ada prediksi umur → skip

    # aturan kerutan
    if age >= 40:
        return frame  # matikan kerutan
    elif age >= 30:
        strength = 0.07
    elif age >= 20:
        strength = 0.15
    else:
        strength = 0.0

    if strength <= 0:
        return frame

    # landmark 106 (buffalo_l)
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return frame

    lm = np.array(lm)

    # titik bawah mata (landmark)
    # kiri  :  94, 95, 96, 97, 98
    # kanan :  101,102,103,104,105
    under_eye_idx = [94,95,96,97,98, 101,102,103,104,105]

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)

    # buat area di bawah mata
    for idx in under_eye_idx:
        x, y = lm[idx]
        x = int(x)
        y = int(y)

        # area kerutan sedikit di bawah mata
        cv2.circle(mask, (x, y + 5), 9, 1.0, -1)

    # haluskan mask
    mask = cv2.GaussianBlur(mask, (31, 31), 0)

    # buat versi tajam (sharpen) untuk meniru kerutan
    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    wrinkled = cv2.filter2D(frame, -1, sharp_kernel)

    # blend proporsional sesuai strength
    mask3 = np.dstack([mask]*3)
    final = frame * (1 - mask3 * strength) + wrinkled * (mask3 * strength)
    final = np.clip(final, 0, 255).astype(np.uint8)

    return final
