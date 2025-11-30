import cv2
import numpy as np

def enhance_under_eye_wrinkles(frame, face):
    import cv2
    import numpy as np
    age = getattr(face, "age", None)
    if age is None:
        return frame

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

    if age >= 40:
        dark_strength = 0.05
    elif age >= 30:
        dark_strength = 0.10
    elif age >= 20:
        dark_strength = 0.14
    elif age >= 13:
        dark_strength = 0.18
    else:
        dark_strength = 0.0

    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return frame

    lm = np.array(lm)
    under_eye_idx = [94, 95, 96, 97, 98, 101, 102, 103, 104, 105]

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)

    for idx in under_eye_idx:
        x, y = lm[idx]
        cv2.circle(mask, (int(x), int(y + 5)), 9, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask3 = np.dstack([mask] * 3)

    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    wrinkled = cv2.filter2D(frame, -1, sharp_kernel)

    final = frame * (1 - mask3 * strength) + wrinkled * (mask3 * strength)
    final = final.astype(np.uint8)

    if dark_strength > 0:
        f32 = final.astype(np.float32)
        darkened = f32 * (1.0 - (mask3 * dark_strength))
        final = (darkened * mask3 + f32 * (1 - mask3)).astype(np.uint8)

    return final
