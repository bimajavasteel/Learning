import cv2
import numpy as np

# ============================================================
#  Landmark region helper
# ============================================================

def _poly(frame_shape, pts, dilation=3):
    """
    Membuat polygon mask berdasarkan landmark.
    pts: list koordinat
    dilation: memperbesar area agar transisi halus
    """
    mask = np.zeros(frame_shape[:2], np.float32)
    hull = cv2.convexHull(np.array(pts).astype(np.int32))

    cv2.fillConvexPoly(mask, hull, 1.0)

    if dilation > 0:
        k = dilation * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask


# ============================================================
#  Main wrinkle-map builder
# ============================================================

def build_wrinkle_map(face, frame_shape, expression: str):
    """
    Membuat peta area kerutan berdasarkan:
      - ekspresi (smile, frown, open_mouth, neutral)
      - landmark 106 buffalo_l
    """

    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return np.zeros(frame_shape[:2], np.float32)

    lm = np.array(lm)
    H, W = frame_shape[:2]

    wrinkle_mask = np.zeros((H, W), np.float32)

    # ========== REGION DEFINITIONS (berbasis landmark 106) ==========
    # Crow’s feet (ujung mata)
    left_crow = lm[[94, 95, 96, 97]]
    right_crow = lm[[101, 102, 103, 104]]

    # Under eye
    under_left = lm[[94, 95, 96, 97, 98]]
    under_right = lm[[101, 102, 103, 104, 105]]

    # Nasolabial fold (area pipi senyum)
    naso_left = lm[[46, 47, 58, 67, 68]]
    naso_right = lm[[53, 54, 56, 65, 66]]

    # Smile lines (lip corner)
    smile_left = lm[[48, 49, 59, 60]]
    smile_right = lm[[53, 54, 64, 65]]

    # Forehead (atas alis)
    forehead = lm[[17, 18, 19, 20, 21, 22, 23, 24]] + np.array([0, -25])

    # Glabella (antara alis → frown lines)
    glabella = lm[[21, 22, 27]] + np.array([0, -10])

    # Chin / under-mouth
    chin = lm[[57, 58, 59, 60]] + np.array([0, 18])

    # ============================================================
    #  EXPRESSION → ACTIVE REGIONS
    # ============================================================

    if expression == "smile":
        wrinkle_mask += _poly((H, W), left_crow)
        wrinkle_mask += _poly((H, W), right_crow)
        wrinkle_mask += _poly((H, W), naso_left)
        wrinkle_mask += _poly((H, W), naso_right)
        wrinkle_mask += _poly((H, W), smile_left)
        wrinkle_mask += _poly((H, W), smile_right)

    elif expression == "open_mouth":
        wrinkle_mask += _poly((H, W), chin)
        wrinkle_mask += _poly((H, W), naso_left)
        wrinkle_mask += _poly((H, W), naso_right)

    elif expression == "frown":
        wrinkle_mask += _poly((H, W), glabella)
        wrinkle_mask += _poly((H, W), forehead)

    else:  # neutral
        wrinkle_mask += _poly((H, W), under_left, dilation=2)
        wrinkle_mask += _poly((H, W), under_right, dilation=2)

    # Normalisasi (0–1)
    wrinkle_mask = np.clip(wrinkle_mask, 0, 1)

    # Halus
    wrinkle_mask = cv2.GaussianBlur(wrinkle_mask, (49, 49), 0)

    return wrinkle_mask


# ============================================================
#  Apply wrinkle enhancement only on wrinkle map
# ============================================================

def apply_smart_wrinkle_map(frame, face, expression: str, strength: float):
    """
    Menguatkan detail kerutan hanya pada area mask yang sesuai ekspresi.
    """

    if strength <= 0:
        return frame

    x1, y1, x2, y2 = map(int, face.bbox)
    H, W = frame.shape[:2]

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame

    mask = build_wrinkle_map(face, crop.shape, expression)
    mask3 = np.dstack([mask] * 3)

    base = crop.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), 3)
    high = base - blur

    wrinkled = base + high * (strength * 1.8)
    blended = base * (1 - mask3) + wrinkled * (mask3 * strength * 1.2)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    frame[y1:y2, x1:x2] = blended
    return frame
