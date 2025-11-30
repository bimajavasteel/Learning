import cv2
import numpy as np

# ============================================================
# 1. EXPRESSION DETECTOR
# ============================================================

def detect_expression(face) -> str:
    """
    Mendeteksi ekspresi wajah menggunakan landmark 106 buffalo_l.
    Kategori:
      - smile
      - frown
      - open_mouth
      - neutral
    """

    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return "neutral"

    lm = np.array(lm)

    mouth_top = lm[52]
    mouth_bottom = lm[58]
    mouth_left = lm[48]
    mouth_right = lm[54]

    brow_left = lm[19]
    brow_right = lm[24]
    brow_center = lm[21]

    # ukur jarak mulut vertical & horizontal
    mouth_open_dist = abs(mouth_bottom[1] - mouth_top[1])
    mouth_width = abs(mouth_right[0] - mouth_left[0])

    # frown (kerutan dahi)
    brow_frown_dist = abs(brow_center[1] - brow_left[1])

    if mouth_open_dist > 6:
        return "open_mouth"
    if mouth_width > 40:
        return "smile"
    if brow_frown_dist > 5:
        return "frown"

    return "neutral"


# ============================================================
# 2. BASELINE WRINKLE FROM AGE
# ============================================================

def compute_wrinkle_strength(age: float) -> float:
    """
    Wrinkle strength berdasarkan umur:
    40+ → tidak ditambah lagi
    30–39 → 0.25
    20–29 → 0.35
    13–19 → 0.55
    """
    try:
        age = float(age)
    except:
        return 0.0

    if age >= 40:
        return 0.0
    elif age >= 30:
        return 0.25
    elif age >= 20:
        return 0.35
    elif age >= 13:
        return 0.55
    return 0.0


def full_face_wrinkle(frame, face, strength: float):
    """
    Menambah detail kerutan ke seluruh area wajah (high-pass sharpen).
    Dipakai sebagai baseline sebelum ekspresi / wrinkle map.
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

    base = crop.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), 3)
    high = base - blur

    enhanced = base + high * (strength * 1.8)
    result = cv2.addWeighted(base, 1 - strength, enhanced, strength, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    frame[y1:y2, x1:x2] = result
    return frame


# ============================================================
# 3. EXPRESSION WRINKLE (WRINKLE BOOSTER)
# ============================================================

def apply_expression_wrinkle(frame, face, expression: str, base_strength: float):
    """
    Booster kerutan berdasarkan ekspresi:
      smile       → +30%
      frown       → +40%
      open_mouth  → +15%
      neutral     → +0%
    """

    if base_strength <= 0:
        return frame

    if expression == "smile":
        strength = base_strength * 1.30
    elif expression == "open_mouth":
        strength = base_strength * 1.15
    elif expression == "frown":
        strength = base_strength * 1.40
    else:
        strength = base_strength

    x1, y1, x2, y2 = map(int, face.bbox)
    H, W = frame.shape[:2]

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame

    base = crop.astype(np.float32)
    blur = cv2.GaussianBlur(base, (0, 0), 3)
    high = base - blur

    wrinkle = base + high * (strength * 2.0)
    result = cv2.addWeighted(base, 1 - strength, wrinkle, strength, 0)

    frame[y1:y2, x1:x2] = np.clip(result, 0, 255).astype(np.uint8)
    return frame


# ============================================================
# 4. SMART WRINKLE MAP (region-aware wrinkles)
# ============================================================

def _poly_mask(shape, pts, dilation=3):
    mask = np.zeros(shape[:2], np.float32)
    hull = cv2.convexHull(np.array(pts).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1.0)

    if dilation > 0:
        k = dilation * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return mask


def build_wrinkle_map(face, shape, expression: str):

    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return np.zeros(shape[:2], np.float32)

    lm = np.array(lm)
    H, W = shape[:2]

    mask = np.zeros((H, W), np.float32)

    # Crow’s feet
    left_crow = lm[[94, 95, 96, 97]]
    right_crow = lm[[101, 102, 103, 104]]

    # Under-eye
    under_left = lm[[94,95,96,97,98]]
    under_right = lm[[101,102,103,104,105]]

    # Nasolabial folds
    naso_left = lm[[46,47,58,67,68]]
    naso_right = lm[[53,54,56,65,66]]

    # Smile lines
    smile_left = lm[[48,49,59,60]]
    smile_right = lm[[53,54,64,65]]

    # Forehead lines
    forehead = lm[[17,18,19,20,21,22,23,24]] + np.array([0, -25])

    # Glabella (frown lines)
    glabella = lm[[21,22,27]] + np.array([0, -10])

    # Chin area
    chin = lm[[57,58,59,60]] + np.array([0, 18])

    # Expression mapping
    if expression == "smile":
        mask += _poly_mask((H, W), left_crow)
        mask += _poly_mask((H, W), right_crow)
        mask += _poly_mask((H, W), naso_left)
        mask += _poly_mask((H, W), naso_right)
        mask += _poly_mask((H, W), smile_left)
        mask += _poly_mask((H, W), smile_right)

    elif expression == "frown":
        mask += _poly_mask((H, W), glabella)
        mask += _poly_mask((H, W), forehead)

    elif expression == "open_mouth":
        mask += _poly_mask((H, W), chin)
        mask += _poly_mask((H, W), naso_left)
        mask += _poly_mask((H, W), naso_right)

    else:  # neutral
        mask += _poly_mask((H, W), under_left, dilation=2)
        mask += _poly_mask((H, W), under_right, dilation=2)

    mask = np.clip(mask, 0, 1)
    mask = cv2.GaussianBlur(mask, (49, 49), 0)

    return mask


def apply_smart_wrinkle_map(frame, face, expression: str, strength: float):

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

    detail = base + high * (strength * 1.8)
    blended = base * (1 - mask3) + detail * (mask3 * strength * 1.2)

    frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame
