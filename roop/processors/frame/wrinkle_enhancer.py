import cv2
import numpy as np

# ================================================================
#  PERLIN NOISE GENERATOR (untuk tekstur pori & kerutan halus)
# ================================================================
def generate_perlin_noise(h, w, scale=35.0, seed=0):
    np.random.seed(seed)
    gx = np.random.rand(h, w) * 2 - 1
    gy = np.random.rand(h, w) * 2 - 1
    grad = np.stack([gx, gy], axis=-1)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x / scale
    y = y / scale

    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1

    def dot_grid(ix, iy):
        ix = np.clip(ix, 0, w-1)
        iy = np.clip(iy, 0, h-1)
        g = grad[iy, ix]
        dx = x - ix
        dy = y - iy
        return g[...,0] * dx + g[...,1] * dy

    n00 = dot_grid(x0, y0)
    n10 = dot_grid(x1, y0)
    n01 = dot_grid(x0, y1)
    n11 = dot_grid(x1, y1)

    def smooth(t): return t*t*(3 - 2*t)

    sx = smooth(x - x0)
    sy = smooth(y - y0)

    nx0 = n00 * (1 - sx) + n10 * sx
    nx1 = n01 * (1 - sx) + n11 * sx
    nxy = nx0 * (1 - sy) + nx1 * sy

    noise = (nxy - nxy.min()) / (nxy.max() - nxy.min())
    return noise.astype(np.float32)


# ================================================================
#  MASK LOWER EYELID (berbasis landmark 106 buffalo_l)
# ================================================================
def build_under_eye_mask(lm, h, w):
    # koordinat landmark bawah mata (left+right)
    under_idx = [94, 95, 96, 97, 98, 101, 102, 103, 104, 105]

    mask = np.zeros((h, w), np.float32)
    pts = []

    for idx in under_idx:
        x, y = lm[idx]
        pts.append([int(x), int(y) + 4])

    pts = np.array(pts, dtype=np.int32)

    # fill polygon mengikuti kontur eyelid
    cv2.fillPoly(mask, [pts], 1.0)

    # blur besar supaya natural
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    return mask


# ================================================================
#  DARK EYE CIRCLES
# ================================================================
def add_dark_eye_circle(frame, mask, strength):
    darkness = (mask * (strength * 45)).astype(np.float32)
    darkened = frame.astype(np.float32)
    darkened[..., 0] -= darkness
    darkened[..., 1] -= darkness * 0.8
    darkened[..., 2] -= darkness * 0.7
    return np.clip(darkened, 0, 255).astype(np.uint8)


# ================================================================
#  MAIN WRINKLE + DARK ENHANCER
# ================================================================
def enhance_wrinkles_after_gfpgan(frame, face):
    """
    Versi final:
    - kerutan Perlin
    - dark circle
    - mask akurat lower eyelid (2D106)
    - bekerja SETELAH GFPGAN
    - tracking friendly (pakai bbox & landmark face)
    """

    age = getattr(face, "age", None)
    if age is None:
        return frame

    # ==========================================================
    # LOGIKA UMUR (pakai versi kamu)
    # ==========================================================
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

    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return frame

    lm = np.array(lm)
    h, w = frame.shape[:2]

    # ==========================================================
    # MASK bawah mata (kontur mengikuti landmark)
    # ==========================================================
    mask = build_under_eye_mask(lm, h, w)
    mask3 = np.dstack([mask] * 3)

    # ==========================================================
    # PERLIN NOISE untuk kerutan halus
    # ==========================================================
    noise = generate_perlin_noise(h, w, scale=23, seed=int(age*13))
    noise = cv2.GaussianBlur(noise, (7, 7), 0)
    noise = (noise * 255).astype(np.uint8)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

    wrinkled = frame.astype(np.float32) + (noise.astype(np.float32) - 128) * (strength * 0.45)

    # ==========================================================
    # Tambahkan DARK EYE CIRCLE natural
    # ==========================================================
    darkened = add_dark_eye_circle(wrinkled, mask, strength)

    # ==========================================================
    # Blending halus
    # ==========================================================
    final = frame * (1 - mask3 * strength) + darkened * (mask3 * strength)
    return np.clip(final, 0, 255).astype(np.uint8)
