import cv2
import numpy as np

# ================================================================
#  PERLIN NOISE GENERATOR (versi very detailed)
# ================================================================
def generate_perlin_noise(h, w, scale=3.0, seed=0):
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
#  MASK LOWER EYELID (Gaussian kecil 4px)
# ================================================================
def build_under_eye_mask(lm, h, w):
    under_idx = [94, 95, 96, 97, 98, 101, 102, 103, 104, 105]

    mask = np.zeros((h, w), np.float32)
    pts = []

    for idx in under_idx:
        x, y = lm[idx]
        pts.append([int(x), int(y) + 2])  # lebih presisi

    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1.0)

    # Gaussian kecil 4px agar mask tetap tajam & tidak melebar
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


# ================================================================
#  DARK CIRCLE BOOST
# ================================================================
def add_dark_eye_circle(frame, mask, strength):
    factor = 50
    darkness = (mask * (strength * factor)).astype(np.float32)

    darkened = frame.astype(np.float32)
    darkened[..., 0] -= darkness
    darkened[..., 1] -= darkness * 0.75
    darkened[..., 2] -= darkness * 0.7
    return np.clip(darkened, 0, 255).astype(np.uint8)


# ================================================================
#  MICRO-SHARPEN khusus bawah mata
# ================================================================
def micro_sharpen(image, mask, amount=1.4):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = np.clip(image + (image - blur) * amount, 0, 255).astype(np.uint8)
    mask3 = np.dstack([mask] * 3)
    return sharpened * mask3 + image * (1 - mask3)


# ================================================================
#  MAIN WRINKLE ENHANCER (FINAL REAL DETAIL)
# ================================================================
def enhance_wrinkles_after_gfpgan(frame, face):

    age = getattr(face, "age", None)
    if age is None:
        return frame

    # logika umur asli
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

    # MASK presisi
    mask = build_under_eye_mask(lm, h, w)
    mask3 = np.dstack([mask] * 3)

    # PERLIN ultra-detail
    noise = generate_perlin_noise(h, w, scale=3, seed=int(age * 5))
    noise = cv2.GaussianBlur(noise, (3, 3), 0)     # blur 1px (3 kernel)
    noise = cv2.cvtColor((noise * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    wrinkle_strength = strength * 0.9
    wrinkled = frame.astype(np.float32) + (noise.astype(np.float32) - 128) * wrinkle_strength

    # DARK CIRCLE
    darkened = add_dark_eye_circle(wrinkled, mask, strength)

    # MICRO SHARP khusus bawah mata
    sharp = micro_sharpen(darkened.astype(np.uint8), mask, amount=1.4)

    # FINAL — blending frame asli dikurangi (lebih kuat detail)
    blend_strength = strength * 1.1
    final = frame * (1 - mask3 * blend_strength) + sharp * (mask3 * blend_strength)

    return np.clip(final, 0, 255).astype(np.uint8)
