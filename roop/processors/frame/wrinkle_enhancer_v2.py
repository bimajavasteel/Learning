import cv2
import numpy as np

# ================================================================
#  PERLIN NOISE GENERATOR — HARD MODE (lebih kasar & kuat)
# ================================================================
def generate_perlin_noise(h, w, scale=3.0, seed=0):
    np.random.seed(seed)
    gx = np.random.randn(h, w)
    gy = np.random.randn(h, w)
    grad = np.stack([gx, gy], axis=-1)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x = x / scale
    y = y / scale

    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1

    def dot(ix, iy):
        ix = np.clip(ix, 0, w-1)
        iy = np.clip(iy, 0, h-1)
        g = grad[iy, ix]
        dx = x - ix
        dy = y - iy
        return g[...,0] * dx + g[...,1] * dy

    n00 = dot(x0, y0)
    n10 = dot(x1, y0)
    n01 = dot(x0, y1)
    n11 = dot(x1, y1)

    def smooth(t): return t * t * (3 - 2 * t)

    sx = smooth(x - x0)
    sy = smooth(y - y0)

    nx0 = n00 * (1 - sx) + n10 * sx
    nx1 = n01 * (1 - sx) + n11 * sx
    nxy = nx0 * (1 - sy) + nx1 * sy

    # normalisasi
    noise = (nxy - nxy.min()) / (nxy.max() - nxy.min())
    return noise.astype(np.float32)


# ================================================================
#  MASK BAWAH MATA — SUPER PRECISE (blur 3px)
# ================================================================
def build_under_eye_mask(lm, h, w):
    under_idx = [94,95,96,97,98,101,102,103,104,105]
    mask = np.zeros((h, w), np.float32)

    pts = []
    for i in under_idx:
        x, y = lm[i]
        pts.append([int(x), int(y) + 1])

    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)  # 3px super tight
    return mask


# ================================================================
#  DARK CIRCLE — HARD MODE
# ================================================================
def add_dark_eye_circle(frame, mask, strength):
    # dark factor lebih tinggi
    factor = 90
    darkness = mask * (strength * factor)

    darkened = frame.astype(np.float32)
    darkened[...,0] -= darkness       # blue
    darkened[...,1] -= darkness * 0.85
    darkened[...,2] -= darkness * 0.82
    return np.clip(darkened, 0, 255).astype(np.uint8)


# ================================================================
#  MICRO-CONTRAST EXTREME
# ================================================================
def micro_contrast(img, mask, amount=2.0):
    blur = cv2.GaussianBlur(img, (1,1), 0)
    mc = np.clip(img + (img - blur) * amount, 0, 255).astype(np.uint8)
    mask3 = np.dstack([mask]*3)
    return mc * mask3 + img * (1 - mask3)


# ================================================================
#  MAIN WRINKLE ENHANCER — HARD MODE
# ================================================================
def enhance_wrinkles_after_gfpgan(frame, face):

    age = getattr(face, "age", None)
    if age is None:
        return frame

    if age >= 40:
        strength = 0.00
    elif age >= 30:
        strength = 0.35
    elif age >= 20:
        strength = 0.48
    elif age >= 13:
        strength = 0.65
    else:
        strength = 0.0

    if strength <= 0:
        return frame

    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return frame

    lm = np.array(lm)
    h, w = frame.shape[:2]

    mask = build_under_eye_mask(lm, h, w)
    mask3 = np.dstack([mask]*3)

    # PERLIN HARD MODE
    noise = generate_perlin_noise(h, w, 3, seed=int(age*7))
    noise = cv2.GaussianBlur(noise, (1,1), 0)
    noise = cv2.cvtColor((noise*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    wr = frame.astype(np.float32) + (noise.astype(np.float32) - 128) * (strength * 1.8)

    dark = add_dark_eye_circle(wr, mask, strength)

    sharp = micro_contrast(dark, mask, amount=1.8)

    # HARD BLEND (frame asli minim)
    final = frame * (1 - mask3 * (strength * 0.25)) + sharp * (mask3 * (strength * 1.75))

    return np.clip(final, 0, 255).astype(np.uint8)
