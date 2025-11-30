import cv2
import numpy as np
import math
import random

# ============================================================
#  PERLIN NOISE (multi-octave untuk pori kulit realistis)
# ============================================================

def perlin_noise(width, height, scale=20, octaves=4, persistence=0.5, lacunarity=2.0):
    def f(x):
        return 6*x**5 - 15*x**4 + 10*x**3

    def lerp(a, b, t):
        return a + t * (b - a)

    gradients = {}
    def gradient(ix, iy):
        if (ix, iy) not in gradients:
            angle = random.random() * 2 * math.pi
            gradients[(ix, iy)] = (math.cos(angle), math.sin(angle))
        return gradients[(ix, iy)]

    noise = np.zeros((height, width), np.float32)

    for y in range(height):
        for x in range(width):
            xf = x / scale
            yf = y / scale

            ix = int(xf)
            iy = int(yf)

            fx = xf - ix
            fy = yf - iy

            tl = np.dot(gradient(ix, iy), (fx, fy))
            tr = np.dot(gradient(ix+1, iy), (fx-1, fy))
            bl = np.dot(gradient(ix, iy+1), (fx, fy-1))
            br = np.dot(gradient(ix+1, iy+1), (fx-1, fy-1))

            u = f(fx)
            v = f(fy)

            noise[y, x] = lerp(lerp(tl, tr, u), lerp(bl, br, u), v)

    result = np.zeros_like(noise)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        scaled = cv2.resize(noise, (int(width * frequency), int(height * frequency)))
        scaled = cv2.resize(scaled, (width, height))

        result += scaled * amplitude

        amplitude *= persistence
        frequency *= lacunarity

    result = cv2.normalize(result, None, 0, 1.0, cv2.NORM_MINMAX)
    return result


# ============================================================
#  WRINKLE + PORES + MICROTEXTURE
# ============================================================

def enhance_under_eye_wrinkles(frame, face):
    """
    Versi C (Hybrid):
    - wrinkle penambah
    - pori kulit (multi-octave Perlin)
    - microtexture halus
    """

    age = getattr(face, "age", None)
    if age is None:
        return frame

    # ---------------------------------------------------------
    #  ATURAN STRENGTH KERUTAN SESUAI PERMINTAAN
    # ---------------------------------------------------------
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

    under_eye_idx = [94, 95, 96, 97, 98, 101, 102, 103, 104, 105]

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)

    # ---------------------------------------------------------
    #  MASK KHUSUS AREA BAWAH MATA
    # ---------------------------------------------------------
    for idx in under_eye_idx:
        x, y = lm[idx]
        x = int(x)
        y = int(y)
        cv2.circle(mask, (x, y + 5), 12, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (41, 41), 0)

    # ---------------------------------------------------------
    #  PERLIN MULTI-OCTAVE SEBAGAI PORI KULIT
    # ---------------------------------------------------------
    pores = perlin_noise(
        width=w,
        height=h,
        scale=18,       # skala kecil = detail halus
        octaves=5,      # lebih banyak oktav = lebih realistis
        persistence=0.55,
        lacunarity=2.3
    )

    pores = cv2.GaussianBlur(pores, (7, 7), 0)
    pores3 = np.dstack([pores * 255] * 3).astype(np.uint8)

    # ---------------------------------------------------------
    #  KERUTAN (SHARPENING KECIL)
    # ---------------------------------------------------------
    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 6, -1],
        [0, -1, 0]
    ])

    wrinkled = cv2.filter2D(frame, -1, sharp_kernel)

    # ---------------------------------------------------------
    #  BLEND: wrinkle + pores → hybrid
    # ---------------------------------------------------------
    mask3 = np.dstack([mask] * 3)

    pores_strength = strength * 0.65      # pori lebih lembut daripada wrinkle
    wrinkle_strength = strength * 0.45    # wrinkle soft supaya tidak kasar

    combined = (
        frame * (1 - mask3 * (pores_strength + wrinkle_strength)) +
        wrinkled * (mask3 * wrinkle_strength) +
        pores3 * (mask3 * pores_strength)
    )

    final = np.clip(combined, 0, 255).astype(np.uint8)
    return final
