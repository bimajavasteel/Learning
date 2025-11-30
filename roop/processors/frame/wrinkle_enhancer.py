import cv2
import numpy as np

def enhance_under_eye_wrinkles(frame, face):
    """
    Improved under-eye wrinkle + dark circle enhancer.
    - Samakan logika umur untuk wrinkle & darkening (strength)
    - Gunakan polygon mask (convex hull) mengikuti landmark
    - Adaptive radius & blur berdasarkan ukuran bbox
    - Sharpen hanya pada crop, bukan keseluruhan frame
    - Blend & darken halus sehingga tidak menghasilkan edge/halo
    """
    age = getattr(face, "age", None)
    if age is None:
        return frame

    # =========================
    # 1) AGE-BASED STRENGTH (SAMA untuk wrinkle & dark)
    # =========================
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

    dark_strength = strength  # disamakan seperti permintaan

    # =========================
    # 2) landmark + bbox safe
    # =========================
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return frame
    lm = np.array(lm, dtype=np.float32)

    # gunakan titik bawah mata (adaptif)
    under_eye_idx = [94,95,96,97,98, 101,102,103,104,105]
    pts = lm[under_eye_idx]

    # bounding box area kerja (tambahan margin)
    x1, y1, x2, y2 = map(int, face.bbox)
    h_frame, w_frame = frame.shape[:2]
    pad_x = max(8, int((x2 - x1) * 0.15))
    pad_y = max(8, int((y2 - y1) * 0.10))
    sx = max(0, x1 - pad_x); sy = max(0, y1 - pad_y)
    ex = min(w_frame, x2 + pad_x); ey = min(h_frame, y2 + pad_y)

    # normalize pts relatif ke crop origin
    rel_pts = pts.copy()
    rel_pts[:,0] = rel_pts[:,0] - sx
    rel_pts[:,1] = rel_pts[:,1] - sy

    crop = frame[sy:ey, sx:ex]
    if crop is None or crop.size == 0:
        return frame

    ch, cw = crop.shape[:2]

    # =========================
    # 3) buat mask tear-trough (convex hull + offset ke bawah)
    # =========================
    # shift landmark sedikit ke bawah supaya menutupi tear-trough alami
    offset_y = max(1, int((ey - sy) * 0.04))
    shifted = rel_pts + np.array([0, offset_y], dtype=np.float32)

    try:
        hull = cv2.convexHull(shifted.astype(np.int32))
    except Exception:
        hull = np.array(shifted.astype(np.int32))

    mask = np.zeros((ch, cw), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)

    # perbesar area horizontal sedikit agar mengikuti kontur pipi
    # dilasi via gaussian blur with kernel dependent on crop size
    blur_k = int(max(3, min(ch, cw) * 0.12))
    if blur_k % 2 == 0:
        blur_k += 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    mask3 = np.dstack([mask]*3)

    # =========================
    # 4) sharpen hanya di crop
    # =========================
    base = crop.astype(np.float32)

    # gunakan bilateral + small gaussian sebelum sharpening untuk mengurangi noise
    smooth = cv2.bilateralFilter(base.astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)
    blur = cv2.GaussianBlur(smooth, (0,0), sigmaX=2)
    high_pass = smooth - blur

    # kontrol kekuatan sharpen relatif ke strength dan ukuran crop
    sharpened = base + high_pass * (strength * 2.0)

    # =========================
    # 5) Blend wrinkle (lebih halus)
    # =========================
    # blend hanya di area mask, dengan soft transition: gunakan power curve untuk mask
    mask_soft = np.clip(mask, 0.0, 1.0)
    mask_soft = np.power(mask_soft, 0.9)  # sedikit memperkuat tengah mask
    mask3_soft = np.dstack([mask_soft]*3)

    result_wrinkle = base * (1.0 - mask3_soft * strength) + sharpened * (mask3_soft * strength)
    result_wrinkle = np.clip(result_wrinkle, 0, 255).astype(np.float32)

    # =========================
    # 6) Dark-circle (apply under-eye darkening) — mengikuti same strength
    # =========================
    if dark_strength > 0:
        # buat dark multiply hanya pada mask area, dengan very subtle gaussian
        dark_amount = dark_strength * 0.6  # scale down supaya tidak over-dark
        darked = result_wrinkle * (1.0 - (mask3_soft * dark_amount))
        # combine: darked inside mask, original outside
        result_final = darked * mask3_soft + result_wrinkle * (1.0 - mask3_soft)
        result_final = np.clip(result_final, 0, 255).astype(np.uint8)
    else:
        result_final = np.clip(result_wrinkle, 0, 255).astype(np.uint8)

    # =========================
    # 7) paste back ke frame dengan feather edge untuk safety
    # =========================
    # optional extra blur on full crop border to avoid seam
    edge_blur_k = int(max(3, min(ch, cw) * 0.02))
    if edge_blur_k % 2 == 0:
        edge_blur_k += 1
    alpha = cv2.GaussianBlur(mask3_soft, (edge_blur_k, edge_blur_k), 0)

    out_crop = (result_final.astype(np.float32) * alpha + base * (1.0 - alpha)).astype(np.uint8)
    frame[sy:ey, sx:ex] = out_crop

    return frame
