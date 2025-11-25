# ======= FULL FACE MORPH BLEND (DELAUNAY) - Integrasi ROOP (Siap Paste) =======
# Persyaratan: insightface buffalo_l sudah ada (provides 68-landmark). Tidak perlu library tambahan.
# Pastikan file ini diimport di modul yang sama dengan fungsi process_frame/process_video.

from typing import Any, List, Callable, Optional, Tuple
import copy
import math
import threading
import cv2
import numpy as np

# --- jika modul lain sudah diimport di file utama, jangan ulangi. Ini lengkap untuk ditempel. ---
# Mengasumsikan roop.globals, get_face_swapper(), adapt_bbox_for_pose() sudah ada di file utama.

# ----------------- Helper untuk landmark 68 (InsightFace buffalo_l) -----------------
def _ensure_68_kps_from_face(face: Any) -> Optional[np.ndarray]:
    """
    Ambil 68 landmark 2D dari objek face (InsightFace buffalo_l).
    Prioritas:
    - face.landmark_3d_68 (3D projected) -> gunakan x,y
    - face.kps (jika sudah ada 68) -> gunakan
    Return: numpy array shape (68,2) float32 atau None
    """
    if face is None:
        return None

    # Cek landmark_3d_68 (InsightFace buffalo_l biasanya punya)
    if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
        kps = np.array(face.landmark_3d_68, dtype=np.float32)
        if kps.ndim == 2 and kps.shape[0] >= 68:
            # ambil 68, gunakan kolom (x,y)
            return kps[:68, :2].copy()

    # fallback: face.kps (sering face.kps berisi 106 atau 68)
    if hasattr(face, "kps") and face.kps is not None:
        kps = np.array(face.kps, dtype=np.float32)
        if kps.ndim == 2 and kps.shape[0] >= 68:
            return kps[:68, :2].copy()

    return None

def _rect_from_points(points: np.ndarray) -> Tuple[int,int,int,int]:
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return (x_min, y_min, x_max, y_max)

def _get_delaunay_triangles(rect: Tuple[int,int,int,int], points: np.ndarray) -> List[Tuple[int,int,int]]:
    """
    rect: (x,y,w,h)
    points: Nx2 (float)
    return: list of index triplet per triangle
    """
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangle_list = subdiv.getTriangleList()
    pts = points.tolist()
    triangles_idx = []

    def _find_index(pt):
        # cari index terdekat
        best_i = -1
        best_d = 1e9
        for i, p in enumerate(pts):
            d = (p[0]-pt[0])**2 + (p[1]-pt[1])**2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    for t in triangle_list:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])
        i1 = _find_index(p1)
        i2 = _find_index(p2)
        i3 = _find_index(p3)
        if i1 is None or i2 is None or i3 is None:
            continue
        if i1 != i2 and i2 != i3 and i1 != i3:
            triangles_idx.append((i1, i2, i3))

    # dedup
    unique = []
    seen = set()
    for tri in triangles_idx:
        key = tuple(sorted(tri))
        if key not in seen:
            seen.add(key)
            unique.append(tri)
    return unique

def _warp_triangle(img_src: np.ndarray, img_dst: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> None:
    """
    Warp triangle region dari img_src -> img_dst in-place pada img_dst.
    t_src, t_dst: (3,2) float32
    """
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        return

    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t_src[i][0] - x1), (t_src[i][1] - y1)))
        t2_rect.append(((t_dst[i][0] - x2), (t_dst[i][1] - y2)))

    t1_rect = np.float32(t1_rect)
    t2_rect = np.float32(t2_rect)

    # crop source patch
    src_patch = img_src[y1:y1+h1, x1:x1+w1]
    if src_patch.size == 0:
        return

    # Affine transform
    M = cv2.getAffineTransform(t1_rect, t2_rect)
    warped_patch = cv2.warpAffine(src_patch, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # mask untuk triangle dst
    mask = np.zeros((h2, w2), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), 255)

    dst_area = img_dst[y2:y2+h2, x2:x2+w2]
    if dst_area.shape[0] != h2 or dst_area.shape[1] != w2:
        # shape mismatch safety
        dst_area = cv2.resize(dst_area, (w2, h2))

    mask_3ch = cv2.merge([mask, mask, mask])
    masked_dst = cv2.bitwise_and(dst_area, cv2.bitwise_not(mask_3ch))
    masked_src = cv2.bitwise_and(warped_patch, mask_3ch)
    result = cv2.add(masked_dst, masked_src)
    img_dst[y2:y2+h2, x2:x2+w2] = result

# ----------------- Fungsi morph utama -----------------
def morph_target_to_source(full_frame: np.ndarray, target_face: Any, source_face: Any, shape_mix: float = 1.0) -> np.ndarray:
    """
    Lakukan full Delaunay face morph: geometri wajah target dimodifikasi
    ke intermediate shape = target + shape_mix * (source - target)
    Return frame baru (np.uint8)
    """
    # ambil kps 68
    src_kps = _ensure_68_kps_from_face(source_face)
    tgt_kps = _ensure_68_kps_from_face(target_face)
    if src_kps is None or tgt_kps is None:
        # fallback: return original
        return full_frame

    img = full_frame.copy().astype(np.uint8)
    h, w = img.shape[:2]

    # clamp kps ke image bounds
    src_kps[:,0] = np.clip(src_kps[:,0], 0, w-1)
    src_kps[:,1] = np.clip(src_kps[:,1], 0, h-1)
    tgt_kps[:,0] = np.clip(tgt_kps[:,0], 0, w-1)
    tgt_kps[:,1] = np.clip(tgt_kps[:,1], 0, h-1)

    # intermediate points
    pts_inter = tgt_kps + shape_mix * (src_kps - tgt_kps)

    # bounding rect from target points with margin
    x_min, y_min, x_max, y_max = _rect_from_points(tgt_kps)
    margin = max(10, int(0.02 * max(w, h)))
    rect = (int(x_min - margin), int(y_min - margin), int(x_max - x_min + 2*margin), int(y_max - y_min + 2*margin))

    # build Delaunay triangles using target points (so triangles map source->intermediate)
    try:
        triangles = _get_delaunay_triangles(rect, tgt_kps)
    except Exception:
        return full_frame

    morphed = img.copy()

    for tri in triangles:
        i1, i2, i3 = tri
        t_src = np.array([tgt_kps[i1], tgt_kps[i2], tgt_kps[i3]], dtype=np.float32)  # from target pos
        t_dst = np.array([pts_inter[i1], pts_inter[i2], pts_inter[i3]], dtype=np.float32) # to intermediate
        try:
            _warp_triangle(img, morphed, t_src, t_dst)
        except Exception:
            continue

    # blending area wajah (feather)
    hull = cv2.convexHull(np.int32(pts_inter))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask_blur = cv2.GaussianBlur(mask, (31,31), 0)

    mask_f = mask_blur.astype(np.float32) / 255.0
    mask_f = cv2.merge([mask_f, mask_f, mask_f])

    out = (morphed.astype(np.float32)*mask_f + img.astype(np.float32)*(1.0-mask_f)).astype(np.uint8)
    return out

# ----------------- Integrasi swap_face (replace existing) -----------------
def swap_face(source_face: Any, target_face: Any, temp_frame: np.ndarray) -> np.ndarray:
    """
    Flow:
    1) adapt_bbox_for_pose(target_face)
    2) morph target geometry -> intermediate (mengikuti source) (shape_mix configurable)
    3) run inswapper on morphed frame
    4) colour-correct & seamlessClone blending (mengembalikan ke tone morphed_frame)
    """
    if source_face is None or target_face is None:
        return temp_frame

    # adapt bbox dulu
    try:
        adapt_bbox_for_pose(target_face, temp_frame.shape)
    except Exception:
        pass

    # ambil shape_mix dari globals jika ada (default 0.9)
    shape_mix = getattr(roop.globals, 'shape_mix', 0.9)
    # clamp
    shape_mix = float(max(0.0, min(1.0, shape_mix)))

    # morph target geometry dulu (hasil: morphed_frame)
    try:
        morphed_frame = morph_target_to_source(temp_frame, target_face, source_face, shape_mix=shape_mix)
    except Exception:
        morphed_frame = temp_frame.copy()

    # jalankan inswapper pada morphed frame
    try:
        swapped_frame = get_face_swapper().get(
            morphed_frame,
            target_face,
            source_face,
            paste_back=True
        )
    except Exception:
        # fallback ke swap pada original jika inswapper error
        try:
            swapped_frame = get_face_swapper().get(
                temp_frame,
                target_face,
                source_face,
                paste_back=True
            )
        except Exception:
            return temp_frame

    # lakukan colour correction / seamless clone dari swapped_frame -> morphed_frame
    # gunakan hull dari target_face kps (intermediate) jika ada
    try:
        tgt_kps = _ensure_68_kps_from_face(target_face)
        if tgt_kps is not None:
            # setelah morph, center clone di bbox center target_face
            x1, y1, x2, y2 = target_face.bbox.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(swapped_frame.shape[1], x2); y2 = min(swapped_frame.shape[0], y2)
            center = ((x1 + x2)//2, (y1 + y2)//2)

            hull = cv2.convexHull(np.int32(tgt_kps))
            mask = np.zeros(swapped_frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)

            # ensure mask non-empty
            if mask.sum() > 0:
                # seamless clone swapped_frame onto morphed_frame using mask
                try:
                    seamless = cv2.seamlessClone(swapped_frame, morphed_frame, mask, center, cv2.NORMAL_CLONE)
                    return seamless
                except Exception:
                    # fallback simple alpha blend
                    mask_f = cv2.GaussianBlur(mask, (31,31), 0).astype(np.float32)/255.0
                    mask_3 = cv2.merge([mask_f, mask_f, mask_f])
                    blended = (swapped_frame.astype(np.float32)*mask_3 + morphed_frame.astype(np.float32)*(1-mask_3)).astype(np.uint8)
                    return blended
    except Exception:
        pass

    return swapped_frame

# -----------------------------------------------------------------------------
# NOTE:
# - Jika kamu ingin tracking/face_reference tetap valid setelah morph (mis. untuk tracking di frame selanjutnya),
#   kamu bisa mengupdate target_face.kps menjadi pts_inter hasil morph. Itu butuh perubahan kecil di process_frames/process_video.
# - Parameter roop.globals.shape_mix dapat ditambahkan di konfigurasi UI agar mudah tuning.
# -----------------------------------------------------------------------------
# END OF MORPH BLEND MODULE
