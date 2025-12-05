import cv2
import numpy as np
from roop.face_analyser import detect_occlusion
from roop.occlusion_utils import composite_with_mask


class FaceSwapper:
    """
    Wrapper untuk model inswapper / GFPGAN / dll.
    Pastikan fungsi `swap_full_frame()` mengembalikan full-frame hasil swap.
    """

    def __init__(self, model):
        self.model = model

    def swap_full_frame(self, frame, source_face, target_face):
        return self.model.get(
            frame.copy(),
            target_face,
            source_face,
            paste_back=True
        )


def process_frame(frame, faces, source_faces, swapper):
    """
    frame: BGR
    faces: list face detection
    source_faces: mapping
    swapper: FaceSwapper
    """

    out = frame.copy()

    for face in faces:
        track_id = getattr(face, "track_id", None) or getattr(face, "id", None)

        src = source_faces.get(track_id, None)
        if src is None:
            continue

        swapped = swapper.swap_full_frame(out, src, face)

        is_occ, mask = detect_occlusion(face, out, track_id)

        if mask is None:
            # Tidak ada mask → jika occluded, jangan paste; jika tidak occluded paste semuanya
            if is_occ:
                continue
            out = swapped
        else:
            out = composite_with_mask(out, swapped, mask)

    return out
