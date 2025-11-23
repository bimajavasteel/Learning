# (FILE 2) – occlusion_handler.py
# -------------------------------

import cv2
import numpy as np

def apply_occlusion_mask(face_crop, occ_mask):
    # invert: 1=visible  /  0=occluded
    visible = 1 - occ_mask.astype(np.float32)
    visible = cv2.GaussianBlur(visible, (31,31), 0)
    visible = visible[...,None]

    return (face_crop * visible).astype("uint8"), visible
