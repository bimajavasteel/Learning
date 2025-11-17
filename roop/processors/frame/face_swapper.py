from typing import Any, List, Callable, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np
from scipy import ndimage
import os
from numba import jit
import time

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'

# Cache untuk performa
FACE_CACHE = {}
CACHE_SIZE = 50

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/datasets/OwlMaster/gg2/resolve/main/inswapper_128.onnx'])
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()
    FACE_CACHE.clear()

@jit(nopython=True, fastmath=True)
def fast_color_transfer(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Color transfer yang dioptimalkan dengan Numba"""
    target = target.astype(np.float32)
    source = source.astype(np.float32)
    
    # Calculate mean and std
    target_mean = np.mean(target, axis=(0, 1))
    target_std = np.std(target, axis=(0, 1))
    source_mean = np.mean(source, axis=(0, 1))
    source_std = np.std(source, axis=(0, 1))
    
    # Avoid division by zero
    source_std = np.where(source_std == 0, 1, source_std)
    
    # Color transfer
    result = (target - source_mean) * (target_std / source_std) + target_mean
    return np.clip(result, 0, 255).astype(np.uint8)

def adaptive_face_alignment(source_face: Face, target_face: Face, temp_frame: Frame) -> Tuple[Frame, np.ndarray]:
    """Alignment adaptif berdasarkan pose dan ekspresi wajah"""
    try:
        # Extract landmarks
        source_landmarks = source_face.landmark_2d_106
        target_landmarks = target_face.landmark_2d_106
        
        # Calculate face orientation
        source_pose = calculate_face_pose(source_landmarks)
        target_pose = calculate_face_pose(target_landmarks)
        
        # Adaptive scaling based on face distance and angle
        scale_factor = calculate_adaptive_scale(source_pose, target_pose)
        
        # Enhanced face matching dengan affine transformation
        if len(source_landmarks) == len(target_landmarks) and len(source_landmarks) > 0:
            # Use key facial points for better alignment
            key_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Important facial landmarks
            
            src_points = np.array([source_landmarks[i] for i in key_points], dtype=np.float32)
            dst_points = np.array([target_landmarks[i] for i in key_points], dtype=np.float32)
            
            # Calculate transformation matrix dengan RANSAC untuk outlier rejection
            transform_matrix, inliers = cv2.estimateAffinePartial2D(
                src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            
            if transform_matrix is not None:
                # Apply scaling to transformation
                transform_matrix[:2, :2] *= scale_factor
                
                h, w = temp_frame.shape[:2]
                aligned_frame = cv2.warpAffine(temp_frame, transform_matrix, (w, h), flags=cv2.INTER_LINEAR)
                
                return aligned_frame, transform_matrix
        
        return temp_frame, np.eye(2, 3, dtype=np.float32)
        
    except Exception as e:
        print(f"Face alignment error: {e}")
        return temp_frame, np.eye(2, 3, dtype=np.float32)

def calculate_face_pose(landmarks: np.ndarray) -> dict:
    """Calculate face pose from landmarks"""
    try:
        # Simple pose estimation from facial landmarks
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        mouth_center = landmarks[13]
        
        # Calculate angles and distances
        eye_center = (left_eye + right_eye) / 2
        vertical_ratio = np.linalg.norm(nose_tip - eye_center) / np.linalg.norm(mouth_center - nose_tip)
        horizontal_ratio = np.linalg.norm(left_eye - right_eye) / np.linalg.norm(eye_center - nose_tip)
        
        return {
            'vertical_ratio': vertical_ratio,
            'horizontal_ratio': horizontal_ratio,
            'eye_distance': np.linalg.norm(left_eye - right_eye)
        }
    except:
        return {'vertical_ratio': 1.0, 'horizontal_ratio': 1.0, 'eye_distance': 1.0}

def calculate_adaptive_scale(source_pose: dict, target_pose: dict) -> float:
    """Calculate adaptive scaling factor based on face poses"""
    try:
        # Scale based on eye distance ratio
        scale_eye = target_pose['eye_distance'] / source_pose['eye_distance'] if source_pose['eye_distance'] > 0 else 1.0
        
        # Scale based on face proportions
        scale_vertical = target_pose['vertical_ratio'] / source_pose['vertical_ratio'] if source_pose['vertical_ratio'] > 0 else 1.0
        scale_horizontal = target_pose['horizontal_ratio'] / source_pose['horizontal_ratio'] if source_pose['horizontal_ratio'] > 0 else 1.0
        
        # Combined scale with weights
        combined_scale = (scale_eye * 0.6 + scale_vertical * 0.2 + scale_horizontal * 0.2)
        
        # Limit scale changes to avoid extreme distortions
        return np.clip(combined_scale, 0.7, 1.3)
    except:
        return 1.0

def advanced_color_correction(swapped_face: Frame, target_frame: Frame, target_face: Face) -> Frame:
    """Advanced color correction dengan multiple techniques"""
    try:
        if target_face is None:
            return swapped_face
        
        # Extract target face region
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        target_region = target_frame[y1:y2, x1:x2]
        
        if target_region.size == 0 or swapped_face.size == 0:
            return swapped_face
        
        # Resize jika diperlukan
        if swapped_face.shape != target_region.shape:
            swapped_face = cv2.resize(swapped_face, (target_region.shape[1], target_region.shape[0]))
        
        # Method 1: Fast LAB color transfer
        try:
            swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB)
            corrected_lab = fast_color_transfer(swapped_lab, target_lab)
            result1 = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        except:
            result1 = swapped_face
        
        # Method 2: Histogram matching per channel
        try:
            result2 = swapped_face.copy()
            for i in range(3):
                result2[:,:,i] = histogram_matching(result2[:,:,i], target_region[:,:,i])
        except:
            result2 = swapped_face
        
        # Method 3: Reinhard color transfer
        try:
            result3 = reinhard_color_transfer(swapped_face, target_region)
        except:
            result3 = swapped_face
        
        # Blend all methods
        blend_ratio = 0.6
        temp_result = cv2.addWeighted(result1, 0.4, result2, 0.3, 0)
        final_result = cv2.addWeighted(temp_result, blend_ratio, result3, 0.3, 0)
        
        # Preserve details from original swapped face
        final_result = cv2.addWeighted(final_result, 0.8, swapped_face, 0.2, 0)
        
        return final_result
        
    except Exception as e:
        print(f"Advanced color correction error: {e}")
        return swapped_face

def histogram_matching(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Histogram matching untuk color correction"""
    try:
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
        
        # Get unique values and their counts
        s_values, s_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        
        # Calculate cumulative distributions
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        
        # Interpolate to find new pixel values
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        
        return interp_t_values[s_idx].reshape(oldshape)
    except:
        return source

def reinhard_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reinhard color transfer algorithm"""
    try:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # Calculate mean and std
        s_mean, s_std = np.mean(source_lab, axis=(0,1)), np.std(source_lab, axis=(0,1))
        t_mean, t_std = np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))
        
        # Color transfer
        result_lab = (source_lab - s_mean) * (t_std / s_std) + t_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    except:
        return source

def create_advanced_mask(face: Face, frame_shape: Tuple[int, int], landmarks: np.ndarray) -> np.ndarray:
    """Create advanced mask dengan facial landmark awareness"""
    mask = np.zeros(frame_shape[:2], dtype=np.float32)
    
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Create base elliptical mask
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Elliptical mask
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 1.0, -1)
        
        # Enhanced mask menggunakan facial landmarks
        if landmarks is not None and len(landmarks) >= 68:
            # Create convex hull dari facial landmarks
            hull_points = []
            for i in range(17):  # Jawline
                hull_points.append((int(landmarks[i][0]), int(landmarks[i][1])))
            for i in range(17, 27):  # Eyebrows and nose
                hull_points.append((int(landmarks[i][0]), int(landmarks[i][1])))
            
            if hull_points:
                hull = np.array(hull_points, dtype=np.int32)
                cv2.fillConvexPoly(mask, hull, 1.0)
        
        # Multi-level Gaussian blur untuk smooth transition
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        
        # Enhance edges
        mask = np.clip(mask * 1.2, 0, 1)
        
        return mask
        
    except Exception as e:
        print(f"Advanced mask creation error: {e}")
        # Fallback
        x1, y1, x2, y2 = map(int, face.bbox)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return mask

def motion_aware_blending(swapped_face: Frame, target_frame: Frame, target_face: Face, prev_face: Frame = None) -> Frame:
    """Motion-aware blending untuk wajah bergerak cepat"""
    try:
        if target_face is None:
            return target_frame
            
        x1, y1, x2, y2 = map(int, target_face.bbox)
        h, w = target_frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Ensure swapped face has correct size
        face_height, face_width = y2 - y1, x2 - x1
        if swapped_face.shape[0] != face_height or swapped_face.shape[1] != face_width:
            swapped_face = cv2.resize(swapped_face, (face_width, face_height))
        
        # Create advanced mask dengan landmarks
        mask = create_advanced_mask(target_face, target_frame.shape, target_face.landmark_2d_106)
        mask_region = mask[y1:y2, x1:x2]
        
        # Ensure mask has correct dimensions
        if mask_region.shape != swapped_face.shape[:2]:
            mask_region = cv2.resize(mask_region, (swapped_face.shape[1], swapped_face.shape[0]))
        
        # Motion compensation jika ada previous frame
        if prev_face is not None:
            try:
                # Simple motion estimation
                prev_gray = cv2.cvtColor(prev_face[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(target_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Apply motion blur to mask untuk smooth transition
                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_factor = np.clip(flow_magnitude / 10.0, 0, 1)
                mask_region = cv2.GaussianBlur(mask_region, (15 + int(np.mean(motion_factor) * 10), 15 + int(np.mean(motion_factor) * 10)), 0)
            except:
                pass
        
        # Create 3-channel mask
        mask_3d = np.stack([mask_region] * 3, axis=-1)
        
        # Multi-resolution blending
        pyramid_levels = 3
        blended_face = multi_resolution_blend(swapped_face, target_frame[y1:y2, x1:x2], mask_region, pyramid_levels)
        
        # Final composition
        result = target_frame.copy()
        result[y1:y2, x1:x2] = blended_face
        
        return result
        
    except Exception as e:
        print(f"Motion aware blending error: {e}")
        return simple_blending(swapped_face, target_frame, target_face)

def multi_resolution_blend(src: Frame, dst: Frame, mask: np.ndarray, levels: int) -> Frame:
    """Multi-resolution blending untuk transisi yang lebih halus"""
    try:
        # Gaussian pyramid untuk source, destination, dan mask
        G_src = [src.astype(np.float32)]
        G_dst = [dst.astype(np.float32)]
        G_mask = [mask.astype(np.float32)]
        
        # Build pyramids
        for i in range(levels):
            G_src.append(cv2.pyrDown(G_src[-1]))
            G_dst.append(cv2.pyrDown(G_dst[-1]))
            G_mask.append(cv2.pyrDown(G_mask[-1]))
        
        # Laplacian pyramids
        L_src = [G_src[levels - 1]]
        L_dst = [G_dst[levels - 1]]
        
        for i in range(levels - 1, 0, -1):
            size = (G_src[i-1].shape[1], G_src[i-1].shape[0])
            L_src_l = G_src[i-1] - cv2.pyrUp(G_src[i], dstsize=size)
            L_dst_l = G_dst[i-1] - cv2.pyrUp(G_dst[i], dstsize=size)
            L_src.append(L_src_l)
            L_dst.append(L_dst_l)
        
        L_src.reverse()
        L_dst.reverse()
        
        # Blend pyramids
        blended_pyramid = []
        for i in range(levels):
            mask_expanded = np.stack([G_mask[i]] * 3, axis=-1) if len(G_mask[i].shape) == 2 else G_mask[i]
            blended = L_src[i] * mask_expanded + L_dst[i] * (1.0 - mask_expanded)
            blended_pyramid.append(blended)
        
        # Reconstruct
        result = blended_pyramid[0]
        for i in range(1, levels):
            size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + blended_pyramid[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        print(f"Multi-resolution blend error: {e}")
        # Fallback to simple blending
        mask_3d = np.stack([mask] * 3, axis=-1)
        return (src * mask_3d + dst * (1.0 - mask_3d)).astype(np.uint8)

def enhance_face_quality(face: Frame) -> Frame:
    """Enhanced face quality improvement"""
    try:
        if face is None:
            return face
            
        # Mild sharpening dengan adaptive kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.15
        
        sharpened = cv2.filter2D(face, -1, kernel)
        
        # Adaptive bilateral filter
        denoised = cv2.bilateralFilter(sharpened, 5, 15, 15)
        
        # Contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
        
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return face

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame, prev_frame: Frame = None) -> Frame:
    """Enhanced face swapping dengan semua optimisasi"""
    try:
        start_time = time.time()
        
        # Apply face alignment
        aligned_frame, transform_matrix = adaptive_face_alignment(source_face, target_face, temp_frame)
        
        # Get basic face swap
        swapped_result = get_face_swapper().get(aligned_frame, target_face, source_face, paste_back=False)
        
        # Ensure proper format
        swapped_frame = ensure_frame_format(swapped_result)
        if swapped_frame is None:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply advanced color correction
        swapped_frame = advanced_color_correction(swapped_frame, temp_frame, target_face)
        
        # Enhance face quality
        swapped_frame = enhance_face_quality(swapped_frame)
        
        # Apply motion-aware blending
        result_frame = motion_aware_blending(swapped_frame, temp_frame, target_face, prev_frame)
        
        processing_time = time.time() - start_time
        if processing_time > 0.1:  # Log slow processing
            print(f"Face swap processing time: {processing_time:.3f}s")
        
        return result_frame
        
    except Exception as e:
        print(f"Enhanced face swap error: {e}")
        # Fallback to original face swapper
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def ensure_frame_format(frame: Any) -> Optional[Frame]:
    """Ensure the frame is in correct numpy array format"""
    if frame is None:
        return None
    
    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
        return frame
    
    if isinstance(frame, tuple):
        try:
            frame_array = np.array(frame)
            if frame_array.size > 0:
                return frame_array
        except:
            pass
    
    return None

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame, prev_frame: Frame = None) -> Frame:
    """Process single frame dengan semua optimisasi"""
    try:
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame, prev_frame)
        else:
            target_face = find_similar_face(temp_frame, reference_face)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame, prev_frame)
        return temp_frame
    except Exception as e:
        print(f"Process frame error: {e}")
        return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Process multiple frames dengan frame-to-frame consistency"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        reference_face = None if roop.globals.many_faces else get_face_reference()
        
        prev_frame = None
        
        for i, temp_frame_path in enumerate(temp_frame_paths):
            try:
                temp_frame = cv2.imread(temp_frame_path)
                if temp_frame is not None:
                    result = process_frame(source_face, reference_face, temp_frame, prev_frame)
                    cv2.imwrite(temp_frame_path, result)
                    prev_frame = temp_frame  # Update previous frame untuk motion compensation
                if update:
                    update()
            except Exception as e:
                print(f"Error processing frame {temp_frame_path}: {e}")
                continue
    except Exception as e:
        print(f"Process frames error: {e}")

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process single image dengan optimisasi"""
    try:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
        result = process_frame(source_face, reference_face, target_frame)
        cv2.imwrite(output_path, result)
    except Exception as e:
        print(f"Process image error: {e}")

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video dengan motion-aware processing"""
    try:
        if not roop.globals.many_faces and not get_face_reference():
            reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
            reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
            set_face_reference(reference_face)
        roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
    except Exception as e:
        print(f"Process video error: {e}")
