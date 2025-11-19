import threading
from typing import Any, Optional, List
import insightface
import numpy as np
import torch
import cv2

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
ADAFACE_MODEL = None
THREAD_LOCK = threading.Lock()

# NEW: AdaFace model loader
def get_adaface_model() -> Any:
    global ADAFACE_MODEL
    
    with THREAD_LOCK:
        if ADAFACE_MODEL is None:
            try:
                # Untuk Kaggle, gunakan model yang compatible
                from adaface import adaface_net
                import torch.backends.cudnn as cudnn
                
                # Load model AdaFace
                model = adaface_net.build_model('adaface_ir101_webface12m')
                checkpoint = torch.load('../models/adaface_ir101_webface12m.pt', 
                                      map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                model.load_state_dict(checkpoint)
                model.eval()
                
                # Optimasi untuk Kaggle GPU
                if torch.cuda.is_available():
                    cudnn.benchmark = True
                    model = model.cuda()
                
                ADAFACE_MODEL = model
                print("[AdaFace] Model loaded successfully")
            except Exception as e:
                print(f"[AdaFace] Error loading model: {e}")
                # Fallback ke InsightFace
                ADAFACE_MODEL = "FALLBACK"
    return ADAFACE_MODEL

# NEW: Extract AdaFace embedding
def extract_adaface_embedding(face: Face, frame: Frame) -> Optional[np.ndarray]:
    try:
        if not hasattr(face, 'bbox'):
            return None
            
        # Extract face region dari frame
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        
        # Boundary check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        
        # Preprocessing untuk AdaFace
        face_crop = cv2.resize(face_crop, (112, 112))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = np.transpose(face_crop, (2, 0, 1))  # HWC to CHW
        face_crop = (face_crop / 255.0 - 0.5) / 0.5  # Normalization
        face_tensor = torch.FloatTensor(face_crop).unsqueeze(0)
        
        # Move to GPU jika available
        if torch.cuda.is_available():
            face_tensor = face_tensor.cuda()
        
        # Extract embedding
        with torch.no_grad():
            embedding = get_adaface_model()(face_tensor)
            embedding = embedding.cpu().numpy().flatten()
            
        return embedding
        
    except Exception as e:
        print(f"[AdaFace] Embedding extraction failed: {e}")
        return None

# MODIFIED: get_face_analyser dengan AdaFace support
def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=roop.globals.execution_providers,
                allowed_modules=['detection', 'recognition']
            )
            FACE_ANALYSER.prepare(ctx_id=0)
            
            # Pre-load AdaFace model
            if getattr(roop.globals, 'use_adaface', False):
                get_adaface_model()
                
    return FACE_ANALYSER

def clear_face_analyser() -> None:
    global FACE_ANALYSER, ADAFACE_MODEL
    FACE_ANALYSER = None
    ADAFACE_MODEL = None

def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        try:
            face = many_faces[position]
            # NEW: Attach AdaFace embedding jika enabled
            if getattr(roop.globals, 'use_adaface', False):
                setattr(face, 'adaface_embedding', extract_adaface_embedding(face, frame))
            return face
        except IndexError:
            face = many_faces[-1]
            if getattr(roop.globals, 'use_adaface', False):
                setattr(face, 'adaface_embedding', extract_adaface_embedding(face, frame))
            return face
    return None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    if frame is None or frame.size == 0:
        return None
    try:
        faces = get_face_analyser().get(frame)
        # NEW: Extract AdaFace embeddings untuk semua faces
        if faces and getattr(roop.globals, 'use_adaface', False):
            for face in faces:
                setattr(face, 'adaface_embedding', extract_adaface_embedding(face, frame))
        return faces
    except (ValueError, RuntimeError) as e:
        print(f"[FaceAnalyser] Skipped invalid frame: {e}")
        return None

# MODIFIED: find_similar_face dengan AdaFace comparison
def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        use_adaface = getattr(roop.globals, 'use_adaface', False)
        
        if use_adaface and hasattr(reference_face, 'adaface_embedding'):
            # NEW: AdaFace comparison
            ref_embedding = reference_face.adaface_embedding
            if ref_embedding is not None:
                best_face = None
                best_distance = float('inf')
                
                for face in many_faces:
                    if hasattr(face, 'adaface_embedding') and face.adaface_embedding is not None:
                        # Cosine similarity distance
                        distance = 1 - np.dot(ref_embedding, face.adaface_embedding) / (
                            np.linalg.norm(ref_embedding) * np.linalg.norm(face.adaface_embedding)
                        )
                        
                        adaface_threshold = getattr(roop.globals, 'adaface_threshold', 0.5)
                        if distance < adaface_threshold and distance < best_distance:
                            best_distance = distance
                            best_face = face
                
                return best_face
        
        # FALLBACK: Original InsightFace comparison
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = np.sum((face.normed_embedding - reference_face.normed_embedding) ** 2)
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
