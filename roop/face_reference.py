from typing import Optional, Any
import numpy as np

from roop.typing import Face

FACE_REFERENCE = None
ADAFACE_REFERENCE = None  # NEW: Store AdaFace embedding separately

def get_face_reference() -> Optional[Face]:
    return FACE_REFERENCE

def set_face_reference(face: Face) -> None:
    global FACE_REFERENCE, ADAFACE_REFERENCE

    FACE_REFERENCE = face
    
    # NEW: Extract and store AdaFace embedding if enabled
    if face is not None and hasattr(face, 'adaface_embedding'):
        ADAFACE_REFERENCE = face.adaface_embedding
        print(f"[AdaFace] Reference embedding stored - shape: {ADAFACE_REFERENCE.shape}")
    else:
        ADAFACE_REFERENCE = None

def get_adaface_reference() -> Optional[np.ndarray]:
    """NEW: Get stored AdaFace embedding for comparison"""
    return ADAFACE_REFERENCE

def clear_face_reference() -> None:
    global FACE_REFERENCE, ADAFACE_REFERENCE

    FACE_REFERENCE = None
    ADAFACE_REFERENCE = None  # NEW: Clear both references

# NEW: Enhanced face reference for better AdaFace support
def create_enhanced_face_reference(face: Face, frame: Any = None) -> dict:
    """
    Create comprehensive face reference for hybrid matching
    """
    reference = {
        'insightface_obj': face,
        'bbox': getattr(face, 'bbox', None),
        'landmarks': getattr(face, 'landmark_2d_106', None),
        'insightface_embedding': getattr(face, 'normed_embedding', None),
        'adaface_embedding': getattr(face, 'adaface_embedding', None),
        'timestamp': getattr(face, 'timestamp', None)
    }
    
    return reference

# NEW: Quick validation for reference quality
def validate_reference_quality(face: Face) -> bool:
    """
    Validate if the reference face has sufficient quality for matching
    """
    if face is None:
        return False
        
    # Check basic attributes
    if not hasattr(face, 'bbox'):
        return False
        
    # Check embedding quality
    use_adaface = False
    try:
        import roop.globals
        use_adaface = getattr(roop.globals, 'use_adaface', False)
    except:
        pass
        
    if use_adaface:
        if hasattr(face, 'adaface_embedding') and face.adaface_embedding is not None:
            embedding_norm = np.linalg.norm(face.adaface_embedding)
            return 0.5 < embedding_norm < 2.0  # Reasonable embedding norm range
    else:
        if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
            return True
            
    return False
