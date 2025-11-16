# config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FaceSwapConfig:
    # Blending settings
    blend_ratio: float = 0.7
    blend_levels: int = 4
    edge_feather: int = 15
    mask_smoothness: int = 7
    
    # Color correction settings
    color_correction_strength: float = 0.8
    color_balance_strength: float = 0.6
    histogram_matching: bool = True
    adaptive_lighting: bool = True
    
    # Quality enhancement settings
    sharpness_enhance: float = 1.1
    denoise_strength: int = 5
    texture_preservation: bool = True
    quality_enhance: bool = True
    
    # Performance settings
    enable_motion_compensation: bool = True
    enable_real_time_processing: bool = False
    max_face_size: int = 1024
    
    # Advanced settings
    pyramid_levels: int = 4
    gaussian_kernel: int = 5
    bilateral_filter_d: int = 5
    bilateral_filter_sigma: int = 75
    
    # Debug settings
    debug_mode: bool = False
    save_intermediate: bool = False

# Preset configurations
PRESETS = {
    'quality': FaceSwapConfig(
        blend_ratio=0.8,
        blend_levels=5,
        color_correction_strength=0.9,
        sharpness_enhance=1.3,
        quality_enhance=True,
        texture_preservation=True,
        pyramid_levels=5
    ),
    'fast': FaceSwapConfig(
        blend_ratio=0.6,
        blend_levels=2,
        color_correction_strength=0.7,
        sharpness_enhance=1.0,
        enable_real_time_processing=True,
        pyramid_levels=2
    ),
    'extreme_motion': FaceSwapConfig(
        blend_ratio=0.75,
        enable_motion_compensation=True,
        color_correction_strength=0.85,
        blend_levels=4,
        texture_preservation=True,
        pyramid_levels=4
    ),
    'low_light': FaceSwapConfig(
        color_correction_strength=0.9,
        adaptive_lighting=True,
        denoise_strength=8,
        sharpness_enhance=1.1,
        blend_levels=4
    ),
    'realistic': FaceSwapConfig(
        blend_ratio=0.65,
        color_correction_strength=0.75,
        sharpness_enhance=1.1,
        blend_levels=4,
        texture_preservation=True,
        adaptive_lighting=True,
        pyramid_levels=4
    )
}

# Global configuration instance
current_config = FaceSwapConfig()

def set_config(preset: str = None, **kwargs):
    """
    Set configuration using preset or custom values
    """
    global current_config
    
    if preset and preset in PRESETS:
        current_config = PRESETS[preset]
        print(f"✅ Preset '{preset}' loaded successfully!")
    
    # Update with custom values
    for key, value in kwargs.items():
        if hasattr(current_config, key):
            setattr(current_config, key, value)
            if preset:  # Only print if using preset with customizations
                print(f"✅ Custom setting: {key} = {value}")
    
    return current_config

def get_config() -> FaceSwapConfig:
    """Get current configuration"""
    return current_config

def show_config():
    """Display current configuration"""
    config = get_config()
    print("\n🎯 Current Face Swap Configuration:")
    print("=" * 50)
    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        print(f"  {field:25}: {value}")
    print("=" * 50)

def list_presets():
    """List all available presets"""
    print("\n📁 Available Presets:")
    print("=" * 30)
    for preset_name in PRESETS.keys():
        print(f"  • {preset_name}")
    print("=" * 30)
