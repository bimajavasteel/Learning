# --- AUTO DOWNLOAD MODEL ---
import os
from roop.utilities import conditional_download, resolve_relative_path

def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128.onnx',
        'https://huggingface.co/somanchiu/reswapper/resolve/main/reswapper_256-1567500.pth'
    ])
    return True


# --- PRIORITY: RESWAPPER FIRST ---
from roop.processors.frame.reswapper import ReSwapperWrapper

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            res_path = resolve_relative_path('../models/reswapper_256-1567500.pth')
            if os.path.exists(res_path):
                FACE_SWAPPER = ReSwapperWrapper(res_path)
                return FACE_SWAPPER

            # fallback inswapper
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )

    return FACE_SWAPPER
