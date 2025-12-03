import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

import roop.globals

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# Monkey patch SSL for MacOS
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


# ===================================================================
#  FFmpeg Wrapper (lebih aman + log error)
# ===================================================================
def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)

    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print("[FFMPEG ERROR] →", e.output.decode(errors="ignore"))
        return False
    except Exception as e:
        print("[FFMPEG ERROR] →", str(e))
        return False


# ===================================================================
#  FPS Detector (lebih aman)
# ===================================================================
def detect_fps(target_path: str) -> float:
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=nokey=1:noprint_wrappers=1', target_path
    ]

    try:
        output = subprocess.check_output(command).decode().strip()

        # Case "30"
        if output.isdigit():
            return float(output)

        # Case "30000/1001"
        num, den = output.split('/')
        return int(num) / int(den)

    except Exception:
        return 30.0


# ===================================================================
#  Helpers Path
# ===================================================================
def get_temp_directory_path(target_path: str) -> str:
    target_path = Path(target_path)
    return str(target_path.parent / TEMP_DIRECTORY / target_path.stem)


def get_temp_output_path(target_path: str) -> str:
    return str(Path(get_temp_directory_path(target_path)) / TEMP_VIDEO_FILE)


# ===================================================================
#  Extract Video Frames
# ===================================================================
def extract_frames(target_path: str, fps: float = 30) -> bool:
    temp_dir = get_temp_directory_path(target_path)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    temp_quality = roop.globals.temp_frame_quality * 31 // 100
    fmt = roop.globals.temp_frame_format

    return run_ffmpeg([
        '-hwaccel', 'auto',
        '-i', target_path,
        '-q:v', str(temp_quality),
        '-pix_fmt', 'rgb24',
        '-vf', f'fps={fps}',
        os.path.join(temp_dir, f'%04d.{fmt}')
    ])


# ===================================================================
#  Create Video
# ===================================================================
def create_video(target_path: str, fps: float = 30) -> bool:
    temp_dir = get_temp_directory_path(target_path)
    temp_out = get_temp_output_path(target_path)

    fmt = roop.globals.temp_frame_format
    encoder = roop.globals.output_video_encoder
    q = (roop.globals.output_video_quality + 1) * 51 // 100

    args = [
        '-hwaccel', 'auto',
        '-r', str(fps),
        '-i', os.path.join(temp_dir, f'%04d.{fmt}'),
        '-c:v', encoder,
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
    ]

    if encoder in ['libx264', 'libx265', 'libvpx']:
        args.extend(['-crf', str(q)])
    elif encoder in ['h264_nvenc', 'hevc_nvenc']:
        args.extend(['-cq', str(q)])

    args.extend(['-y', temp_out])
    return run_ffmpeg(args)


# ===================================================================
#  Restore Audio
# ===================================================================
def restore_audio(target_path: str, output_path: str) -> None:
    temp_out = get_temp_output_path(target_path)

    done = run_ffmpeg([
        '-i', temp_out,
        '-i', target_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', output_path
    ])

    if not done:
        move_temp(target_path, output_path)


# ===================================================================
#  Frame Utils
# ===================================================================
def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_dir = get_temp_directory_path(target_path)
    fmt = roop.globals.temp_frame_format
    return glob.glob(os.path.join(glob.escape(temp_dir), f'*.{fmt}'))


# ===================================================================
#  Output Path Normalization
# ===================================================================
def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        source_stem = Path(source_path).stem
        target_stem = Path(target_path).stem
        ext = Path(target_path).suffix

        if Path(output_path).is_dir():
            return str(Path(output_path) / f"{source_stem}-{target_stem}{ext}")

    return output_path


# ===================================================================
#  Temp ops
# ===================================================================
def create_temp(target_path: str) -> None:
    Path(get_temp_directory_path(target_path)).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_out = Path(get_temp_output_path(target_path))
    output_path = Path(output_path)

    if temp_out.is_file():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(temp_out), str(output_path))


def clean_temp(target_path: str) -> None:
    temp_dir = Path(get_temp_directory_path(target_path))
    parent_dir = temp_dir.parent

    try:
        if not roop.globals.keep_frames and temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception:
        pass

    try:
        if parent_dir.exists() and not any(parent_dir.iterdir()):
            parent_dir.rmdir()
    except Exception:
        pass


# ===================================================================
#  File Type Utils
# ===================================================================
def has_image_extension(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in ('.png', '.jpg', '.jpeg', '.webp')


def is_image(path: str) -> bool:
    if Path(path).is_file():
        mime, _ = mimetypes.guess_type(path)
        return mime and mime.startswith("image/")
    return False


def is_video(path: str) -> bool:
    if Path(path).is_file():
        mime, _ = mimetypes.guess_type(path)
        return mime and mime.startswith("video/")
    return False


# ===================================================================
#  Download helper (stabil, cepat)
# ===================================================================
def conditional_download(directory: str, urls: List[str]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for url in urls:
        out_path = directory / os.path.basename(url)

        if not out_path.exists():
            try:
                req = urllib.request.urlopen(url)
                total = int(req.headers.get('Content-Length', 0))

                with tqdm(total=total, unit='B', unit_scale=True, desc=f"Downloading {out_path.name}") as pbar:
                    urllib.request.urlretrieve(
                        url, out_path,
                        reporthook=lambda c, b, t: pbar.update(b)
                    )
            except Exception as e:
                print("[DOWNLOAD ERROR] →", str(e))


# ===================================================================
#  Resolve relative paths
# ===================================================================
def resolve_relative_path(path: str) -> str:
    return str(Path(__file__).parent.joinpath(path).resolve())
