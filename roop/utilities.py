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


# ===================================================================
#  MAC SSL FIX
# ===================================================================
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


# ===================================================================
#  FFmpeg Wrapper
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
#  FPS DETECTOR (dipakai core.py)
# ===================================================================
def detect_fps(target_path: str) -> float:
    command = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        target_path
    ]

    try:
        output = subprocess.check_output(command).decode().strip()

        if output.isdigit():
            return float(output)

        if "/" in output:
            num, den = output.split('/')
            return float(num) / float(den)

        return float(output)

    except Exception:
        return 30.0


# ===================================================================
#  PROBE CODEC + GPU DECODER AVAILABILITY
# ===================================================================
def probe_codec(target_path: str) -> Optional[str]:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        target_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return out if out else None
    except Exception:
        return None


def is_decoder_available(decoder: str) -> bool:
    try:
        dec = subprocess.check_output(['ffmpeg', '-hide_banner', '-decoders'],
                                      stderr=subprocess.STDOUT).decode().lower()
        return decoder.lower() in dec
    except Exception:
        return False


def choose_gpu_decoder_for_codec(codec: str) -> Optional[str]:
    if not codec:
        return None
    codec = codec.lower()

    if codec in ('h264', 'avc1', 'mpeg4'):
        return 'h264_cuvid'
    if codec in ('hevc', 'h265'):
        return 'hevc_cuvid'

    return None


# ===================================================================
#  PATH HELPERS
# ===================================================================
def get_temp_directory_path(target_path: str) -> str:
    target_path = Path(target_path)
    return str(target_path.parent / TEMP_DIRECTORY / target_path.stem)


def get_temp_output_path(target_path: str) -> str:
    return str(Path(get_temp_directory_path(target_path)) / TEMP_VIDEO_FILE)


# ===================================================================
#  MAIN: GPU FRAME EXTRACTOR + CPU FALLBACK
# ===================================================================
def extract_frames(target_path: str, fps: float = 30.0) -> bool:
    return extract_frames_gpu(target_path, fps=fps, force_gpu=False)


def extract_frames_gpu(target_path: str, fps: float = 30.0, force_gpu: bool = False) -> bool:
    temp_dir = get_temp_directory_path(target_path)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    fmt = roop.globals.temp_frame_format
    q = roop.globals.temp_frame_quality * 31 // 100

    codec = probe_codec(target_path)
    gpu_decoder = choose_gpu_decoder_for_codec(codec)

    # ===========================================================
    #  ATTEMPT GPU DECODE (CUVID)
    # ===========================================================
    if gpu_decoder and is_decoder_available(gpu_decoder):
        print(f"[extract_frames_gpu] GPU decode → {gpu_decoder} (codec={codec})")

        args = [
            '-hwaccel', 'cuda',
            '-hwaccel_output_format', 'cuda',
            '-c:v', gpu_decoder,
            '-i', target_path,
            '-q:v', str(q),

            # FIX UTAMA → CUDA scale → fps
            '-vf', f'scale_cuda=format=rgb0,fps={fps}',

            '-pix_fmt', 'rgb24',
            os.path.join(temp_dir, f"%04d.{fmt}")
        ]

        ok = run_ffmpeg(args)
        if ok:
            print("[extract_frames_gpu] sukses GPU decode.")
            return True

        print("[extract_frames_gpu] GPU decode gagal.")
        if force_gpu:
            raise RuntimeError("FFmpeg gagal memakai GPU decoder.")
        print("[extract_frames_gpu] fallback ke CPU decode...")

    else:
        if force_gpu:
            raise RuntimeError(f"GPU decoder tak tersedia untuk codec={codec}")
        print(f"[extract_frames_gpu] GPU decoder tidak tersedia, fallback CPU. (codec={codec})")

    # ===========================================================
    #  CPU FALLBACK
    # ===========================================================
    cpu_args = [
        '-hwaccel', 'auto',
        '-i', target_path,
        '-q:v', str(q),
        '-pix_fmt', 'rgb24',
        '-vf', f'fps={fps}',
        os.path.join(temp_dir, f"%04d.{fmt}")
    ]

    ok = run_ffmpeg(cpu_args)
    if ok:
        print("[extract_frames_gpu] CPU decode sukses.")
    else:
        print("[extract_frames_gpu] CPU decode gagal.")

    return ok


# ===================================================================
#  CREATE VIDEO
# ===================================================================
def create_video(target_path: str, fps: float = 30.0) -> bool:
    temp_dir = get_temp_directory_path(target_path)
    temp_out = get_temp_output_path(target_path)

    fmt = roop.globals.temp_frame_format
    enc = roop.globals.output_video_encoder
    q = (roop.globals.output_video_quality + 1) * 51 // 100

    args = [
        '-hwaccel', 'auto',
        '-r', str(fps),
        '-i', os.path.join(temp_dir, f"%04d.{fmt}"),
        '-c:v', enc,
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1'
    ]

    if enc in ['libx264', 'libx265', 'libvpx']:
        args.extend(['-crf', str(q)])
    elif enc in ['h264_nvenc', 'hevc_nvenc']:
        args.extend(['-cq', str(q)])

    args.extend(['-y', temp_out])
    return run_ffmpeg(args)


# ===================================================================
#  AUDIO RESTORE
# ===================================================================
def restore_audio(target_path: str, output_path: str) -> None:
    temp_out = get_temp_output_path(target_path)

    ok = run_ffmpeg([
        '-i', temp_out,
        '-i', target_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', output_path
    ])

    if not ok:
        move_temp(target_path, output_path)


# ===================================================================
#  FRAME PATHS
# ===================================================================
def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_dir = get_temp_directory_path(target_path)
    fmt = roop.globals.temp_frame_format
    return glob.glob(os.path.join(glob.escape(temp_dir), f"*.{fmt}"))


# ===================================================================
#  OUTPUT NORMALIZE
# ===================================================================
def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        src = Path(source_path).stem
        tgt = Path(target_path).stem
        ext = Path(target_path).suffix

        if Path(output_path).is_dir():
            return str(Path(output_path) / f"{src}-{tgt}{ext}")

    return output_path


# ===================================================================
#  TEMP OPS
# ===================================================================
def create_temp(target_path: str) -> None:
    Path(get_temp_directory_path(target_path)).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_out = Path(get_temp_output_path(target_path))
    output_path = Path(output_path)

    if temp_out.exists():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(temp_out), str(output_path))


def clean_temp(target_path: str) -> None:
    temp_dir = Path(get_temp_directory_path(target_path))
    parent = temp_dir.parent

    try:
        if not roop.globals.keep_frames and temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception:
        pass

    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        pass


# ===================================================================
#  FILE TYPE CHECK
# ===================================================================
def has_image_extension(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in ('.png', '.jpg', '.jpeg', '.webp')


def is_image(path: str) -> bool:
    if Path(path).is_file():
        m, _ = mimetypes.guess_type(path)
        return m and m.startswith("image/")
    return False


def is_video(path: str) -> bool:
    if Path(path).is_file():
        m, _ = mimetypes.guess_type(path)
        return m and m.startswith("video/")
    return False


# ===================================================================
#  DOWNLOADER
# ===================================================================
def conditional_download(directory: str, urls: List[str]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for url in urls:
        out_file = directory / os.path.basename(url)

        if not out_file.exists():
            try:
                req = urllib.request.urlopen(url)
                total = int(req.headers.get('Content-Length', 0))

                with tqdm(total=total, unit='B', unit_scale=True,
                          desc=f"Downloading {out_file.name}") as pb:
                    urllib.request.urlretrieve(
                        url, out_file,
                        reporthook=lambda c, b, t: pb.update(b)
                    )
            except Exception as e:
                print("[DOWNLOAD ERROR] →", str(e))


# ===================================================================
#  PATH RESOLVER
# ===================================================================
def resolve_relative_path(path: str) -> str:
    return str(Path(__file__).parent.joinpath(path).resolve())
