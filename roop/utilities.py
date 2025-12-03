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
#  SSL Fix for MacOS
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
#  FPS DETECTOR — required by core.py
# ===================================================================
def detect_fps(target_path: str) -> float:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        target_path
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        if out.isdigit():
            return float(out)
        if "/" in out:
            n, d = out.split("/")
            return float(n) / float(d)
        return float(out)
    except Exception:
        return 30.0


# ===================================================================
#  CODEC PROBE + CUVID CHECK
# ===================================================================
def probe_codec(path: str) -> Optional[str]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nokey=1:noprint_wrappers=1",
        path
    ]
    try:
        return subprocess.check_output(cmd).decode().strip()
    except:
        return None


def is_decoder_available(name: str) -> bool:
    try:
        dec = subprocess.check_output(["ffmpeg", "-hide_banner", "-decoders"]).decode().lower()
        return name.lower() in dec
    except:
        return False


def choose_gpu_decoder(codec: str) -> Optional[str]:
    if not codec:
        return None
    codec = codec.lower()

    if codec in ("h264", "avc1", "mpeg4"):
        return "h264_cuvid"

    if codec in ("hevc", "h265"):
        return "hevc_cuvid"

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
#  EXTRACT FRAMES (GPU + HWDOWNLOAD FIX)
# ===================================================================
def extract_frames(target_path: str, fps: float = 30.0) -> bool:
    return extract_frames_gpu(target_path, fps, force_gpu=False)


def extract_frames_gpu(target_path: str, fps: float = 30.0, force_gpu: bool = False) -> bool:
    temp_dir = get_temp_directory_path(target_path)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    fmt = roop.globals.temp_frame_format
    qv = roop.globals.temp_frame_quality * 31 // 100

    codec = probe_codec(target_path)
    gpu_decoder = choose_gpu_decoder(codec)

    # ---------------------------------------------------------------
    #  TRY GPU DECODE (CUVID)
    # ---------------------------------------------------------------
    if gpu_decoder and is_decoder_available(gpu_decoder):
        print(f"[extract_frames_gpu] GPU decode → {gpu_decoder} (codec={codec})")

        args = [
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-c:v", gpu_decoder,
            "-i", target_path,

            # HWDOWNLOAD FIX — the ONLY working chain in Kaggle
            "-vf", f"hwdownload,format=yuv420p,fps={fps}",

            "-pix_fmt", "rgb24",
            "-q:v", str(qv),
            os.path.join(temp_dir, f"%04d.{fmt}")
        ]

        ok = run_ffmpeg(args)
        if ok:
            print("[extract_frames_gpu] sukses GPU decode.")
            return True

        print("[extract_frames_gpu] GPU decode gagal.")
        if force_gpu:
            raise RuntimeError("FFmpeg gagal memakai GPU decoder.")
        print("[extract_frames_gpu] fallback CPU decode...")

    else:
        print(f"[extract_frames_gpu] GPU decoder tidak tersedia (codec={codec}), CPU fallback.")

    # ---------------------------------------------------------------
    #  CPU FALLBACK
    # ---------------------------------------------------------------
    cpu_args = [
        "-hwaccel", "auto",
        "-i", target_path,
        "-q:v", str(qv),
        "-pix_fmt", "rgb24",
        "-vf", f"fps={fps}",
        os.path.join(temp_dir, f"%04d.{fmt}")
    ]

    ok = run_ffmpeg(cpu_args)
    if ok:
        print("[extract_frames_gpu] CPU decode sukses.")
    return ok


# ===================================================================
#  VIDEO CREATION
# ===================================================================
def create_video(target_path: str, fps: float = 30.0) -> bool:
    temp_dir = get_temp_directory_path(target_path)
    out = get_temp_output_path(target_path)
    fmt = roop.globals.temp_frame_format
    enc = roop.globals.output_video_encoder
    qv = (roop.globals.output_video_quality + 1) * 51 // 100

    args = [
        "-hwaccel", "auto",
        "-r", str(fps),
        "-i", os.path.join(temp_dir, f"%04d.{fmt}"),
        "-c:v", enc,
        "-pix_fmt", "yuv420p",
        "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1"
    ]

    if enc in ("libx264", "libx265", "libvpx"):
        args.extend(["-crf", str(qv)])
    elif enc in ("h264_nvenc", "hevc_nvenc"):
        args.extend(["-cq", str(qv)])

    args.extend(["-y", out])
    return run_ffmpeg(args)


# ===================================================================
#  AUDIO RESTORE
# ===================================================================
def restore_audio(target_path: str, out: str) -> None:
    temp_out = get_temp_output_path(target_path)

    ok = run_ffmpeg([
        "-i", temp_out,
        "-i", target_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-y", out
    ])

    if not ok:
        move_temp(target_path, out)


# ===================================================================
#  FRAME PATHS
# ===================================================================
def get_temp_frame_paths(target_path: str) -> List[str]:
    temp = get_temp_directory_path(target_path)
    fmt = roop.globals.temp_frame_format
    return glob.glob(os.path.join(glob.escape(temp), f"*.{fmt}"))


# ===================================================================
#  OUTPUT PATH NORMALIZATION
# ===================================================================
def normalize_output_path(src, tgt, out):
    if src and tgt and out:
        src_name = Path(src).stem
        tgt_name = Path(tgt).stem
        ext = Path(tgt).suffix
        if Path(out).is_dir():
            return str(Path(out) / f"{src_name}-{tgt_name}{ext}")
    return out


# ===================================================================
#  TEMP FILE OPS
# ===================================================================
def create_temp(path: str) -> None:
    Path(get_temp_directory_path(path)).mkdir(parents=True, exist_ok=True)


def move_temp(path: str, out: str) -> None:
    src = Path(get_temp_output_path(path))
    out = Path(out)

    if src.exists():
        if out.exists():
            out.unlink()
        shutil.move(str(src), str(out))


def clean_temp(path: str) -> None:
    temp = Path(get_temp_directory_path(path))
    parent = temp.parent

    try:
        if not roop.globals.keep_frames and temp.exists():
            shutil.rmtree(temp)
    except:
        pass

    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except:
        pass


# ===================================================================
#  IMAGE / VIDEO CHECKERS
# ===================================================================
def has_image_extension(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in (".png", ".jpg", ".jpeg", ".webp")


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
                total = int(req.headers.get("Content-Length", 0))

                with tqdm(total=total, unit="B", unit_scale=True,
                          desc=f"Downloading {out_file.name}") as pb:
                    urllib.request.urlretrieve(
                        url, out_file,
                        reporthook=lambda c, b, t: pb.update(b)
                    )
            except Exception as e:
                print("[DOWNLOAD ERROR] →", str(e))


# ===================================================================
#  RESOLVE RELATIVE PATH
# ===================================================================
def resolve_relative_path(path: str) -> str:
    return str(Path(__file__).parent.joinpath(path).resolve())
