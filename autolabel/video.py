"""视频抽帧导入：ffmpeg 定帧率 / 场景切换采样，帧携带时间戳写入索引。"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("未找到 ffmpeg，请先安装（brew install ffmpeg）")
    return path


def probe_video(path: str) -> dict:
    """读取视频元信息。"""
    ffprobe = shutil.which("ffprobe")
    info = {"path": os.path.abspath(path), "duration": None, "fps": None,
            "width": None, "height": None, "frames": None}
    if not ffprobe:
        return info
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration",
           "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        print(f"[warn] ffprobe 失败: {exc.stderr.strip()[:200]}")
        return info
    values: dict = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values.setdefault(key.strip(), value.strip())
    if values.get("avg_frame_rate") and values["avg_frame_rate"] != "0/0":
        num, _, den = values["avg_frame_rate"].partition("/")
        try:
            info["fps"] = float(num) / float(den or 1)
        except ValueError:
            pass
    duration = values.get("duration")
    if duration and duration != "N/A":
        info["duration"] = float(duration)
    for key in ("width", "height", "nb_frames"):
        if values.get(key) and values[key].isdigit():
            info[{"nb_frames": "frames"}.get(key, key)] = int(values[key])
    return info


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "clip"


def extract_frames(
    video: str,
    dataset_root: str,
    fps: float = 0.5,
    max_frames: int = 600,
    scene: Optional[float] = None,
    start: float = 0.0,
    duration: Optional[float] = None,
    max_side: int = 1600,
    dedupe: str = "perceptual",
    subdir: str = "",
    keep_temp: bool = False,
    lib: str = "pose",
) -> dict:
    """抽帧并导入数据集。

    :param fps: 每秒采样帧数（0.5 = 每 2 秒一帧）；scene 模式下忽略
    :param scene: 场景切换阈值（0~1，常用 0.25~0.4），设置后按镜头切点取帧
    :param lib: 目标标注库（``pose`` / ``objects``），素材仍落在共享池里
    """
    from .dataset import import_prepared

    video = os.path.abspath(os.path.expanduser(video))
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    info = probe_video(video)
    if duration is None and info.get("duration"):
        duration = max(0.0, info["duration"] - start)
    stem = _safe_stem(video)
    tmp = tempfile.mkdtemp(prefix="autolabel_frames_")
    try:
        prepared, extracted = _run_ffmpeg(video, tmp, stem, fps, scene, start, duration,
                                          max_side, max_frames, info)
        result = import_prepared(dataset_root, prepared, dedupe=dedupe, max_side=max_side,
                                 subdir=subdir, lib=lib)
    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
    result.update({"video": video, "video_info": info, "extracted": extracted,
                   "mode": "scene" if scene else "fps", "sampled": len(prepared)})
    return result


def _run_ffmpeg(
    video: str, tmp: str, stem: str, fps: float, scene: Optional[float], start: float,
    duration: Optional[float], max_side: int, max_frames: int, info: dict,
) -> Tuple[List[Tuple[str, str, dict]], int]:
    binary = ffmpeg_bin()
    cmd = [binary, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", video]
    scale = f"scale='min({int(max_side)},iw)':-2"
    if scene:
        vf = f"select='gt(scene,{scene})',showinfo,{scale}"
        cmd += ["-vf", vf, "-fps_mode", "vfr"]
    else:
        vf = f"fps={float(fps)},{scale}"
        cmd += ["-vf", vf]
    cmd += ["-frames:v", str(int(max_frames)), "-q:v", "2", "-start_number", "0",
            os.path.join(tmp, "%06d.jpg")]
    print("[info] ffmpeg: " + " ".join(cmd[-6:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败: {proc.stderr.strip()[:400]}")
    frames = sorted(f for f in os.listdir(tmp) if f.lower().endswith(".jpg"))
    scene_times = _parse_showinfo_times(proc.stderr) if scene else []
    prepared: List[Tuple[str, str, dict]] = []
    for i, name in enumerate(frames):
        if scene and i < len(scene_times):
            timestamp = start + scene_times[i]
        else:
            timestamp = start + (i / float(fps) if fps else 0.0)
        ms = int(round(timestamp * 1000))
        prepared.append((
            os.path.join(tmp, name),
            f"{stem}_t{ms}",
            {"video": video, "time_sec": round(timestamp, 3), "frame_index": i,
             "video_fps": info.get("fps")},
        ))
    return prepared, len(frames)


def _parse_showinfo_times(stderr: str) -> List[float]:
    return [float(m) for m in re.findall(r"pts_time:([0-9]+\.?[0-9]*)", stderr or "")]
