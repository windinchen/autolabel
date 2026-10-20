"""视频抽帧导入：优先 ffmpeg，缺 ffmpeg 时用 OpenCV 兜底。

「这台机器能不能抽帧」是装了什么决定的：ffmpeg 在 Windows 上默认没有、macOS 上要
Homebrew 才有，而这两个平台都是要跑这个项目的——只认 ffmpeg 就等于「没装工具时
视频导入永远空白，还不说为什么」。所以抽帧走两级：ffmpeg（最快，而且只有它支持
按镜头切换取样）→ OpenCV（跟着标注依赖一起来，几乎总在）。两级都没有才报缺依赖，
并把该装哪一个写清楚。
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple


# ffprobe 只是读元信息，正常不到 1 秒；卡住基本等于文件损坏或挂载盘失联
PROBE_TIMEOUT = 60
# 抽帧超时下限/上限（秒）：真实长视频要留够时间，但绝不能无限占住后台任务线程
FFMPEG_TIMEOUT_MIN = 300.0
FFMPEG_TIMEOUT_MAX = 3600.0
# 读不到帧率时按这个算采样间隔（绝大多数手机/监控录像都在这附近）
FALLBACK_FPS = 25.0
# 采样密度下限：再稀就退化成「整段视频只取一两帧」，多半是参数填错了
MIN_SAMPLE_FPS = 0.02
JPEG_QUALITY = 92


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(f"未找到 ffmpeg：{decoder()['note']}")
    return path


def has_opencv() -> bool:
    """这台机器有没有 OpenCV 可用作兜底解码器。

    只查包在不在（``find_spec`` 不会真的 import cv2），因为 ``import cv2`` 要几百毫秒，
    而这个函数在渲染「系统设置」面板时每次都要问一遍。
    """
    try:
        return importlib.util.find_spec("cv2") is not None
    except (ImportError, ValueError):
        return False


def decoder() -> dict:
    """这台机器的视频解码能力：用哪一级、缺什么、该装什么。

    界面上「导入视频后一片空白」的第一嫌疑就是这里，所以「系统设置」面板要能直接
    答出来，而不是让人去猜是不是软件坏了。
    """
    ffmpeg = bool(shutil.which("ffmpeg"))
    opencv = has_opencv()
    name = "ffmpeg" if ffmpeg else ("opencv" if opencv else "")
    labels = {"ffmpeg": "ffmpeg（最快，支持按镜头切换取样）",
              "opencv": "OpenCV 内置解码（够用）",
              "": "缺解码器，视频导不进来"}
    notes = {"ffmpeg": "",
             "opencv": "没找到 ffmpeg，改用 OpenCV 抽帧：按镜头切换取样会自动换成定时取样",
             "": "装 ffmpeg（macOS 用 brew install ffmpeg、Windows 用 winget install ffmpeg）"
                 "或 pip install opencv-python，之后就能导入视频"}
    return {"name": name, "label": labels[name], "ffmpeg": ffmpeg, "opencv": opencv,
            "scene": ffmpeg, "ready": bool(name), "note": notes[name]}


def decode_timeout(span: float = 0.0) -> float:
    """本次抽帧的超时秒数：按要处理的片段长度推导，可用环境变量覆盖。

    默认给「片段时长的 4 倍 + 5 分钟」，低配 CPU 上跑 4K 长视频也够；
    ``AUTOLABEL_FFMPEG_TIMEOUT`` 设为正数时直接用它（调试/大机器逃生阀）。
    两级解码器共用这一份预算——占住后台线程的都是抽帧，不管它用谁。
    """
    override = str(os.environ.get("AUTOLABEL_FFMPEG_TIMEOUT") or "").strip()
    if override:
        try:
            wanted = float(override)
        except ValueError:
            wanted = 0.0
        if wanted > 0:
            return wanted
    span = max(0.0, float(span or 0.0))
    return min(FFMPEG_TIMEOUT_MAX, max(FFMPEG_TIMEOUT_MIN, FFMPEG_TIMEOUT_MIN + span * 4))


# 老名字留着：测试与外部调用都写的是 ffmpeg_timeout，而它管的事现在叫「抽帧预算」
ffmpeg_timeout = decode_timeout


# --------------------------------------------------------------------------- //
# 元信息
# --------------------------------------------------------------------------- //
def _blank_info(path: str) -> dict:
    return {"path": os.path.abspath(path), "duration": None, "fps": None,
            "width": None, "height": None, "frames": None, "backend": ""}


def probe_video(path: str) -> dict:
    """读取视频元信息：有 ffprobe 用 ffprobe，否则退到 OpenCV。

    取不到元信息不是坏事——采样间隔按请求的 fps 算就够了，只有「时长」会缺，
    而缺时长最多是少抽几帧，不该让整次导入失败。
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probed = _probe_ffprobe(ffprobe, path)
        if probed is not None:
            return probed
    probed = _probe_opencv(path)
    if probed is not None:
        return probed
    return _blank_info(path)


def _probe_ffprobe(ffprobe: str, path: str) -> Optional[dict]:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration",
           "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True,
                             stdin=subprocess.DEVNULL, timeout=PROBE_TIMEOUT).stdout
    except subprocess.TimeoutExpired:
        print(f"[warn] ffprobe 超过 {PROBE_TIMEOUT} 秒未返回，跳过元信息读取: {os.path.basename(path)}")
        return None
    except (subprocess.CalledProcessError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"[warn] ffprobe 失败: {detail.strip()[:200]}")
        return None
    values: dict = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values.setdefault(key.strip(), value.strip())
    info = _blank_info(path)
    info["backend"] = "ffprobe"
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


def _probe_opencv(path: str) -> Optional[dict]:
    """没有 ffprobe 时向 OpenCV 要元信息。

    帧数在某些编码上不准（可变帧率/流式 mp4 会给出 0 或估计值），所以只在
    两处用得到它：告诉用户视频多长、以及用户显式指定片段长度时的换算。
    """
    cv2 = _cv2()
    if cv2 is None:
        return None
    try:
        with _capture(cv2, path) as cap:
            if not cap.isOpened():
                return None
            fps = _usable_fps(cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    except Exception:  # noqa: BLE001 解码器崩了不该带走整个进程
        return None
    info = _blank_info(path)
    info["backend"] = "opencv"
    info["fps"] = fps
    info["frames"] = frames or None
    info["width"], info["height"] = width or None, height or None
    info["duration"] = round(frames / fps, 3) if fps and frames > 0 else None
    return info


def _usable_fps(value: object) -> Optional[float]:
    """把元信息里的帧率收成能用的小数：0、负数、四位数的可变帧率垃圾都不算。"""
    try:
        fps = float(value or 0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return fps if 0.0 < fps < 1000.0 else None


def _sample_interval(src_fps: float, fps: float) -> float:
    """相邻两帧之间要跨过多少个源帧——跨距保留小数，别四舍五入成整数。

    整数步长是「抽出来的张数对不上」的源头之一：25 fps 的录像要「每秒 2 帧」，
    ``round(25 / 2)`` 得 12，实际间隔变成 0.48 秒，一条 21.5 秒的录像就多抽 2 帧，
    用户照 ``21.5 × 2 = 43`` 数就对不上。跨距留小数后取样点落在 0 / 0.5 / 1.0 秒，
    张数正好是 ``floor(时长 × fps) + 1``，和 ffmpeg 的 ``fps`` 滤镜同一个口径。
    """
    return max(1.0, float(src_fps) / max(MIN_SAMPLE_FPS, float(fps) or 0.5))


def planned_frames(duration: float, fps: float) -> int:
    """按「时长 × 频率」这段视频本来该抽几帧（第 0 秒那帧算在内）；时长未知返回 0。

    完成消息拿它来判断「是不是被最多帧数截断了」，用户也能自己乘一遍核对——
    21.5 秒按每秒 2 帧就是 43 帧，抽出来 20 帧又不说原因，就像软件在丢素材。
    """
    span = max(0.0, float(duration or 0.0))
    rate = max(MIN_SAMPLE_FPS, float(fps or 0.0) or 0.5)
    return int(span * rate) + 1 if span > 0 else 0


def sampling_rule(fps: object) -> str:
    """抽帧频率说人话：``0.5`` → 「每 2 秒 1 帧」，``2`` → 「每秒 2 帧」；不像数字就返回空。

    消息里只写「抽出 43 帧」，用户还得去翻设置才知道按什么频率抽的；把口径跟数字
    一起报出来，他拿视频时长乘一下就知道这个数对不对。
    以前「镜头切换取样不可用」的提示写作「每 0.5 秒取 1 帧」——0.5 是**频率**不是
    间隔，读起来正好把倍数说反了。
    """
    try:
        rate = float(fps)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if not rate > 0:
        return ""
    return f"每 {1.0 / rate:g} 秒 1 帧" if rate <= 1 else f"每秒 {rate:g} 帧"


# --------------------------------------------------------------------------- //
# 抽帧
# --------------------------------------------------------------------------- //
def extract_frames(
    video: str,
    dataset_root: str,
    fps: float = 0.5,
    max_frames: int = 600,
    scene: Optional[float] = None,
    start: float = 0.0,
    duration: Optional[float] = None,
    max_side: int = 1600,
    dedupe: str = "exact",
    subdir: str = "",
    keep_temp: bool = False,
    lib: str = "pose",
    register: bool = True,
) -> dict:
    """抽帧并导入数据集。

    :param fps: 每秒采样帧数（0.5 = 每 2 秒一帧）；scene 模式下忽略
    :param scene: 场景切换阈值（0~1，常用 0.25~0.4），设置后按镜头切点取帧；
        只有 ffmpeg 认这个参数，用 OpenCV 兜底时自动退回按 ``fps`` 定时取样
    :param start: 从第几秒开始
    :param duration: 只处理这么长的片段；不给就按元信息里的时长（读不到就取到结尾）
    :param dedupe: 默认 ``exact``——只跳过字节完全相同的帧。抽出来的帧本来就是挑过的，
        而且几乎不会字节相同；按 ``perceptual`` 去重会把画面相近的帧当成同一帧丢掉，
        几条同场景的录像一起导，264 帧能只剩 46 帧，看着就像抽帧在丢素材
    :param lib: 目标标注库（``pose`` / ``objects``），素材仍落在共享池里
    :param register: ``False`` 表示抽出来的帧只进素材池，不进标注库索引
    """
    from .dataset import import_prepared

    video = os.path.abspath(os.path.expanduser(video))
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    info = probe_video(video)
    # 只有显式要求的片段长度才交给解码器当停止条件：OpenCV 报的帧数不可靠，
    # 拿它反推的时长去截断会悄悄少抽后半段，宁可多解到 max_frames 为止。
    clip = max(0.0, float(duration)) if duration else 0.0
    cap = max(1, int(max_frames))
    stem = _safe_stem(video)
    tmp = tempfile.mkdtemp(prefix="autolabel_frames_")
    try:
        prepared, extracted, backend, note = decode_frames(
            video, tmp, stem, fps, scene, float(start or 0.0), clip, max_side, cap, info)
        result = import_prepared(dataset_root, prepared, dedupe=dedupe, max_side=max_side,
                                 subdir=subdir, lib=lib, register=register)
    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
    # 「最多帧数」截断要自己说一声：用户按 时长 × 频率 乘出来的数总比实际大，
    # 少了一批又没一句解释，看起来就像抽帧在丢帧。
    span = clip or max(0.0, float(info.get("duration") or 0.0) - max(0.0, float(start or 0.0)))
    planned = 0 if (scene and backend == "ffmpeg") else planned_frames(span, fps)
    if planned > cap and extracted >= cap:
        note = (note + "；" if note else "") + (
            f"整段该抽 {planned} 帧，已按「最多帧数」截到 {cap} 帧（要全抽就调大上限）")
    result.update({"video": video, "video_info": info, "extracted": extracted,
                   "mode": "scene" if (scene and backend == "ffmpeg") else "fps",
                   "backend": backend, "note": note, "sampled": len(prepared)})
    return result


def decode_frames(
    video: str, tmp: str, stem: str, fps: float, scene: Optional[float], start: float,
    duration: float, max_side: int, max_frames: int, info: dict,
) -> Tuple[list, int, str, str]:
    """按这台机器现阶段的解码能力抽帧，返回 ``(待导入帧, 抽出帧数, 用的解码器, 补充说明)``。

    ffmpeg 优先：它一次进程解完整段，比逐帧解码快一个量级，而且是唯一能按镜头
    切点取帧的。缺它时退回 OpenCV，并如实说明「按镜头取样退成了定时取样」。
    """
    if shutil.which("ffmpeg"):
        prepared, extracted = _run_ffmpeg(video, tmp, stem, fps, scene, start,
                                          duration, max_side, max_frames, info)
        return prepared, extracted, "ffmpeg", ""
    if has_opencv():
        note = ""
        if scene:
            note = f"没有 ffmpeg，按镜头切换取样用不了，已改成{sampling_rule(fps) or f'每秒取 {fps} 帧'}"
        prepared, extracted, timed_out = _run_opencv(video, tmp, stem, fps, start,
                                                     duration, max_side, max_frames, info)
        if timed_out:
            note = (note + "；" if note else "") + (
                f"解码超过预算 {int(decode_timeout(span=duration or (info.get('duration') or 0.0)))} 秒"
                f"，只保留前 {extracted} 帧（可用 AUTOLABEL_FFMPEG_TIMEOUT 放宽，单位秒）")
        return prepared, extracted, "opencv", note
    raise RuntimeError(f"抽不出帧：{decoder()['note']}")


def _run_ffmpeg(
    video: str, tmp: str, stem: str, fps: float, scene: Optional[float], start: float,
    duration: Optional[float], max_side: int, max_frames: int, info: dict,
) -> Tuple[List[Tuple[str, str, dict]], int]:
    binary = ffmpeg_bin()
    cmd = [binary, "-hide_banner", "-loglevel", "error", "-ss", f"{float(start or 0.0):.3f}"]
    clip = float(duration or 0.0)
    if not clip and info.get("duration"):
        clip = max(0.0, float(info["duration"]) - float(start or 0.0))
    if clip > 0:
        cmd += ["-t", f"{clip:.3f}"]
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
    limit = decode_timeout(clip or (info.get("duration") or 0.0))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=limit)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"抽帧超过 {int(limit)} 秒被中止，视频可能损坏或过长；"
            "可用环境变量 AUTOLABEL_FFMPEG_TIMEOUT 放宽上限（单位秒）"
        ) from None
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败: {proc.stderr.strip()[:400]}")
    frames = sorted(f for f in os.listdir(tmp) if f.lower().endswith(".jpg"))
    scene_times = _parse_showinfo_times(proc.stderr) if scene else []
    picked: List[Tuple[float, str]] = []
    for i, name in enumerate(frames):
        timestamp = (start + scene_times[i]) if (scene and i < len(scene_times)) \
            else start + (i / float(fps) if fps else 0.0)
        picked.append((timestamp, name))
    return _entries(tmp, picked, stem, video, info), len(frames)


def _run_opencv(
    video: str, tmp: str, stem: str, fps: float, start: float,
    duration: float, max_side: int, max_frames: int, info: dict,
) -> Tuple[List[Tuple[str, str, dict]], int, bool]:
    """用 OpenCV 抽帧。见 :func:`_sample_opencv` 的采样办法。"""
    cv2 = _cv2()
    if cv2 is None:
        raise RuntimeError("OpenCV 没能加载，无法抽帧：pip install opencv-python 后重试")
    try:
        with _capture(cv2, video) as cap:
            if not cap.isOpened():
                raise RuntimeError(f"这台机器的解码器打不开 {os.path.basename(video)}："
                                   "编码不支持或文件已损坏（装上 ffmpeg 可以再试一次）")
            picked, timed_out = _sample_opencv(cv2, cap, tmp, fps, start, duration,
                                               max_side, max_frames, info)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 解码器崩了要报在任务上，不能报成"没有 ffmpeg"
        raise RuntimeError(f"OpenCV 打不开这个视频：{type(exc).__name__}: {exc}") from exc
    return _entries(tmp, picked, stem, video, info), len(picked), timed_out


def _sample_opencv(cv2, cap, tmp: str, fps: float, start: float, duration: float,
                   max_side: int, max_frames: int, info: dict):
    """``grab()`` 跳过不要的帧、只对留下的帧 ``retrieve()``。

    逐帧 ``read()`` 会把每一帧都转成数组，长视频上慢得没法用；``grab()`` 只推进
    解码器不取像素，省掉的就是那部分转换。超时不报错而是收尾返回已抽到的帧——
    半截帧比整次失败有用得多，说明写在返回的 note 里。

    取样点按 :func:`_sample_interval` 的跨距排在 0 / 0.5 / 1.0 秒这样的整格上，
    「每秒 2 帧、21.5 秒」就是不多不少 43 帧，用户自己乘一遍就能核对。
    """
    src_fps = _usable_fps(info.get("fps")) or _usable_fps(cap.get(cv2.CAP_PROP_FPS)) or FALLBACK_FPS
    interval = _sample_interval(src_fps, fps)
    first = max(0, int(round(max(0.0, float(start or 0.0)) * src_fps)))
    if first and not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
        first = 0  # 这个容器不支持定位，老实从头解
    limit = int(float(duration) * src_fps) if duration else 0
    deadline = time.monotonic() + decode_timeout(duration or (info.get("duration") or 0.0))
    picked: List[Tuple[float, str]] = []
    timed_out = False
    seen = 0
    want = float(first)  # 下一个该留的源帧号：起点 + 已留帧数 × 跨距
    while len(picked) < max(1, int(max_frames)):
        if not cap.grab():
            break
        # grab() 之后 POS_FRAMES 指的是下一帧；某些容器它一直返回 0，退回自己数
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
        if pos < first:
            pos = first + seen
        seen += 1
        if limit and seen > limit:
            break
        if pos + 0.5 < want:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break
        name = f"{len(picked):06d}.jpg"
        _write_jpeg(cv2, frame, os.path.join(tmp, name), max_side)
        picked.append((pos / src_fps, name))
        want = first + interval * len(picked)
        if time.monotonic() > deadline:
            timed_out = True
            break
    return picked, timed_out


def _wants_ascii_link(path: str) -> bool:
    """要不要为这个路径多挂一次 ASCII 硬链接：Windows 且路径带非 ASCII 字符。"""
    return os.name == "nt" and not os.path.abspath(path).isascii()


@contextmanager
def _capture(cv2, path: str):
    """打开视频；Windows 上打不开且路径带中文时，先挂一个 ASCII 名字的硬链接再试。

    Windows 的 ``cv2.VideoCapture`` 对非 ASCII 路径没把握（和 ``cv2.imwrite`` 只认
    ANSI 是同一个坑），而这个项目的数据集目录十有八九带中文——「素材池导入视频
    在 mac 上好的、Windows 上一帧都没有」就是它。硬链接不复制数据、同盘瞬间完成、
    解完就删，代价接近零；挂不上（跨盘或文件系统不支持）就照原样报错，
    行为和没加这段一样。
    """
    cap = cv2.VideoCapture(path)
    link = ""
    if not cap.isOpened() and _wants_ascii_link(path):
        cap.release()
        handle, link = tempfile.mkstemp(suffix=os.path.splitext(path)[1].lower() or ".mp4")
        os.close(handle)
        try:
            os.unlink(link)             # 硬链接不能顶着一个已存在的文件
            os.link(path, link)
        except OSError:
            link = ""
        cap = cv2.VideoCapture(link) if link else cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()
        if link:
            try:
                os.unlink(link)
            except OSError:
                pass


def _cv2():
    try:
        import cv2

        return cv2
    except Exception:  # noqa: BLE001 装坏了和没装一样，都该往兜底/报错走
        return None


def _write_jpeg(cv2, frame, path: str, max_side: int) -> None:
    """把一帧 BGR 存成 JPEG：交给 PIL 写，不用 ``cv2.imwrite``。

    Windows 上 ``cv2.imwrite`` 遇到中文路径会静默写出不来的文件（它只认 ANSI），
    而这个项目的数据集目录十有八九带中文；PIL 走的是 unicode 路径，稳。
    """
    from PIL import Image

    im = Image.fromarray(frame[:, :, ::-1])
    side = int(max_side or 0)
    if side > 0 and max(im.size) > side:
        im.thumbnail((side, side), Image.LANCZOS)
    im.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True)


def _entries(tmp: str, picked: List[Tuple[float, str]], stem: str, video: str,
             info: dict) -> List[Tuple[str, str, dict]]:
    """统一成 :func:`dataset.import_prepared` 要的 ``(文件, 文件名主干, 时间戳字段)``。

    文件名主干带毫秒时间戳而不是帧序号：同一批帧重导一次名字不变，
    去重才认得出「就是这一帧」，也不会出现第二次导入全算新素材。
    """
    entries: List[Tuple[str, str, dict]] = []
    for i, (timestamp, name) in enumerate(picked):
        ms = int(round(max(0.0, float(timestamp)) * 1000))
        entries.append((
            os.path.join(tmp, name),
            f"{stem}_t{ms}",
            {"video": video, "time_sec": round(max(0.0, float(timestamp)), 3),
             "frame_index": i, "video_fps": info.get("fps")},
        ))
    return entries


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "clip"


def _parse_showinfo_times(stderr: str) -> List[float]:
    return [float(m) for m in re.findall(r"pts_time:([0-9]+\.?[0-9]*)", stderr or "")]


__all__ = ["decode_frames", "decode_timeout", "decoder", "extract_frames", "ffmpeg_bin",
           "ffmpeg_timeout", "has_opencv", "planned_frames", "probe_video", "sampling_rule"]
