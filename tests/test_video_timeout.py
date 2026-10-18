"""ffmpeg / ffprobe 的超时保护：卡死的外部进程不能永久占住后台任务线程。"""

import os
import time

import pytest

from autolabel import video


def make_fake_binary(tmp_path, name, body, monkeypatch):
    """写一个假的 ffmpeg/ffprobe 可执行文件并把它放到 PATH 最前面。

    用 ``exec`` 替换掉 sh 本身，否则 kill 只杀到 sh，留下的孙子进程会继续
    持有管道，让 subprocess 在超时后反而多等一大截。
    """
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    script = bindir / name
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return str(script)


def test_ffmpeg_timeout_defaults_scale_with_duration_and_clamp(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_FFMPEG_TIMEOUT", raising=False)
    assert video.ffmpeg_timeout(0) == video.FFMPEG_TIMEOUT_MIN
    assert video.ffmpeg_timeout(120) == video.FFMPEG_TIMEOUT_MIN + 480
    assert video.ffmpeg_timeout(999999) == video.FFMPEG_TIMEOUT_MAX

    monkeypatch.setenv("AUTOLABEL_FFMPEG_TIMEOUT", "7")
    assert video.ffmpeg_timeout(999999) == 7.0
    monkeypatch.setenv("AUTOLABEL_FFMPEG_TIMEOUT", "不是数字")
    assert video.ffmpeg_timeout(0) == video.FFMPEG_TIMEOUT_MIN


def test_hung_ffmpeg_is_aborted_with_readable_error(tmp_path, monkeypatch):
    make_fake_binary(tmp_path, "ffmpeg", "exec sleep 30", monkeypatch)
    monkeypatch.setenv("AUTOLABEL_FFMPEG_TIMEOUT", "1")
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 16)

    started = time.time()
    with pytest.raises(RuntimeError) as exc:
        video._run_ffmpeg(str(video_path), str(tmp_path / "out"), "clip",
                          1.0, None, 0.0, 10.0, 1600, 600, {})
    elapsed = time.time() - started
    assert "抽帧超过" in str(exc.value)
    assert "AUTOLABEL_FFMPEG_TIMEOUT" in str(exc.value)
    assert elapsed < 10, f"超时没有生效，白等了 {elapsed:.1f} 秒"


def test_fast_fake_ffmpeg_still_yields_frames(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FAKE_OUT", str(out))
    make_fake_binary(tmp_path, "ffmpeg", 'printf "" > "$FAKE_OUT/000000.jpg"; exit 0', monkeypatch)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 16)

    prepared, extracted = video._run_ffmpeg(
        str(video_path), str(out), "clip", 1.0, None, 0.0, 1.0, 1600, 1, {"fps": 25.0})
    assert extracted == 1 and len(prepared) == 1
    assert prepared[0][1] == "clip_t0"
    assert prepared[0][2]["video_fps"] == 25.0


def test_hung_ffprobe_degrades_to_unknown_metadata(tmp_path, monkeypatch):
    make_fake_binary(tmp_path, "ffprobe", "exec sleep 30", monkeypatch)
    monkeypatch.setattr(video, "PROBE_TIMEOUT", 1)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00" * 16)

    started = time.time()
    info = video.probe_video(str(video_path))
    assert time.time() - started < 10
    assert info["duration"] is None and info["fps"] is None
    assert info["path"].endswith("clip.mp4")
