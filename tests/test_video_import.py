"""视频导入素材：解码器兜底 + 每个模块上传视频都真的去抽帧。

这里守两件曾经都坏过的事：

1. **抽帧只认 ffmpeg**。macOS 上没装 Homebrew、Windows 上默认都没有 ffmpeg，
   于是视频导入必然失败——失败本身不可怕，可怕的是界面上只是"什么都没出现"。
2. **只有姿态页那个 video 任务会抽帧**。素材池页上传走 ``pool`` 任务、提示词标注与
   目标检测标注把图片和视频混在同一个任务里，三处都把 .mp4 当图片交给解码器，
   一张都进不来。
"""

import os

import numpy as np
import pytest
from PIL import Image

from autolabel import upload as upload_module
from autolabel import video
from autolabel.dataset import init_dataset, load_index
from autolabel.pool import load_catalog

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autolabel", "static")


def read_static(name: str) -> str:
    with open(os.path.join(STATIC, name), encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------------------- //
# 解码器现状
# --------------------------------------------------------------------------- //
def test_decoder_reports_a_usable_backend_or_how_to_fix_it(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: False)
    info = video.decoder()
    assert info["ready"] is False and info["name"] == ""
    # 缺什么、该装什么必须写在同一句话里，Windows 与 macOS 的装法都要给
    assert "opencv-python" in info["note"]
    assert "brew install ffmpeg" in info["note"] and "winget install ffmpeg" in info["note"]


def test_ffmpeg_is_preferred_and_needs_no_apology(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    info = video.decoder()
    assert info["name"] == "ffmpeg" and info["ready"] and info["scene"]
    assert info["note"] == ""


def test_opencv_fallback_admits_that_scene_sampling_is_lost(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: True)
    info = video.decoder()
    assert info["name"] == "opencv" and info["ready"] and info["scene"] is False


# --------------------------------------------------------------------------- //
# 抽帧分派
# --------------------------------------------------------------------------- //
def _fake_run_opencv(box):
    def run(clip, tmp, stem, fps, start, duration, max_side, max_frames, info):
        box.update({"fps": fps, "start": start, "duration": duration,
                    "max_frames": max_frames, "max_side": max_side})
        Image.new("RGB", (32, 24)).save(os.path.join(tmp, "000000.jpg"))
        return [(os.path.join(tmp, "000000.jpg"), f"{stem}_t500",
                 {"video": clip, "time_sec": 0.5, "frame_index": 0, "video_fps": 25.0})], 1, False
    return run


def test_scene_sampling_without_ffmpeg_degrades_instead_of_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: True)
    monkeypatch.setattr(video, "_run_opencv", _fake_run_opencv({}))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)

    prepared, extracted, backend, note = video.decode_frames(
        str(clip), str(tmp_path), "clip", 0.5, 0.3, 0.0, 0.0, 1600, 600, {"fps": 25.0})
    assert backend == "opencv" and extracted == 1 and len(prepared) == 1
    assert "按镜头切换取样用不了" in note and "每 0.5 秒" in note


def test_decode_without_any_backend_names_both_install_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: False)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)
    with pytest.raises(RuntimeError) as exc:
        video.decode_frames(str(clip), str(tmp_path), "clip", 0.5, None, 0.0, 0.0, 1600, 600, {})
    assert "opencv-python" in str(exc.value)


def test_extract_frames_result_carries_the_backend_used(tmp_path, monkeypatch):
    """完成消息与批次详情都要说"用的哪个解码器、抽了几帧"，字段得先在结果里。"""
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: True)
    box = {}
    monkeypatch.setattr(video, "_run_opencv", _fake_run_opencv(box))
    captured = {}

    def fake_import_prepared(root, prepared, **kwargs):
        captured.update({"count": len(prepared), "kwargs": kwargs})
        return {"imported": len(prepared), "reused": 0, "skipped_exact": 0,
                "skipped_similar": 0, "failed": 0, "imported_files": [],
                "register": kwargs.get("register"), "lib": kwargs.get("lib")}

    monkeypatch.setattr("autolabel.dataset.import_prepared", fake_import_prepared)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)

    result = video.extract_frames(str(clip), str(tmp_path / "ds"), fps=0.5, register=False)
    assert result["backend"] == "opencv" and result["extracted"] == 1
    assert result["mode"] == "fps" and result["register"] is False
    assert captured["kwargs"]["register"] is False and captured["count"] == 1


# --------------------------------------------------------------------------- //
# 真的用 OpenCV 解一段视频（装了 cv2 才跑）
# --------------------------------------------------------------------------- //
@pytest.fixture()
def clip(tmp_path):
    """一段 6 秒、每帧画面都不同的 mp4，用来验证采样间隔与时间戳。"""
    cv2 = pytest.importorskip("cv2", reason="没有 OpenCV 就没有兜底解码可验证")
    writer = cv2.VideoWriter(str(tmp_path / "busy.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    rng = np.random.default_rng(11)
    for i in range(150):
        frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        frame[:, (i * 3) % 160:] = 0
        writer.write(frame)
    writer.release()
    assert (tmp_path / "busy.mp4").stat().st_size > 0
    return str(tmp_path / "busy.mp4")


def test_probe_uses_opencv_when_ffprobe_is_missing(clip, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    info = video.probe_video(clip)
    assert info["backend"] == "opencv"
    assert info["fps"] and abs(info["fps"] - 25.0) < 0.5
    assert (info["width"], info["height"]) == (160, 120)
    assert info["duration"] and abs(info["duration"] - 6.0) < 0.5


def test_opencv_sampling_keeps_the_requested_interval(clip, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    tmp = os.path.join(os.path.dirname(clip), "out")
    os.makedirs(tmp, exist_ok=True)
    info = video.probe_video(clip)

    entries, extracted, timed_out = video._run_opencv(
        clip, tmp, "busy", 0.5, 0.0, 0.0, 1600, 600, info)
    assert timed_out is False and extracted == 3, "每 2 秒 1 帧，6 秒该出 3 帧"
    stamps = [row[2]["time_sec"] for row in entries]
    assert stamps == [0.0, 2.0, 4.0]
    assert [row[1] for row in entries] == ["busy_t0", "busy_t2000", "busy_t4000"]
    assert all(os.path.exists(row[0]) for row in entries)


def test_opencv_sampling_caps_at_max_frames(clip, monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    tmp = os.path.join(os.path.dirname(clip), "cap")
    os.makedirs(tmp, exist_ok=True)
    _entries, extracted, _timed_out = video._run_opencv(
        clip, tmp, "busy", 5.0, 0.0, 0.0, 1600, 8, video.probe_video(clip))
    assert extracted == 8, "抽满「最多帧数」就该停下，别把整段解完"


def test_real_extract_frames_land_in_the_pool(clip, tmp_path):
    """整条链跑通：上传视频 → 抽帧 → 素材池里有东西、来处写的是原文件名。"""
    root = str(tmp_path / "real")
    init_dataset(root)
    stored = upload_module.save_upload(root, "现场录像.mp4", open(clip, "rb").read())
    store = upload_module.JobStore(root)
    job = upload_module.Job(id="vidjob", kind="pool", files=[stored["stored"]],
                            params={"fps": 0.5, "max_frames": 600})
    store._start_batch(job)
    assert store._ingest(job) == 3
    catalog = load_catalog(root)
    assert len(catalog["items"]) == 3
    for item in catalog["items"].values():
        assert item.get("video") == "现场录像.mp4", "临时上传名不该出现在界面上"
        assert item.get("batch") == job.batch_id
    assert job.ingest_note() == "（视频 1 个抽出 3 帧、进池 3 帧）"


def test_windows_gets_an_ascii_link_when_the_path_is_chinese(tmp_path, monkeypatch):
    """Windows + 中文路径：OpenCV 打不开时挂 ASCII 硬链接重试，别直接判视频死了。"""
    calls = []

    class FakeCap:
        def __init__(self, path):
            self.path = path
            self.opened = os.path.abspath(path).isascii()

        def isOpened(self):  # noqa: N802 照着 cv2 的命名
            return self.opened

        def release(self):
            calls.append(("release", self.path))

    class FakeCv2:
        @staticmethod
        def VideoCapture(path):  # noqa: N802
            calls.append(("open", path))
            return FakeCap(path)

    target = tmp_path / "现场录像.mp4"
    target.write_bytes(b"\x00" * 16)
    monkeypatch.setattr(video, "_wants_ascii_link", lambda path: True)
    opened = []

    with video._capture(FakeCv2, str(target)) as cap:
        opened.append(cap)
    assert opened[0].isOpened(), "硬链接那次必须打开成功"
    # 第一次按原路径（打不开）→ 挂 ASCII 硬链接再开一次 → 收尾把链接删掉
    opens = [row[1] for row in calls if row[0] == "open"]
    assert len(opens) == 2
    assert not os.path.abspath(opens[0]).isascii() and os.path.abspath(opens[1]).isascii()
    assert not os.path.exists(opens[1]), "临时硬链接要删干净，别在临时目录里堆视频"


def test_ascii_path_on_windows_is_opened_directly(tmp_path, monkeypatch):
    calls = []

    class FakeCap:
        def __init__(self, path):
            self.path = path

        def isOpened(self):  # noqa: N802
            return True

        def release(self):
            pass

    class FakeCv2:
        @staticmethod
        def VideoCapture(path):  # noqa: N802
            calls.append(path)
            return FakeCap(path)

    monkeypatch.setattr(video, "_wants_ascii_link", lambda path: False)
    with video._capture(FakeCv2, str(tmp_path / "clip.mp4")):
        pass
    assert len(calls) == 1, "本来就打得开，不该多挂一次链接"


# --------------------------------------------------------------------------- //
# 每个模块都要会抽帧
# --------------------------------------------------------------------------- //
@pytest.mark.parametrize("kind", ["pool", "video", "prompt_upload", "detect_upload"])
def test_every_upload_kind_routes_videos_to_extraction(monkeypatch, tmp_path, kind):
    root = str(tmp_path / "ds")
    init_dataset(root)
    stored = upload_module.save_upload(root, "a.mp4", b"\x00" * 16)
    still = upload_module.save_upload(root, "b.jpg", b"\x00" * 16)
    calls = {"video": [], "image": []}

    def fake_extract(path, dataset_root, **kwargs):
        calls["video"].append(os.path.basename(path))
        calls["video_kwargs"] = kwargs
        return {"imported": 2, "reused": 0, "skipped_exact": 0, "skipped_similar": 0,
                "failed": 0, "extracted": 2, "backend": "opencv", "note": "",
                "imported_files": [(path, f"帧_{i}.jpg") for i in range(2)]}

    def fake_import_images(dataset_root, paths, **kwargs):
        calls["image"] = [os.path.basename(p) for p in paths]
        return {"imported": 1, "reused": 0, "skipped_exact": 0, "skipped_similar": 0,
                "failed": 0, "imported_files": [(paths[0], "图片_0.jpg")]}

    monkeypatch.setattr(video, "extract_frames", fake_extract)
    monkeypatch.setattr("autolabel.dataset.import_images", fake_import_images)

    job = upload_module.Job(id=f"job-{kind}", kind=kind,
                            files=[stored["stored"], still["stored"]],
                            params={"fps": 1.0, "max_frames": 300})
    store = upload_module.JobStore(root)
    assert store._ingest(job) == 3, "视频帧 + 图片都要算进库"
    assert [name.split("_", 1)[1] for name in calls["video"]] == ["a.mp4"], "视频必须走抽帧"
    assert [name.split("_", 1)[1] for name in calls["image"]] == ["b.jpg"], "图片必须走直接入库"
    # 素材池页只落池不进库，其余模块要落进自己的库
    assert calls["video_kwargs"]["register"] is (kind != "pool")
    assert calls["video_kwargs"]["fps"] == 1.0 and calls["video_kwargs"]["max_frames"] == 300


def test_videos_only_upload_does_not_touch_the_image_importer(monkeypatch, tmp_path):
    root = str(tmp_path / "ds")
    init_dataset(root)
    stored = upload_module.save_upload(root, "a.mp4", b"\x00" * 16)
    monkeypatch.setattr(video, "extract_frames", lambda *a, **k: {
        "imported": 0, "reused": 0, "skipped_exact": 0, "skipped_similar": 0, "failed": 0,
        "extracted": 0, "backend": "opencv", "note": "", "imported_files": []})
    monkeypatch.setattr("autolabel.dataset.import_images",
                        lambda *a, **k: pytest.fail("纯视频任务不该再走图片导入器"))
    job = upload_module.Job(id="onlyvideo", kind="pool", files=[stored["stored"]], params={})
    assert upload_module.JobStore(root)._ingest(job) == 0
    assert job.ingest_note() == "（视频 1 个抽出 0 帧、进池 0 帧）"


def test_library_index_records_the_readable_video_name(monkeypatch, tmp_path):
    """库索引里的来源也要是可读名：复核页那行「来源视频」以前是临时目录全路径。"""
    root = str(tmp_path / "ds")
    init_dataset(root)
    stored = upload_module.save_upload(root, "厨房B面.mp4", b"\x00" * 16)
    monkeypatch.setattr(video, "extract_frames", lambda path, dataset_root, **kwargs: {
        "imported": 1, "reused": 0, "skipped_exact": 0, "skipped_similar": 0, "failed": 0,
        "extracted": 1, "backend": "ffmpeg", "note": "",
        "imported_files": [(path, "帧_0.jpg")]})
    from autolabel.dataset import ensure_lib

    ensure_lib(root, "pose")
    index = load_index(root, "pose")
    index.upsert("帧_0.jpg", width=10, height=10)
    index.save()

    job = upload_module.Job(id="named", kind="video", files=[stored["stored"]], params={})
    upload_module.JobStore(root)._ingest(job)
    assert load_index(root, "pose").get("帧_0.jpg")["video"] == "厨房B面.mp4"


# --------------------------------------------------------------------------- //
# 界面上要看得见
# --------------------------------------------------------------------------- //
def test_settings_payload_reports_the_decoder(monkeypatch):
    from autolabel import settings as system

    monkeypatch.setattr(video, "has_opencv", lambda: False)
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    data = system.payload()
    assert data["video"]["ready"] is False
    assert "opencv-python" in data["video"]["note"]
    assert "视频抽帧" in system.describe()


@pytest.mark.parametrize("page", ["pool.html", "review.html", "objects.html", "detect.html"])
def test_upload_dialog_warns_before_a_video_is_wasted(page):
    """四个上传弹层都要有解码器提示位：视频白传一次就够烦了。"""
    html = read_static(page)
    assert 'data-decoder-note' in html, page
    common = read_static("common.js")
    assert "paintDecoderNotices" in common and "loadDecoderInfo" in common
