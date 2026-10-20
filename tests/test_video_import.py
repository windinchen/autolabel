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
def _fake_run_opencv(box, count: int = 1):
    def run(clip, tmp, stem, fps, start, duration, max_side, max_frames, info):
        box.update({"fps": fps, "start": start, "duration": duration,
                    "max_frames": max_frames, "max_side": max_side})
        entries = []
        for i in range(count):
            name = f"{i:06d}.jpg"
            Image.new("RGB", (32, 24)).save(os.path.join(tmp, name))
            entries.append((os.path.join(tmp, name), f"{stem}_t{i * 1000}",
                            {"video": clip, "time_sec": float(i), "frame_index": i,
                             "video_fps": 25.0}))
        return entries, count, False
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
    # 0.5 是**频率**不是间隔，以前提示写成「每 0.5 秒取 1 帧」，倍数正好说反
    assert "按镜头切换取样用不了" in note and "每 2 秒 1 帧" in note


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
    # 抽出来的帧本来就是挑过的，默认不能再来一次近似去重
    assert captured["kwargs"]["dedupe"] == "exact"


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
    assert job.ingest_note() == "（视频 1 个抽出 3 帧、进池 3 帧，取样每 2 秒 1 帧）"


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
    # 没填频率就没有口径可报，但抽了几帧还得说
    assert job.ingest_note() == "（视频 1 个抽出 0 帧、进池 0 帧）"

    rated = upload_module.Job(id="rated", kind="pool", files=[], params={"fps": 3.0})
    rated.videos = [{"video": "a.mp4", "extracted": 0, "imported": 0, "note": ""}]
    assert "取样每秒 3 帧" in rated.ingest_note()


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


# --------------------------------------------------------------------------- //
# 张数要能自证：取样口径、最多帧数截断、去重吃掉多少
# --------------------------------------------------------------------------- //
def test_sampling_lands_on_the_requested_grid(clip, monkeypatch):
    """「每秒 2 帧」就得到 时长 × 2 帧，用户自己乘一遍就能核对。

    以前步长取整数：25 fps 的视频 ``round(25/2) = 12``，实际间隔变成 0.48 秒，
    6 秒的片子出 13 帧而不是 12 帧——多抽的没人抱怨，少抽的才被发现，
    但根子是同一个：取样点没落在请求的格子上。
    """
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    tmp = os.path.join(os.path.dirname(clip), "grid")
    os.makedirs(tmp, exist_ok=True)

    entries, extracted, _timed_out = video._run_opencv(
        clip, tmp, "busy", 2.0, 0.0, 0.0, 1600, 600, video.probe_video(clip))
    stamps = [row[2]["time_sec"] for row in entries]
    assert extracted == 12, f"6 秒 × 每秒 2 帧该是 12 帧，实际 {extracted} 帧"
    assert stamps[0] == 0.0
    gaps = [round(b - a, 2) for a, b in zip(stamps, stamps[1:])]
    assert set(gaps) <= {0.48, 0.52}, f"取样点没排在 0.5 秒的格子上：{gaps}"


def test_a_capped_run_says_the_uncapped_number(tmp_path, monkeypatch):
    """被「最多帧数」截断要报出整段本来该抽几帧：否则用户乘出来的数永远比实际大。"""
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: True)
    monkeypatch.setattr(video, "_run_opencv", _fake_run_opencv({}, count=5))
    monkeypatch.setattr(video, "probe_video",
                        lambda path: {"duration": 60.0, "fps": 25.0, "backend": "opencv"})
    root = str(tmp_path / "ds")
    init_dataset(root)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)

    result = video.extract_frames(str(clip), root, fps=1.0, max_frames=5)
    assert result["extracted"] == 5
    assert "整段该抽 61 帧" in result["note"], result["note"]
    assert "截到 5 帧" in result["note"], result["note"]


def test_an_uncapped_run_does_not_claim_a_truncation(tmp_path, monkeypatch):
    """没到上限就别硬说「已截断」：说错一次的解释比不解释更让人怀疑别的数字。"""
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    monkeypatch.setattr(video, "has_opencv", lambda: True)
    monkeypatch.setattr(video, "_run_opencv", _fake_run_opencv({}, count=3))
    monkeypatch.setattr(video, "probe_video",
                        lambda path: {"duration": 8.0, "fps": 25.0, "backend": "opencv"})
    root = str(tmp_path / "ds")
    init_dataset(root)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 8)

    result = video.extract_frames(str(clip), root, fps=0.5, max_frames=600)
    assert "最多帧数" not in result["note"], result["note"]


def test_video_frames_are_not_deduped_by_sight_anymore(monkeypatch, tmp_path):
    """抽帧默认只跳过完全相同的帧：画面相近就丢，同场景的录像会互相吃掉。

    实测 5 条同机位录像抽出 264 帧，按旧的「近似去重」只剩 46 帧，
    而提示写成「与已有素材内容相同」——用户知道自己没重传文件，只当软件在编理由。
    """
    root = str(tmp_path / "ds")
    init_dataset(root)
    stored = upload_module.save_upload(root, "a.mp4", b"\x00" * 16)
    seen = {}

    def fake_extract(path, dataset_root, **kwargs):
        seen["dedupe"] = kwargs.get("dedupe")
        return {"imported": 1, "reused": 0, "skipped_exact": 0, "skipped_similar": 0,
                "failed": 0, "extracted": 1, "backend": "opencv", "note": "",
                "imported_files": [(path, "帧_0.jpg")]}

    monkeypatch.setattr(video, "extract_frames", fake_extract)
    store = upload_module.JobStore(root)

    store._ingest(upload_module.Job(id="plain", kind="pool",
                                    files=[stored["stored"]], params={}))
    assert seen["dedupe"] == "exact", seen

    store._ingest(upload_module.Job(id="asked", kind="pool", files=[stored["stored"]],
                                    params={"video_dedupe": "perceptual"}))
    assert seen["dedupe"] == "perceptual", "用户真要省张数时还得能选近似去重"

    junk = upload_module.Job(id="junk", kind="pool", files=[stored["stored"]],
                             params={"video_dedupe": "looks-like"})
    with pytest.raises(ValueError) as err:
        store._ingest(junk)
    assert "视频抽帧去重方式" in str(err.value)


def test_the_note_names_the_rule_and_calls_similar_frames_what_they_are():
    """完成消息：抽出多少、按什么抽、少了的为什么少，一句话里说完。"""
    job = upload_module.Job(id="note", kind="pool", files=[], params={"fps": 2.0})
    job.videos = [{"video": "现场.mp4", "extracted": 264, "imported": 46, "note": ""}]
    job.counts = {"similar": 218}

    note = job.ingest_note()
    assert "视频 1 个抽出 264 帧、进池 46 帧" in note, note
    assert "取样每秒 2 帧" in note, note
    assert "218 张画面与已有素材接近" in note and "近似去重" in note, note
    assert "内容相同" not in note, "近似去重不该说成内容相同"


@pytest.mark.parametrize("page", ["pool.html", "review.html", "objects.html", "detect.html"])
def test_dedupe_choice_is_three_way_and_safe_by_default(page):
    """去重要摊开成三档、默认「只跳过完全相同的帧」：一个勾选框把吃帧藏在默认里。"""
    html = read_static(page)
    assert 'id="up-video-dedupe"' in html, page
    assert '<option value="exact" selected>' in html, page
    assert '<option value="perceptual">' in html, page
    assert '<option value="none">' in html, page
    assert "up-keep-similar" not in html, f"{page} 还留着旧的「保留近重复帧」勾选"


@pytest.mark.parametrize("script", ["app.js", "pool.js", "objects.js"])
def test_upload_scripts_send_the_chosen_dedupe(script):
    assert 'readSelect("up-video-dedupe", "exact")' in read_static(script), script


# --------------------------------------------------------------------------- //
# 命令行：抽了多少、按什么口径、少掉的谁吃掉的
# --------------------------------------------------------------------------- //
def test_video_command_accepts_the_flag_name_auto_uses():
    """`auto` 里的抽帧去重叫 `--video-dedupe`，`video` 里以前只认 `--dedupe`——
    文档里那句 `--fps 2 --video-dedupe none` 照抄进 `video` 会被 argparse 拒成退出码 2。
    同一个参数得两个写法都认，`--dedupe` 留着不破坏老脚本。"""
    from autolabel.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["video", "ds", "clip.mp4"]).dedupe == "exact"
    assert parser.parse_args(
        ["video", "ds", "clip.mp4", "--video-dedupe", "perceptual"]).dedupe == "perceptual"
    assert parser.parse_args(["video", "ds", "clip.mp4", "--dedupe", "none"]).dedupe == "none"


def test_frame_summary_names_the_rule_and_splits_the_two_skips():
    """少掉的帧要说清是谁吃掉的：近似去重和「最多帧数」截断，补救办法完全不同。"""
    from autolabel.cli import _frame_summary

    lines = _frame_summary({"extracted": 43, "mode": "fps", "skipped_exact": 2,
                            "skipped_similar": 7, "reused": 3,
                            "note": "整段该抽 61 帧，已按「最多帧数」截到 43 帧（要全抽就调大上限）"},
                           2.0, ingested=36)
    assert lines[0].startswith("抽出 43 帧（fps，每秒 2 帧）→ 入库 36 帧"), lines[0]
    assert "近似去重 7" in lines[0] and "内容相同 2" in lines[0]
    assert "复用素材池 3" in lines[0]
    assert lines[1].startswith("整段该抽 61 帧"), lines

    scene = _frame_summary({"extracted": 9, "mode": "scene"}, 0.5, ingested=9)
    assert scene == ["抽出 9 帧（scene）→ 入库 9 帧"], "按镜头切点取样没有固定频率可报"


def test_video_command_prints_the_cap_that_ate_the_frames(tmp_path, monkeypatch, capsys):
    """截断提示以前只待在 ``result["note"]`` 里，命令行用户永远看不到——
    他只看到「抽出 5 帧」，跟视频时长乘频率乘出来的 44 帧对不上。"""
    from autolabel.cli import main

    root = str(tmp_path / "ds")
    init_dataset(root)
    monkeypatch.setattr(video, "extract_frames", lambda *a, **kw: {
        "imported": 5, "reused": 0, "skipped_exact": 0, "skipped_similar": 0, "failed": 0,
        "imported_files": [(None, f"images/f{i}.jpg") for i in range(5)],
        "extracted": 5, "mode": "fps", "backend": "opencv",
        "video_info": {"duration": 21.5, "width": 2560, "height": 1440, "fps": 25.0},
        "note": "整段该抽 44 帧，已按「最多帧数」截到 5 帧（要全抽就调大上限）"})

    assert main(["video", root, str(tmp_path / "clip.mp4"), "--fps", "2", "--max-frames", "5"]) == 0
    out = capsys.readouterr().out
    assert "每秒 2 帧" in out, out
    assert "整段该抽 44 帧" in out, out
