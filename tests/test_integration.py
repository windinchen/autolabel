"""需要真实权重的端到端测试；缺权重时自动跳过。"""

import os
import shutil
import subprocess

import pytest

from autolabel.backends import BackendSettings
from autolabel.dataset import import_images, label_path_for, load_index
from autolabel.yolo_pose import parse_label_file, validate_label_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "yolo11n-pose.pt")
SAMPLE = os.path.join(ROOT, "models", "bus.jpg")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(WEIGHTS) and os.path.exists(SAMPLE)),
    reason="缺少 models/yolo11n-pose.pt 或 models/bus.jpg，跳过端到端测试",
)


@pytest.fixture()
def real_dataset(tmp_path):
    from autolabel.dataset import init_dataset

    root = str(tmp_path / "real")
    init_dataset(root, name="real")
    # 复用 fixture 目录结构：直接导入真实照片
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy2(SAMPLE, src / "bus.jpg")
    import_images(root, [str(src)])
    return root


def test_end_to_end_pose_labeling(real_dataset):
    from autolabel.pipeline import label_dataset

    settings = BackendSettings(model=WEIGHTS, device="cpu", imgsz=320, flip_tta=False, max_det=20)
    stats = label_dataset(real_dataset, settings, verbose=False)
    assert stats["labeled"] == 1 and stats["failed"] == 0
    # bus.jpg 里有 4 个人
    assert stats["people"] >= 3

    idx = load_index(real_dataset)
    rel, item = next(iter(idx.items.items()))
    assert item["label_signature"] is not None
    persons = parse_label_file(label_path_for(real_dataset, rel), 17)
    assert len(persons) == stats["people"]
    for person in persons:
        assert len(person.keypoints) == 17
        assert 0.0 < person.w <= 1.0 and 0.0 < person.h <= 1.0
        x1, y1, x2, y2 = person.xyxy(item["width"], item["height"])
        visible = person.keypoints_xyxy(item["width"], item["height"])
        if visible:
            assert x1 <= max(p[0] for p in visible) and x2 >= min(p[0] for p in visible)
    assert not validate_label_file(label_path_for(real_dataset, rel), item["width"], item["height"], 17)


def test_labeling_is_resumable_and_invalidates_on_param_change(real_dataset):
    from autolabel.pipeline import label_dataset

    settings = BackendSettings(model=WEIGHTS, device="cpu", imgsz=320, flip_tta=False)
    first = label_dataset(real_dataset, settings, verbose=False)
    assert first["labeled"] == 1
    again = label_dataset(real_dataset, settings, verbose=False)
    assert again["labeled"] == 0  # 参数未变 -> 断点续跑跳过
    changed = label_dataset(real_dataset, BackendSettings(model=WEIGHTS, device="cpu", imgsz=416,
                                                         flip_tta=False), verbose=False)
    assert changed["labeled"] == 1  # imgsz 变化 -> 需要重标


def test_review_then_export_roundtrip(real_dataset):
    from autolabel.export import export_dataset
    from autolabel.pipeline import label_dataset
    from autolabel.review import ReviewStore

    label_dataset(real_dataset, BackendSettings(model=WEIGHTS, device="cpu", imgsz=320,
                                                flip_tta=False), verbose=False)
    rel = next(iter(load_index(real_dataset).items))
    store = ReviewStore(real_dataset)
    payload = store.read_item(rel)["objects"]
    payload[0]["keypoints"][3]["v"] = 0
    store.write_item(rel, payload, status="approved")
    result = export_dataset(real_dataset, out=os.path.join(real_dataset, "exports", "it"),
                            val_ratio=0.0, mode="copy", coco=True)
    assert result["train_images"] == 1
    exported = os.path.join(result["out"], "labels", "train", "bus.txt")
    persons = parse_label_file(exported, 17)
    assert persons[0].keypoints[3].v == 0


def test_auto_command_runs_import_label_qc_coords(tmp_path):
    """一条命令跑完：图片入库 → 标注 → QC → 坐标文件。"""
    import csv

    from autolabel.cli import main

    root = str(tmp_path / "auto")
    code = main(["auto", root, SAMPLE, "--model", WEIGHTS, "--device", "cpu", "--imgsz", "320"])
    assert code == 0
    coord_csv = os.path.join(root, "coords")
    tag = sorted(os.listdir(coord_csv))[0]
    with open(os.path.join(coord_csv, tag, "coords.csv"), "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 3 and rows[0]["image"] == "bus.jpg"
    assert os.path.exists(os.path.join(root, "qc", "report.md"))


def test_auto_command_accepts_video_and_keeps_timestamps(tmp_path):
    """视频走抽帧分支，坐标文件里要带 time_sec / frame_index。"""
    import csv

    if not shutil.which("ffmpeg"):
        pytest.skip("缺少 ffmpeg，跳过视频用例")
    from autolabel.cli import main

    clip = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", "3", "-i", SAMPLE,
                    "-vf", "fps=1,scale=480:-2", "-pix_fmt", "yuv420p", clip], check=True)
    root = str(tmp_path / "autovideo")
    code = main(["auto", root, clip, "--video-dedupe", "none", "--model", WEIGHTS,
                 "--device", "cpu", "--imgsz", "320"])
    assert code == 0
    tag = sorted(os.listdir(os.path.join(root, "coords")))[0]
    with open(os.path.join(root, "coords", tag, "coords.csv"), "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 2
    assert all(r["video"].endswith("clip.mp4") for r in rows)
    times = {r["time_sec"] for r in rows}
    assert "0.0" in times and len(times) >= 2, "多帧要各自带上时间点"
    assert {r["frame_index"] for r in rows} == {str(i) for i in range(len(rows) // 3)}
