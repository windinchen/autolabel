"""坐标文件导出：CSV / 长表 / JSONL 的字段与数值一致性。"""

import csv
import json
import os

from _helpers import idx_names, write_label
from autolabel import dataset as ds
from autolabel import export as ex
from autolabel.coords import wide_columns, write_coordinate_files

# 一个人：bbox 归一化 cxcywh + 17 个关键点（含 v=0/1/2 三种可见性）
SAMPLE = ("0 0.500000 0.500000 0.200000 0.400000 " +
          " ".join(f"{0.30 + i * 0.02:.6f} {0.20 + i * 0.03:.6f} {[0, 1, 2][i % 3]}"
                   for i in range(17)))


def seed(dataset, statuses=None):
    names = sorted(idx_names(dataset))
    for rel in names:
        write_label(dataset, rel, [SAMPLE])
    idx = ds.load_index(dataset)
    for i, rel in enumerate(names):
        item = idx.get(rel)
        item.update(objects=1, score=0.77, width=640 if i == 0 else 400,
                    height=480 if i == 0 else 300)
        if statuses and statuses.get(rel):
            item["status"] = statuses[rel]
    idx.save()
    return names


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_wide_csv_one_row_per_person_with_pixel_coords(dataset):
    names = seed(dataset)
    result = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t1"))
    rows = read_rows(os.path.join(result["out"], "coords.csv"))

    assert len(rows) == 2  # 一行一个人
    row = next(r for r in rows if r["image"] == names[0])
    assert row["status"] == "unreviewed" and row["class_name"] == "person"
    assert row["image_width"] == "640" and row["person_index"] == "0"
    # bbox：cx=0.5,cy=0.5,w=0.2,h=0.4 -> 像素 xyxy
    assert row["bbox_x1"] == "256.0" and row["bbox_y1"] == "144.0"
    assert row["bbox_x2"] == "384.0" and row["bbox_y2"] == "336.0"
    assert row["bbox_cx"] == "0.5" and row["bbox_h"] == "0.4"
    # 关键点 0（v=0）：x=0.30*640=192, y=0.20*480=96
    assert row["kp01_nose_x"] == "192.0" and row["kp01_nose_y"] == "96.0"
    assert row["kp01_nose_v"] == "0" and row["kp03_right_eye_v"] == "2"
    assert len(result["meta"]["keypoint_names"]) == 17
    assert wide_columns(result["meta"]["keypoint_names"])[0] == "image"


def test_long_csv_has_one_row_per_keypoint(dataset):
    seed(dataset)
    result = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t2"),
                                    formats=("long",))
    rows = read_rows(os.path.join(result["out"], "coords_long.csv"))
    assert len(rows) == 2 * 17
    assert {r["keypoint_name"] for r in rows} >= {"nose", "left_shoulder", "right_ankle"}
    assert all(len(r["x"]) and len(r["y"]) for r in rows), "未标注点也要给出坐标"


def test_jsonl_keeps_pixel_and_normalized_in_sync(dataset):
    names = seed(dataset)
    result = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t3"),
                                    formats=("jsonl",))
    with open(os.path.join(result["out"], "coords.jsonl"), "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert [r["image"] for r in records] == names
    rec = records[0]
    assert rec["width"] == 640 and len(rec["objects"][0]["keypoints"]) == 17
    for kp in rec["objects"][0]["keypoints"]:
        assert kp["x"] == round(kp["x_norm"] * rec["width"], 2)
        assert kp["y"] == round(kp["y_norm"] * rec["height"], 2)
    meta = json.load(open(os.path.join(result["out"], "meta.json"), "r", encoding="utf-8"))
    assert meta["num_keypoints"] == 17 and meta["files"]["coords.jsonl"] == "coords.jsonl"


def test_video_frame_metadata_is_carried_into_files(dataset):
    names = seed(dataset)
    idx = ds.load_index(dataset)
    idx.get(names[0]).update(video="/tmp/walk.mp4", time_sec=12.5, frame_index=7)
    idx.save()
    result = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t4"))
    rows = read_rows(os.path.join(result["out"], "coords.csv"))
    video_row = next(r for r in rows if r["image"] == names[0])
    still_row = next(r for r in rows if r["image"] == names[1])
    assert video_row["video"] == "/tmp/walk.mp4"
    assert video_row["time_sec"] == "12.5" and video_row["frame_index"] == "7"
    assert still_row["video"] == "" and still_row["time_sec"] == ""


def test_rejected_images_excluded_unless_requested(dataset):
    all_names = sorted(idx_names(dataset))
    seed(dataset, statuses={all_names[0]: "rejected"})
    rejected = all_names[0]
    excluded = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t5"))
    assert excluded["images"] == 1
    assert excluded["records"][0]["image"] != rejected
    everything = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t6"),
                                        exclude_status=())
    assert everything["images"] == 2


def test_missing_label_skipped_or_kept(dataset):
    names = seed(dataset)
    os.remove(ds.label_path_for(dataset, names[1]))
    skipped = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t7"))
    assert skipped["images"] == 1
    assert skipped["meta"]["skipped"][0]["image"] == names[1]
    kept = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t8"),
                                 require_label=False)
    assert kept["images"] == 2
    assert kept["records"][1]["objects"] == []


def test_export_with_coords_adds_split_column(dataset):
    seed(dataset)
    result = ex.export_dataset(dataset, out=os.path.join(dataset, "exports", "c"),
                              val_ratio=0.5, mode="copy", coords=True)
    coord_dir = result["coords"]
    rows = read_rows(os.path.join(coord_dir, "coords.csv"))
    assert len(rows) == 2 and {r["split"] for r in rows} == {"train", "val"}
    manifest = json.load(open(os.path.join(result["out"], "manifest.json"), "r", encoding="utf-8"))
    assert manifest["coords"] == coord_dir
    assert manifest["coords_counts"] == {"images": 2, "persons": 2}


def test_one_broken_label_skips_only_that_frame(dataset):
    """一张坏标签不能拉着一整批坐标文件陪葬（以前整批不写、manifest 也留不下来）。"""
    from autolabel.coords import collect_records

    names = seed(dataset)
    broken = "0 0.500000 0.500000 0.200000 " + " ".join(
        f"{0.30 + i * 0.02:.6f} {0.20 + i * 0.03:.6f} 2" for i in range(16))   # 少一个关键点
    write_label(dataset, names[0], [broken])

    records, meta = collect_records(dataset)
    kept = {rec["image"] for rec in records}
    assert names[1] in kept, kept                       # 好图照常出记录
    assert names[0] not in kept, kept                   # 坏的那帧被跳，不是抛出去
    reasons = {row["image"]: row["reason"] for row in meta["skipped"]}
    assert names[0] in reasons and "标签解析失败" in reasons[names[0]], reasons


def test_coordinate_files_still_written_when_a_label_is_broken(dataset):
    """跳帧之后文件要照样落地，界面上的「最近导出」才看得见这次到底导了什么。"""
    names = seed(dataset)
    write_label(dataset, names[0], ["0 0.5 0.5"])      # 残行：连 bbox 都不全
    result = write_coordinate_files(dataset, out=os.path.join(dataset, "coords", "broken"))
    for name in ("coords.csv", "coords.jsonl", "summary.json"):
        assert os.path.exists(os.path.join(result["out"], name)), name
    with open(os.path.join(result["out"], "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["images"] == len(names) - 1, summary
    assert any("标签解析失败" in row["reason"] for row in summary["skipped_items"]), summary
