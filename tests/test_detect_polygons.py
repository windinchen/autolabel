"""分割权重（``-seg`` / ``yoloe-*-seg``）的轮廓落盘。

v1.4.5 之前分割权重出的轮廓只记一个 ``has_mask`` 布尔，形状当场丢掉，标签里只有
方框——分割权重跑完只能当检测权重用。现在轮廓按**归一化点列**接在 5 列方框后面，
导出训练集时才换成 YOLO 标准 segment 行。这里守住四件事：

* mask 解析失败**只丢轮廓**，不能把整张图的框带走（轮廓是加分项，不是前提）；
* 轮廓与方框对不上（上游改了 mask 的尺寸口径）就当没轮廓，别把灰边缩略图当形状；
* 复核页只回传方框，保存时轮廓必须按框认领回来，否则点一次「接受」形状就没了；
* 一条轮廓都没有时，导出结果与老版本一致（5 列 + ``task: detect``）。

推理一律用替身：真跑分割权重要下几百 MB 权重，而这里要验的是形状怎么传下去。
"""
import csv
import json
import os
import types

import numpy as np
import pytest

from autolabel import detect as dt
from autolabel import fields as fd
from autolabel import prompt as pr
from autolabel import qc
from autolabel.backends import mask_polygons
from autolabel.dataset import adopt_from_pool, load_index
from autolabel.library import LIB_DETECT, LIB_POSE
from autolabel.review import ReviewStore

from test_detect import FakeBackend


@pytest.fixture()
def seg_ds(dataset):
    """两帧素材同时进目标检测库（与 test_detect 的夹具同一做法）。"""
    rels = [rel for rel, _ in load_index(dataset, LIB_POSE).ordered_items()]
    adopt_from_pool(dataset, rels, lib=LIB_DETECT)
    return dataset


def first_rel(root):
    return [rel for rel, _ in load_index(root, LIB_DETECT).ordered_items()][0]


def blob(shape=(100, 160), box=(20, 10, 60, 90)):
    """一张合成 mask（``masks.data`` 是 ``(N, H, W)``，单张也要带第 0 维）。"""
    mask = np.zeros(shape, dtype="float32")
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1.0
    return mask


def bbox(points):
    xs = [points[i] for i in range(0, len(points), 2)]
    ys = [points[i] for i in range(1, len(points), 2)]
    return min(xs), min(ys), max(xs), max(ys)


class Result:
    """最像 ultralytics 结果的替身：只有 ``masks.data`` 这一件事是真的。"""

    def __init__(self, masks):
        self.masks = types.SimpleNamespace(data=masks)


# --------------------------------------------------------------------------- #
# mask → 轮廓
# --------------------------------------------------------------------------- #
def test_mask_polygons_returns_normalized_outline():
    """整块 mask 要能变成归一化点列，外接范围与画的那块方框对得上。"""
    polygons = mask_polygons(Result(blob()[None]), [0], [(0.25, 0.5, 0.25, 0.8)])
    assert len(polygons) == 1
    points = polygons[0]
    assert len(points) >= 6 and len(points) % 2 == 0
    assert all(0.0 <= value <= 1.0 for value in points)
    x1, y1, x2, y2 = bbox(points)
    assert abs(x1 - 0.125) < 0.03 and abs(x2 - 0.375) < 0.03
    assert abs(y1 - 0.1) < 0.03 and abs(y2 - 0.9) < 0.03


def test_mask_polygons_aligns_with_the_boxes_it_was_given():
    """第几个框拿第几个轮廓：错位比没轮廓更糟（把人的轮廓套到车上）。"""
    masks = np.stack([blob(box=(100, 20, 150, 95)), blob(box=(10, 10, 60, 90))])
    right, left = (0.78125, 0.575, 0.3125, 0.75), (0.21875, 0.5, 0.3125, 0.8)
    polygons = mask_polygons(Result(masks), [0, 1], [right, left])
    assert len(polygons[0]) >= 6 and len(polygons[1]) >= 6
    assert bbox(polygons[0])[0] > bbox(polygons[1])[2], "第 0 个轮廓应该在画面右边"
    # 框和轮廓配错对时不是"歪一点"，是两个都别给：宁可只留方框
    assert mask_polygons(Result(masks), [0, 1], [left, right]) == [[], []]


def test_mask_polygons_drops_an_outline_that_disagrees_with_the_box():
    """轮廓和方框不是一回事（比如 mask 还留在带灰边的推理尺寸上）就整块丢掉。"""
    polygons = mask_polygons(Result(blob()[None]), [0], [(0.9, 0.5, 0.15, 0.6)])
    assert polygons == [[]]


@pytest.mark.parametrize("masks", [
    None,                                        # 普通检测权重：压根没有 masks
    object(),                                    # 认不出的对象
    np.zeros((100, 160), dtype="float32"),       # 少了一维
    np.zeros((2, 100, 160), dtype="float32"),    # 有 mask 但整张是空的
])
def test_mask_polygons_never_takes_the_boxes_down_with_it(masks):
    """解析不动就交白卷：分割是加分项，不能因为没轮廓把这一张的框全扔掉。"""
    polygons = mask_polygons(Result(masks), [0, 1], [(0.5, 0.5, 0.2, 0.2)] * 2)
    assert polygons == [[], []]


def test_predict_attaches_polygons_and_keeps_the_boxes():
    """走一遍真实 :meth:`predict`：框一个不少，能对上框的轮廓才贴上去。"""
    from autolabel.backends import BackendSettings, DetectionBackend

    class Boxes:
        cls = np.array([2, 5])
        conf = np.array([0.91, 0.62])
        xyxy = np.array([[20.0, 10.0, 60.0, 90.0], [100.0, 20.0, 150.0, 95.0]])

        def __len__(self):
            return len(self.cls)

    class Model:
        def predict(self, image, **kwargs):
            masks = np.stack([blob(box=(20, 10, 60, 90)), blob(box=(0, 0, 8, 8))])
            return [types.SimpleNamespace(orig_shape=(100, 160, 3), boxes=Boxes(),
                                          masks=types.SimpleNamespace(data=masks), obb=None)]

    backend = DetectionBackend(BackendSettings(model="yolov8x-seg.pt", device="cpu"),
                              "yolov8x-seg.pt")
    backend._model = Model()
    backend.task = "segment"
    backend.names = {2: "car", 5: "bus"}
    backend._device = "cpu"
    backend._half = False
    found = backend.predict(np.zeros((100, 160, 3), dtype="uint8"), class_ids=[2, 5])
    assert [row.class_id for row in found] == [2, 5]
    assert all(row.has_mask for row in found)
    assert len(found[0].poly) >= 6, "轮廓对得上却没贴上，等于白跑一次分割"
    assert found[1].poly == [], "这块 mask 在画面左上角，跟第 2 号方框根本不是一个目标"


# --------------------------------------------------------------------------- #
# 标签文件：5 列方框 + 轮廓尾列
# --------------------------------------------------------------------------- #
def test_label_file_roundtrip_keeps_the_polygon(seg_ds):
    rel = first_rel(seg_ds)
    fd.upsert_field(seg_ds, name="car", label="车", lib=LIB_DETECT)
    pr.write_detections(seg_ds, rel, [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.4, h=0.6,
                                                   poly=[0.3, 0.2, 0.7, 0.25, 0.5, 0.8])],
                        LIB_DETECT)
    back = pr.read_detections(seg_ds, rel, LIB_DETECT)
    assert back[0].poly == [0.3, 0.2, 0.7, 0.25, 0.5, 0.8]
    assert [row[:5] for row in fd.read_label_entries(fd_label(seg_ds, rel))] == [
        (0, 0.5, 0.5, 0.4, 0.6)], "前 5 列必须还是那个方框，老逻辑才读得对"
    assert fd.validate_box_label_file(fd_label(seg_ds, rel)) == [], "带轮廓不该被当成多写的列"


def fd_label(root, rel):
    from autolabel.library import label_path_for

    return label_path_for(root, rel, LIB_DETECT)


def write_raw_label(root, rel, body):
    path = fd_label(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def test_plain_box_label_is_still_exactly_five_columns(seg_ds):
    """没有轮廓时一个字都不许多：这正是老训练脚本能直接吃的那份标签。"""
    line = fd.format_label_row([0, 0.5, 0.5, 0.2, 0.3])
    assert line == "0 0.500000 0.500000 0.200000 0.300000"
    assert fd.format_segment_row([0, 0.3, 0.2, 0.7, 0.25, 0.5, 0.8]).split()[0] == "0"
    assert len(fd.format_segment_row([0, 0.3, 0.2, 0.7, 0.25, 0.5, 0.8]).split()) == 7


@pytest.mark.parametrize("body, code", [
    # 点数不成对：最后一个点只剩 x
    ("0 0.5 0.5 0.2 0.3 0.30 0.20 0.70\n", "bad_polygon"),
    # 只有两个点，连多边形都围不起来
    ("0 0.5 0.5 0.2 0.3 0.30 0.20 0.70 0.40\n", "bad_polygon"),
    # 点跑到图外
    ("0 0.5 0.5 0.2 0.3 0.30 0.20 1.70 0.40 0.50 0.80\n", "poly_range"),
])
def test_broken_polygon_tail_is_reported_not_swallowed(seg_ds, body, code):
    """读取时坏轮廓静默当没轮廓，校验时必须报出来——否则用户永远不知道形状丢了。"""
    issues = fd.validate_box_label_file(write_raw_label(seg_ds, first_rel(seg_ds), body))
    assert [row.kind for row in issues] == [code]


def test_odd_polygon_tail_does_not_lose_the_box(seg_ds):
    """坏尾列只该丢掉轮廓，方框还得读得出来（一个框都不能悄悄少）。"""
    write_raw_label(seg_ds, first_rel(seg_ds), "0 0.5 0.5 0.2 0.3 0.30 0.20 0.70\n")
    assert fd.read_label_rows(fd_label(seg_ds, first_rel(seg_ds))) == [(0, 0.5, 0.5, 0.2, 0.3)]
    assert pr.read_detections(seg_ds, first_rel(seg_ds), LIB_DETECT)[0].poly == []


# --------------------------------------------------------------------------- #
# 复核保存：轮廓跟着框走
# --------------------------------------------------------------------------- #
def test_carry_polygons_moves_the_outline_with_the_box():
    old = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2, poly=[0.4, 0.4, 0.6, 0.4, 0.5, 0.6])
    new = pr.Detection(cls=0, cx=0.6, cy=0.5, w=0.4, h=0.2)
    pr.carry_polygons([new], [old])
    assert new.poly, "框只是挪了个位置，轮廓不该就这么没了"
    # 老框横跨 0.4~0.6，新框横跨 0.4~0.8：轮廓要按新框拉宽，而不是留在原地
    assert new.poly[0] == pytest.approx(0.4) and new.poly[1] == pytest.approx(0.4)
    assert new.poly[2] == pytest.approx(0.8)
    assert new.poly[4] == pytest.approx(0.6), "框变宽一倍，轮廓要跟着拉宽"


def test_carry_polygons_matches_every_box_when_positions_swap():
    """两个框交换位置：轮廓要各跟各的主人，不能被彼此顶掉一个。"""
    def box(cx):
        return pr.Detection(cls=0, cx=cx, cy=0.5, w=0.2, h=0.2)

    left, right = box(0.2), box(0.6)
    left.poly = [0.1, 0.4, 0.3, 0.4, 0.2, 0.6]
    right.poly = [0.5, 0.4, 0.7, 0.4, 0.6, 0.6]
    swapped = [box(0.6), box(0.2)]
    pr.carry_polygons(swapped, [left, right])
    assert swapped[0].poly and swapped[1].poly, "换位之后不该有框丢掉轮廓"
    assert swapped[0].poly[0] == pytest.approx(0.5), "右边的框该拿右边那份轮廓"
    assert swapped[1].poly[0] == pytest.approx(0.1)


def test_carry_polygons_leaves_manual_and_unmatched_boxes_alone():
    manual = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2, flags=["manual"])
    elsewhere = pr.Detection(cls=0, cx=0.05, cy=0.05, w=0.06, h=0.06)
    old = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2, poly=[0.4, 0.4, 0.6, 0.4, 0.5, 0.6])
    pr.carry_polygons([manual, elsewhere], [old])
    assert manual.poly == [], "人工画的方框硬套一个轮廓是编数据"
    assert elsewhere.poly == [], "另一个目标隔得老远，轮廓不是它的"


def test_saving_from_the_review_page_keeps_the_polygon(seg_ds):
    """复核页只回传方框（画框的 UI 不编辑轮廓），保存时得把轮廓认回来。"""
    rel = first_rel(seg_ds)
    fd.upsert_field(seg_ds, name="car", label="车", lib=LIB_DETECT)
    pr.write_detections(seg_ds, rel, [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.4, h=0.6,
                                                   score=0.9,
                                                   poly=[0.3, 0.2, 0.7, 0.25, 0.5, 0.8],
                                                   flags=["masked"])], LIB_DETECT)
    store = ReviewStore(seg_ds)
    store.write_prompt_item(rel, [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.6}],
                           status="approved", lib=LIB_DETECT)
    rows = fd.read_label_entries(fd_label(seg_ds, rel))
    assert len(rows) == 1 and len(rows[0][5]) == 6, "点了一次接受，轮廓没了"
    saved = store.read_prompt_item(rel, LIB_DETECT)
    assert "masked" in saved["flags"], "轮廓还在，却不再说明这框出自分割权重"
    assert "masked" in saved["objects"][0]["flags"]
    assert store.read_prompt_item(rel, LIB_DETECT)["objects"][0]["poly"] == [
        [round(0.3 * 640, 2), round(0.2 * 480, 2)], [round(0.7 * 640, 2), round(0.25 * 480, 2)],
        [round(0.5 * 640, 2), round(0.8 * 480, 2)]]


# --------------------------------------------------------------------------- #
# 导出方框 / 分割两种口径
# --------------------------------------------------------------------------- #
class SegmentBackend(FakeBackend):
    """分割权重替身：每个框都带一个三角形轮廓。"""

    def predict(self, image, class_ids=(), conf=None):
        found = super().predict(image, class_ids=class_ids, conf=conf)
        for item in found:
            item.poly = [item.cx - 0.09, item.cy - 0.14, item.cx + 0.09, item.cy - 0.12,
                         item.cx, item.cy + 0.14]
        return found


def test_export_writes_segment_rows_when_there_are_outlines(seg_ds, monkeypatch):
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: SegmentBackend(
        model="yolov8x-seg.pt"))
    dt.add_model_classes(seg_ds, "yolov8x-seg.pt", ["person"])
    dt.run_detect_labeling(seg_ds, verbose=False)
    out = dt.export_detect_dataset(seg_ds, out=os.path.join(seg_ds, "exports", "seg"))

    lines = []
    for split in ("train", "val"):
        base = os.path.join(out["out"], "labels", split)
        for name in sorted(os.listdir(base)):
            lines += [row.split() for row in open(os.path.join(base, name),
                                                  encoding="utf-8").read().splitlines() if row.strip()]
    assert lines, "导出集是空的"
    assert all(len(row) >= 7 and (len(row) - 1) % 2 == 0 for row in lines), \
        "有轮廓的框要写成 class x1 y1 ... 的标准 segment 行"
    assert out["counts"]["segment_boxes"] == len(lines) and out["counts"]["box_only"] == 0
    assert out["task"] == "segment"

    assert "分割" in out["segment_note"] or "segment" in out["segment_note"]
    with open(out["data_yaml"], encoding="utf-8") as fh:
        text = fh.read()
    assert "task: segment" in text and "YOLO segment" in text


def test_export_without_outlines_is_unchanged(seg_ds, monkeypatch):
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: FakeBackend())
    dt.add_model_classes(seg_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(seg_ds, verbose=False)
    out = dt.export_detect_dataset(seg_ds, out=os.path.join(seg_ds, "exports", "det"))
    assert out["task"] == "detect" and out["segment_note"] == ""
    assert out["counts"]["segment_boxes"] == 0
    for split in ("train", "val"):
        base = os.path.join(out["out"], "labels", split)
        for name in sorted(os.listdir(base)):
            for row in open(os.path.join(base, name), encoding="utf-8").read().splitlines():
                if row.strip():
                    assert len(row.split()) == 5, "没有轮廓还写多出列，老的 detection 训练会读错位"
    with open(out["data_yaml"], encoding="utf-8") as fh:
        assert "task: detect" in fh.read()


def test_mixed_export_labels_both_kinds(seg_ds, monkeypatch):
    """一半有轮廓一半没有：两种行都能进分割训练，但口径必须在结果里说清。"""
    class HalfBackend(SegmentBackend):
        def predict(self, image, class_ids=(), conf=None):
            found = super().predict(image, class_ids=class_ids, conf=conf)
            for position, item in enumerate(found):
                if position % 2:
                    item.poly = []
            return found

    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: HalfBackend())
    dt.add_model_classes(seg_ds, "yolov8x-seg.pt", ["person", "bicycle"])
    dt.run_detect_labeling(seg_ds, verbose=False)
    out = dt.export_detect_dataset(seg_ds, out=os.path.join(seg_ds, "exports", "mix"))
    assert out["counts"]["segment_boxes"] and out["counts"]["box_only"]
    assert "另" in out["segment_note"]
    widths = set()
    for split in ("train", "val"):
        base = os.path.join(out["out"], "labels", split)
        for name in sorted(os.listdir(base)):
            for row in open(os.path.join(base, name), encoding="utf-8").read().splitlines():
                if row.strip():
                    widths.add(len(row.split()))
    assert widths == {5, 7}


def test_reusing_an_export_still_reports_segment(seg_ds, monkeypatch):
    """内容没变时第二次导出直接复用旧目录，复用那条路也得报 segment。

    复用分支返回的是从 ``yolo_meta.json`` 现拼的字典，漏一个字段界面上就变成
    "同一份训练集，第一次说分割、第二次说检测"。
    """
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: SegmentBackend())
    dt.add_model_classes(seg_ds, "yolov8x-seg.pt", ["person"])
    dt.run_detect_labeling(seg_ds, verbose=False)
    first = dt.export_detect_dataset(seg_ds)
    again = dt.export_detect_dataset(seg_ds)
    assert again.get("reused") and again["out"] == first["out"]
    assert again["task"] == "segment" and again["segment_note"]

    # 老目录的 meta 里没有 task 这一项：按框数倒推，不能默认成 detection
    meta_path = os.path.join(again["out"], "yolo_meta.json")
    meta = json.loads(open(meta_path, encoding="utf-8").read())
    meta.pop("task", None)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    assert dt.export_detect_dataset(seg_ds)["task"] == "segment"


# --------------------------------------------------------------------------- #
# 口径：轮廓已落盘之后，标记和坐标表怎么说
# --------------------------------------------------------------------------- #
def test_masked_is_no_longer_a_pending_review_reason():
    """``masked`` 现在只是「出自分割权重、轮廓已落盘」，不是缺陷，不该催用户复核。"""
    item = {"detect_status": "unreviewed", "detect_objects": 2, "detect_flags": ["masked"]}
    assert not qc.needs_review(item, LIB_DETECT)
    assert qc.needs_review({**item, "detect_flags": ["small_box"]}, LIB_DETECT)


def test_coordinate_table_counts_outline_points(seg_ds, monkeypatch):
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: SegmentBackend())
    dt.add_model_classes(seg_ds, "yolov8x-seg.pt", ["person"])
    dt.run_detect_labeling(seg_ds, verbose=False)
    body = dt.render_detect_coords(seg_ds, fmt="csv")[1].decode("utf-8")
    rows = list(csv.DictReader(body.splitlines()))
    assert rows and all(int(row["poly_points"]) == 3 for row in rows)
