"""导出目录的命名、软链与 flip_idx 回归测试。

第 4 轮合理性检查（导出正确性）实测到三件事：

1. 导出目录是平的（只有 `images/train`、`images/val`），旧代码只取 basename，
   于是 `a/frame_0001.jpg` 和 `b/frame_0001.jpg` 抢同一个文件名——后放的把前面的
   连图片带标签一起盖掉。4 张图导完磁盘上只剩 2 个文件，manifest 与 COCO 里却写 4 张，
   训练时看起来一切正常，只有样本数对不上；
2. `data.yaml` 的 `flip_idx` 直接切 `COCO17_FLIP_IDX[:n]`，14 点模型会写出越界下标；
3. `--mode symlink` 用绝对路径，导出目录一搬家（或换挂载点）整批链接全断。
"""

import csv
import json
import os
import shutil

import pytest
import yaml
from PIL import Image

from _helpers import idx_names, write_label
from autolabel import dataset as ds
from autolabel import export as ex
from autolabel import fields as field_store
from autolabel import prompt as pr
from autolabel import yolo_pose as yp
from autolabel.dataset import import_images, init_dataset, load_index

SAMPLE17 = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 17)
# 同一张图只给 14 个点：声明 17 点的数据集里，这种标签以前会让 COCO 三元组整体错位
SAMPLE14 = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 14)


def _jpeg(path, color, size=(160, 120)):
    Image.new("RGB", size, color).save(path, quality=95)


def _listing(out_root, split, kind="images"):
    folder = os.path.join(out_root, kind, split)
    return sorted(os.listdir(folder))


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture()
def nested_dataset(tmp_path):
    """两个子目录里放着同名素材的数据集——正是旧代码会互相覆盖的那个形状。"""
    root = str(tmp_path / "nested")
    init_dataset(root, name="nested")
    palette = {"a": [(220, 30, 30), (30, 210, 30)], "b": [(30, 40, 220), (220, 210, 30)]}
    for sub, colors in palette.items():
        source = tmp_path / f"src_{sub}"
        source.mkdir()
        for index, color in enumerate(colors, start=1):
            _jpeg(source / f"frame_{index:04d}.jpg", color)
        import_images(root, [str(source)], subdir=sub, dedupe="exact")
    for rel in sorted(idx_names(root)):
        write_label(root, rel, [SAMPLE17])
    assert len(sorted(idx_names(root))) == 4
    return root


def test_nested_same_names_no_longer_eat_each_other(nested_dataset, tmp_path):
    out = os.path.join(str(tmp_path), "exp-flat-name")
    result = ex.export_dataset(nested_dataset, out=out, val_ratio=0.0, mode="symlink")
    images = _listing(out, "train")
    assert images == ["a__frame_0001.jpg", "a__frame_0002.jpg",
                      "b__frame_0001.jpg", "b__frame_0002.jpg"]
    assert _listing(out, "train", kind="labels") == [name.replace(".jpg", ".txt") for name in images]
    assert result["train"] == 4 and result["val"] == 0
    # 软链必须指向真实图片：断链时 getsize 直接抛错，正好当探针
    for name in images:
        assert os.path.getsize(os.path.join(out, "images", "train", name)) > 0
    manifest = _load(os.path.join(out, "manifest.json"))
    listed = [row["image"] for row in manifest["split"]]
    assert len(listed) == 4 and len(set(listed)) == 4


def test_coco_records_follow_the_unique_names(nested_dataset, tmp_path):
    out = os.path.join(str(tmp_path), "exp-coco")
    ex.export_dataset(nested_dataset, out=out, val_ratio=0.0, coco=True)
    coco = _load(os.path.join(out, "coco_keypoints.json"))
    names = [row["file_name"] for row in coco["images"]]
    assert len(names) == 4 and len(set(names)) == 4
    assert set(names) == set(_listing(out, "train"))


def test_flat_dataset_names_are_unchanged(dataset, tmp_path):
    """平铺数据集（绝大多数真实情况）导出名必须一字不变，别为了修 bug 折腾老用户。"""
    for rel in sorted(idx_names(dataset)):
        write_label(dataset, rel, [SAMPLE17])
    out = os.path.join(str(tmp_path), "exp-plain")
    ex.export_dataset(dataset, out=out, val_ratio=0.0, mode="copy")
    assert _listing(out, "train") == ["img0.jpg", "img1.jpg"]
    assert _listing(out, "train", kind="labels") == ["img0.txt", "img1.txt"]


def test_default_symlinks_are_relative_and_survive_a_move(nested_dataset, tmp_path):
    """默认导出到数据集内的 `exports/`：整个数据集搬家后软链仍然连得上。

    绝对路径的软链在「换机器 / 换挂载点 / 改数据集目录名」时全断，只能重导一份；
    相对链接记的是 `../../../images/...`，只要导出目录还在数据集里就跟着一块走。
    """
    out = os.path.join(nested_dataset, "exports", "linked")
    ex.export_dataset(nested_dataset, out=out, val_ratio=0.0, mode="symlink")
    link = os.path.join(out, "images", "train", "a__frame_0001.jpg")
    assert os.path.islink(link)
    assert not os.path.isabs(os.readlink(link))
    relocated = os.path.join(str(tmp_path), "renamed-dataset")
    shutil.move(nested_dataset, relocated)
    with open(os.path.join(relocated, "exports", "linked", "images", "train",
                           "a__frame_0001.jpg"), "rb") as fh:
        assert fh.read(2) == b"\xff\xd8"


@pytest.mark.parametrize("num_kpt", [17, 14, 9, 5, 2, 1, 24, 0])
def test_flip_idx_is_always_in_range(num_kpt):
    """flip_idx 必须是 0..n-1 的一个排列，而且左右配对要对称。"""
    idx = ex.flip_idx_for(num_kpt)
    assert len(idx) == num_kpt
    assert sorted(idx) == list(range(num_kpt))
    assert all(idx[partner] == index for index, partner in enumerate(idx))
    if num_kpt:
        assert max(idx) < num_kpt
    if num_kpt == 17:
        assert idx == list(yp.COCO17_FLIP_IDX)


def test_data_yaml_flip_idx_matches_keypoint_count(nested_dataset, tmp_path):
    """14 点模型以前会往 data.yaml 写 `flip_idx: [..., 14]`，而 14 点里根本没有 14 号点。"""
    idx = load_index(nested_dataset)
    idx.data["kpt_shape"] = [14, 3]
    idx.save()
    out = os.path.join(str(tmp_path), "exp-14pt")
    ex.export_dataset(nested_dataset, out=out, val_ratio=0.0, mode="copy")
    with open(os.path.join(out, "data.yaml"), encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    assert meta["kpt_shape"] == [14, 3]
    assert meta["flip_idx"] == ex.flip_idx_for(14)
    assert len(meta["flip_idx"]) == 14 and max(meta["flip_idx"]) < 14


def test_coco_keypoints_are_padded_to_the_declared_shape(nested_dataset, tmp_path):
    """少给点的标签要补零，否则后面每 keypoints 都整体错位、坐标全是错的。"""
    write_label(nested_dataset, sorted(idx_names(nested_dataset))[0], [SAMPLE14])
    out = os.path.join(str(tmp_path), "exp-pad")
    ex.export_dataset(nested_dataset, out=out, val_ratio=0.0, coco=True)
    coco = _load(os.path.join(out, "coco_keypoints.json"))
    assert coco["annotations"], "至少要有一条注解可查"
    assert {len(ann["keypoints"]) for ann in coco["annotations"]} == {51}
    assert {ann["num_keypoints"] for ann in coco["annotations"]} == {14, 17}


def test_coords_split_column_matches_export_names(nested_dataset, tmp_path):
    """坐标文件的 split 是按导出名对上的，两边命名规则必须一致。"""
    out = os.path.join(str(tmp_path), "exp-coords")
    result = ex.export_dataset(nested_dataset, out=out, val_ratio=0.5, coords=True, mode="copy")
    with open(os.path.join(result["coords"], "coords.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"train", "val"}
    exported = set(_listing(out, "train")) | set(_listing(out, "val"))
    assert {ex.export_stem(row["image"]) + ".jpg" for row in rows} == exported


def test_detection_export_also_avoids_name_collisions(tmp_path):
    """提示词标注的导出走的是同一套命名，别只修姿态那一半。"""
    root = str(tmp_path / "objects")
    init_dataset(root, name="objects")
    for sub, color in (("a", (200, 40, 40)), ("b", (40, 40, 200))):
        source = tmp_path / f"detect_src_{sub}"
        source.mkdir()
        _jpeg(source / "frame_0001.jpg", color)
        import_images(root, [str(source)], subdir=sub, lib=ds.LIB_OBJECTS)
    registered = field_store.upsert_field(root, name="bottle", label="瓶子")
    class_id = int(registered["class_id"])
    index = load_index(root, ds.LIB_OBJECTS)
    for rel in ("a/frame_0001.jpg", "b/frame_0001.jpg"):
        pr.write_detections(root, rel, [pr.Detection(cls=class_id, cx=0.5, cy=0.5, w=0.2, h=0.3)])
        index.upsert(rel, prompt_objects=1, prompt_status=ds.STATUS_UNREVIEWED)
    index.save()
    out = os.path.join(str(tmp_path), "exp-detect")
    result = pr.export_detection_dataset(root, out=out, val_ratio=0.0)
    images = sorted(set(_listing(out, "train")) | set(_listing(out, "val")))
    assert images == ["a__frame_0001.jpg", "b__frame_0001.jpg"]
    assert result["counts"]["train"] + result["counts"]["val"] == 2


# --- 第 7 轮体检（导出与训练可用性）实测到的三件事 ---

@pytest.mark.parametrize("name", ["安全帽: 大", "背带裤", "true", "no", "3d", "a#b", "", "null"])
def test_pose_data_yaml_survives_awkward_class_names(dataset, tmp_path, name):
    """类名里带 `:` 会让整份 data.yaml 解析失败，训练一帧都跑不起来。

    姿态库以前把类名原样拼进 YAML（``1: 安全帽: 大``），``yaml.safe_load`` 直接
    报 "mapping values are not allowed here"；``true``/``no`` 这类还会被 YAML
    悄悄变成布尔值，训练时的类别名就成了 True。
    """
    for rel in sorted(idx_names(dataset)):
        write_label(dataset, rel, [SAMPLE17])
    idx = load_index(dataset)
    idx.data["class_names"] = {"0": name}
    idx.save()
    out = os.path.join(str(tmp_path), "exp-odd-name")
    ex.export_dataset(dataset, out=out, val_ratio=0.0, mode="copy")
    with open(os.path.join(out, "data.yaml"), encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)          # 解析不了在这里就炸
    assert meta["names"] == {0: name}


def test_yaml_scalar_only_quotes_what_yaml_would_misread():
    """纯标识符保持明文（人和 diff 都好读），会被改含义的才加引号。"""
    assert ex.yaml_scalar("person") == "person"
    assert ex.yaml_scalar("potted plant") == '"potted plant"'
    assert ex.yaml_scalar("helmet-color") == "helmet-color"
    for risky in ("安全帽: 大", "true", "No", "null", "3d", "", "a#b"):
        assert yaml.safe_load("k: " + ex.yaml_scalar(risky))["k"] == risky


def test_nested_and_flat_names_do_not_eat_each_other(tmp_path):
    """``a/f.jpg`` 压平后是 ``a__f``，正好和真实存在的 ``a__f.jpg`` 撞名。

    旧写法后写的把先写的连图带标签一起盖掉：3 张图导出后磁盘只剩 2 个文件，
    清单和返回值却都按 3 张计，训练时没人看得出来少了帧。
    """
    root = str(tmp_path / "clash")
    init_dataset(root, name="clash")
    nested = tmp_path / "src_nested"
    nested.mkdir()
    _jpeg(nested / "f.jpg", (10, 200, 30))
    flat = tmp_path / "src_flat"
    flat.mkdir()
    _jpeg(flat / "a__f.jpg", (200, 20, 90))
    import_images(root, [str(nested)], subdir="a", dedupe="none")
    import_images(root, [str(flat)], dedupe="none")
    rels = sorted(idx_names(root))
    assert rels == ["a/f.jpg", "a__f.jpg"]
    for rel in rels:
        write_label(root, rel, [SAMPLE17])

    out = os.path.join(str(tmp_path), "exp-clash")
    result = ex.export_dataset(root, out=out, val_ratio=0.0, mode="copy")
    images = _listing(out, "train")
    assert len(images) == 2, f"两张图导出后磁盘只剩 {images}"
    assert len(set(images)) == 2
    assert result["train"] == 2
    labels = _listing(out, "train", kind="labels")
    assert [os.path.splitext(n)[0] for n in labels] == [os.path.splitext(n)[0] for n in images]
    # 盖掉的帧在 COCO 里也会一起消失，顺手一起验
    coco_out = os.path.join(str(tmp_path), "exp-clash-coco")
    ex.export_dataset(root, out=coco_out, val_ratio=0.0, coco=True, mode="copy")
    assert len(_load(os.path.join(coco_out, "coco_keypoints.json"))["images"]) == 2


def test_reexport_into_the_same_out_prunes_rejected_frames(dataset, tmp_path):
    """同一个 `--out` 反复导出时，这一版不要的帧必须从磁盘上清掉。

    只覆盖同名文件的旧写法，会把上一版留下、这一版已被驳回的帧留在 ``images/train``
    里：训练集悄悄带着用户判定不要的帧，界面却只报剩下的张数。
    """
    for rel in sorted(idx_names(dataset)):
        write_label(dataset, rel, [SAMPLE17])
    out = os.path.join(str(tmp_path), "exp-stale")
    first = ex.export_dataset(dataset, out=out, val_ratio=0.0, mode="copy")
    assert first["train"] == 2 and _listing(out, "train") == ["img0.jpg", "img1.jpg"]

    idx = load_index(dataset)
    idx.items["img1.jpg"]["status"] = "rejected"
    idx.save()
    second = ex.export_dataset(dataset, out=out, val_ratio=0.0, mode="copy")
    assert second["train"] == 1
    assert _listing(out, "train") == ["img0.jpg"], "被驳回的帧还留在导出目录里"
    assert _listing(out, "train", kind="labels") == ["img0.txt"]
    assert second["pruned"] == 2                       # 图片 + 标签
    manifest = _load(os.path.join(out, "manifest.json"))
    assert manifest["counts"]["pruned"] == 2


def test_detection_export_prunes_and_quotes(tmp_path):
    """检测库导出走同一套清理与转义，别只修姿态那一半。"""
    root = str(tmp_path / "det")
    init_dataset(root, name="det")
    source = tmp_path / "det_src"
    source.mkdir()
    for index, color in enumerate([(200, 40, 40), (40, 40, 200)]):
        _jpeg(source / f"d{index}.jpg", color)
    import_images(root, [str(source)], lib=ds.LIB_DETECT)
    registered = field_store.upsert_field(root, name="potted plant", label="盆栽", lib=ds.LIB_DETECT)
    class_id = int(registered["class_id"])
    index = load_index(root, ds.LIB_DETECT)
    for rel in ("d0.jpg", "d1.jpg"):
        pr.write_detections(root, rel,
                            [pr.Detection(cls=class_id, cx=0.5, cy=0.5, w=0.2, h=0.3)],
                            lib=ds.LIB_DETECT)
        index.upsert(rel, detect_objects=1, detect_status=ds.STATUS_UNREVIEWED)
    index.save()

    out = os.path.join(str(tmp_path), "exp-det")
    pr.export_detection_dataset(root, out=out, val_ratio=0.0, lib=ds.LIB_DETECT)
    with open(os.path.join(out, "data.yaml"), encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    assert meta["names"] == {0: "potted plant"}

    index = load_index(root, ds.LIB_DETECT)
    index.items["d1.jpg"]["detect_status"] = "rejected"
    index.save()
    second = pr.export_detection_dataset(root, out=out, val_ratio=0.0, lib=ds.LIB_DETECT)
    assert second["train"] == 1
    assert _listing(out, "train") == ["d0.jpg"], "被驳回的图还留在导出目录里"
    assert second["counts"]["pruned"] == 2
