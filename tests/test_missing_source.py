"""第 16–17 轮体检：索引里还记着、`images/` 里图已经不在了。

实测把一张已接受的图从磁盘上移走之后：

1. `export --mode copy` / `hardlink` 直接崩，报的还是一句裸的
   ``[Errno 2] No such file or directory``，两万张的导出停在一半；
2. `export --mode symlink` **静默产出一根指向空气的软链**，manifest 照报
   "train 4 张"，用户要到训练读图时才发现；
3. `status` 只说"缺标签 0"、`validate` 只说没问题、`coords` 照旧给这一帧写一行
   ——整个工具链没有一处告诉用户"这张图没了"。

现在三处同一个口径：缺源文件的帧不进导出集、不进坐标文件，并且
`status` / `validate` / manifest / 命令行提示都会点名是哪几张。
"""
import json
import os

import pytest

from _helpers import write_label
from autolabel import cli
from autolabel import prompt as pr
from autolabel import fields as field_store
from autolabel.coords import write_coordinate_files
from autolabel.dataset import (import_images, init_dataset, load_index, missing_sources,
                      source_missing)
from autolabel.export import MISSING_SOURCE_REASON, export_dataset
from autolabel.qc import dataset_stats

ROW = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 17)
VICTIM = "img1.jpg"
SURVIVOR = "img0.jpg"


def _approve_everything(root):
    """两张图都写标签并标成 accepted，这样它们一定会被导出选中。"""
    index = load_index(root)
    for rel in sorted(index.items):
        write_label(root, rel, [ROW])
        index.upsert(rel, status="approved", objects=1, score=0.9, labeled_at="2026-10-19 00:00")
    index.save()


def _hide(root, rel):
    """把图片挪出 `images/`（用改名，别真删，测试收尾还要放回去）。"""
    path = os.path.join(os.path.abspath(root), "images", rel)
    os.rename(path, path + ".hold")
    return path


def _exported_names(out_root):
    found = []
    for split in ("train", "val"):
        folder = os.path.join(out_root, "images", split)
        if os.path.isdir(folder):
            found += sorted(os.listdir(folder))
    return found


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- 判定本身 ------------------------------------------------------------- #

def test_source_missing_only_reports_files_that_are_really_gone(dataset):
    _approve_everything(dataset)
    assert source_missing(dataset, SURVIVOR) is False
    assert source_missing(dataset, VICTIM) is False
    _hide(dataset, VICTIM)
    assert source_missing(dataset, VICTIM) is True
    assert missing_sources(dataset, [SURVIVOR, VICTIM]) == [VICTIM]


def test_source_missing_treats_traversal_as_gone(dataset):
    """索引里躺着一条越界路径：既不是一帧图，也不该被当成"文件在"。"""
    assert source_missing(dataset, "../../etc/passwd") is True


# --- 三种导出模式同一行为 ------------------------------------------------- #

@pytest.mark.parametrize("mode", ["symlink", "copy", "hardlink"])
def test_export_skips_missing_frame_in_every_mode(dataset, tmp_path, mode):
    _approve_everything(dataset)
    _hide(dataset, VICTIM)
    out = str(tmp_path / f"out-{mode}")

    result = export_dataset(dataset, out=out, val_ratio=0.0, mode=mode)

    assert result["missing_source"] == 1
    assert result["train_images"] == 1 and result["val_images"] == 0
    assert _exported_names(out) == [SURVIVOR], "缺源文件的帧混进导出目录了"
    for name in _exported_names(out):
        assert os.path.exists(os.path.join(out, "images", "train", name)), \
            f"{name} 是根读不出来的软链"
    # 缺的那帧标签还在磁盘上，不能被拷成一个"有图片没了、标签还在"的样本
    assert not os.path.exists(os.path.join(out, "labels", "train", VICTIM))


def test_copy_mode_no_longer_dies_on_a_gone_file(dataset, tmp_path):
    """以前 copy 模式抛 FileNotFoundError 整批失败，报错里连帧名都没有。"""
    _approve_everything(dataset)
    _hide(dataset, VICTIM)
    out = str(tmp_path / "out-copy")

    result = export_dataset(dataset, out=out, mode="copy")

    assert result["missing_source"] == 1
    manifest = _load(os.path.join(out, "manifest.json"))
    assert manifest["counts"]["missing_source"] == 1
    reasons = [row["reason"] for row in manifest["skipped"] if row["image"] == VICTIM]
    assert reasons == [MISSING_SOURCE_REASON]


def test_exporting_again_after_the_file_returns_is_clean(dataset, tmp_path):
    _approve_everything(dataset)
    path = _hide(dataset, VICTIM)
    first = export_dataset(dataset, out=str(tmp_path / "again"), mode="copy")
    os.rename(path + ".hold", path)
    second = export_dataset(dataset, out=str(tmp_path / "again"), mode="copy")
    assert first["missing_source"] == 1 and second["missing_source"] == 0
    assert second["train_images"] + second["val_images"] == 2, "图放回去之后应该重新进导出集"


def test_all_frames_gone_explains_why(dataset, tmp_path):
    _approve_everything(dataset)
    for rel in sorted(load_index(dataset).items):
        _hide(dataset, rel)
    with pytest.raises(ValueError) as caught:
        export_dataset(dataset, out=str(tmp_path / "empty"), mode="copy")
    assert "images/" in str(caught.value)


# --- 统计、校验、坐标 ----------------------------------------------------- #

def test_dataset_stats_counts_missing_files(dataset):
    _approve_everything(dataset)
    assert dataset_stats(dataset)["missing_files"] == 0
    _hide(dataset, VICTIM)
    stats = dataset_stats(dataset)
    assert stats["missing_files"] == 1
    assert stats["missing_images"] == [VICTIM]
    # 图没了不等于标签没了：标签还在磁盘上，两个格子必须分得开
    assert stats["missing_labels"] == 0


def test_status_names_the_missing_file(dataset, capsys):
    _approve_everything(dataset)
    _hide(dataset, VICTIM)
    capsys.readouterr()
    assert cli.main(["status", dataset]) == 0
    out = capsys.readouterr().out
    assert "缺文件 1 张" in out and VICTIM in out


def test_status_stays_quiet_when_everything_is_there(dataset, capsys):
    _approve_everything(dataset)
    assert cli.main(["status", dataset]) == 0
    assert "缺文件" not in capsys.readouterr().out


def test_validate_reports_missing_source(dataset, capsys):
    _approve_everything(dataset)
    _hide(dataset, VICTIM)
    capsys.readouterr()
    assert cli.main(["validate", dataset, "--lib", "pose"]) == 1
    out = capsys.readouterr().out
    assert "missing_source" in out
    assert "[姿态] 缺源文件: " + VICTIM in out


def test_coords_skip_the_gone_frame(dataset):
    _approve_everything(dataset)
    _hide(dataset, VICTIM)
    out = os.path.join(str(dataset), "coords-run")

    result = write_coordinate_files(dataset, out=out, exclude_status=())

    assert result["images"] == 1
    with open(os.path.join(out, "coords.csv"), encoding="utf-8") as fh:
        assert VICTIM not in fh.read()
    summary = _load(os.path.join(out, "summary.json"))
    assert [row["reason"] for row in summary["skipped_items"] if row["image"] == VICTIM] == \
        [MISSING_SOURCE_REASON]


# --- 提示词库 / 检测库是另一条导出路径 ------------------------------------ #

def test_detection_export_skips_and_reports_missing_source(tmp_path):
    from PIL import Image

    root = str(tmp_path / "det")
    init_dataset(root, name="det")
    source = tmp_path / "det_src"
    source.mkdir()
    for index, color in enumerate([(200, 40, 40), (40, 40, 200)]):
        Image.new("RGB", (160, 120), color).save(source / f"d{index}.jpg")
    import_images(root, [str(source)], lib="objects")
    registered = field_store.upsert_field(root, name="bottle", label="瓶子")
    class_id = int(registered["class_id"])
    index = load_index(root, "objects")
    for rel in ("d0.jpg", "d1.jpg"):
        pr.write_detections(root, rel,
                            [pr.Detection(cls=class_id, cx=0.5, cy=0.5, w=0.2, h=0.3)])
        index.upsert(rel, prompt_objects=1, prompt_status="approved")
    index.save()
    os.rename(os.path.join(root, "images", "d1.jpg"), os.path.join(root, "images", "d1.jpg.hold"))

    out = str(tmp_path / "exp")
    result = pr.export_detection_dataset(root, out=out, val_ratio=0.0)

    assert result["counts"]["missing_source"] == 1
    assert result["train"] == 1
    assert _exported_names(out) == ["d0.jpg"]
    assert "源文件缺失" in result["skipped_note"]
    meta = _load(os.path.join(out, "prompt_meta.json"))
    assert meta["counts"]["missing_source"] == 1


def test_detection_export_says_so_when_nothing_is_left(tmp_path):
    """唯一的框画在丢了的图上：报错要提一句"另有 N 帧源文件找不到"。"""
    from PIL import Image

    root = str(tmp_path / "det2")
    init_dataset(root, name="det2")
    source = tmp_path / "det2_src"
    source.mkdir()
    Image.new("RGB", (160, 120), (200, 40, 40)).save(source / "only.jpg")
    import_images(root, [str(source)], lib="objects")
    registered = field_store.upsert_field(root, name="bottle", label="瓶子")
    index = load_index(root, "objects")
    pr.write_detections(root, "only.jpg",
                        [pr.Detection(cls=int(registered["class_id"]), cx=0.5, cy=0.5, w=0.2, h=0.3)])
    index.upsert("only.jpg", prompt_objects=1, prompt_status="approved")
    index.save()
    os.rename(os.path.join(root, "images", "only.jpg"), os.path.join(root, "images", "only.jpg.hold"))

    with pytest.raises(ValueError) as caught:
        pr.export_detection_dataset(root, out=str(tmp_path / "exp2"), val_ratio=0.0)
    assert "源文件" in str(caught.value)
