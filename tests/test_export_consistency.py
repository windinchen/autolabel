"""第 18 轮体检：导出 / QC / 坐标三处口径必须一致。

这一批全部是上一轮「已核实但没动」的清单，每条都本机实测复现过：

1. 「背景图」按标签文件**字节数**判定 —— 只有一个换行的标签算"有人"，
   ``--max-empty-ratio`` 对整批帧失效，导出报 empty 0 而同一份库 QC 报"无人图 3"；
2. 类别号越界没人拦 —— COCO 里 ``category_id=8`` 对上 ``categories=[1]``，
   按 id 查类别的评估脚本当场炸；坐标文件把类别号写成 class_name="7"，凭空多一类；
3. ``export --coords`` 把被背景图挤掉、缺标签的帧也写进坐标 —— 同一份 manifest 里
   ``train+val=1`` 和 ``coords_counts.images=4`` 并存，split 空着还不解释；
4. 软链→硬链→拷贝三级全失败时抛裸 ``OSError``：英文 traceback + 半截导出目录；
5. 子目录压进文件名后没有长度/保留名兜底 —— 90 个汉字的素材名在 ext4 上
   导出 ``[Errno 36] File name too long``，``CON.jpg`` 在 Windows 上落不了地；
6. QC 报告不提"原图缺失"，还把缺标签的帧从"平均每图/无人图"的分母里悄悄摘掉，
   ``--include-reviewed`` 时又一边数已定论的帧一边说"已定论的不占位"。
"""
import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from _helpers import idx_names, write_label
from autolabel import cli
from autolabel import dataset as ds
from autolabel import export as ex
from autolabel.coords import write_coordinate_files
from autolabel.qc import dataset_stats, write_report

ROW = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 17)


def row_with_class(class_id):
    return ROW.replace("0 0.500000", f"{class_id} 0.500000", 1)


@pytest.fixture()
def library(tmp_path):
    """造一个只有合成图片的最小姿态库。"""
    def build(count=4, name="ds"):
        source = tmp_path / f"{name}-src"
        source.mkdir()
        for index in range(count):
            frame = np.full((240, 320, 3), 30 + index * 25, np.uint8)
            Image.fromarray(frame).save(source / f"f{index}.jpg")
        root = tmp_path / name
        ds.init_dataset(str(root), name=name)
        ds.import_images(str(root), [str(source)])
        return str(root)
    return build


def mark_labeled(root, rels, **extra):
    index = ds.load_index(root)
    for rel in rels:
        index.upsert(rel, objects=1, score=0.9, width=320, height=240, **extra)
    index.save()


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def report_text(root):
    with open(os.path.join(root, "qc", "report.md"), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# 1. 背景图按框数判定
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("content,expected", [
    ("", 0), ("\n", 0), ("  \n\n", 0),
    (ROW + "\n", 1), ("\n" + ROW + "\n", 1), (ROW + "\n" + ROW, 2),
])
def test_label_object_count_counts_rows_not_bytes(tmp_path, content, expected):
    from autolabel.yolo_pose import label_object_count

    path = tmp_path / "label.txt"
    path.write_text(content, encoding="utf-8")
    assert label_object_count(str(path)) == expected
    assert label_object_count(str(tmp_path / "不存在的标签.txt")) == 0


def test_blank_label_file_counts_as_background(library, tmp_path):
    """4 张图里 3 张的标签只有一个换行：必须全算背景图并被上限压掉。"""
    root = library(4)
    names = sorted(idx_names(root))
    write_label(root, names[0], [ROW])
    for rel in names[1:]:
        write_label(root, rel, [""])
    mark_labeled(root, names)

    result = ex.export_dataset(root, out=str(tmp_path / "out"), val_ratio=0.5,
                               max_empty_ratio=0.1, mode="copy")
    manifest = json.load(open(os.path.join(result["out"], "manifest.json"), encoding="utf-8"))
    assert manifest["counts"]["empty_dropped"] == 3
    assert "背景图超出比例上限" in result["skipped_note"]
    # 同一份库的 QC 说 3 张无人图，导出就不许说 0 张背景图
    assert dataset_stats(root)["persons_per_image"]["empty_images"] == 3


# --------------------------------------------------------------------------- #
# 2. 类别号越界
# --------------------------------------------------------------------------- #

def test_out_of_range_class_is_skipped_not_exported(library, tmp_path):
    root = library(2)
    names = sorted(idx_names(root))
    write_label(root, names[0], [ROW])
    write_label(root, names[1], [row_with_class(7)])
    mark_labeled(root, names)

    out = str(tmp_path / "out")
    result = ex.export_dataset(root, out=out, val_ratio=0.0, coco=True, mode="copy")
    manifest = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    assert result["train_images"] == 1
    assert manifest["counts"]["class_out_of_range"] == 1
    assert "类别号不在当前类别表" in result["skipped_note"]

    coco = json.load(open(os.path.join(out, "coco_keypoints.json"), encoding="utf-8"))
    assert {c["id"] for c in coco["categories"]} == {1}
    assert {a["category_id"] for a in coco["annotations"]} == {1}


def test_out_of_range_hint_points_at_the_setting(library, tmp_path):
    """全部帧都越界时要指真正能改的那一处，而不是继续劝 --include-all。"""
    root = library(2)
    names = sorted(idx_names(root))
    for rel in names:
        write_label(root, rel, [row_with_class(7)])
    mark_labeled(root, names)

    with pytest.raises(ValueError) as caught:
        ex.export_dataset(root, out=str(tmp_path / "out"), val_ratio=0.0, mode="copy")
    assert "autolabel validate" in str(caught.value)


def test_coords_skip_unknown_class_instead_of_inventing_a_name(library, tmp_path):
    root = library(2)
    names = sorted(idx_names(root))
    write_label(root, names[0], [ROW])
    write_label(root, names[1], [row_with_class(7)])
    mark_labeled(root, names)

    result = write_coordinate_files(root, out=os.path.join(root, "coords", "cls"))
    assert result["images"] == 1
    assert any("类别号" in row["reason"] for row in result["meta"]["skipped"])
    assert {row["class_name"] for row in read_rows(os.path.join(result["out"], "coords.csv"))} == {"person"}


# --------------------------------------------------------------------------- #
# 3. 坐标文件对齐真正导出的帧
# --------------------------------------------------------------------------- #

def test_export_coords_covers_exactly_the_exported_frames(library, tmp_path):
    root = library(4)
    names = sorted(idx_names(root))
    write_label(root, names[0], [ROW])
    for rel in names[1:]:
        write_label(root, rel, [""])
    mark_labeled(root, names)

    result = ex.export_dataset(root, out=str(tmp_path / "out"), val_ratio=0.5,
                               mode="copy", coords=True, max_empty_ratio=0.1)
    manifest = json.load(open(os.path.join(result["out"], "manifest.json"), encoding="utf-8"))
    exported = manifest["counts"]["train"] + manifest["counts"]["val"]
    assert exported == 1 and manifest["coords_counts"]["images"] == exported

    rows = read_rows(os.path.join(result["coords"], "coords.csv"))
    assert len(rows) == 1 and rows[0]["split"] in {"train", "val"}
    split_dir = os.path.join(result["out"], "images", rows[0]["split"])
    assert len(os.listdir(split_dir)) == 1


# --------------------------------------------------------------------------- #
# 4. 放文件失败的报错
# --------------------------------------------------------------------------- #

def test_place_error_message_translates_errno_and_offers_a_way_out():
    message = ex.place_error_message("symlink", "/tmp/a.jpg", "/tmp/out", OSError(36, "File name too long"))
    assert "软链接" in message and "文件名太长" in message and "--mode copy" in message
    assert "没有权限" not in message
    final = ex.place_error_message("copy", "/tmp/a.jpg", "/tmp/out", OSError(13, "Permission denied"))
    assert "直接拷贝" in final and "没有权限" in final and "--mode copy" not in final


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root 读 000 文件也成功")
def test_copy_failure_says_why_and_leaves_no_half_export(library, capsys):
    root = library(2)
    names = sorted(idx_names(root))
    for rel in names:
        write_label(root, rel, [ROW])
    mark_labeled(root, names)
    victim = os.path.join(os.path.abspath(root), "images", names[0])
    os.chmod(victim, 0o000)
    try:
        assert cli.main(["export", root, "--mode", "copy"]) == 1
        err = capsys.readouterr().err
    finally:
        os.chmod(victim, 0o644)

    assert "[error]" in err and "把文件放进导出目录失败" in err and "没有权限" in err
    assert "Traceback" not in err
    exports = os.path.join(root, "exports")
    assert not os.path.isdir(exports) or not os.listdir(exports)


# --------------------------------------------------------------------------- #
# 5. 导出文件名兜底
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rel,expected", [
    ("CON.jpg", "CON_x"),
    ("cam/LPT9.jpg", "cam__LPT9"),          # 只有整段等于保留名才要改，前缀不算
    ("a:b c.jpg", "a_b_c"),
    ("主会议室/第3帧.jpg", "主会议室__第3帧"),
    ("正常帧_0001.jpg", "正常帧_0001"),
    ("frame.0001.jpg", "frame.0001"),
    (".hidden.jpg", "hidden"),
])
def test_export_stem_stays_writable_and_readable(rel, expected):
    assert ex.export_stem(rel) == expected


def test_export_stem_stays_inside_the_byte_budget():
    stem = ex.export_stem("主" * 90 + ".jpg")
    assert stem.startswith("主") and len(stem) > 20
    assert len(stem.encode("utf-8")) <= ex.STEM_BYTE_BUDGET


def test_truncated_long_names_still_do_not_overwrite_each_other():
    taken = {}
    first = ex.unique_stem(taken, ex.export_stem("a" * 200 + ".jpg"), "x/" + "a" * 200 + ".jpg")
    second = ex.unique_stem(taken, ex.export_stem("a" * 199 + "b.jpg"), "x/" + "a" * 199 + "b.jpg")
    assert first != second


# --------------------------------------------------------------------------- #
# 6. QC 报告的口径
# --------------------------------------------------------------------------- #

def test_qc_report_names_missing_images(library):
    root = library(3)
    names = sorted(idx_names(root))
    for rel in names:
        write_label(root, rel, [ROW])
    mark_labeled(root, names)
    os.remove(os.path.join(os.path.abspath(root), "images", names[1]))

    write_report(root)
    text = report_text(root)
    assert "原图缺失 1" in text
    assert "原图缺失：1 张" in text

    assert dataset_stats(root)["missing_files"] == 1


def test_qc_report_says_what_the_average_was_divided_by(library):
    """4 张图只标了 2 张：不能写成"平均每图 1.0、无人图 0"就完事。"""
    root = library(4)
    names = sorted(idx_names(root))
    for rel in names[:2]:
        write_label(root, rel, [ROW])
    mark_labeled(root, names)

    write_report(root)
    text = report_text(root)
    assert "只按有标签的 2 张算" in text and "缺标签 2 张" in text
    assert dataset_stats(root)["persons_per_image"]["counted"] == 2


def test_pending_line_stops_claiming_settled_rows_are_excluded(library):
    root = library(2)
    names = sorted(idx_names(root))
    for rel in names:
        write_label(root, rel, [ROW])
    mark_labeled(root, [names[0]], status="approved", flags=["crowded"])
    mark_labeled(root, [names[1]], status="rejected", flags=["tiny_box"])

    write_report(root)
    assert "待复核：0 张" in report_text(root) and "已定论 2 张不占位" in report_text(root)

    write_report(root, include_reviewed=True)
    text = report_text(root)
    assert "待复核：2 张" in text and "已定论的 2 张也算进来了" in text
    assert "不占位" not in text


# --------------------------------------------------------------------------- #
# 7. 空库别再报成功
# --------------------------------------------------------------------------- #

def test_coords_on_empty_library_warns_instead_of_claiming_success(library, capsys):
    root = library(2)
    out = os.path.join(root, "coords", "empty")
    assert cli.main(["coords", root, "--out", out]) == 0
    out_text = capsys.readouterr().out
    assert "[warn]" in out_text and "[ok]" not in out_text
    assert json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))["no_data"] is True
