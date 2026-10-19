"""第 11–13 轮体检（状态名校验 / reset 删除范围 / validate 覆盖三个库）的回归测试。

盯三件事：

1. ``reset --status`` 写错必须当场报错——以前 ``aprovied`` 只会得到一句
   「已退回 0 张」加退出码 0，用户以为执行过了；
2. ``reset --labels`` 只能删这次真被退回的那些，并且标签没了要把索引里由标签
   派生的计数一起抹掉，否则 ``status`` 的人数继续报「有 1 个人」；
3. ``validate`` 要把提示词库与检测库的五列标签也查了——坏行在
   ``fields.read_label_rows`` 里是静默跳过的，不查就永远没人知道框悄悄少了。
"""
import io
import os

import pytest
from _helpers import write_label

from autolabel import cli
from autolabel.dataset import adopt_from_pool, label_path_for, load_index
from autolabel.fields import load_registry, save_registry
from autolabel.library import LIB_DETECT

ROW = "0.5 0.5 0.2 0.6 0.9 " + " ".join(["0.3 0.4 2"] * 17)
BOX = "0 0.5 0.5 0.2 0.4"


def mark_reviewed(root, statuses=("approved", "rejected")):
    """给前 N 张图打上复核状态，并写出标签 + 由标签派生的索引计数。"""
    idx = load_index(root)
    rels = sorted(idx.items)[: len(statuses)]
    for rel, status in zip(rels, statuses):
        write_label(root, rel, [ROW])
        idx.upsert(rel, status=status, objects=1, score=0.9, label_signature="sig-keep",
                   labeled_at="2026-10-19 00:00")
    idx.save()
    return rels


def fill_detect_lib(root):
    """把素材池里的图加进目标检测库，并登记一个类别（class_id=0）。"""
    pose = load_index(root)
    rels = sorted(pose.items)
    adopt_from_pool(root, rels, lib=LIB_DETECT)
    registry = load_registry(root, LIB_DETECT)
    registry["fields"] = [{"class_id": 0, "name": "person", "label": "人"}]
    save_registry(root, registry, LIB_DETECT)
    return rels


# --- 第 11 轮：状态名拼错当场拦 ------------------------------------------- #

def test_reset_typo_status_exits_2_with_options(dataset, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["reset", dataset, "--status", "aprovied"])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "不认识的状态「aprovied」" in err
    assert "approved" in err and "rejected" in err


def test_reset_accepts_valid_status_list(dataset, capsys):
    mark_reviewed(dataset)
    capsys.readouterr()
    assert cli.main(["reset", dataset, "--status", "approved,rejected"]) == 0
    out = capsys.readouterr().out
    assert "已退回 2 张" in out
    idx = load_index(dataset)
    assert all(item["status"] == "unreviewed" for item in idx.items.values())


# --- 第 12 轮：--labels 的删除范围与索引清理 ------------------------------ #

def test_reset_labels_deletes_only_the_reset_subset(dataset):
    approved, rejected = mark_reviewed(dataset)
    assert cli.main(["reset", dataset, "--status", "rejected", "--labels"]) == 0

    assert os.path.exists(label_path_for(dataset, approved)), "已接受样本的标签被误删了"
    assert not os.path.exists(label_path_for(dataset, rejected))


def test_reset_labels_clears_label_derived_index_fields(dataset, capsys):
    approved, rejected = mark_reviewed(dataset)
    assert cli.main(["reset", dataset, "--status", "rejected", "--labels"]) == 0
    idx = load_index(dataset)

    item = idx.items[rejected]
    for field in ("objects", "score", "label_signature", "labeled_at"):
        assert field not in item, f"标签已删但索引还留着 {field}={item[field]}"
    assert item["status"] == "unreviewed"

    kept = idx.items[approved]
    assert kept["objects"] == 1 and kept["label_signature"] == "sig-keep"


def test_reset_labels_reports_the_scope(dataset, capsys):
    mark_reviewed(dataset)
    capsys.readouterr()
    assert cli.main(["reset", dataset, "--status", "rejected", "--labels"]) == 0
    out = capsys.readouterr().out
    assert "删掉标签 1 个" in out and "同一批" in out


def test_reset_reports_nothing_to_reset(dataset, capsys):
    mark_reviewed(dataset, statuses=("approved", "approved"))
    assert cli.main(["reset", dataset, "--status", "rejected"]) == 0
    out = capsys.readouterr().out
    assert "已退回 0 张" in out and "这批里没有可退回的复核状态" in out


# --- 第 13 轮：validate 三个库 + 五列标签 -------------------------------- #

def test_validate_pose_only_when_asked(dataset):
    mark_reviewed(dataset)
    assert cli.main(["validate", dataset, "--lib", "pose"]) == 0


def test_validate_detect_catches_broken_box_rows(dataset, capsys):
    rels = fill_detect_lib(dataset)
    write_label(dataset, rels[0], [BOX, "7 0.4 0.4 0.1 0.1", "0 0.5 0.5 0 0.3", "0 0.5 0.5"],
                lib=LIB_DETECT)
    assert cli.main(["validate", dataset, "--lib", "detect"]) == 1
    out = capsys.readouterr().out
    assert "bad_class=1" in out            # 类别号不在本库类别表里
    assert "bbox_degenerate=1" in out      # w=0 的退化框
    assert "bad_columns=1" in out          # 只有 4 列的行（读取时会被静默跳过）
    assert "[检测]" in out


def test_validate_default_covers_all_libraries(dataset, capsys):
    """默认 `--lib all`：检测库的坏标签不能因为「validate 只管姿态」而漏网。"""
    rels = fill_detect_lib(dataset)
    write_label(dataset, rels[0], [BOX])
    write_label(dataset, rels[1], ["broken"], lib=LIB_DETECT)
    assert cli.main(["validate", dataset]) == 1
    out = capsys.readouterr().out
    assert "bad_columns=1" in out and "查的是：姿态标注、提示词标注、目标检测" in out


def test_validate_orphan_labels_named_by_library(dataset, capsys):
    rels = fill_detect_lib(dataset)
    write_label(dataset, rels[0], [BOX], lib=LIB_DETECT)
    write_label(dataset, rels[0], [ROW])            # 姿态库标签齐
    write_label(dataset, rels[1], [ROW])
    orphan = label_path_for(dataset, "deleted_frame.txt".replace(".txt", ""), LIB_DETECT)
    os.makedirs(os.path.dirname(orphan), exist_ok=True)
    with open(orphan + ".txt", "w", encoding="utf-8") as fh:
        fh.write(BOX + "\n")
    assert cli.main(["validate", dataset, "--lib", "detect"]) == 1
    out = capsys.readouterr().out
    assert "孤儿标签" in out and "[检测]" in out


def test_validate_empty_library_is_not_an_error(dataset, capsys):
    assert cli.main(["validate", dataset, "--lib", "detect"]) == 0
    assert "都还没有素材" in capsys.readouterr().out


# --- 第 13 轮附：接口层「非法状态」要说人话 ------------------------------ #

def test_set_status_bad_value_is_explained_in_chinese(dataset):
    from autolabel.review import ReviewStore

    store = ReviewStore(dataset)
    rel = sorted(load_index(dataset).items)[0]
    with pytest.raises(ValueError) as caught:
        store.set_status(rel, "aprovied")
    message = str(caught.value)
    assert "复核状态只能是" in message and "aprovied" in message


# --- 第 28 轮体检：索引里那笔框数账和标签文件对不对得上 ------------------- #

def _write_boxes(root, rel, text):
    """绕过标注流水线，直接决定磁盘上的标签文件里有什么。"""
    path = label_path_for(root, rel, LIB_DETECT)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _fill_books(dataset, counts):
    """给检测库里每张图按 ``counts`` 写框，并把索引里的框数设成给定值。

    ``counts`` 里给的是"索引里记几个"，磁盘上永远写 1 个框——两边一致与否
    完全由这个数说了算，断言才不会被别的问题干扰。
    """
    fill_detect_lib(dataset)
    rels = sorted(load_index(dataset, LIB_DETECT).items)
    index = load_index(dataset, LIB_DETECT)
    for rel in rels:
        _write_boxes(dataset, rel, BOX + "\n")
        claimed = counts(rel)
        if claimed is None:
            index.items[rel].pop("detect_objects", None)
        else:
            index.upsert(rel, detect_objects=claimed, detect_status="approved")
    index.save()
    return rels


def test_validate_names_index_and_label_box_count_drift(dataset, capsys):
    """手工加过框之后，``validate`` 得说出"索引记 2 个、文件里 1 个"。"""
    rels = _fill_books(dataset, lambda rel: 2)
    capsys.readouterr()

    assert cli.main(["validate", dataset, "--lib", LIB_DETECT]) == 1
    out = capsys.readouterr().out
    assert "count_mismatch=" + str(len(rels)) in out
    assert f"{rels[0]}: 索引记 2 个框，标签文件里是 1 个" in out


def test_validate_flags_boxes_the_index_thinks_were_never_labeled(dataset, capsys):
    """标签文件里有框、索引却说"没标过"：导出会把这张图整张跳过。

    用户看到的现象是"明明标了框，导出来一个都没有"，
    而 ``validate`` 以前只查格式，一点征兆都不报。
    """
    rels = _fill_books(dataset, lambda rel: None)
    capsys.readouterr()

    assert cli.main(["validate", dataset, "--lib", LIB_DETECT]) == 1
    out = capsys.readouterr().out
    assert "索引记 没标过，标签文件里是 1 个" in out
    assert f"{rels[0]}" in out


def test_validate_stays_quiet_when_the_books_match(dataset, capsys):
    _fill_books(dataset, lambda rel: 1)
    capsys.readouterr()

    assert cli.main(["validate", dataset, "--lib", LIB_DETECT]) == 0
    assert "count_mismatch" not in capsys.readouterr().out


def test_validate_does_not_double_report_a_missing_label_file(dataset, capsys):
    """标签文件整个没了：只报缺标签，别再补一条"索引记 N 个框"。"""
    rels = _fill_books(dataset, lambda rel: 1)
    os.remove(label_path_for(dataset, rels[0], LIB_DETECT))
    capsys.readouterr()

    assert cli.main(["validate", dataset, "--lib", LIB_DETECT]) == 1
    out = capsys.readouterr().out
    assert "missing_label" in out and "count_mismatch" not in out
