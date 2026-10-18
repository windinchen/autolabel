"""命令行合理性：参数错了当场说清楚，路径打错别静默「成功」，也别甩 Python 栈。"""

import json
import os

import pytest
from PIL import Image

from autolabel.cli import main

from _helpers import idx_names, write_label

SAMPLE17 = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 17)


def batches_of(root):
    path = os.path.join(root, ".autolabel", "batches.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["batches"]


def test_missing_dataset_reports_plain_words(capsys, tmp_path):
    assert main(["status", str(tmp_path / "ghost")]) == 1
    err = capsys.readouterr().err
    assert "[error]" in err and "init" in err
    assert "Traceback" not in err  # 不许再把 Python 栈甩给普通用户


def test_unknown_batch_id_reports_plain_words(dataset, capsys):
    assert main(["batches", dataset, "--delete", "b_ghost"]) == 1
    err = capsys.readouterr().err
    assert "[error]" in err and "b_ghost" in err and "Traceback" not in err


@pytest.mark.parametrize("flag,value", [
    ("--val-ratio", "1.5"), ("--val-ratio", "0"),
    ("--conf", "5"), ("--iou", "-1"), ("--imgsz", "0"),
    ("--max-side", "-10"), ("--jpeg-quality", "101"), ("--port", "70000"),
])
def test_out_of_range_numbers_are_rejected_early(dataset, capsys, flag, value):
    command = "export" if flag == "--val-ratio" else "label"
    if flag in ("--port",):
        command = "review"
    with pytest.raises(SystemExit) as exit_info:
        main([command, dataset, flag, value])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err and flag in err


def test_sane_numbers_still_pass(dataset, source_dir, capsys):
    """校验不能顺手把合法写法挡掉：0 秒起、0 表示不限制、边界值都要能过。"""
    # 先给一张图写上有内容的标签：没标签时导出会如实报错（见 test_business_rules），
    # 这条测试要验的是"参数合法"，不能被前置条件挡住。
    write_label(dataset, sorted(idx_names(dataset))[0], [SAMPLE17])
    assert main(["export", dataset, "--val-ratio", "0.25", "--seed", "-3"]) == 0
    assert main(["qc", dataset, "--top", "5", "--render", "0"]) == 0
    assert main(["pool", dataset, "--limit", "0"]) == 0
    capsys.readouterr()


def test_video_start_zero_is_allowed(dataset, capsys):
    """`--start 0` 是最常见的写法，不能被「必须大于 0」挡掉。"""
    assert main(["video", dataset, "/tmp/不存在的 clip.mp4", "--start", "0", "--fps", "1"]) == 1
    err = capsys.readouterr().err
    assert "得大于 0" not in err  # 报的是文件不存在，不是参数不合法
    assert "--start" not in err


def test_import_from_typo_path_fails_loudly(dataset, capsys, tmp_path):
    """路径打错以前会「导入成功 0 张」并留一个 0 张的空批次。"""
    before = len(batches_of(dataset))
    assert main(["import", dataset, str(tmp_path / "typo-dir")]) == 1
    out = capsys.readouterr()
    assert "路径不存在" in (out.out + out.err)
    assert "[error]" in out.err
    assert len(batches_of(dataset)) == before  # 没有多出一个空批次


def test_import_repeat_is_explained_not_recorded(dataset, source_dir, capsys):
    """重复导入要说「没有新图入库」，而不是造一个空批次糊弄过去。"""
    assert main(["import", dataset, source_dir]) == 0
    capsys.readouterr()
    before = len(batches_of(dataset))
    assert main(["import", dataset, source_dir]) == 0
    out = capsys.readouterr()
    assert "没有新图入库" in out.out
    assert len(batches_of(dataset)) == before


def test_partial_missing_sources_still_import_the_rest(dataset, capsys, tmp_path):
    """一部分路径写错：继续导入能用的，同时逐条 warn。"""
    extra = tmp_path / "extra"
    extra.mkdir()
    Image.new("RGB", (64, 48), (20, 120, 200)).save(extra / "extra0.jpg")
    assert main(["import", dataset, str(extra), str(tmp_path / "typo")]) == 0
    out = capsys.readouterr()
    assert "路径不存在" in (out.out + out.err)
    assert "[ok] 扫描 1 · 导入 1" in out.out
    assert any(batch.get("items") for batch in batches_of(dataset))


def test_unknown_coord_format_is_rejected(dataset, capsys):
    """`--formats xml` 以前会「导出成功」但一个文件都不生成。"""
    assert main(["coords", dataset, "--formats", "xml"]) == 1
    err = capsys.readouterr().err
    assert "未知坐标格式" in err and "csv" in err


def test_auto_without_anything_importable_stops_early(dataset, source_dir, capsys):
    """一张都没进来就往下跑，会产出一堆空文件看着像成功。"""
    assert main(["import", dataset, source_dir]) == 0
    capsys.readouterr()
    assert main(["auto", dataset, source_dir]) == 1
    captured = capsys.readouterr()
    assert "没有可导入的图片" in (captured.out + captured.err)
    last = batches_of(dataset)[-1]
    assert last["status"] == "failed"
