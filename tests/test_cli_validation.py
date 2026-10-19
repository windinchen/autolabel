"""命令行合理性：参数错了当场说清楚，路径打错别静默「成功」，也别甩 Python 栈。"""

import json
import os

import pytest
from PIL import Image

from autolabel.cli import main

from _helpers import idx_names, write_label
from autolabel.dataset import load_index

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


def test_detect_rejects_unknown_images_before_any_side_effect(dataset, monkeypatch, capsys):
    """``--images`` 打错一个字：一行说清楚，别留下失败批次和打不开的幽灵行。

    以前一次留三样脏东西——``model_failed`` 的索引记录（界面多一行没有宽高的图）、
    一条失败的批次记录，再加 8 行 Python 栈。
    """
    from autolabel import detect as dt
    from autolabel.batches import load_batches
    from autolabel.library import LIB_DETECT
    from test_detect import use_backend

    use_backend(monkeypatch)
    dt.add_model_classes(dataset, "yolo11n.pt", ["person"])
    assert main(["detect", dataset, "--images", "ghost.jpg", "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "[error]" in err and "ghost.jpg" in err and "Traceback" not in err
    assert load_batches(dataset, LIB_DETECT) == []
    assert load_index(dataset, LIB_DETECT).get("ghost.jpg") is None


def test_prompt_rejects_unknown_images_before_any_side_effect(dataset, monkeypatch, capsys):
    """提示词标注同理：路径打错连一次模型请求都不该发，也不该建批次。"""
    from autolabel import fields as fd
    from autolabel import models as model_manager
    from autolabel.batches import load_batches
    from autolabel.library import LIB_OBJECTS

    fd.upsert_field(dataset, name="bottle", label="瓶子", prompt_hint="桌上的水瓶")
    profile = model_manager.upsert_profile({"name": "本地 Ollama",
                                           "base_url": "http://x/v1", "model": "qwen2.5vl:7b"})
    monkeypatch.setattr(model_manager, "get_profile", lambda pid: dict(profile, image_limit=768))
    assert main(["prompt", dataset, "--images", "ghost.jpg", "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "[error]" in err and "ghost.jpg" in err and "Traceback" not in err
    assert load_batches(dataset, LIB_OBJECTS) == []
    assert load_index(dataset, LIB_OBJECTS).get("ghost.jpg") is None


def test_runtime_failures_also_report_plain_words(dataset, monkeypatch, capsys):
    """整批标注失败（显存不够、域名解析不了）也不该配一栈 traceback。

    本项目 ``raise RuntimeError`` 的地方全是「用户能自己解决」的运行时状况，
    文案本来就是一句人话，栈只会把它埋在最底下。
    """
    from autolabel import detect as dt
    from test_detect import use_backend

    def boom(*args, **kwargs):
        raise RuntimeError("0 张全部标注失败：显存不够")

    use_backend(monkeypatch)
    dt.add_model_classes(dataset, "yolo11n.pt", ["person"])
    monkeypatch.setattr(dt, "run_detect_labeling", boom)
    assert main(["detect", dataset, "--quiet"]) == 1
    err = capsys.readouterr().err
    assert "[error]" in err and "显存不够" in err and "Traceback" not in err


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


@pytest.mark.parametrize("command", ["export", "coords", "auto"])
def test_keep_status_accepts_comma_and_rejects_typos(dataset, source_dir, capsys, command):
    """``--keep-status`` 与兄弟参数一样认逗号，拼错的状态当场拦。

    以前写错一个字母不会报错，只会当成"没写"——用户以为多留了一批样本，
    其实导出范围一个字没变。
    """
    from autolabel.cli import _exclude_from_args, build_parser

    parser = build_parser()
    # auto 的图片来源是必填位置参数，先补一个占位
    head = [command, dataset] + ([source_dir] if command == "auto" else []) + ["--only-approved"]
    args = parser.parse_args([*head, "--keep-status", "rejected,edited"])
    assert _exclude_from_args(args) == ["unreviewed"]
    spaced = parser.parse_args([*head, "--keep-status", "rejected", "edited"])
    assert _exclude_from_args(spaced) == _exclude_from_args(args)

    capsys.readouterr()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args([*head, "--keep-status", "aprovied"])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "不认识的状态" in err and "rejected" in err and "Traceback" not in err


def test_delete_images_accepts_comma_separated_list(dataset, capsys):
    names = sorted(idx_names(dataset))
    for name in names:
        write_label(dataset, name, [SAMPLE17])

    assert main(["batches", dataset, "--delete-images", ",".join(names), "--lib", "pose"]) == 0
    out = capsys.readouterr().out

    assert "移除图片 2 张" in out
    assert idx_names(dataset) == set()
    # 只摘了姿态引用，提示词库与素材文件都该还在
    assert len(load_index(dataset, "objects").items) == 2
    assert os.path.exists(os.path.join(dataset, "images", names[0]))


def test_pool_add_accepts_comma_separated_list(dataset, capsys):
    names = sorted(idx_names(dataset))

    assert main(["pool", dataset, "--add", ",".join(names), "--lib", "detect"]) == 0
    out = capsys.readouterr().out

    assert "新增 2 张" in out and "池里查无 0 张" in out
    assert set(load_index(dataset, "detect").items) == set(names)


def test_pool_add_that_matches_nothing_fails_loudly(dataset, capsys):
    """池里查无此文件：不能顶着一行 [ok] + 退出码 0 过去，脚本里的 ``&&`` 会白过。"""
    assert main(["pool", dataset, "--add", "打错的名字.jpg", "--lib", "detect"]) == 1
    out = capsys.readouterr().out
    assert "[error]" in out and "素材池里没有这些文件" in out and "打错的名字.jpg" in out
    assert set(load_index(dataset, "detect").items) == set()


@pytest.mark.parametrize("lib,relabel,flag", [
    ("pose", "label", "--force"),
    ("objects", "prompt", "--overwrite"),
    ("detect", "detect", "--overwrite"),
])
def test_duplicate_import_hint_names_that_library_command(dataset, capsys, lib, relabel, flag):
    """重复导入时指路的命令必须是对着那个库的。

    以前三句都写成 ``autolabel label --force``——那是姿态库的重标命令，往检测库导
    素材的人照着敲，得到的是「姿态库还没有类别」之类毫不相干的第二句报错。
    """
    source = os.path.join(dataset, "images", sorted(idx_names(dataset))[0])
    main(["import", dataset, source, "--lib", lib])   # 先让它进库（或认领素材池）
    capsys.readouterr()
    assert main(["import", dataset, source, "--lib", lib]) == 0
    out = capsys.readouterr().out
    assert "没有新图入库" in out
    assert f"autolabel {relabel} {dataset} {flag}" in out
    # 另一种去处：素材池里已有、只是没进本库，得指去 pool --add
    assert "autolabel pool" in out


def test_pool_add_partial_still_exits_zero(dataset, capsys):
    """对一半错一半：加进去的那张算成功，但要单独点出没找到的名字。"""
    names = sorted(idx_names(dataset))
    assert main(["pool", dataset, "--add", f"{names[0]},打错的名字.jpg", "--lib", "detect"]) == 0
    out = capsys.readouterr().out
    assert "[ok]" in out and "新增 1 张" in out and "打错的名字.jpg" in out
    assert set(load_index(dataset, "detect").items) == {names[0]}


def test_batch_delete_accepts_comma_separated_ids(dataset, capsys):
    from autolabel.batches import create_batch

    ids = [create_batch(dataset, name=f"批次{i}", kind="image")["id"] for i in (1, 2)]

    assert main(["batches", dataset, "--delete", ",".join(ids), "--force"]) == 0
    out = capsys.readouterr().out

    assert "删除批次 2 个" in out
    assert ids[0] not in {row["id"] for row in batches_of(dataset)}


@pytest.mark.parametrize("command", [
    ["label", "{ds}", "--device", "cuda:1"],
    ["review", "{ds}", "--device", "cuda:1"],
    ["launch", "{ds}", "--device", "0"],
    ["service-run", "--device", "cuda:0"],
    ["auto", "{ds}", "{src}", "--device", "mps"],
    ["detect", "{ds}", "--device", "cuda:1"],
    ["settings", "--device", "cuda:1"],
])
def test_device_flag_accepts_a_card_number(dataset, command):
    """多卡机器上「一个实例跑一块卡」是常见需求，``--device cuda:1`` 以前
    会被 choices 拒成退出码 2，而帮助里明明写着可以填设备编号。"""
    from autolabel.cli import build_parser

    argv = [item.format(ds=dataset, src=dataset) for item in command]
    namespace = build_parser().parse_args(argv)
    assert namespace.device in ("cuda:1", "cuda:0", "0", "mps")


@pytest.mark.parametrize("value", ["cuda:9", "gp u", "metal", "tpu"])
def test_device_flag_still_rejects_nonsense(dataset, value):
    from autolabel.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["label", dataset, "--device", value])
