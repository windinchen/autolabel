"""导出训练集的图片放法：产物要能压缩、拷贝、带去别的机器继续训练。

用户报「导出 yolo 训练集，图片只有快捷方式，没法训练」。拿他自己那份产物验过：
默认软链接导出的目录原地能训，一压成 zip 每张图就只剩 **43 字节**的链接记录，
``data.yaml`` 里还写着这台电脑上的绝对路径，换台机器连目录都找不到。
"""

import json
import os
import shutil

import pytest
import yaml

from autolabel.cli import main
from autolabel.export import export_dataset, export_dir_links, materialize_export, place_mode

from _helpers import write_label

SAMPLE17 = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 17)
JPEG_MAGIC = b"\xff\xd8"


def labelled(dataset):
    rels = sorted(item for item in os.listdir(os.path.join(dataset, "images")))
    for rel in rels:
        write_label(dataset, rel, [SAMPLE17])
    names = sorted(os.listdir(os.path.join(dataset, "labels")))
    assert rels and names == [os.path.splitext(item)[0] + ".txt" for item in rels]
    return rels


def export_pose(dataset, **kwargs):
    rels = labelled(dataset)
    return export_dataset(dataset, **kwargs), rels


def relocate(out, tmp_path):
    """按「压缩再解压」的方式搬家：链接原样带走、不跟着目标走，而且换个层级。

    Python 自带的 ``shutil.copytree`` 默认会跟链接取内容，那样测试就漏掉了真实
    症状；``symlinks=True`` 加上更深的路径才等价于 zip 到别的机器上。
    """
    target = os.path.join(str(tmp_path), "m", "o", "d", os.path.basename(out))
    shutil.copytree(out, target, symlinks=True)
    return target


def images_of(export_dir):
    for split in ("train", "val"):
        base = os.path.join(export_dir, "images", split)
        for name in sorted(os.listdir(base)):
            yield os.path.join(base, name)


# --------------------------------------------------------------------------- //
# 默认放法：硬链接
# --------------------------------------------------------------------------- //

def test_default_export_leaves_real_files(dataset, tmp_path):
    """默认导出的图片必须是真实文件：同盘硬链接不占额外空间，压缩拷贝都带得走。"""
    result, rels = export_pose(dataset)
    assert result["mode"] == "hardlink" and result["mode_label"] == "硬链接"
    assert export_dir_links(result["out"]) == []
    pictures = list(images_of(result["out"]))
    assert len(pictures) == len(rels)
    for path in pictures:
        assert not os.path.islink(path), path
        assert os.stat(path).st_nlink >= 2, "硬链接应该和素材池共用同一份数据"
        with open(path, "rb") as fh:
            assert fh.read(2) == JPEG_MAGIC, path
    # 搬家之后还能读：这才是"能压去别的机器训练"的可执行定义
    moved = relocate(result["out"], tmp_path)
    for path in images_of(moved):
        with open(path, "rb") as fh:
            assert fh.read(2) == JPEG_MAGIC, path


def test_symlink_export_warns_that_the_pack_is_broken(dataset, tmp_path):
    """仍要软链接可以，但必须当场说清这份产物压走会废。"""
    result, _rels = export_pose(dataset, mode="symlink")
    assert result["mode"] == "symlink"
    assert export_dir_links(result["out"]), "这份导出应该全是软链接"
    assert "空壳" in result["mode_note"], result["mode_note"]
    moved = relocate(result["out"], tmp_path)
    broken = [path for path in images_of(moved) if not os.path.isfile(path)]
    assert broken, "软链接搬到别的层级后应当已经断链"


# --------------------------------------------------------------------------- //
# data.yaml：不许写死这台机器的路径
# --------------------------------------------------------------------------- //

def test_data_yaml_does_not_bake_in_this_machines_path(dataset):
    """写死绝对路径时，换机器训练第一步就报 images not found；省略 path 才会用本文件所在目录。"""
    result, _rels = export_pose(dataset)
    with open(os.path.join(result["out"], "data.yaml"), encoding="utf-8") as fh:
        text = fh.read()
    assert "path:" not in text, text
    data = yaml.safe_load(text)
    assert data["train"] == "images/train" and data["val"] == "images/val"
    assert "不写 path" in text, "留一行说明，下一个人才不会又把绝对路径加回去"


def test_detection_data_yaml_is_portable_too(dataset):
    """提示词标注 / 目标检测标注那套导出与姿态库同口径。"""
    from autolabel import fields as fd
    from autolabel import prompt as pr
    from autolabel.dataset import load_index

    fd.upsert_field(dataset, name="bottle", label="水瓶")
    pr.write_detections(dataset, "img0.jpg",
                        [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4)], lib="objects")
    idx = load_index(dataset, "objects")
    idx.upsert("img0.jpg", prompt_objects=1, prompt_status="unreviewed")
    idx.save()
    info = pr.export_detection_dataset(dataset, lib="objects")
    assert info["mode"] == "hardlink"
    with open(info["data_yaml"], encoding="utf-8") as fh:
        text = fh.read()
    assert "path:" not in text, text
    assert export_dir_links(info["out"]) == []


# --------------------------------------------------------------------------- //
# 旧导出就地救回来
# --------------------------------------------------------------------------- //

def test_materialize_swaps_old_links_for_real_files(dataset, tmp_path):
    """改默认之前导出的目录不必重标：就地换掉放法，划分和标签一个字节都不动。"""
    result, _rels = export_pose(dataset, mode="symlink")
    before = sorted(os.listdir(os.path.join(result["out"], "labels", "train")))
    report = materialize_export(result["out"])
    assert report["converted"] >= 2 and report["broken"] == []
    assert export_dir_links(result["out"]) == []
    assert report["placed_label"] == "硬链接"
    assert sorted(os.listdir(os.path.join(result["out"], "labels", "train"))) == before
    manifest = json.load(open(os.path.join(result["out"], "manifest.json"), encoding="utf-8"))
    assert manifest["mode"] == "hardlink" and manifest.get("materialized_at")
    moved = relocate(result["out"], tmp_path)
    for path in images_of(moved):
        with open(path, "rb") as fh:
            assert fh.read(2) == JPEG_MAGIC, path


def test_materialize_reports_dangling_links_instead_of_lying(dataset):
    """链接那头已经被删的救不回来，得把名字报出来，别报「已换好」。"""
    result, _rels = export_pose(dataset, mode="symlink")
    victim = export_dir_links(result["out"])[0]
    os.remove(victim)                                  # 删掉一个链接，少一张不影响别人
    os.symlink(os.path.join(dataset, "images", "根本没有这张.jpg"), victim + ".dangling")
    report = materialize_export(result["out"])
    assert any(name.endswith(".dangling") for name in report["broken"]), report
    # 悬空的那条救不回来，原样留着并报名字；其余的都得换成真实文件
    leftover = export_dir_links(result["out"])
    assert len(leftover) == 1 and leftover[0].endswith(".dangling"), leftover


def test_materialize_on_a_clean_dir_says_so(dataset):
    result, _rels = export_pose(dataset)
    report = materialize_export(result["out"])
    assert report["converted"] == 0 and "本来就是真实文件" in report["mode_note"]
    with pytest.raises(ValueError, match="导出目录不存在"):
        materialize_export(os.path.join(result["out"], "没有这个目录"))


# --------------------------------------------------------------------------- //
# 参数口径：界面 / 命令行 / 系统设置三处同一个默认
# --------------------------------------------------------------------------- //

def test_place_mode_validates_and_reads_system_settings(monkeypatch):
    assert place_mode() == "hardlink"
    assert place_mode("SYMLINK") == "symlink"
    monkeypatch.setenv("AUTOLABEL_EXPORT_MODE", "copy")
    assert place_mode(None) == "copy" and place_mode("") == "copy"
    with pytest.raises(ValueError, match="hardlink"):
        place_mode("快捷方式")


def test_cli_export_and_settings_share_the_setting(dataset, capsys):
    rels = labelled(dataset)
    assert main(["export", dataset, "--mode", "symlink"]) == 0
    out = capsys.readouterr().out
    assert "图片：软链接" in out, out
    target = [line.split(" -> ", 1)[1].split("（")[0] for line in out.splitlines()
              if line.startswith("[ok] 导出 -> ")][0]
    assert main(["export", dataset, "--materialize", target]) == 0
    second = capsys.readouterr().out
    assert "已换成硬链接" in second, second

    assert main(["settings", "--export-mode", "copy"]) == 0
    with open(os.path.expanduser(os.environ["AUTOLABEL_SETTINGS"]), encoding="utf-8") as fh:
        assert json.load(fh)["export_mode"] == "copy"
    assert main(["export", dataset]) == 0
    third = capsys.readouterr().out
    assert "图片：直接拷贝" in third, third
    assert rels


def test_settings_panel_offers_the_three_placements():
    from autolabel import settings as system

    names = [item["value"] for item in system.EXPORT_MODE_CHOICES]
    assert names == ["hardlink", "copy", "symlink"]
    static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "autolabel", "static", "common.js")
    with open(static, encoding="utf-8") as fh:
        script = fh.read()
    assert 'id="set-export-mode"' in script
    assert 'saveSettings({ export_mode: value })' in script
