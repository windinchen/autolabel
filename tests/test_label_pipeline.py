"""重标不该吃掉人工结果，QC 几何判断要按像素算，框要按真正参与监督的关键点定。"""

import pytest

from autolabel import backends as be
from autolabel import pipeline
from autolabel.backends import BackendSettings
from autolabel.dataset import label_path_for, load_index
from autolabel.yolo_pose import (Keypoint, Person, merge_person_flips, parse_label_file,
                        person_sanity_flags)


def make_person(cx=0.5, cy=0.5, w=0.2, h=0.4, score=0.9, n=17, v=2, conf=0.9):
    return Person(cls=0, cx=cx, cy=cy, w=w, h=h, score=score,
                  keypoints=[Keypoint(cx, min(cy + i * 0.01, 1), v, conf) for i in range(n)])


class FakeEnsemble:
    """替换真实推理：每回都返回同一个人，只看流水线怎么决定「标谁 / 覆盖谁」。"""

    def __init__(self, settings):
        self.settings = settings

    def load(self):
        return self

    @property
    def display_names(self):
        return [f"k{i}" for i in range(17)]

    @property
    def source_tag(self):
        return "fake"

    def infer(self, image):
        return [make_person()]


@pytest.fixture()
def fake_backend(monkeypatch):
    monkeypatch.setattr(pipeline, "build_backend", lambda settings: FakeEnsemble(settings))


def label(root, **kwargs):
    return pipeline.label_dataset(root, BackendSettings(device="cpu"), verbose=False, **kwargs)


def approve_with_human_edit(root, rel):
    """模拟人在复核界面改过骨骼并点了接受。"""
    path = label_path_for(root, rel)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("0 0.11 0.22 0.33 0.44 " + " ".join(["0.5 0.5 2"] * 17) + "\n")
    idx = load_index(root)
    idx.upsert(rel, status="approved")
    idx.save()
    return path


def test_human_approved_edits_survive_param_change(dataset, fake_backend, capsys):
    """换模型 / 换 imgsz 重标：人工接受过的标签文件与状态都不许被模型默默盖掉。"""
    assert label(dataset)["labeled"] == 2
    rel = next(iter(load_index(dataset).items))
    path = approve_with_human_edit(dataset, rel)
    human_text = open(path, encoding="utf-8").read()

    stats = pipeline.label_dataset(dataset, BackendSettings(device="cpu", imgsz=960), verbose=False)
    assert stats["labeled"] == 1  # 只重标没被人工碰过的那张
    assert stats["skipped_human"] == 1
    assert open(path, encoding="utf-8").read() == human_text
    assert load_index(dataset).get(rel)["status"] == "approved"
    assert len(parse_label_file(path, 17)) == 1
    assert parse_label_file(path, 17)[0].cx == pytest.approx(0.11)


def test_force_still_relabels_human_reviewed(dataset, fake_backend):
    """`--force` 是明确说「连人工的一起重标」，这时才允许覆盖。"""
    label(dataset)
    rel = next(iter(load_index(dataset).items))
    path = approve_with_human_edit(dataset, rel)
    stats = pipeline.label_dataset(dataset, BackendSettings(device="cpu"), only_missing=False,
                                   verbose=False)
    assert stats["skipped_human"] == 0
    assert stats["labeled"] == 2
    assert not open(path, encoding="utf-8").read().startswith("0 0.11")


def test_edited_by_hand_is_proticed_too(dataset, fake_backend):
    """状态只到「已修改」（还没点接受）同样是人工成果，一样不许盖。"""
    label(dataset)
    rel = next(iter(load_index(dataset).items))
    path = label_path_for(dataset, rel)
    open(path, "w", encoding="utf-8").write("0 0.9 0.9 0.05 0.05 " + " ".join(["0.9 0.9 2"] * 17) + "\n")
    idx = load_index(dataset)
    idx.upsert(rel, status="edited")
    idx.save()
    stats = pipeline.label_dataset(dataset, BackendSettings(device="cpu", conf=0.4), verbose=False)
    assert stats["skipped_human"] == 1
    assert open(path, encoding="utf-8").read().startswith("0 0.9 0.9")


def test_resume_survives_interrupted_run_with_new_params(dataset, fake_backend):
    """中断时索引头部的参数指纹没来得及更新：已按新参数标好的那些仍要被跳过。"""
    settings = BackendSettings(device="cpu", imgsz=416)
    assert pipeline.label_dataset(dataset, settings, verbose=False)["labeled"] == 2
    idx = load_index(dataset)
    idx.data["settings"] = {"signature": "旧参数留下的指纹"}  # 模拟跑到一半被强杀
    idx.save()
    again = pipeline.label_dataset(dataset, settings, verbose=False)
    assert again["labeled"] == 0
    assert again["total"] == 0


def test_all_human_reviewed_explains_why_nothing_ran(dataset, fake_backend, capsys):
    label(dataset)
    for rel in list(load_index(dataset).items):
        approve_with_human_edit(dataset, rel)
    capsys.readouterr()
    stats = pipeline.label_dataset(dataset, BackendSettings(device="cpu", imgsz=960), verbose=True)
    assert stats["labeled"] == 0
    assert "人工已复核 2 张" in capsys.readouterr().out


def person_from_pixels(w_px, h_px, img_w, img_h):
    """按像素框造一个人（w/h 是归一化值，换个图片尺寸就得换算一次）。"""
    return make_person(w=w_px / img_w, h=h_px / img_h)


def test_odd_aspect_uses_pixel_geometry():
    """长宽比要看像素：同一副骨架换张图（横屏 / 竖屏）不该改结论。"""
    # 100x450 像素（比例 4.5）本来正常，但在 16:9 横屏里归一化比例被拉到 8.0 -> 以前误报
    assert "odd_aspect" not in person_sanity_flags(person_from_pixels(100, 450, 1920, 1080), 1920, 1080)
    # 100x900 像素（比例 9）明显是断链，竖屏视频里归一化只剩 5.1 -> 以前漏报
    assert "odd_aspect" in person_sanity_flags(person_from_pixels(100, 900, 1080, 1920), 1080, 1920)
    # 同样的像素比例，横屏竖屏必须给同一个结论
    assert "odd_aspect" not in person_sanity_flags(person_from_pixels(100, 450, 1080, 1920), 1080, 1920)
    assert "odd_aspect" in person_sanity_flags(person_from_pixels(100, 900, 1920, 1080), 1920, 1080)


def test_box_expands_only_over_supervised_keypoints(monkeypatch):
    """conf 低到判为 v=0 的杂点不该再把包围盒撑大——框要和落盘标签里的 v 一致。"""
    person = make_person(cx=0.5, cy=0.5, w=0.1, h=0.2, score=0.9)
    person.keypoints[3] = Keypoint(x=0.95, y=0.95, v=2, conf=0.02)  # 远处的杂点，阈值下该判 v=0
    backend = be.PoseBackend(BackendSettings(flip_tta=False, visible_conf=0.35, bbox_margin=0.0))
    monkeypatch.setattr(backend, "_predict", lambda image, flipped=False: [person])
    out = backend.infer(__import__("numpy").zeros((200, 200, 3), dtype="uint8"))
    assert out[0].keypoints[3].v == 0
    box = out[0].xyxy(200, 200)
    assert box[2] < 0.95 * 200  # 框的右边界没被那个不监督的杂点拖过去


def test_ensemble_merge_box_follows_merged_keypoints():
    """两个模型融合后骨架会往中间挪，框也得跟着挪，否则自己造出 keypoints_outside_box。"""
    primary = make_person(cx=0.3, cy=0.5, w=0.1, h=0.4, score=0.9)
    secondary = make_person(cx=0.7, cy=0.5, w=0.1, h=0.4, score=0.9)
    out = merge_person_flips(primary, secondary)
    assert out.cx == pytest.approx(0.5)
    assert out.w == pytest.approx(0.1)
    assert "odd_aspect" not in person_sanity_flags(out, 1920, 1080)
