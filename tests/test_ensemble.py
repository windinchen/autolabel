"""多模型共识与翻转融合的逻辑测试（不依赖权重）。"""

import pytest

from autolabel import backends as be
from autolabel.yolo_pose import Keypoint, Person


def make_person(cx, cy, w=0.2, h=0.4, score=0.9, n=17, v=2):
    return Person(cx=cx, cy=cy, w=w, h=h, score=score,
                  keypoints=[Keypoint(min(max(cx + i * 0.001, 0), 1), min(max(cy + i * 0.01, 0), 1), v, 0.8)
                             for i in range(n)])


class FakeBackend:
    """替换真实推理，只验证融合/打标逻辑。"""

    def __init__(self, persons, settings):
        self._persons = persons
        self.settings = settings
        self.keypoint_names = tuple(f"k{i}" for i in range(len(persons[0].keypoints))) if persons else ()
        self.num_keypoints = len(persons[0].keypoints) if persons else 0

    def load(self):
        return self

    @property
    def display_names(self):
        return list(self.keypoint_names)

    def infer(self, image):
        return [Person(cls=p.cls, cx=p.cx, cy=p.cy, w=p.w, h=p.h, score=p.score,
                       keypoints=list(p.keypoints)) for p in self._persons]


def build_ensemble(groups, settings):
    ensemble = be.PoseEnsemble(settings)
    ensemble.backends = [FakeBackend(g, settings) for g in groups]
    return ensemble


def test_confirmed_when_models_agree():
    settings = be.BackendSettings(visibility_policy="all_visible")
    ensemble = build_ensemble([[make_person(0.5, 0.5)], [make_person(0.5, 0.5, score=0.7)]], settings)
    out = ensemble.infer(_image())
    assert len(out) == 1
    assert "multi_model_confirmed" in out[0].flags


def test_secondary_only_detection_is_flagged():
    """辅助模型多检出一个人：补进结果并标记来源，便于人工确认是否误检。"""
    settings = be.BackendSettings(visibility_policy="all_visible")
    primary = [make_person(0.2, 0.5)]
    secondary = [make_person(0.2, 0.5), make_person(0.8, 0.5)]
    out = build_ensemble([primary, secondary], settings).infer(_image())
    assert len(out) == 2
    assert sum("only_secondary_model" in p.flags for p in out) == 1
    assert not any("model_disagreement" in p.flags for p in out)


def test_primary_detection_without_second_opinion_is_flagged():
    """主模型检出但辅助模型没 matched 的目标：标成分歧，交给复核界面优先曝光。"""
    settings = be.BackendSettings(visibility_policy="all_visible")
    primary = [make_person(0.2, 0.5), make_person(0.8, 0.5)]
    secondary = [make_person(0.2, 0.5)]
    out = build_ensemble([primary, secondary], settings).infer(_image())
    assert len(out) == 2
    assert sum("model_disagreement" in p.flags for p in out) == 1
    assert not any("only_secondary_model" in p.flags for p in out)
    # 被两个模型共同确认的那个保留了更高的分数
    assert max(p.score for p in out) == pytest.approx(0.9)


def test_flip_fusion_prefers_high_confidence_keypoints():
    settings = be.BackendSettings()
    forward = [make_person(0.5, 0.5, score=0.9)]
    forward[0].keypoints[0] = Keypoint(0.50, 0.30, 2, 0.9)
    mirrored = [make_person(0.5, 0.5, score=0.6)]
    mirrored[0].keypoints[0] = Keypoint(0.54, 0.32, 2, 0.1)
    fused = be.fuse_flips(forward, mirrored, 1000, 1000, settings)
    assert len(fused) == 1
    assert fused[0].keypoints[0].x == pytest.approx(0.504, abs=1e-3)
    assert fused[0].score == pytest.approx(0.9)


def test_flip_fusion_keeps_unmatched_mirrored_detection():
    settings = be.BackendSettings()
    fused = be.fuse_flips([make_person(0.2, 0.5)], [make_person(0.2, 0.5), make_person(0.85, 0.5)],
                          1000, 1000, settings)
    assert len(fused) == 2


def test_visibility_policies():
    person = make_person(0.5, 0.5)
    for k, conf in zip(person.keypoints, [0.9, 0.2, 0.05] + [0.5] * 14):
        k.conf = conf
    settings = be.BackendSettings(visibility_policy="threshold", visible_conf=0.35, occluded_conf=0.15)
    be.assign_visibility(person, settings)
    assert [person.keypoints[0].v, person.keypoints[1].v, person.keypoints[2].v] == [2, 1, 0]

    person2 = make_person(0.5, 0.5)
    person2.keypoints[1].conf = 0.2
    be.assign_visibility(person2, be.BackendSettings(visibility_policy="binary", visible_conf=0.35))
    assert person2.keypoints[1].v == 0

    person3 = make_person(0.5, 0.5)
    person3.keypoints[2].conf = 0.01
    be.assign_visibility(person3, be.BackendSettings(visibility_policy="all_visible"))
    assert person3.keypoints[2].v == 2


def test_finalize_box_modes():
    person = make_person(0.5, 0.5, w=0.05, h=0.05)
    person.keypoints = [Keypoint(0.2, 0.2, 2, 0.9), Keypoint(0.8, 0.8, 2, 0.9)]
    kept = be.finalize_box(make_person(0.5, 0.5, w=0.05, h=0.05), 1000, 1000,
                           be.BackendSettings(box_mode="model"))
    assert kept.w == pytest.approx(0.05)
    expanded = be.finalize_box(person, 1000, 1000, be.BackendSettings(box_mode="expand", bbox_margin=0.0))
    assert expanded.w * 1000 >= 600
    by_kpt = be.finalize_box(person, 1000, 1000, be.BackendSettings(box_mode="keypoints", bbox_margin=0.0))
    assert by_kpt.w * 1000 == pytest.approx(600, abs=1)


def _image():
    import numpy as np

    return np.zeros((100, 100, 3), dtype="uint8")
