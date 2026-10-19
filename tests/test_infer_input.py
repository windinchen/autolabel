"""喂给模型的东西：色彩顺序与左右关键点配对。

这两件事错了都不报错，只会让标注质量悄悄变差，最难查：

1. ultralytics 收到 ndarray 时**不做任何色彩转换**，直接当 BGR 用（它自己的
   加载器 ``data/loaders.py`` 写明假定 BGR）。我们读图拿到的是 RGB，所以进模型
   前必须正好转一次；以前姿态后端在 ``infer`` 和 ``_predict`` 各转了一次，
   等于把原始 RGB 交进去，R/B 长期互换。
2. 镜像融合（flip TTA）要把镜像结果反映射回原图方向，左右关键点得按**模型自己**
   的配对表换下标。一律套 COCO 表的话，自定义关键点权重的左肩会和第二根什么
   么都可能是右手指平均到一起。
"""
import numpy as np
import pytest

from autolabel import backends as be
from autolabel.yolo_pose import COCO17_FLIP_IDX, Keypoint, Person, mirror_person


class NoDetections:
    """最小结果对象：姿态后端只看 boxes / keypoints / orig_shape。"""

    orig_shape = (8, 8)
    boxes = None
    keypoints = None


class RecordingModel:
    def __init__(self):
        self.sources = []

    def predict(self, *args, **kwargs):
        # 检测后端按位置传图，姿态后端用 source=，两种都得接住
        self.sources.append(args[0] if args else kwargs["source"])
        return [NoDetections()]


@pytest.fixture()
def pose_backend(monkeypatch):
    backend = be.PoseBackend(be.BackendSettings(flip_tta=False, device="cpu"))
    fake = RecordingModel()
    monkeypatch.setattr(backend, "load", lambda: fake)
    backend.fake = fake
    return backend


def _red_on_the_left_corner():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 10            # R
    rgb[..., 2] = 30            # B
    return rgb


def test_pose_backend_hands_the_model_bgr_exactly_once(pose_backend):
    import cv2

    rgb = _red_on_the_left_corner()
    pose_backend._predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    given = pose_backend.fake.sources[0]
    assert given[..., 0][0][0] == 30, "蓝通道该是蓝通道：再转一次就把 R/B 换回去了"
    assert given[..., 2][0][0] == 10


def test_detection_backend_hands_the_model_bgr_once():
    """检测后端只转一次，两条后端口径必须一致，否则同一张图两个库分数没法比。"""
    backend = be.DetectionBackend(be.BackendSettings(model="yolo11n.pt", device="cpu"))
    fake = RecordingModel()
    backend.load = lambda: fake
    backend.names = {0: "person"}
    rgb = _red_on_the_left_corner()
    try:
        backend.predict(rgb)
    except Exception:                              # noqa: BLE001 结果对象够用了，别的全忽略
        pass
    given = fake.sources[0]
    assert given[..., 0][0][0] == 30 and given[..., 2][0][0] == 10
    assert not np.array_equal(given, rgb), "检测后端拿到的是 RGB，必须真的转一次"


@pytest.mark.parametrize("raw,count,wanted", [
    ([0, 2, 1], 3, [0, 2, 1]),                                  # 模型自带就用它
    (list(COCO17_FLIP_IDX), 17, list(COCO17_FLIP_IDX)),
    (None, 5, list(COCO17_FLIP_IDX[:5])),                       # 缺了就退回 COCO
    ([0, 1, 9], 3, list(COCO17_FLIP_IDX[:3])),                  # 下标越界不算数
    ([0, 0, 1], 3, list(COCO17_FLIP_IDX[:3])),                  # 不是排列不算数
    ([0, 1], 3, list(COCO17_FLIP_IDX[:3])),                     # 长度不齐不算数
    (["a", 1], 2, list(COCO17_FLIP_IDX[:2])),                   # 脏数据不算数
])
def test_flip_indices_only_trusts_a_complete_table(raw, count, wanted):
    assert be.flip_indices(raw, count) == wanted


def test_mirror_person_follows_the_models_own_pairing():
    """自定义权重的 1/2 号点是反着配的：套 COCO 表会把左右平均错一对。"""
    custom = [2, 1, 0]
    person = Person(cls=0, cx=0.2, cy=0.5, w=0.2, h=0.4, score=0.9, keypoints=[
        Keypoint(x=0.10, y=0.1, v=2, conf=0.9),
        Keypoint(x=0.20, y=0.2, v=2, conf=0.8),     # 模型口径里的"左"
        Keypoint(x=0.30, y=0.3, v=2, conf=0.7),
    ])
    by_model = mirror_person(person, 100, 100, flip_idx=custom)
    by_coco = mirror_person(person, 100, 100, flip_idx=list(COCO17_FLIP_IDX[:3]))
    # 自定义表：0 号点的镜像落到 2 号位；COCO 表：0 号点原地不动
    assert [k.x for k in by_model.keypoints] == pytest.approx([0.70, 0.80, 0.90])
    assert [k.x for k in by_coco.keypoints] == pytest.approx([0.90, 0.70, 0.80])


def test_mirror_person_ignores_a_mismatched_table():
    """拿别个模型的表（长度对不上）时退回 COCO，别把下标越界抛给用户。"""
    person = Person(cls=0, cx=0.2, cy=0.5, w=0.2, h=0.4, score=0.9,
                    keypoints=[Keypoint(x=0.1, y=0.1, v=2, conf=0.9)] * 3)
    assert [k.x for k in mirror_person(person, 100, 100, flip_idx=[2, 1]).keypoints] == \
        [k.x for k in mirror_person(person, 100, 100).keypoints]
