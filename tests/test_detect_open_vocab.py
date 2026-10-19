"""开放词表权重（YOLOE / YOLO-World）：类别由用户现填的英文词决定。

推理照旧用替身。这里要守住的是三件事：

- 识别口径和 ultralytics 的路由一致（文件名里有 ``-world`` / ``yoloe`` 才算开放词表）；
- 权重自带的 COCO-80 不能再被当成"这次能标什么"摆到界面上；
- 中文词必须当场挡住——CLIP 文本编码器只认英文，填中文不报错、不警告，
  只是一个框都找不到（本机实测：set_classes(["人","狗","猫"]) → 零检出）。
"""
import sys
import types

import pytest

from autolabel import detect as dt
from autolabel import fields as fd
from autolabel import settings as system_settings
from autolabel.backends import (MAX_VOCAB_WORDS, BackendSettings, Detection,
                                DetectionBackend, clean_vocab, open_vocab_kind)

from test_detect import MODEL_CLASSES, use_backend

from autolabel.dataset import adopt_from_pool, load_index
from autolabel.library import LIB_DETECT, LIB_POSE


@pytest.fixture()
def ov_ds(dataset):
    """两帧素材同时进目标检测库（与 test_detect 的夹具同一做法，改名是为了不互相遮蔽）。"""
    rels = [rel for rel, _ in load_index(dataset, LIB_POSE).ordered_items()]
    adopt_from_pool(dataset, rels, lib=LIB_DETECT)
    return dataset


# --------------------------------------------------------------------------- #
# 识别与词表清洗
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename, kind", [
    ("yolov8s-world.pt", "YOLO-World"),
    ("yolov8x-worldv2.pt", "YOLO-World"),
    ("yoloe-26x-seg.pt", "YOLOE"),
    ("yoloe-11l-seg.pt", "YOLOE"),
    ("/abs/path/yolov8m-world.pt", "YOLO-World"),
    ("yolo11n.pt", ""),
    ("yolo11n-pose.pt", ""),
    ("yolov8x.pt", ""),
])
def test_open_vocab_kind_follows_the_ultralytics_naming(filename, kind):
    # ultralytics 的 YOLO() 按文件名里有没有 -world / yoloe 决定实例化哪个类。
    # 两边口径不一致就会出现「模型按开放词表加载、界面却按 COCO 显示类别」
    assert open_vocab_kind(filename) == kind


def test_clean_vocab_rejects_chinese_because_it_silently_finds_nothing():
    with pytest.raises(ValueError) as err:
        clean_vocab(["person", "狗"])
    assert "中文" in str(err.value)


def test_clean_vocab_dedupes_ignores_blanks_and_keeps_order():
    assert clean_vocab([" person ", "dog,,cat", "person", "", "  traffic   cone "]) == \
        ["person", "dog", "cat", "traffic cone"]


def test_clean_vocab_splits_on_the_separators_people_actually_type():
    # 中文输入法下的全角逗号、分号，以及整段粘进来的换行，都得切开：
    # 只认 ASCII 逗号的话，"person，dog" 会变成一个带逗号的词，模型谁也认不出
    assert clean_vocab(["person，dog；car\ncup"]) == ["person", "dog", "car", "cup"]


def test_clean_vocab_rejects_long_words_and_too_many_words():
    with pytest.raises(ValueError, match="最长"):
        clean_vocab(["a" * 60])
    with pytest.raises(ValueError, match="一次最多"):
        clean_vocab([f"cls{i}" for i in range(MAX_VOCAB_WORDS + 1)])
    with pytest.raises(ValueError, match="先填"):
        clean_vocab([])


# --------------------------------------------------------------------------- #
# 后端
# --------------------------------------------------------------------------- #
class StubWorldModel:
    """开放词表权重替身：set_classes 之后类别表就变成填进去的词。"""

    task = "detect"

    def __init__(self):
        self.names = dict(MODEL_CLASSES)        # 出厂自带那张 COCO 表
        self.calls = []

    def set_classes(self, classes):
        self.calls.append(list(classes))
        self.names = {index: name for index, name in enumerate(classes)}


def stub_world_backend(monkeypatch, model="yolov8s-world.pt"):
    from autolabel import backends

    stub = StubWorldModel()
    module = types.ModuleType("ultralytics")
    module.YOLO = lambda path: stub
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(backends, "ensure_weights", lambda name, **kwargs: f"/models/{name}")
    monkeypatch.setattr(backends, "resolve_device", lambda device: "cpu")
    return DetectionBackend(BackendSettings(model=model, device="cpu")), stub


def test_open_vocab_weight_does_not_pretend_the_baked_coco_table_is_available(monkeypatch):
    backend, _stub = stub_world_backend(monkeypatch)
    backend.load()
    assert backend.open_vocab == "YOLO-World"
    # 没填词表之前"能标什么"就是空的；把 COCO-80 摆出去等于骗人
    assert backend.class_names() == {}


def test_a_plain_weight_still_reports_its_own_class_table(monkeypatch):
    backend, _stub = stub_world_backend(monkeypatch, model="yolo11n.pt")
    backend.load()
    assert backend.open_vocab == ""
    assert backend.class_names() == dict(MODEL_CLASSES)


def test_set_vocab_maps_words_to_positions(monkeypatch):
    backend, stub = stub_world_backend(monkeypatch)
    names = backend.set_vocab(["safety helmet", "person", "dog"])
    assert stub.calls == [["safety helmet", "person", "dog"]]
    assert names == {0: "safety helmet", 1: "person", 2: "dog"}
    assert backend.class_names() == names


def test_set_vocab_is_a_no_op_when_the_words_are_unchanged(monkeypatch):
    backend, stub = stub_world_backend(monkeypatch)
    backend.set_vocab(["person", "dog"])
    backend.set_vocab(["person", "dog"])
    assert len(stub.calls) == 1                 # 文本嵌入只算一次
    backend.set_vocab(["person", "cat"])
    assert len(stub.calls) == 2


def test_a_plain_weight_says_why_it_cannot_take_a_vocab(monkeypatch):
    backend, _stub = stub_world_backend(monkeypatch, model="yolo11n.pt")
    backend.load()
    with pytest.raises(ValueError) as err:
        backend.set_vocab(["person"])
    text = str(err.value)
    assert "普通权重" in text and "开放词表" in text


def test_missing_text_encoder_becomes_an_actionable_chinese_message(monkeypatch):
    backend, stub = stub_world_backend(monkeypatch)
    backend.load()

    def boom(classes):
        raise ImportError(
            "Ultralytics requirement ['git+https://github.com/ultralytics/CLIP.git'] not found")

    monkeypatch.setattr(stub, "set_classes", boom)
    with pytest.raises(RuntimeError) as err:
        backend.set_vocab(["person"])
    text = str(err.value)
    assert "CLIP" in text and "联网" in text


# --------------------------------------------------------------------------- #
# 检测库：登记、配对、真正跑起来
# --------------------------------------------------------------------------- #
class FakeWorldBackend:
    """开放词表权重的检测库替身：填过词表才认类别，没填一个都不认。"""

    def __init__(self, model="yolov8s-world.pt", task="detect"):
        self.model_name = model
        self.task = task
        self.open_vocab = open_vocab_kind(model)
        self.vocab = []
        self.names = {}
        self.requests = []

    def describe(self):
        from autolabel.backends import TASK_NAMES

        return {"model": self.model_name, "task": self.task,
                "task_name": TASK_NAMES.get(self.task, self.task),
                "class_count": len(self.names), "device": "cpu",
                "open_vocab": self.open_vocab, "vocab": list(self.vocab)}

    def class_names(self):
        return dict(self.names)

    def set_vocab(self, words):
        self.vocab = clean_vocab(words)
        self.names = {index: word for index, word in enumerate(self.vocab)}
        return dict(self.names)

    def predict(self, image, class_ids=(), conf=None):
        wanted = [int(value) for value in class_ids] if class_ids else sorted(self.names)
        self.requests.append((tuple(wanted), conf))
        return [Detection(class_id=value, class_name=self.names[value], score=0.7,
                          cx=0.5, cy=0.5, w=0.2, h=0.3) for value in wanted]


def test_open_vocab_classes_come_from_what_the_user_types(ov_ds, monkeypatch):
    use_backend(monkeypatch, FakeWorldBackend())
    payload = dt.add_model_classes(ov_ds, "yolov8s-world.pt", ["traffic cone", "person"])
    assert payload["open_vocab"] == "YOLO-World"
    assert payload["not_in_model"] == []        # 开放词表不存在"权重认不出"
    assert {"traffic cone", "person"} <= {row["name"] for row in payload["fields"]}


def test_open_vocab_needs_the_words_before_it_can_register_anything(ov_ds, monkeypatch):
    use_backend(monkeypatch, FakeWorldBackend())
    with pytest.raises(ValueError, match="先填"):
        dt.add_model_classes(ov_ds, "yolov8s-world.pt", [])


def test_open_vocab_matching_is_positional_and_never_leaves_a_class_out(ov_ds, monkeypatch):
    use_backend(monkeypatch, FakeWorldBackend())
    dt.add_model_classes(ov_ds, "yolov8s-world.pt", ["person", "dog", "car"])
    matched = dt.match_model_classes(ov_ds, "yolov8s-world.pt")
    assert matched["missing"] == []
    assert matched["open_vocab"] == "YOLO-World"
    assert matched["class_count"] == 3
    assert {row["name"]: model_id for row, model_id in matched["pairs"]} == \
        {"person": 0, "dog": 1, "car": 2}


def test_a_chinese_class_in_the_registry_is_refused_before_the_run(ov_ds, monkeypatch):
    use_backend(monkeypatch, FakeWorldBackend())
    dt.add_model_classes(ov_ds, "yolov8s-world.pt", ["person"])
    # 登记之后手工把类名改成中文（模拟老库里留下中文名、或别的库导进来的表）
    registry = fd.load_registry(ov_ds, dt.LIB)
    registry["fields"][0]["name"] = "人"
    fd.save_registry(ov_ds, registry, dt.LIB)
    with pytest.raises(ValueError, match="中文"):
        dt.match_model_classes(ov_ds, "yolov8s-world.pt")


def test_run_applies_the_vocab_to_the_backend_that_actually_predicts(ov_ds, monkeypatch):
    backend = FakeWorldBackend()
    use_backend(monkeypatch, backend)
    dt.add_model_classes(ov_ds, "yolov8s-world.pt", ["person", "dog"])
    stats = dt.run_detect_labeling(ov_ds, model="yolov8s-world.pt", verbose=False)
    assert backend.vocab == ["person", "dog"]
    assert stats["open_vocab"] == "YOLO-World"
    assert stats["labeled"] == 2 and stats["boxes"] == 4


def test_a_plain_weight_still_goes_through_its_own_class_table(ov_ds, monkeypatch):
    # 普通权重走的是"类名去权重表里找"那条路，这条测试锁住两条路不会互相串味
    use_backend(monkeypatch)
    dt.add_model_classes(ov_ds, "yolo11n.pt", ["person", "bus"])
    matched = dt.match_model_classes(ov_ds, "yolo11n.pt")
    assert not matched.get("open_vocab")
    assert sorted(row["name"] for row, _ in matched["pairs"]) == ["bus", "person"]


def test_local_models_marks_open_vocab_weights(monkeypatch):
    from autolabel import models as model_manager

    present = ["yolo11n.pt", "yolov8s-world.pt", "yoloe-26x-seg.pt", "yolo11n-pose.pt"]
    monkeypatch.setattr(model_manager, "weight_names", lambda dir_=None: sorted(present))
    monkeypatch.setattr(model_manager, "model_status", lambda name, weights_dir=None: {
        "ready": True, "size_mb": 12.0, "path": f"/models/{name}"})
    rows = {row["name"]: row for row in dt.local_models()}
    assert rows["yolov8s-world.pt"]["open_vocab"] == "YOLO-World"
    assert rows["yoloe-26x-seg.pt"]["open_vocab"] == "YOLOE"
    assert rows["yoloe-26x-seg.pt"]["task"] == "segment"      # 分割档照样能用
    assert "开放词表" in rows["yoloe-26x-seg.pt"]["task_name"]
    assert rows["yolo11n.pt"]["open_vocab"] == ""


# --------------------------------------------------------------------------- #
# 系统设置里的依赖状况
# --------------------------------------------------------------------------- #
def test_settings_payload_reports_open_vocab_readiness(monkeypatch):
    from autolabel import backends

    monkeypatch.setattr(backends, "text_encoder_status", lambda: {
        "installed": False, "cached_bytes": 353976522, "note": "第一次用会自动装好"})
    monkeypatch.setattr(system_settings, "detect", lambda: {
        value: {"available": value == "cpu", "reason": "", "names": [], "label": value}
        for value in ("auto", "cuda", "mps", "cpu")})
    data = system_settings.payload(probe=True)
    assert data["open_vocab"]["cached_bytes"] == 353976522
    text = system_settings.describe(data)
    assert "开放词表" in text and "338" in text


def test_settings_payload_skips_the_import_when_not_probing(monkeypatch):
    from autolabel import backends

    calls = []
    monkeypatch.setattr(backends, "text_encoder_status", lambda: calls.append(1) or {})
    system_settings.payload(probe=False)
    assert calls == []      # import clip 会连带拉起 torch，平时切页不该等它
