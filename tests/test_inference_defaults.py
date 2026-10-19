"""系统设置里的推理默认值 + 「差一点就过阈值」的候选回收（v1.4.7）。

两件事是同一场故障的两面：开放词表标一张白 T 恤照，界面只写一句"没找到"，
其实模型给了 ``shirt`` 0.026 分——既没人告诉用户该降阈值，界面上也没有地方
把输入分辨率提上去（实测同一张图 640→1024，这个词从 0.026 变成 0.553）。
这一组守住四条：

* 默认值只有「系统设置」一处能改，命令行与界面任务显式填的才算覆盖；
* 存盘时坏值当场挡住，读文件时坏值退回默认（一个手写坏的文件不该弄崩服务）；
* 阈值挡掉的候选要交回来给人看，但**不落盘**，标注库里不能混进废框；
* 已经在别的图上出过框的类别不算"差一点"，别把用户往错方向引。
"""
import json
import os
import types

import numpy as np
import pytest

from autolabel import detect as dt
from autolabel import settings as system
from autolabel.backends import BackendSettings, DetectionBackend
from autolabel.dataset import adopt_from_pool, label_path_for, load_index
from autolabel.library import LIB_DETECT, LIB_POSE

from test_detect import FakeBackend
from test_settings import probe_off  # noqa: F401  复用"只有 CPU"的探测夹具

assert probe_off is not None


@pytest.fixture()
def detect_ds(dataset):
    """两帧素材进目标检测库（与 test_detect 同一做法）。"""
    rels = [rel for rel, _ in load_index(dataset, LIB_POSE).ordered_items()]
    adopt_from_pool(dataset, rels, lib=LIB_DETECT)
    return dataset


def first_rel(root):
    return [rel for rel, _ in load_index(root, LIB_DETECT).ordered_items()][0]


def use_backend(monkeypatch, backend):
    """把推理层整个换掉，同时记下检测层真正建出来的那份参数。"""
    seen = {}

    def spy(settings, dir_=None):
        seen["settings"] = settings
        return backend

    monkeypatch.setattr(dt, "_backend", spy)
    return seen


class MixedBackend(FakeBackend):
    """第一个类别稳过阈值，其余类别全部只有 0.05 分：模拟"词对了但部分图没有"。"""

    def predict(self, image, class_ids=(), conf=None, floor=None):
        wanted = [int(value) for value in class_ids] if class_ids else sorted(self.names)
        found = []
        for offset, class_id in enumerate(wanted):
            if class_id not in self.names:
                continue
            found.append(_row(class_id, self.names[class_id],
                              0.9 if offset == 0 else 0.05, floor, conf))
        return found


class SpreadBackend(FakeBackend):
    """一次出好几个互不重叠的低分框，分数逐个降低：用来看候选的排序与上限。"""

    def predict(self, image, class_ids=(), conf=None, floor=None):
        from autolabel.backends import Detection

        wanted = [int(value) for value in class_ids] if class_ids else sorted(self.names)
        out = []
        for offset, class_id in enumerate(wanted):
            if class_id not in self.names:
                continue
            score = round(0.20 - 0.01 * offset, 4)
            out.append(Detection(class_id=class_id, class_name=self.names[class_id],
                                 score=score, cy=0.5, w=0.05, h=0.05,
                                 cx=0.05 + 0.9 * offset / max(1, len(wanted) - 1),
                                 near_miss=floor is not None and conf is not None
                                 and score < conf))
        return out


def _row(class_id, name, score, floor, conf):
    from autolabel.backends import Detection

    return Detection(class_id=class_id, class_name=name, score=score,
                     cx=0.5, cy=0.5, w=0.2, h=0.3,
                     near_miss=floor is not None and conf is not None and score < conf)


# --------------------------------------------------------------------------- //
# 一、默认值只有系统设置一处
# --------------------------------------------------------------------------- //
def test_label_defaults_carry_the_two_new_knobs():
    system.save({"imgsz": 1024, "conf": 0.05})
    assert system.label_defaults() == {"device": "auto", "half": False,
                                       "imgsz": 1024, "conf": 0.05}


def test_detect_follows_the_saved_resolution():
    """界面上把分辨率调高，检测就必须真按 1024 建后端——存了没人读是最坏的一种假。"""
    system.save({"imgsz": 1024})
    built = dt.system_settings()
    assert built.imgsz == 1024 and built.conf == 0.25


def test_explicit_arguments_beat_the_saved_ones():
    system.save({"imgsz": 1280, "conf": 0.4})
    built = dt.system_settings(imgsz=640, conf=0.05)
    assert built.imgsz == 640 and built.conf == 0.05
    # 空值不算"显式指定"：命令行没填就该用系统设置，不能被空串顶掉
    assert dt.system_settings(imgsz=None, device="").imgsz == 1280


def test_the_batch_run_actually_uses_the_saved_resolution(detect_ds, monkeypatch):
    system.save({"imgsz": 1024, "conf": 0.05})
    seen = use_backend(monkeypatch, FakeBackend())
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    stats = dt.run_detect_labeling(detect_ds, verbose=False)
    assert seen["settings"].imgsz == 1024 and stats["imgsz"] == 1024
    assert seen["settings"].conf == 0.05 and stats["conf"] == 0.05


def test_a_per_run_conf_override_wins(detect_ds, monkeypatch):
    system.save({"conf": 0.05})
    seen = use_backend(monkeypatch, FakeBackend())
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    stats = dt.run_detect_labeling(detect_ds, conf=0.6, imgsz=640, verbose=False)
    assert seen["settings"].conf == 0.6 and stats["conf"] == 0.6
    assert seen["settings"].imgsz == 640


def test_raising_the_resolution_relabels_instead_of_claiming_already_done(detect_ds, monkeypatch):
    """参数指纹里必须带分辨率：不然提了分辨率，几千张图全被"同参数已标过"跳过。"""
    use_backend(monkeypatch, FakeBackend())
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    first = dt.run_detect_labeling(detect_ds, imgsz=640, verbose=False)
    assert first["labeled"] == 2
    again = dt.run_detect_labeling(detect_ds, imgsz=640, verbose=False)
    assert again["protected"] == 2 and again["labeled"] == 0, "同参数就该跳过"
    lifted = dt.run_detect_labeling(detect_ds, imgsz=1024, verbose=False)
    assert lifted["protected"] == 0 and lifted["labeled"] == 2, "换了分辨率就是换了参数，要重标"


# --------------------------------------------------------------------------- //
# 二、坏值进不来，坏文件压不垮
# --------------------------------------------------------------------------- //
@pytest.mark.parametrize("value", ["1024", 1024, 1024.0, " 768 "])
def test_sane_resolutions_are_accepted(value):
    assert system.as_imgsz(value) in (768, 1024)


@pytest.mark.parametrize("value", ["", "abc", 0, 64, 4096, None])
def test_broken_resolutions_fall_back_but_saving_them_is_an_error(value):
    assert system.as_imgsz(value) == system.DEFAULTS["imgsz"]
    with pytest.raises(ValueError) as caught:
        system.as_imgsz(value, strict=True)
    assert "输入分辨率" in str(caught.value)


def test_confidence_is_refused_before_it_silently_eats_every_box():
    """填 0.99 的症状是"所有框都没了"，比当场报错难查十倍。"""
    assert system.as_conf("0.05") == 0.05
    for bad in ("1.5", "0", "-1", "看你的", None):
        with pytest.raises(ValueError) as caught:
            system.as_conf(bad, strict=True)
        assert "置信度" in str(caught.value)
    assert system.as_conf("看你的") == system.DEFAULTS["conf"]


def test_a_hand_broken_file_still_reads_a_whole_settings(tmp_path):
    with open(system.path(), "w", encoding="utf-8") as fh:
        fh.write('{"device": "cpu", "imgsz": "很大", "conf": 99}')
    assert system.load() == {"device": "cpu", "half": False, "imgsz": 640, "conf": 0.25}


def test_launch_arguments_can_pin_resolution_for_one_instance_only(monkeypatch):
    system.save({"imgsz": 640, "conf": 0.25})
    monkeypatch.setenv("AUTOLABEL_IMGSZ", "1024")
    monkeypatch.setenv("AUTOLABEL_CONF", "0.02")
    assert system.load()["imgsz"] == 1024 and system.load()["conf"] == 0.02
    # 存盘时不能被本进程的启动参数改写：那是临时值，不是用户真正选过的东西
    assert system.load(ignore_env=True) == {"device": "auto", "half": False,
                                            "imgsz": 640, "conf": 0.25}


# --------------------------------------------------------------------------- //
# 三、阈值挡掉的候选要看得见，但不能落盘
# --------------------------------------------------------------------------- //
def test_near_misses_are_reported_without_entering_the_labels(detect_ds, monkeypatch):
    use_backend(monkeypatch, FakeBackend(score=0.05))
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    stats = dt.run_detect_labeling(detect_ds, conf=0.25, verbose=False)

    assert stats["boxes"] == 0
    assert stats["near_miss"] == [{"name": "person", "score": 0.05}]
    rel = first_rel(detect_ds)
    assert load_index(detect_ds, LIB_DETECT).get(rel)["detect_objects"] == 0
    assert "none_found" in (load_index(detect_ds, LIB_DETECT).get(rel)["detect_flags"])
    path = label_path_for(detect_ds, rel, LIB_DETECT)
    assert not os.path.exists(path) or open(path, encoding="utf-8").read().strip() == ""


def test_the_hint_says_the_score_and_what_to_do_about_it(detect_ds, monkeypatch):
    use_backend(monkeypatch, FakeBackend(score=0.026))
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    stats = dt.run_detect_labeling(detect_ds, conf=0.25, verbose=False)
    hint = dt.near_miss_hint(stats)
    assert "person 0.026" in hint
    assert "0.025" in hint, "提示里要给一个能直接填进去的阈值"
    assert "1024" in hint and "置信度" in hint


def test_a_class_that_hit_somewhere_else_is_not_called_a_near_miss(detect_ds, monkeypatch):
    """同一次任务里出过框的类别别再报"差一点"：那是别的图上没有这个东西。"""
    use_backend(monkeypatch, MixedBackend())
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person", "bicycle"])
    stats = dt.run_detect_labeling(detect_ds, conf=0.25, verbose=False)
    assert stats["boxes"] > 0
    assert len(stats["near_miss"]) == 1, "只有一个类别全程没出框"
    assert dt.near_miss_hint({"near_miss": [], "conf": 0.25}) == ""


def test_near_miss_report_is_capped_and_ordered(detect_ds, monkeypatch):
    names = {index: f"cls{index}" for index in range(dt.NEAR_MISS_TOP + 4)}
    use_backend(monkeypatch, SpreadBackend(names=names))
    wanted = list(names.values())
    dt.add_model_classes(detect_ds, "yolo11n.pt", wanted)
    stats = dt.run_detect_labeling(detect_ds, conf=0.25, verbose=False)
    assert len(stats["near_miss"]) == dt.NEAR_MISS_TOP, "候选要报得完，也要报得不多"
    scores = [row["score"] for row in stats["near_miss"]]
    assert scores == sorted(scores, reverse=True), "最接近阈值的排最前"


# --------------------------------------------------------------------------- //
# 四、真实后端的 floor 语义（不联网、不下载权重）
# --------------------------------------------------------------------------- //
class _Row(list):
    def tolist(self):
        return list(self)


class _Boxes:
    """检测结果里那个可迭代张量：真实对象有长度，替身也得有。"""

    def __init__(self, scores):
        self.conf = list(scores)
        self.cls = [0] * len(scores)
        self.xyxy = [_Row([10.0, 10.0, 60.0, 90.0]) for _ in scores]

    def __len__(self):
        return len(self.conf)


class _Model:
    """最像 ultralytics 结果的替身：只把 ``conf`` 与框的读数当真。"""

    def __init__(self, scores, masks=None):
        self.scores = scores
        self.masks = masks
        self.calls = []

    def predict(self, image, **kwargs):
        """真实模型会按传进去的 conf 自己滤框，替身不滤就测不出 floor 的区别。"""
        self.calls.append(kwargs)
        threshold = float(kwargs.get("conf") or 0)
        kept = [value for value in self.scores if value >= threshold]
        rows = self.masks[:len(kept)] if self.masks is not None else None
        self.last_rows = len(kept)
        return [types.SimpleNamespace(orig_shape=(100, 160), boxes=_Boxes(kept),
                                      masks=types.SimpleNamespace(data=rows))]


class _StubBackend(DetectionBackend):
    def __init__(self, scores, masks=None):
        super().__init__(BackendSettings(device="cpu", conf=0.25), "fake.pt")
        self.names = {0: "t-shirt"}
        self.task = "detect"
        self._stub = _Model(scores, masks)

    def load(self):
        return self._stub


IMAGE = np.zeros((100, 160, 3), dtype="uint8")


def test_floor_is_off_unless_the_caller_asks_for_it():
    """不传 floor 时行为与上一版逐字节一致：低分框压根不进结果。"""
    pytest.importorskip("cv2")
    backend = _StubBackend([0.9, 0.05])
    kept = backend.predict(IMAGE, conf=0.25)
    assert [round(item.score, 2) for item in kept] == [0.9]
    assert backend._stub.calls[-1]["conf"] == 0.25


def test_floor_hands_back_the_rejected_candidates_for_explanation():
    pytest.importorskip("cv2")
    backend = _StubBackend([0.9, 0.05])
    both = backend.predict(IMAGE, conf=0.25, floor=0.01)
    assert [round(item.score, 2) for item in both] == [0.9, 0.05]
    assert [item.near_miss for item in both] == [False, True]
    assert backend._stub.calls[-1]["conf"] == 0.01, "要按 floor 跑才收得到候选"


def test_outlines_are_only_extracted_for_boxes_that_passed():
    """轮廓比找框贵得多：候选框不该去要轮廓，否则压到 0.01 时白烧一大片算力。"""
    pytest.importorskip("cv2")
    backend = _StubBackend([0.9, 0.05, 0.02], masks=np.zeros((3, 32, 50), dtype="float32"))
    asked = {}

    def spy(result, want, boxes):
        asked["slots"] = list(want)
        return [[0.1, 0.1, 0.4, 0.1, 0.25, 0.8] for _ in want]

    import autolabel.backends as backends

    monkey = pytest.MonkeyPatch()
    monkey.setattr(backends, "mask_polygons", spy)
    try:
        found = backend.predict(IMAGE, conf=0.25, floor=0.01)
    finally:
        monkey.undo()
    assert asked["slots"] == [0], "只有过线的那一框该去要轮廓"
    assert [item.poly for item in found][1:] == [[], []]


# --------------------------------------------------------------------------- //
# 五、接口
# --------------------------------------------------------------------------- //
def test_the_settings_api_accepts_the_two_new_knobs(server, probe_off):
    base, _store = server
    status, data = post_json(base, "/api/settings", {"imgsz": 1024, "conf": 0.05})
    assert status == 200
    assert data["saved"] == {"device": "auto", "half": False, "imgsz": 1024, "conf": 0.05}

    status, data = post_json(base, "/api/settings", {"imgsz": 99})
    assert status == 400 and "输入分辨率" in data["error"]
    status, data = post_json(base, "/api/settings", {"conf": 3})
    assert status == 400 and "置信度" in data["error"]

    status, data = get_json(base, "/api/settings")
    assert data["saved"]["imgsz"] == 1024 and data["effective"]["conf"] == 0.05


def test_a_detect_run_can_carry_its_own_resolution(server, dataset):
    """界面不填就别传：留空的意思是"用系统设置那一档"，不是"按 640 跑"。"""
    _base, store = server
    rels = [rel for rel, _ in load_index(dataset, LIB_POSE).ordered_items()]
    adopt_from_pool(dataset, rels, lib=LIB_DETECT)
    captured = []

    def spy(kind, files, params):
        captured.append(params)
        return types.SimpleNamespace(payload=lambda: {"params": params})

    store.jobs.submit = spy
    store.run_detect_job({"classes": ["person"], "imgsz": 1024})
    assert captured[0]["imgsz"] == 1024
    store.run_detect_job({"classes": ["person"]})
    assert "imgsz" not in captured[1]
    with pytest.raises(ValueError) as caught:
        store.run_detect_job({"classes": ["person"], "imgsz": "很大"})
    assert "输入分辨率" in str(caught.value)


def post_json(base, path, payload):
    import urllib.error
    import urllib.request

    request = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get_json(base, path):
    import urllib.request

    with urllib.request.urlopen(base + path, timeout=30) as response:
        return response.status, json.loads(response.read())


def test_one_batch_run_loads_the_weight_once(monkeypatch, detect_ds, tmp_path):
    """一次批量标注只准加载一遍权重。

    类名配对和真跑推理共用同一份后端缓存，而缓存键带着输入分辨率和设备：
    以前配类别那一路一律按系统默认值取（1280），真跑按这次填的 640，缓存当场
    分成两把，几百 MB 的权重复加载一遍——日志里两行「加载检测模型」，
    看着像整批跑了两遍，还白等好几秒。
    """
    from autolabel import fields as fd
    from autolabel import models as model_manager

    fd.add_classes(detect_ds, ["person"], source="test", lib=LIB_DETECT)
    monkeypatch.setattr(system, "label_defaults",
                        lambda: {"device": "auto", "half": False, "imgsz": 1280, "conf": 0.25})
    monkeypatch.setattr(model_manager, "resolve_weight_in_dir",
                        lambda name, dir_=None: os.path.join(str(tmp_path), os.path.basename(name)))
    monkeypatch.setattr(dt, "_BACKENDS", {})

    loads = []

    class CountingBackend(FakeBackend):
        def __init__(self, settings, model_name=""):
            super().__init__(model=model_name or settings.model)
            self.settings = settings

        def load(self):
            loads.append((self.settings.device, self.settings.imgsz))
            return self

    monkeypatch.setattr(dt, "DetectionBackend", CountingBackend)
    stats = dt.run_detect_labeling(detect_ds, model="yolo11n.pt", classes=["person"],
                                   imgsz=640, device="cpu", verbose=False)
    assert stats["labeled"] > 0
    assert loads == [("cpu", 640)], f"权重加载了 {len(loads)} 次：{loads}"


def test_profile_follows_the_saved_device(monkeypatch, tmp_path):
    """配类别那一路的设备要跟系统设置一致：留空写成 "auto" 会和真跑那次的
    具体设备名对不上，同一份权重又要多加载一遍。"""
    from autolabel import models as model_manager

    monkeypatch.setattr(system, "label_defaults",
                        lambda: {"device": "mps", "half": False, "imgsz": 1024, "conf": 0.25})
    monkeypatch.setattr(model_manager, "resolve_weight_in_dir",
                        lambda name, dir_=None: os.path.join(str(tmp_path), os.path.basename(name)))
    monkeypatch.setattr(dt, "_BACKENDS", {})

    seen = {}

    class SpyBackend(FakeBackend):
        def __init__(self, settings, model_name=""):
            super().__init__(model=model_name or settings.model)
            seen["device"] = settings.device
            seen["imgsz"] = settings.imgsz

        def load(self):
            return self

    monkeypatch.setattr(dt, "DetectionBackend", SpyBackend)
    dt.model_profile("yolo11n.pt")
    assert seen == {"device": "mps", "imgsz": 1024}, seen
    dt.model_profile("yolo11n.pt", device="cpu", imgsz=640)
    assert seen == {"device": "cpu", "imgsz": 640}, seen
