"""目标检测标注：权重清单、类别登记表、批量出框、库隔离与接口契约。

推理一律用替身（把 ``detect._backend`` 换掉）：装一次真权重要几百 MB，
而这里要验的是"类别号怎么换算、框落到哪个库、接口返回什么键"，跟模型无关。
"""
import json
import os
import time

import pytest

from autolabel import detect as dt
from autolabel import fields as fd
from autolabel import prompt as pr
from autolabel import upload as up
from autolabel.backends import TASK_NAMES, Detection
from autolabel.dataset import adopt_from_pool, load_index
from autolabel.library import LIB_DETECT, LIB_OBJECTS, LIB_POSE, label_path_for, labels_dir
from test_prompt_web import get, post

#: 替身"自带"的类别表：编号沿用 COCO，方便验证同名类别跨权重不换号
MODEL_CLASSES = {0: "person", 1: "bicycle", 2: "car", 5: "bus", 39: "bottle"}


class FakeBackend:
    """检测后端替身：不加载权重、不跑推理，勾了几个类别就每张出几个框。"""

    def __init__(self, model="yolo11n.pt", names=None, task="detect", score=0.87):
        self.model_name = model
        self.names = dict(MODEL_CLASSES if names is None else names)
        self.task = task
        self.score = score
        self.requests = []

    def describe(self):
        return {"model": os.path.basename(self.model_name), "task": self.task,
                "task_name": TASK_NAMES.get(self.task, self.task),
                "class_count": len(self.names), "device": "cpu"}

    def class_names(self):
        return dict(self.names)

    def predict(self, image, class_ids=(), conf=None):
        wanted = [int(value) for value in class_ids] if class_ids else sorted(self.names)
        self.requests.append((tuple(wanted), conf))
        found = []
        for offset, class_id in enumerate(wanted):
            if class_id not in self.names:
                continue
            found.append(Detection(class_id=class_id, class_name=self.names[class_id],
                                   score=self.score, cx=0.1 + 0.1 * offset, cy=0.5,
                                   w=0.2, h=0.3, has_mask=(self.task == "segment")))
        return found


def use_backend(monkeypatch, backend=None, **kwargs):
    """detect 里"取后端"只有 :func:`detect._backend` 一个入口，换掉它就换掉了整个推理层。"""
    backend = backend or FakeBackend(**kwargs)
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: backend)
    return backend


@pytest.fixture()
def detect_ds(dataset):
    """两帧素材同时进目标检测库：素材池共用一份文件，索引各记各的。"""
    rels = [rel for rel, _ in load_index(dataset, LIB_POSE).ordered_items()]
    assert adopt_from_pool(dataset, rels, lib=LIB_DETECT)["added"] == len(rels)
    return dataset


def rels_of(root, lib=LIB_DETECT):
    return [rel for rel, _ in load_index(root, lib).ordered_items()]


def used_ids(root, lib=LIB_DETECT):
    return {row["name"]: int(row["class_id"]) for row in fd.list_fields(root, lib)}


def all_rows(root, lib=LIB_DETECT):
    base = labels_dir(root, lib)
    rows = []
    for name in sorted(os.listdir(base)):
        rows.extend(fd.read_label_rows(os.path.join(base, name)))
    return rows


# --------------------------------------------------------------------------- #
# 权重清单：能不能用在目标检测库，界面要先问清楚
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,task", [
    ("yolo11n.pt", "detect"), ("yolo11s-seg.pt", "segment"), ("yolo11n-obb.pt", "obb"),
    ("yolo11n-pose.pt", "pose"), ("yolo11n-cls.pt", "classify"),
    ("mydata.pt", "detect"), ("README.md", ""),
])
def test_task_of_reads_the_filename_suffix(filename, task):
    assert dt.task_of(filename) == task


def test_local_models_keeps_pose_weights_out_of_the_dropdown(monkeypatch):
    """同一个 models/ 目录里姿态权重最常见，挡下来还得告诉他回哪去。"""
    from autolabel import models as model_manager

    present = {"yolo11n.pt": (True, 5.4), "yolo11n-pose.pt": (True, 6.0), "yolo11s.pt": (False, 0.0)}
    monkeypatch.setattr(model_manager, "weight_names", lambda dir_=None: sorted(present))
    monkeypatch.setattr(model_manager, "model_status", lambda name, weights_dir=None: {
        "ready": present[name][0], "size_mb": present[name][1], "path": f"models/{name}"})

    rows = {row["name"]: row for row in dt.local_models("models")}
    assert rows["yolo11n-pose.pt"]["usable"] is False
    assert "姿态标注" in rows["yolo11n-pose.pt"]["note"]
    assert rows["yolo11n.pt"]["usable"] is True and rows["yolo11n.pt"]["installed"] is True
    # 体积只能问 model_status：预置但没下载的权重按文件大小数会全是 0 字节
    assert rows["yolo11n.pt"]["bytes"] == int(5.4 * 1048576)
    assert rows["yolo11s.pt"]["installed"] is False and rows["yolo11s.pt"]["downloadable"] is True
    assert "还没下载" in rows["yolo11s.pt"]["note"]
    assert [row["name"] for row in dt.detectable_models("models")] == ["yolo11n.pt", "yolo11s.pt"]


def test_resolve_model_prefers_explicit_then_prefs_then_default(detect_ds):
    assert dt.DEFAULT_DETECT_MODEL == "yolo11n.pt"
    assert dt.resolve_model("yolo11s.pt", detect_ds) == "yolo11s.pt"
    assert dt.resolve_model("", detect_ds) == dt.DEFAULT_DETECT_MODEL
    dt.save_prefs(detect_ds, model="yolo11m.pt", classes=["person"], conf=0.3)
    assert dt.resolve_model("", detect_ds) == "yolo11m.pt"
    prefs = dt.load_prefs(detect_ds)
    assert prefs["model"] == "yolo11m.pt" and prefs["classes"] == ["person"] and prefs["conf"] == 0.3


def test_model_profile_reports_chinese_class_labels(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    profile = dt.model_profile("yolo11n.pt")
    assert profile["task"] == "detect" and profile["task_name"] == "目标检测"
    assert profile["class_count"] == len(MODEL_CLASSES)
    person = next(row for row in profile["classes"] if row["name"] == "person")
    assert person["class_id"] == 0 and person["label"] == "人"


# --------------------------------------------------------------------------- #
# 类别登记表：同名永不换号，删掉的号也不回收
# --------------------------------------------------------------------------- #
def test_add_model_classes_registers_only_the_picked_names(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    payload = dt.add_model_classes(detect_ds, "yolo11n.pt", ["person", "bus"])
    assert payload["added"] == 2 and payload["updated"] == 0
    assert payload["model_class_count"] == len(MODEL_CLASSES)
    assert {row["name"] for row in payload["fields"]} == {"person", "bus"}
    assert payload["fields"][0]["source"] == "yolo11n.pt"
    assert all(row["label"] for row in payload["fields"])

    again = dt.add_model_classes(detect_ds, "yolo11n.pt", ["bus", "person"])
    assert again["added"] == 0 and again["updated"] == 2
    assert used_ids(detect_ds) == used_ids(detect_ds)


def test_add_model_classes_reports_names_the_model_does_not_know(detect_ds, monkeypatch):
    """模型报不出来的类别不许登记：收了它，下次「按全部类别标注」会整批报错。"""
    use_backend(monkeypatch)
    payload = dt.add_model_classes(detect_ds, "yolo11n.pt", ["person", "gas stove", "BUS"])
    assert payload["not_in_model"] == ["gas stove"]
    assert sorted(row["name"] for row in payload["fields"]) == ["bus", "person"]
    # 大小写一律按权重原样收（写成 BUS 也收成 bus）：类名要进 data.yaml，
    # 也得和模型回报的值一模一样，按名对号才不会落空
    assert {row["class_id"] for row in payload["fields"]} == {0, 1}
    assert dt.run_detect_labeling(detect_ds, verbose=False)["missing_classes"] == []


def test_model_class_names_may_contain_spaces(detect_ds, monkeypatch):
    """COCO 里 potted plant / fire hydrant 这类带空格的名字必须收得进来。"""
    use_backend(monkeypatch, FakeBackend(names={0: "person", 58: "potted plant",
                                                60: "dining table"}))
    payload = dt.add_model_classes(detect_ds, "yolo11n.pt", ["all"] and ["person", "potted plant",
                                                                        "dining table"])
    assert payload["added"] == 3 and payload["skipped"] == []
    assert {row["name"] for row in payload["fields"]} == {"person", "potted plant", "dining table"}
    # 提示词库不吃这条宽松规则：类名是多模态大模型要原样回报的 JSON 值
    with pytest.raises(ValueError, match="字段名"):
        fd.upsert_field(detect_ds, name="water bottle", lib=LIB_OBJECTS)


def test_class_numbers_survive_a_weight_swap_but_are_never_reused(detect_ds, monkeypatch):
    """换权重时同类别必须同一个人的号；删掉再建也不能顶回旧号，否则老框集体指错东西。"""
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person", "bus", "car"])
    before = used_ids(detect_ds)
    dt.run_detect_labeling(detect_ds, verbose=False)

    # 换成只认 person / scooter 的自训练权重：person 的号一个都不能变
    use_backend(monkeypatch, FakeBackend(names={0: "person", 7: "scooter"}, model="run5.pt"))
    dt.add_model_classes(detect_ds, "run5.pt", ["person", "scooter"])
    after = used_ids(detect_ds)
    assert after["person"] == before["person"]
    assert after["scooter"] not in set(before.values())

    fd.delete_field(detect_ds, "bus", drop_boxes=True, lib=LIB_DETECT)
    assert all(int(row[0]) != before["bus"] for row in all_rows(detect_ds))
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["bus"])
    assert used_ids(detect_ds)["bus"] != before["bus"]


def test_delete_class_without_drop_boxes_refuses_and_names_the_box_count(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(detect_ds, verbose=False)
    with pytest.raises(ValueError, match="drop-boxes"):
        fd.delete_field(detect_ds, "person", lib=LIB_DETECT)


# --------------------------------------------------------------------------- #
# 批量标注：框落哪、索引写什么、人工结果什么时候不许覆盖
# --------------------------------------------------------------------------- #
def test_run_detect_labeling_writes_boxes_and_detect_index(detect_ds, monkeypatch):
    backend = use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person", "bus"])
    stats = dt.run_detect_labeling(detect_ds, verbose=False)
    assert stats["labeled"] == 2 and stats["failed"] == 0 and stats["protected"] == 0
    assert stats["boxes"] == 4 and stats["missing_classes"] == []
    assert stats["task_name"] == "目标检测" and stats["model"] == "yolo11n.pt"

    ids = used_ids(detect_ds)
    idx = load_index(detect_ds, LIB_DETECT)
    for rel in rels_of(detect_ds):
        item = idx.get(rel)
        assert item["detect_objects"] == 2
        assert item["detect_status"] == "unreviewed"
        assert item["detect_signature"] and item["detect_model"] == "yolo11n.pt"
        assert item["detect_classes"] == sorted(ids.values())
        assert item["detect_error"] == ""
        rows = fd.read_label_rows(label_path_for(detect_ds, rel, LIB_DETECT))
        assert sorted({int(row[0]) for row in rows}) == sorted(ids.values())
        assert all(len(row) == 5 and 0.0 <= row[1] <= 1.0 for row in rows)
    # 筛选要下推给模型（传的是模型自己的类别号，不是本库号）：不是全表跑完再丢
    assert sorted(backend.requests[0][0]) == [0, 5]


def test_run_detect_labeling_leaves_pose_and_objects_alone(detect_ds, monkeypatch):
    """三库共用素材池，但框各归各的目录与索引——这是"独立素材库"的全部意义。"""
    use_backend(monkeypatch)
    fd.upsert_field(detect_ds, name="bottle", label="瓶子", lib=LIB_OBJECTS)
    rel = rels_of(detect_ds)[0]
    pr.write_detections(detect_ds, rel, [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2)], LIB_OBJECTS)
    objects_before = json.dumps(load_index(detect_ds, LIB_OBJECTS).data, sort_keys=True)

    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(detect_ds, verbose=False)

    assert os.path.isdir(labels_dir(detect_ds, LIB_DETECT))
    assert not os.path.exists(label_path_for(detect_ds, rel, LIB_POSE))
    assert json.dumps(load_index(detect_ds, LIB_OBJECTS).data, sort_keys=True) == objects_before
    assert load_index(detect_ds, LIB_POSE).get(rel).get("objects") in (None, 0)


def test_second_run_protects_reviewed_images_and_overwrite_beats_it(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(detect_ds, verbose=False)
    first = rels_of(detect_ds)[0]
    idx = load_index(detect_ds, LIB_DETECT)
    idx.upsert(first, detect_status="approved")
    idx.save()

    # 换阈值 → 指纹变了，理论上会重标；但人工接受过的那张仍然不许动
    stats = dt.run_detect_labeling(detect_ds, conf=0.4, verbose=False)
    assert stats["protected"] == 1 and stats["labeled"] == 1

    forced = dt.run_detect_labeling(detect_ds, conf=0.4, overwrite=True, verbose=False)
    assert forced["protected"] == 0 and forced["labeled"] == 2
    assert load_index(detect_ds, LIB_DETECT).get(first)["detect_status"] == "edited"


def test_run_reports_classes_the_loaded_weight_cannot_see(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person", "bus"])
    use_backend(monkeypatch, FakeBackend(names={0: "person"}))
    stats = dt.run_detect_labeling(detect_ds, verbose=False)
    assert stats["missing_classes"] == ["bus"] and stats["boxes"] == 2

    use_backend(monkeypatch, FakeBackend(names={1: "bicycle"}, model="other.pt"))
    with pytest.raises(ValueError, match="认不出这些类别"):
        dt.run_detect_labeling(detect_ds, verbose=False)


def test_run_needs_a_class_registry_first(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    with pytest.raises(ValueError, match="还没有类别"):
        dt.run_detect_labeling(detect_ds, verbose=False)


def test_a_broken_image_fails_alone_not_the_whole_batch(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    target = rels_of(detect_ds)[0]
    os.remove(os.path.join(detect_ds, "images", target))
    stats = dt.run_detect_labeling(detect_ds, verbose=False)
    assert stats["failed"] == 1 and stats["labeled"] == 1
    item = load_index(detect_ds, LIB_DETECT).get(target)
    assert "model_failed" in (item["detect_flags"] or []) and item["detect_error"]


def test_segment_weight_is_accepted_as_a_box_source(detect_ds, monkeypatch):
    use_backend(monkeypatch, task="segment")
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    stats = dt.run_detect_labeling(detect_ds, verbose=False)
    assert stats["task"] == "segment" and stats["task_name"] == "实例分割"
    assert stats["labeled"] == 2


# --------------------------------------------------------------------------- #
# 指纹、坐标、导出
# --------------------------------------------------------------------------- #
def test_signature_tracks_model_classes_and_threshold_not_the_device(detect_ds):
    from autolabel.backends import BackendSettings

    base = BackendSettings(model="yolo11n.pt", device="mps")
    same = dt.signature_for("yolo11n.pt", [0, 1], BackendSettings(model="yolo11n.pt", device="cuda"))
    assert dt.signature_for("yolo11n.pt", [0, 1], base) == same
    assert dt.signature_for("yolo11n.pt", [0, 2], base) != same
    assert dt.signature_for("yolo11s.pt", [0, 1], base) != same
    assert dt.signature_for("yolo11n.pt", [0, 1],
                            BackendSettings(model="yolo11n.pt", conf=0.5)) != same


def test_coordinate_files_and_export_only_carry_detect_boxes(detect_ds, monkeypatch, tmp_path):
    import csv

    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(detect_ds, verbose=False)
    out = dt.write_detect_coordinate_files(detect_ds, out=str(tmp_path / "coords"))
    assert out["images"] == 2 and out["boxes"] == 2 and os.path.isdir(out["out"])
    assert set(out["files"]) == {"objects.csv", "objects.jsonl", "meta.json", "summary.json"}

    filename, body = dt.render_detect_coords(detect_ds, fmt="csv")
    assert filename.endswith("detect-objects.csv")
    rows = list(csv.DictReader(body.decode("utf-8").splitlines()))
    assert len(rows) == 2 and {row["class_name"] for row in rows} == {"person"}
    assert rows[0]["display_name"] == "人"

    exported = dt.export_detect_dataset(detect_ds, out=str(tmp_path / "ds"))
    assert exported["images"] == 2 and sum(exported["boxes"].values()) == 2
    assert [row["name"] for row in exported["classes"]] == ["person"]
    with open(exported["data_yaml"], encoding="utf-8") as fh:
        assert "person" in fh.read()


def test_detect_summary_names_the_library_unit_and_model(detect_ds, monkeypatch):
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(detect_ds, verbose=False)
    summary = dt.detect_summary(detect_ds)
    assert summary["unit"] == "框" and summary["model"] == "yolo11n.pt"
    assert summary["images"] == 2 and summary["boxes"] == 2 and summary["labeled"] == 2
    assert summary["per_class"] == {"person": 2} and summary["lib"] == LIB_DETECT
    assert summary["classes_remembered"] == ["person"]
    assert all(row["usable"] for row in summary["models"])


# --------------------------------------------------------------------------- #
# 接口契约：一个页面服务两个框库，全靠 lib 参数分流
# --------------------------------------------------------------------------- #
def test_upload_kind_decides_the_library():
    assert up.lib_for_kind("detect") == LIB_DETECT
    assert up.lib_for_kind("detect_upload") == LIB_DETECT
    assert up.lib_for_kind("prompt") == LIB_OBJECTS
    assert up.lib_for_kind("import") == LIB_POSE


def test_module_summary_keeps_each_library_tally_separate():
    """门户按模块汇总：顶层 labeled 是姿态库的无前缀键，别拿它当"本模块已标注"。"""
    from autolabel.review import module_summary

    zero = {"labeled": 0, "boxes": 0, "people": 0, "reviewed": 0, "unreviewed": 0, "flagged": 0}
    rows = [{"module": LIB_DETECT, "items": 1, "labeled": 0, **{"pose": dict(zero),
             "prompt": dict(zero), "detect": {"labeled": 1, "boxes": 5, "reviewed": 0,
                                              "unreviewed": 1, "flagged": 1}}}]
    summary = module_summary(rows)
    assert summary[LIB_DETECT]["labeled"] == 1 and summary[LIB_DETECT]["boxes"] == 5
    assert summary[LIB_POSE]["labeled"] == 0 and summary[LIB_OBJECTS]["prompt_labeled"] == 0


def test_dataset_payload_reports_the_detect_library(server):
    base, store = server
    rels = [rel for rel, _ in load_index(store.root, LIB_POSE).ordered_items()]
    adopt_from_pool(store.root, rels, lib=LIB_DETECT)
    status, body, _ = get(base, "/api/dataset")
    assert status == 200
    assert len(body["detect_items"]) == 2
    assert {row["lib"] for row in body["libs"]} == {LIB_POSE, LIB_OBJECTS, LIB_DETECT}
    assert body["detect"]["images"] == 2
    assert body["detect_classes"]["fields"] == []


def test_fields_are_scoped_per_library_over_http(server):
    base, _ = server
    status, body, _ = post(base, "/api/fields/save", {"name": "bottle", "label": "瓶子", "lib": "detect"})
    assert status == 200 and body["lib"] == "detect"
    assert [row["name"] for row in body["fields"]] == ["bottle"]

    status, objects, _ = get(base, "/api/fields")
    assert status == 200 and objects["fields"] == [] and objects["lib"] == LIB_OBJECTS
    status, detect, _ = get(base, "/api/fields?lib=detect")
    assert detect["fields"][0]["class_id"] == 0 and detect["fields"][0]["label"] == "瓶子"


def test_detect_models_endpoint_separates_usable_weights(server):
    base, store = server
    status, body, _ = get(base, "/api/detect/models")
    assert status == 200
    assert all(row["usable"] for row in body["models"])
    assert all(not row["usable"] for row in body["unusable"])
    assert "prefs" in body and body["device"]
    pose_rows = [row for row in body["unusable"] if row["task"] == "pose"]
    assert all("姿态标注" in row["note"] for row in pose_rows)


def test_detect_classes_add_and_item_roundtrip_use_the_detect_library(server, monkeypatch):
    base, store = server
    rels = [rel for rel, _ in load_index(store.root, LIB_POSE).ordered_items()]
    adopt_from_pool(store.root, rels, lib=LIB_DETECT)
    use_backend(monkeypatch)

    status, body, _ = post(base, "/api/detect/classes/add", {"model": "yolo11n.pt",
                                                            "names": ["person", "bus"]})
    assert status == 200 and body["ok"] is True and body["added"] == 2
    ids = {row["name"]: int(row["class_id"]) for row in body["fields"]}

    status, item, _ = post(base, "/api/prompt/item", {
        "image": rels[0], "lib": "detect", "status": "approved",
        "objects": [{"cls": ids["person"], "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3, "score": 0.9}]})
    assert status == 200 and item["detect_objects"] == 1 and item["detect_status"] == "approved"
    assert "prompt_objects" not in item

    # 读接口把库前缀摘掉成通用键（status/objects），前端一个页面才能服务两个库
    status, read, _ = get(base, f"/api/prompt/item?image={rels[0]}&lib=detect")
    assert status == 200 and read["lib"] == LIB_DETECT
    assert read["status"] == "approved" and len(read["objects"]) == 1
    assert read["objects"][0]["cls"] == ids["person"]
    # 提示词库完全没被这次写入碰到
    assert len(pr.read_detections(store.root, rels[0], LIB_OBJECTS)) == 0

    status, unknown, _ = post(base, "/api/prompt/item", {
        "image": rels[0], "lib": "detect", "objects": [{"cls": 77, "cx": .5, "cy": .5, "w": .2, "h": .2}]})
    assert status == 400 and "不在目标检测的类别表里" in unknown["error"]


def test_detect_prefs_set_the_default_weight_and_refuse_pose_weights(server, monkeypatch):
    """「设为检测默认」只认可用在检测的权重：姿态权重设进来会在跑标注时才炸，太晚。"""
    base, store = server
    names = [row["name"] for row in dt.local_models()]
    pose = next((row["name"] for row in dt.local_models() if not row["usable"]), "")

    status, body, _ = post(base, "/api/detect/prefs", {"model": "yolo11s.pt"})
    if status != 200:      # 本机没有这个权重时应该直说，而不是静默接受
        assert "本机没有" in body["error"]
    else:
        assert body["model"] == "yolo11s.pt" and body["prefs"]["model"] == "yolo11s.pt"
        assert dt.resolve_model("", store.root) == "yolo11s.pt"

    if pose:
        status, body, _ = post(base, "/api/detect/prefs", {"model": pose})
        assert status == 400 and "姿态" in body["error"]
    status, body, _ = post(base, "/api/detect/prefs", {"model": ""})
    assert status == 400 and "没有指定" in body["error"]
    assert names and all(row for row in names)


def test_prompt_export_honours_the_library_in_the_body(server, monkeypatch, tmp_path):
    base, store = server
    rels = [rel for rel, _ in load_index(store.root, LIB_POSE).ordered_items()]
    adopt_from_pool(store.root, rels, lib=LIB_DETECT)
    use_backend(monkeypatch)
    post(base, "/api/detect/classes/add", {"model": "yolo11n.pt", "names": ["person"]})
    assert dt.run_detect_labeling(store.root, verbose=False)["labeled"] == 2

    status, body, _ = post(base, "/api/prompt/export", {"lib": "detect", "scope": "all",
                                                        "out": str(tmp_path / "ds")})
    assert status == 200 and sum(body["export"]["boxes"].values()) == 2
    assert [row["name"] for row in body["export"]["classes"]] == ["person"]
    status, empty, _ = post(base, "/api/prompt/export", {"scope": "all", "out": str(tmp_path / "obj")})
    assert status == 400 and "还没有" in empty["error"]


def test_detect_run_job_labels_the_whole_library_and_says_so(server, monkeypatch):
    base, store = server
    rels = [rel for rel, _ in load_index(store.root, LIB_POSE).ordered_items()]
    adopt_from_pool(store.root, rels, lib=LIB_DETECT)
    use_backend(monkeypatch)
    post(base, "/api/detect/classes/add", {"model": "yolo11n.pt", "names": ["person", "bus"]})

    status, job, _ = post(base, "/api/detect/run", {"model": "yolo11n.pt",
                                                    "classes": ["person"], "batch_name": "接口自检"})
    assert status == 202, job
    assert job["kind"] == "detect"

    deadline = time.time() + 30
    while time.time() < deadline:
        _status, jobs, _headers = get(base, "/api/jobs")
        row = next((item for item in jobs["jobs"] if item["id"] == job["id"]), None)
        if row and row["finished"]:
            break
        time.sleep(0.1)
    assert row and row["ok"] is True, row
    assert row["result"]["boxes"] == 2 and row["result"]["labeled"] == 2
    assert "标注 2 张" in row["message"]
    assert any(batch["name"] == "接口自检" for batch in get(base, "/api/batches")[1]["batches"])


# --------------------------------------------------------------------------- //
# 推理层的两个真实坑：分割权重的 masks、坐标 meta 的列说明
# --------------------------------------------------------------------------- //
class MaskRows:
    """像 ultralytics 的 ``masks.data``：多元素张量套 ``bool()`` 直接抛异常。"""

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __bool__(self):
        if self.count != 1:
            raise ValueError("The truth value of an array with more than one element is ambiguous.")
        return True


def _segment_backend():
    """一个装满替身结果的检测后端：不加载权重，但要走真实的 :meth:`predict`。"""
    import numpy as np

    from autolabel.backends import BackendSettings, DetectionBackend

    class Boxes:
        cls = np.array([2, 5])
        conf = np.array([0.91, 0.62])
        xyxy = np.array([[10.0, 10.0, 60.0, 90.0], [70.0, 20.0, 120.0, 100.0]])

        def __len__(self):
            return len(self.cls)

    class Result:
        orig_shape = (100, 160, 3)
        boxes = Boxes()
        masks = type("Masks", (), {"data": MaskRows(2)})()
        obb = None

    class Model:
        def predict(self, image, **kwargs):
            return [Result()]

    backend = DetectionBackend(BackendSettings(model="yolo11n-seg.pt", device="cpu"),
                              "yolo11n-seg.pt")
    backend._model = Model()
    backend.task = "segment"
    backend.names = {2: "car", 5: "bus"}
    backend._device = "cpu"
    backend._half = False
    return backend


def test_segment_masks_do_not_break_predict():
    """分割权重的 ``masks.data`` 是 (N,H,W) 张量：给它套 ``bool()`` 会抛
    "truth value is ambiguous"，真机跑 ``yolo11n-seg.pt`` 时每张图都失败。"""
    import numpy as np

    found = _segment_backend().predict(np.zeros((100, 160, 3), dtype="uint8"), class_ids=[2, 5])
    assert [row.class_id for row in found] == [2, 5]
    assert all(row.has_mask for row in found), "有 mask 却没记下来，界面就看不出框出自分割权重"


def test_segment_boxes_are_flagged_at_image_level(detect_ds, monkeypatch):
    """分割权重的框要留下 ``masked`` 记号：轮廓被压成水平框这件事不能悄悄发生。"""
    use_backend(monkeypatch, FakeBackend(model="yolo11n-seg.pt", task="segment"))
    dt.add_model_classes(detect_ds, "yolo11n-seg.pt", ["bus"])
    stats = dt.run_detect_labeling(detect_ds, model="yolo11n-seg.pt", classes=["bus"])
    assert stats["labeled"] >= 1
    rows = [load_index(detect_ds, LIB_DETECT).get(rel) or {} for rel in rels_of(detect_ds)]
    assert any("masked" in (row.get("detect_flags") or []) for row in rows)
    assert "masked" in dt.DETECT_FLAG_HINTS, "标记名进了索引却没有中文解释"


def test_detect_coordinate_columns_speak_detect(detect_ds, monkeypatch):
    """坐标 meta.json 的列说明按库给：目标检测库别再写 prompt_stale / vlm_failed。"""
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["bus", "person"])
    dt.run_detect_labeling(detect_ds, model="yolo11n.pt", classes=["bus", "person"])
    _records, meta = dt.collect_detect_records(detect_ds)
    columns = meta["columns"]
    assert meta["lib"] == LIB_DETECT
    assert columns["status"].startswith("目标检测")
    assert "model_stale" in columns["image_flags"] and "prompt_stale" not in columns["image_flags"]

    _pose_records, prompt_meta = pr.collect_object_records(detect_ds, lib=LIB_OBJECTS)
    assert prompt_meta["columns"]["status"].startswith("提示词标注")
    assert "vlm_failed" in prompt_meta["columns"]["image_flags"]


def test_edited_boxes_are_equally_protected_from_relabel(detect_ds, monkeypatch):
    """改过但还没点「接受」的框同样是人工成果。

    保护名单以前只认 approved，检测库自己抄了一份规则还漏了 edited——
    手工把框挪过、还没定论的图，下一次批量出框直接给人抹掉了。
    """
    use_backend(monkeypatch)
    dt.add_model_classes(detect_ds, "yolo11n.pt", ["person"])
    dt.run_detect_labeling(detect_ds, verbose=False)
    first = rels_of(detect_ds)[0]
    idx = load_index(detect_ds, LIB_DETECT)
    idx.upsert(first, detect_status="edited")
    idx.save()

    stats = dt.run_detect_labeling(detect_ds, conf=0.4, verbose=False)
    assert stats["protected"] == 1 and stats["labeled"] == 1, stats

    forced = dt.run_detect_labeling(detect_ds, conf=0.4, overwrite=True, verbose=False)
    assert forced["protected"] == 0 and forced["labeled"] == 2, forced
