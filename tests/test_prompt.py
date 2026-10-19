"""提示词标注流水线：提示词拼装、回复解析容错、批量跑标、坐标与数据集导出。"""
import json
import os

import pytest

from autolabel import fields as fd
from autolabel import prompt as pr
from autolabel.dataset import load_index, prompt_label_path_for


def two_fields(root):
    bottle = fd.upsert_field(root, name="bottle", label="瓶子", prompt_hint="桌上或手里的水瓶")
    backpack = fd.upsert_field(root, name="backpack", label="背包", prompt_hint="椅上或肩上的背包")
    return [bottle, backpack]


def mark_labeled(root, rel, detections, **extra):
    pr.write_detections(root, rel, detections)
    idx = load_index(root, "objects")
    fields = {"prompt_objects": len(detections), "prompt_status": "unreviewed"}
    fields.update(extra)
    idx.upsert(rel, **fields)
    idx.save()


# --------------------------------------------------------------------------- //
# 提示词
# --------------------------------------------------------------------------- //
def test_build_prompt_lists_name_label_and_hint(dataset):
    text = pr.build_prompt(two_fields(dataset), "只标画面正中的人")
    assert "- bottle（瓶子）：桌上或手里的水瓶" in text
    assert "- backpack（背包）：椅上或肩上的背包" in text
    assert "补充说明：只标画面正中的人" in text
    assert '"objects":[{"field":"字段名","bbox":[x1,y1,x2,y2]}]' in text
    assert "0-1000" in text


def test_signature_changes_with_hint_and_profile(dataset):
    rows = two_fields(dataset)
    base = pr.signature_for(rows, "p1")
    assert pr.signature_for(rows, "p1") == base
    assert pr.signature_for(rows, "p2") != base
    assert pr.signature_for(rows, "p1", extra="多一句") != base
    changed = [dict(rows[0], prompt_hint="换个说法"), rows[1]]
    assert pr.signature_for(changed, "p1") != base


# --------------------------------------------------------------------------- //
# 解析容错
# --------------------------------------------------------------------------- //
@pytest.fixture()
def fields_for_parse(dataset):
    return two_fields(dataset)


def test_parse_plain_objects_payload(fields_for_parse):
    text = '{"objects":[{"field":"bottle","bbox":[100,200,300,600],"score":0.8}]}'
    found, guesses = pr.parse_objects(text, fields_for_parse)
    assert len(found) == 1 and not guesses
    assert found[0].cls == 0
    assert abs(found[0].cx - 0.2) < 1e-6 and abs(found[0].cy - 0.4) < 1e-6
    assert found[0].score == 0.8


def test_parse_tolerates_fenced_json_and_bare_array(fields_for_parse):
    fenced = '```json\n{"objects":[{"field":"bottle","bbox":[0,0,500,500]}]}\n```'
    bare = '[{"field":"backpack","bbox":[0,0,500,500]}]'
    assert len(pr.parse_objects(fenced, fields_for_parse)[0]) == 1
    found = pr.parse_objects(bare, fields_for_parse)[0]
    assert len(found) == 1 and found[0].cls == 1


def test_parse_tolerates_alternative_container_keys(fields_for_parse):
    for key in ("detections", "items", "results"):
        text = '{"%s":[{"field":"bottle","bbox":[0,0,500,500]}]}' % key
        assert len(pr.parse_objects(text, fields_for_parse)[0]) == 1, key


def test_parse_tolerates_prose_wrapped_braces(fields_for_parse):
    text = '我看到一个瓶子：{"field": "bottle", "bbox": [10, 20, 110, 220]} 就这样。'
    found, guesses = pr.parse_objects(text, fields_for_parse)
    assert len(found) == 1 and not guesses
    assert abs(found[0].w - 0.1) < 1e-6


def test_parse_accepts_xywh_and_dict_boxes(fields_for_parse):
    xywh = '{"objects":[{"field":"bottle","bbox_xywh":[300,400,200,400]}]}'
    as_dict = '{"objects":[{"field":"bottle","bbox":{"x1":100,"y1":200,"x2":300,"y2":600}}]}'
    as_cxcywh = '{"objects":[{"field":"bottle","bbox":{"cx":200,"cy":400,"w":200,"h":400}}]}'
    first = pr.parse_objects(xywh, fields_for_parse)[0][0]
    assert abs(first.cx - 0.3) < 1e-6 and abs(first.cy - 0.4) < 1e-6
    assert abs(first.w - 0.2) < 1e-6 and abs(first.h - 0.4) < 1e-6
    for text in (as_dict, as_cxcywh):
        found = pr.parse_objects(text, fields_for_parse)[0][0]
        assert abs(found.cx - 0.2) < 1e-6 and abs(found.cy - 0.4) < 1e-6


def test_parse_handles_normalised_and_pixel_coordinates(fields_for_parse):
    tiny = pr.parse_objects('{"objects":[{"field":"bottle","bbox":[0.1,0.2,0.3,0.6]}]}',
                            fields_for_parse)[0][0]
    assert abs(tiny.cx - 0.2) < 1e-6 and abs(tiny.h - 0.4) < 1e-6
    thousand = pr.parse_objects('{"objects":[{"field":"bottle","bbox":[100,200,300,600]}]}',
                                fields_for_parse)[0][0]
    assert abs(thousand.cx - 0.2) < 1e-6 and abs(thousand.h - 0.4) < 1e-6
    assert abs(tiny.cx - thousand.cx) < 1e-9


def test_parse_matches_chinese_label_and_fuzzy_names(fields_for_parse):
    cn = pr.parse_objects('{"objects":[{"field":"瓶子","bbox":[0,0,500,500]}]}', fields_for_parse)[0]
    assert len(cn) == 1 and cn[0].cls == 0
    fuzzy = pr.parse_objects('{"objects":[{"field":"Water Bottle","bbox":[0,0,500,500]}]}',
                             fields_for_parse)[0]
    assert len(fuzzy) == 1 and fuzzy[0].cls == 0


def test_parse_falls_back_to_only_field_when_field_is_missing(fields_for_parse):
    only = [fields_for_parse[0]]
    found, guesses = pr.parse_objects('{"objects":[{"bbox":[0,0,500,500]}]}', only)
    assert len(found) == 1 and found[0].cls == 0 and not guesses


def test_parse_reports_unknown_field_instead_of_guessing(fields_for_parse):
    found, guesses = pr.parse_objects('{"objects":[{"field":"banana","bbox":[0,0,500,500]}]}',
                                      fields_for_parse)
    assert found == [] and guesses == ["banana"]


def test_parse_plain_text_reply_yields_no_boxes_without_error(fields_for_parse):
    found, guesses = pr.parse_objects("画面里没有看到这些目标。", fields_for_parse)
    assert found == [] and guesses == []


def test_parse_dedupes_overlapping_boxes_of_same_field(fields_for_parse):
    text = ('{"objects":[{"field":"bottle","bbox":[100,100,300,500]},'
            '{"field":"bottle","bbox":[102,101,302,502]}]}')
    assert len(pr.parse_objects(text, fields_for_parse)[0]) == 1
    both = ('{"objects":[{"field":"bottle","bbox":[0,0,300,500]},'
            '{"field":"bottle","bbox":[500,0,800,500]}]}')
    assert len(pr.parse_objects(both, fields_for_parse)[0]) == 2


def test_parse_clamps_out_of_range_boxes(fields_for_parse):
    found = pr.parse_objects('{"objects":[{"field":"bottle","bbox":[-40,20,2600,900]}]}',
                             fields_for_parse)[0]
    assert len(found) == 1
    assert -1e-9 <= found[0].cx - found[0].w / 2 and found[0].cx + found[0].w / 2 <= 1 + 1e-9


# --------------------------------------------------------------------------- //
# 标签读写与人工标记
# --------------------------------------------------------------------------- //
def test_write_and_read_detections_roundtrip(dataset):
    detections = [pr.Detection(cls=0, cx=0.5, cy=0.25, w=0.2, h=0.3, score=0.9)]
    pr.write_detections(dataset, "img0.jpg", detections)
    path = prompt_label_path_for(dataset, "img0.jpg")
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert lines == ["0 0.500000 0.250000 0.200000 0.300000"]
    back = pr.read_detections(dataset, "img0.jpg")
    assert len(back) == 1 and back[0].cls == 0 and back[0].score == 0.0


def test_empty_label_file_means_seen_but_absent(dataset):
    pr.write_detections(dataset, "img0.jpg", [])
    assert os.path.exists(prompt_label_path_for(dataset, "img0.jpg"))
    assert pr.read_detections(dataset, "img0.jpg") == []


def test_apply_manual_marks_restores_flags_after_reload(dataset):
    pr.write_detections(dataset, "img0.jpg",
                       [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2),
                        pr.Detection(cls=1, cx=0.2, cy=0.2, w=0.1, h=0.1)])
    reloaded = pr.read_detections(dataset, "img0.jpg")
    assert all("manual" not in d.flags for d in reloaded)
    pr.apply_manual_marks(reloaded, [1, 9, None])
    assert reloaded[1].flags == ["manual"] and reloaded[1].score == 1.0
    assert reloaded[0].flags == []


def test_detection_payload_carries_pixel_xyxy():
    payload = pr.Detection(cls=2, cx=0.5, cy=0.5, w=0.5, h=0.25).payload(640, 480)
    assert payload["xyxy"] == [160.0, 180.0, 480.0, 300.0]
    assert payload["cls"] == 2


# --------------------------------------------------------------------------- //
# 质量标记
# --------------------------------------------------------------------------- //
def test_box_flags_marks_small_edge_duplicate_and_unknown():
    small = pr.Detection(cls=0, cx=0.5, cy=0.5, w=6 / 640, h=6 / 480)
    assert "small_box" in pr.box_flags([small], 640, 480, [0])[0]
    edge = pr.Detection(cls=0, cx=0.001, cy=0.5, w=0.1, h=0.2)
    assert "edge_box" in pr.box_flags([edge], 640, 480, [0])[0]
    unknown = pr.Detection(cls=7, cx=0.5, cy=0.5, w=0.2, h=0.2)
    assert "unknown_class" in pr.box_flags([unknown], 640, 480, [0])[0]
    dup = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.2)
    dup2 = pr.Detection(cls=0, cx=0.505, cy=0.5, w=0.2, h=0.2)
    assert "dup_box" in pr.box_flags([dup, dup2], 640, 480, [0])[0]


def test_image_flags_and_priority_surface_problems(dataset):
    detections = [pr.Detection(cls=0, cx=0.5, cy=0.5, w=6 / 640, h=6 / 480)]
    per_box = pr.box_flags(detections, 640, 480, [0])
    assert pr.image_flags(detections, per_box) == ["small_box"]
    assert pr.image_flags([], []) == ["none_found"]
    assert "prompt_stale" in pr.image_flags([], [], signature="a", item_signature="b")
    assert "vlm_failed" in pr.image_flags([], [], failed=True)
    low = {"prompt_score": 0.1, "prompt_flags": ["small_box"], "prompt_objects": 1,
           "prompt_status": "unreviewed"}
    clean = {"prompt_score": 0.95, "prompt_flags": [], "prompt_objects": 3,
             "prompt_status": "approved"}
    assert pr.prompt_priority(low) > pr.prompt_priority(clean)


# --------------------------------------------------------------------------- //
# 批量跑标
# --------------------------------------------------------------------------- //
@pytest.fixture()
def one_profile(monkeypatch):
    from autolabel import models as model_manager

    profile = model_manager.upsert_profile({"name": "本地 Ollama", "base_url": "http://x/v1",
                                            "model": "qwen2.5vl:7b"})
    monkeypatch.setattr(model_manager, "get_profile", lambda pid: dict(profile, image_limit=768))
    return profile


def test_run_skips_images_already_done_with_same_signature(dataset, one_profile, monkeypatch):
    fields = two_fields(dataset)
    calls = []
    reply = '{"objects":[{"field":"bottle","bbox":[100,100,300,500]}]}'
    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url: calls.append(prompt) or reply)
    signature = pr.signature_for(fields, one_profile["id"])
    first = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile)
    assert first["labeled"] == 2 and first["boxes"] == 2
    assert len(calls) == 2
    idx = load_index(dataset, "objects")
    assert idx.get("img0.jpg")["prompt_signature"] == signature
    again = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile)
    assert again["images"] == 0 and len(calls) == 2


def test_run_overwrite_relabels_and_extra_hint_changes_scope(dataset, one_profile, monkeypatch):
    fields = two_fields(dataset)
    calls = []
    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url: calls.append(prompt)
                        or '{"objects":[{"field":"bottle","bbox":[100,100,300,500]}]}')
    pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile)
    assert pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile,
                                 overwrite=True)["labeled"] == 2
    assert len(calls) == 4
    pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile, extra_hint="只标近景")
    assert len(calls) == 6
    assert any("补充说明：只标近景" in prompt for prompt in calls)


def test_run_protects_human_work_without_overwrite(dataset, one_profile, monkeypatch):
    """人工接受过/画过框的图不能被批量重标抹掉，除非显式 overwrite。"""
    fields = two_fields(dataset)
    calls = []
    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url: calls.append(prompt)
                        or '{"objects":[{"field":"bottle","bbox":[100,100,300,500]}]}')
    pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile)
    assert len(calls) == 2

    idx = load_index(dataset, "objects")
    idx.upsert("img0.jpg", prompt_status="approved")
    idx.upsert("img1.jpg", prompt_manual=1, prompt_manual_boxes=[0])
    idx.save()

    # 换提示词就是换签名，正常情况下两张都该重标，但人碰过的必须原样留下
    stats = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile,
                                   extra_hint="只标桌上的", verbose=False)
    assert stats["protected"] == 2 and stats["images"] == 0 and len(calls) == 2

    stats = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile,
                                   extra_hint="只标桌上的", overwrite=True, verbose=False)
    assert stats["protected"] == 0 and stats["labeled"] == 2 and len(calls) == 4


def test_relabel_resets_stale_manual_marks(dataset, one_profile, monkeypatch):
    """重标后人工标记与类别号要跟着重写，别留指向不存在框的悬空标记。"""
    fields = two_fields(dataset)
    bottle_cls = int(fields[0]["class_id"])
    mark_labeled(dataset, "img0.jpg", [], prompt_manual=2, prompt_manual_boxes=[0, 1],
                 prompt_classes=[9], prompt_status="edited",
                 prompt_signature=pr.signature_for(fields, one_profile["id"]))

    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url:
                        '{"objects":[{"field":"bottle","bbox":[100,100,300,500]}]}')
    stats = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile,
                                   overwrite=True, verbose=False)
    assert stats["labeled"] == 2

    item = load_index(dataset, "objects").get("img0.jpg")
    assert item["prompt_objects"] == 1
    assert item["prompt_manual"] == 0 and item["prompt_manual_boxes"] == []
    assert item["prompt_classes"] == [bottle_cls]
    # 人工已确认过的图重标后回到「已编辑」，需要重新过目
    assert item["prompt_status"] == "edited"
    assert len(pr.read_detections(dataset, "img0.jpg")) == 1


def test_run_records_unknown_fields_and_empty_images(dataset, one_profile, monkeypatch):
    fields = two_fields(dataset)
    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url: '{"objects":[{"field":"banana",'
                                                         '"bbox":[100,100,300,500]}]}')
    stats = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile)
    assert stats["unknown_fields"] == ["banana"]
    assert stats["boxes"] == 0
    assert "none_found" in load_index(dataset, "objects").get("img0.jpg")["prompt_flags"]


def test_run_marks_failed_images_and_raises_when_all_fail(dataset, one_profile, monkeypatch):
    from autolabel.vlm import VLMError

    def boom(profile, prompt, data_url):
        raise VLMError("域名解析失败")

    monkeypatch.setattr("autolabel.vlm.chat", boom)
    with pytest.raises(RuntimeError) as exc:
        pr.run_prompt_labeling(dataset, fields=two_fields(dataset), profile=one_profile)
    assert "全部失败" in str(exc.value)
    item = load_index(dataset, "objects").get("img0.jpg")
    assert "vlm_failed" in item["prompt_flags"]
    assert "域名解析失败" in item["prompt_error"]


def test_run_partial_failure_keeps_going(dataset, one_profile, monkeypatch):
    fields = two_fields(dataset)

    calls = {"n": 0}

    def flaky(profile, prompt, data_url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("单图炸了")
        return '{"objects":[{"field":"bottle","bbox":[100,100,300,500]}]}'

    monkeypatch.setattr("autolabel.vlm.chat", flaky)
    stats = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile, verbose=False)
    assert stats["failed"] == 1 and stats["labeled"] == 1
    failed = [rel for rel, item in load_index(dataset, "objects").ordered_items()
              if "vlm_failed" in (item.get("prompt_flags") or [])]
    assert len(failed) == 1


def test_run_requires_fields_and_profile(dataset, one_profile):
    with pytest.raises(ValueError) as exc:
        pr.run_prompt_labeling(dataset, fields=[], profile=one_profile)
    assert "字段" in str(exc.value)


def test_resolve_profile_errors_in_chinese_when_nothing_configured(dataset):
    with pytest.raises(ValueError) as exc:
        pr.resolve_profile("不存在的配置")
    assert "模型管理" in str(exc.value)


# --------------------------------------------------------------------------- //
# 坐标与数据集导出
# --------------------------------------------------------------------------- //
def test_render_object_coords_csv_has_one_row_per_box(dataset):
    two_fields(dataset)
    mark_labeled(dataset, "img0.jpg", [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4, score=0.7)])
    mark_labeled(dataset, "img1.jpg", [pr.Detection(cls=1, cx=0.3, cy=0.4, w=0.2, h=0.2)])
    name, body = pr.render_object_coords(dataset, "csv")
    assert name.endswith("-objects.csv")
    text = body.decode("utf-8")
    header = text.splitlines()[0].split(",")
    assert header[:5] == ["image", "status", "box_index", "class_id", "class_name"]
    assert len(text.strip().splitlines()) == 3
    assert "bottle" in text and "backpack" in text


def test_render_object_coords_jsonl_is_one_image_per_line(dataset):
    two_fields(dataset)
    mark_labeled(dataset, "img0.jpg", [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4)])
    name, body = pr.render_object_coords(dataset, "jsonl")
    assert name.endswith("-objects.jsonl")
    rows = [json.loads(line) for line in body.decode("utf-8").strip().splitlines()]
    assert len(rows) == 1 and rows[0]["objects"][0]["class_name"] == "bottle"
    assert rows[0]["objects"][0]["bbox_cxcywh_norm"][0] == 0.5


def test_write_object_coordinate_files_creates_all_artifacts(dataset):
    two_fields(dataset)
    mark_labeled(dataset, "img0.jpg", [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4)])
    info = pr.write_object_coordinate_files(dataset, out=os.path.join(dataset, "coords", "t1"))
    for key in ("objects.csv", "objects.jsonl", "meta.json", "summary.json"):
        assert os.path.exists(info["files"][key]), key
    assert info["boxes"] == 1
    summary = json.load(open(info["files"]["summary.json"], encoding="utf-8"))
    assert summary["per_field"] == {"bottle": 1} and summary["images"] == 1
    assert summary["images_empty"] == 0 and summary["boxes"] == 1
    meta = json.load(open(info["files"]["meta.json"], encoding="utf-8"))
    assert {f["name"] for f in meta["fields"]} == {"bottle", "backpack"}
    assert meta["skipped"][0]["reason"] == "未做提示词标注"


def test_export_detection_dataset_remaps_classes_to_contiguous_ids(dataset):
    two_fields(dataset)
    helmet = fd.upsert_field(dataset, name="helmet", label="头盔")
    assert helmet["class_id"] == 2
    fd.delete_field(dataset, "bottle")
    mark_labeled(dataset, "img0.jpg", [pr.Detection(cls=1, cx=0.5, cy=0.5, w=0.2, h=0.4)])
    mark_labeled(dataset, "img1.jpg", [pr.Detection(cls=2, cx=0.3, cy=0.4, w=0.2, h=0.2)])
    info = pr.export_detection_dataset(dataset, out=os.path.join(dataset, "exports", "d1"))
    assert [c["class_id"] for c in info["classes"]] == [1, 2]
    assert [c["export_id"] for c in info["classes"]] == [0, 1]
    assert info["train"] + info["val"] == 2
    yaml_text = open(info["data_yaml"], encoding="utf-8").read()
    assert "  0: backpack" in yaml_text and "  1: helmet" in yaml_text
    assert "# names 显示名：backpack = 背包" in yaml_text
    assert "0: bottle" not in yaml_text
    class_ids = set()
    for split in ("train", "val"):
        base = os.path.join(info["out"], "labels", split)
        for name in sorted(os.listdir(base)):
            for line in open(os.path.join(base, name), encoding="utf-8").read().splitlines():
                if line.split():
                    class_ids.add(int(line.split()[0]))
    assert class_ids == {0, 1}


def test_export_detection_dataset_rejects_rejected_images(dataset):
    two_fields(dataset)
    mark_labeled(dataset, "img0.jpg", [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4)],
                 prompt_status="rejected")
    mark_labeled(dataset, "img1.jpg", [pr.Detection(cls=1, cx=0.3, cy=0.4, w=0.2, h=0.2)])
    info = pr.export_detection_dataset(dataset, out=os.path.join(dataset, "exports", "d2"))
    assert info["train"] + info["val"] == 1


def test_export_detection_dataset_needs_something_to_export(dataset):
    two_fields(dataset)
    with pytest.raises(ValueError) as exc:
        pr.export_detection_dataset(dataset, out=os.path.join(dataset, "exports", "d3"))
    assert "先跑一次标注" in str(exc.value)


def test_prompt_summary_counts_status_flags_and_per_field(dataset):
    two_fields(dataset)
    mark_labeled(dataset, "img0.jpg", [pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4)],
                 prompt_status="approved", prompt_flags=["small_box"])
    mark_labeled(dataset, "img1.jpg", [])
    summary = pr.prompt_summary(dataset)
    assert summary["images"] == 2 and summary["labeled"] == 2
    assert summary["boxes"] == 1
    assert summary["status_counts"] == {"approved": 1, "unreviewed": 1}
    assert summary["flag_counts"] == {"small_box": 1}
    assert summary["per_field"] == {"bottle": 1}
    assert summary["flag_hints"]["small_box"]


def test_edited_boxes_count_as_human_work_too(dataset, one_profile, monkeypatch):
    """提示词库那份保护名单同样漏过 edited：状态只到「已修改」也必须绕开。"""
    fields = two_fields(dataset)
    calls = []
    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url: calls.append(prompt)
                        or '{"objects":[{"field":"bottle","bbox":[100,100,300,500]}]}')
    pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile)
    assert len(calls) == 2

    idx = load_index(dataset, "objects")
    idx.upsert("img0.jpg", prompt_status="edited")
    idx.save()

    stats = pr.run_prompt_labeling(dataset, fields=fields, profile=one_profile,
                                   extra_hint="只标桌上的", verbose=False)
    assert stats["protected"] == 1 and stats["labeled"] == 1, stats
    assert len(calls) == 3, "只有没被人工碰过的那张该再问一次模型"


# --------------------------------------------------------------------------- //
# 逐框置信度：5 列标签不落这一列，靠签名表认回来
# --------------------------------------------------------------------------- //
def test_box_signature_is_stable_across_a_round_trip_through_the_label_file(dataset):
    """签名是逐框分数的钥匙：同一个框不管排第几、读回来第几次，都得是同一把钥匙。

    ``*_manual_boxes`` 按顺序下标记人工框，用户删一个框后面全错位；分数要是也这么
    存，「模型给的 0.83」就会贴到别的框身上，比不贴更坏。
    """
    a = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4, score=0.83)
    b = pr.Detection(cls=1, cx=0.25, cy=0.3, w=0.1, h=0.2, score=0.5)
    assert pr.box_signature(a) != pr.box_signature(b)
    # 类别号不同就是两个框，几何完全一样也要分开
    assert pr.box_signature(a) != pr.box_signature(
        pr.Detection(cls=1, cx=0.5, cy=0.5, w=0.2, h=0.4))

    pr.write_detections(dataset, "img0.jpg", [a, b])
    again = pr.read_detections(dataset, "img0.jpg")

    assert [pr.box_signature(d) for d in again] == [pr.box_signature(a), pr.box_signature(b)], \
        "落盘再读回来签名变了，分数表就永久认不回这些框"
    assert pr.box_signature(a) == pr.box_signature(
        pr.Detection(cls=0, cx=0.5000001, cy=0.5, w=0.2, h=0.4)), "精度要和标签文件一致（6 位小数）"


def test_scores_come_back_by_signature_not_by_position(dataset):
    """读回来的框全是 0 分，只有靠签名才能把分数贴回**原来那个**框。"""
    kept = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4, score=0.83)
    silent = pr.Detection(cls=1, cx=0.25, cy=0.3, w=0.1, h=0.2, score=0.0)
    stored = pr.score_map([kept, silent])
    assert stored == {pr.box_signature(kept): 0.83}, "没有分数的框不该往表里写 0"

    pr.write_detections(dataset, "img0.jpg", [kept, silent])
    back = pr.read_detections(dataset, "img0.jpg")
    assert [d.score for d in back] == [0.0, 0.0], "标签不落分数，读回来必然全 0"

    pr.apply_scores(back, stored)
    assert back[0].score == pytest.approx(0.83)
    assert back[1].score == 0.0

    # 换了顺序也要贴对：贴错框比显示 — 坏得多
    flipped = pr.read_detections(dataset, "img0.jpg")[::-1]
    pr.apply_scores(flipped, stored)
    assert [round(d.score, 2) for d in flipped] == [0.0, 0.83]

    for junk in ("不是表", None, {"随便": "乱码"}, {"随便": 0}):
        pr.apply_scores(back, junk)


def test_manual_marks_beat_the_stored_score(dataset):
    """人工框在索引里按 1.0 计，签名表不能把它改回模型当年给的分。"""
    box = pr.Detection(cls=0, cx=0.5, cy=0.5, w=0.2, h=0.4, score=0.7)
    stored = pr.score_map([box])
    pr.write_detections(dataset, "img0.jpg", [box])
    back = pr.apply_scores(pr.apply_manual_marks(pr.read_detections(dataset, "img0.jpg"), [0]),
                          stored)
    assert back[0].score == 1.0 and "manual" in back[0].flags
