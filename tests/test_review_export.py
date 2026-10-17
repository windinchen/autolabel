import json
import os

import pytest
import yaml

from _helpers import idx_names, write_label
from test_prompt_web import add_fields, get, post
from autolabel import dataset as ds
from autolabel import export as ex
from autolabel import qc
from autolabel import review as rv

SAMPLE = "0 0.500000 0.500000 0.200000 0.400000 " + " ".join(
    f"{0.42 + i * 0.005:.6f} {0.35 + i * 0.02:.6f} 2" for i in range(17))


def seed(dataset, first_only=True):
    names = sorted(idx_names(dataset))
    write_label(dataset, names[0], [SAMPLE])
    if not first_only:
        write_label(dataset, names[1], [SAMPLE])
    idx = ds.load_index(dataset)
    for rel in names:
        item = idx.get(rel)
        item["objects"] = 1
        item["score"] = 0.77
    idx.save()
    return names


def test_write_item_writes_label_and_backs_up_original(dataset):
    first = seed(dataset, first_only=True)[0]
    store = rv.ReviewStore(dataset)
    item = store.read_item(first)
    assert len(item["objects"]) == 1
    payload = item["objects"]
    payload[0]["keypoints"][0]["x"] = 0.99
    payload[0]["keypoints"][0]["v"] = 0
    result = store.write_item(first, payload, status="approved")
    assert result["status"] == "approved"
    assert result["objects"][0]["keypoints"][0]["v"] == 0
    # YOLO 标签不保存模型置信度，索引需保留历史分值，否则会被误判为低置信
    assert ds.load_index(dataset).get(first)["score"] == pytest.approx(0.77)
    assert os.path.exists(os.path.join(dataset, ".autolabel", "originals", "img0.txt"))
    lines = open(ds.label_path_for(dataset, first), encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 1
    assert len(lines[0].split()) == 5 + 17 * 3


def test_write_item_marks_edited_by_default(dataset):
    first = seed(dataset)[0]
    store = rv.ReviewStore(dataset)
    result = store.write_item(first, store.read_item(first)["objects"])
    assert result["status"] == ds.STATUS_EDITED


def test_heal_bbox_expands_around_keypoints():
    person = rv.persons_from_payload(
        [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.04, "h": 0.04,
          "keypoints": [{"x": 0.1, "y": 0.1, "v": 2}, {"x": 0.9, "y": 0.9, "v": 2}]}], 1000, 1000)[0]
    x1, y1, x2, y2 = person.xyxy(1000, 1000)
    assert x1 < 100 and y1 < 100 and x2 > 900 and y2 > 900


def test_persons_from_payload_clamps_out_of_range():
    person = rv.persons_from_payload(
        [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
          "keypoints": [{"x": 1.7, "y": -0.4, "v": 2}]}], 800, 600)[0]
    k = person.keypoints[0]
    assert 0.0 <= k.x <= 1.0 and 0.0 <= k.y <= 1.0


def test_undo_restores_previous_label(dataset):
    first = seed(dataset)[0]
    store = rv.ReviewStore(dataset)
    original = open(ds.label_path_for(dataset, first), encoding="utf-8").read()
    store.write_item(first, [], status="rejected")
    assert open(ds.label_path_for(dataset, first), encoding="utf-8").read().strip() == ""
    store.undo(first)
    assert open(ds.label_path_for(dataset, first), encoding="utf-8").read() == original
    assert ds.load_index(dataset).get(first)["status"] == ds.STATUS_UNREVIEWED


def test_status_validation_and_unknown_image(dataset):
    first = seed(dataset)[0]
    store = rv.ReviewStore(dataset)
    assert store.set_status(first, "rejected")["status"] == "rejected"
    with pytest.raises(ValueError):
        store.set_status(first, "weird")
    with pytest.raises(KeyError):
        store.read_item("nope.jpg")


def test_dataset_payload_shape(dataset):
    seed(dataset)
    payload = rv.ReviewStore(dataset).dataset_payload()
    assert payload["num_keypoints"] == 17
    assert len(payload["keypoint_names"]) == 17
    assert payload["items"] and {"image", "priority", "flags", "status"} <= set(payload["items"][0])
    assert payload["skeleton"] and max(max(e) for e in payload["skeleton"]) < 17


def test_qc_flags_drive_priority(dataset):
    first = seed(dataset)[0]
    idx = ds.load_index(dataset)
    idx.upsert(first, flags=["no_person"], object_flags=["tiny_box"], score=0.2, visible_ratio=0.1)
    idx.save()
    flagged = qc.flagged_items(dataset)
    assert [rel for rel, _item, _p in flagged] == [first]
    stats = qc.write_report(dataset)
    assert stats["flag_counts"]["no_person"] == 1
    report = open(stats["report"], encoding="utf-8").read()
    assert "优先复核清单" in report and first in report
    assert os.path.exists(os.path.join(dataset, "qc", "flags.csv"))


def test_export_splits_and_writes_data_yaml(dataset):
    names = seed(dataset, first_only=False)
    idx = ds.load_index(dataset)
    idx.upsert(names[1], status="rejected")
    idx.save()
    result = ex.export_dataset(dataset, out=os.path.join(dataset, "exports", "t"), val_ratio=0.5,
                              seed=3, mode="copy")
    assert result["train_images"] == 1  # rejected 被排除
    assert result["val_images"] == 0    # 只剩 1 张可导出时不强拆 val
    data = yaml.safe_load(open(os.path.join(result["out"], "data.yaml"), encoding="utf-8"))
    assert data["kpt_shape"] == [17, 3]
    assert data["flip_idx"] == [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    assert data["names"] == {0: "person"}
    manifest = json.load(open(os.path.join(result["out"], "manifest.json"), encoding="utf-8"))
    assert any(s["reason"].startswith("status=rejected") for s in manifest["skipped"])
    label = open(os.path.join(result["out"], "labels", "train", "img0.txt"), encoding="utf-8").read()
    assert label.strip().split()[0] == "0"


def test_export_coco_and_empty_background_cap(dataset):
    names = sorted(idx_names(dataset))
    write_label(dataset, names[0], [SAMPLE])
    for rel in names:
        write_label(dataset, rel, [SAMPLE] if rel == names[0] else [])
    idx = ds.load_index(dataset)
    for rel in names:
        idx.upsert(rel, objects=1 if rel == names[0] else 0, width=640, height=480)
    idx.save()
    out = os.path.join(dataset, "exports", "coco")
    result = ex.export_dataset(dataset, out=out, val_ratio=0.0, mode="copy", coco=True,
                              max_empty_ratio=0.0)
    coco = json.load(open(os.path.join(out, "coco_keypoints.json"), encoding="utf-8"))
    assert coco["categories"][0]["keypoints"][0] == "nose"
    assert coco["categories"][0]["skeleton"][0] == [1, 2]
    ann = coco["annotations"][0]
    assert len(ann["keypoints"]) == 51
    assert ann["num_keypoints"] == 17
    assert len(ann["bbox"]) == 4 and ann["area"] > 0
    # max_empty_ratio=0 时背景图不应进入导出集
    assert result["train_images"] == 1


def test_export_always_keeps_one_validation_image(dataset):
    """val_ratio 很小时也要留 1 张验证图，否则训练时拿不到 pose 指标。"""
    names = seed(dataset, first_only=False)
    for rel in names:
        write_label(dataset, rel, [SAMPLE])
    result = ex.export_dataset(dataset, out=os.path.join(dataset, "exports", "v"), val_ratio=0.01,
                              seed=1, mode="copy")
    assert result["train_images"] == 1 and result["val_images"] == 1
    assert os.path.exists(os.path.join(result["out"], "labels", "val", "img0.txt")) or \
        os.path.exists(os.path.join(result["out"], "labels", "val", "img1.txt"))


def test_export_val_never_eats_whole_dataset(dataset):
    names = seed(dataset, first_only=False)
    for rel in names:
        write_label(dataset, rel, [SAMPLE])
    result = ex.export_dataset(dataset, out=os.path.join(dataset, "exports", "full"), val_ratio=1.0,
                              seed=1, mode="copy")
    assert result["train_images"] >= 1 and result["val_images"] >= 1


def test_video_showinfo_timestamps():
    from autolabel.video import _parse_showinfo_times

    stderr = "frame:0  pts:1024  pts_time:1.5\nframe:1  pts:2048  pts_time:3.25\nnoise\n"
    assert _parse_showinfo_times(stderr) == [1.5, 3.25]
    assert _parse_showinfo_times("") == []


def test_backend_settings_signature_tracks_parameters():
    from autolabel.backends import BackendSettings
    from autolabel.pipeline import settings_signature

    base = BackendSettings()
    assert settings_signature(base) == settings_signature(BackendSettings())
    assert settings_signature(base) != settings_signature(BackendSettings(imgsz=960))
    assert settings_signature(base) != settings_signature(BackendSettings(model="yolo11s-pose.pt"))


# --------------------------------------------------------------------------- #
# 界面上的「YOLO 训练集」按钮：POST /api/export 落地成可直接训练的标准目录
# --------------------------------------------------------------------------- #
def test_api_export_pose_builds_standard_yolo_dataset(server):
    base, store = server
    names = seed(store.root, first_only=False)
    out = os.path.join(store.root, "exports", "web-pose")
    status, payload, _ = post(base, "/api/export", {"out": out, "val_ratio": 0.5, "tag": "web"})
    assert status == 200, payload
    info = payload["export"]
    assert info["train"] + info["val"] == len(names)
    assert info["people"]["train"] + info["people"]["val"] == len(names)
    assert info["data_yaml"] == os.path.join(out, "data.yaml")
    text = open(info["data_yaml"], encoding="utf-8").read()
    assert "kpt_shape: [17, 3]" in text and "train: images/train" in text
    assert os.path.exists(os.path.join(out, "manifest.json"))
    placed = [os.path.join(dirpath, name)
              for dirpath, _dirs, files in os.walk(os.path.join(out, "labels"))
              for name in files if name.endswith(".txt")]
    assert len(placed) == len(names)


def test_api_export_pose_exports_coco_when_asked(server):
    base, store = server
    seed(store.root, first_only=False)
    out = os.path.join(store.root, "exports", "web-coco")
    status, payload, _ = post(base, "/api/export", {"out": out, "tag": "coco", "coco": True})
    assert status == 200, payload
    coco = os.path.join(payload["export"]["out"], "coco_keypoints.json")
    assert os.path.exists(coco)
    coco_data = json.load(open(coco, encoding="utf-8"))
    assert coco_data["annotations"] and len(coco_data["images"]) == 2
    assert len(coco_data["annotations"][0]["keypoints"]) == 51


def test_api_export_pose_scope_only_keeps_approved(server):
    base, store = server
    names = seed(store.root, first_only=False)
    store.write_item(names[0], store.read_item(names[0])["objects"], status="approved")
    out = os.path.join(store.root, "exports", "web-approved")
    status, payload, _ = post(base, "/api/export", {"out": out, "scope": "approved", "tag": "ap"})
    assert status == 200, payload
    info = payload["export"]
    assert info["scope"] == "approved"
    assert info["train"] + info["val"] == 1


def test_api_export_pose_rejects_when_nothing_to_export(server):
    base, store = server
    status, payload, _ = post(base, "/api/export", {})
    assert status == 400
    assert "骨骼标签" in payload["error"]
    # 只有未复核标签时，approved 范围同样导不出东西，要给同样的话而不是空目录
    seed(store.root, first_only=True)
    status, payload, _ = post(base, "/api/export", {"scope": "approved"})
    assert status == 400
    assert "骨骼标签" in payload["error"]


def test_api_export_pose_rejects_unknown_scope(server):
    base, store = server
    seed(store.root, first_only=True)
    status, payload, _ = post(base, "/api/export", {"scope": "everything"})
    assert status == 400
    assert "导出范围" in payload["error"] or "scope" in payload["error"]


def test_yolo_dataset_buttons_shipped_next_to_json_exports():
    """两个模块页顶栏与门户导出行都要有「YOLO 训练集」入口，和坐标 JSON 同级。"""
    static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "autolabel", "static")
    for page in ("review.html", "objects.html"):
        html = open(os.path.join(static, page), encoding="utf-8").read()
        assert "坐标 JSONL" in html, page
        assert 'id="btn-export-yolo"' in html, page
        assert 'id="export-note"' in html, page
        assert html.index("坐标 JSONL") < html.index("btn-export-yolo"), page
    portal = open(os.path.join(static, "portal.js"), encoding="utf-8").read()
    assert '"/api/export"' in portal and '"export-pose"' in portal
    assert '"export-detect"' in portal


def test_dataset_payload_reports_recent_exports(server):
    """导出产物是目录，路径要靠 /api/dataset 常驻回给界面。"""
    base, store = server
    seed(store.root, first_only=False)
    status, payload, _ = post(base, "/api/export", {"tag": "recent", "val_ratio": 0.5})
    assert status == 200, payload
    _status, dataset, _ = get(base, "/api/dataset")
    rows = [row for row in dataset["exports_recent"] if row["kind"] == "pose"]
    assert rows, dataset["exports_recent"]
    assert rows[0]["train"] + rows[0]["val"] == 2
    assert os.path.exists(rows[0]["data_yaml"])
    assert rows[0]["at"]
    # 最新的排在最前
    assert rows[0]["path"] == payload["export"]["out"]


def test_recent_exports_marks_detection_kind(server):
    base, store = server
    add_fields(store.root, "bottle", "backpack")
    for offset, image in enumerate(sorted(ds.load_index(store.root).items)):
        post(base, "/api/prompt/item",
             {"image": image,
              "objects": [{"cls": offset, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}],
              "status": "approved"})
    status, payload, _ = post(base, "/api/prompt/export", {"tag": "recent"})
    assert status == 200, payload
    _status, dataset, _ = get(base, "/api/dataset")
    rows = [row for row in dataset["exports_recent"] if row["kind"] == "detect"]
    assert rows and rows[0]["train"] + rows[0]["val"] == 2


def test_repeat_pose_export_reuses_the_same_folder(server):
    """标签没改时连点导出不再堆重复目录；一改关键点就必须重新导出，不能拿旧目录糊弄。"""
    base, store = server
    names = seed(store.root, first_only=False)
    status, first, _ = post(base, "/api/export", {"val_ratio": 0.5})
    assert status == 200, first
    assert not first["export"].get("reused")

    status, again, _ = post(base, "/api/export", {"val_ratio": 0.5})
    assert status == 200, again
    assert again["export"].get("reused") is True, again
    assert again["export"]["out"] == first["export"]["out"]
    assert again["export"]["train"] + again["export"]["val"] == 2

    # 手工微调一个数值：文件字节数没变，只看大小会漏判，所以这里必须重导
    label = ds.label_path_for(store.root, names[0])
    body = open(label, encoding="utf-8").read().strip()
    fields = body.split(" ")
    fields[1] = "0.499999"
    with open(label, "w", encoding="utf-8") as fh:
        fh.write(" ".join(fields) + "\n")
    status, third, _ = post(base, "/api/export", {"val_ratio": 0.5})
    assert status == 200, third
    assert not third["export"].get("reused")
    assert third["export"]["out"] != first["export"]["out"]


def test_export_scope_change_is_not_treated_as_duplicate(server):
    """导出范围不同就不是同一份产物，不能复用旧目录。"""
    base, store = server
    seed(store.root, first_only=False)
    status, first, _ = post(base, "/api/export", {"val_ratio": 0.5})
    assert status == 200, first
    status, same, _ = post(base, "/api/export", {"val_ratio": 0.5, "scope": "all"})
    assert status == 200, same
    assert not same["export"].get("reused")
    assert same["export"]["out"] != first["export"]["out"]


def test_undo_without_history_reports_plain_words(server):
    """没保存过时按撤销：要说「这张图还没保存过」，别把 Python 异常原文甩到界面上。"""
    base, store = server
    first = seed(store.root)[0]
    status, payload, _ = post(base, "/api/undo", {"image": first})
    assert status == 400, payload
    message = payload["error"]
    assert "没有可撤销的修改" in message, message
    assert "未知对象" not in message and "'" not in message, message


def test_detection_export_reuses_and_reacts_to_edits(server):
    """提示词库同样只在内容变化时生成新目录，删掉一个框就得重导。"""
    base, store = server
    add_fields(store.root, "bottle", "backpack")
    images = sorted(ds.load_index(store.root).items)
    for offset, image in enumerate(images):
        post(base, "/api/prompt/item",
             {"image": image,
              "objects": [{"cls": offset, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}],
              "status": "approved"})
    status, first, _ = post(base, "/api/prompt/export", {})
    assert status == 200, first
    assert not first["export"].get("reused")

    status, again, _ = post(base, "/api/prompt/export", {})
    assert status == 200, again
    assert again["export"].get("reused") is True, again
    assert again["export"]["out"] == first["export"]["out"]
    assert again["export"]["classes"], again

    # 真的挪一下框，内容才算变了
    post(base, "/api/prompt/item",
         {"image": images[0], "objects": [{"cls": 0, "cx": 0.33, "cy": 0.5, "w": 0.2, "h": 0.3}],
          "status": "approved"})
    status, third, _ = post(base, "/api/prompt/export", {})
    assert status == 200, third
    assert not third["export"].get("reused")
    assert third["export"]["out"] != first["export"]["out"]


def test_shortcut_help_only_advertises_bound_keys():
    """页面右侧的快捷键说明不能写没绑定的键，否则用户照着按只会以为界面坏了。"""
    import re

    static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "autolabel", "static")
    for page, script in (("review.html", "app.js"), ("objects.html", "objects.js")):
        html = open(os.path.join(static, page), encoding="utf-8").read()
        js = open(os.path.join(static, script), encoding="utf-8").read()
        advertised = set()
        for line in re.findall(r"<div><span>([^<]+)</span>", html):
            for token in re.split(r"[/\uFF08\uFF09\s]+", line):
                if len(token) == 1 and token.isalpha() and token.isascii():
                    advertised.add(token.lower())
        assert advertised, page
        missing = sorted(key for key in advertised if f'case "{key}"' not in js)
        assert not missing, f"{page} 写了没绑定的快捷键：{missing}"

    objects = open(os.path.join(static, "objects.js"), encoding="utf-8").read()
    assert 'case "0"' in objects  # 帮助里写了「0 删框」
    review_html = open(os.path.join(static, "review.html"), encoding="utf-8").read()
    assert 'id="up-batch-name"' in review_html
    app = open(os.path.join(static, "app.js"), encoding="utf-8").read()
    assert 'batch_name: $("up-batch-name")' in app


def test_reveal_export_opens_folder_and_blocks_stray_paths(server, monkeypatch):
    """「没找到导出文件」要靠系统文件管理器定位，同时不能借这个接口打开任意路径。"""
    import subprocess

    base, store = server
    seed(store.root, first_only=False)
    opened = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: opened.append(cmd))

    status, payload, _ = post(base, "/api/export/reveal", {"path": "/etc/passwd"})
    assert status == 400, payload
    status, payload, _ = post(base, "/api/export/reveal", {})
    assert status == 400, payload
    status, payload, _ = post(base, "/api/export/reveal",
                              {"path": os.path.join(store.root, "exports", "not-here")})
    assert status == 400, payload
    assert opened == []

    status, payload, _ = post(base, "/api/export", {"tag": "reveal"})
    assert status == 200, payload
    out = payload["export"]["out"]
    status, payload, _ = post(base, "/api/export/reveal", {"path": out})
    assert status == 200, payload
    assert payload["path"] == os.path.realpath(out)
    assert len(opened) == 1 and opened[0][-1] == os.path.realpath(out)


def test_favicon_route_serves_icon_instead_of_404(server):
    """页面虽然内联了图标，但浏览器仍可能自动请求 /favicon.ico，不能留一条红色 404。"""
    import urllib.request

    base, _store = server
    with urllib.request.urlopen(base + "/favicon.ico", timeout=5) as res:
        body = res.read()
    assert res.status == 200
    assert res.headers.get_content_type().startswith("image/")
    assert body[:4] == b"\x00\x00\x01\x00"  # ICO 文件头


def test_export_note_ui_ships_reveal_and_copy_buttons():
    """导出行必须自带「打开所在目录 + 复制路径」，否则用户还是不知道文件在哪。"""
    static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "autolabel", "static")
    common = open(os.path.join(static, "common.js"), encoding="utf-8").read()
    assert "function exportRevealButton" in common
    assert "function revealExport" in common
    assert "/api/export/reveal" in common
    assert 'el.append(text, exportRevealButton(row.path), exportCopyButton("复制路径", row.path)' in common
    # 三个导出入口都要在导出成功后自动帮忙开一次目录
    for page in ("app.js", "objects.js", "portal.js"):
        script = open(os.path.join(static, page), encoding="utf-8").read()
        assert "revealExport(info.out" in script, page
        assert "openedSuffix(opened)" in script, page

