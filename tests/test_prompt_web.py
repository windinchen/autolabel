"""门户 + 提示词标注二级模块的界面契约：页面可取、字段增删改、检测框读写、批量任务与导出。"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from autolabel import fields as fd
from autolabel.dataset import load_index, prompt_label_path_for
from test_upload import jpeg_bytes, upload


def get(base, path):
    request = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            ctype = response.headers.get("Content-Type", "")
            body = response.read()
            if ctype.startswith("application/json"):
                return response.status, json.loads(body), response.headers
            return response.status, body, response.headers
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8", "replace")), exc.headers
        except (json.JSONDecodeError, UnicodeDecodeError):
            return exc.code, raw, exc.headers


def post(base, path, payload):
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read()), response.headers
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw), exc.headers
        except json.JSONDecodeError:
            return exc.code, {"error": raw}, exc.headers


def add_fields(root, *names):
    hints = {"bottle": "桌上或手里的水瓶", "backpack": "椅上或肩上的背包"}
    return [fd.upsert_field(root, name=name, label=name.title(), prompt_hint=hints.get(name, ""))
            for name in names]


@pytest.fixture()
def fake_vlm(monkeypatch):
    """一条假的多模态配置 + 固定回复，避免测试联网。"""
    from autolabel import models as model_manager

    profile = model_manager.upsert_profile({"name": "假模型", "base_url": "http://x/v1",
                                            "model": "m"})
    monkeypatch.setattr(model_manager, "get_profile", lambda pid: dict(profile, image_limit=512))
    monkeypatch.setattr("autolabel.vlm.chat",
                        lambda profile, prompt, data_url:
                        '{"objects":[{"field":"bottle","bbox":[100,100,400,700],"score":0.66}]}')
    return profile


# --------------------------------------------------------------------------- //
# 页面与静态资源
# --------------------------------------------------------------------------- //
@pytest.mark.parametrize("path,page,title", [
    ("/", "portal", "工作台"),
    ("/portal", "portal", "工作台"),
    ("/index.html", "portal", "工作台"),
    ("/pose", "pose", "姿态标注"),
    ("/review", "pose", "姿态标注"),
    ("/objects", "objects", "提示词标注"),
    ("/models", "models", "模型管理"),
])
def test_pages_serve_with_module_marker(server, path, page, title):
    status, body, headers = get(server[0], path)
    text = body.decode("utf-8")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert f'data-page="{page}"' in text
    assert title in text


def test_module_nav_and_shared_assets_are_served(server):
    for asset in ("/common.js", "/portal.js", "/objects.js", "/models.js", "/app.js", "/app.css"):
        status, body, headers = get(server[0], asset)
        assert status == 200, asset
        assert "javascript" in headers["Content-Type"] or "text/css" in headers["Content-Type"]
    status, common, _ = get(server[0], "/common.js")
    text = common.decode("utf-8")
    for shared in ("function initNav", "function confirmBox", "const MODULES", "async function postJson"):
        assert shared in text, shared
    # 顶栏导航与确认弹窗骨架必须出现在每一个页面上
    for path in ("/pose", "/objects", "/models", "/"):
        _status, body, _ = get(server[0], path)
        text = body.decode("utf-8")
        assert 'id="nav-modules"' in text, path
        assert 'id="confirm-modal"' in text, path
        assert "/common.js" in text, path


def test_pose_page_no_longer_houses_model_panel(server):
    """姿态页不藏模型面板，但要去提示词标注的路得留着。

    跳法换了：竖条里那枚图标撤了（五枚同形方框并排，反而看不出自己在哪页），
    模块跳转只剩顶栏中文标签与「⋯」菜单，两处都由 common.js 按 MODULES 现渲染，
    所以 HTML 里看得见的是占位，不是写死的 href。
    """
    _status, body, _ = get(server[0], "/pose")
    text = body.decode("utf-8")
    assert 'id="model-modal"' not in text
    assert '<div id="menu-modules"></div>' in text and 'id="nav-modules"' in text


def test_unknown_route_returns_json_404(server):
    status, payload, _ = get(server[0], "/nope")
    assert status == 404 and "没有这个接口" in payload["error"]


# --------------------------------------------------------------------------- //
# 字段管理
# --------------------------------------------------------------------------- //
def test_dataset_payload_ships_fields_and_prompt_overview(server):
    add_fields(server[1].root, "bottle")
    _status, payload, _ = get(server[0], "/api/dataset")
    assert [f["name"] for f in payload["fields"]["fields"]] == ["bottle"]
    assert payload["fields"]["fields"][0]["color"].startswith("#")
    assert payload["prompt"]["flag_hints"]["small_box"]
    # 提示词状态在 object_items 里，姿态页用的 items 不再掺 prompt_* 字段
    assert payload["items"][0].get("prompt_status") is None
    assert payload["object_items"][0]["prompt_status"] == "unreviewed"
    assert payload["object_items"][0]["prompt_objects"] is None
    assert payload["pool"]["files"] == 2 and payload["pool"]["shared"] == 2


def test_fields_api_roundtrip(server):
    _status, created, _ = post(server[0], "/api/fields/save",
                               {"field": {"name": "bottle", "label": "瓶子",
                                          "prompt_hint": "桌上或手里的水瓶"}})
    assert created["field"]["class_id"] == 0
    assert created["field"]["label"] == "瓶子"
    _status, listed, _ = get(server[0], "/api/fields")
    assert [f["name"] for f in listed["fields"]] == ["bottle"]
    _status, updated, _ = post(server[0], "/api/fields/save",
                               {"field": {"id": created["field"]["id"], "name": "bottle",
                                          "label": "水瓶", "prompt_hint": "换个描述"}})
    assert updated["field"]["class_id"] == 0 and updated["field"]["label"] == "水瓶"
    assert len(updated["fields"]) == 1
    status, payload, _ = post(server[0], "/api/fields/delete", {"id": created["field"]["id"]})
    assert status == 200 and payload["name"] == "bottle"
    assert post(server[0], "/api/fields/delete", {"id": "gone"})[0] == 400


def test_field_api_rejects_bad_names_with_chinese_hint(server):
    status, payload, _ = post(server[0], "/api/fields/save", {"field": {"name": "2 bad"}})
    assert status == 400
    assert "显示名" in payload["error"]
    assert post(server[0], "/api/fields/save", {"field": {}})[0] == 400


def test_delete_field_requires_drop_boxes_when_boxes_exist(server):
    saved = add_fields(server[1].root, "bottle")[0]
    fd.write_label_rows(fd.fields_path(server[1].root).replace("fields.json", "")
                        + "noop", [])
    fd.write_label_rows(prompt_label_path_for(server[1].root, "img0.jpg"),
                        [[saved["class_id"], 0.5, 0.5, 0.2, 0.3]])
    status, payload, _ = post(server[0], "/api/fields/delete", {"name": "bottle"})
    assert status == 400 and "框" in payload["error"]
    status, payload, _ = post(server[0], "/api/fields/delete",
                              {"name": "bottle", "drop_boxes": True})
    assert status == 200 and payload["boxes_removed"] == 1
    assert fd.read_label_rows(prompt_label_path_for(server[1].root, "img0.jpg")) == []


# --------------------------------------------------------------------------- //
# 单张图的检测框读写
# --------------------------------------------------------------------------- //
def test_prompt_item_write_read_and_undo(server):
    add_fields(server[1].root, "bottle", "backpack")
    image = sorted(load_index(server[1].root).items)[0]
    status, item, _ = get(server[0], "/api/prompt/item?image=" + urllib.parse.quote(image))
    assert status == 200 and item["objects"] == []
    assert item["status"] == "unreviewed" and item["width"] > 0

    objects = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.4, "score": 0.9, "flags": []},
               {"cls": 1, "cx": 0.2, "cy": 0.3, "w": 0.1, "h": 0.1, "score": 1, "flags": ["manual"]}]
    status, saved, _ = post(server[0], "/api/prompt/item", {"image": image, "objects": objects})
    assert status == 200
    assert saved["prompt_objects"] == 2 and saved["prompt_status"] == "edited"
    assert saved["prompt_manual"] == 1 and saved["prompt_manual_boxes"] == [1]
    assert saved["prompt_classes"] == [0, 1]
    lines = open(prompt_label_path_for(server[1].root, image), encoding="utf-8").read().strip()
    assert [len(line.split()) for line in lines.splitlines()] == [5, 5]

    _status, reloaded, _ = get(server[0], "/api/prompt/item?image=" + urllib.parse.quote(image))
    assert len(reloaded["objects"]) == 2
    assert reloaded["objects"][1]["flags"] == ["manual"]
    assert reloaded["objects"][1]["score"] == 1.0
    assert reloaded["has_backup"] is True

    status, back, _ = post(server[0], "/api/prompt/undo", {"image": image})
    assert status == 200 and back["prompt_objects"] == 0
    assert back["prompt_status"] == "unreviewed"


def test_prompt_item_rejects_unknown_class_and_zero_area(server):
    add_fields(server[1].root, "bottle")
    image = sorted(load_index(server[1].root).items)[0]
    status, payload, _ = post(server[0], "/api/prompt/item",
                              {"image": image, "objects": [{"cls": 9, "cx": 0.5, "cy": 0.5,
                                                            "w": 0.2, "h": 0.2}]})
    assert status == 400
    assert "类别号 9" in payload["error"] and "bottle" in payload["error"]
    status, payload, _ = post(server[0], "/api/prompt/item",
                              {"image": image, "objects": [{"cls": 0, "cx": 0.5, "cy": 0.5,
                                                            "w": 0, "h": 0.2}]})
    assert status == 400 and "宽高" in payload["error"]
    assert get(server[0], "/api/prompt/item?image=nope.jpg")[0] == 404


def test_prompt_status_endpoint_and_validation(server):
    add_fields(server[1].root, "bottle")
    image = sorted(load_index(server[1].root).items)[0]
    status, payload, _ = post(server[0], "/api/prompt/status", {"image": image, "status": "approved"})
    assert status == 200 and payload["prompt_status"] == "approved"
    assert load_index(server[1].root).get(image)["status"] == "unreviewed"
    bad = post(server[0], "/api/prompt/status", {"image": image, "status": "随便"})
    assert bad[0] == 400


def test_undo_with_kind_prompt_touches_only_prompt_labels(server):
    add_fields(server[1].root, "bottle")
    image = sorted(load_index(server[1].root).items)[0]
    pose_line = "0 0.5 0.5 0.2 0.4 " + " ".join("0.1 0.2 2" for _ in range(17))
    from _helpers import write_label

    write_label(server[1].root, image, [pose_line])
    post(server[0], "/api/prompt/item",
         {"image": image, "objects": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.4}]})
    status, payload, _ = post(server[0], "/api/undo", {"image": image, "kind": "prompt"})
    assert status == 200 and payload["prompt_objects"] == 0
    assert len(open(os.path.join(server[1].root, "labels", "img0.txt"),
                    encoding="utf-8").read().strip().splitlines()) == 1


# --------------------------------------------------------------------------- //
# 批量任务与导出
# --------------------------------------------------------------------------- //
def wait_for_job(base, job_id, tries=200):
    """等任务收尾。被「清除任务」从列表里删掉的记录同样算结束。"""
    for _ in range(tries):
        _status, payload, _ = get(base, "/api/jobs")
        job = next((j for j in payload["jobs"] if j["id"] == job_id), None)
        if job is None:
            return {"finished": True, "dropped": True}
        if job["finished"]:
            return job
        time.sleep(0.1)
    raise AssertionError("任务没有结束")


def test_prompt_run_labels_images_through_job_queue(server, fake_vlm):
    add_fields(server[1].root, "bottle")
    base = server[0]
    status, job, _ = post(base, "/api/prompt/run", {"fields": ["bottle"], "vlm_profile": fake_vlm["id"]})
    assert status == 202 and job["kind"] == "prompt"
    done = wait_for_job(base, job["id"])
    assert done["ok"] is True, done
    assert done["result"]["labeled"] == 2 and done["result"]["boxes"] == 2
    idx = load_index(server[1].root, "objects")
    for image in idx.items:
        assert idx.get(image)["prompt_objects"] == 1
        assert idx.get(image)["prompt_profile"] == "假模型"
        assert os.path.exists(prompt_label_path_for(server[1].root, image))
    _status, summary, _ = get(base, "/api/prompt/summary")
    assert summary["prompt"]["labeled"] == 2 and summary["prompt"]["boxes"] == 2
    assert summary["fields"]["fields"][0]["boxes"] == 2


def test_prompt_run_reports_protected_human_work(server, fake_vlm):
    """人工接受过/画过框的图算进 protected，完成消息要说明为什么只标了 0 张。"""
    base = server[0]
    add_fields(server[1].root, "bottle")
    first = wait_for_job(base, post(base, "/api/prompt/run",
                                   {"fields": ["bottle"], "vlm_profile": fake_vlm["id"]})[1]["id"])
    assert first["result"]["protected"] == 0

    images = sorted(load_index(server[1].root).items)
    post(base, "/api/prompt/status", {"image": images[0], "status": "approved"})
    post(base, "/api/prompt/item",
         {"image": images[1], "objects": [{"cls": 0, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3,
                                           "flags": ["manual"]}]})

    again = wait_for_job(base, post(base, "/api/prompt/run",
                                   {"fields": ["bottle"], "prompt_extra": "换个说法"})[1]["id"])
    assert again["ok"] is True, again
    assert again["result"]["labeled"] == 0 and again["result"]["protected"] == 2
    assert "未覆盖" in again["message"]


def test_prompt_run_reuses_task_store_and_honours_clear(server, fake_vlm, monkeypatch):
    import autolabel.prompt as pr

    add_fields(server[1].root, "bottle")
    monkeypatch.setattr(pr, "label_image",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("慢")))
    _status, job, _ = post(server[0], "/api/prompt/run", {"fields": ["bottle"]})
    status, payload, _ = post(server[0], "/api/jobs/clear", {})
    assert status == 200
    assert payload["cancelled"] + payload["cancelling"] >= 1
    done = wait_for_job(server[0], job["id"])
    assert done["finished"] is True


def test_prompt_run_validates_input(server, fake_vlm):
    assert post(server[0], "/api/prompt/run", {"fields": ["nope"]})[0] == 400
    add_fields(server[1].root, "bottle")
    assert post(server[0], "/api/prompt/run", {"fields": ["bottle"],
                                              "vlm_profile": "ghost"})[0] == 400


def test_prompt_coords_download_is_attachment(server):
    add_fields(server[1].root, "bottle", "backpack")
    image = sorted(load_index(server[1].root).items)[0]
    post(server[0], "/api/prompt/item",
         {"image": image, "objects": [{"cls": 1, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}]})
    for fmt, needle in (("csv", "class_name"), ("jsonl", "backpack")):
        status, body, headers = get(server[0], "/api/prompt/coords?format=" + fmt)
        assert status == 200, fmt
        assert "attachment;" in headers["Content-Disposition"]
        assert f"ds-objects.{fmt}" in headers["Content-Disposition"]
        # jsonl 会被 get() 直接解析成对象，这里统一还原成文本再比对
        text = body.decode("utf-8") if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False)
        assert needle in text, fmt
    status, payload, _ = get(server[0], "/api/prompt/coords?format=xml")
    assert status == 400 and "未知格式" in payload["error"]


def test_prompt_export_builds_yolo_dataset(server):
    add_fields(server[1].root, "bottle", "backpack")
    for offset, image in enumerate(sorted(load_index(server[1].root).items)):
        post(server[0], "/api/prompt/item",
             {"image": image,
              "objects": [{"cls": offset, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}],
              "status": "approved"})
    out = os.path.join(server[1].root, "exports", "web-detect")
    status, payload, _ = post(server[0], "/api/prompt/export",
                              {"out": out, "val_ratio": 0.5, "tag": "web"})
    assert status == 200
    info = payload["export"]
    assert info["train"] + info["val"] == 2
    assert os.path.exists(os.path.join(out, "data.yaml"))
    assert {c["name"] for c in info["classes"]} == {"bottle", "backpack"}
    assert info["boxes"]["train"] + info["boxes"]["val"] == 2


def test_portal_data_comes_from_one_dataset_call(server):
    add_fields(server[1].root, "bottle")
    image = sorted(load_index(server[1].root).items)[0]
    post(server[0], "/api/prompt/item",
         {"image": image, "objects": [{"cls": 0, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3,
                                       "flags": ["manual"]}]})
    _status, payload, _ = get(server[0], "/api/dataset")
    labeled = [i for i in payload["object_items"] if i["prompt_objects"]]
    assert len(labeled) == 1
    assert labeled[0]["prompt_objects"] == 1
    assert labeled[0]["prompt_priority"] >= 0
    assert payload["prompt"]["boxes"] == 1
    assert payload["batches"] is not None
    assert "vlm" in payload and "model_rows" in payload


def test_dataset_endpoint_picks_up_external_index_writes(server):
    """服务跑着的时候用命令行/脚本导入图片，刷新界面就该看到，不该要重启。"""
    import shutil

    from autolabel.dataset import import_images

    base, store = server
    root = store.root
    assert len(get(base, "/api/dataset")[1]["items"]) == 2

    first = os.path.join(root, "images", sorted(load_index(root).items)[0])
    drop = os.path.join(os.path.dirname(root), "cli_src")
    os.makedirs(drop, exist_ok=True)
    shutil.copyfile(first, os.path.join(drop, "from_cli.jpg"))
    import_images(root, [drop], dedupe="none")

    items = get(base, "/api/dataset")[1]["items"]
    assert len(items) == 3
    assert any(str(i["image"]).endswith("from_cli.jpg") for i in items)


# --------------------------------------------------------------------------- //
# 门户批次管理：按模块看进度、展开看文件、按张删除
# --------------------------------------------------------------------------- //
def test_batches_payload_groups_progress_by_module(server):
    from autolabel.batches import attach_items, create_batch, finish_batch

    root = server[1].root
    image = sorted(load_index(root).items)[0]
    batch = create_batch(root, name="姿态批次", kind="video")
    attach_items(root, batch["id"], [image])
    finish_batch(root, batch["id"], "done")

    _status, payload, _ = get(server[0], "/api/batches")
    rows = {row["id"]: row for row in payload["batches"]}
    assert rows[batch["id"]]["module"] == "pose"
    # 姿态库 = 这次建的批次 + 剩下 1 张的"未分组"虚拟行，两本账分开算
    pose_side = {key: payload["by_module"]["pose"][key]
                 for key in ("label", "batches", "items", "labeled", "people",
                             "reviewed", "unreviewed", "flagged")}
    assert pose_side == {"label": "姿态标注", "batches": 2, "items": 2, "labeled": 0,
                         "people": 0, "reviewed": 0, "unreviewed": 2, "flagged": 0}
    assert payload["by_module"]["objects"]["batches"] == 1
    assert payload["by_module"]["objects"]["items"] == 2
    assert payload["module_names"]["objects"] == "提示词标注"

    # 门户只调一次 /api/dataset 也要拿到同一份汇总
    _status2, ds, _ = get(server[0], "/api/dataset")
    assert ds["by_module"]["pose"]["batches"] == 2
    assert {row["id"] for row in ds["batches"]} == {row["id"] for row in payload["batches"]}


def test_batch_files_and_single_image_delete(server):
    base, store = server
    add_fields(store.root, "bottle")
    images = sorted(load_index(store.root).items)
    post(base, "/api/prompt/item",
         {"image": images[0], "objects": [{"cls": 0, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}]})

    _status, payload, _ = get(base, "/api/batch/files?id=__ungrouped__:pose")
    assert sorted(row["image"] for row in payload["files"]) == images
    row = next(row for row in payload["files"] if row["image"] == images[0])
    # 手工画过框的图状态是 edited，模型自动标的才是 unreviewed
    assert row["prompt_objects"] == 1 and row["prompt_status"] == "edited"
    assert row["status"] == "unreviewed" and row["bytes"] > 0
    assert row["shared"] is True

    # 「从本库移除」只摘姿态库引用：素材文件与提示词库的框都还在
    status, only_pose, _ = post(base, "/api/images/delete", {"images": [images[0]], "lib": "pose"})
    assert status == 200, only_pose
    assert images[0] not in load_index(store.root).items
    assert images[0] in load_index(store.root, "objects").items
    assert os.path.exists(prompt_label_path_for(store.root, images[0]))
    assert os.path.exists(os.path.join(store.root, "images", images[0]))

    # 「删素材」两个库一起摘，文件和框都清掉
    status, gone, _ = post(base, "/api/images/delete",
                           {"images": [images[0]], "lib": "objects", "mode": "purge"})
    assert status == 200 and gone["items"] == 1 and gone["files_removed"] >= 2
    assert images[0] not in load_index(store.root, "objects").items
    assert images[1] in load_index(store.root).items
    assert not os.path.exists(prompt_label_path_for(store.root, images[0]))
    assert not os.path.exists(os.path.join(store.root, "images", images[0]))
    assert os.path.exists(gone["snapshot"])

    assert post(base, "/api/images/delete", {"images": []})[0] == 400
    assert "不在索引里" in post(base, "/api/images/delete", {"images": ["ghost.jpg"]})[1]["error"]
    assert get(base, "/api/batch/files")[0] == 400


def test_prompt_upload_job_labels_new_uploads_without_pose(server, fake_vlm):
    """提示词模块独立上传：入库后直接出检测框，不跑姿态模型。"""
    base, store = server
    add_fields(store.root, "bottle")
    before = set(load_index(store.root, "objects").items)

    status, _headers, body = upload(base, "kitchen.jpg", jpeg_bytes(7))
    assert status == 200, body
    stored = json.loads(body)["stored"]

    status, job, _ = post(base, "/api/prompt/upload",
                         {"files": [stored], "fields": ["bottle"],
                          "vlm_profile": fake_vlm["id"], "batch_name": "提示词独立上传"})
    assert status == 202 and job["kind"] == "prompt_upload"
    done = wait_for_job(base, job["id"])
    assert done["ok"] is True, json.dumps(done, ensure_ascii=False)
    assert done["result"]["imported"] == 1 and done["result"]["labeled"] == 1
    assert done["result"]["boxes"] == 1

    added = set(load_index(store.root, "objects").items) - before
    assert len(added) == 1
    new = added.pop()
    item = load_index(store.root, "objects").get(new)
    assert item["prompt_objects"] == 1 and "objects" not in item
    # 素材池里落了一份，姿态库还没有引用它
    assert new not in load_index(store.root).items
    assert os.path.exists(prompt_label_path_for(store.root, new))
    assert not os.path.exists(os.path.join(store.root, ".autolabel", "uploads", stored))

    rows = {row["id"]: row for row in get(base, "/api/batches")[1]["batches"]}
    row = rows[done["batch_id"]]
    assert row["module"] == "objects" and row["name"] == "提示词独立上传"
    assert row["prompt"]["boxes"] == 1 and row["pose"]["labeled"] == 0
    # 提示词库 = 新批次 + 两张老图留下的"未分组"虚拟行
    assert get(base, "/api/dataset")[1]["by_module"]["objects"]["batches"] == 2


def test_prompt_upload_without_labeling_just_imports(server, fake_vlm):
    base, store = server
    before = set(load_index(store.root, "objects").items)
    _status, _headers, body = upload(base, "only-import.jpg", jpeg_bytes(11))
    stored = json.loads(body)["stored"]

    _status, job, _ = post(base, "/api/prompt/upload",
                          {"files": [stored], "label": False, "batch_name": "只入库"})
    done = wait_for_job(base, job["id"])
    assert done["ok"] is True, done
    assert done["result"]["imported"] == 1 and done["result"]["labeled"] == 0

    added = set(load_index(store.root, "objects").items) - before
    assert len(added) == 1
    assert load_index(store.root, "objects").get(added.pop()).get("prompt_objects") is None


def test_prompt_upload_validates_input(server, fake_vlm):
    base, store = server
    assert "没有收到文件" in post(base, "/api/prompt/upload", {"files": []})[1]["error"]

    add_fields(store.root, "bottle")
    _status, _headers, body = upload(base, "bad-field.jpg", jpeg_bytes(13))
    stored = json.loads(body)["stored"]
    assert "未知字段" in post(base, "/api/prompt/upload",
                            {"files": [stored], "fields": ["ghost"]})[1]["error"]
    assert "上传文件不存在" in post(base, "/api/prompt/upload",
                              {"files": ["ghost_uploader.jpg"], "fields": ["bottle"],
                               "vlm_profile": fake_vlm["id"]})[1]["error"]
