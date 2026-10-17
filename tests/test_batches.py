"""批次登记、按批次筛选与整批删除。"""

import json
import os
import shutil

import pytest
from _helpers import idx_names, write_label
from test_upload import SAMPLE, call, jpeg_bytes, post_json, upload, wait_job

from autolabel.batches import (
    UNGROUPED_ID,
    attach_items,
    batch_file_rows,
    batch_name,
    create_batch,
    delete_batches,
    delete_images,
    display_name,
    finish_batch,
    list_batches,
    load_batches,
    module_for,
    reconcile_with_disk,
    save_batches,
)
from autolabel.cli import main
from autolabel.dataset import images_dir, label_path_for, load_index, resolve_image

POSE_UNGROUPED = f"{UNGROUPED_ID}:pose"
OBJECTS_UNGROUPED = f"{UNGROUPED_ID}:objects"


def objects_index(dataset):
    return load_index(dataset, "objects")


def test_batch_registration_and_ungrouped(dataset):
    names = sorted(idx_names(dataset))
    batch = create_batch(dataset, name="测试批次", kind="image", source="cli", uploads=["a.jpg", "b.jpg"])
    assert batch["status"] == "running"

    assert attach_items(dataset, batch["id"], [names[0]]) == 1
    finish_batch(dataset, batch["id"], "done", "导入 1 张")

    rows = {row["id"]: row for row in list_batches(dataset)}
    assert rows[batch["id"]]["name"] == "测试批次"
    assert rows[batch["id"]]["status"] == "done"
    assert rows[batch["id"]]["items"] == 1
    assert rows[batch["id"]]["message"] == "导入 1 张"
    assert rows[POSE_UNGROUPED]["items"] == 1  # 另一张没挂批次
    assert rows[OBJECTS_UNGROUPED]["items"] == 2  # 提示词库的未分组是分开的两本账
    assert rows[batch["id"]]["lib"] == "pose"
    assert load_index(dataset).get(names[0])["batch_id"] == batch["id"]


def test_delete_batch_removes_files_and_keeps_snapshot(dataset):
    """提示词库不再引用这张图后，整批删除才会真的把素材文件删掉。"""
    names = sorted(idx_names(dataset))
    write_label(dataset, names[0], [SAMPLE])
    # 先把提示词库的引用摘掉：这张素材就只剩姿态库在用了
    delete_images(dataset, [names[0]], lib="objects")
    batch = create_batch(dataset, name="整批删除", kind="image")
    attach_items(dataset, batch["id"], [names[0]])
    finish_batch(dataset, batch["id"], "done")

    result = delete_batches(dataset, [batch["id"]])

    assert result["items"] == 1 and result["files_removed"] >= 2
    assert not os.path.exists(resolve_image(dataset, names[0]))
    assert not os.path.exists(label_path_for(dataset, names[0]))
    assert names[0] not in load_index(dataset).items
    assert batch["id"] not in {b["id"] for b in load_batches(dataset)}
    assert names[1] in load_index(dataset).items  # 只删选中批次

    with open(result["snapshot"], encoding="utf-8") as fh:
        snap = json.load(fh)
    assert snap["items"][names[0]]["label_lines"] == [SAMPLE]
    assert snap["batches"][0]["name"] == "整批删除"


def test_running_batch_cannot_be_deleted(dataset):
    names = sorted(idx_names(dataset))
    batch = create_batch(dataset, name="还在标")
    attach_items(dataset, batch["id"], [names[0]])
    with pytest.raises(ValueError, match="仍在标注中"):
        delete_batches(dataset, [batch["id"]])
    assert names[0] in load_index(dataset).items


def test_delete_ungrouped_can_keep_files(dataset):
    names = sorted(idx_names(dataset))
    result = delete_batches(dataset, [UNGROUPED_ID], keep_files=True)
    assert result["items"] == 2 and result["files_removed"] == 0
    assert not load_index(dataset).items
    assert os.path.exists(resolve_image(dataset, names[0]))


def test_delete_regenerates_coords_when_present(dataset):
    from autolabel.coords import write_coordinate_files

    names = sorted(idx_names(dataset))
    write_label(dataset, names[0], [SAMPLE])
    write_coordinate_files(dataset)
    batch = create_batch(dataset, name="带坐标")
    attach_items(dataset, batch["id"], [names[0]])
    finish_batch(dataset, batch["id"], "done")

    result = delete_batches(dataset, [batch["id"]])

    assert os.path.isdir(result["coords"])
    with open(os.path.join(result["coords"], "coords.csv"), encoding="utf-8") as fh:
        assert len(fh.read().strip().splitlines()) == 1  # 人已删完，只剩表头


def test_batch_name_and_display_name():
    assert batch_name("video", ["a.mp4", "b.mp4"]) == "视频 2 个 · a.mp4 等"
    assert batch_name("image", ["x.jpg"]) == "图片 1 个 · x.jpg"
    assert batch_name("image", [], "用户上传 10 月") == "用户上传 10 月"
    assert display_name("1a2b3c4d_秘密 视频.mp4") == "秘密 视频.mp4"


def test_cli_import_registers_batch(dataset, tmp_path):
    source = tmp_path / "cli_src"
    source.mkdir()
    (source / "新图.jpg").write_bytes(jpeg_bytes(31))

    assert main(["import", dataset, str(source), "--batch-name", "命令行批次"]) == 0

    rows = list_batches(dataset)
    assert rows[0]["name"] == "命令行批次"
    assert rows[0]["source"] == "cli" and rows[0]["items"] == 1 and rows[0]["status"] == "done"
    ungrouped = {row["id"]: row["items"] for row in rows if row["virtual"]}
    assert ungrouped[POSE_UNGROUPED] == 2


def json_of(result):
    status, _headers, body = result
    return status, json.loads(body)


def test_batches_api_lists_and_deletes(server, dataset):
    base, _store = server
    status, payload = json_of(call(base, "/api/batches"))
    assert status == 200
    assert sorted(row["id"] for row in payload["batches"]) == [OBJECTS_UNGROUPED, POSE_UNGROUPED]
    by_id = {row["id"]: row for row in payload["batches"]}
    assert by_id[POSE_UNGROUPED]["items"] == 2 and by_id[POSE_UNGROUPED]["lib"] == "pose"
    assert by_id[OBJECTS_UNGROUPED]["items"] == 2 and by_id[OBJECTS_UNGROUPED]["lib"] == "objects"

    status, payload = json_of(call(base, "/api/dataset"))
    assert all(row["batch_id"] is None for row in payload["items"])

    status, payload = json_of(post_json(base, "/api/batches/delete", {"ids": ["ghost"]}))
    assert status == 400 and "批次不存在" in payload["error"]
    assert json_of(post_json(base, "/api/batches/delete"))[0] == 400

    # 不带库名的 __ungrouped__ = 两个库的未分组一起清
    status, payload = json_of(post_json(base, "/api/batches/delete", {"ids": [UNGROUPED_ID]}))
    assert status == 200, payload
    # items 按素材张数算（两个库引用同一批图），removals 才是"库×图"的次数
    assert payload["items"] == 2 and payload["removals"] == 4
    assert payload["files_removed"] >= 2

    assert json_of(call(base, "/api/dataset"))[1]["items"] == []
    assert json_of(call(base, "/api/dataset"))[1]["object_items"] == []
    assert json_of(call(base, "/api/batches"))[1]["batches"] == []


def test_upload_job_creates_batch(server, dataset):
    base, _store = server
    stored = json.loads(upload(base, "批次图片.jpg", jpeg_bytes(41))[2])["stored"]
    job_id = json.loads(post_json(base, "/api/jobs", {
        "kind": "image", "files": [stored], "params": {"label": False, "batch_name": "十月素材"}})[2])["id"]
    job = wait_job(base, job_id)

    assert job["ok"] is True, job["error"]
    assert job["batch_id"]
    rows = {row["id"]: row for row in list_batches(dataset)}
    row = rows[job["batch_id"]]
    assert row["name"] == "十月素材" and row["status"] == "done"
    assert row["source"] == "upload" and row["kind"] == "image"
    assert row["items"] == 1 and row["files"] == ["批次图片.jpg"]

    status, payload = json_of(call(base, "/api/dataset"))
    assert status == 200 and payload["items"][0]["batch_id"] == job["batch_id"]


def test_stale_running_batch_needs_force(dataset):
    """服务中断留下的 running 批次：命令行默认拒绝，--force 才放行。"""
    names = sorted(idx_names(dataset))
    batch = create_batch(dataset, name="服务中断的批次")
    attach_items(dataset, batch["id"], [names[0]])

    with pytest.raises(ValueError, match="仍在标注中"):
        delete_batches(dataset, [batch["id"]])
    assert batch["id"] in {row["id"] for row in list_batches(dataset)}

    result = delete_batches(dataset, [batch["id"]], force=True)

    assert result["items"] == 1
    assert batch["id"] not in {row["id"] for row in list_batches(dataset)}
    assert names[0] not in load_index(dataset).items


def test_api_deletes_stale_running_batch_without_force(server, dataset):
    """没有活跃任务指向它时，界面删除不该被历史 running 状态卡住。"""
    base, _store = server
    names = sorted(idx_names(dataset))
    batch = create_batch(dataset, name="重启后遗留的批次")
    attach_items(dataset, batch["id"], [names[0]])

    status, payload = json_of(post_json(base, "/api/batches/delete", {"ids": [batch["id"]]}))

    assert status == 200, payload
    assert payload["items"] == 1
    assert batch["id"] not in {row["id"] for row in json_of(call(base, "/api/batches"))[1]["batches"]}


# --------------------------------------------------------------------------- //
# 启动对账：导入中途被打断，素材不能悄悄丢掉
# --------------------------------------------------------------------------- //
def test_reconcile_recovers_orphans_and_closes_interrupted_batch(dataset):
    names = sorted(idx_names(dataset))
    orphan = "recovered_frame_9999.jpg"
    shutil.copy2(resolve_image(dataset, names[0]), os.path.join(images_dir(dataset), orphan))
    batch = create_batch(dataset, name="被中断的上传", kind="image")

    report = reconcile_with_disk(dataset)

    assert report["recovered"] == 1 and orphan in report["items"]
    item = load_index(dataset).get(orphan)
    assert item["batch_id"] == batch["id"]
    assert item["status"] == "unreviewed" and item["width"] > 0
    assert load_batches(dataset)[0]["status"] == "interrupted"

    again = reconcile_with_disk(dataset)
    assert again["recovered"] == 0 and again["interrupted"] == []


def test_reconcile_leaves_retained_files_out_of_index(dataset):
    names = sorted(idx_names(dataset))
    batch = create_batch(dataset, name="只删记录", kind="image")
    attach_items(dataset, batch["id"], list(names))
    finish_batch(dataset, batch["id"], "done")

    result = delete_batches(dataset, [batch["id"]], keep_files=True)

    assert result["items"] == len(names) and result["files_removed"] == 0
    assert os.path.exists(resolve_image(dataset, names[0]))
    assert reconcile_with_disk(dataset)["recovered"] == 0
    assert set(load_index(dataset).items) == set()


def test_reconcile_reports_index_items_whose_files_vanished(dataset):
    names = sorted(idx_names(dataset))
    os.remove(resolve_image(dataset, names[0]))

    report = reconcile_with_disk(dataset)

    assert report["missing_files"] == 1 and report["recovered"] == 0


# --------------------------------------------------------------------------- //
# 模块归属、单张清单与按张删除（门户批次管理）
# --------------------------------------------------------------------------- //
def _batch_with_two_images(root, name="双图批次", kind="image"):
    """建一个把两张图都挂上的已完成批次。"""
    names = sorted(idx_names(root))
    batch = create_batch(root, name=name, kind=kind)
    attach_items(root, batch["id"], list(names))
    finish_batch(root, batch["id"], "done")
    return batch, names


def test_batch_module_follows_kind_or_explicit_field(dataset):
    assert module_for({"kind": "image"}) == "pose"
    assert module_for({"kind": "video"}) == "pose"
    assert module_for({"kind": "prompt"}) == "objects"
    assert module_for({"kind": "prompt_upload"}) == "objects"
    assert module_for({"kind": "nonsense"}) == "unknown"
    assert module_for({"kind": "image", "module": "prompt"}) == "objects"
    assert module_for({"kind": "image", "module": "objects"}) == "objects"
    # 没显式给 lib 时按 kind 推导，两个库各归各的批次表
    assert create_batch(dataset, name="提示词上传", kind="prompt_upload")["module"] == "objects"
    assert create_batch(dataset, name="姿态上传", kind="video")["module"] == "pose"
    assert load_batches(dataset, "objects")[-1]["name"] == "提示词上传"
    assert load_batches(dataset, "pose")[-1]["name"] == "姿态上传"


def test_batch_rows_split_progress_per_module(dataset):
    """批次行的骨骼/检测框两列 = 该批次那几张素材的真实进度，与展开清单一致。"""
    batch, names = _batch_with_two_images(dataset)
    idx = load_index(dataset)
    idx.upsert(names[0], objects=1, status="approved")
    idx.upsert(names[1], objects=2, status="unreviewed")
    idx.save()
    oidx = objects_index(dataset)
    oidx.upsert(names[0], prompt_objects=2, prompt_status="approved", prompt_flags=["small_box"])
    oidx.upsert(names[1], prompt_objects=1, prompt_status="edited")
    oidx.save()
    prompt_batch = create_batch(dataset, name="提示词批次", kind="prompt", lib="objects")
    attach_items(dataset, prompt_batch["id"], names, lib="objects")
    finish_batch(dataset, prompt_batch["id"], "done")

    pose_side = {"labeled": 2, "people": 3, "reviewed": 1, "unreviewed": 1, "flagged": 0}
    prompt_side = {"labeled": 2, "boxes": 3, "reviewed": 2, "unreviewed": 0, "flagged": 1}

    row = next(r for r in list_batches(dataset) if r["id"] == batch["id"])
    assert row["module"] == "pose" and row["module_name"] == "姿态标注"
    assert row["lib"] == "pose" and row["items"] == 2
    assert row["pose"] == pose_side
    # 同一批素材也被提示词库引用了：骨骼行要把对面的检测框进度一起显示出来，
    # 否则会出现「展开清单写已标 2 框，批次行写 0 框」的自相矛盾
    assert row["prompt"] == prompt_side
    assert row["labeled"] == 2 and row["prompt_boxes"] == 3

    prow = next(r for r in list_batches(dataset) if r["id"] == prompt_batch["id"])
    assert prow["module"] == "objects" and prow["module_name"] == "提示词标注"
    assert prow["lib"] == "objects"
    assert prow["prompt"] == prompt_side
    assert prow["pose"] == pose_side

    # 两本账各记各的：删掉提示词库的标签，只有检测框那一列掉，骨骼列不动
    for rel in names:
        oidx = objects_index(dataset)
        oidx.upsert(rel, prompt_objects=None, prompt_status="unreviewed", prompt_flags=[])
        oidx.save()
    after = next(r for r in list_batches(dataset) if r["id"] == batch["id"])
    assert after["pose"] == pose_side
    assert after["prompt"] == {"labeled": 0, "boxes": 0, "reviewed": 0, "unreviewed": 2, "flagged": 0}

    # 批次行的小计必须等于展开文件清单的逐张汇总
    rows = batch_file_rows(dataset, batch["id"])
    assert sum(1 for r in rows if r["prompt_objects"]) == after["prompt"]["labeled"]
    assert sum(int(r["objects"] or 0) for r in rows) == after["pose"]["people"]


def test_batch_file_rows_report_both_modules(dataset):
    batch, names = _batch_with_two_images(dataset)
    write_label(dataset, names[0], [SAMPLE])
    idx = load_index(dataset)
    idx.upsert(names[0], objects=1, status="approved")
    idx.save()
    oidx = objects_index(dataset)
    oidx.upsert(names[0], prompt_objects=2, prompt_status="edited", prompt_flags=["dup_box"])
    oidx.save()

    rows = batch_file_rows(dataset, batch["id"])
    assert len(rows) == 2
    first = next(row for row in rows if row["image"] == names[0])
    assert first["name"] == os.path.basename(names[0])
    assert first["status"] == "approved" and first["objects"] == 1
    assert first["prompt_status"] == "edited" and first["prompt_flags"] == ["dup_box"]
    assert first["bytes"] > 0 and first["width"] > 0
    assert first["shared"] is True  # 提示词库也引用了这张图

    second = next(row for row in rows if row["image"] == names[1])
    assert second["status"] == "unreviewed" and second["objects"] is None
    assert second["prompt_status"] == "unreviewed" and second["prompt_objects"] is None
    assert batch_file_rows(dataset, POSE_UNGROUPED) == []


def test_delete_images_only_touches_picked_files(dataset):
    batch, names = _batch_with_two_images(dataset)
    write_label(dataset, names[0], [SAMPLE])
    stem = os.path.splitext(names[0])[0]
    prompt_label = os.path.join(dataset, "prompt_labels", stem + ".txt")
    os.makedirs(os.path.dirname(prompt_label), exist_ok=True)
    with open(prompt_label, "w", encoding="utf-8") as fh:
        fh.write("0 0.5 0.5 0.2 0.3\n")
    backup = os.path.join(dataset, ".autolabel", "objects", "originals", stem + ".txt")
    history = os.path.join(dataset, ".autolabel", "objects", "history", stem)
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    os.makedirs(history, exist_ok=True)
    for path in (backup, os.path.join(history, "1.txt")):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("0 0.4 0.4 0.2 0.2\n")

    result = delete_images(dataset, [names[0]], mode="purge")

    assert result["items"] == 1 and result["files_removed"] >= 3
    assert names[0] not in load_index(dataset).items
    assert names[1] in load_index(dataset).items
    assert not os.path.exists(prompt_label) and not os.path.exists(resolve_image(dataset, names[0]))
    # 提示词的原始备份与历史快照也要跟着清掉，否则删过的图留一堆孤儿文件
    assert not os.path.exists(backup) and not os.path.isdir(history)
    assert names[0] not in objects_index(dataset).items

    kept = next(b for b in load_batches(dataset) if b["id"] == batch["id"])
    assert kept["items"] == [names[1]]
    with open(result["snapshot"], encoding="utf-8") as fh:
        snap = json.load(fh)
    assert snap["mode"] == "purge"
    assert snap["items"][names[0]]["label_lines"] == [SAMPLE]


def test_delete_images_drops_emptied_batch_record(dataset):
    batch, names = _batch_with_two_images(dataset)

    delete_images(dataset, list(names), mode="purge")

    assert batch["id"] not in {b["id"] for b in load_batches(dataset)}
    assert list_batches(dataset) == []
    assert load_index(dataset).items == {}


def test_delete_images_keeps_files_when_asked(dataset):
    batch, names = _batch_with_two_images(dataset)

    result = delete_images(dataset, [names[0]], mode="purge", keep_files=True)

    assert result["files_removed"] == 0 and result["items"] == 1
    assert os.path.exists(resolve_image(dataset, names[0]))
    kept = next(b for b in load_batches(dataset) if b["id"] == batch["id"])
    assert kept["items"] == [names[1]]
    # 保留文件是刻意的，启动对账不该把它当成孤儿重新长回索引
    assert reconcile_with_disk(dataset)["recovered"] == 0


def test_delete_images_guardrails(dataset):
    batch, names = _batch_with_two_images(dataset)

    with pytest.raises(ValueError) as busy:
        delete_images(dataset, [names[0]], active_ids={batch["id"]})
    assert "正在标注中" in str(busy.value)

    with pytest.raises(ValueError) as unknown:
        delete_images(dataset, ["不存在的图.jpg"])
    assert "不在索引里" in str(unknown.value)

    with pytest.raises(ValueError):
        delete_images(dataset, [])


def test_delete_images_keeps_batch_when_registry_items_are_stale(dataset):
    """登记表 items 比索引少（导入中途被打断会这样）时，删一张不许带走整批。"""
    batch, names = _batch_with_two_images(dataset)
    records = load_batches(dataset)
    for record in records:
        if record["id"] == batch["id"]:
            record["items"] = [names[0]]  # 索引里两张都挂着，登记表只记下一张
    save_batches(dataset, records)

    delete_images(dataset, [names[0]])

    kept = next(b for b in load_batches(dataset) if b["id"] == batch["id"])
    assert kept["items"] == []
    row = next(r for r in list_batches(dataset) if r["id"] == batch["id"])
    assert row["items"] == 1  # 门户按索引显示，剩下那张还在，批次不能凭空消失


def test_delete_images_does_not_touch_unrelated_batches(dataset):
    """删一个批次的图，不该把别的批次记录（包括 0 张的空批次）顺手清掉。"""
    batch, names = _batch_with_two_images(dataset, name="有图批次")
    empty = create_batch(dataset, name="空批次", kind="image")
    finish_batch(dataset, empty["id"], "done", "没有新图入库")

    delete_images(dataset, [names[0]])

    ids = {b["id"] for b in load_batches(dataset)}
    assert batch["id"] in ids and empty["id"] in ids


def test_cli_batches_deletes_single_images(dataset, capsys):
    """命令行也能按张删：与门户「删这张」同一套口径、两种粒度与快照。"""
    batch, names = _batch_with_two_images(dataset)

    assert main(["batches", dataset, "--delete-images", names[0]]) == 0
    out = capsys.readouterr().out
    # 默认只摘姿态库引用：提示词库还引用着同一份素材，文件必须留着
    assert "移除图片 1 张（仅姿态标注）" in out and "素材文件保留" in out
    assert names[0] not in idx_names(dataset)
    assert names[0] in objects_index(dataset).items
    assert os.path.exists(resolve_image(dataset, names[0]))
    kept = next(b for b in load_batches(dataset) if b["id"] == batch["id"])
    assert kept["items"] == [names[1]]
    assert os.path.exists(resolve_image(dataset, names[1]))

    assert main(["batches", dataset, "--delete-images", names[1], "--lib", "objects",
                 "--mode", "purge"]) == 0
    out = capsys.readouterr().out
    assert "两个库都摘" in out and "清理文件 1 个" in out
    assert names[1] not in objects_index(dataset).items and names[1] not in idx_names(dataset)
    assert not os.path.exists(resolve_image(dataset, names[1]))
