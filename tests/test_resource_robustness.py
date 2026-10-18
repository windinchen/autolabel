"""资源与健壮性回归测试。

第 5 轮合理性检查用 1500 张的合成数据集实测，抓到四类"小数据集上完全看不出来"的问题：

1. 启动对账每扫到一张图就重新解析一遍对面库的 `index.json`，是 O(N²)——1500 张要 1 秒，
   两万张按同一条曲线要几分钟，用户看到的就是"点了启动没反应"；
2. `/api/dataset` 把同一份姿态清单在 `items` 与 `pose_items` 里各序列化一遍，
   `object_items` 还同时给 `prompt_*` 和一套同义旧键，1500 张时响应体 725 KB，其中三成纯属重复；
3. `.autolabel/deleted/` 的删除快照永远只增不减；
4. `/image/`、`/thumb/` 也进访问日志，翻一屏队列终端就被刷屏，真报错被冲走。
"""

import json
import os
import time

import pytest
from PIL import Image

from _helpers import idx_names
from autolabel import batches as bt
from autolabel import dataset as ds
from autolabel.library import deleted_dir
from test_prompt_web import get

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "autolabel", "static")


@pytest.fixture()
def interrupted_import(tmp_path):
    """模拟「导入到一半被强杀」：图已经落盘，姿态库索引里却没有它们。"""
    root = str(tmp_path / "crashed")
    ds.init_dataset(root, name="crashed")
    source = tmp_path / "many"
    source.mkdir()
    for index in range(40):
        Image.new("RGB", (48, 32), (index * 5 % 255, 60, 120)).save(source / f"s{index:03d}.jpg")
    ds.import_images(root, [str(source)], dedupe="none")
    assert len(idx_names(root)) == 40
    idx = ds.load_index(root)
    for rel in sorted(idx_names(root)):
        idx.remove(rel)
    idx.save()
    assert list(idx_names(root)) == [], "素材应留在磁盘上，只是索引里没有"
    return root


def test_reconcile_reads_the_other_index_once_per_library(interrupted_import, monkeypatch):
    """对账不能按张数重复解析对面库的索引：那是 O(N^2)。"""
    calls = []
    real = bt.load_index_or_none

    def counting(root, lib=None, *args, **kwargs):
        calls.append(lib)
        return real(root, lib, *args, **kwargs) if lib is not None else real(root, *args, **kwargs)

    monkeypatch.setattr(bt, "load_index_or_none", counting)
    report = bt.reconcile_with_disk(interrupted_import)
    assert report["recovered"] == 40
    # 每个库各一次 + 批次收尾时的一两次；旧实现这里会是 40 张 × 每张一次
    assert len(calls) <= 8, f"index.json 被解析了 {len(calls)} 次，退化成按张重读"
    assert len(idx_names(interrupted_import)) == 40


def test_reconcile_is_faster_than_a_full_index_rescan_per_image(interrupted_import):
    """40 张的合成库要在人感知不到的时间里对完，同时留足回归余量。"""
    started = time.time()
    bt.reconcile_with_disk(interrupted_import)
    assert time.time() - started < 2.0


def test_reconcile_still_leaves_images_owned_by_the_other_library(interrupted_import):
    """把对面库的索引提到循环外之后，"两边都没记才算孤儿"这条判断必须照旧。"""
    images = os.path.join(interrupted_import, "images")
    rels = sorted(name for name in os.listdir(images) if name.endswith(".jpg"))
    assert len(rels) == 40, "素材文件都得还在磁盘上"
    taken = rels[:10]
    objects = ds.load_index(interrupted_import, "objects")
    for rel in taken:
        objects.upsert(rel, width=48, height=32)
    objects.save()
    report = bt.reconcile_with_disk(interrupted_import)
    pose = ds.load_index(interrupted_import, "pose")
    assert report["recovered"] == 30
    for rel in taken:
        assert pose.get(rel) is None, "提示词库已经认领的图被姿态库抢走了"
    for rel in rels[10:]:
        assert pose.get(rel) is not None, "真正的孤儿图没被补回索引"
    assert ds.load_index(interrupted_import, "objects").get(taken[0]) is not None


def server_dataset(server):
    base, store = server
    _status, payload, _ = get(base, "/api/dataset")
    return payload


def test_dataset_payload_sends_the_pose_list_once(server):
    """`items` 与 `pose_items` 是同一份数据，序列化两遍等于白多发一半字节。"""
    payload = server_dataset(server)
    assert "pose_items" not in payload
    assert len(payload["items"]) == 2
    for key in ("image", "status", "objects", "score", "flags", "priority"):
        assert key in payload["items"][0]


def test_object_rows_only_ship_the_prompt_keys(server):
    """提示词库的行只带 `prompt_*`：界面上没有一处读那几个同义旧键。"""
    from autolabel import fields as field_store
    from test_prompt_web import post

    field_store.upsert_field(server[1].root, name="bottle", label="瓶子")
    image = sorted(idx_names(server[1].root))[0]
    post(server[0], "/api/prompt/item",
         {"image": image, "objects": [{"cls": 0, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}]})
    payload = server_dataset(server)
    row = next(r for r in payload["object_items"] if r["image"] == image)
    assert row["prompt_objects"] == 1
    assert row["prompt_status"] in ("unreviewed", "edited")
    for legacy in ("status", "objects", "score", "flags", "priority"):
        assert legacy not in row


def test_frontend_still_reads_the_pose_list_through_a_fallback():
    """后端删了 `pose_items`，界面上每个读它的地方都得留 `|| ds.items` 兜底。"""
    portal = open(os.path.join(STATIC, "portal.js"), encoding="utf-8").read()
    app = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert portal.count("ds.pose_items || ds.items") == 2
    assert app.count("ds.pose_items || ds.items") == 1
    assert "state.ds.items" in app


def test_deleted_snapshots_are_pruned_to_the_recent_window(dataset):
    """逐张删两万张不该攒出两万个文件：快照只留最近一窗。"""
    folder = deleted_dir(dataset)
    os.makedirs(folder, exist_ok=True)
    for index in range(260):
        with open(os.path.join(folder, f"2026{index:08d}.json"), "w", encoding="utf-8") as handle:
            json.dump({}, handle)
    first = sorted(idx_names(dataset))[0]
    result = bt.delete_images(dataset, [first], lib="pose", mode="purge", regenerate=False)
    left = sorted(os.listdir(folder))
    assert result["items"] == 1
    assert len(left) == bt.DELETE_SNAPSHOT_KEEP
    assert os.path.basename(result["snapshot"]) in left, "刚写的快照被自己清掉了"
    assert "202600000000.json" not in left, "最老的快照没清掉"
    assert "202600000259.json" in left, "最近一窗的快照被误删"


def test_thumbnail_requests_do_not_spam_the_terminal(server, capsys):
    """一屏队列要拉几十张缩略图，全打出来真报错就被冲走了。"""
    image = sorted(idx_names(server[1].root))[0]
    get(server[0], f"/thumb/{image}")
    get(server[0], f"/image/{image}")
    get(server[0], "/api/dataset")
    noise = capsys.readouterr().err
    assert "/thumb/" not in noise and "/image/" not in noise and "/api/" not in noise
    get(server[0], "/pose")
    assert "/pose" in capsys.readouterr().err, "页面访问日志也不打了，出问题时没线索"


def test_leftover_upload_files_are_swept_at_startup(dataset):
    """进程中途被打断时，`uploads/` 里的原文件不能永远占着磁盘（一次视频就几百 MB）。"""
    from autolabel import upload as up

    folder = up.uploads_dir(dataset)
    os.makedirs(folder, exist_ok=True)
    stale = os.path.join(folder, "yesterday.mp4")
    fresh = os.path.join(folder, "just-now.jpg")
    for path in (stale, fresh):
        with open(path, "wb") as handle:
            handle.write(b"x" * 4096)
    long_ago = time.time() - 3 * 24 * 3600
    os.utime(stale, (long_ago, long_ago))
    assert up.prune_stale_uploads(dataset) == {"files": 1, "bytes": 4096}
    assert not os.path.exists(stale), "上次遗留的上传文件没清掉"
    assert os.path.exists(fresh), "另一个还在跑的服务刚收下的文件被误删"
    assert up.prune_stale_uploads(dataset)["files"] == 0
    review_py = open(os.path.join(os.path.dirname(STATIC), "review.py"), encoding="utf-8").read()
    assert "prune_stale_uploads(" in review_py, "启动流程没接上传清理"
