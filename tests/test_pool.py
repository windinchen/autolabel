"""素材池：素材归池、标注归库。

这一页新加的口径全在这里钉住：只进池不入库的素材必须记账（否则下次导入认不出重复、
启动对账会把它抢进姿态库）；分类是素材层的属性，三个模块看到同一份；
删素材看的是「还有哪个库在用」，没入库的就是真删。
"""

import json
import os

import numpy as np
import pytest
from PIL import Image
from test_upload import call, jpeg_bytes, upload, wait_job

from autolabel import pool as pool_catalog
from autolabel.batches import delete_images, reconcile_with_disk
from autolabel.dataset import (
    adopt_from_pool,
    images_dir,
    import_images,
    libs_holding_image,
    load_index,
    pool_items,
)

JSON = {"Content-Type": "application/json"}


@pytest.fixture()
def fresh_dir(tmp_path):
    """和 fixture 素材毫不相干的两张新图：专供「只进池」的导入用。"""
    target = tmp_path / "fresh"
    target.mkdir()
    for i in range(2):
        arr = np.random.default_rng(1000 + i).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(arr).save(target / f"fresh{i}.jpg")
    return str(target)


def index_count(root, lib):
    return len(load_index(root, lib).items)


def ledger(root):
    return pool_catalog.load_catalog(root)["items"]


def pool_rows(root):
    return {row["image"]: row for row in pool_items(root)}


def import_only(root, source, **kwargs):
    stats = import_images(root, [source], dedupe="exact", register=False, **kwargs)
    assert stats["register"] is False
    return sorted(rel for _src, rel in stats["imported_files"]), stats


# --------------------------------------------------------------------------- //
# 只进池：文件落盘 + 记账，三个库一张都不加
# --------------------------------------------------------------------------- //
def test_import_register_false_lands_in_pool_only(dataset, fresh_dir):
    """``register=False`` 只落文件 + 记账：三个库的索引不动，但来处、尺寸、哈希都在账上。"""
    before = {lib: index_count(dataset, lib) for lib in ("pose", "objects", "detect")}
    imported, _stats = import_only(dataset, fresh_dir)
    assert imported, "没导进任何东西，这条测试等于没跑"
    for rel in imported:
        assert os.path.exists(os.path.join(images_dir(dataset), rel)), "素材文件没落盘"
        assert not libs_holding_image(dataset, rel), "只进池的素材不该被任何库引用"
        item = ledger(dataset)[rel]
        assert item.get("sha256"), "没记内容哈希：下次导入认不出这是同一帧"
        assert item.get("bytes") and item.get("width"), "没记体积与尺寸：素材池页只能说个名字"
    assert {lib: index_count(dataset, lib) for lib in before} == before, "只进池的导入掺进了库"
    assert set(pool_catalog.pool_only_rels(dataset)) == set(imported)


def test_pool_only_files_survive_startup_reconcile(dataset, fresh_dir):
    """启动对账不许把「只进池」的图当孤儿图抢进姿态库——那等于替用户做了他没做的决定。"""
    imported, _stats = import_only(dataset, fresh_dir)
    before = index_count(dataset, "pose")
    result = reconcile_with_disk(dataset)
    assert result["recovered"] == 0, f"对账抢走了只进池的素材：{result}"
    assert index_count(dataset, "pose") == before
    assert set(pool_catalog.pool_only_rels(dataset)) >= set(imported)


def test_importing_the_same_frame_twice_does_not_duplicate(dataset, fresh_dir):
    """先去重再记账：同一帧第二次导入既不复制文件，也不记第二笔账。"""
    imported, _first = import_only(dataset, fresh_dir)
    files_before = len(os.listdir(images_dir(dataset)))
    _again, stats = import_only(dataset, fresh_dir)
    assert stats["imported"] == 0 and stats["skipped_exact"] == len(imported), stats
    assert len(os.listdir(images_dir(dataset))) == files_before, "同一帧被复制成了两份"
    assert set(ledger(dataset)) >= set(imported)


def test_adopt_pool_only_into_detect(dataset, fresh_dir):
    """从池里挑进检测库：文件不复制，来处批次跟着走，账上不再认为它「没入库」。"""
    imported, _stats = import_only(dataset, fresh_dir)
    batch = pool_catalog.create_batch(dataset, name="进池一批", uploads=["fresh"])
    assert pool_catalog.attach_items(dataset, batch["id"], imported) == len(imported)
    pool_catalog.finish_batch(dataset, batch["id"], "done", "进池 2 张")

    rows = pool_rows(dataset)
    for rel in imported:
        assert rows[rel]["pool_only"] is True
        assert rows[rel]["pool_batch_name"] == "进池一批", "素材池页会说不出这批是哪来的"
        assert rows[rel]["pool_batch"] == batch["id"]
        assert rows[rel]["batch_id"] == "", "库批次键串进来了：这素材还没进任何库"

    stats = adopt_from_pool(dataset, imported, lib="detect")
    assert stats["added"] == len(imported), stats
    assert set(stats["added_items"]) == set(imported), stats
    for rel in imported:
        assert libs_holding_image(dataset, rel) == ["detect"]
        assert os.path.exists(os.path.join(images_dir(dataset), rel)), "入库不该复制文件"
    assert index_count(dataset, "detect") == len(imported)
    assert index_count(dataset, "pose") == 2, "别的库不该被连带掺水"
    assert not (set(pool_catalog.pool_only_rels(dataset)) & set(imported))


def test_delete_pool_only_removes_file_and_ledger(dataset, fresh_dir):
    """没入库的素材删除就是真删：文件、账本条目一起清，批次里的引用也不留。"""
    imported, _stats = import_only(dataset, fresh_dir)
    batch = pool_catalog.create_batch(dataset, name="要删的一批")
    pool_catalog.attach_items(dataset, batch["id"], imported)

    result = delete_images(dataset, imported, lib="detect", mode="purge")
    assert result["pool_only"] == len(imported), result
    assert result["files_removed"] >= len(imported), result
    for rel in imported:
        assert not os.path.exists(os.path.join(images_dir(dataset), rel)), "文件还躺在 images/"
        assert rel not in ledger(dataset), "账本留了孤账：下次导入会认为它还在"
    assert batch["id"] not in {row["id"] for row in pool_catalog.batches(dataset)}
    assert set(pool_rows(dataset)) == {"img0.jpg", "img1.jpg"}


def test_forget_batch_keeps_files(dataset, fresh_dir):
    """删批次只是忘掉来处：素材本身还在池里，一张都不该被连带删掉。"""
    imported, _stats = import_only(dataset, fresh_dir)
    batch = pool_catalog.create_batch(dataset, name="忘掉我")
    pool_catalog.attach_items(dataset, batch["id"], imported)

    result = pool_catalog.forget_batch(dataset, batch["id"])
    assert result["ok"] and result["detached"] == len(imported)
    assert batch["id"] not in {row["id"] for row in pool_catalog.batches(dataset)}
    for rel in imported:
        assert os.path.exists(os.path.join(images_dir(dataset), rel)), "删批次不该删素材"
        assert str(ledger(dataset)[rel].get("batch") or "") != batch["id"], \
            "来处还指向已经不存在的批次：按这批筛永远筛不到"


# --------------------------------------------------------------------------- //
# 分类：素材层的属性，只存一份
# --------------------------------------------------------------------------- //
def test_category_crud(dataset):
    """新建 / 改名 / 重名拦截 / 空名拦截 / 删分类只把素材退回未分类。"""
    created = pool_catalog.save_category(dataset, "户外", note="光线杂", icon="ic-scene")
    row = created["category"]
    assert row["name"] == "户外" and row["icon"] == "ic-scene"
    with pytest.raises(ValueError):
        pool_catalog.save_category(dataset, "户外")
    with pytest.raises(ValueError):
        pool_catalog.save_category(dataset, "   ")
    with pytest.raises(ValueError):
        pool_catalog.save_category(dataset, "不存在", category_id="c_nope")

    renamed = pool_catalog.save_category(dataset, "室内", category_id=row["id"])
    assert renamed["category"]["name"] == "室内"
    assert [item["id"] for item in pool_catalog.categories(dataset)] == [row["id"]]
    assert pool_catalog.category_map(dataset) == {row["id"]: "室内"}

    deleted = pool_catalog.delete_category(dataset, row["id"])
    assert deleted["ok"] and deleted["uncategorized"] == 0
    assert pool_catalog.categories(dataset) == []
    with pytest.raises(ValueError):
        pool_catalog.delete_category(dataset, row["id"])


def test_category_icon_falls_back_to_the_one_we_have(server):
    """分类图标只准从 sprite 里那几枚挑：乱传会让分类条上出现一个空洞。"""
    status, _headers, body = call(server[0], "/api/pool/category",
                                 data=json.dumps({"name": "穿T恤", "icon": "ic-ghost"}).encode(),
                                 headers=JSON)
    assert status == 200, body
    assert json.loads(body)["category"]["icon"] == "ic-tag", "非法图标要回落到默认那枚"


def test_assign_category_is_shared_by_three_libraries(dataset, fresh_dir):
    """在池里分了类，三个模块看到的是同一份；取消分类不能把池独占的记账一起清掉。"""
    imported, _stats = import_only(dataset, fresh_dir)
    cat = pool_catalog.save_category(dataset, "待复查")["category"]

    assigned = pool_catalog.assign(dataset, imported, cat["id"])
    assert assigned["set"] == len(imported)
    rows = pool_rows(dataset)
    for rel in imported:
        assert rows[rel]["category"] == cat["id"]
        assert rows[rel]["category_name"] == "待复查"

    cleared = pool_catalog.assign(dataset, imported, "")
    assert cleared["cleared"] == len(imported)
    rows = pool_rows(dataset)
    for rel in imported:
        assert rel in ledger(dataset), "只进池的素材账本被清空了：对账会把它抢进姿态库"
        assert rows[rel]["category"] == "" and rows[rel]["category_name"] == "未分类"

    with pytest.raises(ValueError):
        pool_catalog.assign(dataset, imported, "c_nope")


def test_uncategorizing_a_library_image_drops_the_ledger_row(dataset):
    """库内素材的账本只补分类这一个字段：取消分类后整条该删掉，不留空壳。"""
    cat = pool_catalog.save_category(dataset, "顺手一笔")["category"]
    pool_catalog.assign(dataset, ["img0.jpg"], cat["id"])
    assert ledger(dataset)["img0.jpg"] == {"category": cat["id"]} or \
        set(ledger(dataset)["img0.jpg"]) <= {"category", "assigned_at"}
    pool_catalog.assign(dataset, ["img0.jpg"], "")
    assert "img0.jpg" not in ledger(dataset)


def test_category_summary_counts_every_frame_once(dataset, fresh_dir):
    """分类汇总的张数必须和素材数对得上：凑出来的数会比不报更坑。"""
    imported, _stats = import_only(dataset, fresh_dir)
    cat = pool_catalog.save_category(dataset, "有框的")["category"]
    pool_catalog.assign(dataset, imported[:1], cat["id"])
    rows = pool_rows(dataset).values()
    summary = {row["id"]: row["files"] for row in pool_catalog.category_summary(dataset, list(rows))}
    assert sum(summary.values()) == len(pool_rows(dataset))
    assert summary[cat["id"]] == 1


# --------------------------------------------------------------------------- //
# 页面与接口：素材池是一等页面，不再靠各模块顺带看一眼
# --------------------------------------------------------------------------- //
def test_pool_page_and_script_are_served(server):
    base = server[0]
    status, _headers, body = call(base, "/pool")
    assert status == 200 and b'id="pool-grid"' in body
    status, _headers, body = call(base, "/materials")
    assert status == 200 and b'id="pool-grid"' in body, "老地址得落到新页"
    status, _headers, body = call(base, "/pool.js")
    assert status == 200 and b"function poolCard" in body


def test_pool_api_view_all_carries_dataset_categories_and_batches(server, dataset, fresh_dir):
    base = server[0]
    imported, _stats = import_only(dataset, fresh_dir)
    cat = pool_catalog.save_category(dataset, "户外")["category"]
    pool_catalog.assign(dataset, imported, cat["id"])
    batch = pool_catalog.create_batch(dataset, name="对账单")
    pool_catalog.attach_items(dataset, batch["id"], imported)

    status, _headers, body = call(base, "/api/pool?view=all")
    assert status == 200, body
    data = json.loads(body)
    assert data["dataset"]["name"], "顶栏「自动标注 · 数据集名」会空一截"
    assert data["view"] == "all" and len(data["all"]) == len(pool_rows(dataset))
    assert {row["id"]: row["files"] for row in data["categories"]}[cat["id"]] == len(imported), \
        "分类条上的张数是凑的"
    assert any(row["id"] == batch["id"] for row in data["batches"]), "来处那列会一片空白"
    assert data["stats"]["files"] == len(pool_rows(dataset))
    assert data["stats"]["orphan"] >= len(imported)

    # 各模块的「从素材池添加」只要本库还没有的，且不给全量清单
    status, _headers, body = call(base, "/api/pool?lib=detect")
    picker = json.loads(body)
    assert picker["view"] == "picker" and picker["all"] == []
    # detect 库这时还空着，池里四张对它来说都是新的（含只进了姿态库的那两张）
    assert {row["image"] for row in picker["files"]} == set(pool_rows(dataset))

    # 挑进 detect 之后再来要清单，这几张必须消失，否则同一张图能被反复添加
    status, _headers, body = call(base, "/api/pool/adopt",
                                 data=json.dumps({"images": imported, "lib": "detect"}).encode(),
                                 headers=JSON)
    assert status == 200, body
    status, _headers, body = call(base, "/api/pool?lib=detect")
    picker = json.loads(body)
    assert {row["image"] for row in picker["files"]} == set(pool_rows(dataset)) - set(imported), \
        "已经在本库的素材还留在挑选列表里"


def test_pool_category_and_assign_endpoints(server, dataset):
    base = server[0]
    status, _headers, body = call(base, "/api/pool/category",
                                 data=json.dumps({"name": "球场", "note": "逆光多"}).encode(),
                                 headers=JSON)
    assert status == 200, body
    cat = json.loads(body)["category"]
    assert cat["note"] == "逆光多"

    rel = sorted(pool_rows(dataset))[0]
    status, _headers, body = call(base, "/api/pool/assign",
                                 data=json.dumps({"images": [rel], "category": cat["id"]}).encode(),
                                 headers=JSON)
    assert status == 200 and json.loads(body)["set"] == 1, body
    assert ledger(dataset)[rel]["category"] == cat["id"]

    status, _headers, body = call(base, "/api/pool/assign", data=json.dumps({"images": []}).encode(),
                                 headers=JSON)
    assert status == 400 and "没有选择" in body.decode("utf-8"), "空选择要挨一句人话"

    status, _headers, body = call(base, "/api/pool/category/delete",
                                 data=json.dumps({"id": cat["id"]}).encode(), headers=JSON)
    assert status == 200 and json.loads(body)["uncategorized"] == 1, body
    assert "category" not in ledger(dataset).get(rel, {}), "删分类得把素材退回未分类"


def test_pool_upload_job_pools_without_touching_any_library(server, dataset):
    """素材池页的导入任务：文件进池就完事，不进库、不标注；删除走的也是池的口径。"""
    base = server[0]
    status, _headers, body = upload(base, "poolcase.jpg", jpeg_bytes(4242))
    assert status == 200, body
    stored = json.loads(body)["stored"]
    before = {lib: index_count(dataset, lib) for lib in ("pose", "objects", "detect")}

    status, _headers, body = call(base, "/api/jobs",
                                 data=json.dumps({"kind": "pool", "files": [stored],
                                                  "params": {"batch_name": "试池子"}}).encode(),
                                 headers=JSON)
    assert status in (200, 202), body
    job = wait_job(base, json.loads(body)["id"])
    assert job["ok"], job
    assert {lib: index_count(dataset, lib) for lib in before} == before, "只进池的任务掺进了库"

    fresh = sorted(set(pool_rows(dataset)) - {"img0.jpg", "img1.jpg"})
    assert fresh, "导入的图没出现在池清单里"
    row = pool_rows(dataset)[fresh[0]]
    assert row["pool_only"] is True and row["pool_batch_name"] == "试池子"

    status, _headers, body = call(base, "/api/images/delete",
                                 data=json.dumps({"images": fresh, "mode": "purge"}).encode(),
                                 headers=JSON)
    assert status == 200, body
    assert json.loads(body)["pool_only"] == len(fresh)
    assert not (set(pool_rows(dataset)) & set(fresh))
