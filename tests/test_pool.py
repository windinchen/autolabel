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
from autolabel.batches import (
    attach_items,
    create_batch,
    delete_batches,
    delete_images,
    finish_batch,
    list_batches,
    list_snapshots,
    reconcile_with_disk,
    restore_deletion,
)
from autolabel.dataset import (
    adopt_from_pool,
    images_dir,
    import_images,
    label_path_for,
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


# --------------------------------------------------------------------------- //
# 导入记录：池批次 + 各模块导入的批次，一屏看全
# --------------------------------------------------------------------------- //
def test_pool_batch_view_lists_module_import_batches(server, dataset, fresh_dir):
    """从模块「从素材池添加」出来的批次也得进批次视图，且不与池批次撞名。

    以前这里只列池账本的批次，于是素材池页一边写着「86 个素材」，
    批次视图一边说「还没有导入记录」——来处明明记在各库的登记表里，
    却没地方看，也没法按批退回去。
    """
    base = server[0]
    imported, _stats = import_only(dataset, fresh_dir)
    status, _headers, body = call(base, "/api/pool/adopt",
                                 data=json.dumps({"images": imported, "lib": "detect"}).encode(),
                                 headers=JSON)
    assert status == 200, body
    adopted = json.loads(body)["batch_id"]

    data = json.loads(call(base, "/api/pool?view=all")[2])
    rows = {row["id"]: row for row in data["batches"]}
    assert adopted in rows, "各模块导入的批次在批次视图里查无此批"
    row = rows[adopted]
    assert row["module_name"] == "目标检测标注", row
    assert not str(row["name"]).startswith("素材池"), \
        "和池批次撞名，两条并排时分不清哪条是池导入、哪条是模块添加"
    # items 要给数量：给清单的话界面上会印成「a.jpg,b.jpg 个素材」
    assert row["items"] == len(imported) and isinstance(row["items"], int), row
    assert row["in_pool"] == len(imported), row
    for item in data["all"]:
        if item["image"] in set(imported):
            assert item["pool_batch_name"], "来处那列空着"


def test_batch_files_endpoint_answers_for_pool_batches(server, dataset, fresh_dir):
    """点开一个只进了池的批次也要有清单：那批素材本来就不在任何库索引里。"""
    base = server[0]
    imported, _stats = import_only(dataset, fresh_dir)
    batch = pool_catalog.create_batch(dataset, name="只在池里")
    pool_catalog.attach_items(dataset, batch["id"], imported)

    status, _headers, body = call(base, "/api/batch/files?id=" + batch["id"])
    assert status == 200, body
    files = json.loads(body)["files"]
    assert {row["name"] for row in files} == {os.path.basename(rel) for rel in imported}, files
    assert all(row["category_name"] for row in files), "清单里连分类都没有，看了等于没看"


def test_retiring_a_pool_batch_keeps_the_provenance(dataset, fresh_dir):
    """「只摘引用」之后素材还在池里，批次行和来处都得留着，只把状态改成已退回。"""
    imported, _stats = import_only(dataset, fresh_dir)
    batch = pool_catalog.create_batch(dataset, name="退回但留着来处")
    pool_catalog.attach_items(dataset, batch["id"], imported)
    pool_catalog.finish_batch(dataset, batch["id"])

    result = delete_batches(dataset, [batch["id"]], keep_files=True)
    assert result["files_removed"] == 0, result
    rows = {row["id"]: row for row in pool_catalog.batches(dataset)}
    assert batch["id"] in rows, "素材还在池里就把批次行删了：这批素材从此变成「没有批次」"
    assert rows[batch["id"]]["status"] == pool_catalog.STATUS_RETIRED, rows[batch["id"]]
    for rel in imported:
        assert os.path.exists(os.path.join(images_dir(dataset), rel)), "只摘引用不该删文件"
        assert str(ledger(dataset)[rel].get("batch")) == batch["id"], "来处被抹掉了"


def test_purging_a_pool_batch_takes_the_row_and_the_files(dataset, fresh_dir):
    """勾了「连原图文件一起删」才是真删：文件、账本、批次行一次清干净。"""
    imported, _stats = import_only(dataset, fresh_dir)
    batch = pool_catalog.create_batch(dataset, name="连图一起走")
    pool_catalog.attach_items(dataset, batch["id"], imported)
    pool_catalog.finish_batch(dataset, batch["id"])

    result = delete_batches(dataset, [batch["id"]], keep_files=False)
    assert result["files_removed"] >= len(imported), result
    assert batch["id"] not in {row["id"] for row in pool_catalog.batches(dataset)}
    for rel in imported:
        assert not os.path.exists(os.path.join(images_dir(dataset), rel))
        assert rel not in ledger(dataset)


def test_import_can_categorize_on_the_way_in(server, dataset):
    """导入弹层里选的分类要在落池那一刻打上，不用事后再去框一遍。"""
    base = server[0]
    cat = pool_catalog.save_category(dataset, "夜场")["category"]
    status, _headers, body = upload(base, "night.jpg", jpeg_bytes(777))
    assert status == 200, body
    stored = json.loads(body)["stored"]

    status, _headers, body = call(base, "/api/jobs",
                                 data=json.dumps({"kind": "pool", "files": [stored],
                                                  "params": {"batch_name": "夜拍",
                                                             "category": cat["id"]}}).encode(),
                                 headers=JSON)
    assert status in (200, 202), body
    job = wait_job(base, json.loads(body)["id"])
    assert job["ok"], job

    fresh = sorted(set(ledger(dataset)) - {"img0.jpg", "img1.jpg"})
    assert fresh, "导入的素材没进池账本"
    assert all(str(ledger(dataset)[rel].get("category")) == cat["id"] for rel in fresh), \
        "导入时选的分类没生效"


def test_delete_reports_batch_ids_not_dicts(dataset):
    """删完回给前端的批次清单必须是 ID：整串 dict 印在提示条上没人看得懂。"""
    rel = sorted(load_index(dataset, "pose").items)[0]
    batch = create_batch(dataset, name="要清掉的", kind="image", lib="pose")
    attach_items(dataset, batch["id"], [rel], lib="pose")

    result = delete_images(dataset, [rel], lib="pose", mode="lib")
    assert batch["id"] in result["batches_updated"], result
    for item in result["batches_updated"]:
        assert "{" not in item and "'" not in item, result["batches_updated"]


# --------------------------------------------------------------------------- //
# 批次记录：导入的、标注的一屏看全，删错了还能原样捞回来
# --------------------------------------------------------------------------- //
def test_annotation_batches_are_listed_too(server, dataset):
    """跑过的那轮批量标注要在批次记录里找得到。

    只列导入批次时，用户跑完「批量姿态标注 · 343 人」回到素材池页翻不到这批，
    只能理解为「系统没记下我干的活」。现在一次列全，用 action 分清导入还是标注。
    """
    base = server[0]
    rel = sorted(load_index(dataset, "pose").items)[0]
    batch = create_batch(dataset, name="批量姿态标注 2 个", kind="pose", lib="pose")
    attach_items(dataset, batch["id"], [rel], lib="pose")
    finish_batch(dataset, batch["id"], "done", "完成：标注 1 张 · 8 个人")

    rows = {row["id"]: row for row in json.loads(call(base, "/api/pool?view=all")[2])["batches"]}
    assert batch["id"] in rows, "标注批次在批次记录里查无此批"
    row = rows[batch["id"]]
    assert row["action"] == "label" and row["action_name"] == "标注", row
    assert int(row["items"]) >= 1, "标注批次一个素材都没挂上，界面上就是一条空批次"


def test_module_import_stamps_the_provenance(server, dataset):
    """从标注模块导进来的素材也得在池账本上盖来处章。

    以前只有素材池页自己的导入会盖章，模块导入的素材在池账本上是「没有批次」，
    于是素材池页按批次筛筛不到、按批次删也删不到——用户报的正是这一条。
    """
    base = server[0]
    status, _headers, body = upload(base, "mod.jpg", jpeg_bytes(9001))
    assert status == 200, body
    stored = json.loads(body)["stored"]
    status, _headers, body = call(base, "/api/jobs",
                                 data=json.dumps({"kind": "image", "files": [stored],
                                                  "params": {"label": False}}).encode(),
                                 headers=JSON)
    assert status in (200, 202), body
    job = wait_job(base, json.loads(body)["id"])
    assert job["ok"], job

    fresh = sorted(set(ledger(dataset)) - {"img0.jpg", "img1.jpg"})
    assert fresh, "模块导入的素材没进池账本"
    stamps = {str(ledger(dataset)[rel].get("batch") or "") for rel in fresh}
    assert stamps and "" not in stamps, "模块导入没盖来处章：素材池页按批次筛筛不到"
    rows = {row["id"]: row for row in pool_catalog.import_records(dataset)}
    for stamp in stamps:
        assert stamp in rows, f"来处指向 {stamp}，批次记录里却没这一批"
        assert rows[stamp]["action"] == "import", rows[stamp]


def test_provenance_sticks_to_the_first_import(dataset, fresh_dir):
    """同一帧被第二个模块再导一次，不该把「它从哪来」改成后面那次。"""
    imported, _stats = import_only(dataset, fresh_dir)
    first = pool_catalog.create_batch(dataset, name="第一次上传")
    second = pool_catalog.create_batch(dataset, name="第二个模块又挑了一次")
    pool_catalog.attach_items(dataset, first["id"], imported)
    pool_catalog.attach_items(dataset, second["id"], imported)
    for rel in imported:
        assert str(ledger(dataset)[rel].get("batch")) == first["id"], "来处被后面那次改挂了"


def test_retiring_a_library_import_batch_keeps_row_and_stamp(dataset):
    """模块导入的批次「只摘引用」也得不留行、留来处，和池批次同一套规矩。"""
    rel = sorted(load_index(dataset, "pose").items)[0]
    batch = create_batch(dataset, name="模块导入的一批", kind="image", lib="pose")
    attach_items(dataset, batch["id"], [rel], lib="pose")
    pool_catalog.attach_items(dataset, batch["id"], [rel])
    finish_batch(dataset, batch["id"], "done", "入库 1 张")

    delete_batches(dataset, [batch["id"]], keep_files=True)
    rows = {row["id"]: row for row in pool_catalog.import_records(dataset)}
    assert batch["id"] in rows, "只摘引用就把批次行删了：这批素材从此没有来处"
    assert rows[batch["id"]]["status"] == pool_catalog.STATUS_RETIRED, rows[batch["id"]]
    assert str(ledger(dataset)[rel].get("batch")) == batch["id"], "素材还在池里，来处不该抹"


def test_purging_a_library_import_batch_clears_the_stamp(dataset):
    """连原图一起删才是真删：批次行、索引、来处章一次清干净，不留暗账。"""
    rel = sorted(load_index(dataset, "pose").items)[0]
    batch = create_batch(dataset, name="整批清掉", kind="image", lib="pose")
    attach_items(dataset, batch["id"], [rel], lib="pose")
    pool_catalog.attach_items(dataset, batch["id"], [rel])
    finish_batch(dataset, batch["id"], "done", "入库 1 张")

    delete_batches(dataset, [batch["id"]], keep_files=False)
    assert batch["id"] not in {row["id"] for row in list_batches(dataset)}
    assert batch["id"] not in {row["id"] for row in pool_catalog.import_records(dataset)}
    assert not str(ledger(dataset).get(rel, {}).get("batch") or ""), "素材都没了还留着来处"


def test_restore_brings_back_what_a_batch_delete_took(dataset):
    """删掉的批次要能原样装回来：标签原文、索引条目、批次记录，一样都不靠重跑。"""
    rel = sorted(load_index(dataset, "pose").items)[0]
    label = label_path_for(dataset, rel, "pose")
    with open(label, "w", encoding="utf-8") as handle:
        handle.write("0 0.5 0.5 0.2 0.3 0.9 0.9 0.02\n" * 3)
    batch = create_batch(dataset, name="删错的那批", kind="image", lib="pose")
    attach_items(dataset, batch["id"], [rel], lib="pose")
    finish_batch(dataset, batch["id"], "done", "入库 1 张")

    delete_batches(dataset, [batch["id"]], keep_files=True)
    assert rel not in load_index(dataset, "pose").items
    assert not os.path.exists(label), "前提：删除确实把标签文件摘掉了"

    snapshot = list_snapshots(dataset, limit=1)[0]
    assert snapshot["restorable"] == 1, snapshot

    back = restore_deletion(dataset)
    assert back["items"] == 1 and back["labels"] == 1, back
    assert back["batches"] >= 1, back
    assert rel in load_index(dataset, "pose").items, "图片没装回索引"
    assert load_index(dataset, "pose").items[rel].get("batch_id") == batch["id"]
    assert os.path.exists(label), "标签原文没写回来"
    assert snapshot["name"] not in {row["name"] for row in list_snapshots(dataset)}, \
        "捞过的快照要吃掉，不然按钮点第二次会重复装回"


def test_restore_endpoint_says_so_when_nothing_is_left(server, dataset):
    """没有快照时「捞回上次删除」得给一句人话，而不是一串裸异常。"""
    base = server[0]
    status, _headers, body = call(base, "/api/batches/restore", data=json.dumps({}).encode(),
                                 headers=JSON)
    assert status == 400, body
    assert "快照" in body.decode("utf-8"), body


# --------------------------------------------------------------------------- //
# 空账：账上有图、盘上没图，会把重新导入的同一批图挡在门外
# --------------------------------------------------------------------------- //
def test_purging_the_last_holder_clears_the_ledger_row(dataset, fresh_dir):
    """连原图一起删：池账本的条目得跟着走，否则这批图的 sha256 变成空账。

    用户实测踩到的：一批图删掉再重新上传，传 255 张只落 169 张——删的时候只抹了
    「来处章」，账上的 sha256 还留着，去重一撞就当成"池里已经有了"，文件根本没落盘。
    """
    imported, _stats = import_only(dataset, fresh_dir)
    adopt_from_pool(dataset, imported, lib="detect")
    batch = create_batch(dataset, name="整批清掉再重导", kind="detect_upload", lib="detect")
    attach_items(dataset, batch["id"], imported, lib="detect")
    pool_catalog.attach_items(dataset, batch["id"], imported)
    finish_batch(dataset, batch["id"], "done", "入库 2 张")
    assert set(imported) <= set(ledger(dataset)), "前提：素材账本里确实记着这批图"

    delete_batches(dataset, [batch["id"]], keep_files=False)

    for rel in imported:
        assert not os.path.exists(os.path.join(images_dir(dataset), rel)), rel
        assert rel not in ledger(dataset), f"{rel} 的文件删了，素材账本还记着一条空账"

    again = import_images(dataset, [fresh_dir], dedupe="exact", register=False)
    assert again["skipped_exact"] == 0, "重导的同一批图被空账当成重复跳过了"
    assert again["imported"] == len(imported), again


def test_import_ignores_a_hash_whose_file_is_gone(dataset, fresh_dir):
    """去重要认图不认账：手工删过 images/ 之后，同一批图还能重新进池。"""
    imported, _stats = import_only(dataset, fresh_dir)
    for rel in imported:
        os.remove(os.path.join(images_dir(dataset), rel))

    again = import_images(dataset, [fresh_dir], dedupe="exact", register=False)
    assert again["imported"] == len(imported) and again["skipped_exact"] == 0, again
    for _src, rel in again["imported_files"]:
        assert os.path.exists(os.path.join(images_dir(dataset), rel)), rel


def test_startup_reconcile_clears_ledger_rows_for_missing_files(dataset, fresh_dir):
    """启动对账要把空账清干净：素材池页的张数和磁盘说的是同一句话。"""
    imported, _stats = import_only(dataset, fresh_dir)
    for rel in imported:
        os.remove(os.path.join(images_dir(dataset), rel))

    report = reconcile_with_disk(dataset)
    assert report["ledger_pruned"] == len(imported), report
    assert not (set(imported) & set(ledger(dataset))), "空账还在，重导还会被挡"
    assert reconcile_with_disk(dataset)["ledger_pruned"] == 0, "对账要可重入"


def test_pool_job_reports_which_files_were_dropped(server, dataset):
    """任务完成消息要讲清少掉的张数去哪了，不能拿"入库 255 张"糊过去。"""
    base = server[0]
    status, _headers, body = upload(base, "第一次.jpg", jpeg_bytes(9901))
    assert status == 200, body
    first = json.loads(body)["stored"]
    status, _headers, body = call(base, "/api/jobs",
                                 data=json.dumps({"kind": "pool", "files": [first]}).encode(),
                                 headers=JSON)
    assert status in (200, 202), body
    job = wait_job(base, json.loads(body)["id"])
    assert job["ok"] and job["result"]["imported"] == 1, job
    assert "与已有素材内容相同" not in job["message"], job["message"]

    # 同一个内容换个文件名再传一次：张数要报 0，并且说清是被去重跳过的
    status, _headers, body = upload(base, "同一个内容又传一次.jpg", jpeg_bytes(9901))
    second = json.loads(body)["stored"]
    status, _headers, body = call(base, "/api/jobs",
                                 data=json.dumps({"kind": "pool", "files": [second]}).encode(),
                                 headers=JSON)
    job = wait_job(base, json.loads(body)["id"])
    assert job["ok"] and job["result"]["imported"] == 0, job
    assert "与已有素材内容相同" in job["message"], job["message"]
