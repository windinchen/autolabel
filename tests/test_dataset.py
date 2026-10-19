import json
import os

import pytest
from PIL import Image

from autolabel import dataset as ds
from _helpers import idx_names, write_label


def test_init_creates_layout(tmp_path):
    root = str(tmp_path / "fresh")
    idx = ds.init_dataset(root, name="fresh", class_names={0: "person"})
    for sub in ("images", "labels", "qc", "exports", ".autolabel"):
        assert os.path.isdir(os.path.join(root, sub)), sub
    data = json.loads(open(ds.index_path(root), encoding="utf-8").read())
    assert data["kpt_shape"] == [17, 3]
    assert data["class_names"] == {"0": "person"}
    assert "v=1" in json.dumps(data, ensure_ascii=False) or data["visibility_meaning"]["1"]
    # 幂等：重复 init 不清空已有内容
    idx.upsert("a.jpg", width=10, height=10)
    idx.save()
    ds.init_dataset(root)
    assert "a.jpg" in ds.load_index(root).items


def test_fixture_imported_two_unique_images(dataset):
    idx = ds.load_index(dataset)
    assert len(idx.items) == 2
    for rel, item in idx.items.items():
        assert item["status"] == ds.STATUS_UNREVIEWED
        assert item["sha256"] and item["width"] > 0


def test_exact_dedupe_skips_same_bytes(dataset, tmp_path):
    dup = tmp_path / "again.jpg"
    first = os.path.join(ds.images_dir(dataset), sorted(idx_names(dataset))[0])
    Image.open(first).save(dup, "JPEG")
    stats = ds.import_images(dataset, [str(dup)], dedupe="exact")
    assert stats["skipped_exact"] == 1
    assert stats["imported"] == 0
    assert len(ds.load_index(dataset).items) == 2


def test_reimporting_the_pool_folder_reuses_instead_of_copying(dataset):
    """把 ``images/`` 自己再喂回导入口：认出是同一张，不许再复制一份双胞胎。

    搬库、拿导出目录回灌、脚本里图省事直接喂 images 下的文件都会撞上这个。
    账上 ``sha256`` 记的是导入时那张原图，盘上躺的是规范化副本，两份字节不同——
    过去只比原图哈希，整张批认不出重复，原样再落一遍 ``xxx_1.jpg``，
    盘上凭空多一半，去重统计还照样好看，谁都不知道图是被自己复制的。
    """
    from autolabel import dataset as ds

    before = sorted(os.listdir(ds.images_dir(dataset)))
    assert before
    sources = [os.path.join(ds.images_dir(dataset), rel) for rel in before]
    stats = ds.import_images(dataset, sources, dedupe="exact", lib="detect")
    assert stats["imported"] == 0, stats
    assert stats["reused"] == len(before), stats
    assert sorted(os.listdir(ds.images_dir(dataset))) == before, "不该多出一份副本"
    assert len(ds.load_index(dataset, "detect").items) == len(before)


def test_old_ledger_recognises_its_own_files_by_path(dataset):
    """老库账本里只有原图哈希：把 ``images/`` 喂回来照样不许复制——认路径，不只看哈希。

    新字段 ``stored_sha256`` 只在新导入时才有，已经建好的库（比如手上那个两百多张的）
    账本里补不出来，所以认自己人这件事不能只靠哈希表。
    """
    from autolabel import dataset as ds
    from autolabel import pool as pool_catalog

    before = sorted(os.listdir(ds.images_dir(dataset)))
    assert before
    catalog = pool_catalog.load_catalog(dataset)
    for item in catalog["items"].values():
        item.pop(ds.STORED_SHA_KEY, None)
    pool_catalog.save_catalog(dataset, catalog)
    index = ds.load_index(dataset, "pose")
    for item in index.items.values():
        item.pop(ds.STORED_SHA_KEY, None)
    index.save()

    sources = [os.path.join(ds.images_dir(dataset), rel) for rel in before]
    stats = ds.import_images(dataset, sources, dedupe="exact", lib="detect")
    assert stats["imported"] == 0, stats
    assert stats["reused"] == len(before), stats
    assert sorted(os.listdir(ds.images_dir(dataset))) == before, "老库也不该复制出双胞胎"


def test_pool_only_item_records_the_bytes_actually_stored(dataset, tmp_path):
    """只进池的素材要把「落盘字节」的哈希也记一笔，否则下一轮回灌认不出它。"""
    from autolabel import dataset as ds
    from autolabel import pool as pool_catalog

    source = tmp_path / "只进池.jpg"
    Image.new("RGB", (40, 30), "white").save(source)
    stats = ds.import_images(dataset, [str(source)], lib="pose", register=False)
    assert stats["imported"] == 1, stats
    rel = stats["imported_files"][0][1]
    item = pool_catalog.load_catalog(dataset)["items"][rel]
    stored = ds.file_sha256(os.path.join(ds.images_dir(dataset), rel))
    assert item["stored_sha256"] == stored, "记的要是盘上这份字节"
    assert item["sha256"] == ds.file_sha256(str(source)), "原图哈希还得留着，重传原件靠它"
    # 只进池的素材不进任何库索引：各模块看不到它，但去重要认得
    assert rel not in ds.load_index(dataset, "pose").items


def test_perceptual_dedupe_catches_near_duplicate(dataset, tmp_path):
    import numpy as np

    base = np.full((300, 400, 3), 90, np.uint8)
    base[50:200, 60:320] = 180
    Image.fromarray(base).save(tmp_path / "p1.jpg")
    noisy = base.copy()
    noisy[100:110, 100:120] = 120  # 仅极小区域差异
    Image.fromarray(noisy).save(tmp_path / "p2.jpg")
    stats = ds.import_images(dataset, [str(tmp_path / "p1.jpg"), str(tmp_path / "p2.jpg")],
                             dedupe="perceptual")
    assert stats["imported"] == 1
    assert stats["skipped_similar"] == 1


def test_import_resizes_oversized_images(tmp_path):
    root = str(tmp_path / "big")
    ds.init_dataset(root)
    src = tmp_path / "huge.jpg"
    Image.new("RGB", (4000, 2000), (10, 20, 30)).save(src, "JPEG")
    ds.import_images(root, [str(src)], max_side=1000)
    idx = ds.load_index(root)
    item = list(idx.items.values())[0]
    assert max(item["width"], item["height"]) == 1000


def test_import_strips_exif_rotation(tmp_path):
    root = str(tmp_path / "exif")
    ds.init_dataset(root)
    src = tmp_path / "rot.jpg"
    Image.new("RGB", (600, 200), (5, 5, 5)).save(src, "JPEG", exif=exif_with_rotation(6))
    ds.import_images(root, [str(src)])
    item = list(ds.load_index(root).items.values())[0]
    # Orientation=6 -> 导入后应已展开为竖图
    assert (item["width"], item["height"]) == (200, 600)


def exif_with_rotation(orientation: int) -> bytes:
    from PIL.Image import Exif

    exif = Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


def test_label_path_mapping_keeps_subdirs(dataset):
    assert ds.label_path_for(dataset, "img0.jpg").endswith(os.path.join("labels", "img0.txt"))
    assert ds.label_path_for(dataset, "images/img1.jpg").endswith(os.path.join("labels", "img1.txt"))
    assert ds.label_path_for(dataset, "sub/a.png").endswith(os.path.join("labels", "sub", "a.txt"))


def test_resolve_image_blocks_traversal(dataset):
    assert ds.resolve_image(dataset, "img0.jpg").endswith("images/img0.jpg")
    base = os.path.abspath(ds.images_dir(dataset))
    # 绝对路径与 ".." 都会被净化成 images/ 内的相对路径，不会读到系统文件
    for candidate in ("/etc/passwd", "../../etc/passwd", "a/../../b.jpg"):
        assert ds.resolve_image(dataset, candidate).startswith(base)
    with pytest.raises(ValueError):
        ds.resolve_image(dataset, "///")


def test_sync_missing_labels_creates_backgrounds(dataset):
    names = sorted(idx_names(dataset))
    write_label(dataset, names[0], ["0 0.5 0.5 0.2 0.4 " + "0.1 0.1 2 " * 17])
    assert ds.sync_missing_labels(dataset) == len(names) - 1
    for rel in idx_names(dataset):
        assert os.path.exists(ds.label_path_for(dataset, rel))
    # 第二次没有缺口，不应重复写盘
    assert ds.sync_missing_labels(dataset) == 0


def test_index_roundtrip_is_atomic(dataset):
    idx = ds.load_index(dataset)
    idx.upsert("x.jpg", status=ds.STATUS_APPROVED, objects=3)
    idx.save()
    reloaded = ds.load_index(dataset)
    assert reloaded.get("x.jpg")["status"] == ds.STATUS_APPROVED
    leftovers = [n for n in os.listdir(os.path.dirname(ds.index_path(dataset))) if ".tmp" in n]
    assert leftovers == [], f"临时文件没清干净：{leftovers}"
