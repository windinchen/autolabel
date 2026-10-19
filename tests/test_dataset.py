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
