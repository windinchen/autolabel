import os
import sys
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autolabel.dataset import adopt_from_pool, import_images, init_dataset  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_model_paths(tmp_path, monkeypatch):
    """默认把模型登记表、多模态配置与 API Key 文件都指到 tmp。

    否则单个没显式隔离的测试就会读到（甚至按旧格式回写）开发者真实的
    ``configs/vlm.json``——顺手把 ``key_ref`` 冲掉过一次，这里从根上挡住。
    """
    monkeypatch.setenv("AUTOLABEL_MODEL_CONFIG", str(tmp_path / "configs" / "models.json"))
    monkeypatch.setenv("AUTOLABEL_VLM_CONFIG", str(tmp_path / "configs" / "vlm.json"))
    monkeypatch.setenv("AUTOLABEL_SECRETS", str(tmp_path / "secrets.json"))


@pytest.fixture()
def source_dir(tmp_path):
    """导入用的源目录：两张合成图 + 一张字节级完全相同的副本（名字排最后）。"""
    source = tmp_path / "src"
    source.mkdir()
    for i, (w, h) in enumerate([(640, 480), (400, 300)]):
        xs = np.linspace(0, 255, w, dtype=np.uint8)
        arr = np.repeat(xs[None, :, None], h, axis=0)
        arr = np.dstack([arr, 255 - arr, np.full((h, w), 128, np.uint8)])
        Image.fromarray(arr).save(source / f"img{i}.jpg")
    # 字节级完全相同的第三张（名字排在最后，保证 img0/img1 先入库），用于验证精确去重
    (source / "zz_dup.jpg").write_bytes((source / "img0.jpg").read_bytes())
    return str(source)


@pytest.fixture()
def dataset(source_dir, tmp_path):
    """两张合成图片构成的最小数据集。"""
    root = tmp_path / "ds"
    init_dataset(str(root), name="pytest")
    stats = import_images(str(root), [source_dir], dedupe="exact")
    # 两个库都引用同一批素材：这才是分库之后最常见的真实状态
    adopt_from_pool(str(root), [rel for _src, rel in stats["imported_files"]], lib="objects")
    return str(root)


@pytest.fixture()
def server(dataset):
    """在随机端口上起一个真实的复核服务，返回 (base_url, store)。"""
    from autolabel.review import ReviewHandler, ReviewStore

    previous = getattr(ReviewHandler, "store", None)
    store = ReviewStore(dataset)
    ReviewHandler.store = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ReviewHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base, store
    finally:
        httpd.shutdown()
        httpd.server_close()
        ReviewHandler.store = previous


def write_label(root, image_rel, lines):
    from autolabel.dataset import label_path_for

    path = label_path_for(root, image_rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(line + "\n" for line in lines))
    return path
