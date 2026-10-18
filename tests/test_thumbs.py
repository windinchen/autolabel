"""队列缩略图端点：小图 + 长缓存，坏图必须回退原图而不是报错。"""

import io
import os

from PIL import Image

from autolabel.library import thumbs_dir
from test_prompt_web import get  # 与其它 Web 测试同一套助手


def thumb_names(dataset):
    folder = thumbs_dir(dataset)
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def test_thumb_route_serves_small_jpeg_with_long_cache(server, dataset):
    base, _store = server
    status, body, headers = get(base, "/thumb/img0.jpg")
    assert status == 200, body
    assert headers.get("Content-Type") == "image/jpeg"
    assert "max-age=" in (headers.get("Cache-Control") or "")
    with Image.open(io.BytesIO(body)) as img:
        assert img.size == (320, 240)  # 原图 640x480，长边缩到 320
    assert len(body) < os.path.getsize(os.path.join(dataset, "images", "img0.jpg"))
    # 主图仍然不缓存：人工重导同名图时不能看到旧内容
    status, _body, headers = get(base, "/image/img0.jpg")
    assert status == 200
    assert headers.get("Cache-Control") == "no-store"


def test_thumb_is_reused_until_the_image_changes(server, dataset):
    base, _store = server
    first_status, first_body, _headers = get(base, "/thumb/img0.jpg")
    second_status, second_body, _headers = get(base, "/thumb/img0.jpg")
    assert first_status == second_status == 200
    assert first_body == second_body
    assert len(thumb_names(dataset)) == 1  # 第二次直接复用，没有重新生成

    # 换掉原图（尺寸与字节数都变）→ 缓存键变，生成新缩略图且小图不放大
    target = os.path.join(dataset, "images", "img0.jpg")
    Image.new("RGB", (200, 120), (10, 200, 30)).save(target, "JPEG")
    status, body, _headers = get(base, "/thumb/img0.jpg")
    assert status == 200
    with Image.open(io.BytesIO(body)) as img:
        assert img.size == (200, 120)
    assert len(thumb_names(dataset)) == 2


def test_broken_image_falls_back_to_original(server, dataset):
    base, _store = server
    target = os.path.join(dataset, "images", "img1.jpg")
    with open(target, "wb") as fh:
        fh.write(b"\x00\x01not-a-jpeg")
    status, body, headers = get(base, "/thumb/img1.jpg")
    assert status == 200, body
    assert b"not-a-jpeg" in body  # 回退原图，缩略图失败不该把图片请求变成 500
    assert headers.get("Content-Type") == "image/jpeg"
    assert thumb_names(dataset) == []


def test_filmstrip_uses_thumb_route_but_main_viewer_keeps_full_image():
    for name in ("app.js", "objects.js"):
        source = open(os.path.join(os.path.dirname(__file__), "..", "autolabel", "static", name),
                      encoding="utf-8").read()
        assert '"/thumb/"' in source, name  # 队列拉小图
        assert '"/image/"' in source, name  # 主图仍是全分辨率


def test_prune_thumbs_only_removes_stale_files(dataset):
    import time

    from autolabel.thumbs import prune_thumbs

    folder = thumbs_dir(dataset)
    os.makedirs(folder, exist_ok=True)
    fresh = os.path.join(folder, "fresh.jpg")
    stale = os.path.join(folder, "stale.jpg")
    stray = os.path.join(folder, "keep.txt")
    for path in (fresh, stale, stray):
        with open(path, "wb") as fh:
            fh.write(b"x")
    old = time.time() - 40 * 86400
    os.utime(stale, (old, old))

    assert prune_thumbs(dataset) == 1
    assert thumb_names(dataset) == ["fresh.jpg", "keep.txt"]  # 新的与不相干的都留着
