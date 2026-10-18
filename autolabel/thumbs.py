"""队列缩略图：把素材池里的大图缩成小 JPEG 缓存在数据集里，滚动列表不再拉原图。

队列一次要显示上百张缩略图，直接给原图有两个代价：一是每张几 MB 的解码 +
传输把滚动拖成幻灯片；二是所有响应都带 ``no-store``，浏览器每次翻页都重新拉。
这里按 ``(相对路径, 修改时间, 字节数)`` 做缓存键——图片一换键就变，
所以缩略图 URL 可以放心让浏览器长期缓存，不需要失效逻辑。
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Tuple

from .library import thumbs_dir

THUMB_MAX_SIDE = 320
THUMB_QUALITY = 72
# 缓存目录只增不减会慢慢积累几十上百 MB，启动时清掉久未访问的
THUMB_MAX_AGE_DAYS = 30
# 文件名里已经编进了内容指纹，可以放心让浏览器存一周
THUMB_CACHE_CONTROL = "public, max-age=604800"


# 交给 PIL 之前先验文件头：ultralytics 会给 PIL 挂懒加载插件，`Image.open()`
# 认不出的文件会触发它去联网装 pi-heif，图片请求能在请求线程里卡十几秒
_IMAGE_MAGICS = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM")


def looks_like_image(path: str) -> bool:
    """只看文件头判断是不是我们能缩的位图，不做解码。"""
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        return False
    if head.startswith(b"RIFF"):
        return head[8:12] == b"WEBP"
    return any(head.startswith(magic) for magic in _IMAGE_MAGICS)


def thumb_key(image_rel: str, mtime_ns: int, size: int) -> str:
    """缩略图文件名：同一张图同一份内容永远得到同一个键。"""
    raw = f"{image_rel}|{mtime_ns}|{size}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:20] + ".jpg"


def thumb_size(width: int, height: int, max_side: int = THUMB_MAX_SIDE) -> Tuple[int, int]:
    """等比缩放后的目标尺寸；小图不放大。"""
    if width <= 0 or height <= 0:
        return 0, 0
    scale = min(1.0, float(max_side) / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def build_thumbnail(root: str, path: str, image_rel: str) -> str:
    """生成或复用缩略图，返回文件路径。

    任何一步失败（PIL 缺失、图片损坏、只读目录）都返回空串，让调用方回退原图：
    缩略图只是加速手段，绝不能因为它把图片请求变成 500。
    """
    try:
        stat = os.stat(path)
    except OSError:
        return ""
    if not looks_like_image(path):
        return ""
    folder = thumbs_dir(root)
    target = os.path.join(folder, thumb_key(image_rel, stat.st_mtime_ns, stat.st_size))
    if os.path.exists(target):
        return target
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - Pillow 是硬依赖，缺了就只能回退原图
        return ""
    tmp = f"{target}.{os.getpid()}.tmp"
    try:
        os.makedirs(folder, exist_ok=True)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            with open(tmp, "wb") as handle:
                image.save(handle, "JPEG", quality=THUMB_QUALITY, optimize=True)
        os.replace(tmp, target)
    except Exception:  # noqa: BLE001 - 坏图/写盘失败都回退原图
        try:
            os.remove(tmp)
        except OSError:
            pass
        return ""
    return target


def prune_thumbs(root: str, max_age_days: int = THUMB_MAX_AGE_DAYS) -> int:
    """删掉超过 ``max_age_days`` 没被再生成过的缩略图，返回删除数量。

    图片被删掉之后缩略图没法反查归属，只能靠时间兜底；正在用的图下次访问
    会就地重新生成，所以清干净也不影响正确性。
    """
    folder = thumbs_dir(root)
    if not os.path.isdir(folder):
        return 0
    deadline = time.time() - max(1, int(max_age_days)) * 86400
    removed = 0
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        try:
            if not name.endswith((".jpg", ".tmp")) or os.path.getmtime(path) >= deadline:
                continue
            os.remove(path)
            removed += 1
        except OSError:  # 别的进程刚好在写，跳过就是了
            continue
    return removed
