"""数据集目录布局、标注库索引与导入（含精确/感知去重）。

目录布局（**素材池共享，标注库独立**，详见 :mod:`autolabel.library`）::

    <dataset>/
      images/                 # 共享素材池：统一规范化后的图片（RGB、EXIF 已展开）
      labels/                 # 姿态库标签，与图片同名（YOLO pose 格式）
      prompt_labels/          # 提示词库标签（YOLO detection 格式）
      qc/  exports/  coords/  # 三个库的产出各写各的文件名
      .autolabel/
        index.json            # 姿态库索引：图片元信息 + 骨骼复核状态 + 标注参数
        batches.json          # 姿态库批次登记表
        originals/ history/   # 姿态库的人工原件与撤销快照
        fields.json           # 提示词库的字段登记表
        objects/              # 提示词库：独立索引 + 独立批次 + 独立备份
          index.json batches.json originals/ history/
        deleted/              # 各库共用的删除快照（快照里记 lib）

同一个池里的图片可以同时被三个库引用，删除时只摘本库引用；界面上区分
「从本库移除」和「删素材」，只有所有库都不再引用的图片才会真正落盘删除。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .library import (LIB_DETECT, LIB_NAMES, LIB_OBJECTS, LIB_POSE, LIBRARIES, backups_dir,
                      deleted_dir, ensure_lib, exports_dir, history_root, images_dir,
                      index_path, labels_dir, meta_dir, normalize,
                      pool_files, qc_dir)
from .atomic import atomic_write_json
from .library import field_key
from .library import status_key as library_status_key
from .yolo_pose import COCO17_FLIP_IDX, COCO17_KEYPOINTS

INDEX_VERSION = 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

STATUS_UNREVIEWED = "unreviewed"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_EDITED = "edited"

# 人工已经定论的两个状态：接受 = 确认要进训练集，驳回 = 明确不要。
# 「等你复核」「优先队列」「导出默认范围」全以这条线为界——界面前后、命令行与
# 网页必须是同一个口径，否则同一个词在三处给出三个数，用户不知道该信哪个。
# 注意与下面的 HUMAN_STATUSES（approved+edited，"人工碰过就别覆盖"）区分：
# 那条线保护成果，这条线判断还需不需要人再看一眼，用途不同不能混用。
#: 复核状态的全集：命令行参数、界面请求、索引落盘三处同一个口径，别各写一份
ALL_STATUSES = (STATUS_UNREVIEWED, STATUS_EDITED, STATUS_APPROVED, STATUS_REJECTED)

DEFINITIVE_STATUSES = (STATUS_APPROVED, STATUS_REJECTED)

#: 人工在复核界面碰过的两个状态：接受过的、手工改过的。批量重标默认绕开它们。
#: 这条线以前只有姿态流水线认（定义在 pipeline 里），检测与提示词两处各自抄了一遍
#: 且都漏了 edited——"改过但还没点接受"的框会被下一次批量标注悄悄抹掉。
#: 现在三处共用这一个定义，别再出现第三种说法。
HUMAN_STATUSES = (STATUS_APPROVED, STATUS_EDITED)


# --------------------------------------------------------------------------- #
# 路径工具
# --------------------------------------------------------------------------- #

def prompt_labels_dir(root: str) -> str:
    """提示词库标签目录（旧调用名保留，等价于 ``labels_dir(root, LIB_OBJECTS)``）。"""
    return labels_dir(root, LIB_OBJECTS)


def label_path_for(root: str, image_rel: str, lib: str = LIB_POSE) -> str:
    """images/foo.jpg -> labels/foo.txt（提示词库为 prompt_labels/foo.txt）。"""
    rel = image_rel.replace("\\", "/")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return os.path.join(labels_dir(root, lib), os.path.splitext(rel)[0] + ".txt")


def prompt_label_path_for(root: str, image_rel: str) -> str:
    """images/foo.jpg -> prompt_labels/foo.txt（YOLO detection 格式）。"""
    return label_path_for(root, image_rel, LIB_OBJECTS)


def resolve_image(root: str, image_rel: str) -> str:
    """把索引里的相对路径安全映射到 images/ 下的绝对路径（阻断越界穿越）。"""
    base = os.path.abspath(images_dir(root))
    rel = image_rel.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"非法图片路径: {image_rel!r}")
    path = os.path.abspath(os.path.join(base, *parts))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"图片路径越界: {image_rel!r}")
    return path


def source_missing(root: str, image_rel: str) -> bool:
    """索引里有记录、磁盘上却没有这张图（手工删了 / 外接盘没接上）。

    以前导出端对这种情况要么整批崩（copy/hardlink 抛 FileNotFoundError，
    报的还是一句裸的 ``[Errno 2]``），要么悄悄产出一根指向空气的软链
    （symlink：manifest 照报"train 4 张"，要到训练读图时才炸）。三处口径必须
    先有一个共同的判定，才谈得上统一行为。
    """
    try:
        path = resolve_image(root, image_rel)
    except ValueError:
        return True
    return not os.path.isfile(path)


def missing_sources(root: str, rels: Iterable[str]) -> List[str]:
    """挑出索引里指向丢失文件的相对路径，保持传入顺序。"""
    return [rel for rel in rels if source_missing(root, rel)]


def unknown_images(index: "DatasetIndex", rels: Iterable[str]) -> List[str]:
    """点名的图片里哪些压根不在本库索引，保持点名的先后并去重。

    ``--images`` 打错一个字，以前会照样 ``upsert`` 出一条 ``model_failed`` 的
    幽灵记录：界面凭空多出一行没有宽高的图，张数、未标数、缩略图列表全跟着对
    不上。界面那条路有 :meth:`ReviewStore._require_item` 把关，命令行这条路以前
    没人管——而它恰恰是最容易打错的一条。
    """
    out: List[str] = []
    for rel in rels:
        text = str(rel)
        if text not in out and index.get(text) is None:
            out.append(text)
    return out


# --------------------------------------------------------------------------- #
# 索引
# --------------------------------------------------------------------------- #

@dataclass
class DatasetIndex:
    """一个标注库的索引。``lib`` 决定 :meth:`save` 落到哪份 ``index.json``。"""

    root: str
    data: dict
    lib: str = LIB_POSE

    def __post_init__(self) -> None:
        self.lib = normalize(self.lib)

    @property
    def lib_name(self) -> str:
        return LIB_NAMES[self.lib]

    # --- 便捷访问 ---
    @property
    def name(self) -> str:
        return self.data.get("name", os.path.basename(os.path.abspath(self.root)))

    @property
    def class_names(self) -> Dict[int, str]:
        return {int(k): v for k, v in self.data.get("class_names", {"0": "person"}).items()}

    @property
    def num_keypoints(self) -> int:
        return int(self.data.get("kpt_shape", [len(COCO17_KEYPOINTS), 3])[0])

    @property
    def items(self) -> Dict[str, dict]:
        return self.data.setdefault("items", {})

    def get(self, image_rel: str) -> Optional[dict]:
        return self.items.get(image_rel.replace("\\", "/"))

    def upsert(self, image_rel: str, **fields) -> dict:
        key = image_rel.replace("\\", "/")
        item = self.items.setdefault(key, {})
        item.update(fields)
        return item

    def remove(self, image_rel: str) -> None:
        self.items.pop(image_rel.replace("\\", "/"), None)

    def ordered_items(self) -> List[Tuple[str, dict]]:
        return sorted(self.items.items(), key=lambda kv: kv[0])

    def save(self) -> None:
        self.data["updated_at"] = now_str()
        self.data["lib"] = self.lib
        atomic_write_json(index_path(self.root, self.lib), self.data, sort_keys=True)


def status_of(item: Optional[dict], lib: str = LIB_POSE) -> str:
    """取一张图的复核状态。每库用自己的键（``status`` / ``prompt_status`` /
    ``detect_status``），所以删库、重建索引不会把别的库的状态串过来。"""
    return str((item or {}).get(library_status_key(lib)) or STATUS_UNREVIEWED)


def is_definitive(item: Optional[dict], lib: str = LIB_POSE) -> bool:
    """人工点过「接受」或「驳回」= 已定论，这一帧不再占用「等你复核」。"""
    return status_of(item, lib) in DEFINITIVE_STATUSES


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def init_dataset(root: str, name: Optional[str] = None,
                 class_names: Optional[Dict[int, str]] = None,
                 num_keypoints: int = len(COCO17_KEYPOINTS)) -> DatasetIndex:
    """创建（或补齐）数据集骨架，已存在时不破坏已有内容。"""
    root = os.path.abspath(root)
    os.makedirs(images_dir(root), exist_ok=True)
    os.makedirs(exports_dir(root), exist_ok=True)
    for lib in LIBRARIES:
        ensure_lib(root, lib)
    idx = _init_pose_index(root, name, class_names, num_keypoints)
    for lib in LIBRARIES:
        if lib != LIB_POSE:
            _init_box_index(root, lib, name)
    return idx


def _init_pose_index(root: str, name: Optional[str] = None,
                     class_names: Optional[Dict[int, str]] = None,
                     num_keypoints: int = len(COCO17_KEYPOINTS)) -> DatasetIndex:
    """建/读姿态库索引。"""
    path = index_path(root, LIB_POSE)
    if os.path.exists(path):
        return load_index(root, LIB_POSE)
    data = {
        "version": INDEX_VERSION,
        "lib": LIB_POSE,
        "name": name or os.path.basename(root),
        "created_at": now_str(),
        "updated_at": now_str(),
        "class_names": {str(k): v for k, v in (class_names or {0: "person"}).items()},
        "kpt_shape": [num_keypoints, 3],
        "keypoint_names": list(COCO17_KEYPOINTS[:num_keypoints]),
        "flip_idx": list(COCO17_FLIP_IDX[:num_keypoints]),
        "label_format": "class cx cy w h (x y v)*n  (归一化)",
        "visibility_meaning": {"0": "未标注/忽略", "1": "遮挡但参与监督", "2": "可见"},
        "settings": {},
        "items": {},
    }
    idx = DatasetIndex(root=root, data=data, lib=LIB_POSE)
    idx.save()
    return idx


def _init_box_index(root: str, lib: str = LIB_OBJECTS,
                    name: Optional[str] = None) -> DatasetIndex:
    """建/读一个「框」类库的索引（提示词库与目标检测库结构一样，只差库名）。"""
    lib = normalize(lib)
    if os.path.exists(index_path(root, lib)):
        return load_index(root, lib)
    data = {
        "version": INDEX_VERSION,
        "lib": lib,
        "name": name or os.path.basename(root),
        "created_at": now_str(),
        "updated_at": now_str(),
        "label_format": "class cx cy w h (归一化, YOLO detection)",
        "settings": {},
        "items": {},
    }
    idx = DatasetIndex(root=root, data=data, lib=lib)
    idx.save()
    return idx


def _init_objects_index(root: str, name: Optional[str] = None) -> DatasetIndex:
    """建/读提示词库索引（旧调用名保留）。"""
    return _init_box_index(root, LIB_OBJECTS, name)


def load_index(root: str, lib: str = LIB_POSE) -> DatasetIndex:
    """读某个库的索引；文件缺失时抛 :class:`FileNotFoundError`。"""
    root = os.path.abspath(root)
    lib = normalize(lib)
    path = index_path(root, lib)
    if not os.path.exists(path):
        raise FileNotFoundError(f"不是有效的自动标注数据集（缺少 {path}），先执行 init")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("items", {})
    return DatasetIndex(root=root, data=data, lib=lib)


def load_index_or_none(root: str, lib: str = LIB_POSE) -> Optional[DatasetIndex]:
    """读不到就返回 None：对账/删除时要跨库看引用，缺一个库不该直接报错。"""
    try:
        return load_index(root, lib)
    except (FileNotFoundError, OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 图片导入
# --------------------------------------------------------------------------- #

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_size(path: str) -> Tuple[int, int]:
    """只读文件头拿宽高（不整张解码）：给"手工拷进 images/ 的素材"补尺寸用。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001 读不出就当不知道，别拖垮整张清单
        return 0, 0


def _load_rgb(path: str, max_side: Optional[int] = None):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            from PIL import Image as _I

            w, h = im.size
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), _I.LANCZOS)
        return im.copy()


#: 复用素材池时从另一个库带过来的素材元信息（谁导入的、多大、什么尺寸）
POOL_KEYS = ("width", "height", "sha256", "bytes", "dhash", "imported_at", "source", "video")


def _entry_defaults(lib: str) -> dict:
    """新库条目的初始状态字段：每个库只记自己的状态与计数，互不掺水。

    姿态库用不带前缀的 ``status`` / ``objects`` / ``flags``（沿用老数据集的键名），
    提示词库与目标检测库各自带前缀，这样跨库复制元信息时不会把状态串味。
    """
    name = normalize(lib)
    return {field_key(name, "status"): STATUS_UNREVIEWED,
            field_key(name, "objects"): None, field_key(name, "flags"): []}


def pool_entries(root: str, lib: str = LIB_POSE) -> Dict[str, dict]:
    """三个库索引合并出的「素材池视角」：``相对路径 -> 条目``，用于跨库去重与复用。"""
    from . import pool as pool_catalog

    merged: Dict[str, dict] = {}
    for other in LIBRARIES:
        other_idx = load_index_or_none(root, other)
        if other_idx is None:
            continue
        for rel, item in other_idx.items.items():
            merged.setdefault(rel, item)
    merged.pop("__missing__", None)
    # 只进池、还没被任何库引用的素材记在池账本里：漏掉它们，下一次导入认不出这是重复帧
    # （同一个视频抽两遍就落双份），「从素材池添加」也会回一句"池里查无此图"
    for rel, item in pool_catalog.pool_items_map(root).items():
        if rel not in merged:
            merged[rel] = {name: value for name, value in item.items()
                           if name in POOL_KEYS and value is not None}
    return merged


def libs_holding_image(root: str, image_rel: str) -> List[str]:
    """这张素材现在被哪些库引用（删文件前先问这个）。"""
    rel = str(image_rel).replace("\\", "/")
    holders = []
    for lib in LIBRARIES:
        other = load_index_or_none(root, lib)
        if other is not None and rel in other.items:
            holders.append(lib)
    return holders


def adopt_pool_entry(target: DatasetIndex, source_item: Optional[dict], rel: str) -> bool:
    """把素材池里已有的图片登记进本库，不重新落盘。返回是否真的新增了条目。"""
    if not source_item:
        return False
    try:
        resolve_image(target.root, rel)
    except ValueError:
        return False
    if not os.path.exists(resolve_image(target.root, rel)):
        return False
    fields = {key: source_item[key] for key in POOL_KEYS if key in source_item}
    fields.update(_entry_defaults(target.lib))
    target.upsert(rel, **fields)
    return True


def pool_items(root: str) -> List[dict]:
    """素材池清单：每个文件在哪些库里被引用了，供「从素材池添加」与删除确认用。

    ``in_<库名>`` 是各库的单独开关（门户按它决定"这张要不要出现在本库"），
    ``libs`` 是引用它的库名列表——判断"删文件会不会影响别的库"只看这一个，
    两个库的时代可以瞎写 ``in_pose and in_objects``，三个库就不够用了。
    """
    from . import pool as pool_catalog

    indexes = {lib: load_index_or_none(root, lib) for lib in LIBRARIES}
    catalog = pool_catalog.load_catalog(root)
    rows: List[dict] = []
    for rel in pool_files(root):
        found = {lib: (idx.get(rel) if idx is not None else None) for lib, idx in indexes.items()}
        # 只进池的图三个库索引里都没有，尺寸与大小得从池账本补，否则界面上全是 0 KB
        own = catalog["items"].get(rel) or {}
        base = dict(next((item for item in found.values() if item), {}))
        for name in pool_catalog.POOL_ITEM_KEYS:
            if base.get(name) is None and own.get(name) is not None:
                base[name] = own[name]
        if base.get("bytes") is None and base.get("width") is None:
            # 三处都查不到（既没入库也没记账）：按磁盘现状补，卡片才不必写"尺寸未知"
            path = os.path.join(images_dir(root), rel)
            try:
                base["bytes"] = os.path.getsize(path)
            except OSError:
                base["bytes"] = 0
            width, height = image_size(path)
            base["width"], base["height"] = width or None, height or None
        holders = [lib for lib, item in found.items() if item is not None]
        rows.append({
            "image": rel, "name": os.path.basename(rel),
            "bytes": int(base.get("bytes") or 0),
            "width": base.get("width"), "height": base.get("height"),
            "imported_at": base.get("imported_at"),
            "batch_id": base.get("batch_id") or "",
            "video": os.path.basename(str(base["video"])) if base.get("video") else "",
            "in_pose": found[LIB_POSE] is not None, "in_objects": found[LIB_OBJECTS] is not None,
            "in_detect": found[LIB_DETECT] is not None, "libs": holders,
            "status": (found[LIB_POSE] or {}).get("status") or "",
            "prompt_status": (found[LIB_OBJECTS] or {}).get("prompt_status") or "",
            "detect_status": (found[LIB_DETECT] or {}).get("detect_status") or "",
        })
    rows.sort(key=lambda row: str(row.get("imported_at") or ""), reverse=True)
    return pool_catalog.attach(root, rows, catalog=catalog)


def adopt_from_pool(root: str, image_rels: Sequence[str], lib: str = LIB_POSE) -> dict:
    """把素材池里已有的图片添加到某个库（不复制文件），返回统计。"""
    root = os.path.abspath(root)
    lib = normalize(lib)
    idx = load_index(root, lib)
    pool = pool_entries(root, lib)
    stats = {"requested": 0, "added": 0, "already": 0, "missing": [], "added_items": []}
    for rel in image_rels:
        rel = str(rel).replace("\\", "/")
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        if not rel:
            continue
        stats["requested"] += 1
        if idx.get(rel) is not None:
            stats["already"] += 1
            continue
        if adopt_pool_entry(idx, pool.get(rel), rel):
            stats["added"] += 1
            stats["added_items"].append(rel)
        else:
            stats["missing"].append(rel)
    if stats["added"]:
        idx.save()
    stats["lib"] = lib
    return stats


def _unique_name(dst_dir: str, stem: str, ext: str) -> str:
    candidate = stem + ext
    i = 1
    while os.path.exists(os.path.join(dst_dir, candidate)):
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate


def import_images(
    root: str,
    sources: Sequence[str],
    recursive: bool = True,
    max_side: int = 1600,
    jpeg_quality: int = 95,
    dedupe: str = "exact",
    subdir: str = "",
    on_progress=None,
    lib: str = LIB_POSE,
    register: bool = True,
) -> dict:
    """把图片规范化导入共享素材池，并登记进 ``lib`` 指定的标注库。

    :param dedupe: ``none`` | ``exact``（内容哈希）| ``perceptual``（dHash 近似重复）
    :param lib: 目标标注库，``pose`` 或 ``objects``
    :param register: ``False`` 表示**只进素材池**，不进任何标注库索引；这类素材只在
        池账本里记一笔，各模块要用时再从池里挑（``adopt_from_pool``）

    去重看的是**素材池**（三个库的并集）：另一个库已经存过同一帧时不再复制文件，
    只把已有文件登记进本库，计入 ``reused``。本库已有才算 ``skipped_*``。
    """
    from .image_utils import dhash64, hamming

    lib = normalize(lib)
    idx = load_index(root, lib)
    pool = pool_entries(root, lib)
    ensure_lib(os.path.abspath(root), lib)
    dst_root = os.path.join(images_dir(root), subdir) if subdir else images_dir(root)
    os.makedirs(dst_root, exist_ok=True)

    files = collect_image_files(sources, recursive=recursive)
    exact_seen: Dict[str, str] = {v.get("sha256", ""): k for k, v in pool.items() if v.get("sha256")}
    phash_seen: List[Tuple[int, str]] = []
    if dedupe == "perceptual":
        for rel, item in pool.items():
            digest = item.get("dhash")
            if digest is not None:
                phash_seen.append((int(digest), rel))

    stats = {"scanned": len(files), "imported": 0, "reused": 0, "skipped_exact": 0,
             "skipped_similar": 0, "failed": 0, "imported_files": [], "lib": lib}

    def _reuse(rel: str) -> str:
        """命中池里已有的同一帧：复用文件，本库新增条目。"""
        if not register:
            # 只进池：走到这里说明池里已经有同一帧，本库索引一个字节都不动
            return ""
        if idx.get(rel) is not None:
            return ""
        return rel if adopt_pool_entry(idx, pool.get(rel), rel) else ""

    for src in files:
        try:
            digest = file_sha256(src)
            if dedupe in ("exact", "perceptual") and digest in exact_seen:
                rel = exact_seen[digest]
                if not source_missing(root, rel):
                    if _reuse(rel):
                        stats["reused"] += 1
                        stats["imported_files"].append((src, rel))
                        if on_progress:
                            on_progress(rel, stats)
                    else:
                        stats["skipped_exact"] += 1
                    continue
                # 空账：账上记着这张图、盘上却没有（删批次时只摘了来处章、手工删过
                # images/、上次导入断在半路）。拿它当重复，新图就既落不了盘也不报错，
                # 传 255 张只进 169 张就是这么丢的。去重要认图，不能只认账。
                exact_seen.pop(digest, None)
                pool.pop(rel, None)
            im = _load_rgb(src, max_side=max_side)
            ph = dhash64(im) if dedupe == "perceptual" else None
            if ph is not None:
                hit = next((rel for digest, rel in phash_seen
                            if hamming(digest, ph) <= 5 and not source_missing(root, rel)), None)
                if hit:
                    rel = _reuse(hit)
                    if rel:
                        stats["reused"] += 1
                        stats["imported_files"].append((src, rel))
                        if on_progress:
                            on_progress(rel, stats)
                    else:
                        stats["skipped_similar"] += 1
                    continue
            ext = ".jpg"
            stem = os.path.splitext(os.path.basename(src))[0]
            stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:80] or "img"
            name = _unique_name(dst_root, stem, ext)
            abs_dst = os.path.join(dst_root, name)
            im.save(abs_dst, "JPEG", quality=jpeg_quality, optimize=True)
            rel = os.path.relpath(abs_dst, images_dir(root)).replace("\\", "/")
            w, h = im.size
            fields = dict(width=w, height=h, sha256=digest, bytes=os.path.getsize(abs_dst),
                          imported_at=now_str(), source=os.path.abspath(src))
            fields.update(_entry_defaults(lib))
            if ph is not None:
                fields["dhash"] = ph
            if register:
                idx.upsert(rel, **fields)
            else:
                from . import pool as pool_catalog

                pool_catalog.record_file(root, rel, fields)
            exact_seen[digest] = rel
            if ph is not None:
                phash_seen.append((ph, rel))
            stats["imported"] += 1
            stats["imported_files"].append((src, rel))
            if on_progress:
                on_progress(rel, stats)
        except Exception as exc:  # noqa: BLE001 - 单张失败不阻断整批
            stats["failed"] += 1
            print(f"[warn] 导入失败 {src}: {type(exc).__name__}: {exc}")
    if register:
        idx.save()
    stats["register"] = bool(register)
    return stats


def collect_image_files(sources: Iterable[str], recursive: bool = True) -> List[str]:
    """展开输入路径（文件/目录混合），返回去重后的图片列表。"""
    out: List[str] = []
    seen = set()
    for source in sources:
        source = os.path.abspath(os.path.expanduser(source))
        if os.path.isdir(source):
            walker = os.walk(source) if recursive else [(source, [], os.listdir(source))]
            for dirpath, dirnames, filenames in walker:
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                for fn in sorted(filenames):
                    if os.path.splitext(fn)[1].lower() in IMAGE_EXTS and not fn.startswith("."):
                        p = os.path.join(dirpath, fn)
                        if p not in seen:
                            seen.add(p)
                            out.append(p)
        elif os.path.isfile(source):
            if source not in seen:
                seen.add(source)
                out.append(source)
        else:
            print(f"[warn] 路径不存在，跳过: {source}")
    return out


def sync_missing_labels(root: str, lib: str = LIB_POSE) -> int:
    """给某库索引里缺标签文件的图片补空标签，返回补建数量。"""
    lib = normalize(lib)
    idx = load_index(root, lib)
    n = 0
    for rel, item in idx.ordered_items():
        lp = label_path_for(root, rel, lib)
        if not os.path.exists(lp):
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            open(lp, "w", encoding="utf-8").close()
            n += 1
    if n:
        idx.save()
    return n


def import_prepared(
    root: str,
    prepared: Sequence[Tuple[str, str, Optional[dict]]],
    dedupe: str = "perceptual",
    max_side: int = 1600,
    jpeg_quality: int = 95,
    subdir: str = "",
    lib: str = LIB_POSE,
    register: bool = True,
) -> dict:
    """按指定文件名导入图片，用于视频帧这类需要携带时间戳的来源。

    :param prepared: (源文件, 目标文件名主干, 附加索引字段) 列表
    :param lib: 目标标注库，去重同样以共享素材池为准
    :param register: ``False`` 表示只进素材池，不进标注库索引（素材页上传视频用）
    """
    from .image_utils import dhash64, hamming

    lib = normalize(lib)
    idx = load_index(root, lib)
    pool = pool_entries(root, lib)
    ensure_lib(os.path.abspath(root), lib)
    dst_root = os.path.join(images_dir(root), subdir) if subdir else images_dir(root)
    os.makedirs(dst_root, exist_ok=True)
    exact_seen = {v.get("sha256"): k for k, v in pool.items() if v.get("sha256")}
    phash_seen = [(int(v["dhash"]), k) for k, v in pool.items() if v.get("dhash") is not None]

    stats = {"scanned": len(prepared), "imported": 0, "reused": 0, "skipped_exact": 0,
             "skipped_similar": 0, "failed": 0, "imported_files": [], "lib": lib}

    def _reuse(rel: str) -> str:
        if not register:
            # 只进池：池里已有同一帧就是不落新文件，别库的索引也不该被这次导入改动
            return ""
        if idx.get(rel) is not None:
            return ""
        return rel if adopt_pool_entry(idx, pool.get(rel), rel) else ""

    for src, stem, extra in prepared:
        try:
            digest = file_sha256(src)
            if dedupe in ("exact", "perceptual") and digest in exact_seen:
                rel = exact_seen[digest]
                if not source_missing(root, rel):
                    reused = _reuse(rel)
                    if reused:
                        fields = dict(extra or {})
                        if fields:
                            idx.upsert(reused, **fields)
                        stats["reused"] += 1
                        stats["imported_files"].append((src, reused))
                    else:
                        stats["skipped_exact"] += 1
                    continue
                # 空账不算重复（同 :func:`import_images`）：抽帧也走这条，视频导两遍才不会丢帧
                exact_seen.pop(digest, None)
                pool.pop(rel, None)
            im = _load_rgb(src, max_side=max_side)
            ph = dhash64(im) if dedupe == "perceptual" else None
            similar = next((rel for d, rel in phash_seen
                            if ph is not None and hamming(d, ph) <= 5
                            and not source_missing(root, rel)), None)
            if similar is not None:
                reused = _reuse(similar)
                if reused:
                    fields = dict(extra or {})
                    if fields:
                        idx.upsert(reused, **fields)
                    stats["reused"] += 1
                    stats["imported_files"].append((src, reused))
                else:
                    stats["skipped_similar"] += 1
                continue
            stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:80] or "frame"
            name = _unique_name(dst_root, stem, ".jpg")
            abs_dst = os.path.join(dst_root, name)
            im.save(abs_dst, "JPEG", quality=jpeg_quality, optimize=True)
            rel = os.path.relpath(abs_dst, images_dir(root)).replace("\\", "/")
            w, h = im.size
            fields = dict(width=w, height=h, sha256=digest, bytes=os.path.getsize(abs_dst),
                          imported_at=now_str(), source=os.path.abspath(src))
            fields.update(_entry_defaults(lib))
            if ph is not None:
                fields["dhash"] = ph
            fields.update(extra or {})
            if register:
                idx.upsert(rel, **fields)
            else:
                from . import pool as pool_catalog

                pool_catalog.record_file(root, rel, fields)
            exact_seen[digest] = rel
            if ph is not None:
                phash_seen.append((ph, rel))
            stats["imported"] += 1
            stats["imported_files"].append((src, rel))
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            print(f"[warn] 导入失败 {src}: {type(exc).__name__}: {exc}")
    if register:
        idx.save()
    stats["register"] = bool(register)
    return stats


# ``library`` 提供的路径工具在这里重新导出，历史调用（``from .dataset import images_dir``）
# 与测试里的 ``ds.index_path(...)`` 都还能用；新增代码请直接从 :mod:`autolabel.library` 取。
__all__ = [
    "COCO17_KEYPOINTS", "IMAGE_EXTS", "INDEX_VERSION", "LIB_OBJECTS", "LIB_POSE", "LIBRARIES",
    "ALL_STATUSES", "STATUS_APPROVED", "STATUS_EDITED", "STATUS_REJECTED",
    "STATUS_UNREVIEWED", "DatasetIndex",
    "DEFINITIVE_STATUSES", "HUMAN_STATUSES",
    "adopt_from_pool", "backups_dir", "collect_image_files", "deleted_dir", "ensure_lib",
    "exports_dir", "file_sha256", "history_root", "images_dir", "import_images",
    "import_prepared", "index_path", "init_dataset", "label_path_for", "labels_dir",
    "load_index", "load_index_or_none", "meta_dir", "missing_sources", "now_str", "pool_items",
    "prompt_label_path_for", "prompt_labels_dir", "qc_dir", "resolve_image",
    "source_missing", "sync_missing_labels", "unknown_images",
]
