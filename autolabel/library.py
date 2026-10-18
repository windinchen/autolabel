"""标注库（library）：姿态标注与提示词标注各自一套索引、批次、备份，互不干扰。

设计口径——**素材池共享，标注库独立**：

::

    <dataset>/
      images/                 共享素材池（同一帧只落盘一次，两个库都能引用）
      labels/                 姿态库标签（YOLO pose 5 列 + 17 关键点）
      prompt_labels/          提示词库标签（YOLO detection 5 列）
      qc/  coords/  exports/  两套产出各写各的文件名，互不覆盖
      .autolabel/
        index.json            姿态库索引（保持原位，老数据集零迁移）
        batches.json          姿态库批次
        originals/ history/   姿态库的人工原件与撤销快照
        fields.json           提示词库的字段登记表
        objects/              提示词库（独立索引 + 独立批次 + 独立备份）
          index.json
          batches.json
          originals/ history/
        deleted/              两库共用的删除快照（快照里记 lib，知道是从哪个库删的）

为什么要分库：共用一套索引时，「在提示词库里删一张图」会把姿态库的骨骼标注一起带走，
两个模块的统计也总是互相掺水。分成两个库之后每个库只对自己的状态负责，
删图只摘自己的引用；素材文件等两个库都不要了才真删。
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

LIB_POSE = "pose"
LIB_OBJECTS = "objects"
LIBRARIES: Tuple[str, ...] = (LIB_POSE, LIB_OBJECTS)
LIB_NAMES = {LIB_POSE: "姿态标注", LIB_OBJECTS: "提示词标注"}


def normalize(lib: Optional[str]) -> str:
    """把各种写法（``pose`` / ``objects`` / ``prompt`` / 空）归一到库名。"""
    value = str(lib or LIB_POSE).strip().lower()
    if value in ("", LIB_POSE, "pose", "skeleton"):
        return LIB_POSE
    if value in (LIB_OBJECTS, "object", "prompt", "detection", "obj"):
        return LIB_OBJECTS
    raise ValueError(f"未知标注库: {lib}（可用：{'、'.join(LIBRARIES)}）")


def lib_name(lib: Optional[str]) -> str:
    return LIB_NAMES[normalize(lib)]


def lib_dirs(root: str, lib: Optional[str] = LIB_POSE) -> List[str]:
    """该库需要存在的目录（不含素材池）。"""
    name = normalize(lib)
    dirs = [meta_dir(root)]
    if name == LIB_POSE:
        dirs += [labels_dir(root), qc_dir(root), backups_dir(root, name), history_root(root, name)]
    else:
        dirs += [labels_dir(root, name), objects_meta_dir(root),
                 backups_dir(root, name), history_root(root, name)]
    return dirs


# --- 基础目录 -------------------------------------------------------------- #

def meta_dir(root: str) -> str:
    return os.path.join(root, ".autolabel")


def objects_meta_dir(root: str) -> str:
    """提示词库自己的元数据目录。"""
    return os.path.join(meta_dir(root), LIB_OBJECTS)


def images_dir(root: str) -> str:
    """共享素材池：两个库的图片都放这里，同一帧只存一份。"""
    return os.path.join(root, "images")


def labels_dir(root: str, lib: Optional[str] = LIB_POSE) -> str:
    return os.path.join(root, "prompt_labels" if normalize(lib) == LIB_OBJECTS else "labels")


def qc_dir(root: str) -> str:
    return os.path.join(root, "qc")


def exports_dir(root: str) -> str:
    return os.path.join(root, "exports")


def coords_dir(root: str) -> str:
    return os.path.join(root, "coords")


def deleted_dir(root: str) -> str:
    return os.path.join(meta_dir(root), "deleted")


def thumbs_dir(root: str) -> str:
    """队列缩略图缓存目录：可随时删，缺了就地重新生成。"""
    return os.path.join(meta_dir(root), "thumbs")


# --- 每个库一份 ------------------------------------------------------------ #

def index_path(root: str, lib: Optional[str] = LIB_POSE) -> str:
    name = normalize(lib)
    if name == LIB_OBJECTS:
        return os.path.join(objects_meta_dir(root), "index.json")
    return os.path.join(meta_dir(root), "index.json")


def batches_path(root: str, lib: Optional[str] = LIB_POSE) -> str:
    name = normalize(lib)
    if name == LIB_OBJECTS:
        return os.path.join(objects_meta_dir(root), "batches.json")
    return os.path.join(meta_dir(root), "batches.json")


def backups_dir(root: str, lib: Optional[str] = LIB_POSE) -> str:
    """首次人工修改前的原件备份目录（每库一套，撤销与恢复都靠它）。"""
    if normalize(lib) == LIB_OBJECTS:
        return os.path.join(objects_meta_dir(root), "originals")
    return os.path.join(meta_dir(root), "originals")


def history_root(root: str, lib: Optional[str] = LIB_POSE) -> str:
    if normalize(lib) == LIB_OBJECTS:
        return os.path.join(objects_meta_dir(root), "history")
    return os.path.join(meta_dir(root), "history")


def label_path_for(root: str, image_rel: str, lib: Optional[str] = LIB_POSE) -> str:
    """``images/foo.jpg`` -> 该库的 ``<labels>/foo.txt``。"""
    rel = str(image_rel).replace("\\", "/")
    if rel.startswith("images/"):
        rel = rel[len("images/"):]
    return os.path.join(labels_dir(root, lib), os.path.splitext(rel)[0] + ".txt")


# --- 迁移 ------------------------------------------------------------------ #

def legacy_dirs(root: str) -> Dict[str, str]:
    """旧布局里属于提示词库、需要搬进 ``.autolabel/objects/`` 的目录。"""
    return {
        "originals": os.path.join(meta_dir(root), "prompt_originals"),
        "history": os.path.join(meta_dir(root), "prompt_history"),
    }


def split_legacy_index(root: str) -> Tuple[int, int]:
    """把旧索引里的 ``prompt_*`` 键拆到提示词库索引。

    :return: ``(搬动的条目数, 顺手清掉的失效 batch_id 数)``

    旧布局一张图一份索引条目、两套状态混在一起；分库后姿态库只留骨骼相关的键，
    提示词库只留 ``prompt_*`` 与素材元信息（宽高/哈希/导入时间等，两库各存一份，
    因为「谁导入的」属于库的批次信息）。
    """
    from .dataset import DatasetIndex  # 延迟导入，避免和 dataset 形成循环

    legacy = os.path.join(meta_dir(root), "index.json")
    if not os.path.exists(legacy):
        return 0, 0
    with open(legacy, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = data.get("items") or {}

    # 批次归属必须跟着批次表走：旧索引里的 batch_id 可能指向那张已经被划给提示词库
    # 的批次，留着会让姿态库以为自己属于一个本库里不存在的批次（主页多出"记录缺失"虚行）
    from .batches import load_batches  # 延迟导入避免循环

    pose_batch_ids = {str(row.get("id")) for row in load_batches(root, LIB_POSE)}
    objects_batch_ids = {str(row.get("id")) for row in load_batches(root, LIB_OBJECTS)}
    pool_keys = ("width", "height", "sha256", "bytes", "dhash", "imported_at",
                 "source", "video", "batch_id")
    objects_items: Dict[str, dict] = {}
    stripped = 0
    for rel, item in items.items():
        moved = {k: v for k, v in item.items() if str(k).startswith("prompt_")}
        owner = str(item.get("batch_id") or "")
        if moved:
            entry = {k: item[k] for k in pool_keys if k in item}
            if owner and owner not in objects_batch_ids:
                entry.pop("batch_id", None)
            entry.update(moved)
            entry.setdefault("prompt_status", "unreviewed")
            objects_items[rel] = entry
            for key in list(item):
                if str(key).startswith("prompt_"):
                    item.pop(key, None)
        if owner and owner not in pose_batch_ids:
            item.pop("batch_id", None)
            stripped += 1

    if objects_items:
        target = index_path(root, LIB_OBJECTS)
        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "lib": LIB_OBJECTS, "name": data.get("name", ""),
                           "created_at": data.get("created_at", ""),
                           "updated_at": data.get("updated_at", ""),
                           "label_format": "class cx cy w h（归一化，YOLO detection）",
                           "settings": {}, "items": objects_items},
                          fh, ensure_ascii=False, indent=2, sort_keys=True)
    if objects_items or stripped:
        DatasetIndex(root=root, data=data, lib=LIB_POSE).save()
    return len(objects_items), stripped


def split_legacy_batches(root: str) -> int:
    """把旧批次表按 ``kind``/``module`` 拆成两份，返回搬进提示词库的批次数。"""
    from .batches import load_batches, save_batches  # 延迟导入避免循环

    legacy = os.path.join(meta_dir(root), "batches.json")
    if not os.path.exists(legacy):
        return 0
    rows = load_batches(root)
    prompt_rows = [b for b in rows
                   if str(b.get("kind", "")).startswith("prompt") or str(b.get("module", "")) == LIB_OBJECTS
                   or str(b.get("module", "")) == "prompt"]
    if not prompt_rows:
        return 0
    target = batches_path(root, LIB_OBJECTS)
    if not os.path.exists(target):
        for row in prompt_rows:
            row["module"] = LIB_OBJECTS
        save_batches(root, prompt_rows, LIB_OBJECTS)
    pose_rows = [b for b in rows if b not in prompt_rows]
    for row in pose_rows:
        row.setdefault("module", LIB_POSE)
    save_batches(root, pose_rows, LIB_POSE)
    return len(prompt_rows)


def ensure_objects_index(root: str) -> bool:
    """补齐缺失的提示词库索引，返回是否新建了文件。

    纯姿态数据集从没跑过提示词标注时，``split_legacy_index`` 没有 ``prompt_*``
    可搬，提示词库索引不会被创建；而复核服务的存储层要求两本索引都在，
    于是界面直接起不来。迁移时顺手补一本空的，纯姿态数据集也能开网页。
    """
    from .dataset import _init_objects_index, load_index_or_none  # 延迟导入避免循环

    if os.path.exists(index_path(root, LIB_OBJECTS)):
        return False
    pose = load_index_or_none(root, LIB_POSE)
    name = (pose.data.get("name") if pose else None) or None
    ensure_lib(root, LIB_OBJECTS)
    _init_objects_index(root, name)
    return True


def migrate(root: str) -> dict:
    """把旧布局的数据集迁到分库布局；已经分库的目录重复调用无副作用。"""
    root = os.path.abspath(root)
    result = {"moved_items": 0, "moved_batches": 0, "moved_dirs": [], "created": [],
              "stripped_batch_ids": 0}
    if not os.path.isdir(meta_dir(root)):
        return result
    result["moved_batches"] = split_legacy_batches(root)
    result["moved_items"], result["stripped_batch_ids"] = split_legacy_index(root)
    if ensure_objects_index(root):
        result["created"].append(os.path.relpath(index_path(root, LIB_OBJECTS), root))
    for key, old in legacy_dirs(root).items():
        if not os.path.isdir(old):
            continue
        new = backups_dir(root, LIB_OBJECTS) if key == "originals" else history_root(root, LIB_OBJECTS)
        if os.path.exists(new):
            # 目标已存在：把老文件挪过去，同名跳过，避免覆盖新库里的更新
            for name in sorted(os.listdir(old)):
                src = os.path.join(old, name)
                dst = os.path.join(new, name)
                if os.path.exists(dst):
                    continue
                os.makedirs(new, exist_ok=True)
                shutil.move(src, dst)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.makedirs(os.path.dirname(new), exist_ok=True)
            shutil.move(old, new)
        result["moved_dirs"].append(f"{key} -> {os.path.relpath(new, root)}")
    return result


def ensure_lib(root: str, lib: Optional[str] = LIB_POSE) -> None:
    """补齐该库需要的目录（幂等）。"""
    for path in lib_dirs(root, lib):
        os.makedirs(path, exist_ok=True)


def pool_files(root: str) -> List[str]:
    """素材池里的文件名列表（相对 ``images/``），供删除时判断是否还有引用。"""
    base = images_dir(root)
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            out.append(os.path.relpath(os.path.join(dirpath, fn), base).replace("\\", "/"))
    return sorted(out)


def describe() -> List[dict]:
    """给界面/命令行用的库清单（顺序固定：姿态在前）。"""
    return [{"lib": lib, "name": LIB_NAMES[lib]} for lib in LIBRARIES]


__all__: Sequence[str] = [
    "LIBRARIES", "LIB_NAMES", "LIB_OBJECTS", "LIB_POSE", "batches_path", "backups_dir",
    "coords_dir", "deleted_dir", "describe", "ensure_lib", "exports_dir", "history_root",
    "images_dir", "index_path", "label_path_for", "labels_dir", "legacy_dirs", "lib_dirs",
    "lib_name", "meta_dir", "migrate", "normalize", "objects_meta_dir", "pool_files",
    "qc_dir", "split_legacy_batches", "split_legacy_index",
]
