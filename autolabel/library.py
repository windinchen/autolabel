"""标注库（library）：姿态 / 提示词 / 目标检测各一套索引、批次、备份，互不干扰。

设计口径——**素材池共享，标注库独立**：

::

    <dataset>/
      images/                 共享素材池（同一帧只落盘一次，三个库都能引用）
      labels/                 姿态库标签（YOLO pose 5 列 + 17 关键点）
      prompt_labels/          提示词库标签（YOLO detection 5 列）
      detect_labels/          目标检测库标签（YOLO detection 5 列，类别号来自本库登记表）
      qc/  coords/  exports/  各库产出各写各的文件名，互不覆盖
      .autolabel/
        index.json            姿态库索引（保持原位，老数据集零迁移）
        batches.json          姿态库批次
        originals/ history/   姿态库的人工原件与撤销快照
        fields.json           提示词库的字段登记表
        objects/              提示词库（独立索引 + 独立批次 + 独立备份）
          index.json
          batches.json
          originals/ history/
        detect/               目标检测库（同上，另有一张类别登记表与一份上次选择）
          index.json
          batches.json
          classes.json        类别登记表：类名 -> 永不复用的类别号
          prefs.json          上次用的权重与勾选的类别
        deleted/              各库共用的删除快照（快照里记 lib，知道是从哪个库删的）

为什么要分库：共用一套索引时，「在提示词库里删一张图」会把姿态库的骨骼标注一起带走，
各模块的统计也总是互相掺水。分成独立的库之后每个库只对自己的状态负责，
删图只摘自己的引用；素材文件等所有库都不要了才真删。
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

LIB_POSE = "pose"
LIB_OBJECTS = "objects"
LIB_DETECT = "detect"
LIBRARIES: Tuple[str, ...] = (LIB_POSE, LIB_OBJECTS, LIB_DETECT)
LIB_NAMES = {LIB_POSE: "姿态标注", LIB_OBJECTS: "提示词标注", LIB_DETECT: "目标检测标注"}

#: 每个库一份的路径与字段名。加库只改这张表——以前到处写 ``if lib == objects else pose``，
#: 第三个库进来就得改成三层嵌套，迟早有一处漏掉，漏掉的表现是"标签写进了别人的目录"。
#:   labels      标签目录（相对数据集根）
#:   meta        自有元数据目录（相对 ``.autolabel/``；空串表示直接用 ``.autolabel/``）
#:   prefix      索引条目里本库字段的键名前缀（姿态库沿用老数据集不带前缀的写法，
#:               提示词库历史上叫 ``prompt_*``，前缀错了状态就会串到别的库去）
#:   task        该库吃的 YOLO 任务类型（给模型管理判断兼容性）
#:   unit        界面上一个标注对象叫什么
LIB_SPECS: Dict[str, Dict[str, str]] = {
    LIB_POSE: {"labels": "labels", "meta": "", "prefix": "", "task": "pose", "unit": "骨骼"},
    LIB_OBJECTS: {"labels": "prompt_labels", "meta": LIB_OBJECTS, "prefix": "prompt",
                  "task": "detect", "unit": "框"},
    LIB_DETECT: {"labels": "detect_labels", "meta": LIB_DETECT, "prefix": "detect",
                 "task": "detect", "unit": "框"},
}

#: 命令行与界面里见过的各种写法 → 规范库名
ALIASES = {
    "": LIB_POSE, "pose": LIB_POSE, "skeleton": LIB_POSE, "keypoint": LIB_POSE,
    "objects": LIB_OBJECTS, "object": LIB_OBJECTS, "prompt": LIB_OBJECTS, "obj": LIB_OBJECTS,
    "detect": LIB_DETECT, "det": LIB_DETECT, "detection": LIB_DETECT,
}


def spec(lib: Optional[str] = LIB_POSE) -> Dict[str, str]:
    """该库的目录 / 字段口径。"""
    return LIB_SPECS[normalize(lib)]


def normalize(lib: Optional[str]) -> str:
    """把各种写法（``pose`` / ``objects`` / ``prompt`` / ``det`` / 空）归一到库名。"""
    value = str(lib or LIB_POSE).strip().lower()
    if value not in ALIASES:
        raise ValueError(f"未知标注库: {lib}（可用：{'、'.join(LIBRARIES)}）")
    return ALIASES[value]


def lib_name(lib: Optional[str]) -> str:
    return LIB_NAMES[normalize(lib)]


def lib_task(lib: Optional[str] = LIB_POSE) -> str:
    """该库要什么任务的模型：``pose`` 或 ``detect``。"""
    return spec(lib)["task"]


def field_key(lib: Optional[str] = LIB_POSE, field: str = "status") -> str:
    """该库索引条目里某字段的键名：``field_key("objects", "objects")`` -> ``prompt_objects``。

    姿态库不带前缀（``status`` / ``objects`` / ``flags``），那是老数据集的写法。
    """
    prefix = spec(lib)["prefix"]
    return f"{prefix}_{field}" if prefix else str(field)


def label_field_keys(lib: Optional[str] = LIB_POSE) -> Tuple[str, ...]:
    """该库里**由标签文件派生**的索引字段：标签文件被删掉时这些计数必须一起抹掉。

    留着不删，索引会继续报「这张有 1 个人 · 已标过」，而磁盘上标签早没了——
    门户与 ``status`` 的「人数」立刻开始说谎（``缺标签`` 那一列又说没标过）。
    姿态库历史上把参数指纹写成不带前缀的 ``label_signature``，单独补进来。
    """
    name = normalize(lib)
    # ``scores`` 是框类库的逐框置信度表（按几何签名存），标签没了它也就没人认得了
    fields = ["objects", "score", "scores", "classes", "flags", "object_flags",
              "visible_ratio", "signature", "labeled_at", "error"]
    keys = [field_key(name, item) for item in fields]
    if name == LIB_POSE:
        keys.append("label_signature")
    return tuple(keys)


def status_key(lib: Optional[str] = LIB_POSE) -> str:
    """该库在索引条目里用哪个键存复核状态。"""
    return field_key(lib, "status")


def other_libs(lib: Optional[str] = LIB_POSE) -> Tuple[str, ...]:
    """除本库以外的库（删图时要问它们还要不要这张图）。"""
    name = normalize(lib)
    return tuple(item for item in LIBRARIES if item != name)


def lib_dirs(root: str, lib: Optional[str] = LIB_POSE) -> List[str]:
    """该库需要存在的目录（不含素材池）。"""
    name = normalize(lib)
    dirs = [meta_dir(root), labels_dir(root, name)]
    if name != LIB_POSE:
        dirs.append(meta_dir_for(root, name))
    if name == LIB_POSE:
        dirs.append(qc_dir(root))
    dirs += [backups_dir(root, name), history_root(root, name)]
    return dirs


# --- 基础目录 -------------------------------------------------------------- #

def meta_dir(root: str) -> str:
    return os.path.join(root, ".autolabel")


def meta_dir_for(root: str, lib: Optional[str] = LIB_POSE) -> str:
    """该库自己的元数据目录；姿态库沿用 ``.autolabel/``，老数据集零迁移。"""
    name = normalize(lib)
    sub = spec(name)["meta"]
    return os.path.join(meta_dir(root), sub) if sub else meta_dir(root)


def objects_meta_dir(root: str) -> str:
    """提示词库自己的元数据目录（旧调用名保留）。"""
    return meta_dir_for(root, LIB_OBJECTS)


def images_dir(root: str) -> str:
    """共享素材池：各库的图片都放这里，同一帧只落盘一次。"""
    return os.path.join(root, "images")


def labels_dir(root: str, lib: Optional[str] = LIB_POSE) -> str:
    return os.path.join(root, spec(lib)["labels"])


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
    return os.path.join(meta_dir_for(root, lib), "index.json")


def batches_path(root: str, lib: Optional[str] = LIB_POSE) -> str:
    return os.path.join(meta_dir_for(root, lib), "batches.json")


def backups_dir(root: str, lib: Optional[str] = LIB_POSE) -> str:
    """首次人工修改前的原件备份目录（每库一套，撤销与恢复都靠它）。"""
    return os.path.join(meta_dir_for(root, lib), "originals")


def history_root(root: str, lib: Optional[str] = LIB_POSE) -> str:
    return os.path.join(meta_dir_for(root, lib), "history")


def registry_path(root: str, lib: Optional[str] = LIB_OBJECTS) -> str:
    """类别登记表：提示词库叫 ``fields.json``（沿用旧名，老数据集不迁），
    目标检测库叫 ``classes.json``。两者结构一致：类名 → 永不复用的类别号。"""
    name = normalize(lib)
    if name == LIB_POSE:
        raise ValueError("姿态库按关键点标注，没有类别登记表")
    if name == LIB_OBJECTS:
        # 提示词库的 fields.json 一直在 .autolabel/ 根上，沿用旧位置，老数据集不动它
        return os.path.join(meta_dir(root), "fields.json")
    return os.path.join(meta_dir_for(root, name), "classes.json")


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
    # 延迟导入，避免和 dataset 形成循环
    from .dataset import DatasetIndex, load_index_or_none

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
    pool_keys = ("width", "height", "sha256", "stored_sha256", "bytes", "dhash", "imported_at",
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
        existing = load_index_or_none(root, LIB_OBJECTS)
        if existing is None:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "lib": LIB_OBJECTS, "name": data.get("name", ""),
                           "created_at": data.get("created_at", ""),
                           "updated_at": data.get("updated_at", ""),
                           "label_format": "class cx cy w h（归一化，YOLO detection）",
                           "settings": {}, "items": objects_items},
                          fh, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            # 目标索引已经存在（先跑过一次 ``init``，或上次迁移只做了一半）时得**并**
            # 进去，不能整份跳过：下面照样会把 ``prompt_*`` 从姿态库里剥掉，
            # 于是提示词标注两边都没留，而命令行还报"已迁移 N 条"
            for rel, entry in objects_items.items():
                existing.upsert(rel, **{**entry, **(existing.get(rel) or {})})
            existing.save()
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


def ensure_lib_index(root: str, lib: str) -> bool:
    """补齐缺失的库索引，返回是否新建了文件。

    新加的库在老数据集里从没有过索引，而复核服务的存储层要求每本索引都在，
    缺一个界面就整页起不来——启动时顺手补一本空的比报错友好。
    """
    from .dataset import _init_box_index, load_index_or_none  # 延迟导入避免循环

    name = normalize(lib)
    if name == LIB_POSE or os.path.exists(index_path(root, name)):
        return False
    pose = load_index_or_none(root, LIB_POSE)
    label = (pose.data.get("name") if pose else None) or None
    ensure_lib(root, name)
    _init_box_index(root, name, label)
    return True


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
    for lib in LIBRARIES:
        if ensure_lib_index(root, lib):
            result["created"].append(os.path.relpath(index_path(root, lib), root))
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
    """给界面/命令行用的库清单（顺序固定：姿态 → 提示词 → 目标检测）。"""
    return [{"lib": lib, "name": LIB_NAMES[lib], "task": spec(lib)["task"],
             "unit": spec(lib)["unit"], "labels": spec(lib)["labels"]} for lib in LIBRARIES]


__all__: Sequence[str] = [
    "ALIASES", "LIB_DETECT", "LIB_NAMES", "LIB_OBJECTS", "LIB_POSE", "LIB_SPECS",
    "LIBRARIES", "batches_path", "backups_dir", "coords_dir", "deleted_dir", "describe",
    "ensure_lib", "exports_dir", "history_root", "images_dir", "index_path", "label_path_for",
    "labels_dir", "legacy_dirs", "lib_dirs", "lib_name", "lib_task", "meta_dir",
    "meta_dir_for", "migrate", "normalize", "objects_meta_dir", "other_libs", "pool_files",
    "ensure_lib_index", "field_key", "qc_dir", "registry_path", "spec",
    "split_legacy_batches",
    "split_legacy_index", "status_key",
]
