"""批次分组与清理：一次导入/标注任务记为一个批次，可按批次查看、筛选、删除。

**每个标注库一套批次表**：姿态库 ``.autolabel/batches.json``、提示词库
``.autolabel/objects/batches.json``。批次 ID 全局唯一，所以只给 ID 时能自动定位到
所属库（``resolve_lib``），前端删批次不用多传参数。

删除批次时连带删掉本库的标签与复核历史，并在 ``.autolabel/deleted/`` 留一份快照
（索引字段 + 标签原文，快照里记 ``lib``），误删可以照着快照恢复。
素材图片是**三个库共享的池**：只有所有库都不再引用同一帧时才会真的删文件，
否则只是把本库的引用摘掉（界面上分别叫「从本库移除」和「删素材」）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover - 非 POSIX 平台退化成只有进程内锁
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from .dataset import (
    IMAGE_EXTS,
    STATUS_UNREVIEWED,
    file_sha256,
    images_dir,
    label_path_for,
    load_index,
    load_index_or_none,
    now_str,
    resolve_image,
)
from .atomic import atomic_write_json
from .library import (
    LIBRARIES,
    LIB_DETECT,
    LIB_NAMES,
    LIB_OBJECTS,
    LIB_POSE,
    backups_dir,
    batches_path,
    deleted_dir,
    field_key,
    history_root,
    normalize,
    other_libs,
    qc_dir,
    spec,
)

# 索引里没有 batch_id 的历史数据，归到这个虚拟批次里，同样可以整批删除
UNGROUPED_ID = "__ungrouped__"
UNGROUPED_NAME = "未分组（历史导入）"
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
# 批次归属的二级模块就是它所属的标注库：每个库一批批次，互不掺水
MODULE_POSE = LIB_POSE
MODULE_PROMPT = LIB_OBJECTS
MODULE_DETECT = LIB_DETECT
MODULE_UNKNOWN = "unknown"
MODULE_NAMES = dict(LIB_NAMES)
MODULE_NAMES[MODULE_UNKNOWN] = "未识别"
#: 门户批次表里各模块小计的键名。提示词库历史上叫 ``prompt``，沿用不改，
#: 免得前端和导出脚本跟着一起抖；新库按库名给键。
MODULE_ROW_KEYS = {LIB_POSE: "pose", LIB_OBJECTS: "prompt", LIB_DETECT: "detect"}
# 按张删除的两种粒度
DELETE_LIB = "lib"        # 只摘本库引用，素材文件留在池里
DELETE_PURGE = "purge"    # 三个库一起摘，并把素材文件删掉
DELETE_MODES = (DELETE_LIB, DELETE_PURGE)
_SAFE = re.compile(r"[^0-9A-Za-z._\u4e00-\u9fff-]+")
_LOCK = threading.RLock()

_HELD = threading.local()


@contextmanager
def _file_lock(root: str) -> Iterator[None]:
    """批次表的跨进程可重入锁（三个库共用一把，避免互相踩写）。

    ``_LOCK`` 只挡得住同一个进程里的线程；命令行 ``autolabel import`` 和复核服务
    是两个进程，各自「读全表 → 改 → 写全表」，撞上就会把对方刚写的批次记录覆盖掉。
    这里用 ``.autolabel/batches.lock`` 上的 flock 串行化，进锁后一律重新从磁盘载入。

    必须可重入：``reconcile_with_disk()`` 握锁期间还会调 ``attach_items()``，而 flock
    是按打开的文件描述符判定归属的，同线程再开一个描述符等锁会直接把自己锁死。
    """
    path = os.path.join(os.path.abspath(root), ".autolabel", "batches.lock")
    held = getattr(_HELD, "paths", None)
    if held is None:
        held = _HELD.paths = set()
    if path in held:  # 本线程已经握着这把锁，直接放行
        yield
        return
    handle = None
    if fcntl is not None:  # pragma: no branch
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handle = open(path, "a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError:  # pragma: no cover - 只读目录等极端情况，宁可少一层保护
            if handle is not None:
                handle.close()
            handle = None
    held.add(path)
    try:
        with _LOCK:
            yield
    finally:
        held.discard(path)
        if handle is not None:  # pragma: no cover
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _other(lib: str) -> str:
    """老调用名：返回"排在第一位的其它库"。三个库之后别再用它做判断，
    判断素材还有没有人引用要看 ``other_libs``（见 ``_holder_count``）。"""
    peers = other_libs(lib)
    return peers[0] if peers else normalize(lib)


def _peers(root: str, lib: str) -> Dict[str, object]:
    """其它库的索引副本（读不到的库直接不给，调用方按"没人引用"处理）。"""
    peers: Dict[str, object] = {}
    for other in other_libs(lib):
        idx = load_index_or_none(root, other)
        if idx is not None:
            peers[other] = idx
    return peers


# --------------------------------------------------------------------------- #
# 读写批次表
# --------------------------------------------------------------------------- #

def load_batches(root: str, lib: str = LIB_POSE) -> List[dict]:
    """读某个库的批次登记表；文件缺失或损坏时按空表处理。"""
    path = batches_path(root, lib)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("batches") if isinstance(data, dict) else data
    return [b for b in (items or []) if isinstance(b, dict) and b.get("id")]


def save_batches(root: str, batches: Sequence[dict], lib: str = LIB_POSE) -> None:
    atomic_write_json(batches_path(root, lib),
                      {"batches": list(batches), "updated_at": now_str()})


def module_for(record: dict) -> str:
    """批次属于哪个二级模块（= 所属标注库）。

    优先读显式写入的 ``module``/``lib``；历史批次可能写的是老取值 ``prompt``，
    再兜底按 ``kind`` 推导（``image``/``video`` 是姿态库，``prompt*`` 是提示词库）。
    """
    for key in ("module", "lib"):
        value = str(record.get(key) or "").strip()
        if value == "prompt":
            return MODULE_PROMPT
        if value in MODULE_NAMES and value != MODULE_UNKNOWN:
            try:
                return normalize(value)
            except ValueError:
                continue
    kind = str(record.get("kind") or "")
    if kind.startswith("detect"):
        return MODULE_DETECT
    if kind.startswith("prompt") or kind.startswith("object"):
        return MODULE_PROMPT
    if kind in ("image", "video", "pose", "pool"):
        return MODULE_POSE
    return MODULE_UNKNOWN


def resolve_lib(root: str, batch_id: str) -> str:
    """按批次 ID 定位它属于哪个库；找不到时退回姿态库。"""
    batch_id = str(batch_id or "")
    for lib in LIBRARIES:
        if any(str(row.get("id")) == batch_id for row in load_batches(root, lib)):
            return lib
    return LIB_POSE


def create_batch(root: str, *, name: str, kind: str = "image", source: str = "cli",
                 params: Optional[dict] = None, uploads: Optional[Sequence[str]] = None,
                 module: str = "", lib: str = "") -> dict:
    """在指定库里登记一个新批次，状态先记 running，等流水线结束时再收尾。

    没显式给 ``lib`` 时按 ``module``/``kind`` 推导（``prompt*`` 归提示词库），
    推不出来才落到姿态库。
    """
    if lib:
        lib = normalize(lib)
    else:
        derived = module_for({"kind": kind, "module": module})
        lib = derived if derived in LIBRARIES else LIB_POSE
    batch = {
        "id": f"b_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}",
        "name": name, "kind": kind, "source": source,
        "module": lib, "lib": lib,
        "created_at": now_str(), "finished_at": None, "status": STATUS_RUNNING,
        "params": {k: v for k, v in (params or {}).items() if v is not None},
        "uploads": [str(u) for u in (uploads or [])][:30],
        "items": [], "message": "",
    }
    with _file_lock(root):
        batches = load_batches(root, lib)
        batches.append(batch)
        save_batches(root, batches[-300:], lib)
    return batch


def attach_items(root: str, batch_id: str, items: Sequence[str], lib: str = "") -> int:
    """把导入成功的图片挂到批次上：本库索引写 ``batch_id``，登记表补 ``items``。"""
    items = [str(rel).replace("\\", "/") for rel in items if rel]
    if not items:
        return 0
    with _file_lock(root):
        lib = normalize(lib) if lib else resolve_lib(root, batch_id)
        idx = load_index(root, lib)
        for rel in items:
            idx.upsert(rel, batch_id=batch_id)
        idx.save()
        batches = load_batches(root, lib)
        batch = next((b for b in batches if b.get("id") == batch_id), None)
        if batch is not None:
            known = set(batch.get("items", []))
            batch.setdefault("items", []).extend(rel for rel in items if rel not in known)
            save_batches(root, batches, lib)
    return len(items)


def finish_batch(root: str, batch_id: str, status: str = "done", message: str = "",
                 lib: str = "") -> None:
    with _file_lock(root):
        lib = normalize(lib) if lib else resolve_lib(root, batch_id)
        batches = load_batches(root, lib)
        batch = next((b for b in batches if b.get("id") == batch_id), None)
        if batch is None:
            return
        batch["status"] = status
        batch["finished_at"] = now_str()
        if message:
            batch["message"] = message
        save_batches(root, batches, lib)


# --------------------------------------------------------------------------- #
# 启动对账
# --------------------------------------------------------------------------- #

def reconcile_with_disk(root: str) -> dict:
    """启动对账（三个库都跑）：补齐已落盘但索引缺席的图，并给中断批次收尾。

    导入/标注跑到一半被打断（重启复核服务、断电、强杀进程）时，图片副本往往已经
    写进共享素材池 ``images/`` 而索引还没来得及保存，界面上看着像素材凭空消失，
    批次也会一直挂在"标注中"删不掉。这里以磁盘为准补索引，并把无人认领的 running
    批次判定为中断。

    素材池几个库共用，所以「孤儿」的定义是**所有库的索引都不要这张图**；
    补进哪个库看当时还有没有正在跑的批次，都没有就归到姿态库的未分组。

    :return: ``{"recovered": 补回张数, "items": 补回的相对路径样本,
              "interrupted": 判定中断的批次 ID, "missing_files": 索引有记录但原图没了的张数,
              "by_lib": 每个库的同类统计}``
    """
    root = os.path.abspath(root)
    report = {"recovered": 0, "items": [], "interrupted": [], "missing_files": 0, "by_lib": {}}
    missing: set = set()
    with _file_lock(root):
        tombstones = _kept_file_tombstones(root)
        running_by_lib = {lib: [row for row in load_batches(root, lib)
                                if row.get("status") == STATUS_RUNNING] for lib in LIBRARIES}
        for lib in LIBRARIES:
            others = [row for other in LIBRARIES if other != lib for row in running_by_lib[other]]
            one = _reconcile_lib(root, lib, tombstones, others)
            report["by_lib"][lib] = one
            report["recovered"] += one["recovered"]
            missing.update(one["missing"])
            report["items"].extend(one["items"])
            report["interrupted"].extend(one["interrupted"])
        report["items"] = report["items"][:20]
        # 同一帧可能被几个库都记着，只缺一块盘，按张数说人才不会被数字吓到
        report["missing_files"] = len(missing)
    return report


def _reconcile_lib(root: str, lib: str, tombstones: set,
                   other_running: Sequence[dict] = ()) -> dict:
    one = {"recovered": 0, "items": [], "interrupted": [], "missing": []}
    idx = load_index_or_none(root, lib)
    if idx is None:
        return one
    base = images_dir(root)
    referenced: List[str] = []
    for dirpath, _dirs, names in os.walk(base):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), base).replace("\\", "/")
            if rel in tombstones:
                continue
            referenced.append(rel)
    # 对面库的索引读一次就够：以前每扫到一张图就重新解析一遍它的 index.json，
    # 1500 张的库光启动对账就要 1 秒、两万张要几分钟（实测是 O(N^2)）。
    other_indexes = [load_index_or_none(root, other) or _EmptyIndex()
                     for other in other_libs(lib)]
    orphans = [rel for rel in referenced
               if idx.get(rel) is None and all(o.get(rel) is None for o in other_indexes)]
    one["missing"] = [rel for rel in idx.items if not os.path.exists(resolve_image(root, rel))]
    batches = load_batches(root, lib)
    running = [row for row in batches if row.get("status") == STATUS_RUNNING]
    recovered = []
    if orphans and _should_claim(lib, running, other_running):
        for rel in orphans:
            fields = _recovered_fields(root, rel)
            if fields is None:
                continue
            idx.upsert(rel, **fields)
            recovered.append(rel)
    if recovered:
        idx.save()  # 先落盘，attach_items 会重新读索引再挂 batch_id
        if running:
            newest = max(running, key=lambda row: str(row.get("created_at") or ""))
            attach_items(root, str(newest["id"]), recovered, lib=lib)
        one["recovered"] = len(recovered)
        one["items"] = recovered[:20]
    for record in running:
        record["status"] = STATUS_INTERRUPTED
        record["finished_at"] = record.get("finished_at") or now_str()
        record["message"] = "复核服务重启时任务仍在进行，已判定为中断；已入库的图片与标注保留"
        one["interrupted"].append(str(record.get("id")))
    if running:
        save_batches(root, batches, lib)
    return one


class _EmptyIndex:
    """另一个库还没建索引时的空替身，只为统一 ``.get()`` 调用。"""

    @staticmethod
    def get(_rel: str):
        return None


def _should_claim(lib: str, running: Sequence[dict],
                  other_running: Sequence[dict] = ()) -> bool:
    """孤儿图归谁：有在跑批次的库优先，都没有才归姿态库（姿态库是这套工具的老房东）。

    ``other_running`` 是对面库里还在跑的批次：本库闲着、对面正在标这批素材时，
    先让对面认领，否则姿态库会按遍历顺序把提示词库跑了一半的图抢走。
    """
    if running:
        return True
    if other_running:
        return False
    return lib == LIB_POSE


def _kept_file_tombstones(root: str) -> set:
    """删除时勾了"保留图片文件"的文件名：它们留在磁盘上是刻意的，不算孤儿。"""
    folder = deleted_dir(root)
    if not os.path.isdir(folder):
        return set()
    kept: set = set()
    names = sorted((name for name in os.listdir(folder) if name.endswith(".json")), reverse=True)
    for name in names[:20]:  # 快照可能攒很多，只看最近 20 份足够覆盖上一次会话
        try:
            with open(os.path.join(folder, name), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("keep_files") or data.get("mode") == DELETE_LIB:
            kept.update(str(rel).replace("\\", "/") for rel in (data.get("items") or {}))
    return kept


def _recovered_fields(root: str, image_rel: str, lib: str = LIB_POSE) -> Optional[dict]:
    """从磁盘上的图片副本反推索引条目；读不出来的半截坏图直接跳过。"""
    from PIL import Image

    from .dataset import _entry_defaults

    path = resolve_image(root, image_rel)
    try:
        with Image.open(path) as im:
            width, height = im.size
            im.load()
        fields = dict(width=int(width), height=int(height), sha256=file_sha256(path),
                      bytes=os.path.getsize(path), imported_at=now_str(),
                      source=os.path.abspath(path), recovered_at=now_str())
        fields.update(_entry_defaults(lib))
        return fields
    except Exception:  # noqa: BLE001 坏图不能拖垮整个启动流程
        return None


# --------------------------------------------------------------------------- #
# 汇总与列表
# --------------------------------------------------------------------------- #

#: 补齐时每个库要搬过去的字段：本库自己那份永远为准，别库的只补缺。
#: 姿态库额外带 ``visible_ratio`` 与历史遗留的 ``object_flags``。
_EXTRA_FIELDS = ("score", "visible_ratio", "object_flags", "manual", "manual_boxes", "signature")


def _lib_fields(lib: str) -> Tuple[str, ...]:
    """该库在索引条目里占用的键名（跨库补齐时用）。"""
    name = normalize(lib)
    keys = [field_key(name, item) for item in ("status", "objects", "flags", "score")]
    keys += [item for item in _EXTRA_FIELDS
             if item in ("visible_ratio", "object_flags") and name == LIB_POSE
             or item in ("manual", "manual_boxes", "signature") and name != LIB_POSE]
    return tuple(keys)


def _cross_view(item: dict, peers: Optional[Dict[str, dict]], lib: str) -> dict:
    """把本库条目和其它库对同一条素材的条目合成"这条素材的完整进度"。

    门户一行里并排显示骨骼 / 提示词框 / 检测框三列，只读本库字段就会出现
    「文件清单写着已标 1 人，批次行却写 0 人」的自相矛盾。每一列都以负责它的
    那个库为准，本库读不到就从对应库的条目里补。
    """
    if not peers:
        return item
    merged = dict(item)
    for other in LIBRARIES:
        src = item if other == normalize(lib) else (peers.get(other) or {})
        for key in _lib_fields(other):
            merged[key] = src.get(key)
    return merged


#: 复核状态四种，每个库各计一份
STATUS_TALLY = ("approved", "rejected", "edited", "unreviewed")


def _unit_key(lib: str) -> str:
    """该库的"标了多少个东西"叫什么：姿态数骨骼，框类库数框。"""
    return "people" if spec(lib)["unit"] == "骨骼" else "boxes"


def _tally_key(lib: str, name: str) -> str:
    """汇总结果里该库某一项的键名：姿态库不带前缀（老键名），其余带库前缀。"""
    prefix = spec(lib)["prefix"]
    return f"{prefix}_{name}" if prefix else name


def _summarize_lib(items: Sequence[dict], lib: str) -> Dict[str, int]:
    name = normalize(lib)
    unit = _unit_key(name)
    tally = {_tally_key(name, key): 0 for key in ("labeled", unit, "flagged") + STATUS_TALLY}
    for item in items:
        objects = item.get(field_key(name, "objects"))
        status = str(item.get(field_key(name, "status")) or "")
        if objects is None and not status:
            continue                        # 这条素材在本库里还没开始
        if objects is not None:
            tally[_tally_key(name, "labeled")] += 1
            tally[_tally_key(name, unit)] += int(objects or 0)
        flags = set(item.get(field_key(name, "flags")) or [])
        if name == LIB_POSE:
            flags |= set(item.get("object_flags") or [])
        if flags:
            tally[_tally_key(name, "flagged")] += 1
        if status in STATUS_TALLY:
            tally[_tally_key(name, status)] += 1
        else:
            tally[_tally_key(name, "unreviewed")] += 1
    return tally


def summarize(items: Sequence[dict]) -> dict:
    """按索引条目汇总批次规模（不解析标签）。

    每个库的条目只带自己那一组键（姿态库不带前缀、提示词库 ``prompt_*``、
    目标检测库 ``detect_*``），这里逐库套同一套算法，加库不用改这里。
    """
    counts: Dict[str, int] = {"items": len(items)}
    for lib in LIBRARIES:
        counts.update(_summarize_lib(items, lib))
    return counts


def list_batches(root: str, lib: str = "") -> List[dict]:
    """批次列表（新→旧），带实时统计；``lib`` 为空时合并三个库。

    合并时每个库都跳过「登记表的 ID 已被另一个库认领」的情况，避免同一素材在
    几个库互相引用批次时出现重复行。未挂批次的数据归入各自库的「未分组」。
    """
    root = os.path.abspath(root)
    wanted = [normalize(lib)] if lib else list(LIBRARIES)
    merged = not lib
    rows: List[dict] = []
    for target in wanted:
        idx = load_index_or_none(root, target)
        peer_indexes = _peers(root, target)
        # 合并列出时，某个库索引里引用了「其实属于另一个库」的批次 ID 就不再补虚行，
        # 否则主页会出现两条同名批次
        others = {str(row.get("id")) for other in LIBRARIES if other != target
                  for row in load_batches(root, other)} if merged else set()
        members: Dict[str, List[dict]] = {}
        ungrouped: List[dict] = []
        if idx is not None:
            for rel, item in idx.ordered_items():
                view = _cross_view(
                    item, {own: (idx2.get(rel) if idx2 is not None else None)
                           for own, idx2 in peer_indexes.items()}, target)
                batch_id = str(item.get("batch_id") or "")
                if batch_id:
                    members.setdefault(batch_id, []).append(view)
                else:
                    ungrouped.append(view)
        seen = set()
        for batch in load_batches(root, target):
            batch_id = str(batch.get("id"))
            seen.add(batch_id)
            rows.append(_row(batch, members.pop(batch_id, []), lib=target))
        for batch_id, group in members.items():  # 登记表被手工删过，但索引里还有引用
            if batch_id in others:
                continue
            rows.append(_row({"id": batch_id, "name": f"批次 {batch_id}（记录缺失）",
                              "kind": "unknown", "source": "unknown", "status": "-",
                              "created_at": group[0].get("imported_at")}, group, lib=target))
        if ungrouped:
            rows.append(_row({"id": f"{UNGROUPED_ID}:{target}", "name": UNGROUPED_NAME,
                              "kind": "mixed", "source": "-", "status": "-",
                              "created_at": min((i.get("imported_at") or "" for i in ungrouped),
                                                default="")}, ungrouped,
                             virtual=True, lib=target))
    rows.sort(key=lambda r: (r.get("created_at") or "", r.get("id") or ""), reverse=True)
    return rows


def batch_names(root: str) -> Dict[str, str]:
    """``batch_id -> 批次名``，给界面显示用。"""
    return {row["id"]: row["name"] for row in list_batches(root)}


def batch_items(root: str, batch_id: str, lib: str = "") -> List[str]:
    """某批次当前仍在该库索引里的图片相对路径。"""
    if str(batch_id).startswith(f"{UNGROUPED_ID}:"):
        # 未分组行的 ID 形如 __ungrouped__:objects，后半段才是库名
        batch_id, lib = str(batch_id).split(":", 1)
    lib = normalize(lib) if lib else resolve_lib(root, batch_id)
    idx = load_index_or_none(root, lib)
    if idx is None:
        return []
    if batch_id == UNGROUPED_ID:
        return [rel for rel, item in idx.ordered_items() if not item.get("batch_id")]
    return [rel for rel, item in idx.ordered_items() if item.get("batch_id") == batch_id]


def batch_file_rows(root: str, batch_id: str, lib: str = "") -> List[dict]:
    """某批次下的单张清单，给门户「展开文件」和按张删除用（新→旧按入库时间）。"""
    lib = normalize(lib) if lib else resolve_lib(root, batch_id)
    idx = load_index_or_none(root, lib)
    peers = _peers(root, lib)
    rows: List[dict] = []
    for rel in batch_items(root, batch_id, lib):
        item = idx.get(rel) if idx else None
        if item is None:
            continue
        # 每一列都以负责它的那个库为准：本库就用自己的条目，别库去别库的索引里取
        views = {own: (item if own == lib else ((peers[own].get(rel) if own in peers else None) or {}))
                 for own in LIBRARIES}
        flags = sorted(set(list(views[LIB_POSE].get("flags") or [])
                           + list(views[LIB_POSE].get("object_flags") or [])))
        row = {
            "image": rel, "name": os.path.basename(rel), "lib": lib,
            "bytes": int(item.get("bytes") or 0), "imported_at": item.get("imported_at"),
            "width": item.get("width"), "height": item.get("height"),
            "flags": flags,
            # 其它库是否也引用了这张图：决定"删素材"会不会影响对面
            "shared": any(other is not None and other.get(rel) is not None
                          for other in peers.values()),
            "video": os.path.basename(str(item.get("video"))) if item.get("video") else "",
        }
        for own in LIBRARIES:
            row[field_key(own, "status")] = views[own].get(field_key(own, "status")) or (
                STATUS_UNREVIEWED if own == lib else "")
            row[field_key(own, "objects")] = views[own].get(field_key(own, "objects"))
            if own != LIB_POSE:
                row[field_key(own, "flags")] = sorted(set(views[own].get(field_key(own, "flags")) or []))
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("imported_at") or ""), reverse=True)
    return rows


def _index_key(idx, rel: str) -> str:
    """把 ``images/a.jpg`` 这类带目录的写法归一回索引键（``a.jpg``）。"""
    if idx.get(rel) is not None:
        return rel
    if rel.startswith("images/"):
        bare = rel[len("images/"):]
        if idx.get(bare) is not None:
            return bare
    return rel


# --------------------------------------------------------------------------- #
# 删除
# --------------------------------------------------------------------------- #

def delete_images(root: str, image_rels: Sequence[str], *, lib: str = LIB_POSE,
                  mode: str = DELETE_LIB, keep_files: Optional[bool] = None,
                  regenerate: bool = True,
                  active_ids: Optional[Iterable[str]] = None) -> dict:
    """按单张删除图片与标注，批次记录保留，只把这些图从批次 ``items`` 里摘掉。

    与 :func:`delete_batches` 的区别是粒度——门户「文件管理」里勾掉几张传错的图，
    不该连带丢掉整批。删除前同样在 ``.autolabel/deleted/`` 写快照（含各库标签原文）。

    :param lib: 从哪个库删（``pose`` / ``objects`` / ``detect``），给 ``all`` 表示三个库的引用一起摘
    :param mode: ``lib`` 只摘引用、素材留在池里；``purge`` 三个库一起摘并删素材文件
    :param keep_files: 仅 ``purge`` 有效，保留 ``images/`` 下的原图只删记录
    :param active_ids: 仍在跑任务的批次 ID；这些批次里的图默认拒绝删除
    """
    root = os.path.abspath(root)
    # ``batches`` 的 ``--lib`` 默认值就是 all。以前它一路 ``normalize("all")`` 之前
    # 被 CLI 换成空串、再落到 pose——「三个库都摘掉这张图」于是变成只摘了姿态，
    # 还回一句「仅姿态标注」，检测库和提示词库里那张图照旧躺着。
    all_libs = str(lib or "").strip().lower() in ("", "all", "*")
    lib = LIB_POSE if all_libs else normalize(lib)
    mode = str(mode or DELETE_LIB).strip().lower()
    if mode not in DELETE_MODES:
        raise ValueError(f"未知删除方式: {mode}（可用：{'、'.join(DELETE_MODES)}）")
    if keep_files is None:
        keep_files = mode == DELETE_LIB
    rels = [str(rel).replace("\\", "/") for rel in (image_rels or []) if str(rel).strip()]
    if not rels:
        raise ValueError("没有指定要删除的图片")
    live = None if active_ids is None else {str(b) for b in active_ids}
    targets = list(LIBRARIES) if (mode == DELETE_PURGE or all_libs) else [lib]
    with _file_lock(root):
        snapshots: Dict[str, dict] = {}
        chosen: Dict[str, List[str]] = {}
        missing: List[str] = []
        busy: List[str] = []
        absent: set = set()
        found: set = set()
        owner_before: Dict[str, Dict[str, str]] = {}
        removed_files = 0
        for target in targets:
            idx = load_index_or_none(root, target)
            if idx is None:
                continue
            keys = [_index_key(idx, rel) for rel in rels]
            for rel in keys:
                if idx.get(rel) is None:
                    absent.add(rel)
                    if target == lib and not all_libs:
                        missing.append(rel)
                    continue
                found.add(rel)
                if live and str(idx.items[rel].get("batch_id") or "") in live:
                    busy.append(rel)
                    continue
                chosen.setdefault(target, []).append(rel)
            owner_before[target] = {rel: str(idx.items[rel].get("batch_id") or "")
                                    for rel in chosen.get(target, [])}
        if all_libs:
            # 三个库都翻过了还没这张图，才算"不在索引里"
            missing = sorted(absent - found)
        if busy:
            raise ValueError(f"{len(set(busy))} 张图片正在标注中（批次还在跑任务），"
                             f"先在主页点「清除任务」再删，例如 {os.path.basename(busy[0])}")
        if not any(chosen.values()):
            raise ValueError("这些图片不在索引里：" + "、".join(sorted(set(missing))[:5]))

        snapshot: dict = {"deleted_at": now_str(), "keep_files": keep_files, "mode": mode,
                          "lib": lib, "batches": [], "items": {}}
        for target in targets:
            idx = load_index_or_none(root, target)
            if idx is None:
                continue
            for rel in chosen.get(target, []):
                entry = _item_snapshot(root, rel, idx.items[rel], target)
                snapshots[rel] = _merge_snapshot(snapshots[rel], entry) \
                    if rel in snapshots else entry
                removed_files += _remove_lib_files(root, target, rel)
                idx.remove(rel)
            _drop_from_batches(root, target, owner_before.get(target, {}), snapshot, idx)
            idx.save()
        # 素材文件：只有所有库都不再引用（且这次是 purge）才真删
        if mode == DELETE_PURGE and not keep_files:
            for rel in sorted({rel for group in chosen.values() for rel in group}):
                removed_files += _maybe_remove_pool(root, lib, rel)
        snapshot["items"] = snapshots
        snapshot_path = _write_snapshot(root, snapshot) if snapshots else ""

    result = {"items": len(snapshots),
              # 同一张素材可能从几个库各摘一次，removals 记"库×图"的粒度
              "removals": sum(len(group) for group in chosen.values()),
              "files_removed": removed_files,
              "snapshot": snapshot_path, "keep_files": keep_files, "mode": mode, "lib": lib,
              "libs": [t for t in targets if chosen.get(t)],
              "missing": sorted(set(missing))[:10],
              "batches_updated": sorted({str(b) for b in snapshot.get("batches", [])
                                         if isinstance(b, dict)})}
    result["batches_emptied"] = [b["id"] for b in snapshot.get("batches", [])
                                 if isinstance(b, dict) and b.get("emptied")]
    if regenerate and snapshots:
        result.update(_regenerate(root))
    return result


def _drop_from_batches(root: str, lib: str, owners: Dict[str, str], snapshot: dict,
                       idx) -> None:
    """把删掉的图从本库批次的 ``items`` 里摘掉，删空且无残留的批次顺手清理。

    :param owners: ``图片 -> 所属批次 ID``，必须在改索引**之前**采集（改完就查不到了）
    :param idx: 已经摘掉这些条目的**内存索引**，判空要以它为准
    """
    if not owners:
        return
    batches = load_batches(root, lib)
    gone_set = {str(rel) for rel in owners}
    # 批次还剩几张以索引为准：登记表里的 items 可能因为导入中断而偏少，
    # 只按 items 判空会把还有图的批次误删（门户里表现为整批凭空消失）
    left: Dict[str, int] = {}
    for _rel, item in (idx.ordered_items() if idx is not None else []):
        owner = str(item.get("batch_id") or "")
        if owner:
            left[owner] = left.get(owner, 0) + 1
    touched = {owner for owner in owners.values() if owner}
    emptied_ids = []
    for record in batches:
        batch_id = str(record.get("id"))
        if batch_id not in touched:
            continue  # 这次没动到这个批次，别顺手改写它
        members = [str(rel).replace("\\", "/") for rel in (record.get("items") or [])]
        record["items"] = [rel for rel in members if rel not in gone_set]
        emptied = not record["items"] and not left.get(batch_id)
        snapshot["batches"].append({"id": batch_id, "name": record.get("name"),
                                    "emptied": emptied})
        if emptied:
            emptied_ids.append(batch_id)
    if emptied_ids:
        keep = set(emptied_ids)
        save_batches(root, [b for b in batches if str(b.get("id")) not in keep], lib)
    else:
        save_batches(root, batches, lib)


def delete_batches(root: str, batch_ids: Sequence[str], *, keep_files: bool = False,
                   regenerate: bool = True, active_ids: Optional[Iterable[str]] = None,
                   force: bool = False) -> dict:
    """删除批次对应的图片、标注与记录，并写一份可恢复的快照。

    只给批次 ID 就够了：库由 :func:`resolve_lib` 自动判定。素材图片只有在
    另一个库不再引用时才会删，否则只摘本库引用。

    :param keep_files: 只删索引条目与批次记录，保留素材文件
    :param regenerate: 删除后重算坐标文件与 QC 报告（若原本已生成过）
    :param active_ids: 当前确有任务在跑的批次 ID；传 None 表示无从得知（命令行），
                       此时凡是 running 都按“仍在标注”拒绝
    :param force: 忽略“仍在标注”的保护，服务中断留下的僵尸批次用这个
    """
    root = os.path.abspath(root)
    ids = [str(b) for b in batch_ids if str(b)]
    if not ids:
        raise ValueError("没有指定要删除的批次")
    live = None if active_ids is None else {str(b) for b in active_ids}
    # 「未分组」在每个库里各有一个桶，不带库名的写法等价于三个库一起清
    expanded: List[str] = []
    for batch_id in ids:
        if batch_id == UNGROUPED_ID:
            expanded.extend(f"{UNGROUPED_ID}:{lib}" for lib in LIBRARIES)
        else:
            expanded.append(batch_id)
    with _file_lock(root):
        plan: List[Tuple[str, dict, List[str]]] = []
        registry: Dict[str, List[dict]] = {}
        for batch_id in expanded:
            real_id = batch_id
            if batch_id.startswith(f"{UNGROUPED_ID}:"):
                real_id, lib = batch_id.split(":", 1)
            else:
                lib = resolve_lib(root, batch_id)
            registry.setdefault(lib, load_batches(root, lib))
            if real_id == UNGROUPED_ID:
                record = {"id": UNGROUPED_ID, "name": UNGROUPED_NAME, "kind": "mixed",
                          "source": "unknown", "status": "unknown", "module": lib}
            else:
                record = next((b for b in registry[lib] if str(b.get("id")) == batch_id), None)
                if record is None:
                    raise ValueError(f"批次不存在: {batch_id}")
                running = record.get("status") == STATUS_RUNNING
                if running and not force and (live is None or batch_id in live):
                    raise ValueError(f"批次「{record.get('name')}」仍在标注中，请先点「清除任务」，"
                                     f"或加 --force 强删")
                registry[lib] = [b for b in registry[lib] if str(b.get("id")) != batch_id]
            plan.append((lib, record, batch_items(root, batch_id, lib)))

        snapshot: dict = {"deleted_at": now_str(), "keep_files": keep_files,
                          "mode": "batch", "lib": plan[0][0] if plan else LIB_POSE,
                          "batches": [], "items": {}}
        removed_files = 0
        removals = 0
        for lib, record, rels in plan:
            idx = load_index_or_none(root, lib)
            if idx is None:
                continue
            kept = []
            for rel in rels:
                item = idx.items.get(rel)
                if item is None:
                    continue
                entry = _item_snapshot(root, rel, item, lib)
                snapshot["items"][rel] = (_merge_snapshot(snapshot["items"][rel], entry)
                                          if rel in snapshot["items"] else entry)
                kept.append(rel)
                removals += 1
                removed_files += _remove_lib_files(root, lib, rel)
                if not keep_files:
                    removed_files += _maybe_remove_pool(root, lib, rel)
                idx.remove(rel)
            snapshot["batches"].append({**record, "lib": lib, "items": kept})
            idx.save()
        snapshot["batches"] = [row for row in snapshot["batches"] if row.get("items")
                               or str(row.get("id")) != UNGROUPED_ID]
        for lib, rows in registry.items():
            save_batches(root, rows, lib)
        snapshot_path = _write_snapshot(root, snapshot) if snapshot["items"] else ""

    result = {"batches": sorted({str(record["id"]) for _lib, record, _rels in plan}),
              "items": len(snapshot["items"]), "removals": removals,
              "files_removed": removed_files,
              "snapshot": snapshot_path, "keep_files": keep_files,
              "libs": sorted({lib for lib, _r, _rels in plan})}
    if regenerate and result["items"]:
        result.update(_regenerate(root))
    return result


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #

def module_counts(counts: Dict[str, int], lib: str) -> dict:
    """从汇总结果里挑出某个库的进度小计（门户批次表一行里并排显示用）。"""
    name = normalize(lib)
    unit = _unit_key(name)
    picked = {key: int(counts.get(_tally_key(name, key)) or 0)
              for key in ("labeled", unit, "flagged") + STATUS_TALLY}
    return {unit: picked[unit], "labeled": picked["labeled"],
            "reviewed": sum(picked[key] for key in ("approved", "rejected", "edited")),
            "unreviewed": picked["unreviewed"], "flagged": picked["flagged"]}


def _row(record: dict, items: Sequence[dict], *, virtual: bool = False,
         lib: str = LIB_POSE) -> dict:
    files = [display_name(u) for u in record.get("uploads", [])][:6]
    counts = summarize(items)
    module = module_for({**record, "lib": lib})
    return {
        "id": str(record.get("id")), "name": record.get("name") or str(record.get("id")),
        "kind": record.get("kind", "unknown"), "source": record.get("source", "unknown"),
        "lib": lib, "module": module, "module_name": MODULE_NAMES.get(module, module),
        # 每个模块各自的进度小计：pose / prompt / detect
        **{MODULE_ROW_KEYS[item]: module_counts(counts, item) for item in LIBRARIES},
        "status": record.get("status", "unknown"), "created_at": record.get("created_at"),
        "finished_at": record.get("finished_at"), "message": record.get("message", ""),
        "params": record.get("params", {}), "files": files, "file_count": len(record.get("uploads", [])),
        "virtual": virtual, **counts,
    }


def display_name(stored: str) -> str:
    """去掉上传时加的 8 位哈希前缀，还原成人看得懂的文件名。"""
    name = os.path.basename(str(stored))
    return re.sub(r"^[0-9a-f]{8}_", "", name)


def batch_name(kind: str, names: Sequence[str], override: str = "") -> str:
    """给批次起个可读名字：优先用用户填的，否则按来源文件名生成。"""
    if override:
        return str(override)[:80]
    names = [str(n) for n in names if n]
    label = {"video": "视频", "prompt": "提示词标注", "prompt_upload": "提示词素材",
             "detect": "目标检测", "detect_upload": "检测素材",
             "pool": "素材池"}.get(kind, "图片")
    count = len(names) or 1
    head = names[0][:40] if names else "未命名"
    return f"{label} {count} 个 · {head} 等" if len(names) > 1 else f"{label} {count} 个 · {head}"


def _item_snapshot(root: str, rel: str, item: dict, lib: str = LIB_POSE) -> dict:
    """删除前的可恢复记录：各库的标签原文都留着，事后能分开找回。"""
    entry = dict(item)
    entry["lib"] = lib
    entry["label_lines"] = _read_text(label_path_for(root, rel, lib))
    for name in LIBRARIES:
        entry[f"{name}_label_lines"] = _read_text(label_path_for(root, rel, name))
    return entry


def _merge_snapshot(existing: dict, new: dict) -> dict:
    """同一张图被几个库同批删除时合并记录，标签原文以非空的那份为准。"""
    merged = {**existing, **new}
    for key in tuple(f"{name}_label_lines" for name in LIBRARIES) + ("label_lines",):
        merged[key] = existing.get(key) or new.get(key) or []
    return merged


def _read_text(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def _remove_lib_files(root: str, lib: str, rel: str) -> int:
    """删掉一张图在**某个库**里的痕迹：标签、人工原件备份、撤销历史。"""
    stem = os.path.splitext(rel)[0]
    removed = 0
    targets = [label_path_for(root, rel, lib),
               os.path.join(backups_dir(root, lib), stem + ".txt")]
    if lib == LIB_POSE:
        targets.append(os.path.join(qc_dir(root), "overlays", stem + ".jpg"))
    for path in targets:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    history = os.path.join(history_root(root, lib), stem)
    if os.path.isdir(history):
        shutil.rmtree(history, ignore_errors=True)
        removed += 1
    return removed


def _remove_pool_files(root: str, rel: str) -> int:
    """删掉共享素材池里的图片本体（只在所有库都不要时调用）。"""
    try:
        os.remove(resolve_image(root, rel))
        return 1
    except (OSError, ValueError):
        return 0


def _maybe_remove_pool(root: str, lib: str, rel: str) -> int:
    """还有别的库引用这张图就只摘本库引用，不删文件。"""
    for other in _peers(root, lib).values():
        if other.get(rel) is not None:  # type: ignore[union-attr]
            return 0
    removed = _remove_pool_files(root, rel)
    if removed:
        stem = os.path.splitext(rel)[0]
        try:
            os.remove(os.path.join(qc_dir(root), "overlays", stem + ".jpg"))
        except OSError:
            pass
    return removed


DELETE_SNAPSHOT_KEEP = 200
DELETE_SNAPSHOT_PRUNE_MARGIN = 25


def _write_snapshot(root: str, snapshot: dict) -> str:
    folder = deleted_dir(root)
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(folder, f"{stamp}.json")
    tail = 1
    while os.path.exists(path):
        path = os.path.join(folder, f"{stamp}-{tail}.json")
        tail += 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)
    _prune_snapshots(folder)
    return path


_SNAPSHOT_NAME = re.compile(r"^(\d{8}-\d{6})(?:-(\d+))?\.json$")


def _snapshot_order(name: str) -> Tuple[str, int]:
    """按"哪一份先写"给快照文件名排序。

    同一秒里的多份靠 ``-1``、``-2`` 后缀区分，但按字符串排会乱：``-10`` 排在
    ``-2`` 前面，而没有后缀的那一份（其实是一个秒里**最先**写的）反倒排到最后。
    清理是从旧往新删，切点正好落在同一秒里时就会删掉更新的、留下更旧的。
    认不出的名字排到最后，永远不会被当成旧快照删掉。
    """
    match = _SNAPSHOT_NAME.match(name)
    if not match:
        return (name, 10 ** 9)
    return (match.group(1), int(match.group(2) or 0))


def _prune_snapshots(folder: str) -> int:
    """删除快照只留最近 ``DELETE_SNAPSHOT_KEEP`` 份，返回清掉的份数。

    快照是删错图之后的补救材料，但在界面上逐张删两万张就会攒出两万个小文件，
    而启动对账只回头看最近 20 份（见 ``_kept_file_tombstones``）。留出余量再
    一次清一批，免得每次删除都去动整个目录。
    """
    try:
        names = sorted((name for name in os.listdir(folder) if name.endswith(".json")),
                       key=_snapshot_order)
    except OSError:
        return 0
    if len(names) <= DELETE_SNAPSHOT_KEEP + DELETE_SNAPSHOT_PRUNE_MARGIN:
        return 0
    removed = 0
    for name in names[:len(names) - DELETE_SNAPSHOT_KEEP]:
        try:
            os.remove(os.path.join(folder, name))
            removed += 1
        except OSError:
            pass
    return removed


def _regenerate(root: str) -> dict:
    """删除后让坐标文件与 QC 报告跟上（只在原本已生成过时重算，纯 CPU 操作）。"""
    out: dict = {}
    if os.path.isdir(os.path.join(root, "coords")):
        try:
            from .coords import write_coordinate_files

            out["coords"] = str(write_coordinate_files(root, tag="batch-delete")["out"])
        except Exception:  # noqa: BLE001 重算失败不影响删除本身
            out["coords_error"] = "坐标文件重算失败，可手动执行 coords 命令"
    if os.path.exists(os.path.join(root, "qc", "report.md")):
        try:
            from .qc import write_report

            out["report"] = str(write_report(root).get("report"))
        except Exception:  # noqa: BLE001
            out["report_error"] = "QC 报告重算失败，可手动执行 qc 命令"
    return out
