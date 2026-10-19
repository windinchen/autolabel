"""素材池（pool）：三个标注库共用的素材仓库，以及素材自己的分类与来处。

口径——**素材归池，标注归库**：

::

    <dataset>/
      images/                 素材本体，同一帧只落盘一次
      .autolabel/
        pool.json             池自己的账本：分类表 + 导入批次 + 每张素材的归属

以前「素材从哪来、还剩多少、要不要删」只能在各标注库里看（批次表记的是各库的引用），
想单独管素材、给素材分类就得三个库来回跳。分池之后：上传可以只进池不入库，
各库想标什么就从池里挑（``/api/pool/adopt``）；删素材看的是「还有哪些库在用」
（:func:`dataset.libs_holding_image`），分类跟着素材走，不会因为换了个库就丢。

分类和批次都是**素材层的属性，不是标注层的属性**：同一张图在三个库里是同一份文件，
所以只存一份，三个模块看到的分类永远一致。``未分类`` 是虚拟分类（不落库），
删掉一个真实分类时底下的素材退回未分类，一张素材都不会被连带删掉。

``pool.json`` 里的 ``items`` 只在两种情况下记账：素材被分了类，或者素材**只进了池、
还没有任何库引用**（这种"池独占"的图必须记账，否则下一次导入认不出它是重复帧，
启动对账也会把它当成孤儿图塞进姿态库）。库内已经有索引条目的图，状态仍以各库自己的
``index.json`` 为准，池账本只补素材层的字段，两边不会互相覆盖。
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Dict, List, Optional, Sequence

from .atomic import atomic_write_json
from .dataset import image_size, images_dir, now_str
from .library import meta_dir

#: 分类数量与名称长度上限：分类是给人扫一眼的，不是拿来建目录树的
CATEGORY_MAX = 40
CATEGORY_NAME_MAX = 24
NOTE_MAX = 80
#: 池批次 ID 的前缀：库批次是裸 hex，池批次 ``p_xxxxx``，删除与对账都按这个前缀分流
POOL_BATCH_PREFIX = "p_"
#: 虚拟分类：没分类的素材，界面上叫「未分类」
UNCATEGORIZED = ""
#: 分类图标只准从这几个里挑（前端 sprite 里的 id），否则一屏五种画风
CATEGORY_ICONS = ("ic-tag", "ic-person", "ic-object", "ic-scene", "ic-video")
#: 池独占素材的批次状态（和批次表同一套词，界面上不用另学一套）
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_INTERRUPTED = "interrupted"

#: 素材层可以从池账本补进"素材池视角"的字段。别把 ``status`` 之类标注意义的值放进来，
#: 那会串库（同一张图在姿态库已复核、在检测库还没标，谁也不该覆盖谁）
POOL_ITEM_KEYS = ("sha256", "bytes", "width", "height", "dhash", "imported_at", "source", "video")


def catalog_path(root: str) -> str:
    """池账本的位置（``.autolabel/pool.json``）。"""
    return os.path.join(meta_dir(root), "pool.json")


def _blank() -> dict:
    return {"version": 1, "categories": [], "batches": [], "items": {}}


def load_catalog(root: str) -> dict:
    """读池账本；缺文件/坏文件都退化成空账本，不拦启动。"""
    path = catalog_path(root)
    if not os.path.exists(path):
        return _blank()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):  # 坏账本当空的，下次写的时候自然重建
        return _blank()
    if not isinstance(data, dict):
        return _blank()
    rows = data.get("categories")
    batches = data.get("batches")
    items = data.get("items")
    return {
        "version": 1,
        "categories": [row for row in (rows if isinstance(rows, list) else [])
                       if isinstance(row, dict) and row.get("id")],
        "batches": [row for row in (batches if isinstance(batches, list) else [])
                    if isinstance(row, dict) and row.get("id")],
        "items": {str(rel): row for rel, row in (items if isinstance(items, dict) else {}).items()
                  if isinstance(row, dict) and str(rel)},
    }


def _save(root: str, catalog: dict) -> dict:
    atomic_write_json(catalog_path(root), catalog)
    return catalog


def save_catalog(root: str, catalog: dict) -> dict:
    """把整本池账本写回磁盘。跨模块改完账本统一走这里，别自己拼原子写。"""
    return _save(root, catalog)


def _key(rel: str) -> str:
    text = str(rel or "").replace("\\", "/").strip()
    return text[len("images/"):] if text.startswith("images/") else text


# --------------------------------------------------------------------------- #
# 分类
# --------------------------------------------------------------------------- #

def categories(root: str) -> List[dict]:
    """分类表（建类时间正序，界面顺序与这里一致）。"""
    return list(load_catalog(root)["categories"])


def category_map(root: str) -> Dict[str, str]:
    """``分类 ID -> 分类名``。"""
    return {str(row["id"]): str(row.get("name") or "") for row in categories(root)}


def _clean_name(name: str) -> str:
    text = " ".join(str(name or "").split())
    if not text:
        raise ValueError("分类名不能为空")
    return text[:CATEGORY_NAME_MAX]


def save_category(root: str, name: str, note: str = "", icon: str = "",
                  category_id: str = "") -> dict:
    """新建 / 改名一个分类。给了 ``category_id`` 就改那一条，否则新建。"""
    text = _clean_name(name)
    catalog = load_catalog(root)
    rows = catalog["categories"]
    wanted = str(category_id or "").strip()
    if wanted:
        target = next((row for row in rows if str(row["id"]) == wanted), None)
        if target is None:
            raise ValueError(f"没有这个分类（可能刚被删掉）：{wanted}")
        if any(str(row.get("name")) == text and str(row["id"]) != wanted for row in rows):
            raise ValueError(f"已经有叫「{text}」的分类了")
        target["name"] = text
        target["note"] = str(note or "").strip()[:NOTE_MAX]
        if icon in CATEGORY_ICONS:
            target["icon"] = icon
        target["updated_at"] = now_str()
        row = target
    else:
        if any(str(row.get("name")) == text for row in rows):
            raise ValueError(f"已经有叫「{text}」的分类了")
        if len(rows) >= CATEGORY_MAX:
            raise ValueError(f"分类最多 {CATEGORY_MAX} 个，先删几个再建")
        used = {str(row["id"]) for row in rows}
        row = {"id": "c" + uuid.uuid4().hex[:8], "name": text,
               "note": str(note or "").strip()[:NOTE_MAX],
               "icon": icon if icon in CATEGORY_ICONS else "ic-tag",
               "created_at": now_str()}
        while row["id"] in used:  # 极小概率撞 id，换一个
            row["id"] = "c" + uuid.uuid4().hex[:8]
        rows.append(row)
    _save(root, catalog)
    return {"ok": True, "category": row, "categories": rows}


def delete_category(root: str, category_id: str) -> dict:
    """删分类；底下的素材退回「未分类」，素材文件一张都不动。"""
    wanted = str(category_id or "").strip()
    catalog = load_catalog(root)
    rows = catalog["categories"]
    target = next((row for row in rows if str(row["id"]) == wanted), None)
    if target is None:
        raise ValueError(f"没有这个分类（可能刚被删掉）：{wanted}")
    catalog["categories"] = [row for row in rows if str(row["id"]) != wanted]
    moved = 0
    for item in catalog["items"].values():
        if str(item.get("category") or "") == wanted:
            item.pop("category", None)
            moved += 1
    _save(root, catalog)
    return {"ok": True, "deleted": target, "uncategorized": moved,
            "categories": catalog["categories"]}


def assign(root: str, image_rels: Sequence[str], category_id: str = "") -> dict:
    """给一批素材设分类；``category_id`` 为空表示取消分类。"""
    wanted = str(category_id or "").strip()
    catalog = load_catalog(root)
    if wanted and not any(str(row["id"]) == wanted for row in catalog["categories"]):
        raise ValueError(f"没有这个分类（可能刚被删掉）：{wanted}")
    stamp = now_str()
    set_count = cleared = 0
    for rel in image_rels:
        key = _key(rel)
        if not key:
            continue
        item = catalog["items"].setdefault(key, {})
        if wanted:
            item["category"] = wanted
            set_count += 1
        else:
            item.pop("category", None)
            item.pop("assigned_at", None)
            cleared += 1
            if not item:
                catalog["items"].pop(key, None)
                continue
        item["assigned_at"] = stamp
    _save(root, catalog)
    return {"ok": True, "category": wanted, "set": set_count, "cleared": cleared,
            "requested": len([rel for rel in image_rels if str(rel).strip()])}


# --------------------------------------------------------------------------- #
# 导入批次（素材的来处）
# --------------------------------------------------------------------------- #

def batches(root: str) -> List[dict]:
    """池的导入批次（新→旧）。"""
    rows = list(load_catalog(root)["batches"])
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows


def batch_map(root: str) -> Dict[str, str]:
    """``批次 ID -> 批次名``。"""
    return {str(row["id"]): str(row.get("name") or "") for row in batches(root)}


def create_batch(root: str, *, name: str, kind: str = "pool", source: str = "ui",
                 uploads: Optional[Sequence[str]] = None) -> dict:
    """记一次进池（上传 / 抽帧 / 命令行导入）。"""
    batch = {
        "id": POOL_BATCH_PREFIX + uuid.uuid4().hex[:10],
        "name": str(name or "素材导入")[:60], "kind": kind, "source": source,
        "created_at": now_str(), "finished_at": None, "status": STATUS_RUNNING,
        "uploads": [str(u) for u in (uploads or [])][:30],
        "items": [], "message": "",
    }
    catalog = load_catalog(root)
    catalog["batches"].append(batch)
    catalog["batches"] = catalog["batches"][-300:]
    _save(root, catalog)
    return batch


def attach_items(root: str, batch_id: str, image_rels: Sequence[str]) -> int:
    """把进池成功的素材挂到批次上（只进池的图靠这里记来处，库索引里没有它们）。"""
    rels = [_key(rel) for rel in image_rels if str(rel).strip()]
    rels = [rel for rel in rels if rel]
    if not rels:
        return 0
    catalog = load_catalog(root)
    batch = next((row for row in catalog["batches"] if str(row.get("id")) == str(batch_id)), None)
    if batch is not None:
        known = set(batch.get("items") or [])
        batch.setdefault("items", []).extend(rel for rel in rels if rel not in known)
    for rel in rels:
        item = catalog["items"].setdefault(rel, {})
        item["batch"] = str(batch_id)
        if item.get("bytes") is None and item.get("width") is None:
            # 素材本来只在某个库的索引里（池账本还没它）：就地按磁盘把素材层字段补齐，
            # 只记一个批次号会让素材池页这张卡片写「尺寸未知 · 0 KB」
            path = os.path.join(images_dir(root), rel)
            try:
                item["bytes"] = os.path.getsize(path)
            except OSError:
                continue
            width, height = image_size(path)
            item["width"], item["height"] = width or None, height or None
            item.setdefault("imported_at", now_str())
    _save(root, catalog)
    return len(rels)


def finish_batch(root: str, batch_id: str, status: str = STATUS_DONE, message: str = "") -> None:
    catalog = load_catalog(root)
    batch = next((row for row in catalog["batches"] if str(row.get("id")) == str(batch_id)), None)
    if batch is None:
        return
    batch["status"] = status
    batch["finished_at"] = now_str()
    if message:
        batch["message"] = str(message)[:200]
    _save(root, catalog)


def forget_batch(root: str, batch_id: str) -> dict:
    """批次记录随素材一起删掉时调用（素材文件已删，不再留空批次行）。"""
    catalog = load_catalog(root)
    before = len(catalog["batches"])
    catalog["batches"] = [row for row in catalog["batches"] if str(row.get("id")) != str(batch_id)]
    removed = before - len(catalog["batches"])
    detached = 0
    if removed:
        # 素材上的「来处」也要一起忘掉：指着一个已经不存在的批次，按批次筛就永远筛不到它
        for item in catalog["items"].values():
            if str(item.get("batch") or "") == str(batch_id):
                item.pop("batch", None)
                detached += 1
        _save(root, catalog)
    return {"ok": bool(removed), "removed": removed, "detached": detached}


# --------------------------------------------------------------------------- #
# 素材记账
# --------------------------------------------------------------------------- #

def record_file(root: str, image_rel: str, fields: dict, batch_id: str = "") -> dict:
    """只进池的素材在池账本里记一笔（库索引里没有它，靠这笔认重复、认来处）。"""
    key = _key(image_rel)
    if not key:
        raise ValueError("素材路径为空")
    catalog = load_catalog(root)
    item = catalog["items"].setdefault(key, {})
    for name in POOL_ITEM_KEYS:
        if fields.get(name) is not None:
            item[name] = fields[name]
    if batch_id:
        item["batch"] = str(batch_id)
    _save(root, catalog)
    return item


def pool_items_map(root: str) -> Dict[str, dict]:
    """池账本里的素材条目（原样返回，调用方别改）。"""
    return dict(load_catalog(root)["items"])


def pool_only_rels(root: str) -> List[str]:
    """只进了池、还没有任何库引用的素材：启动对账不许把它们当孤儿图抢进姿态库。"""
    from .dataset import libs_holding_image

    out = []
    for rel in load_catalog(root)["items"]:
        if not libs_holding_image(root, rel):
            out.append(rel)
    return out


def prune(root: str, image_rels: Sequence[str]) -> dict:
    """素材文件真被删了之后，把账本里的条目一起清掉，不留孤账。

    批次跟着一起瘦身：删掉的素材从批次 ``items`` 里摘掉，一批素材全删光的批次行也删掉。
    否则素材池页「按来处」会留一排点进去空白的 0 张批次，让人以为素材还在。
    正在导入、还没挂上素材的批次不动（它 ``items`` 本来就是空的）。
    """
    catalog = load_catalog(root)
    gone = set()
    removed = 0
    for rel in image_rels:
        key = _key(rel)
        if not key:
            continue
        gone.add(key)
        if catalog["items"].pop(key, None) is not None:
            removed += 1
    emptied = 0
    if gone:
        keep: List[dict] = []
        for row in catalog["batches"]:
            had = list(row.get("items") or [])
            left = [rel for rel in had if _key(rel) not in gone]
            if len(left) == len(had):
                keep.append(row)
                continue
            if left:
                row["items"] = left
                keep.append(row)
            else:
                emptied += 1  # 这批的素材全没了，批次行留着只会误导
        catalog["batches"] = keep
    if removed or emptied:
        _save(root, catalog)
    return {"ok": True, "removed": removed, "batches_emptied": emptied}


def attach(root: str, rows: List[dict], catalog: Optional[dict] = None) -> List[dict]:
    """给 :func:`dataset.pool_items` 的每行补上分类与来处（原地改行，返回同一列表）。

    ``catalog`` 允许调用方把已经读好的账本传进来，省一次读盘（大池子逐行读会被拖慢）。
    """
    catalog = catalog or load_catalog(root)
    names = {str(row["id"]): str(row.get("name") or "") for row in catalog["categories"]}
    batch_names = {str(row["id"]): str(row.get("name") or "") for row in catalog["batches"]}
    # 命令行导入的素材只有各库的导入批次、没有池批次，来处要回落到那份名字，
    # 否则素材池页上"这批是哪来的"会出现成片空白
    library_batches: Dict[str, str] = {}
    if any(not batch_names.get(str(row.get("batch") or "")) for row in rows):
        from .batches import batch_names as library_batch_names

        library_batches = library_batch_names(root)
    for row in rows:
        rel = str(row.get("image") or "")
        item = catalog["items"].get(rel) or {}
        cid = str(item.get("category") or UNCATEGORIZED)
        if cid and cid not in names:  # 分类被删过但账本没清干净，退化成未分类
            cid = UNCATEGORIZED
        row["category"] = cid
        row["category_name"] = names.get(cid) or "未分类"
        batch_id = str(item.get("batch") or "") or str(row.get("batch_id") or "")
        row["pool_batch"] = batch_id
        row["pool_batch_name"] = batch_names.get(batch_id) or library_batches.get(batch_id) or ""
        # 只进了池、三个库都没引用：界面上叫「未入库」，各模块的「从素材池添加」才捡得到它
        row["pool_only"] = not (row.get("libs") or [])
    return rows


def category_summary(root: str, rows: Optional[List[dict]] = None) -> List[dict]:
    """按分类汇总张数与体积，给素材池的分类条和工作台摘要用。"""
    if rows is None:
        from .dataset import pool_items

        rows = attach(root, pool_items(root))
    buckets: Dict[str, dict] = {}
    for row in rows:
        cid = str(row.get("category") or UNCATEGORIZED)
        bucket = buckets.setdefault(cid, {
            "id": cid, "name": row.get("category_name") or "未分类",
            "files": 0, "bytes": 0, "unassigned": cid == UNCATEGORIZED})
        bucket["files"] += 1
        bucket["bytes"] += int(row.get("bytes") or 0)
    ordered = [buckets[str(row["id"])] for row in categories(root) if str(row["id"]) in buckets]
    if UNCATEGORIZED in buckets:
        ordered.append(buckets[UNCATEGORIZED])
    return ordered


__all__: Sequence[str] = [
    "CATEGORY_ICONS", "CATEGORY_MAX", "CATEGORY_NAME_MAX", "NOTE_MAX",
    "POOL_BATCH_PREFIX", "POOL_ITEM_KEYS", "STATUS_DONE", "STATUS_INTERRUPTED",
    "STATUS_RUNNING", "UNCATEGORIZED",
    "assign", "attach", "attach_items", "batch_map", "batches", "catalog_path", "categories",
    "category_map", "category_summary", "create_batch", "delete_category", "finish_batch",
    "forget_batch", "load_catalog", "pool_items_map", "pool_only_rels", "prune",
    "record_file", "save_catalog", "save_category",
]
