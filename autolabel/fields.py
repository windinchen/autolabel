"""字段登记表：提示词标注里「要标什么」的最小单元。

一个字段 = 一个 YOLO detection 类别，由三部分组成：

- ``name``       英文类名，写进标签与 ``data.yaml``，也是模型回报时必填的 ``field`` 值
- ``label``      中文显示名，界面上给人看
- ``prompt_hint`` 描述提示词，告诉多模态大模型「什么样的东西算这个字段」

存储位置 ``<dataset>/.autolabel/fields.json``::

    {
      "version": 1,
      "next_class_id": 2,
      "fields": [
        {"id": "f_1a2b3c4d", "class_id": 0, "name": "bottle", "label": "瓶子",
         "prompt_hint": "桌上或手里的水瓶、纸杯、易拉罐", "color": "#4ade80",
         "created_at": "2026-10-17 09:00:00", "updated_at": "2026-10-17 09:00:00"}
      ]
    }

``class_id`` 在创建时分配、**永不复用**：删字段时若仍有框引用它，必须显式
``drop_boxes=True`` 把那些框一起删掉，否则类别号会错位、把已标数据喂错类别。
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import now_str

FIELDS_VERSION = 1
MAX_FIELDS = 80
MAX_NAME_LEN = 32
MAX_LABEL_LEN = 40
MAX_HINT_LEN = 600
# 类名要进 data.yaml 也要被模型原样回报，只允许 ASCII，中文名放 label
_NAME_RE = re.compile(r"^[A-Za-z][0-9A-Za-z_-]*$")
# 界面画框用的调色板，按顺序分配，字段多了循环使用
PALETTE = (
    "#4ade80", "#f87171", "#60a5fa", "#fbbf24", "#c084fc", "#2dd4bf",
    "#fb923c", "#a3e635", "#f472b6", "#38bdf8", "#facc15", "#a78bfa",
)

_LOCK = threading.RLock()


def fields_path(root: str) -> str:
    return os.path.join(os.path.abspath(root), ".autolabel", "fields.json")


def empty_registry() -> dict:
    return {"version": FIELDS_VERSION, "next_class_id": 0, "fields": [], "updated_at": ""}


def load_registry(root: str) -> dict:
    """读登记表；文件缺失或损坏时按空表处理（不抛异常，界面不该因为空数据集报错）。"""
    path = fields_path(root)
    if not os.path.exists(path):
        return empty_registry()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return empty_registry()
    if not isinstance(data, dict):
        return empty_registry()
    fields = [f for f in (data.get("fields") or []) if isinstance(f, dict) and f.get("name")]
    registry = dict(data)
    registry["fields"] = sorted(fields, key=lambda f: int(f.get("class_id", 0)))
    registry["next_class_id"] = int(data.get("next_class_id") or 0)
    # 兜底：登记表被手工改过时，保证 next_class_id 一定大于已用号
    used = [int(f.get("class_id", -1)) for f in registry["fields"]]
    registry["next_class_id"] = max([registry["next_class_id"]] + [n + 1 for n in used if n >= 0])
    return registry


def save_registry(root: str, registry: dict) -> None:
    path = fields_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(registry)
    payload["version"] = FIELDS_VERSION
    payload["updated_at"] = now_str()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def list_fields(root: str) -> List[dict]:
    """按类别号排序的字段列表。"""
    return list(load_registry(root)["fields"])


def field_by_class_id(root: str) -> Dict[int, dict]:
    return {int(f.get("class_id", 0)): f for f in list_fields(root)}


def normalize_name(name: str) -> str:
    """校验并规范化英文类名，不合法直接抛 :class:`ValueError`（错误文案给人看）。"""
    value = str(name or "").strip()
    if not value:
        raise ValueError("字段名不能为空")
    if len(value) > MAX_NAME_LEN:
        raise ValueError(f"字段名最长 {MAX_NAME_LEN} 个字符，当前 {len(value)} 个")
    if not _NAME_RE.match(value):
        raise ValueError("字段名要用英文字母开头，只能含字母/数字/下划线/连字符"
                         f"（要写中文请填「显示名」）：{value}")
    return value


def normalize_hint(hint: str) -> str:
    value = str(hint or "").strip()
    if len(value) > MAX_HINT_LEN:
        raise ValueError(f"描述提示词最长 {MAX_HINT_LEN} 字，当前 {len(value)} 字")
    return value


def _match(field: dict, wanted: str) -> bool:
    key = str(wanted or "").strip()
    if not key:
        return False
    if key in (str(field.get("id")), str(field.get("label"))):
        return True
    # 名字按大小写不敏感匹配：upsert 本来就把 bottle / BOTTLE 当同一条，
    # 这里再严格区分大小写会让「刚改名成 BOTTLE 的字段删不掉」
    return key.lower() == str(field.get("name", "")).lower()


def find_field(root: str, wanted: str) -> Optional[dict]:
    for field in list_fields(root):
        if _match(field, wanted):
            return field
    return None


def resolve_fields(root: str, wanted: Optional[Sequence[str]] = None) -> List[dict]:
    """把「字段名/显示名/ID」列表解析成字段字典列表；留空表示全部字段。"""
    fields = list_fields(root)
    if not wanted:
        return fields
    picked: List[dict] = []
    missing: List[str] = []
    for key in wanted:
        field = find_field(root, key)
        if field is None:
            missing.append(str(key))
        elif field not in picked:
            picked.append(field)
    if missing:
        options = "、".join(f.get("name", "") for f in fields) or "（还没有字段）"
        raise ValueError(f"未知字段 {'、'.join(missing)}；已有字段：{options}"
                         "（用 autolabel fields 或界面「提示词标注」新建）")
    return sorted(picked, key=lambda f: int(f.get("class_id", 0)))


def next_color(root: str) -> str:
    fields = list_fields(root)
    used = {str(f.get("color", "")).lower() for f in fields}
    for color in PALETTE:
        if color not in used:
            return color
    return PALETTE[len(fields) % len(PALETTE)]


def upsert_field(
    root: str,
    *,
    name: str,
    label: str = "",
    prompt_hint: str = "",
    field_id: str = "",
    color: str = "",
) -> dict:
    """新建或更新一个字段；``field_id`` 为空表示新建。"""
    clean_name = normalize_name(name)
    clean_label = str(label or "").strip()[:MAX_LABEL_LEN]
    clean_hint = normalize_hint(prompt_hint)
    clean_color = str(color or "").strip()[:32]
    with _LOCK:
        registry = load_registry(root)
        fields = registry["fields"]
        existing = next((f for f in fields
                         if (field_id and str(f.get("id")) == str(field_id))
                         or str(f.get("name")).lower() == clean_name.lower()), None)
        if existing is not None:
            if field_id and str(existing.get("id")) != str(field_id) and \
                    str(existing.get("name")).lower() == clean_name.lower():
                raise ValueError(f"字段名 {clean_name} 已被「{existing.get('id')}」占用，换个名")
            existing["name"] = clean_name
            existing["label"] = clean_label or clean_name
            existing["prompt_hint"] = clean_hint
            if clean_color:
                existing["color"] = clean_color
            existing["updated_at"] = now_str()
            saved = existing
        else:
            if len(fields) >= MAX_FIELDS:
                raise ValueError(f"字段数量上限 {MAX_FIELDS} 个，先删掉不用的再新建")
            taken = {int(f.get("class_id", -1)) for f in fields}
            class_id = int(registry["next_class_id"])
            while class_id in taken:
                class_id += 1
            saved = {
                "id": f"f_{uuid.uuid4().hex[:8]}", "class_id": class_id, "name": clean_name,
                "label": clean_label or clean_name, "prompt_hint": clean_hint,
                "color": clean_color or next_color(root), "created_at": now_str(),
                "updated_at": now_str(),
            }
            fields.append(saved)
            registry["next_class_id"] = class_id + 1
        registry["fields"] = sorted(fields, key=lambda f: int(f.get("class_id", 0)))
        save_registry(root, registry)
    return saved


def class_id_usage(root: str) -> Dict[int, int]:
    """每个类别号在 ``prompt_labels/`` 里还有多少个框。"""
    counts: Dict[int, int] = {}
    for _rel, rows in _iter_prompt_labels(root):
        for row in rows:
            counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def _iter_prompt_labels(root: str):
    """遍历 ``(相对路径, [(class_id, cx, cy, w, h), ...])``，跳过坏行。"""
    from .dataset import prompt_labels_dir

    base = prompt_labels_dir(root)
    if not os.path.isdir(base):
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if not fn.endswith(".txt") or fn.startswith("."):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, base).replace("\\", "/")
            yield rel, read_label_rows(path)


def read_label_rows(path: str) -> List[Tuple[int, float, float, float, float]]:
    """读 detection 标签，返回 ``(class_id, cx, cy, w, h)`` 列表；坏行直接跳过。"""
    rows: List[Tuple[int, float, float, float, float]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return rows
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append((int(float(parts[0])), float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return rows


def write_label_rows(path: str, rows: Sequence[Sequence[float]]) -> str:
    """写 detection 标签文件（空列表落一个空文件，表示「这张看过了，确实没有目标」）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = "".join(format_label_row(row) + "\n" for row in rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def format_label_row(row: Sequence[float]) -> str:
    """``class cx cy w h``，与 YOLO detection 完全一致（不带多余列，可直接喂训练）。"""
    cls = int(round(float(row[0])))
    values = [max(0.0, min(1.0, float(v))) for v in row[1:5]]
    return " ".join([str(cls)] + [f"{v:.6f}" for v in values])


def delete_field(root: str, wanted: str, *, drop_boxes: bool = False) -> dict:
    """删除字段；仍有框引用它时必须 ``drop_boxes=True``，并顺手把那些框从标签里清掉。"""
    field = find_field(root, wanted)
    if field is None:
        options = "、".join(f.get("name", "") for f in list_fields(root)) or "（还没有字段）"
        raise ValueError(f"未知字段 {wanted}；已有字段：{options}")
    class_id = int(field.get("class_id", -1))
    usage = class_id_usage(root)
    boxes = int(usage.get(class_id, 0))
    if boxes and not drop_boxes:
        raise ValueError(f"字段 {field.get('name')} 还标着 {boxes} 个框；"
                         "确认要一起删掉这些框再加 --drop-boxes（界面里勾「同时删除已标框」）")
    with _LOCK:
        registry = load_registry(root)
        before = len(registry["fields"])
        registry["fields"] = [f for f in registry["fields"] if int(f.get("class_id", -1)) != class_id]
        removed = before - len(registry["fields"])
        if removed == 0:
            raise ValueError(f"字段 {wanted} 已经被删掉了")
        save_registry(root, registry)
        touched = 0
        if boxes:
            touched = _strip_class(root, class_id)
    return {"ok": True, "id": field.get("id"), "name": field.get("name"),
            "class_id": class_id, "boxes_removed": boxes, "labels_touched": touched}


def _strip_class(root: str, class_id: int) -> int:
    """从所有 detection 标签里去掉某个类别号的行，返回改动过的文件数。"""
    from .dataset import prompt_labels_dir

    base = prompt_labels_dir(root)
    touched = 0
    for rel, rows in _iter_prompt_labels(root):
        kept = [row for row in rows if row[0] != class_id]
        if len(kept) == len(rows):
            continue
        write_label_rows(os.path.join(base, rel), kept)
        touched += 1
    return touched


def orphan_class_ids(root: str) -> List[int]:
    """标签里出现、但登记表里已经没有的类别号（删字段/手工改盘留下的尾巴）。"""
    known = {int(f.get("class_id", -1)) for f in list_fields(root)}
    return sorted({cid for cid in class_id_usage(root) if cid not in known})


def fields_payload(root: str) -> dict:
    """给界面/命令行共用的字段清单，带每个字段的框数与占用色。"""
    fields = list_fields(root)
    usage = class_id_usage(root)
    rows = []
    for field in fields:
        class_id = int(field.get("class_id", -1))
        fallback = PALETTE[class_id % len(PALETTE)] if class_id >= 0 else PALETTE[0]
        rows.append({
            "id": field.get("id"), "class_id": class_id, "name": field.get("name"),
            "label": field.get("label") or field.get("name"), "prompt_hint": field.get("prompt_hint", ""),
            "color": str(field.get("color") or fallback),
            "created_at": field.get("created_at"), "updated_at": field.get("updated_at"),
            "boxes": int(usage.get(class_id, 0)),
        })
    return {"fields": rows, "next_class_id": load_registry(root)["next_class_id"],
            "path": fields_path(root), "orphans": orphan_class_ids(root)}
