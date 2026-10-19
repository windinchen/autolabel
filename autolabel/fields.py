"""字段登记表：提示词标注里「要标什么」的最小单元。

一个字段 = 一个 YOLO detection 类别，由三部分组成：

- ``name``       英文类名，写进标签与 ``data.yaml``，也是模型回报时必填的 ``field`` 值
- ``label``      中文显示名，界面上给人看
- ``prompt_hint`` 描述提示词，告诉多模态大模型「什么样的东西算这个字段」

两个 detection 类库共用这一套结构，各自一份登记表，``lib`` 参数决定读哪一份：
提示词标注在 ``<dataset>/.autolabel/fields.json``（沿用老名字，老数据集不迁移），
目标检测在 ``<dataset>/.autolabel/detect/classes.json``。
目标检测的类别来自模型自带类别表，``prompt_hint`` 用不上，改用 ``source`` 记来源模型。

存储位置::

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

from .dataset import load_index, now_str
from .atomic import atomic_write_json
from .library import (LIB_DETECT, LIB_OBJECTS, field_key, labels_dir, lib_name, normalize,
                      registry_path)

FIELDS_VERSION = 1
MAX_FIELDS = 80
#: 目标检测库的类别表来自模型自带（COCO 一种就占满 80 个），上限必须放宽，
#: 否则"把模型认识的类别都收进登记表"这件事根本做不到。
MAX_FIELDS_BY_LIB = {LIB_OBJECTS: MAX_FIELDS, LIB_DETECT: 200}
MAX_NAME_LEN = 32
MAX_LABEL_LEN = 40
MAX_HINT_LEN = 600
# 类名要进 data.yaml 也要被模型原样回报，只允许 ASCII，中文名放 label
_NAME_RE = re.compile(r"^[A-Za-z][0-9A-Za-z_-]*$")
#: 目标检测库额外允许词间空格：COCO 里 ``potted plant``、``fire hydrant`` 这类
#: 名字是模型原样回报的，逼用户改名会让「把模型认识的类别都收进来」漏掉十几个，
#: 而改成下划线又会对不上模型回报的值，标出来的框直接少一批。
_NAME_RE_SPACED = re.compile(r"^[A-Za-z][0-9A-Za-z_-]*(?: [0-9A-Za-z_-]+)*$")
# 界面画框用的调色板，按顺序分配，字段多了循环使用
PALETTE = (
    "#4ade80", "#f87171", "#60a5fa", "#fbbf24", "#c084fc", "#2dd4bf",
    "#fb923c", "#a3e635", "#f472b6", "#38bdf8", "#facc15", "#a78bfa",
)

_LOCK = threading.RLock()


def fields_path(root: str, lib: str = LIB_OBJECTS) -> str:
    """该库的类别登记表路径（提示词库沿用 ``fields.json`` 这个历史名字）。"""
    return registry_path(root, lib)


def max_fields(lib: str = LIB_OBJECTS) -> int:
    """该库的类别数量上限。"""
    return int(MAX_FIELDS_BY_LIB.get(normalize(lib), MAX_FIELDS))


def empty_registry() -> dict:
    return {"version": FIELDS_VERSION, "next_class_id": 0, "fields": [], "updated_at": ""}


def load_registry(root: str, lib: str = LIB_OBJECTS) -> dict:
    """读登记表；文件缺失或损坏时按空表处理（不抛异常，界面不该因为空数据集报错）。"""
    path = fields_path(root, lib)
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


def save_registry(root: str, registry: dict, lib: str = LIB_OBJECTS) -> None:
    payload = dict(registry)
    payload["version"] = FIELDS_VERSION
    payload["updated_at"] = now_str()
    atomic_write_json(fields_path(root, lib), payload)


def list_fields(root: str, lib: str = LIB_OBJECTS) -> List[dict]:
    """按类别号排序的字段列表。"""
    return list(load_registry(root, lib)["fields"])


def field_by_class_id(root: str, lib: str = LIB_OBJECTS) -> Dict[int, dict]:
    return {int(f.get("class_id", 0)): f for f in list_fields(root, lib)}


def normalize_name(name: str, lib: str = LIB_OBJECTS) -> str:
    """校验并规范化英文类名，不合法直接抛 :class:`ValueError`（错误文案给人看）。

    提示词库按严格规则（类名是多模态大模型必须原样回报的 JSON 值，带空格容易回报成
    下划线）；目标检测库允许词间空格，因为这里的类名一律来自权重自带的类别表。
    """
    value = " ".join(str(name or "").split())     # 顺手把连续空白压成一个空格
    if not value:
        raise ValueError("字段名不能为空")
    if len(value) > MAX_NAME_LEN:
        raise ValueError(f"字段名最长 {MAX_NAME_LEN} 个字符，当前 {len(value)} 个")
    spaced = normalize(lib) == LIB_DETECT
    if not (spaced and _NAME_RE_SPACED.match(value)) and not _NAME_RE.match(value):
        raise ValueError("字段名要用英文字母开头，只能含字母/数字/下划线/连字符"
                         + ("（目标检测库还可含词间空格）" if spaced else "")
                         + f"（要写中文请填「显示名」）：{value}")
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


def vocab(lib: str = LIB_OBJECTS) -> Tuple[str, str]:
    """两套叫法、两条添加路径：提示词库叫「字段」就地新建，目标检测库叫「类别」从权重导入。

    以前对检测库也说"未知字段……已有字段……在界面里新建"，可那页面上根本没有
    「新建」按钮，类别是从 YOLO 权重的类别表导入的；把 80 个 COCO 名全列出来
    还会把报错撑成一大坨，真正想说的"这个名字还没导入"反而被埋掉。
    """
    if normalize(lib) == LIB_DETECT:
        return "类别", (f"在界面「{lib_name(lib)}」点「从权重类别表导入」，"
                        f"命令行 autolabel detect --add-classes 类名")
    return "字段", (f"在界面「{lib_name(lib)}」新建，"
                    f"命令行 autolabel fields --add 名字=描述")


def list_options(fields: Sequence[dict], lib: str = LIB_OBJECTS, limit: int = 12) -> str:
    """已有的类别/列个开头，太长就省略——80 个 COCO 名全列出来没人看得完。"""
    names = [str(f.get("name", "")) for f in fields if f.get("name")]
    kind = vocab(lib)[0]
    if not names:
        # 这里不要再补"怎么添加"：外层那句已经把动作说了一遍，
        # 两处都说就成了「…先在界面里新建…）。在界面里新建…」这种复读机。
        return f"（还没有{kind}）"
    head = "、".join(names[:limit])
    if len(names) > limit:
        head += f"……（共 {len(names)} 个）"
    return head


def find_field(root: str, wanted: str, lib: str = LIB_OBJECTS) -> Optional[dict]:
    for field in list_fields(root, lib):
        if _match(field, wanted):
            return field
    return None


def resolve_fields(root: str, wanted: Optional[Sequence[str]] = None,
                   lib: str = LIB_OBJECTS) -> List[dict]:
    """把「字段名/显示名/ID」列表解析成字段字典列表；留空表示全部字段。"""
    fields = list_fields(root, lib)
    if not wanted:
        return fields
    picked: List[dict] = []
    missing: List[str] = []
    for key in wanted:
        field = find_field(root, key, lib)
        if field is None:
            missing.append(str(key))
        elif field not in picked:
            picked.append(field)
    if missing:
        kind, how = vocab(lib)
        raise ValueError(f"未知{kind} {'、'.join(missing)}；已有{kind}：{list_options(fields, lib)}。{how}")
    return sorted(picked, key=lambda f: int(f.get("class_id", 0)))


def next_color(root: str, lib: str = LIB_OBJECTS) -> str:
    fields = list_fields(root, lib)
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
    lib: str = LIB_OBJECTS,
    source: str = "",
) -> dict:
    """新建或更新一个字段；``field_id`` 为空表示新建。"""
    clean_name = normalize_name(name, lib)
    clean_label = str(label or "").strip()[:MAX_LABEL_LEN]
    clean_hint = normalize_hint(prompt_hint)
    clean_color = str(color or "").strip()[:32]
    clean_source = str(source or "").strip()[:80]
    limit = max_fields(lib)
    with _LOCK:
        registry = load_registry(root, lib)
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
            if clean_source:
                # 目标检测库靠这个字段记"这些类别出自哪个权重"，别被空值抹掉
                existing["source"] = clean_source
            if clean_color:
                existing["color"] = clean_color
            existing["updated_at"] = now_str()
            saved = existing
        else:
            if len(fields) >= limit:
                raise ValueError(f"{lib_name(lib)}最多登记 {limit} 个类别，"
                                 "先删掉不用的再新建")
            taken = {int(f.get("class_id", -1)) for f in fields}
            class_id = int(registry["next_class_id"])
            while class_id in taken:
                class_id += 1
            saved = {
                "id": f"f_{uuid.uuid4().hex[:8]}", "class_id": class_id, "name": clean_name,
                "label": clean_label or clean_name, "prompt_hint": clean_hint,
                "source": clean_source, "color": clean_color or next_color(root, lib),
                "created_at": now_str(), "updated_at": now_str(),
            }
            fields.append(saved)
            registry["next_class_id"] = class_id + 1
        registry["fields"] = sorted(fields, key=lambda f: int(f.get("class_id", 0)))
        save_registry(root, registry, lib)
    return saved


def class_id_usage(root: str, lib: str = LIB_OBJECTS) -> Dict[int, int]:
    """每个类别号在本库标签目录里还有多少个框。"""
    counts: Dict[int, int] = {}
    for _rel, rows in _iter_box_labels(root, lib):
        for row in rows:
            counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def _iter_box_labels(root: str, lib: str = LIB_OBJECTS):
    """遍历本库标签目录 ``(相对路径, [(class_id, cx, cy, w, h, 轮廓), ...])``，跳过坏行。

    轮廓是 ``[x1, y1, ...]`` 归一化点列，没有轮廓时为空表；只用前 5 列的调用方
    （框数、类别占用）照旧成立。
    """
    base = labels_dir(root, lib)
    if not os.path.isdir(base):
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if not fn.endswith(".txt") or fn.startswith("."):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, base).replace("\\", "/")
            yield rel, read_label_entries(path)


#: 轮廓尾列的最少列数：3 个点 × (x, y)
POLY_MIN_COLUMNS = 6


def parse_polygon(tokens: Sequence[str]) -> List[float]:
    """把标签尾列解析成归一化多边形 ``[x1, y1, x2, y2, ...]``；不合规就返回空表。

    读的时候**不报错**：尾列写坏了宁可当没轮廓（框还要留着），坏列由
    :func:`validate_box_label_file` 负责报出来。
    """
    if len(tokens) < POLY_MIN_COLUMNS or len(tokens) % 2:
        return []
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        return []
    return [max(0.0, min(1.0, value)) for value in values]


def read_label_entries(path: str) -> List[Tuple[int, float, float, float, float, List[float]]]:
    """读 detection 标签，含可选的轮廓尾列；坏行直接跳过。

    前 5 列含义不变（``class cx cy w h``），轮廓接在后面：只认 5 列的老逻辑
    读到的是同一个框，``-seg`` 权重的轮廓也不再被压成方框丢掉。
    """
    entries: List[Tuple[int, float, float, float, float, List[float]]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return entries
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            entries.append((int(float(parts[0])), float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]), parse_polygon(parts[5:])))
        except ValueError:
            continue
    return entries


def read_label_rows(path: str) -> List[Tuple[int, float, float, float, float]]:
    """读 detection 标签的 ``(class_id, cx, cy, w, h)``；轮廓在这里被抹掉。

    只要方框的调用方用它，要轮廓用 :func:`read_label_entries`。
    """
    return [row[:5] for row in read_label_entries(path)]


def validate_box_label_file(path: str, class_ids: Optional[set] = None) -> List:
    """校验 YOLO detection 标签（``class cx cy w h``，可再接归一化轮廓点列）。

    ``read_label_rows`` 为了不让一行脏数据毁掉整批，坏行是**静默跳过**的——
    标注时这样对，校验时正相反：被跳过的行必须报出来，否则用户永远不知道自己
    丢了几个框。给了 ``class_ids`` 就顺带检查类别号还在不在本库类别表里。
    """
    from .yolo_pose import LabelIssue

    issues: List = []
    if not os.path.exists(path):
        return [LabelIssue(path, "missing_label", "缺少标签文件")]
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) < 5:
                issues.append(LabelIssue(path, "bad_columns",
                                         f"只有 {len(parts)} 列，detection 标签要 5 列"
                                         f"（class cx cy w h）", lineno))
                continue
            tail = parts[5:]
            if tail:
                issues += _polygon_issues(path, lineno, tail)
            try:
                class_id = int(float(parts[0]))
                cx, cy, width, height = (float(value) for value in parts[1:5])
            except ValueError as exc:
                issues.append(LabelIssue(path, "parse_error", str(exc), lineno))
                continue
            if class_id < 0:
                issues.append(LabelIssue(path, "bad_class", f"class_id={class_id} 是负数", lineno))
            elif class_ids is not None and class_id not in class_ids:
                issues.append(LabelIssue(path, "bad_class",
                                         f"class_id={class_id} 不在本库类别表里", lineno))
            for name, value in (("cx", cx), ("cy", cy), ("w", width), ("h", height)):
                if not (0.0 <= value <= 1.0):
                    issues.append(LabelIssue(path, "bbox_range", f"{name}={value:.4f} 越界", lineno))
            if width <= 0 or height <= 0:
                issues.append(LabelIssue(path, "bbox_degenerate",
                                         f"w={width:.4f} h={height:.4f}", lineno))
    return issues


def _polygon_issues(path: str, lineno: int, tail: Sequence[str]) -> List:
    """轮廓尾列的格式检查：点数、成对、取值范围。"""
    from .yolo_pose import LabelIssue

    issues: List = []
    if len(tail) % 2:
        return [LabelIssue(path, "bad_polygon",
                           f"轮廓多出 {len(tail)} 列，点的坐标要成对写（x y）", lineno)]
    if len(tail) < POLY_MIN_COLUMNS:
        return [LabelIssue(path, "bad_polygon",
                           f"轮廓只有 {len(tail) // 2} 个点，多边形至少要 3 个点", lineno)]
    try:
        points = [float(value) for value in tail]
    except ValueError as exc:
        return [LabelIssue(path, "parse_error", f"轮廓坐标不是数字：{exc}", lineno)]
    outside = [value for value in points if not (0.0 <= value <= 1.0)]
    if outside:
        issues.append(LabelIssue(path, "poly_range",
                                f"轮廓有 {len(outside)} 个坐标越界（要在 0~1 之间，"
                                f"例如 {outside[0]:.4f}）", lineno))
    return issues


def write_label_rows(path: str, rows: Sequence[Sequence[float]]) -> str:
    """写 detection 标签文件（空列表落一个空文件，表示「这张看过了，确实没有目标」）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = "".join(format_label_row(row) + "\n" for row in rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def format_label_row(row: Sequence[float]) -> str:
    """``class cx cy w h``；带轮廓时再接归一化点列 ``x1 y1 ...``。

    不带轮廓时与 YOLO detection 完全一致，可直接喂 ``yolo detect train``；
    带轮廓时前 5 列一个字符都不变，老的读取逻辑照样拿到同一个框。
    """
    cls = int(round(float(row[0])))
    values = [max(0.0, min(1.0, float(v))) for v in row[1:5]]
    cells = [str(cls)] + [f"{v:.6f}" for v in values]
    tail = [max(0.0, min(1.0, float(v))) for v in row[5:]]
    if len(tail) >= POLY_MIN_COLUMNS and len(tail) % 2 == 0:
        cells += [f"{v:.6f}" for v in tail]
    return " ".join(cells)


def format_segment_row(row: Sequence[float]) -> str:
    """标准 YOLO **segment** 行：``class x1 y1 x2 y2 ...``（不带方框列）。

    导出的训练集用这个格式，轮廓才真的进得去分割训练；本库自己的标签文件用
    :func:`format_label_row`（方框优先，轮廓是附加信息）。
    """
    cls = int(round(float(row[0])))
    points = [max(0.0, min(1.0, float(v))) for v in row[1:]]
    if len(points) < POLY_MIN_COLUMNS or len(points) % 2:
        raise ValueError(f"分割标签要至少 3 个点（{POLY_MIN_COLUMNS} 列坐标），现在只有 {len(points)} 列")
    return " ".join([str(cls)] + [f"{v:.6f}" for v in points])


def delete_field(root: str, wanted: str, *, drop_boxes: bool = False,
                 lib: str = LIB_OBJECTS) -> dict:
    """删除字段；仍有框引用它时必须 ``drop_boxes=True``，并顺手把那些框从标签里清掉。"""
    field = find_field(root, wanted, lib)
    if field is None:
        kind, how = vocab(lib)
        raise ValueError(f"未知{kind} {wanted}；已有{kind}：{list_options(list_fields(root, lib), lib)}。{how}")
    class_id = int(field.get("class_id", -1))
    usage = class_id_usage(root, lib)
    boxes = int(usage.get(class_id, 0))
    if boxes and not drop_boxes:
        raise ValueError(f"{vocab(lib)[0]} {field.get('name')} 还标着 {boxes} 个框；"
                         "确认要一起删掉这些框再加 --drop-boxes（界面里勾「同时删除已标框」）")
    with _LOCK:
        registry = load_registry(root, lib)
        before = len(registry["fields"])
        registry["fields"] = [f for f in registry["fields"] if int(f.get("class_id", -1)) != class_id]
        removed = before - len(registry["fields"])
        if removed == 0:
            raise ValueError(f"{vocab(lib)[0]} {wanted} 已经被删掉了")
        save_registry(root, registry, lib)
        touched = 0
        if boxes:
            touched = _strip_class(root, class_id, lib)
    return {"ok": True, "id": field.get("id"), "name": field.get("name"),
            "class_id": class_id, "boxes_removed": boxes, "labels_touched": touched}


def flatten_label_row(entry: Sequence[object]) -> List[float]:
    """``(class_id, cx, cy, w, h, [轮廓点])`` → 一行标签要的扁平数字列表。"""
    tail = entry[5] if len(entry) > 5 else []
    return [float(entry[0])] + [float(v) for v in entry[1:5]] + [float(v) for v in (tail or [])]


def _strip_class(root: str, class_id: int, lib: str = LIB_OBJECTS) -> int:
    """从本库所有 detection 标签里去掉某个类别号的行，返回改动过的文件数。

    顺手把索引里每张图的框数改成和磁盘一致。以前这一步只重写标签文件，
    索引继续按删之前的数字报"这张图 3 个框"：列表页写着 3 个、点开编辑器只剩 1 个，
    `status` 的合计也跟着虚高——同一件事在同一个界面上出现两个数。
    """
    base = labels_dir(root, lib)
    index = load_index(root, lib)
    # 标签目录里是 ``x0.txt``，索引里是 ``x0.jpg``：按主干名对上，认不出的
    # （孤儿标签）跳过，不该为它凭空造一条索引
    image_of_stem = {os.path.splitext(rel)[0]: rel for rel in index.items}
    count_key = field_key(lib, "objects")
    touched = 0
    for rel, rows in _iter_box_labels(root, lib):
        kept = [flatten_label_row(row) for row in rows if row[0] != class_id]
        if len(kept) == len(rows):
            continue
        write_label_rows(os.path.join(base, rel), kept)
        touched += 1
        image = image_of_stem.get(os.path.splitext(rel)[0])
        if image is not None:
            index.upsert(image, **{count_key: len(kept)})
    if touched:
        index.save()
    return touched


def box_counts_by_image(root: str, lib: str = LIB_OBJECTS) -> Dict[str, int]:
    """本库每个标签文件里到底有几个框，键是**图片主干名**（``x0``，不带扩展名）。

    索引里的框数是写标签时顺手记的一笔账，手工改过标签、或写标签与写索引之间
    被打断，两边就会分叉：列表页的框数和编辑器里真正的框各说各话。校验需要的是
    "磁盘上几个"这一份事实，所以单独提供。
    """
    counts: Dict[str, int] = {}
    for rel, rows in _iter_box_labels(root, lib):
        counts[os.path.splitext(rel)[0]] = len(rows)
    return counts


def orphan_class_ids(root: str, lib: str = LIB_OBJECTS) -> List[int]:
    """标签里出现、但登记表里已经没有的类别号（删字段/手工改盘留下的尾巴）。"""
    known = {int(f.get("class_id", -1)) for f in list_fields(root, lib)}
    return sorted({cid for cid in class_id_usage(root, lib) if cid not in known})


def fields_payload(root: str, lib: str = LIB_OBJECTS) -> dict:
    """给界面/命令行共用的类别清单，带每个类别的框数、占用色与来源。"""
    fields = list_fields(root, lib)
    usage = class_id_usage(root, lib)
    rows = []
    for field in fields:
        class_id = int(field.get("class_id", -1))
        fallback = PALETTE[class_id % len(PALETTE)] if class_id >= 0 else PALETTE[0]
        rows.append({
            "id": field.get("id"), "class_id": class_id, "name": field.get("name"),
            "label": field.get("label") or field.get("name"), "prompt_hint": field.get("prompt_hint", ""),
            "source": field.get("source", ""),
            "color": str(field.get("color") or fallback),
            "created_at": field.get("created_at"), "updated_at": field.get("updated_at"),
            "boxes": int(usage.get(class_id, 0)),
        })
    return {"fields": rows, "next_class_id": load_registry(root, lib)["next_class_id"],
            "path": fields_path(root, lib), "orphans": orphan_class_ids(root, lib),
            "lib": normalize(lib), "lib_name": lib_name(lib), "max_fields": max_fields(lib)}


def _iter_prompt_labels(root: str):
    """旧调用名：等价于 ``_iter_box_labels(root, LIB_OBJECTS)``，提示词库的老代码还在用。"""
    return _iter_box_labels(root, LIB_OBJECTS)


def add_classes(
    root: str,
    names: Sequence[str],
    *,
    source: str = "",
    lib: str = LIB_DETECT,
    labels: Optional[Dict[str, str]] = None,
) -> dict:
    """把模型自带的一批类别名收进登记表（目标检测库的主路径）。

    已存在的类别只补来源与中文名，**类别号一律不动**：换权重时同名类别必须
    继续用同一个号，否则先前标好的框会集体指错东西。

    :param names: 模型给的英文类名列表（``DetectionBackend.class_names()`` 的值）
    :param source: 这批类别出自哪个权重，写进 ``source`` 给界面看
    :param labels: 手工指定的中文名，优先于对照表
    """
    from .classnames import zh_for

    added: List[dict] = []
    updated: List[dict] = []
    skipped: List[str] = []
    for raw in names:
        name = str(raw or "").strip()
        if not name:
            continue
        existed = find_field(root, name, lib) is not None
        label = str((labels or {}).get(name) or "").strip() or zh_for(name)
        try:
            saved = upsert_field(root, name=name, label=label, source=source, lib=lib)
        except ValueError as exc:
            skipped.append(f"{name}（{exc}）")
            continue
        (updated if existed else added).append(saved)
    payload = fields_payload(root, lib)
    payload.update({"added": len(added), "updated": len(updated), "skipped": skipped,
                    "added_names": [f.get("name") for f in added][:80]})
    return payload
