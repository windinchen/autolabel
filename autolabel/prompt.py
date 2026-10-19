"""提示词标注：用「字段名 + 描述提示词」让多模态大模型找画面里的物品，产出 YOLO detection 框。

和姿态标注的关系：

- 姿态标注产出 ``labels/``（``class cx cy w h`` + 17×3 关键点），提示词标注产出
  ``prompt_labels/``（``class cx cy w h``），两套标签分目录存放，互不覆盖；
- 类别表来自 ``autolabel/fields.py`` 的字段登记表，类别号 = 字段的 ``class_id``；
- 索引里另开一组 ``prompt_*`` 字段记复核状态，姿态的 ``status`` 不受影响。

本模块的检测框机制（读写标签、逐框质量标记、坐标导出、导出训练集）是**按库参数化**的：
默认 ``lib=objects`` 服务提示词标注，目标检测库（``lib=detect``）由
``autolabel/detect.py`` 带同样的参数复用同一套代码，不复制第二份。

坐标约定与 ``vlm.py`` 一致：内部全用 0-1 归一化，提示词里让模型回 0-1000 整数。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import fields as field_store
from .library import (
    LIB_DETECT,
    LIB_OBJECTS,
    field_key,
    label_path_for,
    labels_dir,
    lib_name,
    normalize as normalize_lib,
    spec,
)
from .dataset import (
    HUMAN_STATUSES,
    STATUS_APPROVED,
    STATUS_EDITED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    load_index,
    now_str,
    resolve_image,
    source_missing,
)

VALID_PROMPT_STATUS = {STATUS_UNREVIEWED, STATUS_APPROVED, STATUS_EDITED, STATUS_REJECTED}
COORD_SCALE = 1000.0
# 短边不足这个像素数的框基本没法训练，直接打标记让人复核
MIN_BOX_PIXEL = 12
# 同字段两个框重叠到这个程度就算重复标注
DUP_IOU = 0.9
DEFAULT_EXCLUDE_STATUS = {"rejected"}

PROMPT_FLAG_HINTS = {
    "no_fields": "还没有字段：先在「提示词标注」页新建要标的物品，再去跑标注",
    "none_found": "这张图一个框都没有：可能画面里确实没有，也可能整张漏标",
    "small_box": "有框太小（短边不足 12 像素），训练时容易学不动",
    "edge_box": "有框贴着画面边缘，目标可能被裁断",
    "dup_box": "同字段有两个高度重叠的框，重复标注",
    "unknown_class": "标签里有登记表里不存在的类别号（字段被删过或手工改过盘）",
    "prompt_stale": "标注用的字段/描述与当前设置不一致，建议重新标注",
    "vlm_failed": "多模态大模型这次没答上（网络/超时/解析失败），框是旧的或缺失",
    "manual": "人工补画或重画的框（不是模型给的），导出时置信度按 1 计",
}

#: 目标检测库的标记说明：框的质量标记与上面同一套，只有"整图级"的两条换了名字，
#: 因为失败原因是本地模型而不是多模态大模型。
DETECT_FLAG_HINTS = {
    "no_classes": "还没有类别：先在「目标检测」页从模型类别表里勾选要标的类别",
    "none_found": "这张图一个框都没有：可能画面里确实没有，也可能模型没认出来",
    "small_box": "有框太小（短边不足 12 像素），训练时容易学不动",
    "edge_box": "有框贴着画面边缘，目标可能被裁断",
    "dup_box": "同类别有两个高度重叠的框，重复标注",
    "unknown_class": "标签里有登记表里不存在的类别号（类别被删过或手工改过盘）",
    "model_stale": "标注用的权重/类别/阈值与当前设置不一致，建议重新标注",
    "model_failed": "这次推理失败（权重缺失/显存不足/读图失败），框是旧的或缺失",
    "masked": "这一轮出自分割权重：多边形轮廓只按外接水平框记录，想要轮廓得换标注方式",
    "manual": "人工补画或重画的框（不是模型给的），导出时置信度按 1 计",
}

OBJECT_COLUMNS = [
    "image", "status", "box_index", "class_id", "class_name", "display_name",
    "image_width", "image_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    "box_flags", "image_flags", "image_score", "time_sec", "video", "source",
]

OBJECT_COLUMN_DOC = {
    "image": "images/ 下的相对路径",
    "status": "提示词标注的复核状态 unreviewed/approved/edited/rejected",
    "box_index": "同一张图内的框序号，从 0 开始",
    "class_id": "YOLO detection 类别号（= 字段登记表里的 class_id）",
    "class_name": "字段英文类名，写进 data.yaml 的 names",
    "display_name": "字段中文显示名",
    "bbox_x1..y2": "边界框，像素坐标 xyxy",
    "bbox_cx..h": "边界框，归一化 cxcywh（与标签文件一致）",
    "box_flags": "这个框的质量标记（small_box/edge_box/dup_box/unknown_class）",
    "image_flags": "整图标记（none_found/prompt_stale/vlm_failed）",
    "image_score": "该图所有框的平均置信度，人工补画的框按 1 计",
}


def object_column_doc(lib: str = LIB_OBJECTS) -> dict:
    """列说明按库改口径：状态字段与整图标记的键名两个框类库不一样。

    照抄提示词库那份给目标检测库，``meta.json`` 会告诉用户标记叫
    ``prompt_stale``/``vlm_failed``，而 CSV 里躺着的是 ``model_stale``/``model_failed``。
    """
    name = normalize_lib(lib)
    doc = dict(OBJECT_COLUMN_DOC)
    if name == LIB_OBJECTS:
        return doc
    doc["status"] = "目标检测的复核状态 unreviewed/approved/edited/rejected"
    doc["class_id"] = "YOLO detection 类别号（= 本库类别登记表里的 class_id，同名永不换号）"
    doc["image_flags"] = "整图标记（none_found/model_stale/model_failed/masked）"
    doc["image_score"] = "该图所有框的平均置信度（模型给的分数），人工补画的框按 1 计"
    return doc


# --------------------------------------------------------------------------- #
# 数据结构与标签读写
# --------------------------------------------------------------------------- #

@dataclass
class Detection:
    """一个检测框，坐标按图片宽高归一化。"""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    score: float = 0.0
    flags: List[str] = field(default_factory=list)

    @property
    def xyxy_norm(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)

    def xyxy(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy_norm
        return (x1 * width, y1 * height, x2 * width, y2 * height)

    def row(self) -> List[float]:
        return [float(self.cls), self.cx, self.cy, self.w, self.h]

    def payload(self, width: int, height: int) -> dict:
        x1, y1, x2, y2 = self.xyxy(width, height)
        return {"cls": int(self.cls), "cx": round(self.cx, 6), "cy": round(self.cy, 6),
                "w": round(self.w, 6), "h": round(self.h, 6), "score": round(self.score, 4),
                "flags": list(self.flags),
                "xyxy": [round(v, 2) for v in (x1, y1, x2, y2)]}


def read_detections(root: str, image_rel: str, lib: str = LIB_OBJECTS) -> List[Detection]:
    path = label_path_for(root, image_rel, lib)
    return [Detection(cls=cls, cx=cx, cy=cy, w=w, h=h)
            for cls, cx, cy, w, h in field_store.read_label_rows(path)]


def apply_manual_marks(detections: List[Detection], indices: Sequence[int]) -> List[Detection]:
    """把索引里记录的人工补画框标回 ``manual``。

    YOLO detection 的 5 列标签不落 score 和 flags，重读磁盘后框会「失忆」，
    看起来像模型自己画的、置信度还是 0，所以人工痕迹单独记在索引里再贴回来。
    """
    for index in indices or []:
        try:
            position = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= position < len(detections):
            detection = detections[position]
            detection.flags = sorted(set(list(detection.flags) + ["manual"]))
            if detection.score <= 0:
                detection.score = 1.0
    return detections


def write_detections(root: str, image_rel: str, detections: Sequence[Detection],
                     lib: str = LIB_OBJECTS) -> str:
    return field_store.write_label_rows(label_path_for(root, image_rel, lib),
                                       [d.row() for d in detections])


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #

def build_prompt(fields: Sequence[dict], extra: str = "") -> str:
    """把字段表拼成给多模态大模型的标注指令。"""
    lines = ["你是图像标注助手。请在图里找出下列目标，每个目标给一个贴边的边界框。", "目标字段："]
    for field_row in fields:
        name = field_row.get("name", "")
        label = str(field_row.get("label") or "").strip()
        hint = str(field_row.get("prompt_hint") or "").strip()
        head = f"- {name}"
        if label and label != name:
            head += f"（{label}）"
        if hint:
            head += f"：{hint}"
        lines.append(head)
    lines += [
        f"坐标用 0-{int(COORD_SCALE)} 的整数，表示相对图片宽高的位置：左上角 (0,0)，右下角 (1000,1000)。",
        "field 字段必须严格照抄上面给出的英文字段名，不要自己造词、不要翻译。",
        "框要贴着目标边缘；同一个目标只标一个框；目标之间不要互相重叠成两个框。",
        "看不清、不确定、只是像的目标不要标；被遮挡超过一半的目标不要标；画面里没有就返回空列表。",
        "只输出 JSON，不要任何解释文字，格式：",
        '{"objects":[{"field":"字段名","bbox":[x1,y1,x2,y2]}]}',
    ]
    extra = str(extra or "").strip()
    if extra:
        lines.insert(1, f"补充说明：{extra}")
    return "\n".join(lines)


def signature_for(fields: Sequence[dict], profile_id: str, extra: str = "") -> str:
    """字段/描述/后端任一变了就换签名：用来判断哪些图需要重标。"""
    payload = json.dumps({
        "fields": [[int(f.get("class_id", -1)), f.get("name", ""),
                    str(f.get("prompt_hint") or "")] for f in fields],
        "profile": str(profile_id or ""), "extra": str(extra or "").strip(),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 解析模型回复
# --------------------------------------------------------------------------- #

def _field_matchers(fields: Sequence[dict]) -> List[Tuple[str, dict]]:
    """建「可接受写法 -> 字段」表：英文名、中文名、去分隔符的小写名都能对上。"""
    table: List[Tuple[str, dict]] = []
    for row in fields:
        keys = {str(row.get("name", "")), str(row.get("label", "")), str(row.get("id", ""))}
        for key in keys:
            clean = str(key).strip().lower()
            if clean:
                table.append((clean, row))
        squeezed = str(row.get("name", "")).strip().lower().replace("_", "").replace("-", "")
        if squeezed:
            table.append((squeezed, row))
    return table


def _match_field(raw: object, matchers: Sequence[Tuple[str, dict]], fallback: Optional[dict]) -> Optional[dict]:
    text = str(raw or "").strip().lower()
    if not text:
        return fallback
    for key, row in matchers:
        if text == key:
            return row
    for key, row in matchers:
        if len(key) >= 3 and (key in text or text in key):
            return row
    return fallback


def _coord_scale(values: Sequence[float]) -> float:
    """兼容模型回 0-1 / 0-1000 / 像素三种写法。"""
    peak = max((abs(float(v)) for v in values), default=0.0)
    if peak <= 1.5:
        return 1.0
    if peak <= 2000.0:
        return COORD_SCALE
    return peak if peak > 0 else 1.0


XYXY_KEYS = ("bbox", "box", "bbox_xyxy", "xyxy", "rect", "coordinates", "location")
XYWH_KEYS = ("bbox_xywh", "xywh", "cxcywh", "box_cxcywh")


def _box_values(entry: dict) -> Optional[List[float]]:
    """取原始坐标四元组（未归一化），统一成 xyxy；xywh 写法先换算。"""
    raw = next((entry[key] for key in XYXY_KEYS + XYWH_KEYS if key in entry), None)
    as_xywh = any(key in entry for key in XYWH_KEYS) and not any(key in entry for key in XYXY_KEYS)
    if isinstance(raw, dict):
        if all(k in raw for k in ("x1", "y1", "x2", "y2")):
            return [float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"])]
        if all(k in raw for k in ("cx", "cy", "w", "h")):
            cx, cy, w, h = (float(raw["cx"]), float(raw["cy"]), float(raw["w"]), float(raw["h"]))
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        return None
    if isinstance(raw, (list, tuple)):
        flat: List[float] = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                flat.extend(float(v) for v in item)
            else:
                flat.append(float(item))
        if len(flat) >= 4:
            values = flat[:4]
            return list(_from_cxcywh(values)) if as_xywh else values
    return None


def _from_cxcywh(values: Sequence[float]) -> Tuple[float, float, float, float]:
    cx, cy, w, h = [float(v) for v in values]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def parse_objects(text: str, fields: Sequence[dict]) -> Tuple[List[Detection], List[str]]:
    """把模型回复解析成检测框；返回 ``(框列表, 没认出来的字段名)``。"""
    from .vlm import VLMError, json_block

    try:
        block = json_block(str(text or ""))
    except VLMError:
        # 模型回了「画面里没有目标」这类纯文字答复，不算失败，就是零个框
        return [], []

    matchers = _field_matchers(fields)
    only = list(fields)[0] if len(fields) == 1 else None
    guesses: List[str] = []
    detections: List[Detection] = []
    for entry in _object_entries(block):
        raw_field = next((entry[k] for k in ("field", "label", "class", "class_name", "name", "category")
                          if entry.get(k) not in (None, "")), "")
        row = _match_field(raw_field, matchers, only)
        if row is None:
            if str(raw_field or "").strip():
                guesses.append(str(raw_field).strip())
            continue
        values = _box_values(entry)
        if not values:
            continue
        scale = _coord_scale(values)
        x1, y1, x2, y2 = [float(v) / scale for v in values]
        box = _clamp_box(x1, y1, x2, y2)
        if box is None:
            continue
        score = _entry_score(entry)
        detections.append(Detection(cls=int(row.get("class_id", 0)), cx=box[0], cy=box[1],
                                   w=box[2], h=box[3], score=score))
    return dedupe(detections), guesses


def _object_entries(block: str) -> List[dict]:
    """从 JSON 主体里取对象列表，兼容 ``{"objects":[...]}``、裸数组、``{"detections":[...]}``。"""
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return _scan_entries(block)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("objects", "detections", "items", "results", "boxes", "labels"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if _box_values(data) or data.get("field") or data.get("name"):
            return [data]
    return []


def _scan_entries(block: str) -> List[dict]:
    """整段 JSON 解析失败时，逐个抠 ``{...}`` 小对象救回能用的框。"""
    out: List[dict] = []
    depth, start = 0, -1
    for index, char in enumerate(block):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    row = json.loads(block[start:index + 1])
                    if isinstance(row, dict):
                        out.append(row)
                except json.JSONDecodeError:
                    pass
    return out


def _entry_score(entry: dict) -> float:
    for key in ("score", "confidence", "conf", "prob"):
        if isinstance(entry.get(key), (int, float)):
            return max(0.0, min(1.0, float(entry[key])))
    return 0.5  # 大模型不回报置信度，给个中性占位分


def _clamp_box(x1: float, y1: float, x2: float, y2: float) -> Optional[Tuple[float, float, float, float]]:
    lo_x, hi_x = max(0.0, min(x1, x2)), min(1.0, max(x1, x2))
    lo_y, hi_y = max(0.0, min(y1, y2)), min(1.0, max(y1, y2))
    w, h = hi_x - lo_x, hi_y - lo_y
    if w <= 0.002 or h <= 0.002:
        return None
    return (round((lo_x + hi_x) / 2, 6), round((lo_y + hi_y) / 2, 6), round(w, 6), round(h, 6))


def dedupe(detections: Sequence[Detection]) -> List[Detection]:
    """同字段高度重叠的框只留面积最大的那个，并把 dup_box 标记留在被丢的框上（供统计）。"""
    kept: List[Detection] = []
    for item in sorted(detections, key=lambda d: d.w * d.h, reverse=True):
        dup = next((other for other in kept if other.cls == item.cls
                    and iou(other, item) >= DUP_IOU), None)
        if dup is None:
            kept.append(item)
        else:
            dup.flags = sorted(set(dup.flags) | {"dup_box"})
    return kept


def iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy_norm
    bx1, by1, bx2, by2 = b.xyxy_norm
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# 质量标记
# --------------------------------------------------------------------------- #

def box_flags(detections: Sequence[Detection], width: int, height: int,
              known: Optional[Sequence[int]] = None) -> List[List[str]]:
    """逐框质量标记：过小、贴边、重复、类别号失联。"""
    known_ids = None if known is None else set(known)
    out: List[List[str]] = []
    for index, item in enumerate(detections):
        flags = list(item.flags)
        if width and height:
            x1, y1, x2, y2 = item.xyxy(width, height)
            if min(x2 - x1, y2 - y1) < MIN_BOX_PIXEL:
                flags.append("small_box")
            if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
                flags.append("edge_box")
        else:
            if min(item.w, item.h) * 1000 < MIN_BOX_PIXEL:
                flags.append("small_box")
        for other in detections[index + 1:]:
            if other.cls == item.cls and iou(item, other) >= DUP_IOU:
                flags.append("dup_box")
        if known_ids is not None and item.cls not in known_ids:
            flags.append("unknown_class")
        out.append(sorted(set(flags)))
    return out


def image_flags(detections: Sequence[Detection], flags_per_box: Sequence[Sequence[str]],
                signature: str = "", item_signature: str = "", failed: bool = False,
                stale_flag: str = "prompt_stale", failed_flag: str = "vlm_failed") -> List[str]:
    """整图级标记，和姿态那套 flags 分开命名，避免界面筛选串味。

    ``stale_flag``/``failed_flag`` 按库换名：提示词库是 ``prompt_stale``/``vlm_failed``，
    目标检测库是 ``model_stale``/``model_failed``——同样是"这批框过期了"，
    但原因和补救动作完全不同，界面要能一眼分开说。
    """
    flags: List[str] = []
    if not detections:
        flags.append("none_found")
    if any("small_box" in f for f in flags_per_box):
        flags.append("small_box")
    if any("edge_box" in f for f in flags_per_box):
        flags.append("edge_box")
    if any("dup_box" in f for f in flags_per_box):
        flags.append("dup_box")
    if any("unknown_class" in f for f in flags_per_box):
        flags.append("unknown_class")
    if any("masked" in f for f in flags_per_box):
        flags.append("masked")
    if signature and item_signature and signature != item_signature:
        flags.append(stale_flag)
    if failed:
        flags.append(failed_flag)
    return sorted(set(flags))


def prompt_priority(item: dict, lib: str = LIB_OBJECTS) -> int:
    """检测框类库的复核优先级（越大越该先看）；``lib`` 决定读哪一组状态字段。

    两个框库的标记名有一部分是同义的（``vlm_failed``/``model_failed``、
    ``prompt_stale``/``model_stale``），这里一次认全，避免新库进来时漏项。
    """
    status = item.get(field_key(lib, "status"))
    flags = set(item.get(field_key(lib, "flags")) or [])
    score = float(item.get(field_key(lib, "score")) or 0.0)
    if status == STATUS_REJECTED:
        return 0
    weight = 60 if flags & {"vlm_failed", "model_failed"} else 0
    weight += 40 if "unknown_class" in flags else 0
    weight += 30 if flags & {"dup_box", "small_box"} else 0
    weight += 20 if flags & {"prompt_stale", "model_stale"} else 0
    weight += 10 if "none_found" in flags else 0
    weight += 5 if "edge_box" in flags else 0
    if status == STATUS_UNREVIEWED:
        weight += 3
    weight += max(0.0, 30 - score * 30)
    return int(weight)


# --------------------------------------------------------------------------- #
# 标注流水线
# --------------------------------------------------------------------------- #

def label_image(root: str, image_rel: str, fields: Sequence[dict], profile: dict,
                *, extra_hint: str = "") -> dict:
    """问一次多模态大模型，把结果写成 ``prompt_labels/<同名>.txt``。"""
    import numpy as np

    from . import vlm as vlm_module

    if not fields:
        raise ValueError("还没有可标注的字段：先新建至少一个字段（autolabel fields --add bottle=描述）")
    source = resolve_image(root, image_rel)
    if not os.path.exists(source):
        raise FileNotFoundError(image_rel)
    from .dataset import _load_rgb

    image = np.asarray(_load_rgb(source), dtype=np.uint8)
    prompt = build_prompt(fields, extra_hint)
    data_url = vlm_module.encode_image(image, max_side=int(profile.get("image_limit") or 1024))
    text = vlm_module.chat(profile, prompt, data_url)
    detections, guesses = parse_objects(text, fields)
    return {"detections": detections, "unknown_fields": guesses, "reply_chars": len(text),
            "prompt_chars": len(prompt)}


def resolve_profile(wanted: str = "") -> dict:
    """按名称/ID 找多模态配置；留空退到默认配置，再退到「只配了一条就用它」。"""
    from . import models as model_manager

    text = str(wanted or "").strip()
    config = model_manager.load_vlm()
    profiles = config.get("profiles") or []
    profile = model_manager.find_profile(text) if text else None
    if text and profile is None:
        options = "、".join(str(p.get("name", "")) for p in profiles) or "（还没有）"
        raise ValueError(f"找不到多模态配置 {text}；可选：{options}（在「模型管理」页或 "
                         "autolabel models --add-vlm 里添加）")
    if profile is None:
        default_id = str(config.get("default") or "")
        profile = next((p for p in profiles if str(p.get("id")) == default_id), None)
    if profile is None and len(profiles) == 1:
        profile = profiles[0]
    if profile is None:
        raise ValueError("没有可用的多模态大模型配置：先在「模型管理」页添加本地（Ollama/LM Studio/"
                         "vLLM）或云端 OpenAI 兼容端点配置，再跑提示词标注")
    return model_manager.get_profile(profile["id"])


def run_prompt_labeling(
    root: str,
    *,
    fields: Optional[Sequence[dict]] = None,
    profile: Optional[dict] = None,
    images: Optional[Sequence[str]] = None,
    limit: int = 0,
    overwrite: bool = False,
    extra_hint: str = "",
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    verbose: bool = True,
) -> dict:
    """批量提示词标注：逐图问模型、写框、更新索引的 ``prompt_*`` 字段。"""
    picked = list(fields) if fields is not None else field_store.list_fields(root)
    if not picked:
        raise ValueError("还没有可标注的字段：先新建字段再跑（界面「提示词标注」或 autolabel fields --add）")
    profile = resolve_profile(str((profile or {}).get("id") or ""))

    idx = load_index(root, LIB_OBJECTS)
    targets = [str(rel) for rel in images] if images else [rel for rel, _ in idx.ordered_items()]
    signature = signature_for(picked, profile["id"], extra_hint)
    protected = 0
    if not overwrite:
        # 重复跑只补新图：同签名标过的、人工接受过 / 改过的、人工画过框的都不动，
        # 否则模型一句"没找到"就把人花过时间的框抹掉了（实测踩过）
        kept = []
        for rel in targets:
            item = idx.get(rel) or {}
            relabeled = (item.get("prompt_signature") == signature
                         and item.get("prompt_objects") is not None)
            human = (item.get("prompt_status") in HUMAN_STATUSES
                     or int(item.get("prompt_manual") or 0) > 0)
            if relabeled or human:
                protected += 1
                continue
            kept.append(rel)
        targets = kept
    if limit and limit > 0:
        targets = targets[:limit]

    stats = {"labeled": 0, "boxes": 0, "failed": 0, "skipped": 0, "protected": protected,
             "images": len(targets), "unknown_fields": [],
             "fields": [f.get("name") for f in picked],
             "profile": profile.get("name", ""), "last_error": ""}
    known_ids = [int(f.get("class_id", -1)) for f in picked]
    for done, rel in enumerate(targets, start=1):
        if should_stop and should_stop():
            stats["skipped"] += len(targets) - done + 1
            break
        item = idx.get(rel) or {}
        try:
            result = label_image(root, rel, picked, profile, extra_hint=extra_hint)
        except Exception as exc:  # noqa: BLE001 单图失败不该中断整批
            stats["failed"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {exc}"
            idx.upsert(rel, prompt_flags=sorted(set(list(item.get("prompt_flags") or [])) | {"vlm_failed"}),
                       prompt_signature=signature, prompt_error=stats["last_error"][:300])
            idx.save()
            if verbose:
                print(f"[warn] {rel} 标注失败：{stats['last_error']}")
            if progress:
                progress(done, len(targets))
            continue
        detections = result["detections"]
        per_box = box_flags(detections, int(item.get("width") or 0), int(item.get("height") or 0), known_ids)
        write_detections(root, rel, detections)
        flags = image_flags(detections, per_box, signature=signature,
                            item_signature=str(item.get("prompt_signature") or ""))
        scores = [d.score for d in detections] or [0.0]
        status = item.get("prompt_status") or STATUS_UNREVIEWED
        if status not in VALID_PROMPT_STATUS:
            status = STATUS_UNREVIEWED
        if status in (STATUS_APPROVED, STATUS_EDITED):
            status = STATUS_EDITED  # 重标过的人工结果要重新确认
        idx.upsert(rel, prompt_objects=len(detections), prompt_status=status,
                   prompt_flags=flags, prompt_signature=signature, prompt_error="",
                   prompt_score=round(sum(scores) / len(scores), 4), prompt_labeled_at=now_str(),
                   prompt_profile=profile.get("name", ""),
                   # 这批框全部出自模型：人工标记与类别号都要跟着重写，否则索引里会
                   # 留下指向不存在框的人工标记（撤销与导出都会被带偏）
                   prompt_manual=0, prompt_manual_boxes=[],
                   prompt_classes=sorted({detection.cls for detection in detections}))
        idx.save()
        stats["labeled"] += 1
        stats["boxes"] += len(detections)
        stats["unknown_fields"].extend(result["unknown_fields"])
        if verbose:
            print(f"[ok] {rel}：{len(detections)} 个框"
                  + (f"，模型还回了不认识的字段 {'、'.join(result['unknown_fields'])}"
                     if result["unknown_fields"] else ""))
        if progress:
            progress(done, len(targets))

    stats["unknown_fields"] = sorted(set(stats["unknown_fields"]))
    if stats["failed"] and stats["labeled"] == 0:
        raise RuntimeError(f"{stats['failed']} 张全部失败，最后一条错误：{stats['last_error'] or '未知'}")
    return stats


# --------------------------------------------------------------------------- #
# 汇总与坐标导出
# --------------------------------------------------------------------------- #

def prompt_summary(root: str, lib: str = LIB_OBJECTS) -> dict:
    """本库总览（只数索引与标签行数，不额外解码图片，成本低）。"""
    lib = normalize_lib(lib)
    idx = load_index(root, lib)
    counts = {"images": len(idx.items), "labeled": 0, "unreviewed": 0, "reviewed": 0}
    status_counts: Dict[str, int] = {}
    flag_counts: Dict[str, int] = {}
    per_field: Dict[str, int] = {}
    boxes = 0
    names = {int(f.get("class_id", -1)): f.get("name", "") for f in field_store.list_fields(root, lib)}
    for _rel, item in idx.ordered_items():
        if item.get(field_key(lib, "objects")) is None:
            counts["unreviewed"] += 1
            continue
        counts["labeled"] += 1
        status = str(item.get(field_key(lib, "status")) or STATUS_UNREVIEWED)
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in (STATUS_APPROVED, STATUS_EDITED):
            counts["reviewed"] += 1
        for flag in item.get(field_key(lib, "flags")) or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    for _rel, rows in field_store._iter_box_labels(root, lib):
        boxes += len(rows)
        for row in rows:
            name = names.get(row[0]) or f"class_{row[0]}"
            per_field[name] = per_field.get(name, 0) + 1
    return {**counts, "boxes": boxes, "status_counts": status_counts,
            "flag_counts": flag_counts, "per_field": per_field, "per_class": per_field,
            "lib": lib, "lib_name": lib_name(lib),
            "flag_hints": DETECT_FLAG_HINTS if lib == LIB_DETECT else PROMPT_FLAG_HINTS,
            "orphans": field_store.orphan_class_ids(root, lib)}


def collect_object_records(root: str, exclude_status: Sequence[str] = ("rejected",),
                           lib: str = LIB_OBJECTS) -> Tuple[List[dict], dict]:
    """把本库标签目录解析成带像素坐标的记录，供 CSV/JSONL 与界面复用。"""
    lib = normalize_lib(lib)
    idx = load_index(root, lib)
    names = {int(f.get("class_id", -1)): f for f in field_store.list_fields(root, lib)}

    def key(name: str) -> str:
        return field_key(lib, name)

    skip = set(exclude_status)
    records: List[dict] = []
    skipped: List[dict] = []
    for rel, item in idx.ordered_items():
        status = str(item.get(key("status")) or STATUS_UNREVIEWED)
        if status in skip:
            skipped.append({"image": rel, "reason": f"{key('status')}={status}"})
            continue
        if item.get(key("objects")) is None:
            skipped.append({"image": rel, "reason": f"未做{lib_name(lib)}"})
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        detections = apply_manual_marks(read_detections(root, rel, lib),
                                       item.get(key("manual_boxes")) or [])
        per_box = box_flags(detections, width, height, list(names))
        objects = []
        for index, detection in enumerate(detections):
            field_row = names.get(detection.cls) or {}
            payload = detection.payload(width, height)
            objects.append({
                "index": index, "cls": detection.cls,
                "class_name": field_row.get("name") or f"class_{detection.cls}",
                "display_name": field_row.get("label") or field_row.get("name") or "",
                "bbox_xyxy": payload["xyxy"],
                "bbox_cxcywh_norm": [detection.cx, detection.cy, detection.w, detection.h],
                "flags": per_box[index],
            })
        records.append({
            "image": rel, "status": status, "width": width, "height": height,
            "image_score": item.get(key("score")), "image_flags": list(item.get(key("flags")) or []),
            # 这批框出自谁：提示词库是多模态配置名，目标检测库是权重文件名
            "prompt_profile": item.get("prompt_profile") or "",
            "labeled_by": item.get(key("profile")) or item.get(key("model")) or "",
            "time_sec": item.get("time_sec"),
            "video": item.get("video"), "source": item.get("source"), "objects": objects,
        })
    meta = {
        "lib": lib, "lib_name": lib_name(lib),
        "fields": [{"class_id": cid, "name": row.get("name"), "label": row.get("label"),
                    "prompt_hint": row.get("prompt_hint", ""),
                    "source": row.get("source", "")} for cid, row in sorted(names.items())],
        "source_dataset": os.path.abspath(root), "generated_at": now_str(),
        "filter": {"exclude_status": sorted(skip)}, "columns": object_column_doc(lib),
        "skipped": skipped,
    }
    return records, meta


def _write_object_wide(path: str, records: Sequence[dict]) -> int:
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        import csv

        writer = csv.DictWriter(fh, fieldnames=OBJECT_COLUMNS)
        writer.writeheader()
        for rec in records:
            for obj in rec["objects"]:
                writer.writerow({
                    "image": rec["image"], "status": rec["status"], "box_index": obj["index"],
                    "class_id": obj["cls"], "class_name": obj["class_name"],
                    "display_name": obj["display_name"], "image_width": rec["width"],
                    "image_height": rec["height"],
                    "bbox_x1": obj["bbox_xyxy"][0], "bbox_y1": obj["bbox_xyxy"][1],
                    "bbox_x2": obj["bbox_xyxy"][2], "bbox_y2": obj["bbox_xyxy"][3],
                    "bbox_cx": obj["bbox_cxcywh_norm"][0], "bbox_cy": obj["bbox_cxcywh_norm"][1],
                    "bbox_w": obj["bbox_cxcywh_norm"][2], "bbox_h": obj["bbox_cxcywh_norm"][3],
                    "box_flags": "|".join(obj["flags"]), "image_flags": "|".join(rec["image_flags"]),
                    "image_score": rec["image_score"] if rec["image_score"] is not None else "",
                    "time_sec": rec["time_sec"] if rec["time_sec"] is not None else "",
                    "video": rec["video"] or "", "source": rec["source"] or "",
                })
                rows += 1
    return rows


def write_object_coordinate_files(root: str, out: Optional[str] = None, tag: Optional[str] = None,
                                  exclude_status: Sequence[str] = ("rejected",),
                                  lib: str = LIB_OBJECTS) -> dict:
    """生成本库的坐标文件（CSV + JSONL + meta + summary）。"""
    from .coords import coords_dir

    lib = normalize_lib(lib)
    records, meta = collect_object_records(root, exclude_status=exclude_status, lib=lib)
    stamp = tag or now_str().replace(" ", "_").replace(":", "")
    out_root = os.path.abspath(out or os.path.join(coords_dir(root), f"{stamp}-{spec(lib)['prefix']}"))
    os.makedirs(out_root, exist_ok=True)
    files = {
        "objects.csv": os.path.join(out_root, "objects.csv"),
        "objects.jsonl": os.path.join(out_root, "objects.jsonl"),
    }
    rows = _write_object_wide(files["objects.csv"], records)
    with open(files["objects.jsonl"], "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta["files"] = list(files)
    summary = {
        "generated_at": meta["generated_at"], "source_dataset": meta["source_dataset"],
        "images": len(records), "images_with_boxes": sum(1 for r in records if r["objects"]),
        "images_empty": sum(1 for r in records if not r["objects"]), "boxes": rows,
        "per_field": {}, "status_counts": {}, "filter": meta["filter"],
    }
    for rec in records:
        summary["status_counts"][rec["status"]] = summary["status_counts"].get(rec["status"], 0) + 1
        for obj in rec["objects"]:
            summary["per_field"][obj["class_name"]] = summary["per_field"].get(obj["class_name"], 0) + 1
    summary["lib"] = meta["lib"]
    summary["lib_name"] = meta["lib_name"]
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    files["meta.json"] = os.path.join(out_root, "meta.json")
    files["summary.json"] = os.path.join(out_root, "summary.json")
    return {"out": out_root, "files": files, "images": len(records), "boxes": rows}


OBJECT_DOWNLOAD_NAMES = {"csv": "objects.csv", "jsonl": "objects.jsonl"}


def render_object_coords(root: str, fmt: str = "csv",
                         exclude_status: Sequence[str] = ("rejected",),
                         lib: str = LIB_OBJECTS) -> Tuple[str, bytes]:
    """按需生成单个坐标文件（不落盘），给界面的「下载坐标」用。"""
    if fmt not in OBJECT_DOWNLOAD_NAMES:
        raise ValueError(f"未知格式: {fmt}（可选 {', '.join(OBJECT_DOWNLOAD_NAMES)}）")
    lib = normalize_lib(lib)
    tmp = tempfile.mkdtemp(prefix="autolabel_%s_" % spec(lib)["prefix"])
    try:
        result = write_object_coordinate_files(os.path.abspath(root), out=tmp,
                                               exclude_status=exclude_status, lib=lib)
        with open(result["files"][OBJECT_DOWNLOAD_NAMES[fmt]], "rb") as fh:
            body = fh.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.basename(os.path.abspath(root)))
    prefix = spec(lib)["prefix"]
    name = OBJECT_DOWNLOAD_NAMES[fmt] if prefix == "prompt" else f"{prefix}-{OBJECT_DOWNLOAD_NAMES[fmt]}"
    return f"{safe or 'dataset'}-{name}", body


def export_detection_dataset(
    root: str,
    out: Optional[str] = None,
    *,
    val_ratio: float = 0.2,
    seed: int = 7,
    exclude_status: Sequence[str] = ("rejected",),
    max_empty_ratio: float = 0.15,
    mode: str = "symlink",
    tag: Optional[str] = None,
    lib: str = LIB_OBJECTS,
) -> dict:
    """导出可直接喂 ``yolo detect train`` 的目录（train/val + data.yaml）。

    类别号会被重映射成连续的 0..N-1：登记表里删过字段时，标签里的 ``class_id``
    可能不连续，而 YOLO 要求 ``names`` 的键从 0 连续递增。
    """
    from .export import (MISSING_SOURCE_REASON, _place, content_fingerprint,
                         find_reusable_export, prune_stale, rank_for_split, skipped_summary,
                         split_val_set, unique_export_dir, unique_stem, yaml_comment, yaml_scalar)

    lib = normalize_lib(lib)
    records, meta = collect_object_records(root, exclude_status=exclude_status, lib=lib)
    # 与姿态库同一口径：源图已经不在磁盘上的帧不进导出集（宁可少一帧，也不能让
    # 训练集里躺一张读不出来的图），原因照样写进 skipped 给用户看。
    missing = [rec["image"] for rec in records
               if source_missing(root, str(rec["image"] or ""))]
    if missing:
        gone = set(missing)
        records = [rec for rec in records if rec["image"] not in gone]
    used = sorted({obj["cls"] for rec in records for obj in rec["objects"]})
    if not used:
        raise ValueError(f"还没有任何{lib_name(lib)}框可导出：先跑一次标注"
                         f"（界面「{lib_name(lib)}」，命令行 autolabel "
                         f"{'prompt' if lib == LIB_OBJECTS else 'detect'}）"
                         + (f"；另有 {len(missing)} 帧的源文件在 images/ 里找不到了" if missing else ""))
    remap = {cls: index for index, cls in enumerate(used)}
    names = {row["class_id"]: row for row in meta["fields"]}

    labeled = [rec for rec in records if rec["objects"]]
    empty = [rec for rec in records if not rec["objects"]]
    # 与姿态库同一套稳定划分，理由见 export.split_val_set()
    labeled = rank_for_split(labeled, seed, key=lambda rec: rec["image"])
    empty = rank_for_split(empty, seed, key=lambda rec: rec["image"])
    keep_empty = int(len(labeled) * max_empty_ratio / max(1e-6, 1 - max_empty_ratio)) if labeled else len(empty)
    keep_empty = max(0, keep_empty)
    empty_dropped = empty[keep_empty:]
    empty = empty[:keep_empty]

    selected = labeled + empty
    val_set = split_val_set([rec["image"] for rec in selected], val_ratio, seed)
    fingerprint = content_fingerprint(
        [(rec["image"], "\n".join(
            "{}:{:.6f},{:.6f},{:.6f},{:.6f}".format(
                obj["cls"], *obj["bbox_cxcywh_norm"]) for obj in rec["objects"]).encode("utf-8"))
         for rec in selected],
        params=json.dumps({"val_ratio": val_ratio, "max_empty_ratio": max_empty_ratio,
                           "mode": mode, "seed": seed, "exclude": sorted(exclude_status),
                           "used": used, "filter": meta.get("filter"), "lib": lib},
                          sort_keys=True, default=str))
    # 与姿态库同理：内容没变就复用最近那份，别在 exports/ 里堆重复目录
    kind = "detect" if lib == LIB_OBJECTS else "yolo_detect"
    meta_name = "prompt_meta.json" if lib == LIB_OBJECTS else "yolo_meta.json"
    suffix = "detect" if lib == LIB_OBJECTS else "yolo-detect"
    if not out and not tag:
        reused = find_reusable_export(root, fingerprint, kind=kind)
        if reused is not None:
            return reused
    tag = tag or now_str().replace(" ", "_").replace(":", "")
    if out:
        out_root = os.path.abspath(out)
    else:
        out_root = unique_export_dir(os.path.abspath(os.path.join(root, "exports", f"{tag}-{suffix}")))
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0, "boxes": {"train": 0, "val": 0}, "empty": len(empty),
              "empty_dropped": len(empty_dropped), "dropped": 0, "missing_source": len(missing)}
    written: Dict[str, set] = {"train": set(), "val": set()}
    taken: Dict[str, str] = {}
    for rec in selected:
        split = "val" if rec["image"] in val_set else "train"
        from .export import export_stem

        stem = unique_stem(taken, export_stem(rec["image"]), rec["image"])
        dst_img = os.path.join(out_root, "images", split, stem + os.path.splitext(rec["image"])[1])
        dst_lbl = os.path.join(out_root, "labels", split, stem + ".txt")
        _place(resolve_image(root, rec["image"]), dst_img, mode)
        lines: List[str] = []
        for obj in rec["objects"]:
            if obj["cls"] not in remap:
                counts["dropped"] += 1
                continue
            cx, cy, w, h = obj["bbox_cxcywh_norm"]
            lines.append(field_store.format_label_row([remap[obj["cls"]], cx, cy, w, h]))
        with open(dst_lbl, "w", encoding="utf-8") as fh:  # 永远写 5 列，删掉的类别不留尾巴
            fh.write("".join(line + "\n" for line in lines))
        counts[split] += 1
        counts["boxes"][split] += len(lines)
        written[split].add(stem)
    # 与姿态库一致：重复导到同一个目录时，这一版不要的帧得从磁盘上清掉
    counts["pruned"] = prune_stale(out_root, written)

    lines = [f"# 由自动标注系统生成 on {now_str()}（{lib_name(lib)} / YOLO detection）",
             f"path: {out_root}", "train: images/train", "val: images/val"]
    labels = []
    for cls in used:
        field_row = names.get(cls) or {}
        name = str(field_row.get("name") or f"class_{cls}")
        labels.append("# names 显示名：%s = %s"
                      % (yaml_comment(name), yaml_comment(field_row.get("label") or name)))
    lines.append("names:")
    for cls in used:
        field_row = names.get(cls) or {}
        label = str(field_row.get("name") or f"class_{cls}")
        # 带空格的类别名（COCO 的 potted plant 之类）在 YAML 里是合法明文，
        # 但含 : # 等符号就会解析错位，交给 export.yaml_scalar 统一判断
        lines.append(f"  {remap[cls]}: {yaml_scalar(label)}")
    data_yaml = "\n".join(lines[:1] + labels + lines[1:]) + "\n"
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(data_yaml)
    with open(os.path.join(out_root, meta_name), "w", encoding="utf-8") as fh:
        json.dump({"generated_at": now_str(), "source_dataset": meta["source_dataset"],
                   "lib": lib, "lib_name": lib_name(lib),
                   "fields": meta["fields"], "class_remap": {str(k): v for k, v in remap.items()},
                   "counts": counts, "filter": meta["filter"],
                   "max_empty_ratio": max_empty_ratio,
                   # 被比例上限压掉的空帧只在这里留痕，不然用户只看到张数变少
                   "empty_dropped_images": [rec["image"] for rec in empty_dropped][:200],
                   "fingerprint": fingerprint}, fh, ensure_ascii=False, indent=2)
    reasons = [(str(row.get("image") or ""), str(row.get("reason") or ""))
               for row in (meta.get("skipped") or []) if isinstance(row, dict)]
    reasons += [(rec["image"], f"背景图超出比例上限（上限 {max_empty_ratio:.0%}）")
                for rec in empty_dropped]
    reasons += [(rel, MISSING_SOURCE_REASON) for rel in missing]
    return {"out": out_root, "images": len(selected), "train": counts["train"], "val": counts["val"],
            "skipped": len(reasons), "skipped_note": skipped_summary(reasons),
            "boxes": counts["boxes"], "classes": [
                {"class_id": cls, "export_id": remap[cls],
                 "name": (names.get(cls) or {}).get("name") or f"class_{cls}"} for cls in used],
            "counts": counts, "data_yaml": os.path.join(out_root, "data.yaml")}


def write_prompt_files(root: str, lib: str = LIB_OBJECTS) -> int:
    """给标注过但缺文件的图补空标签，返回补建数量（空文件=看过确实没目标）。"""
    lib = normalize_lib(lib)
    idx = load_index(root, lib)
    made = 0
    for rel, item in idx.ordered_items():
        if item.get(field_key(lib, "objects")) is None:
            continue
        path = label_path_for(root, rel, lib)
        if not os.path.exists(path):
            field_store.write_label_rows(path, [])
            made += 1
    if made:
        os.makedirs(labels_dir(root, lib), exist_ok=True)
    return made
