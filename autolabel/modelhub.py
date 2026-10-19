"""YOLO 模型目录与离线搜索：给「模型管理」页的搜索框当数据层。

``models.PRESET_CATALOG`` 只有姿态一档（13 条），界面要按任务搜模型（检测 / 分割 /
旋转框 / 分类）时没有数据可用。这里在不改 ``models.py`` 的前提下补一份更全的目录：

- 姿态那 13 条从 ``models.PRESET_CATALOG`` 原样带过来，字段一个不减
  （``map_pose``/``preset_visible`` 这些现有代码还在读）；
- 其余任务按 ultralytics/assets release v8.4.0 的实际字节数补齐，体积全部经 HEAD
  核实过；``sha256`` 只填本机 ``models/`` 里已有文件的真实值，其余留空串；
- 参数规模与精度取自 Ultralytics 官方模型表（2026-10 版），官方没给的填 ``None``
  并在中文说明里写明；
- 离线匹配 ``search()`` 是主路径，联网搜索 ``search_remote()`` / ``search_payload()``
  只是加分项：断网、超时、限流都必须安静降级，绝不抛异常；
- 下载不另起一套逻辑，直接复用 ``models.start_download`` / ``backends.ensure_weights``。

每行都带界面可以直接渲染的别名字段（``task_label``/``intro``/``params``/``bytes``/
``url``/``installed``），不必再在前端做一次字段改名。``capability`` 与 ``task`` 取同
一个值：姿态仍是 ``models.CAPABILITY_POSE``（``"pose"``），其余任务用任务名本身。
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

from . import models
from .backends import weights_url

# --------------------------------------------------------------------------- #
# 任务口径
# --------------------------------------------------------------------------- #

#: 目录里出现的任务，顺序即界面分组顺序（姿态在前，其余任务跟在后面）
TASK_VALUES: Tuple[str, ...] = ("pose", "detect", "segment", "obb", "classify")
TASK_LABELS: Dict[str, str] = {
    "pose": "人体姿态",
    "detect": "目标检测",
    "segment": "实例分割",
    "obb": "旋转框检测",
    "classify": "图像分类",
}

#: 权重发布页：所有目录条目的字节数都在这个 release 上核实
RELEASE_URL = "https://github.com/ultralytics/assets/releases/tag/v8.4.0"

FAMILY_DOCS: Dict[str, str] = {
    "yolo11": "https://docs.ultralytics.com/models/yolo11/",
    "yolov8": "https://docs.ultralytics.com/models/yolov8/",
    "yolo12": "https://docs.ultralytics.com/models/yolo12/",
    "yolo26": "https://docs.ultralytics.com/models/yolo26/",
}

#: 类别表来源：界面直接显示，避免用户下完才发现类别表不是自己要的
CLASS_NOTES: Dict[str, str] = {
    "pose": "关键点表 COCO-17（模型自带，17 个人体关键点）",
    "detect": "类别表 COCO-80（模型自带 names：人、车、动物等 80 类）",
    "segment": "类别表 COCO-80（80 类，模型自带 names，多边形轮廓由模型给）",
    "obb": "类别表 DOTA-15（航拍、遥感常见 15 类，模型自带 names）",
    "classify": "类别表 ImageNet-1000（1000 类；score 是 top-1 准确率，不是 mAP）",
}

SIZE_NAMES: Dict[str, str] = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}

# --------------------------------------------------------------------------- #
# 目录条目
# --------------------------------------------------------------------------- #


def _family(stem: str) -> str:
    """从文件名里取系列名（``yolo11n-seg`` -> ``yolo11``），认不出返回空串。"""
    match = re.match(r"(yolov\d+|yolo\d+)", str(stem or "").lower())
    return match.group(1) if match else ""


def _row(filename: str, display_name: str, task: str, params_m: Optional[float],
         score: Optional[float], expected_bytes: int, speed: str, description: str,
         recommended_for: str, sha256: str = "") -> dict:
    """拼一条目录记录：体积按字节算 MB，出处按系列给文档地址。"""
    stem = filename[:-3] if filename.endswith(".pt") else filename
    return {
        "id": stem.lower(),
        "filename": filename,
        "display_name": display_name,
        "task": task,
        "task_name": TASK_LABELS.get(task, task),
        "capability": task,
        "params_m": params_m,
        "score": score,
        "expected_bytes": int(expected_bytes),
        "expected_mb": round(int(expected_bytes) / 1048576, 1),
        "size_verified": True,
        "sha256": sha256,
        "speed": speed,
        "description": description,
        "recommended_for": recommended_for,
        "downloadable": True,
        "installed": False,
        "homepage": FAMILY_DOCS.get(_family(stem), RELEASE_URL),
        "num_keypoints": 17 if task == "pose" else 0,
        "class_note": CLASS_NOTES.get(task, ""),
    }


#: 除姿态之外的目录。字节数取 ultralytics/assets v8.4.0 的 Content-Length（全部实测 200），
#: 参数与精度取官方模型表；``yolo11n.pt`` 本机已有，顺带填上真实 SHA256。
EXTRA_CATALOG: Tuple[dict, ...] = (
    # --- 目标检测（COCO-80）---
    _row("yolo11n.pt", "YOLO11-nano 目标检测", "detect", 2.6, 39.5, 5613764,
         "最快，CPU 也能实时",
         "最小的通用检测权重，5.4 MB，CPU 上也能几百毫秒一张，用来先跑通流程和大批量抽帧预筛最合算。"
         "精度是这一代里最低的，小目标和重叠目标容易漏。",
         "试跑验证、大批量预标注、CPU 或轻薄本",
         sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    _row("yolo11s.pt", "YOLO11-small 目标检测", "detect", 9.4, 47.0, 19313732, "快",
         "速度、精度、体积最均衡的一档，比 nano 高约 7.5 个 mAP，单张仍在百毫秒级。"
         "没有特殊要求就从这档开始。",
         "日常目标检测默认选择"),
    _row("yolo11m.pt", "YOLO11-medium 目标检测", "detect", 20.1, 51.5, 40684120, "中等",
         "比 small 明显更准，小目标和重叠目标召回更好，代价是体积和耗时都翻一倍。"
         "人多车多的画面值得换这档。",
         "精度优先、机器有独显或 Apple Silicon"),
    _row("yolo11l.pt", "YOLO11-large 目标检测", "detect", 25.3, 53.4, 51387343, "慢",
         "large 档，比 medium 再高不到两点而体积只多 10 MB，显存够的话可以直接上。"
         "再往上的 xlarge 收益已经很边际。",
         "成片级高质量标注"),
    _row("yolo11x.pt", "YOLO11-xlarge 目标检测", "detect", 56.9, 54.7, 114636239, "最慢（本目录内）",
         "YOLO11 检测的顶配，109 MB，比 large 只高一点三分，下载和推理都要留时间。",
         "追求极限精度、机器很强"),
    _row("yolov8n.pt", "YOLOv8-nano 目标检测", "detect", 3.2, 37.3, 6549796, "最快",
         "上一代最小检测权重，接口与 v11 完全一致，同精度下比 v11-nano 略大略慢，"
         "一般只在复用旧工程或与 v11 组双模型时用。",
         "复用旧项目权重、与 v11 做双模型对照"),
    _row("yolov8s.pt", "YOLOv8-small 目标检测", "detect", 11.2, 44.9, 22588772, "快",
         "上一代 small 档，mAP 比 v11-small 低两点，胜在生态里现成教程和二次训练权重最多。",
         "多模型共识里的第二票"),
    _row("yolov8m.pt", "YOLOv8-medium 目标检测", "detect", 25.9, 50.2, 52136884, "中等",
         "上一代 medium 档，精度与 v11-medium 接近但体积更大，适合和 v11 交叉验证。",
         "和 v11-medium 交叉验证"),
    _row("yolov8l.pt", "YOLOv8-large 目标检测", "detect", 43.7, 52.9, 87792836, "慢",
         "上一代 large 档，参数比 v11-large 多近一倍而精度只差半分，性价比一般。",
         "显存富余、想比 v11 多一票"),
    _row("yolov8x.pt", "YOLOv8-xlarge 目标检测", "detect", 68.2, 53.9, 136890692, "最慢（本目录内）",
         "上一代最大的检测权重，130 MB，本目录里最占地方的一档。",
         "只做精度对照"),
    _row("yolo26n.pt", "YOLO26-nano 目标检测", "detect", 2.4, 40.9, 5544453, "快",
         "最新一代 YOLO26 的最小检测权重，5.3 MB。官方说明这是端到端（免 NMS）设计，"
         "表里的 40.9 是走 NMS 的那一头，免 NMS 推理是 40.1。新版权重要新版 ultralytics "
         "才读得动，报加载错就先升级。",
         "想试新一代模型、愿意多花人工复核"),
    _row("yolo26s.pt", "YOLO26-small 目标检测", "detect", 9.5, 48.6, 20422725, "快",
         "YOLO26 的 small 档，官方表里比 v11-small 高约 1.6 个 mAP，体积相近。"
         "建议先小批量试标，和 yolo11s.pt 对比框准不准再决定长期用哪个。",
         "新一代对照实验"),
    _row("yolo12n.pt", "YOLO12-nano 目标检测", "detect", 2.6, 40.6, 5595063, "快",
         "YOLO12 用注意力机制替换了部分卷积，nano 档同等参数量下比 v11-nano 更准，"
         "代价是注意力算子更吃显存、CPU 上更慢。",
         "同等体积下想多要一点精度"),
    _row("yolo12s.pt", "YOLO12-small 目标检测", "detect", 9.3, 48.0, 19006455, "中等",
         "YOLO12 的 small 档，mAP 略高于 v11-small 但速度不如它，GPU 上差距比 CPU 上小。",
         "精度优先于人力的批量标注"),
    # --- 实例分割（COCO-80）---
    _row("yolo11n-seg.pt", "YOLO11-nano 实例分割", "segment", 2.9, 38.9, 6182636,
         "最快，CPU 也能实时",
         "最小的分割权重，出框之外还给每个目标一份多边形轮廓，5.9 MB。适合要轮廓、"
         "又不想等太久的批量预标，轮廓边缘会比 medium 毛一点。",
         "轮廓级预标注、CPU 试跑"),
    _row("yolo11s-seg.pt", "YOLO11-small 实例分割", "segment", 10.1, 46.6, 20669228, "快",
         "分割档里最均衡的一档，比 nano 高约 7.7 个 mAP，轮廓明显更贴合，单张几百毫秒。"
         "分割任务建议从这档开始。",
         "实例分割日常默认"),
    _row("yolo11m-seg.pt", "YOLO11-medium 实例分割", "segment", 22.4, 51.5, 45400152, "中等",
         "更准的分割权重，细长目标和贴边目标的轮廓少断裂，耗时约为 small 的两倍。",
         "精度优先的分割标注、有独显"),
    _row("yolov8n-seg.pt", "YOLOv8-nano 实例分割", "segment", 3.4, 36.7, 7071756, "最快",
         "上一代最小分割权重，接口一致，同精度下比 v11-nano 略大略慢，主要在沿用旧工程时用。",
         "与 v11 分割模型做对照"),
    _row("yolov8s-seg.pt", "YOLOv8-small 实例分割", "segment", 11.8, 44.6, 23914764, "快",
         "上一代 small 分割档，mAP 比 v11-small 低约两点，二次训练的资料最多，适合自己微调。",
         "自己微调分割模型时的起点"),
    # --- 旋转框检测（DOTA-15）---
    _row("yolo11n-obb.pt", "YOLO11-nano 旋转框检测", "obb", 2.7, 78.4, 5795654, "最快",
         "旋转框（OBB）最小权重，输出带角度的框，航拍、遥感、斜着摆的物体比正框贴合得多。"
         "官方分数是在 DOTAv1（1024 输入）上测的，不能和 COCO 的分数横向比。",
         "航拍、遥感、倾斜摆放物体的批量标注"),
    _row("yolo11s-obb.pt", "YOLO11-small 旋转框检测", "obb", 9.7, 79.5, 19934534, "快",
         "旋转框里最均衡的一档，比 nano 更能扛密集小目标（车位、货架），耗时约为 nano 的两倍。",
         "旋转框日常默认"),
    _row("yolov8n-obb.pt", "YOLOv8-nano 旋转框检测", "obb", 3.1, 78.0, 6567590, "最快",
         "上一代旋转框 nano 权重，分数与 v11-nano 相当，接口一致，主要在沿用旧工程时用。",
         "与 v11-obb 交叉验证"),
    # --- 图像分类（ImageNet-1000）---
    _row("yolo11n-cls.pt", "YOLO11-nano 图像分类", "classify", 2.8, 70.0, 5790624, "最快",
         "分类权重只回答「这张图属于哪一类」，不给框也不给轮廓，适合按类别分拣素材、给数据集"
         "打标签。官方 70.0 是 ImageNet-1k 的 top-1 准确率（top-5 为 89.4），"
         "和检测档的 mAP 口径不同，别拿两个数字横向比。",
         "素材按类分拣、给数据集打标签"),
    _row("yolov8n-cls.pt", "YOLOv8-nano 图像分类", "classify", 2.7, 69.0, 5563076, "最快",
         "上一代分类 nano，top-1 准确率比 v11-nano 低一分，体积差不多，一般只在沿用旧工程时用。"
         "分数同样是 top-1 口径。",
         "与 v11 分类模型对照"),
)


def _pose_rows() -> List[dict]:
    """姿态条目从 ``models.PRESET_CATALOG`` 原样带过来，只补搜索和界面要用的新字段。"""
    rows: List[dict] = []
    for preset in models.PRESET_CATALOG:
        row = dict(preset)
        row["task"] = "pose"
        row["task_name"] = TASK_LABELS["pose"]
        row["capability"] = str(preset.get("capability") or models.CAPABILITY_POSE)
        row["score"] = preset.get("map_pose")
        row["homepage"] = FAMILY_DOCS.get(_family(str(row.get("filename") or "")), RELEASE_URL)
        kps = preset.get("num_keypoints") or 17
        keypoint_set = str(preset.get("keypoint_set") or "COCO-17")
        row["class_note"] = f"关键点表 {keypoint_set}（模型自带 {kps} 个人体关键点）"
        rows.append(row)
    return rows


def _base_rows() -> List[dict]:
    """完整目录（不含本机安装状态）：姿态在前，其余任务按 EXTRA_CATALOG 的顺序跟在后面。"""
    return _pose_rows() + [dict(row) for row in EXTRA_CATALOG]


# --------------------------------------------------------------------------- #
# 本机安装状态
# --------------------------------------------------------------------------- #


def installed_names(dir_: Optional[str] = None) -> List[str]:
    """权重目录里真实存在的 ``.pt``（小写文件名）。

    ``models.weight_names()`` 会把「目录里预置但还没下载」的权重也列出来，所以这里再按
    文件是否存在过一遍，避免把「能下」报成「已有」；半截的 ``.pt.part``/``.pt.bad``
    同样不算已安装。
    """
    dir_ = models.weights_dir() if dir_ is None else dir_
    found: List[str] = []
    for name in models.weight_names(dir_):
        bare = os.path.basename(str(name or ""))
        if bare.lower().endswith(".pt") and os.path.isfile(os.path.join(dir_, bare)):
            lowered = bare.lower()
            if lowered not in found:
                found.append(lowered)
    return found


def _by_filename(filename: str, rows: Optional[Sequence[dict]] = None) -> Optional[dict]:
    """按文件名取目录条目（大小写不敏感，可带路径前缀）。"""
    bare = os.path.basename(str(filename or "").strip().replace("\\", "/")).lower()
    if not bare:
        return None
    for row in (catalog() if rows is None else rows):
        if str(row.get("filename") or "").lower() == bare:
            return row
    return None


# --------------------------------------------------------------------------- #
# 对外：目录 / 任务判定
# --------------------------------------------------------------------------- #


def catalog(task: str = "") -> List[dict]:
    """全部可下载模型，每行带中文说明、体积、下载地址和本机安装状态。

    ``task`` 只保留该任务的条目；认不出的任务返回空列表（界面上把「没有这一类」当成空
    结果处理，比抛错好）。
    """
    wanted = str(task or "").strip().lower()
    if wanted and wanted not in TASK_VALUES:
        return []
    installed = set(installed_names())
    rows: List[dict] = []
    for base in _base_rows():
        row = dict(base)
        filename = str(row.get("filename") or "")
        row["task_label"] = TASK_LABELS.get(str(row.get("task")), str(row.get("task") or ""))
        row["intro"] = str(row.get("description") or "")
        row["params"] = row.get("params_m")
        row["bytes"] = int(row.get("expected_bytes") or 0)
        row["url"] = weights_url(filename) if filename.endswith(".pt") else ""
        row["installed"] = filename.lower() in installed
        rows.append(row)
    if wanted:
        rows = [row for row in rows if row.get("task") == wanted]
    return rows


def task_of(filename: str) -> str:
    """按 ultralytics 命名猜任务：``-pose``/``-seg``/``-obb``/``-cls``，认不出的 yolo 名按检测。

    完全看不出是 YOLO 权重的（``bus.pt``、``myweights.pt``）返回空串，让调用方去问模型。
    """
    base = os.path.basename(str(filename or "").strip().replace("\\", "/")).lower()
    stem = base[:-3] if base.endswith(".pt") else base
    stem = stem.split(".")[0]
    if not stem:
        return ""
    for suffix, task in (("-pose", "pose"), ("-seg", "segment"), ("-segment", "segment"),
                         ("-obb", "obb"), ("-cls", "classify"), ("-classify", "classify")):
        if suffix in f"-{stem}-":
            return task
    if re.match(r"^(yolo|yolov|rtdetr)", stem):
        return "detect"
    return ""


# --------------------------------------------------------------------------- #
# 离线模糊搜索
# --------------------------------------------------------------------------- #

#: 中文任务词：搜「分割」也能找到 ``-seg`` 权重，不必记英文后缀
TASK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "pose": ("pose", "姿态", "骨骼", "骨架", "关键点", "人体"),
    "detect": ("detect", "检测", "目标检测"),
    "segment": ("seg", "segment", "分割", "实例分割", "轮廓"),
    "obb": ("obb", "旋转框", "旋转", "航拍", "遥感", "dota"),
    "classify": ("cls", "classify", "分类", "分拣"),
}

#: ``nano``/``n`` 这类尺寸词按档位匹配，不做子串匹配（否则 n 能匹配到几乎所有文件）
SIZE_TERMS: Dict[str, str] = {"nano": "n", "tiny": "n", "small": "s", "medium": "m",
                              "large": "l", "xlarge": "x", "n": "n", "s": "s", "m": "m",
                              "l": "l", "x": "x"}

SPEED_TERMS = frozenset(("快", "最快", "速度", "实时", "fast"))
QUALITY_TERMS = frozenset(("精度", "准", "准度", "质量", "mAP", "map"))
SMALL_TERMS = frozenset(("小", "体积小", "轻量", "省", "占用低"))
LARGE_TERMS = frozenset(("大", "大模型", "最大"))
HINT_TERMS = SPEED_TERMS | QUALITY_TERMS | SMALL_TERMS | LARGE_TERMS

# 排序分档：文件名精确 > 前缀 > 文件名包含 > 任务词 > 速度/精度/体积提示 > 描述关键词
SCORE_EXACT = 10000
SCORE_PREFIX = 6000
SCORE_SUBSTRING = 4500
SCORE_TASK = 3000
SCORE_HINT = 900
SCORE_SIZE = 1200
SCORE_KEYWORD = 1000
#: 提示词（快 / 精度 / 小）只保留该维度在前半的条目
HINT_KEEP_AT = 0.5


def _size_of(stem: str) -> str:
    """文件名里的尺寸档（``yolo11n-seg`` -> ``n``），认不出返回空串。"""
    match = re.search(r"[-_]?(\d*)([nmlsx])(?:$|-)", str(stem or "").lower())
    return match.group(2) if match else ""


def _haystack(row: dict) -> str:
    keys = ("filename", "display_name", "description", "recommended_for", "speed",
            "class_note", "task_name")
    return " ".join(str(row.get(key) or "") for key in keys).lower()


def _task_hit(term: str, task: str) -> bool:
    words = set(TASK_KEYWORDS.get(task, ())) | {task, TASK_LABELS.get(task, "")}
    for word in words:
        word = str(word or "").lower()
        if not word:
            continue
        if term == word or (len(term) >= 2 and (term in word or word in term)):
            return True
    return False


def _speed_norm(speed: str) -> float:
    text = str(speed or "").strip()
    if not text:
        return 0.0
    if text.startswith("最慢"):
        return 0.05
    if text.startswith("最"):                      # 最快
        return 1.0
    if text.startswith("慢"):
        return 0.2
    if text.startswith("中"):
        return 0.4
    return 0.75                                    # 快


def _score_range(rows: Sequence[dict]) -> Dict[str, Tuple[float, float]]:
    """每个任务内部的分数区间：分类档是准确率、检测档是 mAP，只能各自排名。"""
    ranges: Dict[str, Tuple[float, float]] = {}
    for row in rows:
        value = row.get("score")
        if value is None:
            continue
        task = str(row.get("task") or "")
        low, high = ranges.get(task, (float(value), float(value)))
        ranges[task] = (min(low, float(value)), max(high, float(value)))
    return ranges


def _quality_norm(row: dict, ranges: Dict[str, Tuple[float, float]]) -> float:
    value = row.get("score")
    if value is None:
        return 0.0
    low, high = ranges.get(str(row.get("task") or ""), (0.0, 0.0))
    if high <= low:
        return 1.0
    return (float(value) - low) / (high - low)


def _small_norm(row: dict, max_mb: float) -> float:
    if max_mb <= 0:
        return 0.0
    return max(0.0, 1.0 - float(row.get("expected_mb") or 0.0) / max_mb)


#: 碎片的信息量门槛：命中超过这个比例条目的词组（``模型``、``标注``）等于没给线索
WINDOW_STOP_RATIO = 0.5
#: 短词（``旋转框``、``行人检测``）本来就搜得准，只有整句才需要切片兜底
_SLICE_MIN_LEN = 5


def _slices(term: str) -> List[str]:
    """把中文长词切成二字窗口（``适合标注行人`` → 适合/合标/标注/注行/行人）。

    中文没有空格，``我想找一个适合标注行人的小模型`` 在按符号切词之后就是**一个**
    词条，只能整句去 ``in`` 描述文本，结果永远搜不到。切成窗口后按命中比例给分，
    整句命中（``SCORE_KEYWORD``）仍然比碎片命中高得多，排序不会因为切词变糊。
    英文词交给空格切，这里不切。
    """
    if term.isascii() or len(term) < _SLICE_MIN_LEN:
        return []
    return [term[index:index + 2] for index in range(len(term) - 1)]


def _informative_windows(terms: Sequence[str], haystacks: Sequence[str]) -> Dict[str, List[str]]:
    """按"命中了多少条目录"筛掉没有信息量的中文碎片。

    ``不存在的模型`` 切出来的碎片里只有 ``模型`` 能命中，而目录里几乎条条描述都带
    "模型"两个字——这种碎片命中等于没说，留着它就变成"随便敲一句都返回全部"。
    反过来 ``轮廓`` 只在分割档的描述里出现，留着才能把分割权重排上来。
    """
    total = max(1, len(haystacks))
    kept_by_term: Dict[str, List[str]] = {}
    for term in terms:
        kept = []
        for window in _slices(term):
            hits = sum(1 for text in haystacks if window in text)
            if 0 < hits / total <= WINDOW_STOP_RATIO:
                kept.append(window)
        kept_by_term[term] = kept
    return kept_by_term


def _hint_score(term: str, row: dict, ranges: Dict[str, Tuple[float, float]],
                max_mb: float) -> int:
    if term in SPEED_TERMS:
        norm = _speed_norm(str(row.get("speed") or ""))
    elif term in QUALITY_TERMS:
        norm = _quality_norm(row, ranges)
    elif term in SMALL_TERMS:
        norm = _small_norm(row, max_mb)
    elif term in LARGE_TERMS:
        norm = 1.0 - _small_norm(row, max_mb)
    else:
        return 0
    return int(SCORE_HINT * norm) if norm >= HINT_KEEP_AT else 0


def _hint_score_partial(term: str, row: dict, ranges: Dict[str, Tuple[float, float]],
                        max_mb: float) -> int:
    """长句里夹着倾向词也认：``最省显存的`` 该找到体积最小的那几个。

    只在"词是句子里的一部分"时用（整词命中由 :func:`_hint_score` 负责，不重复计分）。
    权重本来就低（≤900）且只保留该维度前一半的条目，误命中一个"大/小"不会翻车。
    """
    best = 0
    for group in (SPEED_TERMS, QUALITY_TERMS, SMALL_TERMS, LARGE_TERMS):
        for word in group:
            if word != term and word in term:
                best = max(best, _hint_score(word, row, ranges, max_mb))
    return best


def _match_score(row: dict, terms: Sequence[str], ranges: Dict[str, Tuple[float, float]],
                 max_mb: float, windows: Optional[Dict[str, List[str]]] = None) -> int:
    stem = str(row.get("filename") or "").lower()
    if stem.endswith(".pt"):
        stem = stem[:-3]
    haystack = _haystack(row)
    size = _size_of(stem)
    total = 0
    for term in terms:
        if term == stem:
            total += SCORE_EXACT
        elif len(term) >= 2 and stem.startswith(term):
            total += SCORE_PREFIX
        elif len(term) >= 2 and term in stem:
            total += SCORE_SUBSTRING
        if _task_hit(term, str(row.get("task") or "")):
            total += SCORE_TASK
        if size and SIZE_TERMS.get(term) == size:
            total += SCORE_SIZE
        total += _hint_score(term, row, ranges, max_mb) + _hint_score_partial(
            term, row, ranges, max_mb)
        # 单个英文字母做全文匹配噪声太大（几乎每行都能命中），尺寸词交给上面处理
        if term not in HINT_TERMS and (len(term) >= 2 or not term.isascii()) and term in haystack:
            total += SCORE_KEYWORD
        else:
            kept = (windows or {}).get(term) or []
            all_windows = _slices(term)
            if kept and all_windows:
                total += int(SCORE_KEYWORD * len(kept) / len(all_windows))
    return total


def search(query: str = "", task: str = "") -> List[dict]:
    """离线模糊匹配：不联网、不加载模型，中文英文都能搜。

    排序：文件名精确 > 前缀 > 文件名包含 > 任务词（``分割``→segment、``旋转框``→obb、
    ``骨骼``/``姿态``→pose、``分类``→classify）> 速度/精度/体积倾向（``快``/``精度``/``小``）
    > 描述关键词。同分按目录顺序。空 query 返回全部（仍受 ``task`` 过滤）。
    """
    rows = catalog(task)
    # 搜索框里粘一大段话不必照单全收：取前 200 字足够表达意图，也免得逐行乘上去
    text = str(query or "").strip().lower()[:200]
    if not text:
        return rows
    terms = [term for term in re.split(r"[^0-9a-z\u4e00-\u9fff.+-]+", text) if term]
    if not terms:
        return []
    ranges = _score_range(rows)
    max_mb = max((float(row.get("expected_mb") or 0.0) for row in rows), default=0.0)
    windows = _informative_windows(terms, [_haystack(row) for row in rows])
    scored = []
    for position, row in enumerate(rows):
        score = _match_score(row, terms, ranges, max_mb, windows)
        if score > 0:
            scored.append((-score, position, row))
    scored.sort()
    return [item[2] for item in scored]


# --------------------------------------------------------------------------- #
# 联网增强搜索（可选，失败必须安静降级）
# --------------------------------------------------------------------------- #

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories?q="
REMOTE_NOTE = "社区权重未经校验，下载前建议核对文件大小"
NEEDS_QUERY_NOTE = "联网检索要先输入关键词，下面先是本地模型目录里的匹配结果"
LOCAL_ONLY_NOTE = "只匹配了本地模型目录（未联网）"
EMPTY_QUERY = "没有填写搜索关键词"
REMOTE_LIMIT = 20
REMOTE_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "autolabel-modelhub"}


def github_query(query: str) -> str:
    """给 GitHub 的查询词：没提到 yolo 的补一个，避免搜出一堆无关仓库。

    顺手截到 120 字：粘一整段说明进来时，GitHub 那边只会回一个无关的大杂烩，
    不如拿前半句去搜。
    """
    text = str(query or "").strip()[:120].strip()
    if not text:
        return ""
    return text if "yolo" in text.lower() else f"{text} yolo"


def _remote_failure(reason: str) -> str:
    return reason


def _remote_outcome(query: str, timeout: float) -> Tuple[List[dict], str, str]:
    """联网搜一次，并折算成（结果，错误原因，状态）。

    状态只有三种：``needs_query`` 没填关键词（压根不该联网，不算失败）、``ok`` 真的
    请求成功（命中 0 条也算）、``failed`` 断网/超时/限流这类真连不上。
    """
    results, error = _remote_fetch(query, timeout)
    if error == EMPTY_QUERY:
        return [], "", "needs_query"
    return results, _remote_failure(error), ("failed" if error else "ok")


def _remote_note(error: str, state: str = "") -> str:
    """联网没成功时那句人话：只说原因，「下面是本地结果」由界面补一次，不重复两遍。"""
    if state == "needs_query":
        return NEEDS_QUERY_NOTE
    return f"没连上 GitHub（{error}）" if error else REMOTE_NOTE


def _remote_fetch(query: str, timeout: float) -> Tuple[List[dict], str]:
    """去 GitHub 搜仓库，返回（结果列表，错误原因）。任何异常都换成错误原因，不往外抛。"""
    term = github_query(query)
    if not term:
        return [], EMPTY_QUERY
    url = GITHUB_SEARCH_URL + urllib.parse.quote(term)
    request = urllib.request.Request(url, headers=dict(REMOTE_HEADERS))
    try:
        with urllib.request.urlopen(request, timeout=max(0.5, float(timeout))) as response:
            body = response.read().decode("utf-8", "replace")
        payload = json.loads(body or "{}")
    except Exception as exc:  # noqa: BLE001 断网/超时/限流/返回不是 JSON 都得安静降级
        return [], f"{type(exc).__name__}: {exc}"
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], "GitHub 返回的内容看不懂（可能被限流或需要登录），已退回本地目录"
    results: List[dict] = []
    for item in items[:REMOTE_LIMIT]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("full_name") or item.get("name") or "").strip()
        link = str(item.get("html_url") or "").strip()
        if not name or not link:
            continue
        description = str(item.get("description") or "").strip()
        results.append({
            "name": name,
            "html_url": link,
            "description": description,
            "origin": "GitHub",
            # 界面直接渲染用的别名字段：社区仓库没有体积与参数量，填 0 / None
            "filename": "",
            "task": task_of(name),
            "task_label": TASK_LABELS.get(task_of(name), ""),
            "intro": description,
            "params": None,
            "bytes": 0,
            "url": link,
            "installed": False,
        })
    return results, ""


def search_remote(query: str, timeout: float = 6.0) -> dict:
    """联网搜社区里的 YOLO 权重仓库，附带本地目录里的匹配结果。

    返回 ``{"ok","query","local","results","note","error","remote_state"}``。GitHub 不可达、
    超时、限流、返回看不懂都只把 ``ok`` 置为 ``False`` 并填 ``error``，``local`` 照常给出，
    绝不抛异常。没填关键词时根本不联网，``remote_state`` 为 ``needs_query`` 且不算失败。
    """
    text = str(query or "").strip()
    results, error, state = _remote_outcome(text, timeout)
    return {"ok": state != "failed", "query": text, "local": search(text),
            "results": results, "note": _remote_note(error, state), "error": error,
            "remote_state": state}


def search_payload(q: str = "", task: str = "", remote: bool = False,
                   dataset_root: str = "") -> dict:
    """搜索框的一次完整应答：``remote=False`` 时纯本地、一个网络请求都不发。

    ``local`` 是本地目录里的匹配项（字段界面可直接渲染），``results`` 是 GitHub 上的社区
    仓库（只有 ``remote=True`` 时才有内容）。``remote_state`` 说明联网走到哪一步：
    ``skipped`` 没开联网、``needs_query`` 没填关键词、``ok`` 联网成功、``failed`` 真连不上，
    只有最后一种才 ``ok=False``。断网/超时/限流一律写进 ``note``/``error``，不抛异常。
    ``dataset_root`` 预留给调用方判兼容性，这一层还不需要它。
    """
    query = str(q or "").strip()
    local = search(query, task)
    shared = {"query": query, "task": str(task or "").strip().lower(), "local": local,
              "dataset_root": str(dataset_root or "")}
    if not remote:
        return dict(shared, ok=True, results=[], note=LOCAL_ONLY_NOTE, error="",
                    remote_state="skipped")
    results, error, state = _remote_outcome(query, 6.0)
    return dict(shared, ok=state != "failed",
                results=results if state == "ok" else [],
                note=_remote_note(error, state), error=error, remote_state=state)


# --------------------------------------------------------------------------- #
# 下载：完全复用 models.start_download / backends.ensure_weights
# --------------------------------------------------------------------------- #


def download_url(filename: str, mirror: Optional[str] = None) -> str:
    """下载地址（默认带镜像前缀），和下载记录、界面展示用同一个算法。"""
    bare = os.path.basename(str(filename or "").strip().replace("\\", "/"))
    return weights_url(bare, mirror) if bare.endswith(".pt") else ""


def download(filename: str, force: bool = False, mirror: Optional[str] = None,
             background: bool = True) -> dict:
    """下载目录里的权重：进度、校验、失败重试都在 ``models.start_download`` 里，这里不重复实现。"""
    bare = os.path.basename(str(filename or "").strip().replace("\\", "/"))
    if not bare.endswith(".pt"):
        raise ValueError(f"只能下载 .pt 权重：{filename}")
    return models.start_download(bare, force=force, mirror=mirror, background=background)


def download_state(name: str) -> Optional[dict]:
    """下载进度（同一份后台状态），没在下载就返回 ``None``。"""
    return models.download_state(name)


__all__ = ["TASK_VALUES", "TASK_LABELS", "catalog", "search", "search_payload",
           "search_remote", "task_of", "installed_names", "download", "download_state",
           "download_url"]
