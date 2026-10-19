"""目标检测标注：用本地 YOLO 检测 / 分割 / 旋转框权重批量出框。

和「提示词标注」的分工（界面上也按这个口径写，别让用户猜）：

- **提示词标注**：多模态大模型按你写的字段描述找物品。类别你自己定，可以联网，
  慢（每张一次请求），胜在灵活——仓库里没见过的箱子也能标。
- **目标检测**：本地 YOLO 权重按它**自带的类别表**（COCO-80 等）出框。不联网、
  快（单张几十毫秒），类别固定——先选权重、再勾权重认识的类别，剩下的交给机器。

产物与归属：

::

    <dataset>/detect_labels/<同名>.txt       YOLO detection 5 列，可直接喂训练
    <dataset>/.autolabel/detect/index.json   本库索引（detect_* 状态）
    <dataset>/.autolabel/detect/classes.json 类别登记表：类名 -> 永不复用的类别号
    <dataset>/.autolabel/detect/prefs.json   上次用的权重与勾选的类别

类别号这件事必须讲清楚：模型自带的是模型自己的编号（COCO 里 bus=5），
但标签文件里的编号由**本库登记表**决定，且**同名类别永不换号**。换权重时
同名类别继续用同一个号，先前标好的框才不会一夜之间指错东西。
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import fields as field_store
from .backends import DETECT_TASKS, TASK_NAMES, BackendSettings, DetectionBackend
from .classnames import zh_for
from .dataset import (
    HUMAN_STATUSES,
    STATUS_APPROVED,
    STATUS_EDITED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    _load_rgb,
    load_index,
    now_str,
)
from .atomic import atomic_write_json
from .library import LIB_DETECT, labels_dir, lib_name, meta_dir_for
from .prompt import (
    DETECT_FLAG_HINTS,
    Detection,
    box_flags,
    collect_object_records,
    dedupe,
    export_detection_dataset,
    image_flags,
    read_detections,
    render_object_coords,
    write_detections,
    write_object_coordinate_files,
)

#: 本模块服务的标注库，写在这里省得满文件拼字符串
LIB = LIB_DETECT
VALID_DETECT_STATUS = {STATUS_UNREVIEWED, STATUS_APPROVED, STATUS_EDITED, STATUS_REJECTED}
#: 界面上一次最多勾多少个类别：再多数框就没什么意义了，纯粹拖慢推理
MAX_RUN_CLASSES = 200
#: 用户没选权重时的兜底。不能沿用 ``BackendSettings.model``——那是 ``yolo11n-pose.pt``，
#: 拿姿态权重跑检测要等加载完才报错，而且报错不会告诉用户该换成什么。
DEFAULT_DETECT_MODEL = "yolo11n.pt"


# --------------------------------------------------------------------------- #
# 模型与类别
# --------------------------------------------------------------------------- #

def resolve_model(model: str = "", root: str = "") -> str:
    """把「没选权重」落成一个真能跑的检测权重：显式指定 → 上次用的 → 默认值。"""
    name = str(model or "").strip()
    if not name and root:
        name = str(load_prefs(root).get("model") or "").strip()
    return name or DEFAULT_DETECT_MODEL


def task_of(filename: str) -> str:
    """按 Ultralytics 的命名猜任务类型（``yolo11n-seg.pt`` -> ``segment``）。

    只在"想知道这个权重大概能干什么"时用，真正判任务一律以加载后
    ``model.task`` 为准：用户完全可能把文件改名。猜不出返回空串。
    """
    name = str(filename or "").lower()
    for suffix, task in (("-pose", "pose"), ("-seg", "segment"), ("-obb", "obb"),
                         ("-cls", "classify"), ("-detect", "detect")):
        if suffix in name:
            return task
    return "detect" if name.endswith((".pt", ".onnx", ".engine")) else ""


def local_models(dir_: Optional[str] = None) -> List[dict]:
    """本机已有的权重清单（**不加载模型**，几百毫秒就能列完）。

    ``usable`` 表示能不能用在目标检测库：按文件名猜是检测类的算可用，
    认不出任务型的也算可用（真实任务由 :class:`DetectionBackend` 加载时把关），
    只有明确是姿态/分类这类对不上的才标成不可用，并在 ``note`` 里说清为什么。
    """
    from . import models as model_manager

    dir_ = model_manager.weights_dir() if dir_ is None else dir_
    rows: List[dict] = []
    for name in model_manager.weight_names(dir_):
        task = task_of(name)
        usable = (not task) or task in DETECT_TASKS
        note = ""
        if not usable:
            note = (f"这是「{TASK_NAMES.get(task, task)}」权重，检测库用不了；"
                    + ("人体骨骼请回姿态标注" if task == "pose" else "该任务的产出不是检测框"))
        # 只能用 model_status：weight_names 会把「目录里预置但还没下载」的也列出来，
        # 按文件大小判会全部显示 0 字节，界面就分不清"有"和"能下载"
        status = model_manager.model_status(name, weights_dir=dir_)
        rows.append({"name": name, "task": task or "unknown",
                     "task_name": TASK_NAMES.get(task, "未知（按文件名猜不出）"),
                     "usable": usable, "note": note,
                     "installed": bool(status["ready"]),
                     "bytes": int(float(status["size_mb"] or 0) * 1048576)})
        if usable and not status["ready"] and not note:
            row = rows[-1]
            row["note"] = "还没下载，选它开始标注时会自动下（几 MB 到几百 MB，看这一档多大）"
            row["downloadable"] = True
    return rows


def detectable_models(dir_: Optional[str] = None) -> List[dict]:
    """能用在目标检测库的那部分权重（界面下拉框只该看到这些）。"""
    return [row for row in local_models(dir_) if row["usable"]]


def model_profile(model: str, device: str = "", dir_: Optional[str] = None) -> dict:
    """加载权重并回报它是什么任务、认得出哪些类（第一次加载会下载权重，慢）。"""
    settings = BackendSettings(model=resolve_model(model), device=device or "auto")
    backend = _backend(settings, dir_)
    described = backend.describe()
    described["classes"] = [{"class_id": cid, "name": name, "label": zh_for(name) or name}
                            for cid, name in sorted(backend.class_names().items())]
    return described


#: 已加载的后端：权重加载一次几十毫秒到几秒，批量标注里每张都重加载会白等一半时间
_BACKENDS: Dict[Tuple[str, str, int, bool], DetectionBackend] = {}


def _backend(settings: BackendSettings, dir_: Optional[str] = None) -> DetectionBackend:
    """取（必要时加载）检测后端，并把权重路径对齐到本机 ``models/``。

    缓存键只带会改变结果的参数：同一张显卡、同一个输入尺寸重复取就直接复用。
    ``conf`` 不在键里——它是每次推理单独传的，换阈值不必重新加载权重。
    """
    from . import models as model_manager

    path = model_manager.resolve_weight_in_dir(settings.model, dir_)
    settings = BackendSettings(**dict(settings.to_dict(), model=path or settings.model))
    key = (os.path.abspath(path or settings.model), settings.device, settings.imgsz, bool(settings.half))
    backend = _BACKENDS.get(key)
    if backend is None:
        backend = DetectionBackend(settings, path or settings.model)
        backend.load()
        _BACKENDS[key] = backend
    return backend


def pick_classes(root: str, wanted: Optional[Sequence[str]] = None) -> List[dict]:
    """本次要标的类别（登记表里的条目）；留空表示登记表里全部。"""
    return field_store.resolve_fields(root, wanted, LIB)


def match_model_classes(root: str, model: str, wanted: Optional[Sequence[str]] = None,
                        device: str = "") -> dict:
    """把登记的类别名对到模型自带的类别号上，返回能用的与对不上的。

    对不上是常态而不是错误：上一轮用 COCO 权重标了 ``bottle``，这次换成自训练的
    3 类权重，``bottle`` 就落空了。落空的类别要明说，别默默少标一堆框。
    """
    picked = pick_classes(root, wanted)
    if not picked:
        raise ValueError(f"{lib_name(LIB)}还没有类别：先在界面上从模型类别表里勾选，"
                         f"或跑 autolabel detect --model {model or 'yolo11n.pt'} --add-classes bus,person")
    profile = model_profile(resolve_model(model, root), device)
    by_name = {str(row.get("name", "")).strip().lower(): int(row["class_id"])
               for row in profile["classes"]}
    pairs: List[Tuple[dict, int]] = []
    missing: List[str] = []
    for row in picked:
        key = str(row.get("name", "")).strip().lower()
        if key in by_name:
            pairs.append((row, by_name[key]))
        else:
            missing.append(str(row.get("name") or key))
    return {"model": profile["model"], "task": profile["task"], "task_name": profile["task_name"],
            "device": profile["device"], "class_count": profile["class_count"],
            "pairs": pairs, "missing": missing,
            "requested": [row.get("name") for row in picked]}


def add_model_classes(root: str, model: str, names: Optional[Sequence[str]] = None,
                      device: str = "", dir_: Optional[str] = None) -> dict:
    """把模型自带的类别收进本库登记表（界面「从模型导入类别」）。

    ``names`` 留空表示全量导入。中文名优先用 ``classnames`` 对照表，英文类名
    原样保留——它要写进 ``data.yaml``，也得和模型回报的值对得上。
    """
    profile = model_profile(resolve_model(model, root), device, dir_)
    table = {str(row["name"]): int(row["class_id"]) for row in profile["classes"]}
    known = {key.lower(): key for key in table}
    wanted, unknown = [], []
    for raw in (names or table):
        name = str(raw or "").strip()
        if not name:
            continue
        # 只登记这个权重真的报得出的类别：收进来一个它不认识的，
        # 之后「按本库全部类别标注」会整批报"认不出这些类别"，等于自己埋雷
        match = known.get(name.lower())
        (wanted.append(match) if match else unknown.append(name))
    payload = field_store.add_classes(root, wanted, source=os.path.basename(model), lib=LIB)
    payload.update({"model": profile["model"], "task": profile["task"],
                    "task_name": profile["task_name"], "model_class_count": profile["class_count"],
                    "not_in_model": unknown})
    return payload


# --------------------------------------------------------------------------- #
# 参数指纹与上次选择
# --------------------------------------------------------------------------- #

def signature_for(model: str, class_ids: Sequence[int], settings: BackendSettings) -> str:
    """参数指纹：换权重、换类别或改阈值就该重标。

    ``device``/``half`` 不进指纹（见 ``pipeline.settings_signature``）：它们只改快慢，
    算进去会让"从 MPS 切到 CPU"变成把整个库重跑一遍。
    """
    payload = {"model": os.path.basename(str(model)), "classes": sorted(int(c) for c in class_ids),
               "imgsz": settings.imgsz, "conf": settings.conf, "iou": settings.iou,
               "max_det": settings.max_det}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:12]


def prefs_path(root: str) -> str:
    return os.path.join(meta_dir_for(root, LIB), "prefs.json")


def load_prefs(root: str) -> dict:
    """上次用的权重与勾选的类别，界面上次关掉再开还记得。"""
    try:
        with open(prefs_path(root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(root: str, *, model: str = "", classes: Optional[Sequence[str]] = None,
               conf: Optional[float] = None) -> dict:
    prefs = load_prefs(root)
    if model:
        prefs["model"] = os.path.basename(str(model))
    if classes is not None:
        prefs["classes"] = [str(name) for name in classes]
    if conf is not None:
        prefs["conf"] = float(conf)
    prefs["updated_at"] = now_str()
    atomic_write_json(prefs_path(root), prefs)
    return prefs


# --------------------------------------------------------------------------- #
# 单张与批量标注
# --------------------------------------------------------------------------- #

def label_image(root: str, image_rel: str, backend: DetectionBackend,
                pairs: Sequence[Tuple[dict, int]], conf: Optional[float] = None) -> dict:
    """跑一张图：模型出的类别号换成本库类别号，再打质量标记。"""
    from .dataset import resolve_image

    source = resolve_image(root, image_rel)
    if not os.path.exists(source):
        raise FileNotFoundError(image_rel)
    # _load_rgb 给的是 PIL 图，检测后端要的是 numpy（RGB），中间这步不能省
    image = np.asarray(_load_rgb(source), dtype=np.uint8)
    wanted = [int(model_id) for _row, model_id in pairs]
    found = backend.predict(image, class_ids=wanted, conf=conf)
    # 模型类别号 -> 本库类别号（同名类别永不换号，全靠这张表）
    to_local = {int(model_id): int(row.get("class_id", -1)) for row, model_id in pairs}
    labels = {int(model_id): str(row.get("name", "")) for row, model_id in pairs}
    detections: List[Detection] = []
    for item in found:
        cls = to_local.get(int(item.class_id), -1)
        if cls < 0:
            continue
        detections.append(Detection(cls=cls, cx=item.cx, cy=item.cy, w=item.w, h=item.h,
                                    score=item.score,
                                    flags=["masked"] if item.has_mask else []))
    detections = dedupe(detections)
    return {"detections": detections, "model": os.path.basename(backend.model_name),
            "model_classes": sorted({labels.get(int(item.class_id), item.class_name)
                                     for item in found}),
            "raw": len(found)}


def run_detect_labeling(
    root: str,
    *,
    model: str = "",
    classes: Optional[Sequence[str]] = None,
    images: Optional[Sequence[str]] = None,
    limit: int = 0,
    overwrite: bool = False,
    device: str = "",
    conf: Optional[float] = None,
    settings: Optional[BackendSettings] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    verbose: bool = True,
) -> dict:
    """批量目标检测：逐张跑本地权重、写 ``detect_labels/``、更新索引 ``detect_*``。

    :param model: 权重文件名或路径，留空用系统设置里的默认权重
    :param classes: 要标的类别（类名/显示名/ID），留空 = 登记表里全部
    :param overwrite: 默认跳过人工接受过、改过、手工画过框的图，勾了才覆盖
    """
    root = os.path.abspath(root)
    matched = match_model_classes(root, model, classes, device)
    pairs = matched["pairs"]
    if not pairs:
        raise ValueError(
            f"{os.path.basename(matched['model'])} 认不出这些类别：{'、'.join(matched['missing'])}"
            f"（它是「{matched['task_name']}」模型，共 {matched['class_count']} 类）。"
            "去「目标检测」页重新勾选，或换一个包含这些类别的权重")
    if len(pairs) > MAX_RUN_CLASSES:
        raise ValueError(f"一次最多标 {MAX_RUN_CLASSES} 个类别，当前勾了 {len(pairs)} 个")

    base = settings or BackendSettings()
    resolved = BackendSettings(model=matched["model"], device=device or base.device,
                               imgsz=base.imgsz, max_det=base.max_det, iou=base.iou,
                               half=base.half,
                               conf=float(conf) if conf is not None else base.conf)
    backend = _backend(resolved)
    local_ids = [int(row.get("class_id", -1)) for row, _ in pairs]
    signature = signature_for(matched["model"], local_ids, resolved)
    known_names = {str(row.get("name", "")).lower() for row, _ in pairs}

    idx = load_index(root, LIB)
    targets = [str(rel) for rel in images] if images else [rel for rel, _ in idx.ordered_items()]
    protected = 0
    if not overwrite:
        kept = []
        for rel in targets:
            item = idx.get(rel) or {}
            relabeled = (item.get("detect_signature") == signature
                         and item.get("detect_objects") is not None)
            human = (item.get("detect_status") in HUMAN_STATUSES
                     or int(item.get("detect_manual") or 0) > 0)
            if relabeled or human:
                protected += 1
                continue
            kept.append(rel)
        targets = kept
    if limit and limit > 0:
        targets = targets[:limit]

    stats = {"labeled": 0, "boxes": 0, "failed": 0, "skipped": 0, "protected": protected,
             "images": len(targets), "missing_classes": matched["missing"],
             "classes": sorted(known_names), "model": matched["model"],
             "task": matched["task"], "task_name": matched["task_name"],
             "device": matched["device"], "conf": resolved.conf, "last_error": ""}
    if matched["missing"] and verbose:
        print(f"[warn] 这个模型认不出：{'、'.join(matched['missing'])}（这些类别本次不产出框）")

    for done, rel in enumerate(targets, start=1):
        if should_stop and should_stop():
            stats["skipped"] += len(targets) - done + 1
            break
        item = idx.get(rel) or {}
        try:
            result = label_image(root, rel, backend, pairs, conf=resolved.conf)
        except Exception as exc:  # noqa: BLE001 单图失败不该拖垮整批
            stats["failed"] += 1
            stats["last_error"] = f"{type(exc).__name__}: {exc}"
            idx.upsert(rel, detect_flags=sorted(set(list(item.get("detect_flags") or []))
                                               | {"model_failed"}),
                       detect_signature=signature, detect_error=stats["last_error"][:300])
            idx.save()
            if verbose:
                print(f"[warn] {rel} 标注失败：{stats['last_error']}")
            if progress:
                progress(done, len(targets))
            continue
        detections = result["detections"]
        per_box = box_flags(detections, int(item.get("width") or 0),
                            int(item.get("height") or 0), local_ids)
        write_detections(root, rel, detections, LIB)
        flags = image_flags(detections, per_box, signature=signature,
                            item_signature=str(item.get("detect_signature") or ""),
                            stale_flag="model_stale", failed_flag="model_failed")
        scores = [detection.score for detection in detections] or [0.0]
        status = item.get("detect_status") or STATUS_UNREVIEWED
        if status not in VALID_DETECT_STATUS:
            status = STATUS_UNREVIEWED
        if status in (STATUS_APPROVED, STATUS_EDITED):
            status = STATUS_EDITED      # 重标过的人工结果要重新确认一遍
        idx.upsert(rel, detect_objects=len(detections), detect_status=status,
                   detect_flags=flags, detect_signature=signature, detect_error="",
                   detect_score=round(sum(scores) / len(scores), 4),
                   detect_labeled_at=now_str(), detect_model=matched["model"],
                   # 这批框全部出自模型：人工标记与类别号都要跟着重写，
                   # 否则索引里会留下指向不存在框的人工标记（撤销与导出会被带偏）
                   detect_manual=0, detect_manual_boxes=[],
                   detect_classes=sorted({detection.cls for detection in detections}))
        idx.save()
        stats["labeled"] += 1
        stats["boxes"] += len(detections)
        if verbose:
            print(f"[ok] {rel}：{len(detections)} 个框（{matched['model']} · {len(pairs)} 类）")
        if progress:
            progress(done, len(targets))

    if stats["failed"] and stats["labeled"] == 0:
        raise RuntimeError(f"{stats['failed']} 张全部失败，最后一条错误：{stats['last_error'] or '未知'}")
    save_prefs(root, model=matched["model"], classes=stats["classes"], conf=resolved.conf)
    return stats


# --------------------------------------------------------------------------- #
# 总览 / 坐标 / 导出（全部复用提示词库那套，只是换个库名）
# --------------------------------------------------------------------------- #

def detect_summary(root: str) -> dict:
    """目标检测库总览：数量、状态分布、按类别统计、上次用的权重。"""
    from .prompt import prompt_summary

    summary = prompt_summary(root, LIB)
    prefs = load_prefs(root)
    summary["model"] = prefs.get("model") or ""
    summary["models"] = detectable_models()
    summary["classes_remembered"] = list(prefs.get("classes") or [])
    summary["flag_hints"] = DETECT_FLAG_HINTS
    summary["unit"] = "框"
    return summary


def collect_detect_records(root: str, exclude_status: Sequence[str] = ("rejected",)) -> Tuple[List[dict], dict]:
    return collect_object_records(root, exclude_status=exclude_status, lib=LIB)


def write_detect_coordinate_files(root: str, out: Optional[str] = None, tag: Optional[str] = None,
                                  exclude_status: Sequence[str] = ("rejected",)) -> dict:
    return write_object_coordinate_files(root, out=out, tag=tag,
                                         exclude_status=exclude_status, lib=LIB)


def render_detect_coords(root: str, fmt: str = "csv",
                         exclude_status: Sequence[str] = ("rejected",)) -> Tuple[str, bytes]:
    return render_object_coords(root, fmt, exclude_status=exclude_status, lib=LIB)


def export_detect_dataset(root: str, out: Optional[str] = None, **kwargs) -> dict:
    kwargs.pop("lib", None)
    return export_detection_dataset(root, out, lib=LIB, **kwargs)


def detect_labels_dir(root: str) -> str:
    return labels_dir(root, LIB)


def read_detect_detections(root: str, image_rel: str) -> List[Detection]:
    return read_detections(root, image_rel, LIB)


__all__ = [
    "DETECT_FLAG_HINTS", "LIB", "MAX_RUN_CLASSES", "VALID_DETECT_STATUS",
    "add_model_classes", "collect_detect_records", "detect_labels_dir", "detect_summary",
    "detectable_models", "export_detect_dataset", "label_image", "load_prefs",
    "DEFAULT_DETECT_MODEL", "local_models", "match_model_classes", "model_profile",
    "pick_classes", "prefs_path", "read_detect_detections", "render_detect_coords",
    "resolve_model", "run_detect_labeling", "save_prefs",
    "signature_for", "task_of", "write_detect_coordinate_files",
]
