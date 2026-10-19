"""模型管理：可浏览的模型目录（含中文说明）+ 姿态权重清单/删除/补下 + 多模态配置与连通性验证。

``ModelRegistry`` 把三类来源合成一份统一清单：内置可下载目录（``PRESET_CATALOG``，
可在 ``configs/models.json`` 的 ``catalog`` 里增删改）、用户注册的本地权重、
多模态大模型配置。每一行都带中文显示名、能力、来源、状态、预计体积、说明、
备注、最后使用时间与「和当前数据集是否兼容」。

权重清单直接扫描 ``models/`` 目录，所以用户自己丢进去的权重也能看到、能选、能删；
删除只允许动这个目录里的裸文件名，用户在配置里指定的绝对路径不碰。

多模态大模型配置放在 ``configs/vlm.json``（可用环境变量 ``AUTOLABEL_VLM_CONFIG``
覆盖），本地服务（Ollama / LM Studio / vLLM）和云端服务（OpenAI / 通义千问等
OpenAI 兼容端点）用同一套字段描述。API Key 不进 ``vlm.json``，只写进
``~/.autolabel/secrets.json``，配置里留 ``vlm:<id>`` 这样的引用名；接口返回时一律掩码。

用户登记与备注放在 ``configs/models.json``（可用 ``AUTOLABEL_MODEL_CONFIG`` 覆盖）：
注册路径、默认模型、备注说明、下载记录、最后使用时间，都不写 API Key。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from .atomic import atomic_write_json

from . import secrets as secret_store
from .backends import (UI_MODEL_CHOICES, ensure_weights, model_status, partial_paths,
                       weights_url)
from .dataset import load_index, now_str

DEFAULT_WEIGHTS_DIR = "models"
DEFAULT_VLM_CONFIG = os.path.join("configs", "vlm.json")
DEFAULT_MODEL_CONFIG = os.path.join("configs", "models.json")
KIND_LOCAL = "local"
KIND_CLOUD = "cloud"
MAX_PROFILES = 50
MODEL_CONFIG_VERSION = 1
MAX_DOWNLOAD_RECORDS = 50
# 配置读写与删除并发进行时，保证同一份 JSON 不被后写覆盖
_LOCK = threading.RLock()

# 云端权重下载镜像沿用 backends 里的默认值
DOWNLOAD_TIMEOUT = 30

# --------------------------------------------------------------------------- #
# 分层分类：能力 / 来源 / 状态 / 出处
# --------------------------------------------------------------------------- #

CAPABILITY_POSE = "pose"
CAPABILITY_VLM = "vlm"
CAPABILITY_CUSTOM = "custom_task"
CAPABILITIES = (CAPABILITY_POSE, CAPABILITY_VLM, CAPABILITY_CUSTOM)
CAPABILITY_LABELS = {
    CAPABILITY_POSE: "姿态估计（YOLO-pose，快且稳）",
    CAPABILITY_VLM: "多模态大模型（逐图询问，慢但灵活）",
    CAPABILITY_CUSTOM: "自定义任务（自带关键点定义）",
}

SOURCE_PRESET = "preset"          # 在内置目录里，本地还没有文件
SOURCE_LOCAL = "local"            # 用户自己登记的本地文件 / 本地服务
SOURCE_DOWNLOADED = "downloaded"  # 由本工具经镜像下载落盘
SOURCE_EXTERNAL = "external"      # 指向外部服务（云端 API）
SOURCE_LABELS = {
    SOURCE_PRESET: "内置目录",
    SOURCE_LOCAL: "本地",
    SOURCE_DOWNLOADED: "已下载",
    SOURCE_EXTERNAL: "外部服务",
}

STATUS_AVAILABLE = "available"
STATUS_INCOMPLETE = "incomplete"
STATUS_MISSING = "missing"
STATUS_INCOMPATIBLE = "incompatible"
STATUS_LABELS = {
    STATUS_AVAILABLE: "可用",
    STATUS_INCOMPLETE: "文件不完整",
    STATUS_MISSING: "未下载",
    STATUS_INCOMPATIBLE: "不兼容/暂不可用",
}

ORIGIN_ULTRALYTICS = "ultralytics"
ORIGIN_HUGGINGFACE = "huggingface"
ORIGIN_OLLAMA = "ollama"
ORIGIN_LMSTUDIO = "lmstudio"
ORIGIN_OPENAI = "openai"
ORIGIN_DASHSCOPE = "dashscope"
ORIGIN_CUSTOM = "custom"
ORIGIN_LABELS = {
    ORIGIN_ULTRALYTICS: "Ultralytics 官方发布",
    ORIGIN_HUGGINGFACE: "Hugging Face",
    ORIGIN_OLLAMA: "Ollama（本地）",
    ORIGIN_LMSTUDIO: "LM Studio（本地）",
    ORIGIN_OPENAI: "OpenAI 兼容端点",
    ORIGIN_DASHSCOPE: "阿里通义千问",
    ORIGIN_CUSTOM: "自定义",
}


def capability_label(value: str) -> str:
    return CAPABILITY_LABELS.get(str(value or ""), str(value or "未知能力"))


def source_label(value: str) -> str:
    return SOURCE_LABELS.get(str(value or ""), str(value or "未知来源"))


def status_label(value: str) -> str:
    return STATUS_LABELS.get(str(value or ""), str(value or "未知状态"))


def origin_label(value: str) -> str:
    return ORIGIN_LABELS.get(str(value or ""), str(value or "未知出处"))


# --------------------------------------------------------------------------- #
# 内置可下载目录（可被 configs/models.json 的 catalog 增删改）
# --------------------------------------------------------------------------- #
# 体积按 ultralytics/assets release v8.4.0 的实际字节数填写：n/s/m/v8m 由本机权重文件
# 算出并带 SHA256 校验值，其余由 HEAD 取 Content-Length。参数规模与 mAP 取自 Ultralytics
# 官方模型表（约值），官方未列明的留 None 并在中文说明里注明。
def _entry(**fields) -> dict:
    base = {
        "id": "", "display_name": "", "capability": CAPABILITY_POSE,
        "origin": ORIGIN_ULTRALYTICS, "filename": "", "params_m": None, "map_pose": None,
        "expected_mb": 0.0, "expected_bytes": 0, "size_verified": False, "sha256": "",
        "num_keypoints": 17, "keypoint_set": "COCO-17", "speed": "", "description": "",
        "recommended_for": "", "downloadable": True, "preset_visible": False,
    }
    base.update(fields)
    return base


PRESET_CATALOG: tuple = (
    _entry(id="yolo11n-pose", filename="yolo11n-pose.pt", display_name="YOLO11-nano 姿态",
           params_m=2.9, map_pose=50.4, expected_mb=6.0, expected_bytes=6255593,
           size_verified=True,
           sha256="869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
           speed="最快，CPU 也能实时", preset_visible=True,
           description="最小的姿态权重，下载快、显存占用低，适合先跑通流程和大批量抽帧预筛。"
                       "精度是 YOLO11 这一代里最低的，小人和重叠关节点容易漂。",
           recommended_for="试跑验证、大批量预标注、CPU 或轻薄本"),
    _entry(id="yolo11s-pose", filename="yolo11s-pose.pt", display_name="YOLO11-small 姿态",
           params_m=10.1, map_pose=58.5, expected_mb=19.4, expected_bytes=20373161,
           size_verified=True,
           sha256="1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba",
           speed="快", preset_visible=True,
           description="速度、精度、体积最均衡的一档，也是本项目的默认权重。绝大多数人工复核"
                       "场景够用：错点少，单张几百毫秒。",
           recommended_for="日常默认选择"),
    _entry(id="yolo11m-pose", filename="yolo11m-pose.pt", display_name="YOLO11-medium 姿态",
           params_m=20.9, map_pose=62.7, expected_mb=40.5, expected_bytes=42459307,
           size_verified=True,
           sha256="29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2",
           speed="中等", preset_visible=True,
           description="比 small 明显更准，速度慢一点，适合人多、重叠、小目标多的画面。"
                       "有 Apple Silicon 或独显时优先考虑这档。",
           recommended_for="精度优先、机器性能足够"),
    _entry(id="yolo11l-pose", filename="yolo11l-pose.pt", display_name="YOLO11-large 姿态",
           params_m=26.5, map_pose=64.9, expected_mb=50.7, expected_bytes=53164402, speed="慢",
           description="大模型档，精度接近这一代上限，需要更大显存和更长的等待时间。"
                       "和 medium 相比提升有限，建议先用 medium 对比再换。",
           recommended_for="成片级高质量标注"),
    _entry(id="yolo11x-pose", filename="yolo11x-pose.pt", display_name="YOLO11-xlarge 姿态",
           params_m=56.0, map_pose=65.9, expected_mb=113.0, expected_bytes=118481010,
           speed="最慢（本目录内）",
           description="YOLO11 这一代最大的姿态权重，113 MB，下载和推理都要留时间，"
                       "精度收益到 large 以后已经很边际。",
           recommended_for="追求极限精度、机器很强"),
    _entry(id="yolov8n-pose", filename="yolov8n-pose.pt", display_name="YOLOv8-nano 姿态",
           params_m=3.3, map_pose=50.4, expected_mb=6.5, expected_bytes=6832633, speed="最快",
           description="上一代 YOLOv8 的最小姿态权重，接口与 v11 完全一致。同精度下比 "
                       "v11-nano 略大略慢，一般只在复用旧工程时用它。",
           recommended_for="复用旧项目权重、与 v11 做双模型对照"),
    _entry(id="yolov8s-pose", filename="yolov8s-pose.pt", display_name="YOLOv8-small 姿态",
           params_m=11.6, map_pose=60.0, expected_mb=22.4, expected_bytes=23513657, speed="快",
           description="上一代 small 档，官方 mAP 与 v11-small 接近但参数更多，适合和 v11 "
                       "组成共识模型，看两代分歧点在哪。",
           recommended_for="多模型共识里的第二票"),
    _entry(id="yolov8m-pose", filename="yolov8m-pose.pt", display_name="YOLOv8-medium 姿态",
           params_m=26.8, map_pose=63.0, expected_mb=50.8, expected_bytes=53266977,
           size_verified=True,
           sha256="dbe539ea268db2534390942cfdf206e521f376f19e5415967a57f6a2ddfa3c90",
           speed="中等",
           description="上一代 medium 档，精度与 v11-medium 接近但体积更大，生态里现成教程"
                       "和二次训练权重最多。",
           recommended_for="和 v11-medium 交叉验证"),
    _entry(id="yolov8l-pose", filename="yolov8l-pose.pt", display_name="YOLOv8-large 姿态",
           params_m=46.8, map_pose=65.0, expected_mb=85.3, expected_bytes=89396617, speed="慢",
           description="上一代 large 档，参数明显多于 v11-large 而精度只高一到两点，性价比一般。",
           recommended_for="显存富余、想比 v11 多一票"),
    _entry(id="yolov8x-pose", filename="yolov8x-pose.pt", display_name="YOLOv8-xlarge 姿态",
           params_m=68.2, map_pose=65.5, expected_mb=133.0, expected_bytes=139420105,
           speed="最慢（本目录内）",
           description="上一代最大的姿态权重，133 MB，本目录里最占地方的一档。",
           recommended_for="只做精度对照"),
    _entry(id="yolo26n-pose", filename="yolo26n-pose.pt", display_name="YOLO26-nano 姿态",
           expected_mb=7.5, expected_bytes=7878574, speed="快",
           description="最新一代 YOLO26 的最小姿态权重，按 Ultralytics 说明是端到端（免 NMS）"
                       "设计；官方暂未公布这一档的参数规模与 mAP，先当实验档用。",
           recommended_for="想试新一代模型、愿意多花人工复核"),
    _entry(id="yolo26s-pose", filename="yolo26s-pose.pt", display_name="YOLO26-small 姿态",
           expected_mb=23.0, expected_bytes=24151790, speed="快",
           description="YOLO26 的 small 档，体积比 v11-small 略大。官方未列明参数与 mAP，"
                       "建议先小批量试标，和 yolo11s-pose 对比点准不准再决定长期用哪个。",
           recommended_for="新一代对照实验"),
    _entry(id="yolo26m-pose", filename="yolo26m-pose.pt", display_name="YOLO26-medium 姿态",
           expected_mb=46.8, expected_bytes=49036866, speed="中等",
           description="YOLO26 的 medium 档，官方未列明参数与 mAP。加载方式与 v11 一致，"
                       "新版权重偶尔要升级 ultralytics 包才读得动，报加载错就先升级。",
           recommended_for="新一代精度对照"),
)

PRESET_BY_FILENAME = {str(e["filename"]).lower(): dict(e) for e in PRESET_CATALOG}
PRESET_BY_ID = {str(e["id"]).lower(): dict(e) for e in PRESET_CATALOG}

# VLM 配置项到「出处」的映射：同一套 OpenAI 兼容协议，只按服务分开标注
VLM_ORIGIN_BY_PROVIDER = {
    "ollama": ORIGIN_OLLAMA, "lmstudio": ORIGIN_LMSTUDIO, "vllm": ORIGIN_CUSTOM,
    "openai": ORIGIN_OPENAI, "dashscope": ORIGIN_DASHSCOPE, "custom": ORIGIN_OPENAI,
}


def weights_dir() -> str:
    """权重目录：默认 ``models/``，可用 ``AUTOLABEL_WEIGHTS_DIR`` 覆盖。"""
    return os.environ.get("AUTOLABEL_WEIGHTS_DIR") or DEFAULT_WEIGHTS_DIR


def vlm_config_path() -> str:
    """多模态大模型配置文件路径：默认 ``configs/vlm.json``。"""
    return os.environ.get("AUTOLABEL_VLM_CONFIG") or DEFAULT_VLM_CONFIG


def model_config_path() -> str:
    """模型登记表路径：默认 ``configs/models.json``，只存路径/备注/下载记录，不存 Key。"""
    return os.environ.get("AUTOLABEL_MODEL_CONFIG") or DEFAULT_MODEL_CONFIG


# --------------------------------------------------------------------------- #
# 姿态权重
# --------------------------------------------------------------------------- #

def partial_weight(dir_: str, name: str) -> str:
    """半截下载/失败残留的文件属于哪个权重；不是残留返回空串。

    临时名固定成 ``x.pt.part``（重启后还能续传），并发由文件锁挡住；
    ``x.pt.<任何东西>.part`` 也一起认，用户手工改过名或历史版本留下的都能识别。
    """
    if name.endswith(".part"):
        head = name[: -len(".part")]
    elif name.endswith(".bad"):
        head = name[: -len(".bad")]
    else:
        return ""
    if head.endswith(".pt"):
        return head
    marker = head.rfind(".pt.")
    return head[: marker + 3] if marker > 0 else ""


def weight_names(dir_: Optional[str] = None) -> List[str]:
    """候选权重 = 界面预置 + 权重目录里实际存在的 ``.pt``（含自定义权重）。"""
    dir_ = weights_dir() if dir_ is None else dir_
    found: List[str] = []
    if os.path.isdir(dir_):
        for name in sorted(os.listdir(dir_)):
            if name.endswith(".pt") and os.path.isfile(os.path.join(dir_, name)):
                found.append(name)
            elif partial_weight(dir_, name):
                # 下载中断的残留（``x.pt.part`` / ``x.pt.4211.3.part`` / ``x.pt.bad``）
                # 也要能被看到、被删掉，否则只剩一个看不见又占着磁盘的文件
                bare = partial_weight(dir_, name)
                if bare not in found:
                    found.append(bare)
    # 目录里的权重按「推荐档在前」的顺序排在前面，用户自己放进来的自定义权重跟在后面
    presets = _catalog_weight_names() or list(UI_MODEL_CHOICES)
    return presets + [n for n in found if n not in presets]


def task_hint(name: str) -> Tuple[str, str]:
    """按文件名推「这是哪类权重」＋中文名，给界面贴类别标签用；认不出给空串。

    ``modelhub`` 顶部 import 了本模块，所以这里必须延迟导入，不然两个模块互相等。
    """
    from .backends import open_vocab_kind
    from .modelhub import TASK_LABELS, task_of

    task = task_of(name)
    kind = open_vocab_kind(name)
    label = str(TASK_LABELS.get(task) or "")
    if kind:
        label = f"{label or '目标检测'} · 开放词表（{kind}）"
    return task, label


def weight_info(name: str, dir_: Optional[str] = None, dataset_kps: int = 0) -> dict:
    """单个权重的状态：是否就绪、体积、来源、半截文件残留，外加目录里的中文说明。"""
    dir_ = weights_dir() if dir_ is None else dir_
    info = dict(model_status(name, weights_dir=dir_))
    info["task"], info["task_name"] = task_hint(name)
    path = str(info["path"])
    info["source"] = "disk" if os.path.isfile(path) else "preset"
    info["custom"] = _registry().get(name) is None
    info["mtime"] = (time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
                     if os.path.isfile(path) else None)
    leftover = 0
    for path_part in partial_paths(dir_, name):
        try:
            leftover += os.path.getsize(path_part)
        except OSError:
            pass
    info["partial_mb"] = round(leftover / 1048576, 1)
    info.update(_registry().weight_overlay(name, dataset_kps=dataset_kps))
    return info


def list_weights(dir_: Optional[str] = None, dataset_root: str = "") -> List[dict]:
    """权重清单，供「模型管理」面板与标注下拉框共用。

    ``dataset_root`` 非空时每行会带上「和本数据集关键点数量是否兼容」，只读一次索引。
    """
    kps, _name = dataset_keypoints(dataset_root) if dataset_root else (0, "")
    return [weight_info(name, dir_, dataset_kps=kps) for name in weight_names(dir_)]


def resolve_weight_in_dir(name: str, dir_: Optional[str] = None) -> str:
    """把权重名解析到权重目录里的绝对路径。

    ``yolo11n.pt`` 和 ``models/yolo11n.pt`` 都认——帮助里写的就是「权重文件名或
    路径」，而 ``--model models/yolo11n.pt`` 是抄命令行时最自然的写法，以前这条
    会被自己拒掉。指向目录外的路径仍然拒绝：那几个调用方会**删文件**，
    误删用户放在别处的权重比报一句错严重得多。
    """
    dir_ = weights_dir() if dir_ is None else dir_
    raw = str(name or "").strip().replace("\\", "/")
    base = os.path.basename(raw)
    if not base.endswith(".pt"):
        raise ValueError("只能管理权重目录里的 .pt 文件")
    if base != raw and os.path.dirname(os.path.abspath(raw)) != os.path.abspath(dir_):
        raise ValueError(f"只能管理 {dir_}/ 目录下的权重，别的路径请先拷进来：{raw}")
    return os.path.join(os.path.abspath(dir_), base)


def delete_weight(name: str, active_names: Sequence[str] = (), dir_: Optional[str] = None) -> dict:
    """删除权重文件（连同半截的 ``.part``/``.bad``）；正在被任务使用的模型不许删。"""
    path = resolve_weight_in_dir(name, dir_)
    base = os.path.basename(path)
    active = {str(a) for a in active_names}
    if base in active:
        raise ValueError(f"{base} 正在被标注任务使用，请先在上传面板点「清除任务」")
    resolved_dir = os.path.dirname(path)
    targets = [path, path + ".bad"] + partial_paths(resolved_dir, base)
    removed, freed = [], 0
    for target in targets:
        if not os.path.exists(target):
            continue
        freed += os.path.getsize(target)
        os.remove(target)
        removed.append(os.path.basename(target))
    if not removed:
        raise FileNotFoundError(f"权重不存在：{base}")
    return {"name": base, "removed": removed, "freed_mb": round(freed / 1048576, 1),
            "weights": list_weights(dir_)}


# --- 后台下载 --- #
_DOWNLOADS: Dict[str, dict] = {}
_DL_LOCK = threading.Lock()


def downloads() -> List[dict]:
    with _DL_LOCK:
        return [dict(task) for task in _DOWNLOADS.values()]


def download_state(name: str) -> Optional[dict]:
    base = os.path.basename(str(name or "").strip())
    with _DL_LOCK:
        task = _DOWNLOADS.get(base)
        return dict(task) if task else None


def start_download(name: str, force: bool = False, mirror: Optional[str] = None,
                   background: bool = True) -> dict:
    """下载（或重下）一个权重；同名任务在跑就直接复用，不重复起线程。"""
    dir_ = weights_dir()
    path = resolve_weight_in_dir(name, dir_)
    base = os.path.basename(path)
    with _DL_LOCK:
        current = _DOWNLOADS.get(base)
        if current and current.get("stage") == "running":
            return dict(current)
        task = {"name": base, "stage": "running", "message": "准备下载", "done": 0, "total": 0,
                "percent": 0, "started_at": now_str(), "finished_at": None,
                "ok": None, "error": "", "force": bool(force)}
        _DOWNLOADS[base] = task

    def work() -> None:
        started = time.time()
        entry = _registry().get(base) or {}
        record = {"model_id": str(entry.get("id") or base), "name": base,
                  "url": weights_url(base, mirror), "mirror": _mirror_label(mirror),
                  "sha256_expected": str(entry.get("sha256") or ""), "sha256_actual": "",
                  "sha256_match": None, "expected_bytes": int(entry.get("expected_bytes") or 0),
                  "bytes": 0, "size_mb": 0.0, "temp_name": os.path.basename(path) + ".part",
                  "force": bool(force), "ok": False, "error": "", "seconds": 0.0,
                  "started_at": task.get("started_at") or now_str(), "finished_at": ""}

        def on_progress(done: int, total: int) -> None:
            with _DL_LOCK:
                task["done"], task["total"] = done, total
                task["percent"] = int(done * 100 / total) if total else 0
                task["message"] = (f"下载中 {done / 1048576:.1f}/{total / 1048576:.1f} MB"
                                   if total else f"已下载 {done / 1048576:.1f} MB")

        try:
            # ensure_weights 内部会把网络异常吞掉退回原路径，这里把原因接出来给人看
            failures: List[str] = []
            ensure_weights(base, weights_dir=dir_, mirror=mirror, verbose=False,
                           force=force, progress=on_progress, timeout=DOWNLOAD_TIMEOUT,
                           on_error=lambda exc: failures.append(f"{type(exc).__name__}: {exc}"))
            info = model_status(base, weights_dir=dir_)
            record["bytes"] = int(os.path.getsize(path)) if os.path.isfile(path) else 0
            record["size_mb"] = round(record["bytes"] / 1048576, 1)
            record["sha256_actual"] = sha256_file(path) if info["ready"] else ""
            if record["sha256_expected"] and record["sha256_actual"]:
                record["sha256_match"] = (record["sha256_actual"].lower()
                                          == record["sha256_expected"].lower())
            record["ok"] = bool(info["ready"])
            record["error"] = "" if info["ready"] else (failures[-1] if failures else "文件仍不完整")
            record["seconds"] = round(time.time() - started, 1)
            record["finished_at"] = now_str()
            _registry().record_download(record)
            with _DL_LOCK:
                task["stage"] = "done" if info["ready"] else "failed"
                task["ok"] = bool(info["ready"])
                task["finished_at"] = now_str()
                if info["ready"]:
                    task["message"] = f"就绪 {info['size_mb']} MB"
                    if record["sha256_match"] is False:
                        # 体积对得上但哈希不一致（镜像偶发），只提示不拦，避免误伤可用文件
                        task["message"] += "（SHA256 与目录记录不一致，建议删除后重下）"
                else:
                    task["message"] = "下载失败，文件仍不完整"
                    task["error"] = failures[-1] if failures else "下载结果不是完整的权重包，可再点一次重试"
            return
        except Exception as exc:  # noqa: BLE001 后台线程必须吞掉异常
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] = round(time.time() - started, 1)
            record["finished_at"] = now_str()
            _registry().record_download(record)
            with _DL_LOCK:
                task["stage"] = "failed"
                task["ok"] = False
                task["finished_at"] = now_str()
                task["error"] = f"{type(exc).__name__}: {exc}"
                task["message"] = "下载失败"

    if background:
        threading.Thread(target=work, name=f"autolabel-weight-{base}", daemon=True).start()
    else:
        work()
    with _DL_LOCK:
        return dict(_DOWNLOADS[base])


# --------------------------------------------------------------------------- #
# 多模态大模型配置
# --------------------------------------------------------------------------- #

# 预设只填默认地址，用户可改；OpenAI 兼容端点都能用同一套请求格式
VLM_PRESETS: tuple = (
    {"id": "ollama", "label": "Ollama（本地）", "kind": KIND_LOCAL,
     "base_url": "http://localhost:11434/v1", "model": "qwen2.5vl:7b", "needs_key": False},
    {"id": "lmstudio", "label": "LM Studio（本地）", "kind": KIND_LOCAL,
     "base_url": "http://localhost:1234/v1", "model": "", "needs_key": False},
    {"id": "vllm", "label": "vLLM（本地/内网）", "kind": KIND_LOCAL,
     "base_url": "http://localhost:8000/v1", "model": "Qwen2.5-VL-7B-Instruct", "needs_key": False},
    {"id": "openai", "label": "OpenAI（云端）", "kind": KIND_CLOUD,
     "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "needs_key": True},
    {"id": "dashscope", "label": "阿里通义千问 VL（云端）", "kind": KIND_CLOUD,
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "model": "qwen-vl-max", "needs_key": True},
    {"id": "custom", "label": "自定义（OpenAI 兼容）", "kind": KIND_CLOUD,
     "base_url": "", "model": "", "needs_key": False},
)

PRESET_BY_ID = {p["id"]: p for p in VLM_PRESETS}
_KEY_FIELDS = ("provider", "base_url", "model")


def empty_vlm_config() -> dict:
    return {"profiles": [], "default": None}


def load_vlm(path: Optional[str] = None) -> dict:
    """读配置；文件缺失或损坏按空表处理，不让界面打不开。"""
    path = vlm_config_path() if path is None else path
    if not os.path.exists(path):
        return empty_vlm_config()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return empty_vlm_config()
    if not isinstance(data, dict):
        return empty_vlm_config()
    profiles = [p for p in (data.get("profiles") or []) if isinstance(p, dict) and p.get("id")]
    default = data.get("default")
    if default and not any(p["id"] == default for p in profiles):
        default = profiles[0]["id"] if profiles else None
    migrated = _migrate_keys(profiles)
    if migrated:
        with _LOCK:
            save_vlm({"profiles": profiles, "default": default}, path)
    for profile in profiles:
        # 明文 Key 只在这一刻回到内存里，落盘的配置永远只有 key_ref
        profile["api_key"] = secret_store.get_secret(str(profile.get("key_ref") or ""))
    return {"profiles": profiles, "default": default}


def key_ref_for(profile_id: str) -> str:
    """某条 VLM 配置在 Key 文件里的引用名。"""
    ref = f"vlm:{profile_id}"
    return ref if secret_store.valid_ref(ref) else ""


def _migrate_keys(profiles: Sequence[dict]) -> bool:
    """把老配置里残留的明文 ``api_key`` 挪进 Key 文件，返回是否改动了内容。"""
    moved = False
    for profile in profiles:
        plain = str(profile.pop("api_key", "") or "")
        if not plain or profile.get("key_ref"):
            continue
        ref = key_ref_for(str(profile.get("id") or ""))
        if not ref:
            continue
        secret_store.put_secret(ref, plain)
        profile["key_ref"] = ref
        moved = True
    return moved


def save_vlm(config: dict, path: Optional[str] = None) -> None:
    path = vlm_config_path() if path is None else path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    profiles = [{k: v for k, v in item.items() if k != "api_key"}
                for item in (config.get("profiles") or []) if isinstance(item, dict)]
    payload = {"profiles": profiles, "default": config.get("default"), "updated_at": now_str()}
    atomic_write_json(path, payload)


def mask_key(key: str) -> str:
    """API Key 掩码：只露后 4 位，界面回显用它，看不到原文。"""
    key = str(key or "")
    if not key:
        return ""
    if len(key) <= 8:
        # 短 Key 全打星等于没提示，留首尾各一个字符便于比对
        return f"{key[0]}***{key[-1]}" if len(key) > 2 else "**"
    return f"{key[:3]}…{key[-4:]}"


def normalize_base_url(base_url: str) -> str:
    """补协议头、去掉尾斜杠，避免拼出 ``//chat/completions`` 这种地址。"""
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("请填写服务地址（base_url）")
    if "://" not in base:
        base = "http://" + base
    if not base.lower().startswith(("http://", "https://")):
        raise ValueError("服务地址只支持 http/https")
    return base.rstrip("/")


def normalize_profile(raw: dict, existing: Optional[dict] = None) -> dict:
    """校验并补全一条 VLM 配置；``existing`` 用于继承未改动字段与掩码后的 Key。"""
    raw = dict(raw or {})
    preset = PRESET_BY_ID.get(str(raw.get("provider") or ""))
    kind = str(raw.get("kind") or (preset or {}).get("kind") or KIND_CLOUD)
    if kind not in (KIND_LOCAL, KIND_CLOUD):
        raise ValueError(f"类型只能是本地或云端，收到 {kind}")
    name = (str(raw.get("name") or "").strip() or str((existing or {}).get("name") or "").strip()
            or str((preset or {}).get("label") or "").strip())
    if not name:
        raise ValueError("请填写配置名称")
    profile = {
        "id": str(raw.get("id") or (existing or {}).get("id") or f"vlm_{uuid.uuid4().hex[:8]}"),
        "name": name[:40],
        "provider": str(raw.get("provider") or (preset or {}).get("id") or "custom"),
        "kind": kind,
        "base_url": normalize_base_url(raw.get("base_url") or (existing or {}).get("base_url") or ""),
        "model": str(raw.get("model") or (existing or {}).get("model") or "").strip()[:80],
        # 多模态标注单张可能要几十秒，默认给足余量（连通性测试仍按此值等待）
        "timeout": _clamp_int(raw.get("timeout") if raw.get("timeout") is not None
                              else (existing or {}).get("timeout"), 60, 1, 300),
        # 单条配置的推理参数：一次回复的 token 上限、送图的最长边、是否强制 JSON 模式
        "max_tokens": _clamp_int(raw.get("max_tokens") if raw.get("max_tokens") is not None
                                 else (existing or {}).get("max_tokens"), 4096, 64, 32768),
        "image_limit": _clamp_int(raw.get("image_limit") if raw.get("image_limit") is not None
                                  else (existing or {}).get("image_limit"), 1024, 256, 4096),
        "json_mode": _as_bool(raw.get("json_mode") if raw.get("json_mode") is not None
                              else (existing or {}).get("json_mode"), False),
        "remark": str(raw.get("remark") or (existing or {}).get("remark") or "")[:200],
    }
    if not profile["model"]:
        raise ValueError("请填写模型名（例如 qwen2.5vl:7b 或 gpt-4o-mini）")
    # Key 进 Key 文件，配置里只留引用名；api_key 只在内存里流转到落盘前一刻
    key = _resolve_key(raw, existing, profile)
    profile["key_ref"] = str((existing or {}).get("key_ref") or key_ref_for(profile["id"]) or "")
    profile["api_key"] = key
    if profile["key_ref"]:
        secret_store.put_secret(profile["key_ref"], key)
    kept = (existing or {}).get("last_test")
    if kept and all(str((existing or {}).get(f)) == str(profile.get(f)) for f in _KEY_FIELDS):
        profile["last_test"] = kept
    return profile


def _resolve_key(raw: dict, existing: Optional[dict], profile: dict) -> str:
    """掩码值或空串表示「不改」，避免前端回填时把掩码写回真实 Key。"""
    incoming = str(raw.get("api_key") if raw.get("api_key") is not None else "")
    stored = str((existing or {}).get("api_key") or "")
    if incoming and incoming != mask_key(stored) and not incoming.startswith("sk-…"):
        return incoming.strip()[:200]
    return stored


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def public_profile(profile: dict) -> dict:
    """对外返回的配置：Key 掩码 + 是否已填。"""
    out = {k: v for k, v in profile.items() if k != "api_key"}
    key = str(profile.get("api_key") or "")
    out["api_key_masked"] = mask_key(key)
    out["has_key"] = bool(key)
    return out


def vlm_choices() -> dict:
    """上传弹层用的精简清单：只给「能选哪几条多模态配置」和最近测试是否通过。"""
    config = load_vlm()
    choices: List[dict] = []
    for profile in config["profiles"]:
        tested = profile.get("last_test") or {}
        choices.append({
            "id": profile["id"],
            "name": profile.get("name", ""),
            "kind": profile.get("kind", "local"),
            "provider": profile.get("provider", "custom"),
            "model": profile.get("model", ""),
            "ok": bool(tested.get("ok")),
            "tested": bool(tested),
            "checked_at": str(tested.get("checked_at", "")),
        })
    return {"default": config.get("default") or "", "profiles": choices}


def vlm_payload(dataset_root: str = "") -> dict:
    """界面一次要拿的全部模型信息（权重 + 下载状态 + VLM 配置）。"""
    config = load_vlm()
    registry = _registry()
    return {
        "weights": list_weights(dataset_root=dataset_root),
        "weights_dir": os.path.abspath(weights_dir()),
        "downloads": downloads(),
        "models": registry.models(dataset_root=dataset_root),
        "catalog": registry.catalog(),
        "defaults": registry.defaults(),
        "secrets_path": secret_store.secrets_path(),
        "model_config_path": os.path.abspath(model_config_path()),
        "download_records": registry.download_records(),
        "vlm": {
            "config_path": os.path.abspath(vlm_config_path()),
            "presets": [dict(p) for p in VLM_PRESETS],
            "profiles": [public_profile(p) for p in config["profiles"]],
            "default": config["default"],
        },
    }


def upsert_profile(raw: dict) -> dict:
    """新增或更新一条 VLM 配置；同名配置在没有 id 时会被更新而不是复制。"""
    with _LOCK:
        config = load_vlm()
        profiles = config["profiles"]
        raw = dict(raw or {})
        existing = None
        wanted = str(raw.get("id") or "").strip()
        if wanted:
            existing = next((p for p in profiles if p["id"] == wanted), None)
            if existing is None:
                raise ValueError(f"未知配置: {wanted}")
        else:
            existing = next((p for p in profiles
                             if str(p.get("name")) == str(raw.get("name") or "").strip()), None)
        profile = normalize_profile(raw, existing)
        if existing is None and len(profiles) >= MAX_PROFILES:
            raise ValueError(f"配置数量已达上限 {MAX_PROFILES} 条，先删掉不用的")
        if existing is None:
            clash = next((p for p in profiles if str(p.get("name")) == profile["name"]), None)
            if clash is not None:
                raise ValueError(f"已有同名配置「{profile['name']}」，换个名字或直接编辑它")
            profiles.append(profile)
        else:
            profiles[[p["id"] for p in profiles].index(existing["id"])] = profile
        config["default"] = config["default"] or profile["id"]
        save_vlm(config)
    return public_profile(profile)


def delete_profile(profile_id: str) -> dict:
    with _LOCK:
        config = load_vlm()
        before = len(config["profiles"])
        gone = [p for p in config["profiles"] if p["id"] == profile_id]
        config["profiles"] = [p for p in config["profiles"] if p["id"] != profile_id]
        if len(config["profiles"]) == before:
            raise ValueError(f"未知配置: {profile_id}")
        if config.get("default") == profile_id:
            config["default"] = config["profiles"][0]["id"] if config["profiles"] else None
        save_vlm(config)
        # Key 得跟着配置一起走。配置文件里只留 ``key_ref``，真 Key 在 Key 文件里；
        # 只删配置的话那串明文就再没人引用了——界面里看不见、也删不掉，
        # 用户以为换掉/删掉了，磁盘上其实还躺着一把幽灵密钥。
        for profile in gone:
            ref = str(profile.get("key_ref") or "")
            if ref:
                secret_store.delete_secret(ref)
    return config_summary(config)


def set_default(profile_id: str) -> dict:
    with _LOCK:
        config = load_vlm()
        if profile_id and not any(p["id"] == profile_id for p in config["profiles"]):
            raise ValueError(f"未知配置: {profile_id}")
        config["default"] = profile_id or None
        save_vlm(config)
    return config_summary(config)


def record_test(profile_id: str, result: dict) -> None:
    """把连通性结果写回配置，界面能一眼看到上次是否可用。"""
    with _LOCK:
        config = load_vlm()
        profile = next((p for p in config["profiles"] if p["id"] == profile_id), None)
        if profile is None:
            return
        profile["last_test"] = result
        save_vlm(config)


def config_summary(config: dict) -> dict:
    return {"profiles": [public_profile(p) for p in config["profiles"]],
            "default": config.get("default")}


def get_profile(profile_id: str) -> dict:
    config = load_vlm()
    profile = next((p for p in config["profiles"] if p["id"] == profile_id), None)
    if profile is None:
        raise ValueError(f"未知配置: {profile_id}")
    return profile


def find_profile(wanted: str) -> Optional[dict]:
    """按配置 ID 或名称找一条多模态配置，找不到返回 ``None``（不抛异常）。

    命令行允许写人类可读的名称，所以顺序是：ID 精确 → 名称精确（忽略大小写）→ 名称前缀。
    """
    text = str(wanted or "").strip()
    if not text:
        return None
    profiles = load_vlm()["profiles"]
    hit = next((p for p in profiles if p["id"] == text), None)
    if hit:
        return hit
    lowered = text.lower()
    hit = next((p for p in profiles if str(p.get("name", "")).lower() == lowered), None)
    if hit:
        return hit
    return next((p for p in profiles if str(p.get("name", "")).lower().startswith(lowered)), None)


def test_saved_profile(profile_id: str) -> dict:
    """测试一条已保存的配置，并把结果写回。"""
    profile = get_profile(profile_id)
    result = test_profile(profile)
    record_test(profile_id, result)
    return result


def test_profile(profile: dict) -> dict:
    """连通性验证：先列模型（便宜），失败再发一条极小的 chat 请求。

    返回的 message 直接给人看，所以要指出下一步动作（服务没起 / Key 不对 / 地址少了 /v1）。
    """
    started = time.time()
    timeout = _clamp_int(profile.get("timeout"), 15, 1, 300)
    headers = {"User-Agent": "autolabel-vlm"}
    key = str(profile.get("api_key") or "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    base = normalize_base_url(profile.get("base_url") or "")

    notes: List[str] = []
    status, body, err = _request(base + "/models", headers, timeout)
    models = _extract_models(body) if status == 200 else []
    if status == 200 and models:
        return _test_result(True, started, f"连接正常，可见 {len(models)} 个模型："
                           + "、".join(models[:5]) + ("…" if len(models) > 5 else ""), models)
    if status == 200:
        notes.append("/models 返回空列表")
    elif err:
        notes.append(err)

    payload = {"model": str(profile.get("model") or ""), "temperature": 0,
               "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    status2, body2, err2 = _request(base + "/chat/completions", headers, timeout, payload)
    if status2 == 200:
        suffix = f"（/models 不可用：{notes[0]}）" if notes else ""
        return _test_result(True, started, f"连接正常，chat 接口可用{suffix}", models)
    detail = err2 or err or (notes[0] if notes else "接口无响应")
    return _test_result(False, started, detail, models)


def _test_result(ok: bool, started: float, message: str, models: List[str]) -> dict:
    return {"ok": bool(ok), "message": message, "latency_ms": int((time.time() - started) * 1000),
            "models": models[:20], "checked_at": now_str()}


def _request(url: str, headers: Dict[str, str], timeout: int,
             payload: Optional[dict] = None):
    """发一个 HTTP 请求，返回 (状态码, 文本, 人话错误)；不抛异常。"""
    data = None
    request = urllib.request.Request(url, headers=dict(headers))
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, data=data, timeout=timeout) as resp:
            return int(getattr(resp, "status", 200) or 200), _read_text(resp), ""
    except urllib.error.HTTPError as exc:
        body = _read_text(exc)
        return int(exc.code), body, _http_error_hint(int(exc.code), url, body)
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        return 0, "", _url_error_hint(reason, url)
    except Exception as exc:  # noqa: BLE001 超时/编码问题统一成文案
        return 0, "", f"{type(exc).__name__}: {exc}"


def _read_text(resp) -> str:
    try:
        raw = resp.read()
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw or "")


def _http_error_hint(code: int, url: str, body: str) -> str:
    """把状态码翻译成人能照着做的下一步。"""
    snippet = " ".join(body.split())[:160]
    if code in (401, 403):
        return f"{code} 未授权：API Key 不对或没有该模型权限" + (f"（{snippet}）" if snippet else "")
    if code == 404:
        return ("404 找不到接口：检查服务地址是否完整（多数服务要写成 …/v1），"
                "以及模型名是否存在" + (f"（{snippet}）" if snippet else ""))
    if code == 429:
        return "429 请求过于频繁或额度不足：稍后再试，或换个模型/Key"
    if 500 <= code < 600:
        return f"{code} 服务端报错：确认模型已加载" + (f"（{snippet}）" if snippet else "")
    return f"HTTP {code}：{url}" + (f"（{snippet}）" if snippet else "")


def _url_error_hint(reason: str, url: str) -> str:
    lowered = reason.lower()
    if "refused" in lowered or "unavailable" in lowered:
        return (f"连不上 {url}：本地服务没启动？Ollama 先跑 `ollama serve`，"
                f"LM Studio 打开本地服务端")
    if "timed out" in lowered or "timeout" in lowered:
        return f"请求超时（{url}）：模型可能还在加载，把超时调大再试"
    if "ssl" in lowered or "certificate" in lowered:
        return f"证书校验失败（{url}）：检查 https 证书或改用 http"
    if "nodename" in lowered or "name or service not known" in lowered or "nodenotnames" in lowered:
        return f"域名解析失败（{url}）：检查地址拼写与网络"
    return f"请求失败（{url}）：{reason}"


def _extract_models(body: str) -> List[str]:
    """从 /models 响应里挑模型名，兼容 OpenAI 与 Ollama 两种格式。"""
    try:
        data = json.loads(body or "")
    except json.JSONDecodeError:
        return []
    raw: List[object] = []
    if isinstance(data, dict):
        raw = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        raw = data
    names: List[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            value = item.get("model") or item.get("name") or item.get("id")
            if value:
                names.append(str(value))
    return names[:50]


# --------------------------------------------------------------------------- #
# 统一模型清单：内置目录 + 本地注册 + 多模态配置
# --------------------------------------------------------------------------- #

# 目录条目里允许被 configs/models.json 覆盖的字段（hidden 属于用户自己的标记）
_OVERRIDABLE_KEYS = ("display_name", "capability", "origin", "filename", "params_m", "map_pose",
                     "expected_mb", "expected_bytes", "size_verified", "sha256", "num_keypoints",
                     "keypoint_set", "speed", "description", "recommended_for", "downloadable",
                     "preset_visible")


def normalize_model_id(value: str) -> str:
    """把用户写的模型标识收敛成裸名：去路径、去 ``.pt`` 后缀。"""
    text = os.path.basename(str(value or "").strip().replace("\\", "/"))
    if text.endswith(".pt"):
        text = text[: -len(".pt")]
    return text.strip()


def empty_model_config() -> dict:
    return {"version": MODEL_CONFIG_VERSION, "defaults": {}, "catalog": {}, "local": [],
            "downloads": [], "last_used": {}, "metadata": {}}


def load_model_config(path: Optional[str] = None) -> dict:
    """读模型登记表；文件缺失或损坏按空表处理，模型管理面板要始终打得开。"""
    path = model_config_path() if path is None else path
    config = empty_model_config()
    if not os.path.exists(path):
        return config
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return config
    if not isinstance(data, dict):
        return config
    for key in ("defaults", "catalog", "last_used", "metadata"):
        value = data.get(key)
        if isinstance(value, dict):
            config[key] = {str(k): v for k, v in value.items()}
    for key in ("local", "downloads"):
        value = data.get(key)
        if isinstance(value, list):
            config[key] = [v for v in value if isinstance(v, dict)]
    return config


def save_model_config(config: dict, path: Optional[str] = None) -> None:
    """原子写回登记表（0600 足够，里面没有任何 Key）。"""
    path = model_config_path() if path is None else path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    merged = load_model_config(path)
    merged.update({k: v for k, v in (config or {}).items() if k in empty_model_config()})
    merged["version"] = MODEL_CONFIG_VERSION
    merged["updated_at"] = now_str()
    atomic_write_json(path, merged, newline=True)


def dataset_keypoints(root: str) -> "tuple[int, str]":
    """读数据集的关键点数量与名称，用于兼容性判断；读不到返回 ``(0, "")``。"""
    if not root:
        return 0, ""
    try:
        index = load_index(root)
        return int(index.num_keypoints or 0), str(index.name or "")
    except Exception:  # noqa: BLE001 数据集没初始化也要能列出模型
        return 0, ""


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """算文件 SHA256，用来核对下载结果与目录里的校验值；读不到返回空串。"""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _is_zip(path: str) -> bool:
    import zipfile

    return bool(path) and os.path.isfile(path) and zipfile.is_zipfile(path)


def _entry_path(entry: dict, weights: str) -> str:
    """模型对应的本地文件路径：注册过的绝对路径优先，否则回到权重目录。"""
    raw = str((entry or {}).get("path") or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(raw))
    filename = str((entry or {}).get("filename") or "").strip()
    return os.path.join(os.path.abspath(weights), filename) if filename else ""


def _clean_override(raw: object) -> dict:
    """只保留允许的目录覆盖字段，避免手改 JSON 时塞进无关键。"""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, object] = {}
    for key in _OVERRIDABLE_KEYS:
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    return out


def _blank_row(ident: str, filename: str) -> dict:
    """目录里没有的自定义权重也要有一行可展示的记录。"""
    return {"id": ident or filename, "display_name": filename or ident, "capability": CAPABILITY_POSE,
            "origin": ORIGIN_CUSTOM, "filename": filename, "params_m": None, "map_pose": None,
            "expected_mb": 0.0, "expected_bytes": 0, "size_verified": False, "sha256": "",
            "num_keypoints": 17, "keypoint_set": "COCO-17", "speed": "",
            "description": "用户自行放入权重目录的自定义权重，参数与精度以文件自带元数据为准。",
            "recommended_for": "自定义训练/二次微调的权重", "downloadable": False,
            "preset_visible": False}


def _mirror_label(mirror: Optional[str] = None) -> str:
    """下载记录里的镜像名：空串表示直连 GitHub。"""
    from .backends import DEFAULT_MIRROR

    value = (DEFAULT_MIRROR if mirror is None else mirror)
    return str(value or "").strip() or "直连"


def _resolve_default_ids(rows: Sequence[dict], explicit: Optional[dict] = None) -> Dict[str, str]:
    """每类能力实际生效的默认模型：用户显式设的优先，其次「已就绪的推荐档」。"""
    explicit = dict(explicit or {})
    effective: Dict[str, str] = {}
    for capability in CAPABILITIES:
        known = [row for row in rows if str(row.get("capability")) == capability]
        chosen = str(explicit.get(capability) or "")
        if chosen and any(str(row.get("id")) == chosen for row in known):
            effective[capability] = chosen
            continue
        ready = [row for row in known if row.get("ready")]
        preferred = next((row for row in ready if row.get("preset_visible")), None) \
            or (ready[0] if ready else (known[0] if known else None))
        effective[capability] = str(preferred["id"]) if preferred else ""
    return effective


class ModelRegistry:
    """把内置可下载目录、本地注册模型、多模态配置合成一份带中文说明的统一清单。

    所有路径都在调用时重新解析环境变量，所以换权重目录、换配置文件、测试里 monkeypatch
    都能立刻生效，不需要重建实例。
    """

    def __init__(self, config_path: Optional[str] = None, weights_root: Optional[str] = None):
        self._config_path = config_path
        self._weights_root = weights_root

    # --- 基本属性 ---
    @property
    def config_path(self) -> str:
        return model_config_path() if self._config_path is None else os.path.abspath(self._config_path)

    @property
    def weights_root(self) -> str:
        return weights_dir() if self._weights_root is None else os.path.abspath(self._weights_root)

    def load(self) -> dict:
        return load_model_config(self.config_path)

    def save(self, config: dict) -> None:
        save_model_config(config, self.config_path)

    # --- 内置可下载目录 ---
    def catalog(self, include_hidden: bool = False) -> List[dict]:
        """内置目录 + 用户在 ``catalog`` 里追加/改写的条目，顺序即界面展示顺序。"""
        overrides = self.load().get("catalog") or {}
        rows: List[dict] = []
        seen = set()
        for preset in PRESET_CATALOG:
            row = dict(preset)
            row.update(_clean_override(overrides.get(row["id"])))
            seen.add(row["id"])
            rows.append(row)
        for key in sorted(overrides):
            if key in seen:
                continue
            value = _clean_override(overrides.get(key))
            filename = str(value.get("filename") or (key if key.endswith(".pt") else f"{key}.pt"))
            row = _blank_row(key, filename)
            row["description"] = ""
            row.update(value)
            rows.append(row)
        if include_hidden:
            return rows
        return [row for row in rows if not (overrides.get(row["id"]) or {}).get("hidden")]

    def catalog_downloadable(self) -> List[dict]:
        """「可下载模型目录」：带中文说明、预计体积和下载地址。"""
        config = self.load()
        last_used = config.get("last_used") or {}
        rows: List[dict] = []
        for row in self.catalog():
            if not row.get("downloadable") or not row.get("filename"):
                continue
            item = self._file_state(row)
            item["url"] = weights_url(str(row["filename"]))
            item["mirror"] = _mirror_label()
            item["last_used"] = str(last_used.get(str(row["id"])) or "")
            item["capability_label"] = capability_label(item.get("capability"))
            item["source_label"] = source_label(item.get("source"))
            item["origin_label"] = origin_label(item.get("origin"))
            item["status_label"] = status_label(item.get("status"))
            rows.append(item)
        return rows

    # --- 本地注册的模型 ---
    def local_rows(self) -> List[dict]:
        rows: List[dict] = []
        for item in self.load().get("local") or []:
            ident = str(item.get("id") or "").strip()
            if not ident:
                continue
            row = _blank_row(ident, str(item.get("filename") or f"{ident}.pt"))
            row["path"] = str(item.get("path") or "")
            for key in ("display_name", "capability", "origin", "description", "recommended_for",
                        "num_keypoints", "keypoint_set", "speed", "params_m"):
                if item.get(key) not in (None, ""):
                    row[key] = item[key]
            row["added_at"] = str(item.get("added_at") or "")
            row["registered"] = True
            rows.append(row)
        return rows

    # --- 多模态配置映射成统一行 ---
    def vlm_rows(self) -> List[dict]:
        last_used = self.load().get("last_used") or {}
        rows: List[dict] = []
        for profile in load_vlm()["profiles"]:
            pid = str(profile["id"])
            tested = profile.get("last_test") or {}
            failed = bool(tested) and not tested.get("ok")
            reasons = [] if not failed else [
                f"最近一次连通性测试未通过：{tested.get('message') or '原因未知'}"]
            if not tested:
                reasons = ["还没做过连通性测试，先到「模型管理」点「保存并测试」"]
            rows.append({
                "id": pid,
                "display_name": str(profile.get("name") or pid),
                "capability": CAPABILITY_VLM,
                "origin": VLM_ORIGIN_BY_PROVIDER.get(str(profile.get("provider")), ORIGIN_CUSTOM),
                "source": SOURCE_LOCAL if profile.get("kind") == KIND_LOCAL else SOURCE_EXTERNAL,
                "filename": "", "path": "", "downloadable": False, "preset_visible": False,
                "exists": True, "ready": bool(tested.get("ok")), "verified": bool(tested.get("ok")),
                "size_mb": 0.0, "expected_mb": 0.0, "params_m": None, "map_pose": None,
                "num_keypoints": 17, "keypoint_set": "COCO-17",
                "model": str(profile.get("model") or ""),
                "base_url": str(profile.get("base_url") or ""),
                "kind": str(profile.get("kind") or KIND_CLOUD),
                "provider": str(profile.get("provider") or "custom"),
                "timeout": _clamp_int(profile.get("timeout"), 60, 1, 300),
                "max_tokens": _clamp_int(profile.get("max_tokens"), 4096, 64, 32768),
                "image_limit": _clamp_int(profile.get("image_limit"), 1024, 256, 4096),
                "json_mode": _as_bool(profile.get("json_mode"), False),
                "key_ref": str(profile.get("key_ref") or ""),
                "has_key": bool(profile.get("api_key")),
                "api_key_masked": mask_key(profile.get("api_key") or ""),
                "description": f"{profile.get('name') or pid}：{'本地服务' if profile.get('kind') == KIND_LOCAL else '云端 API'}"
                               f" {profile.get('base_url') or ''} 上的 {profile.get('model') or '未填模型名'}。"
                               "逐张图询问，速度明显慢于 YOLO，但换场景不用重训权重。",
                "recommended_for": "画面里没有人体姿态先验、或 YOLO 权重明显标错的样本",
                "speed": "慢（单张数十秒）",
                "status": STATUS_INCOMPATIBLE if failed else STATUS_AVAILABLE,
                "checked_at": str(tested.get("checked_at") or ""),
                "compatibility_reasons": reasons,
                # 多模态配置那句「备注」是 vlm.json 里的 remark，属于配置本身，不是模型备注
                "note": str(profile.get("remark") or ""),
                "last_used": str(last_used.get(pid) or ""),
            })
        for row in rows:
            row["status_label"] = status_label(row["status"])
        return rows

    # --- 查询 ---
    def get(self, model_id: str) -> Optional[dict]:
        """按模型 ID 或权重文件名找一条目录/本地模型（不含多模态配置）。"""
        key = normalize_model_id(model_id)
        if not key:
            return None
        needles = {key, f"{key}.pt"}
        for row in self.catalog(include_hidden=True) + self.local_rows():
            haystack = {normalize_model_id(str(row.get("id") or "")),
                        normalize_model_id(str(row.get("filename") or "")),
                        normalize_model_id(str(os.path.basename(str(row.get("path") or ""))))}
            if needles & haystack:
                return row
        return None

    def get_any(self, model_id: str) -> Optional[dict]:
        """同上，但允许指向一条多模态配置。"""
        row = self.get(model_id)
        if row is not None:
            return row
        text = str(model_id or "").strip()
        return next((r for r in self.vlm_rows() if r["id"] == text), None)

    def _file_state(self, row: dict) -> dict:
        """给目录/本地行补上磁盘状态：路径、体积、是否就绪、分层来源与状态。"""
        out = dict(row)
        path = _entry_path(row, self.weights_root)
        size = os.path.getsize(path) if path and os.path.isfile(path) else 0
        ready = _is_zip(path) if size else False
        out["path"] = path
        out["exists"] = bool(size)
        out["size_mb"] = round(size / 1048576, 1)
        out["ready"] = bool(ready)
        leftover = 0
        if path:
            for partial in partial_paths(os.path.dirname(path), os.path.basename(path)):
                try:
                    leftover += os.path.getsize(partial)
                except OSError:
                    pass
        out["partial_mb"] = round(leftover / 1048576, 1)
        preset = str(row.get("origin")) == ORIGIN_ULTRALYTICS and not row.get("path")
        if not size:
            out["status"] = STATUS_MISSING
            out["source"] = SOURCE_PRESET if preset else SOURCE_LOCAL
        elif not ready:
            out["status"] = STATUS_INCOMPLETE
            out["source"] = SOURCE_DOWNLOADED if preset else SOURCE_LOCAL
        else:
            out["status"] = STATUS_AVAILABLE
            out["source"] = SOURCE_DOWNLOADED if preset else SOURCE_LOCAL
        if str(row.get("capability")) != CAPABILITY_POSE:
            out["ready"] = bool(size)
        return out

    def compatibility(self, entry: dict, dataset_kps: int = 0, dataset_name: str = "") -> dict:
        """判「这个模型能不能用在当前数据集上」，返回 ``compatible`` 与中文理由。"""
        reasons: List[str] = []
        kps = int((entry or {}).get("num_keypoints") or 0)
        label = str((entry or {}).get("keypoint_set") or "未标注关键点集")
        if dataset_kps and kps and dataset_kps != kps:
            reasons.append(f"当前数据集{dataset_name or ''}是 {dataset_kps} 关键点，该模型输出 {kps} 关键点"
                           f"（{label}），标出来的骨骼没法直接和已有标签合并")
        if str((entry or {}).get("status")) == STATUS_INCOMPLETE:
            reasons.append("权重文件不完整，删掉重下后才能用")
        reasons.extend(list((entry or {}).get("compatibility_reasons") or []))
        return {"compatible": not reasons, "compatibility_reasons": reasons}

    def models(self, dataset_root: str = "", capability: str = "") -> List[dict]:
        """统一清单：内置目录 + 本地注册 + 多模态，每行带状态、备注、最后使用与兼容性。"""
        kps, ds_name = dataset_keypoints(dataset_root) if dataset_root else (0, "")
        config = self.load()
        last_used = config.get("last_used") or {}
        defaults = config.get("defaults") or {}
        metadata = config.get("metadata") or {}
        listed = [self._file_state(row) for row in self.catalog()] + \
                 [self._file_state(row) for row in self.local_rows()]
        known = {normalize_model_id(str(row.get("id") or "")) for row in listed}
        known |= {normalize_model_id(str(row.get("filename") or "")) for row in listed}
        rows: List[dict] = []
        for base in listed + self._disk_rows(known):
            row = dict(base)
            row.setdefault("compatibility_reasons", [])
            row["last_used"] = str(last_used.get(row["id"]) or "")
            row["metadata"] = dict(metadata.get(row["id"]) or {})
            row["default"] = defaults.get(str(row.get("capability"))) == row["id"]
            verdict = self.compatibility(row, dataset_kps=kps, dataset_name=ds_name)
            row.update(verdict)
            if not verdict["compatible"] and row["status"] == STATUS_AVAILABLE \
                    and any("关键点" in r for r in verdict["compatibility_reasons"]):
                row["status"] = STATUS_INCOMPATIBLE
            rows.append(row)
        for row in self.vlm_rows():
            row["default"] = defaults.get(CAPABILITY_VLM) == row["id"]
            row["compatible"] = not row["compatibility_reasons"]
            row["local_path"] = ""
            rows.append(row)
        for row in rows:
            row.setdefault("partial_mb", 0.0)
            row.setdefault("expected_bytes", 0)
            row.setdefault("mtime", None)
            row.setdefault("local_path", row.get("path", ""))
            row["capability_label"] = capability_label(row.get("capability"))
            row["source_label"] = source_label(row.get("source"))
            row["origin_label"] = origin_label(row.get("origin"))
            row["status_label"] = status_label(row.get("status"))
        effective = _resolve_default_ids(rows, defaults)
        for row in rows:
            row["default_explicit"] = str(defaults.get(str(row.get("capability"))) or "") == row["id"]
            row["default"] = bool(effective.get(str(row.get("capability")))) \
                and effective.get(str(row.get("capability"))) == row["id"]
        if capability:
            rows = [row for row in rows if row.get("capability") == capability]
        return rows

    def _disk_rows(self, known: set) -> List[dict]:
        """权重目录里已经有文件、但目录和登记表都不认识的那些。

        症状很具体：用户刚把 ``yolo26x-pose.pt`` 下进 ``models/``，标注页的下拉框里
        还是没有它——因为清单是按「我们认识哪些模型」列的，不是按磁盘上有什么列的。
        文件既然在目录里就必须出现在清单里：任务按文件名推，中文名先用文件名顶着，
        等官方目录补上说明再换。
        """
        rows: List[dict] = []
        for name in weight_names(self.weights_root):
            ident = normalize_model_id(name)
            if not ident or ident in known:
                continue
            task, label = task_hint(name)
            row = _blank_row(ident, name)
            row["capability"] = task or CAPABILITY_POSE
            row["origin"] = ORIGIN_ULTRALYTICS if task else ORIGIN_CUSTOM
            row["downloadable"] = bool(task)
            row["task_label"] = label
            from .modelhub import lookup        # 延迟导入：modelhub 顶部 import 了本模块

            entry = lookup(name) or {}
            # 目录里已经写好中文说明的（含官方完整清单）一律沿用，别让文件名顶替说明
            for key in ("display_name", "description", "recommended_for", "speed",
                        "num_keypoints", "keypoint_set"):
                if entry.get(key) not in (None, "", 0):
                    row[key] = entry[key]
            if not entry:
                row["description"] = (
                    f"权重目录里的{label or '未识别'}权重，官方目录还没写它的中文说明；"
                    f"参数与精度以文件自带元数据为准。" if task
                    else _blank_row(ident, name)["description"])
                row["recommended_for"] = ("本机已有，直接可用" if task
                                          else "自定义训练/二次微调的权重")
            state = self._file_state(row)
            # 文件名候选里混着「目录预置但磁盘上根本没有」的那些，这里必须按文件过滤：
            # 预置档一旦漏进「本机已有」，标注页会出现一个点开就跑不动的下拉项
            if not state.get("exists") and not state.get("partial_mb"):
                continue
            rows.append(state)
        return rows

    def describe(self, model_id: str, dataset_root: str = "") -> dict:
        """单个模型的中文详情，找不到抛 ``KeyError``。"""
        key = normalize_model_id(model_id)
        for row in self.models(dataset_root=dataset_root):
            if key in {normalize_model_id(str(row.get("id") or "")),
                       normalize_model_id(str(row.get("filename") or "")), row.get("id")}:
                if row.get("downloadable") and row.get("filename"):
                    # 单独看一个模型时也要能抄到下载地址，不必再去 --browse 里翻
                    row["url"] = weights_url(str(row["filename"]))
                    row["mirror"] = _mirror_label()
                return row
        raise KeyError(f"未知模型: {model_id}（用 models --browse 看完整目录）")

    def weight_overlay(self, name: str, dataset_kps: int = 0) -> dict:
        """给权重行补分层字段；目录里没有的自定义权重也给一份默认值。"""
        entry = self.get(name) or _blank_row(normalize_model_id(name), str(name))
        row = self._file_state(entry)
        verdict = self.compatibility(row, dataset_kps=dataset_kps)
        row.update(verdict)
        if not verdict["compatible"] and row["status"] == STATUS_AVAILABLE \
                and any("关键点" in r for r in verdict["compatibility_reasons"]):
            row["status"] = STATUS_INCOMPATIBLE
        keep = ("model_id", "display_name", "capability", "capability_label", "origin",
                "origin_label", "status", "status_label", "description",  # 没有 note：备注说明已取消
                "recommended_for", "params_m", "map_pose", "expected_mb", "expected_bytes",
                "num_keypoints", "keypoint_set", "speed", "downloadable", "preset_visible",
                "compatible", "compatibility_reasons")
        row["model_id"] = row["id"]
        row["capability_label"] = capability_label(row.get("capability"))
        row["origin_label"] = origin_label(row.get("origin"))
        row["status_label"] = status_label(row.get("status"))
        config = self.load()
        row["last_used"] = str((config.get("last_used") or {}).get(str(row["id"])) or "")
        out = {k: row[k] for k in keep if k in row}
        out["source_layer"] = row["source"]
        out["source_label"] = source_label(row["source"])
        out["last_used"] = row["last_used"]
        return out

    # --- 用户可写的部分 ---
    def defaults(self) -> dict:
        """每类能力的默认模型；没显式设置时给一个稳妥的兜底。"""
        config = self.load()
        defaults = _resolve_default_ids(self.models(), dict(config.get("defaults") or {}))
        labelled = {capability: {"id": defaults.get(capability, ""),
                                "label": capability_label(capability)} for capability in CAPABILITIES}
        return {"ids": defaults, "labels": labelled}

    def default_for(self, capability: str = CAPABILITY_POSE) -> str:
        return str(self.defaults()["ids"].get(capability) or "")

    def set_default(self, model_id: str, capability: str = "") -> dict:
        entry = self.get_any(model_id)
        if entry is None:
            raise KeyError(f"未知模型: {model_id}（用 models --browse 看完整目录）")
        cap = str(capability or entry.get("capability") or "")
        if cap not in CAPABILITIES:
            raise ValueError(f"能力只能是 {'/'.join(CAPABILITIES)}，收到 {cap}")
        if cap != entry.get("capability"):
            raise ValueError(f"{entry.get('display_name')} 属于"
                             f"{capability_label(entry.get('capability'))}，不能当"
                             f"{capability_label(cap)}的默认模型")
        with _LOCK:
            config = self.load()
            defaults = dict(config.get("defaults") or {})
            defaults[cap] = str(entry["id"])
            config["defaults"] = defaults
            self.save(config)
        return self.defaults()

    def register_local(self, path: str, capability: str = CAPABILITY_POSE, display_name: str = "",
                       origin: str = ORIGIN_CUSTOM, model_id: str = "",
                       num_keypoints: int = 0) -> dict:
        """登记一个本地权重/模型：只记路径和说明，文件本身不动。"""
        raw = os.path.abspath(os.path.expanduser(str(path or "").strip()))
        if not raw:
            raise ValueError("请填写模型文件路径")
        if not os.path.isfile(raw):
            raise FileNotFoundError(f"文件不存在：{raw}")
        if capability not in CAPABILITIES:
            raise ValueError(f"能力只能是 {'/'.join(CAPABILITIES)}，收到 {capability}")
        if origin not in ORIGIN_LABELS:
            raise ValueError(f"出处只能是 {'/'.join(sorted(ORIGIN_LABELS))}，收到 {origin}")
        base = os.path.basename(raw)
        if capability == CAPABILITY_POSE and not _is_zip(raw):
            raise ValueError(f"{base} 看起来不是完整的权重包（YOLO 权重是 zip 结构），"
                             f"可能还在下载或已经损坏")
        existing_in_catalog = self.get(base)
        if existing_in_catalog is not None and str(existing_in_catalog.get("path", "")) in ("", raw):
            # 本来就在目录里（或已经放进权重目录）就别再登记一遍，多一条只会多一处对不上
            return {"id": str(existing_in_catalog["id"]), "already_in_catalog": True,
                    "registered": False}
        ident = normalize_model_id(model_id) or os.path.splitext(base)[0]
        clash = self.get(ident)
        if clash is not None and str(clash.get("path", "")) != raw:
            raise ValueError(f"模型 ID「{ident}」已被 {clash.get('filename') or clash.get('path')} 占用，"
                             f"请用 --id 换个名字")
        row = {"id": ident, "filename": base, "path": raw, "capability": capability,
               "origin": origin, "display_name": (display_name or os.path.splitext(base)[0])[:40],
               "added_at": now_str()}
        if num_keypoints:
            row["num_keypoints"] = int(num_keypoints)
        with _LOCK:
            config = self.load()
            rows = [item for item in (config.get("local") or []) if str(item.get("id")) != ident]
            rows.append(row)
            config["local"] = rows
            self.save(config)
        return {"id": ident, "registered": True, "already_in_catalog": False, "path": raw}

    def forget(self, model_id: str) -> dict:
        """取消登记本地模型：只删登记表里的条目，文件留在原地不动。"""
        ident = normalize_model_id(model_id)
        with _LOCK:
            config = self.load()
            rows = [item for item in (config.get("local") or []) if str(item.get("id")) != ident]
            if len(rows) == len(config.get("local") or []):
                raise KeyError(f"「{ident}」不是登记过的本地模型；内置目录项想不显示，"
                               f"就在 {self.config_path} 的 catalog.{ident}.hidden 里隐藏")
            config["local"] = rows
            defaults = dict(config.get("defaults") or {})
            for key, value in list(defaults.items()):
                if str(value) == ident:
                    defaults.pop(key, None)
            config["defaults"] = defaults
            self.save(config)
        return {"id": ident, "forgotten": True}

    # --- 运行时回报 ---
    def record_use(self, model_id: str) -> None:
        """记一笔「什么时候真用过这个模型」；失败不打断标注。"""
        ident = normalize_model_id(model_id)
        if not ident:
            return
        try:
            with _LOCK:
                config = self.load()
                used = dict(config.get("last_used") or {})
                used[ident] = now_str()
                config["last_used"] = used
                self.save(config)
        except Exception:  # noqa: BLE001 记录失败不该影响推理
            pass

    def record_metadata(self, model_id: str, meta: dict) -> None:
        """记下加载时读到的真实元数据（关键点数、设备、版本），用于解释兼容性问题。"""
        ident = normalize_model_id(model_id)
        if not ident or not isinstance(meta, dict):
            return
        allowed = {k: v for k, v in meta.items()
                   if k in ("num_keypoints", "keypoint_set", "device", "task", "ultralytics",
                            "imgsz", "params_m") and v not in (None, "")}
        allowed["recorded_at"] = now_str()
        try:
            with _LOCK:
                config = self.load()
                metadata = dict(config.get("metadata") or {})
                merged = dict(metadata.get(ident) or {})
                merged.update(allowed)
                metadata[ident] = merged
                config["metadata"] = metadata
                self.save(config)
        except Exception:  # noqa: BLE001
            pass

    def record_download(self, record: dict) -> dict:
        """追加一条下载记录：URL、SHA256、字节数、临时文件名、耗时、成败。"""
        row = {k: v for k, v in dict(record or {}).items() if v is not None}
        row.setdefault("started_at", now_str())
        row.setdefault("finished_at", now_str())
        with _LOCK:
            config = self.load()
            rows = list(config.get("downloads") or [])
            rows.append(row)
            config["downloads"] = rows[-MAX_DOWNLOAD_RECORDS:]
            if row.get("ok"):
                used = dict(config.get("last_used") or {})
                ident = normalize_model_id(str(row.get("model_id") or row.get("name") or ""))
                if ident:
                    used[ident] = now_str()
                    config["last_used"] = used
            self.save(config)
        return row

    def download_records(self, model_id: str = "", limit: int = 20) -> List[dict]:
        rows = list(self.load().get("downloads") or [])
        if model_id:
            key = normalize_model_id(model_id)
            rows = [row for row in rows
                    if key in {normalize_model_id(str(row.get("model_id") or "")),
                               normalize_model_id(str(row.get("name") or ""))}]
        rows = rows[-max(1, int(limit)):]
        for row in rows:
            url = str(row.get("url") or "")
            row["url_masked"] = (url.split("?")[0] if len(url) > 160 else url)
        return rows[::-1]

    def payload(self, dataset_root: str = "") -> dict:
        """一次给全「模型管理」面板：清单 + 可下载目录 + 默认值 + 下载记录 + 路径。"""
        return {
            "models": self.models(dataset_root=dataset_root),
            "catalog": self.catalog_downloadable(),
            "defaults": self.defaults(),
            "download_records": self.download_records(),
            "model_config_path": self.config_path,
            "weights_dir": self.weights_root,
            "secrets_path": secret_store.secrets_path(),
            "capability_labels": dict(CAPABILITY_LABELS),
            "status_labels": dict(STATUS_LABELS),
            "source_labels": dict(SOURCE_LABELS),
            "origin_labels": dict(ORIGIN_LABELS),
        }


_REGISTRY = ModelRegistry()


def registry() -> ModelRegistry:
    """进程内共享的注册表实例（路径每次调用时重新解析）。"""
    return _REGISTRY


def _registry() -> ModelRegistry:
    return _REGISTRY


def _catalog_weight_names() -> List[str]:
    """目录里的权重文件名，顺序即展示顺序（推荐的 n/s/m 在最前）。"""
    names: List[str] = []
    for row in _registry().catalog():
        if row.get("capability") != CAPABILITY_POSE:
            continue
        filename = str(row.get("filename") or "")
        if filename and filename not in names:
            names.append(filename)
    return names


def list_models(dataset_root: str = "", capability: str = "") -> List[dict]:
    """统一模型清单，界面与命令行共用。"""
    return _registry().models(dataset_root=dataset_root, capability=capability)


def model_catalog() -> List[dict]:
    """可下载模型目录（含中文说明与下载地址）。"""
    return _registry().catalog_downloadable()


def describe_model(model_id: str, dataset_root: str = "") -> dict:
    return _registry().describe(model_id, dataset_root=dataset_root)


def set_model_default(model_id: str, capability: str = "") -> dict:
    return _registry().set_default(model_id, capability)


def default_model(capability: str = CAPABILITY_POSE) -> str:
    return _registry().default_for(capability)


def register_local_model(**kwargs) -> dict:
    return _registry().register_local(**kwargs)


def forget_model(model_id: str) -> dict:
    return _registry().forget(model_id)


def record_model_use(model_id: str) -> None:
    _registry().record_use(model_id)


def record_model_load(model_id: str, meta: Optional[dict] = None) -> None:
    """加载/使用成功后回报：记一笔最后使用时间，顺带记下真实元数据。"""
    registry = _registry()
    registry.record_use(model_id)
    if meta:
        registry.record_metadata(model_id, meta)


def download_records(model_id: str = "", limit: int = 20) -> List[dict]:
    return _registry().download_records(model_id, limit)
