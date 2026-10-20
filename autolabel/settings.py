"""系统设置：推理用哪块算力、要不要开半精度，存在这台机器的一份文件里。

默认落在 ``~/.autolabel/settings.json``（可用环境变量 ``AUTOLABEL_SETTINGS`` 覆盖）。
放在家目录而不是数据集里，是因为这里记的是**这台机器的算力**：换显卡、换芯片跟人走，
跟某一份数据没关系；跟着数据集走还会造成"同一份设置，四个数据集四个值"。

优先级要说清楚，系统设置是**兜底默认**而不是最高优先级：
命令行 ``--device`` / ``--config`` 写明的值最大，界面上单个任务填的参数其次，
两处都没写才用这里的。否则会出现"命令行明明指定了 cuda，却被家目录里一个文件改了"。

设备选了但这台机器没有时不报错、也不静默换：跑之前打一行降级说明，
用能用的那档继续跑。标注不会因为换设备失败，只是快慢不同。
"""

from __future__ import annotations

import json
import os
import platform
from typing import Dict, Optional, Tuple

DEFAULT_SETTINGS = os.path.join("~", ".autolabel", "settings.json")

#: 界面上的顺序就是这个顺序；"自动选择" 排第一，因为绝大多数人不该去动它。
DEVICE_CHOICES: Tuple[Dict[str, str], ...] = (
    {"value": "auto", "label": "自动选择", "hint": "有 N 卡用 CUDA，其次 Apple 加速，最后 CPU"},
    {"value": "cuda", "label": "NVIDIA 显卡（CUDA）", "hint": "只在装了 NVIDIA 显卡的机器上可用"},
    {"value": "mps", "label": "Apple 加速（MPS）", "hint": "Apple Silicon（M 系列）的 GPU 加速"},
    {"value": "cpu", "label": "纯 CPU", "hint": "最稳，也最慢；排查显卡问题时用"},
)
DEVICE_VALUES = tuple(item["value"] for item in DEVICE_CHOICES)

#: 命令行 ``--device`` 认的全部写法：四个大档 + 双卡机器上的卡号写法。
#: 界面面板只展示前四个（卡号在面板里没意义），但启动参数必须能写卡号——
#: 清单和文档都教人写 ``cuda:1``，argparse 不认的话症状是后台任务一起来就
#: 退出码 2，然后被服务管理器每 10 秒拉起一次刷日志。
DEVICE_ARGS = DEVICE_VALUES + ("0", "1", "cuda:0", "cuda:1")
DEVICE_LABELS = {item["value"]: item["label"] for item in DEVICE_CHOICES}

DEFAULTS = {"device": "auto", "half": False, "imgsz": 640, "conf": 0.25,
            "export_mode": "hardlink"}

#: 能被改写的设置项：设置面板、命令行、复核服务三处共用这一份清单。
#: 新增一档只改这里——以前是各处各写一份，加 export_mode 时漏了服务那条白名单，
#: 症状是「界面上选了直接拷贝、点保存，回来一看还是硬链接」。
SETTINGS_KEYS = ("device", "half", "imgsz", "conf", "export_mode")

#: 导出训练集时把图片「放」进导出目录的三种方式，界面里的顺序就是这个顺序。
#: 硬链接排第一是踩过坑的：软链接在原地能训，但把它压成 zip 或拷去别的机器，
#: 每张图片就剩下几十个字节的链接记录，症状是「训练集里全是快捷方式，没法训」。
#: 硬链接和真文件没有区别，同一个盘上还不额外占磁盘。
EXPORT_MODE_CHOICES: Tuple[Dict[str, str], ...] = (
    {"value": "hardlink", "label": "硬链接（推荐）",
     "hint": "跟真文件一样，能直接压缩、拷贝、上传到别的机器训练；同一个磁盘不额外占空间"},
    {"value": "copy", "label": "直接拷贝",
     "hint": "最稳，但多一份磁盘占用；导出到 U 盘或另一个分区时用这个"},
    {"value": "symlink", "label": "软链接（最省）",
     "hint": "最省磁盘，但只能在原位置训练：一压缩或一拷走就变成几十字节的空壳"},
)
EXPORT_MODE_VALUES = tuple(item["value"] for item in EXPORT_MODE_CHOICES)
EXPORT_MODE_LABELS = {item["value"]: item["label"] for item in EXPORT_MODE_CHOICES}

#: 输入分辨率档位。界面上只摆这三档（命令行允许 320–1920 之间任意整数）。
#: 这一档对能不能出框的影响比想象中大：本机实测同一张白 T 恤棚拍照，
#: 开放词表的 ``shirt`` 在 640 上只有 0.026 分，提到 1024 变成 0.553——
#: 同一个模型、同一张图，一边是"一个框都没有"，一边是稳稳过默认阈值。
IMGSZ_CHOICES: Tuple[Dict[str, object], ...] = (
    {"value": 640, "label": "640（标准）", "hint": "最快，绝大多数场景够用"},
    {"value": 1024, "label": "1024（更细）", "hint": "服装、小物件、开放词表更容易出框，约慢 2–3 倍"},
    {"value": 1280, "label": "1280（最细）", "hint": "只为难标的类别开，耗时和显存都明显上涨"},
)
IMGSZ_VALUES = tuple(item["value"] for item in IMGSZ_CHOICES)
IMGSZ_MIN, IMGSZ_MAX = 320, 1920

#: 置信度上界收在 0.95：填 1.0 等于"只收满分"，实际会把所有框都丢光
CONF_MIN, CONF_MAX = 0.01, 0.95

#: 开关类字段的合法写法。``bool("false")`` 在 Python 里是 ``True``，
#: 所以字符串绝不能直接 ``bool()``——见 :func:`as_bool`。
TRUE_WORDS = ("1", "true", "yes", "on", "y", "开", "是")
FALSE_WORDS = ("0", "false", "no", "off", "n", "关", "否")


def as_bool(value: object, *, default: bool = False, strict: bool = False) -> bool:
    """把开关读成真正的布尔值，字符串按字面意思判。

    实测踩过：界面上把半精度**关掉**，请求里带的是字符串 ``"false"``，
    而 ``bool("false") == True``，存下来半精度是**开着**的——而且界面还显示"已保存"。
    读文件时宽松（坏值退回默认，不让一个手写坏的文件把服务弄崩），
    存用户改动时严格（认不出就报错，别让一个填错的开关悄悄改了行为）。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in TRUE_WORDS:
            return True
        if token in FALSE_WORDS:
            return False
    if value is None:
        return default
    if strict:
        raise ValueError(f'半精度只能选「开」或「关」，现在是「{value}」')
    return default


def as_imgsz(value: object, *, strict: bool = False) -> int:
    """把输入分辨率读成一个能用的整数：读不懂就退回默认值。

    档位不必死守界面那三个——命令行 ``--imgsz 768`` 是合理需求，所以只卡
    区间不卡枚举。但 320 以下模型基本啥都认不出，1920 以上显存容易炸，
    超出区间就当填错了。
    """
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        number = 0
    if not IMGSZ_MIN <= number <= IMGSZ_MAX:
        if strict:
            raise ValueError(f"输入分辨率只能在 {IMGSZ_MIN}–{IMGSZ_MAX} 之间，现在是「{value}」")
        return int(DEFAULTS["imgsz"])
    return number


def as_conf(value: object, *, strict: bool = False) -> float:
    """把置信度阈值读成 (0, 1) 之间的小数，读不懂就退回默认值。"""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        number = -1.0
    if not CONF_MIN <= number <= CONF_MAX:
        if strict:
            raise ValueError(f"置信度阈值只能在 {CONF_MIN}–{CONF_MAX} 之间，现在是「{value}」"
                             "（填太高的症状是所有框都被丢掉，界面上看着像模型没认出东西）")
        return float(DEFAULTS["conf"])
    return round(number, 4)

def as_export_mode(value: object, *, strict: bool = False) -> str:
    """把「导出时图片怎么放」读成三种之一，读不懂就退回默认值。

    这一档决定产物拿去别的机器还能不能训练，所以认不出时**不能**猜：
    界面上选了「直接拷贝」、存下来变成软链接，用户要到压缩打包才发现图片是空壳。
    """
    mode = str(value if value is not None else "").strip().lower()
    if mode in EXPORT_MODE_VALUES:
        return mode
    if strict:
        raise ValueError(f"导出的图片放法只能选 {' / '.join(EXPORT_MODE_VALUES)}，现在是「{value}」")
    return str(DEFAULTS["export_mode"])


#: 启动参数 / 环境变量覆盖：想"一个实例跑一块卡"就在起服务时带上 ``review --device cuda``，
#: 它只影响这一个进程，不改家目录里那份设置，两个实例各用各的卡互不干扰。
ENV_DEVICE = "AUTOLABEL_DEVICE"
ENV_HALF = "AUTOLABEL_HALF"
ENV_IMGSZ = "AUTOLABEL_IMGSZ"
ENV_CONF = "AUTOLABEL_CONF"
ENV_EXPORT_MODE = "AUTOLABEL_EXPORT_MODE"

#: 同一次服务里只探测一次：import torch 不算便宜，每次开菜单都探测会白等。
_probe_cache: Optional[dict] = None


def path() -> str:
    """系统设置文件路径：默认 ``~/.autolabel/settings.json``。"""
    return os.path.abspath(os.path.expanduser(os.environ.get("AUTOLABEL_SETTINGS") or DEFAULT_SETTINGS))


def normalize(raw: Optional[dict]) -> dict:
    """补齐缺项并挡住不认识的值，返回干净的一份设置。"""
    raw = raw if isinstance(raw, dict) else {}
    device = str(raw.get("device") or DEFAULTS["device"]).strip().lower()
    if device in ("0", "1", "cuda:0", "cuda:1"):     # 命令行能写的多卡写法，界面上归到 CUDA
        device = "cuda"
    if device not in DEVICE_VALUES:
        device = DEFAULTS["device"]
    half = as_bool(raw.get("half", DEFAULTS["half"]), default=DEFAULTS["half"])
    imgsz = as_imgsz(raw.get("imgsz", DEFAULTS["imgsz"]))
    conf = as_conf(raw.get("conf", DEFAULTS["conf"]))
    export_mode = as_export_mode(raw.get("export_mode", DEFAULTS["export_mode"]))
    return {"device": device, "half": half, "imgsz": imgsz, "conf": conf,
            "export_mode": export_mode}


def env_override() -> dict:
    """本进程的环境变量覆盖（``AUTOLABEL_DEVICE`` / ``AUTOLABEL_HALF``），没设就是空。"""
    out: Dict[str, object] = {}
    raw_device = os.environ.get(ENV_DEVICE, "").strip().lower()
    if raw_device:
        # 卡号写法（1 / cuda:1）要原样留着：归一成 "cuda" 之后"用哪块卡"就丢了，
        # 双卡机器上「一卡一实例」的两个实例会全挤到 0 号卡，另一块空转还没提示
        out["device"] = raw_device if raw_device in DEVICE_ARGS else \
            normalize({"device": raw_device})["device"]
    raw_half = os.environ.get(ENV_HALF, "").strip()
    if raw_half:
        out["half"] = as_bool(raw_half, default=DEFAULTS["half"])
    raw_imgsz = os.environ.get(ENV_IMGSZ, "").strip()
    if raw_imgsz:
        out["imgsz"] = as_imgsz(raw_imgsz)
    raw_conf = os.environ.get(ENV_CONF, "").strip()
    if raw_conf:
        out["conf"] = as_conf(raw_conf)
    raw_export_mode = os.environ.get(ENV_EXPORT_MODE, "").strip().lower()
    if raw_export_mode:
        out["export_mode"] = as_export_mode(raw_export_mode)
    return out


def load(*, ignore_env: bool = False) -> dict:
    """读设置；文件不存在或写坏了都回默认值，不抛错。

    默认把环境变量（启动参数）叠上去，那是本进程真正该用的值；``ignore_env=True``
    只拿文件里那份——存盘时必须用它，否则一次保存就把用户真正的选择
    换成了启动参数里的临时值。
    """
    merged = dict(DEFAULTS)
    try:
        with open(path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        merged = normalize(merged)
    else:
        if isinstance(data, dict):
            merged["device"] = data.get("device", merged["device"])
            merged["half"] = data.get("half", merged["half"])
            merged["imgsz"] = data.get("imgsz", merged["imgsz"])
            merged["conf"] = data.get("conf", merged["conf"])
            merged["export_mode"] = data.get("export_mode", merged["export_mode"])
        merged = normalize(merged)
    if not ignore_env:
        merged.update(env_override())
    return merged


def save(patch: Optional[dict] = None) -> dict:
    """改几项存回去（只认 device / half / imgsz / conf / export_mode），返回完整设置。"""
    wanted = {k: v for k, v in (patch or {}).items() if v is not None}
    if "device" in wanted:
        asked = str(wanted["device"]).strip().lower()
        if asked not in DEVICE_ARGS:
            raise ValueError(f"不认识的设备「{wanted['device']}」，可选：{' / '.join(DEVICE_VALUES)}")
    # 半精度也要挡住：填进来一个看不懂的词，静默按"开"存下去比报错难查得多
    if "half" in wanted:
        wanted["half"] = as_bool(wanted["half"], strict=True)
    if "imgsz" in wanted:
        wanted["imgsz"] = as_imgsz(wanted["imgsz"], strict=True)
    if "conf" in wanted:
        wanted["conf"] = as_conf(wanted["conf"], strict=True)
    if "export_mode" in wanted:
        wanted["export_mode"] = as_export_mode(wanted["export_mode"], strict=True)
    next_settings = normalize({**load(ignore_env=True), **wanted})
    target = path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(next_settings, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return next_settings


def detect(force: bool = False) -> dict:
    """探测这台机器实际能用哪些算力，顺带把显卡型号取出来给人看。

    认不出来的都算"不可用"并写清原因——界面上宁可显示"探测不到"，
    也不该把一个按不下去的按钮摆成能按的样子。
    """
    global _probe_cache
    if _probe_cache is not None and not force:
        return _probe_cache
    devices: Dict[str, dict] = {
        item["value"]: {"value": item["value"], "label": item["label"], "hint": item["hint"],
                        "available": False, "names": [], "reason": ""}
        for item in DEVICE_CHOICES
    }
    devices["cpu"]["available"] = True
    devices["cpu"]["names"] = [_cpu_label()]
    devices["auto"]["available"] = True
    try:
        import torch
    except Exception as exc:  # noqa: BLE001  没装、装坏、架构不对都算
        devices["mps"]["reason"] = f"探测不到 PyTorch（{type(exc).__name__}）"
        devices["cuda"]["reason"] = devices["mps"]["reason"]
        _probe_cache = _with_auto_note(devices)
        return _probe_cache

    try:
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count() or 0)
            devices["cuda"]["available"] = count > 0
            devices["cuda"]["names"] = [_safe_device_name(torch, i) for i in range(count)] or ["CUDA 显卡"]
            if not devices["cuda"]["available"]:
                devices["cuda"]["reason"] = "CUDA 报告可用但数不到卡，试试重装驱动"
        else:
            devices["cuda"]["reason"] = _cuda_miss_reason(torch)
    except Exception as exc:  # noqa: BLE001
        devices["cuda"]["reason"] = f"显卡探测失败（{type(exc).__name__}）"

    try:
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            devices["mps"]["available"] = True
            devices["mps"]["names"] = [_mac_label()]
        else:
            devices["mps"]["reason"] = _mps_miss_reason(torch)
    except Exception as exc:  # noqa: BLE001
        devices["mps"]["reason"] = f"Apple 加速探测失败（{type(exc).__name__}）"

    _probe_cache = _with_auto_note(devices)
    return _probe_cache


def _with_auto_note(devices: dict) -> dict:
    """自动档要说明"它会挑谁"，否则界面上选中它时人不知道自己实际在用哪块卡。"""
    picked = next((devices[v]["label"] for v in ("cuda", "mps") if devices[v]["available"]),
                  devices["cpu"]["label"])
    names = [next(iter(devices[v]["names"]), "") for v in ("cuda", "mps") if devices[v]["available"]]
    detail = "、".join(n for n in names if n)
    devices["auto"]["names"] = [f"本次会用 {picked}" + (f"（{detail}）" if detail else "")]
    return devices


def _safe_device_name(torch_module, index: int) -> str:
    try:
        return str(torch_module.cuda.get_device_name(index))
    except Exception:  # noqa: BLE001  取名字失败不该整个探测失败
        return "CUDA 显卡"


def _cpu_label() -> str:
    chip = platform.machine() or "CPU"
    count = os.cpu_count() or 0
    return f"{chip} · {count} 核" if count else chip


def _mac_label() -> str:
    brand = " ".join(str(part) for part in platform.mac_ver()[0:1] if part)
    return f"Apple Silicon（{platform.machine()}）" if platform.system() == "Darwin" else (brand or "MPS")


def _cuda_miss_reason(torch_module) -> str:
    """CUDA 不可用时说人话：最常见的两种原因分开讲。"""
    if getattr(torch_module.version, "cuda", None) in (None, ""):
        return "这个 PyTorch 不带 CUDA（装的是 CPU 版或 macOS 版），换显卡也还是 CPU"
    return "没检测到可用的 NVIDIA 显卡，检查驱动或 nvidia-smi"


def _mps_miss_reason(torch_module) -> str:
    version = str(getattr(torch_module, "__version__", "") or "")
    if platform.system() != "Darwin":
        return "MPS 只在 macOS 上有"
    # 带本地后缀的构建号（2.4.1+cpu、2.6.0a0+git1234）比版本号常见，
    # 不去掉那截会被当成"解析不出来"，于是把一句好端端的 MPS 说成"版本太旧"
    core = version.split("+", 1)[0]
    if version and core.count(".") and _atleast(core, "1.12"):
        return "这台 Mac 不支持 MPS（Intel 机型或系统太旧），用 CPU"
    return "PyTorch 版本太旧，没有 MPS 支持（需要 1.12 以上）"


def _atleast(version: str, minimum: str) -> bool:
    from .version import compare

    return compare(version, minimum) >= 0


def resolve(requested: str = "") -> Tuple[str, str]:
    """把设置里的设备落到这台机器实际能用的那一档，返回 ``(设备, 降级说明)``。

    没填就用系统设置里的。有卡就用选的，没卡退到能用的档并带回一句话，
    让命令行和界面都能把"我其实退到 CPU 了"讲出来，而不是闷声变慢。
    """
    wanted = normalize({"device": requested or load()["device"]})["device"]
    if wanted == "cpu":
        return "cpu", ""
    facts = detect()
    if wanted == "auto":
        for value in ("cuda", "mps"):
            if facts[value]["available"]:
                return value, ""
        return "cpu", ""
    if facts[wanted]["available"]:
        return wanted, ""
    fallback = ""
    for value in ("cuda", "mps", "cpu"):
        if facts[value]["available"]:
            fallback = value
            break
    reason = facts[wanted]["reason"] or "这台机器上没有"
    return fallback or "cpu", (f"{DEVICE_LABELS.get(wanted, wanted)}不可用（{reason}），"
                               f"本次改用 {DEVICE_LABELS.get(fallback, 'CPU')}")


def label_defaults() -> dict:
    """给标注任务用的兜底参数：界面任务与命令行在没显式指定时都读这里。

    分辨率和置信度放这儿而不是各模块自己写死，是因为它们**经常要一起调**：
    开放词表标服装类时"降阈值 + 提分辨率"才出得来框，两处各设一处就会
    出现"界面上改了、命令行还在用旧值"。
    """
    saved = load()
    return {"device": saved["device"], "half": saved["half"],
            "imgsz": saved["imgsz"], "conf": saved["conf"]}


def payload(probe: bool = False) -> dict:
    """给界面「系统设置」和 ``autolabel settings`` 共用的一份数据。"""
    saved = load(ignore_env=True)      # 面板里选中的是文件里那份
    running = load()                   # 本实例真正会用的那份（可能来自启动参数）
    devices = detect() if probe else {}
    effective, note = (resolve(running["device"]) if probe else (running["device"], ""))
    override = env_override()
    # 开放词表那套（CLIP 文本编码器）只在打开面板时查一次：import clip 会连带
    # 拉起 torch，平时切页不值得为它多等这一秒
    from . import backends as inference   # 延迟导入：这个文件平时只读一份 JSON

    open_vocab = inference.text_encoder_status() if probe else {}
    from . import video as video_tools   # 只查有没有解码器，不 import cv2，平时代价几乎为零

    return {
        "open_vocab": open_vocab,
        "video": video_tools.decoder(),
        "saved": saved,
        "env": {k: DEVICE_LABELS.get(v, v) if k == "device" else v for k, v in override.items()},
        "devices": devices,
        "choices": [dict(item) for item in DEVICE_CHOICES],
        "imgsz_choices": [dict(item) for item in IMGSZ_CHOICES],
        "export_mode_choices": [dict(item) for item in EXPORT_MODE_CHOICES],
        "effective": {"device": effective, "half": running["half"], "note": note,
                      "imgsz": running["imgsz"], "conf": running["conf"],
                      "export_mode": running["export_mode"],
                      "from_env": bool(override)},
        "path": path(),
    }


def describe(raw: Optional[dict] = None) -> str:
    """``autolabel settings`` 打印的那几行。"""
    data = payload(probe=True)
    saved, devices = data["saved"], data["devices"]
    lines = [f"系统设置文件 {data['path']}",
             f"推理设备 {DEVICE_LABELS[saved['device']]}（{saved['device']}）",
             f"半精度 {'开' if saved['half'] else '关'}",
             f"输入分辨率 {saved['imgsz']} · 默认置信度 {saved['conf']}",
             f"导出训练集 图片用{EXPORT_MODE_LABELS[saved['export_mode']]}（{saved['export_mode']}）"]
    usable = [devices[v]["label"] for v in DEVICE_VALUES if v != "auto" and devices[v]["available"]]
    lines.append(f"这台机器可用 {' / '.join(usable) or '只有 CPU'}")
    if saved["device"] == "auto" and devices["auto"]["names"]:
        lines.append(f"  · {devices['auto']['names'][0]}")
    for value in ("cuda", "mps"):
        facts = devices[value]
        if not facts["available"] and facts["reason"]:
            lines.append(f"  · {facts['label']}：{facts['reason']}")
    names = [n for n in (devices.get(saved["device"], {}).get("names") or []) if n]
    if names and saved["device"] != "auto":
        lines.append(f"  · 识别到 {'、'.join(names)}")
    if data["effective"]["note"]:
        lines.append(f"[降级] {data['effective']['note']}")
    open_vocab = data.get("open_vocab") or {}
    if open_vocab:
        cached = open_vocab.get("cached_bytes") or 0
        state = "已就绪" if open_vocab.get("installed") else "缺依赖"
        lines.append(f"开放词表权重（YOLOE / YOLO-World）{state}"
                     + (f" · 文本编码器权重已缓存 {round(cached / 1048576)} MB" if cached else ""))
        if not open_vocab.get("installed"):
            lines.append(f"  · {open_vocab.get('note') or ''}")
    video = data.get("video") or {}
    lines.append(f"视频抽帧 {video.get('label') or '未知'}"
                 + (f"：{video.get('note')}" if video.get("note") else ""))
    if data["env"]:
        got = "、".join(
            (DEVICE_LABELS.get(v, v) if k == "device" else ("开" if v else "关"))
            for k, v in sorted(data["env"].items()))
        lines.append(f"  · 本进程被启动参数覆盖为 {got}（只影响这一个实例，不改上面的文件值）")
    lines.append("命令行 --device / --config 写明的值优先于这里")
    return "\n".join(lines)


__all__ = ["DEFAULTS", "DEVICE_ARGS", "DEVICE_CHOICES", "DEVICE_LABELS",
           "DEVICE_VALUES", "ENV_DEVICE", "ENV_EXPORT_MODE", "ENV_HALF",
           "EXPORT_MODE_CHOICES", "EXPORT_MODE_LABELS", "EXPORT_MODE_VALUES",
           "SETTINGS_KEYS",
           "as_export_mode", "describe", "detect", "env_override", "label_defaults", "load",
           "normalize", "path", "payload", "resolve", "save"]
