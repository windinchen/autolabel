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
DEVICE_LABELS = {item["value"]: item["label"] for item in DEVICE_CHOICES}

DEFAULTS = {"device": "auto", "half": False}

#: 启动参数 / 环境变量覆盖：想"一个实例跑一块卡"就在起服务时带上 ``review --device cuda``，
#: 它只影响这一个进程，不改家目录里那份设置，两个实例各用各的卡互不干扰。
ENV_DEVICE = "AUTOLABEL_DEVICE"
ENV_HALF = "AUTOLABEL_HALF"

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
    half = bool(raw.get("half", DEFAULTS["half"]))
    return {"device": device, "half": half}


def env_override() -> dict:
    """本进程的环境变量覆盖（``AUTOLABEL_DEVICE`` / ``AUTOLABEL_HALF``），没设就是空。"""
    out: Dict[str, object] = {}
    raw_device = os.environ.get(ENV_DEVICE, "").strip()
    if raw_device:
        out["device"] = normalize({"device": raw_device})["device"]
    raw_half = os.environ.get(ENV_HALF, "").strip()
    if raw_half in ("0", "1", "true", "false", "yes", "no"):
        out["half"] = raw_half.lower() in ("1", "true", "yes")
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
        merged = normalize(merged)
    if not ignore_env:
        merged.update(env_override())
    return merged


def save(patch: Optional[dict] = None) -> dict:
    """改几项存回去（只认 device / half），返回存好之后的完整设置。"""
    next_settings = normalize({**load(ignore_env=True), **(patch or {})})
    wanted = {k: v for k, v in (patch or {}).items() if v is not None}
    if "device" in wanted:
        asked = str(wanted["device"]).strip().lower()
        if asked not in DEVICE_VALUES and asked not in ("0", "1", "cuda:0", "cuda:1"):
            raise ValueError(f"不认识的设备「{wanted['device']}」，可选：{' / '.join(DEVICE_VALUES)}")
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
    if version and version.split("+")[0].count(".") and _atleast(version, "1.12"):
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
    """给标注任务用的兜底参数：界面任务与命令行在没显式指定时都读这里。"""
    saved = load()
    return {"device": saved["device"], "half": saved["half"]}


def payload(probe: bool = False) -> dict:
    """给界面「系统设置」和 ``autolabel settings`` 共用的一份数据。"""
    saved = load(ignore_env=True)      # 面板里选中的是文件里那份
    running = load()                   # 本实例真正会用的那份（可能来自启动参数）
    devices = detect() if probe else {}
    effective, note = (resolve(running["device"]) if probe else (running["device"], ""))
    override = env_override()
    return {
        "saved": saved,
        "env": {k: DEVICE_LABELS.get(v, v) if k == "device" else v for k, v in override.items()},
        "devices": devices,
        "choices": [dict(item) for item in DEVICE_CHOICES],
        "effective": {"device": effective, "half": running["half"], "note": note,
                      "from_env": bool(override)},
        "path": path(),
    }


def describe(raw: Optional[dict] = None) -> str:
    """``autolabel settings`` 打印的那几行。"""
    data = payload(probe=True)
    saved, devices = data["saved"], data["devices"]
    lines = [f"系统设置文件 {data['path']}",
             f"推理设备 {DEVICE_LABELS[saved['device']]}（{saved['device']}）",
             f"半精度 {'开' if saved['half'] else '关'}"]
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
    if data["env"]:
        got = "、".join(
            (DEVICE_LABELS.get(v, v) if k == "device" else ("开" if v else "关"))
            for k, v in sorted(data["env"].items()))
        lines.append(f"  · 本进程被启动参数覆盖为 {got}（只影响这一个实例，不改上面的文件值）")
    lines.append("命令行 --device / --config 写明的值优先于这里")
    return "\n".join(lines)


__all__ = ["DEFAULTS", "DEVICE_CHOICES", "DEVICE_LABELS", "DEVICE_VALUES", "ENV_DEVICE",
           "ENV_HALF", "describe", "detect", "env_override", "label_defaults", "load",
           "normalize", "path", "payload", "resolve", "save"]
