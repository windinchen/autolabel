"""版本管理：版本号只在这一处定义，界面上常驻显示，并和 GitHub 上的发布版比对提示更新。

提升规则（别再靠记忆）：日常改动一律 **+0.01**（``1.2.1 → 1.2.2``），
只有特意安排的大版本才动前两位。改法 ``autolabel version --bump``，
它只改本文件里的 ``VERSION``，其余地方（``--version``、界面、接口）全部跟着走。

联网检查遵守三条：不挡启动（后台线程 + 半天缓存）、不编造（GitHub 更旧就照说更旧，
不弹"有更新"）、不联网也不报错（提示"未检查"而不是崩）。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from datetime import datetime
from typing import Callable, Optional, Tuple

DEVELOPER = "windinchen"
REPO = "windinchen/autolabel"
VERSION = "1.4.14"

OWNER = REPO.split("/", 1)[0]
HOME_PAGE_URL = f"https://github.com/{OWNER}"
REPO_URL = f"https://github.com/{REPO}"
RELEASES_URL = f"{REPO_URL}/releases"
RELEASE_API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RAW_VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/main/autolabel/version.py"

DEFAULT_CACHE = os.path.join("~", ".autolabel", "update.json")
CACHE_TTL = 12 * 3600          # 半天查一次：界面上每次开页只读缓存，不该为版本号去等网络
TIMEOUT = 2.5                  # 网络再慢也不值得让用户多等这么久
_VERSION_PATTERN = re.compile(r'^VERSION\s*=\s*["\']([\w.\-]+)["\']', re.M)
#: 写回文件的版本号必须是三段数字：``1.2`` 在比较时等于 ``1.2.0``，
#: 界面上写着"当前版本 v1.2"、更新提示却按 1.2.0 算，看着就像版本号没生效
_STRICT_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_TAG_PATTERN = re.compile(r"^v?(\d+(?:\.\d+){0,2}(?:[-.\w]+)?)$", re.I)


def cache_path() -> str:
    """更新检查缓存路径：默认 ``~/.autolabel/update.json``（可用环境变量覆盖）。"""
    return os.path.abspath(os.path.expanduser(os.environ.get("AUTOLABEL_UPDATE_CACHE") or DEFAULT_CACHE))


def network_enabled() -> bool:
    """``AUTOLABEL_OFFLINE=1`` 时完全不碰网络（离线机器、跑测试都靠它）。"""
    return os.environ.get("AUTOLABEL_OFFLINE", "").strip().lower() not in ("1", "true", "yes", "on")


def normalized(text: object) -> str:
    """``"v1.5.0"`` / ``"V1.5"`` → ``"1.5.0"`` / ``"1.5"``；认不出来原样返回去掉 v 的串。

    GitHub 的 tag 习惯带 ``v`` 前缀，界面上又要自己拼 ``v`` + 版本号，
    不归一化就会显示成 ``v v1.5.0``，比对时也多一层心智负担。
    """
    match = _TAG_PATTERN.match(str(text or "").strip())
    return match.group(1) if match else str(text or "").strip().lstrip("vV")


def parse_version(text: object) -> Tuple[int, ...]:
    """``"v1.2.10"`` / ``"1.2"`` → ``(1, 2, 10)``；解析不出来给 ``(0,)``，比较时永远最旧。"""
    match = _TAG_PATTERN.match(str(text or "").strip())
    if not match:
        return (0,)
    core = match.group(1).split("-", 1)[0]
    parts = [int(p) for p in core.split(".") if p.isdigit()]
    return tuple(parts or [0])


def compare(left: object, right: object) -> int:
    """按数字逐段比版本，缺的段补 0：``1.2`` == ``1.2.0``。"""
    a, b = parse_version(left), parse_version(right)
    width = max(len(a), len(b))
    a += (0,) * (width - len(a))
    b += (0,) * (width - len(b))
    return (a > b) - (a < b)


def bump(version: Optional[str] = None, part: str = "patch") -> str:
    """算下一个版本号。默认 ``patch``（+0.01），``minor`` / ``major`` 留给特意安排的大版本。"""
    parts = list(parse_version(version if version is not None else VERSION))
    parts = (parts + [0, 0, 0])[:3]
    if part == "major":
        parts = [parts[0] + 1, 0, 0]
    elif part == "minor":
        parts = [parts[0], parts[1] + 1, 0]
    elif part == "patch":
        parts[2] += 1
    else:
        raise ValueError(f"未知的版本粒度：{part}（可选 patch / minor / major）")
    return ".".join(str(p) for p in parts)


def version_file(path: Optional[str] = None) -> str:
    return os.path.abspath(path or __file__)


def rewrite(new_version: str, path: Optional[str] = None) -> Tuple[str, str]:
    """把新版本号写回本文件，返回 ``(旧版本, 新版本)``。只动 ``VERSION`` 这一行。"""
    target = version_file(path)
    cleaned = str(new_version or "").strip().lstrip("vV")
    if not _STRICT_VERSION.match(cleaned):
        raise ValueError(f"版本号要写成三段数字（例如 1.2.3），现在是「{new_version}」")
    with open(target, "r", encoding="utf-8") as fh:
        source = fh.read()
    match = _VERSION_PATTERN.search(source)
    if not match:
        raise ValueError(f"没在 {target} 里找到 VERSION 定义，版本管理脚本可能被改坏了")
    old = match.group(1)
    updated = source[:match.start(1)] + cleaned + source[match.end(1):]
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(updated)
    return old, cleaned


def _http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={
        "User-Agent": f"autolabel/{VERSION}",
        "Accept": "application/vnd.github+json, text/plain;q=0.5",
    })
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", "replace")


def fetch_latest(fetcher: Optional[Callable[[str], str]] = None) -> Tuple[str, str]:
    """取 GitHub 上最新的版本号，返回 ``(版本, 来源)``；取不到返回 ``("", 原因)``。

    先认 Releases 的 tag（发布流程认的东西），没有 Releases 再退回仓库里的
    ``autolabel/version.py``——直接推 main 不发 release 的时候也能提示到。
    """
    fetch = fetcher or _http_get
    problems = []
    try:
        data = json.loads(fetch(RELEASE_API_URL))
        tag = normalized(data.get("tag_name") or data.get("name") or "")
        if parse_version(tag) != (0,):
            return tag, "release"
        problems.append("GitHub 还没有发布过版本")
    except Exception as exc:  # noqa: BLE001  断网 / 限流 / 代理都是这里的日常
        problems.append(f"Releases 接口没通（{type(exc).__name__}）")
    try:
        match = _VERSION_PATTERN.search(fetch(RAW_VERSION_URL))
        if match:
            return match.group(1), "source"
        problems.append("仓库里的 version.py 没读到版本号")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"仓库文件没读到（{type(exc).__name__}）")
    return "", " · ".join(problems) or "没取到版本"


def _as_float(value: object) -> float:
    """缓存里的时间戳认不出来就当"从没查过"（0），别让一个坏字段拖垮接口。"""
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_text(value: object) -> str:
    """只收字符串和数字：缓存被手改成 ``{"latest": ["1.9.0"]}`` 时，
    ``str(...)`` 会给出 ``"['1.9.0']"``，界面照着渲染就是一串 Python 字面量。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def _read_cache() -> dict:
    """读更新检查缓存。这是用户机器上一个会被手改、被半截写入污染的普通文件：
    ``checked_at`` 是字符串时 ``float()`` 当场 ValueError，``/api/version`` 整个 500，
    顶栏版本徽标和启动时的后台检查一起没。所有字段在这里一次性收敛成安全类型。
    """
    path = cache_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        "latest": _as_text(data.get("latest")),
        "source": _as_text(data.get("source")),
        "note": _as_text(data.get("note")),
        "checked_text": _as_text(data.get("checked_text")),
        "checked_at": _as_float(data.get("checked_at")),
    }


def _write_cache(latest: str, source: str, note: str) -> dict:
    record = {"latest": latest, "source": source, "note": note, "checked_at": time.time(),
              "checked_text": datetime.now().strftime("%Y-%m-%d %H:%M")}
    path = cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
    except OSError:
        pass    # 缓存写不进去不影响本次结果，下次再查就是
    return record


def status(fetcher: Optional[Callable[[str], str]] = None, force: bool = False) -> dict:
    """界面与命令行共用的版本状态：当前版本、GitHub 版本、有没有得升、去哪儿看。

    ``force`` 是用户在菜单里点「检查更新」；平时只读半天内的缓存。
    GitHub 比本地旧时 ``update_available`` 为假，但把 ``ahead`` 与说明带出去——
    开发中的检出比发布版新，弹"有更新"反而骗人。
    """
    cache = _read_cache()
    fresh = bool(cache.get("checked_at")) and time.time() - _as_float(cache.get("checked_at")) < CACHE_TTL
    if force or not fresh:
        if not network_enabled():
            record = _write_cache(str(cache.get("latest") or ""), "offline",
                                 "已按离线设置跳过检查（AUTOLABEL_OFFLINE）")
        else:
            latest, source = fetch_latest(fetcher)
            record = _write_cache(latest, source, "" if latest else source)
    else:
        record = cache
    latest = str(record.get("latest") or "")
    checked = parse_version(latest) != (0,)
    return {
        "current": VERSION,
        "developer": DEVELOPER,
        "developer_url": HOME_PAGE_URL,
        "repo_url": REPO_URL,
        "releases_url": RELEASES_URL,
        "latest": latest,
        "checked": checked,
        "update_available": bool(checked and compare(latest, VERSION) > 0),
        "ahead": bool(checked and compare(latest, VERSION) < 0),
        "checked_at": float(record.get("checked_at") or 0),
        "checked_text": str(record.get("checked_text") or ""),
        "source": str(record.get("source") or "none"),
        "note": str(record.get("note") or ""),
    }


def start_background_check() -> threading.Thread:
    """启动时后台查一次，绝不让工作台为了一个版本号多等一秒。"""
    from threading import Thread

    def run() -> None:
        info = status()
        if info["update_available"]:
            print(f"[info] GitHub 上有新版本 {info['latest']}（当前 {VERSION}）：{RELEASES_URL}")

    thread = Thread(target=run, name="autolabel-update-check", daemon=True)
    thread.start()
    return thread


def describe(info: Optional[dict] = None) -> str:
    """命令行 ``autolabel version`` 的那几行。"""
    lines = [f"版本 {VERSION}",
             f"开发者 {DEVELOPER} · {HOME_PAGE_URL}",
             f"仓库 {REPO_URL}（发布页 {RELEASES_URL}）"]
    info = status() if info is None else info
    if not info["checked"]:
        lines.append(f"GitHub 版本：没查到（{info['note'] or '未检查'}）· 加 --check 重试")
    elif info["update_available"]:
        lines.append(f"GitHub 最新 {info['latest']}：可以升级 → {RELEASES_URL}")
    elif info["ahead"]:
        lines.append(f"GitHub 最新 {info['latest']}：本地比它新，是开发中的版本")
    else:
        lines.append(f"GitHub 最新 {info['latest']}：已经是最新版")
    lines.append(f"上次检查 {info['checked_text'] or '从未'} · 缓存 {cache_path()}")
    return "\n".join(lines)


__all__ = ["DEVELOPER", "HOME_PAGE_URL", "REPO", "REPO_URL", "RELEASES_URL", "VERSION",
           "bump", "cache_path", "compare", "describe", "fetch_latest",
           "network_enabled", "normalized", "parse_version", "rewrite",
           "start_background_check", "status"]
