"""一键更新：把「GitHub 上有新版本」变成「点一下就升上去」。

只认 fast-forward。本地有未提交的改动、或者历史已经分叉，都不动仓库——把原因和
该跑的命令原样写在结果里让用户自己决定。覆盖掉用户刚才半小时的手艺，比不更新糟得多。

安装形态有三种，界面必须说实话：
``git``   —— 检出仓库，能一键更新；
``plain`` —— 拷贝/压缩包装的（没有 ``.git``），只能提示"去发布页下载新版"；
``pip``   —— 用包管理器装的，提示 ``pip install -U``。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from . import platform_utils
from .version import RELEASES_URL, REPO, VERSION

GIT_TIMEOUT = 180          # 一次 fetch 慢网络下也要得完
_TAG_LINE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?)$")

_STATE_LOCK = threading.Lock()
_STATE: dict = {
    "stage": "idle",        # idle / running / done / failed
    "message": "",
    "kind": "",             # git / plain / pip
    "branch": "",
    "from_version": "",
    "to_version": "",
    "from_commit": "",
    "to_commit": "",
    "detail": [],
    "dirty": [],
    "restartable": False,
    "started_at": "",
    "finished_at": "",
}


def _set(**fields) -> None:
    with _STATE_LOCK:
        _STATE.update(fields)


def state() -> dict:
    """当前更新状态（副本，随便给界面轮询）。"""
    with _STATE_LOCK:
        snapshot = dict(_STATE)
        snapshot["detail"] = list(_STATE.get("detail") or [])
        snapshot["dirty"] = list(_STATE.get("dirty") or [])
    snapshot["current_version"] = VERSION
    snapshot["releases_url"] = RELEASES_URL
    return snapshot


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- //
# 这一份安装是什么形态、在哪个仓库
# --------------------------------------------------------------------------- //
def repo_root(start: Optional[str] = None) -> str:
    """本包所在的仓库根（含 ``.git``）；不是 git 检出就返回空串。"""
    current = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(6):
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return ""


def install_kind() -> str:
    """git 检出 / pip 装的 / 拷贝的。判据是 ``.git`` 目录与 dist-info。"""
    if repo_root():
        return "git"
    marker = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        entries = os.listdir(marker)
    except OSError:
        entries = []
    if any(str(name).lower().startswith("autolabel-") and str(name).endswith(".dist-info")
           for name in entries):
        return "pip"
    return "plain"


def git_bin() -> str:
    path = shutil.which("git")
    if path:
        return path
    return ""


def run_git(args: List[str], cwd: str, timeout: int = GIT_TIMEOUT) -> Tuple[int, str, str]:
    """跑一条 git，返回 ``(退出码, stdout, stderr)``；git 不存在给 ``(127, "", 提示)``。"""
    exe = git_bin()
    if not exe:
        return 127, "", "这台机器上没有 git 命令"
    try:
        done = subprocess.run([exe] + list(args), cwd=cwd, capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"
    return int(done.returncode), (done.stdout or "").strip(), (done.stderr or "").strip()


def branch_name(root: str) -> str:
    code, out, _err = run_git(["rev-parse", "--abbrev-ref", "HEAD"], root, timeout=15)
    return out if code == 0 and out else ""


def head_commit(root: str) -> str:
    code, out, _err = run_git(["rev-parse", "--short", "HEAD"], root, timeout=15)
    return out if code == 0 else ""


def dirty_files(root: str, limit: int = 12) -> List[str]:
    """未提交改动的文件名（含未跟踪）。更新前必须问一句，否则 pull 会糊掉它们。"""
    # quotepath=off：本项目的路径里有中文，默认的八进制转义会把文件名糊成
    # "\344\270\255\346\226\207"，界面上念不出来，也认不出是不是自己那个文件
    code, out, _err = run_git(["-c", "core.quotepath=off", "status", "--porcelain"],
                              root, timeout=30)
    if code != 0:
        return []
    names = []
    for line in out.splitlines():
        path = line[3:].strip() if len(line) > 3 else ""
        if " -> " in path:        # 重命名行是「旧名 -> 新名」，用户关心的是新名
            path = path.split(" -> ", 1)[1]
        if path:
            names.append(path)
    return names[:limit] if limit else names


def remote_name(root: str) -> str:
    code, out, _err = run_git(["config", "--get", "remote.origin.url"], root, timeout=15)
    if code != 0 or not out:
        return ""
    return "origin"


def upstream_name(root: str) -> str:
    code, out, _err = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
                              root, timeout=15)
    return out if code == 0 and out else ""


def version_tags(root: str) -> List[str]:
    """本地已知的版本号标签（``v1.4.2`` 这种），按新旧排好。"""
    code, out, _err = run_git(["tag", "--list"], root, timeout=30)
    if code != 0:
        return []
    found: List[str] = []
    for line in out.splitlines():
        match = _TAG_LINE.match(line.strip())
        if match:
            found.append(match.group(1))
    from .version import parse_version

    return sorted(found, key=parse_version)


def newest_local_tag(root: str) -> str:
    tags = version_tags(root)
    return tags[-1] if tags else ""


# --------------------------------------------------------------------------- //
# 能更新吗 / 更新
# --------------------------------------------------------------------------- //
def preflight() -> dict:
    """能不能一键更新，不能的话原因说清楚（界面上据此决定按钮点不点得动）。"""
    kind = install_kind()
    if kind != "git":
        return {"kind": kind, "ok": False,
                "reason": "这不是 git 检出，一键更新只认 git 装的版本",
                "manual": (f"用包管理器升级：pip install -U {REPO.split('/', 1)[1]}"
                           if kind == "pip" else f"去发布页下载新版：{RELEASES_URL}")}
    if not git_bin():
        return {"kind": kind, "ok": False, "reason": "这台机器上没有 git 命令",
                "manual": "装好 Git 再点，或手动执行 git pull --ff-only"}
    root = repo_root()
    dirty = dirty_files(root)
    if dirty:
        return {"kind": kind, "ok": False, "dirty": dirty,
                "reason": f"有 {len(dirty)} 个文件还没提交，更新会动到它们",
                "manual": "先提交或暂存这些改动，再点一键更新"}
    if not upstream_name(root):
        return {"kind": kind, "ok": False,
                "reason": "当前分支没有上游分支（没跟 origin/main 绑定）",
                "manual": "手动执行 git branch --set-upstream-to=origin/main"}
    return {"kind": kind, "ok": True, "reason": "", "manual": "",
            "branch": branch_name(root), "from_commit": head_commit(root)}


def perform(pull: bool = True, runner: Optional[Callable] = None) -> dict:
    """真的更新一次：fetch → 快进 → 记结果。任何一步不对就停下并把原因留在状态里。"""
    # 带上 timeout 形参：调用处会按命令传不同超时（fetch 慢、rev-list 快），
    # 自定义 runner 少收一个参数就会在快进那一步 TypeError，测试里一跑就炸
    git = runner or (lambda args, cwd, timeout=GIT_TIMEOUT: run_git(args, cwd, timeout=timeout))
    _set(stage="running", message="联网取最新版本…", detail=[], restartable=False,
         started_at=now_text(), finished_at="", to_version="", to_commit="",
         from_version=VERSION)
    check = preflight()
    kind = check.get("kind") or install_kind()
    _set(kind=kind, dirty=list(check.get("dirty") or []))
    if not check.get("ok"):
        _set(stage="failed", message=str(check.get("reason") or "不能一键更新"),
             detail=[str(check.get("manual") or "")], finished_at=now_text())
        return state()

    root = repo_root()
    _set(branch=str(check.get("branch") or ""), from_commit=str(check.get("from_commit") or ""))
    code, out, err = git(["fetch", "--tags", "--prune", "origin"], root)
    if code != 0:
        _set(stage="failed", message="联网取更新失败",
             detail=[line for line in (err or out or "").splitlines() if line][:6],
             finished_at=now_text())
        return state()

    upstream = upstream_name(root)
    code, out, _err = git(["rev-list", "--count", f"HEAD..{upstream}"], root, timeout=30)
    behind = int(out) if out.isdigit() else 0
    if not behind:
        _set(stage="done", message="仓库已经是最新，没有可快进的提交",
             to_commit=head_commit(root), to_version=newest_local_tag(root) or VERSION,
             finished_at=now_text())
        return state()

    if not pull:
        _set(stage="done", message=f"已取到更新（落后 {behind} 个提交），未改动工作区",
             to_version=newest_local_tag(root), restartable=False, finished_at=now_text())
        return state()

    _set(message=f"快进 {behind} 个提交…")
    code, out, err = git(["merge", "--ff-only", upstream], root)
    if code != 0:
        _set(stage="failed", message="快进失败（本地历史与远端分叉了）",
             detail=[line for line in (err or out or "").splitlines() if line][:6]
             + ["手动处理：git status 看清分叉，再决定 rebase 还是另开分支"],
             finished_at=now_text())
        return state()

    _set(stage="done",
         message=f"已更新到 {head_commit(root)}（{behind} 个提交），重启服务后生效",
         to_commit=head_commit(root), to_version=newest_local_tag(root),
         restartable=True, finished_at=now_text())
    return state()


def start(pull: bool = True, worker: Optional[Callable] = None) -> dict:
    """后台跑 :func:`perform`，界面轮询 :func:`state` 看进度。已在跑就复用。"""
    current = state()
    if current.get("stage") == "running":
        return current
    _set(stage="running", message="准备更新…", started_at=now_text(), finished_at="",
         detail=[], restartable=False)
    run = worker or perform

    def work() -> None:
        try:
            run(pull=pull)
        except Exception as exc:  # noqa: BLE001  更新失败不该把服务带崩
            _set(stage="failed", message=f"更新出错：{type(exc).__name__}",
                 detail=[str(exc)], finished_at=now_text())

    threading.Thread(target=work, name="autolabel-one-click-update", daemon=True).start()
    return state()


# --------------------------------------------------------------------------- //
# 更新完重启自己：新代码要重新 import 才算升上来
# --------------------------------------------------------------------------- //
def restart_argv(argv: Optional[List[str]] = None) -> List[str]:
    """用当前解释器重新拼一条能起服务的命令。

    ``python -m autolabel review ...`` 的 ``sys.argv[0]`` 是 ``__main__.py`` 的路径，
    照原样 exec 会变成"跑那个文件"，丢掉 ``-m`` 之后 sys.path 里的工作目录；
    这种形态要显式写回 ``-m autolabel``。
    """
    args = list(sys.argv if argv is None else argv)
    head = [sys.executable]
    if args and os.path.basename(str(args[0])) == "__main__.py":
        return head + ["-u", "-m", "autolabel"] + args[1:]
    return head + args


def schedule_restart(delay: float = 1.0, sleeper: Optional[Callable] = None,
                     reexec: Optional[Callable] = None) -> dict:
    """延迟一点再重启：先把这条 HTTP 响应发完，不然界面只会看到一个断掉的请求。"""
    _set(message="服务重启中…")

    def run() -> None:
        (sleeper or time.sleep)(delay)
        try:
            (reexec or _reexec)()
        except Exception as exc:  # noqa: BLE001  重启不了就说人话，让用户自己重启
            _set(stage="failed", message=f"自动重启失败：{type(exc).__name__}",
                 detail=[f"请手动重启服务：{' '.join(restart_argv())}"])

    threading.Thread(target=run, name="autolabel-restart", daemon=True).start()
    return state()


def _reexec(argv: Optional[List[str]] = None) -> None:
    """换掉当前进程映像。Windows 的 ``os.execv`` 会把控制台搞乱，改成先起新的再退。"""
    command = restart_argv(argv)
    cwd = os.getcwd()
    if platform_utils.is_windows():
        subprocess.Popen(command, cwd=cwd)       # noqa: S603  命令由我们自己拼
        os._exit(0)
    os.execv(command[0], command)


__all__ = ["GIT_TIMEOUT", "VERSION", "branch_name", "dirty_files", "git_bin",
           "install_kind", "newest_local_tag", "perform", "preflight", "repo_root",
           "restart_argv", "run_git", "schedule_restart", "start", "state",
           "upstream_name", "version_tags"]
