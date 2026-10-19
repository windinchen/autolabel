"""跨平台的差异集中在这一处。

这个项目一开始只跑 macOS，``lsof``、``/tmp``、``open -R``、``security`` 这些系统
专属命令散在 zsh 启动脚本里。要同时支持 Windows，判断散着写就得全库找一遍，
漏一处就是「mac 上好好的，Windows 上莫名其妙起不来」。所以约定：
**别的模块只问「该用哪个」，不自己写 ``sys.platform`` 分支**。

这里只放两类东西：一是纯粹的「哪个系统用哪条命令」，二是家目录/日志目录这种
每个系统规矩不同的路径。业务逻辑一律不放进来。
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

MACOS = "macos"
WINDOWS = "windows"
LINUX = "linux"

#: macOS 的「受保护文件夹」：后台任务（launchd 直接起的，不是终端的子进程）默认
#: 读不到里面的文件，而且**不弹窗问权限**，只会闷头崩。实测这三个都读不到，
#: 家目录下的普通文件夹读得到。
PROTECTED_HOME_DIRS: Tuple[str, ...] = ("Desktop", "Documents", "Downloads")

#: ``uv`` 的常见位置。``${VAR}`` 形式在 POSIX 上就是同名环境变量，统一走
#: :func:`_expand`。macOS 上官方 install.sh 早先装到 ``Library/Python/3.9/bin``，
#: 现在装到 ``.local/bin``，两个都认。
UV_CANDIDATES: Tuple[str, ...] = (
    "~/.local/bin/uv",
    "~/Library/Python/3.9/bin/uv",
    "/usr/local/bin/uv",
    "/opt/homebrew/bin/uv",
)
WINDOWS_UV_CANDIDATES: Tuple[str, ...] = (
    r"%USERPROFILE%\.local\bin\uv.exe",
    r"%LOCALAPPDATA%\Programs\uv\uv.exe",
    r"%USERPROFILE%\.cargo\bin\uv.exe",
)

#: ``openssl`` 的常见位置。签本地证书要用它：macOS/Linux 自带，Windows 上它一般
#: 藏在 Git for Windows 里（普通用户机器上「有 Git」比「有 OpenSSL」常见得多）。
OPENSSL_CANDIDATES: Tuple[str, ...] = (
    "/usr/bin/openssl",
    "/opt/homebrew/bin/openssl",
    "/usr/local/bin/openssl",
)
WINDOWS_OPENSSL_CANDIDATES: Tuple[str, ...] = (
    r"%ProgramFiles%\Git\usr\bin\openssl.exe",
    r"%ProgramFiles(x86)%\Git\usr\bin\openssl.exe",
    r"%LOCALAPPDATA%\Programs\Git\usr\bin\openssl.exe",
    r"%ProgramFiles%\OpenSSL-Win64\bin\openssl.exe",
    r"%SystemRoot%\System32\openssl.exe",
)


def current() -> str:
    """``"macos"`` / ``"windows"`` / ``"linux"``。别的模块拿这个决定用哪条命令。"""
    if sys.platform == "darwin":
        return MACOS
    if sys.platform.startswith("win"):
        return WINDOWS
    return LINUX


def is_macos() -> bool:
    return current() == MACOS


def is_windows() -> bool:
    return current() == WINDOWS


def _expand(raw: str) -> Path:
    """先展 ``${VAR}`` / ``%VAR%``，再展 ``~``。Windows 和 POSIX 的写法都能吃。"""
    return Path(os.path.expanduser(os.path.expandvars(str(raw))))


def state_dir() -> Path:
    """这台机器的用户级数据目录：``~/.autolabel``（可用 ``AUTOLABEL_HOME`` 整体挪走）。

    设置、密钥、版本缓存、日志、服务启动片都在它下面。**这里刻意不用 ``/tmp``
    或 ``%TEMP%``**：Windows 的存储感知会定期清临时目录，崩溃现场跟着一起没了，
    而崩溃现场正是第二天排查时唯一能看的东西。
    """
    raw = os.environ.get("AUTOLABEL_HOME")
    return _expand(raw) if raw else Path(os.path.expanduser("~/.autolabel"))


def log_dir() -> Path:
    return state_dir() / "logs"


def service_log(port: object) -> Path:
    """某个实例的日志路径。端口不是数字时按字符串拼，调用方自己保证合法。"""
    return log_dir() / f"autolabel-{port}.log"


def legacy_logs(port: object) -> List[Path]:
    """老版本写在临时目录里的日志（macOS 是 ``/tmp``，Windows 是 ``%TEMP%``）。

    只用来「顺手也看一眼」，不再往那儿写——临时目录会被系统清，崩溃现场跟着没。
    """
    stem = f"autolabel-{port}.log"
    out = [Path(tempfile.gettempdir()) / stem]
    posix_tmp = Path("/tmp") / stem          # macOS/Linux 上老脚本写死的正是这里
    if posix_tmp not in out:
        out.append(posix_tmp)
    return out


def log_paths(port: object) -> List[Path]:
    """新位置 + 老位置，按读取优先级排好。状态页取日志时按这个顺序找。"""
    return [service_log(port), *legacy_logs(port)]


def find_uv() -> Optional[Path]:
    """找到 ``uv`` 就返回路径，找不到返回 ``None``。

    先 ``PATH``，再逐个试已知安装位置。用 ``is_file()`` 而不是 ``-x``：Windows 上
    没有执行位的概念，``access(X_OK)`` 结果不可信。
    """
    found = shutil.which("uv")
    if found:
        return Path(found)
    candidates = WINDOWS_UV_CANDIDATES if is_windows() else UV_CANDIDATES
    for raw in candidates:
        path = _expand(raw)
        if path.is_file():
            return path
    return None


def uv_argv(uv: object, packages: Sequence[str], *, python: str = "3.11",
            offline: bool = True) -> List[str]:
    """``uv run`` 前缀，把「装依赖」和「起进程」合成一步。"""
    argv = [str(uv), "run"]
    if offline:
        argv += ["--offline", "--quiet"]
    for package in packages:
        argv += ["--with", package]
    if python:
        argv += ["--python", python]
    argv += ["python3"]
    return argv


def find_openssl() -> Optional[Path]:
    """找到 ``openssl`` 就返回路径，找不到返回 ``None``。"""
    found = shutil.which("openssl")
    if found:
        return Path(found)
    candidates = WINDOWS_OPENSSL_CANDIDATES if is_windows() else OPENSSL_CANDIDATES
    for raw in candidates:
        path = _expand(raw)
        if path.is_file():
            return path
    return None


def openssl_missing_hint() -> str:
    """找不到 openssl 时的出路。真正会看到这句的是 Windows——mac 自带。"""
    if is_windows():
        return ("装 Git for Windows，它自带 openssl：https://git-scm.com/download/win\n"
                "        装完重开命令行再试；不想装就用 HTTP："
                "python -m autolabel review --no-ssl（摄像头会不可用）")
    if is_macos():
        return "macOS 自带 openssl，多半是 PATH 被改短了：确认里面还有 /usr/bin"
    return "装一下 openssl：Debian/Ubuntu 用 sudo apt install openssl，Fedora 用 sudo dnf install"


def force_utf8_streams() -> None:
    """把标准输出/错误固定在 UTF-8。

    Windows 把输出重定向进文件时按本地代码页写字节，中文机器是 cp936，而本项目
    满屏的 ``✓ ✗ ▸`` 都不在 cp936 里——写第一行日志就 ``UnicodeEncodeError``。
    双击开窗口时看不见（控制台走 Unicode API），只有输出进日志文件才炸，最难查。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        if str(getattr(stream, "encoding", "") or "").replace("-", "").lower() == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def probe_address(host: object) -> Tuple[object, str]:
    """探端口该连哪个地址、用哪个地址族。

    绑在通配地址上时改探 ``127.0.0.1``：拿 ``::`` 当**连接**目标是未定义行为
    （macOS 上直接报错），而且 AF_INET 的套接字拿 ``::`` 去连永远连不上——
    端口明明有人听也会说没人，症状是「服务起好了，但脚本死等 90 秒」。
    """
    text = str(host).strip().strip("[]")
    if text in ("", "*", "0.0.0.0", "127.0.0.1", "::", "::0"):
        return socket.AF_INET, "127.0.0.1"
    if ":" in text:
        return socket.AF_INET6, text
    return socket.AF_INET, text


def port_busy(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    """端口有没有人监听。

    用 ``connect`` 试一下，不用 ``lsof``——Windows 没有 lsof，而这是最关键的判断
    （「服务到底起没起来」全看它），必须每个系统都准。
    """
    try:
        target = int(port)
    except (TypeError, ValueError):
        return False
    family, address = probe_address(host)
    where = (address, target, 0, 0) if family == socket.AF_INET6 else (address, target)
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex(where) == 0


#: 通配地址：绑它等于「本机所有网卡都能访问」，给用户看的地址要折回 localhost
WILDCARD_HOSTS: Tuple[str, ...] = ("", "*", "0.0.0.0", "::", "::0")


def normalize_host(host: object) -> str:
    """把用户写的监听地址收成干净形式：去掉 IPv6 外面那对方括号。"""
    text = str(host).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def bind_family(host: object) -> Tuple[int, str]:
    """**监听**该地址该用哪个地址族、实际绑哪个地址。

    必须和 :func:`probe_address` 分开：探活时 ``::`` 要折成 ``127.0.0.1``
    （拿通配地址当*连接*目标是未定义行为），而监听时 ``::`` 就得真的绑 ``::``。
    以前这里没有对应判断，服务端的地址族写死 IPv4，于是 ``--host ::1`` 与
    ``--host ::`` 直接在 ``bind()`` 上抛 ``gaierror``——可证书早就签了 ``::1``、
    启动提示也写着 ``::`` 能用，等于宣称支持却起不来。
    """
    text = normalize_host(host)
    if text in ("", "*"):
        return socket.AF_INET, "0.0.0.0"
    if text in ("::", "::0"):
        return socket.AF_INET6, "::"
    if ":" in text:
        return socket.AF_INET6, text
    return socket.AF_INET, text


def url_for_host(host: object, port: int, scheme: str = "https") -> str:
    """给用户敲进浏览器的地址。

    IPv6 字面量必须带方括号：``https://::1:8765/`` 粘进地址栏是打不开的，
    而 ``--host ::`` 折成 ``localhost`` 才点得动（``::`` 本身不是合法主机名）。
    """
    text = normalize_host(host)
    if text in WILDCARD_HOSTS:
        text = "localhost"
    if ":" in text:
        text = f"[{text}]"
    return f"{scheme}://{text}:{int(port)}/"


def listening_pid_argv(port: int) -> List[str]:
    """查「在听的进程是谁」的命令。拆成纯函数是为了能在 mac 上单测 Windows 那条。"""
    if is_windows():
        script = ("$c = Get-NetTCPConnection -LocalPort %d -State Listen "
                  "-ErrorAction SilentlyContinue | Select-Object -First 1; "
                  "if ($c) { $c.OwningProcess }" % int(port))
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    return ["lsof", "-nP", "-tiTCP:%d" % int(port), "-sTCP:LISTEN"]


def listening_pid(port: int) -> Optional[int]:
    """在听的进程 PID；查不到返回 ``None``。

    调用方**不能把 ``None`` 当成「就是我」**——它只代表认不出来。用在状态页：
    「端口上有人在听」不等于「我这个后台实例起来了」，多半是用户自己开的工作台。
    """
    argv = listening_pid_argv(port)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (done.stdout or "").splitlines():
        token = line.strip()
        if token.isdigit():
            return int(token)
    return None


def reveal_argv(path: object) -> List[str]:
    """在系统文件管理器里定位一个文件/目录。

    Windows 的 ``explorer`` 只认 ``/select,`` 这种连笔写法，而且**永远返回非零**
    退出码，所以调用方要用 ``Popen`` 并且不看返回码。
    """
    target = os.path.normpath(str(path))
    if is_macos():
        return ["open", "-R", target]
    if is_windows():
        return ["explorer", f"/select,{target}"]
    return ["xdg-open", target if os.path.isdir(target) else os.path.dirname(target)]


def open_argv(url: str) -> List[str]:
    """用默认浏览器打开一个网址（Windows 上 ``start`` 是 cmd 内置，得绕一下）。"""
    if is_windows():
        return ["cmd", "/c", "start", "", url]
    if is_macos():
        return ["open", url]
    return ["xdg-open", url]


def remove_hint(path: object) -> str:
    """「删掉这个文件」在本系统怎么敲。提示语里的 ``rm`` 在 Windows 上没用。"""
    return f'del "{path}"' if is_windows() else f"rm {path}"


def service_backend() -> str:
    """常驻服务用哪个系统机制：``launchd`` / ``taskscheduler`` / ``none``。"""
    return {MACOS: "launchd", WINDOWS: "taskscheduler"}.get(current(), "none")


def protected_parent(path: object) -> Optional[str]:
    """路径落在 macOS 受保护文件夹里就返回那个文件夹名，否则 ``None``。

    只有 macOS 有这道限制，别的系统一律返回 ``None``。
    """
    if not is_macos():
        return None
    try:
        resolved = Path(str(path)).expanduser().resolve()
    except OSError:
        return None
    home = Path(os.path.expanduser("~")).resolve()
    for name in PROTECTED_HOME_DIRS:
        guard = home / name
        try:
            resolved.relative_to(guard)
        except ValueError:
            continue
        return name
    return None


def tcc_hint(path: object) -> List[str]:
    """读不到项目文件时给用户的出路，实测可靠的那条排在第一位。"""
    if protected_parent(path) is None:
        return []
    return [
        "macOS 不让后台任务读「桌面 / 文稿 / 下载」这三个文件夹，也不会弹窗问你要权限。",
        "① 实测可靠：把整个项目挪出这三个文件夹（比如 ~/work/自动标注），再跑一次注册命令；",
        "② 或「系统设置 → 隐私与安全性 → 完全磁盘访问权限」里按 +（⌘⇧G 手输路径）加入 "
        "/bin/zsh。",
        "   给「终端」授权多半没用——后台任务是 launchd 直接起的，不是终端的子进程。",
    ]


__all__ = [
    "LINUX", "MACOS", "PROTECTED_HOME_DIRS", "WINDOWS",
    "current", "find_openssl", "find_uv", "force_utf8_streams", "is_macos",
    "is_windows", "legacy_logs", "openssl_missing_hint",
    "listening_pid", "listening_pid_argv", "log_dir", "log_paths", "open_argv",
    "bind_family", "normalize_host", "url_for_host",
    "port_busy", "probe_address", "protected_parent", "remove_hint", "reveal_argv",
    "service_backend",
    "service_log", "state_dir", "tcc_hint", "uv_argv",
]


# --------------------------------------------------------------------------- #
# 跨进程文件锁
# --------------------------------------------------------------------------- #

def _posix_fcntl():
    try:
        import fcntl
    except ImportError:          # Windows 上没有这个模块
        return None
    return fcntl


def _win_msvcrt():
    try:
        import msvcrt
    except ImportError:
        return None
    return msvcrt


def file_lock_supported() -> bool:
    """这台机器有没有真正的跨进程文件锁（POSIX 用 ``flock``，Windows 用 ``msvcrt``）。"""
    return _posix_fcntl() is not None or _win_msvcrt() is not None


def file_lock_acquire(handle, blocking: bool = False) -> bool:
    """给一个已打开的文件句柄占独占锁，占不到返回 ``False``。

    Windows 上没有 ``flock``，只有 ``msvcrt.locking``——它锁的是"从当前读写位置起
    的若干字节"，所以必须先 ``seek(0)``。以前 Windows 一律当成"没别人"，
    结果是两个实例同时写素材账本会互相覆盖、同时下同一个权重会把对方正在
    追加的 ``.part`` 截断，而这两种情况在 mac 上都复现不了。

    ``blocking=True`` 与 ``flock`` 同一口径：一直等到拿到锁为止（进程一退出系统就
    自动放锁，不会留下没人解的死锁）。
    """
    fcntl = _posix_fcntl()
    if fcntl is not None:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
            return True
        except OSError:
            return False
    msvcrt = _win_msvcrt()
    if msvcrt is None:
        return True                      # 哪种锁都没有：别因为锁挡了正事
    while True:
        try:
            handle.seek(0)
        except OSError:
            pass
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            if not blocking:
                return False
            time.sleep(0.05)


def file_lock_release(handle) -> None:
    """放掉 :func:`file_lock_acquire` 占的锁；句柄关掉时系统也会自己放，所以失败不报。"""
    fcntl = _posix_fcntl()
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        return
    msvcrt = _win_msvcrt()
    if msvcrt is None:
        return
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
