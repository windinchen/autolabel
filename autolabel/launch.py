"""起复核工作台：环境自检 → 起服务 → 端口真监听了才开浏览器。

这套流程原来写在 zsh 里（``运行工作台.command`` / ``后台实例.command``），只适配
macOS：``lsof``、``/tmp``、``$HOME/Library/Python/3.9/bin/uv`` 全是 mac 写法。
收进 Python 之后 mac 和 Windows 共用同一份判断，也进得了测试——原先那两条启动
链路（双击启动 / 后台常驻）各自抄了一遍「等端口」「失败打日志」，改一处忘另一处。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from . import platform_utils

#: 自检用的浅依赖：不加载推理框架，零点几秒出结果，缺文件/语法错当场就说清楚
LIGHT_PACKAGES: Tuple[str, ...] = ("numpy", "pillow", "pyyaml", "requests")
#: 真启动才带推理框架（ultralytics 拖 torch，冷启动要几十秒）
FULL_PACKAGES: Tuple[str, ...] = (*LIGHT_PACKAGES, "ultralytics")

#: 日志超过这个大小就先滚成 ``.1``，别让后台跑几个月吃掉磁盘
LOG_ROLL_BYTES = 2 * 1024 * 1024
#: 从日志文件名里认端口：``autolabel-8765.log`` 和滚出来的 ``autolabel-8765.log.1``
#: 都要认得出。早先按「最后一个 ``-`` 后面」取，滚动一次的日志名里带 ``.log``，
#: 于是拿 ``8765.log`` 去拼老路径，永远找不到。
LOG_PORT_PATTERN = re.compile(r"autolabel-(\d{1,5})")


def read_log(path: object, lines: int = 25) -> str:
    """日志尾几行；文件不存在时把老位置也找一遍。"""
    target = Path(str(path))
    if not target.exists():
        found = LOG_PORT_PATTERN.search(target.name)
        if found:
            for fallback in platform_utils.legacy_logs(found.group(1)):
                if fallback.exists():
                    target = fallback
                    break
    try:
        kept = target.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join(f"    {line}" for line in kept)


def rotate_log(path: object, *, limit: int = LOG_ROLL_BYTES) -> bool:
    """日志太大就先滚成 ``.1``。返回有没有滚。

    只在「新进程即将接管这个日志」时调用：崩溃重试时最需要看到的正是上一次
    留下的那几行，直接清空等于把线索擦了。
    """
    target = Path(str(path))
    try:
        if not target.is_file() or target.stat().st_size <= limit:
            return False
        target.replace(target.with_name(target.name + ".1"))
    except OSError:
        return False
    return True


def cert_ready() -> bool:
    """本地证书齐不齐。缺了服务照样起，只是走 HTTP，浏览器会关掉摄像头权限。"""
    from . import ca

    return bool(os.path.exists(ca.leaf_cert_path()) and os.path.exists(ca.leaf_key_path()))


def ensure_cert(echo: Callable[[str], None] = print) -> bool:
    """缺证书就现签一张并装进当前用户的信任列表。

    原来这一步写在 ``运行工作台.command`` 里，Windows 上就没有然后了——界面能开，
    摄像头录不了像，用户完全不知道是证书的事。放进这里两条链路都管，而且
    签失败**不算启动失败**：HTTP 看图复核照样能用，只是摄像头那条不能用。
    """
    from . import ca

    try:
        if cert_ready():
            return True
        echo("▸ 第一次用，生成本地证书（往当前用户的信任列表里加一条 AutoLabel 根证书）")
        ca.issue()
        done = ca.install_trust()
        echo(("✓ " if done.get("ok") else "! ") + str(done.get("note", "")))
        return cert_ready()
    except Exception as exc:                                # noqa: BLE001  证书不该挡住开工
        echo(f"! 证书没装好，本次走 HTTP：看图标注照常，摄像头用不了。{exc}")
        return False


def write_pid_file(process: object, path: object) -> Optional[Path]:
    """记下真正在监听端口的进程 PID，返回文件路径（写失败返回 ``None``）。

    服务管理器给的 PID 不是那个：macOS 的 launchd 记的是启动片那条
    ``uv run`` 的 PID，而 ``uv run`` 会**再 fork 一个** python 出来监听端口
    （实测两个 PID 差 1）；Windows 任务计划程序更是不给 PID。拿它去和
    ``lsof`` 看到的监听者比对，结论永远是「不是我起的」，于是窗口一关、
    屏幕上还写着就绪。让自己人写个条，比对就没这么玄学了。
    """
    target = Path(str(path))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(int(process.pid)), encoding="ascii")
    except (OSError, TypeError, ValueError):
        return None
    return target


def drop_pid_file(path: Optional[Path], pid: object) -> None:
    """进程退出后清掉 PID 条，但只清自己写的那一份（别把新实例的擦掉）。"""
    if path is None:
        return
    try:
        if path.read_text(encoding="ascii").strip() == str(int(pid)):
            path.unlink()
    except (OSError, ValueError):
        pass


def interpreter(root: object, *, framework: bool = True) -> Tuple[List[str], str]:
    """返回「怎么起一个 Python」的 argv 前缀，以及一句给人看的说明。

    优先 ``uv run --with ...``：它把「装依赖」和「起进程」合成一步，用户不用建
    venv。找不到 uv 就退回当前解释器——这时依赖得自己装好，说明里会讲。
    """
    packages = FULL_PACKAGES if framework else LIGHT_PACKAGES
    uv = platform_utils.find_uv()
    if uv:
        return platform_utils.uv_argv(uv, packages), f"uv（{uv}）"
    return [sys.executable], f"当前 Python（{sys.executable}，依赖需自己装好）"


def import_check(root: object) -> Tuple[bool, str]:
    """只 import 界面与后端那一层，不加载推理框架。

    改坏了哪个模块、少了哪个文件，在这里就说清楚，不让用户对着浏览器的
    「无法访问此网站」猜。
    """
    prefix, _ = interpreter(root, framework=False)
    argv = [*prefix, "-c", "import autolabel.review"]
    try:
        done = subprocess.run(argv, cwd=str(root), capture_output=True, text=True,
                              timeout=180, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False, "依赖自检超时了（可能在第一次下载依赖），稍等再试一次。"
    except OSError as exc:
        return False, f"没法执行 `{argv[0]}`：{exc}"
    if done.returncode == 0:
        return True, ""
    body = (done.stderr or done.stdout or "").strip()
    if "No module named" in body:
        return False, ("少了代码文件或者依赖没装齐（下面 No module named 后面那个）。"
                       "未提交的改动被误删时，跑 git status 看看少了什么。")
    return False, body.splitlines()[0][:200] if body else "自检失败（没输出）"


def serve_command(prefix: Sequence[str], dataset: str, port: int,
                  host: str = "127.0.0.1", device: Optional[str] = None,
                  half: Optional[bool] = None) -> List[str]:
    """拼 ``autolabel review`` 的参数。

    没写 device 就一个 ``--device`` 都不传：让「系统设置」里存的那档算力生效，
    否则启动器会把用户设好的 CUDA 又按回「自动选择」。
    """
    # -u 关掉解释器的输出缓冲：服务是被重定向进日志文件的，Python 默认切成
    # 8KB 块缓冲，症状就是「工作台明明开着，日志一个字都没有」——而启动后前几行
    # （用的哪块算力、证书加载没、降级警告）正是排查时最想看的那几行。
    argv = [*prefix, "-u", "-m", "autolabel", "review", str(dataset),
            "--port", str(int(port)), "--host", str(host), "--no-browser"]
    if device:
        argv += ["--device", str(device)]
    if half:
        argv += ["--half"]
    return argv


def wait_for_port(port: int, timeout: float = 90.0, host: str = "127.0.0.1") -> bool:
    """端口真的开始监听才返回。以前固定 sleep 12 秒，冷启动或起不来时都白等。"""
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if platform_utils.port_busy(port, host):
            return True
        time.sleep(1)
    return platform_utils.port_busy(port, host)


#: 只绑本机的地址：给用户看的地址统一折成 ``localhost``——证书签的就是它，
#: 而且它同时解析到 127.0.0.1 与 ::1，两种都好使
LOCALHOST_ALIASES: Tuple[str, ...] = ("", "*", "0.0.0.0", "127.0.0.1", "::", "::0")


def url_for(port: int, host: str = "127.0.0.1", secure: bool = True) -> str:
    """给用户敲进浏览器的地址：本机地址折成 localhost，IPv6 字面量补上方括号。

    ``--host ::1`` 以前会拼出 ``https://::1:8765/``，粘进地址栏根本打不开。
    """
    shown = "localhost" if platform_utils.normalize_host(host) in LOCALHOST_ALIASES else host
    return platform_utils.url_for_host(shown, port, "https" if secure else "http")


def launch(root: object, dataset: str, *, port: int = 8765, host: str = "127.0.0.1",
           device: Optional[str] = None, half: Optional[bool] = None,
           open_browser: bool = True, log_path: Optional[object] = None,
           guard_port: bool = True, wait_seconds: float = 90.0,
           pid_file: Optional[object] = None,
           echo: Callable[[str], None] = print) -> int:
    """起工作台。返回进程退出码；后台常驻（``service-run``）也走这里。

    :param guard_port: 端口已被占用就让路退出 0。后台任务不这样会每 10 秒撞一次
        端口、被 launchd 反复拉起，日志全是没用的报错。
    """
    root_path = Path(str(root)).expanduser().resolve()
    log = Path(str(log_path)) if log_path else platform_utils.service_log(port)

    problems: List[str] = []
    if not (root_path / "autolabel").is_dir():
        problems.append(f"这个目录里没有 autolabel 代码：{root_path}\n"
                        "        多半是只把启动脚本单独拷走了，请回到项目目录再跑一次。")
    if not (root_path / str(dataset)).is_dir():
        problems.append(f"数据集目录不存在：{dataset}\n"
                        f"        先建一个：python3 -m autolabel init {dataset}")
    if problems:
        for line in problems:
            echo(f"✗ {line}")
        return 1

    secure = ensure_cert(echo)
    url = url_for(port, host, secure=secure)

    if guard_port and platform_utils.port_busy(port, host):
        pid = platform_utils.listening_pid(port)
        echo(f"! 端口 {port} 已经有程序在听了（{'PID ' + str(pid) if pid else ' PID 未知'}）"
             "，这个实例先不起。")
        echo(f"  多半是工作台已经开着：{url}")
        echo("  想让新实例接管：先关掉在跑的那个，再重跑一次。")
        return 0

    echo(f"▸ 自检环境（{platform_utils.current()}）")
    ok_import, why = import_check(root_path)
    if not ok_import:
        echo(f"✗ 代码自检没通过，服务先不起，免得浏览器开一张空白页：\n        {why}")
        return 1
    prefix, note = interpreter(root_path)
    echo(f"✓ 环境正常（解释器：{note}）")

    log.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(log)
    command = serve_command(prefix, dataset, port, host, device, half)
    echo(f"▸ 启动工作台（数据集 {dataset} · 端口 {port} · 日志 {log}）")
    try:
        handle = open(log, "ab", buffering=0)
        process = subprocess.Popen(command, cwd=str(root_path), stdout=handle,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except OSError as exc:
        echo(f"✗ 起不来：{exc}")
        return 1

    keeper = write_pid_file(process, pid_file) if pid_file else None
    try:
        deadline = time.time() + float(wait_seconds)
        while time.time() < deadline:
            if process.poll() is not None:
                echo(f"✗ 服务启动后就退出了（退出码 {process.returncode}），最后几行日志：")
                tail = read_log(log)
                if tail:
                    echo(tail)
                if platform_utils.port_busy(port, host):
                    echo(f"✗ 端口 {port} 被别的程序占着。换一个再试。")
                return 1
            if platform_utils.port_busy(port, host):
                echo(f"✓ 服务已就绪：{url}")
                if open_browser:
                    try:
                        subprocess.Popen(platform_utils.open_argv(url),
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except OSError as exc:
                        echo(f"  没能自动打开浏览器（{exc}），手动访问 {url}")
                process.wait()
                return process.returncode or 0
            time.sleep(1)

        echo(f"! 等了 {int(wait_seconds)} 秒端口还没监听，可能在第一次准备环境或下载权重，"
             f"窗口先留着；日志在 {log}")
        process.wait()
        return process.returncode or 0
    finally:
        drop_pid_file(keeper, process.pid)


__all__ = ["FULL_PACKAGES", "LIGHT_PACKAGES", "LOG_PORT_PATTERN", "cert_ready",
           "drop_pid_file",
           "ensure_cert", "import_check", "interpreter", "launch", "read_log",
           "rotate_log", "serve_command", "url_for", "wait_for_port",
           "write_pid_file"]
