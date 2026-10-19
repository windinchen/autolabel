"""复核工作台的常驻服务：登录自起、崩了自动拉起，不占终端窗口。

macOS 用 **launchd**（``~/Library/LaunchAgents`` 里一个 plist），Windows 用
**任务计划程序**（登录触发 + 失败自动重启）。两个系统的脾气不一样，但要对用户
交代的事一样，所以「注册 / 状态 / 重启 / 注销」的流程只写一遍，系统差异全压在
:func:`backend` 选出来的那两个小类里。

三条踩过的坑，别再改回去：

1. **launchd 的 ``ProgramArguments`` 里不能出现中文路径**——会被传成乱码
   （``/bin/zsh: can't open input file: /Users/…/项…注/…``），退出码 127，
   每 10 秒重来一次。所以 plist 里只写 ``~/.autolabel/services/<名字>.sh``
   这个纯 ASCII 的启动片，由它再交给项目里的实例。
2. **macOS 的后台任务读不到「桌面 / 文稿 / 下载」**，而且不弹窗问权限。启动片
   先真读一次项目里的文件，读不到就写清楚该点哪里再 ``exit 0``；配合 plist 里的
   ``KeepAlive = {SuccessfulExit: false}``，launchd 不再每 10 秒空转重试刷日志。
3. **端口上有人在听 ≠ 我这个实例起来了**——多半是用户自己双击开的工作台。
   就绪判定必须确认「在听的那个人就是我起的」，否则窗口一关服务就没了，
   而屏幕上还写着「已就绪」。
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

from . import platform_utils
from .version import DEVELOPER

CONF_NAME = os.path.join("configs", "后台实例.conf")
CONF_ENV = "AUTOLABEL_SERVICE_CONF"
#: 实例名要写进 launchd 标签、启动片文件名，不能带中文；长度也留足余量
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"
#: ``Get-ScheduledTask`` 的 ``State``（Windows）与 ``launchctl list`` 的读数都往这里对
READY_STATES = ("ready", "queued", "not running", "就绪")
RUNNING_STATES = ("running", "正在运行")
DISABLED_STATES = ("disabled", "已禁用")


class ServiceError(Exception):
    """一条标题 + 若干行解释。全挤在一行用户读不下去。"""

    def __init__(self, head: str, *details: str) -> None:
        super().__init__(head)
        self.head = head
        self.details = list(details)

    def lines(self) -> List[str]:
        return [self.head, *[f"  {line}" for line in self.details]]


@dataclass(frozen=True)
class Instance:
    """清单里的一行：名字 端口 数据集 设备 半精度。"""

    name: str
    port: int
    dataset: str
    device: str = ""            # 空串 = 跟「系统设置」里存的那档走
    half: bool = False

    def env_lines(self) -> List[str]:
        """给启动片/排查用的键值行，格式和旧版 ``print-env`` 保持一致。"""
        return [f"PORT={self.port}", f"DATASET={self.dataset}",
                f"DEVICE={self.device or ''}", f"HALF={'yes' if self.half else 'no'}"]


def conf_path(root: object) -> Path:
    """清单路径。默认在项目里，测试或第二份清单用 ``AUTOLABEL_SERVICE_CONF`` 指到别处。"""
    raw = os.environ.get(CONF_ENV)
    if raw:
        return Path(raw).expanduser()
    return Path(str(root)).expanduser() / CONF_NAME


def read_conf(path: object, *, root: Optional[object] = None) -> List[Instance]:
    """解析清单。任何一行不合规就抛 :class:`ServiceError`，把行号和原因说清楚。"""
    target = Path(str(path))
    if not target.is_file():
        hint = ["照下面这个格式建一个（一行一个实例）：",
                "  mps 8765 datasets/demo - no",
                "  cuda 8766 datasets/demo cuda no"]
        example = Path(str(target)).with_name(Path(str(target)).name + ".example")
        if example.is_file():
            hint = ["最省事的是拷项目里那份模板，再按自己的端口和数据集改一行：",
                    f"  cp {example} {target}   （Windows：copy {example} {target}）",
                    *hint]
        raise ServiceError(f"没有实例清单：{target}", *hint)
    base = Path(str(root)).expanduser() if root else target.parent
    seen: Dict[str, Instance] = {}
    order: List[str] = []
    try:
        body = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ServiceError(f"清单读不出来：{target}", f"系统说：{exc}") from exc
    for lineno, raw in enumerate(body.splitlines(), start=1):
        fields = raw.split()
        if not fields or fields[0].startswith("#"):
            continue
        while len(fields) < 5:
            fields.append("")
        name, port, dataset, device, half = fields[:5]

        if not re.match(NAME_PATTERN, name):
            raise ServiceError(
                f"{target} 第 {lineno} 行的实例名不合规：{name}",
                "只能用字母、数字、- 和 _——它要写进服务标签和启动片文件名，不能带中文")
        if name in seen:
            raise ServiceError(f"{target} 第 {lineno} 行实例名重复：{name}")
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ServiceError(f"{target} 第 {lineno} 行端口不是数字：{port}（实例 {name}）")
        dataset_dir = Path(dataset)
        if not dataset_dir.is_absolute():
            dataset_dir = base / dataset_dir
        if not dataset_dir.is_dir():
            raise ServiceError(
                f"{target} 第 {lineno} 行的数据集不存在：{dataset}（实例 {name}）",
                f"先 python -m autolabel init {dataset}，或把这一行改成已有目录")
        # 存解析后的绝对路径：后台任务的工作目录是项目根，但 launchd /
        # 任务计划程序给的环境千差万别，相对路径在这儿只是一颗延时引爆的雷
        seen[name] = Instance(name=name, port=int(port), dataset=str(dataset_dir),
                             device="" if device == "-" else device,
                             half=half.lower() in ("yes", "y", "true", "1", "on"))
        order.append(name)
    if not order:
        raise ServiceError(f"{target} 里一个实例都没有",
                           "去掉注释至少留一行：mps 8765 datasets/demo - no")
    return [seen[name] for name in order]


def services_dir() -> Path:
    """启动片放这儿：路径纯 ASCII，launchd / 任务计划程序只认这种。"""
    return platform_utils.state_dir() / "services"


def pid_file(name: str) -> Path:
    """实例自己写的 PID 条。为什么需要它见 :func:`autolabel.launch.write_pid_file`。"""
    return services_dir() / f"{name}.pid"


def pidfile_pid(name: str) -> Optional[int]:
    """读 PID 条。读不到 / 里面不是数字都返回 ``None``（表示"认不出来"，不是"没有"）。"""
    try:
        text = pid_file(name).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return int(text) if text.isdigit() else None


# --------------------------------------------------------------------------- #
# 两个系统各一个后端。命令都拆成「纯函数返回 argv」，这样在 mac 上也能单测
# Windows 那一份，不必真有一台 Windows 机器。
# --------------------------------------------------------------------------- #

class Backend:
    name = ""
    unit_suffix = ""

    def unit_path(self, name: str) -> Path:
        raise NotImplementedError

    def render_unit(self, inst: Instance) -> str:
        raise NotImplementedError

    def shim_path(self, inst: Instance) -> Path:
        return services_dir() / f"{inst.name}{self.shim_suffix}"

    def render_shim(self, inst: Instance, root: object) -> str:
        raise NotImplementedError

    def register_argv(self, inst: Instance) -> List[str]:
        raise NotImplementedError

    def unregister_argv(self, name: str) -> List[str]:
        raise NotImplementedError

    def restart_argv(self, name: str) -> List[str]:
        raise NotImplementedError

    def query_argv(self, name: str) -> List[str]:
        raise NotImplementedError

    def parse_state(self, stdout: str) -> Tuple[bool, bool, Optional[int]]:
        raise NotImplementedError

    def extra_paths(self, inst: Instance) -> List[Path]:
        return []

    def prepare(self, inst: Instance, root: object) -> None:
        """注册前需要额外做的事（Windows 要建目录联接），默认没有。"""

    def label(self, name: str) -> str:
        return name


class LaunchdBackend(Backend):
    """macOS：``~/Library/LaunchAgents/com.<开发者>.autolabel.<名字>.plist``。"""

    name = "launchd"
    unit_suffix = ".plist"
    shim_suffix = ".sh"

    def label(self, name: str) -> str:
        dev = str(DEVELOPER or "windinchen").strip().lower().replace(" ", "")
        return f"com.{dev}.autolabel.{name}"

    def agents_dir(self) -> Path:
        return Path(os.path.expanduser("~/Library/LaunchAgents"))

    def unit_path(self, name: str) -> Path:
        return self.agents_dir() / f"{self.label(name)}{self.unit_suffix}"

    def render_unit(self, inst: Instance) -> str:
        shim = self.shim_path(inst)
        log = str(platform_utils.service_log(inst.port))
        home = os.path.expanduser("~")
        rows = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
            '<plist version="1.0">',
            "<dict>",
            f"  <key>Label</key><string>{xml_escape(self.label(inst.name))}</string>",
            "  <key>ProgramArguments</key>",
            "  <array>",
            "    <string>/bin/zsh</string>",
            f"    <string>{xml_escape(str(shim))}</string>",
            f"    <string>{xml_escape(inst.name)}</string>",
            "  </array>",
            "  <key>EnvironmentVariables</key>",
            "  <dict>",
            f"    <key>HOME</key><string>{xml_escape(home)}</string>",
            "  </dict>",
            "  <key>RunAtLoad</key><true/>",
            "  <key>KeepAlive</key>",
            "  <dict><key>SuccessfulExit</key><false/></dict>",
            "  <key>ThrottleInterval</key><integer>10</integer>",
            f"  <key>StandardOutPath</key><string>{xml_escape(log)}</string>",
            f"  <key>StandardErrorPath</key><string>{xml_escape(log)}</string>",
            "</dict>",
            "</plist>",
        ]
        return "\n".join(rows) + "\n"

    def render_shim(self, inst: Instance, root: object) -> str:
        prefix, _ = _runner_prefix()
        project = str(Path(str(root)).expanduser())
        quoted = " ".join(_shell_quote(part) for part in prefix)
        lines = [
            "#!/bin/zsh",
            "# 由 autolabel service 生成，别手改——下次注册会覆盖它。",
            "# 项目路径带中文，直接写进 launchd 的 ProgramArguments 会被传成乱码，"
            "所以先落在这儿。",
            f"PROJECT={_shell_quote(project)}",
            "# 被「桌面文件夹」权限挡住时，-r / -e 这类判断照样说\"可读\"，只有真读一次才知道。",
            "# 说清原因再 exit 0，launchd 就不会每 10 秒空转重试一次。",
            'if ! head -1 "$PROJECT/autolabel/version.py" >/dev/null 2>&1; then',
            '  print -P "%F{red}✗%f 后台任务读不到项目里的文件：$PROJECT"',
        ]
        lines += ['  print -r -- "  %s"' % _zsh_double_quote(line)
                  for line in platform_utils.tcc_hint(project)]
        lines += [
            '  exit 0',
            'fi',
            'cd -- "$PROJECT" || exit 0',
            f"exec {quoted} -m autolabel service-run {_shell_quote(inst.name)}",
        ]
        return "\n".join(lines) + "\n"

    def register_argv(self, inst: Instance) -> List[str]:
        return ["launchctl", "bootstrap", _gui_domain(), str(self.unit_path(inst.name))]

    def unregister_argv(self, name: str) -> List[str]:
        return ["launchctl", "bootout", f"{_gui_domain()}/{self.label(name)}"]

    def restart_argv(self, name: str) -> List[str]:
        return ["launchctl", "kickstart", "-k", f"{_gui_domain()}/{self.label(name)}"]

    def query_argv(self, name: str) -> List[str]:
        return ["launchctl", "list", self.label(name)]

    def parse_state(self, stdout: str) -> Tuple[bool, bool, Optional[int]]:
        loaded = '"Label"' in stdout or "PID" in stdout or "ProgramArguments" in stdout
        pid: Optional[int] = None
        for line in stdout.splitlines():
            token = line.strip()
            if token.startswith('"PID" ='):
                digits = token.partition("=")[2].strip().rstrip(";").strip()
                if digits.isdigit():
                    pid = int(digits)
        return loaded, pid is not None, pid


class TaskSchedulerBackend(Backend):
    """Windows：登录触发的计划任务 + 失败自动重启（对应 launchd 的 KeepAlive）。"""

    name = "taskscheduler"
    unit_suffix = ".task.xml"
    shim_suffix = ".cmd"

    def label(self, name: str) -> str:
        return f"AutoLabel-{name}"

    def unit_path(self, name: str) -> Path:
        return services_dir() / f"{self.label(name)}{self.unit_suffix}"

    def work_link(self, inst: Instance) -> Path:
        """指向项目目录的目录联接。

        用它是因为 ``.cmd`` 文件是按 OEM 代码页解析的，把中文路径写进文件里容易
        变乱码；``cd /d \"%~dp0<名字>.work\"`` 全程 ASCII，中文只留在文件系统里。
        """
        return services_dir() / f"{inst.name}.work"

    def extra_paths(self, inst: Instance) -> List[Path]:
        return [self.work_link(inst)]

    def prepare(self, inst: Instance, root: object) -> None:
        link = self.work_link(inst)
        target = Path(str(root)).expanduser().resolve()
        if _link_target_is(link, target):
            return
        _remove_path(link)
        done = subprocess.run(junction_argv(link, target), capture_output=True, text=True,
                              shell=_needs_shell(link))
        if done.returncode != 0:
            raise ServiceError(
                f"建不了指向项目的目录联接：{link}",
                "任务计划程序要靠它绕开中文路径，没有就只能把项目挪到英文目录。",
                f"系统说：{(done.stderr or done.stdout or '').strip()[:180]}")

    def render_shim(self, inst: Instance, root: object) -> str:
        prefix, _ = _runner_prefix()
        project = str(Path(str(root)).expanduser())
        # 先试 ASCII 联接；联接没建成时退回直接 cd（写文件时按 OEM 代码页编码）
        link = self.work_link(inst)
        jump = (f'cd /d "%~dp0{link.name}"' if _link_target_is(link, Path(project))
                else f'cd /d "{project}"')
        joined = " ".join(_cmd_quote(part) for part in prefix)
        return "\r\n".join([
            "@echo off",
            "rem 由 autolabel service 生成，别手改——下次注册会覆盖它。",
            "rem 端口被占就 exit 0，任务计划程序只认「非零退出」才重启，"
            "别让它在用户自己开着的工作台后面反复撞端口。",
            jump,
            joined + " -m autolabel service-run " + inst.name,
            "exit /b %ERRORLEVEL%",
            "",
        ])

    def render_unit(self, inst: Instance) -> str:
        return "\r\n".join([
            f"task={self.label(inst.name)}",
            f"shim={self.shim_path(inst)}",
            f"port={inst.port}",
            "",
        ])

    def register_argv(self, inst: Instance) -> List[str]:
        return ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                "Bypass", "-Command", register_script(inst, self)]

    def unregister_argv(self, name: str) -> List[str]:
        return ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                "Bypass", "-Command",
                f"Unregister-ScheduledTask -TaskName '{self.label(name)}' "
                "-Confirm:$false -ErrorAction SilentlyContinue"]

    def restart_argv(self, name: str) -> List[str]:
        return ["schtasks", "/End", "/TN", self.label(name)] + ["&"] + \
            ["schtasks", "/Run", "/TN", self.label(name)]

    def query_argv(self, name: str) -> List[str]:
        """查任务状态用 PowerShell，不用 ``schtasks``。

        ``schtasks /Query`` 连**值**都跟着系统语言翻译（中文系统上是
        「状态: 正在运行」），换台机器就对不上；``Get-ScheduledTask.State``
        永远返回英文枚举，省掉一整套本地化匹配。
        """
        script = ("$t = Get-ScheduledTask -TaskName '%s' -ErrorAction SilentlyContinue; "
                  "if ($t) { $t.State }" % _ps_quote(self.label(name)))
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]

    def parse_state(self, stdout: str) -> Tuple[bool, bool, Optional[int]]:
        state = ""
        for line in stdout.splitlines():
            token = line.strip()
            if token:
                state = token.lower()       # PowerShell 只吐一个词，取最后一个非空行
        if not state:
            return False, False, None
        known = READY_STATES + RUNNING_STATES + DISABLED_STATES + ("unknown",)
        return state in known, state in RUNNING_STATES, None


def register_script(inst: Instance, backend: TaskSchedulerBackend) -> str:
    """PowerShell 注册脚本。拆出来是为了能在 mac 上单测内容。

    ``-RestartCount`` / ``-RestartInterval`` 对应 launchd 的
    ``KeepAlive = {SuccessfulExit: false}``：非零退出才重启，所以启动片里
    「端口被占」和「读不到项目」都必须 ``exit 0``。
    """
    shim = str(backend.shim_path(inst))
    return (
        f"$a = New-ScheduledTaskAction -Execute '{_ps_quote(shim)}'; "
        "$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew "
        "-RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) "
        "-ExecutionTimeLimit ([TimeSpan]::Zero); "
        "$t = New-ScheduledTaskTrigger -AtLogOn; "
        f"Register-ScheduledTask -TaskName '{_ps_quote(backend.label(inst.name))}' "
        "-Action $a -Settings $s -Trigger $t -Force | Out-Null"
    )


def junction_argv(link: object, target: object) -> List[str]:
    """建目录联接的命令。Windows 用 ``mklink /J``（不需要管理员），
    其它系统退回符号链接（由 :func:`_make_junction_posix` 直接建）。"""
    return ["cmd", "/c", "mklink", "/J", str(link), str(target)]


def backend(name: Optional[str] = None) -> Optional[Backend]:
    """按名字或当前系统挑后端。"""
    wanted = (name or platform_utils.service_backend()).lower()
    table = {"launchd": LaunchdBackend, "taskscheduler": TaskSchedulerBackend}
    maker = table.get(wanted)
    return maker() if maker else None


def _gui_domain() -> str:
    return f"gui/{os.getuid()}" if hasattr(os, "getuid") else "gui/500"


def _runner_prefix() -> Tuple[List[str], str]:
    from .launch import interpreter

    return interpreter(None, framework=True)


def _shell_quote(value: object) -> str:
    """单引号包住，里面的单引号按 zsh 的写法绕出来。

    一律加引号：项目路径里有空格、括号、中文都常见，"看着没特殊字符就不加"
    省下的是几个字节，赔进去的是一台起不来的机器。
    """
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _zsh_double_quote(value: object) -> str:
    """塞进 zsh 双引号里的一行提示：``\\  "  $  ` `` 这四个字符在双引号里仍有特殊含义。

    提示文案是我们自己写的，眼下没有这几个字符；但加一次转义的成本是一行，
    省下的将来是「启动片里少了一行指引、后台任务默默失败」这种难查现场。
    """
    text = str(value)
    for char in ("\\", '"', "$", "`"):
        text = text.replace(char, "\\" + char)
    return text


def _cmd_quote(value: object) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _ps_quote(value: object) -> str:
    return str(value).replace("'", "''")


def _link_target_is(link: object, target: object) -> bool:
    path = Path(str(link))
    if not path.exists():
        return False
    try:
        return Path(str(path)).resolve() == Path(str(target)).expanduser().resolve()
    except OSError:
        return False


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _needs_shell(path: object) -> bool:
    return platform_utils.is_windows()


# --------------------------------------------------------------------------- #
# 面向用户的四个动作：注册 / 状态 / 重启 / 注销。流程只写一遍，
# 系统差异全在上面那两个后端里。
# --------------------------------------------------------------------------- #

def _query(backend_: Backend, name: str) -> Tuple[bool, bool, Optional[int]]:
    try:
        done = run(backend_.query_argv(name))
    except OSError:
        return False, False, None
    if done.returncode != 0:
        return False, False, None
    return backend_.parse_state(done.stdout or "")


def ascii_only(path: object) -> bool:
    """服务只会读纯 ASCII 路径下的东西，中文路径要提前拦住说清楚。"""
    return all(32 <= ord(char) < 127 for char in str(path))


def _note_listener(port: int, echo: Callable[[str], None]) -> None:
    pid = platform_utils.listening_pid(port)
    if pid:
        echo(f"  端口 {port} 现在是 PID {pid} 在听——多半是你自己开着的工作台，窗口一关就断")


def explain_down(port: int, echo: Callable[[str], None], root: object = None) -> None:
    """实例没起来时把最可能的原因翻成人话，别只甩几行日志。"""
    from .launch import read_log

    for path in platform_utils.log_paths(port):
        tail = read_log(path)
        if tail:
            echo(tail)
            body = tail
            if "读不到项目" in body or "can't open input file" in body:
                echo("这不是端口被占：后台任务读不到项目里的文件。")
                # 问的是"项目在不在受保护文件夹里"，就得用项目自己的路径，
                # 拿当前工作目录判断等于问错了对象
                for line in platform_utils.tcc_hint(root or Path.cwd()):
                    echo(f"  {line}")
                echo("  办好之前想先用界面：双击 运行工作台.command（Windows 双击 运行工作台.bat），"
                     "前台进程不受这条限制。")
            return


def write_unit(backend_: Backend, inst: Instance, root: object) -> Path:
    """把启动片与任务定义写到磁盘上，返回任务定义的路径。"""
    shim = backend_.shim_path(inst)
    if not ascii_only(shim):
        raise ServiceError(
            f"启动片的路径带非 ASCII 字符，服务起不来：{shim}",
            "家目录名带中文就会撞上这条（Windows 上就是 C:\\Users\\<中文名>）。两条出路：",
            "  ① 只把「服务用的目录」另指到纯英文路径：设环境变量 AUTOLABEL_HOME=D:\\autolabel"
            "（macOS/Linux 写 ~/autolabel-svc），重开终端再注册一次；",
            "  ② 先不用常驻服务：双击 运行工作台 照常干活，这条限制只管后台任务，不管前台。")
    services_dir().mkdir(parents=True, exist_ok=True)
    backend_.prepare(inst, root)
    body = backend_.render_shim(inst, root)
    tmp = shim.with_name(shim.name + ".tmp")
    try:
        tmp.write_bytes(body.encode(_shim_encoding(), errors="replace"))
        os.replace(tmp, shim)
        os.chmod(shim, 0o755)
    except OSError as exc:
        _remove_path(tmp)
        raise ServiceError(f"生成启动片失败：{shim}", f"系统说：{exc}") from exc
    unit = backend_.unit_path(inst.name)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit_tmp = unit.with_name(unit.name + ".tmp")
    try:
        unit_tmp.write_text(backend_.render_unit(inst), encoding="utf-8")
        os.replace(unit_tmp, unit)
        os.chmod(unit, 0o644)
    except OSError as exc:
        _remove_path(unit_tmp)
        raise ServiceError(f"生成任务定义失败：{unit}", f"系统说：{exc}") from exc
    return unit


def _shim_encoding() -> str:
    """Windows 的 ``.cmd`` 是按 OEM 代码页解析的，写 UTF-8 会让中文路径变乱码。"""
    if not platform_utils.is_windows():
        return "utf-8"
    try:
        import ctypes

        return "cp%d" % ctypes.windll.kernel32.GetOEMCP()      # type: ignore[attr-defined]
    except Exception:                                          # noqa: BLE001
        return "utf-8"


def _verify_ready(backend_: Backend, inst: Instance, *, timeout: float = 40.0,
                  echo: Callable[[str], None] = print) -> str:
    """返回 ``ok`` / ``foreign`` / ``down``。

    ``foreign`` 是「端口上有人在听，但那个人不是我起的实例」——多半是用户自己
    开着的工作台。把这种情况报成就绪，用户关掉窗口服务就没了。
    """
    from .launch import wait_for_port

    if not wait_for_port(inst.port, timeout):
        return "down"
    _, _, backend_pid = _query(backend_, inst.name)
    listener = platform_utils.listening_pid(inst.port)
    # 优先认实例自己写的 PID 条：服务管理器给的 PID 常常是包装进程（uv / cmd），
    # 真正在听端口的是它 fork 出来的孩子，直接比一定不相等
    mine = pidfile_pid(inst.name) or backend_pid
    if mine and listener and listener == mine:
        echo(f"✓ {inst.name} → {url_for(inst.port)}（PID {mine}）")
        return "ok"
    if mine and listener is None:
        # 认不出在听的是谁（没有 lsof 之类的工具），宁可保守也别乱报就绪
        echo(f"! {inst.name} · 端口 {inst.port} 有人在听，但认不出是不是这个实例")
        return "foreign"
    echo(f"! {inst.name} · 端口 {inst.port} 有人在听，但不是这个后台实例"
         f"（PID {listener or '未知'}）")
    echo("  多半是你自己双击开了工作台，那个窗口一关服务就断。")
    echo(f"  想让后台接管：关掉那个窗口，再跑 autolabel service --restart {inst.name}。")
    return "foreign"


def url_for(port: int) -> str:
    from .launch import url_for as _url

    return _url(port)


def install(root: object, *, dry_run: bool = False,
            echo: Callable[[str], None] = print) -> int:
    """按清单注册全部实例。清单里删掉的实例会被顺手注销，不留僵尸任务。"""
    backend_ = backend()
    if backend_ is None:
        echo(f"✗ 这个系统（{platform_utils.current()}）还没有常驻服务后端，"
             "双击 运行工作台 照常能用。")
        return 1
    root_path = Path(str(root)).expanduser()
    instances = read_conf(conf_path(root_path), root=root_path)
    echo(f"▸ 读实例清单 {conf_path(root_path)}（{len(instances)} 个）")
    skipped = 0
    for inst in instances:
        unit = write_unit(backend_, inst, root_path)
        if dry_run:
            echo(f"✓ {inst.name} · 端口 {inst.port} · 任务定义与启动片已写出（没碰服务管理器）")
            echo(f"  {unit}")
            continue
        loaded, _, _ = _query(backend_, inst.name)
        if loaded:
            echo(f"▸ {inst.name} · 先摘掉在跑的旧任务")
            run(backend_.unregister_argv(inst.name))
            _wait_port_free(inst.port, 10)
        if platform_utils.port_busy(inst.port):
            skipped += 1
            echo(f"! {inst.name} · 端口 {inst.port} 还被别的程序占着，这个实例先不起")
            echo("  多半是你自己双击过 运行工作台。要么关掉那个窗口再跑一次，")
            echo("  要么把清单里这一行的端口换一个。")
            continue
        from .launch import rotate_log

        # 显式注册就是一次全新的开始：旧日志里上次失败的原因会把人带偏
        for path in platform_utils.log_paths(inst.port):
            if path.exists():
                rotate_log(path)
                break
        done = run(backend_.register_argv(inst))
        if done.returncode != 0:
            raise ServiceError(
                f"{inst.name} · 注册失败（{backend_.label(inst.name)}）",
                f"服务管理器拒绝了 {unit}",
                (done.stderr or done.stdout or "").strip()[:200])
        echo(f"✓ {inst.name} · 端口 {inst.port} · 已注册（登录/开机自动加载）")
    prune_stale(backend_, instances, root_path, dry_run=dry_run, echo=echo)
    if dry_run:
        echo("")
        echo("这是 --dry-run：服务管理器没被动过，在跑的还是原来的定义。")
        return 0
    echo("")
    echo("▸ 等服务起来")
    failed = 0
    for inst in instances:
        loaded, _, _ = _query(backend_, inst.name)
        verdict = _verify_ready(backend_, inst, echo=echo)
        if verdict == "ok":
            continue
        if verdict == "foreign":
            skipped += 1
            continue
        failed += 1
        if loaded:
            echo(f"! {inst.name} · 40 秒没监听：")
            explain_down(inst.port, echo, root_path)
        else:
            echo(f"! {inst.name} · 服务管理器没接这个任务，见 {backend_.unit_path(inst.name)}")
    echo("")
    if failed:
        echo(f"✗ 有 {failed} 个实例没起来，上面是原因。")
        return 1
    if skipped:
        echo(f"  另外 {skipped} 个因为端口被占跳过了，见上面。")
    echo("✓ 常驻服务就位。以后开机自己就起来，不用再双击任何东西。")
    return 0


def _wait_port_free(port: int, tries: int = 10) -> None:
    import time as _time

    for _ in range(tries):
        if not platform_utils.port_busy(port):
            return
        _time.sleep(1)


def prune_stale(backend_: Backend, instances: Sequence[Instance], root: object,
                *, dry_run: bool = False,
                echo: Callable[[str], None] = print) -> None:
    """清单里删掉的实例，别让服务管理器按旧定义反复重启一个不存在的实例。"""
    keep = {inst.name for inst in instances}
    for path in _known_units(backend_):
        name = _unit_name(backend_, path)
        if name is None or name in keep:
            continue
        if not dry_run:
            run(backend_.unregister_argv(name))
            _remove_path(path)
            _remove_path(services_dir() / f"{name}{backend_.shim_suffix}")
            _remove_path(pid_file(name))
        echo(f"! 清单里已经没有 {name}，注销并清掉它留下的任务定义和启动片")


def _known_units(backend_: Backend) -> List[Path]:
    folder = backend_.unit_path("_").parent
    try:
        return sorted(folder.glob(f"*{backend_.unit_suffix}"))
    except OSError:
        return []


def _unit_name(backend_: Backend, path: Path) -> Optional[str]:
    """从任务定义的文件名还原实例名；**不是我们的前缀就返回** ``None``。

    ``~/Library/LaunchAgents`` 是全体 App 共用的目录，里面躺着 Steam、Clash Verge、
    蒲公英……的任务定义。早先这里对不上前缀也照样把整个文件名当实例名返回，于是
    「清理清单里删掉的实例」变成了「把这台机器上所有不认识的任务逐个 bootout +
    删文件」——用户只会发现装了本软件之后 Steam 的清理任务没了。
    """
    stem = path.name[: -len(backend_.unit_suffix)]
    prefix = backend_.label("")
    if not stem.startswith(prefix):
        return None
    return stem[len(prefix):].lstrip(".-") or None


def status(root: object, *, echo: Callable[[str], None] = print) -> int:
    backend_ = backend()
    if backend_ is None:
        echo(f"✗ 这个系统（{platform_utils.current()}）还没有常驻服务后端。")
        return 1
    root_path = Path(str(root)).expanduser()
    instances = read_conf(conf_path(root_path), root=root_path)
    echo(f"常驻服务实例（清单 {conf_path(root_path)} · 后端 {backend_.name}）")
    for inst in instances:
        label = backend_.label(inst.name)
        loaded, running, pid = _query(backend_, inst.name)
        note = "算力跟系统设置" if not inst.device else f"算力指定为 {inst.device}"
        if not loaded:
            if backend_.unit_path(inst.name).exists():
                echo(f"! {inst.name} 的任务定义已写好，但服务管理器还没加载 · {url_for(inst.port)}")
                _note_listener(inst.port, echo)
                echo(f"  下次登录会自己加载；现在就想跑：autolabel service --restart {inst.name}")
            else:
                echo(f"✗ {inst.name} 未注册 · {url_for(inst.port)}")
                echo("  跑一次 autolabel service 就注册上了")
            echo(f"  {note} · 日志 {platform_utils.service_log(inst.port)}")
            echo("")
            continue
        mine = pidfile_pid(inst.name) or pid
        if running:
            echo(f"✓ {inst.name} 在跑 · PID {mine or '未知'} · {url_for(inst.port)}"
                 f" · 数据集 {inst.dataset}")
        else:
            echo(f"✗ {inst.name} 已注册但没在跑 · {url_for(inst.port)} · {label}")
            _note_listener(inst.port, echo)
        echo(f"  {note} · 日志 {platform_utils.service_log(inst.port)}")
        if not running:
            explain_down(inst.port, echo, root_path)
            echo("")
    return 0


def restart(root: object, target: Optional[str] = None,
            *, echo: Callable[[str], None] = print) -> int:
    backend_ = backend()
    if backend_ is None:
        echo(f"✗ 这个系统（{platform_utils.current()}）还没有常驻服务后端。")
        return 1
    root_path = Path(str(root)).expanduser()
    instances = [inst for inst in read_conf(conf_path(root_path), root=root_path)
                 if not target or inst.name == target]
    if not instances:
        echo(f"✗ 清单里没有实例：{target}")
        return 1
    done = 0
    for inst in instances:
        loaded, _, _ = _query(backend_, inst.name)
        if not loaded:
            # 注册时端口被占会跳过加载，任务定义其实已经写好——这里顺手补上
            unit = backend_.unit_path(inst.name)
            if unit.exists() and run(backend_.register_argv(inst)).returncode == 0:
                echo(f"✓ {inst.name} 已加载并启动（端口 {inst.port}）")
                done += 1
                continue
            echo(f"✗ {inst.name} 还没注册，先跑 autolabel service")
            continue
        if run(backend_.restart_argv(inst.name)).returncode == 0:
            echo(f"✓ {inst.name} 已重启（端口 {inst.port}）")
            done += 1
        else:
            echo(f"✗ {inst.name} 重启失败（{backend_.label(inst.name)}）")
    if not done:
        return 1
    failed = 0
    for inst in instances:
        if _verify_ready(backend_, inst, echo=echo) == "down":
            echo(f"! {inst.name} · 40 秒没监听，日志最后几行：")
            explain_down(inst.port, echo, root_path)
            failed += 1
    return 1 if failed else 0


def remove(root: object, target: Optional[str] = None,
           *, echo: Callable[[str], None] = print) -> int:
    backend_ = backend()
    if backend_ is None:
        echo(f"✗ 这个系统（{platform_utils.current()}）还没有常驻服务后端。")
        return 1
    root_path = Path(str(root)).expanduser()
    instances = [inst for inst in read_conf(conf_path(root_path), root=root_path)
                 if not target or inst.name == target]
    for inst in instances:
        run(backend_.unregister_argv(inst.name))
        echo(f"✓ {inst.name} 已注销")
        for path in [backend_.unit_path(inst.name), backend_.shim_path(inst),
                     pid_file(inst.name), *backend_.extra_paths(inst)]:
            _remove_path(path)
    echo("  清单里的实例都处理完了；要恢复常驻服务再跑一次 autolabel service")
    return 0


def instance_for(root: object, name: str) -> Instance:
    root_path = Path(str(root)).expanduser()
    for inst in read_conf(conf_path(root_path), root=root_path):
        if inst.name == name:
            return inst
    raise ServiceError(f"清单里没有实例：{name}",
                       "改过 configs/后台实例.conf 就重跑一次 autolabel service")


def run(argv: Sequence[str], **kw: object) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, **kw)  # type: ignore[arg-type]


__all__ = ["Backend", "CONF_ENV", "CONF_NAME", "Instance", "LaunchdBackend",
           "ServiceError", "TaskSchedulerBackend", "ascii_only", "backend",
           "conf_path", "explain_down", "install", "instance_for", "junction_argv",
           "pid_file", "pidfile_pid", "read_conf", "register_script", "remove",
           "restart", "services_dir", "status", "write_unit"]
