"""起工作台这条链路（autolabel/launch.py）。

以前这套流程写在 ``运行工作台.command`` 和 ``后台实例.command`` 里，两份 zsh 各抄
一遍，还只适配 macOS。收进 Python 之后要守住的老坑：

1. **端口真开始监听了才开浏览器**——以前固定 sleep 12 秒，冷启动或压根起不来时
   浏览器只会演一张"无法访问此网站"，用户完全不知道背后发生了什么；
2. **不写 ``--device`` 就一个 ``--device`` 都不传**，让「系统设置」里存的那档算力
   生效，否则启动器会把用户设好的 CUDA 又按回"自动选择"；
3. **总是带 ``--no-browser``**：起服务的人负责开浏览器，服务端自己别再开一次
   （以前两边各开一次，屏幕上两个标签页）；
4. **端口被占就退出码 0**，后台任务才不会每 10 秒撞一次端口刷日志。
"""
import os
import sys
from pathlib import Path

import pytest

from autolabel import launch as L


class FakeProcess:
    """替掉真进程：只记录被怎么起来，以及"退出了没有"。"""

    def __init__(self, pid=4242, exit_code=None):
        self.pid = pid
        self.returncode = exit_code
        self._exit = exit_code
        self.waited = False

    def poll(self):
        return self._exit

    def wait(self):
        self.waited = True
        return self._exit or 0


@pytest.fixture()
def project(tmp_path, monkeypatch):
    """一个最小"项目"：有 autolabel 目录和一个数据集，证书环节直接放过。"""
    (tmp_path / "autolabel").mkdir()
    (tmp_path / "datasets" / "demo").mkdir(parents=True)
    monkeypatch.setattr(L, "ensure_cert", lambda echo=(lambda *_: None): True)
    monkeypatch.setattr(L, "import_check", lambda _root: (True, ""))
    monkeypatch.setattr(L, "interpreter", lambda _root, framework=True: ([sys.executable], "假解释器"))
    return tmp_path


@pytest.fixture(autouse=True)
def _port_starts_empty(monkeypatch):
    """默认当作「端口是空的」：这台机器上常驻服务很可能正在听 8765。

    ``launch()`` 第一件事就是端口守卫，端口真被占着它会直接让路退出 0，于是
    「自检失败不该开空白页」这类用例走的是另一条分支；更糟的是 ``listening_pid``
    会去调真的 ``lsof``，而用例为了拦 ``Popen`` 换掉的替身接不住那种调用。
    用户把工作台开着跑测试就红一片——而这恰恰是最常发生的场景。
    需要「端口被占」的用例自己在正文里覆盖这条默认值。
    """
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: False)
    monkeypatch.setattr(L.platform_utils, "listening_pid", lambda *_a, **_k: None)


class Spawn:
    """记录 Popen 调用，并按脚本安排"端口什么时候开始监听"。"""

    def __init__(self, monkeypatch, busy_after=0, process=None):
        self.calls = []
        self.process = process or FakeProcess()
        self.busy_after = busy_after
        self.probes = 0
        monkeypatch.setattr(L.subprocess, "Popen", self._popen)
        monkeypatch.setattr(L.time, "sleep", lambda _s: None)

    def _popen(self, argv, **_kw):
        self.calls.append([str(part) for part in argv])
        return self.process

    def busy(self, _port, _host="127.0.0.1", **_kw):
        self.probes += 1
        return self.probes > self.busy_after


def collect():
    lines = []
    return lines, lines.append


def _repo_root():
    """真实仓库根目录：这条测试跑的就是一次真 import。"""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




# --------------------------------------------------------------------------- #
# 拼命令：三条老坑各钉一条
# --------------------------------------------------------------------------- #
def test_serve_command_always_passes_no_browser():
    """服务端不许自己开浏览器：以前启动器和服务各开一次，一屏两个标签页。"""
    argv = L.serve_command(["py"], "datasets/demo", 8765)
    assert "--no-browser" in argv
    assert argv[-4:] == ["review", "datasets/demo", "--port", "8765"] or "review" in argv


def test_serve_command_keeps_the_log_unbuffered():
    """``-u`` 必须带上：日志是排查时唯一能看的东西，不能压在缓冲区里。

    服务的 stdout 被重定向进日志文件，Python 于是默认切成 8KB 块缓冲——症状是
    「工作台开着，日志是空的」，等它攒够 8KB 时用户早就重启三回了。
    """
    argv = L.serve_command(["py"], "datasets/demo", 8765)
    assert argv[argv.index("py") + 1] == "-u", "解释器参数得紧跟解释器，放到 -m 后面就成了程序参数"


def test_serve_command_leaves_device_to_the_settings_panel():
    """没指定算力就一个 --device 都不传——传了就把用户存的 CUDA 按回"自动选择"。"""
    plain = L.serve_command(["py"], "datasets/demo", 8765)
    assert "--device" not in plain
    with_device = L.serve_command(["py"], "datasets/demo", 8765, device="cuda")
    assert with_device[with_device.index("--device") + 1] == "cuda"


def test_serve_command_only_adds_half_when_asked():
    assert "--half" not in L.serve_command(["py"], "d", 1, half=None)
    assert "--half" not in L.serve_command(["py"], "d", 1, half=False)
    assert "--half" in L.serve_command(["py"], "d", 1, half=True)


def test_url_for_shows_localhost_rather_than_the_wildcard():
    assert L.url_for(8765, "0.0.0.0") == "https://localhost:8765/"
    assert L.url_for(8765, "192.168.1.9", secure=False) == "http://192.168.1.9:8765/"


# --------------------------------------------------------------------------- #
# 解释器：有 uv 用 uv，没有就退回当前 Python 并把话说清楚
# --------------------------------------------------------------------------- #
def test_interpreter_prefers_uv(monkeypatch):
    monkeypatch.setattr(L.platform_utils, "find_uv", lambda: Path("/bin/uv"))
    argv, note = L.interpreter("/project")
    assert argv[0] == "/bin/uv" and "ultralytics" in argv
    assert "uv" in note


def test_interpreter_falls_back_to_current_python(monkeypatch):
    monkeypatch.setattr(L.platform_utils, "find_uv", lambda: None)
    argv, note = L.interpreter("/project")
    assert argv == [sys.executable]
    assert "依赖需自己装好" in note, "退回裸 Python 时必须说明依赖得自备，否则用户等不到原因"


def test_light_check_does_not_load_the_inference_framework(monkeypatch):
    """自检只 import 界面那层，不拖 torch——不然每次双击都要先等几十秒。"""
    monkeypatch.setattr(L.platform_utils, "find_uv", lambda: Path("/bin/uv"))
    argv, _note = L.interpreter("/project", framework=False)
    assert "ultralytics" not in argv and "numpy" in argv


def test_import_check_passes_on_a_healthy_tree(monkeypatch, tmp_path):
    """自检是真跑一次 ``import autolabel.review``，不是数文件在不在。"""
    monkeypatch.setattr(L, "interpreter", lambda _root, framework=True: ([sys.executable], "当前 Python"))
    ok, why = L.import_check(_repo_root())
    assert ok is True and why == ""


def test_import_check_speaks_chinese_when_a_module_is_missing(monkeypatch, tmp_path):
    """缺文件 / 改坏模块时，要在窗口里说人话，而不是让用户对着浏览器的"无法访问"猜。"""
    monkeypatch.setattr(L, "interpreter", lambda _root, framework=True: (
        [sys.executable, "-c", "import definitely_not_a_real_module"], "假解释器"))
    ok, why = L.import_check(tmp_path)
    assert ok is False
    assert "No module named" in why, "少了依赖/文件要点名到具体是哪一个"


# --------------------------------------------------------------------------- #
# 日志：滚存与读尾巴
# --------------------------------------------------------------------------- #
def test_rotate_log_rolls_only_a_oversized_file(tmp_path):
    log = tmp_path / "autolabel-8765.log"
    log.write_text("小文件", encoding="utf-8")
    assert L.rotate_log(log, limit=1024) is False
    log.write_text("x" * 50, encoding="utf-8")
    assert L.rotate_log(log, limit=10) is True
    assert (tmp_path / "autolabel-8765.log.1").read_text(encoding="utf-8") == "x" * 50
    assert not log.exists(), "滚存后旧名要让位给新日志"


def test_rotate_log_survives_a_missing_file(tmp_path):
    assert L.rotate_log(tmp_path / "不存在.log") is False


def test_read_log_tails_and_indents(tmp_path):
    log = tmp_path / "autolabel-8765.log"
    log.write_text("\n".join("行%d" % i for i in range(1, 41)), encoding="utf-8")
    tail = L.read_log(log, lines=3)
    assert tail.splitlines() == ["    行38", "    行39", "    行40"]


def test_read_log_finds_the_old_temp_location(tmp_path, monkeypatch):
    """日志位置搬过一次（``/tmp`` → ``~/.autolabel/logs``），老现场还得读得到。"""
    legacy = tmp_path / "tmp" / "autolabel-8765.log"
    legacy.parent.mkdir()
    legacy.write_text("上次崩在这里\n", encoding="utf-8")
    monkeypatch.setattr(L.platform_utils.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    assert "上次崩在这里" in L.read_log(tmp_path / "logs" / "autolabel-8765.log")


@pytest.mark.parametrize("name", ["autolabel-8765.log", "autolabel-8765.log.1"])
def test_read_log_recognises_a_rolled_log_when_falling_back(tmp_path, monkeypatch, name):
    """滚动一次的日志文件名尾巴多个 ``.1``：按「最后一个 - 后面」取端口会取出
    ``8765.log``，于是永远找不到老位置的现场。"""
    legacy = tmp_path / "tmp" / "autolabel-8765.log"
    legacy.parent.mkdir()
    legacy.write_text("上次崩在这里\n", encoding="utf-8")
    monkeypatch.setattr(L.platform_utils.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    assert "上次崩在这里" in L.read_log(tmp_path / "logs" / name)


def test_read_log_on_a_missing_file_is_silent(tmp_path):
    """查状态时日志十有八九还不存在，这里不能把异常抛到用户脸上。"""
    assert L.read_log(tmp_path / "没这个文件.log") == ""


# --------------------------------------------------------------------------- #
# PID 条：服务管理器给的 PID 不是那个在听端口的
# --------------------------------------------------------------------------- #
def test_write_pid_file_records_the_child(tmp_path):
    target = L.write_pid_file(FakeProcess(pid=4321), tmp_path / "svc" / "mps.pid")
    assert target.read_text(encoding="ascii") == "4321"


def test_drop_pid_file_only_removes_its_own(tmp_path):
    path = tmp_path / "mps.pid"
    path.write_text("111", encoding="ascii")
    L.drop_pid_file(path, 222)
    assert path.exists(), "新实例写下的条不能被旧实例擦掉"
    L.drop_pid_file(path, 111)
    assert not path.exists()


def test_write_pid_file_never_breaks_the_launch(tmp_path):
    """写不进去（目录只读、PID 奇怪）就当没有这条，绝不能因此起不了服务。"""
    assert L.write_pid_file(FakeProcess(pid="不是数字"), tmp_path / "a.pid") is None
    assert L.write_pid_file(FakeProcess(), tmp_path / "占位" / "x.pid") is not None
    (tmp_path / "占位").joinpath("x.pid").unlink()


# --------------------------------------------------------------------------- #
# launch()：三条失败路径 + 一条成功路径
# --------------------------------------------------------------------------- #
def test_launch_refuses_a_missing_dataset(project):
    lines, echo = collect()
    assert L.launch(project, "datasets/没有这个", echo=echo) == 1
    assert any("autolabel init" in line for line in lines), "要给出下一步命令"


def test_launch_refuses_a_directory_without_the_code(tmp_path):
    lines, echo = collect()
    assert L.launch(tmp_path, "datasets/demo", echo=echo) == 1
    assert any("没有 autolabel 代码" in line for line in lines)


def test_launch_steps_aside_when_the_port_is_taken(project, monkeypatch):
    """多半是用户自己开着的工作台。这里必须**退出码 0**：后台任务不这样会每
    10 秒撞一次端口，日志里全是没用的报错。"""
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: True)
    monkeypatch.setattr(L.platform_utils, "listening_pid", lambda _p: 99)
    spawn = Spawn(monkeypatch)
    lines, echo = collect()
    assert L.launch(project, "datasets/demo", echo=echo) == 0
    assert spawn.calls == [], "端口被占就不该再起一个"
    assert any("已经有程序在听" in line for line in lines)


def test_launch_does_not_open_a_blank_page_when_self_check_fails(project, monkeypatch):
    lines, echo = collect()
    monkeypatch.setattr(L, "import_check", lambda _root: (False, "少了 autolabel.qc"))
    spawn = Spawn(monkeypatch)
    assert L.launch(project, "datasets/demo", echo=echo) == 1
    assert spawn.calls == []
    assert any("代码自检没通过" in line for line in lines)


def test_launch_opens_browser_only_after_the_port_listens(project, monkeypatch):
    """端口没监听之前不许开浏览器——这是这条链路存在的理由。"""
    spawn = Spawn(monkeypatch, busy_after=2)      # 前两次探测都没监听
    monkeypatch.setattr(L.platform_utils, "port_busy", spawn.busy)
    opened = []
    monkeypatch.setattr(L.platform_utils, "open_argv", lambda url: opened.append(url) or ["true"])
    lines, echo = collect()
    assert L.launch(project, "datasets/demo", echo=echo) == 0
    assert [call[0] for call in spawn.calls][0] == sys.executable
    assert opened == ["https://localhost:8765/"]
    assert any("服务已就绪" in line for line in lines)


def test_launch_reports_the_log_tail_when_the_server_dies(project, monkeypatch):
    log = project / "run.log"
    log.write_text("Traceback...\nImportError: 少了东西\n", encoding="utf-8")
    Spawn(monkeypatch, process=FakeProcess(exit_code=3))
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: False)
    lines, echo = collect()
    assert L.launch(project, "datasets/demo", log_path=log, echo=echo) == 1
    assert any("退出码 3" in line for line in lines)
    assert any("ImportError" in line for line in lines)


def test_launch_writes_and_clears_the_pid_file(project, monkeypatch):
    """PID 条存在的意义：状态页要能证明"在听端口的那个人就是我起的"。"""
    spawn = Spawn(monkeypatch)
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: True)
    pid = project / "services" / "mps.pid"
    seen = {}

    real_write = L.write_pid_file

    def spy(process, path):
        target = real_write(process, path)
        seen["content"] = target.read_text(encoding="ascii") if target else ""
        return target

    monkeypatch.setattr(L, "write_pid_file", spy)
    monkeypatch.setattr(L.platform_utils, "listening_pid", lambda _p: spawn.process.pid)
    L.launch(project, "datasets/demo", pid_file=pid, guard_port=False,
             echo=lambda *_: None)
    assert seen["content"] == str(spawn.process.pid)
    assert not pid.exists(), "服务退出后要留干净，否则下次注册会误判成还在跑"


def test_launch_warns_but_keeps_the_window_when_startup_is_slow(project, monkeypatch):
    """第一次准备环境/下载权重可能超过等待时间：窗口留着，别直接关掉。"""
    Spawn(monkeypatch)
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: False)
    lines, echo = collect()
    assert L.launch(project, "datasets/demo", wait_seconds=0, echo=echo) == 0
    assert any("窗口先留着" in line for line in lines)


def test_wait_for_port_returns_at_once_when_it_is_up(monkeypatch):
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: True)
    assert L.wait_for_port(8765, timeout=0.1) is True


def test_wait_for_port_gives_up(monkeypatch):
    monkeypatch.setattr(L.platform_utils, "port_busy", lambda *_a, **_k: False)
    monkeypatch.setattr(L.time, "sleep", lambda _s: None)
    assert L.wait_for_port(8765, timeout=0.01) is False


def test_cert_ready_looks_for_both_files(tmp_path, monkeypatch):
    from autolabel import ca

    monkeypatch.setattr(ca, "leaf_cert_path", lambda: str(tmp_path / "leaf.pem"))
    monkeypatch.setattr(ca, "leaf_key_path", lambda: str(tmp_path / "leaf-key.pem"))
    assert L.cert_ready() is False
    (tmp_path / "leaf.pem").write_text("x", encoding="utf-8")
    assert L.cert_ready() is False, "只有证书没有私钥照样起不了 HTTPS"
    (tmp_path / "leaf-key.pem").write_text("x", encoding="utf-8")
    assert L.cert_ready() is True


def test_ensure_cert_never_blocks_the_workbench(monkeypatch):
    """缺 openssl 也得能看图标注：证书失败只降级成 HTTP，不能把启动顶掉。"""
    from autolabel import ca

    def boom(*_a, **_k):
        raise RuntimeError("找不到 openssl 命令")

    monkeypatch.setattr(L, "cert_ready", lambda: False)
    monkeypatch.setattr(ca, "issue", boom)
    lines, echo = collect()
    assert L.ensure_cert(echo) is False
    assert any("摄像头用不了" in line for line in lines)
