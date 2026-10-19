"""平台差异的验收单（autolabel/platform_utils.py）。

这个项目的规矩是"别的模块只问该用哪条命令，不自己写 sys.platform 分支"，
前提是这里每条命令都给对。以前判断散在 zsh 启动脚本里，Windows 上就是
"mac 好好的，这边莫名其妙起不来"，而且没人能一眼看出漏了哪一处。

每条 Windows 分支都在 mac 上跑：命令都是"返回 argv 的纯函数"，不需要真有一台
Windows 机器——代价是这里验的是"命令写对了没"，不是"Windows 上真跑通了"。
"""
import io
import os
import socket
import sys
from pathlib import Path

import pytest

from autolabel import platform_utils as pu


# --------------------------------------------------------------------------- #
# 我是谁：current() 判错，后面每条分支都会跟着错
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    ("darwin", pu.MACOS), ("win32", pu.WINDOWS), ("linux", pu.LINUX),
    ("freebsd", pu.LINUX),
])
def test_current_recognises_every_os(monkeypatch, value, expected):
    monkeypatch.setattr(sys, "platform", value)
    assert pu.current() == expected


@pytest.mark.parametrize("value,backend", [
    ("darwin", "launchd"), ("win32", "taskscheduler"), ("linux", "none"),
])
def test_service_backend_per_os(monkeypatch, value, backend):
    """常驻服务的后端只能按系统选；Linux 现在还没实现，要说"没有"而不是硬凑一个。"""
    monkeypatch.setattr(sys, "platform", value)
    assert pu.service_backend() == backend


# --------------------------------------------------------------------------- #
# 家目录与日志：日志不能再放临时目录
# --------------------------------------------------------------------------- #
def test_state_dir_can_be_moved_wholesale(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOLABEL_HOME", str(tmp_path / "别处的数据"))
    assert pu.state_dir() == tmp_path / "别处的数据"


def test_state_dir_defaults_to_dot_autolabel_under_home(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTOLABEL_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pu.state_dir() == tmp_path / ".autolabel"


def test_service_log_is_not_in_a_temp_folder(monkeypatch, tmp_path):
    """日志以前写 ``/tmp``：Windows 的存储感知会定期清临时目录，
    崩溃现场跟着没，第二天排查时什么都看不到。"""
    monkeypatch.setenv("AUTOLABEL_HOME", str(tmp_path / "state"))
    log = pu.service_log(8765)
    assert log.parent == tmp_path / "state" / "logs"
    assert log.name == "autolabel-8765.log"
    assert not str(log).startswith("/tmp")


def test_legacy_logs_are_read_but_not_written(tmp_path, monkeypatch):
    """老位置的日志要能顺手读到（老任务的现场），但新日志一律写新位置。"""
    monkeypatch.setenv("AUTOLABEL_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(pu.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    legacy = pu.legacy_logs(8765)
    assert legacy[0] == tmp_path / "tmp" / "autolabel-8765.log"
    assert Path("/tmp/autolabel-8765.log") in legacy
    assert len(legacy) == len(set(legacy)), "同一台机器上两个位置重了要合并"
    assert pu.log_paths(8765)[0] == pu.service_log(8765), "新位置优先级必须最高"


# --------------------------------------------------------------------------- #
# 端口：这是"服务到底起没起来"唯一的依据，必须每个系统都准
# --------------------------------------------------------------------------- #
def test_port_busy_sees_a_real_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert pu.port_busy(port) is True
        assert pu.port_busy(port + 1) is False


@pytest.mark.parametrize("junk", [None, "", "abc", 0])
def test_port_busy_says_no_for_junk_instead_of_crashing(junk):
    """清单里的端口写歪了（或还没解析出来）时不能抛异常——调用方在起服务的路上。"""
    assert pu.port_busy(junk) is False


def test_listening_pid_argv_per_os(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert pu.listening_pid_argv(8765) == ["lsof", "-nP", "-tiTCP:8765", "-sTCP:LISTEN"]
    monkeypatch.setattr(sys, "platform", "win32")
    argv = pu.listening_pid_argv(8766)
    assert argv[:3] == ["powershell", "-NoProfile", "-NonInteractive"]
    assert "Get-NetTCPConnection" in argv[-1] and "8766" in argv[-1]


def test_listening_pid_reads_the_number(monkeypatch):
    class Done:
        returncode, stdout, stderr = 0, "  4242\n", ""

    monkeypatch.setattr(pu.subprocess, "run", lambda *a, **k: Done())
    assert pu.listening_pid(8765) == 4242


def test_listening_pid_none_is_not_me(monkeypatch):
    """查不到就返回 None：调用方不能把"认不出"当成"就是我"。"""
    class Done:
        returncode, stdout, stderr = 1, "", "lsof: command not found"

    monkeypatch.setattr(pu.subprocess, "run", lambda *a, **k: Done())
    assert pu.listening_pid(8765) is None


def test_listening_pid_survives_a_missing_tool(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("没有这个命令")

    monkeypatch.setattr(pu.subprocess, "run", boom)
    assert pu.listening_pid(8765) is None


# --------------------------------------------------------------------------- #
# 打开文件管理器 / 打开浏览器
# --------------------------------------------------------------------------- #
def test_reveal_argv_windows_uses_the_select_switch(monkeypatch, tmp_path):
    """``explorer`` 只认 ``/select,路径`` 这种连笔写法，分开传就只会打开"我的电脑"。"""
    monkeypatch.setattr(sys, "platform", "win32")
    target = tmp_path / "导出结果"
    argv = pu.reveal_argv(str(target))
    assert argv[0] == "explorer"
    assert argv[1].startswith("/select,") and argv[1].endswith("导出结果")


def test_reveal_argv_mac_uses_open_dash_r(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    target = tmp_path / "导出结果"
    assert pu.reveal_argv(str(target)) == ["open", "-R", str(target)]


@pytest.mark.parametrize("value,head", [("darwin", ["open"]), ("win32", ["cmd", "/c", "start"]),
                                        ("linux", ["xdg-open"])])
def test_open_argv_per_os(monkeypatch, value, head):
    monkeypatch.setattr(sys, "platform", value)
    assert pu.open_argv("https://localhost:8765/")[:len(head)] == head


# --------------------------------------------------------------------------- #
# macOS 的"受保护文件夹"：后台任务读不到，而且不弹窗问
# --------------------------------------------------------------------------- #
def test_protected_parent_flags_the_three_guarded_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    inside = tmp_path / "Desktop" / "项目任务" / "自动标注"
    inside.mkdir(parents=True)
    assert pu.protected_parent(inside) == "Desktop"
    assert pu.protected_parent(tmp_path / "work" / "自动标注") is None


def test_protected_parent_is_a_noop_off_mac(monkeypatch, tmp_path):
    """只有 macOS 有这道限制。别的系统要是也报"受保护"，用户会被指着改一个不存在的问题。"""
    monkeypatch.setattr(sys, "platform", "win32")
    assert pu.protected_parent(Path(os.path.expanduser("~")) / "Desktop" / "x") is None


def test_tcc_hint_gives_the_proven_way_first(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    inside = tmp_path / "Desktop" / "p"
    inside.mkdir(parents=True)
    hint = "\n".join(pu.tcc_hint(inside))
    # 出路必须排在原因后面，而且第一条要是实测可靠的那条（挪目录），
    # 不是"去系统设置里授权"——给「终端」授权对 launchd 直起的任务没用
    assert hint.index("①") < hint.index("②")
    assert "挪出" in hint and "完全磁盘访问权限" in hint
    assert pu.tcc_hint(tmp_path / "work") == []


# --------------------------------------------------------------------------- #
# 找工具：uv 与 openssl
# --------------------------------------------------------------------------- #
class _Done:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_find_uv_accepts_a_candidate_path(monkeypatch, tmp_path):
    fake = tmp_path / "uv"
    monkeypatch.setattr(pu.shutil, "which", lambda _n: None)
    monkeypatch.setattr(pu, "UV_CANDIDATES", (str(fake),))
    assert pu.find_uv() is None
    fake.write_text("")
    assert pu.find_uv() == fake


def test_find_uv_uses_path_first(monkeypatch):
    monkeypatch.setattr(pu.shutil, "which", lambda _n: "/opt/weird/uv")
    assert pu.find_uv() == Path("/opt/weird/uv")


def test_find_openssl_knows_git_for_windows(monkeypatch, tmp_path):
    """Windows 上 openssl 一般不单独装，而是跟着 Git for Windows 一起来。
    写死 ``openssl`` 会"明明装了却说找不到"，这是最容易白折腾半小时的一种坑。"""
    fake = tmp_path / "openssl.exe"
    fake.write_text("")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(pu.shutil, "which", lambda _n: None)
    monkeypatch.setattr(pu, "WINDOWS_OPENSSL_CANDIDATES", (str(tmp_path / "没这个.exe"),
                                                           str(fake)))
    assert pu.find_openssl() == fake


def test_find_openssl_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(pu.shutil, "which", lambda _n: None)
    monkeypatch.setattr(pu, "OPENSSL_CANDIDATES", ("/nonexistent/openssl",))
    monkeypatch.setattr(pu, "WINDOWS_OPENSSL_CANDIDATES", ("C:\\nonexistent.exe",))
    assert pu.find_openssl() is None


def test_openssl_hint_offers_http_fallback(monkeypatch):
    """装不上 openssl 也得能干活：HTTP 只是摄像头不能用，看图标注照常。"""
    monkeypatch.setattr(sys, "platform", "win32")
    assert "--no-ssl" in pu.openssl_missing_hint()
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "macOS" in pu.openssl_missing_hint()


def test_uv_argv_prefix():
    argv = pu.uv_argv("/bin/uv", ["numpy", "pillow"])
    assert argv[:2] == ["/bin/uv", "run"]
    assert argv.count("--with") == 2 and "numpy" in argv
    assert argv[-2:] == ["--python", "3.11"] or argv[-1] == "python3"
    assert pu.uv_argv("/bin/uv", [], python="")[-1] == "python3"


# --------------------------------------------------------------------------- #
# 输出编码：Windows 特有的"第一行日志就崩"
# --------------------------------------------------------------------------- #
def test_force_utf8_streams_fixes_a_narrow_console(monkeypatch):
    """重定向进文件时用的是本地代码页，中文机器的 cp936 里没有 ✓ ✗ ▸。"""
    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    pu.force_utf8_streams()
    assert stream.encoding.replace("-", "").lower() == "utf8"
    stream.write("✓ 服务已就绪")          # 以前这里 UnicodeEncodeError
    stream.flush()
    assert "✓".encode("utf-8") in stream.buffer.getvalue()


def test_force_utf8_streams_leaves_utf8_alone(monkeypatch):
    class Fine:
        encoding = "UTF-8"

        def reconfigure(self, **_kw):
            raise AssertionError("已经是 UTF-8 就别再动它")

    monkeypatch.setattr(sys, "stdout", Fine())
    pu.force_utf8_streams()


def test_force_utf8_streams_tolerates_odd_streams(monkeypatch):
    """被重定向成日志文件、被测试框架换掉、被别的库包装过——都不能因为这段兜底把启动搞崩。"""
    monkeypatch.setattr(sys, "stdout", object())
    monkeypatch.setattr(sys, "stderr", None)
    pu.force_utf8_streams()
