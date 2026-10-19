"""常驻服务这条链路（autolabel/service.py + 四个薄壳启动器）。

macOS 用 launchd，Windows 用任务计划程序，机制不同但要对用户交代的事一样，所以
注册 / 状态 / 重启 / 注销只写一份，系统差异全压在两个后端类里。钉住的老坑：

1. **任务定义里不能出现中文路径**——launchd 会把它传成乱码（``can't open input
   file: /Users/…/项…注/…``，退出码 127），任务每 10 秒重来一次。所以定义里只写
   ``~/.autolabel/services/<名字>`` 这个纯 ASCII 的启动片，由它再交给项目；
2. **启动片必须真读一次项目里的文件**——被「桌面文件夹」权限挡住时 ``-r`` / ``-e``
   照样说"可读"，只有真读才知道；说清原因再 ``exit 0``，服务管理器才不会每 10 秒
   空转重试刷日志（它只认非零退出）；
3. **端口上有人在听 ≠ 我这个实例起来了**——多半是用户自己双击开的工作台，报成
   就绪的话窗口一关服务就没了，屏幕上还写着"运行中"；
4. **日志不进临时目录**——Windows 的存储感知会清 ``%TEMP%``，崩溃现场跟着没。

原先这些逻辑写在 ``后台服务.command`` / ``后台实例.command`` 两份 zsh 里，测试只能
隔着 zsh 敲；现在直接测 Python，Windows 那条分支在 mac 上也能验（命令都是纯函数）。
"""
import os
import plistlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from autolabel import platform_utils, service

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH_CMD = os.path.join(ROOT, "运行工作台.command")
SERVICE_CMD = os.path.join(ROOT, "后台服务.command")
LAUNCH_BAT = os.path.join(ROOT, "运行工作台.bat")
SERVICE_BAT = os.path.join(ROOT, "后台服务.bat")
ZSH = "/bin/zsh"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """把家目录换掉：plist / 启动片 / PID 条 / 日志全落在 tmp 里，绝不碰真机。"""
    real_home = tmp_path / "home"
    (real_home / "Desktop" / "自动标注").mkdir(parents=True)
    monkeypatch.delenv("AUTOLABEL_HOME", raising=False)
    monkeypatch.setenv("HOME", str(real_home))
    return real_home


@pytest.fixture()
def conf(home, monkeypatch):
    """临时实例清单：一个跟系统设置走，一个指定 cuda + 半精度。"""
    dataset = home / "Desktop" / "自动标注" / "datasets" / "svc"
    dataset.mkdir(parents=True)
    path = home / "instances.conf"
    path.write_text(
        "# 测试清单\n"
        "\n"
        "mps 8765 %s - no\n"
        "cuda 8766 %s cuda yes\n" % (dataset, dataset),
        encoding="utf-8")
    monkeypatch.setenv(service.CONF_ENV, str(path))
    return path


@pytest.fixture(autouse=True)
def fake_runner(monkeypatch):
    """启动片里的解释器固定成一个假 uv：测试进程自己就跑在 uv 里，
    ``PATH`` 上不一定还有 uv，不钉住的话生成的启动片内容会飘。"""
    prefix = ["/bin/uv", "run", "--offline", "--quiet", "--with", "numpy",
              "--with", "ultralytics", "--python", "3.11", "python3"]
    monkeypatch.setattr(service, "_runner_prefix", lambda: (prefix, "uv"))
    return prefix


def read_conf_lines(path, root=None):
    return service.read_conf(path, root=root)


# --------------------------------------------------------------------------- #
# 薄壳启动器：四个文件都在，且真的只剩壳
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [LAUNCH_CMD, SERVICE_CMD])
def test_command_launchers_ship_and_are_executable(path):
    assert os.path.isfile(path), "双击启动器不能丢：这是新手唯一知道的入口"
    assert os.access(path, os.X_OK), "没有执行位，双击会变成「用文本编辑器打开」"


@pytest.mark.parametrize("path", [LAUNCH_CMD, SERVICE_CMD])
def test_command_launchers_are_valid_zsh(path):
    if not os.path.exists(ZSH):
        pytest.skip("这台机器没有 zsh")
    assert subprocess.run([ZSH, "-n", path]).returncode == 0


@pytest.mark.parametrize("path,keyword", [
    (LAUNCH_CMD, "运行工作台.command"),
    (SERVICE_CMD, "后台服务.command"),
])
def test_command_launchers_help_lists_the_usage(path, keyword):
    out = subprocess.run([ZSH, path, "--help"], capture_output=True, text=True).stdout
    assert keyword in out and "端口" in out


@pytest.mark.parametrize("path,keyword", [
    (LAUNCH_BAT, "运行工作台.bat"),
    (SERVICE_BAT, "后台服务.bat"),
])
def test_bat_launchers_ship_with_the_windows_usage(path, keyword):
    raw = Path(path).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "cmd 不认 UTF-8 BOM，第一行会被吃掉"
    assert b"\r\n" in raw, "cmd 对只有 LF 的批处理不稳，双击可能只跑第一行"
    text = raw.decode("utf-8")
    assert keyword in text and "chcp 65001" in text, "不切 UTF-8 代码页，中文提示全是乱码"
    assert 'cd /d "%~dp0"' in text, "不切到自己所在目录，双击时工作目录是 system32"


@pytest.mark.parametrize("path,command", [(LAUNCH_BAT, "autolabel launch"),
                                          (SERVICE_BAT, "autolabel service"),
                                          (LAUNCH_CMD, "autolabel launch"),
                                          (SERVICE_CMD, "autolabel service")])
def test_launchers_are_thin_shells(path, command):
    """壳里不许留业务逻辑：一条 ``lsof``、一处写死的 ``/tmp`` 日志，都是以前
    mac / Windows 各抄一份之后对不上的地方。"""
    text = Path(path).read_text(encoding="utf-8")
    assert command in text
    verbs = set(re.findall(r"-m autolabel ([A-Za-z-]+)", text))
    assert verbs == {command.split()[1]}, \
        "一个启动器只该调一个子命令：uv 分支和系统 Python 分支写成两个不同的命令，" \
        "装了 uv 的机器就会双击 后台服务 却打开工作台（%s）" % sorted(verbs)
    assert "lsof" not in text, "端口判断已经进 Python（Windows 没有 lsof）"
    assert "/tmp/autolabel" not in text, "日志位置已经统一到 ~/.autolabel/logs"
    assert len(text.splitlines()) < 70, "壳越来越厚就说明逻辑又漏回 shell 了"


def test_command_launcher_translates_the_old_env_vars(tmp_path):
    """``PORT=8766 DEVICE=cuda ./运行工作台.command`` 这类老写法得真的变成参数。

    把 ``uv`` 换成一个「把自己的参数原样打出来」的假脚本：翻译错了当场看见，
    不用真起服务。这一条曾经漏过去过一回——``${NAME}`` 少写 ``(P)`` 把**变量名**
    当值传了出去，每次双击都变成 ``--port PORT --device DEVICE``，参数校验直接
    报错，窗口一闪什么都没起来。
    """
    if not os.path.exists(ZSH):
        pytest.skip("这台机器没有 zsh")
    home = tmp_path / "home"
    (home / "Library" / "Python" / "3.9" / "bin").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text("#!/bin/zsh\nprint -r -- \"$@\"\n", encoding="utf-8")
    fake.chmod(0o755)
    base = dict(os.environ, HOME=str(home), PATH=str(bin_dir),
                AUTOLABEL_HOME=str(home / ".autolabel"))
    for name in ("PORT", "DEVICE", "HALF"):
        base.pop(name, None)

    def run(**extra):
        got = subprocess.run([ZSH, LAUNCH_CMD, "datasets/demo", "--no-browser"],
                             env=dict(base, **extra), capture_output=True, text=True)
        assert got.returncode == 0, got.stderr
        return got.stdout

    plain = run()
    assert "datasets/demo" in plain and "--no-browser" in plain
    assert "--port" not in plain and "--device" not in plain, "没设环境变量就别硬塞参数"
    both = run(PORT="8766", DEVICE="cuda:1")
    assert "--port 8766" in both and "--device cuda:1" in both, both
    assert "--port PORT" not in both and "--device DEVICE" not in both, "把变量名当值传出去了"
    assert "--half" not in run(HALF="no"), "写 no 也算开半精度会让人莫名其妙"
    assert "--half" in run(HALF="yes")


def test_bat_launcher_translates_the_same_env_vars():
    """同一套老写法 Windows 也得认：两边能力不对等，用户只会说「mac 上好使」。

    macOS 之外跑不了 cmd，所以钉的是「三个变量确实被拼成了参数并真的传下去」。
    """
    text = Path(LAUNCH_BAT).read_text(encoding="utf-8")
    for var, flag in (("PORT", "--port"), ("DEVICE", "--device"), ("HALF", "--half")):
        assert f"if defined {var}" in text, f"少了 {var} 的翻译"
        assert flag in text, flag
    assert text.count("launch %EXTRA% %*") == 2, "拼好的参数得真的传下去，两条分支都要"


def test_the_dedicated_runner_script_is_gone():
    """``后台实例.command`` 已经并进 ``autolabel service-run``：留着它就是第三份
    "等端口 + 打日志"的实现，改一处忘两处。"""
    assert not os.path.exists(os.path.join(ROOT, "后台实例.command"))


# --------------------------------------------------------------------------- #
# 实例清单：错在哪一行、怎么改，都要说清楚
# --------------------------------------------------------------------------- #
def test_conf_parses_both_device_styles(home, conf):
    got = read_conf_lines(conf)
    assert [row.name for row in got] == ["mps", "cuda"]
    assert got[0].device == "" and got[0].half is False
    assert got[1].device == "cuda" and got[1].half is True
    assert got[1].port == 8766
    assert got[1].env_lines() == ["PORT=8766", "DATASET=%s" % got[1].dataset,
                                  "DEVICE=cuda", "HALF=yes"]


def test_conf_accepts_a_gpu_index_and_keeps_the_card_number(home, tmp_path):
    """``cuda:1`` 必须能填，而且要留着卡号：收成 ``cuda`` 会让双卡机器上
    两个实例全挤到 0 号卡，第二块卡闲着还都以为自己在用它。"""
    real = tmp_path / "ds"
    real.mkdir()
    path = tmp_path / "gpu.conf"
    path.write_text("one 8770 %s CUDA:1 no\ntwo 8771 %s 1 yes\n" % (real, real),
                    encoding="utf-8")

    got = service.read_conf(path)

    assert [row.device for row in got] == ["cuda:1", "1"], "大小写要认，卡号不能丢"


def test_conf_rejects_an_unknown_device_before_registration(home, tmp_path):
    """填错设备现在就得拦住：这个值会原样交给 ``review --device``，
    后台任务一起来就退出码 2，再被服务管理器每 10 秒拉起一次，日志刷爆。"""
    real = tmp_path / "ds"
    real.mkdir()
    path = tmp_path / "bad-device.conf"
    path.write_text("worker 8765 %s to-the-moon no\n" % real, encoding="utf-8")

    with pytest.raises(service.ServiceError) as caught:
        service.read_conf(path)

    assert "设备不认识" in str(caught.value)
    advice = " ".join(caught.value.details)
    assert "cuda:1" in advice and "mps" in advice, "报错要把能填的档位全列出来"
    assert "退出码 2" in advice, "还要说清不拦会怎么炸，用户才知道这一步不是可改可不改"


@pytest.mark.parametrize("line,keyword,at", [
    ("中文 8765 {ds} - no", "实例名不合规", 1),
    ("dup 8765 {ds} - no\ndup 8766 {ds} - no", "实例名重复", 2),
    ("bad notaport {ds} - no", "端口不是数字", 1),
    ("miss 8765 不存在的目录 - no", "数据集不存在", 1),
])
def test_conf_rejects_each_bad_line_with_its_own_message(home, tmp_path, line, keyword, at):
    real = tmp_path / "ds"
    real.mkdir()
    path = tmp_path / "bad.conf"
    # 除了"数据集不存在"那条，其余都要用真目录：不然四条全撞在同一个报错上
    path.write_text(line.format(ds=real) + "\n", encoding="utf-8")
    with pytest.raises(service.ServiceError) as caught:
        service.read_conf(path)
    assert keyword in caught.value.head
    # 重复名是"这一行和第 1 行撞了"，行号要指出第 2 行，指第 1 行会让人改错一行
    assert "第 %d 行" % at in caught.value.head, "不说行号，用户得自己数"


def test_conf_relative_dataset_is_resolved_against_the_project(home, tmp_path):
    dataset = tmp_path / "datasets" / "rel"
    dataset.mkdir(parents=True)
    path = tmp_path / "rel.conf"
    path.write_text("rel 8770 datasets/rel - no\n", encoding="utf-8")
    row = service.read_conf(path, root=tmp_path)[0]
    assert Path(row.dataset).is_absolute() and Path(row.dataset).is_dir()


@pytest.mark.parametrize("line,dataset,device,half", [
    ("mps 8765 datasets/demo - no", "datasets/demo", "", False),
    ("mps 8765 datasets/demo - yes", "datasets/demo", "", True),
    ("cuda 8766 datasets/demo cuda no", "datasets/demo", "cuda", False),
    ("cuda 8766 datasets/素材 批次1 cuda yes", "datasets/素材 批次1", "cuda", True),
    ("cpu 8767 datasets/素材 批次1", "datasets/素材 批次1", "", False),
    ("mps 8765 datasets/cuda", "datasets/cuda", "", False),
    ("mps 8765 datasets/demo cuda", "datasets/demo", "cuda", False),
    ("mps 8765 datasets/yes - no", "datasets/yes", "", False),
])
def test_conf_dataset_path_may_contain_spaces(home, tmp_path, line, dataset, device, half):
    """目录名带空格是常态（``项目 任务/素材 批次1``）：整份清单不该因此读不动。

    以前一律按空格切五段，``datasets/素材 批次1`` 被切成「数据集 = datasets/素材、
    设备 = 批次1」，报的是「设备不认识：批次1」，看着像填错了算力。
    认法改成：实例名 / 端口 / 设备 / 半精度四段天然不含空格，剩下的整段归数据集。
    """
    root = tmp_path / "项目 任务"
    (root / dataset).mkdir(parents=True)
    path = tmp_path / "带空格.conf"
    path.write_text(line + "\n", encoding="utf-8")

    row = service.read_conf(path, root=root)[0]
    assert row.dataset == str(root / dataset)
    assert row.device == device and row.half is half


def test_conf_device_error_explains_the_spaced_path_trap(home, tmp_path):
    """五段又带空格的写法认不出来时，报错要直接点破这一条，而不是只列可选算力。"""
    root = tmp_path / "proj"
    (root / "datasets" / "素材").mkdir(parents=True)
    path = tmp_path / "trap.conf"
    path.write_text("mps 8765 datasets/素材 批次1 -\n", encoding="utf-8")

    with pytest.raises(service.ServiceError) as caught:
        service.read_conf(path, root=root)

    assert "设备不认识" in caught.value.head
    assert "数据集目录名里有空格" in "\n".join(caught.value.details)


def test_conf_of_only_comments_says_so(home, tmp_path):
    path = tmp_path / "empty.conf"
    path.write_text("# 注释\n\n   # 又是注释\n", encoding="utf-8")
    with pytest.raises(service.ServiceError) as caught:
        service.read_conf(path)
    assert "一个实例都没有" in caught.value.head


def test_missing_conf_shows_the_format(home, tmp_path):
    with pytest.raises(service.ServiceError) as caught:
        service.read_conf(tmp_path / "没有.conf")
    assert "mps 8765 datasets/demo - no" in " ".join(caught.value.details)


def test_the_shipped_example_conf_actually_parses(tmp_path):
    """模板是给「新克隆的机器第一次注册」用的，写坏了就是第一脚踩空——当真清单解析一遍。

    清单本身（端口、数据集、用哪块卡）是本机偏好、不进仓库，所以仓库里能被用户
    照着抄的只有这份 ``.example``。
    """
    example = os.path.join(ROOT, "configs", "后台实例.conf.example")
    assert os.path.isfile(example), "模板要进仓库，不然新机器只能猜清单长什么样"
    root = tmp_path / "proj"
    (root / "datasets" / "demo").mkdir(parents=True)
    rows = service.read_conf(example, root=root)
    assert [row.name for row in rows] == ["mps"], "模板只放开一个每台机器都有的实例，其余留注释"
    assert rows[0].port == 8765 and rows[0].device == "" and rows[0].half is False


def test_missing_conf_points_at_the_template_when_it_is_there(tmp_path):
    """有模板就指路，没模板别瞎指——指一个不存在的路径比不提还误事。"""
    (tmp_path / "后台实例.conf.example").write_text("mps 8765 datasets/demo - no\n",
                                                    encoding="utf-8")
    with pytest.raises(service.ServiceError) as caught:
        service.read_conf(tmp_path / "后台实例.conf")
    assert "后台实例.conf.example" in " ".join(caught.value.details)

    with pytest.raises(service.ServiceError) as other:
        service.read_conf(tmp_path / "另一份.conf")
    assert "example" not in " ".join(other.value.details)


def test_instance_for_names_the_conf_that_needs_a_rerun(home, conf, monkeypatch):
    monkeypatch.setenv(service.CONF_ENV, str(conf))
    assert service.instance_for(home, "mps").port == 8765
    with pytest.raises(service.ServiceError) as caught:
        service.instance_for(home, "没有这个实例")
    assert "重跑" in " ".join(caught.value.details)


# --------------------------------------------------------------------------- #
# macOS 后端：plist 是 launchd 要读的，格式与"只有 ASCII 路径"是硬要求
# --------------------------------------------------------------------------- #
def test_launchd_label_follows_the_developer_name(home):
    backend = service.backend("launchd")
    from autolabel.version import DEVELOPER

    assert backend.label("mps") == "com.%s.autolabel.mps" % DEVELOPER.strip().lower()
    assert backend.label("mps").isascii()


def test_plist_is_what_launchd_will_read(home, conf):
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    data = plistlib.loads(backend.render_unit(inst).encode("utf-8"))
    assert data["Label"] == backend.label("mps")
    assert data["ProgramArguments"][0] == "/bin/zsh"
    assert data["ProgramArguments"][1].endswith("services/mps.sh")
    assert data["EnvironmentVariables"]["HOME"] == str(home)
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] == 10


def test_plist_paths_stay_ascii(home, conf):
    """实例放在中文目录里是常态：定义里出现中文就是每 10 秒崩一次。"""
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    data = plistlib.loads(backend.render_unit(inst).encode("utf-8"))
    for key in ("ProgramArguments",):
        for item in data[key]:
            assert item.isascii(), item
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert data[key].isascii(), data[key]


def test_plist_escapes_the_xml_metacharacters(home, conf, monkeypatch):
    """家目录里带 ``&`` 或空格都不稀奇，plist 不转义就直接解析失败。"""
    monkeypatch.setattr(service, "services_dir",
                        lambda: home / "a&b c" / "services")
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    body = backend.render_unit(inst)
    assert "a&amp;b c" in body, "路径里有 & < > 时 plist 会解析失败"
    data = plistlib.loads(body.encode("utf-8"))
    assert "a&b c" in data["ProgramArguments"][1]


def test_plist_logs_go_to_the_persistent_folder(home, conf):
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    data = plistlib.loads(backend.render_unit(inst).encode("utf-8"))
    assert data["StandardOutPath"].startswith(str(home / ".autolabel" / "logs"))


def test_shim_really_reads_a_project_file(home, conf):
    """被「桌面文件夹」权限挡住时 ``-r`` 会说谎，只有真读一次才知道。"""
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    shim = backend.render_shim(inst, home / "Desktop" / "自动标注")
    assert 'head -1 "$PROJECT/autolabel/version.py"' in shim
    assert "exit 0" in shim.split("fi")[0], "读不到就得先退出 0，别让 launchd 空转重试"
    assert "挪出" in shim, "要把该点哪里写出来：只说「读不到」用户没法动手"


def test_shim_hands_off_to_the_python_entry_point(home, conf):
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[1]
    shim = backend.render_shim(inst, home / "Desktop" / "自动标注")
    assert "-m autolabel service-run 'cuda'" in shim
    assert "ultralytics" in shim, "后台实例要带推理框架，否则标注一跑就报缺依赖"
    assert "--offline" in shim


@pytest.mark.parametrize("folder", ["带 空格和括号(的)目录", "带'单引号'的目录",
                                   "带$美元和`反引号`的目录", "plain dir"])
def test_shim_quotes_everything(home, conf, tmp_path, folder):
    """项目路径里有空格、括号、单引号都很常见：省引号省下几个字节，赔进去一台起不来。

    只钉一件事——把生成的那一行交给 zsh 自己解析，解析回来的必须和原路径一字不差。
    """
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    messy = tmp_path / folder
    messy.mkdir()
    shim = backend.render_shim(inst, messy)
    assignment = [row for row in shim.splitlines() if row.startswith("PROJECT=")]
    assert len(assignment) == 1
    assert assignment[0].startswith("PROJECT='"), "一律用单引号包住，别挑「看着没特殊字符就不加」"
    if not os.path.exists(ZSH):
        pytest.skip("这台机器没有 zsh")
    probe = subprocess.run([ZSH, "-c", "%s; print -r -- $PROJECT" % assignment[0]],
                           capture_output=True, text=True)
    assert probe.stdout.strip() == str(messy), probe.stderr


def test_write_unit_lands_ascii_shim_with_exec_bit(home, conf):
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    unit = service.write_unit(backend, inst, home / "Desktop" / "自动标注")
    shim = backend.shim_path(inst)
    assert unit.is_file() and shim.is_file()
    assert shim.stat().st_mode & 0o111, "启动片没有执行位，launchd 直接拒绝"
    if os.path.exists(ZSH):
        assert subprocess.run([ZSH, "-n", str(shim)]).returncode == 0
    assert service.pid_file("mps").parent == shim.parent


def test_write_unit_refuses_a_non_ascii_home(home, conf):
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    monkey_home = home / "中文家"
    monkey_home.mkdir()
    original = service.services_dir
    service.services_dir = lambda: monkey_home / "服务"
    try:
        with pytest.raises(service.ServiceError) as caught:
            service.write_unit(backend, inst, home)
        assert "非 ASCII" in caught.value.head
    finally:
        service.services_dir = original


def test_dry_run_writes_files_without_touching_the_service_manager(home, conf, monkeypatch):
    called = []
    monkeypatch.setattr(service, "run",
                        lambda argv, **kw: called.append(list(argv)) or _done())
    lines = []
    assert service.install(home / "Desktop" / "自动标注", dry_run=True,
                           echo=lines.append) == 0
    assert called == [], "dry-run 不该碰 launchctl"
    assert (home / "Library" / "LaunchAgents" / "com.windinchen.autolabel.mps.plist").is_file()
    assert (home / ".autolabel" / "services" / "mps.sh").is_file()
    assert any("--dry-run" in line for line in lines)


def test_install_bootouts_the_old_job_before_re_registering(home, conf, monkeypatch):
    """第二次点「注册」时旧任务还挂在 launchd 里：不先摘掉，bootstrap 会被拒。"""
    calls = []
    probe = _launchctl(loaded=("mps", "cuda"), pids={"mps": 4242, "cuda": 4243})

    def fake_run(argv, **kw):
        calls.append([str(part) for part in argv])
        return probe(argv)

    monkeypatch.setattr(service, "run", fake_run)
    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: False)
    monkeypatch.setattr(service, "_verify_ready", lambda *_a, **_k: "ok")
    service.install(home / "Desktop" / "自动标注", echo=lambda *_: None)
    verbs = [call[1] for call in calls if call[0] == "launchctl"]
    assert "bootout" in verbs and "bootstrap" in verbs
    assert verbs.index("bootout") < verbs.index("bootstrap"), "顺序反了就成了自己摘自己"


def test_install_refuses_to_spawn_behind_someone_elses_port(home, conf, monkeypatch):
    """端口被自己开着的工作台占着：跳过这个实例并说清两条出路，别硬抢。"""
    monkeypatch.setattr(service, "run", lambda argv, **kw: _done())
    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: True)
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: 777)
    lines = []
    service.install(home / "Desktop" / "自动标注", echo=lines.append)
    text = "\n".join(lines)
    assert "还被别的程序占着" in text and "--restart" in text
    assert not any(call[1] == "bootstrap" for call in [] if len(call) > 1)


def _done(returncode=0, stdout=""):
    class Done:
        pass

    got = Done()
    got.returncode, got.stdout, got.stderr = returncode, stdout, ""
    return got


def _launchctl(loaded=(), pids=None):
    """按 ``launchctl`` 的真实脾气造返回值。

    ``launchctl list <标签>`` 只在任务**已加载**时返回 0 并打出 ``"PID" = …``；
    没加载时返回 1、输出为空。替身要是不分这两种情况，「未注册 / 定义了没加载 /
    加载了没在跑」三态在测试里就是糊的，改实现的人也看不出自己改坏了。
    """
    pids = pids or {}

    def run(argv, **_kw):
        argv = [str(part) for part in argv]
        if argv[:2] == ["launchctl", "list"] and len(argv) > 2:
            label = argv[2]
            name = label.rsplit(".", 1)[-1]
            if name not in loaded:
                return _done(returncode=1)
            return _done(stdout=chr(34) + "PID" + chr(34) + " = %s;" % (pids.get(name) or "-")
                         + "\n" + chr(34) + "Label" + chr(34) + " = " + label + ";")
        return _done()

    return run


# --------------------------------------------------------------------------- #
# 就绪判定：三种结果必须分得开
# --------------------------------------------------------------------------- #
@pytest.fixture()
def registered(home, conf, monkeypatch):
    monkeypatch.setattr(service, "run", lambda argv, **kw: _done(stdout=""))
    inst = read_conf_lines(conf)[0]
    service.write_unit(service.backend("launchd"), inst, home / "Desktop" / "自动标注")
    return inst


def test_ready_requires_the_listener_to_be_ours(registered, monkeypatch):
    backend = service.backend("launchd")
    monkeypatch.setattr(service, "wait_for_port_for_test", True, raising=False)
    monkeypatch.setattr("autolabel.launch.wait_for_port", lambda *_a, **_k: True)
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: 4321)
    service.pid_file("mps").write_text("4321", encoding="ascii")
    lines = []
    assert service._verify_ready(backend, registered, echo=lines.append) == "ok"


def test_a_foreign_listener_is_reported_as_foreign(registered, monkeypatch):
    """端口上有人在听，但那是用户自己双击开的工作台——窗口一关就断。"""
    backend = service.backend("launchd")
    monkeypatch.setattr("autolabel.launch.wait_for_port", lambda *_a, **_k: True)
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: 9999)
    service.pid_file("mps").write_text("4321", encoding="ascii")
    lines = []
    assert service._verify_ready(backend, registered, echo=lines.append) == "foreign"
    assert any("窗口一关服务就断" in line for line in lines)


def test_no_listener_at_all_is_down(registered, monkeypatch):
    backend = service.backend("launchd")
    monkeypatch.setattr("autolabel.launch.wait_for_port", lambda *_a, **_k: False)
    assert service._verify_ready(backend, registered, echo=lambda *_: None) == "down"


# --------------------------------------------------------------------------- #
# 状态页：未注册 / 定义写好但没加载 / 在跑，三态要分得清
# --------------------------------------------------------------------------- #
def _status_lines(home, conf, monkeypatch, loaded=(), pids=None, listener=None):
    """把 launchd 的返回值换成可控的，再抓一遍 ``autolabel service --status`` 的输出。"""
    monkeypatch.setattr(service, "run", _launchctl(loaded=loaded, pids=pids))
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: listener)
    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: True)
    lines = []
    service.status(home / "Desktop" / "自动标注", echo=lines.append)
    return "\n".join(lines)


def test_status_says_unregistered_plainly(home, conf, monkeypatch):
    text = _status_lines(home, conf, monkeypatch)
    # 指路要指到人真敲得动的命令：仓库没装 console script，提示里写
    # ``autolabel service`` 照抄只会得到 command not found
    assert "未注册" in text and "后台服务" in text, text


def test_status_tells_a_written_but_unloaded_job_apart(home, conf, monkeypatch):
    """plist 已经写好、只是 launchd 还没加载（下次登录会自己起来）——这跟"没注册"
    是两件事，混在一起说用户会反复重跑注册。"""
    inst = read_conf_lines(conf)[0]
    service.write_unit(service.backend("launchd"), inst, home / "Desktop" / "自动标注")
    text = _status_lines(home, conf, monkeypatch)
    assert "任务定义已写好" in text
    assert "--restart mps" in text


def test_status_reports_a_running_job_with_its_pid(home, conf, monkeypatch):
    """同一屏要能分出「这个在跑、那个没在跑」，只看一个全局状态是不够的。"""
    backend = service.backend("launchd")
    for every in read_conf_lines(conf):
        service.write_unit(backend, every, home / "Desktop" / "自动标注")
    # 两个都已被 launchd 加载，只有 mps 有 PID：cuda 属于「注册了但这会儿没进程」
    monkeypatch.setattr(service, "run",
                        _launchctl(loaded=("mps", "cuda"), pids={"mps": 4321}))
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: None)
    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: True)
    service.pid_file("mps").write_text("4321", encoding="ascii")
    lines = []
    service.status(home / "Desktop" / "自动标注", echo=lines.append)
    text = "\n".join(lines)
    assert "mps 在跑 · PID 4321" in text
    assert "cuda 已注册但没在跑" in text


def test_status_catches_a_process_that_never_started_listening(home, conf, monkeypatch):
    """launchd 说「在跑」、端口却没人听：进程卡在加载或起的根本不是这个实例。

    以前状态页只转述服务管理器的话，照样打「✓ 在跑 · http://…」，
    用户照着地址点开却是「无法访问」，而屏幕上明明写着运行中。
    """
    monkeypatch.setattr(service, "run", _launchctl(loaded=("mps", "cuda"), pids={"mps": 4321}))
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: None)
    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: False)
    service.pid_file("mps").parent.mkdir(parents=True, exist_ok=True)
    service.pid_file("mps").write_text("4321", encoding="ascii")
    lines = []
    code = service.status(home / "Desktop" / "自动标注", echo=lines.append)
    text = "\n".join(lines)

    assert "进程在跑（PID 4321），但端口 8765 没人监听" in text, text
    assert "mps 在跑 ·" not in text, "端口没听就不许打这一句"
    assert "模型还在加载" in text, "要给一句「稍后再看一次」，别只报坏消息"
    assert code == 1, "注册了却没在服务，退出码不能还是 0"


def test_port_busy_instances_are_not_counted_twice(home, conf, monkeypatch):
    """端口被占而跳过的实例根本没加载，第二遍别再拿去验一次、数两次。"""
    monkeypatch.setattr(service, "run", lambda argv, **kw: _done())
    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: True)
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: 777)
    checked = []
    monkeypatch.setattr(service, "_verify_ready",
                        lambda _b, inst, **_k: checked.append(inst.name) or "foreign")
    lines = []
    assert service.install(home / "Desktop" / "自动标注", echo=lines.append) == 0
    text = "\n".join(lines)

    assert checked == [], "跳过的实例不该再被验一遍"
    assert "还被别的程序占着" in text
    assert "另外 2 个因为端口被占跳过了" in text, text
    assert text.count("端口被占跳过") == 1


def test_ready_wait_matches_the_launcher():
    """注册那条路等 40 秒、实例那条路等 90 秒：同一台冷启动机器会一边报失败一边说正常。"""
    from autolabel import launch

    assert service.READY_WAIT == 90.0
    assert service.READY_WAIT == launch.wait_for_port.__defaults__[0]


def test_plist_carries_autolabel_home_when_it_is_set(home, conf, monkeypatch):
    """注册时把服务用的目录另指过，任务定义就得带上，否则后台读的是默认 ~/.autolabel。"""
    backend = service.backend("launchd")
    inst = read_conf_lines(conf)[0]
    assert "AUTOLABEL_HOME" not in backend.render_unit(inst), "没设过就别凭空造一个"

    monkeypatch.setenv("AUTOLABEL_HOME", str(home / "autolabel 数据"))
    data = plistlib.loads(backend.render_unit(inst).encode("utf-8"))
    assert data["EnvironmentVariables"]["AUTOLABEL_HOME"] == str(home / "autolabel 数据")
    assert data["EnvironmentVariables"]["HOME"] == str(home)


def test_windows_shim_carries_autolabel_home_too(home, conf, monkeypatch):
    """任务计划程序不继承用户的环境变量：不设这一行，后台和前台就是两份配置。"""
    backend = service.backend("taskscheduler")
    inst = read_conf_lines(conf)[1]
    assert "AUTOLABEL_HOME" not in backend.render_shim(inst, home / "Desktop" / "自动标注")

    monkeypatch.setenv("AUTOLABEL_HOME", "D:\\autolabel")
    shim = backend.render_shim(inst, home / "Desktop" / "自动标注")
    assert 'set "AUTOLABEL_HOME=D:\\autolabel"' in shim


def test_status_explains_a_tcc_failure_in_plain_words(home, conf, monkeypatch):
    """日志里写着"读不到项目里的文件"时，要顺手把出路写出来，别只甩日志行。"""
    log = platform_utils.service_log(8765)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("✗ 后台任务读不到项目里的文件：/Users/x/Desktop/p\n", encoding="utf-8")
    lines = []
    service.explain_down(8765, lines.append, root=home / "Desktop" / "自动标注")
    text = "\n".join(lines)
    assert "读不到项目里的文件" in text and "挪出" in text
    assert "运行工作台" in text, "还要告诉用户前台双击不受这条限制"


# --------------------------------------------------------------------------- #
# 注销与清理：清单里删掉的实例不能留个僵尸任务
# --------------------------------------------------------------------------- #
def test_remove_deletes_unit_shim_and_pid(home, conf, monkeypatch):
    inst = read_conf_lines(conf)[0]
    backend = service.backend("launchd")
    unit = service.write_unit(backend, inst, home / "Desktop" / "自动标注")
    service.pid_file("mps").write_text("1", encoding="ascii")
    monkeypatch.setattr(service, "run", lambda argv, **kw: _done())
    assert service.remove(home / "Desktop" / "自动标注", "mps", echo=lambda *_: None) == 0
    assert not unit.exists() and not backend.shim_path(inst).exists()
    assert not service.pid_file("mps").exists()
    assert backend.unit_path("cuda").parent.exists()


def test_prune_stale_clears_a_deleted_instance(home, conf, monkeypatch):
    """清单里删了一行，任务定义还在的话服务管理器会一直重启一个不存在的实例。"""
    backend = service.backend("launchd")
    for alive in read_conf_lines(conf):
        service.write_unit(backend, alive, home / "Desktop" / "自动标注")
    stale = service.Instance("gone", 8790, "datasets/demo")
    unit = service.write_unit(backend, stale, home)
    service.pid_file("gone").write_text("1", encoding="ascii")
    calls = []
    monkeypatch.setattr(service, "run", lambda argv, **kw: calls.append(list(argv)) or _done())
    lines = []
    service.prune_stale(backend, read_conf_lines(conf), home, echo=lines.append)
    assert not unit.exists() and not service.pid_file("gone").exists()
    assert backend.shim_path(stale).exists() is False
    assert any("gone" in line for line in lines)
    assert calls and calls[0][:2] == ["launchctl", "bootout"]
    assert backend.unit_path("mps").exists(), "还在清单里的实例不能被顺手清掉"


def test_prune_stale_never_touches_someone_elses_jobs(home, conf, monkeypatch):
    """``~/Library/LaunchAgents`` 是全体 App 共用的目录，清僵尸只能认自己的前缀。

    实机这一目录里躺着 Steam、Clash Verge、蒲公英的任务定义：早先的写法对不上
    前缀也照样把整个文件名当实例名，「清掉清单里删掉的实例」于是变成「把这台
    机器上所有不认识的任务逐个 bootout + 删文件」。
    """
    backend = service.backend("launchd")
    agents = backend.agents_dir()
    agents.mkdir(parents=True, exist_ok=True)
    foreign = agents / "com.valvesoftware.steamclean.plist"
    near_miss = agents / "com.windinchen.autolabelx.ghost.plist"
    for extra in (foreign, near_miss):
        extra.write_text("<plist/>", encoding="utf-8")
    calls = []
    monkeypatch.setattr(service, "run", lambda argv, **kw: calls.append(list(argv)) or _done())
    lines = []
    service.prune_stale(backend, read_conf_lines(conf), home, echo=lines.append)
    assert foreign.is_file(), "别人的任务定义被删了"
    assert near_miss.is_file(), "前缀只是「很像」也不算我们的"
    assert calls == [], "别人的任务一个都不该 bootout"
    assert not any("steamclean" in line for line in lines)


@pytest.mark.parametrize("backend_name,filename,wanted", [
    ("launchd", "com.windinchen.autolabel.mps.plist", "mps"),
    ("launchd", "com.windinchen.autolabel.ai.hermes.plist", "ai.hermes"),
    ("launchd", "com.valvesoftware.steamclean.plist", None),
    ("launchd", "ai.hermes.gateway.plist", None),
    ("launchd", "com.windinchen.autolabelx.ghost.plist", None),
    ("launchd", "com.windinchen.autolabel..plist", None),
    ("taskscheduler", "AutoLabel-cuda.task.xml", "cuda"),
    ("taskscheduler", "SteamClean.task.xml", None),
])
def test_unit_name_only_recognises_its_own_prefix(backend_name, filename, wanted):
    backend = service.backend(backend_name)
    assert service._unit_name(backend, Path(filename)) == wanted


# --------------------------------------------------------------------------- #
# Windows 后端：这一半在 mac 上跑，验的是"命令写对了没"
# --------------------------------------------------------------------------- #
def win(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    return service.backend("taskscheduler")


def test_windows_task_name_is_readably_english(monkeypatch, home):
    backend = win(monkeypatch)
    assert backend.label("mps") == "AutoLabel-mps"
    assert backend.unit_path("mps").parent == service.services_dir()


def test_windows_query_uses_powershell_not_schtasks(monkeypatch, home):
    """``schtasks`` 连**值**都跟着系统语言翻译（中文系统是「状态: 正在运行」），
    换台机器就对不上；``Get-ScheduledTask.State`` 永远返回英文枚举。"""
    backend = win(monkeypatch)
    argv = backend.query_argv("mps")
    assert argv[0] == "powershell" and "Get-ScheduledTask" in argv[-1]
    assert "AutoLabel-mps" in argv[-1]


def test_windows_restart_needs_no_cmd_shell(monkeypatch, home):
    """``&`` 是 **cmd 的** 命令分隔符：不带 shell 把它交给 ``schtasks``，它只当
    是多出来的参数报「无效参数」，Windows 上每次「重启」都会失败。"""
    argv = win(monkeypatch).restart_argv("mps")
    assert "&" not in argv, "argv 里没有 shell，& 只是个普通参数，不会被解析成分隔符"
    assert argv[0] == "powershell" and "/End" not in " ".join(argv)
    script = argv[-1]
    assert "Stop-ScheduledTask" in script and "Start-ScheduledTask" in script
    assert "'AutoLabel-mps'" in script


def test_windows_restart_waits_for_the_old_instance_to_leave(monkeypatch, home):
    """任务是 ``IgnoreNew``：旧实例还没退干净就下发起动，等于什么都没发生，
    但两条命令都返回成功——屏幕上会写「已重启」。"""
    script = win(monkeypatch).restart_argv("mps")[-1]
    stop, start = script.index("Stop-ScheduledTask"), script.index("Start-ScheduledTask")
    assert stop < start
    assert "State -ne 'Running'" in script and "Start-Sleep" in script, "先等旧的退了再启动"


def test_prune_stale_clears_the_windows_junction_too(monkeypatch, home, conf):
    """Windows 的实例还留一个指向项目的目录联接，清僵尸时得连它一起清掉。"""
    backend = win(monkeypatch)
    services = service.services_dir()
    services.mkdir(parents=True, exist_ok=True)
    ghost = services / "gone.work"
    os.symlink(home / "Desktop" / "自动标注", ghost)
    unit = backend.unit_path("gone")
    unit.write_text("task=AutoLabel-gone\n", encoding="utf-8")
    monkeypatch.setattr(service, "run", lambda argv, **kw: _done())
    service.prune_stale(backend, read_conf_lines(conf), home, echo=lambda *_: None)
    assert not unit.exists()
    assert not ghost.is_symlink(), "联接没清：下次同名实例会被指到别处"


@pytest.mark.parametrize("state,loaded,running", [
    ("Running", True, True), ("Ready", True, False), ("Disabled", True, False),
    ("Queued", True, False), ("", False, False),
])
def test_windows_state_parsing(monkeypatch, home, state, loaded, running):
    got_loaded, got_running, pid = win(monkeypatch).parse_state(state + "\n")
    assert (got_loaded, got_running, pid) == (loaded, running, None)


def test_windows_register_script_mirrors_keep_alive(monkeypatch, home, conf):
    """launchd 的 ``KeepAlive={SuccessfulExit:false}`` 在 Windows 上是
    ``-RestartCount`` + ``-RestartInterval``：非零退出才重启。"""
    backend = win(monkeypatch)
    script = service.register_script(read_conf_lines(conf)[0], backend)
    assert "-RestartCount 999" in script and "-RestartInterval" in script
    assert "-AtLogOn" in script, "登录触发对应 launchd 的 RunAtLoad"
    assert "-MultipleInstances IgnoreNew" in script, "重复触发不许起第二个实例抢端口"
    assert "-TaskName 'AutoLabel-mps'" in script
    assert "-ExecutionTimeLimit ([TimeSpan]::Zero)" in script, "不然跑满 3 天被系统杀掉"


@pytest.fixture()
def linked(home, conf):
    """补一个指向项目的目录联接（mac 上用符号链接代替，效果一样：能解析到项目）。"""
    backend = service.TaskSchedulerBackend()
    inst = read_conf_lines(conf)[0]
    project = home / "Desktop" / "自动标注"
    link = backend.work_link(inst)
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(project, link)
    return backend, inst, project


def test_windows_shim_stays_off_the_chinese_path(linked):
    """.cmd 按 OEM 代码页解析，中文路径写进去容易变乱码；目录联接让 ``%~dp0``
    拼出来的全 ASCII，中文只留在文件系统里。"""
    backend, inst, project = linked
    shim = backend.render_shim(inst, project)
    assert "\r\n" in shim, "cmd 对只有 LF 的批处理不稳"
    assert 'cd /d "%~dp0mps.work"' in shim, "联接在位时就不该把中文路径写进 .cmd"
    assert "autolabel service-run mps" in shim
    assert "@echo off" in shim and "exit /b %ERRORLEVEL%" in shim


def test_windows_shim_falls_back_to_the_real_path(monkeypatch, home, conf):
    """联接没建成（磁盘不支持、被安全软件挡了）就直接 cd 过去，
    能不能成由 ``write_unit`` 的 ``prepare`` 决定，壳自己先别拒。"""
    backend = win(monkeypatch)
    inst = read_conf_lines(conf)[0]
    project = home / "Desktop" / "自动标注"
    shim = backend.render_shim(inst, project)      # 联接不在 → 直接 cd 到项目
    assert 'cd /d "%s"' % project in shim


def test_windows_junction_command(monkeypatch, tmp_path):
    assert service.junction_argv(tmp_path / "mps.work", tmp_path / "项目") == [
        "cmd", "/c", "mklink", "/J", str(tmp_path / "mps.work"), str(tmp_path / "项目")]


def test_cmd_shim_is_written_in_the_console_codepage(monkeypatch, home):
    """``.cmd`` 要按 OEM 代码页落盘：写 UTF-8 的话中文路径在 cmd 里是乱码。"""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert service._shim_encoding() == "utf-8"


def test_windows_backend_is_only_picked_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert service.backend() is not None and service.backend().name == "launchd"
    monkeypatch.setattr(sys, "platform", "win32")
    assert service.backend().name == "taskscheduler"
    monkeypatch.setattr(sys, "platform", "linux")
    assert service.backend() is None


def test_every_backend_exposes_the_same_verbs():
    for name in ("launchd", "taskscheduler"):
        backend = service.backend(name)
        inst = service.Instance("mps", 8765, "datasets/demo")
        assert backend.render_unit(inst) and backend.render_shim(inst, "/tmp/proj")
        assert backend.register_argv(inst) and backend.unregister_argv("mps")
        assert backend.restart_argv("mps") and backend.query_argv("mps")
        assert backend.parse_state("") is not None


# --------------------------------------------------------------------------- #
# 服务入口自己：端口被占必须退出码 0
# --------------------------------------------------------------------------- #
def test_service_run_guard_is_a_clean_zero(monkeypatch, project_root):
    """非零退出 = "失败"，服务管理器会一直重启；端口被占不是失败，是让路。"""
    from autolabel import launch as launch_module

    monkeypatch.setattr(platform_utils, "port_busy", lambda *_a, **_k: True)
    monkeypatch.setattr(platform_utils, "listening_pid", lambda _p: 1234)
    called = []
    monkeypatch.setattr(launch_module.subprocess, "Popen",
                        lambda *a, **k: called.append(a))
    assert launch_module.launch(project_root, "datasets/demo", echo=lambda *_: None) == 0
    assert called == []


@pytest.fixture()
def project_root(tmp_path, monkeypatch):
    """一个像样的项目目录：有 autolabel 包，也有 datasets/demo。"""
    root = tmp_path / "proj"
    (root / "autolabel").mkdir(parents=True)
    (root / "datasets" / "demo").mkdir(parents=True)
    from autolabel import launch as launch_module

    # 签证书这步在测试里只要「别挡路」。必须用 monkeypatch：直接写模块属性
    # 不会还原，后面 test_launch.py 拿到的是这个假 ensure_cert，怎么断言都是 True
    monkeypatch.setattr(launch_module, "ensure_cert",
                        lambda echo=(lambda *_: None): True)
    return str(root)


def test_ascii_only_guard_rejects_a_translated_path():
    assert service.ascii_only("/Users/me/.autolabel/services/mps.sh") is True
    assert service.ascii_only("/Users/我/服务/mps.sh") is False
