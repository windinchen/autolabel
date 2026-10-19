"""后台服务脚本（后台服务.command / 后台实例.command）的回归测试。

这两个脚本负责"开机自起、崩了自动拉起"，而它们踩过的三个坑都靠测试钉住：

1. 启动脚本以前写在 ``/tmp``，重启就被清掉，服务第二天起不来——现在文件必须在项目里；
2. 项目路径带中文，直接写进 launchd 的 ``ProgramArguments`` 会被传成乱码，
   任务每 10 秒崩一次——所以 plist 里只能出现纯 ASCII 路径；
3. 软链启动器必须按"解析后的真实位置"找项目目录，``dirname "$0"`` 在这儿会指错。

清单默认是 ``configs/后台实例.conf``（本机偏好，不进仓库），测试一律用
``AUTOLABEL_SERVICE_CONF`` 指到临时文件，并把 ``HOME`` 也换掉，绝不碰真机上的
``~/Library/LaunchAgents`` 和 ``launchctl``。
"""
import os
import plistlib
import shutil
import stat
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANAGER = os.path.join(ROOT, "后台服务.command")
RUNNER = os.path.join(ROOT, "后台实例.command")
REAL_CONF = os.path.join(ROOT, "configs", "后台实例.conf")
ZSH = shutil.which("zsh")

pytestmark = pytest.mark.skipif(ZSH is None, reason="这台机器上没有 zsh，后台服务脚本用不上")


@pytest.fixture()
def service_env(tmp_path):
    """临时清单 + 临时 HOME：脚本写 plist / 建软链都落在 tmp_path 里。"""
    dataset = tmp_path / "datasets" / "svc"
    dataset.mkdir(parents=True)
    conf = tmp_path / "instances.conf"
    conf.write_text(
        "# 测试清单\n"
        "mps 8765 %s - no\n"
        "cuda 8766 %s cuda yes\n" % (dataset, dataset),
        encoding="utf-8",
    )
    env = dict(os.environ, HOME=str(tmp_path / "home"), AUTOLABEL_SERVICE_CONF=str(conf))
    return {"env": env, "tmp": tmp_path, "conf": conf, "dataset": str(dataset)}


def run(args, env=None, cwd=ROOT):
    return subprocess.run([ZSH] + args, cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=60)


def texts(result):
    return (result.stdout or "") + (result.stderr or "")


def _conf_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def test_both_scripts_shipped_and_executable():
    for path in (MANAGER, RUNNER):
        assert os.path.isfile(path), f"缺文件：{path}"
        assert os.stat(path).st_mode & stat.S_IXUSR, f"没有执行权限：{path}"


@pytest.mark.parametrize("path", [MANAGER, RUNNER])
def test_scripts_are_syntactically_valid(path):
    result = run(["-n", path])
    assert result.returncode == 0, texts(result)


@pytest.mark.parametrize("script,keyword", [(MANAGER, "后台服务.command"), (RUNNER, "后台实例.command")])
def test_help_explains_usage(script, keyword):
    result = run([script, "--help"])
    assert result.returncode == 0, texts(result)
    assert keyword in texts(result)
    assert "用法" in texts(result)


def test_print_env_returns_the_instance_parameters(service_env):
    result = run([MANAGER, "print-env", "cuda"], env=service_env["env"])
    assert result.returncode == 0, texts(result)
    import shlex

    values = dict(item.split("=", 1) for item in shlex.split(result.stdout))
    assert values == {"PORT": "8766", "DATASET": service_env["dataset"],
                      "DEVICE": "cuda", "HALF": "yes"}


@pytest.mark.parametrize("name,keyword", [
    ("nope", "清单里没有实例"),
])
def test_print_env_names_the_missing_instance_in_chinese(service_env, name, keyword):
    result = run([MANAGER, "print-env", name], env=service_env["env"])
    assert result.returncode != 0
    assert keyword in texts(result)


def test_conf_rejects_non_ascii_instance_name(tmp_path):
    """实例名要写进 launchd 标签和软链文件名，带中文等于埋一颗雷。"""
    conf = tmp_path / "bad.conf"
    conf.write_text("姿态 8765 . - no\n", encoding="utf-8")
    result = run([MANAGER, "--status"], env=dict(os.environ, AUTOLABEL_SERVICE_CONF=str(conf)))
    assert result.returncode != 0
    assert "实例名不合规" in texts(result)


def test_conf_rejects_missing_dataset(tmp_path):
    conf = tmp_path / "bad.conf"
    conf.write_text("mps 8765 /definitely/not/here - no\n", encoding="utf-8")
    result = run([MANAGER, "--status"], env=dict(os.environ, AUTOLABEL_SERVICE_CONF=str(conf)))
    assert result.returncode != 0
    assert "数据集不存在" in texts(result)


def test_conf_rejects_non_numeric_port(tmp_path):
    conf = tmp_path / "bad.conf"
    conf.write_text("mps 87xx . - no\n", encoding="utf-8")
    result = run([MANAGER, "--status"], env=dict(os.environ, AUTOLABEL_SERVICE_CONF=str(conf)))
    assert result.returncode != 0
    assert "端口不是数字" in texts(result)


def test_conf_with_only_comments_says_so(tmp_path):
    conf = tmp_path / "empty.conf"
    conf.write_text("# 只有注释\n\n", encoding="utf-8")
    result = run([MANAGER, "--status"], env=dict(os.environ, AUTOLABEL_SERVICE_CONF=str(conf)))
    assert result.returncode != 0
    assert "一个实例都没有" in texts(result)


def test_plist_only_contains_ascii_paths_and_stays_alive(service_env):
    result = run([MANAGER, "plist", "mps"], env=service_env["env"])
    assert result.returncode == 0, texts(result)
    data = plistlib.loads(result.stdout.encode("utf-8"))
    label = data["Label"]
    assert label.endswith(".autolabel.mps") and label.startswith("com.")
    args = data["ProgramArguments"]
    assert args[0] == "/bin/zsh"
    shim = args[1]
    # 这一条是整个设计的核心：launchd 传非 ASCII 路径会变成乱码
    assert shim.isascii(), f"启动路径不能带非 ASCII 字符：{shim}"
    assert shim.endswith(os.sep + "mps.sh")
    assert args[2] == "mps"                      # 实例名是唯一的入参，其余每次启动现读清单
    assert data["RunAtLoad"] is True
    # 只认非正常退出：启动器读不到项目时会说明原因并 exit 0，这时别再空转重试
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] >= 10         # 起不来时别把 CPU 烧在重试上
    assert data["StandardErrorPath"] == "/tmp/autolabel-8765.log"


def test_dry_run_lands_plist_and_shim_without_launchd(service_env):
    result = run([MANAGER, "--dry-run"], env=service_env["env"])
    assert result.returncode == 0, texts(result)
    agent_dir = service_env["tmp"] / "home" / "Library" / "LaunchAgents"
    plists = sorted(p.name for p in agent_dir.glob("*.plist"))
    assert len(plists) == 2 and all(p.endswith(".plist") for p in plists)
    shim = service_env["tmp"] / "home" / ".autolabel" / "services" / "mps.sh"
    assert shim.is_symlink() and os.path.realpath(shim) == os.path.realpath(RUNNER)
    assert "--dry-run" in texts(result)


def test_plist_written_by_dry_run_is_the_one_launchd_will_read(service_env):
    run([MANAGER, "--dry-run"], env=service_env["env"])
    agent_dir = service_env["tmp"] / "home" / "Library" / "LaunchAgents"
    written = agent_dir.glob("*.cuda.plist")
    data = plistlib.loads(next(written).read_bytes())
    rendered = run([MANAGER, "plist", "cuda"], env=service_env["env"])
    assert data == plistlib.loads(rendered.stdout.encode("utf-8"))


def test_runner_finds_the_project_through_a_symlink(tmp_path):
    """plist 里写的是 ~/.autolabel/services/<名字>.sh，靠软链指回项目。

    所以启动器必须按解析后的真实位置定位项目目录——测试直接跑这根软链，
    能打印用法就说明 ``${0:P:h}`` 生效了。
    """
    link = tmp_path / "mps.sh"
    link.symlink_to(RUNNER)
    result = run([str(link), "--help"])
    assert result.returncode == 0, texts(result)
    assert "用法" in texts(result)


def test_runner_rejects_unknown_flag_with_usage_hint(service_env):
    result = run([RUNNER, "--nope"], env=service_env["env"])
    assert result.returncode == 2
    assert "不认识的参数" in texts(result)


def test_missing_conf_says_what_to_write(tmp_path):
    result = run([MANAGER, "--status"],
                 env=dict(os.environ, AUTOLABEL_SERVICE_CONF=str(tmp_path / "none.conf")))
    assert result.returncode != 0
    body = texts(result)
    assert "没有实例清单" in body and "mps 8765" in body


@pytest.mark.skipif(not os.path.isfile(REAL_CONF), reason="本机没有后台实例清单")
def test_real_conf_is_launchd_safe():
    """真机上那份清单必须能被同一套校验通过（名字纯 ASCII、端口不重复、数据集存在）。"""
    result = run([MANAGER, "--status"])          # 不注入临时清单，读的就是开发者这份
    assert result.returncode == 0, texts(result)
    ports = []
    for line in _conf_lines(REAL_CONF):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        assert fields[0].isascii(), f"实例名要写进 launchd 标签，别用中文：{fields[0]}"
        ports.append(fields[1])
    assert len(ports) == len(set(ports)), f"端口重复：{ports}"

