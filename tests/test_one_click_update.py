"""一键更新：宁可什么都不动，也不许把用户没提交的改动糊掉。

界面顶栏那句「GitHub 上有新版本」以前只是一条链接，用户还得回终端敲 git pull；
现在点一下就走完 fetch → 快进 → 提示重启。这条链上最容易伤人的三处：

1. 工作区脏着就 pull —— 刚才手标的半小时没了；
2. 本地历史和远端分叉了还硬合 —— 冲突留在工作区里，界面却显示"已更新"；
3. 拷贝安装 / pip 安装的版本没有 .git，却照样给一个点得动的更新按钮。

这一组把这些边界钉住：拒绝要给出可照抄的下一步，成功要说清"重启才生效"。
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

from autolabel import update as updater


@pytest.fixture(autouse=True)
def idle_state():
    """每条测试都从"没在更新"开始，别拿上一条留下的 running 做判断。"""
    updater._set(stage="idle", message="", kind="", branch="", from_version="",
                 to_version="", from_commit="", to_commit="", detail=[], dirty=[],
                 restartable=False, started_at="", finished_at="")
    yield


def fake_repo(monkeypatch, kind="git", dirty=None, upstream="origin/main"):
    """把一个假仓库的判据铺好：在不在 git 里、脏不脏、上游是谁。"""
    monkeypatch.setattr(updater, "install_kind", lambda: kind)
    monkeypatch.setattr(updater, "repo_root", lambda start=None: "/tmp/fake-repo")
    monkeypatch.setattr(updater, "git_bin", lambda: "/usr/bin/git")
    monkeypatch.setattr(updater, "dirty_files", lambda root, limit=12: list(dirty or []))
    monkeypatch.setattr(updater, "upstream_name", lambda root: upstream)
    monkeypatch.setattr(updater, "branch_name", lambda root: "main")
    monkeypatch.setattr(updater, "head_commit", lambda root: "abc1234")
    monkeypatch.setattr(updater, "newest_local_tag", lambda root: "1.4.20")
    return "/tmp/fake-repo"


class Runner:
    """假 git：按命令行前缀给结果，并把每次调用记下来，好断言"没多跑一步"。"""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, args, cwd, timeout=updater.GIT_TIMEOUT):
        self.calls.append((list(args), cwd, timeout))
        for prefix, outcome in self.results.items():
            if " ".join(args).startswith(prefix):
                return outcome
        return 0, "", ""

    def names(self):
        return [" ".join(args) for args, _cwd, _timeout in self.calls]


# --------------------------------------------------------------------------- #
# 一、能不能更新：不能就要说清为什么、下一步敲什么
# --------------------------------------------------------------------------- #

def test_dirty_worktree_blocks_the_update_and_lists_the_files(monkeypatch):
    fake_repo(monkeypatch, dirty=["autolabel/pool.py", "configs/vlm.json"])
    check = updater.preflight()
    assert check["ok"] is False
    assert "还没提交" in check["reason"]
    assert "autolabel/pool.py" in check["dirty"]
    assert "提交" in check["manual"] or "暂存" in check["manual"]


def test_copied_install_is_told_to_use_the_release_page(monkeypatch):
    fake_repo(monkeypatch, kind="plain")
    check = updater.preflight()
    assert check["ok"] is False and check["kind"] == "plain"
    assert updater.RELEASES_URL in check["manual"]


def test_pip_install_is_told_to_use_the_package_manager(monkeypatch):
    fake_repo(monkeypatch, kind="pip")
    check = updater.preflight()
    assert check["ok"] is False and check["kind"] == "pip"
    assert "pip install -U" in check["manual"]


def test_branch_without_upstream_is_refused(monkeypatch):
    fake_repo(monkeypatch, upstream="")
    check = updater.preflight()
    assert check["ok"] is False and "上游分支" in check["reason"]
    assert "set-upstream" in check["manual"]


def test_clean_repo_is_allowed_and_reports_where_it_starts(monkeypatch):
    fake_repo(monkeypatch)
    check = updater.preflight()
    assert check["ok"] is True and check["branch"] == "main" and check["from_commit"] == "abc1234"


# --------------------------------------------------------------------------- #
# 二、真的更新：只认快进，任何一步不对都当场停下
# --------------------------------------------------------------------------- #

def test_fast_forward_succeeds_and_asks_for_a_restart(monkeypatch):
    fake_repo(monkeypatch)
    runner = Runner({"fetch": (0, "", ""), "rev-list": (0, "3", ""), "merge": (0, "Fast-forward", "")})
    result = updater.perform(runner=runner)
    assert result["stage"] == "done" and result["restartable"] is True
    assert "3 个提交" in result["message"] and "重启" in result["message"]
    assert result["to_version"] == "1.4.20"
    assert any(call.startswith("merge --ff-only") for call in runner.names())


def test_nothing_behind_is_reported_as_already_latest(monkeypatch):
    fake_repo(monkeypatch)
    runner = Runner({"fetch": (0, "", ""), "rev-list": (0, "0", "")})
    result = updater.perform(runner=runner)
    assert result["stage"] == "done" and result["restartable"] is False
    assert "没有可快进的提交" in result["message"]
    assert not any(call.startswith("merge") for call in runner.names())


def test_fetch_failure_never_touches_the_worktree(monkeypatch):
    fake_repo(monkeypatch)
    runner = Runner({"fetch": (128, "", "fatal: unable to access origin")})
    result = updater.perform(runner=runner)
    assert result["stage"] == "failed" and "联网取更新失败" in result["message"]
    assert "fatal: unable to access origin" in result["detail"]
    assert not any(call.startswith("merge") for call in runner.names())


def test_diverged_history_fails_with_a_manual_route(monkeypatch):
    fake_repo(monkeypatch)
    runner = Runner({"fetch": (0, "", ""), "rev-list": (0, "2", ""),
                     "merge": (1, "", "Not possible to fast-forward")})
    result = updater.perform(runner=runner)
    assert result["stage"] == "failed" and "分叉" in result["message"]
    assert any("git status" in line for line in result["detail"])


def test_pull_false_fetches_but_leaves_the_worktree_alone(monkeypatch):
    fake_repo(monkeypatch)
    runner = Runner({"fetch": (0, "", ""), "rev-list": (0, "5", "")})
    result = updater.perform(pull=False, runner=runner)
    assert result["stage"] == "done" and result["restartable"] is False
    assert "5 个提交" in result["message"] and "未改动工作区" in result["message"]
    assert not any(call.startswith("merge") for call in runner.names())


def test_dirty_repo_never_even_reaches_the_network(monkeypatch):
    fake_repo(monkeypatch, dirty=["README.md"])
    runner = Runner()
    result = updater.perform(runner=runner)
    assert result["stage"] == "failed" and result["dirty"] == ["README.md"]
    assert runner.calls == []


def test_runner_must_accept_the_timeout_argument(monkeypatch):
    """调用处会按命令传不同超时；runner 少收一个参数就会在快进那步 TypeError。"""
    fake_repo(monkeypatch)
    seen = []

    def runner(args, cwd, timeout=updater.GIT_TIMEOUT):
        seen.append(timeout)
        if args[0] == "rev-list":
            return 0, "1", ""
        return 0, "", ""

    result = updater.perform(runner=runner)
    assert result["stage"] == "done", result["message"]
    assert seen and all(timeout for timeout in seen)


# --------------------------------------------------------------------------- #
# 三、后台执行与重启
# --------------------------------------------------------------------------- #

def test_start_reuses_a_running_job_instead_of_running_twice(monkeypatch):
    gate = threading.Event()
    done = threading.Event()
    calls = []

    def worker(pull=True):
        calls.append(pull)
        gate.wait(3)
        done.set()

    first = updater.start(worker=worker)
    second = updater.start(worker=worker)
    assert first["stage"] == "running" and second["stage"] == "running"
    gate.set()
    assert done.wait(3)
    assert len(calls) == 1, "同一时间只能有一个更新在跑"


def test_worker_crash_becomes_a_failure_not_a_dead_service():
    def worker(pull=True):
        raise RuntimeError("磁盘满了")

    result = updater.start(worker=worker)
    for _attempt in range(60):
        result = updater.state()
        if result["stage"] != "running":
            break
        threading.Event().wait(0.05)
    assert result["stage"] == "failed" and "RuntimeError" in result["message"]
    assert "磁盘满了" in result["detail"]


def test_restart_argv_keeps_the_module_form():
    """``python -m autolabel`` 起的服务，argv[0] 是 __main__.py；照抄会丢掉 -m。"""
    argv = ["/repo/autolabel/__main__.py", "review", "ds", "--port", "8765"]
    rebuilt = updater.restart_argv(argv)
    assert rebuilt[:4] == [updater.sys.executable, "-u", "-m", "autolabel"]
    assert rebuilt[4:] == ["review", "ds", "--port", "8765"]
    plain = updater.restart_argv(["/usr/local/bin/autolabel", "review"])
    assert plain == [updater.sys.executable, "/usr/local/bin/autolabel", "review"]


def test_schedule_restart_reports_back_before_it_leaves():
    woke = threading.Event()
    jumped = threading.Event()
    result = updater.schedule_restart(delay=0.01, sleeper=lambda _seconds: woke.set(),
                                      reexec=lambda: jumped.set())
    assert result["message"] == "服务重启中…"
    assert jumped.wait(3), "重启没执行，界面会一直卡在「重启中」"


def test_failed_restart_tells_the_user_how_to_do_it_by_hand(monkeypatch):
    monkeypatch.setattr(updater, "restart_argv", lambda argv=None: ["autolabel", "review"])

    def boom():
        raise OSError("execv 不行")

    updater.schedule_restart(delay=0.01, sleeper=lambda _s: None, reexec=boom)
    for _attempt in range(60):
        state = updater.state()
        if state["stage"] == "failed":
            break
        threading.Event().wait(0.05)
    assert "请手动重启服务" in updater.state()["detail"][0]


# --------------------------------------------------------------------------- #
# 四、状态与接口：界面轮询的字段必须一直在
# --------------------------------------------------------------------------- #

def test_state_always_carries_the_version_and_help_urls():
    snapshot = updater.state()
    for key in ("stage", "message", "kind", "current_version", "releases_url", "restartable"):
        assert key in snapshot
    from autolabel.version import VERSION

    assert snapshot["current_version"] == VERSION


def test_state_snapshot_is_a_copy(monkeypatch):
    updater._set(dirty=["a.py"])
    first = updater.state()
    first["dirty"].append("塞进来的")
    assert updater.state()["dirty"] == ["a.py"], "状态被外面改脏了，轮询会看到假进度"


def get_json(url, data=None):
    request = urllib.request.Request(url, data=json.dumps(data or {}).encode("utf-8")
                                     if data is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_api_update_reports_the_state_and_starts_the_job(server, monkeypatch):
    base, _store = server
    snapshot = get_json(f"{base}/api/update")
    assert snapshot["stage"] == "idle"

    monkeypatch.setattr(updater, "start", lambda pull=True: {"stage": "running", "pull": pull})
    started = get_json(f"{base}/api/update", {})
    assert started["stage"] == "running"


def test_api_update_restart_is_acknowledged(server, monkeypatch):
    base, _store = server
    monkeypatch.setattr(updater, "schedule_restart",
                        lambda delay=1.0: {"stage": "done", "delay": delay})
    assert get_json(f"{base}/api/update/restart", {"delay": 0.5})["delay"] == 0.5


def test_version_badge_states_where_the_number_came_from(server, monkeypatch):
    """顶栏那句「GitHub 最新 vX」必须标出来源：发布版 / 仓库标签 / 仓库代码不是一回事。"""
    base, store = server
    monkeypatch.setenv("AUTOLABEL_OFFLINE", "1")
    payload = store.version_payload(force=True)
    assert payload["source_label"], "版本信息里没有来源标注，界面只能干巴巴写「GitHub 最新」"
    from autolabel.version import source_label

    assert source_label("release") == "发布版"
    assert source_label("tag") == "仓库标签"
    assert source_label("offline") == "离线跳过"
    assert source_label("weird") == "weird", "认不出的来源要原样给出，别编个好听的"
