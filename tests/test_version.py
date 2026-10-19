"""版本管理与更新提示：版本号只有一个来源、日常只升 0.01、GitHub 比本地旧不许谎报有更新。

这一组盯的是"版本管理"本身最容易烂掉的几处：

1. 界面上写死一份版本号（升到 1.2.1 之后顶栏还挂着 v1.1）；
2. 升版本时手改三处，漏掉 ``--version`` 或 CHANGELOG；
3. 本地是开发版（比 GitHub 的发布版新）却弹"有更新"，用户点了什么也没发生；
4. 没联网时报错甩栈，或者干脆一声不吭让人以为已经最新。
"""

import json
import os
import re
import urllib.request

import pytest

from autolabel import version as versioning

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "autolabel", "static")
PAGES = ("portal.html", "review.html", "objects.html", "models.html")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """更新检查缓存挪到 tmp，别拿开发者机器上那份半天缓存来影响断言。"""
    monkeypatch.setenv("AUTOLABEL_UPDATE_CACHE", str(tmp_path / "update.json"))
    monkeypatch.delenv("AUTOLABEL_OFFLINE", raising=False)


def read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as fh:
        return fh.read()


def get_json(url):
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def fake_fetcher(release=None, raw=None):
    """不联网的 GitHub：Releases 接口和 raw 文件各自给一份假数据。"""
    def fetch(url):
        if "api.github.com" in url:
            if release is None:
                raise OSError("没有发布过版本")
            return json.dumps({"tag_name": release})
        if raw is None:
            raise OSError("raw 读不到")
        return raw
    return fetch


# --------------------------------------------------------------------------- #
# 一、版本号只有一个来源，升版只改一行
# --------------------------------------------------------------------------- #

def test_version_number_lives_in_one_place():
    """``autolabel.__version__`` 必须等于 version.py 里的 VERSION，界面与 CLI 都从这里取。"""
    import autolabel

    assert autolabel.__version__ == versioning.VERSION
    assert re.fullmatch(r"\d+\.\d+\.\d+", versioning.VERSION), "版本号要写成 1.2.3 这样"
    assert autolabel.DEVELOPER == versioning.DEVELOPER == "windinchen"
    assert versioning.HOME_PAGE_URL == "https://github.com/windinchen"


def test_daily_bump_follows_plus_001():
    """约定：日常改动一律 +0.01（patch 段 +1），别让每次改动都去动大版本。"""
    assert versioning.bump("1.2.1") == "1.2.2"
    assert versioning.bump("1.2.9") == "1.2.10"     # 段内加一，不是小数点算术
    assert versioning.bump("v1.2") == "1.2.1"


def test_major_and_minor_bumps_reset_lower_parts():
    assert versioning.bump("1.2.9", part="minor") == "1.3.0"
    assert versioning.bump("1.2.9", part="major") == "2.0.0"


def test_bump_rejects_unknown_part():
    with pytest.raises(ValueError) as exc:
        versioning.bump("1.2.1", part="huge")
    assert "未知的版本粒度" in str(exc.value)


def test_rewrite_only_touches_the_version_line(tmp_path):
    """``version --bump`` 只许改 ``VERSION`` 一行，其它内容（含注释）一个字都不动。"""
    source = os.path.join(os.path.dirname(versioning.__file__), "version.py")
    target = str(tmp_path / "version.py")
    with open(source, encoding="utf-8") as fh:
        original = fh.read()
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(original)

    old, new = versioning.rewrite("9.9.9", path=target)
    assert (old, new) == (versioning.VERSION, "9.9.9")
    with open(target, encoding="utf-8") as fh:
        changed = fh.read()
    diff = [i for i, (a, b) in enumerate(zip(original.splitlines(), changed.splitlines())) if a != b]
    assert len(diff) == 1, f"改动行号 {diff}，应当只有 VERSION 那一行"
    assert versioning.VERSION not in changed.splitlines()[diff[0]].replace('"9.9.9"', "")
    assert re.search(r'^VERSION = "9\.9\.9"$', changed, re.M), "新版本号没写对位置"


# --------------------------------------------------------------------------- #
# 二、和 GitHub 比对：只认真正更新的版本
# --------------------------------------------------------------------------- #

def test_compare_tolerates_v_prefix_and_short_versions():
    assert versioning.compare("v1.2", "1.2.0") == 0
    assert versioning.compare("1.10.0", "1.9.9") > 0
    assert versioning.compare("不是版本", "1.0.0") < 0, "解析不出来的要当最旧，不能当最新"


def test_fetch_latest_prefers_release_tag():
    """GitHub 的 tag 带 v 前缀，取回来就归一化成纯数字，界面才能自己拼显示。"""
    latest, source = versioning.fetch_latest(fetcher=fake_fetcher(release="v1.5.0"))
    assert (latest, source) == ("1.5.0", "release")
    assert versioning.compare(latest, "1.4.9") > 0


def test_fetch_latest_falls_back_to_repository_file():
    """没发过 release、只是把 main 推上去了，也要能提示到更新。"""
    latest, source = versioning.fetch_latest(fetcher=fake_fetcher(raw='VERSION = "1.6.2"\n'))
    assert (latest, source) == ("1.6.2", "source")


def test_update_is_only_claimed_when_github_is_newer():
    newer = versioning.status(fetcher=fake_fetcher(release="99.0.0"), force=True)
    assert newer["update_available"] is True and newer["checked"] is True

    same = versioning.status(fetcher=fake_fetcher(release="v" + versioning.VERSION), force=True)
    assert same["update_available"] is False and same["ahead"] is False

    # 开发中的检出比发布版新：说清楚"本地更新"，但别弹"有更新"骗人去点
    older = versioning.status(fetcher=fake_fetcher(release="0.1.0"), force=True)
    assert older["update_available"] is False and older["ahead"] is True


def test_offline_check_admits_it_instead_of_lying(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_OFFLINE", "1")
    info = versioning.status(force=True)
    assert info["checked"] is False and info["update_available"] is False
    assert info["source"] == "offline" and "离线" in info["note"]

    def always_fail(_url):
        raise OSError("网络不可达")

    broken = versioning.status(fetcher=always_fail, force=True)
    assert broken["checked"] is False and broken["note"], "取不到版本必须说清为什么"
    assert "已经是最新版" not in versioning.describe(broken)


# --------------------------------------------------------------------------- #
# 三、接口与界面：四个页面都得挂着版本号，且数字不写死
# --------------------------------------------------------------------------- #

def test_api_version_endpoint_matches_backend(server, monkeypatch):
    base, store = server
    monkeypatch.setattr(versioning, "fetch_latest", lambda fetcher=None: ("99.0.0", "release"))
    body = get_json(base + "/api/version?refresh=1")
    assert body["current"] == versioning.VERSION
    assert body["developer"] == "windinchen" and body["developer_url"].endswith("windinchen")
    assert body["repo_url"].endswith("/autolabel") and body["releases_url"].endswith("/releases")
    assert body["update_available"] is True and body["latest"] == "99.0.0"
    assert get_json(base + "/api/version")["latest"] == "99.0.0", "第二次读缓存也要给同一个结论"


def test_every_page_shows_version_and_the_about_block():
    """顶栏徽标、更新提示条、「关于」菜单在四个页面一处都不能少。"""
    for name in PAGES:
        html = read(name)
        assert 'id="version-chip"' in html, f"{name} 顶栏没有版本徽标"
        assert 'id="update-bar"' in html, f"{name} 缺更新提示条"
        assert 'id="about-dev"' in html and ">windinchen<" in html, f"{name} 没写开发者"
        assert 'id="about-check"' in html, f"{name} 的菜单里没有检查更新"
        assert 'symbol id="ic-refresh"' in html, f"{name} 缺检查更新用的图标"


def test_frontend_never_hardcodes_a_version():
    """界面只渲染后端给的数字；写死一份就会出现顶栏 v1.1、命令行 1.2 的笑话。"""
    common = read("common.js")
    assert not re.search(r'["\']\d+\.\d+\.\d+["\']', common), "common.js 里出现了写死的版本号"
    assert '"/api/version"' in common and "function initVersionWidget" in common
    assert "initVersionWidget();" in common.split("function initNav() {", 1)[1].split("\n}", 1)[0], \
        "版本控件没挂进四个页面共用的 initNav()"
    assert "checkForUpdates" in common.split("function appearanceCommands() {", 1)[1], \
        "命令面板搜不到检查更新"


def test_rewrite_only_accepts_three_part_versions(tmp_path):
    """``version --bump`` 之外还有人手填版本号：``1.2`` 得挡住。

    比较时 ``1.2`` == ``1.2.0``，写成 ``1.2`` 之后顶栏显示 v1.2、更新提示按 1.2.0
    算，用户只觉得版本号没生效。四段同理，别让它悄悄进 main。
    """
    target = tmp_path / "version_copy.py"
    target.write_text('VERSION = "1.2.4"\n', encoding="utf-8")

    for bad in ("1.2", "1.2.3.4", "abc", "", "1.2.x"):
        with pytest.raises(ValueError, match="三段数字"):
            versioning.rewrite(bad, path=str(target))

    old, new = versioning.rewrite("v1.3.0", path=str(target))
    assert (old, new) == ("1.2.4", "1.3.0"), "带 v 前缀要归一化后再写回文件"
    assert target.read_text(encoding="utf-8") == 'VERSION = "1.3.0"\n'


# --- 第 18 轮体检：更新缓存是用户机器上一个会坏掉的普通文件 --------------- #

def _write_cache(payload):
    with open(versioning.cache_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


@pytest.mark.parametrize("checked_at", ["abc", {"x": 1}, [1], True, None, "", "2026-10-19"])
def test_broken_checked_at_does_not_break_the_badge(checked_at):
    """``checked_at`` 认不出来只能当"没查过"，不能让 ``float()`` 当场炸。

    ``~/.autolabel/update.json`` 会被手改、被半截写入污染。以前 ``checked_at``
    是字符串时 ``status()`` 直接 ValueError，``/api/version`` 变 500，顶栏版本
    徽标和启动时的后台检查线程一起没——一个徽标拖垮一整页。
    """
    _write_cache({"latest": "9.9.9", "checked_at": checked_at, "source": "release"})

    info = versioning.status()

    assert info["current"] == versioning.VERSION
    assert isinstance(info["checked_at"], float)


def test_non_scalar_latest_is_not_rendered_as_a_python_literal():
    """``{"latest": ["1.9.0"]}`` 以前会被 ``str()`` 渲染成 ``"['1.9.0']"``。"""
    _write_cache({"latest": ["1.9.0"], "checked_at": 2 ** 31 + 1})

    info = versioning.status()

    assert info["latest"] == "" and info["checked"] is False
    assert "[" not in json.dumps(info)


def test_half_written_cache_file_is_just_an_empty_cache():
    """JSON 只写了一半：按"没查过"处理，不抛。"""
    with open(versioning.cache_path(), "w", encoding="utf-8") as fh:
        fh.write('{"latest": "1.9.0", "checked_')

    info = versioning.status(fetcher=lambda url: '{"tag_name": "v2.0.0"}')

    assert info["latest"] == "2.0.0" and info["update_available"] is True


def test_offline_with_a_broken_cache_still_explains_itself(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_OFFLINE", "1")
    _write_cache({"latest": "1.9.0", "checked_at": "昨天"})

    info = versioning.status(force=True)

    assert info["latest"] == "1.9.0" and "离线" in info["note"]
