"""文案里出现的命令名必须真的存在（第 32 轮体检）。

界面提示和报错里经常顺手写「命令行 autolabel xxx」，而命令名是靠手抄的：
以前导出空库时提示的是 ``autolabel auto / annotate``，可 ``annotate``
从来没注册过，用户照着敲只会得到 ``invalid choice`` 第二次打击。

这条测试把 ``autolabel/**``（含 static 里的 JS/HTML）里所有
``autolabel <子命令>`` 形式的提法抓出来，逐个比对真实子命令表。
"""
import argparse
import io
import os
import re

import pytest

from autolabel import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIRS = ("autolabel",)
SRC_EXTS = (".py", ".js", ".html", ".css", ".md")

# autolabel download / autolabel --version 这类全局写法不是子命令，白名单放行
STOPWORDS = {"--help", "-h", "--version", "cli", "review-server"}

# 「autolabel auto / label」「autolabel import、auto」「autolabel export|coords」
SEP = r"\s*(?:/|、|或|,|，|\|)\s*"
PATTERN = re.compile(r"autolabel\s+((?:[a-z][\w-]*)(?:" + SEP + r"[a-z][\w-]*)*)")


def _sources():
    for name in SRC_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, name)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.endswith(SRC_EXTS):
                    yield os.path.join(dirpath, fn)


def _real_commands():
    """从 CLI 定义里取真实子命令名，别在测试里再抄一份清单。"""
    parser = cli.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("CLI 里没有子命令表，测试要跟着改")


def _mentioned():
    """返回 {命令名: 出现位置} —只保留像命令的小写词。"""
    hits = {}
    for path in _sources():
        text = io.open(path, encoding="utf-8").read()
        for match in PATTERN.finditer(text):
            for word in re.split(SEP, match.group(1)):
                if word and word not in STOPWORDS:
                    hits.setdefault(word, os.path.relpath(path, ROOT))
    return hits


def test_real_command_list_is_complete():
    cmds = _real_commands()
    assert len(cmds) >= 20, f"子命令数量异常（{len(cmds)}），疑似解析方式失效"
    assert {"auto", "label", "import", "export", "validate"} <= cmds


@pytest.mark.parametrize("word,where", sorted(_mentioned().items()))
def test_every_documented_command_exists(word, where):
    assert word in _real_commands(), f"{where} 里提示的 autolabel {word} 并不是真实命令"


def test_hint_no_longer_mentions_annotate():
    """曾经的那条错提示不能悄悄回来。"""
    for path in _sources():
        text = io.open(path, encoding="utf-8").read()
        assert "auto / annotate" not in text and "auto/annotate" not in text, path
