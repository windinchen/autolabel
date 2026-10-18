"""快捷键不许和浏览器抢键，翻页要接住读图失败——都是复核界面上真实会撞到的事。"""
import json
import os
import urllib.error
import urllib.request

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "autolabel", "static")


def read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as fh:
        return fh.read()


def keydown_body(name):
    source = read(name)
    start = source.index("function onKeyDown(e)")
    end = source.index("\n}\n", start)
    return source[start:end]


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_command_shortcuts_stop_hijacking_the_page():
    """⌘A / ⌘D / ⌘F / ⌘J / ⌘B / ⌘1… 归浏览器，不该顺手翻页、改可见性或跳去模型页。"""
    for page in ("app.js", "objects.js"):
        body = keydown_body(page)
        assert "e.metaKey || e.ctrlKey" in body, f"{page} 的快捷键没挡修饰键组合"
        guard = body.index("e.metaKey || e.ctrlKey")
        assert guard < body.index("switch (e.key)"), f"{page} 的修饰键守卫来得太晚"
        assert guard > body.index('["INPUT", "SELECT", "TEXTAREA"]'), f"{page} 守卫位置不对"
        # 守卫之后必须直接 return，剩下的组合交回给浏览器
        assert "return;" in body[guard:body.index("switch (e.key)")]


def test_cmd_z_is_undo_on_both_pages():
    """撤销的通用键是 ⌘Z；裸 z 仍是「适应窗口」，但 ⌘Z 不该只重置视图。"""
    for page, undo_call in (("app.js", "undo();"), ("objects.js", "undoItem();")):
        body = keydown_body(page)
        guard = body[body.index("e.metaKey || e.ctrlKey"):body.index("switch (e.key)")]
        assert 'e.key === "z"' in guard and undo_call in guard, f"{page} 的 ⌘Z 没接去撤销"


def test_item_load_reports_failures_instead_of_blanking():
    """图在另一个标签页被删时，要给一句人话，而不是白画布 + 控制台里的 TypeError。"""
    source = read("app.js")
    assert 'getJson("/api/item?image="' in source, "读标注没走会报错的 getJson"
    assert "state.item = await res.json()" not in source, "读标注还在忽略 HTTP 状态码"
    assert "读不到这张图的标注" in source and 'toast("读不到 ' in source


def test_stale_item_responses_are_dropped():
    """快速连翻时先发的请求可能后到，过期结果不许再把当前张换成旧图。"""
    source = read("app.js")
    assert "let loadSeq = 0;" in source
    assert source.count("if (seq !== loadSeq) return;") >= 2, "只挡住了一处，图片加载回来还是会覆盖"


def test_boot_says_why_the_dataset_did_not_load():
    source = read("app.js")
    assert "数据集读不出来" in source, "首屏拉不到数据集时静默白屏"


def test_unknown_object_error_has_no_python_quotes(server):
    """后端早就回了 404 + 中文，但 KeyError 会把路径带着引号吐出来，界面上一看就是程序报错。"""
    base, _store = server
    status, body = get(base, "/api/item?image=ghost.jpg")
    assert status == 404
    assert body["error"] == "未知对象: ghost.jpg", body
