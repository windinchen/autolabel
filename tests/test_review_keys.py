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


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


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


def test_missing_image_error_says_where_it_went(server):
    """后端早就回了 404 + 中文，但读标注那条路径只剩「未知对象: ghost.jpg」。

    图在另一个标签页被删是常事，界面上这句话得说清图去哪了、该怎么办，
    而不是留一句像程序内部报错的半截英文。
    """
    base, _store = server
    status, body = get(base, "/api/item?image=ghost.jpg")
    assert status == 404
    message = body["error"]
    assert "未知对象" not in message and "'" not in message, message
    assert "已不在" in message and "刷新页面" in message, message
    assert "ghost.jpg" in message, message


def test_saving_boxes_reports_the_library_by_name(server):
    """存框时图已经不在：要说「已不在目标检测标注里」，画完才发现比翻页更早撞上。"""
    base, _store = server
    status, body = post(base, "/api/prompt/item",
                        {"image": "ghost.jpg", "lib": "detect",
                         "objects": [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]})
    assert status == 404, body
    assert "目标检测标注" in body["error"] and "未知对象" not in body["error"], body


def test_download_poll_without_a_record_reads_as_a_sentence(server):
    """查下载进度查不到记录：以前抛 KeyError，界面拼成「未知对象: 没有 x.pt 的下载记录」。"""
    base, _store = server
    status, body = post(base, "/api/models/download", {"name": "从没下载过.pt"})
    assert status == 404, body
    assert "下载记录" in body["error"] and "未知对象" not in body["error"], body
