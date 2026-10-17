"""新壳层的界面契约：四个页面必须是同一套壳层、状态词只有一份、筛选项都得真的能筛、
门户不许把实现细节当主词。

这些都是 UI 重构里真踩过的坑：状态词表只写在姿态页导致提示词页整页报错，
右侧动作区用 sticky 把目标列表整个盖住，筛选下拉里多一个没人处理的值，
以及门户/模型页迟迟没跟上新壳、顶栏和左导航条各长各的。
"""
import os
import re

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "autolabel", "static")

# 后端会写回的状态原值，和 qc / review 里的口径保持一致
KNOWN_STATUS = {"unreviewed", "approved", "edited", "rejected"}
# 界面层自己加的聚合项：待复核（含已修改）、仅带标记、未标注、全部
POSE_FILTERS = {"pending", "flagged", "all"} | KNOWN_STATUS
OBJECTS_FILTERS = POSE_FILTERS | {"unlabeled"}


def read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as fh:
        return fh.read()


def filter_options(html):
    block = re.search(r'<select id="filter".*?</select>', html, re.S)
    assert block, "页面里没有筛选下拉"
    return re.findall(r'<option value="([^"]+)"', block.group(0))


def test_status_words_are_defined_once_and_shared():
    """状态词表只允许在 common.js 出现一份：重复 const 会让整页脚本直接不执行。"""
    common = read("common.js")
    assert "const STATUS_LABEL" in common and "const statusLabel" in common
    for page in ("app.js", "objects.js"):
        source = read(page)
        assert "const STATUS_LABEL" not in source, f"{page} 又自己定义了一遍状态词"
        assert "const statusLabel" not in source, f"{page} 又自己定义了一遍状态词"
        assert "statusLabel(" in source, f"{page} 没有用统一状态词"


def test_both_review_pages_ship_the_new_shell():
    """两个标注页必须是同一套壳层：竖排队列、HUD、撤销条、命令面板、速查表、可滚动的面板。"""
    for page in ("review.html", "objects.html"):
        html = read(page)
        for anchor in ('id="rail"', 'id="queue-col"', 'id="hud"', 'id="undo-bar"',
                       'id="panel-scroll"', 'id="palette-modal"', 'id="help-modal"',
                       'id="capsule-count"'):
            assert anchor in html, f"{page} 缺 {anchor}"
        assert "$(\"capsule-count\")" in read("objects.js" if page == "objects.html" else "app.js"), \
            f"{page} 的待确认数没有回填"
    css = read("app.css")
    actions = re.search(r"\.panel-block\.actions \{[^}]*\}", css).group(0)
    assert "sticky" not in actions, "动作区用 sticky 会把上方列表盖住，必须留在面板底部"


def test_every_filter_option_is_actually_understood():
    """下拉里出现的筛选值必须被 JS 认识，否则选了等于没选，用户只会以为界面坏了。"""
    for page, allowed in (("review.html", POSE_FILTERS), ("objects.html", OBJECTS_FILTERS)):
        options = filter_options(read(page))
        assert "pending" in options, f"{page} 默认筛选应当是「待复核」"
        unknown = sorted(set(options) - allowed)
        assert not unknown, f"{page} 有没人处理的筛选项：{unknown}"


# --------------------------------------------------------------------------- //
# 门户与模型管理：文档型页面也必须共用同一套壳层
# --------------------------------------------------------------------------- //
DOC_PAGES = {"portal.html": "portal.js", "models.html": "models.js"}


def test_document_pages_share_the_same_shell():
    """门户和模型管理不许留旧顶栏：左导航条、命令面板、速查表、溢出菜单一个都不能少。"""
    css = read("app.css")
    assert ".shell-row" in css, "缺 .shell-row：文档页的左导航条没地方放"
    for page, script in DOC_PAGES.items():
        html = read(page)
        for anchor in ('id="rail"', 'class="shell-row"', 'id="btn-palette"', 'id="palette-modal"',
                       'id="help-modal"', '<div class="menu" id="more-menu" hidden>',
                       'id="nav-modules"', 'id="confirm-modal"'):
            assert anchor in html, f"{page} 缺 {anchor}"
        source = read(script)
        for call in ("registerShellCommands(", "initShellChrome(", "shellKeys(e)"):
            assert call in source, f"{script} 没接壳层：{call}"


def test_document_pages_handle_every_key_they_advertise():
    """速查表里写出来的单键必须在同页 JS 里真的能按，否则等于骗用户。"""
    for page, script in DOC_PAGES.items():
        html = read(page)
        source = read(script)
        for key in set(re.findall(r"<div><span>([A-Za-z])</span>", html)):
            assert f'case "{key.lower()}"' in source, f"{page} 宣传了 {key} 键，{script} 里却没处理"


# --------------------------------------------------------------------------- //
# 外观：白天 / 夜晚 · 字号缩放
# --------------------------------------------------------------------------- //
ALL_PAGES = {"review.html": "app.js", "objects.html": "objects.js",
             "portal.html": "portal.js", "models.html": "models.js"}


def test_appearance_controls_are_wired_on_every_page():
    """外观是全站设置：四页都得有首屏脚本、菜单控件位，速查表也得写上快捷键。"""
    css = read("app.css")
    assert ':root[data-theme="light"]' in css, "没有白天主题变量块"
    assert ".menu-seg" in css, "缺外观分段选择器样式"
    for step in ('[data-fs="0"]', '[data-fs="2"]', '[data-fs="3"]'):
        assert step in css, f"缺字号档位 {step}"
    for page in ALL_PAGES:
        html = read(page)
        assert 'id="appearance-ctl"' in html, f"{page} 的 ⋯ 菜单没有外观控件位"
        assert "autolabel.appearance" in html, f"{page} 缺首屏外观脚本，切页会闪一下默认主题"
        assert "dataset.theme" in html and "dataset.fs" in html, f"{page} 首屏脚本没写 data-theme / data-fs"
        for advertised in ("⌥T", "⌥=", "⌥-"):
            assert advertised in html, f"{page} 速查表没写 {advertised}"
    common = read("common.js")
    for api in ("function setThemePref", "function toggleTheme", "function stepFont",
                "function renderAppearanceControls", "function canvasPalette"):
        assert api in common, f"common.js 缺 {api}()"
    assert "appearanceCommands()" in read("common.js").split("function registerShellCommands")[1], \
        "外观动作没进命令面板，⌘K 搜不到"


def test_every_font_size_follows_the_scale():
    """字号必须全部走 --fs，漏一条就是「放大字号后有一处大小没变」。"""
    css = read("app.css")
    fixed = [d for d in re.findall(r"font-size:\s*([^;]+);", css) if "var(--fs)" not in d]
    assert not fixed, f"这些字号没接缩放：{fixed}"
    assert "font: calc(13.5px * var(--fs))/1.5" in css, "body 的字号没接缩放"


def test_theme_dependent_colors_are_not_hardcoded():
    """`:root` 之外不许写死颜色：写死的深色到白天就隐形。"""
    css = read("app.css")
    outside = re.sub(r":root[^{]*\{.*?\n\}", "", css, flags=re.S)
    literals = sorted(set(re.findall(r"#[0-9A-Fa-f]{3,8}\b|rgba?\([^)]*\)", outside)))
    assert not literals, f"这些颜色没走 token：{literals}"


def test_canvas_draws_with_theme_colors_not_literals():
    """画布上的颜色和字号也得跟外观走，否则白天背景上会出现白字白线。"""
    for page in ("app.js", "objects.js"):
        source = read(page)
        assert "canvasPalette()" in source, f"{page} 画布没取外观配色"
        assert "cfont(" in source, f"{page} 画布字号没接缩放"
        dead = [probe for probe in (
            'fillStyle = "#',
            "fillStyle = '#",
            'fillStyle = "rgba(',
            'strokeStyle = "#',
            "strokeStyle = '#",
            'strokeStyle = "rgba(',
        ) if probe in source]
        assert not dead, f"{page} 画布还在用写死的颜色：{dead}"
        assert 'addEventListener("appearancechange"' in source, f"{page} 换主题后没重画画布"


def test_overlays_on_photos_use_bright_ink():
    """压在照片上的深色底板两种主题都是深的，里面的字必须是恒定偏亮的「亮墨」。

    白天模式的强调色是为了读浅色面板而压深的，直接搬到深色底板上就成深字压深底：
    实拍时 HUD 的「4 处待确认」和画布上的 ``#1 k13/17`` 几乎看不见。
    """
    css = read("app.css")
    for token in ("--veil-ink", "--veil-ink-2", "--veil-ink-3", "--veil-hairline",
                  "--veil-jade", "--veil-amber", "--veil-coral",
                  "--c-photo-amber", "--c-photo-sky", "--c-photo-coral"):
        assert token + ":" in css, f"缺深色底板用的亮墨 token {token}"
    for sel in ("#hud {", "#hud .segs i", "#hud .chip.bad", "#hud .chip.ok", "#hud .pos",
                  "#draw-hint {"):
        assert sel in css, f"样式 {sel} 不见了"
        block = css.split(sel, 1)[1].split("}", 1)[0]
        assert "var(--veil-" in block, f"{sel} 压在深色底板上却没走亮墨 token"
        dead = [name for name in ("var(--ink", "var(--amber)", "var(--jade)", "var(--coral)",
                                 "var(--sky)", "var(--hairline)") if name in block]
        assert not dead, f"{sel} 还在用主题色，白天会出现深字压深底：{dead}"
    assert "photoLabelInk(" in read("app.js"), "画布标签的文字没走亮墨，白天读不清"

def test_quality_flags_are_shown_in_chinese():
    """质量标记在界面上必须说中文：原始 key 只留在 tooltip 里对着报表查。

    实拍时状态区整块写着 ``limb_length_mismatch``，用户根本不知道在说什么。
    """
    from autolabel.prompt import PROMPT_FLAG_HINTS
    from autolabel.qc import FLAG_HINTS

    common = read("common.js")
    block = re.search(r"const FLAG_LABEL = \{(.*?)\n\};", common, re.S)
    assert block, "common.js 缺质量标记的中文名表 FLAG_LABEL"
    labeled = dict(re.findall(r'(\w+):\s*"([^"]+)"', block.group(1)))
    for flag in sorted(set(FLAG_HINTS) | set(PROMPT_FLAG_HINTS)):
        assert flag in labeled, f"标记 {flag} 没有中文名"
        assert re.search(r"[\u4e00-\u9fff]", labeled[flag]), f"标记 {flag} 的标签不是中文"
    assert "const flagLabel" in common, "common.js 缺 flagLabel()，两页会各写各的"
    for page in ("app.js", "objects.js"):
        assert "flagLabel(" in read(page), f"{page} 还在直接显示标记 key"


def test_portal_leads_with_next_actions_and_drops_internalese():
    """门户先回答「下一步做什么」，并且不许出现素材池/两库共用/二级模块这类实现细节词。"""
    html = read("portal.html")
    portal_js = read("portal.js")
    assert 'id="next-actions"' in html, "门户缺行动流区块"
    assert "renderNextActions(" in portal_js and "renderNextActions(ds);" in portal_js, "行动流没被渲染"
    assert "slice(0, 3)" in portal_js, "行动流要限量，否则又变成一屏卡片墙"
    for banned in ("素材池", "两库共用", "二级模块"):
        assert banned not in html, f"门户界面文案还留着「{banned}」"
        block = portal_js[portal_js.index("const MODULE_CARDS"):portal_js.index("function buildStats")]
        assert banned not in block, f"门户卡片/数字文案还留着「{banned}」"
    assert ".stat-strip" in read("app.css"), "门户概览应该是紧凑数字条，不是十几张卡片"
