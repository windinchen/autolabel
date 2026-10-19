"""新壳层的界面契约：五个页面必须是同一套壳层、状态词只有一份、筛选项都得真的能筛、
门户不许把实现细节当主词。

这些都是 UI 重构里真踩过的坑：状态词表只写在姿态页导致提示词页整页报错，
右侧动作区用 sticky 把目标列表整个盖住，筛选下拉里多一个没人处理的值，
以及门户/模型页迟迟没跟上新壳、顶栏和左导航条各长各的。
"""
import json
import os
import re
import shutil
import subprocess

import pytest

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
DOC_PAGES = {"portal.html": "portal.js", "pool.html": "pool.js", "models.html": "models.js"}


def test_document_pages_share_the_same_shell():
    """工作台 / 素材池 / 模型管理不许留旧顶栏：左导航条、命令面板、速查表、溢出菜单一个都不能少。"""
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
             "portal.html": "portal.js", "pool.html": "pool.js", "models.html": "models.js"}


def test_appearance_controls_are_wired_on_every_page():
    """外观是全站设置：四页都得有首屏脚本、系统设置入口，速查表也得写上快捷键。"""
    css = read("app.css")
    assert ':root[data-theme="light"]' in css, "没有白天主题变量块"
    assert ".seg button" in css, "缺分段选择器样式"
    for step in ('[data-fs="0"]', '[data-fs="2"]', '[data-fs="3"]'):
        assert step in css, f"缺字号档位 {step}"
    for page in ALL_PAGES:
        html = read(page)
        assert 'id="btn-settings"' in html, f"{page} 的 ⋯ 菜单没有系统设置入口"
        assert "autolabel.appearance" in html, f"{page} 缺首屏外观脚本，切页会闪一下默认主题"
        assert "dataset.theme" in html and "dataset.fs" in html, f"{page} 首屏脚本没写 data-theme / data-fs"
        for advertised in ("⌥T", "⌥=", "⌥-", "⌥S"):
            assert advertised in html, f"{page} 速查表没写 {advertised}"
    common = read("common.js")
    for api in ("function setThemePref", "function toggleTheme", "function stepFont",
                "function renderSettingsPanel", "function canvasPalette"):
        assert api in common, f"common.js 缺 {api}()"
    assert "appearanceCommands()" in read("common.js").split("function registerShellCommands")[1], \
        "外观动作没进命令面板，⌘K 搜不到"


def test_appearance_settings_live_in_the_settings_panel_only():
    """白天/夜晚与字号收进「系统设置」一处：菜单里不许再散落一份裸控件。

    同一件事有两个入口，改天就会有一个忘了同步——界面偏好只留一个家。
    """
    common = read("common.js")
    assert "appearance-ctl" not in common, "菜单内联的外观控件位还在，说明没收进系统设置"
    for page in ALL_PAGES:
        assert "appearance-ctl" not in read(page), f"{page} 还留着菜单内的外观控件位"
    panel = common.split("function buildSettingsModal()", 1)[-1]
    for control in ('id="set-theme"', 'id="set-font"', 'id="set-devices"', 'id="set-half"'):
        assert control in panel, f"系统设置面板缺 {control}"
    assert "initSettings()" in common.split("function initNav()", 1)[1], "系统设置入口没接线"


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
    for banned in ("两库共用", "二级模块", "素材索引"):
        assert banned not in html, f"门户界面文案还留着「{banned}」"
        block = portal_js[portal_js.index("const MODULE_CARDS"):portal_js.index("function buildStats")]
        assert banned not in block, f"门户卡片/数字文案还留着「{banned}」"
    # 素材池是正式模块：门户只给一句摘要 + 两个入口，批次清单不许在门户再留一份
    assert 'id="sec-pool"' in html and 'href="/pool"' in html and 'href="/pool?upload=1"' in html
    assert "renderPoolBrief(" in portal_js and "renderPoolBrief(ds);" in portal_js
    for leftover in ('id="sec-batches"', 'id="portal-batches"', "renderBatches(", "batchState"):
        assert leftover not in html + portal_js, f"门户还留着批次管理：{leftover}"
    assert ".stat-strip" in read("app.css"), "门户概览应该是紧凑数字条，不是十几张卡片"


# --------------------------------------------------------------------------- //
# 模型管理：目录只看已下载、备注说明已取消
# --------------------------------------------------------------------------- #
def run_models_function(name, state, argument, extra=()):
    """把 models.js 里依赖 modelState 的纯函数抠出来交给 node 求值。

    过滤规则用正则猜是猜不准的：一个 `||` 写错，「搜到了却看不见」就得靠肉眼逛十遍。
    ``extra`` 是这条函数还要用的顶层常量名（如 MODULE_DEFAULT_TARGETS）：原文里把那段
    字面量抠出来交给 node 求值，测试断的就是界面真正会用的那份映射，不是测试自己抄的副本。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("本机没有 node，跳过前端纯函数求值")
    source = read("models.js")
    params, body = source.split("function %s(" % name, 1)[1].split(") {", 1)
    body = body.split("\n}", 1)[0]
    literals = []
    for const_name in extra:
        block = re.search(r"const %s = (\[.*?\n\]);" % const_name, source, re.S)
        assert block, f"models.js 里找不到常量 {const_name}"
        literals.append(block.group(1))
    names = ["modelState"] + [p.strip() for p in params.split(",")] + list(extra)
    # 函数体与中文文案走环境变量，别在 shell 引号里被折腾一遍
    script = ("const names = JSON.parse(process.argv[1]);"
              "const values = JSON.parse(process.argv[2]);"
              "const held = process.argv.slice(3).map((text) => Function(`return (${text})`)());"
              "const fn = Function(...names, process.env.FN_BODY);"
              "process.stdout.write(JSON.stringify(fn(...values, ...held)));")
    done = subprocess.run([node, "-e", script, json.dumps(names), json.dumps([state, argument]),
                           *literals], capture_output=True, text=True, timeout=60,
                          env=dict(os.environ, FN_BODY=body))
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


CATALOG_ROWS = [
    {"id": "yolo11n", "task": "detect", "installed": True},
    {"id": "yolov8x", "task": "detect", "installed": False},
    {"id": "ollama-qwen", "task": "vlm", "installed": False, "ready": True},
]


def test_catalog_defaults_to_the_whole_official_catalog():
    """目录默认给全量（下载入口比「只看已下载」更常被找），勾上开关才收窄到本机已有的。"""
    html = read("models.html")
    source = read("models.js")

    assert 'class="switch" id="model-installed-only"' in html, "「只看已下载」默认得是关的"
    assert 'aria-checked="false"' in html, "开关的无障碍状态要和默认一致"
    assert "installedOnly: false" in source
    assert "visibleCatalogRows(" in source and "toggleInstalledOnly()" in source
    assert "另有" in source and "未下载没显示" in source, "藏掉多少条要说出来，否则像目录只剩几个"

    default = run_models_function("visibleCatalogRows",
                                  {"installedOnly": False, "query": "", "filter": ""}, CATALOG_ROWS)
    assert [row["id"] for row in default["shown"]] == [row["id"] for row in CATALOG_ROWS], default
    assert default["hidden"] == 0, "默认就该看全量：没下载的也得能点下载"
    narrowed = run_models_function("visibleCatalogRows",
                                   {"installedOnly": True, "query": "", "filter": ""}, CATALOG_ROWS)
    assert [row["id"] for row in narrowed["shown"]] == ["yolo11n", "ollama-qwen"], narrowed
    assert narrowed["hidden"] == 1


def test_searching_the_catalog_beats_the_installed_only_switch():
    """搜到了却看不见是最气人的一种「搜不到」：搜索或筛任务时开关必须让路。"""
    for state in ({"installedOnly": True, "query": "yolov8x", "filter": ""},
                  {"installedOnly": True, "query": "", "filter": "detect"},
                  {"installedOnly": False, "query": "", "filter": ""}):
        got = run_models_function("visibleCatalogRows", state, CATALOG_ROWS)
        assert len(got["shown"]) == len(CATALOG_ROWS), (state, got)
        assert got["hidden"] == 0, (state, got)


def test_model_page_has_no_leftover_note_ui():
    """备注说明整条功能删了：输入框、保存按钮、接口调用都不许留残骸。"""
    html = read("models.html")
    source = read("models.js")

    for leftover in ("model-reg-note", "model-note-row", "note-save", "saveModelNote",
                     "bindModelNote"):
        assert leftover not in html + source, f"备注功能还留着 {leftover}"


def test_local_weight_list_only_lists_real_files():
    """「本机权重」按磁盘上真有的文件列，未下载的预置档不在这里占行。"""
    source = read("models.js")

    assert "function localWeightRows()" in source
    assert "for (const info of localWeightRows())" in source or \
        "const rows = localWeightRows();" in source


# --------------------------------------------------------------------------- //
# 1.4.10：模块改名、模型列表接管「默认」、目录可更新
# --------------------------------------------------------------------------- #

def read_repo_file(*parts):
    root = os.path.dirname(STATIC)
    with open(os.path.join(root, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_third_module_is_named_detection_labeling_everywhere():
    """模块对外一律叫「目标检测标注」，数据库名照旧是「目标检测库」：库名会进文件名和导出。"""
    common = read("common.js")
    portal = read("portal.js")
    objects = read("objects.js")
    detect_html = read("detect.html")
    library = read_repo_file("library.py")

    assert 'name: "目标检测标注"' in common, "左侧导航与顶栏用的就是 MODULES 里这个名字"
    assert 'name: "目标检测标注"' in objects and 'runLabel: "批量目标检测标注"' in objects
    assert "<title>目标检测标注</title>" in detect_html
    assert '"批量目标检测"' not in objects + detect_html + portal
    assert 'LIB_NAMES = {' in library and 'LIB_DETECT: "目标检测标注"' in library
    assert "目标检测库" in read_repo_file("static", "portal.html") or "目标检测库" in portal


def test_weight_section_is_the_single_place_that_sets_defaults():
    """「本机权重」改名「模型列表」，并把设默认这个动作整个接过来：一件事不许有两个入口。"""
    html = read("models.html")
    source = read("models.js")

    assert ">模型列表<" in html and "本机权重" not in html, "区块标题和说明得一起改过来"
    assert "设为XX默认" in html or "设为" in html
    assert "function renderModuleDefaults(" in source and "setModuleDefault(" in source
    # 目录卡片只剩下载按钮：以前目录和权重列表都能设默认，两处早晚对不上
    assert '<button class="def"></button>' not in source
    assert "setDetectDefault(" not in source and "setModelDefault(" not in source
    assert "function setModuleDefault(" in source
    css = read("app.css")
    assert ".module-defaults" in css and ".mod-default.on" in css, "默认状态要有描边，不能只靠文字"


def test_each_module_offers_only_the_weight_kinds_it_can_run():
    """哪档权重能被哪个标注模块设为默认，读的是 models.js 里那份真映射。"""
    def labels(task):
        got = run_models_function("modulesForWeight", {}, {"task": task},
                                 extra=("MODULE_DEFAULT_TARGETS",))
        return [item["label"] for item in got]

    assert labels("pose") == ["姿态标注"]
    assert labels("detect") == labels("segment") == labels("obb") == ["目标检测标注"]
    assert labels("classify") == [], "图像分类只出一张图一个标签，两个模块都不吃它"
    # 认不出任务的自定义权重两边都给，真正能不能跑交给后端把关
    assert labels("") == ["姿态标注", "目标检测标注"]


def test_catalog_refresh_is_wired_end_to_end():
    """「更新目录」从按钮到接口到那句来源说明，一条链都得在。"""
    html = read("models.html")
    source = read("models.js")
    review = read_repo_file("review.py")

    assert 'id="model-catalog-refresh"' in html and 'id="model-catalog-source"' in html
    assert "async function refreshCatalog()" in source
    assert 'postJson("/api/modelhub/refresh"' in source
    assert 'renderCatalogSource();' in source
    # 按钮里是「图标 + 两个 span」：整块 textContent 一写图标就没了，只能改内层那两个
    assert 'id="model-catalog-refresh-text"' in html and 'id="model-catalog-refresh-time"' in html
    refresh = source.split("async function refreshCatalog()", 1)[1].split("\n}", 1)[0]
    assert "button.textContent =" not in refresh, "整块写 textContent 会把按钮里的图标抹掉"
    assert "function paintCatalogRefresh()" in source and "paintCatalogRefresh();" in source
    assert "更新于 " in source, "按钮上要写清上次核对的时间，不然没人知道该不该点"
    assert "从未核对" in source, "从没核对过的情况也得说清楚"
    assert '"/api/modelhub/refresh"' in review and "refresh_model_catalog" in review
    assert "更新模型目录" in source, "命令面板也得找得到这件事"


def test_pose_dropdown_only_offers_pose_weights():
    """姿态标注的模型下拉框只摆姿态与自定义权重，而且只摆本机已有的：
    混一个检测权重进来，跑起来只会回一句「这个权重没有关键点」。"""
    source = read("app.js")
    body = source.split("function renderModelOptions", 1)[1].split("\n}", 1)[0]

    assert 'for (const capability of ["pose", "vlm", "custom_task"])' in body
    assert "all.filter((r) => r.ready && r.compatible !== false)" in body
    for other in ("detect", "segment", "obb", "classify"):
        assert other not in body, f"姿态页的下拉框不该认识「{other}」权重"


# --------------------------------------------------------------------------- //
# 最左边那一竖条：只管本页动作，模块跳转只有一处代码在生成
# --------------------------------------------------------------------------- //
PAGES = ("portal.html", "pool.html", "review.html", "objects.html", "detect.html", "models.html")
MODULE_HREFS = ('href="/"', 'href="/pool"', 'href="/pose"', 'href="/objects"',
                'href="/detect"', 'href="/models"')


def rail_block(html):
    block = re.search(r'<div id="rail">.*?\n  </div>', html, re.S)
    assert block, "页面里没有左侧竖条"
    return block.group(0)


def test_module_jumps_are_generated_in_one_place_with_one_placeholder_per_page():
    """模块跳转只由 MODULES 渲染，各页在「⋯」菜单里留一个占位。

    以前每页自己写死一排，加一个库要改五个文件：模型管理页就漏过目标检测，
    兜底代码又把补出来的那条插到「关于」底下，顺序和各页都不一样。
    """
    common = read("common.js")
    assert "ensureRailModules" not in common, "竖条补图标的老路已废，不许引回来"
    for page in PAGES:
        assert '<div id="menu-modules"></div>' in read(page), f"{page} 缺模块跳转占位"


def test_rail_holds_only_this_pages_own_actions():
    """竖条不再挂模块跳转：五枚同形图标并排放，反而看不出自己在哪一页。"""
    for page in PAGES:
        rail = rail_block(read(page))
        for href in MODULE_HREFS:
            assert href not in rail, f"{page} 的竖条里还留着模块跳转 {href}"
        assert "rail-spacer" in rail, f"{page} 的竖条没把帮助钉在底部"


def test_import_has_exactly_one_entrance_and_it_is_top_right():
    """「导入素材」六页只有右上角一处，模型管理页压根不管素材。

    以前它同时挂在右上角和竖条上（工作台还有第三处：素材池摘要卡里的按钮），
    两颗长得一样的图标去的是同一个弹层，用户只能猜哪颗是正主。现在竖条只管
    本页面板开关，导入的位置六页统一在右上角，快捷键（W / U）不变。
    """
    for page in PAGES:
        rail = rail_block(read(page))
        assert 'id="rail-upload"' not in rail, f"{page} 的竖条又长出第二颗导入按钮"
        assert "pool?upload=1" not in rail, f"{page} 的竖条又挂回导入捷径"
    assert 'id="btn-import" href="/pool?upload=1"' in read("portal.html"), \
        "工作台的导入要指向素材池，别再塞进某个库"
    for page in ("pool.html", "review.html", "objects.html", "detect.html"):
        assert 'id="btn-upload"' in read(page), f"{page} 右上角缺导入按钮"
    assert 'id="btn-upload"' not in read("models.html"), "模型管理页不该管素材"
    assert 'onUpload' not in read("common.js"), "竖条导入的绑定已废，别引回来"


def test_portal_next_step_separates_annotating_from_reviewing():
    """「下一步」得先分清还没标和标了没复核：从素材池刚加进来的库一张都没标。

    上一版这种库会被写成「有 2 帧本机权重出的框没复核」，可它一个框都还没有，
    按这句话点进复核队列只会看到一片空白，等于让主页说了句假话。
    """
    source = read("portal.js")
    assert "poseUntouched" in source and "detectUntouched" in source
    assert 'values.pose_unreviewed && values.pose_labeled' in source, "没标的库不该说「骨骼没复核」"
    assert 'values.detect_unreviewed && values.detect_labeled' in source, "同上，检测库口径一致"
    assert "还没生成骨骼" in source and "还没跑过" in source
    assert 'chip: "目标检测标注"' in source, "模块名要写全，别再用「检测标注」这个第三名字"


def test_help_button_is_the_same_button_on_every_page():
    """六页那颗「说明」必须是同一个 id、同一处接线：姿态页曾经点了没反应。

    接线只在 :func:`initShellChrome` 里做一次，各页不再自己补一条 ``onclick``，
    否则加一页要么漏绑（点了没动静态按钮），要么两条语句互相覆盖看不出来。
    """
    for page in PAGES:
        assert 'id="rail-help"' in read(page), f"{page} 的说明按钮没跟着统一 id"
        assert 'id="btn-help"' not in read(page), f"{page} 还留着老 id，那根线已经不在这里了"
    common = read("common.js")
    assert 'bind("rail-help", () => openHelp(true));' in common
    assert 'bind("btn-help"' not in common
    for page in ("portal.js", "pool.js", "models.js"):
        assert '$("rail-help").onclick' not in read(page), f"{page} 又自己绑了一遍说明按钮"


def test_icons_say_one_thing_each():
    """同一枚图标在不同页不许表示两件不同的事：类别管理和批次管理不能共用一摞纸。"""
    for page in ("objects.html", "detect.html"):
        rail = rail_block(read(page))
        assert 'id="rail-fields" title="字段管理（1）"' in rail or \
               'id="rail-fields" title="类别管理（1）"' in rail, page
        assert "#ic-list" in rail, f"{page} 的类别/字段管理还在用批次那摞纸"
        assert "#ic-layers" not in rail, f"{page} 的竖条里批次图标串场了"


def test_selected_state_uses_the_right_language():
    """天蓝说「你在这／选中的是这个」，琥珀说「这步该做／开关开着」，底色差一档在白天主题下等于没选。"""
    css = read("app.css")
    tab = re.search(r"\.nav-module\.active \{([^}]*)\}", css)
    assert tab, "找不到顶栏当前模块的样式"
    assert "--sky" in tab.group(1), "顶栏当前模块要用选中色，别只把底色调深一档"
    toggle = re.search(r"\.iconbtn\.on \{([^}]*)\}", css)
    assert toggle and "--amber" in toggle.group(1), "开关态的琥珀描边没了，28px 的小块亮不起来"
    assert "border-color" in toggle.group(1)


def test_pose_page_got_the_batch_panel_the_other_two_already_had():
    """三个标注页都要能对着「库里已有的图」直接批量标注，快捷键同一枚 G。"""
    html = read("review.html")
    for anchor in ('id="btn-run"', 'id="rail-run"', 'id="run-modal"', 'id="run-scope"',
                   'id="run-start"', 'id="run-log"', "批量姿态标注"):
        assert anchor in html, f"review.html 缺 {anchor}"
    source = read("app.js")
    assert 'postJson("/api/pose/run"' in source and "function initRun()" in source
    assert 'case "g": case "G": openRun(true)' in source
    assert "initRun();" in source, "面板没接线的话按 G 只会静悄悄什么也不做"
