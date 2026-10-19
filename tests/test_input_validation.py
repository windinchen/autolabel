"""第 1 轮体检（输入校验与错误提示）的回归测试。

盯四件事：

1. 界面传来的开关按字面意思判——``bool("false") == True`` 这个坑不能再踩；
2. 数字参数（置信度、张数上限、抽帧频率、最长边）填错当场说中文，
   不许穿到 ultralytics / PIL 里炸出英文 traceback 让异步任务悄悄失败；
3. 参数错的报错要盖过"库里还没图"，否则真正写错的框被盖住；
4. 目标检测库说"类别"、提示词库说"字段"，别再对检测库说"去界面里新建字段"。
"""
import inspect
import io
import json
import os
import urllib.error
import urllib.request

import pytest

from autolabel import fields as fields_module
from autolabel.library import LIB_DETECT
from autolabel.video import extract_frames

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "autolabel", "static")


def read_static(name):
    with io.open(os.path.join(STATIC, name), encoding="utf-8") as handle:
        return handle.read()


def post(base, path, payload):
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# --------------------------------------------------------------------------- //
# 一、半精度开关：关就是关
# --------------------------------------------------------------------------- //
@pytest.mark.parametrize("spelling", ["false", "False", " FALSE ", "0", "off", "关", "no", 0, False])
def test_the_ui_saying_false_actually_turns_half_precision_off(server, spelling):
    """界面上关掉半精度，存下来必须真是关的。

    ``bool("false")`` 在 Python 里是 ``True``：以前界面关掉半精度、请求带字符串
    ``"false"``，存进去是**开着**的，而且界面还回一句"已保存"。
    """
    base, _store = server
    status, data = post(base, "/api/settings", {"half": spelling})
    assert status == 200, data
    assert data["saved"]["half"] is False, spelling
    status, data = get(base, "/api/settings")
    assert data["saved"]["half"] is False


@pytest.mark.parametrize("spelling", ["true", "1", "on", "开", "yes", 1])
def test_the_ui_saying_true_turns_half_precision_on(server, spelling):
    base, _store = server
    status, data = post(base, "/api/settings", {"half": spelling})
    assert status == 200 and data["saved"]["half"] is True, (spelling, data)


def test_a_word_that_is_not_a_switch_is_refused(server):
    """看不懂的开关要报错，别按"开"悄悄存下去。"""
    base, _store = server
    status, data = post(base, "/api/settings", {"half": "也许"})
    assert status == 400, data
    assert "半精度" in data["error"], data


# --------------------------------------------------------------------------- //
# 二、批量标注的数字参数：中文报错，别丢给底层
# --------------------------------------------------------------------------- //
def test_detect_run_rejects_conf_outside_the_unit_interval(server):
    """``conf=-1`` 以前收 202，等异步任务起来才吐英文栈。"""
    base, _store = server
    for value, phrase in [(-1, "置信度阈值不能小于"), (1.5, "置信度阈值不能大于")]:
        status, data = post(base, "/api/detect/run", {"conf": value})
        assert status == 400, (value, data)
        assert phrase in data["error"], (value, data)


def test_detect_run_rejects_a_limit_that_is_not_a_whole_number(server):
    base, _store = server
    status, data = post(base, "/api/detect/run", {"limit": "七个"})
    assert status == 400 and "标注张数上限要填数字" in data["error"], data
    status, data = post(base, "/api/detect/run", {"limit": 1.5})
    assert status == 400 and "标注张数上限要填整数" in data["error"], data


def test_a_bad_number_beats_the_empty_library_message(server):
    """空库 + 填错参数时报参数错——"库里还没图"会把真正写错的框盖掉。"""
    base, store = server
    assert len(store.detect_index.items) == 0, "本用例的前提是检测库为空"
    status, data = post(base, "/api/detect/run", {"conf": 3})
    assert status == 400 and "置信度阈值" in data["error"], data
    assert "还没有图片" not in data["error"], data


@pytest.mark.parametrize("key,phrase", [("fps", "视频抽帧频率要大于 0"),
                                        ("max_frames", "视频最多抽帧数要大于 0"),
                                        ("max_side", "图片最长边像素要填数字")])
def test_detect_upload_validates_its_numbers_early(server, key, phrase):
    """三个数以前直接 ``float(...)``/``int(...)``，报错是英文的而且发生在异步任务里。"""
    base, _store = server
    value = {"fps": 0, "max_frames": -3, "max_side": "一大圈"}[key]
    status, data = post(base, "/api/detect/upload", {"files": ["片段.mp4"], key: value})
    assert status == 400, (key, data)
    assert phrase in data["error"], (key, data)


def test_prompt_upload_shares_the_same_number_guard(server):
    """两个上传入口以前是复制粘贴的同一句 ``float(payload.get("fps") or 0.5)``。

    走 ``label=false``（只入库不标注）那条：提示词库要先有多模态配置才谈得上标注，
    而抽帧频率这类数字跟标不标注无关，填错了就该当场说。
    """
    base, _store = server
    status, data = post(base, "/api/prompt/upload", {"files": ["a.jpg"], "fps": 0, "label": False})
    assert status == 400 and "视频抽帧频率要大于 0" in data["error"], data
    status, data = post(base, "/api/prompt/upload", {"files": ["a.jpg"], "max_side": 0, "label": False})
    assert status == 400 and "图片最长边像素要大于 0" in data["error"], data
    status, data = post(base, "/api/prompt/upload", {"files": ["a.jpg"], "max_frames": "many",
                                                     "label": False})
    assert status == 400 and "视频最多抽帧数要填数字" in data["error"], data


def test_nan_is_not_a_number_you_can_label_with(server):
    """``float("NaN")`` 能过，但 NaN 存进配置比报错难查得多。"""
    base, _store = server
    status, data = post(base, "/api/detect/run", {"conf": "NaN"})
    assert status == 400 and "置信度阈值要填数字" in data["error"], data


# --------------------------------------------------------------------------- //
# 三、两套叫法：检测库说"类别"，提示词库说"字段"
# --------------------------------------------------------------------------- //
def test_detect_library_stops_calling_classes_fields(dataset):
    """对检测库说"未知字段……在界面里新建"是谎话：那页只有「从权重类别表导入」。"""
    for i in range(80):
        fields_module.upsert_field(dataset, name=f"cls{i}", lib=LIB_DETECT)
    with pytest.raises(ValueError) as caught:
        fields_module.resolve_fields(dataset, ["ghost"], lib=LIB_DETECT)
    message = str(caught.value)
    assert "未知类别 ghost" in message, message
    assert "未知字段" not in message and "新建" not in message, message
    assert "从权重类别表导入" in message, message
    # 80 个 COCO 名全列出来会把真正那句"这个名字没导入"埋掉
    assert "共 80 个" in message and message.count("cls") <= 12, message


def test_objects_library_keeps_the_word_field(dataset):
    """提示词库的措辞是原来那句，界面里确实有「新建字段」。"""
    with pytest.raises(ValueError) as caught:
        fields_module.resolve_fields(dataset, ["ghost"])
    assert "未知字段 ghost" in str(caught.value)
    assert "新建" in str(caught.value)


def test_web_delete_reports_the_right_word_per_library(server):
    """界面上删一个不存在的类别，报错要按那一页的叫法说。"""
    base, _store = server
    status, data = post(base, "/api/fields/delete", {"name": "ghost", "lib": "detect"})
    assert status == 400 and "未知类别" in data["error"], data
    assert "未知字段" not in data["error"], data
    status, data = post(base, "/api/fields/delete", {"name": "ghost"})
    assert status == 400 and "未知字段" in data["error"], data


# --------------------------------------------------------------------------- //
# 四、界面读数与默认值：同一件事不许有两个数
# --------------------------------------------------------------------------- //
def test_the_frame_cap_default_is_the_same_number_everywhere(server, monkeypatch):
    """界面不填「最多帧数」时，必须用命令行/抽帧函数同一个默认，别再 300 对 600。"""
    from autolabel.upload import Job

    base, store = server
    captured = []

    def fake_submit(kind, files, params=None):
        captured.append(dict(params or {}))
        return Job(id="jobx", kind=kind, files=list(files), params=dict(params or {}))

    monkeypatch.setattr(store.jobs, "submit", fake_submit)
    canonical = inspect.signature(extract_frames).parameters["max_frames"].default
    status, data = post(base, "/api/detect/upload", {"files": ["片段.mp4"]})
    assert status == 202, data
    assert captured[0]["max_frames"] == canonical == 600, captured
    # 其余两个数同样不许在界面与底层各写一套
    assert captured[0]["fps"] == inspect.signature(extract_frames).parameters["fps"].default
    assert captured[0]["max_side"] == inspect.signature(extract_frames).parameters["max_side"].default


@pytest.mark.parametrize("page", ["review.html", "objects.html", "detect.html"])
def test_upload_forms_default_to_the_same_frame_cap(page):
    """三个上传表单的默认值也要是同一个数，不然换个页面结果就变了。"""
    assert 'id="up-max-frames" value="600"' in read_static(page), page


def test_the_frontend_stops_reading_numbers_with_or():
    """``parseFloat(v) || 默认`` 会把用户填的 0 当假值吞掉，改读成诚实的 readNumber()。"""
    common = read_static("common.js")
    assert "const readNumber = (id, fallback)" in common
    assert "const DEFAULT_VAL_RATIO = 0.2;" in common
    objects = read_static("objects.js")
    for stale in ['|| 0.25', 'parseInt($("up-max-frames")']:
        assert stale not in objects, stale
    assert 'readNumber("up-max-frames", 600)' in objects
    # 导出比例以前在三个文件里各写死一遍 0.2
    for name in ("app.js", "objects.js", "portal.js"):
        assert "val_ratio: 0.2" not in read_static(name), name


def test_the_export_ratio_is_actually_adjustable():
    """服务端一直支持调 val_ratio，界面上却只有写死的 0.2：门户导出面板现在有输入框。"""
    assert 'id="export-val-ratio"' in read_static("portal.html")
    assert 'readNumber("export-val-ratio", DEFAULT_VAL_RATIO)' in read_static("portal.js")


# --------------------------------------------------------------------------- //
# 五、兜底提示也得是中文，而且不许复读
# --------------------------------------------------------------------------- //
def test_empty_library_error_says_how_once(dataset):
    """空库时报错要说清去哪新建，但一遍就够，别把提示复读两遍。"""
    with pytest.raises(ValueError) as caught:
        fields_module.resolve_fields(dataset, ["ghost"])
    message = str(caught.value)
    assert "未知字段 ghost" in message and "新建" in message, message
    assert message.count("新建") == 1, message
    assert message.count("fields --add") == 1, message


def test_unknown_endpoint_says_so_in_chinese(server):
    """界面与服务端版本不匹配时打到的空接口，以前只弹一个英文 not found。"""
    base, _store = server
    status, data = get(base, "/api/definitely-not-an-endpoint")
    assert status == 404, data
    assert "没有这个接口" in data["error"] and "/api/definitely-not-an-endpoint" in data["error"], data
