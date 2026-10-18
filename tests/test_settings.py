"""系统设置：推理设备（CUDA / Apple 加速 / CPU）与半精度的存取、降级和界面入口。

这一组盯的是最容易骗人的几处：

1. 设置存了却没人读——界面上选了 CPU，标注其实还在 MPS 上跑；
2. 这台机器没有的设备摆成能按的样子，跑起来才崩在 ultralytics 的英文栈里；
3. 换设备把参数指纹改了，几千张标好的图被判成"参数变了"重跑；
4. 命令行明明写了 ``--device``，却被家目录里一个文件改掉。
"""

import json
import os
import sys
import types
import urllib.error
import urllib.request

import pytest

from autolabel import settings as system
from autolabel.backends import BackendSettings

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "autolabel", "static")
PAGES = ("portal.html", "review.html", "objects.html", "models.html")


def read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as fh:
        return fh.read()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def post(base, path, payload):
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def fake_torch(cuda=(), mps=False, cuda_build="12.4"):
    """造一个假 torch：有哪几张卡、MPS 开不开、这版带不带 CUDA，全由参数说。"""
    module = types.ModuleType("torch")
    module.__version__ = "2.4.0"
    module.version = types.SimpleNamespace(cuda=cuda_build)
    module.cuda = types.SimpleNamespace(
        is_available=lambda: bool(cuda),
        device_count=lambda: len(cuda),
        get_device_name=lambda i: cuda[i],
    )
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: bool(mps)))
    return module


@pytest.fixture()
def probe_off(monkeypatch):
    """默认把探测钉死成"只有 CPU"，别让开发机的 MPS 混进断言。"""
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    system._probe_cache = None
    yield
    system._probe_cache = None


# --------------------------------------------------------------------------- //
# 一、存取：文件只装该装的两项，不认识的值不进门
# --------------------------------------------------------------------------- //

def test_missing_file_falls_back_to_defaults():
    assert system.load() == {"device": "auto", "half": False}
    assert not os.path.exists(system.path())


def test_settings_file_is_the_machine_one_not_the_dataset(tmp_path, monkeypatch):
    """默认落 ~/.autolabel/settings.json；换数据集不该换一份设置。"""
    monkeypatch.delenv("AUTOLABEL_SETTINGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert system.path() == os.path.join(str(tmp_path), ".autolabel", "settings.json")


def test_save_writes_only_the_two_known_keys():
    system.save({"device": "cpu", "half": True, "顺手塞进来的": "x"})
    with open(system.path(), encoding="utf-8") as fh:
        assert json.load(fh) == {"device": "cpu", "half": True}


def test_unknown_device_is_refused_with_the_choices_listed():
    with pytest.raises(ValueError) as caught:
        system.save({"device": "tpu"})
    message = str(caught.value)
    assert "tpu" in message and "cuda" in message and "mps" in message


def test_multi_gpu_spellings_fold_into_cuda():
    """命令行能写 --device 0 / cuda:1，存进设置统一收成 cuda，读的时候再核对卡号。"""
    for spelling in ("0", "1", "cuda:1"):
        assert system.normalize({"device": spelling})["device"] == "cuda"
    assert system.normalize({"device": "乱七八糟"})["device"] == "auto"


def test_broken_file_does_not_take_the_whole_ui_down(monkeypatch):
    os.makedirs(os.path.dirname(system.path()), exist_ok=True)
    with open(system.path(), "w", encoding="utf-8") as fh:
        fh.write("{这文件被谁改坏了")
    assert system.load() == {"device": "auto", "half": False}


# --------------------------------------------------------------------------- //
# 二、探测：有什么卡说什么话，没有的就写清为什么没有
# --------------------------------------------------------------------------- //

def test_detect_reports_cuda_names(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        fake_torch(cuda=("NVIDIA GeForce RTX 4090", "NVIDIA T4")))
    system._probe_cache = None
    facts = system.detect(force=True)
    assert facts["cuda"]["available"] is True
    assert facts["cuda"]["names"] == ["NVIDIA GeForce RTX 4090", "NVIDIA T4"]
    assert facts["cpu"]["available"] is True


def test_auto_row_says_which_card_it_will_actually_use(monkeypatch):
    """自动档必须说"这次会用谁"，否则用户不知道自己实际跑在哪。"""
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    system._probe_cache = None
    facts = system.detect(force=True)
    assert facts["auto"]["names"][0].startswith("本次会用 Apple 加速")


def test_cpu_only_torch_explains_why_cuda_is_greyed_out(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_build=None))
    system._probe_cache = None
    facts = system.detect(force=True)
    assert facts["cuda"]["available"] is False
    assert "不带 CUDA" in facts["cuda"]["reason"]


def test_missing_torch_is_reported_not_raised(monkeypatch):
    """没装 torch 也要给出得体的说法，而不是把 ImportError 甩到界面上。"""
    monkeypatch.setitem(sys.modules, "torch", None)
    system._probe_cache = None
    facts = system.detect(force=True)
    assert facts["cpu"]["available"] is True
    assert "PyTorch" in facts["cuda"]["reason"] and "PyTorch" in facts["mps"]["reason"]


def test_probe_is_cached_until_forced(monkeypatch):
    """探测比读一个 JSON 贵得多：不 force 就用缓存，插上卡想立刻看到就 force。"""
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    system._probe_cache = None
    first = system.detect()
    assert system.detect() is first, "没 force 就该复用同一份探测结果"
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=("RTX 4090",)))
    assert system.detect()["cuda"]["available"] is False
    assert system.detect(force=True)["cuda"]["available"] is True


# --------------------------------------------------------------------------- //
# 三、降级：选了没有的设备不报错，改用能用的并把话说出来
# --------------------------------------------------------------------------- //

def test_resolve_uses_the_chosen_card_when_it_exists(probe_off, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=("RTX 4090",)))
    system._probe_cache = None
    assert system.resolve("cuda") == ("cuda", "")


def test_resolve_falls_back_and_says_so(probe_off):
    device, note = system.resolve("cuda")
    assert device == "cpu"
    assert "不可用" in note and "改用" in note


def test_explicit_cpu_never_warns(probe_off):
    assert system.resolve("cpu") == ("cpu", "")


def test_auto_prefers_cuda_over_mps_over_cpu(probe_off, monkeypatch):
    system._probe_cache = None
    assert system.resolve("auto")[0] == "cpu"
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    system._probe_cache = None
    assert system.resolve("auto")[0] == "mps"
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=("RTX 4090",), mps=True))
    system._probe_cache = None
    assert system.resolve("auto")[0] == "cuda"


def test_saved_setting_is_what_resolve_uses_when_nothing_passed(probe_off):
    system.save({"device": "cuda"})
    device, note = system.resolve("")
    assert device == "cpu" and "CUDA" in note


# --------------------------------------------------------------------------- //
# 四、推理层：设备编号、半精度、参数指纹
# --------------------------------------------------------------------------- //

def test_backends_maps_cuda_to_device_zero(probe_off, monkeypatch):
    from autolabel.backends import resolve_device

    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=("RTX 4090", "T4")))
    system._probe_cache = None
    assert resolve_device("cuda") == "0"
    assert resolve_device("1") == "1", "多卡时用户填的卡号不能被吃掉"


def test_backends_degrades_instead_of_crashing(probe_off, capsys):
    """只有 CPU 的机器上点名要 MPS：降级并说清楚，而不是甩 ultralytics 的英文栈。"""
    from autolabel.backends import resolve_device

    system._probe_cache = None
    assert resolve_device("mps") == "cpu"
    assert "改用" in capsys.readouterr().out


def test_backends_warns_when_asked_card_is_missing(probe_off, monkeypatch, capsys):
    from autolabel.backends import resolve_device

    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=("只有一张卡",)))
    system._probe_cache = None
    resolve_device("3")
    printed = capsys.readouterr().out
    assert "第 3 张" in printed and "只识别到 1 张" in printed


def test_half_precision_only_sticks_on_nvidia(probe_off, capsys):
    from autolabel.backends import resolve_half

    assert resolve_half(True, "0") is True
    assert resolve_half(True, "cpu") is False
    assert "半精度" in capsys.readouterr().out
    assert resolve_half(False, "0") is False


def test_device_and_half_do_not_change_the_relabel_fingerprint():
    """换算力只是换个引擎，不该让几千张已标好的图重跑一遍。"""
    from autolabel.pipeline import settings_signature

    base = settings_signature(BackendSettings())
    assert settings_signature(BackendSettings(device="cpu")) == base
    assert settings_signature(BackendSettings(device="cuda", half=True)) == base
    assert settings_signature(BackendSettings(imgsz=960)) != base, "真正改结果的参数还得算进指纹"


# --------------------------------------------------------------------------- //
# 五、优先级：命令行写明的最大，系统设置只兜底
# --------------------------------------------------------------------------- //

def test_cli_flag_beats_the_saved_setting(probe_off):
    from autolabel.cli import load_settings

    system.save({"device": "cpu"})
    args = types.SimpleNamespace(config=None, device="mps", extra_models=None)
    assert load_settings(args).device == "mps"


def test_saved_setting_applies_when_cli_says_nothing(probe_off):
    from autolabel.cli import load_settings

    system.save({"device": "cpu", "half": True})
    args = types.SimpleNamespace(config=None, device=None, extra_models=None)
    settings = load_settings(args)
    assert (settings.device, settings.half) == ("cpu", True)


def test_ui_jobs_pick_up_the_saved_device(probe_off, monkeypatch):
    """界面上的导入任务不带设备，必须自己补上系统设置里的值，否则那个面板就是摆设。"""
    import autolabel.pipeline as pipeline
    from autolabel.upload import Job, JobStore

    system.save({"device": "cpu", "half": True})
    captured = {}

    def fake_label(root, settings, **kwargs):
        captured["settings"] = settings
        return {"imported": 0}

    monkeypatch.setattr(pipeline, "label_dataset", fake_label)
    store = JobStore.__new__(JobStore)            # 只验参数拼装，不真的跑标注
    store.root = "某数据集"
    store._label(Job(id="j1", kind="label", files=[], params={"model": "yolo11n-pose.pt"}))
    assert (captured["settings"].device, captured["settings"].half) == ("cpu", True)


def test_job_params_beat_the_saved_device(probe_off, monkeypatch):
    """单个任务自己填了设备就该听它的，系统设置只在不填时兜底。"""
    import autolabel.pipeline as pipeline
    from autolabel.upload import Job, JobStore

    system.save({"device": "cpu"})
    captured = {}
    monkeypatch.setattr(pipeline, "label_dataset",
                        lambda root, settings, **kw: captured.setdefault("settings", settings))
    store = JobStore.__new__(JobStore)
    store.root = "某数据集"
    store._label(Job(id="j2", kind="label", files=[], params={"device": "mps"}))
    assert captured["settings"].device == "mps"


# --------------------------------------------------------------------------- //
# 六、接口与界面入口
# --------------------------------------------------------------------------- //

def test_settings_api_round_trip(server, probe_off):
    base, _store = server
    status, data = get(base, "/api/settings")
    assert status == 200
    assert data["saved"] == {"device": "auto", "half": False}
    assert [c["value"] for c in data["choices"]] == ["auto", "cuda", "mps", "cpu"]

    status, data = post(base, "/api/settings", {"device": "cpu"})
    assert status == 200 and data["saved"]["device"] == "cpu"

    status, data = post(base, "/api/settings", {"device": "tpu"})
    assert status == 400 and "tpu" in data["error"]


def test_settings_api_warns_when_the_choice_is_unusable(server, probe_off):
    base, _store = server
    status, data = post(base, "/api/settings", {"device": "cuda"})
    assert status == 200 and "用不了" in data["note"]


def test_get_without_probe_does_not_touch_torch(server, probe_off, monkeypatch):
    """平时切页只读 JSON；探测显卡留到真的打开面板那一刻。"""
    base, _store = server

    def explode(*args, **kwargs):
        raise AssertionError("不该在没带 probe 的时候探测")

    monkeypatch.setattr(system, "detect", explode)
    status, data = get(base, "/api/settings")
    assert status == 200 and data["devices"] == {}


def test_settings_entry_is_wired_on_every_page():
    for page in PAGES:
        html = read(page)
        assert 'id="btn-settings"' in html, f"{page} 没有系统设置入口"
        assert 'href="#ic-gear"' in html, f"{page} 缺齿轮图标"


def test_frontend_reads_the_device_list_from_the_server():
    """设备清单由后端给，前端不许自己硬编一份，否则装了 4090 界面还说不认识。"""
    common = read("common.js")
    assert '"/api/settings"' in common
    assert "saveSettings({ device:" in common
    assert "NVIDIA GeForce" not in common, "前端不该写死显卡型号"
