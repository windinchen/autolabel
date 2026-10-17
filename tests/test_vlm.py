"""多模态大模型标注后端：提示词、图片编码、JSON 解析、后端选择与端到端标注。"""

import base64
import json

import numpy as np
import pytest

from autolabel import models, vlm
from autolabel.backends import BackendSettings, PoseEnsemble, build_backend
from autolabel.yolo_pose import VIS_UNLABELED, VIS_VISIBLE, parse_label_file


def chat_body(reply: str) -> str:
    return json.dumps({"choices": [{"message": {"content": reply}}]})


def person_json(scale: float = 1000.0) -> dict:
    """一个站姿的人，坐标按 scale 缩放（1000 表示整数相对坐标，1 表示小数）。"""
    def fmt(value: float):
        return round(value) if scale >= 10 else round(value, 3)

    raw = [
        (0.50, 0.12, 2),   # nose
        (0.54, 0.10, 2),   # left_eye
        (0.46, 0.10, 2),   # right_eye
        (0.58, 0.12, 2),   # left_ear
        (0.42, 0.12, 2),   # right_ear
        (0.62, 0.30, 2),   # left_shoulder
        (0.38, 0.30, 2),   # right_shoulder
        (0.70, 0.48, 1),   # left_elbow
        (0.30, 0.48, 1),   # right_elbow
        (0.74, 0.64, 2),   # left_wrist
        (0.26, 0.64, 2),   # right_wrist
        (0.56, 0.70, 2),   # left_hip
        (0.44, 0.70, 2),   # right_hip
        (0.58, 0.86, 2),   # left_knee
        (0.42, 0.86, 2),   # right_knee
        (0.60, 0.98, 2),   # left_ankle
        (0.40, 0.98, 2),   # right_ankle
    ]
    return {"keypoints": [[fmt(x * scale), fmt(y * scale), v] for x, y, v in raw]}


@pytest.fixture()
def vlm_config(tmp_path, monkeypatch):
    """把多模态配置写到 tmp，并预置一条本地配置。"""
    monkeypatch.setenv("AUTOLABEL_VLM_CONFIG", str(tmp_path / "configs" / "vlm.json"))
    return tmp_path


@pytest.fixture()
def profile(vlm_config):
    saved = models.upsert_profile({
        "name": "本地桩", "provider": "ollama", "kind": "local",
        "base_url": "http://127.0.0.1:11434/v1", "model": "qwen2.5vl:7b", "api_key": "secret-key",
    })
    return models.get_profile(saved["id"])  # 内部副本才带 api_key 明文


# --------------------------------------------------------------------------- #
# 提示词与图片编码
# --------------------------------------------------------------------------- #

def test_prompt_lists_all_keypoints_and_json_shape():
    prompt = vlm.build_prompt()
    assert "nose" in prompt and "right_ankle" in prompt
    assert "0-1000" in prompt
    assert '"people"' in prompt
    assert "共17项" in prompt


def test_encode_image_is_jpeg_data_url_downscaled():
    image = np.zeros((2400, 1200, 3), dtype=np.uint8)
    data_url = vlm.encode_image(image, max_side=640)
    assert data_url.startswith("data:image/jpeg;base64,")

    from PIL import Image
    from io import BytesIO

    raw = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(BytesIO(raw)) as im:
        assert im.format == "JPEG"
        assert max(im.size) == 640  # 长边压到上限，控制请求体


# --------------------------------------------------------------------------- #
# 解析模型回复
# --------------------------------------------------------------------------- #

def test_parse_people_normalizes_scaled_coordinates():
    persons = vlm.parse_people(json.dumps({"people": [person_json(1000.0)]}))
    assert len(persons) == 1
    person = persons[0]
    assert len(person.keypoints) == 17
    assert person.keypoints[0].x == pytest.approx(0.5, abs=0.01)
    assert person.keypoints[0].v == VIS_VISIBLE
    assert person.score == vlm.DEFAULT_SCORE
    assert person.flags == [vlm.VLM_FLAG]
    # bbox 必须包住所有可见点，否则训练时会“点在线段外、框没包住”
    xs = [k.x for k in person.keypoints if k.v != VIS_UNLABELED]
    ys = [k.y for k in person.keypoints if k.v != VIS_UNLABELED]
    x1, y1, x2, y2 = person.xyxy(1, 1)
    assert x1 <= min(xs) and x2 >= max(xs)
    assert y1 <= min(ys) and y2 >= max(ys)


def test_parse_people_accepts_markdown_fence_and_decimal_coords():
    reply = "好的：\n```json\n" + json.dumps({"people": [person_json(1.0)]}) + "\n```\n就这些"
    persons = vlm.parse_people(reply)
    assert len(persons) == 1
    assert persons[0].keypoints[5].x == pytest.approx(0.62, abs=0.01)


def test_parse_people_pads_missing_keypoints_and_drops_empty_ones():
    entry = {"keypoints": [[500, 120, 2], [540, 100, 2], [0, 0, 2]]}
    persons = vlm.parse_people(json.dumps({"people": [entry]}))
    assert len(persons) == 1
    kpts = persons[0].keypoints
    assert len(kpts) == 17
    assert kpts[2].v == VIS_UNLABELED          # 全零点当成“没标”
    assert all(k.v == VIS_UNLABELED for k in kpts[3:])

    only_zero = {"people": [{"keypoints": [[0, 0, 0]] * 17}]}
    assert vlm.parse_people(json.dumps(only_zero)) == []


def test_parse_people_clamps_visibility_and_uses_model_bbox():
    entry = {"bbox": [100, 100, 900, 950],
             "keypoints": [[500, 300, 7]] + [[0, 0, 0]] * 16}
    person = vlm.parse_people(json.dumps({"people": [entry]}))[0]
    assert person.keypoints[0].v == VIS_VISIBLE  # 非法可见性收到 2
    assert person.w > 0.5                        # 用了模型给的框


def test_parse_people_rejects_garbage():
    with pytest.raises(vlm.VLMError) as exc:
        vlm.parse_people("图里没有人，无法标注。")
    assert "JSON" in str(exc.value)
    with pytest.raises(vlm.VLMError):
        vlm.parse_people('{"people": [')
    with pytest.raises(vlm.VLMError):
        vlm.parse_people('{"people": "none"}')


def test_parse_people_recovers_from_broken_json():
    """模型偶发漏逗号时，要把能救的人救回来，而不是整张判失败。"""
    broken = ('{"people": [{"bbox": [62, 375, 295, 835], "keypoints": [[178, 410, 2], [190, 400, 2]'
              ' [165, 402, 2], [205, 412, 1]]}, {"keypoints": [[10, 20, 2], [30, 40, 1]]}]}')
    persons = vlm.parse_people(broken)
    assert len(persons) == 2
    assert persons[0].keypoints[0].x == pytest.approx(0.178, abs=0.001)
    assert persons[0].w > 0  # bbox 仍然带上了


def test_parse_people_salvages_truncated_array_and_raises_when_hopeless():
    persons = vlm.parse_people('{"people": [{"keypoints": [[100, 200, 2], [300, 400, 2]')
    assert len(persons) == 1 and len(persons[0].keypoints) == 17
    with pytest.raises(vlm.VLMError, match="残缺"):
        vlm.parse_people('{"people": [{"bbox": [1, 2, 3, 4]')


def test_parse_people_accepts_empty_list_and_persons_alias():
    assert vlm.parse_people('{"people": []}') == []
    assert len(vlm.parse_people(json.dumps({"persons": [person_json()]}))) == 1


# --------------------------------------------------------------------------- #
# 发请求
# --------------------------------------------------------------------------- #

def test_chat_sends_image_and_key(monkeypatch, profile):
    calls = []

    def fake_request(url, headers, timeout, payload=None):
        calls.append({"url": url, "headers": dict(headers), "timeout": timeout, "payload": payload})
        return 200, chat_body('{"people": []}'), ""

    monkeypatch.setattr(models, "_request", fake_request)
    reply = vlm.chat(profile, "标一下", "data:image/jpeg;base64,AAAA", max_tokens=8)

    assert reply == '{"people": []}'
    sent = calls[0]
    assert sent["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer secret-key"
    assert sent["payload"]["max_tokens"] == 8
    assert sent["payload"]["model"] == "qwen2.5vl:7b"
    kinds = [part["type"] for part in sent["payload"]["messages"][0]["content"]]
    assert kinds == ["text", "image_url"]


def test_chat_raises_with_http_hint(monkeypatch, profile):
    monkeypatch.setattr(models, "_request",
                        lambda *a, **k: (401, "{}", "401 未授权：API Key 不对或没有该模型权限"))
    with pytest.raises(vlm.VLMError, match="未授权"):
        vlm.chat(profile, "标一下", "data:image/jpeg;base64,AAAA")


def test_chat_raises_on_empty_content(monkeypatch, profile):
    monkeypatch.setattr(models, "_request", lambda *a, **k: (200, chat_body("   "), ""))
    with pytest.raises(vlm.VLMError, match="空内容"):
        vlm.chat(profile, "标一下", "data:image/jpeg;base64,AAAA")


# --------------------------------------------------------------------------- #
# 后端选择与流水线
# --------------------------------------------------------------------------- #

def test_build_backend_switches_on_vlm_profile():
    assert isinstance(build_backend(BackendSettings()), PoseEnsemble)
    assert isinstance(build_backend(BackendSettings(vlm_profile="本地桩")), vlm.VLMBackend)


def test_backend_load_accepts_id_or_name(profile):
    backend = vlm.VLMBackend(BackendSettings(vlm_profile="本地桩"))
    assert backend.load().profile["id"] == profile["id"]
    assert backend.source_tag == f"vlm:{profile['id']}"
    assert "本地" in backend.describe()

    missing = vlm.VLMBackend(BackendSettings(vlm_profile="不存在的配置"))
    with pytest.raises(vlm.VLMError) as exc:
        missing.load()
    assert "模型管理" in str(exc.value)
    assert "本地桩" in str(exc.value)  # 报错要把可选项列出来

    with pytest.raises(vlm.VLMError):
        vlm.VLMBackend(BackendSettings()).load()


def test_backend_infer_returns_persons(monkeypatch, profile):
    monkeypatch.setattr(models, "_request",
                        lambda *a, **k: (200, chat_body(json.dumps({"people": [person_json()]})), ""))
    backend = vlm.VLMBackend(BackendSettings(vlm_profile=profile["id"]))
    backend.load()
    persons = backend.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(persons) == 1
    assert len(persons[0].keypoints) == 17
    assert backend.last_latency_ms >= 0


def test_label_dataset_uses_vlm_backend(tmp_path, monkeypatch, profile):
    from PIL import Image

    from autolabel.dataset import import_images, init_dataset, label_path_for, load_index
    from autolabel.pipeline import label_dataset, settings_signature

    source = tmp_path / "shot.png"
    Image.new("RGB", (320, 240), (120, 130, 140)).save(source)
    root = tmp_path / "ds"
    init_dataset(str(root))
    import_images(str(root), [str(source)])

    monkeypatch.setattr(models, "_request",
                        lambda *a, **k: (200, chat_body(json.dumps({"people": [person_json()]})), ""))
    stats = label_dataset(str(root), BackendSettings(vlm_profile="本地桩"), verbose=False)

    assert stats["labeled"] == 1 and stats["people"] == 1 and stats["failed"] == 0
    rel = next(iter(load_index(str(root)).items))
    persons = parse_label_file(label_path_for(str(root), rel))
    assert len(persons) == 1 and len(persons[0].keypoints) == 17

    item = load_index(str(root)).get(rel)
    assert item["labeled_by"] == f"vlm:{profile['id']}"
    assert vlm.VLM_FLAG in item["object_flags"]  # 保证整批进人工复核清单
    # 换后端等于换参数指纹，避免把 YOLO 的标签当成多模态已标完
    assert settings_signature(BackendSettings(vlm_profile="本地桩")) != settings_signature(BackendSettings())


def test_label_dataset_survives_single_image_failure(tmp_path, monkeypatch, profile):
    from PIL import Image

    from autolabel.dataset import import_images, init_dataset
    from autolabel.pipeline import label_dataset

    root = tmp_path / "ds"
    init_dataset(str(root))
    sources = []
    for index, name in enumerate(("a.png", "b.png")):
        path = tmp_path / name
        # 两张图必须不同，否则会被入库时的完全重复去重吃掉一张
        Image.new("RGB", (200, 200), (90 + index * 60, 90, 90)).save(path)
        sources.append(str(path))
    import_images(str(root), sources)

    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0, "", "请求超时（http://127.0.0.1:11434/v1/chat/completions）"
        return 200, chat_body('{"people": []}'), ""

    monkeypatch.setattr(models, "_request", flaky)
    stats = label_dataset(str(root), BackendSettings(vlm_profile=profile["id"]), verbose=False)

    assert stats["failed"] == 1 and stats["labeled"] == 1  # 一张失败不中断整批


# --------------------------------------------------------------------------- #
# 给界面的清单与命令行参数
# --------------------------------------------------------------------------- #

def test_vlm_choices_hide_api_key(profile):
    profile_id = profile["id"]
    models.record_test(profile_id, {"ok": True, "message": "连接正常", "latency_ms": 5,
                                   "models": ["qwen2.5vl:7b"], "checked_at": "2026-10-16 23:59"})
    payload = models.vlm_choices()
    text = json.dumps(payload, ensure_ascii=False)
    assert "secret-key" not in text
    assert payload["default"] == profile_id
    row = payload["profiles"][0]
    assert (row["id"], row["ok"], row["tested"]) == (profile_id, True, True)
    assert row["kind"] == "local"


def test_cli_vlm_flag_reaches_settings():
    from autolabel.cli import build_parser, load_settings

    args = build_parser().parse_args(["label", "datasets/x", "--vlm", "本地 Ollama"])
    assert load_settings(args).vlm_profile == "本地 Ollama"
    assert load_settings(build_parser().parse_args(["label", "datasets/x"])).vlm_profile == ""


def test_dataset_payload_exposes_vlm_choices(tmp_path, monkeypatch, profile):
    from autolabel.dataset import init_dataset
    from autolabel.review import ReviewStore

    root = tmp_path / "ds"
    init_dataset(str(root))
    store = ReviewStore(str(root))
    payload = store.dataset_payload()
    assert payload["vlm"]["profiles"][0]["name"] == "本地桩"
    assert "api_key" not in json.dumps(payload["vlm"], ensure_ascii=False)


def test_job_payload_reports_vlm_profile():
    from autolabel.upload import Job

    job = Job(id="j1", kind="image", files=["a.jpg"], params={"model": "yolo11n-pose.pt",
                                                             "vlm_profile": "vlm_abc"})
    assert job.payload()["vlm_profile"] == "vlm_abc"
    assert Job(id="j2", kind="image", files=["a.jpg"], params={}).payload()["vlm_profile"] == ""


def test_submit_resolves_vlm_profile_name_and_rejects_unknown(profile):
    from autolabel.upload import JobStore

    resolved = JobStore._normalize_params({"model": "vlm:名字", "vlm_profile": profile["id"]})
    assert resolved["vlm_profile"] == profile["id"]
    assert resolved["vlm_name"] == "本地桩"  # 任务与批次里显示人话名字

    by_name = JobStore._normalize_params({"vlm_profile": "本地桩"})
    assert by_name["vlm_profile"] == profile["id"]

    plain = JobStore._normalize_params({"model": "yolo11n-pose.pt", "vlm_profile": "  "})
    assert "vlm_profile" not in plain and "vlm_name" not in plain

    with pytest.raises(ValueError, match="找不到多模态配置"):
        JobStore._normalize_params({"vlm_profile": "没这条"})


def test_load_hint_names_the_multimodal_model(profile, monkeypatch, tmp_path):
    from autolabel.upload import JobStore

    monkeypatch.setenv("AUTOLABEL_WEIGHTS_DIR", str(tmp_path / "models"))
    store = JobStore(str(tmp_path))
    job = type("J", (), {"params": {"vlm_profile": "本地桩"}})()
    assert "多模态大模型 本地桩" in store._load_hint(job)
    assert "逐图" in store._load_hint(job)
