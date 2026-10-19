"""模型管理：权重清单/删除/后台下载 + 多模态大模型配置与连通性验证。"""

import io
import json
import os
import urllib.error
import zipfile

import pytest

from autolabel import models
from autolabel import secrets


def zip_bytes(payload: bytes = b"weights") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.txt", payload)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload, status: int = 200):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload).encode("utf-8")
        self.payload = payload
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk, self.payload = self.payload, b""
        else:
            chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int, body: str = "boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/v1/models", code, body, {}, io.BytesIO(body.encode()))


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """把权重目录与 VLM 配置都指到 tmp，避免碰仓库里的真实文件。"""
    weights = tmp_path / "models"
    weights.mkdir()
    monkeypatch.setenv("AUTOLABEL_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("AUTOLABEL_VLM_CONFIG", str(tmp_path / "configs" / "vlm.json"))
    return {"weights": str(weights), "config": str(tmp_path / "configs" / "vlm.json")}


# --------------------------------------------------------------------------- #
# 姿态权重
# --------------------------------------------------------------------------- #

def test_list_weights_covers_presets_and_custom(workspace):
    with open(os.path.join(workspace["weights"], "my-pose.pt"), "wb") as fh:
        fh.write(zip_bytes())

    rows = models.list_weights()
    by_name = {row["name"]: row for row in rows}

    # 预置权重按 n/s/m 排在前面，自定义权重跟在后面
    assert [row["name"] for row in rows[:3]] == list(models.UI_MODEL_CHOICES)
    assert "my-pose.pt" in by_name
    assert by_name["my-pose.pt"]["ready"] is True
    assert by_name["my-pose.pt"]["custom"] is True
    assert by_name["my-pose.pt"]["source"] == "disk"
    assert by_name[models.UI_MODEL_CHOICES[0]]["ready"] is False
    assert by_name[models.UI_MODEL_CHOICES[0]]["source"] == "preset"


@pytest.mark.parametrize("bad", ["models/x.pt", "/abs/x.pt", "x.txt", ""])
def test_resolve_weight_in_dir_rejects_outside_files(workspace, bad):
    with pytest.raises(ValueError):
        models.resolve_weight_in_dir(bad)


def test_delete_weight_removes_part_and_bad(workspace):
    target = os.path.join(workspace["weights"], "yolo11s-pose.pt")
    with open(target, "wb") as fh:
        fh.write(zip_bytes())
    with open(target + ".part", "wb") as fh:
        fh.write(b"half")
    with open(target + ".bad", "wb") as fh:
        fh.write(b"older")

    result = models.delete_weight("yolo11s-pose.pt")

    assert result["name"] == "yolo11s-pose.pt"
    assert sorted(result["removed"]) == ["yolo11s-pose.pt", "yolo11s-pose.pt.bad", "yolo11s-pose.pt.part"]
    assert not os.path.exists(target)
    assert not any(w["name"] == "yolo11s-pose.pt" and w["size_mb"] for w in result["weights"])


def test_delete_weight_refuses_model_in_use_and_missing_file(workspace):
    target = os.path.join(workspace["weights"], "yolo11m-pose.pt")
    with open(target, "wb") as fh:
        fh.write(zip_bytes())

    with pytest.raises(ValueError, match="正在被标注任务使用"):
        models.delete_weight("yolo11m-pose.pt", active_names=("yolo11m-pose.pt",))
    with pytest.raises(FileNotFoundError):
        models.delete_weight("yolo11x-pose.pt")


def test_start_download_completes_weight(workspace, monkeypatch):
    payload = zip_bytes(b"yolo11l")

    def fake(request, data=None, timeout=None):
        assert request.full_url.startswith("https://")
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    task = models.start_download("yolo11l-pose.pt", background=False)

    assert task["ok"] is True and task["stage"] == "done"
    assert models.model_status("yolo11l-pose.pt", weights_dir=models.weights_dir())["ready"] is True


def test_start_download_reports_failure(workspace, monkeypatch):
    def boom(request, data=None, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    task = models.start_download("yolo11x-pose.pt", background=False)

    assert task["ok"] is False and task["stage"] == "failed"
    assert "connection refused" in task["error"]


# --------------------------------------------------------------------------- #
# 多模态大模型配置
# --------------------------------------------------------------------------- #

def test_vlm_config_starts_empty_when_missing(workspace):
    assert models.load_vlm() == {"profiles": [], "default": None}


def test_upsert_profile_creates_updates_and_masks_key(workspace):
    created = models.upsert_profile({"provider": "ollama", "name": "本地 Ollama",
                                     "base_url": "localhost:11434/v1/", "model": "qwen2.5vl:7b"})

    assert created["id"].startswith("vlm_")
    assert created["kind"] == "local"
    assert created["base_url"] == "http://localhost:11434/v1"   # 补协议头、去尾斜杠
    assert created["has_key"] is False
    assert "api_key" not in created
    assert models.load_vlm()["default"] == created["id"]

    with_key = models.upsert_profile({"id": created["id"], "api_key": "sk-proj-abcdef1234"})
    assert with_key["api_key_masked"] == "sk-…1234"
    assert models.get_profile(created["id"])["api_key"] == "sk-proj-abcdef1234"

    # 前端把掩码原样回传时不能覆盖真实 Key，名称等字段也要继承
    again = models.upsert_profile({"id": created["id"], "api_key": with_key["api_key_masked"],
                                  "remark": "标注前预筛"})
    assert models.get_profile(created["id"])["api_key"] == "sk-proj-abcdef1234"
    assert again["name"] == "本地 Ollama"
    assert again["remark"] == "标注前预筛"
    assert len(models.load_vlm()["profiles"]) == 1


@pytest.mark.parametrize("raw,expect", [
    ({}, "请填写配置名称"),
    ({"name": "A", "base_url": "http://x/v1"}, "请填写模型名"),
    ({"name": "A", "model": "m"}, "服务地址"),
    ({"name": "A", "base_url": "ftp://x/v1", "model": "m"}, "http/https"),
    ({"name": "A", "base_url": "http://x/v1", "model": "m", "kind": "carrier"}, "本地或云端"),
])
def test_upsert_profile_validates_input(workspace, raw, expect):
    with pytest.raises(ValueError, match=expect):
        models.upsert_profile(raw)


def test_default_and_delete_keep_config_consistent(workspace):
    first = models.upsert_profile({"name": "A", "base_url": "http://a/v1", "model": "m"})
    second = models.upsert_profile({"name": "B", "base_url": "http://b/v1", "model": "m"})

    assert models.set_default(second["id"])["default"] == second["id"]
    after_delete = models.delete_profile(second["id"])
    assert after_delete["default"] == first["id"]        # 删掉默认项后自动回落
    with pytest.raises(ValueError, match="未知配置"):
        models.delete_profile("vlm_nope")
    with pytest.raises(ValueError, match="未知配置"):
        models.set_default("vlm_nope")


def test_test_profile_uses_models_endpoint(workspace, monkeypatch):
    def fake(request, data=None, timeout=None):
        return FakeResponse({"data": [{"id": "qwen2.5vl:7b"}, {"id": "llava:13b"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = models.test_profile({"base_url": "http://x/v1", "model": "qwen2.5vl:7b"})

    assert result["ok"] is True
    assert "qwen2.5vl:7b" in result["message"]
    assert result["models"][:2] == ["qwen2.5vl:7b", "llava:13b"]


def test_test_profile_falls_back_to_chat_probe(workspace, monkeypatch):
    seen = []

    def fake(request, data=None, timeout=None):
        seen.append((request.full_url, data))
        if request.full_url.endswith("/models"):
            raise http_error(404, "not found")
        return FakeResponse({"choices": []})

    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = models.test_profile({"base_url": "http://x/v1", "model": "gpt-4o-mini"})

    assert result["ok"] is True
    assert seen[1][0].endswith("/chat/completions")
    assert json.loads(seen[1][1])["max_tokens"] == 1
    assert "/models 不可用" in result["message"]


def test_test_profile_explains_auth_and_refused(workspace, monkeypatch):
    def deny(request, data=None, timeout=None):
        raise http_error(401, "invalid api key")

    monkeypatch.setattr("urllib.request.urlopen", deny)
    unauthorized = models.test_profile({"base_url": "http://x/v1", "model": "m"})
    assert unauthorized["ok"] is False and "API Key" in unauthorized["message"]

    def refused(request, data=None, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refused)
    down = models.test_profile({"base_url": "http://localhost:11434/v1", "model": "m"})
    assert down["ok"] is False and "ollama serve" in down["message"]


def test_record_test_kept_only_for_same_target(workspace):
    profile = models.upsert_profile({"name": "A", "base_url": "http://a/v1", "model": "m"})
    models.record_test(profile["id"], {"ok": True, "message": "可用", "latency_ms": 5})

    assert models.get_profile(profile["id"])["last_test"]["ok"] is True
    renamed = models.upsert_profile({"id": profile["id"], "name": "改名了"})
    assert renamed["last_test"]["ok"] is True           # 只改名不影响连通结论
    moved = models.upsert_profile({"id": profile["id"], "base_url": "http://other/v1"})
    assert "last_test" not in moved                     # 换了服务地址，旧结论作废


# --------------------------------------------------------------------------- #
# HTTP 接口
# --------------------------------------------------------------------------- #

def test_models_api_lists_weights_and_vlm(server, workspace):
    base, _store = server

    with open(os.path.join(workspace["weights"], "extra-pose.pt"), "wb") as fh:
        fh.write(zip_bytes())
    payload = json.loads(_get(base + "/api/models"))

    assert os.path.basename(payload["weights_dir"]) == "models"
    assert any(row["name"] == "extra-pose.pt" and row["ready"] for row in payload["weights"])
    assert {p["id"] for p in payload["vlm"]["presets"]} >= {"ollama", "openai", "custom"}
    assert payload["vlm"]["profiles"] == []


def test_models_api_vlm_roundtrip(server, workspace):
    base, _store = server
    # 127.0.0.1:9 基本必然拒连，既能验证失败链路，又不会真的打到公网
    saved = json.loads(_post(base + "/api/models/vlm/save",
                             {"profile": {"name": "云端", "base_url": "127.0.0.1:9/v1",
                                          "model": "vl", "api_key": "sk-abcdef123456"}}))
    profile_id = saved["profile"]["id"]
    assert saved["profile"]["api_key_masked"] == "sk-…3456"
    assert "sk-abcdef123456" not in json.dumps(saved, ensure_ascii=False)

    tested = json.loads(_post(base + "/api/models/vlm/test", {"id": profile_id}))
    assert tested["result"]["ok"] is False          # 没有真实服务，返回可读错误而不是 500
    assert "连不上" in tested["result"]["message"]
    assert tested["profiles"][0]["last_test"]["ok"] is False

    deleted = json.loads(_post(base + "/api/models/vlm/delete", {"id": profile_id}))
    assert deleted["profiles"] == [] and deleted["default"] is None


def test_models_api_weight_download_and_delete(server, workspace, monkeypatch):
    base, _store = server

    def fake_ensure(model, weights_dir="models", **kwargs):
        path = os.path.join(weights_dir, model)
        os.makedirs(weights_dir, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(zip_bytes())
        return path

    monkeypatch.setattr(models, "ensure_weights", fake_ensure)
    started = json.loads(_post(base + "/api/models/weight/download", {"name": "yolo11l-pose.pt"}))
    assert started["ok"] is True

    for _ in range(50):
        state = json.loads(_post(base + "/api/models/download", {"name": "yolo11l-pose.pt"}))
        if state["task"]["stage"] != "running":
            break
    assert state["task"]["ok"] is True

    deleted = json.loads(_post(base + "/api/models/weight/delete", {"name": "yolo11l-pose.pt"}))
    assert deleted["ok"] is True and deleted["removed"] == ["yolo11l-pose.pt"]


def test_models_api_rejects_unknown_and_traversal(server, workspace):
    base, _store = server
    body = json.loads(_post(base + "/api/models/weight/delete", {"name": "../secrets.pt"}))
    assert "只能管理" in body["error"]
    missing = json.loads(_post(base + "/api/models/vlm/test", {"id": "vlm_nope"}))
    assert "未知配置" in missing["error"]


def test_store_blocks_deleting_weight_used_by_running_job(server, workspace):
    from autolabel.upload import Job

    base, store = server
    target = os.path.join(workspace["weights"], "yolo11n-pose.pt")
    with open(target, "wb") as fh:
        fh.write(zip_bytes())
    job = Job(id="j1", kind="image", files=[], params={"model": "yolo11n-pose.pt"})
    with store.jobs.lock:
        store.jobs.jobs.append(job)

    assert store.active_models() == ["yolo11n-pose.pt"]
    with pytest.raises(ValueError, match="正在被标注任务使用"):
        store.delete_weight("yolo11n-pose.pt")

    job.finished = True
    result = store.delete_weight("yolo11n-pose.pt")
    assert result["removed"] == ["yolo11n-pose.pt"]


def _get(url: str) -> str:
    return _send(url)


def _post(url: str, payload: dict) -> str:
    return _send(url, payload)


def _send(url: str, payload: dict = None) -> str:
    """发请求并原样返回响应体：非 2xx 也要拿到服务端的错误 JSON。"""
    import urllib.request

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.read().decode("utf-8")


def test_deleting_profile_takes_the_key_with_it(workspace):
    """删掉的配置不能留下一把界面里看不见、也删不掉的幽灵密钥。

    Key 存在 Key 文件里，配置文件只留 ``key_ref``：以前 ``delete_profile`` 只删
    配置，那串明文就再没人引用——用户以为已经撤销了，磁盘上其实还躺着。
    """
    created = models.upsert_profile({"name": "云端", "base_url": "https://api.x/v1",
                                     "model": "gpt-4o-mini", "api_key": "sk-secretvalue98765"})
    ref = created["key_ref"]
    assert secrets.get_secret(ref) == "sk-secretvalue98765"

    models.delete_profile(created["id"])

    assert secrets.get_secret(ref) == ""
    assert ref not in secrets.secret_refs()
