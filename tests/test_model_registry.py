"""模型目录与登记表：中文说明、默认值、本地登记、下载记录。

备注说明这个功能在 1.4.9 取消了（目录里的中文说明就是唯一口径），这里留着几条
断言钉住它别再长回来；多模态配置那句 remark 是配置本身，跟模型备注不是一回事。
"""

import io
import json
import os
import urllib.error
import urllib.request
import zipfile

import pytest

from autolabel import cli, models
from autolabel import secrets as secret_store


def zip_bytes(payload: bytes = b"weights") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.txt", payload)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.status = 200
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


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """登记表、权重目录、多模态配置、Key 文件全指向 tmp，不碰仓库与 ~/.autolabel。"""
    weights = tmp_path / "models"
    weights.mkdir()
    monkeypatch.setenv("AUTOLABEL_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("AUTOLABEL_VLM_CONFIG", str(tmp_path / "configs" / "vlm.json"))
    monkeypatch.setenv("AUTOLABEL_MODEL_CONFIG", str(tmp_path / "configs" / "models.json"))
    monkeypatch.setenv("AUTOLABEL_SECRETS", str(tmp_path / "secrets.json"))
    return {"weights": str(weights), "model_config": str(tmp_path / "configs" / "models.json"),
            "vlm_config": str(tmp_path / "configs" / "vlm.json")}


def put_weight(workspace, name: str, payload: bytes = b"weights") -> str:
    """在权重目录里放一个 zip 结构的假权重。"""
    path = os.path.join(workspace["weights"], name)
    with open(path, "wb") as fh:
        fh.write(zip_bytes(payload))
    return path


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# 可下载目录与中文说明
# --------------------------------------------------------------------------- #

def test_catalog_lists_every_downloadable_model_with_chinese_docs(workspace):
    rows = models.model_catalog()

    assert [row["id"] for row in rows[:3]] == ["yolo11n-pose", "yolo11s-pose", "yolo11m-pose"]
    assert len(rows) == len(models.PRESET_CATALOG)
    for row in rows:
        assert row["downloadable"] is True
        assert len(row["description"]) > 20
        assert row["recommended_for"]
        assert row["url"].startswith("https://")
        assert row["expected_mb"] > 0 and row["num_keypoints"] == 17
        assert row["capability_label"] == models.CAPABILITY_LABELS["pose"]
        # 空权重目录里全都还没下载
        assert row["status"] == models.STATUS_MISSING
        assert row["status_label"] == "未下载"


def test_catalog_description_flags_numbers_officially_unpublished(workspace):
    rows = {row["id"]: row for row in models.model_catalog()}

    assert rows["yolo11n-pose"]["params_m"] == 2.9
    assert rows["yolo11n-pose"]["map_pose"] == 50.4
    assert rows["yolo11n-pose"]["sha256"]
    assert rows["yolo26n-pose"]["params_m"] is None
    assert "官方暂未公布" in rows["yolo26n-pose"]["description"]


def test_catalog_status_reflects_downloaded_weight(workspace):
    assert models.model_catalog()[0]["status"] == models.STATUS_MISSING

    put_weight(workspace, "yolo11n-pose.pt")
    rows = {row["id"]: row for row in models.model_catalog()}

    assert rows["yolo11n-pose"]["ready"] is True
    assert rows["yolo11n-pose"]["status"] == models.STATUS_AVAILABLE
    assert rows["yolo11n-pose"]["status_label"] == "可用"
    assert rows["yolo11n-pose"]["source"] == models.SOURCE_DOWNLOADED
    assert rows["yolo11n-pose"]["source_label"] == "已下载"


def test_weight_info_keeps_legacy_keys_and_adds_catalog_info(workspace):
    put_weight(workspace, "yolo11s-pose.pt")

    info = models.weight_info("yolo11s-pose.pt")

    assert info["name"] == "yolo11s-pose.pt"
    assert info["source"] == "disk"          # 老前端依赖 disk/preset 两值
    assert info["custom"] is False
    assert info["ready"] is True
    assert info["model_id"] == "yolo11s-pose"
    assert info["display_name"] == "YOLO11-small 姿态"
    assert info["source_layer"] == models.SOURCE_DOWNLOADED   # 分层来源走新字段
    assert len(info["description"]) > 20


def test_custom_weight_gets_a_placeholder_row(workspace):
    put_weight(workspace, "gym-finetune.pt")

    info = models.weight_info("gym-finetune.pt")

    assert info["custom"] is True
    assert info["model_id"] == "gym-finetune"
    assert "自定义" in info["description"]
    assert info["downloadable"] is False


# --------------------------------------------------------------------------- #
# 备注说明已取消
# --------------------------------------------------------------------------- #

def test_model_note_feature_is_gone(workspace):
    """模型不再有「备注说明」：目录里写好的中文说明就是唯一口径。"""
    assert not hasattr(models, "set_model_note")
    assert not hasattr(models, "MAX_NOTE_CHARS")
    assert "note" not in models.describe_model("yolo11m-pose")
    assert "notes" not in models.empty_model_config()


def test_vlm_profile_keeps_its_own_remark(workspace):
    """多模态配置那句备注写在 vlm.json 的 remark 里，那是配置的一部分，不受影响。"""
    saved = models.upsert_profile({"name": "本地桩", "provider": "ollama", "kind": "local",
                                  "base_url": "http://127.0.0.1:11434/v1",
                                  "model": "qwen2.5vl:7b", "remark": "只在离线机上用"})

    row = next(r for r in models.list_models(capability=models.CAPABILITY_VLM)
               if r["id"] == saved["id"])

    assert row["note"] == "只在离线机上用"
    assert read_json(workspace["vlm_config"])["profiles"][0]["remark"] == "只在离线机上用"


# --------------------------------------------------------------------------- #
# 默认模型
# --------------------------------------------------------------------------- #

def test_default_model_falls_back_to_a_sane_weight(workspace):
    # 一条都没下载时兜底到目录里第一档（推荐的 nano），界面至少有得选
    assert models.default_model(models.CAPABILITY_POSE) == "yolo11n-pose"

    put_weight(workspace, "yolov8s-pose.pt")
    assert models.default_model(models.CAPABILITY_POSE) == "yolov8s-pose"


def test_default_model_can_be_set_and_read_back(workspace):
    defaults = models.set_model_default("yolo11m-pose")

    assert defaults["ids"][models.CAPABILITY_POSE] == "yolo11m-pose"
    assert models.default_model(models.CAPABILITY_POSE) == "yolo11m-pose"
    assert read_json(workspace["model_config"])["defaults"]["pose"] == "yolo11m-pose"
    row = models.describe_model("yolo11m-pose")
    assert row["default"] is True and row["default_explicit"] is True


def test_default_model_rejects_wrong_capability_and_unknown_id(workspace):
    with pytest.raises(ValueError, match="不能当"):
        models.set_model_default("yolo11n-pose", models.CAPABILITY_VLM)
    with pytest.raises(ValueError, match="能力只能是"):
        models.set_model_default("yolo11n-pose", "magic")
    with pytest.raises(KeyError, match="未知模型"):
        models.set_model_default("nope")


# --------------------------------------------------------------------------- #
# 登记本地模型 / 取消登记
# --------------------------------------------------------------------------- #

def test_register_local_model_records_path_without_moving_file(workspace, tmp_path):
    source = tmp_path / "outside" / "gym-pose.pt"
    source.parent.mkdir()
    source.write_bytes(zip_bytes(b"gym"))

    saved = models.register_local_model(path=str(source), display_name="体操自研模型")

    assert saved["registered"] is True and saved["already_in_catalog"] is False
    assert saved["id"] == "gym-pose"
    row = models.describe_model("gym-pose")
    assert row["path"] == str(source) and os.path.isfile(row["path"])
    assert row["display_name"] == "体操自研模型"
    assert row["registered"] is True
    assert row["source"] == models.SOURCE_LOCAL and row["source_label"] == "本地"

    assert models.forget_model("gym-pose")["forgotten"] is True
    assert os.path.isfile(str(source))          # 取消登记只删记录，不动文件
    assert read_json(workspace["model_config"])["local"] == []
    with pytest.raises(KeyError, match="未知模型"):
        models.describe_model("gym-pose")


def test_register_local_model_on_catalog_weight_says_it_is_already_listed(workspace):
    """已经在目录里的权重不重复登记：多一条只会多一处对不上。"""
    path = put_weight(workspace, "yolo11n-pose.pt")

    saved = models.register_local_model(path=path)

    assert saved["already_in_catalog"] is True and saved["registered"] is False
    assert saved["id"] == "yolo11n-pose"
    # 备注取消后这条路径根本不写盘：没登记成功就不该凭空多出一份配置
    assert not os.path.exists(workspace["model_config"]) or \
        read_json(workspace["model_config"])["local"] == []


@pytest.mark.parametrize("name,data,match", [
    ("half.pt", b"not a zip at all", "不是完整的权重包"),
    ("missing.pt", None, "文件不存在"),
])
def test_register_local_model_rejects_broken_or_absent_files(workspace, tmp_path, name, data, match):
    path = tmp_path / name
    if data is not None:
        path.write_bytes(data)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        models.register_local_model(path=str(path))


def test_register_local_model_validates_origin_and_id_clash(workspace, tmp_path):
    source = tmp_path / "dup.pt"
    source.write_bytes(zip_bytes())

    with pytest.raises(ValueError, match="出处只能是"):
        models.register_local_model(path=str(source), origin="taobao")
    with pytest.raises(ValueError, match="能力只能是"):
        models.register_local_model(path=str(source), capability="dance")
    # 显式指定的 ID 撞上目录里已有的模型要拦下来
    with pytest.raises(ValueError, match="已被"):
        models.register_local_model(path=str(source), model_id="yolo11n-pose")


def test_forget_refuses_catalog_entries_and_clears_default(workspace, tmp_path):
    models.set_model_default("yolo11s-pose")

    # 内置目录项不该被「取消登记」误删，默认值也就不会被连带清掉
    with pytest.raises(KeyError, match="内置目录项"):
        models.forget_model("yolo11s-pose")
    assert models.default_model(models.CAPABILITY_POSE) == "yolo11s-pose"

    source = tmp_path / "temp.pt"
    source.write_bytes(zip_bytes())
    saved = models.register_local_model(path=str(source))
    models.set_model_default(saved["id"], models.CAPABILITY_POSE)

    models.forget_model(saved["id"])

    assert read_json(workspace["model_config"])["defaults"] == {}


# --------------------------------------------------------------------------- #
# 兼容性与下载记录
# --------------------------------------------------------------------------- #

def test_keypoint_mismatch_marks_model_incompatible(workspace, dataset, tmp_path):
    source = tmp_path / "body26.pt"
    source.write_bytes(zip_bytes(b"26"))
    models.register_local_model(path=str(source), num_keypoints=26)

    rows = {row["id"]: row for row in models.list_models(dataset_root=dataset)}

    assert rows["body26"]["compatible"] is False
    assert rows["body26"]["status"] == models.STATUS_INCOMPATIBLE
    assert rows["body26"]["status_label"] == "不兼容/暂不可用"
    assert any("26 关键点" in reason for reason in rows["body26"]["compatibility_reasons"])
    assert rows["yolo11n-pose"]["compatible"] is True
    assert models.describe_model("body26", dataset_root=dataset)["compatible"] is False


def test_incomplete_weight_is_reported_as_unusable(workspace):
    with open(os.path.join(workspace["weights"], "yolo11s-pose.pt"), "wb") as fh:
        fh.write(b"half of a weight file")

    row = models.describe_model("yolo11s-pose")

    assert row["status"] == models.STATUS_INCOMPLETE
    assert row["status_label"] == "文件不完整"
    assert row["ready"] is False and row["compatible"] is False
    assert any("不完整" in reason for reason in row["compatibility_reasons"])
    assert models.weight_info("yolo11s-pose.pt")["compatible"] is False


def test_download_writes_a_full_audit_record(workspace, monkeypatch):
    payload = zip_bytes(b"yolo11n-payload")
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda request, data=None, timeout=None: FakeResponse(payload))

    task = models.start_download("yolo11n-pose.pt", background=False)

    assert task["ok"] is True
    records = models.download_records("yolo11n-pose")
    assert len(records) == 1
    record = records[0]
    for key in ("model_id", "url", "mirror", "sha256_expected", "sha256_actual",
                "expected_bytes", "bytes", "size_mb", "temp_name", "seconds", "ok",
                "started_at", "finished_at"):
        assert key in record, key
    assert record["model_id"] == "yolo11n-pose"
    assert record["url"].startswith("https://")
    assert record["bytes"] == len(payload) and record["ok"] is True
    assert record["temp_name"] == "yolo11n-pose.pt.part"
    # 假权重的哈希不可能等于官方值：如实记不一致，但不算下载失败
    assert record["sha256_match"] is False
    assert "SHA256" in task["message"]
    assert models.describe_model("yolo11n-pose")["last_used"]
    assert models.download_records("yolov8x-pose") == []

    # 官方没公布 SHA256 的档位不下结论，只把体积和结果记下来
    task2 = models.start_download("yolo11l-pose.pt", background=False)
    record2 = models.download_records("yolo11l-pose")[0]
    assert task2["ok"] is True and "sha256_match" not in record2
    assert record2["sha256_expected"] == "" and record2["ok"] is True
    assert len(models.download_records()) == 2


# --------------------------------------------------------------------------- #
# 多模态配置的 Key 只存引用
# --------------------------------------------------------------------------- #

def test_plain_api_key_migrates_into_secret_store(workspace):
    os.makedirs(os.path.dirname(workspace["vlm_config"]), exist_ok=True)
    with open(workspace["vlm_config"], "w", encoding="utf-8") as fh:
        json.dump({"profiles": [{"id": "vlm_old", "name": "老配置", "kind": "cloud",
                                 "provider": "openai", "base_url": "https://api.example.com/v1",
                                 "model": "gpt-4o-mini", "api_key": "sk-plain-123456"}],
                   "default": "vlm_old"}, fh, ensure_ascii=False)

    config = models.load_vlm()

    assert config["profiles"][0]["api_key"] == "sk-plain-123456"      # 内存里能拿到明文
    on_disk = read_json(workspace["vlm_config"])["profiles"][0]
    assert "api_key" not in on_disk                                    # 落盘只剩引用
    assert on_disk["key_ref"] == "vlm:vlm_old"
    assert secret_store.get_secret("vlm:vlm_old") == "sk-plain-123456"

    row = models.list_models(capability=models.CAPABILITY_VLM)[0]
    assert row["api_key_masked"] == "sk-…3456"
    assert row["has_key"] is True
    assert "sk-plain-123456" not in json.dumps(row, ensure_ascii=False)


def test_vlm_rows_carry_runtime_parameters(workspace):
    models.upsert_profile({"name": "省显存", "provider": "ollama", "kind": "local",
                           "base_url": "http://127.0.0.1:11434/v1", "model": "qwen2.5vl:7b",
                           "max_tokens": 512, "image_limit": 512, "json_mode": True})

    row = models.list_models(capability=models.CAPABILITY_VLM)[0]

    assert row["max_tokens"] == 512 and row["image_limit"] == 512 and row["json_mode"] is True
    assert row["speed"] and "慢" in row["speed"]
    assert row["capability_label"] == models.CAPABILITY_LABELS["vlm"]
    # 没做过连通性测试：算可用，但要给出中文提示
    assert any("连通性测试" in reason for reason in row["compatibility_reasons"])


def test_models_payload_exposes_paths_and_defaults(workspace):
    payload = models.vlm_payload()

    assert payload["model_config_path"].endswith("models.json")
    assert payload["secrets_path"].endswith("secrets.json")
    assert set(payload["defaults"]) == {"ids", "labels"}
    assert len(payload["catalog"]) == len(models.PRESET_CATALOG)
    assert [row["id"] for row in payload["models"][:1]] == ["yolo11n-pose"]
    assert json.dumps(payload["vlm"], ensure_ascii=False).count('"api_key"') == 0


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def test_cli_browse_prints_catalog_with_download_commands(workspace, capsys):
    assert cli.main(["models", "--browse"]) == 0

    out = capsys.readouterr().out
    assert "可下载模型目录（13 个）" in out
    assert "YOLO11-nano 姿态" in out
    assert "最小的姿态权重" in out
    assert "models --download-weight yolo11n-pose.pt" in out
    # 这 13 档只是姿态精选，不写清「其余去哪找」就会被当成目录的全部
    assert "models --search" in out and "只看已下载" in out


def test_cli_info_prints_detail(workspace, capsys):
    assert cli.main(["models", "--info", "yolo11x-pose"]) == 0

    out = capsys.readouterr().out
    assert "YOLO11-xlarge 姿态" in out
    assert "备注" not in out            # 备注说明已经取消，别再打一行空值
    assert "下载地址" in out and "yolo11x-pose.pt" in out
    assert "（无文件）" in out and "不兼容" not in out

    assert cli.main(["models", "--info", "yolo11n-pose"]) == 0
    detail = capsys.readouterr().out
    assert "SHA256" in detail and "约 2.9M" in detail and "mAP50-95" in detail


def test_cli_default_and_register_roundtrip(workspace, tmp_path, capsys):
    source = tmp_path / "dance.pt"
    source.write_bytes(zip_bytes(b"dance"))

    assert cli.main(["models", "--register", str(source), "--display-name", "街舞权重"]) == 0
    printed = capsys.readouterr().out
    assert "已登记本地模型 dance" in printed

    assert cli.main(["models", "--default", "dance"]) == 0
    assert "默认模型 -> dance" in capsys.readouterr().out

    assert cli.main(["models", "--forget", "dance"]) == 0
    assert "已取消登记 dance" in capsys.readouterr().out


def test_cli_no_longer_accepts_note(workspace, capsys):
    """``--note`` 这个参数已经删掉：留着它等于留一个没人读的字段。"""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["models", "--note", "yolo11n-pose=随便写点什么"])
    assert exit_info.value.code == 2      # argparse 认不出的参数直接退 2，不会走回 handler
    assert "unrecognized arguments" in capsys.readouterr().err


def test_cli_rejects_bad_capability_and_register(workspace, capsys):
    assert cli.main(["models", "--default", "yolo11n-pose", "--capability", "dance"]) == 2
    assert "能力只能是" in capsys.readouterr().out

    assert cli.main(["models", "--register", os.devnull]) == 2
    assert "登记本地模型失败" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 复核界面接口
# --------------------------------------------------------------------------- #

def post(base, path, payload):
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_model_rows_ship_with_chinese_docs(server, workspace):
    base, store = server

    row = next(item for item in store.model_rows() if item["id"] == "yolo11m-pose")

    assert len(row["description"]) > 20
    assert "note" not in row            # 统一清单里不再带备注字段
    assert base


def test_api_note_default_local_add_and_forget(server, workspace, tmp_path):
    base, _store = server
    source = tmp_path / "studio.pt"
    source.write_bytes(zip_bytes(b"studio"))

    status, body = post(base, "/api/models/local/add",
                        {"path": str(source), "display_name": "棚内权重"})
    assert status == 200 and body["ok"] is True
    ident = body["model"]["id"]
    assert any(row["id"] == ident and row["display_name"] == "棚内权重" for row in body["models"])

    status, body = post(base, "/api/models/default", {"id": ident})
    assert status == 200 and body["defaults"]["ids"]["pose"] == ident

    status, body = post(base, "/api/models/local/forget", {"id": ident})
    assert status == 200 and body["models"] and os.path.isfile(str(source))

    status, body = post(base, "/api/models/local/forget", {"id": ident})
    assert status == 404 and "未知对象" in body["error"]


def test_api_model_note_endpoint_is_gone(server, workspace):
    """写备注的接口整条删掉了：留 404 而不是悄悄接受一个没人读的字段。"""
    base, _store = server

    status, body = post(base, "/api/models/note", {"id": "yolo11m-pose", "note": "x"})

    assert status == 404
    assert "/api/models/note" in body["error"]


def test_api_reports_dataset_compatibility(server, workspace, dataset):
    base, store = server

    rows = {row["id"]: row for row in store.model_rows()}

    assert rows["yolo11n-pose"]["compatible"] is True
    assert rows["yolo11n-pose"]["status"] == models.STATUS_MISSING


# --------------------------------------------------------------------------- #
# 权重目录里真有的文件必须出现在清单里
# --------------------------------------------------------------------------- #

def test_unknown_weight_in_models_dir_joins_the_unified_list(workspace):
    """自己练的权重拷进 models/ 就该被看见，不必先跑一遍「登记本地模型」。"""
    put_weight(workspace, "studio-finetune.pt")

    rows = {row["id"]: row for row in models.list_models()}

    assert "studio-finetune" in rows, "清单只按「我们认识哪些模型」列，自己放进去的权重会凭空消失"
    row = rows["studio-finetune"]
    assert row["ready"] is True and row["status"] == models.STATUS_AVAILABLE
    assert row["origin"] == models.ORIGIN_CUSTOM


def test_catalog_weight_dropped_into_models_dir_is_listed_once(workspace):
    """官方权重直接拷进目录也只有一条：同一文件出现两行，默认值就会指错。"""
    put_weight(workspace, "yolo11n-pose.pt")

    rows = [row for row in models.list_models() if row["id"] == "yolo11n-pose"]

    assert len(rows) == 1, rows
    assert rows[0]["ready"] is True


def test_missing_preset_weights_are_never_listed_as_installed(workspace):
    """预置档可以摆出来给人下载，但不能被当成「本机已有」——那是下载按钮的分界。"""
    rows = models.list_models()

    assert rows
    assert not [row for row in rows if row["status"] == models.STATUS_AVAILABLE]


def test_weight_rows_carry_the_task_label_the_badge_needs(workspace):
    """「模型目录」那枚类别标签读的是 task/task_name，开放词表必须在这一步就说清。"""
    pose = models.weight_info("yolo26x-pose.pt")
    open_vocab = models.weight_info("yoloe-26x-seg.pt")

    assert pose["task"] == "pose" and "姿态" in pose["task_name"]
    assert open_vocab["task"] == "segment" and "开放词表" in open_vocab["task_name"]
