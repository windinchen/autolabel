"""模型目录与离线搜索：字段完整性、官方完整清单、中文搜索、联网降级、下载转调。

联网那几条全部 monkeypatch 掉 ``urllib.request.urlopen``，测试不碰真实网络。
"""

import io
import json
import socket
import urllib.error
import zipfile

import pytest

from autolabel import modelhub, models
from autolabel.backends import DETECT_TASKS, TASK_NAMES, weights_url

#: 界面要直接渲染的字段（含 alias），一个都不能少
FIELDS = ("id", "filename", "display_name", "task", "task_name", "task_label", "capability",
          "params_m", "params", "score", "expected_bytes", "bytes", "expected_mb",
          "size_verified", "sha256", "speed", "description", "intro", "recommended_for",
          "downloadable", "installed", "homepage", "num_keypoints", "class_note", "url")

NANO_FILES = ["yolo11n-pose.pt", "yolov8n-pose.pt", "yolo26n-pose.pt", "yolo11n.pt",
              "yolov8n.pt", "yolo26n.pt", "yolo12n.pt", "yolo11n-seg.pt", "yolov8n-seg.pt",
              "yolo11n-obb.pt", "yolov8n-obb.pt", "yolo11n-cls.pt", "yolov8n-cls.pt"]


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

    def read(self, _size: int = -1) -> bytes:
        chunk, self.payload = self.payload, b""
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def weights(tmp_path, monkeypatch):
    """空权重目录：不碰仓库里真实的 models/。"""
    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setenv("AUTOLABEL_WEIGHTS_DIR", str(directory))
    return directory


@pytest.fixture()
def partial_weights(weights):
    """已有两个完整权重 + 一个只下了一半的残留。"""
    for name in ("yolo11n.pt", "yolo11s-pose.pt"):
        (weights / name).write_bytes(zip_bytes())
    (weights / "yolo11s.pt.part").write_bytes(zip_bytes()[:30])
    return weights


# --------------------------------------------------------------------------- #
# 任务口径
# --------------------------------------------------------------------------- #

def test_task_labels_match_backend_labels():
    assert modelhub.TASK_VALUES == ("pose", "detect", "segment", "obb", "classify")
    for task in modelhub.TASK_VALUES:
        assert modelhub.TASK_LABELS[task] == TASK_NAMES[task]


def test_detection_backend_tasks_are_covered():
    """检测库能吃的任务必须都在目录里，否则界面搜到权重却标不了。"""
    assert set(DETECT_TASKS) <= set(modelhub.TASK_VALUES)


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #

def test_catalog_rows_carry_every_rendered_field(weights):
    rows = modelhub.catalog()
    assert rows
    for row in rows:
        assert not [key for key in FIELDS if key not in row], row["filename"]
        assert row["task"] in modelhub.TASK_VALUES
        assert row["task_name"] == modelhub.TASK_LABELS[row["task"]]
        assert row["task_label"] == row["task_name"]
        assert row["intro"] == row["description"] and row["intro"]
        assert row["params"] == row["params_m"]
        assert row["bytes"] == row["expected_bytes"]
        assert row["expected_mb"] == round(row["expected_bytes"] / 1048576, 1)
        assert row["url"] == weights_url(row["filename"])
        assert row["downloadable"] is True
        assert isinstance(row["installed"], bool)
        assert row["homepage"].startswith("https://")


def test_pose_entries_are_carried_over_verbatim(weights):
    """姿态条目原样带过来：现有代码还在读 map_pose / preset_visible / keypoint_set。"""
    by_id = {row["id"]: row for row in modelhub.catalog()}
    assert len(by_id) == len(modelhub.catalog())
    for preset in models.PRESET_CATALOG:
        row = by_id[preset["id"]]
        for key, value in preset.items():
            assert row[key] == value, (preset["id"], key)
        assert row["task"] == "pose" and row["capability"] == "pose"
        assert row["score"] == preset["map_pose"]
        assert row["num_keypoints"] == 17 and "COCO-17" in row["class_note"]


def test_catalog_covers_every_official_release(weights):
    """官方 release 里的每一档都要在目录里。

    以前目录只有精选的 43 条，用户自己下的 ``yolo26x.pt`` 根本搜不到，
    界面就成了「明明有这一档，搜索却说没有」。
    """
    rows = {row["filename"].lower(): row for row in modelhub.catalog()}
    missing = [name for name, _ in modelhub.WIDE_MANIFEST if name.lower() not in rows]
    assert missing == []
    assert len(rows) == len(modelhub.WIDE_MANIFEST) == 160
    counts = {task: sum(1 for row in rows.values() if row["task"] == task)
              for task in modelhub.TASK_VALUES}
    # 五个任务都得有成体量：只堆检测档等于其它任务照旧搜不到
    assert all(counts[task] >= 10 for task in modelhub.TASK_VALUES), counts
    assert modelhub.catalog() == list(modelhub.catalog(""))


def test_wide_entries_are_renderable_but_do_not_invent_numbers(weights):
    """宽目录条目：界面要能直接渲染，但官方没给的精度/参数一律留空。"""
    rows = {row["filename"]: row for row in modelhub.catalog() if not row.get("curated")}
    assert rows, "宽目录一条都没有，说明合并没生效"
    for filename, size in modelhub.WIDE_MANIFEST:
        row = rows.get(filename)
        if row is None:
            continue                        # 与精选条目同名的以精选为准，这里不重复查
        assert row["expected_bytes"] == size, filename
        assert not row["display_name"].endswith(".pt"), filename       # 不许拿文件名当中文名
        assert row["intro"] and row["speed"] and row["recommended_for"], filename
        assert row["class_note"], filename
        assert row["score"] is None and row["params_m"] is None, filename


def test_curated_entries_win_over_wide_templates(weights):
    """同名的宽目录条目不能盖掉手写说明的精选条目。"""
    curated = {row["filename"] for row in modelhub.catalog() if row.get("curated")}
    assert len(curated) == len(models.PRESET_CATALOG) + len(modelhub.EXTRA_CATALOG)
    assert {row["filename"] for row in modelhub.EXTRA_CATALOG} <= curated
    by_name = {row["filename"]: row for row in modelhub.catalog()}
    assert by_name["yolo11n.pt"]["score"] == 39.5          # 精选里核过的 mAP 还在


def test_every_wide_entry_parses_family_and_size(weights):
    """每一档都要推得出系列和尺寸档，否则中文名会变成文件名。"""
    for filename, _ in modelhub.WIDE_MANIFEST:
        stem = filename[:-3]
        assert modelhub._wide_family(stem), filename
        assert modelhub._wide_size(stem), filename


def test_just_downloaded_weight_is_searchable(weights):
    """用户本机真有 yolo26x.pt：搜得到才谈得上「模型搜索全」。"""
    names = {row["filename"] for row in modelhub.search("yolo26x")}
    assert {"yolo26x.pt", "yolo26x-pose.pt", "yolo26x-seg.pt"} <= names


def test_open_vocab_rows_say_the_class_table_is_yours(weights):
    """开放词表那几档必须自己说清「类别自己填」，否则用户会按 COCO-80 理解它。"""
    rows = [row for row in modelhub.catalog()
            if "-world" in row["filename"] or row["filename"].startswith("yoloe")]
    assert len(rows) >= 20        # 官方这两条系列本身就几十档
    for row in rows:
        assert "开放词表" in row["class_note"]
        assert row["downloadable"] and row["task"] in ("detect", "segment")
        # 精度这一列留空：官方数字量的是 LVIS，和本目录里 COCO 的 mAP 不是一把尺子
        assert row["score"] is None
        if row["task"] == "segment":
            # 1.4.8 起分割权重的轮廓真的会落盘，别再说「多边形暂不落盘」
            assert "不落盘" not in row["class_note"], row["filename"]
            assert "轮廓" in row["class_note"], row["filename"]
    hosts = {row["homepage"] for row in rows}
    assert "https://docs.ultralytics.com/models/yolo-world/" in hosts
    assert "https://docs.ultralytics.com/models/yoloe/" in hosts


@pytest.mark.parametrize("filename,expected_bytes", [
    ("yolo11n.pt", 5613764), ("yolo11s.pt", 19313732), ("yolo11m.pt", 40684120),
    ("yolo11l.pt", 51387343), ("yolo11x.pt", 114636239),
    ("yolov8n.pt", 6549796), ("yolov8x.pt", 136890692),
    ("yolo11n-seg.pt", 6182636), ("yolov8s-seg.pt", 23914764),
    ("yolo11n-obb.pt", 5795654), ("yolov8n-obb.pt", 6567590),
    ("yolo11n-cls.pt", 5790624), ("yolov8n-cls.pt", 5563076),
    ("yolo26n.pt", 5544453), ("yolo12s.pt", 19006455),
    ("yolov8s-world.pt", 27169314), ("yolov8x-world.pt", 147961954),
    ("yolov8x-worldv2.pt", 146355704), ("yoloe-26l-seg.pt", 79417413),
    ("yoloe-26x-seg.pt", 171640453), ("yoloe-11l-seg.pt", 70982416),
])
def test_catalog_expected_bytes_match_release_assets(weights, filename, expected_bytes):
    row = next(item for item in modelhub.catalog() if item["filename"] == filename)
    assert row["expected_bytes"] == expected_bytes
    assert row["size_verified"] is True


def test_sha256_only_for_weights_present_locally(weights):
    """本机有的权重填真实校验值，其余留空串，不猜。"""
    rows = {row["filename"]: row for row in modelhub.catalog()}
    assert rows["yolo11n.pt"]["sha256"] == (
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1")
    for filename in ("yolo11s.pt", "yolo11n-seg.pt", "yolo26n.pt", "yolov8n-cls.pt"):
        assert rows[filename]["sha256"] == ""
    for row in rows.values():
        assert row["sha256"] == "" or len(row["sha256"]) == 64
        assert row["speed"] and row["recommended_for"]


def test_official_metrics_missing_are_none_not_zero(weights):
    """官方没给参数规模或精度的一律 None，不能填 0 冒充。"""
    for row in modelhub.catalog():
        for key in ("params_m", "score"):
            assert row[key] is None or isinstance(row[key], (int, float))


def test_catalog_task_filter(weights):
    everything = modelhub.catalog()
    assert modelhub.catalog("") == everything
    for task in modelhub.TASK_VALUES:
        rows = modelhub.catalog(task)
        assert rows and all(row["task"] == task for row in rows)
    assert modelhub.catalog("SEGMENT")[0]["task"] == "segment"
    assert modelhub.catalog("depth") == []      # 认不出的任务给空列表，不抛错


def test_catalog_task_matches_task_of(weights):
    for row in modelhub.catalog():
        assert modelhub.task_of(row["filename"]) == row["task"], row["filename"]


@pytest.mark.parametrize("filename,task", [
    ("yolo11n-pose.pt", "pose"), ("yolov8s-pose.pt", "pose"),
    ("yolo11m-seg.pt", "segment"), ("yolov8n-seg", "segment"),
    ("yolo11s-obb.pt", "obb"), ("yolo11n-cls.pt", "classify"),
    ("yolo11n.pt", "detect"), ("yolo12s.pt", "detect"), ("yolo26n.pt", "detect"),
    ("rtdetr-l.pt", "detect"),
    ("runs/exp/weights/yolo11s-pose.pt", "pose"),
    ("bus.pt", ""), ("myweights.pt", ""), ("", ""), ("bus.jpg", ""),
])
def test_task_of(filename, task):
    assert modelhub.task_of(filename) == task


# --------------------------------------------------------------------------- #
# 本机安装状态
# --------------------------------------------------------------------------- #

def test_installed_flags_ignore_missing_and_partial(weights, partial_weights):
    rows = {row["filename"]: row for row in modelhub.catalog()}
    assert rows["yolo11n.pt"]["installed"] is True
    assert rows["yolo11s-pose.pt"]["installed"] is True
    assert rows["yolo11s.pt"]["installed"] is False          # 只有 .pt.part 残留
    assert rows["yolo11m-pose.pt"]["installed"] is False     # 预置但没下过


def test_installed_names_do_not_report_presets(weights):
    """weight_names() 会列出没下载的预置权重，installed_names() 不能跟着一起报。"""
    assert "yolo11m-pose.pt" in models.weight_names()
    assert modelhub.installed_names() == []
    assert all(row["installed"] is False for row in modelhub.catalog())


def test_installed_names_see_user_dropped_weights(weights):
    (weights / "yolo11m-seg.pt").write_bytes(zip_bytes())
    assert modelhub.installed_names() == ["yolo11m-seg.pt"]
    rows = {row["filename"]: row for row in modelhub.catalog()}
    assert rows["yolo11m-seg.pt"]["installed"] is True
    assert rows["yolo11n-seg.pt"]["installed"] is False


# --------------------------------------------------------------------------- #
# 离线搜索
# --------------------------------------------------------------------------- #

def test_search_empty_query_returns_catalog(weights):
    assert modelhub.search("") == modelhub.catalog()
    assert modelhub.search("   ") == modelhub.catalog()
    assert [r["filename"] for r in modelhub.search("", task="obb")] == [
        r["filename"] for r in modelhub.catalog("obb")]


@pytest.mark.parametrize("query,task,expected", [
    # YOLOE 那三档也算"分割"：任务型是 segment，只是类别表要自己填
    ("分割", "segment", {"yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
                          "yolov8n-seg.pt", "yolov8s-seg.pt",
                          "yoloe-26l-seg.pt", "yoloe-26x-seg.pt", "yoloe-11l-seg.pt"}),
    ("实例分割", "segment", None),
    ("旋转框", "obb", {"yolo11n-obb.pt", "yolo11s-obb.pt", "yolov8n-obb.pt"}),
    ("航拍", "obb", None),
    ("骨骼", "pose", None),
    ("姿态", "pose", None),
    ("关键点", "pose", None),
    ("分类", "classify", {"yolo11n-cls.pt", "yolov8n-cls.pt"}),
])
def test_search_chinese_task_words(weights, query, task, expected):
    rows = modelhub.search(query)
    assert rows and all(row["task"] == task for row in rows), query
    if expected is not None:
        # 宽目录补齐后同一任务词命中的档位多了一倍，这里只保证精选那几档一定在
        assert expected <= {row["filename"] for row in rows}, query


def test_search_exact_filename_beats_everything(weights):
    assert modelhub.search("yolo11n-seg.pt")[0]["filename"] == "yolo11n-seg.pt"
    assert modelhub.search("YOLO11N-SEG")[0]["filename"] == "yolo11n-seg.pt"
    rows = modelhub.search("yolo11n")
    assert [row["filename"] for row in rows] == ["yolo11n.pt", "yolo11n-pose.pt",
                                                 "yolo11n-seg.pt", "yolo11n-obb.pt",
                                                 "yolo11n-cls.pt"]


@pytest.mark.parametrize("query", ["v8", "11", "12", "26"])
def test_search_version_numbers(weights, query):
    rows = modelhub.search(query)
    assert rows and query in rows[0]["filename"]
    if query == "12":
        assert [row["filename"] for row in rows[:5]] == [
            "yolo12n.pt", "yolo12s.pt", "yolo12l.pt", "yolo12m.pt", "yolo12x.pt"]


def test_search_size_words_match_the_size_not_the_letter(weights):
    for query in ("n", "nano"):
        rows = modelhub.search(query)
        assert {row["filename"] for row in rows[:len(NANO_FILES)]} == set(NANO_FILES)
    # x 档：精选里那七档仍排在最前（写好了说明的在前），后面才跟着宽目录的其它 x 档
    assert [row["filename"] for row in modelhub.search("x")][:7] == [
        "yolo11x-pose.pt", "yolov8x-pose.pt", "yolo11x.pt", "yolov8x.pt",
        "yolov8x-world.pt", "yolov8x-worldv2.pt", "yoloe-26x-seg.pt"]
    # 尺寸词按档位匹配，不能把 l 档、c 档混进来（文件名里有 x 不算）
    assert all(modelhub._size_of(row["filename"][:-3]) == "x"
               for row in modelhub.search("x")), [row["filename"] for row in modelhub.search("x")]


def test_search_speed_hint_prefers_fastest(weights):
    rows = modelhub.search("快")
    assert rows
    assert {row["speed"] for row in rows} <= {"最快，CPU 也能实时", "最快", "快"}
    assert rows[0]["speed"].startswith("最快")
    assert modelhub.search("速度")[0]["speed"].startswith("最快")


def test_search_quality_hint_ranks_within_task(weights):
    """分类档是 top-1 准确率、其余是 mAP，只能各自排名后再排。"""
    rows = modelhub.search("精度")
    assert rows and all(row["score"] is not None for row in rows)
    assert rows[0]["filename"] == "yolo11x-pose.pt"
    assert len(rows) < len(modelhub.catalog())


def test_search_small_hint_prefers_smallest(weights):
    """搜「小」/「大」：小的排前面、大的排前面，但不钉死具体哪一个第一。

    体积倾向的分是按全目录的体积区间归一化的（``_score_range`` / ``max_mb``），
    目录里多一档 160 MB 的开放词表权重，最小那几档之间的近似平局就会换人。
    该守的是"最小的打头"，不是"某个文件必须打头"。
    """
    rows = modelhub.search("小")
    smallest = min(row["expected_mb"] for row in modelhub.catalog())
    biggest = max(row["expected_mb"] for row in modelhub.catalog())
    assert rows[0]["expected_mb"] <= smallest + 0.2
    assert all(row["expected_mb"] <= 6.0 for row in rows[:5])
    big = modelhub.search("大")
    assert big and big[0]["expected_mb"] == biggest


def test_search_keyword_bucket_ranks_below_task_words(weights):
    rows = modelhub.search("轮廓")                       # 分类档的说明里也提到轮廓
    assert [row["task"] for row in rows[:5]] == ["segment"] * 5
    assert any(row["task"] == "classify" for row in rows[5:])


def test_search_with_task_filter(weights):
    assert modelhub.search("分割", task="pose") == []
    rows = modelhub.search("nano", task="segment")
    assert [row["filename"] for row in rows[:2]] == ["yolo11n-seg.pt", "yolov8n-seg.pt"]
    assert all(row["task"] == "segment" for row in rows)
    assert all(row["filename"].startswith("yolov8") for row in modelhub.search("v8", task="pose"))


@pytest.mark.parametrize("query", ["zzz", "不存在的模型", "！！！", "  .  "])
def test_search_without_hit_returns_empty(weights, query):
    assert modelhub.search(query) == []


# --------------------------------------------------------------------------- #
# 联网增强：必须能安静降级
# --------------------------------------------------------------------------- #

def github_payload(count: int = 2) -> dict:
    return {"items": [{"full_name": f"someone/yolo{task}-weights",
                       "html_url": f"https://github.com/someone/yolo{task}-weights",
                       "description": f"社区训练的 {task} 权重"} for task in range(count)]}


def boom_factory(exc):
    def boom(*_args, **_kwargs):
        raise exc

    return boom


@pytest.mark.parametrize("exc", [
    OSError("connection refused"),
    socket.timeout("timed out"),
    urllib.error.URLError("no route to host"),
    urllib.error.HTTPError("https://api.github.com", 403, "rate limited", {}, io.BytesIO(b"{}")),
    urllib.error.HTTPError("https://api.github.com", 599, "boom", {}, io.BytesIO(b"boom")),
    ValueError("bad literal"),
])
def test_search_remote_never_raises(weights, monkeypatch, exc):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(exc))
    out = modelhub.search_remote("yolo11n-seg")
    assert out["ok"] is False
    assert out["error"] and type(exc).__name__ in out["error"]
    assert out["results"] == [] and out["remote_state"] == "failed"
    assert "没连上 GitHub" in out["note"] and type(exc).__name__ in out["note"]
    assert {row["filename"] for row in out["local"]} == {
        row["filename"] for row in modelhub.search("yolo11n-seg")}


def test_search_remote_maps_results(weights, monkeypatch):
    seen = {}

    def fake(request, timeout=None):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["agent"] = request.get_header("User-agent")
        return FakeResponse(github_payload())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    out = modelhub.search_remote("旋转框", timeout=3.0)

    assert out["ok"] is True and out["error"] == ""
    assert seen["url"].startswith("https://api.github.com/search/repositories?q=")
    assert "yolo" in seen["url"] and out["query"] == "旋转框"
    assert seen["timeout"] == 3.0 and seen["agent"] == "autolabel-modelhub"
    assert len(out["results"]) == 2
    first = out["results"][0]
    assert set(first) == {"name", "html_url", "description", "origin", "filename", "task",
                          "task_label", "intro", "params", "bytes", "url", "installed"}
    assert first["url"] == first["html_url"] and first["origin"] == "GitHub"
    assert first["installed"] is False and first["params"] is None and first["bytes"] == 0
    assert out["note"] == "社区权重未经校验，下载前建议核对文件大小"


@pytest.mark.parametrize("body", [b"<html>not json</html>", b"", json.dumps({"items": "nope"})])
def test_search_remote_unreadable_payload(weights, monkeypatch, body):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse(body))
    out = modelhub.search_remote("yolo11n")
    assert out["ok"] is False and out["results"] == [] and out["error"]
    assert out["local"]


@pytest.mark.parametrize("payload", [{"items": []}, {"items": [{"id": 7}]}])
def test_search_remote_without_usable_items_is_not_a_failure(weights, monkeypatch, payload):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse(payload))
    out = modelhub.search_remote("yolo11n")
    assert out["ok"] is True and out["error"] == "" and out["results"] == []


def test_search_remote_empty_query_skips_network(weights, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(AssertionError("空关键词不该联网")))
    out = modelhub.search_remote("   ")
    assert out["ok"] is True and out["results"] == [] and out["error"] == ""
    assert out["remote_state"] == "needs_query"               # 没输关键词不算「连不上」
    assert len(out["local"]) == len(modelhub.catalog())       # 空关键词照样给本地目录
    assert "关键词" in out["note"] and "没连上" not in out["note"]


def test_github_query_appends_yolo_only_when_missing():
    assert modelhub.github_query("分割") == "分割 yolo"
    assert modelhub.github_query("YOLO11n-pose") == "YOLO11n-pose"
    assert modelhub.github_query("") == ""


# --------------------------------------------------------------------------- #
# search_payload：界面调的那个口子
# --------------------------------------------------------------------------- #

def test_search_payload_local_only_makes_no_request(weights, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(AssertionError("默认不该联网")))
    out = modelhub.search_payload("分割")
    assert out["ok"] is True and out["error"] == ""
    assert out["results"] == [] and "未联网" in out["note"]
    assert out["query"] == "分割" and out["task"] == ""
    assert all(row["task"] == "segment" for row in out["local"])


def test_search_payload_honours_task_filter_and_dataset_root(weights, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(AssertionError("不该联网")))
    out = modelhub.search_payload("nano", task="segment", dataset_root="datasets/demo")
    assert [row["filename"] for row in out["local"][:2]] == ["yolo11n-seg.pt", "yolov8n-seg.pt"]
    assert all(row["task"] == "segment" for row in out["local"])
    assert out["task"] == "segment" and out["dataset_root"] == "datasets/demo"


def test_search_payload_remote_success(weights, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse(github_payload(1)))
    out = modelhub.search_payload("yolo11n-obb", remote=True)
    assert out["ok"] is True and out["error"] == ""
    assert len(out["results"]) == 1 and out["note"] == modelhub.REMOTE_NOTE
    assert out["local"][0]["filename"] == "yolo11n-obb.pt"
    assert all(row["task"] == "obb" for row in out["local"])


@pytest.mark.parametrize("exc", [OSError("offline"), socket.timeout("slow")])
def test_search_payload_remote_failure_explains_and_keeps_local(weights, monkeypatch, exc):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(exc))
    out = modelhub.search_payload("分割", remote=True)
    assert out["ok"] is False
    assert type(exc).__name__ in out["error"] and "没连上" in out["note"]
    assert out["results"] == [] and all(r["task"] == "segment" for r in out["local"])


def test_search_payload_empty_query_with_remote_on_is_not_a_failure(weights, monkeypatch):
    """开着联网开关却没输关键词：以前会回一句「没连上 GitHub（没有填写搜索关键词）」，自相矛盾。"""
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(AssertionError("空关键词不该联网")))
    out = modelhub.search_payload("", remote=True)
    assert out["ok"] is True and out["error"] == "" and out["results"] == []
    assert out["remote_state"] == "needs_query" and "没连上" not in out["note"]
    assert "关键词" in out["note"]
    assert len(out["local"]) == len(modelhub.catalog())


@pytest.mark.parametrize("remote", [False, True])
def test_search_payload_always_exposes_remote_state(weights, monkeypatch, remote):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(OSError("offline")))
    out = modelhub.search_payload("分割", remote=remote)
    assert out["remote_state"] == ("failed" if remote else "skipped")
    assert out["ok"] is not remote


def test_search_payload_failure_note_does_not_repeat_local_hint(weights, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(OSError("offline")))
    out = modelhub.search_payload("pose", remote=True)
    assert out["note"] == "没连上 GitHub（OSError: offline）"
    assert "本地目录" not in out["note"]        # 「下面是本地结果」由界面补一次就够


def test_search_payload_local_field_shape(weights, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", boom_factory(AssertionError("不该联网")))
    for row in modelhub.search_payload("")["local"]:
        assert {"filename", "task", "task_label", "intro", "params", "bytes", "url",
                "installed"} <= set(row)


# --------------------------------------------------------------------------- #
# 下载：不另起一套逻辑
# --------------------------------------------------------------------------- #

def test_download_delegates_to_models_start_download(weights, monkeypatch):
    seen = {}

    def fake(name, force=False, mirror=None, background=True):
        seen.update(name=name, force=force, mirror=mirror, background=background)
        return {"name": name, "stage": "running"}

    monkeypatch.setattr(models, "start_download", fake)
    assert modelhub.download("models/yolo11n-seg.pt", force=True, mirror="https://m/") == {
        "name": "yolo11n-seg.pt", "stage": "running"}
    assert seen == {"name": "yolo11n-seg.pt", "force": True, "mirror": "https://m/",
                    "background": True}


@pytest.mark.parametrize("bad", ["", "yolo11n", "notes.txt", "models/"])
def test_download_rejects_non_weights(weights, bad):
    with pytest.raises(ValueError):
        modelhub.download(bad)


def test_download_url_reuses_backend_url_builder(weights):
    assert modelhub.download_url("yolo12s.pt") == weights_url("yolo12s.pt")
    assert modelhub.download_url("yolo12s.pt", mirror="") == (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt")
    assert modelhub.download_url("bus.jpg") == ""


def test_download_state_passthrough(weights, monkeypatch):
    monkeypatch.setattr(models, "download_state", lambda name: {"name": name, "percent": 42})
    assert modelhub.download_state("yolo11n.pt") == {"name": "yolo11n.pt", "percent": 42}


# --- 第 19 轮体检：搜索框里的中文整句 ------------------------------------- #

def test_a_whole_chinese_sentence_finds_something(weights):
    """中文没有空格，一整句以前被当成**一个**词条去整串匹配，永远搜不到。

    ``能标轮廓的`` 这种话是最自然的输入，返回 0 条就等于告诉用户"这搜索框没用"。
    """
    rows = modelhub.search("能标轮廓的")

    assert rows, "整句中文不该一律搜不到"
    assert [row["task"] for row in rows[:3]] == ["segment"] * 3


def test_a_vague_sentence_ranks_by_what_it_said(weights):
    """``最省显存的`` 该把体积最小的排在前面，``精度高`` 该把最准的排在前面。"""
    small = modelhub.search("最省显存的")
    big = modelhub.search("精度高")

    # 同上：体积区间随目录变，最小的几档之间会换人，守"最小的打头"就够了
    assert small and small[0]["expected_mb"] <= \
        min(r["expected_mb"] for r in modelhub.catalog()) + 0.2
    assert big and big[0]["params_m"] >= 5.0


def test_a_nonsense_phrase_still_returns_nothing(weights):
    """切片的代价是不能把"没这个东西"变成"返回全部"。

    ``模型`` 两个字几乎条条描述都有，命中它等于没给线索——这种碎片要按命中比例丢掉。
    """
    assert modelhub.search("不存在的模型") == []
    assert modelhub.search("哈哈哈哈哈") == []


@pytest.mark.parametrize("phrase, task", [("旋转框", "obb"), ("分类背景", "classify")])
def test_short_precise_chinese_queries_stay_tight(weights, phrase, task):
    """本来就搜得准的短词不该被切片兜底弄成"全都沾一点"。"""
    rows = modelhub.search(phrase)

    assert rows and {row["task"] for row in rows} == {task}


@pytest.mark.parametrize("huge", ["yolo" * 2500, "分割" * 800, "！" * 5000])
def test_absurdly_long_query_does_not_explode(weights, huge):
    """搜索框里粘一整篇文档：不炸、不卡，结果该是啥是啥。"""
    rows = modelhub.search(huge)

    assert isinstance(rows, list)


def test_github_query_trims_a_pasted_paragraph(weights):
    """粘一整段话去 GitHub 搜只会回来一坨无关仓库，取前半句就够。"""
    assert len(modelhub.github_query("安全帽 " + "模型" * 500)) <= 130


# --------------------------------------------------------------------------- #
# 「更新目录」：官方 release 的最新档位与体积
# --------------------------------------------------------------------------- #

#: 假 release：一档改体积、一档全新、SAM 与深度分支各占一档、再来一个非 .pt 资源
RELEASE_V9 = {
    "tag_name": "v9.1.0",
    "html_url": "https://github.com/ultralytics/assets/releases/tag/v9.1.0",
    "published_at": "2026-11-02T04:05:06Z",
    "assets": [
        {"name": "yolo11n.pt", "size": 6_000_000},
        {"name": "yolo99n-seg.pt", "size": 41_943_040},
        {"name": "sam3_b.pt", "size": 90_000_000},
        {"name": "yolo26l-depth.pt", "size": 70_000_000},
        {"name": "yolo11n-pose.tar.gz", "size": 1},
    ],
}


def fake_release(monkeypatch, payload):
    """把 release 接口换成假数据：这几条测试一次真实网络都不碰。

    刷新会把下载地址改到假 release 那一版，登记下来让 monkeypatch 测试后还原，
    否则下一个断言 ``.../v8.4.0/xxx.pt`` 的测试会莫名其妙地红。
    """
    def respond(request, timeout=0, **_kwargs):
        return FakeResponse(payload)

    monkeypatch.setattr(modelhub.backends, "ASSETS_BASE", modelhub.backends.ASSETS_BASE)
    monkeypatch.setattr(modelhub.backends, "ASSETS_RELEASE", modelhub.backends.ASSETS_RELEASE)
    monkeypatch.setattr(modelhub.urllib.request, "urlopen", respond)


def fake_offline(monkeypatch, reason="名字解释不了"):
    def boom(request, timeout=0, **_kwargs):
        raise urllib.error.URLError(reason)

    monkeypatch.setattr(modelhub.urllib.request, "urlopen", boom)


def test_release_branches_this_library_cannot_run_stay_out(weights):
    """官方 release 里一半资源本库标不了东西：收进目录就是让人白下几十 MB。"""
    for name, want in (("yolo11n.pt", True), ("yolo11n-seg.pt", True), ("yolov8x-worldv2.pt", True),
                       ("yolo26x-pose.pt", True), ("yolo26l-depth.pt", False),
                       ("yolo26m-reid-market.pt", False), ("yolo26s-sem-ade20k.pt", False),
                       ("yolo11n-grayscale.pt", False), ("sam3_b.pt", False),
                       ("FastSAM-x.pt", False), ("mobile_sam.pt", False), ("yolov3u.pt", False),
                       ("yolo11n.torchscript", False), ("", False)):
        assert modelhub.refresh_supported(name) is want, name


def test_refresh_adds_new_tier_and_fixes_sizes(weights, monkeypatch):
    """新档位要带中文名和介绍，体积按官方改正，下载地址跟着换到那一版。"""
    fake_release(monkeypatch, RELEASE_V9)

    result = modelhub.refresh_catalog()
    assert result["ok"] and result["changed"], result
    assert result["added"] == ["yolo99n-seg.pt"]
    assert {"filename": "yolo11n.pt", "old": 5_613_764, "new": 6_000_000} in result["resized"]
    assert result["skipped"] == 2

    rows = {row["filename"]: row for row in modelhub.catalog()}
    fresh = rows["yolo99n-seg.pt"]
    assert not fresh["display_name"].endswith(".pt"), "不许拿文件名当中文名"
    assert fresh["intro"] and fresh["class_note"] and fresh["task"] == "segment"
    assert fresh["expected_mb"] == 40.0
    assert rows["yolo11n.pt"]["expected_bytes"] == 6_000_000
    # 新档位只存在于最新那一版：地址还按 v8.4.0 拼就是 404
    assert modelhub.download_url("yolo99n-seg.pt").endswith("/v9.1.0/yolo99n-seg.pt")


def test_refresh_note_says_what_actually_changed(weights, monkeypatch):
    """那句提示得是真话：新增多少、改了几档体积、多少档用不上，各说各的。"""
    fake_release(monkeypatch, RELEASE_V9)

    note = modelhub.refresh_catalog()["note"]

    assert "新增 1 档" in note and "体积按 v9.1.0 改正" in note
    assert "2 档本库用不上" in note and "已经找不到了" in note

    again = modelhub.refresh_catalog()["note"]
    assert "没有新档位，体积也没变" in again, "刷第二次不该再报一遍新增"


def test_refresh_survives_a_second_identical_run(weights, monkeypatch):
    """连点两次「更新目录」不许把目录刷成两行一档。"""
    fake_release(monkeypatch, RELEASE_V9)
    modelhub.refresh_catalog()
    first = [row["filename"] for row in modelhub.catalog()]

    second = modelhub.refresh_catalog()
    assert second["changed"] is False and second["added"] == []
    assert [row["filename"] for row in modelhub.catalog()] == first


def test_offline_refresh_keeps_the_catalog_it_has(weights, monkeypatch):
    """断网不许把目录刷没：说清原因、缓存保持原样，搜索照旧有结果。"""
    before = modelhub.catalog_info()
    fake_offline(monkeypatch)

    result = modelhub.refresh_catalog()

    assert result["ok"] is False and result["changed"] is False
    assert "没连上 GitHub" in result["note"] and "标注照常能用" in result["note"]
    assert modelhub.catalog_info() == before, "刷新失败不该改动目录来源"
    assert modelhub.search("yolo11n"), "目录还是得能用"


def test_refresh_is_remembered_across_requests(weights, monkeypatch):
    """界面那句「已核对到哪一版、什么时候刷的」靠这份缓存，重启服务也得说得出。"""
    fake_release(monkeypatch, RELEASE_V9)
    modelhub.refresh_catalog()

    info = modelhub.catalog_info()
    assert info["refreshed"] is True and info["release"] == "v9.1.0"
    assert info["published_at"] == "2026-11-02" and info["official_total"] == 4
    assert info["usable_total"] == 2 and info["added"] == 1
    assert info["fetched_at"], "得留下刷新时间，否则用户不知道该不该再点"
    assert modelhub.search_payload("分割")["catalog"]["release"] == "v9.1.0"


def test_catalog_info_names_the_bundled_list_before_any_refresh(weights):
    """没刷过的时候也得说清目录是哪来的，别让人以为界面坏了一半。"""
    info = modelhub.catalog_info()

    assert info["refreshed"] is False and info["release"] == ""
    assert info["builtin_release"] and info["builtin_total"] == len(modelhub.WIDE_MANIFEST)
