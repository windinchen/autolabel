"""「目标检测」二级模块的界面契约：页面可取、静态白名单完整、脚本参数化、检测框按库读写。

这一页与「提示词标注」共用同一份 objects.js（本仓库忌讳同一件事两个入口），
所以这里既测新页面，也反过来守住老页面没被参数化改造写坏。
"""
import io
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

import pytest

from test_upload import jpeg_bytes, upload

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "autolabel", "static")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = "// ---- 库的差异到此为止，下面的正文对两个库必须一模一样 ----"


def read_static(name):
    with io.open(os.path.join(STATIC, name), encoding="utf-8") as handle:
        return handle.read()


def read_review_py():
    with io.open(os.path.join(REPO, "autolabel", "review.py"), encoding="utf-8") as handle:
        return handle.read()


def get(base, path):
    request = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            ctype = response.headers.get("Content-Type", "")
            body = response.read()
            if ctype.startswith("application/json"):
                return response.status, json.loads(body), response.headers
            return response.status, body, response.headers
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8", "replace")), exc.headers
        except (json.JSONDecodeError, UnicodeDecodeError):
            return exc.code, raw, exc.headers


def post(base, path, payload):
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read()), response.headers
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw), exc.headers
        except json.JSONDecodeError:
            return exc.code, {"error": raw}, exc.headers


def adopt_detect(root, *rels):
    """把素材引进目标检测库（检测框接口只认本库索引里有的图）。"""
    from autolabel.dataset import adopt_from_pool

    return adopt_from_pool(root, list(rels), lib="detect")


@pytest.fixture()
def weights(tmp_path, monkeypatch):
    """指向一个只放两副权重的临时目录：一副检测（真能加载）、一副姿态（必须被挡在门外）。"""
    source = os.path.join(REPO, "models")
    wanted = ["yolo11n.pt", "yolo11n-pose.pt"]
    missing = [name for name in wanted if not os.path.exists(os.path.join(source, name))]
    if missing:
        pytest.skip(f"缺少本机权重，跳过检测页集成用例：{'、'.join(missing)}")
    folder = tmp_path / "weights"
    folder.mkdir()
    for name in wanted:
        os.symlink(os.path.join(source, name), str(folder / name))
    monkeypatch.setenv("AUTOLABEL_WEIGHTS_DIR", str(folder))
    return str(folder)


# --------------------------------------------------------------------------- //
# 页面与静态资源
# --------------------------------------------------------------------------- //
def test_detect_page_is_served_with_lib_marker(server):
    """`/detect` 必须带着 data-lib="detect" 出来：脚本靠它选库，缺了就默默写进提示词库。"""
    for path in ("/detect", "/detect/"):
        status, body, headers = get(server[0], path)
        text = body.decode("utf-8")
        assert status == 200, path
        assert "text/html" in headers["Content-Type"]
        assert 'data-page="detect"' in text
        assert 'data-lib="detect"' in text
        assert "目标检测" in text
    # 提示词页不能被同一个值糊到检测库：要么显式写 objects，要么干脆不写（脚本默认就是它）
    _status, body, _ = get(server[0], "/objects")
    text = body.decode("utf-8")
    assert 'data-page="objects"' in text
    assert 'data-lib="detect"' not in text


def test_detect_page_reuses_objects_script_without_a_fork(server):
    """不许为第二页再复制一份脚本：多一个入口就多一份会漂移的逻辑。"""
    detect = read_static("detect.html")
    objects = read_static("objects.html")
    assert 'src="/objects.js"' in detect
    assert 'src="/objects.js"' in objects
    assert "detect.js" not in read_review_py()
    assert not os.path.exists(os.path.join(STATIC, "detect.js"))
    # 运行按钮与类别面板在检测页换了名字
    assert "批量目标检测" in detect
    assert "类别管理" in detect
    assert 'id="cls-import"' in detect


def test_static_whitelist_covers_every_referenced_asset(server):
    """静态文件走白名单：页面引用了却没进白名单，点开就是一屏 404。"""
    review = read_review_py()
    scripts = re.search(r'elif path in \(([^)]*?\.js[^)]*?)\)', review).group(1)
    allowed = set(re.findall(r'"(/[^"]+)"', scripts))
    allowed |= {"/" + name for name in re.findall(r'_send_page\("([^"]+)"\)', review)}
    allowed |= {"/app.css", "/favicon.ico"}
    assert "/objects.js" in allowed and "/app.css" in allowed and "/common.js" in allowed

    referenced = set()
    for name in sorted(os.listdir(STATIC)):
        if not name.endswith(".html"):
            continue
        html = read_static(name)
        for ref in re.findall(r'(?:src|href)="/([^"#?]+)', html):
            if ref.endswith((".js", ".css")):
                referenced.add("/" + ref)
    missing = sorted(referenced - allowed)
    assert not missing, f"页面引用了但白名单没放行：{missing}"

    for asset in sorted(referenced):
        status, _body, headers = get(server[0], asset)
        assert status == 200, asset
        assert "javascript" in headers["Content-Type"] or "text/css" in headers["Content-Type"]
    # 白名单里不许留已经删掉的文件
    for asset in sorted(allowed):
        if asset.endswith((".js", ".css", ".html", ".ico")):
            assert os.path.exists(os.path.join(STATIC, asset.lstrip("/"))), asset


def test_objects_script_stays_library_agnostic():
    """配置表以下不许再出现写死的库名，否则另一个页面被悄悄写坏。"""
    source = read_static("objects.js")
    assert source.count(MARKER) == 1, "两个库的分界注释被删了，正文有没有硬编码就没人管了"
    head, body = source.split(MARKER, 1)
    for lib in ("objects", "detect"):
        assert re.search(r"\n  %s: \{" % lib, head), f"配置表缺 {lib} 那一行"
    assert 'items: "detect_items"' in head and 'items: "object_items"' in head
    assert 'exportKind: "yolo_detect"' in head

    offenders = []
    for match in re.finditer(r'"(objects|detect|prompt)"', body):
        prefix = body[max(0, match.start() - 4):match.start()]
        if prefix.endswith("pk("):        # pk("objects") 是两个库共用的字段名后缀，不是库名
            continue
        offenders.append((body[:match.start()].count("\n") + 1, match.group(0)))
    assert not offenders, f"正文里还有写死的库名：{offenders}"
    for pattern in (r'openPool\(\s*true\s*,\s*"', r'\blib:\s*"', r'location\.href = "/objects"',
                    r'location\.href = "/detect"', r'libItems\(\s*,\s*"'):
        assert not re.search(pattern, body), f"正文里残留硬编码：{pattern}"
    assert body.count("CFG.lib") >= 3 and "pk(" in body


def test_every_shortcut_advertised_by_detect_page_is_bound():
    """速查表里写的单键必须真能按——两个库共用一个脚本，别页写了自己没接。"""
    html = read_static("detect.html")
    js = read_static("objects.js")
    advertised = set()
    for line in re.findall(r"<div><span>([^<]+)</span>", html):
        for token in re.split(r"[/（）\s]+", line):
            if len(token) == 1 and token.isalpha() and token.isascii():
                advertised.add(token.lower())
    assert advertised, "检测页速查表是空的"
    assert "o" in advertised, "检测页要能一键去提示词标注"
    missing = sorted(key for key in advertised if f'case "{key}"' not in js)
    assert not missing, f"检测页宣传了没绑定的快捷键：{missing}"


def test_navigation_offers_the_third_module_everywhere():
    """门户卡片、左侧导航数据、命令面板与批次筛选都要认得目标检测。"""
    common = read_static("common.js")
    assert 'id: "detect", href: "/detect"' in common
    assert "ensureMenuModules" in common and "ensureRailModules" in common
    portal = read_static("portal.js")
    block = portal[portal.index("const MODULE_CARDS"):portal.index("function buildStats")]
    assert 'id: "detect", href: "/detect"' in block
    for banned in ("素材池", "两库共用", "二级模块"):
        assert banned not in block
    assert 'for (const key of ["pose", "objects", "detect"])' in portal
    assert '"export-yolo-detect"' in portal and '"yolo_detect"' in portal
    assert 'key: "detect_items"' in portal
    html = read_static("portal.html")
    assert '<option value="detect">目标检测</option>' in html
    assert "三个标注库" in html


# --------------------------------------------------------------------------- //
# 检测框读写：同一套接口，靠 lib=detect 分流
# --------------------------------------------------------------------------- //
def seed_detect_class_table(root, *names):
    from autolabel import fields as field_store

    for index, name in enumerate(names):
        field_store.upsert_field(root, name=name, label=f"类别{index}", lib="detect")
    return field_store.fields_payload(root, "detect")


def test_box_endpoints_split_by_lib_argument(server, dataset):
    """`lib=detect` 的读写只动检测库，提示词库一个字都不该跟着变。"""
    base, store = server
    images = [row["image"] for row in store.dataset_payload()["object_items"]]
    assert images, "素材应当已经引进提示词库"
    adopt_detect(dataset, *images)
    seed_detect_class_table(dataset, "bus", "person")

    before = {row["image"]: row for row in get(base, "/api/dataset")[1]["object_items"]}
    target = images[0]

    status, item, _ = get(base, "/api/prompt/item?image=" + urllib.parse.quote(target) + "&lib=detect")
    assert status == 200, item
    assert item["lib"] == "detect"

    boxes = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3},
             {"cls": 1, "cx": 0.25, "cy": 0.6, "w": 0.1, "h": 0.2}]
    status, written, _ = post(base, "/api/prompt/item",
                              {"image": target, "objects": boxes, "status": "approved",
                               "lib": "detect"})
    assert status == 200, written
    assert written["detect_objects"] == 2
    assert written["detect_status"] == "approved"

    _status, dataset_payload, _ = get(base, "/api/dataset")
    detect_row = {row["image"]: row for row in dataset_payload["detect_items"]}[target]
    assert detect_row["detect_objects"] == 2 and detect_row["detect_status"] == "approved"
    object_row = {row["image"]: row for row in dataset_payload["object_items"]}[target]
    assert object_row["prompt_objects"] == before[target]["prompt_objects"], "写检测库串到了提示词库"

    # 不写 lib 的默认值必须仍是提示词库：漏带参数的老页面不能莫名改到别的库
    status, defaulted, _ = post(base, "/api/prompt/status", {"image": target, "status": "rejected"})
    assert status == 200, defaulted
    assert defaulted.get("prompt_status") == "rejected"
    _status, again, _ = get(base, "/api/prompt/item?image=" + urllib.parse.quote(target) + "&lib=detect")
    assert again["status"] == "approved", "默认库的状态写到了检测库"

    status, coords, _ = get(base, "/api/prompt/coords?format=csv&lib=detect")
    assert status == 200
    text = coords.decode("utf-8") if isinstance(coords, bytes) else str(coords)
    assert "bus" in text and target in text

    status, summary, _ = get(base, "/api/prompt/summary?lib=detect")
    assert status == 200 and summary["detect"]["boxes"] == 2, summary
    assert summary["detect"]["lib"] == "detect"

    status, undone, _ = post(base, "/api/prompt/undo", {"image": target, "lib": "detect"})
    assert status == 200, undone
    _status, dataset_payload, _ = get(base, "/api/dataset")
    row = {item["image"]: item for item in dataset_payload["detect_items"]}[target]
    assert row["detect_objects"] != 2 or row["detect_status"] != "approved"


def test_field_endpoints_split_by_lib_argument(server, dataset):
    """类别表也按库分家：删检测库的类别不许伤到提示词库的字段登记表。"""
    base, store = server
    images = [row["image"] for row in store.dataset_payload()["object_items"]]
    adopt_detect(dataset, *images)
    from autolabel import fields as field_store

    prompt_field = field_store.upsert_field(dataset, name="bottle", label="水瓶", lib="objects")
    saved = field_store.upsert_field(dataset, name="bus", label="公交车", lib="detect")

    status, payload, _ = get(base, "/api/fields?lib=detect")
    assert status == 200
    assert [row["name"] for row in payload["fields"]] == ["bus"]

    status, saved_row, _ = post(base, "/api/fields/save",
                                {"id": saved["class_id"], "name": "bus", "label": "公共汽车",
                                 "prompt_hint": "路上的公交车", "lib": "detect"})
    assert status == 200, saved_row
    table = get(base, "/api/fields?lib=detect")[1]["fields"]
    assert table[0]["label"] == "公共汽车"
    assert table[0]["class_id"] == saved["class_id"], "改中文名不该换类别号"

    status, removed, _ = post(base, "/api/fields/delete",
                              {"id": saved["id"], "drop_boxes": False, "lib": "detect"})
    assert status == 200, removed
    assert get(base, "/api/fields?lib=detect")[1]["fields"] == []
    still = get(base, "/api/fields")[1]["fields"]
    assert [row["name"] for row in still] == ["bottle"], "删检测库类别带走了提示词库字段"
    assert still[0]["class_id"] == prompt_field["class_id"]


# --------------------------------------------------------------------------- //
# 目标检测专属接口
# --------------------------------------------------------------------------- //
def test_detect_models_lists_usable_and_explains_the_rest(server, weights):
    """用不了的权重不能凭空消失：界面要靠 note 告诉用户为什么它不在下拉框里。"""
    base, _ = server
    status, payload, _ = get(base, "/api/detect/models")
    assert status == 200, payload
    usable = {row["name"]: row for row in payload["models"]}
    unusable = {row["name"]: row for row in payload["unusable"]}
    assert "yolo11n.pt" in usable and usable["yolo11n.pt"]["usable"] is True
    assert "yolo11n-pose.pt" not in usable, "姿态权重混进了检测库下拉框"
    note = unusable["yolo11n-pose.pt"]["note"]
    assert "检测库用不了" in note and "姿态" in note
    assert set(payload["prefs"]) <= {"model", "classes", "conf", "updated_at"}
    assert payload["device"]


def run_static_function(name, argument):
    """把 objects.js 里的一个纯函数抠出来交给 node 求值。

    归并/截断这类文案逻辑写在 JS 里，隔正则是猜不出结果的；node 不在就跳过，
    不为一个提示语把测试绑死在某个运行时上。
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("本机没有 node，跳过前端纯函数求值")
    source = read_static("objects.js")
    params, body = source.split("function %s(" % name, 1)[1].split(") {", 1)
    body = body.split("\n}", 1)[0]
    script = ("const arg = JSON.parse(process.argv[1]);"
              "const fn = Function(...JSON.parse(process.argv[2]), process.argv[3]);"
              "process.stdout.write(JSON.stringify(fn(arg)));")
    done = subprocess.run([node, "-e", script, json.dumps(argument),
                           json.dumps([item.strip() for item in params.split(",")]), body],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


POSE_NOTE = "这是「人体姿态」权重，检测库用不了；人体骨骼请回姿态标注"


def test_unusable_weight_note_says_each_reason_once():
    """一排姿态权重不能把同一句话刷十几遍。

    实测：本机摆了 13 个 ``*-pose.pt``，「用不了的权重」就刷出 13 模一样的
    「这是「人体姿态」权重，检测库用不了；人体骨骼请回姿态标注」，真正要看的
    那句被埋在最底下。归并是展示层的事——接口照旧按文件逐条回 ``note``，
    那份还要给「设为默认」的报错用。
    """
    pose = [{"name": "yolo%s-pose.pt" % tag, "note": POSE_NOTE}
            for tag in ("11n", "11s", "11m", "11l", "11x", "8n", "8s", "8m", "8l", "8x",
                        "26n", "26s", "26m")]
    got = run_static_function("unusableWeightsNote", pose)

    assert got["text"].count("人体骨骼请回姿态标注") == 1, got["text"]
    assert "等 13 个" in got["text"], got["text"]
    assert "yolo11n-pose.pt" in got["text"] and "yolo26m-pose.pt" not in got["text"], \
        "正文只留前三个，其余收进悬停提示"
    for row in pose:
        assert row["name"] in got["title"], "悬停提示要能找回每一个权重"


def test_unusable_weight_note_keeps_single_and_mixed_cases():
    """归并不能把「只有一副用不了」这种最常见的情况写成复数，也不能把不同原因并掉。"""
    only = [{"name": "yolo11n-cls.pt", "note": "这是「图像分类」权重，检测库用不了"}]
    assert run_static_function("unusableWeightsNote", only)["text"] == \
        "用不了的权重 · yolo11n-cls.pt：这是「图像分类」权重，检测库用不了"

    mixed = run_static_function("unusableWeightsNote", [
        {"name": "a-pose.pt", "note": POSE_NOTE},
        {"name": "b-pose.pt", "note": POSE_NOTE},
        {"name": "c-cls.pt", "note": "这是「图像分类」权重，检测库用不了"},
        {"name": "d.pt"},                                  # 后端没给原因
    ])
    assert mixed["text"].count("人体骨骼请回姿态标注") == 1, mixed["text"]
    assert "c-cls.pt" in mixed["text"] and "d.pt" in mixed["text"], mixed["text"]
    assert run_static_function("unusableWeightsNote", [])["text"] == ""


def test_unusable_weight_note_is_rendered_through_the_grouping():
    """三个面板必须都走归并后的文案，别留一处还在逐个拼。"""
    source = read_static("objects.js")
    assert "unusableWeightsNote(modelState.unusable)" in source
    assert "modelState.unusable.map(" not in source, "还有一处逐个拼「名字：原因」"
    assert 'el.title = unusable.title' in source, "完整清单得有地方放"


def test_detect_model_classes_and_import_roundtrip(server, dataset, weights):
    """列权重自带类别 → 导入本库类别表；同名类别永不换号是这功能的底线。"""
    base, _ = server
    status, profile, _ = get(base, "/api/detect/model/classes?model=yolo11n.pt")
    assert status == 200, profile
    assert profile["class_count"] == len(profile["classes"]) > 10
    names = [row["name"] for row in profile["classes"]]
    assert len(names) == len(set(names))
    assert all(re.search(r"[\u4e00-\u9fff]", str(row["label"])) for row in profile["classes"][:5])
    assert {"class_id", "name", "label"} <= set(profile["classes"][0])

    status, first, _ = post(base, "/api/detect/classes/add",
                            {"model": "yolo11n.pt", "names": ["bus", "person"]})
    assert status == 200, first
    assert first["added"] == 2 and first["updated"] == 0 and first["skipped"] == []
    ids = {row["name"]: row["class_id"] for row in first["fields"]}
    assert ids == {"bus": 0, "person": 1} or set(ids) == {"bus", "person"}
    assert first["next_class_id"] == max(ids.values()) + 1

    status, again, _ = post(base, "/api/detect/classes/add",
                           {"model": "yolo11n.pt", "names": ["person", "bicycle", "不存在的类"]})
    assert status == 200, again
    assert again["not_in_model"] == ["不存在的类"]
    assert again["updated"] == 1 and again["added"] == 1
    second = {row["name"]: row["class_id"] for row in again["fields"]}
    assert second["person"] == ids["person"], "同名类别被换了类别号，老标签会集体指错东西"
    assert "不存在的类" not in second

    _status, dataset_payload, _ = get(base, "/api/dataset")
    assert [row["name"] for row in dataset_payload["detect_classes"]["fields"]] == [
        "bus", "person", "bicycle"]
    assert set(second) == {"bus", "person", "bicycle"}
    assert dataset_payload["fields"]["fields"] == [], "导入的类别串进了提示词库字段登记表"


def test_detect_prefs_roundtrip_refills_the_last_choice(server, dataset, weights):
    """上次用的权重 / 类别 / 阈值要能回填，否则每次开页都要重挑一遍。"""
    from autolabel import detect

    base, _ = server
    detect.save_prefs(dataset, model="yolo11n.pt", classes=["bus", "person"], conf=0.42)
    status, payload, _ = get(base, "/api/detect/models")
    assert status == 200
    assert payload["prefs"]["model"] == "yolo11n.pt"
    assert payload["prefs"]["classes"] == ["bus", "person"]
    assert payload["prefs"]["conf"] == 0.42


def test_detect_run_job_rejects_empty_lib_and_accepts_images(server, dataset, monkeypatch):
    """空库要当场说清楚，别丢给用户一个跑完什么也没发生的任务。"""
    base, store = server
    status, body, _ = post(base, "/api/detect/run", {"model": "yolo11n.pt", "classes": ["bus"]})
    assert status == 400, body
    assert "目标检测库" in body["error"]

    from autolabel.upload import Job

    images = [row["image"] for row in store.dataset_payload()["object_items"]]
    adopt_detect(dataset, *images)
    submitted = []

    def fake_submit(kind, files, params=None):
        submitted.append((kind, list(files), dict(params or {})))
        return Job(id="job1", kind=kind, files=list(files), params=dict(params or {}))

    monkeypatch.setattr(store.jobs, "submit", fake_submit)
    status, job, _ = post(base, "/api/detect/run",
                          {"model": "yolo11n.pt", "classes": ["bus"], "conf": 0.3,
                           "overwrite": True, "images": images})
    assert status == 202, job
    assert job["kind"] == "detect"
    assert submitted and submitted[0][0] == "detect"
    assert submitted[0][2]["detect_classes"] == ["bus"]
    assert submitted[0][2]["model"] == "yolo11n.pt"
    assert submitted[0][2]["conf"] == 0.3 and submitted[0][2]["overwrite"] is True


def test_detect_upload_job_submits_a_detect_upload_task(server, dataset, monkeypatch):
    """上传即出框这条路要带权重与类别进任务参数，界面才有东西可复核。"""
    base, _store = server
    _status, _headers, body = upload(base, "detect-frame.jpg", jpeg_bytes(7))
    stored = json.loads(body.decode("utf-8"))["stored"]
    from autolabel.upload import Job

    submitted = []

    def fake_submit(kind, files, params=None):
        submitted.append((kind, list(files), dict(params or {})))
        return Job(id="job2", kind=kind, files=list(files), params=dict(params or {}))

    monkeypatch.setattr(_store.jobs, "submit", fake_submit)
    status, job, _ = post(base, "/api/detect/upload",
                          {"files": [stored], "model": "yolo11n.pt", "classes": ["bus"],
                           "conf": 0.25, "label": True, "fps": 1.0, "max_frames": 20,
                           "batch_name": "检测页导入"})
    assert status == 202, job
    assert job["kind"] == "detect_upload"
    assert submitted[0][0] == "detect_upload"
    assert submitted[0][2]["detect_classes"] == ["bus"]
    assert submitted[0][2]["max_frames"] == 20


def test_detect_export_writes_its_own_yolo_dataset(server, dataset):
    """检测库的导出 kind 是 yolo_detect：与提示词库那份互不覆盖，训练命令也不同。"""
    base, store = server
    images = [row["image"] for row in store.dataset_payload()["object_items"]]
    adopt_detect(dataset, *images)
    seed_detect_class_table(dataset, "bus")
    post(base, "/api/prompt/item",
         {"image": images[0], "objects": [{"cls": 0, "cx": 0.4, "cy": 0.5, "w": 0.2, "h": 0.3}],
          "status": "approved", "lib": "detect"})
    status, payload, _ = post(base, "/api/prompt/export", {"lib": "detect", "val_ratio": 0.0})
    assert status == 200, payload
    info = payload["export"]
    assert info["lib"] == "detect"
    assert "yolo-detect" in info["out"]
    assert [row["name"] for row in info["classes"]] == ["bus"]
    assert os.path.exists(os.path.join(info["out"], "yolo_meta.json")), "检测库的导出元数据名不同"
    _status, dataset_payload, _ = get(base, "/api/dataset")
    rows = [row for row in dataset_payload["exports_recent"] if row["kind"] == "yolo_detect"]
    assert rows, dataset_payload["exports_recent"]
    assert rows[0]["path"] == info["out"]
    # 门户给这一行配的训练命令必须是 detect，不是默认的 pose
    assert "yolo_detect:" in read_static("common.js")
