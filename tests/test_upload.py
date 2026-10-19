"""上传接口、后台标注任务与坐标下载。"""

import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest
from PIL import Image

from _helpers import write_label
from autolabel.batches import load_batches
from autolabel.review import ReviewHandler
from autolabel.upload import kind_for, safe_upload_name

SAMPLE = ("0 0.500000 0.500000 0.200000 0.400000 " +
          " ".join(f"{0.30 + i * 0.02:.6f} {0.20 + i * 0.03:.6f} 2" for i in range(17)))


def jpeg_bytes(seed: int) -> bytes:
    """生成一张与 fixture 不重复的 JPEG，用于走真实导入去重逻辑。"""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (200, 240, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def call(base, path, data=None, headers=None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def upload(base, name, data):
    return call(base, "/api/upload?name=" + urllib.parse.quote(name), data=data,
                headers={"Content-Type": "application/octet-stream"})


def wait_job(base, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _status, _headers, body = call(base, "/api/jobs")
        jobs = json.loads(body)["jobs"]
        job = next(j for j in jobs if j["id"] == job_id)
        if job["finished"]:
            return job
        time.sleep(0.2)
    raise AssertionError(f"任务 {job_id} 超时未完成")


def test_upload_then_import_job_without_model(server, dataset):
    base, _store = server
    status, _headers, body = upload(base, "新照片 01.jpg", jpeg_bytes(1))
    assert status == 200, body
    stored = json.loads(body)["stored"]
    assert stored.endswith("_新照片_01.jpg"), stored  # 空格转下划线，中文保留

    status, _headers, body = call(base, "/api/jobs", data=json.dumps(
        {"kind": "image", "files": [stored], "params": {"label": False}}).encode(),
        headers={"Content-Type": "application/json"})
    assert status == 202, body
    job = wait_job(base, json.loads(body)["id"])
    assert job["ok"] is True, job["error"]
    assert job["result"]["imported"] == 1

    status, _headers, body = call(base, "/api/dataset")
    items = json.loads(body)["items"]
    assert len(items) == 3  # fixture 的 2 张 + 新上传的 1 张
    assert not os.listdir(os.path.join(dataset, ".autolabel", "uploads"))  # 临时文件已清理


def test_coords_download_is_attachment_csv(server, dataset):
    base, _store = server
    from _helpers import idx_names

    write_label(dataset, sorted(idx_names(dataset))[0], [SAMPLE])
    status, headers, body = call(base, "/api/coords?format=csv")
    assert status == 200
    assert "attachment" in headers.get("Content-Disposition", "")
    assert "coords.csv" in headers.get("Content-Disposition", "")
    text = body.decode("utf-8")
    assert "bbox_x1" in text.splitlines()[0]
    assert len(text.strip().splitlines()) == 2  # 表头 + 1 个人


def test_upload_rejects_unknown_extension(server):
    base, _store = server
    status, _headers, body = upload(base, "payload.exe", b"MZ\x00\x00")
    assert status == 400
    assert "不支持" in json.loads(body)["error"]


def test_upload_respects_size_cap(server):
    base, _store = server
    original = ReviewHandler.max_upload_mb
    ReviewHandler.max_upload_mb = 0
    try:
        status, _headers, body = upload(base, "big.jpg", jpeg_bytes(2))
        assert status == 413
        assert "上限" in json.loads(body)["error"]
    finally:
        ReviewHandler.max_upload_mb = original


def test_oversized_upload_reports_limit_instead_of_broken_pipe(server, dataset):
    """超限请求必须把请求体读干净再回 413。

    只回状态码就断开的话，客户端还在往 socket 里塞剩下几 MB，
    看到的是 Broken pipe / 网络错误，用户根本不知道是体积超限。
    """
    base, _store = server
    original = ReviewHandler.max_upload_mb
    ReviewHandler.max_upload_mb = 1
    try:
        payload = jpeg_bytes(7) * 40  # 约 2.3 MB，明确超过下面设的 1 MB 上限
        assert len(payload) > 2 * 1024 * 1024  # 明确超过 1 MB 上限
        status, _headers, body = upload(base, "big.mp4", payload)
        assert status == 413, body
        assert "上限" in json.loads(body)["error"]
    finally:
        ReviewHandler.max_upload_mb = original

    uploads = os.path.join(dataset, ".autolabel", "uploads")
    assert os.listdir(uploads) == []  # 半截文件已清掉
    status, _headers, body = call(base, "/api/dataset")  # 连接状态没被打乱
    assert status == 200, body
    assert len(json.loads(body)["items"]) == 2


def test_multi_megabyte_upload_is_written_whole(server, dataset):
    """流式落盘后字节数必须和上传内容完全一致（边读边写最容易掉尾）。"""
    base, _store = server
    payload = jpeg_bytes(11) * 40  # 跨过分块边界，验证尾块没有掉
    status, _headers, body = upload(base, "clip.mp4", payload)
    assert status == 200, body
    info = json.loads(body)
    assert info["bytes"] == len(payload)
    stored = os.path.join(dataset, ".autolabel", "uploads", info["stored"])
    assert os.path.getsize(stored) == len(payload)


def test_oversized_unsupported_type_still_drains_body(server):
    """类型不支持时也要读完请求体，否则下一个请求会读到上一个的尾巴。"""
    base, _store = server
    status, _headers, body = upload(base, "payload.exe", b"MZ" * (2 << 20))
    assert status == 400, body
    assert "不支持" in json.loads(body)["error"]
    status, _headers, body = call(base, "/api/dataset")
    assert status == 200, body


def test_submit_missing_upload_file_returns_404(server):
    base, _store = server
    status, _headers, body = call(base, "/api/jobs", data=json.dumps(
        {"kind": "image", "files": ["ghost.jpg"]}).encode(),
        headers={"Content-Type": "application/json"})
    assert status == 404
    assert "ghost.jpg" in json.loads(body)["error"]


def test_upload_name_and_kind_sanitising():
    assert safe_upload_name("../../x/秘密 a.jpg") == "秘密_a.jpg"
    assert safe_upload_name("///") == "file"
    assert kind_for(".MP4") == "video" and kind_for(".png") == "image" and kind_for(".exe") == ""


def post_json(base, path, payload=None):
    return call(base, path, data=json.dumps(payload or {}).encode(),
                headers={"Content-Type": "application/json"})


def wait_until(cond, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_clear_jobs_cancels_running_and_drops_queued(server, dataset, monkeypatch):
    """一个任务卡在抽帧、一个在排队，清除后都要停下，临时文件都要删掉。"""
    base, store = server
    release = threading.Event()
    original = store.jobs._ingest

    def slow_ingest(job):
        if job.params.get("hold"):
            assert release.wait(30), "测试没有放行卡住的任务"
        return original(job)

    monkeypatch.setattr(store.jobs, "_ingest", slow_ingest)

    held = json.loads(upload(base, "held.jpg", jpeg_bytes(11))[2])["stored"]
    queued = json.loads(upload(base, "queued.jpg", jpeg_bytes(12))[2])["stored"]
    first = json.loads(post_json(base, "/api/jobs",
                                 {"kind": "image", "files": [held],
                                  "params": {"label": False, "hold": True}})[2])["id"]
    assert wait_until(lambda: any(j["id"] == first and j["stage"] == "importing"
                                  for j in json.loads(call(base, "/api/jobs")[2])["jobs"]))
    second = json.loads(post_json(base, "/api/jobs",
                                  {"kind": "image", "files": [queued],
                                   "params": {"label": False}})[2])["id"]
    assert wait_until(lambda: len(json.loads(call(base, "/api/jobs")[2])["jobs"]) == 2)

    status, _headers, body = post_json(base, "/api/jobs/clear")
    assert status == 200, body
    assert json.loads(body) == {"ok": True, "cancelled": 1, "cancelling": 1, "dropped": 0}

    jobs = json.loads(call(base, "/api/jobs")[2])["jobs"]
    running = next(j for j in jobs if j["id"] == first)
    assert running["cancel_requested"] is True and running["finished"] is False
    assert not os.path.exists(os.path.join(dataset, ".autolabel", "uploads", queued))  # 排队任务的文件已删

    release.set()
    assert wait_until(lambda: all(j["id"] not in (first, second)
                                  for j in json.loads(call(base, "/api/jobs")[2])["jobs"])), "被清除的任务没有退出任务列表"
    assert not os.listdir(os.path.join(dataset, ".autolabel", "uploads"))

    records = load_batches(dataset)  # 被取消的任务登记为 cancelled，排队未开始的不产生批次
    assert [record["status"] for record in records] == ["cancelled"]


def test_clear_jobs_with_nothing_running_is_a_noop(server):
    base, _store = server
    status, _headers, body = post_json(base, "/api/jobs/clear")
    assert status == 200
    assert json.loads(body) == {"ok": True, "cancelled": 0, "cancelling": 0, "dropped": 0}


def test_label_dataset_stops_on_request(dataset):
    from autolabel.backends import BackendSettings
    from autolabel.pipeline import label_dataset

    stats = label_dataset(dataset, BackendSettings(device="cpu"), dry_run=True, verbose=False,
                          should_stop=lambda: True)
    assert stats["labeled"] == 0 and stats["total"] > 0 and stats["cancelled"] == stats["total"]




# --- 第 31 轮体检：界面能传的必须就是命令行能导的 ------------------------ #

def test_the_browser_accepts_exactly_what_the_cli_can_import():
    """同一个后缀不该"命令行导得进、界面里被拒"。

    差的就是 ``.tif``/``.tiff``：命令行导入一路正常，浏览器里传同一张图只得到
    一句"不支持的文件类型"。收进来之后统一转码成 JPEG，界面看的从来不是原图。
    """
    from autolabel import upload as up
    from autolabel.dataset import IMAGE_EXTS as importable

    assert set(up.IMAGE_EXTS) == set(importable)
    assert kind_for(".tif") == "image" and kind_for(".TIFF") == "image"


def test_a_tiff_upload_lands_as_a_normalized_jpeg(dataset):
    """TIFF 进素材池要变成 JPEG：``<img>`` 渲染不了 TIFF，索引尺寸也得是转码后的。"""
    from autolabel import upload as up
    from autolabel.dataset import import_images, load_index

    buf = io.BytesIO()
    Image.new("RGB", (300, 120), (200, 60, 60)).save(buf, "TIFF")
    saved = up.save_upload(dataset, "扫描件.tif", buf.getvalue())

    assert saved["kind"] == "image"
    stats = import_images(dataset, [os.path.join(up.uploads_dir(dataset), saved["stored"])])
    assert stats["imported"] == 1 and stats["failed"] == 0
    rel = stats["imported_files"][0][1]
    assert rel.endswith(".jpg")
    item = load_index(dataset).items[rel]
    assert (item["width"], item["height"]) == (300, 120)


def test_an_unsupported_extension_is_still_refused(dataset):
    from autolabel import upload as up

    with pytest.raises(ValueError, match="不支持的文件类型"):
        up.save_upload(dataset, "报告.pdf", b"%PDF-1.4 fake")


def test_pose_run_route_takes_a_picked_subset(server, dataset, monkeypatch):
    """「批量姿态标注」：库里已有的图直接跑权重，圈了哪几张就只标哪几张。

    这条补的是姿态页一直缺的口子——以前只有「导入素材」会触发自动标注，
    勾了「只入库不标注」传进来的图想补标，只能把素材整批重传。
    """
    from test_prompt_web import post
    from autolabel.upload import Job

    base, store = server
    images = [row["image"] for row in store.dataset_payload()["items"]]
    assert len(images) >= 2
    submitted = []

    def fake_submit(kind, files, params=None):
        submitted.append((kind, list(files), dict(params or {})))
        return Job(id="posejob", kind=kind, files=list(files), params=dict(params or {}))

    monkeypatch.setattr(store.jobs, "submit", fake_submit)
    status, job, _ = post(base, "/api/pose/run",
                          {"model": "yolo11n-pose.pt", "conf": 0.3, "overwrite": True,
                           "images": images[:1]})
    assert status == 202, job
    assert job["kind"] == "pose" and job["file_count"] == 1
    assert submitted[0][0] == "pose"
    assert submitted[0][1] == images[:1]
    assert submitted[0][2]["only"] == images[:1]
    assert submitted[0][2]["conf"] == 0.3 and submitted[0][2]["overwrite"] is True


def test_pose_run_forwards_the_multimodal_profile(server, dataset, monkeypatch):
    """选多模态标骨骼时，vlm_profile 必须单独进任务参数。

    少了它任务会去加载一个叫 vlm:xxx 的 YOLO 权重然后报缺文件；model 里那份
    原样保留是有意的：和「导入素材」那条同一份参数，断点续跑用的指纹才对得上。
    """
    from test_prompt_web import post

    base, store = server
    profile = {"id": "p-1", "name": "本机多模态", "kind": "local", "model": "qwen-vl",
               "base_url": "http://127.0.0.1:9/v1", "api_key": "", "enabled": True}
    monkeypatch.setattr("autolabel.models.find_profile", lambda wanted: profile)
    started = []
    monkeypatch.setattr(store.jobs, "_run", lambda job: started.append(job))
    status, ack, _ = post(base, "/api/pose/run", {"model": "vlm:p-1", "vlm_profile": "p-1"})
    assert status == 202, ack
    assert ack["vlm_profile"] == "p-1" and ack["vlm_name"] == "本机多模态"
    assert started and started[0].params["vlm_profile"] == "p-1"
    assert started[0].params["model"] == "vlm:p-1"


def test_pose_run_refuses_images_that_are_not_in_the_lib(server, dataset):
    """界面传上来的一张图早被删了：当场说清楚，别开一个跑完什么也没发生的任务。"""
    from test_prompt_web import post

    base, _store = server
    status, body, _ = post(base, "/api/pose/run", {"model": "yolo11n-pose.pt",
                                                   "images": ["images/没了.jpg"]})
    assert status == 400, body
    assert "没有收到文件" in body["error"]


def test_pose_job_stages_and_batch_read_like_the_other_two_panels(dataset, monkeypatch):
    """任务的阶段、批次名、产出口径要和提示词 / 目标检测两页的批量面板对得上。"""
    from autolabel.dataset import load_index
    from autolabel.upload import Job, JobStore

    store = JobStore(dataset)
    images = [rel for rel, _ in load_index(dataset).ordered_items()]
    seen = {}

    def fake_label(job):
        seen["only"] = list(job.params.get("only") or [])
        return {"labeled": len(seen["only"]), "people": len(seen["only"]), "flagged": 0,
                "failed": 0, "skipped_human": 0}

    monkeypatch.setattr(store, "_label", fake_label)
    monkeypatch.setattr(store, "_qc", lambda job: "reports/qc.md")
    monkeypatch.setattr(store, "_coords", lambda job: "exports/coords")
    picked = images[:1]
    job = Job(id="pose1", kind="pose", files=picked,
              params={"only": picked, "model": "yolo11n-pose.pt"})
    store._start_batch(job)
    store._run_pose(job)
    assert job.finished and job.ok, job.error
    assert seen["only"] == picked, "圈定范围没传到流水线"
    assert job.result["labeled"] == 1 and job.result["coords"] == "exports/coords"
    mine = [row for row in load_batches(dataset) if row["id"] == job.batch_id]
    assert len(mine) == 1
    assert mine[0]["kind"] == "pose" and mine[0]["status"] == "done"
    assert "批量姿态标注" in mine[0]["name"], mine[0]["name"]
    assert set(mine[0]["items"]) == set(picked), "批次挂的图要和圈定的那批一致"
