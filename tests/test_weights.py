"""权重就绪判定与镜像下载兜底。"""

import io
import os
import zipfile

import pytest

from autolabel.backends import ensure_weights, model_status, weights_ready


def zip_bytes(payload: bytes = b"weights") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.txt", payload)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload: bytes, status: int = 200):
        self.payload = payload
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_weights_ready_detects_truncated_download(tmp_path):
    good = tmp_path / "good.pt"
    good.write_bytes(zip_bytes())
    broken = tmp_path / "broken.pt"
    broken.write_bytes(zip_bytes()[:40])

    assert weights_ready(str(good)) is True
    assert weights_ready(str(broken)) is False
    assert weights_ready(str(tmp_path / "missing.pt")) is False
    assert model_status("broken.pt", weights_dir=str(tmp_path))["ready"] is False


def test_ensure_weights_skips_network_when_ready(tmp_path, monkeypatch):
    target = tmp_path / "fake-pose.pt"
    target.write_bytes(zip_bytes())

    def boom(*_args, **_kwargs):
        raise AssertionError("已就绪不该再走网络")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert ensure_weights("fake-pose.pt", weights_dir=str(tmp_path)) == str(target)


def test_ensure_weights_downloads_through_mirror(tmp_path, monkeypatch):
    payload = zip_bytes(b"yolo11s")
    seen = {}

    def fake(request, timeout=None):
        seen["url"] = request.full_url
        seen["range"] = request.get_header("Range")
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake)

    path = ensure_weights("fake-pose.pt", weights_dir=str(tmp_path),
                          mirror="https://mirror.test/", verbose=False)

    assert path == os.path.join(str(tmp_path), "fake-pose.pt")
    assert weights_ready(path)
    assert seen["url"] == "https://mirror.test/" + (
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/fake-pose.pt")
    assert seen["range"] is None


def test_ensure_weights_resumes_partial_file(tmp_path, monkeypatch):
    whole = zip_bytes(b"0123456789" * 64)
    part = tmp_path / "fake-pose.pt.part"
    part.write_bytes(whole[:100])

    def fake(request, timeout=None):
        start = int(request.get_header("Range").split("=")[1].split("-")[0])
        assert start == 100
        return FakeResponse(whole[start:], status=206)

    monkeypatch.setattr("urllib.request.urlopen", fake)

    path = ensure_weights("fake-pose.pt", weights_dir=str(tmp_path),
                          mirror="https://mirror.test/", verbose=False)

    assert open(path, "rb").read() == whole
    assert not part.exists()


def test_ensure_weights_falls_back_when_mirror_down(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    path = ensure_weights("fake-pose.pt", weights_dir=str(tmp_path),
                          mirror="https://mirror.test/", attempts=2, verbose=False)

    # 不抛错，退回原路径交给 ultralytics 自己的下载器
    assert path == os.path.join(str(tmp_path), "fake-pose.pt")
    assert not os.path.exists(path)


def test_ensure_weights_rejects_partial_zip_result(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake(request, timeout=None):
        calls["n"] += 1
        return FakeResponse(b"not-a-zip")

    monkeypatch.setattr("urllib.request.urlopen", fake)

    path = ensure_weights("fake-pose.pt", weights_dir=str(tmp_path),
                          mirror="https://mirror.test/", attempts=1, verbose=False)

    assert path == os.path.join(str(tmp_path), "fake-pose.pt")
    assert calls["n"] == 1
    assert not os.path.exists(path)


@pytest.mark.parametrize("model", ["models/local.pt", "/abs/path.pt"])
def test_ensure_weights_leaves_explicit_paths_alone(tmp_path, monkeypatch, model):
    def boom(*_args, **_kwargs):
        raise AssertionError("带路径的权重不该由这里代下载")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert ensure_weights(model, weights_dir=str(tmp_path)) == model


# --- 第 8 轮体检（模型与算力）实测到的三件事 ---

def test_mirror_that_ignores_range_still_completes(tmp_path, monkeypatch):
    """镜像吃掉 Range 头、直接回 200 整包时，以前永远判"下载不完整"。

    总长度只在真的收到 206 分段时才该把已下过的字节加进去；否则总量凭空多出一截
    永远等不满，重试几次结果一模一样，用户只看到"下载失败"。
    """
    whole = zip_bytes(b"0123456789" * 64)
    part = tmp_path / "fake-pose.pt.part"
    part.write_bytes(whole[:100])           # 上次没下完的半截

    def fake(request, timeout=None):
        # 服务器不看 Range，一律 200 + 整包
        assert request.get_header("Range") is not None
        return FakeResponse(whole, status=200)

    monkeypatch.setattr("urllib.request.urlopen", fake)
    path = ensure_weights("fake-pose.pt", weights_dir=str(tmp_path),
                          mirror="https://mirror.test/", attempts=1, verbose=False)

    assert weights_ready(path)
    assert open(path, "rb").read() == whole


def test_ensure_weights_defers_to_a_running_download(tmp_path, monkeypatch):
    """同一个权重已经有别的进程/任务在下，就等它下完，别再插一脚。

    两个实例共用一个 ``models/`` 是这本项目的正常用法。以前两边写同一份
    ``x.pt.part``：后来者把对方正在追加的文件截断，换名失败的一方还报
    "已下载部分保留"，其实那份文件早就不在了。
    """
    from autolabel.backends import claim_download

    target = tmp_path / "fake-pose.pt"
    mine, handle = claim_download(str(target))
    assert mine is True

    def finish_later():
        import time as _time

        _time.sleep(0.2)
        target.write_bytes(zip_bytes(b"yolo11m"))
        handle.close()

    import threading

    threading.Thread(target=finish_later, daemon=True).start()

    def boom(*_args, **_kwargs):
        raise AssertionError("别人正在下，这里不该再走网络")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    path = ensure_weights("fake-pose.pt", weights_dir=str(tmp_path),
                          mirror="https://mirror.test/", verbose=False)

    assert path == str(target)
    assert weights_ready(path)


def test_claim_download_lets_only_one_holder_through(tmp_path):
    """锁被占着又等不到结果时，老实承认"没抢到"，而不是硬下第二遍。"""
    from autolabel.backends import claim_download

    target = str(tmp_path / "fake-pose.pt")
    mine, holder = claim_download(target)
    assert mine is True
    try:
        second, _ = claim_download(target, wait=False)
        assert second is False
        # 等一会儿也没人把权重下好：同样返回 False，不阻塞调用方
        started = __import__("time").time()
        third, _ = claim_download(target, deadline=started + 0.3)
        assert third is False
        assert __import__("time").time() - started < 5.0
        # 文件落到位了，等待方就该高高兴兴地拿走结果
        with open(target, "wb") as fh:
            fh.write(zip_bytes(b"done"))
        assert claim_download(target, deadline=__import__("time").time() + 3.0)[0] is False
    finally:
        holder.close()
    assert claim_download(target)[0] is True


def test_inference_honours_the_weights_dir_env(monkeypatch, tmp_path):
    """``AUTOLABEL_WEIGHTS_DIR`` 是对外承诺过的：界面和推理必须看同一个目录。

    以前只有 models.py 认它，推理侧把 ``models/`` 写死——改了变量的用户在界面里
    看到权重"已就绪"，点开始标注却又往 ``./models`` 重新下一份。
    """
    from autolabel import models as model_manager
    from autolabel.backends import default_weights_dir, resolve_model_path

    custom = tmp_path / "custom-weights"
    monkeypatch.setenv("AUTOLABEL_WEIGHTS_DIR", str(custom))

    assert default_weights_dir() == str(custom)
    assert resolve_model_path("yolo11n-pose.pt") == os.path.join(str(custom), "yolo11n-pose.pt")
    assert os.path.isdir(str(custom)), "目录要在解析路径时就建好，ultralytics 才不会写错地方"
    assert model_status("yolo11n-pose.pt")["path"] == os.path.join(str(custom), "yolo11n-pose.pt")
    assert model_manager.weights_dir() == default_weights_dir()
