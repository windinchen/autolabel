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
