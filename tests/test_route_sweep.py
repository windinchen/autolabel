"""全路由终检：任何一个页面/接口都不许回 5xx，兜底错误也得是中文。

界面有 5 个页面、30 多个接口，个别接口只在特定库有数据时才走到（检测页、
提示词页各一份类别表）。这里把它们一网打尽跑一遍，专门盯"服务内部出错"——
那种错误在界面上只剩一串英文，用户只能截图来问。
"""
import json
import urllib.error
import urllib.request

import pytest
from test_prompt_web import get, post


def raw_get(base, path):
    """下载类接口返回的是 CSV / JSONL，不能按 JSON 解析，这里只拿原始字节。"""
    try:
        with urllib.request.urlopen(base + path, timeout=30) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers

PAGES = ["/", "/index.html", "/portal", "/pose", "/review", "/objects", "/detect", "/models",
         "/app.js", "/common.js", "/portal.js", "/objects.js", "/models.js", "/app.css",
         "/favicon.ico"]

APIS = ["/api/dataset", "/api/fields", "/api/fields?lib=objects", "/api/fields?lib=detect",
        "/api/detect/models", "/api/detect/summary", "/api/jobs", "/api/batches",
        "/api/pool", "/api/pool?lib=detect", "/api/models", "/api/version", "/api/settings",
        "/api/settings?probe=1", "/api/prompt/summary", "/api/prompt/summary?lib=detect",
        "/api/batch/files?id=__ungrouped__:pose", "/api/modelhub/search?q=pose"]

DOWNLOADS = ["/api/coords", "/api/coords?format=jsonl&include_all=1",
             "/api/coords?format=long", "/api/prompt/coords?format=jsonl"]

# 目标检测库在这份夹具里压根没数据：空表是正确答案，但表头与文件名不能少
EMPTY_OK_DOWNLOADS = ["/api/prompt/coords?format=csv&lib=detect",
                      "/api/prompt/coords?format=jsonl&lib=detect"]

BAD_GETS = ["/api/item?image=", "/api/item?image=../../etc/passwd", "/api/prompt/item?image=x.jpg",
            "/api/detect/model/classes?model=", "/image/ghost.jpg", "/thumb/ghost.jpg",
            "/api/coords?format=bogus", "/api/nope/nothing"]


def test_every_page_is_served(server):
    base, _store = server
    for path in PAGES:
        status, body, _headers = get(base, path)
        # favicon 在最小环境里可能确实没有，其余页面缺一个就是发布事故
        assert status < 500, (path, status)
        if path != "/favicon.ico" and status == 404:
            pytest.fail(f"页面缺失：{path}")
        assert isinstance(body, (bytes, str)) and body, path


def test_every_read_api_answers_without_crashing(server, dataset):
    base, _store = server
    for path in APIS:
        status, body, _headers = get(base, path)
        assert status < 500, (path, status, str(body)[:200])
        assert status == 200, (path, status, str(body)[:200])
        assert isinstance(body, dict), path
        assert "服务内部出错" not in json.dumps(body, ensure_ascii=False)


def test_coordinate_downloads_carry_real_payloads(server, dataset):
    """坐标下载：CSV/JSONL 都得带上真正的标注，别只有表头就回 200。"""
    from _helpers import idx_names, write_label
    from autolabel.dataset import load_index, prompt_label_path_for

    row = "0 0.5 0.5 0.4 0.6 " + " ".join(["0.5 0.5 2"] * 17)
    names = sorted(idx_names(dataset))
    for name in names:
        write_label(dataset, name, [row])  # 姿态库
        with open(prompt_label_path_for(dataset, name), "w", encoding="utf-8") as fh:
            fh.write("0 0.5 0.5 0.2 0.3 0.9\n")  # 提示词库
    objects = load_index(dataset, "objects")
    for name in names:
        objects.items[name]["prompt_objects"] = 1
    objects.save()
    base, _store = server
    for path in DOWNLOADS:
        status, body, headers = raw_get(base, path)
        assert status == 200, (path, status, str(body)[:200])
        ctype = headers.get("Content-Type", "")
        assert ctype.startswith(("text/csv", "application/json")), (path, ctype)
        assert "attachment" in headers.get("Content-Disposition", ""), path
        text = body.decode("utf-8", "replace")
        assert any(name in text for name in names), (path, text[:200])
        if "jsonl" in path:
            lines = [line for line in text.splitlines() if line.strip()]
            assert len(lines) == len(names), (path, len(lines), text[:200])
            for line in lines:
                assert isinstance(json.loads(line), dict)


def test_empty_library_download_still_names_its_file(server, dataset):
    """空库下载：回 200 + 表头 + 正常文件名，界面"下载坐标"按钮不该拿到坏文件。"""
    base, _store = server
    for path in EMPTY_OK_DOWNLOADS:
        status, body, headers = raw_get(base, path)
        assert status == 200, (path, status, str(body)[:200])
        disposition = headers.get("Content-Disposition", "")
        want = ".jsonl" if "jsonl" in path else ".csv"
        assert "attachment" in disposition and want in disposition, (path, disposition)


def test_bad_get_parameters_get_chinese_words_not_stack(server, dataset):
    base, _store = server
    for path in BAD_GETS:
        status, body, _headers = get(base, path)
        assert status < 500, (path, status, str(body)[:200])
        assert "Traceback" not in str(body)


def test_write_apis_survive_garbage_and_unknown_names(server, dataset):
    base, _store = server
    cases = [
        ("/api/status", {"image": "ghost.jpg", "status": "approved"}),
        ("/api/status", {"image": "", "status": "bogus"}),
        ("/api/item", {"image": "ghost.jpg", "objects": [], "status": "approved"}),
        ("/api/undo", {"image": "ghost.jpg"}),
        ("/api/images/delete", {"images": ["查无此图.jpg"], "lib": "all"}),
        ("/api/images/delete", {"images": ["a.jpg"], "lib": "detect", "mode": "瞎写"}),
        ("/api/images/delete", {"images": [], "lib": "pose"}),
        ("/api/pool/adopt", {"images": ["ghost.jpg"], "lib": "detect"}),
        ("/api/batches/delete", {"ids": ["b_ghost"]}),
        ("/api/batches/delete", {"ids": []}),
        ("/api/jobs", {"kind": "image", "files": [], "params": {}}),
        ("/api/jobs/clear", {}),
        ("/api/export", {"scope": "瞎写的范围"}),
        ("/api/export/reveal", {"path": ""}),
        ("/api/prompt/export", {"lib": "detect"}),
        ("/api/settings", {"device": "瞎写"}),
        ("/api/models/vlm/delete", {"id": "ghost"}),
        ("/api/models/weight/delete", {"name": "ghost.pt"}),
        ("/api/models/local/forget", {"name": "ghost.pt"}),
    ]
    for path, payload in cases:
        status, body, _headers = post(base, path, payload)
        assert status < 500, (path, payload, status, str(body)[:220])
        assert "Traceback" not in str(body)
        if status >= 400:
            # 出错必须说人话：中文说明，且把出错原因写清楚
            text = json.dumps(body, ensure_ascii=False)
            assert any("\u4e00" <= ch <= "\u9fff" for ch in text), (path, payload, body)


def test_catalog_refresh_reports_in_chinese_when_github_is_unreachable(server, monkeypatch):
    """「更新目录」打的是 GitHub：连不上要回中文原因，更不许把 5xx 甩给界面。"""
    from autolabel import modelhub

    real = urllib.request.urlopen

    def only_github_is_down(request, timeout=0, **kwargs):
        """只让 GitHub 连不上：测试自己发的那个 POST 还得照常出去。"""
        if "api.github.com" in getattr(request, "full_url", str(request)):
            raise urllib.error.URLError("连不上")
        return real(request, timeout=timeout, **kwargs)

    monkeypatch.setattr(modelhub.urllib.request, "urlopen", only_github_is_down)
    status, body, _headers = post(server[0], "/api/modelhub/refresh", {})

    assert status == 200, body
    assert body["ok"] is False and "没连上 GitHub" in body["note"]
    assert get(server[0], "/api/modelhub/search?q=pose")[1]["catalog"]["refreshed"] is False


def test_unknown_paths_are_404_on_both_verbs(server):
    base, _store = server
    assert get(base, "/api/does-not-exist")[0] == 404
    assert post(base, "/api/does-not-exist", {})[0] == 404
