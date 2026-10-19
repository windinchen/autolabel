"""第 14–15 轮体检（接口对畸形请求的姿态）的回归测试。

盯两件事：

1. 请求体是**合法 JSON 但不是对象**（``[1,2]`` / ``"hi"`` / ``null``）时，以前
   ``payload.get(...)`` 抛 ``AttributeError``，而那一行在所有 try 之外 —— 连接被
   直接关掉、一个字节响应都没写，界面上只剩「Failed to fetch」，看着像服务挂了；
2. ``Content-Length`` 写坏（``abc`` / 负数）在 ``int()`` 上抛同一个症状。
"""
import json
import socket
import urllib.error
import urllib.request


def post_raw(base, path, body: bytes, length_header=None, ctype="application/json"):
    """手写请求：要看到「连接被掐断、没有任何响应」这种情况，urllib 会把它
    包装成 ``RemoteDisconnected``，容易和真的网络问题混在一起。"""
    host, port = base.split("//", 1)[1].split(":")
    port = int(port)
    size = len(body) if length_header is None else length_header
    head = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {size}\r\nConnection: close\r\n\r\n")
    with socket.create_connection((host, port), timeout=6) as sock:
        sock.sendall(head.encode() + body)
        chunks = []
        while True:
            piece = sock.recv(4096)
            if not piece:
                break
            chunks.append(piece)
    return b"".join(chunks).decode("utf-8", "replace")


def split_response(raw):
    head, _, body = raw.partition("\r\n\r\n")
    return head.splitlines()[0] if head else "", body


def test_non_object_json_body_gets_400_not_a_dead_connection(server, dataset):
    """数组 / 字符串 / null / 数字都得回 400 带中文，而不是把连接关掉。"""
    for body, kind in (b"[1,2]", "数组"), (b'"hi"', "字符串"), (b"null", "null"), (b"42", "数字"):
        raw = post_raw(server[0], "/api/status", body)
        status, payload = split_response(raw)
        assert "400" in status, (body, raw[:120])
        assert kind in json.loads(payload)["error"], payload


def test_broken_json_reports_position(server):
    raw = post_raw(server[0], "/api/status", b'{"image": }')
    status, payload = split_response(raw)
    message = json.loads(payload)["error"]
    assert "400" in status and "JSON 解析失败" in message and "第 1 行" in message


def test_bad_content_length_header_does_not_kill_the_connection(server):
    """以前 ``int("abc")`` 抛在所有 try 之外，客户端拿到的是断链。"""
    for value in ("abc", "-5", ""):
        raw = post_raw(server[0], "/api/status", b"{}", length_header=value)
        status, _payload = split_response(raw)
        assert raw, f"Content-Length={value!r} 时没有任何响应"
        assert "400" in status, (value, raw[:120])


def test_missing_status_field_is_explained(server):
    """``str(None)`` 当成状态传给校验，用户会看到「现在是「None」」这种废话。"""
    raw = post_raw(server[0], "/api/status", b"{}")
    _status, payload = split_response(raw)
    message = json.loads(payload)["error"]
    assert "没有指定复核状态" in message and "None" not in message


def test_unknown_status_lists_options(server, dataset):
    from autolabel.dataset import load_index

    rel = sorted(load_index(dataset).items)[0]
    body = json.dumps({"image": rel, "status": "aprovied"}).encode()
    _status, payload = split_response(post_raw(server[0], "/api/status", body))
    message = json.loads(payload)["error"]
    assert "复核状态只能是" in message and "unreviewed" in message and "aprovied" in message


def test_empty_upload_still_answers_in_chinese(server):
    raw = post_raw(server[0], "/api/upload?name=a.jpg", b"")
    status, payload = split_response(raw)
    assert "400" in status
    assert "缺少文件内容" in json.loads(payload)["error"]


def test_valid_request_still_works(server, dataset):
    """健壮性改动不能把正常请求一起挡掉。"""
    from autolabel.dataset import load_index

    rel = sorted(load_index(dataset).items)[0]
    body = json.dumps({"image": rel, "status": "approved"}).encode()
    status, payload = split_response(post_raw(server[0], "/api/status", body))
    assert "200" in status, payload
    assert json.loads(payload)["status"] == "approved"
    assert load_index(dataset).items[rel]["status"] == "approved"


def _get_error(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=6) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_unknown_endpoint_names_the_path(server):
    """GET / POST 打到一个不存在的接口都要报出**是哪个地址**，别只说 not found。"""
    code, payload = _get_error(server[0], "/api/nope")
    assert code == 404 and "/api/nope" in payload["error"]

    raw = post_raw(server[0], "/api/nope", b"{}")
    status, body = split_response(raw)
    assert "404" in status and "/api/nope" in json.loads(body)["error"]


def test_traversal_in_image_query_stays_inside_the_dataset(server):
    """``?image=../../etc/passwd`` 不能读出数据集以外的文件。"""
    for path in ("/api/item?image=../../etc/passwd", "/image/../../../../etc/passwd"):
        code, payload = _get_error(server[0], path)
        assert code == 404, (path, payload)
        assert "root:" not in json.dumps(payload), f"把 /etc/passwd 的内容吐出去了：{payload}"
