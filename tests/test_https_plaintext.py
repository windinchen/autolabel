# -*- coding: utf-8 -*-
"""同一个端口既要会说 HTTPS，也要听懂有人敲了 ``http://``（review.py 的 WorkbenchServer）。

两件事都是实测踩出来的：

1. 明文 ``http://`` 打到只听 HTTPS 的端口，以前握手在 ``accept()`` 里抛，而标准库把
   ``get_request()`` 的 ``OSError`` 整个吞掉（连 ``handle_error`` 都不走）：浏览器只剩
   一句「响应无效」，服务端日志一行不剩——「网页打不开」根本无从查起；
2. ``--host ::1`` / ``--host ::`` 直接在 ``bind()`` 上抛 ``gaierror``（父类地址族写死
   IPv4），可证书 SAN 早签了 ``::1``、启动提示也写着 ``::``，宣称支持却起不来。
"""
import json
import os
import socket
import ssl
import threading
import time

import pytest

from autolabel import ca
from autolabel.review import ReviewHandler, ReviewStore, WorkbenchServer

needs_openssl = pytest.mark.skipif(not ca.openssl_available(), reason="这台机器没有 openssl")


@pytest.fixture(scope="module")
def cert_pair(tmp_path_factory):
    """签一次证书整份文件共用：openssl 签发要一秒上下，每个用例都签太磨人。"""
    root = tmp_path_factory.mktemp("https-certs")
    previous = os.environ.get("AUTOLABEL_CERTS")
    os.environ["AUTOLABEL_CERTS"] = str(root)
    try:
        ca.issue(force=True)
        yield ca.leaf_cert_path(), ca.leaf_key_path()
    finally:
        if previous is None:
            os.environ.pop("AUTOLABEL_CERTS", None)
        else:
            os.environ["AUTOLABEL_CERTS"] = previous


@pytest.fixture
def workbench(dataset, cert_pair):
    """起真服务并交出一个 ``start(host=…, ssl_context=…)`` 工厂，用例自己决定绑哪。"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_pair[0], keyfile=cert_pair[1])
    previous = getattr(ReviewHandler, "store", None)
    ReviewHandler.store = ReviewStore(dataset)
    opened = []

    def start(host="127.0.0.1", ssl_context=context):
        httpd = WorkbenchServer(host, 0, ReviewHandler, ssl_context=ssl_context)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        opened.append(httpd)
        return httpd.server_address[1], httpd

    yield start
    for httpd in opened:
        httpd.shutdown()
        httpd.server_close()
    ReviewHandler.store = previous


def talk(port, payload, tls=False, family=socket.AF_INET):
    """手写请求发出去，收完响应头就回。

    走裸 socket 是为了看见「连接被掐断、一个字节都没回」这种 urllib 会包装成
    ``RemoteDisconnected`` 的情况——那正是以前敲 http:// 的真实症状。
    """
    where = ("::1", port, 0, 0) if family == socket.AF_INET6 else ("127.0.0.1", port)
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(8)
    sock.connect(where)
    if tls:
        client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client.check_hostname = False
        client.verify_mode = ssl.CERT_NONE
        sock = client.wrap_socket(sock, server_hostname="localhost")
    sock.sendall(payload)
    # 每个请求都带 ``Connection: close``，所以读到 EOF 为止：只读到响应头就返回的话，
    # 响应体可能还在下一个包里，全量跑的时候就会随机拿到空 body（本地单跑却过）
    chunks = []
    while True:
        piece = sock.recv(4096)
        if not piece:
            break
        chunks.append(piece)
    sock.close()
    return b"".join(chunks).decode("latin-1")


def get_line(port, path="/", **kw):
    host = "[::1]" if kw.get("family") == socket.AF_INET6 else "127.0.0.1"
    return talk(port, ("GET %s HTTP/1.1\r\nHost: %s:%d\r\nConnection: close\r\n\r\n"
                       % (path, host, port)).encode(), **kw)


@needs_openssl
def test_plain_http_get_becomes_a_301_to_the_same_port(workbench):
    """少敲一个 s：301 把人送到 https，端口不变，浏览器自己就跳过去了。"""
    port, _httpd = workbench()
    raw = get_line(port, "/detect")
    head = raw.partition("\r\n\r\n")[0].splitlines()
    assert "301" in head[0], raw[:200]
    location = [line for line in head if line.lower().startswith("location")]
    assert location and location[0].split(": ", 1)[1] == f"https://127.0.0.1:{port}/detect", head


@needs_openssl
def test_plain_http_post_is_told_to_use_https_in_chinese(workbench):
    """POST 不能靠 301（有些客户端会退成 GET 把请求体丢了），直接一句中文说清楚。"""
    port, _httpd = workbench()
    raw = talk(port, ("POST /api/status HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Length: 2\r\n"
                      "Connection: close\r\n\r\n{}" % port).encode())
    status, _, body = raw.partition("\r\n\r\n")
    assert "400" in status.splitlines()[0], raw[:200]
    payload = json.loads(body)
    assert "HTTPS" in payload["error"], payload
    assert payload["https"] == f"https://127.0.0.1:{port}/api/status", payload


@needs_openssl
def test_real_https_still_works_on_the_same_port(workbench):
    """加嗅探不能把正经 TLS 一起挡掉——这条是整个改动的底线。"""
    port, _httpd = workbench()
    raw = get_line(port, "/", tls=True)
    assert "200 OK" in raw.splitlines()[0], raw[:200]


@needs_openssl
def test_redirect_keeps_the_https_service_working(workbench):
    """明文那条连接判完之后，同一端口还得继续正常服务。"""
    port, _httpd = workbench()
    assert "301" in get_line(port).splitlines()[0]
    assert "200 OK" in get_line(port, "/pose", tls=True).splitlines()[0]
    assert "200 OK" in get_line(port, "/", tls=True).splitlines()[0]


@needs_openssl
def test_a_plaintext_server_does_not_redirect_to_https(workbench):
    """证书没加载成功时服务本来就是 HTTP，这时候把人往 https 上赶就成死循环了。"""
    port, _httpd = workbench(ssl_context=None)
    assert "200 OK" in get_line(port).splitlines()[0]


@needs_openssl
def test_broken_handshake_says_so_and_keeps_serving(workbench, capsys):
    """握手失败以前只吐一坨 Python 栈、或者直接静默；现在要留下一句人话并且不崩。"""
    port, _httpd = workbench()
    with socket.create_connection(("127.0.0.1", port), timeout=6) as sock:
        sock.sendall(b"\x16\x03\x01\x00\x02\x01\x00")   # 半截 ClientHello
    seen, deadline = "", time.time() + 6
    while time.time() < deadline and "握手没成功" not in seen:
        seen += capsys.readouterr().out
        time.sleep(0.1)
    assert "握手没成功" in seen, seen
    assert "https://" in seen, seen
    assert "200 OK" in get_line(port, "/", tls=True).splitlines()[0]


@needs_openssl
def test_silent_clients_cannot_wedge_the_accept_loop(workbench):
    """浏览器会预热连接：连上来一个字都不说。嗅探必须带超时，否则整个服务卡死。"""
    port, _httpd = workbench()
    for _ in range(3):
        socket.create_connection(("127.0.0.1", port), timeout=6).close()
    assert "200 OK" in get_line(port, "/", tls=True).splitlines()[0]


@needs_openssl
@pytest.mark.parametrize("host", ["::1", "::"])
def test_ipv6_listen_addresses_actually_bind(workbench, host):
    """以前这两种写法在 ``bind()`` 就抛 gaierror，而提示里写着它们能用。"""
    try:
        port, httpd = workbench(host)
    except OSError as exc:                       # 真没有 IPv6 的机器不算这个项目的问题
        pytest.skip(f"这台机器没有 IPv6：{exc}")
    assert httpd.address_family == socket.AF_INET6
    assert port > 0
    raw = get_line(port, "/", tls=True, family=socket.AF_INET6)
    assert "200 OK" in raw.splitlines()[0], raw[:200]


@needs_openssl
def test_wildcard_ipv6_also_answers_ipv4(workbench):
    """绑 ``::`` 时显式关掉 V6ONLY：``localhost`` 不管解析到 127.0.0.1 还是 ::1 都得通。

    系统默认值不统一（Linux 默认双栈、部分 macOS 默认只 IPv6），不写死就会出现
    「这台机能打开、那台机连不上」。
    """
    try:
        port, httpd = workbench("::")
    except OSError as exc:
        pytest.skip(f"这台机器没有 IPv6：{exc}")
    assert httpd.dual_stack is True
    for family in (socket.AF_INET, socket.AF_INET6):
        assert "200 OK" in get_line(port, "/", tls=True, family=family).splitlines()[0], family
