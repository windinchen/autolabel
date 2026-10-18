# -*- coding: utf-8 -*-
"""本地 HTTPS 证书：摄像头只在"安全网页"里开放，这块坏了整个录制功能就没了。"""
import os
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from autolabel import ca
from autolabel.review import local_cert_paths

needs_openssl = pytest.mark.skipif(not ca.openssl_available(), reason="这台机器没有 openssl")


@pytest.fixture
def certdir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_CERTS", str(tmp_path / "certs"))
    return tmp_path / "certs"


@needs_openssl
def test_issue_creates_a_real_ca_pair(certdir):
    ca.issue()
    for path in (ca.ca_cert_path(), ca.ca_key_path(), ca.leaf_cert_path(), ca.leaf_key_path()):
        assert os.path.exists(path), path
    # 真让 openssl 验一次链，比读字段可信
    done = subprocess.run(["openssl", "verify", "-CAfile", ca.ca_cert_path(), ca.leaf_cert_path()],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@needs_openssl
def test_leaf_covers_localhost_and_lan_ip(certdir):
    ca.issue()
    sans = set(ca.sans(ca.leaf_cert_path()))
    assert {"localhost", "127.0.0.1"} <= sans, sans      # 没有 localhost 浏览器直接判地址不合法
    assert "autolabel.local" in sans


@needs_openssl
def test_ca_is_a_ca_and_leaf_is_not(certdir):
    ca.issue()
    ca_text = subprocess.run(["openssl", "x509", "-in", ca.ca_cert_path(), "-noout", "-text"],
                             capture_output=True, text=True).stdout
    leaf_text = subprocess.run(["openssl", "x509", "-in", ca.leaf_cert_path(), "-noout", "-text"],
                               capture_output=True, text=True).stdout
    assert "CA:TRUE" in ca_text.replace(" ", "")
    assert "CA:FALSE" in leaf_text.replace(" ", "")
    assert "TLS Web Server Authentication" in leaf_text   # 不能当别的用


@needs_openssl
def test_issue_is_idempotent_unless_forced(certdir):
    assert ca.issue()["created"] is True
    before = open(ca.leaf_cert_path(), "rb").read()
    assert ca.issue()["created"] is False
    assert open(ca.leaf_cert_path(), "rb").read() == before
    ca.issue(force=True)
    assert open(ca.leaf_cert_path(), "rb").read() != before


@needs_openssl
def test_key_files_are_owner_only(certdir):
    ca.issue()
    for path in (ca.ca_key_path(), ca.leaf_key_path()):
        assert (os.stat(path).st_mode & 0o077) == 0, path


@needs_openssl
def test_days_left_is_sane(certdir):
    ca.issue()
    left = ca.days_left(ca.leaf_cert_path())
    assert left is not None and ca.LEAF_DAYS - 3 <= left <= ca.LEAF_DAYS, left


def test_status_reports_nothing_when_no_certs(certdir):
    info = ca.status()
    assert info["have_cert"] is False and info["trusted"] is False
    assert info["days_left"] is None


@needs_openssl
def test_status_reports_the_cert_after_issue(certdir):
    ca.issue()
    info = ca.status()
    assert info["have_cert"] is True and info["fingerprint"]
    assert info["expires_soon"] is False


def test_uninstall_without_keychain_is_not_an_error(certdir, monkeypatch):
    monkeypatch.setattr(ca, "keychain_fingerprints", lambda: ())
    done = ca.uninstall_trust()
    assert done["ok"] is True and "无需卸载" in done["note"]


def test_install_without_cert_asks_to_issue_first(certdir):
    done = ca.install_trust()
    assert done["ok"] is False and "autolabel ca" in done["note"]


def test_local_cert_paths_follow_the_files(certdir):
    assert local_cert_paths() == (None, None)
    certdir.mkdir(parents=True, exist_ok=True)
    for path in (ca.leaf_cert_path(), ca.leaf_key_path()):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
    assert local_cert_paths() == (ca.leaf_cert_path(), ca.leaf_key_path())


class _Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


@needs_openssl
def test_server_actually_speaks_https_with_those_files(certdir):
    """光有文件不算数：真握手一次，验的是客户端认不认这张证书。"""
    ca.issue()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=ca.leaf_cert_path(), keyfile=ca.leaf_key_path())
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Ping)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    port = httpd.socket.getsockname()[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = ssl.create_default_context(cafile=ca.ca_cert_path())
        with client.wrap_socket(socket.create_connection(("localhost", port)),
                              server_hostname="localhost") as sock:
            assert sock.version() == "TLSv1.2" or sock.version() == "TLSv1.3", sock.version()
            sock.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            data = sock.recv(64)
        assert b"200" in data
    finally:
        httpd.shutdown()
        httpd.server_close()
