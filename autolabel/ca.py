# -*- coding: utf-8 -*-
"""本地 HTTPS 证书：让摄像头、麦克风这类"只有安全网页才开放"的功能能用。

浏览器只在 HTTPS（或 localhost）下开放摄像头，而且现在的浏览器会自动把
http:// 升级成 https://，所以工作台长期跑在自签 HTTPS 上反而省事。

根证书装在**当前用户**的信任域，因此不需要管理员密码，也不会弹
"某某浏览器想修改受限制的系统根证书"那种框：

- macOS：登录钥匙串（``security add-trusted-cert``）；
- Windows：当前用户的「受信任的根证书颁发机构」（``certutil -addstore -user Root``），
  Chrome / Edge 认的就是这个存储；中途会弹一个确认框，点「是」即可；
- 其它系统：只签发不装，提示手动导入。

    autolabel ca                  # 生成 + 装进钥匙串 + 打印状态
    autolabel ca --status         # 只看状态，不改动
    autolabel ca --force          # 重新签发（换机器名、证书过期时用）
    autolabel ca --uninstall      # 从钥匙串摘掉

证书放在项目里的 certs/（已 gitignore），地址用 localhost 与 autolabel.local 都行，
后者需要在 /etc/hosts 里加一行，不配也能用。
"""
from __future__ import annotations

import datetime
import hashlib
import os
import re
import subprocess
import sys
from typing import Dict, Optional, Tuple

from . import platform_utils

# 名称用 ASCII：security 按 -c 搜索中文名会匹配不到
CA_NAME = "AutoLabel Local Dev Root CA"
CA_SEARCH = "AutoLabel Local Dev"
LEAF_HOST = "autolabel.local"
DNS_NAMES = ("localhost", LEAF_HOST)
IP_ADDRESSES = ("127.0.0.1", "::1")
CA_DAYS = 3650          # 根证书十年，免得年年重签
LEAF_DAYS = 825         # 叶子证书两年，够短也不至于被浏览器嫌太长
RENEW_AFTER_DAYS = 45   # 剩不到一个半月就提醒


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def certs_dir() -> str:
    return os.environ.get("AUTOLABEL_CERTS") or os.path.join(project_root(), "certs")


def ca_cert_path() -> str:
    return os.path.join(certs_dir(), "ca.pem")


def ca_key_path() -> str:
    return os.path.join(certs_dir(), "ca-key.pem")


def leaf_cert_path() -> str:
    return os.path.join(certs_dir(), f"{LEAF_HOST}.pem")


def leaf_key_path() -> str:
    return os.path.join(certs_dir(), f"{LEAF_HOST}-key.pem")


def openssl_available() -> bool:
    return platform_utils.find_openssl() is not None


def openssl_bin() -> str:
    """openssl 的路径。找不到时退回裸名字，让系统按 PATH 试一次再报错。

    不能写死 ``"openssl"``：Windows 上它通常只在 Git for Windows 的 ``usr/bin``
    里，而双击启动时那个目录不在 PATH 上——明明装了却报"找不到"最难查。
    """
    found = platform_utils.find_openssl()
    return str(found) if found else "openssl"


def _run(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def _sha1_of_cert(path: str) -> str:
    """证书指纹（大写 SHA-1），用来和钥匙串里的记录比对。"""
    try:
        with open(path, "rb") as handle:
            pem = handle.read()
        import base64

        blocks = _pem_blocks(pem)
        if not blocks:
            return ""
        der = base64.b64decode(b"".join(b"".join(lines) for _, lines in blocks[:1]))
        return hashlib.sha1(der).hexdigest().upper()
    except OSError:
        return ""


def _pem_blocks(pem: bytes):
    """拆 PEM 里的每一段证书，返回 (类型行, base64 行列表)。"""
    blocks, current = [], None
    for line in pem.splitlines():
        if line.startswith(b"-----BEGIN"):
            current = (line, [])
        elif line.startswith(b"-----END") and current:
            blocks.append(current)
            current = None
        elif current:
            current[1].append(line.strip())
    return blocks


def _not_after(path: str) -> str:
    out = _run([openssl_bin(), "x509", "-in", path, "-noout", "-enddate"])
    return out.stdout.strip().split("=", 1)[-1] if out.returncode == 0 else ""


def days_left(path: str) -> Optional[int]:
    """还有几天过期；取不到时间就返回 None。"""
    out = _run([openssl_bin(), "x509", "-in", path, "-noout", "-enddate"])
    if out.returncode:
        return None
    raw = out.stdout.strip().split("=", 1)[-1]
    if not raw:
        return None
    try:
        parsed = datetime.datetime.strptime(raw, "%b %d %H:%M:%S %Y %Z").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    return int((parsed - datetime.datetime.now(datetime.timezone.utc)).total_seconds() // 86400)


def sans(path: str) -> Tuple[str, ...]:
    """证书里的备用名称（沙盒里的 LibreSSL 不一定支持 -ext，只能读文本）。"""
    out = _run([openssl_bin(), "x509", "-in", path, "-noout", "-text"])
    if out.returncode:
        return ()
    body = out.stdout.replace(" ", "").replace("IPAddress:", "IP:")
    found = re.findall(r"(?:DNS|IP):([^,\s\]]+)", body)
    return tuple(dict.fromkeys(found))


def _config_text(cn: str, org: str, sections: Dict[str, str], extra: str = "") -> str:
    lines = ["[req]", "distinguished_name = dn", "string_mask = utf8only", "prompt = no", ""]
    if extra:
        lines.append(extra)
        lines.append("")
    lines += ["[dn]"]
    for key, value in (("C", "CN"), ("O", org), ("CN", cn)):
        lines.append(f"{key} = {value}")
    lines.append("")
    for name, body in sections.items():
        lines.append(f"[{name}]")
        for key, value in body.items():
            lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)


def _alt_names_section() -> Dict[str, str]:
    alt = {}
    for i, name in enumerate(DNS_NAMES, 1):
        alt[f"DNS.{i}"] = name
    for i, ip in enumerate(IP_ADDRESSES, 1):
        alt[f"IP.{i}"] = ip
    return alt


def issue(force: bool = False) -> Dict:
    """签发根证书与 localhost 叶子证书，已存在且没到期就原样留着。"""
    if not openssl_available():
        raise RuntimeError("找不到 openssl 命令，无法签发本地证书\n        "
                           + platform_utils.openssl_missing_hint())
    os.makedirs(certs_dir(), exist_ok=True)
    os.chmod(certs_dir(), 0o700)
    existing = days_left(leaf_cert_path())
    if (not force and existing is not None and existing > RENEW_AFTER_DAYS
            and os.path.exists(ca_cert_path()) and os.path.exists(leaf_key_path())):
        return {"created": False, **status()}

    ca_cnf = os.path.join(certs_dir(), "ca.cnf")
    leaf_cnf = os.path.join(certs_dir(), "leaf.cnf")
    with open(ca_cnf, "w", encoding="utf-8") as handle:
        handle.write(_config_text(
            CA_NAME, "AutoLabel",
            {"v3_ca": {"subjectKeyIdentifier": "hash",
                       "basicConstraints": "critical, CA:TRUE",
                       "keyUsage": "critical, digitalSignature, cRLSign, keyCertSign"}},
            "x509_extensions = v3_ca"))
    with open(leaf_cnf, "w", encoding="utf-8") as handle:
        handle.write(_config_text(
            LEAF_HOST, "AutoLabel",
            {"v3_req": {"basicConstraints": "CA:FALSE",
                        "keyUsage": "critical, digitalSignature, keyEncipherment",
                        "extendedKeyUsage": "serverAuth",
                        "subjectAltName": "@alt"},
             "alt": _alt_names_section()},
            "req_extensions = v3_req"))

    steps = []
    if force or not os.path.exists(ca_cert_path()):
        steps.append([openssl_bin(), "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                      "-keyout", ca_key_path(), "-out", ca_cert_path(),
                      "-days", str(CA_DAYS), "-sha256", "-config", ca_cnf])
    csr = os.path.join(certs_dir(), "leaf.csr")
    steps.append([openssl_bin(), "req", "-new", "-newkey", "rsa:2048", "-nodes",
                  "-keyout", leaf_key_path(), "-out", csr, "-config", leaf_cnf])
    steps.append([openssl_bin(), "x509", "-req", "-in", csr, "-out", leaf_cert_path(),
                  "-CA", ca_cert_path(), "-CAkey", ca_key_path(), "-CAcreateserial",
                  "-days", str(LEAF_DAYS), "-sha256",
                  "-extfile", leaf_cnf, "-extensions", "v3_req"])
    for step in steps:
        done = _run(step)
        if done.returncode:
            raise RuntimeError(f"{step[1]} 失败：{(done.stderr or done.stdout).strip()[:200]}")
    verified = _run([openssl_bin(), "verify", "-CAfile", ca_cert_path(), leaf_cert_path()])
    if verified.returncode:
        raise RuntimeError("签出来的证书链校验不过："
                           f"{(verified.stderr or verified.stdout).strip()[:160]}")
    try:
        os.remove(csr)
    except OSError:
        pass
    for path in (ca_key_path(), leaf_key_path()):
        os.chmod(path, 0o600)
    return {"created": True, **status()}


def trust_store_label() -> str:
    """「根证书装在哪」用当前系统的话说一遍，命令行和界面共用一句。"""
    if platform_utils.is_windows():
        return "受信任根证书（当前用户）"
    if platform_utils.is_macos():
        return "钥匙串信任"
    return "系统信任"


def _certutil(*args: str) -> Tuple[int, str]:
    """跑 certutil 并合并输出。Windows 上「证书到底装没装上」只能问它。"""
    try:
        done = _run(["certutil", *args])
    except OSError as exc:
        return 1, str(exc)
    return done.returncode, ((done.stdout or "") + (done.stderr or "")).strip()


def _store_has_root(fingerprint: str) -> bool:
    """当前用户的「受信任的根证书颁发机构」里有没有这张根证书。

    拿指纹当查询条件而不是证书名：certutil 按名称匹配对空格和大小写都很敏感，
    比对指纹没这些毛病。输出里的指纹是"aa bb cc"这种带空格的写法，先抹干净。
    """
    code, body = _certutil("-store", "-user", "Root", fingerprint)
    if code != 0:
        return False
    return fingerprint.lower() in re.sub(r"[^0-9a-f]", "", body.lower())


def _install_trust_windows() -> Dict:
    if not os.path.exists(ca_cert_path()):
        return {"ok": False, "note": "还没有证书，先跑一次 autolabel ca"}
    want = _sha1_of_cert(ca_cert_path())
    if want and _store_has_root(want):
        return {"ok": True, "note": "根证书已在「受信任的根证书颁发机构」里，无需重复安装",
                "fingerprint": want}
    code, body = _certutil("-addstore", "-user", "Root", ca_cert_path())
    if code != 0:
        return {"ok": False, "note": f"写入受信任根证书失败：{body[:160]}"}
    return {"ok": True,
            "note": "已把根证书加入当前用户的「受信任的根证书颁发机构」（不需要管理员）。"
                    "中途弹的确认框点「是」；装完要完全退出浏览器再打开才生效",
            "fingerprint": want}


def _uninstall_trust_windows() -> Dict:
    want = _sha1_of_cert(ca_cert_path())
    if not want or not _store_has_root(want):
        return {"ok": True, "note": "本机信任列表里没有这张根证书，无需卸载"}
    code, body = _certutil("-delstore", "-user", "Root", want)
    if code != 0:
        return {"ok": False, "note": f"删除失败：{body[:160]}"}
    return {"ok": True, "note": f"已从「受信任的根证书颁发机构」移除「{CA_NAME}」"}


def keychain_fingerprints() -> Tuple[str, ...]:
    """钥匙串里所有证书的 SHA-1。按指纹比，不按名字搜——中文名匹配不可靠。"""
    if sys.platform != "darwin":
        return ()
    out = _run(["security", "find-certificate", "-a", "-Z"])
    return tuple(dict.fromkeys(re.findall(r"SHA-1 hash: ([0-9A-Fa-f]{40})", out.stdout)))


def chain_trusted() -> Tuple[bool, str]:
    """让系统自己判这条链能不能用于 TLS，比查列表准。"""
    if platform_utils.is_windows():
        want = _sha1_of_cert(ca_cert_path())
        if not want:
            return False, "读不到本地根证书的指纹，先跑一次 autolabel ca"
        if _store_has_root(want):
            return True, ""
        return False, "当前用户的「受信任的根证书颁发机构」里没有这张证书"
    if sys.platform != "darwin":
        return False, f"{platform_utils.current()}：需手动把 certs/ca.pem 设为受信任根证书"
    out = _run(["security", "verify-cert", "-p", "ssl", "-c", leaf_cert_path()])
    if out.returncode == 0:
        return True, ""
    return False, (out.stderr or out.stdout).strip()[:200]


def install_trust() -> Dict:
    """把根证书装进**当前用户**的信任域（不碰系统域，免管理员密码）。"""
    if platform_utils.is_windows():
        return _install_trust_windows()
    if sys.platform != "darwin":
        return {"ok": False, "note": f"{platform_utils.current()}：请把 certs/ca.pem 手动设为受信任根证书"}
    if not os.path.exists(ca_cert_path()):
        return {"ok": False, "note": "还没有证书，先跑一次 autolabel ca"}
    want = _sha1_of_cert(ca_cert_path())
    if want and want in keychain_fingerprints() and chain_trusted()[0]:
        return {"ok": True, "note": "根证书已在钥匙串里且已受信，无需重复安装", "fingerprint": want}
    done = _run(["security", "add-trusted-cert", "-r", "trustRoot",
                 "-k", os.path.expanduser("~/Library/Keychains/login.keychain-db"),
                 ca_cert_path()])
    if done.returncode:
        return {"ok": False, "note": f"写入钥匙串失败：{(done.stderr or done.stdout).strip()[:160]}"}
    return {"ok": True, "note": "已把根证书设为受信任（当前用户，不需要管理员密码）",
            "fingerprint": want}


def uninstall_trust() -> Dict:
    if platform_utils.is_windows():
        return _uninstall_trust_windows()
    if sys.platform != "darwin":
        return {"ok": False, "note": f"非 macOS：请手动删除信任列表里的「{CA_NAME}」"}
    keychain = os.path.expanduser("~/Library/Keychains/login.keychain-db")
    fingerprints = keychain_fingerprints()
    want = _sha1_of_cert(ca_cert_path())
    # 按指纹删：证书名带中文时 -c 会匹配不到
    targets = [want] if want in fingerprints else []
    if not targets:
        return {"ok": True, "note": "钥匙串里没有本项目当前这张根证书，无需卸载"}
    failures = []
    for item in targets:
        done = _run(["security", "delete-certificate", "-Z", item, keychain])
        if done.returncode:
            failures.append((done.stderr or done.stdout).strip()[:120])
    if failures:
        return {"ok": False, "note": "删除失败：" + failures[0]}
    return {"ok": True, "note": f"已从钥匙串移除「{CA_NAME}」"}


def status() -> Dict:
    leaf, ca = leaf_cert_path(), ca_cert_path()
    have = os.path.exists(leaf) and os.path.exists(ca) and os.path.exists(leaf_key_path())
    want = _sha1_of_cert(ca) if have else ""
    trusted = chain_trusted()[0] if have else False
    left = days_left(leaf) if have else None
    return {"dir": certs_dir(), "have_cert": have, "days_left": left,
            "trusted": trusted, "fingerprint": want,
            "sans": list(sans(leaf)) if have else [],
            "openssl": openssl_available(), "platform": platform_utils.current(),
            "trust_store": trust_store_label(),
            "expires_soon": bool(left is not None and left <= RENEW_AFTER_DAYS)}


def ensure(*, install: bool = True) -> Tuple[str, str]:
    """给 review 用：证书齐了就直接给路径，缺了就现签（并顺手装信任）。"""
    issue()
    if install:
        install_trust()
    return leaf_cert_path(), leaf_key_path()
