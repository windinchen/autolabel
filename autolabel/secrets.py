"""敏感信息集中存放：API Key 只写在这份文件里，模型配置只保存引用名。

默认落在 ``~/.autolabel/secrets.json``（可用环境变量 ``AUTOLABEL_SECRETS`` 覆盖），
文件权限 0600。这样 ``configs/vlm.json``、``configs/models.json`` 都能放心提交、
放心贴给别人看，里面只有 ``vlm_xxx`` 这种引用名，没有真实 Key。
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List, Optional

DEFAULT_SECRETS = os.path.join("~", ".autolabel", "secrets.json")
_REF_PATTERN = re.compile(r"^[0-9A-Za-z._:-]{1,80}$")
_LOCK = threading.RLock()


def secrets_path() -> str:
    """Key 文件路径：默认 ``~/.autolabel/secrets.json``。"""
    return os.path.abspath(os.path.expanduser(os.environ.get("AUTOLABEL_SECRETS") or DEFAULT_SECRETS))


def load_secrets(path: Optional[str] = None) -> Dict[str, str]:
    """读 Key 表；文件缺失或损坏按空表处理，不能让模型管理面板打不开。"""
    path = secrets_path() if path is None else path
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def save_secrets(secrets: Dict[str, str], path: Optional[str] = None) -> None:
    """原子写回 Key 表，权限收到 0600。"""
    path = secrets_path() if path is None else path
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in sorted(secrets.items()) if v}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:  # 只读挂载之类的文件系统，不因权限设置失败而报错
        pass


def valid_ref(ref: str) -> bool:
    """引用名只允许稳妥字符，避免把路径或 JSON 片段当 key 用。"""
    return bool(_REF_PATTERN.match(str(ref or "")))


def get_secret(ref: str, path: Optional[str] = None) -> str:
    """按引用名取 Key；引用名不存在时返回空串。"""
    if not ref:
        return ""
    return load_secrets(path).get(str(ref), "")


def put_secret(ref: str, value: str, path: Optional[str] = None) -> str:
    """写入 Key 并返回引用名；空值等于删除，返回值仍是引用名。"""
    ref = str(ref or "").strip()
    if not valid_ref(ref):
        raise ValueError(f"Key 引用名只能包含字母、数字和 . _ : -，收到 {ref!r}")
    with _LOCK:
        secrets = load_secrets(path)
        value = str(value or "").strip()
        if value:
            secrets[ref] = value[:400]
        else:
            secrets.pop(ref, None)
        save_secrets(secrets, path)
    return ref


def delete_secret(ref: str, path: Optional[str] = None) -> bool:
    """删除引用；返回是否真的删掉了什么。"""
    with _LOCK:
        secrets = load_secrets(path)
        ref = str(ref or "")
        if ref not in secrets:
            return False
        secrets.pop(ref, None)
        save_secrets(secrets, path)
    return True


def secret_refs(path: Optional[str] = None) -> List[str]:
    """已存的引用名列表，供「模型管理」提示哪些配置还缺 Key。"""
    return sorted(load_secrets(path))


def mask(value: str) -> str:
    """掩码显示：只露前 3 位和后 4 位。"""
    value = str(value or "")
    if not value:
        return ""
    if len(value) <= 8:
        return f"{value[0]}***{value[-1]}" if len(value) > 2 else "**"
    return f"{value[:3]}…{value[-4:]}"
