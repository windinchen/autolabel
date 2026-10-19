"""原子写：整文件换名落盘，临时名带上进程号与线程号。

以前全仓库六个写者（三个库的索引、批次、类别登记表、两份模型配置、Key 表）
都写成 ``path + ".tmp"``——同名。一台机器同时开两个复核实例（一卡一实例是
这本项目的正常用法），或者界面跑着的同时命令行在 ``import``，就会出现：

1. 进程 A 打开 ``index.json.tmp`` 写了一半；
2. 进程 B 也打开**同一个** ``index.json.tmp``，``"w"`` 直接把 A 的内容截断；
3. 谁后 ``os.replace`` 谁就把这份残缺文件变成了正式索引。

索引读不出来，界面上的表现是"库突然空了"，而下一次保存会把这张空表写回磁盘，
真把标注记录冲掉。所以临时名必须每个写者独享。
"""

from __future__ import annotations

import itertools
import json
import os
import threading
from typing import Any, Optional

#: 进程内自增序号。线程号看着能用，其实会被复用：线程退出后 CPython 会把同一个
#: ``get_ident()`` 发给后来者，赶上一次写完之前另开一个就又是同名临时文件。
#: 进程号区分进程，这个序号区分线程，两条合起来才真的不撞车。
_SEQUENCE = itertools.count(1)
_SEQUENCE_LOCK = threading.Lock()


def atomic_write_text(path: str, text: str, *, mode: Optional[int] = None) -> None:
    """把 ``text`` 原子写进 ``path``：先写独享临时文件，再 ``os.replace`` 换名。

    :param mode: 需要收紧权限时用（Key 表要 0600）；文件系统不支持就放过，
        与原来 ``secrets`` 的处理保持一致——存不下密钥比权限松一档更糟。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with _SEQUENCE_LOCK:
        token = next(_SEQUENCE)
    tmp = os.path.join(directory,
                       f".{os.path.basename(path)}.{os.getpid()}.{token}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
        if mode is not None:
            try:
                os.chmod(tmp, mode)
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_json(path: str, payload: Any, *, mode: Optional[int] = None,
                      sort_keys: bool = False, newline: bool = False) -> None:
    """JSON 版原子写，参数与各处原来手写的 ``json.dump`` 一致。"""
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys)
    atomic_write_text(path, body + ("\n" if newline else ""), mode=mode)
