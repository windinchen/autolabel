"""第 4 轮体检（磁盘与索引一致性）：多个写者同时落盘，谁都不许把别人的文件截断。

盯三件事：

1. 临时文件名必须每个写者独享——以前六处都叫 ``xxx.tmp``，两个实例并行会写出残缺文件；
2. 换名之后不留临时文件；
3. Key 表 0600、索引 sort_keys 这些细节不能因为换助手而退化。
"""
import json
import os
import subprocess
import sys
import threading

from autolabel.atomic import atomic_write_json, atomic_write_text
from autolabel.dataset import index_path, load_index
from autolabel.secrets import save_secrets

#: 两个子进程各写这么多份，每份约 1 MB：窗口足够宽，以前那份同名临时文件必炸
ROUNDS = 40
BLOB_BYTES = 1024 * 1024

CHILD = (
    "import json,os,sys;"
    "sys.path.insert(0, os.environ['REPO']);"
    "from autolabel.atomic import atomic_write_text;"
    "path=sys.argv[1]; tag=sys.argv[2]; size=int(sys.argv[3]); rounds=int(sys.argv[4]);"
    "blob='A'*size if tag=='A' else 'b';"
    "body=json.dumps({'tag':tag,'blob':blob});"
    "[atomic_write_text(path, body) for _ in range(rounds)]"
)


def test_concurrent_writers_never_publish_a_truncated_file(tmp_path):
    """一卡一实例同时写同一份索引时，最后的文件必须还是能解析的完整 JSON。"""
    target = tmp_path / "index.json"
    env = dict(os.environ, REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    procs = [subprocess.Popen([sys.executable, "-c", CHILD, str(target), tag,
                               str(BLOB_BYTES), str(ROUNDS)], env=env)
             for tag in ("A", "B")]
    codes = [proc.wait(timeout=180) for proc in procs]
    assert codes == [0, 0], codes
    data = json.loads(target.read_text(encoding="utf-8"))     # 残缺文件在这里就会抛
    assert data["tag"] in ("A", "B")
    assert data["blob"] in ("A" * BLOB_BYTES, "b"), "内容被两个写者搅在一起了"
    assert [n for n in os.listdir(tmp_path) if ".tmp" in n] == []


def test_temp_names_are_unique_per_thread(tmp_path):
    """同一进程里的两个线程也不能共用临时名。"""
    target = tmp_path / "index.json"
    seen = []
    lock = threading.Lock()
    real_replace = os.replace

    def spy(src, dst):
        with lock:
            seen.append(src)
        return real_replace(src, dst)

    import autolabel.atomic as atomic

    original = atomic.os.replace
    atomic.os.replace = spy
    try:
        threads = [threading.Thread(target=atomic_write_text,
                                    args=(str(target), json.dumps({"w": i}))) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        atomic.os.replace = original
    assert len(set(seen)) == len(seen), f"临时名撞车：{seen}"
    assert json.loads(target.read_text(encoding="utf-8")) in [{"w": i} for i in range(8)]


def test_no_temp_files_left_behind(tmp_path):
    target = tmp_path / "nested" / "index.json"
    for i in range(10):
        atomic_write_json(str(target), {"n": i})
    assert json.loads(target.read_text(encoding="utf-8")) == {"n": 9}
    assert [n for n in os.listdir(tmp_path / "nested") if ".tmp" in n] == []


def test_the_index_file_keeps_its_shape(dataset):
    """换成助手之后，索引还得是那份排好序、键名不变的 JSON。"""
    idx = load_index(dataset)
    idx.upsert("x.jpg", status="approved")
    idx.save()
    path = index_path(dataset)
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    assert raw == json.dumps(json.loads(raw), ensure_ascii=False, indent=2, sort_keys=True)
    assert not [n for n in os.listdir(os.path.dirname(path)) if ".tmp" in n]


def test_the_key_table_still_locks_its_permissions(tmp_path, monkeypatch):
    """API Key 落盘必须还是 0600：临时名换了不能把权限一起换掉。"""
    target = tmp_path / "secrets.json"
    monkeypatch.setenv("AUTOLABEL_SECRETS", str(target))
    save_secrets({"openai": "sk-secret"}, str(target))
    assert oct(os.stat(target).st_mode & 0o777) == "0o600", oct(os.stat(target).st_mode & 0o777)
    assert [n for n in os.listdir(tmp_path) if ".tmp" in n] == []
