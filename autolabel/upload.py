"""浏览器上传与后台标注：把网页收到的图片/视频收进数据集，并在后台跑标注流水线。

上传的文件先落在 ``<dataset>/.autolabel/uploads/``，任务成功后删除；
标注在后台线程串行执行（同一时刻只跑一个任务，避免反复加载模型）。
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .dataset import IMAGE_EXTS as IMPORTABLE_IMAGE_EXTS, now_str
from .batches import attach_items, batch_name, create_batch, display_name, finish_batch

from .library import LIB_DETECT, LIB_OBJECTS, LIB_POSE

UPLOAD_CHUNK = 1 << 20  # 上传落盘时的分块大小：1 MB
# 上次进程被打断时留下的上传临时文件，超过这个小时数就在启动时清掉
STALE_UPLOAD_HOURS = 24

#: 浏览器能传的 = 命令行能导的，两处同一个口径。导入时统一转码成 JPEG，
#: 界面从不直接看原图，所以 ``<img>`` 渲染不出来的 TIFF 照样能收——以前就差
#: ``.tif``/``.tiff`` 两个后缀，同一个文件命令行导得进、界面里报"不支持的文件类型"。
IMAGE_EXTS = set(IMPORTABLE_IMAGE_EXTS)
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv", ".wmv",
              ".mts", ".m2ts", ".ts", ".mpg", ".mpeg", ".3gp"}
# 保留中文与常见安全字符，其余替换为下划线
_SAFE = re.compile(r"[^0-9A-Za-z._\u4e00-\u9fff-]+")

DEFAULT_MAX_UPLOAD_MB = 1024


class UploadTooLarge(ValueError):
    """上传体积超过单文件上限；请求体已经读干净，可以安全地回 413。"""


class JobCancelled(Exception):
    """任务被用户在界面上「清除」，用于从流水线阶段提前退出。"""


def duplicate_note(counts: dict) -> str:
    """把"传了 255 张怎么只有 169 张"讲明白：重复跳过多少、读不出多少，没有就返回空串。

    以前完成消息里的"已进池 255 张"把被去重跳过的张数也算进去了，界面上看着一张没少，
    池里却只有 169 个文件——少了东西又不说少了什么，比少进更让人不信任。

    「内容完全相同」和「画面近似」必须分开说：抽出来的视频帧几乎不会字节相同，
    它们是被**近似去重**吃掉的。把它们说成「内容相同」，而用户手上那 255 个文件
    确实一个都不重样，读到的就是「软件在编理由」。
    """
    parts = []
    if counts.get("duplicates"):
        parts.append(f"{counts['duplicates']} 张与已有素材内容相同，没有重复入库")
    if counts.get("similar"):
        parts.append(f"{counts['similar']} 张画面与已有素材接近，"
                     f"按「近似去重」每个画面只留 1 张")
    if counts.get("failed"):
        parts.append(f"{counts['failed']} 个文件读不出图片")
    return f"（{'，'.join(parts)}）" if parts else ""


DEDUPE_MODES = ("none", "exact", "perceptual")


def dedupe_mode(value: object, name: str = "去重方式", default: str = "exact") -> str:
    """把界面递交的去重方式收成 ``none`` / ``exact`` / ``perceptual`` 三种之一。

    这个值决定抽出来的帧会被吃掉多少：``perceptual`` 把画面相近的帧当成同一帧丢掉，
    视频帧几乎全中招——5 条录像抽出 264 帧只剩 46 张就是这么来的。所以默认走
    ``exact``（只跳过字节完全相同的帧），而且传进来一个不认识的词要报错：以前它会
    被下游当成「不去重」或「近似去重」默默改掉行为，比直接失败难查得多。
    """
    mode = str(value if value is not None else "").strip().lower() or default
    if mode not in DEDUPE_MODES:
        raise ValueError(f"{name}只能是 {' / '.join(DEDUPE_MODES)}，现在是 {mode}")
    return mode


def prune_stale_uploads(root: str, max_age_hours: float = STALE_UPLOAD_HOURS) -> dict:
    """清掉上一次进程遗留的上传临时文件，返回 ``{"files": 份数, "bytes": 字节数}``。

    上传的原始文件先落在 ``.autolabel/uploads/``，任务成功后逐个删除；进程中途被打断
    （Ctrl+C、崩溃、重启电脑）时没人再管它们，一次视频上传就是几百 MB，而且界面上
    完全看不见。只动超过 ``max_age_hours`` 没被碰过的文件，另一个还在跑着的复核服务
    刚收下的文件不受影响。
    """
    report = {"files": 0, "bytes": 0}
    folder = uploads_dir(root)
    if not os.path.isdir(folder):
        return report
    deadline = time.time() - max(0.0, float(max_age_hours)) * 3600.0
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return report
    for entry in entries:
        try:
            stat = entry.stat()
            if not entry.is_file(follow_symlinks=False) or stat.st_mtime >= deadline:
                continue
            os.remove(entry.path)
        except OSError:
            continue
        report["files"] += 1
        report["bytes"] += int(stat.st_size)
    return report


def finish_batch_any(root: str, batch_id: str, status: str = "done", message: str = "") -> None:
    """批次收尾：``p_`` 开头的是素材池批次（记在 ``pool.json``），其余是各库批次。

    上传任务只有"批次 ID"这一个线索，靠 ID 前缀分派比每个调用点多问一遍 kind 稳当——
    少分派一次，池批次就会永远挂在"标注中"，界面上删不掉。
    """
    if str(batch_id or "").startswith("p_"):
        from . import pool as pool_catalog

        pool_catalog.finish_batch(root, batch_id, status, message)
        return
    finish_batch(root, batch_id, status, message)


def lib_for_kind(kind: str) -> str:
    """任务类型 → 标注库：``prompt*`` 归提示词库，``detect*`` 归目标检测库，其余归姿态库。"""
    name = str(kind or "")
    if name.startswith("prompt"):
        return LIB_OBJECTS
    if name.startswith("detect"):
        return LIB_DETECT
    return LIB_POSE


def uploads_dir(root: str) -> str:
    path = os.path.join(os.path.abspath(root), ".autolabel", "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def safe_upload_name(name: str) -> str:
    base = os.path.basename(str(name).replace("\\", "/"))
    base = _SAFE.sub("_", base).strip("._")
    return base[:120] or "file"


def _ext(path: str) -> str:
    """小写后缀；上传落盘名带的是原文件名，中文和空格都得受得住。"""
    return os.path.splitext(str(path))[1].lower()


def kind_for(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return ""


def load_hint_text(name: str, size_mb: float = 0.0) -> str:
    """加载权重时给界面的那句话：把「多大、大概多久」写出来，慢就不像卡死。

    经验值：读盘 + 建计算图 + 搬上显存，100 MB 权重量级是十几到几十秒，
    200 MB 以上（26x / yolov8x-seg 这类）到一两分钟都正常。给区间不给准数，
    估快了用户以为坏了，估慢了没人信。
    """
    size = float(size_mb or 0)
    if size <= 0:
        return f"加载模型 {name}…"
    low = max(5, int(size / 6))
    high = max(low + 5, int(size / 1.5))
    return (f"加载模型 {name}（权重 {round(size, 1)} MB，"
            f"读进内存 / 显存约需 {low}–{high} 秒，大模型更久属正常）…")


def save_upload(root: str, name: str, data: bytes, max_mb: int = DEFAULT_MAX_UPLOAD_MB) -> dict:
    """把上传字节写入 uploads 目录，返回带唯一名的存储信息。"""
    if not data:
        raise ValueError("空文件")
    return save_upload_stream(root, name, io.BytesIO(data), len(data), max_mb=max_mb)


def save_upload_stream(root: str, name: str, source, length: int,
                       max_mb: int = DEFAULT_MAX_UPLOAD_MB) -> dict:
    """边读边写把上传落盘，最多消费 ``length`` 字节后返回。

    两条硬性约定，缺一条就会带来线上故障：

    1. **必须把 length 字节读干净**（哪怕已经判定超限或类型不支持）才允许返回或抛错。
       HTTP/1.1 复用连接时，残留在 socket 里的请求体会被下一个请求当成请求行，
       客户端只能看到 ``Broken pipe``，界面上就只剩「网络错误」。
    2. **不能整份进内存**：浏览器一次拖几十条录像很常见，整份 read 会带来和模型
       抢内存的瞬时尖峰，所以这里分块读写。
    """
    if length <= 0:
        raise ValueError("空文件")
    safe = safe_upload_name(name)
    ext = os.path.splitext(safe)[1].lower()
    kind = kind_for(ext)
    if not kind:
        _drain(source, length)
        raise ValueError(f"不支持的文件类型: {ext or safe}（图片 {sorted(IMAGE_EXTS)} / 视频 {sorted(VIDEO_EXTS)}）")
    limit = max(0, int(max_mb)) * 1024 * 1024
    stored = f"{uuid.uuid4().hex[:8]}_{safe}"
    path = os.path.join(uploads_dir(root), stored)
    received = 0
    too_large = False
    try:
        with open(path, "wb") as fh:
            while received < length:
                chunk = source.read(min(UPLOAD_CHUNK, length - received))
                if not chunk:
                    break  # 客户端提前断开，剩下的按中断处理
                received += len(chunk)
                if too_large:
                    continue
                if received > limit:
                    too_large = True
                    continue
                fh.write(chunk)
    except OSError as exc:
        raise ValueError(f"保存上传文件失败: {exc}") from exc
    finally:
        if received < length:
            _drain(source, length - received)
    if received != length:
        _discard(path)
        raise ValueError(f"上传中断：收到 {received} / {length} 字节")
    if too_large:
        _discard(path)
        raise UploadTooLarge(f"文件超过 {max_mb} MB 上限")
    return {"ok": True, "stored": stored, "name": safe, "kind": kind, "bytes": received}


def _drain(source, length: int) -> None:
    """丢弃请求体里剩余的字节，保证连接可以安全复用。"""
    remaining = max(0, int(length))
    while remaining > 0:
        chunk = source.read(min(UPLOAD_CHUNK, remaining))
        if not chunk:
            return
        remaining -= len(chunk)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@dataclass
class Job:
    """一次上传后的处理任务：入库 → 标注 → QC → 坐标导出。"""

    id: str
    kind: str
    files: List[str]
    params: dict
    stage: str = "queued"
    message: str = "排队中"
    done: int = 0
    total: int = 0
    finished: bool = False
    ok: Optional[bool] = None
    cancel_requested: bool = False
    batch_id: str = ""
    ingested: List[str] = field(default_factory=list)
    #: 入库时被去重跳过 / 读不出图片的张数，完成消息要照这个说，不能糊在"入库 N 张"里
    counts: dict = field(default_factory=dict)
    #: 每个视频抽了多少帧、用的哪个解码器，完成消息与「批次详情」都要照着说
    videos: List[dict] = field(default_factory=list)
    frames: int = 0
    result: dict = field(default_factory=dict)
    error: str = ""
    created_at: str = field(default_factory=now_str)
    updated_at: str = field(default_factory=now_str)

    def set_stage(self, stage: str, message: str = "", done: int = None, total: int = None) -> None:
        self.stage = stage
        if message:
            self.message = message
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        self.updated_at = now_str()

    def ingest_note(self) -> str:
        """完成消息里那对括号：少掉的张数 + 视频抽了多少帧，一次说完。

        「我那条 3 分钟的录像怎么池里只有 90 张」是抽帧最容易被当成丢素材的疑问：
        按每 2 秒 1 帧本来就该是 90 帧，把数字和原因一起说出来才不用猜。
        """
        parts = []
        quiet = duplicate_note(self.counts).strip("（）")
        if quiet:
            parts.append(quiet)
        if self.videos:
            got = sum(int(row.get("extracted") or 0) for row in self.videos)
            kept = sum(int(row.get("imported") or 0) for row in self.videos)
            parts.append(f"视频 {len(self.videos)} 个抽出 {got} 帧、进池 {kept} 帧")
            from .video import sampling_rule

            rule = sampling_rule(self.params.get("fps"))
            if rule:
                parts.append(f"取样{rule}")
            parts.extend(sorted({str(row.get("note") or "") for row in self.videos if row.get("note")}))
        return f"（{'，'.join(parts)}）" if parts else ""

    def payload(self) -> dict:
        return {"id": self.id, "kind": self.kind, "files": self.files[:6], "file_count": len(self.files),
                "stage": self.stage, "message": self.message, "done": self.done, "total": self.total,
                "finished": self.finished, "ok": self.ok, "result": self.result, "error": self.error,
                "cancel_requested": self.cancel_requested,
                "batch_id": self.batch_id, "counts": self.counts,
                "frames": self.frames, "videos": self.videos,
                "model": self.params.get("model"),
                "vlm_profile": self.params.get("vlm_profile") or "",
                "vlm_name": self.params.get("vlm_name") or "",
                "created_at": self.created_at, "updated_at": self.updated_at}


class JobStore:
    """任务表 + 后台工作线程；``on_finish`` 用于让复核索引重新读取磁盘。"""

    def __init__(self, root: str, on_finish: Optional[Callable[[], None]] = None,
                 max_mb: int = DEFAULT_MAX_UPLOAD_MB):
        self.root = os.path.abspath(root)
        self.on_finish = on_finish
        self.max_mb = max_mb
        self.jobs: List[Job] = []
        self.lock = threading.Lock()
        self._gate = threading.Semaphore(1)
        self._running_id: Optional[str] = None

    # --- 对外 ---
    def submit(self, kind: str, files: Sequence[str], params: Optional[dict] = None) -> Job:
        if kind not in ("image", "video", "pool", "pose", "prompt", "prompt_upload",
                        "detect", "detect_upload"):
            raise ValueError(f"未知任务类型: {kind}")
        # 只跑标注不入库的那三类，files 是本库索引里的相对路径，不能被 basename 削掉子目录
        rel_paths = kind in ("pose", "prompt", "detect")
        if rel_paths:
            files = [str(f).replace("\\", "/") for f in files if str(f).strip()]
            files = self._known_images(files, lib_for_kind(kind))
        else:
            files = [os.path.basename(f) for f in files if f]
        if not files:
            raise ValueError("没有收到文件")
        if not rel_paths:
            for name in files:
                if not os.path.exists(os.path.join(uploads_dir(self.root), name)):
                    raise FileNotFoundError(f"上传文件不存在或已清理: {name}")
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, files=list(files),
                  params=self._normalize_params(params))
        with self.lock:
            self.jobs.append(job)
            self.jobs = self.jobs[-20:]
        thread = threading.Thread(target=self._run, args=(job,), name=f"autolabel-job-{job.id}", daemon=True)
        thread.start()
        return job

    def _known_images(self, rels: Sequence[str], lib: str = "pose") -> List[str]:
        """任务只接受该库索引里真的存在的图，免得跑起来才一路报缺文件。"""
        from .dataset import load_index

        if not rels:
            return []
        index = load_index(self.root, lib)
        missing = [rel for rel in rels if index.get(rel) is None]
        known = [rel for rel in rels if index.get(rel) is not None]
        if missing:
            from .library import lib_name

            print(f"[warn] {lib_name(lib)}任务跳过不存在的图片 {len(missing)} 张，例如 {missing[0]}")
        return known

    @staticmethod
    def _normalize_params(params: Optional[dict]) -> dict:
        """选了多模态配置就当场解析成 ID + 名称：提交时校验，别等任务跑到标注才发现配置不存在。"""
        params = dict(params or {})
        wanted = str(params.get("vlm_profile") or "").strip()
        if not wanted:
            params.pop("vlm_profile", None)
            return params
        from . import models as model_manager

        profile = model_manager.find_profile(wanted)
        if profile is None:
            options = "、".join(p.get("name", "") for p in model_manager.load_vlm()["profiles"]) or "（还没有）"
            raise ValueError(f"找不到多模态配置 {wanted}；可选：{options}（先在「模型管理」里添加）")
        params["vlm_profile"] = profile["id"]
        params["vlm_name"] = profile.get("name", wanted)
        return params

    def snapshot(self) -> List[dict]:
        with self.lock:
            jobs = list(self.jobs)
        return [job.payload() for job in reversed(jobs)]

    def active(self) -> bool:
        with self.lock:
            return any(not job.finished for job in self.jobs)

    def clear(self) -> dict:
        """清除任务：排队中的立即删除，正在跑的请求取消，并清掉历史记录。

        正在标注的任务不会硬杀线程（模型推理中途强杀会留下半截索引），
        而是等当前图片标完再停，已写入的标注全部保留。

        :return: ``{"cancelled": 立即删除的排队任务数, "cancelling": 正在取消的运行中任务数,
                  "dropped": 清掉的历史任务记录数}``
        """
        counts = {"cancelled": 0, "cancelling": 0, "dropped": 0}
        stale_files: List[str] = []
        with self.lock:
            keep: List[Job] = []
            for job in self.jobs:
                if job.finished:
                    counts["dropped"] += 1
                    continue
                if job.id == self._running_id:
                    if not job.cancel_requested:
                        job.cancel_requested = True
                        job.set_stage(job.stage, "已请求取消，当前图片标完后停止")
                        counts["cancelling"] += 1
                    keep.append(job)
                    continue
                job.cancel_requested = True
                job.set_stage("cancelled", "已取消（未开始）")
                job.finished, job.ok = True, False
                counts["cancelled"] += 1
                stale_files.extend(job.files)
            self.jobs = keep
        for name in stale_files:  # 排队任务的上传临时文件不再需要
            try:
                os.remove(os.path.join(uploads_dir(self.root), name))
            except OSError:
                pass
        self._notify()
        return counts

    # --- 流水线 ---
    def _run(self, job: Job) -> None:
        acquired = self._gate.acquire(timeout=600)
        if not acquired:
            job.set_stage("failed", "等待空闲标注槽位超时")
            job.finished, job.ok = True, False
            self._notify()
            return
        try:
            with self.lock:
                if job.cancel_requested:  # 排队期间就被清除了
                    raise JobCancelled(job.message or "已取消（未开始）")
                self._running_id = job.id
            self._start_batch(job)
            if job.kind == "prompt":
                self._run_prompt(job)
            elif job.kind == "prompt_upload":
                self._run_prompt_upload(job)
            elif job.kind == "pose":
                self._run_pose(job)
            elif job.kind == "detect":
                self._run_detect(job)
            elif job.kind == "detect_upload":
                self._run_detect_upload(job)
            elif job.kind == "pool":
                self._run_pool(job)
            else:
                self._run_import(job)
        except JobCancelled as exc:
            self._attach_batch(job)
            finish_batch_any(self.root, job.batch_id, "cancelled", f"{exc or '任务已取消'}，已完成的标注保留")
            salvaged = self._salvage(job)
            job.result = salvaged
            job.set_stage("cancelled", f"{exc or '任务已取消'}，已完成的标注与坐标已保留")
            job.finished, job.ok = True, False
        except Exception as exc:  # noqa: BLE001 后台线程必须兜住所有异常
            job.error = f"{type(exc).__name__}: {exc}"
            job.set_stage("failed", job.error)
            job.finished, job.ok = True, False
            finish_batch_any(self.root, job.batch_id, "failed", job.error)
        finally:
            with self.lock:
                if self._running_id == job.id:
                    self._running_id = None
                if job.cancel_requested:  # 被清除的任务不再留在任务列表里
                    self.jobs = [item for item in self.jobs if item.id != job.id]
            self._gate.release()
            self._notify()

    def _load_hint(self, job: Job) -> str:
        """权重没下载齐时说清楚要在下载，别让界面看着像卡死。"""
        from .backends import model_status

        name = str(job.params.get("model") or "yolo11n-pose.pt")
        wanted = str(job.params.get("vlm_profile") or "").strip()
        if wanted:
            # 多模态标注是逐图网络请求，慢是常态，先把预期写在进度条上
            from . import models as model_manager

            profile = model_manager.find_profile(wanted)
            if profile is None:
                return f"加载多模态大模型配置 {wanted}…"
            kind = "本地" if profile.get("kind") == "local" else "云端"
            return (f"用多模态大模型 {profile.get('name')}（{kind} · {profile.get('model')}）"
                    f"逐图标注，每张一次网络请求，比 YOLO 慢很多属正常…")
        try:
            info = model_status(name)
        except OSError:
            return "加载模型…"
        if info["ready"]:
            return load_hint_text(name, float(info["size_mb"] or 0))
        note = f"已下载 {info['size_mb']} MB" if info["size_mb"] else "尚未下载"
        return f"首次加载 {name}：正在下载权重（{note}，视网络可能要几分钟）…"

    @contextmanager
    def _loading(self, job: Job, hint: str):
        """把"把权重读进内存/显存"这段单独标成 ``loading``，并每秒报一次已等时长。

        大权重（yolov8x / 26x 这类 100 MB 以上）冷加载要几十秒，以前这段时间被并进
        ``labeling`` 且 0/0，进度条一口气跳到一半再一动不动——看着就是卡死了，很多人
        于是点「清除任务」把正在加载的模型掐了。单独一个阶段 + 一直在动的秒数，
        才能区分"在加载"和"卡住"。
        """
        started = time.time()
        job.set_stage("loading", f"{hint}（已用 0 秒）", done=0, total=0)
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(1.0):
                # 内部的进度回调一旦把阶段推到 labeling，这条心跳就该自己退场
                if job.stage != "loading" or job.finished:
                    return
                job.set_stage("loading", f"{hint}（已用 {int(time.time() - started)} 秒）",
                              done=0, total=0)

        threading.Thread(target=tick, name=f"load-hint-{job.id}", daemon=True).start()
        try:
            yield
        finally:
            stop.set()

    def _check(self, job: Job) -> None:
        """阶段边界检查：收到清除请求就抛出 :class:`JobCancelled`。"""
        if job.cancel_requested:
            raise JobCancelled("任务已取消")

    def _start_batch(self, job: Job) -> None:
        params = job.params
        if job.kind == "pool":
            # 只进池的素材在任何库的索引里都没有条目，批次只能记在池账本上
            from . import pool as pool_catalog

            job.batch_id = pool_catalog.create_batch(
                self.root,
                name=batch_name("pool", [display_name(f) for f in job.files],
                                str(params.get("batch_name") or "")),
                kind="pool", source="upload", uploads=job.files)["id"]
            return
        batch = create_batch(
            self.root,
            name=batch_name(job.kind, [display_name(f) for f in job.files],
                            str(params.get("batch_name") or "")),
            kind=job.kind, source="upload", lib=lib_for_kind(job.kind),
            params={"model": params.get("vlm_name") or params.get("model"), "fps": params.get("fps"),
                    "max_frames": params.get("max_frames"), "label": params.get("label"),
                    "video_dedupe": params.get("video_dedupe"),
                    "vlm_profile": params.get("vlm_profile") or "",
                    "prompt_fields": params.get("prompt_fields") or [],
                    "detect_classes": params.get("detect_classes") or [],
                    "conf": params.get("conf"),
                    "overwrite": bool(params.get("overwrite")),
                    "prompt_extra": params.get("prompt_extra") or "",
                    "targets": len(job.files)},
            uploads=job.files)
        job.batch_id = batch["id"]

    def _stamp_pool_origin(self, job: Job) -> None:
        """素材一落池，就在池账本上把「来处 + 分类」两枚章盖好。

        以前只有素材池页自己的导入盖来处，三个模块导入的素材在池账本上是"没有批次"，
        于是素材池页按批次筛筛不到、按批次删也删不到；分类同理，只能事后框一遍。
        取消的任务也盖——已经进池的那部分素材是真实存在的，没理由让它没来处没分类。
        """
        if not job.ingested:
            return
        from . import pool as pool_catalog

        if job.batch_id:
            pool_catalog.attach_items(self.root, job.batch_id, list(job.ingested))
        wanted = str(job.params.get("category") or "").strip()
        if not wanted:
            return
        try:
            pool_catalog.assign(self.root, list(job.ingested), wanted)
        except ValueError as exc:
            # 分类刚好在这期间被人删了：素材照样进池，只是这批没分成类，说一声不拦路
            job.set_stage(job.stage, f"{job.message}（分类没打上：{exc}）")

    def _attach_batch(self, job: Job) -> None:
        """把已入库的图片挂到批次上（取消时也只挂实际落盘的那部分）。"""
        self._stamp_pool_origin(job)
        if not (job.batch_id and job.ingested) or str(job.batch_id).startswith("p_"):
            return  # 池批次在 _stamp_pool_origin 里已经挂好了
        attach_items(self.root, job.batch_id, job.ingested, lib=lib_for_kind(job.kind))

    def _ingest(self, job: Job) -> int:
        """把任务收到的文件落进池/库：图片直接入库，视频先抽帧。

        **按后缀分派，不按任务类型分派**：素材池页上传的视频走的是 ``pool`` 任务，
        提示词标注和目标检测标注上传时图片和视频混在同一个任务里——以前只有姿态页
        那个 ``video`` 任务会抽帧，其余三处把 .mp4 当图片交给解码器，一张都进不来，
        界面上只剩「已进池 0 张」。谁上传都得是同一个结果。
        """
        from .dataset import import_images

        # 素材池页的导入只落文件、不进库：各库要用再从「从素材池添加」里挑
        register = job.kind != "pool"
        max_side = int(job.params.get("max_side", 1600))
        lib = lib_for_kind(job.kind)
        paths = [os.path.join(uploads_dir(self.root), name) for name in job.files]
        videos = [path for path in paths if kind_for(_ext(path)) == "video"]
        stills = [path for path in paths if kind_for(_ext(path)) != "video"]
        imported = 0
        if videos:
            imported += self._extract(job, videos, register=register, lib=lib)
        if stills:
            stats = import_images(self.root, stills, max_side=max_side,
                                  dedupe=dedupe_mode(job.params.get("dedupe"), "图片去重方式"),
                                  lib=lib, register=register)
            self._tally(job, stats)
            job.ingested.extend(rel for _src, rel in stats.get("imported_files", []))
            imported += int(stats["imported"]) + int(stats["reused"])
        self._attach_batch(job)
        return imported

    def _extract(self, job: Job, paths: List[str], *, register: bool, lib: str) -> int:
        """逐个视频抽帧入库，并把「哪个视频抽了几帧、用的哪个解码器」记在任务上。

        完成消息要能回答「我那条 3 分钟的录像怎么池里只有 90 张」——按 0.5 fps
        抽本来就该是 90 帧，不说清楚就像丢了东西。
        """
        from .video import extract_frames

        imported = 0
        for path in paths:
            self._check(job)
            job.set_stage("importing", f"抽帧 {os.path.basename(path)}")
            result = extract_frames(path, self.root, fps=float(job.params.get("fps", 0.5)),
                                    max_frames=int(job.params.get("max_frames", 600)),
                                    scene=job.params.get("scene"),
                                    start=float(job.params.get("start") or 0.0),
                                    duration=job.params.get("duration"),
                                    max_side=int(job.params.get("max_side", 1600)),
                                    dedupe=dedupe_mode(job.params.get("video_dedupe"),
                                                       "视频抽帧去重方式"),
                                    lib=lib, register=register)
            imported += int(result["imported"]) + int(result.get("reused") or 0)
            self._tally(job, result)
            job.ingested.extend(rel for _src, rel in result.get("imported_files", []))
            self._relabel_video(job, result, path, register=register, lib=lib)
            job.videos.append({"video": display_name(os.path.basename(path)),
                               "extracted": int(result.get("extracted") or 0),
                               "imported": len(result.get("imported_files") or []),
                               "backend": result.get("backend") or "",
                               "note": result.get("note") or ""})
        job.frames = sum(int(row.get("extracted") or 0) for row in job.videos)
        return imported

    def _relabel_video(self, job: Job, result: dict, path: str, *, register: bool, lib: str) -> None:
        """把账本里的「来源视频」换成用户原来那个文件名。

        抽帧时来源写的是 ``.autolabel/uploads/`` 里的临时文件，任务收尾就删了；账本上
        再留着 ``a1b2c3d4_现场录像.mp4`` 这种带哈希前缀的临时名，界面上读起来就像
        「我那个文件怎么变成乱码了」，而它其实只是上传时的去重用的前缀。
        """
        rels = [rel for _src, rel in result.get("imported_files", [])]
        if not rels:
            return
        label = display_name(os.path.basename(path))
        if register:
            from .dataset import load_index

            index = load_index(self.root, lib)
            for rel in rels:
                index.upsert(rel, video=label)
            index.save()
            return
        from . import pool as pool_catalog

        pool_catalog.record_files(self.root, [(rel, {"video": label}) for rel in rels])

    @staticmethod
    def _tally(job: Job, stats: dict) -> None:
        """把去重跳过 / 读取失败的分项累加到任务上，供完成消息原样报出来。

        「完全相同」和「画面近似」分开记：前者是用户真的重传了同一个文件，后者
        多半是同一场戏的相邻帧。混成一句「内容相同」，用户明明没重传，只能认为
        软件在找借口。
        """
        job.counts["duplicates"] = (int(job.counts.get("duplicates") or 0)
                                    + int(stats.get("skipped_exact") or 0))
        job.counts["similar"] = (int(job.counts.get("similar") or 0)
                                + int(stats.get("skipped_similar") or 0))
        job.counts["failed"] = int(job.counts.get("failed") or 0) + int(stats.get("failed") or 0)

    def _label(self, job: Job) -> dict:
        from .backends import BackendSettings
        from .pipeline import label_dataset

        fields = {k: v for k, v in job.params.items()
                  if k in BackendSettings.__dataclass_fields__ and v is not None}
        # 界面上的导入任务不带设备/半精度，这里补系统设置里的值：
        # 「系统设置」选的算力要对界面生效，否则它只是个摆设。
        from . import settings as system

        for key, value in system.label_defaults().items():
            fields.setdefault(key, value)
        if isinstance(fields.get("extra_models"), str):
            fields["extra_models"] = tuple(m.strip() for m in fields["extra_models"].split(",") if m.strip())
        settings = BackendSettings(**fields)

        def progress(done: int, total: int) -> None:
            job.set_stage("labeling", f"标注中 {done}/{total}", done=done, total=total)

        return label_dataset(self.root, settings, verbose=False, progress=progress,
                             only_missing=not bool(job.params.get("overwrite")),
                             limit=int(job.params.get("limit") or 0) or None,
                             only=job.params.get("only") or None,
                             should_stop=lambda: bool(job.cancel_requested))

    def _qc(self, job: Job) -> str:
        from .qc import write_report

        return str(write_report(self.root).get("report"))

    def _coords(self, job: Job) -> str:
        from .coords import write_coordinate_files

        return str(write_coordinate_files(self.root, tag=f"upload-{job.id}")["out"])

    # --- 提示词标注任务：图片早已入库，只是逐图问多模态大模型补检测框 ---
    def _run_prompt(self, job: Job) -> None:
        fields, profile = self._prompt_targets(job)
        names = "、".join(str(row.get("name")) for row in fields)
        total = len(job.files)
        job.set_stage("labeling",
                      f"用 {job.params.get('vlm_name') or profile.get('name')} 逐图找【{names}】，"
                      f"共 {total} 张（每张一次网络请求，比 YOLO 慢很多属正常）0/{total}",
                      done=0, total=total)

        def progress(done: int, total_: int) -> None:
            job.set_stage("labeling", f"提示词标注 {done}/{total_} · 字段 {names}",
                          done=done, total=total_)

        stats = self._prompt_label(job, fields, profile, progress)
        self._check(job)
        job.set_stage("coords", "生成坐标文件")
        coords = self._prompt_coords(job)
        protected = int(stats.get("protected") or 0)
        job.result = {"labeled": stats.get("labeled", 0), "boxes": stats.get("boxes", 0),
                      "failed": stats.get("failed", 0), "protected": protected,
                      "unknown_fields": stats.get("unknown_fields", []),
                      "coords": coords, "batch_id": job.batch_id, "fields": names}
        # 全是"人工保护"时只说"标注 0 张"会让人以为任务没跑起来，把原因写进完成消息
        done = f"完成：标注 {stats.get('labeled', 0)} 张，产出 {stats.get('boxes', 0)} 个框"
        if protected:
            done += f"；{protected} 张人工接受过或画过框的图未覆盖（勾「覆盖已有框」可重标）"
        job.set_stage("done", done)
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    def _prompt_targets(self, job: Job):
        """解析这次任务要标的字段与多模态配置（两个提示词任务共用）。"""
        from .fields import resolve_fields
        from .prompt import resolve_profile

        wanted = job.params.get("prompt_fields") or []
        if isinstance(wanted, str):
            wanted = [item.strip() for item in wanted.split(",") if item.strip()]
        fields = resolve_fields(self.root, list(wanted))
        profile = resolve_profile(str(job.params.get("vlm_profile") or ""))
        return fields, profile

    def _prompt_label(self, job: Job, fields, profile, progress,
                      images: Optional[List[str]] = None) -> dict:
        from .prompt import run_prompt_labeling

        # 提示词上传任务的 job.files 是 uploads/ 里的临时文件名，必须显式传入库后的相对路径
        targets = list(images) if images is not None else (list(job.files) or None)
        return run_prompt_labeling(
            self.root, fields=fields, profile=profile, images=targets,
            overwrite=bool(job.params.get("overwrite")),
            extra_hint=str(job.params.get("prompt_extra") or ""),
            progress=progress, should_stop=lambda: bool(job.cancel_requested), verbose=False)

    def _prompt_coords(self, job: Job) -> str:
        from .prompt import write_object_coordinate_files

        return str(write_object_coordinate_files(self.root, tag=f"prompt-{job.id}")["out"])

    def _run_pool(self, job: Job) -> None:
        """素材池导入：文件落进共享池就结束，不进库、不跑标注、不出坐标。"""
        job.set_stage("importing", f"导入 {len(job.files)} 个文件")
        ingested = self._ingest(job)
        self._check(job)
        self._cleanup(job)
        job.result = {"imported": ingested, "batch_id": job.batch_id}
        job.set_stage("done", f"已进池 {ingested} 张{job.ingest_note()}",
                     done=ingested, total=max(len(job.files), ingested))
        # finished / ok 必须在这儿落定：以前只写了 stage=done，任务条会一直转，
        # 「还在导入中」的保护还会把这批素材一直拦住不让删
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    def _run_import(self, job: Job) -> None:
        """上传任务：入库 → 姿态自动标注 → QC → 坐标导出。"""
        job.set_stage("importing", f"导入 {len(job.files)} 个文件")
        ingested = self._ingest(job)
        job.set_stage("importing", f"入库 {ingested} 张", done=0, total=ingested)

        self._check(job)
        if job.params.get("label", True):
            with self._loading(job, self._load_hint(job)):
                labeled = self._label(job)
            self._check(job)
            job.set_stage("qc", "生成质量报告")
            report = self._qc(job)
        else:
            labeled, report = {}, None

        self._check(job)
        job.set_stage("coords", "生成坐标文件")
        coords = self._coords(job)
        self._cleanup(job)
        job.result = {"imported": ingested, "labeled": labeled.get("labeled", 0),
                      "people": labeled.get("people", 0), "flagged": labeled.get("flagged", 0),
                      "report": report, "coords": coords, "batch_id": job.batch_id}
        job.set_stage("done", f"完成：入库 {ingested} 张{job.ingest_note()}，"
                             f"标注 {labeled.get('labeled', 0)} 张")
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    def _run_prompt_upload(self, job: Job) -> None:
        """提示词标注模块的上传：入库 →（可选）逐图问多模态大模型出框 → 写检测框坐标。

        和姿态页的上传互不打扰：这边不跑 YOLO-pose，产出的框进 ``prompt_labels/``。
        """
        job.set_stage("importing", f"导入 {len(job.files)} 个文件")
        ingested = self._ingest(job)
        rels = list(job.ingested)
        job.set_stage("importing", f"入库 {ingested} 张", done=0, total=len(rels))
        if not rels:
            self._cleanup(job)
            job.result = {"imported": 0, "labeled": 0, "boxes": 0, "batch_id": job.batch_id}
            job.set_stage("done", "完成：没有新图入库"
                         + (job.ingest_note() or "（可能与已有图片重复，或文件里没有可用图片）"))
            job.finished, job.ok = True, True
            finish_batch_any(self.root, job.batch_id, "done", job.message)
            return

        self._check(job)
        stats = {"labeled": 0, "boxes": 0, "failed": 0, "protected": 0, "unknown_fields": []}
        names = ""
        if job.params.get("label", True):
            fields, profile = self._prompt_targets(job)
            names = "、".join(str(row.get("name")) for row in fields)

            def progress(done: int, total: int) -> None:
                job.set_stage("labeling", f"提示词标注 {done}/{total} · 字段 {names}",
                              done=done, total=total)

            job.set_stage("labeling",
                          f"用 {job.params.get('vlm_name') or profile.get('name')} 逐图找【{names}】，"
                          f"共 {len(rels)} 张（每张一次网络请求，比 YOLO 慢很多属正常）0/{len(rels)}",
                          done=0, total=len(rels))
            stats = self._prompt_label(job, fields, profile, progress, images=rels)
            self._check(job)

        job.set_stage("coords", "生成检测框坐标文件")
        coords = self._prompt_coords(job)
        self._cleanup(job)
        job.result = {"imported": ingested, "labeled": stats.get("labeled", 0),
                      "boxes": stats.get("boxes", 0), "failed": stats.get("failed", 0),
                      "protected": stats.get("protected", 0),
                      "unknown_fields": stats.get("unknown_fields", []),
                      "coords": coords, "batch_id": job.batch_id, "fields": names}
        done = (f"完成：入库 {ingested} 张{job.ingest_note()}，"
                f"标注 {stats.get('labeled', 0)} 张，产出 {stats.get('boxes', 0)} 个框")
        if stats.get("failed"):
            done += f"（{stats['failed']} 张没答上）"
        job.set_stage("done", done)
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    # --- 目标检测任务：本地 YOLO 权重出它自带的类别，不联网、比多模态快一个数量级 ---
    def _detect_targets(self, job: Job):
        """这次任务用哪个权重、标哪些类别、跑在什么算力上。"""
        from . import detect
        from . import settings as system

        model = detect.resolve_model(str(job.params.get("model") or ""), self.root)
        wanted = job.params.get("detect_classes") or []
        if isinstance(wanted, str):
            wanted = [item.strip() for item in wanted.split(",") if item.strip()]
        device = str(job.params.get("device") or "") or system.label_defaults().get("device", "")
        conf, imgsz = job.params.get("conf"), job.params.get("imgsz")
        return {"model": model, "classes": [str(item) for item in wanted], "device": device,
                "conf": float(conf) if conf not in (None, "") else None,
                # 没显式指定就留空：让检测层去读「系统设置」，别在这里替它写死 640
                "imgsz": int(imgsz) if imgsz not in (None, "") else None}

    def _detect_label(self, job: Job, targets: dict, images: Optional[List[str]] = None) -> dict:
        from . import detect

        files = list(images) if images is not None else (list(job.files) or None)
        return detect.run_detect_labeling(
            self.root, model=targets["model"], classes=list(targets["classes"]), images=files,
            overwrite=bool(job.params.get("overwrite")), device=targets["device"],
            conf=targets["conf"], imgsz=targets["imgsz"],
            progress=self._progress_writer(job),
            should_stop=lambda: bool(job.cancel_requested), verbose=False)

    def _progress_writer(self, job: Job):
        def progress(done: int, total: int) -> None:
            job.set_stage("labeling", f"目标检测 {done}/{total}", done=done, total=total)

        return progress

    def _detect_load_hint(self, model: str) -> str:
        """检测权重的加载提示：模型既可能是裸文件名，也可能是用户给的绝对路径。"""
        from .backends import default_weights_dir

        path = model if os.path.isabs(model) else os.path.join(default_weights_dir(), model)
        try:
            size_mb = os.path.getsize(path) / 1048576
        except OSError:
            size_mb = 0.0
        return load_hint_text(os.path.basename(model), size_mb)

    def _detect_coords(self, job: Job) -> str:
        from .detect import write_detect_coordinate_files

        return str(write_detect_coordinate_files(self.root, tag=f"detect-{job.id}")["out"])

    def _detect_done_text(self, stats: dict, ingested: int = 0, note: str = "") -> str:
        """完成消息：把"多少张没覆盖""少掉的张哪去了"讲清楚，否则用户以为任务没跑。"""
        parts = [f"入库 {ingested} 张{note}"] if ingested else []
        parts.append(f"标注 {stats.get('labeled', 0)} 张，产出 {stats.get('boxes', 0)} 个框")
        if stats.get("protected"):
            parts.append(f"{stats['protected']} 张人工接受过或画过框的图未覆盖（勾「覆盖已有框」可重标）")
        if stats.get("failed"):
            parts.append(f"{stats['failed']} 张推理失败")
        if stats.get("missing_classes"):
            parts.append(f"该权重认不出 {'、'.join(str(x) for x in stats['missing_classes'])}")
        from . import detect as detect_module

        hint = detect_module.near_miss_hint(stats)
        if hint:
            parts.append(hint)
        return "完成：" + "；".join(parts)

    def _run_pose(self, job: Job) -> None:
        """已入库的图批量跑姿态标注：不导入、不重传，产出只进 ``labels/``。

        姿态页以前只有「导入素材」这一个能触发自动标注的口子：先勾「只入库不标注」
        传进来的、或换了一档更重的权重想重标的，就只能把素材整批重传一遍。
        这条把「已入库」和「要标注」拆开，和提示词标注 / 目标检测标注两页的批量面板同一口径。
        """
        total = len(job.files)
        job.ingested = list(job.files)        # 批次要能列出"这批标的是哪几张"
        self._attach_batch(job)
        with self._loading(job, f"{self._load_hint(job)}"
                                f"本库这批 {total} 张，参数没变且已标过的自动跳过"):
            labeled = self._label(job)
        self._check(job)
        job.set_stage("qc", "生成质量报告")
        report = self._qc(job)
        job.set_stage("coords", "生成坐标文件")
        coords = self._coords(job)
        job.result = {"labeled": labeled.get("labeled", 0), "people": labeled.get("people", 0),
                      "flagged": labeled.get("flagged", 0), "failed": labeled.get("failed", 0),
                      "skipped_human": labeled.get("skipped_human", 0), "report": report,
                      "coords": coords, "batch_id": job.batch_id,
                      "model": str(job.params.get("model") or "")}
        job.set_stage("done",
                      f"完成：标注 {labeled.get('labeled', 0)} 张 · {labeled.get('people', 0)} 个人"
                      + (f" · 人工已复核的 {labeled['skipped_human']} 张没动"
                         if labeled.get("skipped_human") else "")
                      + (f" · 失败 {labeled['failed']} 张" if labeled.get("failed") else ""))
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    def _run_detect(self, job: Job) -> None:
        """已入库的图批量跑目标检测：不导入、不跑姿态，产出只进 ``detect_labels/``。"""
        targets = self._detect_targets(job)
        model, classes = targets["model"], targets["classes"]
        total = len(job.files)
        names = "、".join(classes) or "（本库类别表里全部）"
        with self._loading(job, f"{self._detect_load_hint(model)} "
                                f"标注【{names}】共 {total} 张（本地推理，比多模态大模型快得多）"):
            stats = self._detect_label(job, targets)
        self._check(job)
        job.set_stage("coords", "生成坐标文件")
        coords = self._detect_coords(job)
        job.result = {"labeled": stats.get("labeled", 0), "boxes": stats.get("boxes", 0),
                      "failed": stats.get("failed", 0), "protected": stats.get("protected", 0),
                      "missing_classes": stats.get("missing_classes", []),
                      "model": stats.get("model", model), "coords": coords,
                      "batch_id": job.batch_id, "classes": names}
        job.set_stage("done", self._detect_done_text(stats))
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    def _run_detect_upload(self, job: Job) -> None:
        """目标检测标注模块的上传：入库 →（可选）本地权重出框 → 写检测框坐标。"""
        job.set_stage("importing", f"导入 {len(job.files)} 个文件")
        ingested = self._ingest(job)
        rels = list(job.ingested)
        job.set_stage("importing", f"入库 {ingested} 张", done=0, total=len(rels))
        if not rels:
            self._cleanup(job)
            job.result = {"imported": 0, "labeled": 0, "boxes": 0, "batch_id": job.batch_id}
            job.set_stage("done", "完成：没有新图入库"
                         + (job.ingest_note() or "（可能与已有图片重复，或文件里没有可用图片）"))
            job.finished, job.ok = True, True
            finish_batch_any(self.root, job.batch_id, "done", job.message)
            return

        self._check(job)
        stats = {"labeled": 0, "boxes": 0, "failed": 0, "protected": 0, "missing_classes": []}
        targets = self._detect_targets(job)
        model, classes = targets["model"], targets["classes"]
        names = "、".join(classes) or "（本库类别表里全部）"
        if job.params.get("label", True):
            def progress(done: int, total: int) -> None:
                job.set_stage("labeling", f"目标检测 {done}/{total} · {os.path.basename(model)}",
                              done=done, total=total)

            with self._loading(job, f"{self._detect_load_hint(model)} "
                                    f"标注【{names}】共 {len(rels)} 张（本地推理，比多模态大模型快得多）"):
                stats = self._detect_label(job, targets, images=rels)
            self._check(job)

        job.set_stage("coords", "生成检测框坐标文件")
        coords = self._detect_coords(job)
        self._cleanup(job)
        job.result = {"imported": ingested, "labeled": stats.get("labeled", 0),
                      "boxes": stats.get("boxes", 0), "failed": stats.get("failed", 0),
                      "protected": stats.get("protected", 0),
                      "missing_classes": stats.get("missing_classes", []),
                      "model": stats.get("model", model),
                      "coords": coords, "batch_id": job.batch_id, "classes": names}
        job.set_stage("done", self._detect_done_text(stats, ingested, job.ingest_note()))
        job.finished, job.ok = True, True
        finish_batch_any(self.root, job.batch_id, "done", job.message)

    def _cleanup(self, job: Job) -> None:
        if job.kind in ("prompt", "detect"):  # 只跑标注的任务没有临时上传文件要清
            return
        for name in job.files:
            try:
                os.remove(os.path.join(uploads_dir(self.root), name))
            except OSError:
                pass

    def _salvage(self, job: Job) -> dict:
        """任务取消后，仍把已经标好的部分写进坐标文件。"""
        kind = str(job.kind)
        try:
            if kind.startswith("detect"):
                coords = self._detect_coords(job)
            elif kind.startswith("prompt"):
                coords = self._prompt_coords(job)
            else:
                coords = self._coords(job)
            self._cleanup(job)
            return {"coords": coords, "cancelled": True}
        except Exception:  # noqa: BLE001 收尾失败不影响取消本身
            return {}

    def _notify(self) -> None:
        if self.on_finish:
            try:
                self.on_finish()
            except Exception:  # noqa: BLE001
                pass
