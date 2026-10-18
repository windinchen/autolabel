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
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .dataset import now_str
from .batches import attach_items, batch_name, create_batch, display_name, finish_batch

from .library import LIB_OBJECTS, LIB_POSE

UPLOAD_CHUNK = 1 << 20  # 上传落盘时的分块大小：1 MB
# 上次进程被打断时留下的上传临时文件，超过这个小时数就在启动时清掉
STALE_UPLOAD_HOURS = 24

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv", ".wmv",
              ".mts", ".m2ts", ".ts", ".mpg", ".mpeg", ".3gp"}
# 保留中文与常见安全字符，其余替换为下划线
_SAFE = re.compile(r"[^0-9A-Za-z._\u4e00-\u9fff-]+")

DEFAULT_MAX_UPLOAD_MB = 1024


class UploadTooLarge(ValueError):
    """上传体积超过单文件上限；请求体已经读干净，可以安全地回 413。"""


class JobCancelled(Exception):
    """任务被用户在界面上「清除」，用于从流水线阶段提前退出。"""


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


def lib_for_kind(kind: str) -> str:
    """任务类型 → 标注库：``prompt*`` 归提示词库，其余归姿态库。"""
    return LIB_OBJECTS if str(kind or "").startswith("prompt") else LIB_POSE


def uploads_dir(root: str) -> str:
    path = os.path.join(os.path.abspath(root), ".autolabel", "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def safe_upload_name(name: str) -> str:
    base = os.path.basename(str(name).replace("\\", "/"))
    base = _SAFE.sub("_", base).strip("._")
    return base[:120] or "file"


def kind_for(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return ""


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

    def payload(self) -> dict:
        return {"id": self.id, "kind": self.kind, "files": self.files[:6], "file_count": len(self.files),
                "stage": self.stage, "message": self.message, "done": self.done, "total": self.total,
                "finished": self.finished, "ok": self.ok, "result": self.result, "error": self.error,
                "cancel_requested": self.cancel_requested,
                "batch_id": self.batch_id,
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
        if kind not in ("image", "video", "prompt", "prompt_upload"):
            raise ValueError(f"未知任务类型: {kind}")
        if kind == "prompt":
            # 提示词标注跑的是已入库的图片，files 是 images/ 下的相对路径，不能被 basename 削掉子目录
            files = [str(f).replace("\\", "/") for f in files if str(f).strip()]
            files = self._known_images(files, lib_for_kind(kind))
        else:
            files = [os.path.basename(f) for f in files if f]
        if not files:
            raise ValueError("没有收到文件")
        if kind != "prompt":
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
            print(f"[warn] 提示词任务跳过不存在的图片 {len(missing)} 张，例如 {missing[0]}")
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
            else:
                self._run_import(job)
        except JobCancelled as exc:
            self._attach_batch(job)
            finish_batch(self.root, job.batch_id, "cancelled", f"{exc or '任务已取消'}，已完成的标注保留")
            salvaged = self._salvage(job)
            job.result = salvaged
            job.set_stage("cancelled", f"{exc or '任务已取消'}，已完成的标注与坐标已保留")
            job.finished, job.ok = True, False
        except Exception as exc:  # noqa: BLE001 后台线程必须兜住所有异常
            job.error = f"{type(exc).__name__}: {exc}"
            job.set_stage("failed", job.error)
            job.finished, job.ok = True, False
            finish_batch(self.root, job.batch_id, "failed", job.error)
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
            return f"加载模型 {name}…"
        note = f"已下载 {info['size_mb']} MB" if info["size_mb"] else "尚未下载"
        return f"首次加载 {name}：正在下载权重（{note}，视网络可能要几分钟）…"

    def _check(self, job: Job) -> None:
        """阶段边界检查：收到清除请求就抛出 :class:`JobCancelled`。"""
        if job.cancel_requested:
            raise JobCancelled("任务已取消")

    def _start_batch(self, job: Job) -> None:
        params = job.params
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
                    "overwrite": bool(params.get("overwrite")),
                    "prompt_extra": params.get("prompt_extra") or "",
                    "targets": len(job.files)},
            uploads=job.files)
        job.batch_id = batch["id"]

    def _attach_batch(self, job: Job) -> None:
        """把已入库的图片挂到批次上（取消时也只挂实际落盘的那部分）。"""
        if job.batch_id and job.ingested:
            attach_items(self.root, job.batch_id, job.ingested, lib=lib_for_kind(job.kind))

    def _ingest(self, job: Job) -> int:
        from .dataset import import_images
        from .video import extract_frames

        paths = [os.path.join(uploads_dir(self.root), name) for name in job.files]
        if job.kind == "video":
            imported = 0
            for path in paths:
                self._check(job)
                job.set_stage("importing", f"抽帧 {os.path.basename(path)}")
                result = extract_frames(path, self.root, fps=float(job.params.get("fps", 0.5)),
                                        max_frames=int(job.params.get("max_frames", 600)),
                                        scene=job.params.get("scene"),
                                        max_side=int(job.params.get("max_side", 1600)),
                                        dedupe=str(job.params.get("video_dedupe", "perceptual")),
                                        lib=lib_for_kind(job.kind))
                imported += int(result["imported"]) + int(result.get("reused") or 0)
                job.ingested.extend(rel for _src, rel in result.get("imported_files", []))
            self._attach_batch(job)
            return imported
        stats = import_images(self.root, paths, max_side=int(job.params.get("max_side", 1600)),
                              dedupe=str(job.params.get("dedupe", "exact")),
                              lib=lib_for_kind(job.kind))
        job.ingested.extend(rel for _src, rel in stats.get("imported_files", []))
        self._attach_batch(job)
        return (int(stats["imported"]) + int(stats["reused"])
                + int(stats["skipped_exact"]) + int(stats["skipped_similar"]))

    def _label(self, job: Job) -> dict:
        from .backends import BackendSettings
        from .pipeline import label_dataset

        fields = {k: v for k, v in job.params.items()
                  if k in BackendSettings.__dataclass_fields__ and v is not None}
        if isinstance(fields.get("extra_models"), str):
            fields["extra_models"] = tuple(m.strip() for m in fields["extra_models"].split(",") if m.strip())
        settings = BackendSettings(**fields)

        def progress(done: int, total: int) -> None:
            job.set_stage("labeling", f"标注中 {done}/{total}", done=done, total=total)

        return label_dataset(self.root, settings, verbose=False, progress=progress,
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
        finish_batch(self.root, job.batch_id, "done", job.message)

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

    def _run_import(self, job: Job) -> None:
        """上传任务：入库 → 姿态自动标注 → QC → 坐标导出。"""
        job.set_stage("importing", f"导入 {len(job.files)} 个文件")
        ingested = self._ingest(job)
        job.set_stage("importing", f"入库 {ingested} 张", done=0, total=ingested)

        self._check(job)
        if job.params.get("label", True):
            job.set_stage("labeling", self._load_hint(job))
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
        job.set_stage("done", f"完成：入库 {ingested} 张，标注 {labeled.get('labeled', 0)} 张")
        job.finished, job.ok = True, True
        finish_batch(self.root, job.batch_id, "done", job.message)

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
            job.set_stage("done", "完成：没有新图入库（可能与已有图片重复，或文件里没有可用图片）")
            job.finished, job.ok = True, True
            finish_batch(self.root, job.batch_id, "done", job.message)
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
        done = f"完成：入库 {ingested} 张，标注 {stats.get('labeled', 0)} 张，产出 {stats.get('boxes', 0)} 个框"
        if stats.get("failed"):
            done += f"（{stats['failed']} 张没答上）"
        job.set_stage("done", done)
        job.finished, job.ok = True, True
        finish_batch(self.root, job.batch_id, "done", job.message)

    def _cleanup(self, job: Job) -> None:
        if job.kind == "prompt":  # 提示词任务没有临时上传文件要清
            return
        for name in job.files:
            try:
                os.remove(os.path.join(uploads_dir(self.root), name))
            except OSError:
                pass

    def _salvage(self, job: Job) -> dict:
        """任务取消后，仍把已经标好的部分写进坐标文件。"""
        prompt_kind = job.kind.startswith("prompt")
        try:
            coords = self._prompt_coords(job) if prompt_kind else self._coords(job)
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
