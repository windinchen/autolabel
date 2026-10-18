"""本地人工复核服务：浏览器里拖拽关键点、改可见性、接受/驳回，直接落 YOLO 标签。

只依赖标准库 HTTP 服务，前端为 ``autolabel/static/review.html``（无构建步骤）。
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, List, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:  # 仅类型标注使用，避免导入期拉起推理依赖
    from .upload import JobStore

from .dataset import (
    STATUS_APPROVED,
    STATUS_EDITED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    DatasetIndex,
    adopt_from_pool,
    label_path_for,
    load_index,
    now_str,
    pool_items,
    resolve_image,
)
from .library import (
    LIBRARIES,
    LIB_NAMES,
    LIB_OBJECTS,
    LIB_POSE,
    backups_dir,
    history_root,
    lib_name,
    migrate,
    normalize,
)
from .models import (
    delete_profile,
    delete_weight,
    download_state,
    forget_model,
    list_weights,
    list_models,
    register_local_model,
    set_model_default,
    set_model_note,
    set_default,
    start_download,
    test_saved_profile,
    upsert_profile,
    vlm_payload,
    vlm_choices,
)
from .pipeline import image_level_flags
from .thumbs import THUMB_CACHE_CONTROL, build_thumbnail, prune_thumbs
from .qc import COCO17_SKELETON, FLAG_HINTS, LOW_SCORE, POSITIVE_FLAGS, dataset_stats, review_counts, \
    review_priority
from .batches import (
    MODULE_NAMES,
    batch_file_rows,
    delete_images,
    list_batches,
    reconcile_with_disk,
)
from .yolo_pose import COCO17_KEYPOINTS, Keypoint, Person, parse_label_file, write_label_file
from .dataset import index_path, prompt_label_path_for
from .fields import (
    delete_field,
    fields_payload as fields_payload_for,
    list_fields,
    resolve_fields as resolve_prompt_fields,
    upsert_field,
)
from .prompt import (
    PROMPT_FLAG_HINTS,
    Detection,
    apply_manual_marks,
    box_flags,
    image_flags,
    prompt_priority,
    prompt_summary,
    read_detections,
    resolve_profile,
    write_detections,
)

VALID_STATUS = {STATUS_UNREVIEWED, STATUS_APPROVED, STATUS_EDITED, STATUS_REJECTED}


def module_summary(rows: List[dict]) -> dict:
    """按二级模块（= 标注库）汇总批次规模：门户工作台与 ``/api/batches`` 同一套口径。"""
    out: dict = {}
    for name, label in MODULE_NAMES.items():
        group = [row for row in rows if row.get("module") == name]
        out[name] = {
            "label": label, "batches": len(group),
            "items": sum(int(row.get("items") or 0) for row in group),
            "labeled": sum(int(row.get("labeled") or 0) for row in group),
            "prompt_labeled": sum(int(row.get("prompt_labeled") or 0) for row in group),
            "prompt_boxes": sum(int(row.get("prompt_boxes") or 0) for row in group),
        }
    return out
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

HISTORY_LIMIT = 10


class ReviewStore:
    """复核数据读写：标签文件 + 索引 + 原始备份/历史快照。"""

    def __init__(self, root: str, jobs: Optional[JobStore] = None):
        self.root = os.path.abspath(root)
        # 可重入：改索引时要在同一把锁里再走一遍「重新对齐磁盘」（见 ``_mutate``），
        # 用普通 Lock 会自己把自己锁死
        self.lock = threading.RLock()
        # 老布局（两套状态混在一份索引里）先迁成分库布局，重复调用无副作用
        self.migration = migrate(self.root)
        # 先按磁盘对账，再读索引：上次跑到一半被打断的导入不能悄悄丢掉素材
        self.recovery = reconcile_with_disk(self.root)
        # 姿态库与提示词库各自一份索引：删图只摘本库引用，统计也不再互相掺水
        self.index: DatasetIndex = load_index(self.root, LIB_POSE)
        self.objects_index: DatasetIndex = load_index(self.root, LIB_OBJECTS)
        self._index_stamp = self._disk_stamp()
        if jobs is None:
            from .upload import JobStore as _JobStore

            jobs = _JobStore(self.root, on_finish=self.reload)
        self.jobs = jobs

    def _disk_stamp(self):
        """两个库索引文件的（修改时间, 大小）快照，用来判断磁盘是不是被别人动过。"""
        stamps = []
        for lib in LIBRARIES:
            try:
                stat = os.stat(index_path(self.root, lib))
                stamps.append((lib, stat.st_mtime_ns, stat.st_size))
            except OSError:
                stamps.append((lib, None, None))
        return tuple(stamps)

    def _refresh_if_changed(self) -> None:
        """磁盘索引变了就重读：命令行 `autolabel import`、外部脚本改过盘之后，
        不用重启服务、也不用先点一下界面，新图就该出现在列表里（实测踩过：
        命令行导完 4 张，门户一直显示 0 张）。只在文件真的变过时才解析。"""
        stamp = self._disk_stamp()
        with self.lock:
            if stamp == self._index_stamp:
                return
            self._reload_locked()
            self._index_stamp = stamp

    def reload(self) -> dict:
        """重新读取两个库的磁盘索引：上传/后台标注写入的新图要能立刻出现在界面里。"""
        with self.lock:
            self._reload_locked()
            self._index_stamp = self._disk_stamp()
        return {"ok": True, "images": len(self.index.items),
                "object_images": len(self.objects_index.items)}

    def _reload_locked(self) -> None:
        """调用方必须已经握着 ``self.lock``。"""
        for lib in LIBRARIES:
            idx = load_index(self.root, lib)
            if lib == LIB_POSE:
                self.index = idx
            else:
                self.objects_index = idx

    def _sync_locked(self) -> None:
        """按磁盘重建内存索引并记下时间戳；调用方必须已经握着 ``self.lock``。"""
        self._reload_locked()
        self._index_stamp = self._disk_stamp()

    @contextmanager
    def _mutate(self, resync: bool = False):
        """改索引的标准姿势：一把锁里完成「对齐磁盘 → 改 → 落盘」。

        以前的写法是 ``self.reload()``（拿锁再放锁）之后再来一句
        ``with self.lock:`` 改+存。两次拿锁之间的空隙里，另一个页面或后台线程
        可能已经改过索引，手里的内存副本就成了旧值——整份文件按旧副本写回去，
        对方刚删掉/刚标的记录会一起被抹掉（索引是整表覆盖写，没有字段级合并）。
        删除走的也是整表覆盖，所以删除必须挤进同一把锁，否则「删 20 张、
        刷新又全回来」。

        :param resync: 块内改用 batches/dataset 的助手直接落盘时置 True，
            收尾再按磁盘对齐一次——内存索引不知道别人刚删了什么
        """
        with self.lock:
            self._sync_locked()
            try:
                yield self
            finally:
                if resync:
                    self._sync_locked()
                else:
                    # 块里刚写过盘，时间戳必须跟上，否则下一次读会白解析一遍
                    self._index_stamp = self._disk_stamp()

    # --- 读取 ---
    def dataset_payload(self) -> dict:
        self._refresh_if_changed()
        stats = dataset_stats(self.root)
        batch_rows = list_batches(self.root)
        items = []
        for rel, item in self.index.ordered_items():
            flags = sorted(set(list(item.get("flags", [])) + list(item.get("object_flags", []))))
            items.append({
                "image": rel,
                "width": item.get("width"),
                "height": item.get("height"),
                "status": item.get("status", STATUS_UNREVIEWED),
                "objects": item.get("objects"),
                "score": item.get("score"),
                "visible_ratio": item.get("visible_ratio"),
                "flags": flags,
                "priority": review_priority(item),
                "time_sec": item.get("time_sec"),
                "video": os.path.basename(item["video"]) if item.get("video") else None,
                "batch_id": item.get("batch_id"),
            })
        object_items = []
        for rel, item in self.objects_index.ordered_items():
            # 行里只留 prompt_* 这一套键：界面读的本来就是这套，以前同时给一份
            # status/objects/score/flags/priority 的重复拷贝，两万张的库白胖一截
            object_items.append({
                "image": rel,
                "width": item.get("width"),
                "height": item.get("height"),
                "time_sec": item.get("time_sec"),
                "video": os.path.basename(item["video"]) if item.get("video") else None,
                "batch_id": item.get("batch_id"),
                "prompt_objects": item.get("prompt_objects"),
                "prompt_status": item.get("prompt_status", STATUS_UNREVIEWED),
                "prompt_flags": sorted(set(item.get("prompt_flags") or [])),
                "prompt_score": item.get("prompt_score"),
                "prompt_priority": prompt_priority(item),
            })
        # 「等你复核」由后端一次算清：门户 / 命令行 / 两个复核页共用同一个口径。
        # 以前门户 JS 自己数一遍，姿态库数"所有没定论的"、提示词库数"带标记的"，
        # 同一个数字两套算法；正向标记（多模型结论一致）还被前端当成待确认。
        review = review_counts(items, object_items)
        # 门户要一眼看清"池里有多少素材、每个库各引用了多少"
        pool_rows = pool_items(self.root)
        pool_stats = {
            "files": len(pool_rows),
            "bytes": sum(int(row.get("bytes") or 0) for row in pool_rows),
            "pose": sum(1 for row in pool_rows if row["in_pose"]),
            "objects": sum(1 for row in pool_rows if row["in_objects"]),
            "shared": sum(1 for row in pool_rows if row["in_pose"] and row["in_objects"]),
            "orphan": sum(1 for row in pool_rows if not row["in_pose"] and not row["in_objects"]),
        }
        num_kpt = int(self.index.data.get("kpt_shape", [len(COCO17_KEYPOINTS), 3])[0])
        return {
            "name": self.index.name,
            "root": self.root,
            # 同一份姿态清单以前在 items 与 pose_items 里各序列化一遍：1500 张的
            # 数据集响应体 720 KB，其中 305 KB 纯属第二份拷贝。界面两处写的都是
            # ``ds.pose_items || ds.items``，去掉重复键不用动前端。
            "items": items,
            "object_items": object_items,
            "class_names": {str(k): v for k, v in self.index.class_names.items()},
            "keypoint_names": list(self.index.data.get("keypoint_names", COCO17_KEYPOINTS))[:num_kpt],
            "skeleton": [[a, b] for a, b in COCO17_SKELETON if max(a, b) < num_kpt],
            "num_keypoints": num_kpt,
            "stats": {k: stats[k] for k in ("images", "persons_total", "status_counts", "flag_counts",
                                            "box_height_px", "persons_per_image")},
            "flag_hints": FLAG_HINTS,
            "review": review,
            # 前端判断"这帧要不要人看"要用同一条线：正向标记清单 + 置信度阈值
            "positive_flags": sorted(POSITIVE_FLAGS),
            "low_score": LOW_SCORE,
            "models": list_weights(),
            "vlm": vlm_choices(),
            "model_rows": list_models(dataset_root=self.root),
            "settings": self.index.data.get("settings", {}),
            "batches": batch_rows,
            "by_module": module_summary(batch_rows),
            "libs": [{"lib": lib, "name": LIB_NAMES[lib],
                      "items": len(self.index.items if lib == LIB_POSE
                                   else self.objects_index.items)} for lib in LIBRARIES],
            "pool": pool_stats,
            "pool_files": pool_rows[:400],
            "exports_recent": self.exports_recent(),
            "fields": fields_payload_for(self.root),
            "prompt": prompt_summary(self.root),
            "recovery": self.recovery,
        }

    def read_item(self, image_rel: str) -> dict:
        item = self.index.get(image_rel)
        if item is None:
            raise KeyError(image_rel)
        label = label_path_for(self.root, image_rel)
        persons = parse_label_file(label) if os.path.exists(label) else []
        return {
            "image": image_rel,
            "width": item.get("width"),
            "height": item.get("height"),
            "status": item.get("status", STATUS_UNREVIEWED),
            "flags": sorted(set(list(item.get("flags", [])) + list(item.get("object_flags", [])))),
            "time_sec": item.get("time_sec"),
            "video": item.get("video"),
            "batch_id": item.get("batch_id"),
            "objects": [self._person_payload(p) for p in persons],
            "has_backup": self.undoable(image_rel),
        }

    @staticmethod
    def _person_payload(p: Person) -> dict:
        return {"cls": p.cls, "cx": p.cx, "cy": p.cy, "w": p.w, "h": p.h, "score": p.score,
                "flags": list(p.flags),
                "keypoints": [{"x": k.x, "y": k.y, "v": int(k.v), "conf": round(k.conf, 3)}
                              for k in p.keypoints]}

    # --- 写入 ---
    def write_item(self, image_rel: str, objects: List[dict], status: Optional[str] = None) -> dict:
        # 后台标注可能刚写入新图/新字段，锁内先对齐磁盘状态再改
        with self._mutate():
            item = self.index.get(image_rel)
            if item is None:
                raise KeyError(image_rel)
            item = dict(item)
            h = int(item.get("height") or 0)
            w = int(item.get("width") or 0)
            persons = persons_from_payload(objects, w, h)
            label = label_path_for(self.root, image_rel)
            self._backup(image_rel, label)
            write_label_file(label, persons)
            flags = image_level_flags(h, w, persons)
            # YOLO 标签本身不存模型置信度：人工改过之后沿用索引里的历史分值，
            # 否则复核过的图会因为 score=0 被重新判定为“低置信待复核”。
            scores = [p.score for p in persons if p.score]
            score = round(sum(scores) / len(scores), 4) if scores else float(item.get("score") or 0.0)
            fields = dict(objects=len(persons), score=score,
                          visible_ratio=_visible_ratio(persons),
                          flags=[f for f in flags if not f.startswith("obj:")],
                          object_flags=sorted({fl for p in persons for fl in p.flags}),
                          reviewed_at=now_str())
            if status in VALID_STATUS:
                fields["status"] = status
            elif item.get("status") not in (STATUS_APPROVED, STATUS_REJECTED):
                fields["status"] = STATUS_EDITED
            self.index.upsert(image_rel, **fields)
            self.index.save()
        return {"ok": True, "image": image_rel, **fields,
                "objects": [self._person_payload(p) for p in persons]}

    def set_status(self, image_rel: str, status: str) -> dict:
        if status not in VALID_STATUS:
            raise ValueError(f"非法状态: {status}")
        with self._mutate():
            self.index.upsert(image_rel, status=status, reviewed_at=now_str())
            self.index.save()
        return {"ok": True, "image": image_rel, "status": status}

    def pool_payload(self, lib: str = "") -> dict:
        """共享素材池清单：每张图被哪些库引用，供「从素材池添加」和删除确认用。"""
        rows = pool_items(self.root)
        wanted = normalize(lib) if lib else ""
        return {
            "lib": wanted,
            "lib_name": lib_name(wanted) if wanted else "全部",
            "files": [row for row in rows
                      if not wanted or not row.get(f"in_{wanted}")],
            "stats": {"files": len(rows),
                      "pose": sum(1 for row in rows if row["in_pose"]),
                      "objects": sum(1 for row in rows if row["in_objects"]),
                      "shared": sum(1 for row in rows if row["in_pose"] and row["in_objects"]),
                      "bytes": sum(int(row.get("bytes") or 0) for row in rows)},
            "libs": [{"lib": lib_key, "name": LIB_NAMES[lib_key]} for lib_key in LIBRARIES],
        }

    def adopt_pool(self, image_rels: List[str], lib: str = LIB_POSE,
                   batch_name_text: str = "") -> dict:
        """把素材池里已有的图片添加到某个库，并登记成一个批次（不复制文件）。"""
        from .batches import attach_items, batch_name as make_batch_name, create_batch, finish_batch

        lib = normalize(lib)
        rels = [str(rel) for rel in (image_rels or []) if str(rel).strip()]
        if not rels:
            raise ValueError("没有选择要添加的图片")
        with self._mutate(resync=True):
            batch = create_batch(self.root, name=make_batch_name("pool", [
                os.path.basename(rel) for rel in rels], batch_name_text),
                kind="pool", source="ui", uploads=[os.path.basename(rel) for rel in rels],
                lib=lib)
            stats = adopt_from_pool(self.root, rels, lib)
            if stats["added_items"]:
                attach_items(self.root, batch["id"], stats["added_items"], lib=lib)
            finish_batch(self.root, batch["id"], "done",
                         f"从素材池添加 {stats['added']} 张"
                         + (f"，已在库中 {stats['already']} 张" if stats["already"] else "")
                         + (f"，池里查无 {len(stats['missing'])} 张" if stats["missing"] else ""))
        return {"ok": True, "batch_id": batch["id"], "lib": lib, **stats}

    def batches_payload(self) -> dict:
        """批次列表 + 按二级模块的汇总，门户的批次管理一次取全。"""
        rows = list_batches(self.root)
        return {"batches": rows, "by_module": module_summary(rows),
                "module_names": dict(MODULE_NAMES)}

    def batch_files(self, batch_id: str) -> dict:
        """某批次下的单张清单（点开批次看文件、按张删时用）。"""
        if not str(batch_id or "").strip():
            raise ValueError("没有指定批次")
        return {"batch_id": batch_id, "files": batch_file_rows(self.root, str(batch_id))}

    def delete_images(self, image_rels: List[str], keep_files: bool = False,
                      lib: str = LIB_POSE, mode: str = "lib") -> dict:
        """按张删除图片与标注，正在标注的批次拒绝。

        :param lib: 从哪个库删（``pose`` / ``objects``）
        :param mode: ``lib`` 只摘本库引用（素材留在池里）；``purge`` 两库都摘并删素材文件
        """
        live = {str(job.get("batch_id")) for job in self.jobs.snapshot()
                if not job.get("finished") and job.get("batch_id")}
        with self._mutate(resync=True):
            result = delete_images(self.root, image_rels, lib=lib, mode=mode,
                                   keep_files=keep_files, active_ids=live)
        return result

    def run_prompt_upload_job(self, payload: dict) -> dict:
        """提示词标注模块自己上传素材：入库后直接跑提示词标注，登记成提示词模块的批次。"""
        payload = payload or {}
        wanted = payload.get("fields") or []
        if isinstance(wanted, str):
            wanted = [item.strip() for item in wanted.split(",") if item.strip()]
        files = [str(name) for name in (payload.get("files") or []) if str(name).strip()]
        if not files:
            raise ValueError("没有收到文件，先选择图片或视频")
        # 只入库不标注时不必校验字段与多模态配置，免得连字段都还没建就不能传素材
        if bool(payload.get("label", True)):
            fields = resolve_prompt_fields(self.root, wanted)
            profile = resolve_profile(str(payload.get("vlm_profile") or ""))
        else:
            fields, profile = [], None
        params = {
            "prompt_fields": [field.get("name") for field in fields],
            "prompt_extra": str(payload.get("prompt_extra") or ""),
            "label": bool(payload.get("label", True)),
            "fps": float(payload.get("fps") or 0.5),
            "max_frames": int(payload.get("max_frames") or 300),
            "max_side": int(payload.get("max_side") or 1600),
            "dedupe": str(payload.get("dedupe") or "exact"),
            "video_dedupe": str(payload.get("video_dedupe") or "perceptual"),
            "batch_name": str(payload.get("batch_name") or "")[:80],
        }
        if profile is not None:
            params["vlm_profile"] = profile["id"]
        job = self.jobs.submit("prompt_upload", files, params)
        return job.payload()

    def delete_batches(self, batch_ids: List[str], keep_files: bool = False) -> dict:
        """按批次删除图片与记录；与人工写入共用同一把锁，避免边删边改。"""
        from .batches import delete_batches as _delete_batches

        # 只有真正在跑的任务才算「标注中」；服务重启留下的僵尸批次允许直接删
        live = {str(job.get("batch_id")) for job in self.jobs.snapshot()
                if not job.get("finished") and job.get("batch_id")}
        with self._mutate(resync=True):
            result = _delete_batches(self.root, batch_ids, keep_files=keep_files, active_ids=live)
        return result

    # --- 提示词标注：字段表 + 检测框读写 ---
    def fields_payload(self) -> dict:
        """字段登记表 + 每个字段的已标框数。"""
        return fields_payload_for(self.root)

    def save_field(self, payload: dict) -> dict:
        """新建/更新一个字段（英文类名 + 中文显示名 + 描述提示词）。"""
        payload = payload or {}
        return upsert_field(self.root, name=str(payload.get("name") or ""),
                            label=str(payload.get("label") or ""),
                            prompt_hint=str(payload.get("prompt_hint") or ""),
                            field_id=str(payload.get("id") or ""),
                            color=str(payload.get("color") or ""))

    def remove_field(self, wanted: str, drop_boxes: bool = False) -> dict:
        # drop_boxes 会把该字段的框从所有图里抹掉并重写提示词索引，同样要进锁
        with self._mutate(resync=True):
            return delete_field(self.root, wanted, drop_boxes=drop_boxes)

    def prompt_summary_payload(self) -> dict:
        summary = prompt_summary(self.root)
        return {"prompt": summary, "fields": fields_payload_for(self.root),
                "flag_hints": PROMPT_FLAG_HINTS}

    def read_prompt_item(self, image_rel: str) -> dict:
        """读一张图的检测框，附带像素坐标与逐框质量标记。"""
        item = self.objects_index.get(image_rel)
        if item is None:
            raise KeyError(image_rel)
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        detections = apply_manual_marks(read_detections(self.root, image_rel),
                                       item.get("prompt_manual_boxes") or [])
        known = [int(field.get("class_id", -1)) for field in list_fields(self.root)]
        per_box = box_flags(detections, width, height, known)
        return {
            "image": image_rel, "width": width, "height": height,
            "status": item.get("prompt_status", STATUS_UNREVIEWED),
            "flags": sorted(set(item.get("prompt_flags") or [])),
            "score": item.get("prompt_score"), "time_sec": item.get("time_sec"),
            "video": item.get("video"), "batch_id": item.get("batch_id"),
            "signature": item.get("prompt_signature") or "",
            "stale": bool("prompt_stale" in (item.get("prompt_flags") or [])),
            "error": item.get("prompt_error") or "",
            "objects": [dict(detection.payload(width, height), flags=per_box[index])
                        for index, detection in enumerate(detections)],
            "has_backup": self.undoable(image_rel, "prompt"),
        }

    def write_prompt_item(self, image_rel: str, objects: List[dict],
                          status: Optional[str] = None) -> dict:
        """把前端画好的框写回 ``prompt_labels/``（YOLO detection 5 列）。"""
        with self._mutate():
            item = self.objects_index.get(image_rel)
            if item is None:
                raise KeyError(image_rel)
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            registry = {int(field.get("class_id", -1)): field for field in list_fields(self.root)}
            detections: List[Detection] = []
            for raw in objects or []:
                cls = int(raw.get("cls", -1))
                if cls not in registry:
                    names = "、".join(field.get("name", "") for field in registry.values()) or "（还没有字段）"
                    raise ValueError(f"类别号 {cls} 不在字段表里；当前可用类别：{names}")
                detection = Detection(cls=cls, cx=_clamp01(float(raw.get("cx", 0.5))),
                                      cy=_clamp01(float(raw.get("cy", 0.5))),
                                      w=max(0.0, min(1.0, float(raw.get("w", 0.0)))),
                                      h=max(0.0, min(1.0, float(raw.get("h", 0.0)))),
                                      score=float(raw.get("score") or 0.0),
                                      flags=[str(f) for f in (raw.get("flags") or [])])
                if detection.w <= 0 or detection.h <= 0:
                    raise ValueError("框的宽高必须大于 0")
                detections.append(detection)
            per_box = box_flags(detections, width, height, list(registry))
            manual_indices = [index for index, raw in enumerate(objects or [])
                              if "manual" in (raw.get("flags") or [])]
            scores = [detection.score for detection in detections]
            self._backup(image_rel, prompt_label_path_for(self.root, image_rel),
                         backup_path=self._prompt_backup_path(image_rel),
                         history_dir=self._prompt_history_dir(image_rel))
            write_detections(self.root, image_rel, detections)
            flags = image_flags(detections, per_box)
            if "prompt_stale" in (item.get("prompt_flags") or []):
                flags = sorted(set(flags) | {"prompt_stale"})
            new_status = status if status in VALID_STATUS else (
                item.get("prompt_status") if item.get("prompt_status") in (STATUS_APPROVED, STATUS_REJECTED)
                else STATUS_EDITED)
            fields_written = dict(prompt_objects=len(detections), prompt_status=new_status,
                                  prompt_flags=flags, prompt_manual=len(manual_indices),
                                  prompt_manual_boxes=manual_indices, prompt_error="",
                                  prompt_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
                                  prompt_classes=sorted({detection.cls for detection in detections}),
                                  prompt_reviewed_at=now_str())
            self.objects_index.upsert(image_rel, **fields_written)
            self.objects_index.save()
        return {"ok": True, "image": image_rel, **fields_written,
                "objects": [dict(detection.payload(width, height), flags=per_box[index])
                            for index, detection in enumerate(detections)]}

    def set_prompt_status(self, image_rel: str, status: str) -> dict:
        if status not in VALID_STATUS:
            raise ValueError(f"非法状态: {status}")
        with self._mutate():
            self.objects_index.upsert(image_rel, prompt_status=status,
                                      prompt_reviewed_at=now_str())
            self.objects_index.save()
        return {"ok": True, "image": image_rel, "prompt_status": status}

    def run_prompt_job(self, payload: dict) -> dict:
        """提交一次批量提示词标注任务（走任务队列，可在界面上「清除任务」）。"""
        payload = payload or {}
        wanted = payload.get("fields") or []
        if isinstance(wanted, str):
            wanted = [item.strip() for item in wanted.split(",") if item.strip()]
        fields = resolve_prompt_fields(self.root, wanted)
        images = [str(rel) for rel in (payload.get("images") or []) if str(rel).strip()]
        if not images:
            images = [rel for rel, _ in self.objects_index.ordered_items()]
        limit = int(payload.get("limit") or 0)
        if limit > 0:
            images = images[:limit]
        if not images:
            raise ValueError("数据集里还没有图片，先在「姿态标注」页导入素材")
        params = {"prompt_fields": [field.get("name") for field in fields],
                  "overwrite": bool(payload.get("overwrite")),
                  "prompt_extra": str(payload.get("prompt_extra") or "")}
        if payload.get("vlm_profile"):
            params["vlm_profile"] = str(payload.get("vlm_profile"))
        job = self.jobs.submit("prompt", images, params)
        return job.payload()

    def exports_recent(self, limit: int = 8) -> List[dict]:
        """`exports/` 里最近几份导出结果，界面用它把产物路径常驻显示出来。"""
        from .export import list_exports

        return list_exports(self.root, limit=limit)

    def export_pose_dataset(self, payload: dict) -> dict:
        """导出 YOLO pose 标准训练集：`images/{train,val}` + `labels/{train,val}` + `data.yaml`。

        ``scope`` 决定收哪些状态的图：`reviewed`（默认，剔掉驳回的）、
        `approved`（只要人工接受过的）、`all`（全都要，含未复核与驳回）。
        """
        from .export import export_dataset

        payload = payload or {}
        scope = str(payload.get("scope") or "reviewed")
        table = {"reviewed": ("rejected",), "approved": ("rejected", "unreviewed"), "all": ()}
        if scope not in table:
            raise ValueError(f"未知的导出范围：{scope}（可选 reviewed / approved / all）")
        # 参数先校验再查库存：用户填错 1.5 时，报告"验证集比例不能大于 1"才有用，
        # 报"库里还没标签"会把真正写错的参数盖掉（实测踩过空库 + val_ratio=1.5）。
        val_ratio = self._number(payload, "val_ratio", 0.2, 0.0, 1.0, "验证集比例")
        seed = int(self._number(payload, "seed", 7, None, None, "随机种子", integer=True))
        max_empty_ratio = self._number(payload, "max_empty_ratio", 0.15, 0.0, 1.0, "背景图比例上限")
        mode = str(payload.get("mode") or "symlink")
        ready = self.pose_labelled_count(table[scope])
        if not ready:
            raise ValueError(self.no_pose_label_hint())
        result = export_dataset(
            self.root, out=str(payload.get("out") or "") or None,
            val_ratio=val_ratio, seed=seed, exclude_status=table[scope],
            max_empty_ratio=max_empty_ratio, mode=mode,
            tag=str(payload.get("tag") or "") or None,
            coco=bool(payload.get("coco")),
            coords=bool(payload.get("coords")))
        result["scope"] = scope
        result["data_yaml"] = os.path.join(result["out"], "data.yaml")
        return result

    def version_payload(self, force: bool = False) -> dict:
        """顶栏版本徽标与更新提示的数据：全部出自 :mod:`autolabel.version`。"""
        from . import version as versioning

        return versioning.status(force=force)

    def no_pose_label_hint(self) -> str:
        """「整库没标注」和「这个范围里没带标签的帧」出路不同，别给同一句提示。

        库里一张标签都没有时，让人「点成接受」是白指路——接受不会凭空长出骨骼；
        反过来库里明明有标签、只是被范围滤空了，让人去跑标注同样是绕远路。
        """
        labelled = self.pose_labelled_count()
        if not labelled:
            if not load_index(self.root).items:
                return "姿态库还没有图片：先导入图片或抽帧（命令行 autolabel import / auto）"
            return ("姿态库还没有任何骨骼标签：先跑一次自动标注（命令行 autolabel auto / annotate，"
                    "或在复核界面手动画一副骨骼再保存）")
        return (f"这个导出范围里没有带骨骼标签的帧（库里共 {labelled} 张带标签）："
                "换个导出范围，或先把带标签的帧点成「接受」")

    def pose_labelled_count(self, exclude_status: Sequence[str] = ()) -> int:
        """姿态库里真正带标签文件的图片数，用来在空库时直接说人话而不是导出个空目录。"""
        exclude = set(exclude_status)
        count = 0
        for rel, item in load_index(self.root).ordered_items():
            if str(item.get("status", "unreviewed")) in exclude:
                continue
            path = label_path_for(self.root, rel)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                count += 1
        return count

    @staticmethod
    def _number(payload: dict, key: str, default: float, low: Optional[float], high: Optional[float],
                name: str, integer: bool = False) -> float:
        """界面传来的数字参数：0 是实义值，不能被 ``or`` 吞成默认值。

        实测过 ``val_ratio=0``（整库当训练集）、``seed=0``、``max_empty_ratio=0``
        （不要背景图）三种合法写法，以前统统被 ``payload.get(k) or 默认值`` 改回
        默认值，界面上怎么调都不生效，而命令行同样的写法是生效的——两条路两套口径。
        """
        raw = payload.get(key, None)
        if raw is None or raw == "":
            raw = default
        try:
            value = int(raw) if integer else float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name}要填数字，现在是「{raw}」")
        if integer and float(raw) != value:
            raise ValueError(f"{name}要填整数，现在是「{raw}」")
        if low is not None and value < low:
            raise ValueError(f"{name}不能小于 {low}，现在是 {value}")
        if high is not None and value > high:
            raise ValueError(f"{name}不能大于 {high}，现在是 {value}")
        return float(value)

    def export_detection(self, payload: dict) -> dict:
        """导出 YOLO detection 数据集（train/val + data.yaml）。"""
        from .prompt import export_detection_dataset

        payload = payload or {}
        scope = str(payload.get("scope") or "reviewed")
        table = {"reviewed": ("rejected",), "approved": ("rejected", "unreviewed"), "all": ()}
        if scope not in table:
            raise ValueError(f"未知的导出范围：{scope}（可选 reviewed / approved / all）")
        val_ratio = self._number(payload, "val_ratio", 0.2, 0.0, 1.0, "验证集比例")
        seed = int(self._number(payload, "seed", 7, None, None, "随机种子", integer=True))
        max_empty_ratio = self._number(payload, "max_empty_ratio", 0.15, 0.0, 1.0, "背景图比例上限")
        result = export_detection_dataset(
            self.root, out=str(payload.get("out") or "") or None,
            val_ratio=val_ratio, seed=seed, exclude_status=table[scope],
            max_empty_ratio=max_empty_ratio,
            tag=str(payload.get("tag") or "") or None,
            mode=str(payload.get("mode") or "symlink"))
        result["scope"] = scope
        return result

    def reveal_export(self, payload: dict) -> dict:
        """在系统文件管理器里定位导出目录：产物是目录，光给路径用户还是找不到。"""
        import subprocess
        import sys

        from .library import exports_dir

        raw = str((payload or {}).get("path") or "").strip()
        if not raw:
            raise ValueError("没有指定要打开的导出目录")
        base = os.path.realpath(exports_dir(self.root))
        target = os.path.realpath(os.path.expanduser(raw))
        if target != base and not target.startswith(base + os.sep):
            raise ValueError("只能打开本数据集 exports/ 下的导出结果")
        if not os.path.exists(target):
            raise ValueError(f"导出目录不存在，可能已被清理：{target}")
        if sys.platform == "darwin":
            cmd = ["open", "-R", target]
        elif sys.platform.startswith("win"):
            cmd = ["explorer", os.path.normpath(target)]
        else:
            cmd = ["xdg-open", target if os.path.isdir(target) else os.path.dirname(target)]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            return {"ok": False, "path": target,
                    "message": f"没能自动打开（{exc}），可手动执行：{' '.join(cmd)}"}
        return {"ok": True, "path": target, "message": "已在文件管理器里定位到该目录"}

    # --- 模型管理 ---
    def models_payload(self) -> dict:
        """权重清单 + 下载进度 + 多模态大模型配置，一次给全「模型管理」面板。"""
        return vlm_payload(dataset_root=self.root)

    def active_models(self) -> List[str]:
        """当前仍有未完成任务在用的权重，删除时要把住这道门。"""
        return [str(job.get("model")) for job in self.jobs.snapshot()
                if not job.get("finished") and job.get("model")]

    def delete_weight(self, name: str) -> dict:
        return delete_weight(name, active_names=self.active_models())

    def download_weight(self, name: str, force: bool = False) -> dict:
        task = start_download(name, force=force)
        return {"ok": True, "task": task, "weights": self.models_payload()["weights"]}

    def weight_download_state(self, name: str) -> dict:
        state = download_state(name)
        if state is None:
            raise KeyError(f"没有 {name} 的下载记录")
        return state

    def save_vlm_profile(self, profile: dict) -> dict:
        return upsert_profile(profile)

    def delete_vlm_profile(self, profile_id: str) -> dict:
        return delete_profile(profile_id)

    def set_default_vlm(self, profile_id: str) -> dict:
        return set_default(profile_id)

    def test_vlm_profile(self, profile_id: str) -> dict:
        return test_saved_profile(profile_id)

    def model_rows(self) -> List[dict]:
        """统一模型清单（含中文说明、备注、兼容性与最后使用时间）。"""
        return list_models(dataset_root=self.root)

    def save_model_note(self, model_id: str, note: str) -> dict:
        """给某个模型写备注说明，空串表示清空。"""
        return set_model_note(model_id, note)

    def save_model_default(self, model_id: str, capability: str = "") -> dict:
        return set_model_default(model_id, capability)

    def add_local_model(self, payload: dict) -> dict:
        """登记一个本地权重：只记路径和说明，不动文件。"""
        payload = payload or {}
        return register_local_model(
            path=str(payload.get("path") or ""),
            capability=str(payload.get("capability") or "pose"),
            display_name=str(payload.get("display_name") or ""),
            note=str(payload.get("note") or ""),
            origin=str(payload.get("origin") or "custom"),
            model_id=str(payload.get("id") or ""),
        )

    def remove_local_model(self, model_id: str) -> dict:
        return forget_model(model_id)

    def undo(self, image_rel: str, kind: str = "pose") -> dict:
        """回滚到最近一次历史快照（``kind="prompt"`` 回滚检测框）。"""
        if kind == "prompt":
            return self.undo_prompt(image_rel)
        history = self._history_dir(image_rel)
        with self._mutate():
            snaps = sorted(os.listdir(history)) if os.path.isdir(history) else []
            if not snaps:
                backup = self._backup_path(image_rel)
                if not os.path.exists(backup):
                    raise ValueError("这张图还没有可撤销的修改：它没有被保存过")
                shutil.copy2(backup, label_path_for(self.root, image_rel))
            else:
                latest = snaps[-1]
                shutil.copy2(os.path.join(history, latest), label_path_for(self.root, image_rel))
                os.remove(os.path.join(history, latest))
            persons = parse_label_file(label_path_for(self.root, image_rel))
            self.index.upsert(image_rel, status=STATUS_UNREVIEWED, objects=len(persons),
                              reviewed_at=now_str())
            self.index.save()
        return {"ok": True, "image": image_rel,
                "objects": [self._person_payload(p) for p in persons],
                "status": STATUS_UNREVIEWED}

    def undo_prompt(self, image_rel: str) -> dict:
        """把检测框回滚到上一次快照：删错的框删多了也能救回来。"""
        history = self._prompt_history_dir(image_rel)
        with self._mutate():
            label = prompt_label_path_for(self.root, image_rel)
            snaps = sorted(os.listdir(history)) if os.path.isdir(history) else []
            if snaps:
                shutil.copy2(os.path.join(history, snaps[-1]), label)
                os.remove(os.path.join(history, snaps[-1]))
            else:
                backup = self._prompt_backup_path(image_rel)
                if not os.path.exists(backup):
                    raise ValueError("这张图还没有可撤销的修改：它没有被保存过")
                shutil.copy2(backup, label)
            count = len(read_detections(self.root, image_rel))
            self.objects_index.upsert(image_rel, prompt_status=STATUS_UNREVIEWED,
                                      prompt_objects=count, prompt_reviewed_at=now_str())
            self.objects_index.save()
        return {"ok": True, "image": image_rel, "prompt_objects": count,
                "prompt_status": STATUS_UNREVIEWED}

    def _backup(self, image_rel: str, label: str, *, backup_path: str = "",
                history_dir: str = "") -> None:
        """首次修改前备份原标签；随后按时间戳留有限个历史版本。

        姿态标签与提示词标签共用这套机制，只是各走各的备份目录。
        """
        history = history_dir or self._history_dir(image_rel)
        os.makedirs(history, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        if os.path.exists(label):
            backup = backup_path or self._backup_path(image_rel)
            if not os.path.exists(backup):
                os.makedirs(os.path.dirname(backup), exist_ok=True)
                shutil.copy2(label, backup)
            shutil.copy2(label, os.path.join(history, f"{stamp}.txt"))
        else:
            # 原来一张框都没有：留一个空白快照，撤销才能回到"没画过"
            open(os.path.join(history, f"{stamp}.txt"), "w", encoding="utf-8").close()
        snaps = sorted(os.listdir(history))
        for old in snaps[:-HISTORY_LIMIT]:
            os.remove(os.path.join(history, old))

    def _backup_path(self, image_rel: str) -> str:
        return os.path.join(backups_dir(self.root, LIB_POSE), self._stem(image_rel) + ".txt")

    def _history_dir(self, image_rel: str) -> str:
        return os.path.join(history_root(self.root, LIB_POSE), self._stem(image_rel))

    @staticmethod
    def _stem(image_rel: str) -> str:
        return os.path.splitext(str(image_rel).replace("/", "__"))[0]

    def undoable(self, image_rel: str, kind: str = "pose") -> bool:
        """界面用来判断「撤销」是否可用：有原始备份或有历史快照都算。"""
        if kind == "prompt":
            backup, history = (self._prompt_backup_path(image_rel),
                               self._prompt_history_dir(image_rel))
        else:
            backup, history = self._backup_path(image_rel), self._history_dir(image_rel)
        if os.path.exists(backup):
            return True
        return os.path.isdir(history) and bool(os.listdir(history))

    def _prompt_backup_path(self, image_rel: str) -> str:
        return os.path.join(backups_dir(self.root, LIB_OBJECTS), self._stem(image_rel) + ".txt")

    def _prompt_history_dir(self, image_rel: str) -> str:
        return os.path.join(history_root(self.root, LIB_OBJECTS), self._stem(image_rel))


def persons_from_payload(objects: List[dict], img_w: int, img_h: int) -> List[Person]:
    """把前端 JSON 转成 Person，并做越界裁剪与 bbox 自愈。"""
    persons: List[Person] = []
    for raw in objects:
        kpts = [Keypoint(x=_clamp01(float(k["x"])), y=_clamp01(float(k["y"])),
                         v=int(k.get("v", 2)), conf=float(k.get("conf", 1.0 if int(k.get("v", 2)) else 0.0)))
                for k in raw.get("keypoints", [])]
        person = Person(cls=int(raw.get("cls", 0)), cx=_clamp01(float(raw.get("cx", 0.5))),
                        cy=_clamp01(float(raw.get("cy", 0.5))), w=_clamp01(float(raw.get("w", 0))),
                        h=_clamp01(float(raw.get("h", 0))), keypoints=kpts,
                        score=float(raw.get("score", 0.0)), flags=list(raw.get("flags", [])))
        persons.append(heal_bbox(person, img_w, img_h))
    return persons


def heal_bbox(person: Person, img_w: int, img_h: int) -> Person:
    """关键点被拖动后保证框住所有可见关键点，避免训练时框/点不一致。"""
    visible = person.keypoints_xyxy(img_w, img_h)
    if not visible:
        return person
    x1, y1, x2, y2 = person.xyxy(img_w, img_h)
    x1, y1 = min(x1, min(p[0] for p in visible)), min(y1, min(p[1] for p in visible))
    x2, y2 = max(x2, max(p[0] for p in visible)), max(y2, max(p[1] for p in visible))
    pad_x, pad_y = (x2 - x1) * 0.04, (y2 - y1) * 0.04
    healed = Person.from_xyxy((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
                              img_w, img_h, person.keypoints, person.score, person.cls)
    healed.flags = person.flags
    return healed


def _visible_ratio(persons: List[Person]) -> float:
    total = visible = 0
    for p in persons:
        total += len(p.keypoints)
        visible += sum(1 for k in p.keypoints if k.v != 0)
    return round(visible / total, 3) if total else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore  # 由 server 注入
    protocol_version = "HTTP/1.1"
    max_upload_mb: int = 1024

    # 一屏队列就要拉几十张缩略图，全打出来的话终端只剩刷屏，真正要看的报错被冲走
    QUIET_LOG_PREFIXES = ("/api/", "/image/", "/thumb/")

    def log_message(self, fmt: str, *args) -> None:  # 降噪
        if self.path.startswith(self.QUIET_LOG_PREFIXES):
            return
        super().log_message(fmt, *args)

    # --- 路由 ---
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html", "/portal"):
                self._send_page("portal.html")
            elif path in ("/pose", "/pose/", "/review", "/review/"):
                # /pose 是二级模块「姿态标注」，/review 是改名前的老地址，两者同一页
                self._send_page("review.html")
            elif path in ("/objects", "/objects/"):
                self._send_page("objects.html")
            elif path in ("/models", "/models/"):
                self._send_page("models.html")
            elif path in ("/app.js", "/common.js", "/portal.js", "/objects.js", "/models.js"):
                self._send_file(os.path.join(STATIC_DIR, path.lstrip("/")),
                                "application/javascript; charset=utf-8")
            elif path == "/app.css":
                self._send_file(os.path.join(STATIC_DIR, "app.css"), "text/css; charset=utf-8")
            elif path == "/favicon.ico":
                # 页面已内联 SVG 图标，这里只是兜底，省得浏览器控制台留一条 404
                self._send_file(os.path.join(STATIC_DIR, "favicon.ico"), "image/x-icon")
            elif path == "/api/dataset":
                self._send_json(self.store.dataset_payload())
            elif path == "/api/item":
                image = parse_qs(parsed.query).get("image", [""])[0]
                self._send_json(self.store.read_item(unquote(image)))
            elif path == "/api/fields":
                self._send_json(self.store.fields_payload())
            elif path == "/api/prompt/item":
                image = parse_qs(parsed.query).get("image", [""])[0]
                self._send_json(self.store.read_prompt_item(unquote(image)))
            elif path == "/api/prompt/summary":
                self._send_json(self.store.prompt_summary_payload())
            elif path == "/api/jobs":
                self._send_json({"jobs": self.store.jobs.snapshot()})
            elif path == "/api/batches":
                self._send_json(self.store.batches_payload())
            elif path == "/api/batch/files":
                batch_id = parse_qs(parsed.query).get("id", [""])[0]
                self._send_json(self.store.batch_files(unquote(batch_id)))
            elif path == "/api/pool":
                lib = parse_qs(parsed.query).get("lib", [""])[0]
                self._send_json(self.store.pool_payload(lib))
            elif path == "/api/models":
                self._send_json(self.store.models_payload())
            elif path == "/api/version":
                # 顶栏的版本徽标；?refresh=1 是用户在「关于」里点了「检查更新」
                refresh = parse_qs(parsed.query).get("refresh", ["0"])[0] in ("1", "true", "yes")
                self._send_json(self.store.version_payload(force=refresh))
            elif path == "/api/coords":
                from .coords import render_coords

                query = parse_qs(parsed.query)
                fmt = query.get("format", ["csv"])[0]
                exclude = () if query.get("include_all", ["0"])[0] in ("1", "true") else ("rejected",)
                name, body = render_coords(self.store.root, fmt, exclude_status=exclude)
                ctype = "application/json; charset=utf-8" if fmt == "jsonl" else "text/csv; charset=utf-8"
                self._send_download(name, body, ctype)
            elif path == "/api/prompt/coords":
                from .prompt import render_object_coords

                fmt = parse_qs(parsed.query).get("format", ["csv"])[0]
                name, body = render_object_coords(self.store.root, fmt)
                ctype = "application/json; charset=utf-8" if fmt == "jsonl" else "text/csv; charset=utf-8"
                self._send_download(name, body, ctype)
            elif path.startswith("/image/"):
                self._send_image(unquote(path[len("/image/"):]))
            elif path.startswith("/thumb/"):
                # 队列缩略图走小图 + 长缓存，主图仍旧用 /image/ 看全分辨率
                self._send_image(unquote(path[len("/thumb/"):]), thumb=True)
            else:
                self._send_json({"error": "not found"}, status=404)
        except FileNotFoundError as exc:
            self._send_json({"error": f"文件不存在: {exc}"}, status=404)
        except KeyError as exc:
            self._send_json({"error": f"未知对象: {exc.args[0] if exc.args else ''}"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if parsed.path == "/api/upload":
            self._handle_upload(parsed, length)
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "JSON 解析失败"}, status=400)
            return
        image = unquote(str(payload.get("image", "")))
        try:
            if parsed.path == "/api/item":
                self._send_json(self.store.write_item(image, payload.get("objects", []),
                                                      payload.get("status")))
            elif parsed.path == "/api/status":
                self._send_json(self.store.set_status(image, str(payload.get("status"))))
            elif parsed.path == "/api/undo":
                self._send_json(self.store.undo(image, str(payload.get("kind") or "pose")))
            elif parsed.path == "/api/fields/save":
                saved = self.store.save_field(payload.get("field") or payload)
                self._send_json({"ok": True, "field": saved, **self.store.fields_payload()})
            elif parsed.path == "/api/fields/delete":
                wanted = str(payload.get("id") or payload.get("name") or "").strip()
                if not wanted:
                    raise ValueError("没有指定要删除的字段")
                self._send_json({"ok": True, **self.store.remove_field(
                    wanted, drop_boxes=bool(payload.get("drop_boxes")))})
            elif parsed.path == "/api/prompt/item":
                self._send_json(self.store.write_prompt_item(image, payload.get("objects", []),
                                                             payload.get("status")))
            elif parsed.path == "/api/prompt/status":
                self._send_json(self.store.set_prompt_status(image, str(payload.get("status"))))
            elif parsed.path == "/api/prompt/undo":
                self._send_json(self.store.undo_prompt(image))
            elif parsed.path == "/api/prompt/run":
                self._send_json(self.store.run_prompt_job(payload), status=202)
            elif parsed.path == "/api/prompt/export":
                self._send_json({"ok": True, "export": self.store.export_detection(payload)})
            elif parsed.path == "/api/export":
                self._send_json({"ok": True, "export": self.store.export_pose_dataset(payload)})
            elif parsed.path == "/api/export/reveal":
                self._send_json(self.store.reveal_export(payload))
            elif parsed.path == "/api/prompt/upload":
                self._send_json(self.store.run_prompt_upload_job(payload), status=202)
            elif parsed.path == "/api/images/delete":
                images = [str(item) for item in (payload.get("images") or []) if str(item).strip()]
                if not images:
                    raise ValueError("没有指定要删除的图片")
                self._send_json({"ok": True, **self.store.delete_images(
                    images, keep_files=bool(payload.get("keep_files")),
                    lib=str(payload.get("lib") or "pose"),
                    mode=str(payload.get("mode") or "lib"))})
            elif parsed.path == "/api/pool/adopt":
                rels = [str(item) for item in (payload.get("images") or []) if str(item).strip()]
                self._send_json(self.store.adopt_pool(
                    rels, lib=str(payload.get("lib") or "pose"),
                    batch_name_text=str(payload.get("batch_name") or "")))
            elif parsed.path == "/api/jobs":
                job = self.store.jobs.submit(str(payload.get("kind", "image")),
                                             list(payload.get("files") or []),
                                             payload.get("params") or {})
                self._send_json(job.payload(), status=202)
            elif parsed.path == "/api/jobs/clear":
                self._send_json({"ok": True, **self.store.jobs.clear()})
            elif parsed.path == "/api/batches/delete":
                ids = [str(b) for b in (payload.get("ids") or []) if str(b)]
                if not ids:
                    raise ValueError("没有指定要删除的批次")
                self._send_json({"ok": True, **self.store.delete_batches(
                    ids, keep_files=bool(payload.get("keep_files")))})
            elif parsed.path == "/api/models/weight/download":
                name = str(payload.get("name") or "").strip()
                if not name:
                    raise ValueError("没有指定要下载的权重")
                self._send_json(self.store.download_weight(name, force=bool(payload.get("force"))))
            elif parsed.path == "/api/models/download":
                self._send_json({"ok": True,
                                 "task": self.store.weight_download_state(
                                     str(payload.get("name") or ""))})
            elif parsed.path == "/api/models/weight/delete":
                name = str(payload.get("name") or "").strip()
                if not name:
                    raise ValueError("没有指定要删除的权重")
                self._send_json({"ok": True, **self.store.delete_weight(name)})
            elif parsed.path == "/api/models/vlm/save":
                profile = payload.get("profile") or payload
                self._send_json({"ok": True, "profile": self.store.save_vlm_profile(profile),
                                 **self.store.models_payload()["vlm"]})
            elif parsed.path == "/api/models/vlm/delete":
                self._send_json({"ok": True, **self.store.delete_vlm_profile(
                                    str(payload.get("id") or ""))})
            elif parsed.path == "/api/models/vlm/default":
                self._send_json({"ok": True, **self.store.set_default_vlm(
                                    str(payload.get("id") or ""))})
            elif parsed.path == "/api/models/vlm/test":
                profile_id = str(payload.get("id") or "").strip()
                if not profile_id:
                    raise ValueError("没有指定要测试的配置")
                self._send_json({"ok": True, "result": self.store.test_vlm_profile(profile_id),
                                 **self.store.models_payload()["vlm"]})
            elif parsed.path == "/api/models/note":
                model_id = str(payload.get("id") or "").strip()
                if not model_id:
                    raise ValueError("没有指定要备注的模型")
                self._send_json({"ok": True, **self.store.save_model_note(
                    model_id, str(payload.get("note") or ""))})
            elif parsed.path == "/api/models/default":
                model_id = str(payload.get("id") or "").strip()
                if not model_id:
                    raise ValueError("没有指定要设为默认的模型")
                self._send_json({"ok": True, "defaults": self.store.save_model_default(
                    model_id, str(payload.get("capability") or ""))})
            elif parsed.path == "/api/models/local/add":
                self._send_json({"ok": True, "model": self.store.add_local_model(payload),
                                 "models": self.store.model_rows()})
            elif parsed.path == "/api/models/local/forget":
                model_id = str(payload.get("id") or "").strip()
                if not model_id:
                    raise ValueError("没有指定要取消登记的模型")
                self._send_json({"ok": True, **self.store.remove_local_model(model_id),
                                 "models": self.store.model_rows()})
            else:
                self._send_json({"error": "not found"}, status=404)
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except KeyError as exc:
            self._send_json({"error": f"未知对象: {exc.args[0] if exc.args else ''}"}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _handle_upload(self, parsed, length: int) -> None:
        """原始字节上传（无 multipart 解析）：文件名走 ?name= 查询参数。

        交给 ``save_upload_stream`` 分块落盘：超限/类型不支持时它也会把请求体读干净，
        这里才允许回 413/400，否则浏览器看到的是断链而不是中文提示。
        """
        from .upload import UploadTooLarge, save_upload_stream

        if length <= 0:
            self._send_json({"error": "缺少文件内容"}, status=400)
            return
        query = parse_qs(parsed.query)
        name = unquote(query.get("name", [""])[0])
        try:
            info = save_upload_stream(self.store.root, name, self.rfile, length,
                                      max_mb=self.max_upload_mb)
        except UploadTooLarge:
            self._send_json({"error": f"单个文件超过 {self.max_upload_mb} MB 上限"}, status=413)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            return
        self._send_json(info)

    # --- 响应 ---
    def _send_page(self, filename: str) -> None:
        """门户页与三个二级模块页都从这里发，共用同一份 CSS 与工具脚本。"""
        self._send_file(os.path.join(STATIC_DIR, filename), "text/html; charset=utf-8")

    def _send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, ctype: str, cache: str = "no-store") -> None:
        with open(path, "rb") as fh:
            body = fh.read()
        # 页面/图片用内联渲染，只有坐标下载才带 attachment 头
        self._send_download("", body, ctype, cache=cache)

    def _send_download(self, name: str, body: bytes, ctype: str,
                       cache: str = "no-store") -> None:
        from urllib.parse import quote

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if name:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}")
        # 只有缩略图允许缓存：它的文件名里编了内容指纹，图片一换 URL 就变
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, image_rel: str, thumb: bool = False) -> None:
        path = resolve_image(self.store.root, image_rel)
        if not os.path.exists(path):
            raise FileNotFoundError(image_rel)
        if thumb:
            small = build_thumbnail(self.store.root, path, image_rel)
            if small:
                self._send_file(small, "image/jpeg", cache=THUMB_CACHE_CONTROL)
                return
        ctype = "image/jpeg"
        if path.lower().endswith(".png"):
            ctype = "image/png"
        elif path.lower().endswith((".webp",)):
            ctype = "image/webp"
        self._send_file(path, ctype)


def serve(root: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          max_upload_mb: int = 1024) -> None:
    from . import version as versioning

    store = ReviewStore(root)
    ReviewHandler.store = store
    ReviewHandler.max_upload_mb = max_upload_mb
    # 缩略图缓存只增不减：删掉的图没人认领，启动时清一次久未访问的
    pruned = prune_thumbs(store.root)
    if pruned:
        print(f"[info] 清理缩略图缓存 {pruned} 张（超过 30 天未使用）")
    from .upload import prune_stale_uploads

    # 上传的原始文件在任务成功后才删，中途被打断就没人认领了（一次视频就几百 MB）
    stale = prune_stale_uploads(store.root)
    if stale["files"]:
        print(f"[info] 清理上次遗留的上传临时文件 {stale['files']} 个"
              f"（{stale['bytes'] / 1048576:.1f} MB，超过 24 小时未使用）")
    httpd = ThreadingHTTPServer((host, port), ReviewHandler)
    url = f"http://{host}:{httpd.server_address[1]}/"
    total = len(store.index.items)
    objects_total = len(store.objects_index.items)
    # 用内存里已经读好的两本索引现算，别再为这一行去解析一遍 index.json
    counts = review_counts(list(store.index.items.values()),
                           list(store.objects_index.items.values()))  # noqa: E501
    print(f"[info] 工作台: {url}")
    print(f"[info] 自动标注 v{versioning.VERSION} · 开发者 {versioning.DEVELOPER} · {versioning.REPO_URL}")
    print(f"[info] 数据集 {store.index.name}：姿态库 {total} 张 · 提示词库 {objects_total} 张，"
          f"等你复核 {counts['pending']} 张（其中 {counts['flagged']} 张"
          f"机器自己报了问题，进页面按 F 逐张跳）")
    print("[info] 二级模块：/pose 姿态标注 · /objects 提示词标注 · /models 模型管理")
    print(f"[info] 右上角可上传视频/图片目录（单文件上限 {max_upload_mb} MB），上传后自动标注并生成坐标文件")
    if store.recovery["recovered"] or store.recovery["interrupted"] or store.recovery["missing_files"]:
        print(f"[warn] 启动对账：补回 {store.recovery['recovered']} 张未入库图片，"
              f"判定 {len(store.recovery['interrupted'])} 个批次中断，"
              f"{store.recovery['missing_files']} 条索引记录的原图已丢失")
    print("[info] 快捷键：Q 接受 / E 驳回 / ←→ 或 A D 翻页 / 0 1 2 设置关键点可见性 / X 删除目标 / U 撤销")
    print("[info] Ctrl+C 退出")
    versioning.start_background_check()   # 有新版本自己补一行，不挡启动
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] 已退出，索引与标签已落盘")
    finally:
        httpd.server_close()
        store.index.save()
        store.objects_index.save()
