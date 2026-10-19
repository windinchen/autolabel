"""批量自动标注流水线：读图 -> 推理 -> QC 打分 -> 写 YOLO pose 标签。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .backends import BackendSettings, build_backend
from .dataset import (
    DEFINITIVE_STATUSES,
    HUMAN_STATUSES,
    STATUS_UNREVIEWED,
    label_path_for,
    load_index,
    now_str,
    resolve_image,
)
from .image_utils import load_rgb
from .yolo_pose import Person, person_sanity_flags, write_label_file


#: 人工碰过的状态直接从 dataset 取（唯一出处）：这里只是给老调用留个顺手的别名。
__all__ = ["HUMAN_STATUSES", "image_level_flags", "label_dataset", "settings_signature"]


def _human_reviewed(item: dict) -> bool:
    """这张图的人工结果是否值得保护（接受过 / 手工改过）。"""
    return item.get("status") in HUMAN_STATUSES


def settings_signature(settings: BackendSettings) -> str:
    """用于断点续跑的参数指纹：参数变了就重标。

    ``device`` 与 ``half`` 特意不算进指纹——它们只改快慢，不改标成什么样。
    否则中途从 Apple 加速切到 CPU，几千张已标好的图会被判成"参数变了"重跑一遍。
    """
    payload = {k: v for k, v in settings.to_dict().items() if k not in ("device", "half")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:12]


def blur_score(image: np.ndarray) -> float:
    """拉普拉斯方差清晰度：值越小越糊。"""
    gray = image.mean(axis=2) if image.ndim == 3 else image
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return float("inf")
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    laplacian = np.zeros_like(gray, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            laplacian += kernel[i, j] * padded[i:i + gray.shape[0], j:j + gray.shape[1]]
    return float(laplacian.var())


def image_level_flags(
    height: int,
    width: int,
    persons: List[Person],
    min_side_warn: int = 320,
    crowd_threshold: int = 8,
    blurred: bool = False,
) -> List[str]:
    """图片级 QC 标记（只需宽高，复核服务无需重新解码图片）。"""
    flags: List[str] = []
    if not persons:
        flags.append("no_person")
    if len(persons) >= crowd_threshold:
        flags.append("crowded")
    if min(width, height) < min_side_warn:
        flags.append("low_resolution")
    if blurred:
        flags.append("likely_blurred")
    for person in persons:
        flags.extend(f"obj:{f}" for f in person.flags)
    return sorted(set(flags))


def label_dataset(
    root: str,
    settings: BackendSettings,
    only_missing: bool = True,
    limit: Optional[int] = None,
    render_qc: int = 0,
    dry_run: bool = False,
    verbose: bool = True,
    progress: Optional["Callable[[int, int], None]"] = None,
    should_stop: Optional["Callable[[], bool]"] = None,
    only: Optional[Sequence[str]] = None,
) -> dict:
    """对数据集内图片批量推理并写标签。

    :param only_missing: 断点续跑——跳过参数指纹一致且标签已存在的图片，
        以及人工接受过 / 改过骨骼的图片（`--force` 才会连它们一起重标）
    :param render_qc: 额外渲染前 N 张带骨架叠加图的 QC 预览
    :param progress: 每完成一张回调 (已完成, 总数)，供上传界面的进度条使用
    :param should_stop: 每张前检查，返回 True 就停下来（用于界面「清除任务」）
    :param only: 只标这批图（界面「批量姿态标注」圈定的那批）。留空 = 扫全库，
        命令行与上传任务不传这个参数，行为一点不变
    """
    idx = load_index(root)
    wanted = {str(rel) for rel in only} if only is not None else None
    # 逐图改动先攒在这里，落盘前与磁盘上的索引合并：复核界面在标注期间也能写索引，
    # 直接整体覆盖会把用户在标注过程中点的“接受/驳回”抹掉。
    updates: Dict[str, dict] = {}

    def flush() -> None:
        try:
            target = load_index(root)
        except Exception:  # noqa: BLE001 磁盘索引不可用时退回内存副本
            target = idx
        for rel, fields in updates.items():
            target.upsert(rel, **fields)
        target.save()

    signature = settings_signature(settings)
    previous = idx.data.get("settings", {})
    if previous.get("signature") not in (None, signature) and only_missing and verbose:
        stale = [k for k, v in idx.items.items()
                 if not _human_reviewed(v) and v.get("label_signature") != signature]
        if stale:
            print(f"[info] 标注参数已变化，{len(stale)} 张需要重标（人工已复核的不计入）")

    ensemble = build_backend(settings)
    if not dry_run:
        ensemble.load()

    # 该不该跳交给逐张判断：`_already_labeled` 本来就是按参数指纹比的，
    # 以前额外来一句「参数一变就整批不续跑」，把中断前已经按新参数标好的那些又重标一遍。
    targets: List[tuple] = []
    skipped_human = 0
    for rel, item in idx.ordered_items():
        if wanted is not None and str(rel) not in wanted:
            continue
        if only_missing and _human_reviewed(item) and os.path.exists(label_path_for(root, rel)):
            skipped_human += 1
            continue
        if only_missing and _already_labeled(root, rel, item, signature):
            continue
        targets.append((rel, item))
    if limit:
        targets = targets[:limit]

    stats = {"total": len(targets), "labeled": 0, "people": 0, "empty": 0,
             "flagged": 0, "failed": 0, "skipped_reviewed": 0, "skipped_human": skipped_human,
             "verdict_reset": 0,
             "cancelled": 0}
    if not targets:
        if skipped_human:
            print(f"[info] 没有需要标注的图片（人工已复核 {skipped_human} 张按规则不动，"
                  f"其余已完成；连人工结果一起重标用 --force）")
        else:
            print("[info] 没有需要标注的图片（全部已完成，可用 --force 强制重标）")
        return stats

    start = time.time()
    for n, (rel, item) in enumerate(targets, start=1):
        if should_stop and should_stop():
            stats["cancelled"] = len(targets) - n + 1
            break
        abs_path = resolve_image(root, rel)
        try:
            image = load_rgb(abs_path, max_side=settings.max_side)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            print(f"[warn] 读取失败 {rel}: {type(exc).__name__}: {exc}")
            continue
        h, w = image.shape[:2]
        if dry_run:
            persons: List[Person] = []
        else:
            try:
                persons = ensemble.infer(image)
            except Exception as exc:  # noqa: BLE001 单张推理失败不该中断整批
                stats["failed"] += 1
                print(f"[warn] 推理失败 {rel}: {type(exc).__name__}: {exc}")
                continue
            for person in persons:
                person.flags = sorted(set(person.flags) | set(
                    person_sanity_flags(person, w, h)))
        flags = [f for f in image_level_flags(
            h, w, persons, blurred=blur_score(image) < 40.0) if not f.startswith("obj:")]
        if not dry_run:
            write_label_file(label_path_for(root, rel), persons)
        # 人工结论只对"当初那批框"有效：参数一换内容就是新的，接受过的也得重新确认。
        # 同一套参数 --force 重跑结果本就一样，这种保留原结论，别把人工标记冲没。
        had_verdict = item.get("status") in DEFINITIVE_STATUSES
        keep_verdict = had_verdict and item.get("label_signature") in (None, signature)
        fields = dict(
            width=w, height=h,
            objects=len(persons),
            score=round(float(np.mean([p.score for p in persons])), 4) if persons else 0.0,
            visible_ratio=round(_visible_ratio(persons), 3),
            flags=[f for f in flags if not f.startswith("obj:")],
            object_flags=sorted({f for p in persons for f in p.flags}),
            label_signature=None if dry_run else signature,
            labeled_at=now_str(),
            labeled_by=getattr(ensemble, "source_tag", "") or f"yolo:{settings.model}",
            status=item.get("status") if keep_verdict else STATUS_UNREVIEWED,
        )
        if fields["status"] != STATUS_UNREVIEWED:
            stats["skipped_reviewed"] += 1
        elif had_verdict:
            stats["verdict_reset"] += 1      # 人工结论作废，退回待复核
        idx.upsert(rel, **fields)
        updates.setdefault(rel, {}).update(fields)
        stats["labeled"] += 1
        stats["people"] += len(persons)
        stats["empty"] += int(not persons)
        stats["flagged"] += int(bool(flags) or bool(fields["object_flags"]))
        if render_qc and n <= render_qc and not dry_run:
            _render_qc_image(root, rel, image, persons)
        if verbose and (n % 20 == 0 or n == len(targets)):
            rate = n / max(time.time() - start, 1e-6)
            eta = (len(targets) - n) / max(rate, 1e-6)
            print(f"[{n}/{len(targets)}] {rate:.1f} img/s  ETA {eta:.0f}s  人数累计 {stats['people']}")
        if progress:
            progress(n, len(targets))
        if n % 25 == 0:
            flush()

    def finish() -> None:
        try:
            target = load_index(root)
        except Exception:  # noqa: BLE001
            target = idx
        for rel, fields in updates.items():
            target.upsert(rel, **fields)
        target.data["settings"] = {"signature": signature, **settings.to_dict(),
                                   "keypoint_names": ensemble.display_names}
        target.data["kpt_shape"] = [len(ensemble.display_names), 3]
        target.data["last_labeled_at"] = now_str()
        target.save()

    finish()
    stats["elapsed_sec"] = round(time.time() - start, 1)
    stats["img_per_sec"] = round(stats["labeled"] / max(stats["elapsed_sec"], 1e-6), 2)
    return stats


def _already_labeled(root: str, rel: str, item: dict, signature: str) -> bool:
    """这张图是否已经用**当前这套参数**标好了（断点续跑的最小单位是单张，不是整批）。"""
    return (item.get("label_signature") == signature
            and item.get("objects") is not None
            and os.path.exists(label_path_for(root, rel)))


def _visible_ratio(persons: List[Person]) -> float:
    total = visible = 0
    for p in persons:
        total += len(p.keypoints)
        visible += sum(1 for k in p.keypoints if k.v != 0)
    return visible / total if total else 0.0


def _render_qc_image(root: str, rel: str, image: np.ndarray, persons: List[Person]) -> None:
    import cv2

    from .image_utils import draw_poses
    from .yolo_pose import COCO17_SKELETON

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    canvas = draw_poses(bgr, persons, COCO17_SKELETON)
    out = os.path.join(os.path.abspath(root), "qc", "overlays", os.path.splitext(rel)[0] + ".jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
