"""命令行入口：init / import / video / label / qc / review / export / coords / validate / status /
pool / migrate / fields / prompt / models / reset。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional

from .backends import BackendSettings
from .dataset import (
    STATUS_APPROVED,
    STATUS_UNREVIEWED,
    import_images,
    init_dataset,
    label_path_for,
    load_index,
)
from .library import LIBRARIES, lib_name
from .upload import VIDEO_EXTS
from .yolo_pose import validate_label_file


def _pad(text: object, width: int) -> str:
    """按终端显示宽度补空格：中文占两列，直接用 ``:<10`` 会把表格拉歪。"""
    value = str(text)
    wide = sum(1 for ch in value if unicodedata.east_asian_width(ch) in ("W", "F"))
    return value + " " * max(0, width - len(value) - wide)


def load_settings(args: argparse.Namespace) -> BackendSettings:
    """命令行参数 > YAML 配置 > 内置默认。"""
    overrides: Dict = {}
    config = getattr(args, "config", None)
    if config and os.path.exists(config):
        import yaml

        with open(config, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        overrides.update({k: v for k, v in data.items() if k in BackendSettings.__dataclass_fields__})
    for key in BackendSettings.__dataclass_fields__:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if overrides.get("extra_models") and isinstance(overrides["extra_models"], str):
        overrides["extra_models"] = tuple(m.strip() for m in overrides["extra_models"].split(",") if m.strip())
    return BackendSettings(**overrides)


def cmd_init(args: argparse.Namespace) -> int:
    names = {}
    if args.class_names:
        for i, name in enumerate(args.class_names.split(",")):
            names[i] = name.strip()
    idx = init_dataset(args.dataset, name=args.name, class_names=names or None,
                       num_keypoints=args.num_keypoints)
    print(f"[ok] 数据集已就绪: {idx.root}")
    print("     images/ 素材池（姿态库与提示词库共用）· labels/ qc/ exports/")
    print("     .autolabel/index.json 姿态库 · .autolabel/objects/index.json 提示词库")
    print(f"     类别 {idx.class_names} · 关键点数 {idx.num_keypoints}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from .batches import attach_items, batch_name, create_batch, finish_batch

    batch = create_batch(args.dataset,
                         name=batch_name("image", [os.path.basename(str(s)) for s in args.sources],
                                         args.batch_name or ""),
                         kind="image", source="cli", lib=args.lib,
                         params={"dedupe": args.dedupe, "max_side": args.max_side,
                                 "subdir": args.subdir},
                         uploads=args.sources)
    stats = import_images(args.dataset, args.sources, recursive=not args.no_recursive,
                          max_side=args.max_side, jpeg_quality=args.jpeg_quality, dedupe=args.dedupe,
                          subdir=args.subdir, lib=args.lib)
    attach_items(args.dataset, batch["id"], [rel for _src, rel in stats["imported_files"]],
                 lib=args.lib)
    finish_batch(args.dataset, batch["id"], "done", _import_note(stats["imported"], stats.get("reused")))
    print(f"[ok] 扫描 {stats['scanned']} · 导入 {stats['imported']} · "
          f"重复跳过 {stats['skipped_exact']} · 近似重复跳过 {stats['skipped_similar']} · "
          f"复用素材池 {stats.get('reused', 0)} · 失败 {stats['failed']}"
          f"（入库到 {lib_name(args.lib)}）")
    print(f"     批次 {batch['id']}（用 batches 命令或复核界面「批次管理」可整批删除）")
    return 0


def _import_note(imported: int, reused: object = 0, *, unit: str = "张", verb: str = "导入") -> str:
    """批次备注：入库总数 = 新落盘素材 + 复用素材池里已有的文件。"""
    reused_n = int(reused or 0)
    total = int(imported) + reused_n
    return f"{verb} {total} {unit}" + (f"（复用素材池 {reused_n}）" if reused_n else "")


def cmd_video(args: argparse.Namespace) -> int:
    from .video import extract_frames

    from .batches import attach_items, batch_name, create_batch, finish_batch

    batch = create_batch(args.dataset,
                         name=batch_name("video", [os.path.basename(args.video)], args.batch_name or ""),
                         kind="video", source="cli", lib=args.lib,
                         params={"fps": args.fps, "max_frames": args.max_frames,
                                 "scene": args.scene, "dedupe": args.dedupe},
                         uploads=[args.video])
    result = extract_frames(args.video, args.dataset, fps=args.fps, max_frames=args.max_frames,
                            scene=args.scene, start=args.start, duration=args.duration,
                            max_side=args.max_side, dedupe=args.dedupe, subdir=args.subdir,
                            lib=args.lib)
    attach_items(args.dataset, batch["id"], [rel for _src, rel in result["imported_files"]],
                 lib=args.lib)
    finish_batch(args.dataset, batch["id"], "done",
                 _import_note(result["imported"], result.get("reused"), unit="帧", verb="抽帧导入"))
    print(f"[ok] 视频抽帧：抽出 {result['extracted']} 帧（模式 {result['mode']}）→ "
          f"导入 {result['imported']} 帧，复用素材池 {result.get('reused', 0)} 帧，"
          f"重复跳过 {result['skipped_exact']}/相似 {result['skipped_similar']}"
          f"（入库到 {lib_name(args.lib)}）")
    info = result.get("video_info", {})
    if info.get("duration"):
        print(f"     视频时长 {info['duration']:.1f}s · {info.get('width')}x{info.get('height')} · "
              f"{info.get('fps') and round(info['fps'], 2)} fps")
    print(f"     批次 {batch['id']}（用 batches 命令或复核界面「批次管理」可整批删除）")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    settings = load_settings(args)
    print(f"[info] 模型 {settings.model} · device={settings.device} · imgsz={settings.imgsz} · "
          f"conf={settings.conf} · flip_tta={settings.flip_tta} · 可见性策略={settings.visibility_policy}")
    if settings.visibility_policy == "threshold" and settings.occluded_conf < settings.visible_conf:
        print("[tip] v=1 的关键点同样参与训练监督；只想训高置信点请设 --occluded-conf 等于 --visible-conf")
    from .pipeline import label_dataset

    stats = label_dataset(args.dataset, settings, only_missing=not args.force, limit=args.limit,
                          render_qc=args.render_qc, dry_run=args.dry_run)
    print(f"[ok] 标注完成 {stats['labeled']} 张 · 人数 {stats['people']} · 无人图 {stats['empty']} · "
          f"带标记 {stats['flagged']} · 失败 {stats['failed']}")
    if stats.get("img_per_sec"):
        print(f"     速度 {stats['img_per_sec']} img/s（{stats['elapsed_sec']}s）")
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    from .qc import render_overlays, write_report

    if args.render:
        outputs = render_overlays(args.dataset, limit=args.render, only_flagged=not args.render_all)
        print(f"[ok] 渲染 {len(outputs)} 张叠加图 -> {os.path.join(args.dataset, 'qc', 'overlays')}")
    stats = write_report(args.dataset, top_n=args.top)
    print(f"[ok] 报告 -> {stats['report']}")
    print(f"     图片 {stats['images']} · 人数 {stats['persons_total']} · 待复核 {stats['flagged_count']}")
    top = ", ".join(f"{f}={c}" for f, c in list(stats["flag_counts"].items())[:6])
    if top:
        print(f"     标记分布：{top}")
    if args.json:
        print(json.dumps({k: stats[k] for k in
                          ("images", "persons_total", "status_counts", "flag_counts", "box_height_px")},
                         ensure_ascii=False, indent=2))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from .review import serve

    serve(args.dataset, host=args.host, port=args.port, open_browser=not args.no_browser,
          max_upload_mb=args.max_upload_mb)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_dataset

    exclude = [] if args.include_all else ["rejected"]
    if args.only_approved:
        exclude = [s for s in ("unreviewed", "rejected") if s not in (args.keep_status or [])]
    result = export_dataset(args.dataset, out=args.out, val_ratio=args.val_ratio, seed=args.seed,
                            exclude_status=exclude, max_empty_ratio=args.max_empty_ratio,
                            mode=args.mode, tag=args.tag, coco=args.coco, coords=args.coords)
    print(f"[ok] 导出 -> {result['out']}")
    print(f"     train {result['train_images']} 张 / {result['people']['train']} 人 · "
          f"val {result['val_images']} 张 / {result['people']['val']} 人 · 跳过 {result['skipped']}")
    print(f"     data.yaml: {os.path.join(result['out'], 'data.yaml')}")
    if result.get("coords"):
        print(f"     坐标文件: {result['coords']}（coords.csv / coords_long.csv / coords.jsonl）")
    print(f"     训练命令: yolo pose train data={os.path.join(result['out'], 'data.yaml')} "
          f"model=yolo11s-pose.pt epochs=100 imgsz=640")
    return 0


def _exclude_from_args(args: argparse.Namespace) -> List[str]:
    """把 --only-approved / --include-all / --keep-status 折算成要排除的状态。"""
    exclude = [] if getattr(args, "include_all", False) else ["rejected"]
    if getattr(args, "only_approved", False):
        exclude = [s for s in ("unreviewed", "rejected")
                   if s not in (getattr(args, "keep_status", None) or [])]
    return exclude


def cmd_coords(args: argparse.Namespace) -> int:
    from .coords import DEFAULT_FORMATS, write_coordinate_files

    formats = [f.strip() for f in (args.formats or ",".join(DEFAULT_FORMATS)).split(",") if f.strip()]
    out = getattr(args, "coords_out", None) or getattr(args, "out", None)
    result = write_coordinate_files(args.dataset, formats=formats,
                                    exclude_status=_exclude_from_args(args),
                                    require_label=not args.include_unlabeled,
                                    out=out, tag=getattr(args, "tag", None))
    meta = result["meta"]
    print(f"[ok] 坐标导出 -> {result['out']}")
    for name in sorted(result["files"]):
        print(f"     - {name}")
    print(f"     图片 {result['images']} 张 · 人 {result['persons']} 个 · "
          f"关键点 {result['persons'] * meta['num_keypoints']} 个 · "
          f"已排除状态 {meta['filter']['exclude_status'] or '无'}")
    print("     提示：v=0 不参与 YOLO 监督，v=1（遮挡）与 v=2（可见）都参与")
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """一条命令：视频/图片入库 → 自动标注 → QC 报告 → 坐标文件导出。"""
    from .dataset import index_path, init_dataset

    root = args.dataset
    if not os.path.exists(index_path(root)):
        init_dataset(root, name=args.name)
        print(f"[1/4] 数据集已创建: {root}")

    videos = [s for s in args.sources
              if os.path.splitext(os.path.expanduser(s))[1].lower() in VIDEO_EXTS]
    stills = [s for s in args.sources if s not in videos]

    from .batches import attach_items, batch_name, create_batch, finish_batch
    from .library import LIB_POSE, lib_name

    lib = getattr(args, "lib", LIB_POSE) or LIB_POSE
    batch = create_batch(
        root, name=batch_name("video" if videos and not stills else "image",
                              [os.path.basename(str(s)) for s in args.sources], args.batch_name or ""),
        kind="video" if videos and not stills else "image", source="cli", lib=lib,
        params={"fps": args.fps, "max_frames": args.max_frames, "dedupe": args.dedupe,
                "video_dedupe": args.video_dedupe},
        uploads=args.sources)
    imported: List[str] = []
    if videos:
        from .video import extract_frames

        for video in videos:
            result = extract_frames(video, root, fps=args.fps, max_frames=args.max_frames,
                                    scene=args.scene, start=args.start, duration=args.duration,
                                    max_side=args.max_side, dedupe=args.video_dedupe,
                                    subdir=args.subdir, lib=lib)
            imported.extend(rel for _src, rel in result["imported_files"])
            print(f"[1/4] 视频 {os.path.basename(video)}：抽出 {result['extracted']} 帧 → "
                  f"入库 {len(result['imported_files'])} 帧（近似重复跳过 {result['skipped_similar']}）")
    if stills:
        stats = import_images(root, stills, recursive=not args.no_recursive, max_side=args.max_side,
                              jpeg_quality=args.jpeg_quality, dedupe=args.dedupe, subdir=args.subdir,
                              lib=lib)
        imported.extend(rel for _src, rel in stats["imported_files"])
        print(f"[1/4] 图片：扫描 {stats['scanned']} · 入库 {len(stats['imported_files'])} 张"
              f"（新素材 {stats['imported']} · 复用素材池 {stats.get('reused', 0)}）· "
              f"重复跳过 {stats['skipped_exact'] + stats['skipped_similar']} · 失败 {stats['failed']}")
    attach_items(root, batch["id"], imported, lib=lib)
    print(f"      批次 {batch['id']}（{lib_name(lib)}）")

    if lib != LIB_POSE:
        # 提示词库不跑 YOLO-pose：入库即收尾，骨骼标注 / QC / 骨骼坐标是姿态库那套流水线
        finish_batch(root, batch["id"], "done", _import_note(len(imported)))
        print(f"[2/4] {lib_name(lib)}不跑 YOLO-pose，已跳过骨骼标注、QC 与骨骼坐标")
        print(f"      下一步：python3 -m autolabel prompt {root} --fields <字段名> --vlm <多模态配置> --coords")
        return 0

    settings = load_settings(args)
    print(f"[2/4] 标注：模型 {settings.model} · device={settings.device} · "
          f"imgsz={settings.imgsz} · conf={settings.conf}")
    from .pipeline import label_dataset

    stats = label_dataset(root, settings, only_missing=not args.force, limit=args.limit,
                          render_qc=args.render_qc, dry_run=args.dry_run, verbose=False)
    print(f"      完成 {stats['labeled']} 张 · 人数 {stats['people']} · 无人图 {stats['empty']} · "
          f"带标记 {stats['flagged']} · 失败 {stats['failed']}")
    if stats.get("img_per_sec"):
        print(f"      速度 {stats['img_per_sec']} img/s（{stats['elapsed_sec']}s）")

    from .qc import write_report

    qc_stats = write_report(root)
    print(f"[3/4] QC 报告 -> {qc_stats['report']}（待复核 {qc_stats['flagged_count']} 张）")
    print("[4/4] 导出坐标文件")
    finish_batch(root, batch["id"], "done",
                 f"入库 {len(imported)} 张 · 标注 {stats['labeled']} 张")
    return cmd_coords(args)


def cmd_batches(args: argparse.Namespace) -> int:
    """列出批次，或删除指定批次的图片与记录。"""
    from .batches import UNGROUPED_ID, delete_batches, list_batches

    lib = "" if args.lib == "all" else args.lib

    if args.delete_images:
        from .batches import delete_images

        try:
            result = delete_images(args.dataset, args.delete_images, lib=lib or "pose",
                                   mode=args.mode, keep_files=args.keep_files)
        except ValueError as exc:
            # 文件名是手打的，拼错很常见，别甩整段回溯给用户
            print(f"[error] {exc}")
            return 2
        where = "两个库都摘" if result["mode"] == "purge" else f"仅{lib_name(result['lib'])}"
        kept = result["mode"] == "lib" or result["keep_files"]
        scope = "标签/复核历史，素材文件保留" if kept else "标签/复核历史 + 素材文件"
        print(f"[ok] 移除图片 {result['items']} 张（{where}）· 库引用 {result['removals']} 处 · "
              f"清理文件 {result['files_removed']} 个（范围：{scope}）· "
              f"涉及批次 {len(result['batches_updated'])} 个")
        if result.get("batches_emptied"):
            print(f"     这些批次已删空，记录一并移除：{'、'.join(result['batches_emptied'])}")
        if result.get("missing"):
            print(f"[warn] 索引里找不到：{'、'.join(result['missing'])}")
        if result.get("snapshot"):
            print(f"     快照 -> {result['snapshot']}（索引字段 + 标签原文，可照此恢复）")
        if result.get("coords"):
            print(f"     坐标文件已重算 -> {result['coords']}")
        return 0

    if args.delete:
        result = delete_batches(args.dataset, args.delete, keep_files=args.keep_files,
                                force=args.force)
        scope = "保留文件" if args.keep_files else "标签/QC 叠加图/复核历史"
        print(f"[ok] 删除批次 {len(result['batches'])} 个 · 移除图片 {result['items']} 张 · "
              f"清理文件 {result['files_removed']} 个（范围：{scope}，"
              f"素材只在另一个库不再引用时才删）")
        if result.get("snapshot"):
            print(f"     快照 -> {result['snapshot']}（索引字段 + 标签原文，可照此恢复）")
        if result.get("coords"):
            print(f"     坐标文件已重算 -> {result['coords']}")
        return 0

    rows = list_batches(args.dataset, lib=lib)
    if args.json:
        print(json.dumps({"batches": rows, "libs": lib or "all"}, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("还没有批次记录：import / video / auto 与复核界面上传都会自动登记")
        return 0
    scope = "两个库合计" if not lib else lib_name(lib)
    print(f"共 {len(rows)} 个批次（{scope}）")
    print(f"{_pad('批次 ID', 26)} {_pad('库', 10)} {_pad('状态', 9)} {_pad('图片', 5)} "
          f"{_pad('已标', 5)} {_pad('人数', 5)} {_pad('检测框', 7)}  来源 · 名称")
    for row in rows:
        print(f"{_pad(row['id'], 26)} "
              f"{_pad(row.get('module_name') or row.get('module') or '-', 10)} "
              f"{_pad(row['status'], 9)} {_pad(row['items'], 5)} {_pad(row['labeled'], 5)} "
              f"{_pad(row['people'], 5)} {_pad(row.get('prompt_boxes') or 0, 7)}  "
              f"[{row['source']}/{row['kind']}] {row['name']}")
    print(f"\n整批删除：batches {args.dataset} --delete <批次 ID>"
          f"（未分组用 {UNGROUPED_ID} 一次清两个库，或 {UNGROUPED_ID}:objects；"
          f"加 --keep-files 只删记录、--force 强删僵尸批次）")
    print(f"按张删除：batches {args.dataset} --delete-images images/a.jpg --lib objects "
          f"[--mode lib|purge]（lib 只摘本库引用，purge 连素材文件一起删）")
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    """素材池：看有哪些共享素材、把已有素材添加进某个库。"""
    from .dataset import adopt_from_pool, pool_items

    rows = pool_items(args.dataset)
    if args.add:
        stats = adopt_from_pool(args.dataset, args.add, lib=args.lib)
        print(f"[ok] 添加到{lib_name(stats['lib'])}：新增 {stats['added']} 张 · "
              f"本来就在库里 {stats['already']} 张 · 池里查无 {len(stats['missing'])} 张")
        if stats["missing"]:
            print(f"[warn] 找不到：{'、'.join(stats['missing'][:10])}")
        return 0

    shared = sum(1 for row in rows if row["in_pose"] and row["in_objects"])
    orphan = sum(1 for row in rows if not row["in_pose"] and not row["in_objects"])
    print(f"素材池共 {len(rows)} 个文件 · 姿态库 {sum(1 for r in rows if r['in_pose'])} · "
          f"提示词库 {sum(1 for r in rows if r['in_objects'])} · 两库共用 {shared} · 无人引用 {orphan}")
    for row in rows[:args.limit]:
        mark = ("姿态" if row["in_pose"] else "") + ("/提示词" if row["in_objects"] else "")
        size = f"{row['width']}x{row['height']}" if row["width"] else "未知"
        size_bytes = f"{row['bytes']:,} B"
        print(f"  {_pad(row['image'], 44)} {_pad(size, 11)} {_pad(size_bytes, 12)}  "
              f"在 {mark or '（没被任何库引用）'}")
    if len(rows) > args.limit:
        print(f"  …… 还有 {len(rows) - args.limit} 个，--limit 提高条数")
    print(f"\n添加：pool {args.dataset} --add a.jpg --add b.jpg --lib objects")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """把老布局（提示词库和姿态库混在一本索引）拆成两本库。"""
    from .library import migrate

    result = migrate(args.dataset)
    if not (result["moved_items"] or result["moved_batches"] or result["moved_dirs"]
            or result["stripped_batch_ids"] or result["created"]):
        print("[ok] 已经是分库布局，无需迁移")
        return 0
    extra = (f" · 清掉失效的批次归属 {result['stripped_batch_ids']} 条"
             if result["stripped_batch_ids"] else "")
    print(f"[ok] 已迁移：索引条目 {result['moved_items']} 条 · "
          f"批次记录 {result['moved_batches']} 个{extra}")
    for line in result["created"]:
        print(f"     新建 {line}")
    for line in result["moved_dirs"]:
        print(f"     目录 {line}")
    print("     提示词库现在在 .autolabel/objects/ 与 prompt_labels/，素材仍在 images/ 共用")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """列出并管理姿态权重与多模态大模型配置；不带参数就是清单。"""
    from . import models as model_admin

    failed = 0
    for name in args.download_weight:
        print(f"[info] 下载权重 {name}（优先走镜像，已有半截文件会续传）…")
        task = model_admin.start_download(name, force=args.force, background=False)
        mark = "[ok]" if task.get("ok") else "[warn]"
        print(f"{mark} {name}：{task.get('message')}{'（' + task['error'] + '）' if task.get('error') else ''}")
        failed += 0 if task.get("ok") else 1

    for raw in args.add_vlm:
        try:
            profile = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[error] --add-vlm 不是合法 JSON：{exc}")
            return 2
        saved = model_admin.upsert_profile(profile)
        print(f"[ok] 已保存多模态配置「{saved['name']}」 -> {saved['id']}")

    for profile_id in args.test_vlm:
        result = model_admin.test_saved_profile(profile_id)
        mark = "[ok]" if result["ok"] else "[warn]"
        print(f"{mark} 连通性 {profile_id}：{result['message']}（{result['latency_ms']} ms）")
        failed += 0 if result["ok"] else 1

    if args.default_vlm:
        model_admin.set_default(args.default_vlm)
        print(f"[ok] 默认多模态配置 -> {args.default_vlm}")

    for profile_id in args.delete_vlm:
        model_admin.delete_profile(profile_id)
        print(f"[ok] 已删除多模态配置 {profile_id}")

    for name in args.delete_weight:
        if not args.yes:
            print(f"[warn] 未删除 {name}：删权重会触发下次标注重新下载，确认请加 --yes")
            failed += 1
            continue
        info = model_admin.delete_weight(name)
        print(f"[ok] 已删除权重 {info['name']}（释放 {info['freed_mb']} MB，含 {len(info['removed'])} 个文件）")

    # 先登记再写备注：--register 新生成的 ID 也能在同一句命令里直接备注
    if args.register:
        try:
            saved = model_admin.register_local_model(
                path=args.register, capability=args.capability or "pose",
                display_name=args.display_name or "", origin=args.origin or "custom",
                model_id=args.id or "")
        except (ValueError, FileNotFoundError) as exc:
            print(f"[error] 登记本地模型失败：{exc}")
            return 2
        if saved.get("already_in_catalog"):
            print(f"[ok] {os.path.basename(args.register)} 本来就在目录里（{saved['id']}），"
                  f"直接 --note {saved['id']}=说明 更合适")
        else:
            print(f"[ok] 已登记本地模型 {saved['id']} -> {saved['path']}")

    for item in args.note:
        model_id, _, text = str(item).partition("=")
        if not model_id.strip() or "=" not in str(item):
            print(f"[error] --note 要写成 模型ID=备注文字，收到 {item}")
            return 2
        try:
            saved = model_admin.set_model_note(model_id.strip(), text)
        except KeyError as exc:
            print(f"[error] {exc}")
            return 2
        print(f"[ok] 已备注 {saved['id']}：{saved['note'] or '（已清空）'}")

    for model_id in args.forget:
        try:
            model_admin.forget_model(model_id)
        except KeyError as exc:
            print(f"[error] {exc}")
            failed += 1
            continue
        print(f"[ok] 已取消登记 {model_id}（文件保留在原地）")

    if args.default:
        try:
            defaults = model_admin.set_model_default(args.default, args.capability or "")
        except (KeyError, ValueError) as exc:
            print(f"[error] {exc}")
            return 2
        print(f"[ok] 默认模型 -> {args.default}"
              f"（当前各类默认：{'、'.join(f'{k}={v}' for k, v in defaults['ids'].items() if v)}）")

    if args.info:
        try:
            row = model_admin.describe_model(args.info, dataset_root=args.dataset or "")
        except KeyError as exc:
            print(f"[error] {exc}")
            return 2
        _print_model_detail(row, model_admin)
        return 1 if failed else 0

    payload = model_admin.vlm_payload(dataset_root=args.dataset or "")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if failed else 0

    print(f"姿态权重目录：{os.path.abspath(model_admin.weights_dir())}")
    print(f"模型登记表：  {payload['model_config_path']}（登记路径、默认模型、备注、下载记录）")
    print(f"API Key 文件：{payload['secrets_path']}（Key 只存这里，配置里只留引用名）")

    if args.browse or args.catalog:
        _print_catalog(payload)
    else:
        _print_model_rows(payload, args, model_admin)

    vlm = payload["vlm"]
    print(f"\n多模态大模型配置：{vlm['config_path']}")
    if not vlm["profiles"]:
        print("  还没有配置：在复核界面「模型管理」里添加，或用 --add-vlm 传 JSON 对象")
    else:
        print(f"{'配置 ID':<16} {'类型':<6} {'默认':<5} {'名称':<18} {'服务地址':<38} {'模型':<24} Key · 最近测试")
        for row in vlm["profiles"]:
            test = row.get("last_test") or {}
            tested = ("可用 " + str(test.get("latency_ms")) + "ms") if test.get("ok") \
                else ("不可用" if test else "未测试")
            kind = "本地" if row["kind"] == model_admin.KIND_LOCAL else "云端"
            key = row["api_key_masked"] or "未设置"
            print(f"{row['id']:<16} {kind:<6} {'默认' if row['id'] == vlm['default'] else '-':<5} "
                  f"{row['name']:<18} {row['base_url']:<38} {row['model']:<24} {key} · {tested}")
    if failed:
        print(f"\n[warn] 有 {failed} 项操作未成功，详见上面的输出")
    return 1 if failed else 0


def _display_width(text: str) -> int:
    """显示宽度：中日韩字符占两列，表格才对得齐。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(text))


def _pad(text: object, width: int) -> str:
    return str(text) + " " * max(0, width - _display_width(text))


def _model_size(row: dict) -> str:
    """体积列：本地有多少说多少，没下载就说预计值。"""
    if row.get("size_mb"):
        return f"{row['size_mb']:.1f} MB"
    if row.get("expected_mb"):
        return f"约 {row['expected_mb']:.1f} MB"
    return "-"


def _model_hints(row: dict) -> str:
    """备注行之外的临时提示：默认标记、半截文件、不兼容原因。"""
    hints: List[str] = []
    if row.get("default"):
        hints.append("默认")
    if row.get("partial_mb"):
        hints.append(f"残留半截 {row['partial_mb']} MB，重下可续传")
    if not row.get("compatible", True):
        hints.extend(row.get("compatibility_reasons") or [])
    return " · ".join(hints)


def _print_model_rows(payload: dict, args: argparse.Namespace, model_admin) -> None:
    """按能力分组打印统一清单：中文名、状态、体积、最后使用、备注。"""
    rows = payload.get("models") or []
    if args.capability:
        rows = [row for row in rows if row.get("capability") == args.capability]
    for capability in model_admin.CAPABILITIES:
        group = [row for row in rows if row.get("capability") == capability]
        if not group:
            continue
        print(f"\n{model_admin.capability_label(capability)}")
        print(f"{_pad('模型 ID', 18)}{_pad('中文名', 22)}{_pad('状态', 12)}"
              f"{_pad('体积', 14)}{_pad('最后使用', 20)}备注 / 提示")
        for row in group:
            line = (f"{_pad(row.get('id', ''), 18)}{_pad(row.get('display_name', ''), 22)}"
                    f"{_pad(row.get('status_label', ''), 12)}{_pad(_model_size(row), 14)}"
                    f"{_pad(row.get('last_used') or '-', 20)}")
            hints = _model_hints(row)
            print(line + (hints or (row.get("note") or "-")))
            if row.get("note") and hints:
                print(" " * 76 + "备注：" + str(row["note"]))
    print("\n看单个模型的完整中文说明：models --info <模型 ID>")
    print("写备注说明：            models --note <模型 ID>=说明")
    print("看可下载目录：          models --browse")


def _print_catalog(payload: dict) -> None:
    """可下载模型目录：每个模型一段中文说明，含参数、预计体积与下载命令。"""
    rows = payload.get("catalog") or []
    print(f"\n可下载模型目录（{len(rows)} 个）：先在这里挑，再下载，未下载的模型不能在上传弹层里直接选")
    for row in rows:
        params = f"{row['params_m']:.1f}M 参数" if row.get("params_m") else "参数官方未列明"
        accuracy = (f"，mAP50-95 约 {row['map_pose']}" if row.get("map_pose") else "")
        size = f"预计 {row.get('expected_mb', 0):.1f} MB" if row.get("expected_mb") else "体积未知"
        mark = row.get("status_label", "")
        print(f"\n{_pad(row.get('id', ''), 18)}{_pad(row.get('display_name', ''), 22)}"
              f"{_pad(mark, 12)}{_pad(params, 18)}{size}")
        print(f"  说明：{row.get('description') or '（目录里没写说明）'}")
        print(f"  适合：{row.get('recommended_for') or '（未标注适用场景）'}"
              f" ｜ 速度：{row.get('speed') or '未标注'}"
              f" ｜ 关键点：{row.get('num_keypoints', 17)} 个（{row.get('keypoint_set') or '未标注'}）"
              f" ｜ 精度：{params}{accuracy if accuracy else '，精度官方未列明'}")
        if row.get("note"):
            print(f"  备注：{row['note']}")
        print(f"  下载：python -m autolabel models --download-weight {row.get('filename')}")
    print("\n下载完成后回到「导入素材」的模型下拉框即可选用。")


def _print_model_detail(row: dict, model_admin) -> None:
    """单个模型的中文详情。"""
    records = model_admin.download_records(str(row.get("id") or ""), limit=3)
    fields = [
        ("模型 ID", row.get("id") or "-"),
        ("中文名", row.get("display_name") or "-"),
        ("能力", row.get("capability_label") or "-"),
        ("来源 / 出处", f"{row.get('source_label', '-')} / {row.get('origin_label', '-')}"),
        ("状态", row.get("status_label") or "-"),
        ("本地路径", row.get("local_path") or row.get("path") or "（还没下载到本地）"),
        ("实际体积", f"{row['size_mb']:.1f} MB" if row.get("size_mb") else "（无文件）"),
        ("预计体积", f"约 {row['expected_mb']:.1f} MB" if row.get("expected_mb") else "未标注"),
        ("参数规模", f"约 {row['params_m']}M" if row.get("params_m") else "官方未列明"),
        ("精度", f"mAP50-95 约 {row['map_pose']}" if row.get("map_pose") else "官方未列明"),
        ("关键点", f"{row.get('num_keypoints', 17)} 个（{row.get('keypoint_set') or '未标注'}）"),
        ("速度", row.get("speed") or "未标注"),
        ("适合场景", row.get("recommended_for") or "未标注"),
        ("默认模型", "是" if row.get("default") else "否"),
        ("最后使用", row.get("last_used") or "还没用过"),
        ("兼容性", "与当前数据集兼容" if row.get("compatible", True)
         else "；".join(row.get("compatibility_reasons") or ["不兼容"])),
        ("备注说明", row.get("note") or "（空）"),
    ]
    if row.get("url"):
        fields.append(("下载地址", f"{row['url']}（镜像 {row.get('mirror') or '直连'}）"))
    if row.get("sha256"):
        fields.append(("SHA256", f"{row['sha256'][:16]}…"))
    for label, value in fields:
        print(f"{_pad(label, 12)}：{value}")
    print(f"{'说明':<10}：{row.get('description') or '（目录里没写说明）'}")
    if row.get("metadata"):
        print(f"{'加载实测':<10}：{row['metadata']}")
    if records:
        print("最近下载记录：")
        for item in records:
            mark = "成功" if item.get("ok") else "失败"
            hashed = ("，SHA256 " + ("一致" if item.get("sha256_match") else "不一致")) \
                if item.get("sha256_match") is not None else ""
            print(f"  {item.get('finished_at') or item.get('started_at')} {mark} "
                  f"{item.get('size_mb', 0)} MB，用时 {item.get('seconds', 0)}s，"
                  f"临时文件 {item.get('temp_name', '-')}{hashed}"
                  + (f"，{item.get('error')}" if item.get("error") else ""))


def cmd_fields(args: argparse.Namespace) -> int:
    """字段登记表：提示词标注要标哪些物品、每个物品用什么描述提示词。"""
    from .fields import delete_field, fields_payload, find_field, upsert_field

    for item in args.add or []:
        name, _, hint = str(item).partition("=")
        field = upsert_field(args.dataset, name=name, prompt_hint=hint)
        note = field["prompt_hint"] or "（还没写描述提示词，模型只能凭字段名猜）"
        print(f"[ok] 字段 {field['name']}（类别号 {field['class_id']}）：{note}")
    for item in args.set_label or []:
        name, _, label = str(item).partition("=")
        hit = find_field(args.dataset, name)
        if hit is None:
            raise ValueError(f"未知字段 {name}，先用 --add {name}=描述 新建")
        field = upsert_field(args.dataset, name=hit["name"], label=label,
                             prompt_hint=hit.get("prompt_hint", ""))
        print(f"[ok] 字段 {field['name']} 的显示名改为「{field['label']}」")
    for item in args.delete or []:
        result = delete_field(args.dataset, item, drop_boxes=args.drop_boxes)
        print(f"[ok] 已删除字段 {result['name']}（类别号 {result['class_id']}），"
              f"连带删掉 {result['boxes_removed']} 个框，改了 {result['labels_touched']} 个标签文件")

    payload = fields_payload(args.dataset)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not payload["fields"]:
        print("还没有字段。新建一个：autolabel fields <数据集> --add \"bottle=桌上或手里的水瓶、纸杯、易拉罐\"")
        print("或者在网页「提示词标注」页里填字段名 + 描述提示词。")
        return 0
    print(f"字段登记表 {payload['path']}（{len(payload['fields'])} 个字段）")
    for row in payload["fields"]:
        hint = row["prompt_hint"] or "（没写描述提示词）"
        print(f"  [{row['class_id']:>2}] {row['name']:<14} 显示名 {row['label']:<10} "
              f"框 {row['boxes']:>5}  {hint[:60]}")
    if payload["orphans"]:
        print(f"[warn] 标签里还有登记表里没有的类别号 {payload['orphans']}："
              "多半是删字段时没加 --drop-boxes，导出时这些框会被跳过")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    """提示词标注：让多模态大模型按字段描述找物品，产出 YOLO detection 框。"""
    from .batches import batch_name, create_batch, finish_batch
    from .fields import resolve_fields
    from .prompt import (
        export_detection_dataset,
        prompt_summary,
        resolve_profile,
        run_prompt_labeling,
        write_object_coordinate_files,
    )

    wanted: List[str] = []
    for item in args.fields or []:
        wanted += [piece.strip() for piece in str(item).split(",") if piece.strip()]
    images: List[str] = []
    for item in args.images or []:
        images += [piece.strip() for piece in str(item).split(",") if piece.strip()]

    if args.status:
        summary = prompt_summary(args.dataset)
        print(f"提示词标注：已标 {summary['labeled']}/{summary['images']} 张 · 框 {summary['boxes']} · "
              f"已确认 {summary['reviewed']} · 没标过 {summary['unreviewed']}")
        if summary["per_field"]:
            print("  字段分布 " + " · ".join(f"{k}={v}" for k, v in
                                            sorted(summary["per_field"].items(),
                                                   key=lambda kv: (-kv[1], kv[0]))[:12]))
        if summary["status_counts"]:
            print("  状态 " + " · ".join(f"{k}={v}" for k, v in sorted(summary["status_counts"].items())))
        if summary["flag_counts"]:
            print("  标记 " + " · ".join(f"{k}={v}" for k, v in sorted(summary["flag_counts"].items())))
        return 0

    fields = resolve_fields(args.dataset, wanted)
    profile = resolve_profile(args.vlm or "")
    names = [str(field.get("name")) for field in fields]
    batch = create_batch(args.dataset,
                         name=batch_name("prompt", names, args.batch_name or ""),
                         kind="prompt", source="cli",
                         params={"prompt_fields": names, "vlm_profile": profile["id"],
                                 "overwrite": args.overwrite, "targets": len(images)},
                         uploads=names)
    print(f"[info] 用多模态大模型 {profile.get('name')}（{profile.get('model')}）"
          f"标字段 {'、'.join(names)}；逐图一次请求，比 YOLO 慢很多属正常")
    try:
        stats = run_prompt_labeling(args.dataset, fields=fields, profile=profile,
                                    images=images or None, limit=args.limit or 0,
                                    overwrite=args.overwrite, extra_hint=args.prompt_extra or "",
                                    verbose=not args.quiet)
    except Exception as exc:  # noqa: BLE001 批次要留失败原因
        finish_batch(args.dataset, batch["id"], "failed", str(exc))
        raise
    result = {"coords": "", "export": ""}
    if args.coords:
        written = write_object_coordinate_files(args.dataset, tag=f"cli-{batch['id']}")
        result["coords"] = str(written["out"])
        print(f"[ok] 坐标文件：{result['coords']}（{written['images']} 张 / {written['boxes']} 个框）")
    if args.export:
        exported = export_detection_dataset(args.dataset, out=args.out or None)
        result["export"] = str(exported["out"])
        print(f"[ok] detection 训练集：{result['export']}（train {exported['train']} / "
              f"val {exported['val']}，{exported['data_yaml']}）")
    finish_batch(args.dataset, batch["id"], "done",
                 f"标注 {stats['labeled']} 张，产出 {stats['boxes']} 个框")
    print(f"[ok] 批次 {batch['id']}：标注 {stats['labeled']} 张 · 框 {stats['boxes']} · "
          f"失败 {stats['failed']} · 跳过 {stats['skipped']}")
    if stats.get("protected"):
        print(f"[info] 人工保护 {stats['protected']} 张（已接受或手动画过框，默认不覆盖；要重标加 --overwrite）")
    if stats["unknown_fields"]:
        print(f"[warn] 模型回了字段表里没有的名字 {'、'.join(stats['unknown_fields'][:8])}："
              "这些框被丢掉了，描述提示词写得更具体些会好很多")
    if stats["failed"]:
        print(f"[warn] {stats['failed']} 张失败，最后一条错误：{stats['last_error']}")
    if not args.coords:
        print("     要导出坐标文件加 --coords，直接出训练集加 --export")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    idx = load_index(args.dataset)
    num_kpt = args.num_keypoints or idx.num_keypoints
    issue_count: Dict[str, int] = {}
    samples: List[str] = []
    for rel, item in idx.ordered_items():
        issues = validate_label_file(label_path_for(args.dataset, rel),
                                    int(item.get("width", 0) or 0), int(item.get("height", 0) or 0),
                                    num_keypoints=num_kpt, class_names=idx.class_names)
        for issue in issues:
            issue_count[issue.kind] = issue_count.get(issue.kind, 0) + 1
            if len(samples) < 20:
                samples.append(f"{issue.path}:{issue.line or '-'} [{issue.kind}] {issue.detail}")
    # 目录侧检查：孤儿标签 / 缺图片
    label_root = os.path.join(os.path.abspath(args.dataset), "labels")
    indexed = {os.path.splitext(rel)[0] for rel in idx.items}
    orphans = []
    for dirpath, _dirs, files in os.walk(label_root):
        for fn in files:
            if fn.endswith(".txt"):
                stem = os.path.splitext(os.path.relpath(os.path.join(dirpath, fn), label_root))[0]
                if stem not in indexed:
                    orphans.append(os.path.relpath(os.path.join(dirpath, fn), label_root))
    if orphans:
        issue_count["orphan_label"] = len(orphans)
        samples.extend(f"孤儿标签: {o}" for o in orphans[:10])
    if issue_count:
        print("[warn] 发现问题：" + ", ".join(f"{k}={v}" for k, v in sorted(issue_count.items())))
        for line in samples:
            print("   - " + line)
        return 1
    print(f"[ok] 校验通过：{len(idx.items)} 张图，标签格式与关键点数（{num_kpt}）均一致")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    idx = load_index(args.dataset)
    from .qc import dataset_stats

    stats = dataset_stats(args.dataset)
    print(f"数据集 {idx.name}  ({idx.root})")
    print(f"  姿态库 图片 {stats['images']} · 人数 {stats['persons_total']} · "
          f"缺标签 {stats['missing_labels']}")
    if not stats["images"]:
        # 空库就别打印一堆 0 了，直接告诉用户往哪儿灌素材
        print("  姿态库还没有素材：autolabel import <数据集> --lib pose，或界面「导入素材」")
    else:
        print("  状态 " + " · ".join(f"{k}={v}" for k, v in sorted(stats["status_counts"].items())))
        print(f"  每图人数 mean={stats['persons_per_image']['mean']} "
              f"max={stats['persons_per_image']['max']} 无人={stats['persons_per_image']['empty_images']}")
        print(f"  框高度px p10={stats['box_height_px']['p10']} p50={stats['box_height_px']['p50']} "
              f"p90={stats['box_height_px']['p90']}")
    if stats["flag_counts"]:
        print("  标记 " + " · ".join(f"{k}={v}" for k, v in list(stats["flag_counts"].items())[:8]))
    settings = stats.get("label_settings") or {}
    if settings:
        print(f"  标注参数 model={settings.get('model')} imgsz={settings.get('imgsz')} "
              f"conf={settings.get('conf')} flip_tta={settings.get('flip_tta')}")
    from .prompt import prompt_summary

    prompt = prompt_summary(args.dataset)
    if not prompt["images"]:
        print("  提示词库还没有素材：autolabel import <数据集> --lib objects，"
              "或界面「导入素材」/「从素材池添加」")
        prompt = {"per_field": {}, "flag_counts": {}}
    else:
        print(f"  提示词标注 已标 {prompt['labeled']}/{prompt['images']} 张 · 框 {prompt['boxes']} · "
              f"已确认 {prompt['reviewed']} · 未标 {prompt['unreviewed']}")
    if prompt["per_field"]:
        top = sorted(prompt["per_field"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        print("  字段分布 " + " · ".join(f"{k}={v}" for k, v in top))
    if prompt["flag_counts"]:
        print("  提示词标记 " + " · ".join(f"{k}={v}" for k, v in sorted(prompt["flag_counts"].items())))

    from .dataset import pool_items

    pool = pool_items(args.dataset)
    shared = sum(1 for row in pool if row["in_pose"] and row["in_objects"])
    orphan = sum(1 for row in pool if not row["in_pose"] and not row["in_objects"])
    print(f"  素材池 {len(pool)} 个文件 · 两库共用 {shared} · 无人引用 {orphan}"
          f"（{os.path.join(os.path.abspath(args.dataset), 'images')}）")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """把复核状态退回未复核，方便重跑或换参数后重来。"""

    idx = load_index(args.dataset)
    targets = {s.strip() for s in (args.status or "").split(",") if s.strip()} or None
    changed = 0
    for _rel, item in idx.items.items():
        status = item.get("status", STATUS_UNREVIEWED)
        if status == STATUS_UNREVIEWED:
            continue
        if targets and status not in targets:
            continue
        if args.keep_approved and status == STATUS_APPROVED:
            continue
        item["status"] = STATUS_UNREVIEWED
        item.pop("reviewed_at", None)
        changed += 1
    idx.save()
    removed = 0
    if args.labels:
        for rel in list(idx.items):
            path = label_path_for(args.dataset, rel)
            if os.path.exists(path):
                os.remove(path)
                removed += 1
    print(f"[ok] 已退回 {changed} 张为未复核" + (f"，删除 {removed} 个标签文件" if args.labels else ""))
    print("     提示：标注参数变化时 label 会自动重标，无需手动 reset")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autolabel", description="人体骨骼自动标注系统（YOLO pose）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_dataset(p: argparse.ArgumentParser) -> None:
        p.add_argument("dataset", help="数据集目录")

    def add_label_args(p: argparse.ArgumentParser) -> None:
        """标注参数，label 与 auto 共用。"""
        p.add_argument("--config", help="YAML 参数文件")
        p.add_argument("--model", help="模型名或路径，如 yolo11n-pose.pt / yolo11s-pose.pt")
        p.add_argument("--extra-models", help="逗号分隔的校验模型，用于多模型共识与分歧标记")
        p.add_argument("--vlm", dest="vlm_profile", metavar="配置",
                       help="改用多模态大模型标注，取「模型管理」里的配置 ID 或名称")
        p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu", "0", "1"])
        p.add_argument("--imgsz", type=int)
        p.add_argument("--conf", type=float)
        p.add_argument("--iou", type=float)
        p.add_argument("--max-det", type=int)
        p.add_argument("--half", action="store_true", default=None)
        p.add_argument("--no-flip", dest="flip_tta", action="store_false", default=None,
                       help="关闭水平翻转融合（更快，关键点质量略降）")
        p.add_argument("--visibility-policy", choices=["threshold", "all_visible", "binary"])
        p.add_argument("--visible-conf", type=float)
        p.add_argument("--occluded-conf", type=float)
        p.add_argument("--box-mode", choices=["expand", "model", "keypoints"])
        p.add_argument("--bbox-margin", type=float)
        p.add_argument("--limit", type=int, help="只标注前 N 张，用于试跑")
        p.add_argument("--render-qc", type=int, default=0, help="同时渲染 N 张骨架叠加图")
        p.add_argument("--force", action="store_true", help="忽略断点续跑，全部重标")
        p.add_argument("--dry-run", action="store_true", help="只跑流程不写标签")

    p = sub.add_parser("init", help="创建数据集骨架")
    add_dataset(p)
    p.add_argument("--name", help="数据集名称")
    p.add_argument("--class-names", default="person", help="逗号分隔类别名，默认单类 person")
    p.add_argument("--num-keypoints", type=int, default=17, help="关键点数，COCO 为 17")
    p.set_defaults(func=cmd_init)

    def add_lib(parser: argparse.ArgumentParser, default: str = "pose") -> None:
        parser.add_argument("--lib", choices=list(LIBRARIES), default=default,
                            help="入库到哪个标注库：pose=姿态标注（默认），objects=提示词标注；"
                                 "素材文件两个库共用，images/ 里已有的直接引用，不重复落盘")

    p = sub.add_parser("import", help="导入图片（自动规范化 RGB/EXIF/尺寸并去重）")
    add_dataset(p)
    p.add_argument("sources", nargs="+", help="图片或目录")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--dedupe", choices=["none", "exact", "perceptual"], default="exact")
    p.add_argument("--subdir", default="", help="导入到 images/ 下的子目录")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    add_lib(p)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("video", help="视频抽帧导入（ffmpeg）")
    add_dataset(p)
    p.add_argument("video", help="视频文件")
    p.add_argument("--fps", type=float, default=0.5, help="采样帧率，0.5 = 每 2 秒 1 帧")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--scene", type=float, help="按场景切点抽帧的阈值（0.2~0.4），设置后忽略 --fps")
    p.add_argument("--start", type=float, default=0.0, help="起始秒")
    p.add_argument("--duration", type=float, help="截取时长（秒）")
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--dedupe", choices=["none", "exact", "perceptual"], default="perceptual")
    p.add_argument("--subdir", default="")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    add_lib(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("label", help="批量自动标注")
    add_dataset(p)
    add_label_args(p)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("qc", help="生成质量报告与复核清单")
    add_dataset(p)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--render", type=int, default=0, help="渲染 N 张叠加图")
    p.add_argument("--render-all", action="store_true", help="叠加图覆盖全部图片而非仅带标记的")
    p.add_argument("--json", action="store_true", help="额外打印 JSON 统计")
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("review", help="打开浏览器复核界面")
    add_dataset(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--max-upload-mb", type=int, default=1024, help="网页上传单文件大小上限")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("export", help="导出 YOLO 训练数据集（train/val + data.yaml）")
    add_dataset(p)
    p.add_argument("--out", help="导出目录，默认 <dataset>/exports/<时间戳>")
    p.add_argument("--tag", help="导出子目录名")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-empty-ratio", type=float, default=0.15, help="背景图占比上限")
    p.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    p.add_argument("--coco", action="store_true", help="同时导出 COCO keypoint JSON")
    p.add_argument("--coords", action="store_true", help="同时导出 CSV/JSONL 坐标文件（含 split 列）")
    p.add_argument("--only-approved", action="store_true", help="只导出已接受/已修改的样本")
    p.add_argument("--keep-status", nargs="*", default=[], help="配合 --only-approved 额外保留的状态")
    p.add_argument("--include-all", action="store_true", help="连 rejected 一起导出")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("coords", help="导出坐标文件（宽表 CSV / 长表 CSV / JSONL）")
    add_dataset(p)
    p.add_argument("--out", help="输出目录，默认 <dataset>/coords/<时间戳>")
    p.add_argument("--tag", help="输出子目录名")
    p.add_argument("--formats", default="csv,long,jsonl",
                   help="逗号分隔：csv 一行一个人 / long 一行一个关键点 / jsonl 一行一张图")
    p.add_argument("--only-approved", action="store_true", help="只导出已接受/已修改的样本")
    p.add_argument("--keep-status", nargs="*", default=[], help="配合 --only-approved 额外保留的状态")
    p.add_argument("--include-all", action="store_true", help="连 rejected 一起导出")
    p.add_argument("--include-unlabeled", action="store_true", help="缺标签的图也输出一行（objects 为空）")
    p.set_defaults(func=cmd_coords)

    p = sub.add_parser("auto", help="一条命令跑完：导入/抽帧 → 自动标注 → QC → 坐标导出")
    add_dataset(p)
    p.add_argument("sources", nargs="+", help="图片、图片目录或视频文件（按扩展名自动分流）")
    p.add_argument("--name", help="数据集名称，仅首次创建时使用")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--dedupe", choices=["none", "exact", "perceptual"], default="exact", help="图片去重方式")
    p.add_argument("--video-dedupe", choices=["none", "exact", "perceptual"], default="perceptual",
                   help="视频帧去重方式，默认感知去重以剔除近重复帧")
    p.add_argument("--subdir", default="", help="导入到 images/ 下的子目录")
    p.add_argument("--fps", type=float, default=0.5, help="视频采样帧率，0.5 = 每 2 秒 1 帧")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--scene", type=float, help="按场景切点抽帧的阈值（0.2~0.4），设置后忽略 --fps")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float)
    add_lib(p)
    add_label_args(p)
    p.add_argument("--formats", default="csv,long,jsonl", help="坐标文件格式")
    p.add_argument("--coords-out", help="坐标输出目录，默认 <dataset>/coords/<时间戳>")
    p.add_argument("--only-approved", action="store_true")
    p.add_argument("--keep-status", nargs="*", default=[])
    p.add_argument("--include-all", action="store_true")
    p.add_argument("--include-unlabeled", action="store_true")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("validate", help="校验标签格式")
    add_dataset(p)
    p.add_argument("--num-keypoints", type=int, default=0)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("status", help="数据集概览")
    add_dataset(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("batches", help="按批次查看已标注数据，或整批删除图片与记录")
    add_dataset(p)
    p.add_argument("--lib", choices=["all", *LIBRARIES], default="all",
                   help="看/删哪个库的批次，默认 all 两个库一起看")
    p.add_argument("--delete", action="append", default=[], metavar="批次ID",
                   help="删除该批次的图片与记录，可重复；未分组用 __ungrouped__")
    p.add_argument("--delete-images", action="append", default=[], metavar="图片路径",
                   help="按张删除（a.jpg 或 images/a.jpg 都认），可重复；批次记录里同步摘除，删空的批次连记录一起删")
    p.add_argument("--keep-files", action="store_true", help="只删索引与批次记录，保留图片/标签文件")
    p.add_argument("--mode", choices=["lib", "purge"], default="lib",
                   help="按张删除的粒度：lib 只摘本库引用（默认，素材留给另一个库），"
                        "purge 两个库一起摘并删素材文件")
    p.add_argument("--force", action="store_true",
                   help="忽略「仍在标注中」的保护，用于服务中断留下的僵尸批次")
    p.add_argument("--json", action="store_true", help="以 JSON 输出批次列表")
    p.set_defaults(func=cmd_batches)

    p = sub.add_parser("pool", help="素材池：查看共享素材被哪些库引用，或把已有素材添加进某个库")
    add_dataset(p)
    p.add_argument("--add", action="append", default=[], metavar="图片路径",
                   help="把素材池里的这张图添加进 --lib 指定的库（不复制文件），可重复")
    p.add_argument("--limit", type=int, default=30, help="最多列几条，默认 30")
    add_lib(p)
    p.set_defaults(func=cmd_pool)

    p = sub.add_parser("migrate", help="把老布局数据集迁到分库布局（幂等，可重复执行）")
    add_dataset(p)
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("fields", help="管理提示词标注的字段（要标什么物品、用什么描述提示词）")
    add_dataset(p)
    p.add_argument("--add", action="append", default=[], metavar="字段名=描述提示词",
                   help="新建/更新一个字段，描述提示词写清楚「什么算这个目标」，可重复")
    p.add_argument("--set-label", action="append", default=[], metavar="字段名=中文显示名",
                   help="改字段的中文显示名，可重复")
    p.add_argument("--delete", action="append", default=[], metavar="字段名",
                   help="删除字段，可重复；仍有框时需要 --drop-boxes")
    p.add_argument("--drop-boxes", action="store_true",
                   help="配合 --delete：连带删掉这个字段已经标出的框")
    p.add_argument("--json", action="store_true", help="以 JSON 输出字段表")
    p.set_defaults(func=cmd_fields)

    p = sub.add_parser("prompt", help="提示词标注：按字段描述让多模态大模型找物品，产出 YOLO detection 框")
    add_dataset(p)
    p.add_argument("--fields", action="append", default=[], metavar="字段名",
                   help="只标这些字段（逗号分隔可重复），默认用字段表里全部字段")
    p.add_argument("--images", action="append", default=[], metavar="相对路径",
                   help="只标这些图（逗号分隔可重复），默认整个数据集")
    p.add_argument("--vlm", default="", help="多模态配置名或 ID，默认用「模型管理」里设的默认配置")
    p.add_argument("--limit", type=int, default=0, help="只标前 N 张，用于试跑")
    p.add_argument("--overwrite", action="store_true",
                   help="已经标过的也重标（默认跳过同签名已标的、人工接受过的、人工画过框的图）")
    p.add_argument("--prompt-extra", default="", help="附加到提示词里的全局补充说明")
    p.add_argument("--coords", action="store_true", help="顺带导出坐标文件（CSV + JSONL）")
    p.add_argument("--export", action="store_true", help="顺带导出 YOLO detection 训练集")
    p.add_argument("--out", default="", help="配合 --export：导出目录")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    p.add_argument("--status", action="store_true", help="只看待标/已标汇总，不跑标注")
    p.add_argument("--quiet", action="store_true", help="不逐张打印进度")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("models", help="列出/管理姿态权重与多模态大模型（本地或云端）配置")
    p.add_argument("--dataset", default="",
                   help="兼容性检查用的数据集目录（比对关键点数量，判这个模型能不能用）")
    p.add_argument("--browse", action="store_true",
                   help="浏览可下载模型目录：中文说明、参数规模、预计体积、适合场景")
    p.add_argument("--catalog", action="store_true", help="同 --browse")
    p.add_argument("--info", metavar="模型ID", help="看单个模型的完整中文说明、下载地址与下载记录")
    p.add_argument("--capability", default="", help="只看一类能力：pose | vlm | custom_task")
    p.add_argument("--note", action="append", default=[], metavar="模型ID=说明",
                   help="给模型写备注说明，可重复；说明留空即清空")
    p.add_argument("--register", metavar="路径", help="登记一个本地模型/权重文件（只记路径，不动文件）")
    p.add_argument("--id", default="", help="配合 --register：指定模型 ID，默认取文件名")
    p.add_argument("--display-name", default="", help="配合 --register：模型中文名")
    p.add_argument("--origin", default="", help="配合 --register：出处 custom | huggingface | ultralytics")
    p.add_argument("--forget", action="append", default=[], metavar="模型ID",
                   help="取消登记本地模型（文件保留），可重复")
    p.add_argument("--default", metavar="模型ID", help="设为该能力（pose/vlm）的默认模型")
    p.add_argument("--download-weight", action="append", default=[], metavar="FILE",
                   help="下载或补下指定权重，可重复")
    p.add_argument("--delete-weight", action="append", default=[], metavar="FILE",
                   help="删除权重文件，可重复；需配合 --yes")
    p.add_argument("--add-vlm", action="append", default=[], metavar="JSON",
                   help="新增/更新多模态配置，取值为 JSON 对象，可重复")
    p.add_argument("--delete-vlm", action="append", default=[], metavar="配置ID",
                   help="删除多模态配置，可重复")
    p.add_argument("--default-vlm", metavar="配置ID", help="设为默认多模态配置")
    p.add_argument("--test-vlm", action="append", default=[], metavar="配置ID",
                   help="测试多模态配置连通性，可重复")
    p.add_argument("--force", action="store_true", help="配合 --download-weight：已就绪也重新下载")
    p.add_argument("--yes", action="store_true", help="确认删除权重")
    p.add_argument("--json", action="store_true", help="以 JSON 输出完整清单")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("reset", help="退回复核状态（可选同时删除标签）")
    add_dataset(p)
    p.add_argument("--status", help="只重置这些状态，逗号分隔，如 rejected,edited")
    p.add_argument("--keep-approved", action="store_true", help="保留已接受的样本")
    p.add_argument("--labels", action="store_true", help="同时删除标签文件")
    p.set_defaults(func=cmd_reset)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[info] 已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
