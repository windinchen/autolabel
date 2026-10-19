"""命令行入口：init / import / video / auto / label / detect / qc / review / launch / service /
service-run / export / coords / validate / status / batches / pool / migrate / fields / prompt /
models / settings / version / ca / reset。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .backends import BackendSettings
from .settings import DEVICE_ARGS
from . import __version__
from . import platform_utils
from .dataset import (
    ALL_STATUSES,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    import_images,
    init_dataset,
    label_path_for,
    load_index,
    missing_sources,
    status_of,
    unknown_images,
)
from .library import (
    LIB_DETECT,
    LIB_OBJECTS,
    LIB_POSE,
    LIBRARIES,
    field_key,
    label_field_keys,
    lib_name,
    normalize,
    status_key,
)
from .upload import VIDEO_EXTS
from .yolo_pose import validate_label_file


#: 表格里用的库名短标签（``lib_name`` 给的是全称，列宽只有几格放不下）
LIB_TAGS = {LIB_POSE: "姿态", LIB_OBJECTS: "提示词", LIB_DETECT: "检测"}


def load_settings(args: argparse.Namespace) -> BackendSettings:
    """命令行参数 > YAML 配置 > 系统设置 > 内置默认。

    系统设置（``autolabel settings`` / 界面「系统设置」）只兜底设备和半精度：
    命令行或 YAML 里写明了就用写明的，不会被家目录里一个文件偷偷改掉。
    """
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
    from . import settings as system

    for key, value in system.label_defaults().items():
        overrides.setdefault(key, value)
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


def _missing_sources(sources: Sequence[str]) -> Tuple[List[str], List[str]]:
    """把导入源分成「不存在」与「存在」两组（展开 ~ 与相对路径后再判断）。"""
    missing, existing = [], []
    for source in sources:
        path = os.path.abspath(os.path.expanduser(str(source)))
        (existing if os.path.exists(path) else missing).append(str(source))
    return missing, existing


def cmd_import(args: argparse.Namespace) -> int:
    from .batches import attach_items, batch_name, create_batch, finish_batch

    missing, existing = _missing_sources(args.sources)
    for source in missing:
        print(f"[warn] 路径不存在，跳过: {source}")
    if not existing:
        raise FileNotFoundError(
            f"导入源都不存在：{'、'.join(missing)}；"
            "路径打错了？相对路径请写成绝对路径或先 cd 到上级目录")
    # 先导入再建批次：一条都没入库时不该在批次表里留一个 0 张的空批次
    stats = import_images(args.dataset, existing, recursive=not args.no_recursive,
                          max_side=args.max_side, jpeg_quality=args.jpeg_quality, dedupe=args.dedupe,
                          subdir=args.subdir, lib=args.lib)
    added = [rel for _src, rel in stats["imported_files"]]
    if not added:
        # 内容重复被跳过的图有两种去处，指错一条用户就照着敲出一句不相干的报错：
        # 素材池里躺着但没进本库 —— 用 pool --add 认领；本库里已标过 —— 才是重标。
        # 而「重标」的命令三个库各不相同（label --force / prompt|detect --overwrite）。
        relabel = {"objects": "prompt", "detect": "detect"}.get(normalize(args.lib), "label")
        flag = "--force" if relabel == "label" else "--overwrite"
        print(f"[info] 扫描 {stats['scanned']} 张，没有新图入库："
              f"完全重复 {stats['skipped_exact']} · 近似重复 {stats['skipped_similar']} · "
              f"失败 {stats['failed']}")
        print(f"       素材池已有、只是没进{lib_name(args.lib)}："
              f"autolabel pool {args.dataset} --lib {args.lib} --add <图片>")
        print(f"       已在{lib_name(args.lib)}里要重标："
              f"autolabel {relabel} {args.dataset} {flag}")
        return 0
    batch = create_batch(args.dataset,
                         name=batch_name("image", [os.path.basename(str(s)) for s in args.sources],
                                         args.batch_name or ""),
                         kind="image", source="cli", lib=args.lib,
                         params={"dedupe": args.dedupe, "max_side": args.max_side,
                                 "subdir": args.subdir},
                         uploads=args.sources)
    attach_items(args.dataset, batch["id"], added, lib=args.lib)
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
    if stats.get("skipped_human"):
        print(f"     人工已复核 {stats['skipped_human']} 张未覆盖（连它们一起重标加 --force）")
    if stats.get("img_per_sec"):
        print(f"     速度 {stats['img_per_sec']} img/s（{stats['elapsed_sec']}s）")
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    from .qc import render_overlays, write_report

    if args.render:
        outputs = render_overlays(args.dataset, limit=args.render, only_flagged=not args.render_all)
        print(f"[ok] 渲染 {len(outputs)} 张叠加图 -> {os.path.join(args.dataset, 'qc', 'overlays')}")
    stats = write_report(args.dataset, top_n=args.top, include_reviewed=args.include_reviewed)
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
    from . import settings as settings_module
    from .review import local_cert_paths, serve

    # 起服务时指定的算力只对这个实例生效，不去作家目录里的全局设置，
    # 这样"一个实例跑一块卡"可行，多卡机器上开两个实例互不干扰
    if args.device:
        os.environ[settings_module.ENV_DEVICE] = args.device
    if args.half is not None:
        os.environ[settings_module.ENV_HALF] = "1" if args.half else "0"
    cert = args.ssl_cert or ""
    key = args.ssl_key or ""
    if args.no_ssl:
        cert = key = ""
    elif cert and not key:
        key = cert.replace(".pem", "-key.pem")
    elif not cert:
        cert, key = local_cert_paths()
    serve(args.dataset, host=args.host, port=args.port, open_browser=not args.no_browser,
          max_upload_mb=args.max_upload_mb, cert=cert, key=key)
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    """双击启动的核心：自检 → 起服务 → 端口真监听了才开浏览器。

    macOS 的 ``运行工作台.command`` 和 Windows 的 ``运行工作台.bat`` 都只负责把自己
    交到这里，两条启动链路共用同一份判断，不再各写一遍「等端口」「失败打日志」。
    """
    from .launch import launch

    half = True if args.half else None
    return launch(os.getcwd(), args.dataset, port=args.port, host=args.host,
                  device=args.device, half=half, open_browser=not args.no_browser,
                  wait_seconds=args.wait)


def cmd_service(args: argparse.Namespace) -> int:
    """常驻服务：注册 / 状态 / 重启 / 注销。macOS 走 launchd，Windows 走任务计划程序。"""
    from . import service

    try:
        if args.print_env:
            inst = service.instance_for(os.getcwd(), args.print_env)
            print("\n".join(inst.env_lines()))
            return 0
        if args.show:
            inst = service.instance_for(os.getcwd(), args.show)
            chosen = service.backend()
            if chosen is None:
                print("[error] 这个系统还没有常驻服务后端", file=sys.stderr)
                return 1
            print(chosen.render_unit(inst), end="")
            return 0
        if args.status:
            return service.status(os.getcwd())
        if args.restart is not None:
            return service.restart(os.getcwd(), args.restart or None)
        if args.remove is not None:
            return service.remove(os.getcwd(), args.remove or None)
        return service.install(os.getcwd(), dry_run=args.dry_run)
    except service.ServiceError as exc:
        print(f"[error] {exc.head}", file=sys.stderr)
        for line in exc.details:
            print(f"        {line}", file=sys.stderr)
        return 1


def cmd_service_run(args: argparse.Namespace) -> int:
    """真正被服务管理器拉起来的那一个实例：参数从清单现读，不开浏览器、不占窗口。

    端口被占时**必须退出码 0**：launchd 的 ``KeepAlive`` 与任务计划程序的
    「失败重启」都只认非零退出，非零就等于让它每 10 秒撞一次端口刷日志。
    """
    from . import service
    from .launch import launch

    try:
        inst = service.instance_for(os.getcwd(), args.name) if args.name else \
            service.Instance("manual", args.port or 8765, args.dataset or "datasets/demo",
                             args.device or "", bool(args.half))
    except service.ServiceError as exc:
        print(f"[error] {exc.head}", file=sys.stderr)
        for line in exc.details:
            print(f"        {line}", file=sys.stderr)
        return 1
    return launch(os.getcwd(), inst.dataset, port=inst.port, device=inst.device or None,
                  half=True if inst.half else None, open_browser=False,
                  guard_port=True, wait_seconds=args.wait,
                  pid_file=service.pid_file(inst.name))


def cmd_ca(args: argparse.Namespace) -> int:
    from . import ca

    if not ca.openssl_available():
        print("[error] 找不到 openssl 命令，签不出本地证书", file=sys.stderr)
        print(f"        {platform_utils.openssl_missing_hint()}", file=sys.stderr)
        return 1
    if args.uninstall:
        done = ca.uninstall_trust()
        print(("[ok] " if done["ok"] else "[error] ") + done["note"])
        return 0 if done["ok"] else 1
    if args.status:
        info = ca.status()
    else:
        try:
            result = ca.issue(force=args.force)
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        left = ca.days_left(ca.leaf_cert_path())
        print(f"[ok] {'已签发' if result['created'] else '沿用已有'}本地证书"
              f"（约 {left} 天到期）：{ca.leaf_cert_path()}")
        if not args.no_install:
            done = ca.install_trust()
            print(("[ok] " if done["ok"] else "[warn] ") + done["note"])
        info = ca.status()
    print(f"     目录: {info['dir']}")
    print(f"     地址: {', '.join(info['sans']) or '（无）'}")
    print(f"     有效期: {info['days_left']} 天"
          + ("（快到期，建议 autolabel ca --force 重签）" if info["expires_soon"] else ""))
    print(f"     {info.get('trust_store') or ca.trust_store_label()}: "
          + ("✓ 已设为受信任根证书" if info["trusted"] else "✗ 未信任，浏览器仍会提示不安全"))
    if info["trusted"]:
        print("     打开 https://localhost:8765/ 即可，摄像头可用")
    else:
        print("     若浏览器仍拦：完全退出浏览器再打开（它会缓存证书状态）")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    from . import version as versioning

    if args.bump:
        try:
            old, new = versioning.rewrite(versioning.bump(part=args.bump))
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        print(f"[ok] 版本号 {old} → {new}（改的是 {versioning.version_file()}）")
        print("     日常改动按约定只升 0.01；升完记得同步 demo_docs/CHANGELOG.md，再提交打 tag：")
        print(f"     git add -A && git commit -m 'v{new}' && git tag v{new} && git push origin main v{new}")
        return 0
    print(versioning.describe(versioning.status(force=args.check)))
    return 0


def cmd_settings(args: argparse.Namespace) -> int:
    from . import settings as system

    patch: Dict = {}
    if args.device:
        patch["device"] = args.device
    if args.half is not None:
        patch["half"] = args.half
    if args.reset:
        patch = dict(system.DEFAULTS)
    if patch:
        try:
            saved = system.save(patch)
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        print(f"[ok] 系统设置已存到 {system.path()}")
        print(f"     推理设备 {system.DEVICE_LABELS[saved['device']]} · 半精度 {'开' if saved['half'] else '关'}")
        print("     界面「系统设置」和命令行用的是同一份，改完即生效（正在跑的任务下次启动才算）")
        if args.device and args.device != "auto":
            facts = system.detect(force=True).get(args.device, {})
            if facts and not facts.get("available"):
                print(f"[warn] 这台机器现在用不了{facts.get('label', args.device)}："
                      f"{facts.get('reason') or '原因未知'}，跑的时候会自动降级")
        return 0
    if args.json:
        print(json.dumps(system.payload(probe=True), ensure_ascii=False, indent=2))
        return 0
    print(system.describe())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_dataset

    exclude = _exclude_from_args(args)
    result = export_dataset(args.dataset, out=args.out, val_ratio=args.val_ratio, seed=args.seed,
                            exclude_status=exclude, max_empty_ratio=args.max_empty_ratio,
                            mode=args.mode, tag=args.tag, coco=args.coco, coords=args.coords)
    print(f"[ok] 导出 -> {result['out']}")
    print(f"     train {result['train_images']} 张 / {result['people']['train']} 人 · "
          f"val {result['val_images']} 张 / {result['people']['val']} 人 · 跳过 {result['skipped']}")
    if result.get("skipped_note"):
        # 背景图被比例上限压掉时只报个数字没用，得说清少掉的张数去哪了
        print(f"     剔除明细 {result['skipped_note']}")
    if result.get("empty_dropped"):
        print(f"     背景图（无人）只保留 {result['empty']} 张，上限 --max-empty-ratio "
              f"{args.max_empty_ratio}；想全要就设 1")
    if result.get("missing_source"):
        print(f"     注意：{result['missing_source']} 帧索引里记着、images/ 里却没有了，"
              f"本次已跳过；把图片放回 images/ 或在界面「全部素材」删掉这些帧再导一次")
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
        keep = _flatten_status(getattr(args, "keep_status", None))
        exclude = [s for s in (STATUS_UNREVIEWED, STATUS_REJECTED) if s not in keep]
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
    if result["images"]:
        print(f"[ok] 坐标导出 -> {result['out']}")
    else:
        # 空库照样落 5 个文件、只有一行表头：报 [ok] 会让人以为导出成功了
        print(f"[warn] 姿态标注库里没有可导出的帧，坐标文件只有表头 -> {result['out']}")
    for name in sorted(result["files"]):
        print(f"     - {name}")
    print(f"     图片 {result['images']} 张 · 人 {result['persons']} 个 · "
          f"关键点 {result['persons'] * meta['num_keypoints']} 个 · "
          f"已排除状态 {meta['filter']['exclude_status'] or '无'}")
    print("     提示：v=0 不参与 YOLO 监督，v=1（遮挡）与 v=2（可见）都参与")
    gone = [row for row in (result["meta"].get("skipped") or [])
            if str(row.get("reason", "")).startswith("源文件缺失")]
    if gone:
        print(f"     注意：{len(gone)} 张图索引里记着、images/ 里却没有了，已跳过这些行")
    if not result["images"]:
        from .export import other_library_hint

        hint = other_library_hint(args.dataset, flag="coords")
        if hint:
            print("     提示：姿态标注库里没有可导出的样本" + hint)
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

    missing, existing = _missing_sources(args.sources)
    for source in missing:
        print(f"[warn] 路径不存在，跳过: {source}")
    if not existing:
        raise FileNotFoundError(
            f"导入源都不存在：{'、'.join(missing)}；路径打错了？")
    args.sources = existing
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
    if not imported:
        # 一张都没进来就别再往下标注 / 出坐标，否则会产出一堆空文件看着像成功了
        finish_batch(root, batch["id"], "failed", "没有可导入的图片（源里没有支持的图片/视频，或全部重复）")
        raise ValueError("没有可导入的图片：源里没有支持的图片/视频扩展名，或全部与已有素材重复")

    if lib != LIB_POSE:
        # 提示词库不跑 YOLO-pose：入库即收尾，骨骼标注 / QC / 骨骼坐标是姿态库那套流水线
        finish_batch(root, batch["id"], "done", _import_note(len(imported)))
        print(f"[2/4] {lib_name(lib)}不跑 YOLO-pose，已跳过骨骼标注、QC 与骨骼坐标")
        if lib == LIB_DETECT:
            print(f"      下一步：python3 -m autolabel detect {root} "
                  "--add-classes all --coords（先看这个权重能标什么：--model-classes yolo11n.pt）")
        else:
            print(f"      下一步：python3 -m autolabel prompt {root} "
                  "--fields <字段名> --vlm <多模态配置> --coords")
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
            result = delete_images(args.dataset, _split_multi(args.delete_images), lib=args.lib,
                                   mode=args.mode, keep_files=args.keep_files)
        except ValueError as exc:
            # 文件名是手打的，拼错很常见，别甩整段回溯给用户
            print(f"[error] {exc}")
            return 2
        touched = [str(x) for x in (result.get("libs") or [])] or [result["lib"]]
        where = ("三个库都摘" if result["mode"] == "purge" or len(touched) > 1
                 else f"仅{lib_name(touched[0])}")
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
        result = delete_batches(args.dataset, _split_multi(args.delete), keep_files=args.keep_files,
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
    scope = "三个库合计" if not lib else lib_name(lib)
    print(f"共 {len(rows)} 个批次（{scope}）")
    print(f"{_pad('批次 ID', 26)} {_pad('库', 10)} {_pad('状态', 9)} {_pad('图片', 5)} "
          f"{_pad('已标', 5)} {_pad('人数', 5)} {_pad('检测框', 7)}  来源 · 名称")
    from .batches import MODULE_ROW_KEYS

    for row in rows:
        # 已标 / 框数取这批所属模块的小计：顶层的 labeled / people 是姿态库的，
        # 照抄会让目标检测批次显示成「已标 0 · 检测框 0」
        mine = row.get(MODULE_ROW_KEYS.get(str(row.get("module") or ""), "pose")) or {}
        boxes = int(row.get("prompt_boxes") or 0) + int(row.get("detect_boxes") or 0)
        print(f"{_pad(row['id'], 26)} "
              f"{_pad(row.get('module_name') or row.get('module') or '-', 10)} "
              f"{_pad(row['status'], 9)} {_pad(row['items'], 5)} "
              f"{_pad(mine.get('labeled', row.get('labeled')) or 0, 5)} "
              f"{_pad(row['people'], 5)} {_pad(boxes, 7)}  "
              f"[{row['source']}/{row['kind']}] {row['name']}")
    print(f"\n整批删除：batches {args.dataset} --delete <批次 ID>"
          f"（未分组用 {UNGROUPED_ID} 一次清三个库，或 {UNGROUPED_ID}:detect；"
          f"加 --keep-files 只删记录、--force 强删僵尸批次）")
    print(f"按张删除：batches {args.dataset} --delete-images images/a.jpg --lib objects "
          f"[--mode lib|purge]（lib 只摘本库引用，purge 连素材文件一起删）")
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    """素材池：看有哪些共享素材、把已有素材添加进某个库。"""
    from .dataset import adopt_from_pool, pool_items

    rows = pool_items(args.dataset)
    if args.add:
        stats = adopt_from_pool(args.dataset, _split_multi(args.add), lib=args.lib)
        landed = int(stats["added"]) + int(stats["already"])
        # 一张都没落地却回 [ok]、退出码 0，脚本里 ``&&`` 就白过了：
        # 名字打错的人只看第一行会以为加进去了
        print(f"{'[ok]' if landed else '[error]'} 添加到{lib_name(stats['lib'])}："
              f"新增 {stats['added']} 张 · 本来就在库里 {stats['already']} 张 · "
              f"池里查无 {len(stats['missing'])} 张")
        if stats["missing"]:
            print(f"[warn] 素材池里没有这些文件："
                  f"{'、'.join(stats['missing'][:10])}{'……' if len(stats['missing']) > 10 else ''}")
            print(f"       池里有什么：autolabel pool {args.dataset}；"
                  f"新素材先入库：autolabel import {args.dataset} --lib {args.lib} <图片>")
        return 0 if landed else 1

    # 引用情况一律看 libs 那份清单：只写 ``in_pose and in_objects`` 会把
    # 「只在目标检测库里」的素材报成无人引用，害人去重新认领已经在用的文件
    used = {lib: sum(1 for row in rows if row.get("in_" + lib)) for lib in LIBRARIES}
    shared = sum(1 for row in rows if len(row.get("libs") or []) > 1)
    orphan = sum(1 for row in rows if not (row.get("libs") or []))
    print(f"素材池共 {len(rows)} 个文件 · 姿态库 {used[LIB_POSE]} · 提示词库 {used[LIB_OBJECTS]} · "
          f"检测库 {used[LIB_DETECT]} · 多库共用 {shared} · 无人引用 {orphan}")
    for row in rows[:args.limit]:
        mark = "/".join(LIB_TAGS[lib] for lib in LIBRARIES if row.get("in_" + lib))
        size = f"{row['width']}x{row['height']}" if row["width"] else "未知"
        size_bytes = f"{row['bytes']:,} B"
        print(f"  {_pad(row['image'], 44)} {_pad(size, 11)} {_pad(size_bytes, 12)}  "
              f"在 {mark or '（没被任何库引用）'}")
    if len(rows) > args.limit:
        print(f"  …… 还有 {len(rows) - args.limit} 个，--limit 提高条数")
    print(f"\n添加：pool {args.dataset} --add a.jpg --add b.jpg "
          f"--lib pose|objects|detect（默认 pose；这张图在别的库已经标过的，加进来不覆盖那边的标注）")
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

    if getattr(args, "search", "") or getattr(args, "remote", False):
        return _print_model_search(args)

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
    """显示宽度：全角字符占两列，表格才对得齐（``east_asian_width`` 才是准的）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


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


def _print_model_search(args: argparse.Namespace) -> int:
    """模型搜索：本地模型目录（含官方可下载的那些）+ 可选的 GitHub 社区权重。

    断网/超时不算失败：modelhub 会把 ``ok`` 置 False 并把原因写在 ``note``，
    这里照原话念出来，本地结果照样给。
    """
    from . import modelhub

    payload = modelhub.search_payload(args.search or "", task=getattr(args, "task", "") or "",
                                      remote=bool(getattr(args, "remote", False)),
                                      dataset_root=args.dataset or "")
    local = payload.get("local") or []
    remote = payload.get("results") or []
    query = payload.get("query") or "（空）"
    line = f"搜索「{query}」：模型目录命中 {len(local)} 个"
    if args.remote:
        line += f" · GitHub 命中 {len(remote)} 个"
    print(line)
    if payload.get("note"):
        print(f"  {payload['note']}")
    for row in local:
        size = f"{row.get('expected_mb') or 0:.1f} MB" if row.get("expected_mb") else "体积未知"
        params = f"{row.get('params_m'):.1f}M 参数" if row.get("params_m") else "参数量官方未列明"
        flag = "已在本机" if row.get("installed") else "未下载"
        print(f"\n  {_pad(row.get('filename') or row.get('name') or '', 24)}"
              f"{_pad(row.get('task_name') or row.get('task_label') or '', 12)}"
              f"{_pad(flag, 10)}{_pad(size, 14)}{params}")
        print(f"    说明：{(row.get('description') or row.get('intro') or '（目录里没写说明）')[:120]}")
        if row.get("recommended_for"):
            print(f"    适合：{row['recommended_for']} ｜ 速度：{row.get('speed') or '未标注'}")
        if row.get("class_note"):
            print(f"    类别：{row['class_note']}")
        print(f"    下载：python3 -m autolabel models --download-weight {row.get('filename')}")
    for row in remote:
        print(f"\n  [GitHub] {row.get('name')}  {row.get('html_url')}")
        if row.get("description"):
            print(f"    {str(row['description'])[:120]}")
    if not local and not remote:
        print("\n  没搜到。换个关键词（yolo11 / 姿态 / 分割 / 旋转框），"
              "或者去掉 --task 限制再看一次")
    print("\n下载完就能在「目标检测」「姿态标注」的模型下拉框里选到。")
    return 0


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

    lib = getattr(args, "lib", LIB_OBJECTS) or LIB_OBJECTS
    for item in args.add or []:
        name, _, hint = str(item).partition("=")
        field = upsert_field(args.dataset, name=name, prompt_hint=hint, lib=lib)
        note = field["prompt_hint"] or "（还没写描述提示词，模型只能凭字段名猜）"
        print(f"[ok] 字段 {field['name']}（类别号 {field['class_id']}）：{note}")
    for item in args.set_label or []:
        name, _, label = str(item).partition("=")
        hit = find_field(args.dataset, name, lib)
        if hit is None:
            raise ValueError(f"未知字段 {name}，先用 --add {name}=描述 新建")
        field = upsert_field(args.dataset, name=hit["name"], label=label,
                             prompt_hint=hit.get("prompt_hint", ""), lib=lib)
        print(f"[ok] 字段 {field['name']} 的显示名改为「{field['label']}」")
    for item in args.delete or []:
        result = delete_field(args.dataset, item, drop_boxes=args.drop_boxes, lib=lib)
        print(f"[ok] 已删除字段 {result['name']}（类别号 {result['class_id']}），"
              f"连带删掉 {result['boxes_removed']} 个框，改了 {result['labels_touched']} 个标签文件")

    payload = fields_payload(args.dataset, lib)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not payload["fields"]:
        if payload.get("lib") == LIB_DETECT:
            print("目标检测库还没有类别表。类别来自权重自带，先看能标什么、再导入：")
            print(f"  python3 -m autolabel detect {args.dataset} --model-classes yolo11n.pt")
            print(f"  python3 -m autolabel detect {args.dataset} --model yolo11n.pt --add-classes all")
        else:
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
    if images:
        _require_known_images(args.dataset, LIB_OBJECTS, images)
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
        print(f"[ok] {export_task_name(exported)}训练集：{result['export']}"
              f"（train {exported['train']} / val {exported['val']}，{exported['data_yaml']}）")
        if exported.get("segment_note"):
            print(f"     {exported['segment_note']}")
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


def export_task_name(exported: dict) -> str:
    """导出行前的中文名按实际口径写：分割训练集不该被叫成 detection 训练集。"""
    return "分割（segment）" if str(exported.get("task") or "") == "segment" else "检测（detection）"


def _split_multi(values: Sequence[str]) -> List[str]:
    """把 ``--classes a,b --classes c`` 摊平成一个列表；界面和命令行都能少记一种写法。"""
    out: List[str] = []
    for item in values or []:
        out += [piece.strip() for piece in str(item).split(",") if piece.strip()]
    return out


def _require_known_images(dataset: str, lib: str, images: Sequence[str]) -> None:
    """``--images`` 点名的图必须真在本库里，否则一行说清楚就退出。

    打错一个字，以前一次留下三样脏东西：一条 ``model_failed`` 的幽灵索引记录
    （界面凭空多一行没有宽高、永远打不开的图）、一条失败的批次记录，外加甩给用户
    8 行看不懂的 Python 栈。界面那条路有 ``ReviewStore._require_item`` 把关，
    命令行这条路一直没人管——而它恰恰最容易打错。
    """
    lost = unknown_images(load_index(dataset, lib), images)
    if not lost:
        return
    tail = f" 等 {len(lost)} 张" if len(lost) > 5 else ""
    raise ValueError(f"这些图不在{lib_name(lib)}里：{'、'.join(lost[:5])}{tail}；"
                     f"先入库再标：autolabel import <数据集> --lib {lib} <图片>，"
                     f"素材池里已有的图用 autolabel pool --lib {lib} --add <图片>")


def cmd_detect(args: argparse.Namespace) -> int:
    """目标检测：用本机 YOLO 权重标检测框，类别表由权重自带（非姿态的那类目标检测都走这里）。"""
    from . import detect as detector
    from .batches import batch_name, create_batch, finish_batch
    from .fields import fields_payload
    from .models import weights_dir

    classes = _split_multi(args.classes)
    images = _split_multi(args.images)

    if args.list_models:
        rows = detector.local_models()
        print(f"权重目录：{os.path.abspath(weights_dir())}")
        if not rows:
            print("  本机还没有权重：在界面「模型管理」里搜 yolo 下载，"
                  "或 python3 -m autolabel models --download-weight yolo11n.pt")
            return 0
        for row in rows:
            mark = "可用" if row["usable"] else "用不了"
            note = f"  {row['note']}" if row["note"] else ""
            size = f"{row['bytes'] / 1024 / 1024:.1f} MB" if row.get("installed") else "未下载"
            print(f"  [{mark:<3}] {row['name']:<26} {row['task_name']:<12} {_pad(size, 12)}"
                  f"{_pad('已在本机' if row.get('installed') else '-', 10)}{note}")
        return 0

    if args.model_classes:
        profile = detector.model_profile(args.model_classes, args.device or "")
        if profile.get("open_vocab"):
            # 开放词表权重没有"自带类别表"可列：类别是这次现填的词，
            # 照普通权重打「0 类 + 建议 --add-classes all」等于说废话
            print(f"{os.path.basename(profile['model'])}：{profile['task_name']} · "
                  f"开放词表（{profile['open_vocab']}）· 算力 {profile['device']}")
            print("     类别不来自权重自带类别表，而是你这次填的英文词——想标什么就填什么，"
                  "不必先下载对应权重类别")
            if profile.get("vocab"):
                print(f"  上次用的词表：{'、'.join(profile['vocab'])}")
            print("导入到本库类别表：python3 -m autolabel detect <数据集> --model "
                  f"{os.path.basename(profile['model'])} --add-classes person,dog,traffic cone")
            print("     词表只能填英文（填中文不报错但一个框都找不到），"
                  "一次最多 80 个，每个最长 48 字符")
            return 0
        print(f"{os.path.basename(profile['model'])}：{profile['task_name']} · "
              f"{profile['class_count']} 类 · 算力 {profile['device']}")
        for row in profile["classes"][:args.max_print]:
            print(f"  [{row['class_id']:>3}] {row['name']:<18} {row['label']}")
        if profile["class_count"] > args.max_print:
            print(f"  …还有 {profile['class_count'] - args.max_print} 类，"
                  f"--max-print {profile['class_count']} 看全")
        print("导入到本库类别表：python3 -m autolabel detect <数据集> --model "
              f"{os.path.basename(profile['model'])} --add-classes all")
        return 0

    if args.status:
        summary = detector.detect_summary(args.dataset)
        registry = fields_payload(args.dataset, LIB_DETECT)
        # unreviewed 是"还没标注"的张数，不是"标了没复核"：两个数分开报，
        # 否则会出现"已标 1 张 · 待复核 0"这种看着矛盾的两句
        pending = summary["labeled"] - summary["reviewed"]
        print(f"目标检测：已标 {summary['labeled']}/{summary['images']} 张 · "
              f"框 {summary['boxes']} · 已确认 {summary['reviewed']} · "
              f"没复核 {pending} · 未标 {summary['unreviewed']}")
        print(f"  上次用的权重 {summary['model'] or '（还没跑过）'} · "
              f"类别表 {len(registry['fields'])}/{registry['max_fields']} 类（{registry['path']}）")
        if summary["per_class"]:
            top = sorted(summary["per_class"].items(), key=lambda kv: (-kv[1], kv[0]))[:12]
            print("  类别分布 " + " · ".join(f"{k}={v}" for k, v in top))
        if summary["flag_counts"]:
            print("  标记 " + " · ".join(f"{k}={v}" for k, v in sorted(summary["flag_counts"].items())))
        return 0

    model = detector.resolve_model(args.model or "", args.dataset)
    if args.add_classes:
        wanted = _split_multi(args.add_classes)
        names = None if any(item.lower() == "all" for item in wanted) else wanted
        payload = detector.add_model_classes(args.dataset, model, names, args.device or "")
        not_in = payload.get("not_in_model") or []
        print(f"[ok] 从 {os.path.basename(payload['model'])}（{payload['task_name']} · "
              f"{payload['model_class_count']} 类）导入类别：新增 {payload['added']} · "
              f"复用 {payload['updated']} · 跳过 {len(payload['skipped'])}")
        if payload["added_names"]:
            print("     " + "、".join(str(name) for name in payload["added_names"][:20]))
        if not_in:
            # 这些名字压根没进登记表（add_model_classes 只登记权重整报得出的类别），
            # 说"已登记"会让人以为类别表里多了一行
            print(f"[warn] 这些名字该权重认不出，没有登记：{'、'.join(map(str, not_in[:12]))}"
                  "（换包含这些类别的权重，或在类别表里改掉）")
        classes = classes or [row["name"] for row in payload["fields"]]
        if args.no_label:
            print("[ok] 只导入类别，没有跑标注（加 --no-label 就是这个意思）")
            return 0

    registry = detector.pick_classes(args.dataset, classes)
    if not registry:
        print(f"{lib_name(LIB_DETECT)}还没有类别表。先看权重能标什么，再导入：")
        print(f"  python3 -m autolabel detect {args.dataset} --model-classes {model}")
        print(f"  python3 -m autolabel detect {args.dataset} --model {model} --add-classes all")
        return 2

    if images:
        _require_known_images(args.dataset, LIB_DETECT, images)

    # 批次名直接写"用哪个权重新标了几类"：batch_name() 的"图片 N 个"是给导入用的，
    # 标注批次套用它会被读成"传进来 N 个文件"
    batch = create_batch(args.dataset,
                         name=batch_name("detect", [os.path.basename(model)],
                                         args.batch_name or f"目标检测 {os.path.basename(model)} · "
                                                           f"{len(registry)} 类"),
                         kind="detect", source="cli", lib=LIB_DETECT,
                         params={"model": os.path.basename(model), "detect_classes": classes,
                                 "conf": args.conf, "overwrite": args.overwrite,
                                 "targets": len(images)},
                         uploads=[os.path.basename(model)])
    print(f"[info] 用 {os.path.basename(model)} 标 {len(registry)} 个类别"
          f"（本地推理，不联网，比多模态大模型快得多）")
    try:
        stats = detector.run_detect_labeling(args.dataset, model=model, classes=classes,
                                            images=images or None, limit=args.limit or 0,
                                            overwrite=args.overwrite, device=args.device or "",
                                            conf=args.conf, verbose=not args.quiet)
    except Exception as exc:  # noqa: BLE001 批次要留下失败原因，界面里才看得见
        finish_batch(args.dataset, batch["id"], "failed", str(exc))
        raise
    if stats.get("missing_classes"):
        print(f"[warn] 该权重认不出 {'、'.join(stats['missing_classes'][:10])}："
              "这些类别本次不产出框，换权重或在类别表里改掉")
    if args.coords:
        from .detect import write_detect_coordinate_files

        written = write_detect_coordinate_files(args.dataset, tag=f"cli-{batch['id']}")
        print(f"[ok] 坐标文件：{written['out']}（{written['images']} 张 / {written['boxes']} 个框）")
    if args.export:
        from .detect import export_detect_dataset

        exported = export_detect_dataset(args.dataset, out=args.out or None)
        print(f"[ok] {export_task_name(exported)}训练集：{exported['out']}"
              f"（train {exported['train']} / val {exported['val']}，{exported['data_yaml']}）")
        if exported.get("segment_note"):
            print(f"     {exported['segment_note']}")
    finish_batch(args.dataset, batch["id"], "done",
                 f"用 {os.path.basename(model)} 标注 {stats['labeled']} 张，产出 {stats['boxes']} 个框")
    print(f"[ok] 批次 {batch['id']}：标注 {stats['labeled']} 张 · 框 {stats['boxes']} · "
          f"失败 {stats['failed']} · 跳过 {stats['skipped']}")
    if stats.get("protected"):
        print(f"[info] 人工保护 {stats['protected']} 张（已接受或手动画过框，默认不覆盖；要重标加 --overwrite）")
    if stats["failed"]:
        print(f"[warn] {stats['failed']} 张失败，最后一条错误：{stats['last_error']}")
    if not args.coords and not args.export:
        print("     要导出坐标文件加 --coords，直接出训练集加 --export")
    return 0


def _collect_issues(issues: Sequence[object], issue_count: Dict[str, int],
                    samples: List[str], lib: str) -> None:
    """把一次校验的问题并进料数与样例清单（样例最多 20 条，并标出是哪个库）。"""
    for issue in issues:
        issue_count[issue.kind] = issue_count.get(issue.kind, 0) + 1
        if len(samples) < 20:
            samples.append(f"[{LIB_TAGS.get(lib, lib)}] "
                           f"{issue.path}:{issue.line or '-'} [{issue.kind}] {issue.detail}")


def _orphan_labels(root: str, lib: str, items: Sequence[Tuple[str, dict]]) -> List[str]:
    """标签目录里有、索引里没有的 ``.txt``（删了图没删标签，训练时会多算样本）。"""
    from .library import labels_dir

    label_root = labels_dir(root, lib)
    indexed = {os.path.splitext(rel)[0] for rel, _ in items}
    orphans = []
    for dirpath, _dirs, files in os.walk(label_root):
        for name in files:
            if not name.endswith(".txt"):
                continue
            full = os.path.join(dirpath, name)
            stem = os.path.splitext(os.path.relpath(full, label_root))[0]
            if stem not in indexed:
                orphans.append(os.path.relpath(full, label_root))
    return orphans


def cmd_validate(args: argparse.Namespace) -> int:
    """校验标签格式：三个库都查，``--lib`` 可以只看一个。

    以前只看姿态库。提示词库与检测库的标签是五列的 detection 格式，坏行在
    ``fields.read_label_rows`` 里是**静默跳过**的（标注时这样对，一批里不能
    因为一行脏数据全崩），所以这两个库的坏标签以前没人报——框悄悄少了，
    要到训练结果不行才发现。
    """
    from .fields import box_counts_by_image, list_fields, validate_box_label_file

    root = args.dataset
    libs = list(LIBRARIES) if args.lib == "all" else [normalize(args.lib)]
    issue_count: Dict[str, int] = {}
    samples: List[str] = []
    checked = 0
    kpt_note = ""
    for lib in libs:
        idx = load_index(root, lib)
        items = list(idx.ordered_items())
        if not items:
            continue
        checked += len(items)
        gone = missing_sources(root, [rel for rel, _item in items])
        if gone:
            # 图被手工删了：标签格式再干净也没用，导出会跳帧、训练会读不到图。
            # 这条以前哪儿都不报，只有 export 复制文件时才炸。
            issue_count["missing_source"] = issue_count.get("missing_source", 0) + len(gone)
            samples.extend(f"[{LIB_TAGS.get(lib, lib)}] 缺源文件: {rel}" for rel in gone[:10])
        if lib == LIB_POSE:
            num_kpt = args.num_keypoints or idx.num_keypoints
            kpt_note = f"（关键点数 {num_kpt}）"
            for rel, item in items:
                _collect_issues(validate_label_file(
                    label_path_for(root, rel, lib),
                    int(item.get("width", 0) or 0), int(item.get("height", 0) or 0),
                    num_keypoints=num_kpt, class_names=idx.class_names), issue_count, samples, lib)
        else:
            class_ids = {int(row.get("class_id", -1)) for row in list_fields(root, lib)}
            for rel, _item in items:
                _collect_issues(validate_box_label_file(label_path_for(root, rel, lib), class_ids),
                                issue_count, samples, lib)
            # 索引里那笔框数账和磁盘上的标签文件对不对得上：手工改过标签、
            # 或者上次写在两步之间被打断，列表页就会一直显示一个不存在的数字
            on_disk = box_counts_by_image(root, lib)
            for rel, item in items:
                counted = on_disk.get(os.path.splitext(rel)[0])
                indexed = item.get(field_key(lib, "objects"))
                if counted is None or (indexed is None and counted == 0):
                    continue      # 没有标签文件的，上面按 missing_label 报过了
                if indexed is not None and int(indexed) == counted:
                    continue
                issue_count["count_mismatch"] = issue_count.get("count_mismatch", 0) + 1
                if len(samples) < 20:
                    # 索引里写着"没标过"而标签文件里有框，导出会整张跳过：
                    # 用户看到的就是"我明明标了 3 个框，导出来一个都没有"
                    claimed = "没标过" if indexed is None else f"{indexed} 个框"
                    samples.append(f"[{LIB_TAGS.get(lib, lib)}] {rel}: 索引记 {claimed}，"
                                   f"标签文件里是 {counted} 个（重跑一次标注即可对齐）")
        orphans = _orphan_labels(root, lib, items)
        if orphans:
            issue_count["orphan_label"] = issue_count.get("orphan_label", 0) + len(orphans)
            samples.extend(f"[{LIB_TAGS.get(lib, lib)}] 孤儿标签: {o}" for o in orphans[:10])

    scope = "、".join(lib_name(item) for item in libs)
    if issue_count:
        print("[warn] 发现问题：" + ", ".join(f"{k}={v}" for k, v in sorted(issue_count.items())))
        for line in samples:
            print("   - " + line)
        print(f"     查的是：{scope}")
        return 1
    if not checked:
        print(f"[ok] 没有需要校验的标签：{scope} 都还没有素材")
        return 0
    print(f"[ok] 校验通过：{scope} 共 {checked} 张，标签格式一致{kpt_note}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    idx = load_index(args.dataset)
    from .qc import dataset_stats

    stats = dataset_stats(args.dataset)
    print(f"数据集 {idx.name}  ({idx.root})")
    if stats["images"]:
        print(f"  姿态库 图片 {stats['images']} · 人数 {stats['persons_total']} · "
              f"缺标签 {stats['missing_labels']}")
    if stats.get("missing_files"):
        # 只报数字没用：得说清是哪几张、以及"导出会自动跳过"，否则用户以为导出了
        sample = "、".join(stats.get("missing_images") or [])
        more = "…" if stats["missing_files"] > len(stats.get("missing_images") or []) else ""
        print(f"  缺文件 {stats['missing_files']} 张（索引有记录、images/ 里不在了）：{sample}{more}"
              f" —— 导出会自动跳过，删掉这些帧或把图片放回 images/ 即可对齐")
    if not stats["images"]:
        # 空库就别打印一堆 0 了，直接告诉用户往哪儿灌素材——提示词库与检测库
        # 空着时本来就只给这一行，三个库同一个说法，别让人以为姿态库丢了数据
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
              f"已确认 {prompt['reviewed']} · 未标 {prompt['unreviewed']} · "
              f"没复核 {prompt['labeled'] - prompt['reviewed']}")
    if prompt["per_field"]:
        top = sorted(prompt["per_field"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        print("  字段分布 " + " · ".join(f"{k}={v}" for k, v in top))
    if prompt["flag_counts"]:
        print("  提示词标记 " + " · ".join(f"{k}={v}" for k, v in sorted(prompt["flag_counts"].items())))

    from .detect import detect_summary

    detect = detect_summary(args.dataset)
    if not detect["images"]:
        print("  检测库还没有素材：autolabel import <数据集> --lib detect，"
              "或界面「目标检测」页上传（类别来自权重：autolabel detect <数据集> --model-classes yolo11n.pt）")
    else:
        print(f"  目标检测 已标 {detect['labeled']}/{detect['images']} 张 · 框 {detect['boxes']} · "
              f"已确认 {detect['reviewed']} · 未标 {detect['unreviewed']} · "
              f"没复核 {detect['labeled'] - detect['reviewed']} · 上次权重 {detect['model'] or '（还没跑过）'}")
    if detect["per_class"]:
        top = sorted(detect["per_class"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        print("  检测类别分布 " + " · ".join(f"{k}={v}" for k, v in top))
    if detect["flag_counts"]:
        print("  检测标记 " + " · ".join(f"{k}={v}" for k, v in sorted(detect["flag_counts"].items())))

    from .dataset import pool_items

    pool = pool_items(args.dataset)
    # 只按两个库判断会把「只在检测库里」的素材说成无人引用
    shared = sum(1 for row in pool if len(row.get("libs") or []) > 1)
    orphan = sum(1 for row in pool if not (row.get("libs") or []))
    print(f"  素材池 {len(pool)} 个文件 · 多库共用 {shared} · 无人引用 {orphan}"
          f"（{os.path.join(os.path.abspath(args.dataset), 'images')}）")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """把复核状态退回未复核，方便重跑或换参数后重来。"""

    lib = normalize(args.lib)
    idx = load_index(args.dataset, lib)
    key = status_key(lib)
    wanted = set(args.status or ())
    changed: List[str] = []
    for rel, item in idx.items.items():
        status = status_of(item, lib)
        if status == STATUS_UNREVIEWED:
            continue
        if wanted and status not in wanted:
            continue
        if args.keep_approved and status == STATUS_APPROVED:
            continue
        item[key] = STATUS_UNREVIEWED
        item.pop(field_key(lib, "reviewed_at"), None)
        changed.append(rel)
    # `--labels` 删的是**上面真的被退回那些**：以前它无差别删光整库标签，
    # `--status rejected --labels` 会把已接受样本的标签一起删掉。标签没了还得把
    # 索引里由标签派生的计数一起抹掉，否则索引继续报「这张有 1 个人」，
    # 门户和 `status` 的人数立刻开始说谎。
    removed = 0
    if args.labels:
        keys = label_field_keys(lib)
        for rel in changed:
            path = label_path_for(args.dataset, rel, lib)
            if os.path.exists(path):
                os.remove(path)
                removed += 1
            for field in keys:
                idx.items[rel].pop(field, None)
    idx.save()
    note = ""
    if args.labels:
        note = (f"，删掉标签 {removed} 个（与退回的是同一批，索引里的标注计数一起清）"
                if changed else "，但这批里没有可退回的图，标签一个都没删")
    print(f"[ok] 已退回 {len(changed)} 张为未复核（{lib_name(lib)}）{note}")
    if not changed:
        print("     这批里没有可退回的复核状态；想看现在都是什么状态：autolabel status <数据集>")
    print("     提示：标注参数变化时 label 会自动重标，无需手动 reset")
    return 0


# --- 参数校验：错在参数上就在参数上报，别等流水线跑到一半 ---------- #

def _review_statuses(text: str) -> List[str]:
    """``--status rejected,edited`` 拆成列表；写错的状态当场拦。

    以前拼错一个字母（``aprovied``）只会得到一句「已退回 0 张」，退出码 0，
    用户以为执行过了，其实什么都没发生。
    """
    values: List[str] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item not in ALL_STATUSES:
            raise argparse.ArgumentTypeError(
                f"不认识的状态「{item}」，可选：{'、'.join(ALL_STATUSES)}")
        if item not in values:
            values.append(item)
    return values


def _status_words(text: str) -> List[str]:
    """``--keep-status rejected,edited``：逗号写法摊平，拼错的状态当场拦。

    以前 ``--keep-status aprieved`` 与漏写没区别——它只参与"要不要把某个状态
    留在导出范围里"的判断，拼错的词谁都匹配不上，用户以为多留了一批，
    其实导出的还是那一份。
    """
    return _review_statuses(text)


def _flatten_status(values) -> List[str]:
    """argparse 用 ``type=_status_words`` 后每项是个列表，这里统一摊平成一个列表。"""
    out: List[str] = []
    for value in values or []:
        pieces = value if isinstance(value, (list, tuple)) else str(value).split(",")
        for piece in pieces:
            item = str(piece).strip()
            if item and item not in out:
                out.append(item)
    return out


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"得是整数，收到「{text}」") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"得大于 0，收到 {value}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"得是整数，收到「{text}」") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"不能是负数，收到 {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"得是数字，收到「{text}」") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"得大于 0，收到 {value}")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"得是数字，收到「{text}」") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"不能是负数，收到 {value}")
    return value


def _fraction(text: str) -> float:
    """0~1 之间的比例（不含两端）：验证集占比、置信度阈值这类。"""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"得是数字，收到「{text}」") from None
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"得在 0 和 1 之间（不含两端），收到 {value}")
    return value


def _unit_interval(text: str) -> float:
    """允许取到端点的 0~1：背景图占比这类可以为 0。"""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"得是数字，收到「{text}」") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"得在 0 到 1 之间，收到 {value}")
    return value


def _quality(text: str) -> int:
    value = _positive_int(text)
    if value > 100:
        raise argparse.ArgumentTypeError(f"JPEG 质量得在 1~100，收到 {value}")
    return value


def _port(text: str) -> int:
    value = _positive_int(text)
    if value > 65535:
        raise argparse.ArgumentTypeError(f"端口得在 1~65535，收到 {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autolabel", description="人体骨骼自动标注系统（YOLO pose）")
    parser.add_argument("--version", action="version", version=f"autolabel {__version__}")
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
        # 可选值来自「系统设置」那一处定义，别再手抄一份漏掉 CUDA
        p.add_argument("--device", choices=list(DEVICE_ARGS),
                       help="推理算力：auto / cuda（N 卡）/ mps（Apple 加速）/ cpu，0、1 指定第几块 N 卡")
        p.add_argument("--imgsz", type=_positive_int)
        p.add_argument("--conf", type=_fraction)
        p.add_argument("--iou", type=_fraction)
        p.add_argument("--max-det", type=_positive_int)
        p.add_argument("--half", action="store_true", default=None)
        p.add_argument("--no-flip", dest="flip_tta", action="store_false", default=None,
                       help="关闭水平翻转融合（更快，关键点质量略降）")
        p.add_argument("--visibility-policy", choices=["threshold", "all_visible", "binary"])
        p.add_argument("--visible-conf", type=_fraction)
        p.add_argument("--occluded-conf", type=_fraction)
        p.add_argument("--box-mode", choices=["expand", "model", "keypoints"])
        p.add_argument("--bbox-margin", type=_positive_float)
        p.add_argument("--limit", type=_non_negative_int, help="只标注前 N 张，用于试跑")
        p.add_argument("--render-qc", type=_non_negative_int, default=0, help="同时渲染 N 张骨架叠加图")
        p.add_argument("--force", action="store_true",
                   help="连人工已接受 / 手工改过的图一起重标（默认跳过它们与已按当前参数标好的图）")
        p.add_argument("--dry-run", action="store_true", help="只跑流程不写标签")

    p = sub.add_parser("init", help="创建数据集骨架")
    add_dataset(p)
    p.add_argument("--name", help="数据集名称")
    p.add_argument("--class-names", default="person", help="逗号分隔类别名，默认单类 person")
    p.add_argument("--num-keypoints", type=_positive_int, default=17, help="关键点数，COCO 为 17")
    p.set_defaults(func=cmd_init)

    def add_lib(parser: argparse.ArgumentParser, default: str = "pose") -> None:
        parser.add_argument("--lib", choices=list(LIBRARIES), default=default,
                            help="入库到哪个标注库：pose=姿态标注（默认）、objects=提示词标注、"
                                 "detect=目标检测；素材文件三个库共用，images/ 里已有的直接引用，不重复落盘")

    p = sub.add_parser("import", help="导入图片（自动规范化 RGB/EXIF/尺寸并去重）")
    add_dataset(p)
    p.add_argument("sources", nargs="+", help="图片或目录")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--max-side", type=_positive_int, default=1600)
    p.add_argument("--jpeg-quality", type=_quality, default=95)
    p.add_argument("--dedupe", choices=["none", "exact", "perceptual"], default="exact")
    p.add_argument("--subdir", default="", help="导入到 images/ 下的子目录")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    add_lib(p)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("video", help="视频抽帧导入（ffmpeg）")
    add_dataset(p)
    p.add_argument("video", help="视频文件")
    p.add_argument("--fps", type=_positive_float, default=0.5, help="采样帧率，0.5 = 每 2 秒 1 帧")
    p.add_argument("--max-frames", type=_positive_int, default=600)
    p.add_argument("--scene", type=_fraction, help="按场景切点抽帧的阈值（0.2~0.4），设置后忽略 --fps")
    p.add_argument("--start", type=_non_negative_float, default=0.0, help="起始秒")
    p.add_argument("--duration", type=_positive_float, help="截取时长（秒）")
    p.add_argument("--max-side", type=_positive_int, default=1600)
    p.add_argument("--dedupe", choices=["none", "exact", "perceptual"], default="perceptual")
    p.add_argument("--subdir", default="")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    add_lib(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("label", help="批量自动标注")
    add_dataset(p)
    add_label_args(p)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("qc", help="生成质量报告与复核清单（骨骼质量与复核优先级，只管姿态库）")
    add_dataset(p)
    p.add_argument("--top", type=_positive_int, default=30)
    p.add_argument("--render", type=_non_negative_int, default=0, help="渲染 N 张叠加图")
    p.add_argument("--render-all", action="store_true", help="叠加图覆盖全部图片而非仅带标记的")
    p.add_argument("--include-reviewed", action="store_true",
                   help="复核清单连已接受 / 已驳回的一起列（默认只列还没定论的）")
    p.add_argument("--json", action="store_true", help="额外打印 JSON 统计")
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("review", help="打开浏览器复核界面")
    add_dataset(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=_port, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--max-upload-mb", type=_positive_int, default=1024, help="网页上传单文件大小上限")
    p.add_argument("--ssl-cert", default="",
                   help="HTTPS 证书；不给就自动用 certs/ 里的本地证书（跑过 autolabel ca 就有）")
    p.add_argument("--ssl-key", default="", help="HTTPS 私钥，默认按证书名猜同目录的 -key.pem")
    p.add_argument("--no-ssl", action="store_true",
                   help="退回 HTTP。能用，但浏览器会关掉摄像头，还可能被自动升级成 HTTPS 而打不开")
    p.add_argument("--device", choices=list(DEVICE_ARGS),
                   help="这个实例的推理算力，覆盖系统设置（想一机多卡就多用 --port 起几个实例，"
                        "卡号写 cuda:1 或 1）")
    half_group = p.add_mutually_exclusive_group()
    half_group.add_argument("--half", action="store_true", default=None,
                            help="这个实例开半精度（只有 NVIDIA 显卡提速）")
    half_group.add_argument("--no-half", dest="half", action="store_false",
                            help="这个实例关半精度")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("launch", help="启动复核工作台（自检 → 起服务 → 端口就绪再开浏览器；"
                                      "运行工作台.command / .bat 就是它）")
    p.add_argument("dataset", nargs="?", default="datasets/demo",
                   help="数据集目录，默认 datasets/demo（双击启动时不用填）")
    p.add_argument("--port", type=_positive_int, default=8765, help="监听端口，默认 8765")
    p.add_argument("--host", default="127.0.0.1", help="监听地址，默认只本机可访问")
    p.add_argument("--device", choices=list(DEVICE_ARGS),
                   help="推理算力：auto / cuda（N 卡）/ mps（Apple 加速）/ cpu；"
                        "不填就按「系统设置」里存的那档")
    p.add_argument("--half", action="store_true", help="半精度推理（只在 NVIDIA 显卡上有意义）")
    p.add_argument("--no-browser", action="store_true", help="起好了但不自动开浏览器")
    p.add_argument("--wait", type=_positive_int, default=90,
                   help="最多等多少秒端口还没监听就放弃，默认 90")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("service", help="常驻服务：登录自起、崩了自动拉起"
                                       "（macOS 用 launchd，Windows 用任务计划程序）")
    p.add_argument("--status", action="store_true", help="谁在跑、端口、退出原因、日志尾几行")
    p.add_argument("--restart", nargs="?", const="", metavar="名字",
                   help="改完代码重启（不给名字就全部重启）")
    p.add_argument("--remove", nargs="?", const="", metavar="名字",
                   help="注销并删掉生成的任务定义与启动片")
    p.add_argument("--dry-run", action="store_true", help="只把任务定义写出来，不碰服务管理器")
    p.add_argument("--show", metavar="名字", help="打印这个实例的任务定义（排查用，不改任何东西）")
    p.add_argument("--print-env", metavar="名字", dest="print_env",
                   help="打印这个实例的参数（服务启动片自己用）")
    p.set_defaults(func=cmd_service)

    p = sub.add_parser("service-run", help="常驻服务的实例本体（由服务管理器调用，一般不用手敲）")
    p.add_argument("name", nargs="?", help="清单里的实例名；不填就用手上的参数临时起一个")
    p.add_argument("--port", type=_positive_int)
    p.add_argument("--dataset", help="不读清单时临时指定数据集")
    p.add_argument("--device", choices=list(DEVICE_ARGS), help="不读清单时临时指定算力")
    p.add_argument("--half", action="store_true", help="半精度推理")
    p.add_argument("--wait", type=_positive_int, default=90, help="最多等多少秒端口就绪")
    p.set_defaults(func=cmd_service_run)

    p = sub.add_parser("version", help="看版本号 / 开发者 / 与 GitHub 比对 / 升版本")
    p.add_argument("--check", action="store_true", help="立刻联网重新比对（平时读半天内的缓存）")
    p.add_argument("--bump", nargs="?", const="patch", choices=["patch", "minor", "major"],
                   help="升版本号并写回，默认 patch（+0.01）")
    p.set_defaults(func=cmd_version)

    p = sub.add_parser("settings", help="看/改系统设置：推理设备（CUDA / Apple 加速 / CPU）、半精度")
    p.add_argument("--device", choices=list(DEVICE_ARGS), help="推理用哪块算力")
    half_group = p.add_mutually_exclusive_group()
    half_group.add_argument("--half", action="store_true", default=None, help="开半精度（只有 NVIDIA 显卡提速）")
    half_group.add_argument("--no-half", dest="half", action="store_false", help="关半精度")
    p.add_argument("--reset", action="store_true", help="恢复默认（自动选择 + 关半精度）")
    p.add_argument("--json", action="store_true", help="连探测结果一起打 JSON")
    p.set_defaults(func=cmd_settings)

    p = sub.add_parser("ca", help="生成本地 HTTPS 证书并设为受信任（摄像头要 HTTPS）")
    p.add_argument("--status", action="store_true", help="只看状态，不签发不装信任")
    p.add_argument("--force", action="store_true", help="重新签发（证书过期或换域名时用）")
    p.add_argument("--no-install", action="store_true", help="只签发，不写入钥匙串")
    p.add_argument("--uninstall", action="store_true", help="从钥匙串移除本项目根证书")
    p.set_defaults(func=cmd_ca)

    p = sub.add_parser("export", help="导出 YOLO 训练数据集（train/val + data.yaml）")
    add_dataset(p)
    p.add_argument("--out", help="导出目录，默认 <dataset>/exports/<时间戳>")
    p.add_argument("--tag", help="导出子目录名")
    p.add_argument("--val-ratio", type=_fraction, default=0.2, help="验证集占比，0~1 之间（不含两端）")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-empty-ratio", type=_unit_interval, default=0.15, help="背景图占比上限")
    p.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    p.add_argument("--coco", action="store_true", help="同时导出 COCO keypoint JSON")
    p.add_argument("--coords", action="store_true", help="同时导出 CSV/JSONL 坐标文件（含 split 列）")
    p.add_argument("--only-approved", action="store_true", help="只导出已接受/已修改的样本")
    p.add_argument("--keep-status", nargs="*", default=[], type=_status_words,
                   metavar="状态",
                   help="配合 --only-approved 额外保留的状态，逗号分隔可重复"
                        f"（可选：{'、'.join(ALL_STATUSES)}）")
    p.add_argument("--include-all", action="store_true", help="连 rejected 一起导出")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("coords", help="导出坐标文件（宽表 CSV / 长表 CSV / JSONL）")
    add_dataset(p)
    p.add_argument("--out", help="输出目录，默认 <dataset>/coords/<时间戳>")
    p.add_argument("--tag", help="输出子目录名")
    p.add_argument("--formats", default="csv,long,jsonl",
                   help="逗号分隔：csv 一行一个人 / long 一行一个关键点 / jsonl 一行一张图")
    p.add_argument("--only-approved", action="store_true", help="只导出已接受/已修改的样本")
    p.add_argument("--keep-status", nargs="*", default=[], type=_status_words,
                   metavar="状态",
                   help="配合 --only-approved 额外保留的状态，逗号分隔可重复"
                        f"（可选：{'、'.join(ALL_STATUSES)}）")
    p.add_argument("--include-all", action="store_true", help="连 rejected 一起导出")
    p.add_argument("--include-unlabeled", action="store_true", help="缺标签的图也输出一行（objects 为空）")
    p.set_defaults(func=cmd_coords)

    p = sub.add_parser("auto", help="一条命令跑完：导入/抽帧 → 自动标注 → QC → 坐标导出")
    add_dataset(p)
    p.add_argument("sources", nargs="+", help="图片、图片目录或视频文件（按扩展名自动分流）")
    p.add_argument("--name", help="数据集名称，仅首次创建时使用")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--max-side", type=_positive_int, default=1600)
    p.add_argument("--jpeg-quality", type=_quality, default=95)
    p.add_argument("--dedupe", choices=["none", "exact", "perceptual"], default="exact", help="图片去重方式")
    p.add_argument("--video-dedupe", choices=["none", "exact", "perceptual"], default="perceptual",
                   help="视频帧去重方式，默认感知去重以剔除近重复帧")
    p.add_argument("--subdir", default="", help="导入到 images/ 下的子目录")
    p.add_argument("--fps", type=_positive_float, default=0.5, help="视频采样帧率，0.5 = 每 2 秒 1 帧")
    p.add_argument("--max-frames", type=_positive_int, default=600)
    p.add_argument("--scene", type=_fraction, help="按场景切点抽帧的阈值（0.2~0.4），设置后忽略 --fps")
    p.add_argument("--start", type=_non_negative_float, default=0.0)
    p.add_argument("--duration", type=_positive_float)
    add_lib(p)
    add_label_args(p)
    p.add_argument("--formats", default="csv,long,jsonl", help="坐标文件格式")
    p.add_argument("--coords-out", help="坐标输出目录，默认 <dataset>/coords/<时间戳>")
    p.add_argument("--only-approved", action="store_true")
    p.add_argument("--keep-status", nargs="*", default=[], type=_status_words,
                   metavar="状态")
    p.add_argument("--include-all", action="store_true")
    p.add_argument("--include-unlabeled", action="store_true")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("validate", help="校验标签格式（三个库都查）")
    add_dataset(p)
    p.add_argument("--num-keypoints", type=_non_negative_int, default=0)
    p.add_argument("--lib", choices=["all", *LIBRARIES], default="all",
                   help="只查某个库的标签：all=三个库都查（默认）、pose=姿态标注、"
                        "objects=提示词标注、detect=目标检测")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("status", help="数据集概览")
    add_dataset(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("batches", help="按批次查看已标注数据，或整批删除图片与记录")
    add_dataset(p)
    p.add_argument("--lib", choices=["all", *LIBRARIES], default="all",
                   help="看/删哪个库的批次，默认 all：列表三个库一起看，"
                        "按张删除则三个库的引用一起摘（素材文件仍保留）")
    p.add_argument("--delete", action="append", default=[], metavar="批次ID",
                   help="删除该批次的图片与记录，逗号分隔可重复；未分组用 __ungrouped__")
    p.add_argument("--delete-images", action="append", default=[], metavar="图片路径",
                   help="按张删除（a.jpg 或 images/a.jpg 都认），逗号分隔可重复；"
                        "批次记录里同步摘除，删空的批次连记录一起删")
    p.add_argument("--keep-files", action="store_true", help="只删索引与批次记录，保留图片/标签文件")
    p.add_argument("--mode", choices=["lib", "purge"], default="lib",
                   help="按张删除的粒度：lib 只摘引用（默认，素材文件留给别的库），"
                        "purge 三个库一起摘并把素材文件删掉")
    p.add_argument("--force", action="store_true",
                   help="忽略「仍在标注中」的保护，用于服务中断留下的僵尸批次")
    p.add_argument("--json", action="store_true", help="以 JSON 输出批次列表")
    p.set_defaults(func=cmd_batches)

    p = sub.add_parser("pool", help="素材池：查看共享素材被哪些库引用，或把已有素材添加进某个库")
    add_dataset(p)
    p.add_argument("--add", action="append", default=[], metavar="图片路径",
                   help="把素材池里的这张图添加进 --lib 指定的库（不复制文件），逗号分隔可重复")
    p.add_argument("--limit", type=_non_negative_int, default=30, help="最多列几条，默认 30")
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
    p.add_argument("--lib", choices=[LIB_OBJECTS, LIB_DETECT], default=LIB_OBJECTS,
                   help="哪张类别表：objects=提示词标注（默认，字段名+描述提示词）、"
                        "detect=目标检测（类别来自 YOLO 权重自带，一般用 autolabel detect --add-classes 导入）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出字段表")
    p.set_defaults(func=cmd_fields)

    p = sub.add_parser("prompt", help="提示词标注：按字段描述让多模态大模型找物品，产出 YOLO detection 框")
    add_dataset(p)
    p.add_argument("--fields", action="append", default=[], metavar="字段名",
                   help="只标这些字段（逗号分隔可重复），默认用字段表里全部字段")
    p.add_argument("--images", action="append", default=[], metavar="相对路径",
                   help="只标这些图（逗号分隔可重复），默认整个数据集")
    p.add_argument("--vlm", default="", help="多模态配置名或 ID，默认用「模型管理」里设的默认配置")
    p.add_argument("--limit", type=_non_negative_int, default=0, help="只标前 N 张，用于试跑")
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

    p = sub.add_parser("detect", help="目标检测：用本机 YOLO 权重标检测框（非姿态的目标检测任务都走这里）")
    add_dataset(p)
    p.add_argument("--model", default="",
                   help="权重文件名或路径，默认上次用的那个（没有就 yolo11n.pt）；它决定能标哪些类别")
    p.add_argument("--list-models", action="store_true",
                   help="列出本机权重，并标出哪些能用在目标检测（姿态权重会说明为什么用不了）")
    p.add_argument("--model-classes", metavar="权重",
                   help="列出这个权重自带的类别表（中文名 + 类别号），只看不动数据；"
                        "开放词表权重会说明该填哪些英文词")
    p.add_argument("--add-classes", action="append", default=[], metavar="类名",
                   help="把权重自带的类别导入本库类别表，逗号分隔可重复；all=全部导入。"
                        "开放词表权重（yolov8x-world.pt、yoloe-26x-seg.pt）这里填的就是"
                        "要找的英文词，不限权重自带类别", )
    p.add_argument("--no-label", action="store_true", help="配合 --add-classes：只导入类别，不跑标注")
    p.add_argument("--classes", action="append", default=[], metavar="类名",
                   help="只标这些类别（逗号分隔可重复），默认用本库类别表里全部")
    p.add_argument("--images", action="append", default=[], metavar="相对路径",
                   help="只标这些图（逗号分隔可重复），默认整个目标检测库")
    p.add_argument("--device", choices=list(DEVICE_ARGS), default="",
                   help="推理算力，默认用「系统设置」里那档；0、1 指定第几块 N 卡")
    p.add_argument("--conf", type=_fraction, default=None, help="置信度阈值，默认用系统设置里的值")
    p.add_argument("--limit", type=_non_negative_int, default=0, help="只标前 N 张，用于试跑")
    p.add_argument("--overwrite", action="store_true",
                   help="已经标过的也重标（默认跳过同参数已标的、人工接受过的、人工画过框的图）")
    p.add_argument("--coords", action="store_true", help="顺带导出坐标文件（CSV + JSONL）")
    p.add_argument("--export", action="store_true",
                   help="顺带导出 YOLO 训练集（标签里有轮廓时出 segment 格式）")
    p.add_argument("--out", default="", help="配合 --export：导出目录")
    p.add_argument("--max-print", type=_positive_int, default=40,
                   help="配合 --model-classes：最多打印多少类，默认 40")
    p.add_argument("--batch-name", help="批次名称，用于「批次管理」分组显示")
    p.add_argument("--status", action="store_true", help="只看已标/待复核汇总，不跑标注")
    p.add_argument("--quiet", action="store_true", help="不逐张打印进度")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("models", help="列出/管理姿态权重与多模态大模型（本地或云端）配置")
    p.add_argument("--dataset", default="",
                   help="兼容性检查用的数据集目录（比对关键点数量，判这个模型能不能用）")
    p.add_argument("--browse", action="store_true",
                   help="浏览可下载模型目录：中文说明、参数规模、预计体积、适合场景")
    p.add_argument("--catalog", action="store_true", help="同 --browse")
    p.add_argument("--info", metavar="模型ID", help="看单个模型的完整中文说明、下载地址与下载记录")
    p.add_argument("--capability", default="", help="只看一类能力：pose | vlm | custom_task")
    p.add_argument("--search", metavar="关键词",
                   help="按关键词搜模型（yolo11 / 姿态 / 分割…），本机已登记 + 官方目录一起给；"
                        "配合 --remote 再查 GitHub 上的自制权重")
    p.add_argument("--task", default="", help="配合 --search：只看一类任务 "
                                             "pose | detect | segment | obb | classify")
    p.add_argument("--remote", action="store_true", help="配合 --search：连 GitHub 上的自制权重一起搜（要联网）")
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
    add_lib(p)
    p.add_argument("--status", type=_review_statuses, default=[],
                   help=f"只重置这些状态，逗号分隔（可选 {'、'.join(ALL_STATUSES)}；"
                        f"不填 = 除未复核以外的全部）")
    p.add_argument("--keep-approved", action="store_true", help="保留已接受的样本")
    p.add_argument("--labels", action="store_true",
                   help="同时删除标签文件（只删上面被退回那些，并清掉索引里的标注计数）")
    p.set_defaults(func=cmd_reset)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    # 先固定输出编码：Windows 把输出重定向进日志文件时按本地代码页写字节，
    # 中文机器上满屏的 ✓ ✗ ▸ 一个都不在里头，写第一行就崩，而且只在后台跑时崩
    platform_utils.force_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[info] 已中断")
        return 130
    except (FileNotFoundError, NotADirectoryError, ValueError, KeyError, RuntimeError) as exc:
        # 路径打错、批次不存在、参数越界这类用户自己就能改对的问题：一行说清楚。
        # 以前直接甩 Python 栈，普通用户看到的是 8 行看不懂的英文。
        # RuntimeError 也在列：本项目 raise 它的地方全是「用户能自己解决」的运行时
        # 状况（没装 ffmpeg、找不到 openssl、权重加载失败、整批标注全失败），
        # 文案本来就是一句人话，配一栈 traceback 只会把它埋掉。
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"[error] {message}", file=sys.stderr)
        if os.environ.get("AUTOLABEL_TRACEBACK"):  # 排查用：贴栈出来
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
