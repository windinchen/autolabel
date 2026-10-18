"""导出训练可用的 YOLO pose 数据集（train/val 划分 + data.yaml + COCO JSON）。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import label_path_for, load_index, now_str
from .yolo_pose import COCO17_KEYPOINTS, COCO17_FLIP_IDX, COCO17_SKELETON, parse_label_file

DEFAULT_EXCLUDE_STATUS = {"rejected"}
MISSING_LABEL_REASON = "缺少标签"
EMPTY_CAP_REASON = "背景图超出比例上限"


def skipped_summary(skipped: Sequence[Tuple[str, str]]) -> str:
    """把跳过原因压成一句中文：``（已驳回 3 张 · 背景图超出比例上限 12 张）``。"""
    counts: Dict[str, int] = {}
    for _rel, reason in skipped:
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return ""
    parts = [f"{reason} {num} 张" for reason, num in
             sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "（" + " · ".join(parts) + "）"


def no_sample_hint(skipped: Sequence[Tuple[str, str]]) -> str:
    """没有帧可导出时，只指真正能改的那个开关。

    以前不分原因都提示「加 --include-all / 界面范围改成全部」，实测三行里只有一行
    吃这一套：`export` 根本没有放行缺标签帧的开关（只有 `coords` 有
    --include-unlabeled），背景图吃的又是 --max-empty-ratio。用户照着提示换了范围，
    报错一个字不变，只会以为软件坏了。
    """
    reasons = [reason for _rel, reason in skipped]
    advice = []
    if any(reason.startswith("status=") for reason in reasons):
        advice.append("想连被驳回的帧一起要，命令行加 --include-all，界面里把导出范围改成「全部帧」")
    if MISSING_LABEL_REASON in reasons:
        advice.append("没标注的帧进不了训练集，先跑一次自动标注（命令行 autolabel auto / annotate）")
    if any(reason.startswith(EMPTY_CAP_REASON) for reason in reasons):
        advice.append("背景图是被占比上限压掉的，命令行加 --max-empty-ratio 1 就全要")
    if not advice:
        return "：库里一张图都还没有，先导入图片或抽帧（命令行 autolabel import / auto）"
    return "：" + "；".join(advice)


def export_dataset(
    root: str,
    out: Optional[str] = None,
    val_ratio: float = 0.2,
    seed: int = 7,
    exclude_status: Sequence[str] = ("rejected",),
    require_label: bool = True,
    max_empty_ratio: float = 0.15,
    mode: str = "symlink",
    tag: Optional[str] = None,
    coco: bool = False,
    coords: bool = False,
) -> dict:
    """生成可直接喂给 `yolo pose train` 的目录。

    :param mode: symlink（省空间）| copy
    :param max_empty_ratio: 背景图（无人）在导出数据集中的占比上限，避免负样本淹没正样本
    :param coords: 同时在导出目录里生成 CSV/JSONL 坐标文件（含 split 列）
    """
    idx = load_index(root)
    exclude = set(exclude_status)
    populated: List[Tuple[str, dict]] = []
    empty: List[Tuple[str, dict]] = []
    skipped: List[Tuple[str, str]] = []
    for rel, item in idx.ordered_items():
        status = item.get("status", "unreviewed")
        label = label_path_for(root, rel)
        if status in exclude:
            skipped.append((rel, f"status={status}"))
            continue
        if require_label and not os.path.exists(label):
            skipped.append((rel, MISSING_LABEL_REASON))
            continue
        if os.path.exists(label) and os.path.getsize(label) == 0:
            empty.append((rel, item))
        else:
            populated.append((rel, item))

    # 划分只认每张图自己的哈希名次，不整表 shuffle，理由见 split_val_set()
    populated = rank_for_split(populated, seed)
    empty = rank_for_split(empty, seed)
    keep_empty = int(len(populated) * max_empty_ratio / max(1e-6, 1 - max_empty_ratio)) if populated else len(empty)
    keep_empty = max(0, keep_empty)
    # 以前是先 empty = empty[:keep_empty] 再取 empty[keep_empty:] 记原因——切完再切，
    # 尾巴永远是空的：1500 张背景图被压到 15% 之后，manifest 里一条记录都没有，
    # 命令行也只报"跳过 1"，用户根本不知道少掉的那几百张去哪了。
    dropped_empty = empty[keep_empty:]
    empty = empty[:keep_empty]
    cap = f"{EMPTY_CAP_REASON}（上限 {max_empty_ratio:.0%}）"
    skipped.extend((rel, cap) for rel, _ in dropped_empty)

    selected = populated + empty
    if not selected:
        raise ValueError("没有符合条件的样本可导出" + skipped_summary(skipped)
                         + no_sample_hint(skipped))
    val_set = split_val_set([rel for rel, _item in selected], val_ratio, seed)
    fingerprint = content_fingerprint(
        [(rel, _read_label(root, rel)) for rel, _item in selected],
        params=json.dumps({"val_ratio": val_ratio, "seed": seed,
                           "max_empty_ratio": max_empty_ratio, "mode": mode,
                           "exclude": sorted(exclude), "coco": bool(coco),
                           "coords": bool(coords)}, sort_keys=True))
    # 内容没变就别再堆一份一模一样的目录：用户找不到路径时最容易连点好几次导出
    if not out and not tag:
        reused = find_reusable_export(root, fingerprint, kind="pose")
        if reused is not None:
            return reused
    tag = tag or now_str().replace(" ", "_").replace(":", "")
    if out:
        out_root = os.path.abspath(out)
    else:
        out_root = unique_export_dir(os.path.abspath(os.path.join(root, "exports", tag)))
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_root, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0, "people": {"train": 0, "val": 0}, "empty": len(empty),
              "empty_dropped": len(dropped_empty)}
    coco_records: List[Tuple[str, dict, str, List]] = []
    for rel, item in selected:
        src_img = os.path.join(os.path.abspath(root), "images", rel)
        src_lbl = label_path_for(root, rel)
        try:
            persons = parse_label_file(src_lbl) if os.path.exists(src_lbl) else []
        except ValueError as exc:
            # 一行坏标签不该毁掉整批导出（以前一张坏文件让两万张的导出直接失败，
            # 而且报错在"文件已经拷了一半"之后）。跳过这一帧，原因照常写进 manifest。
            skipped.append((rel, f"标签解析失败：{exc}"))
            continue
        split = "val" if rel in val_set else "train"
        split_unique = export_stem(rel)
        dst_img = os.path.join(out_root, "images", split, split_unique + ".jpg")
        dst_lbl = os.path.join(out_root, "labels", split, split_unique + ".txt")
        _place(src_img, dst_img, mode)
        if os.path.exists(src_lbl):
            _place(src_lbl, dst_lbl, mode)
        counts[split] += 1
        counts["people"][split] += len(persons)
        coco_records.append((split_unique, item, split, persons))

    if counts["train"] + counts["val"] == 0:
        # 坏标签改成只跳单帧之后，"每一帧都读不通"会导出了一个 0 张的训练集还报成功。
        # 一帧都没落地就别留空目录（只在目录确实空着时删，免得抹掉用户自己指定的 --out）。
        leftovers = [name for split in ("train", "val")
                     for name in os.listdir(os.path.join(out_root, "images", split))]
        if not leftovers:
            shutil.rmtree(out_root, ignore_errors=True)
        bad = [reason for _rel, reason in skipped if reason.startswith("标签解析失败")]
        raise ValueError(f"选中的 {len(selected)} 帧标签全都读不通，一张都没导出"
                         f"（{len(bad)} 帧读不通，例：{bad[0] if bad else '标签为空'}）"
                         + "：按提示里的 文件:行号 修掉坏行，或先跑 autolabel validate 查格式")

    names = idx.class_names
    num_kpt = int(idx.data.get("kpt_shape", [17, 3])[0])
    lines = [f"# 由自动标注系统生成 on {now_str()}", f"path: {out_root}", "train: images/train",
             "val: images/val", f"kpt_shape: [{num_kpt}, 3]",
             f"flip_idx: {flip_idx_for(num_kpt)}"]
    lines.append("names:")
    for cid in sorted(names):
        lines.append(f"  {cid}: {names[cid]}")
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    manifest = {
        "generated_at": now_str(),
        "source_dataset": os.path.abspath(root),
        "seed": seed,
        "val_ratio": val_ratio,
        "max_empty_ratio": max_empty_ratio,
        "mode": mode,
        "counts": counts,
        "skipped": [{"image": rel, "reason": reason} for rel, reason in skipped][:2000],
        "split": [{"image": s, "split": sp, "people": len(p)} for s, _, sp, p in coco_records],
        "fingerprint": fingerprint,
    }
    if coco:
        coco_path = os.path.join(out_root, "coco_keypoints.json")
        write_coco_keypoints(coco_path, coco_records, names, num_kpt)
        manifest["coco"] = coco_path

    coord_out = None
    if coords:
        from .coords import write_coordinate_files

        splits = {stem: split for stem, _item, split, _persons in coco_records}
        coord_result = write_coordinate_files(
            root, out=os.path.join(out_root, "coords"), tag="coords",
            exclude_status=tuple(exclude), splits=splits,
        )
        coord_out = coord_result["out"]
        manifest["coords"] = coord_out
        manifest["coords_counts"] = {"images": coord_result["images"],
                                     "persons": coord_result["persons"]}

    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    return {"out": out_root, **counts, "skipped": len(skipped),
            "skipped_note": skipped_summary(skipped),
            "train_images": counts["train"], "val_images": counts["val"],
            "coords": coord_out}


def export_stem(rel: str) -> str:
    """导出目录里用的文件名：把子目录压进名字里。

    导出目录是平的（``images/train/``），只取 basename 的话 ``a/frame_0001.jpg`` 和
    ``b/frame_0001.jpg`` 会抢同一个 ``frame_0001.jpg``——后放的把前面的连图片带标签一起
    盖掉，实测 4 张图导完只剩 2 个文件，manifest 与 COCO 里却还写着 4 张。
    平铺的数据集（绝大多数）算出来和原来一模一样，只有带子目录的会多一段前缀。
    """
    stem = os.path.splitext(str(rel).replace(os.sep, "/"))[0].replace("/", "__").strip("_")
    return stem or "image"


def flip_idx_for(num_kpt: int) -> List[int]:
    """按关键点数给出 flip_idx：配对点落在范围外的改成自己映自己。

    直接切 ``COCO17_FLIP_IDX[:n]`` 在 n<17 时会把「左膝 → 14」这种越界下标写进
    ``data.yaml``（14 点数据集里根本没有第 14 号点），n>17 时又短一截；
    这两种都是自定义/裁剪关键点模型会踩到的。
    """
    total = max(0, int(num_kpt))
    out: List[int] = []
    for index in range(total):
        partner = COCO17_FLIP_IDX[index] if index < len(COCO17_FLIP_IDX) else index
        out.append(partner if partner < total else index)
    return out


def split_bucket(seed: int, rel: str) -> int:
    """单张图的划分桶（0..999999）：只由 ``seed`` + 相对路径决定，与别的数据无关。"""
    digest = hashlib.sha256(f"{seed}|{rel}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1_000_000


def split_val_set(images: Sequence[str], val_ratio: float, seed: int) -> set:
    """按每张图自己的哈希桶决定 val，而不是把整表 shuffle 后取前 N 张。

    整表 shuffle 的名次取决于「这次一共有多少张」：素材越标越多、每次重导一次，
    上一版还在 val 的这帧就可能掉进 train——同一帧既被训练又被用来算指标，
    两次的指标也就没法比较。哈希桶只跟 (seed, 相对路径, val_ratio) 有关，
    数据集再怎么长，已定过划分的图都不搬家（val 只会新增，不会来回翻）。

    代价是 val 张数会围绕 ``val_ratio`` 小幅波动（小数据集更明显），
    换来的是划分可复现、可增量。仍然保住两条老保证：至少 1 张 val
    （否则 ``yolo pose train`` 的 val 阶段是空的、拿不到指标）、至少 1 张 train。
    """
    pool = list(images)
    if len(pool) < 2 or float(val_ratio) <= 0:
        return set()
    cut = max(1, min(999_999, int(round(float(val_ratio) * 1_000_000))))
    ranked = sorted(pool, key=lambda rel: (split_bucket(seed, rel), rel))
    val = {rel for rel in ranked if split_bucket(seed, rel) < cut}
    if not val:
        val = {ranked[0]}
    if len(val) == len(ranked):
        val.discard(ranked[-1])
    return val


def rank_for_split(items, seed: int, key=lambda item: item[0]):
    """按划分桶排序：让 manifest 顺序与背景图抽样也稳定（姿态库传元组，提示词库传字典）。"""
    return sorted(items, key=lambda item: (split_bucket(seed, key(item)), str(key(item))))


def unique_export_dir(path: str) -> str:
    """同一秒里连点两次导出时目录名会撞车：撞了就加 `-2`、`-3`，别把两次的文件混在一个目录里。"""
    if not os.path.exists(path):
        return path
    for index in range(2, 1000):
        candidate = f"{path}-{index}"
        if not os.path.exists(candidate):
            return candidate
    return f"{path}-{int(time.time())}"


def _read_label(root: str, rel: str) -> bytes:
    try:
        with open(label_path_for(root, rel), "rb") as fh:
            return fh.read()
    except OSError:
        return b"<missing>"


def content_fingerprint(pairs: Sequence[Tuple[str, bytes]], params: str = "") -> str:
    """把「这次要导出哪些内容 + 用什么参数导出」压成一个指纹，用来识别重复导出。

    这里刻意读标签原文而不是文件 mtime/size：手工把 `0.500` 改成 `0.501` 时大小不变，
    只看 size 会误判成"没变化"，把用户刚改的结果当成旧导出。
    """
    digest = hashlib.sha1()
    digest.update(params.encode("utf-8"))
    digest.update(b"\n")
    for rel, payload in sorted(pairs, key=lambda pair: pair[0]):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def find_reusable_export(root: str, fingerprint: str, kind: str = "pose") -> Optional[dict]:
    """`exports/` 里最近一份指纹完全相同的导出目录，命中就直接复用，字段与真实导出保持一致。"""
    if not fingerprint:
        return None
    for row in list_exports(root, limit=8):
        if row.get("kind") != kind:
            continue
        meta_name = "prompt_meta.json" if kind == "detect" else "manifest.json"
        try:
            with open(os.path.join(row["path"], meta_name), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("fingerprint") != fingerprint:
            continue
        counts = meta.get("counts") or {}
        if not counts:
            continue
        if kind == "detect":
            fields = {int(row["class_id"]): row for row in meta.get("fields") or []
                      if str(row.get("class_id", "")).lstrip("-").isdigit()}
            remap = meta.get("class_remap") or {}
            classes = [{"class_id": int(cid), "export_id": int(value),
                        "name": (fields.get(int(cid)) or {}).get("name") or f"class_{cid}"}
                       for cid, value in sorted(remap.items(), key=lambda kv: int(kv[1]))]
            return {"out": row["path"], "reused": True, "images": counts.get("train", 0) + counts.get("val", 0),
                    "train": counts.get("train", 0), "val": counts.get("val", 0),
                    "boxes": counts.get("boxes") or {}, "classes": classes, "counts": counts,
                    "data_yaml": row["data_yaml"]}
        return {"out": row["path"], "reused": True,
                "train": counts.get("train", 0), "val": counts.get("val", 0),
                "people": counts.get("people") or {}, "empty": counts.get("empty", 0),
                "skipped": len(meta.get("skipped") or []),
                "skipped_note": skipped_summary(
                    [(str(r.get("image") or ""), str(r.get("reason") or ""))
                     for r in (meta.get("skipped") or []) if isinstance(r, dict)]),
                "train_images": counts.get("train", 0), "val_images": counts.get("val", 0),
                "coords": meta.get("coords"), "data_yaml": row["data_yaml"]}
    return None


def _place(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if mode == "symlink":
        # 相对链接：整个导出目录（或整个数据集）搬家、换机器挂载点变了都还连得上，
        # 绝对路径的软链一挪目录就全断，只能重新导一份。
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"未知导出模式: {mode}")


def write_coco_keypoints(path: str, records: Sequence[Tuple[str, dict, str, List]],
                         names: Dict[int, str], num_kpt: int) -> None:
    """导出 COCO keypoint 注解，便于与第三方工具/评估脚本对接。"""
    images, annotations = [], []
    ann_id = 1
    for i, (file_name, item, _split, persons) in enumerate(records, start=1):
        images.append({"id": i, "file_name": file_name + ".jpg",
                       "width": int(item.get("width", 0)), "height": int(item.get("height", 0))})
        for person in persons:
            w, h = int(item.get("width", 1)), int(item.get("height", 1))
            x1, y1, x2, y2 = person.xyxy(w, h)
            kps: List[float] = []
            num_visible = 0
            for k in person.keypoints[:num_kpt]:
                kps.extend([round(k.x * w, 2), round(k.y * h, 2), int(k.v)])
                num_visible += int(k.v != 0)
            # 少给点的标签（手改过 / 老版本写的）补零，否则 COCO 的三元组会整体错位
            kps.extend([0.0, 0.0, 0.0] * max(0, num_kpt - len(person.keypoints[:num_kpt])))
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": int(person.cls) + 1,
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "area": round(max(0.0, (x2 - x1) * (y2 - y1)), 2),
                "iscrowd": 0, "num_keypoints": num_visible, "keypoints": kps,
                "segmentation": [],
            })
            ann_id += 1
    skeleton = [[a + 1, b + 1] for a, b in COCO17_SKELETON]
    payload = {
        "info": {"description": "auto-labeled human pose dataset", "date_created": now_str()},
        "licenses": [], "images": images,
        "categories": [{"id": int(cid) + 1, "name": name, "supercategory": "person",
                        "keypoints": list(COCO17_KEYPOINTS[:num_kpt]),
                        "skeleton": skeleton if num_kpt == 17 else []}
                       for cid, name in sorted(names.items())],
        "annotations": annotations,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def count_split_images(root_dir: str, split: str) -> int:
    """数一下导出目录里某个 split 的图片数（symlink 也算，默认导出就是软链）。"""
    target = os.path.join(root_dir, "images", split)
    if not os.path.isdir(target):
        return 0
    try:
        with os.scandir(target) as handle:
            return sum(1 for entry in handle if entry.is_file() or entry.is_symlink())
    except OSError:
        return 0


def list_exports(root: str, limit: int = 8) -> List[dict]:
    """列出 `exports/` 里最近的导出结果，让界面能把产物路径常驻显示出来。

    产物是目录不是文件，路径只闪一次的话用户根本找不到，所以这里连 `data.yaml`
    一起给出去，前端可以直接提示"复制这行就能开训"。
    """
    from .library import exports_dir

    base = exports_dir(root)
    if not os.path.isdir(base):
        return []
    rows: List[dict] = []
    try:
        names = os.listdir(base)
    except OSError:
        return []
    for name in names:
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "prompt_meta.json")):
            kind = "detect"
        elif os.path.exists(os.path.join(path, "manifest.json")):
            kind = "pose"
        else:
            kind = "unknown"
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        train = count_split_images(path, "train")
        val = count_split_images(path, "val")
        rows.append({"name": name, "kind": kind, "path": os.path.abspath(path),
                     "data_yaml": os.path.join(os.path.abspath(path), "data.yaml"),
                     "mtime": mtime,
                     "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "",
                     "train": train, "val": val, "total": train + val})
    # 同一分钟里连点多次导出时，按 mtime 再按目录名排，保证最新那份在前
    rows.sort(key=lambda row: (row["mtime"], row["name"]), reverse=True)
    return rows[:max(1, limit)]
