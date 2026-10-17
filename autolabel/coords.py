"""坐标导出：把 YOLO 标签转成 CSV / JSONL 等通用文件，方便自查、对账和二次开发。

输出目录默认 <dataset>/coords/<时间戳>/，包含：
  - coords.csv       宽表，一行一个人，17 个关键点展开成列（Excel / pandas 友好）
  - coords_long.csv  长表，一行一个关键点（做统计、透视表方便）
  - coords.jsonl     一行一张图，嵌套 bbox + 关键点，程序读取方便
  - meta.json        关键点名称、骨架连线、字段含义、筛选条件
  - summary.json     本次导出的数量统计

坐标同时给出像素值和归一化值：v=0 不参与 YOLO 监督，v=1/2 都参与。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import label_path_for, load_index, now_str
from .yolo_pose import COCO17_KEYPOINTS, COCO17_SKELETON, parse_label_file, person_sanity_flags

WIDE_FILE = "coords.csv"
LONG_FILE = "coords_long.csv"
JSONL_FILE = "coords.jsonl"
META_FILE = "meta.json"
SUMMARY_FILE = "summary.json"

DEFAULT_FORMATS = ("csv", "long", "jsonl")

# 宽表里关键点列之外的固定列
BASE_COLUMNS = [
    "image", "split", "status", "person_index", "class_name",
    "image_width", "image_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
    "person_flags", "image_flags", "image_score",
    "time_sec", "frame_index", "video", "source",
]

LONG_COLUMNS = [
    "image", "split", "status", "person_index", "class_name",
    "keypoint_index", "keypoint_name", "x", "y", "v", "x_norm", "y_norm",
]

COLUMN_DOC = {
    "image": "images/ 下的相对路径",
    "split": "导出时的划分（仅 export --coords 时有值）",
    "status": "复核状态 unreviewed/approved/edited/rejected",
    "person_index": "同一张图内的人序号，从 0 开始",
    "bbox_x1..y2": "边界框，像素坐标 xyxy",
    "bbox_cx..h": "边界框，归一化 cxcywh（与 YOLO 标签一致）",
    "kpXX_name_x/y": "关键点像素坐标",
    "kpXX_name_v": "可见性：0 未标注（不参与训练监督），1 遮挡，2 可见",
    "image_score": "该图自动标注的平均置信度",
    "time_sec": "视频抽帧时的原始时间点（秒），非视频图为空",
    "frame_index": "抽帧序号，非视频图为空",
}


def coords_dir(root: str) -> str:
    return os.path.join(os.path.abspath(root), "coords")


def wide_columns(keypoint_names: Sequence[str]) -> List[str]:
    """固定列 + 每个关键点三列（x / y / v）。"""
    columns = list(BASE_COLUMNS)
    for i, name in enumerate(keypoint_names, start=1):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or f"kp{i}"
        base = f"kp{i:02d}_{safe}"
        columns += [f"{base}_x", f"{base}_y", f"{base}_v"]
    return columns


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #

def collect_records(
    root: str,
    exclude_status: Sequence[str] = ("rejected",),
    require_label: bool = True,
    splits: Optional[Dict[str, str]] = None,
) -> Tuple[List[dict], dict]:
    """把标签解析成带像素坐标的记录。

    :return: (逐图记录, 数据集元信息)
    """
    idx = load_index(root)
    num_kpt = idx.num_keypoints
    source_names = list(idx.data.get("keypoint_names") or COCO17_KEYPOINTS)
    names = [source_names[i] if i < len(source_names) else f"kp{i}" for i in range(num_kpt)]
    class_names = idx.class_names
    skip = set(exclude_status)

    records: List[dict] = []
    skipped: List[dict] = []
    for rel, item in idx.ordered_items():
        status = item.get("status", "unreviewed")
        if status in skip:
            skipped.append({"image": rel, "reason": f"status={status}"})
            continue
        label = label_path_for(root, rel)
        if not os.path.exists(label):
            if require_label:
                skipped.append({"image": rel, "reason": "缺少标签"})
                continue
            persons: List = []
        else:
            persons = parse_label_file(label, num_kpt)
        width = int(item.get("width", 0) or 0)
        height = int(item.get("height", 0) or 0)
        stem = os.path.splitext(os.path.basename(rel))[0]
        objects: List[dict] = []
        for pi, person in enumerate(persons):
            x1, y1, x2, y2 = person.xyxy(width, height)
            keypoints = [{"i": ki, "name": names[ki] if ki < len(names) else f"kp{ki}",
                          "x": round(k.x * width, 2), "y": round(k.y * height, 2), "v": int(k.v),
                          "x_norm": round(k.x, 6), "y_norm": round(k.y, 6)}
                         for ki, k in enumerate(person.keypoints)]
            objects.append({
                "index": pi,
                "cls": int(person.cls),
                "class_name": class_names.get(int(person.cls), str(person.cls)),
                "bbox_xyxy": [round(v, 2) for v in (x1, y1, x2, y2)],
                "bbox_cxcywh_norm": [round(person.cx, 6), round(person.cy, 6),
                                     round(person.w, 6), round(person.h, 6)],
                "flags": person_sanity_flags(person, width, height) if width and height else [],
                "keypoints": keypoints,
            })
        records.append({
            "image": rel,
            "stem": stem,
            "split": (splits or {}).get(stem, ""),
            "status": status,
            "width": width,
            "height": height,
            "image_score": item.get("score"),
            "image_flags": list(item.get("flags") or []),
            "time_sec": item.get("time_sec"),
            "frame_index": item.get("frame_index"),
            "video": item.get("video"),
            "source": item.get("source"),
            "objects": objects,
        })

    meta = {
        "keypoint_names": names,
        "class_names": {str(k): v for k, v in class_names.items()},
        "num_keypoints": num_kpt,
        "skeleton": [list(s) for s in COCO17_SKELETON if max(s) < num_kpt],
        "source_dataset": os.path.abspath(root),
        "generated_at": now_str(),
        "filter": {"exclude_status": sorted(skip), "require_label": require_label},
        "columns": COLUMN_DOC,
        "skipped": skipped,
    }
    return records, meta


# --------------------------------------------------------------------------- #
# 写文件
# --------------------------------------------------------------------------- #

def _write_wide(path: str, records: Sequence[dict], names: Sequence[str]) -> int:
    columns = wide_columns(names)
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            for obj in rec["objects"]:
                row = {
                    "image": rec["image"], "split": rec["split"], "status": rec["status"],
                    "person_index": obj["index"], "class_name": obj["class_name"],
                    "image_width": rec["width"], "image_height": rec["height"],
                    "bbox_x1": obj["bbox_xyxy"][0], "bbox_y1": obj["bbox_xyxy"][1],
                    "bbox_x2": obj["bbox_xyxy"][2], "bbox_y2": obj["bbox_xyxy"][3],
                    "bbox_cx": obj["bbox_cxcywh_norm"][0], "bbox_cy": obj["bbox_cxcywh_norm"][1],
                    "bbox_w": obj["bbox_cxcywh_norm"][2], "bbox_h": obj["bbox_cxcywh_norm"][3],
                    "person_flags": "|".join(obj["flags"]),
                    "image_flags": "|".join(rec["image_flags"]),
                    "image_score": rec["image_score"] if rec["image_score"] is not None else "",
                    "time_sec": rec["time_sec"] if rec["time_sec"] is not None else "",
                    "frame_index": rec["frame_index"] if rec["frame_index"] is not None else "",
                    "video": rec["video"] or "", "source": rec["source"] or "",
                }
                for i, kp in enumerate(obj["keypoints"]):
                    base = f"kp{i + 1:02d}_{names[i]}" if i < len(names) else f"kp{i + 1:02d}_kp{i}"
                    row[f"{base}_x"] = kp["x"]
                    row[f"{base}_y"] = kp["y"]
                    row[f"{base}_v"] = kp["v"]
                writer.writerow(row)
                rows += 1
    return rows


def _write_long(path: str, records: Sequence[dict]) -> int:
    rows = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LONG_COLUMNS)
        writer.writeheader()
        for rec in records:
            for obj in rec["objects"]:
                for kp in obj["keypoints"]:
                    writer.writerow({
                        "image": rec["image"], "split": rec["split"], "status": rec["status"],
                        "person_index": obj["index"], "class_name": obj["class_name"],
                        "keypoint_index": kp["i"], "keypoint_name": kp["name"],
                        "x": kp["x"], "y": kp["y"], "v": kp["v"],
                        "x_norm": kp["x_norm"], "y_norm": kp["y_norm"],
                    })
                    rows += 1
    return rows


def _write_jsonl(path: str, records: Sequence[dict]) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def write_coordinate_files(
    root: str,
    out: Optional[str] = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    exclude_status: Sequence[str] = ("rejected",),
    require_label: bool = True,
    tag: Optional[str] = None,
    splits: Optional[Dict[str, str]] = None,
) -> dict:
    """生成坐标文件，返回 {out, files, images, persons}。"""
    records, meta = collect_records(root, exclude_status=exclude_status,
                                    require_label=require_label, splits=splits)
    out_root = os.path.abspath(out or os.path.join(coords_dir(root),
                                                   tag or now_str().replace(" ", "_").replace(":", "")))
    os.makedirs(out_root, exist_ok=True)

    names = ["".join(c if c.isalnum() or c in "-_" else "_" for c in n) or f"kp{i}"
             for i, n in enumerate(meta["keypoint_names"])]
    want = {f.strip().lower() for f in formats if f.strip()}
    files: Dict[str, str] = {}
    rows_wide = rows_long = 0
    if "csv" in want:
        files[WIDE_FILE] = os.path.join(out_root, WIDE_FILE)
        rows_wide = _write_wide(files[WIDE_FILE], records, names)
    if "long" in want:
        files[LONG_FILE] = os.path.join(out_root, LONG_FILE)
        rows_long = _write_long(files[LONG_FILE], records)
    if "jsonl" in want:
        files[JSONL_FILE] = os.path.join(out_root, JSONL_FILE)
        _write_jsonl(files[JSONL_FILE], records)

    persons = sum(len(rec["objects"]) for rec in records)
    status_counts: Dict[str, int] = {}
    for rec in records:
        status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1
    summary = {
        "generated_at": meta["generated_at"],
        "source_dataset": meta["source_dataset"],
        "images": len(records),
        "images_with_people": sum(1 for rec in records if rec["objects"]),
        "images_empty": sum(1 for rec in records if not rec["objects"]),
        "persons": persons,
        "keypoints_written": persons * meta["num_keypoints"],
        "status_counts": status_counts,
        "rows": {WIDE_FILE: rows_wide, LONG_FILE: rows_long, JSONL_FILE: len(records)},
        "filter": meta["filter"],
        "skipped_items": meta["skipped"][:200],
    }
    meta["files"] = {k: os.path.basename(v) for k, v in files.items()}
    with open(os.path.join(out_root, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, SUMMARY_FILE), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    files[META_FILE] = os.path.join(out_root, META_FILE)
    files[SUMMARY_FILE] = os.path.join(out_root, SUMMARY_FILE)
    return {"out": out_root, "files": files, "images": len(records), "persons": persons,
            "records": records, "meta": meta}


DOWNLOAD_NAMES = {"csv": WIDE_FILE, "long": LONG_FILE, "jsonl": JSONL_FILE}


def render_coords(root: str, fmt: str = "csv",
                  exclude_status: Sequence[str] = ("rejected",)) -> Tuple[str, bytes]:
    """按需生成单个坐标文件（不落盘到数据集），给复核界面的“下载坐标”用。"""
    if fmt not in DOWNLOAD_NAMES:
        raise ValueError(f"未知格式: {fmt}（可选 {', '.join(DOWNLOAD_NAMES)}）")
    tmp = tempfile.mkdtemp(prefix="autolabel_coords_")
    try:
        result = write_coordinate_files(root, out=tmp, tag="dl", formats=(fmt,),
                                        exclude_status=exclude_status)
        path = result["files"][DOWNLOAD_NAMES[fmt]]
        with open(path, "rb") as fh:
            body = fh.read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.basename(os.path.abspath(root)))
    name = f"{safe or 'dataset'}-{DOWNLOAD_NAMES[fmt]}"
    return name, body
