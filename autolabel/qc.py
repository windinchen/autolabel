"""质量检查：标记汇总、复核优先级排序、Markdown 报告与叠加图渲染。"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import (LIB_DETECT, LIB_OBJECTS, LIB_POSE, is_definitive, label_path_for,
                      load_index, now_str, resolve_image, source_missing)
from .library import field_key, normalize
from .yolo_pose import COCO17_KEYPOINTS, COCO17_SKELETON, parse_label_file

# 复核优先级权重：越大越该先看
FLAG_WEIGHTS = {
    "no_person": 3.0,
    "crowded": 2.0,
    "low_resolution": 1.0,
    "likely_blurred": 2.0,
    "few_keypoints": 3.0,
    "tiny_box": 2.0,
    "odd_aspect": 2.0,
    "keypoints_outside_box": 3.0,
    "edge_clipped": 2.0,
    "limb_length_mismatch": 3.0,
    "model_disagreement": 3.0,
    "only_secondary_model": 2.0,
    # 多模态大模型标注的坐标精度不如 YOLO-pose，整批拉进人工复核清单
    "vlm_labeled": 1.5,
    # 正向标记：多模型一致反而降低复核优先级
    "multi_model_confirmed": -1.5,
}

FLAG_HINTS = {
    "no_person": "未检测到任何人：确认是纯背景，还是漏检/阈值过高",
    "crowded": "人多重叠，骨架容易串号，建议逐个人核对",
    "low_resolution": "分辨率偏低，关键点误差会偏大",
    "likely_blurred": "疑似运动模糊，考虑剔除或降低该帧权重",
    "few_keypoints": "可见关键点过少，可能是远距/严重遮挡",
    "tiny_box": "目标过小，建议提高 imgsz 或考虑剔除",
    "odd_aspect": "框长宽比异常，可能框错或关键点错位",
    "keypoints_outside_box": "关键点大量落在框外，框与骨架不匹配",
    "edge_clipped": "目标贴边被截断，注意左右手互换",
    "limb_length_mismatch": "左右肢体长度差异过大，疑似左右互换或断链",
    "model_disagreement": "多模型结论不一致，建议人工确认",
    "only_secondary_model": "仅辅助模型检出，可能是误检",
    "multi_model_confirmed": "两个模型结论一致，可信度较高",
    "vlm_labeled": "由多模态大模型标注，骨骼坐标精度弱于 YOLO-pose，建议逐张核对",
}

# 正向标记不构成"待复核"理由：它们是"这件事进展得更好"，不是"这里有问题"
# （masked 从 v1.4.6 起只代表轮廓已落盘，轮廓不再被压成方框，也就不是缺陷了）
POSITIVE_FLAGS = {"multi_model_confirmed", "masked"}

# 平均置信度低于这条线，机器自己都没把握，也算"该让人看一眼"
LOW_SCORE = 0.4


def _lib_key(lib: str) -> str:
    """状态字段归属：认得出的库一律按本库算。

    以前只认 ``objects``/``prompt``，剩下的全落姿态库——于是 ``needs_review(item,
    "detect")`` 读的是姿态库的 ``flags``/``score``，目标检测库的「带标记」永远是 0，
    和同一页批次表里的「带标记 3」自相矛盾。认不出的写法（``all``、空串）才退回姿态库。
    """
    try:
        return normalize(lib)
    except ValueError:
        return LIB_POSE


def issue_flags(item: dict, lib: str = LIB_POSE) -> List[str]:
    """这张图上的"真问题"：去掉「多模型结论一致」这类正向标记。"""
    name = _lib_key(lib)
    if name != LIB_POSE:
        # 两个框类库各有各的标记前缀，只看本库那份，别让隔壁库的标记串味
        flags = list(item.get(field_key(name, "flags")) or [])
    else:
        flags = list(item.get("flags", [])) + list(item.get("object_flags", []))
    return [f for f in flags if f not in POSITIVE_FLAGS]


def needs_review(item: dict, lib: str = LIB_POSE) -> bool:
    """这一帧还需不需要人看一眼：没定论，且机器自己报了问题。

    姿态库的问题是 QC 标记或平均置信度偏低；两个框类库（提示词 / 目标检测）
    是标记或压根还没标过框。
    界面右下角的「待确认 N 处」、红点、``F`` 键跳转、门户的「带标记」全部走这里。
    """
    name = _lib_key(lib)
    if is_definitive(item, name):
        return False
    if issue_flags(item, name):
        return True
    if name != LIB_POSE:
        return item.get(field_key(name, "objects")) is None
    score = item.get("score")
    return score is not None and float(score) < LOW_SCORE


def review_counts(pose_rows: List[dict], prompt_rows: List[dict],
                  detect_rows: Optional[List[dict]] = None) -> dict:
    """各库的「等你复核」计数：pending = 还没定论的帧，与各页默认筛选同口径。

    门户那一格、启动日志、各复核页都读这一份，别各自加一遍——以前门户 JS
    自己数，姿态库数"所有没定论的"、提示词库数"带标记的"，同一个数字两套算法。
    """
    counts = {
        "pose_pending": sum(1 for it in pose_rows if not is_definitive(it, LIB_POSE)),
        "pose_flagged": sum(1 for it in pose_rows if needs_review(it, LIB_POSE)),
        "prompt_pending": sum(1 for it in prompt_rows if not is_definitive(it, LIB_OBJECTS)),
        "prompt_flagged": sum(1 for it in prompt_rows if needs_review(it, LIB_OBJECTS)),
    }
    rows = list(detect_rows or [])
    counts["detect_pending"] = sum(1 for it in rows if not is_definitive(it, LIB_DETECT))
    counts["detect_flagged"] = sum(1 for it in rows if needs_review(it, LIB_DETECT))
    # 合计把所有 *_pending / *_flagged 一起加：以后再加库不会漏进合计
    counts["pending"] = sum(value for key, value in counts.items() if key.endswith("_pending"))
    counts["flagged"] = sum(value for key, value in counts.items() if key.endswith("_flagged"))
    return counts


def review_priority(item: dict) -> float:
    flags = list(item.get("flags", [])) + list(item.get("object_flags", []))
    score = sum(FLAG_WEIGHTS.get(f, 1.0) for f in set(flags))
    score += max(0.0, 1.0 - float(item.get("score", 0.0))) * 2.0
    score += max(0.0, 1.0 - float(item.get("visible_ratio", 1.0))) * 2.0
    return round(score, 3)


def flagged_items(root: str, include_reviewed: bool = False) -> List[Tuple[str, dict, float]]:
    """优先复核清单：还没定论、且机器自己报了问题的帧，按优先级从高到低。

    :param include_reviewed: 连人工已经定论（接受 / 驳回）的帧一起给，用于复盘
        "当初驳回的那批到底是什么问题"。默认不含——已驳回的帧早就排除了出训练集，
        再挂在清单最前面只会让人反复看同一张。
    """
    idx = load_index(root)
    out = []
    for rel, item in idx.ordered_items():
        if not include_reviewed and is_definitive(item, LIB_POSE):
            continue
        flags = issue_flags(item, LIB_POSE)
        if flags or needs_review(item, LIB_POSE):
            out.append((rel, item, review_priority(item)))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out


def dataset_stats(root: str) -> dict:
    """全量统计，用于报告与阈值调优。"""
    idx = load_index(root)
    num_kpt = int(idx.data.get("kpt_shape", [17, 3])[0])
    status_counts: Dict[str, int] = {}
    flag_counts: Dict[str, int] = {}
    per_kpt = [{"name": COCO17_KEYPOINTS[i % len(COCO17_KEYPOINTS)], "v0": 0, "v1": 0, "v2": 0}
               for i in range(num_kpt)]
    persons_total = 0
    persons_per_image: List[int] = []
    box_heights: List[float] = []
    box_scores: List[float] = []
    missing_labels = 0
    bad_labels = 0
    missing_files: List[str] = []
    for rel, item in idx.ordered_items():
        status = item.get("status", "unreviewed")
        status_counts[status] = status_counts.get(status, 0) + 1
        if source_missing(root, rel):
            # 图被手工删了：标签可能还在，所以"缺标签 0"完全不提这件事，
            # 而导出偏偏会在这一帧上崩或留下读不出来的图。必须单开一个格子。
            missing_files.append(rel)
        for flag in list(item.get("flags", [])) + list(item.get("object_flags", [])):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        lp = label_path_for(root, rel)
        if not os.path.exists(lp):
            missing_labels += 1
            continue
        h = int(item.get("height", 0))
        try:
            persons = parse_label_file(lp, num_kpt)
        except ValueError:
            # 坏标签既不缺也不在，统计里必须有个格子，否则"图片 100 / 人数 0"没人解释
            bad_labels += 1
            continue
        persons_per_image.append(len(persons))
        persons_total += len(persons)
        for person in persons:
            box_heights.append(person.h * h if h else person.h)
            if person.score:
                box_scores.append(person.score)
            for i, k in enumerate(person.keypoints[:num_kpt]):
                per_kpt[i][f"v{int(k.v)}"] += 1

    def pct(values, q):
        return round(float(np.percentile(values, q)), 2) if values else 0.0

    return {
        "generated_at": now_str(),
        "dataset": idx.name,
        "images": len(idx.items),
        "missing_labels": missing_labels,
        "missing_files": len(missing_files),
        "missing_images": missing_files[:50],
        "bad_labels": bad_labels,
        "status_counts": status_counts,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
        "persons_total": persons_total,
        "persons_per_image": {
            "mean": round(float(np.mean(persons_per_image)), 2) if persons_per_image else 0.0,
            "median": pct(persons_per_image, 50),
            "max": int(max(persons_per_image)) if persons_per_image else 0,
            "empty_images": int(sum(1 for v in persons_per_image if v == 0)),
            # 缺标签的帧在统计前就 continue 了：不写清分母，"平均每图 1.0、无人图 0"
            # 会让人以为 5 张图都标过，其实只有 2 张有标签
            "counted": len(persons_per_image),
        },
        "box_height_px": {"p10": pct(box_heights, 10), "p50": pct(box_heights, 50),
                          "p90": pct(box_heights, 90), "min": pct(box_heights, 0),
                          "max": pct(box_heights, 100)},
        "model_score": {"mean": pct(box_scores, 50), "p10": pct(box_scores, 10),
                        "p90": pct(box_scores, 90)},
        "keypoint_visibility": per_kpt,
        "label_settings": idx.data.get("settings", {}),
    }


def pending_line(stats: dict, flagged_count: int, include_reviewed: bool) -> str:
    """「待复核」那一行：加了 ``--include-reviewed`` 就不能再说"已定论的不占位"。"""
    settled = stats["status_counts"].get("approved", 0) + stats["status_counts"].get("rejected", 0)
    tail = (f"已定论的 {settled} 张也算进来了" if include_reviewed
            else f"已定论 {settled} 张不占位")
    return f"- 待复核：{flagged_count} 张（既没接受也没驳回、且机器自己报了问题；{tail}）"


def missing_source_line(stats: dict) -> str:
    """原图缺失必须单开一行：缺标签 0 完全不提这件事，而导出偏偏会在这些帧上跳过。"""
    gone = stats.get("missing_files", 0)
    if not gone:
        return ""
    return (f"- 原图缺失：{gone} 张（索引里记着、`images/` 里已经没有了；"
            f"导出时整帧跳过，导出清单里写「源文件缺失」）")


def write_report(root: str, top_n: int = 30, include_reviewed: bool = False) -> dict:
    """生成 qc/report.md、qc/flags.csv、qc/stats.json。"""
    idx = load_index(root)
    stats = dataset_stats(root)
    qc_root = os.path.join(os.path.abspath(root), "qc")
    os.makedirs(qc_root, exist_ok=True)
    with open(os.path.join(qc_root, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    flagged = flagged_items(root, include_reviewed=include_reviewed)
    with open(os.path.join(qc_root, "flags.csv"), "w", encoding="utf-8") as fh:
        fh.write("image,priority,status,persons,score,visible_ratio,flags\n")
        for rel, item, priority in flagged:
            flags = "|".join(sorted(set(list(item.get("flags", [])) + list(item.get("object_flags", [])))))
            fh.write(f'{rel},{priority},{item.get("status","unreviewed")},{item.get("objects","")},'
                     f'{item.get("score","")},{item.get("visible_ratio","")},{flags}\n')

    lines: List[str] = [f"# QC 报告 · {idx.name}", "", f"- 生成时间：{stats['generated_at']}",
                        f"- 标注参数：`{json.dumps(stats['label_settings'], ensure_ascii=False, sort_keys=True)}`",
                        "", "## 总览", "",
                        f"- 图片总数：{stats['images']}（缺标签 {stats['missing_labels']} · "
                        f"标签解析失败 {stats['bad_labels']} · 原图缺失 {stats['missing_files']}）",
                        f"- 人数合计：{stats['persons_total']}，平均每图 {stats['persons_per_image']['mean']} 人，"
                        f"无人图 {stats['persons_per_image']['empty_images']} 张"
                        f"（只按有标签的 {stats['persons_per_image']['counted']} 张算；"
                        f"缺标签 {stats['missing_labels']} 张、读不通 {stats['bad_labels']} 张不占这两项）",
                        "- 复核状态：" + ", ".join(f"{k}={v}" for k, v in sorted(stats["status_counts"].items())),
                        pending_line(stats, len(flagged), include_reviewed),
                        missing_source_line(stats),
                        "", "## 框高度分布（像素）", "",
                        "| p10 | p50 | p90 | min | max |", "| --- | --- | --- | --- | --- |",
                        f"| {stats['box_height_px']['p10']} | {stats['box_height_px']['p50']} | "
                        f"{stats['box_height_px']['p90']} | {stats['box_height_px']['min']} | "
                        f"{stats['box_height_px']['max']} |",
                        "", "> p10 偏小时建议提高 `--imgsz`，否则小人的关键点会被下采样抹平。",
                        "", "## 标记分布", "", "| 标记 | 命中数 | 处理建议 |", "| --- | --- | --- |"]
    for flag, count in stats["flag_counts"].items():
        lines.append(f"| `{flag}` | {count} | {FLAG_HINTS.get(flag, '人工确认')} |")
    lines += ["", "## 关键点可见性分布", "",
              "| 关键点 | v=0 忽略 | v=1 遮挡 | v=2 可见 | 忽略率 |", "| --- | --- | --- | --- | --- |"]
    for row in stats["keypoint_visibility"]:
        total = row["v0"] + row["v1"] + row["v2"] or 1
        lines.append(f"| {row['name']} | {row['v0']} | {row['v1']} | {row['v2']} | "
                     f"{row['v0'] / total:.1%} |")
    lines += ["", "> `v=1` 在 ultralytics 中同样参与监督；想把某类关键点彻底排除，需改成 `v=0`。",
              "", "## 优先复核清单（前 %d）" % top_n, "",
              "| 优先级 | 图片 | 人数 | 置信 | 标记 |", "| --- | --- | --- | --- | --- |"]
    for rel, item, priority in flagged[:top_n]:
        flags = ", ".join(sorted(set(list(item.get("flags", [])) + list(item.get("object_flags", [])))))
        lines.append(f"| {priority} | `{rel}` | {item.get('objects', '-')} | "
                     f"{item.get('score', '-')} | {flags} |")
    lines += ["", "## 下一步", "",
              "1. 打开复核界面逐张确认：`python3 -m autolabel review <dataset>`",
              "2. 只导出已确认的样本：`python3 -m autolabel export <dataset> --only-approved`",
              "3. 训练：`yolo pose train data=<export>/data.yaml model=yolo11s-pose.pt epochs=100`", ""]
    report = os.path.join(qc_root, "report.md")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    stats["report"] = report
    stats["flagged_count"] = len(flagged)
    stats["flagged"] = [{"image": rel, "priority": p, "flags": sorted(set(
        list(item.get("flags", [])) + list(item.get("object_flags", []))))} for rel, item, p in flagged[:top_n]]
    return stats


def render_overlays(root: str, limit: int = 40, only_flagged: bool = True,
                    max_side: int = 1280, show_hidden: bool = True) -> List[str]:
    """渲染带骨架叠加的预览图，用于肉眼快速扫描。"""
    import cv2

    from .image_utils import draw_poses, load_rgb

    idx = load_index(root)
    if only_flagged:
        targets = [rel for rel, _, _ in flagged_items(root)[:limit]]
        if not targets:
            # 没有命中任何标记时退回按文件名渲染，避免 --render 出现空结果
            targets = [rel for rel, _ in idx.ordered_items()][:limit]
    else:
        targets = [rel for rel, _ in idx.ordered_items()][:limit]
    outputs: List[str] = []
    for rel in targets:
        item = idx.get(rel) or {}
        try:
            image = load_rgb(resolve_image(root, rel), max_side=max_side)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 读取失败 {rel}: {exc}")
            continue
        h, w = image.shape[:2]
        try:
            persons = parse_label_file(label_path_for(root, rel))
        except ValueError as exc:
            print(f"[warn] 标签解析失败 {rel}: {exc}")
            continue
        for person in persons:
            person.score = float(person.score or item.get("score") or 0.0)
        canvas = draw_poses(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), persons, COCO17_SKELETON,
                            show_hidden=show_hidden)
        out = os.path.join(os.path.abspath(root), "qc", "overlays", os.path.splitext(rel)[0] + ".jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        outputs.append(out)
    return outputs
