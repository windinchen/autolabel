"""YOLO pose 标签格式的读写、几何变换与校验。

标签行格式（每行一个人，全部数值按图片宽高归一化）::

    class_id cx cy w h  x1 y1 v1  x2 y2 v2 ... xn yn vn

v 的取值沿用 COCO 约定：``0`` 未标注 / ``1`` 遮挡不可见 / ``2`` 可见。

重要：ultralytics 训练时的可见性掩码是 ``v != 0``
（见 ``ultralytics/utils/loss.py`` 中 ``kpt_mask = gt_kpt[..., 2] != 0``），
也就是说 ``v=1`` 依然参与 keypoint 回归监督。若想真正让某个关键点不产生监督，
必须写成 ``v=0``，这是本系统默认把低置信关键点降级为 0 的原因。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# COCO-17 骨架定义
# --------------------------------------------------------------------------- #

COCO17_KEYPOINTS: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# 水平翻转后需要互换左右的关键点下标（0-based）。
COCO17_FLIP_IDX: Tuple[int, ...] = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)

COCO17_SKELETON: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

# 用于肢体对称性检查的成对肢体：(本肢体两端点, 对称肢体两端点)
SYMMETRIC_LIMBS: Tuple[Tuple[Tuple[int, int], Tuple[int, int]], ...] = (
    ((5, 7), (6, 8)),    # 左上臂 / 右上臂
    ((7, 9), (8, 10)),   # 左前臂 / 右前臂
    ((11, 13), (12, 14)),  # 左大腿 / 右大腿
    ((13, 15), (14, 16)),  # 左小腿 / 右小腿
)

VIS_UNLABELED = 0
VIS_OCCLUDED = 1
VIS_VISIBLE = 2


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class Keypoint:
    """单个关键点，坐标按图片宽高归一化。"""

    x: float
    y: float
    v: int = VIS_VISIBLE
    conf: float = 1.0  # 模型给出的原始置信度，仅用于 QC，不写入标签

    def as_xy(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class Person:
    """一个人的检测结果：归一化 bbox（cxcywh）+ 关键点序列。"""

    cls: int = 0
    cx: float = 0.5
    cy: float = 0.5
    w: float = 0.0
    h: float = 0.0
    keypoints: List[Keypoint] = field(default_factory=list)
    score: float = 0.0
    flags: List[str] = field(default_factory=list)

    # --- bbox 互转 ---
    def xyxy(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """归一化 cxcywh 转像素 xyxy。"""
        half_w = self.w * img_w / 2.0
        half_h = self.h * img_h / 2.0
        cx, cy = self.cx * img_w, self.cy * img_h
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    @classmethod
    def from_xyxy(
        cls,
        xyxy: Sequence[float],
        img_w: int,
        img_h: int,
        keypoints: Optional[List[Keypoint]] = None,
        score: float = 0.0,
        cls_id: int = 0,
    ) -> "Person":
        x1, y1, x2, y2 = xyxy[:4]
        return cls(
            cls=cls_id,
            cx=_clamp((x1 + x2) / 2.0 / max(img_w, 1), 0.0, 1.0),
            cy=_clamp((y1 + y2) / 2.0 / max(img_h, 1), 0.0, 1.0),
            w=_clamp(abs(x2 - x1) / max(img_w, 1), 0.0, 1.0),
            h=_clamp(abs(y2 - y1) / max(img_h, 1), 0.0, 1.0),
            keypoints=list(keypoints or []),
            score=score,
        )

    def visible_keypoints(self) -> List[Keypoint]:
        return [k for k in self.keypoints if k.v != VIS_UNLABELED]

    def keypoints_xyxy(self, img_w: int, img_h: int, only_visible: bool = True) -> List[Tuple[float, float]]:
        pts = self.keypoints if not only_visible else self.visible_keypoints()
        return [(k.x * img_w, k.y * img_h) for k in pts]

    def pixel_area(self, img_w: int, img_h: int) -> float:
        return max(self.w * img_w, 0.0) * max(self.h * img_h, 0.0)


# --------------------------------------------------------------------------- #
# 文本读写
# --------------------------------------------------------------------------- #

def format_person(p: Person, ndigits: int = 6) -> str:
    """把一个人序列化成一行 YOLO pose 标签。"""
    parts = [
        str(int(p.cls)),
        _fmt(p.cx, ndigits), _fmt(p.cy, ndigits), _fmt(p.w, ndigits), _fmt(p.h, ndigits),
    ]
    for k in p.keypoints:
        parts.append(_fmt(k.x, ndigits))
        parts.append(_fmt(k.y, ndigits))
        parts.append(str(int(k.v)))
    return " ".join(parts)


def write_label_file(path: str, persons: Iterable[Person]) -> None:
    """写标签文件；没有目标时写空文件（YOLO 背景图约定）。"""
    lines = [format_person(p) for p in persons]
    text = "".join(line + "\n" for line in lines)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def parse_person(line: str, num_keypoints: Optional[int] = None) -> Person:
    """解析一行 YOLO pose 标签。"""
    tokens = line.strip().split()
    if not tokens:
        raise ValueError("空标签行")
    values = [float(t) for t in tokens[1:5]]
    rest = tokens[5:]
    if len(rest) % 3 != 0:
        raise ValueError(f"关键点字段数不是 3 的倍数: {len(rest)}")
    if num_keypoints is not None and len(rest) // 3 != num_keypoints:
        raise ValueError(f"关键点数量 {len(rest) // 3} 与预期 {num_keypoints} 不符")
    keypoints: List[Keypoint] = []
    for i in range(0, len(rest), 3):
        conf = 1.0 if float(rest[i + 2]) != 0 else 0.0
        keypoints.append(Keypoint(x=float(rest[i]), y=float(rest[i + 1]), v=int(float(rest[i + 2])), conf=conf))
    return Person(cls=int(float(tokens[0])), cx=values[0], cy=values[1], w=values[2], h=values[3],
                  keypoints=keypoints, score=0.0)


def parse_label_file(path: str, num_keypoints: Optional[int] = None) -> List[Person]:
    persons: List[Person] = []
    if not os.path.exists(path):
        return persons
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                persons.append(parse_person(raw, num_keypoints))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno} 解析失败: {exc}") from exc
    return persons


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #

def keypoints_bbox(
    keypoints: Sequence[Keypoint],
    margin_ratio: float = 0.06,
    min_side: float = 0.01,
) -> Optional[Tuple[float, float, float, float]]:
    """由可见关键点求归一化 cxcywh 包围盒，带一点外扩。"""
    pts = [k for k in keypoints if k.v != VIS_UNLABELED]
    if not pts:
        return None
    xs = [k.x for k in pts]
    ys = [k.y for k in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    w = max(x2 - x1, min_side)
    h = max(y2 - y1, min_side)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w *= 1.0 + margin_ratio
    h *= 1.0 + margin_ratio
    return _clamp(cx, 0.0, 1.0), _clamp(cy, 0.0, 1.0), _clamp(w, min_side, 1.0), _clamp(h, min_side, 1.0)


def expand_bbox_with_keypoints(
    person: Person,
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.06,
) -> Person:
    """让 bbox 覆盖所有可见关键点（训练 YOLO-pose 时很关键）。"""
    pts = person.keypoints_xyxy(img_w, img_h)
    if not pts:
        return person
    x1, y1, x2, y2 = person.xyxy(img_w, img_h)
    x1 = min(x1, min(p[0] for p in pts))
    y1 = min(y1, min(p[1] for p in pts))
    x2 = max(x2, max(p[0] for p in pts))
    y2 = max(y2, max(p[1] for p in pts))
    pad_x = (x2 - x1) * margin_ratio
    pad_y = (y2 - y1) * margin_ratio
    x1, y1 = x1 - pad_x, y1 - pad_y
    x2, y2 = x2 + pad_x, y2 + pad_y
    out = Person.from_xyxy((x1, y1, x2, y2), img_w, img_h, person.keypoints, person.score, person.cls)
    return out


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1)) + max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1)) - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(
    a_boxes: Sequence[Sequence[float]],
    b_boxes: Sequence[Sequence[float]],
    iou_thresh: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """按 IoU 贪心匹配两组框，返回 (a_idx, b_idx, iou)。"""
    pairs: List[Tuple[float, int, int]] = []
    for i, a in enumerate(a_boxes):
        for j, b in enumerate(b_boxes):
            v = iou_xyxy(a, b)
            if v >= iou_thresh:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, v))
    return out


def merge_person_flips(primary: Person, mirrored: Person) -> Person:
    """融合正向与翻转两路预测（翻转一路已反映射回原方向）。"""
    kps: List[Keypoint] = []
    n = max(len(primary.keypoints), len(mirrored.keypoints))
    for i in range(n):
        a = primary.keypoints[i] if i < len(primary.keypoints) else None
        b = mirrored.keypoints[i] if i < len(mirrored.keypoints) else None
        if a is None and b is None:
            continue
        if a is None or b is None:
            kps.append(a if a is not None else b)
            continue
        if a.v == VIS_UNLABELED and b.v == VIS_UNLABELED:
            kps.append(Keypoint(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0, v=VIS_UNLABELED,
                                conf=max(a.conf, b.conf)))
            continue
        wa = max(a.conf, 1e-6) if a.v != VIS_UNLABELED else 0.0
        wb = max(b.conf, 1e-6) if b.v != VIS_UNLABELED else 0.0
        if wa + wb <= 0:
            kps.append(Keypoint(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0, v=VIS_UNLABELED, conf=0.0))
            continue
        x = (a.x * wa + b.x * wb) / (wa + wb)
        y = (a.y * wa + b.y * wb) / (wa + wb)
        v = max(a.v, b.v)
        kps.append(Keypoint(x=x, y=y, v=v, conf=max(a.conf, b.conf)))
    # 框跟着关键点一起融合：只留一路的框，融合后的骨架会漂到框外，
    # 再被判成 keypoints_outside_box，让人工去修软件自己造出来的不匹配。
    wa, wb = max(primary.score, 1e-6), max(mirrored.score, 1e-6)
    merged = Person(cls=primary.cls,
                    cx=(primary.cx * wa + mirrored.cx * wb) / (wa + wb),
                    cy=(primary.cy * wa + mirrored.cy * wb) / (wa + wb),
                    w=(primary.w * wa + mirrored.w * wb) / (wa + wb),
                    h=(primary.h * wa + mirrored.h * wb) / (wa + wb),
                    keypoints=kps, score=max(primary.score, mirrored.score))
    return merged


def mirror_person(person: Person, img_w: int, img_h: int,
                  flip_idx: Optional[Sequence[int]] = None) -> Person:
    """把「翻转图上的检测结果」反映射回原图方向（含左右关键点互换）。

    ``flip_idx`` 用模型自带的配对表：自训练的自定义关键点权重顺序未必是 COCO，
    一律套 COCO 表会把左肩和右肩平均到一起——标注看着正常，其实每个点都内缩，
    而且不报错。长度对不上（拿的是别个模型的表）就退回 COCO。
    """
    n = len(person.keypoints)
    if flip_idx is not None and len(flip_idx) == n:
        flip = list(flip_idx)
    else:
        flip = COCO17_FLIP_IDX[:n] if n <= len(COCO17_FLIP_IDX) else list(range(n))
    # 注意：既要镜像 x 坐标，也要把左右下标换回来，否则左肩会和右肩对齐求平均。
    kps: List[Keypoint] = [Keypoint(x=0.0, y=0.0, v=VIS_UNLABELED, conf=0.0) for _ in range(n)]
    for src, dst in enumerate(flip):
        if src >= n or dst >= n:
            continue
        k = person.keypoints[src]
        kps[dst] = Keypoint(x=1.0 - k.x, y=k.y, v=k.v, conf=k.conf)
    cx = 1.0 - person.cx
    return Person(cls=person.cls, cx=cx, cy=person.cy, w=person.w, h=person.h,
                  keypoints=kps, score=person.score)


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #

@dataclass
class LabelIssue:
    path: str
    kind: str
    detail: str
    line: Optional[int] = None


def validate_label_file(
    path: str,
    img_w: int,
    img_h: int,
    num_keypoints: int = 17,
    class_names: Optional[Dict[int, str]] = None,
) -> List[LabelIssue]:
    """检查单个标签文件的格式合法性与数值合理性。"""
    issues: List[LabelIssue] = []
    if not os.path.exists(path):
        return [LabelIssue(path, "missing_label", "缺少标签文件")]
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                person = parse_person(raw, num_keypoints)
            except ValueError as exc:
                issues.append(LabelIssue(path, "parse_error", str(exc), lineno))
                continue
            if class_names is not None and person.cls not in class_names:
                issues.append(LabelIssue(path, "bad_class", f"class_id={person.cls} 未定义", lineno))
            for name, value in (("cx", person.cx), ("cy", person.cy), ("w", person.w), ("h", person.h)):
                if not (0.0 <= value <= 1.0):
                    issues.append(LabelIssue(path, "bbox_range", f"{name}={value:.4f} 越界", lineno))
            if person.w <= 0 or person.h <= 0:
                issues.append(LabelIssue(path, "bbox_degenerate", f"w={person.w:.4f} h={person.h:.4f}", lineno))
            for idx, k in enumerate(person.keypoints):
                if not (0.0 <= k.x <= 1.0 and 0.0 <= k.y <= 1.0):
                    issues.append(LabelIssue(path, "kpt_range", f"关键点 {idx} 坐标越界 ({k.x:.3f},{k.y:.3f})", lineno))
                if int(k.v) not in (0, 1, 2):
                    issues.append(LabelIssue(path, "bad_visibility", f"关键点 {idx} v={k.v}", lineno))
    return issues


def person_sanity_flags(
    person: Person,
    img_w: int,
    img_h: int,
    min_box_px: int = 28,
    min_visible_kpts: int = 5,
    limb_ratio: float = 2.2,
    edge_tol: float = 0.002,
) -> List[str]:
    """给单个人打 QC 标签，供人工复核排序使用。"""
    flags: List[str] = []
    visible = person.visible_keypoints()
    if len(visible) < min_visible_kpts:
        flags.append("few_keypoints")
    if person.pixel_area(img_w, img_h) <= 0 or min(person.w * img_w, person.h * img_h) < min_box_px:
        flags.append("tiny_box")
    x1, y1, x2, y2 = person.xyxy(img_w, img_h)
    # 长宽比必须按像素算：归一化后的 h/w 会把图片自己的宽高比乘进去
    # （ar_norm = ar_px × 图宽/图高），同一个人塞进 16:9 横屏被拉高 1.78 倍、
    # 塞进竖屏手机视频又被压掉一半，判不判「异常」全跟着图片形状走。
    px_w, px_h = person.w * img_w, person.h * img_h
    if px_w >= 1.0:
        ar = px_h / px_w
        if ar < 0.25 or ar > 6.0:
            flags.append("odd_aspect")
    if visible:
        outside = sum(1 for k in visible if not (x1 <= k.x * img_w <= x2 and y1 <= k.y * img_h <= y2))
        if outside > len(visible) * 0.35:
            flags.append("keypoints_outside_box")
        edge = sum(
            1 for k in visible
            if k.x <= edge_tol or k.x >= 1 - edge_tol or k.y <= edge_tol or k.y >= 1 - edge_tol
        )
        if edge >= 3 and len(visible) and edge / len(visible) > 0.3:
            flags.append("edge_clipped")
    if img_w and img_h:
        for (a1, a2), (b1, b2) in SYMMETRIC_LIMBS:
            if max(a1, a2, b1, b2) >= len(person.keypoints):
                continue
            ka1, ka2 = person.keypoints[a1], person.keypoints[a2]
            kb1, kb2 = person.keypoints[b1], person.keypoints[b2]
            if min(ka1.v, ka2.v, kb1.v, kb2.v) == VIS_UNLABELED:
                continue
            la = math.hypot((ka2.x - ka1.x) * img_w, (ka2.y - ka1.y) * img_h)
            lb = math.hypot((kb2.x - kb1.x) * img_w, (kb2.y - kb1.y) * img_h)
            if la < 2 or lb < 2:
                continue
            if max(la, lb) / max(min(la, lb), 1e-6) > limb_ratio:
                flags.append("limb_length_mismatch")
                break
    return flags


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _fmt(value: float, ndigits: int) -> str:
    return f"{float(value):.{ndigits}f}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
