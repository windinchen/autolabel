"""图像读取、感知哈希与可视化绘制。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def load_rgb(path: str, max_side: Optional[int] = None) -> np.ndarray:
    """读取为 RGB ndarray，自动展开 EXIF 旋转。"""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            w, h = im.size
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def image_size(path: str) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        im = ImageOps_exif(im)
        return im.size


def ImageOps_exif(im):
    from PIL import ImageOps

    return ImageOps.exif_transpose(im)


def dhash64(image) -> int:
    """9x8 灰度差分哈希，返回 64bit 整数。"""
    from PIL import Image

    if isinstance(image, np.ndarray):
        gray = image.mean(axis=2) if image.ndim == 3 else image
        small = Image.fromarray(gray.astype(np.uint8)).resize((9, 8), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.int16)
    else:
        arr = np.asarray(image.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


# --------------------------------------------------------------------------- #
# 骨架绘制（QC 渲染 / 复核前的快速预览）
# --------------------------------------------------------------------------- #

# BGR，供 cv2 使用；亦可直接给 RGB（仅颜色观感差异）
_SKELETON_COLORS = [
    (0, 255, 255), (255, 255, 0), (0, 165, 255), (255, 0, 255),
    (0, 255, 0), (255, 255, 255), (0, 0, 255),
]


def draw_poses(
    image: np.ndarray,
    persons: Sequence,
    skeleton: Sequence[Tuple[int, int]],
    keypoint_names: Optional[Sequence[str]] = None,
    show_labels: bool = True,
    show_hidden: bool = False,
    thickness: int = 2,
) -> np.ndarray:
    """在图片副本上绘制 bbox / 骨架 / 关键点。"""
    import cv2

    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    for pi, person in enumerate(persons):
        color = _SKELETON_COLORS[pi % len(_SKELETON_COLORS)]
        x1, y1, x2, y2 = person.xyxy(w, h)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
        pts = [(k.x * w, k.y * h, k.v) for k in person.keypoints]
        for a, b in skeleton:
            if a >= len(pts) or b >= len(pts):
                continue
            if pts[a][2] == 0 or pts[b][2] == 0:
                continue
            cv2.line(canvas, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])),
                     color, thickness, cv2.LINE_AA)
        for idx, (px, py, v) in enumerate(pts):
            if v == 0 and not show_hidden:
                cv2.circle(canvas, (int(px), int(py)), 2, (120, 120, 120), -1, cv2.LINE_AA)
                continue
            r = 4 if v == 2 else 3
            cv2.circle(canvas, (int(px), int(py)), r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(px), int(py)), r, (0, 0, 0), 1, cv2.LINE_AA)
        if show_labels:
            visible = sum(1 for p in pts if p[2] != 0)
            text = f"id{pi} {person.score:.2f} k{visible}/{len(pts)}"
            if person.flags:
                text += " " + ",".join(person.flags[:2])
            _put_text(canvas, text, (int(x1) + 2, max(12, int(y1) - 5)), color)
    return canvas


def _put_text(canvas, text: str, org, color) -> None:
    import cv2

    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
