"""多模态视觉大模型标注后端：把图片交给视觉大模型，直接输出 COCO-17 骨骼点。

对外接口与 ``backends.PoseEnsemble`` 对齐（``load`` / ``infer`` / ``display_names``），
所以标注流水线只换后端、不改流程。坐标约定与 YOLO 一致：按图片宽高归一化。
"""

from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import models as model_manager
from .backends import BackendSettings
from .yolo_pose import (
    COCO17_KEYPOINTS,
    VIS_OCCLUDED,
    VIS_UNLABELED,
    VIS_VISIBLE,
    Keypoint,
    Person,
    expand_bbox_with_keypoints,
    keypoints_bbox,
)

# 提示词里让模型用 0-1000 整数回报坐标：多数视觉大模型对整数相对坐标比小数稳
COORD_SCALE = 1000.0
# 大模型不回报置信度，用固定分占位；复核优先级靠 vlm_labeled 标记保证
DEFAULT_SCORE = 0.5
VLM_FLAG = "vlm_labeled"
MAX_IMAGE_SIDE = 1024
# 思考型模型要先写推理再写 JSON，输出上限给小了会只剩思考、正文为空
DEFAULT_MAX_TOKENS = 4096
# 实测本地 8B 级视觉模型标一张要 20~40 秒，配置的超时太小会一直“失败”
MIN_TIMEOUT = 60


class VLMError(RuntimeError):
    """多模态配置缺失、调用失败或返回内容无法解析。"""


def build_prompt(num_keypoints: int = len(COCO17_KEYPOINTS)) -> str:
    """给视觉大模型的标注指令：点名顺序、坐标尺度、可见性语义、只回 JSON。"""
    names = "、".join(f"{i + 1}.{n}" for i, n in enumerate(COCO17_KEYPOINTS[:num_keypoints]))
    return (
        "你是人体姿态标注助手。请标出图中每一个人的身体关键点，坐标要尽量准。\n"
        f"关键点顺序固定为 {num_keypoints} 个：{names}。\n"
        "坐标用 0-1000 的整数，表示相对图片宽高的位置：左上角 (0,0)，右下角 (1000,1000)。\n"
        "每个点的第三个值是可见性：2=明确可见，1=被遮挡但位置可推断，0=画面里看不到该部位。\n"
        "看不到的人不要编，画面里没有人就返回空列表。\n"
        '只输出 JSON，不要任何解释文字，格式：\n'
        '{"people":[{"bbox":[x1,y1,x2,y2],"keypoints":[[x,y,v], ...共%d项]}]}' % num_keypoints
    )


def encode_image(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE, quality: int = 82) -> str:
    """把 RGB 数组压成 data URL，控制边长以免请求体过大。"""
    import base64
    from io import BytesIO

    from PIL import Image

    array = np.ascontiguousarray(image)
    h, w = array.shape[:2]
    longest = max(h, w)
    limit = max(int(max_side), 128)
    if longest > limit:
        ratio = limit / float(longest)
        array = np.asarray(Image.fromarray(array).resize(
            (max(int(w * ratio), 1), max(int(h * ratio), 1)), Image.BILINEAR))
    buffer = BytesIO()
    Image.fromarray(array).convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def chat(profile: dict, prompt: str, image_data_url: str,
         timeout: Optional[int] = None, max_tokens: Optional[int] = None) -> str:
    """发一次带图的 chat 请求，返回模型文本；失败抛 :class:`VLMError`。"""
    from .models import _request  # 复用同一套 HTTP 与错误翻译

    base = model_manager.normalize_base_url(profile.get("base_url") or "")
    headers = {"User-Agent": "autolabel-vlm"}
    key = str(profile.get("api_key") or "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    limit = max(int(timeout or profile.get("timeout") or MIN_TIMEOUT), MIN_TIMEOUT)
    tokens = int(max_tokens or profile.get("max_tokens") or DEFAULT_MAX_TOKENS)
    payload = {
        "model": str(profile.get("model") or ""),
        "temperature": 0,
        "max_tokens": tokens,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
    }
    if profile.get("json_mode"):
        # 部分本地服务不认 response_format，所以默认关掉，用户在配置里显式打开才发
        payload["response_format"] = {"type": "json_object"}
    status, body, err = _request(base + "/chat/completions", headers, max(limit, 5), payload)
    if status != 200:
        raise VLMError(err or f"HTTP {status}：{base}/chat/completions")
    text = _reply_text(body)
    if not text.strip():
        raise VLMError("模型返回空内容：思考型模型可能把输出额度都花在推理上了，"
                       "换非思考模型或调大输出上限再试")
    return text


def _reply_text(body: str) -> str:
    """取回模型正文，兼容 content 为字符串/分段数组；正文为空时回落到思考内容。"""
    try:
        data = json.loads(body or "")
    except json.JSONDecodeError:
        return ""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = _flatten_content((message or {}).get("content"))
    if content.strip():
        return content
    # 有些推理模型把结论只写进 reasoning_content，里面通常也带着最终 JSON
    return _flatten_content((message or {}).get("reasoning_content"))


def _flatten_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 部分服务把分段数组原样返回
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def json_block(text: str) -> str:
    """从模型回复里切出 JSON 主体（去掉 ```json 包裹和前后废话）。"""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S | re.I)
    body = fenced.group(1) if fenced else text
    start = body.find("{")
    if start < 0:
        raise VLMError("模型没返回 JSON：" + " ".join(text.split())[:120])
    end = body.rfind("}")
    # 被截断的回复没有收尾花括号，先原样交给下层按片段抢救
    return body[start:end + 1] if end > start else body[start:]


def parse_people(text: str, num_keypoints: int = len(COCO17_KEYPOINTS)) -> List[Person]:
    """把模型 JSON 转成 ``Person`` 列表；坐标统一到归一化，bbox 兜底包住可见点。"""
    entries = _entries(text)
    persons: List[Person] = []
    for entry in entries:
        person = _one_person(entry, num_keypoints)
        if person is not None:
            persons.append(person)
    return persons


def _entries(text: str) -> List[dict]:
    """先按整段 JSON 解析；大模型偶尔漏逗号，那就退化成逐个抓 keypoints 数组。"""
    try:
        data = json.loads(json_block(text))
        strict = True
    except json.JSONDecodeError:
        data, strict = None, False

    if not strict:
        scanned = _scan_entries(text)
        if scanned:
            return scanned
        raise VLMError("返回的 JSON 残缺到无法解析（多半是漏逗号或被截断）："
                       + " ".join((text or "").split())[:120])

    if isinstance(data, list):
        raw: object = data
    elif isinstance(data, dict):
        raw = data.get("people")
        if raw is None:
            raw = data.get("persons") or data.get("detections") or []
    else:
        raw = []
    if not isinstance(raw, list):
        raise VLMError("people 字段不是列表，无法解析")
    entries = [entry for entry in raw if isinstance(entry, dict)]
    if not entries or any(_point_count(entry) for entry in entries):
        return entries  # 结构完整就直接采信，包括“确实没有人”的空列表
    return _scan_entries(text) or entries


def _point_count(entry: dict) -> int:
    points = entry.get("keypoints") or entry.get("kpts") or entry.get("pose") or []
    return len(points) if isinstance(points, list) else 0


def _scan_entries(text: str) -> List[dict]:
    """从残缺 JSON 里逐个救出 keypoints 数组（数组内只有数字，正则足够稳）。"""
    entries: List[dict] = []
    for match in re.finditer(r'"(?:keypoints|kpts|pose)"\s*:\s*\[', text or ""):
        block = _balanced_array(text, match.end() - 1)
        if not block:
            continue
        points = _points_from_block(block)
        if not points:
            continue
        entry: dict = {"keypoints": points}
        box = _bbox_before(text[:match.start()])
        if box is not None:
            entry["bbox"] = box
        entries.append(entry)
    return entries


def _points_from_block(block: str) -> List[List[float]]:
    points: List[List[float]] = []
    for x, y, v in re.findall(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*(?:,\s*(-?[\d.]+)\s*)?\]", block):
        try:
            points.append([float(x), float(y), float(v) if v else float(VIS_VISIBLE)])
        except ValueError:
            continue
    return points


def _balanced_array(text: str, start: int) -> str:
    """取一个可能没闭合的数组文本；截断就退回最后一个闭合的内层元素。"""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    tail = text.rfind("]", start, min(len(text), start + 40000))
    return text[start:tail + 1] + "]" if tail > start else ""


def _bbox_before(prefix: str) -> Optional[List[float]]:
    index = prefix.rfind('"bbox"')
    if index < 0:
        return None
    bracket = prefix.find("[", index)
    if bracket < 0:
        return None
    try:
        values = json.loads(_balanced_array(prefix, bracket) or "null")
    except json.JSONDecodeError:
        return None
    return values if isinstance(values, list) and len(values) >= 4 else None


def _one_person(entry: object, num_keypoints: int) -> Optional[Person]:
    if not isinstance(entry, dict):
        return None
    raw_points = entry.get("keypoints")
    if raw_points is None:
        raw_points = entry.get("kpts") or entry.get("pose") or []
    if not isinstance(raw_points, list) or not raw_points:
        return None
    scale = _coord_scale(raw_points)
    keypoints: List[Keypoint] = []
    for item in raw_points[:num_keypoints]:
        point = _one_keypoint(item, scale)
        if point is not None:
            keypoints.append(point)
    if len(keypoints) < num_keypoints:  # 少给的点补成“未标注”，保证字段数恒定
        keypoints += [Keypoint(x=0.0, y=0.0, v=VIS_UNLABELED, conf=0.0)
                      for _ in range(num_keypoints - len(keypoints))]
    if not any(k.v != VIS_UNLABELED for k in keypoints):
        return None
    person = Person(keypoints=keypoints, score=DEFAULT_SCORE, flags=[VLM_FLAG])
    box = _box_from_entry(entry.get("bbox"), scale)
    if box is None:
        box = keypoints_bbox(keypoints, margin_ratio=0.06)
    if box is None:
        return None
    person.cx, person.cy, person.w, person.h = box
    # 模型给的框常常偏小，按关键点再撑开一次，避免“点在线段外、框没包住”
    expanded = expand_bbox_with_keypoints(person, 1, 1, margin_ratio=0.02)
    expanded.flags = [VLM_FLAG]
    return expanded


def _one_keypoint(item: object, scale: float) -> Optional[Keypoint]:
    values: Sequence[object]
    if isinstance(item, dict):
        values = [item.get("x"), item.get("y"), item.get("v", item.get("visible", 2))]
    elif isinstance(item, (list, tuple)):
        values = list(item)
    else:
        return None
    if len(values) < 2:
        return None
    try:
        x = float(values[0]) / scale
        y = float(values[1]) / scale
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    v = _vis(values[2] if len(values) > 2 else 2)
    if x <= 0.0 and y <= 0.0:  # 多数模型用全零表示“这个点没标”
        v = VIS_UNLABELED
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return Keypoint(x=x, y=y, v=v, conf=1.0 if v != VIS_UNLABELED else 0.0)


def _vis(value: object) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return VIS_VISIBLE
    if number <= VIS_UNLABELED:
        return VIS_UNLABELED
    if number >= VIS_VISIBLE:
        return VIS_VISIBLE
    return VIS_OCCLUDED


def _coord_scale(raw_points: Sequence[object]) -> float:
    """模型可能回 0-1000 整数，也可能直接回 0-1 小数，按数值范围判断。"""
    numbers: List[float] = []
    for item in raw_points:
        if isinstance(item, dict):
            candidates = [item.get("x"), item.get("y")]
        elif isinstance(item, (list, tuple)):
            candidates = list(item[:2])
        else:
            continue
        for value in candidates:
            try:
                numbers.append(abs(float(value)))
            except (TypeError, ValueError):
                continue
    if numbers and max(numbers) <= 1.5:
        return 1.0
    return COORD_SCALE


def _box_from_entry(raw: object, scale: float) -> Optional[Tuple[float, float, float, float]]:
    """模型给的 xyxy 框转成归一化 cxcywh；不给就返回 None 交给关键点兜底。"""
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1 = float(raw[0]) / scale, float(raw[1]) / scale
        x2, y2 = float(raw[2]) / scale, float(raw[3]) / scale
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx = min(max((x1 + x2) / 2.0, 0.0), 1.0)
    cy = min(max((y1 + y2) / 2.0, 0.0), 1.0)
    return cx, cy, min(max(x2 - x1, 0.01), 1.0), min(max(y2 - y1, 0.01), 1.0)


class VLMBackend:
    """用多模态大模型逐图标骨骼点，接口对齐 :class:`backends.PoseEnsemble`。"""

    def __init__(self, settings: BackendSettings):
        self.settings = settings
        self.profile: Optional[dict] = None
        self.display_names: List[str] = list(COCO17_KEYPOINTS)
        self.last_latency_ms = 0

    def load(self, profile_id: str = "") -> "VLMBackend":
        """按配置 ID 或名称取多模态配置；返回 self，取不到就报清下一步怎么做。"""
        wanted = str(profile_id or getattr(self.settings, "vlm_profile", "") or "").strip()
        if not wanted:
            raise VLMError("没有指定多模态大模型配置")
        profile = model_manager.find_profile(wanted)
        if profile is None:
            options = "、".join(p.get("name", "") for p in model_manager.load_vlm()["profiles"]) or "（还没有）"
            raise VLMError(f"找不到多模态配置 {wanted}；可选：{options}（在界面「模型管理」里添加）")
        self.profile = profile
        model_manager.record_model_load(str(profile.get("id") or ""))
        return self

    @property
    def source_tag(self) -> str:
        """写进索引 ``labeled_by`` 的来源标记。"""
        return f"vlm:{(self.profile or {}).get('id', self.settings.vlm_profile)}"

    def describe(self) -> str:
        profile = self.profile or {}
        kind = "本地" if profile.get("kind") == "local" else "云端"
        return f"多模态大模型 {profile.get('name', '?')}（{kind} · {profile.get('model', '?')}）"

    def infer(self, image: np.ndarray) -> List[Person]:
        if self.profile is None:
            self.load()
        started = time.time()
        side_limit = int((self.profile or {}).get("image_limit") or MAX_IMAGE_SIDE)
        data_url = encode_image(image,
                                max_side=min(int(self.settings.max_side), MAX_IMAGE_SIDE, side_limit))
        text = chat(self.profile or {}, build_prompt(), data_url)
        persons = parse_people(text)
        for person in persons:
            person.score = DEFAULT_SCORE
        self.last_latency_ms = int((time.time() - started) * 1000)
        return persons
