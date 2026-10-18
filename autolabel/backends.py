"""姿态推理后端：ultralytics YOLO-pose + 翻转增强 + 多模型共识。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .yolo_pose import (
    VIS_OCCLUDED,
    VIS_UNLABELED,
    VIS_VISIBLE,
    Keypoint,
    Person,
    expand_bbox_with_keypoints,
    match_by_iou,
    merge_person_flips,
    mirror_person,
)


@dataclass
class BackendSettings:
    """推理与自动标注参数。"""

    model: str = "yolo11n-pose.pt"
    extra_models: Tuple[str, ...] = ()          # 额外模型，用于共识/分歧标记
    device: str = "auto"                        # auto | mps | cuda | cpu | 0
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 300
    half: bool = False
    flip_tta: bool = True
    # 可见性判定
    visibility_policy: str = "threshold"        # threshold | all_visible | binary
    visible_conf: float = 0.35                  # >= 该值 -> v=2
    occluded_conf: float = 0.15                 # >= 该值 -> v=1（threshold 策略下仍参与监督）
    # bbox 处理
    box_mode: str = "expand"                    # expand | model | keypoints
    bbox_margin: float = 0.06
    # 共识
    ensemble_iou: float = 0.3
    # 运行时
    max_side: int = 1600
    # 多模态大模型：非空（配置 ID 或名称）时改用视觉大模型标注，忽略上面的 YOLO 权重
    vlm_profile: str = ""

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def resolve_device(device: str = "auto") -> str:
    """自动选择推理设备。"""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def resolve_model_path(model: str, weights_dir: str = "models") -> str:
    """把 `yolo11n-pose.pt` 解析到本地 weights 目录，缺失时交给 ultralytics 下载。

    ultralytics 会按请求路径的父目录存放下载文件，因此这里显式加上目录前缀，
    避免权重散落在工作区根目录。
    """
    expanded = os.path.expanduser(model)
    if os.path.sep in expanded or os.path.exists(expanded):
        return expanded
    os.makedirs(weights_dir, exist_ok=True)
    return os.path.join(weights_dir, expanded)


# ultralytics 权重发布地址与默认镜像：直连 GitHub 常掉到 10 KB/s 甚至连不上，
# 镜像前缀可用环境变量 AUTOLABEL_WEIGHTS_MIRROR 覆盖（设成空串即回到直连）。
ASSETS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
DEFAULT_MIRROR = os.environ.get("AUTOLABEL_WEIGHTS_MIRROR", "https://gh-proxy.com/")


def weights_url(model: str, mirror: Optional[str] = None) -> str:
    """算出某个权重的下载地址（默认带镜像前缀），下载与展示记录共用同一套算法。"""
    prefix = DEFAULT_MIRROR if mirror is None else mirror
    return (prefix.rstrip("/") + "/" if prefix.strip() else "") + f"{ASSETS_BASE}/{model}"


def weights_ready(path: str) -> bool:
    """权重是否可用：只认 zip 尾部目录，半截下载按不可用处理。"""
    import zipfile

    return bool(path) and os.path.isfile(path) and zipfile.is_zipfile(path)


def ensure_weights(model: str, weights_dir: str = "models", mirror: Optional[str] = None,
                   attempts: int = 3, timeout: int = 30, verbose: bool = True,
                   force: bool = False, progress: Optional[Callable[[int, int], None]] = None,
                   on_error: Optional[Callable[[Exception], None]] = None) -> str:
    """本地权重缺失或只有一半时，先走镜像把它拉全，再交给 ultralytics 加载。

    镜像和 GitHub 的同一个 release asset 字节一致，所以已有的半截文件可以用
    Range 续传，不必从头再来。任何一步失败都不抛错，退回原路径让 ultralytics
    自己的下载器兜底，行为不会比现在更差。

    ``progress(done, total)`` 给界面回报下载进度（字节；total 未知时为 0），
    ``on_error(exc)`` 回报每次失败原因，两者都只用于提示，不改变兜底行为。
    """
    path = resolve_model_path(model, weights_dir)
    if os.path.sep in model or not model.endswith(".pt"):
        return path
    if weights_ready(path) and not force:
        return path
    if force and os.path.exists(path):
        os.replace(path, path + ".bad")   # 先挪走，下载成功后再删，失败时还能人工比对

    prefix = DEFAULT_MIRROR if mirror is None else mirror
    url = weights_url(model, prefix)
    part = path + ".part"
    for try_no in range(1, max(1, attempts) + 1):
        try:
            got = _download_to(url, part, timeout=timeout, verbose=verbose, progress=progress)
            if not weights_ready(part):
                raise OSError("下载结果不是完整的 zip 包")
            os.replace(part, path)
            bad = path + ".bad"
            if os.path.exists(bad):
                os.remove(bad)
            if verbose:
                print(f"[ok] 权重 {model} 就绪（{got / 1048576:.1f} MB）")
            return path
        except Exception as exc:  # noqa: BLE001 镜像不可用要能退回直连
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception:  # noqa: BLE001 回调报错不该影响下载
                    pass
            if verbose:
                print(f"[warn] 第 {try_no} 次经镜像下载 {model} 失败："
                      f"{type(exc).__name__}: {exc}（已下载部分保留，可续传）")
    return path


def _download_to(url: str, dst: str, timeout: int = 30, verbose: bool = True,
                 progress: Optional[Callable[[int, int], None]] = None) -> int:
    """下载单个文件，支持断点续传，返回落盘后的总字节数。"""
    import urllib.error
    import urllib.request

    start = os.path.getsize(dst) if os.path.exists(dst) else 0
    request = urllib.request.Request(url, headers={"User-Agent": "autolabel-weights"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        resume = status == 206 and start > 0
        total = start + int(resp.headers.get("Content-Length") or 0)
        done = start if resume else 0
        with open(dst, "ab" if resume else "wb") as fh:
            if not resume:
                fh.truncate(0)
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress is not None:
                    try:
                        progress(done, total)
                    except Exception:  # noqa: BLE001 进度回调报错不该拖垮下载
                        pass
                if verbose and total:
                    pct = done * 100 // total
                    if pct != (done - len(chunk)) * 100 // total:
                        line = (f"[info] 下载权重 {os.path.basename(dst)} {pct}% "
                                f"({done / 1048576:.1f}/{total / 1048576:.1f} MB)")
                        # 交互式终端里原地刷新，写日志时每 1% 单独一行太吵，改成每 25% 一行
                        if sys.stdout.isatty():
                            print("\r" + line, end="", flush=True)
                        elif pct // 25 != (done - len(chunk)) * 100 // total // 25:
                            print(line)
    if total and done != total:
        raise OSError(f"下载不完整 {done}/{total} 字节")
    if verbose and total and sys.stdout.isatty():
        print()
    return done


# 复核界面下拉框里的候选权重，顺序与 review.html 保持一致
UI_MODEL_CHOICES: Tuple[str, ...] = ("yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt")


def model_status(model: str, weights_dir: str = "models") -> Dict[str, object]:
    """查本地权重是否可用；下载中断留下的半截文件按“未就绪”处理。

    ultralytics 权重是 zip 包，`is_zipfile` 只读尾部目录，能便宜地识破被截断的下载，
    比让用户点完开始才撞到 PytorchStreamReader 报错友好得多。
    """
    import zipfile

    path = resolve_model_path(model, weights_dir)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return {"name": model, "path": path, "size_mb": round(size / 1048576, 1),
            "ready": bool(size) and zipfile.is_zipfile(path)}


def model_inventory(models: Sequence[str] = UI_MODEL_CHOICES, weights_dir: str = "models") -> List[dict]:
    """批量返回候选权重的就绪状态，供界面标注。"""
    return [model_status(name, weights_dir) for name in models]


def weights_hint_retryable(path: str, weights_dir: str = "models") -> bool:
    """是否允许自动重下：只认权重目录里的裸文件名，用户自己给的路径不动。"""
    if not path or not os.path.isfile(path):
        return False
    return os.path.dirname(os.path.abspath(path)) == os.path.abspath(weights_dir)


def weights_hint(path: str, exc: Exception) -> str:
    """权重加载失败时的修复提示：下载被中断（服务重启/断网）是最常见原因。"""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return (f"模型 {path} 加载失败（{type(exc).__name__}: {exc}）；"
            f"当前文件 {size / 1048576:.1f} MB，多半是下载中断导致不完整，"
            f"删掉重新标注会自动重下：rm {path}")


def report_model_use(model: str, num_keypoints: int, names: Sequence[str] = (),
                     device: str = "") -> None:
    """把「真的用过这个权重 + 读到的关键点数」回报给模型管理，异常一律吞掉不影响推理。"""
    try:
        from .models import record_model_load
        from .yolo_pose import COCO17_KEYPOINTS

        keypoint_set = ("COCO-17" if tuple(names or ()) == tuple(COCO17_KEYPOINTS)
                        else f"{num_keypoints}-自定义")
        record_model_load(model, {"num_keypoints": int(num_keypoints),
                                  "keypoint_set": keypoint_set, "device": str(device)})
    except Exception:  # noqa: BLE001 记账失败不该拖垮一次标注
        pass


class PoseBackend:
    """单模型 YOLO-pose 后端。"""

    def __init__(self, settings: BackendSettings, model_name: Optional[str] = None):
        self.settings = settings
        self.model_name = model_name or settings.model
        self._model = None
        self._device = resolve_device(settings.device)
        self.keypoint_names: Tuple[str, ...] = ()
        self.num_keypoints: int = 17

    # --- 生命周期 ---
    def load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        path = ensure_weights(self.model_name)
        print(f"[info] 加载模型 {path} -> device={self._device}")
        try:
            self._model = YOLO(path, task="pose")
        except Exception as exc:  # noqa: BLE001 先试着重下，再给修复建议
            if not self._retry_after_redownload(path, exc, verbose=True):
                raise RuntimeError(weights_hint(path, exc)) from exc
            self._model = YOLO(path, task="pose")
        meta = getattr(self._model, "pose_metadata", None) or {}
        names = meta.get("keypoint_names")
        if names:
            self.keypoint_names = tuple(names)
            self.num_keypoints = len(self.keypoint_names)
        else:
            from .yolo_pose import COCO17_KEYPOINTS

            self.keypoint_names = COCO17_KEYPOINTS
            flip = meta.get("flip_idx")
            self.num_keypoints = len(flip) if flip else len(COCO17_KEYPOINTS)
        report_model_use(self.model_name, self.num_keypoints, self.keypoint_names, self._device)
        return self._model

    def _retry_after_redownload(self, path: str, exc: Exception, verbose: bool) -> bool:
        """权重损坏时经镜像重下一遍，成功就返回 True 让调用方再加载一次。"""
        name = os.path.basename(path)
        if not weights_hint_retryable(path):
            return False
        if verbose:
            print(f"[warn] {name} 加载失败（{type(exc).__name__}），尝试经镜像重新下载…")
        return ensure_weights(name, weights_dir=os.path.dirname(path) or "models",
                              verbose=verbose, force=True, attempts=2) == path

    @property
    def display_names(self) -> List[str]:
        return list(self.keypoint_names)

    # --- 推理 ---
    def _predict(self, image: np.ndarray, flipped: bool = False) -> List[Person]:
        import cv2

        model = self.load()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = model.predict(
            source=bgr,
            device=self._device,
            imgsz=self.settings.imgsz,
            conf=self.settings.conf,
            iou=self.settings.iou,
            max_det=self.settings.max_det,
            half=self.settings.half,
            verbose=False,
        )
        persons: List[Person] = []
        for res in results:
            h, w = res.orig_shape[:2]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            kpt_data = None
            if res.keypoints is not None and getattr(res.keypoints, "data", None) is not None:
                kpt_data = res.keypoints.data.cpu().numpy()  # (N, K, 3) 原图像素坐标
            for i in range(len(xyxy)):
                kps: List[Keypoint] = []
                if kpt_data is not None and i < len(kpt_data):
                    for kx, ky, kc in kpt_data[i]:
                        raw = float(kc)
                        # 兼容早期版本恒等于 2.0 的占位置信度
                        conf = 1.0 if raw >= 1.99 and raw <= 2.01 else min(max(raw, 0.0), 1.0)
                        kps.append(Keypoint(x=clamp01(kx / max(w, 1)), y=clamp01(ky / max(h, 1)),
                                            v=VIS_VISIBLE, conf=conf))
                person = Person.from_xyxy(xyxy[i], w, h, kps, score=float(confs[i]), cls_id=int(clss[i]))
                if flipped:
                    person = mirror_person(person, w, h)
                persons.append(person)
        return persons

    def infer(self, image: np.ndarray) -> List[Person]:
        """输入 RGB ndarray，返回归一化 Person 列表（已做翻转融合与可见性判定）。"""
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 先定可见性、再定框：_predict 里的 v 还是占位的「全可见」，直接拿它扩框会把
        # conf 低到该判 v=0 的杂点也圈进包围盒——框被撑大，还与落盘标签里的 v 自相矛盾。
        persons = [assign_visibility(p, self.settings) for p in self._predict(bgr)]
        if self.settings.flip_tta:
            mirrored = [assign_visibility(p, self.settings)
                        for p in self._predict(cv2.flip(bgr, 1), flipped=True)]
            persons = fuse_flips(persons, mirrored, image.shape[1], image.shape[0], self.settings)
        return [finalize_box(p, image.shape[1], image.shape[0], self.settings) for p in persons]


class PoseEnsemble:
    """多模型共识：主模型 + 校验模型，按 IoU 匹配并标记分歧。"""

    def __init__(self, settings: BackendSettings):
        self.settings = settings
        names = [settings.model] + [m for m in settings.extra_models if m and m != settings.model]
        self.backends = [PoseBackend(settings, name) for name in names]

    def load(self):
        for b in self.backends:
            b.load()
        return self

    @property
    def primary(self) -> PoseBackend:
        return self.backends[0]

    @property
    def display_names(self) -> List[str]:
        return self.primary.display_names

    def infer(self, image: np.ndarray) -> List[Person]:
        grouped: List[List[Person]] = [b.infer(image) for b in self.backends]
        primary = grouped[0]
        if len(grouped) == 1:
            return primary
        for others in grouped[1:]:
            pairs = match_by_iou([p.xyxy(image.shape[1], image.shape[0]) for p in primary],
                                 [p.xyxy(image.shape[1], image.shape[0]) for p in others],
                                 iou_thresh=self.settings.ensemble_iou)
            matched_primary = {i for i, _, _ in pairs}
            matched_secondary = {j for _, j, _ in pairs}
            for i, j, _ in pairs:
                merged = merge_person_flips(primary[i], others[j])
                merged.score = max(primary[i].score, others[j].score)
                merged.flags = sorted(set(primary[i].flags) | {"multi_model_confirmed"})
                primary[i] = merged
            # 两路结论不一致的两个方向都要标：主模型有、辅助模型没有 -> 分歧待确认；
            # 辅助模型有、主模型漏了 -> 记为仅辅助模型检出。
            for i in range(len(primary)):
                if i not in matched_primary:
                    primary[i].flags = sorted(set(primary[i].flags) | {"model_disagreement"})
            missed = [j for j in range(len(others)) if j not in matched_secondary]
            for extra in (others[j] for j in missed):
                extra.flags = ["only_secondary_model"]
            primary.extend(others[j] for j in missed)
        return primary


def build_backend(settings: BackendSettings):
    """按参数挑标注后端：默认 YOLO-pose，指定 ``vlm_profile`` 时改用视觉大模型。"""
    if str(getattr(settings, "vlm_profile", "") or "").strip():
        from .vlm import VLMBackend  # 延迟导入，避免多模态模块拖慢 YOLO 冷启动

        return VLMBackend(settings)
    return PoseEnsemble(settings)


# --------------------------------------------------------------------------- #
# 后处理
# --------------------------------------------------------------------------- #

def fuse_flips(
    persons: List[Person],
    mirrored: List[Person],
    img_w: int,
    img_h: int,
    settings: BackendSettings,
) -> List[Person]:
    """把正向与水平翻转两路预测融合（关键点按置信度加权平均）。"""
    if not mirrored:
        return persons
    if not persons:
        return mirrored
    a_boxes = [p.xyxy(img_w, img_h) for p in persons]
    b_boxes = [p.xyxy(img_w, img_h) for p in mirrored]
    pairs = match_by_iou(a_boxes, b_boxes, iou_thresh=max(settings.ensemble_iou, 0.25))
    mapping = {i: j for i, j, _ in pairs}
    used_b = set(mapping.values())
    out: List[Person] = []
    for i, person in enumerate(persons):
        j = mapping.get(i)
        out.append(merge_person_flips(person, mirrored[j]) if j is not None else person)
    for j, person in enumerate(mirrored):
        if j not in used_b:
            out.append(person)
    return out


def assign_visibility(person: Person, settings: BackendSettings) -> Person:
    """按策略给关键点赋 v 值。

    ``threshold``：conf>=visible_conf -> 2，conf>=occluded_conf -> 1，其余 -> 0。
    注意 ultralytics 把 ``v!=0`` 都当作监督目标，因此 conf 落在 occluded 区间的点
    仍会被学习；若要更保守，把 ``occluded_conf`` 提到 ``visible_conf`` 即可退化为二值。
    """
    for k in person.keypoints:
        if settings.visibility_policy == "all_visible":
            k.v = VIS_VISIBLE
        elif settings.visibility_policy == "binary":
            k.v = VIS_VISIBLE if k.conf >= settings.visible_conf else VIS_UNLABELED
        else:
            if k.conf >= settings.visible_conf:
                k.v = VIS_VISIBLE
            elif k.conf >= settings.occluded_conf:
                k.v = VIS_OCCLUDED
            else:
                k.v = VIS_UNLABELED
    return person


def finalize_box(person: Person, img_w: int, img_h: int, settings: BackendSettings) -> Person:
    """按 box_mode 决定最终 bbox。"""
    if settings.box_mode == "model":
        return person
    if settings.box_mode == "keypoints":
        from .yolo_pose import keypoints_bbox

        bbox = keypoints_bbox(person.keypoints, margin_ratio=settings.bbox_margin)
        if bbox is not None:
            person.cx, person.cy, person.w, person.h = bbox
        return person
    return expand_bbox_with_keypoints(person, img_w, img_h, margin_ratio=settings.bbox_margin)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
