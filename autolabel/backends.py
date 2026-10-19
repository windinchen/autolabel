"""姿态推理后端：ultralytics YOLO-pose + 翻转增强 + 多模型共识。"""

from __future__ import annotations

import os
import re
import sys
import time
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
from .platform_utils import remove_hint


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


def _cuda_card_index(requested: str) -> Optional[int]:
    """``"0"`` / ``"1"`` / ``"cuda:1"`` → 卡号；其它写法 → ``None``。"""
    text = requested.split(":", 1)[1] if requested.startswith("cuda:") else requested
    return int(text) if text.isdigit() else None


def resolve_device(device: str = "auto") -> str:
    """把参数里的设备落成这台机器实际能用的那一档。

    早先"只要不是 auto 就原样返回"，于是在 Intel Mac 上写 ``--device mps``
    会一路走到 ultralytics 里抛一句英文栈，新手根本不知道该改哪儿。现在先探测：
    指定了编号就核对有没有这张卡，其余情况交给 :func:`autolabel.settings.resolve`
    按系统设置降级，并且每次都用中文说明退到了哪一档——标注不会因为换设备失败，
    只是快慢不同，所以降级跑下去比报错有用。
    """
    from . import settings as system

    requested = str(device or "auto").strip().lower()
    index = _cuda_card_index(requested)
    if index is not None:
        cards = system.detect().get("cuda", {}).get("names") or []
        if index < len(cards):
            return str(index)
        print(f"[warn] 指定了第 {index} 张 NVIDIA 显卡，但这台机器只识别到 {len(cards)} 张，改用系统设置里的设备")
        requested = ""
    resolved, note = system.resolve(requested)
    if note:
        print(f"[warn] {note}")
    return "0" if resolved == "cuda" else resolved


def resolve_half(half: bool, device: str) -> bool:
    """半精度只在 NVIDIA 显卡上真的更快：CPU 上 ultralytics 直接不支持。

    这里先挡一层，是为了把话说明白（"半精度在 CPU 上不支持"），
    而不是让用户在 ultralytics 的英文警告里自己猜为什么没提速。
    """
    if not half:
        return False
    text = str(device).strip().lower()
    if text == "cuda" or text.isdigit():
        return True
    print(f"[warn] 半精度在 {device} 上没有加速（只有 NVIDIA 显卡支持），本次按普通精度跑")
    return False


#: 半截下载的临时名与"谁正在下这个权重"的锁文件名
PART_SUFFIX = ".part"
LOCK_SUFFIX = ".lock"


def default_weights_dir() -> str:
    """推理这头该去哪个目录找权重：和「模型管理」面板共用一个口径。

    ``AUTOLABEL_WEIGHTS_DIR`` 是对外承诺过的环境变量（见 08-模型管理.md），
    以前只有 ``models.py`` 认它、推理侧把 ``models/`` 写死。改了变量的用户会看到
    界面里权重"已就绪"，真点开始标注却又往 ``./models`` 重新下一份，白等还多占磁盘。
    """
    from .models import weights_dir

    return weights_dir()


def claim_download(path: str, *, wait: bool = True, deadline: float = 0.0) -> Tuple[bool, object]:
    """尽量独占某个权重的一次下载。返回 ``(是否独占, 锁句柄)``。

    一台机器同时开两个实例（一卡一实例是这本项目的正常用法），或者界面点着「下载」
    的同时标注任务正在补下同一个权重，两边会写同一份 ``x.pt.part``：后开的把对方
    正在追加的文件截断，续传的起点成了对方刚写下的偏移量，换名失败的一方还会报
    "已下载部分保留"而那份文件早就不在。这里用文件锁把第二个下载者挡在门外，
    让它等第一个下完——同一份权重下两遍本来就是白等一倍时间。

    没有 ``flock`` 的平台（Windows）按"没别人"处理，行为与加锁之前一致。
    """
    handle = None
    try:
        import fcntl
    except ImportError:
        return True, None
    try:
        handle = open(path + LOCK_SUFFIX, "a+b")
    except OSError:
        return True, None          # 目录不让建锁文件：照常下载，别因为锁挡了正事
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True, handle
    except OSError:
        handle.close()
        if not wait:
            return False, None
        while time.time() < deadline:
            if weights_ready(path):
                return False, None
            time.sleep(0.5)
            if time.time() >= deadline:
                break
            # 等着的一方也得回头抢锁：先动手的那个可能直接被杀了（服务重启、任务
            # 被取消），锁早就空着却没人接手，干等到超时只能报一句"权重不完整"
            retry = None
            try:
                retry = open(path + LOCK_SUFFIX, "a+b")
                fcntl.flock(retry.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if retry is not None:
                    retry.close()
                continue
            return True, retry
        return False, None


def partial_paths(weights_dir: str, model: str) -> List[str]:
    """某个权重在目录里剩下的半截下载文件。

    界面要能看见它们（"有 12 MB 没下完"）、删除时要连着它们一起删干净，
    否则会留下一个看不见又占着磁盘的文件，用户只能整个目录清空。
    """
    base = os.path.basename(str(model or ""))
    if not base or not os.path.isdir(weights_dir):
        return []
    found = []
    try:
        entries = os.listdir(weights_dir)
    except OSError:
        return []
    for entry in entries:
        if not entry.endswith(PART_SUFFIX):
            continue
        head = entry[: -len(PART_SUFFIX)]
        if head == base or head.startswith(base + "."):
            found.append(os.path.join(weights_dir, entry))
    return sorted(found)


def resolve_model_path(model: str, weights_dir: Optional[str] = None) -> str:
    """把 `yolo11n-pose.pt` 解析到本地 weights 目录，缺失时交给 ultralytics 下载。

    ultralytics 会按请求路径的父目录存放下载文件，因此这里显式加上目录前缀，
    避免权重散落在工作区根目录。
    """
    weights_dir = default_weights_dir() if weights_dir is None else weights_dir
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


def ensure_weights(model: str, weights_dir: Optional[str] = None,
                   mirror: Optional[str] = None,
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
    # 候选地址：先镜像、再直连。以前三次重试打的都是同一个镜像地址，镜像一旦限流
    # 或挂掉就白等三轮超时——注释和提示都写着"退回直连"，得真的退回去
    direct = weights_url(model, "")
    candidates: List[str] = []
    for candidate in (weights_url(model, prefix), direct):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    # 临时名保持固定（``x.pt.part``）：服务重启、断网重连之后还能接着这半截续传，
    # 几十 MB 的权重不必从头再来——这是这个文件存在的意义，不能改成一次性随机名。
    # 并发的问题交给文件锁解决：抢不到锁的一方在这里等对方下完，而不是插一脚进来。
    part = path + PART_SUFFIX
    deadline = time.time() + max(60.0, float(timeout) * max(1, attempts) * 2)
    mine, handle = claim_download(path, deadline=deadline)
    if not mine:
        if verbose:
            print(f"[info] {model} 已经有别的进程/任务在下载，等它下完")
        return path
    try:
        for try_no in range(1, max(1, attempts) + 1):
            url = candidates[min(try_no, len(candidates)) - 1]
            via = "直连" if url == direct else "镜像"
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
                    print(f"[warn] 第 {try_no} 次走{via}下载 {model} 失败："
                          f"{type(exc).__name__}: {exc}（已下载部分保留，可续传）")
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
    if force and not weights_ready(path) and os.path.exists(path + ".bad"):
        # 一次都没下成（多半是这台机器没网）：把原来那份放回去。它虽然加载失败，
        # 但比让用户手上连一份权重都不剩要好，至少人工还能拿它去比对
        try:
            os.replace(path + ".bad", path)
        except OSError:
            pass
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
        length = int(resp.headers.get("Content-Length") or 0)
        # 只有服务器真的按 Range 回了一段（206），已下过的字节才算在总量里。
        # 镜像/代理常常把 Range 头直接吃掉、回 200 + 整包：那份总量加上 start
        # 就凭空多出一截，永远等不满，最后判"下载不完整"，重试几次都是这个结果。
        total = start + length if resume else length
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


def model_status(model: str, weights_dir: Optional[str] = None) -> Dict[str, object]:
    """查本地权重是否可用；下载中断留下的半截文件按“未就绪”处理。

    ultralytics 权重是 zip 包，`is_zipfile` 只读尾部目录，能便宜地识破被截断的下载，
    比让用户点完开始才撞到 PytorchStreamReader 报错友好得多。
    """
    import zipfile

    path = resolve_model_path(model, weights_dir)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return {"name": model, "path": path, "size_mb": round(size / 1048576, 1),
            "ready": bool(size) and zipfile.is_zipfile(path)}


def model_inventory(models: Sequence[str] = UI_MODEL_CHOICES,
                    weights_dir: Optional[str] = None) -> List[dict]:
    """批量返回候选权重的就绪状态，供界面标注。"""
    return [model_status(name, weights_dir) for name in models]


def weights_hint_retryable(path: str, weights_dir: Optional[str] = None) -> bool:
    """是否允许自动重下：只认权重目录里的裸文件名，用户自己给的路径不动。"""
    if not path or not os.path.isfile(path):
        return False
    return os.path.dirname(os.path.abspath(path)) == os.path.abspath(
        default_weights_dir() if weights_dir is None else weights_dir)


def weights_hint(path: str, exc: Exception) -> str:
    """权重加载失败时的修复提示：下载被中断（服务重启/断网）是最常见原因。"""
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return (f"模型 {path} 加载失败（{type(exc).__name__}: {exc}）；"
            f"当前文件 {size / 1048576:.1f} MB，多半是下载中断导致不完整，"
            f"删掉重新标注会自动重下：{remove_hint(path)}")


def flip_indices(raw: object, count: int) -> List[int]:
    """模型自带的左右关键点配对表；缺了或对不上才退回 COCO 的前 N 个。

    只有「长度等于关键点数」且「正好是 0..N-1 的一个排列」才算数：自训练权重
    偶尔会写出一截残缺的 ``flip_idx``，拿它去换下标会直接 IndexError。
    """
    wanted = max(int(count), 0)
    items: List[int] = []
    if isinstance(raw, (list, tuple)):
        try:
            items = [int(value) for value in raw]
        except (TypeError, ValueError):
            items = []
    if len(items) == wanted and sorted(items) == list(range(wanted)):
        return items
    from .yolo_pose import COCO17_FLIP_IDX

    return list(COCO17_FLIP_IDX[:wanted]) if wanted <= len(COCO17_FLIP_IDX) \
        else list(range(wanted))


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
        self._half = resolve_half(settings.half, self._device)
        self.keypoint_names: Tuple[str, ...] = ()
        self.num_keypoints: int = 17
        self.flip_idx: List[int] = []

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
        # 镜像融合要按模型自己的左右配对换下标：自训练的自定义关键点权重顺序未必
        # 是 COCO，一律套 COCO 表会把左肩和右肩平均到一起——错得不报错
        self.flip_idx = flip_indices(meta.get("flip_idx"), self.num_keypoints)
        report_model_use(self.model_name, self.num_keypoints, self.keypoint_names, self._device)
        return self._model

    def _retry_after_redownload(self, path: str, exc: Exception, verbose: bool) -> bool:
        """权重损坏时经镜像重下一遍，成功就返回 True 让调用方再加载一次。"""
        name = os.path.basename(path)
        if not weights_hint_retryable(path):
            return False
        if verbose:
            print(f"[warn] {name} 加载失败（{type(exc).__name__}），尝试经镜像重新下载…")
        # ensure_weights 无论成败都返回同一个 path（下坏了也要把兜底交给
        # ultralytics），拿返回值比等于永远算成功：重下没成还会把原文件挪成
        # .bad，第二次加载就变成一句 FileNotFoundError，中文修复提示反而看不见
        ensure_weights(name, weights_dir=os.path.dirname(path) or "models",
                              verbose=verbose, force=True, attempts=2)
        return weights_ready(path)

    @property
    def display_names(self) -> List[str]:
        return list(self.keypoint_names)

    # --- 推理 ---
    def _predict(self, bgr: np.ndarray, flipped: bool = False) -> List[Person]:
        """跑一次推理。入参必须已经是 BGR：ultralytics 收到 ndarray 时不做任何色彩
        转换，直接当 BGR 用（它自己的加载器写明假定 BGR）。以前这里和
        :meth:`infer` 各转了一次，等于把原始 RGB 交给模型，R/B 长期互换，
        关键点置信度和坐标系统性变差，还一声不吭。
        """
        model = self.load()
        kwargs = dict(source=bgr, device=self._device, imgsz=self.settings.imgsz,
                      conf=self.settings.conf, iou=self.settings.iou,
                      max_det=self.settings.max_det, verbose=False)
        if self._half:          # 8.4 起传 half 会喊弃用，关着就别传
            kwargs["half"] = True
        results = model.predict(**kwargs)
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
                    person = mirror_person(person, w, h, flip_idx=self.flip_idx or None)
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


#: 目标检测库能吃的 YOLO 任务。姿态模型不算：它输出的是关键点，塞进检测库会
#: 把 person 之外的目标全丢掉，得让界面先说清楚而不是标完才发现少了一堆框。
DETECT_TASKS = ("detect", "segment", "obb")
TASK_NAMES = {"detect": "目标检测", "segment": "实例分割", "obb": "旋转框检测",
              "pose": "人体姿态", "classify": "图像分类", "depth": "深度估计"}

#: 开放词表权重的文件名标记。这份判断必须和 ultralytics 自己的路由一致
#: （``YOLO(路径)`` 见到 ``-world`` 走 YOLOWorld、见到 ``yoloe`` 走 YOLOE，
#: 见 ``ultralytics/models/yolo/model.py``），否则会出现「模型按开放词表加载、
#: 界面却按权重自带类别表显示」——用户以为词表生效了，其实一直在跑 COCO-80。
OPEN_VOCAB_MARKERS = (("-world", "YOLO-World"), ("yoloe", "YOLOE"))
#: 一次最多给多少个词。词表要逐个过文本编码器，几百个词纯属拖时间
MAX_VOCAB_WORDS = 80
#: 单个词最长多少字符。CLIP 看的是短句，"a person riding a mountain bike on a road"
#: 这种长描述会明显掉点，界面上限压短一点比事后解释"为什么漏标"划算
MAX_VOCAB_WORD_CHARS = 48
#: 文本编码器权重（CLIP 约 338MB）默认落在项目 ``weights/``，第一次用才下
TEXT_ENCODER_WEIGHTS_DIR = "weights"


def open_vocab_kind(name: str) -> str:
    """按文件名判断是不是开放词表权重，返回 ``"YOLO-World"`` / ``"YOLOE"`` / 空串。

    只看文件名是有意的：判断发生在加载之前，界面要在"选权重"这一步就说清
    类别从哪来，而不是等加载完才发现类别框和权重对不上。
    """
    stem = os.path.splitext(os.path.basename(str(name or "")))[0].lower()
    for marker, kind in OPEN_VOCAB_MARKERS:
        if marker in stem:
            return kind
    return ""


def clean_vocab(words: Sequence[str]) -> List[str]:
    """整理用户填的词表：去空去重、限长限量，非 ASCII 直接报人话。

    一律要求英文是有实测依据的：CLIP 文本编码器只认英文，
    ``["人","狗","猫"]`` 这种中文词表跑完是**零检出**——不报错、不警告，
    只是什么框都没有。宁可在这里挡住，也别让用户对着空标注库猜。
    """
    cleaned: List[str] = []
    seen = set()
    for raw in (words or []):
        # 词表是从输入框/命令行里来的一段自由文本，分隔符不该只认一种：
        # 中文用户很自然会打「person，dog」，按 ASCII 逗号切的那套在这就散了
        for piece in re.split(r"[,;，；\n\r]+", str(raw or "")):
            cleaned.extend(_vocab_one(piece, seen))
    if not cleaned:
        raise ValueError("开放词表权重要先填「这次要找什么」，比如 person、dog、traffic cone")
    if len(cleaned) > MAX_VOCAB_WORDS:
        raise ValueError(f"一次最多填 {MAX_VOCAB_WORDS} 个词，当前 {len(cleaned)} 个"
                         "（词表要逐个过文本编码器，太多只会变慢）")
    return cleaned


def _vocab_one(piece: str, seen: set) -> List[str]:
    """整理单个词：去空白、限长、挡中文、去重。合法返回单元素列表，否则空列表。"""
    word = " ".join(str(piece or "").split())
    if not word:
        return []
    if len(word) > MAX_VOCAB_WORD_CHARS:
        raise ValueError(f"词表里的词最长 {MAX_VOCAB_WORD_CHARS} 个字符，当前 {len(word)} 个："
                         f"{word}（模型看的是短词，长句子会明显漏标）")
    if not word.isascii():
        raise ValueError(f"开放词表只能填英文，中文不生效：{word}"
                         "（文本编码器是英文词表，填中文不会报错，但一个框都找不到）")
    if word.lower() in seen:
        return []
    seen.add(word.lower())
    return [word]


def text_encoder_hint(exc: Exception) -> str:
    """把「文本编码器没装」这类的英文报错翻成能照着做的中文指引。"""
    detail = str(exc)
    if "clip" not in detail.lower():
        return f"设置开放词表失败：{detail}"
    return ("开放词表权重要先算文本嵌入，这一步依赖 CLIP 文本编码器，本机还没装好。"
            "它是 ultralytics 的 git 依赖（不在普通 pip 源里），"
            "首次使用要联网装包并下载约 338MB 的编码器权重，之后一直复用。"
            f"\n原始报错：{type(exc).__name__}: {detail[:200]}")


def text_encoder_status() -> dict:
    """开放词表的可用性概况（系统设置页用；不加载模型，几百毫秒）。"""
    installed = True
    reason = ""
    try:
        import clip  # noqa: F401
    except Exception as exc:  # noqa: BLE001 缺包/装坏了都算不可用
        installed = False
        reason = f"{type(exc).__name__}: {str(exc)[:120]}"
    weights = os.path.join(os.getcwd(), TEXT_ENCODER_WEIGHTS_DIR)
    cached = 0
    if os.path.isdir(weights):
        for base, _dirs, files in os.walk(weights):
            for name in files:
                try:
                    cached += os.path.getsize(os.path.join(base, name))
                except OSError:
                    continue
    if installed:
        note = ""
    elif cached:
        note = ("文本编码器的权重已经在本地，只差装一个 Python 包；"
                "第一次用开放词表权重时会自动装好（要联网，约 10 秒），不用手动操作")
    else:
        note = ("第一次用开放词表权重时会自动装依赖并下载约 338MB 的文本编码器权重"
                "（要联网，之后离线可用），不用手动操作")
    return {"installed": installed, "reason": reason, "weights_dir": weights,
            "cached_bytes": int(cached), "note": note}


@dataclass
class Detection:
    """一个检出目标：类别 + 置信度 + 归一化中心框（和 YOLO detection 一个口径）。"""

    class_id: int
    class_name: str
    score: float
    cx: float
    cy: float
    w: float
    h: float
    has_mask: bool = False

    def as_row(self) -> List[float]:
        return [float(self.class_id), self.cx, self.cy, self.w, self.h]


class DetectionBackend:
    """目标检测 / 实例分割后端：出框，不出关键点。

    与 ``PoseBackend`` 分开是有意的：检测库关心的是「这是什么、在哪」，
    类别表由模型给；姿态库的类别永远是 person，多出来的东西全是关键点。
    混在一个后端里会让两边的参数互相牵制（``imgsz`` 之外还有 ``box_mode``、
    可见性策略这些只对姿态有意义）。
    """

    def __init__(self, settings: BackendSettings, model_name: Optional[str] = None):
        self.settings = settings
        self.model_name = model_name or settings.model
        self._model = None
        self._device = resolve_device(settings.device)
        self._half = resolve_half(settings.half, self._device)
        self.task = ""
        self.names: Dict[int, str] = {}
        #: 开放词表权重是哪一类（``YOLO-World`` / ``YOLOE`` / 空串表示普通权重）
        self.open_vocab = ""
        #: 这次设给模型的词表；同样的词不重复算文本嵌入
        self.vocab: Tuple[str, ...] = ()

    # --- 生命周期 ---
    def load(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        path = ensure_weights(self.model_name)
        print(f"[info] 加载检测模型 {path} -> device={self._device}")
        try:
            self._model = YOLO(path)
        except Exception as exc:  # noqa: BLE001 先试着重下，再给修复建议
            if not self._retry_after_redownload(path, exc, verbose=True):
                raise RuntimeError(weights_hint(path, exc)) from exc
            self._model = YOLO(path)
        task = str(getattr(self._model, "task", "") or "")
        if task and task not in DETECT_TASKS:
            raise ValueError(
                f"{os.path.basename(path)} 是「{TASK_NAMES.get(task, task)}」模型，"
                f"目标检测库要的是 {'、'.join(TASK_NAMES[t] for t in DETECT_TASKS)}；"
                "人体骨骼请回姿态标注里用带 -pose 的权重")
        self.task = task or "detect"
        self.open_vocab = open_vocab_kind(path)
        names = getattr(self._model, "names", None) or {}
        if isinstance(names, dict):
            self.names = {int(k): str(v) for k, v in names.items()}
        if self.open_vocab:
            # 开放词表权重自带的那张表只是出厂默认（YOLO-World 是 COCO-80），
            # 不代表"这次能找什么"。留着它，界面就会拿 COCO 类别去勾，
            # 用户以为在用开放词表，其实一直在跑默认表。
            self.names = {i: word for i, word in enumerate(self.vocab)}
        report_model_use(self.model_name, 0, tuple(self.names.values()), self._device)
        return self._model

    def _retry_after_redownload(self, path: str, exc: Exception, verbose: bool) -> bool:
        name = os.path.basename(path)
        if not weights_hint_retryable(path):
            return False
        if verbose:
            print(f"[warn] {name} 加载失败（{type(exc).__name__}），尝试经镜像重新下载…")
        # ensure_weights 无论成败都返回同一个 path（下坏了也要把兜底交给
        # ultralytics），拿返回值比等于永远算成功：重下没成还会把原文件挪成
        # .bad，第二次加载就变成一句 FileNotFoundError，中文修复提示反而看不见
        ensure_weights(name, weights_dir=os.path.dirname(path) or "models",
                             verbose=verbose, force=True, attempts=2)
        return weights_ready(path)

    # --- 类别表 ---
    def class_names(self) -> Dict[int, str]:
        """模型能认出哪些类（先加载，第一次会慢一点）。

        开放词表权重返回的是**当前词表**，没设词表时是空的：它的类别由用户
        当次填的词决定，权重自带的 COCO-80 只是没人填时的出厂默认。
        """
        self.load()
        return dict(self.names)

    def set_vocab(self, words: Sequence[str]) -> Dict[int, str]:
        """开放词表权重：把「这次要找什么」交给模型，返回新的类别表。

        普通权重没有这一步（类别表在权重里，改不了）；开放词表权重每次换词
        都要重算文本嵌入，首次还要下几百 MB 的文本编码器权重，所以词表没变
        就直接复用。类别号按词的**先后顺序**排（第 0 个词 = 0 号），
        ultralytics 内部也是这个口径。
        """
        model = self.load()
        if not self.open_vocab:
            raise ValueError(f"{os.path.basename(self.model_name)} 是普通权重，类别表在权重里改不了；"
                             "换类别请换成开放词表权重（yolov8x-world.pt、yoloe-26x-seg.pt 这类）")
        cleaned = clean_vocab(words)
        if tuple(cleaned) == self.vocab:
            return dict(self.names)
        try:
            model.set_classes(list(cleaned))
        except Exception as exc:  # noqa: BLE001 缺 CLIP / 嵌入算不动都要翻成人话
            raise RuntimeError(text_encoder_hint(exc)) from exc
        self.vocab = tuple(cleaned)
        self.names = {index: word for index, word in enumerate(cleaned)}
        return dict(self.names)

    # --- 推理 ---
    def predict(self, image: np.ndarray, class_ids: Sequence[int] = (),
                conf: Optional[float] = None) -> List[Detection]:
        """跑一张图；``class_ids`` 非空时把筛选下推给模型（比事后丢弃省时间）。"""
        model = self.load()
        wanted = [int(v) for v in class_ids] if class_ids else None
        if wanted == []:
            return []
        kwargs = dict(imgsz=self.settings.imgsz, conf=self.settings.conf if conf is None else conf,
                      iou=self.settings.iou, max_det=self.settings.max_det,
                      device=self._device, verbose=False)
        # 8.4 起 predict 里传 half 会喊弃用：关着的时候干脆别传
        if self._half:
            kwargs["half"] = True
        if wanted is not None:
            kwargs["classes"] = wanted
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = model.predict(bgr, **kwargs)[0]
        height, width = result.orig_shape[:2]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            boxes = getattr(result, "obb", None)      # 旋转框模型的结果挂在 obb 上
        # masks.data 是 (N, H, W) 的张量，套 bool() 会直接抛"truth value is
        # ambiguous"，整张分割图就全军覆没；只看第 0 维有没有东西
        mask_rows = getattr(getattr(result, "masks", None), "data", None)
        try:
            has_mask = bool(mask_rows is not None and len(mask_rows) > 0)
        except TypeError:                     # 遇上游哪天换成不带长度的对象
            has_mask = mask_rows is not None
        out: List[Detection] = []
        if boxes is None:
            return out
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            name = self.names.get(class_id, str(class_id))
            if wanted is not None and class_id not in wanted:
                continue                              # 个别版本不认 classes，兜底再滤一次
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
            out.append(Detection(
                class_id=class_id, class_name=name, score=float(boxes.conf[i]),
                cx=((x1 + x2) / 2.0) / max(1.0, width), cy=((y1 + y2) / 2.0) / max(1.0, height),
                w=max(0.0, x2 - x1) / max(1.0, width), h=max(0.0, y2 - y1) / max(1.0, height),
                has_mask=has_mask))
        return out

    def describe(self) -> dict:
        """给界面看的模型概况（类别数、任务、认得出哪些类）。"""
        self.load()
        return {"model": os.path.basename(self.model_name), "task": self.task,
                "task_name": TASK_NAMES.get(self.task, self.task),
                "class_count": len(self.names), "device": self._device,
                "open_vocab": self.open_vocab,
                "vocab": list(self.vocab)}


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
