"""常见数据集类别的中文名：模型给的是英文类名，界面要给人看中文。

只有这一处维护对照表。目标检测库把模型类别收进登记表时，用它把 ``label``
（中文显示名）自动填好；表里没有的类名就退回英文，绝不编一个不像的中文名。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

#: COCO-80（Ultralytics 检测/分割/旋转框权重的默认类别表，顺序与模型 ``names`` 一致）
COCO_ZH: Tuple[Tuple[str, str], ...] = (
    ("person", "人"), ("bicycle", "自行车"), ("car", "汽车"), ("motorcycle", "摩托车"),
    ("airplane", "飞机"), ("bus", "公交车"), ("train", "火车"), ("truck", "卡车"),
    ("boat", "船"), ("traffic light", "红绿灯"), ("fire hydrant", "消防栓"),
    ("stop sign", "停车标志"), ("parking meter", "停车计时器"), ("bench", "长椅"),
    ("bird", "鸟"), ("cat", "猫"), ("dog", "狗"), ("horse", "马"), ("sheep", "绵羊"),
    ("cow", "奶牛"), ("elephant", "大象"), ("bear", "熊"), ("zebra", "斑马"),
    ("giraffe", "长颈鹿"), ("backpack", "双肩包"), ("umbrella", "雨伞"),
    ("handbag", "手提包"), ("tie", "领带"), ("suitcase", "行李箱"), ("frisbee", "飞盘"),
    ("skis", "滑雪板"), ("snowboard", "单板滑雪板"), ("sports ball", "球"),
    ("kite", "风筝"), ("baseball bat", "球棒"), ("baseball glove", "棒球手套"),
    ("skateboard", "滑板"), ("surfboard", "冲浪板"), ("tennis racket", "网球拍"),
    ("bottle", "瓶子"), ("wine glass", "高脚杯"), ("cup", "杯子"), ("fork", "叉子"),
    ("knife", "刀"), ("spoon", "勺子"), ("bowl", "碗"), ("banana", "香蕉"),
    ("apple", "苹果"), ("sandwich", "三明治"), ("orange", "橙子"), ("broccoli", "西兰花"),
    ("carrot", "胡萝卜"), ("hot dog", "热狗"), ("pizza", "披萨"), ("donut", "甜甜圈"),
    ("cake", "蛋糕"), ("chair", "椅子"), ("couch", "沙发"), ("potted plant", "盆栽"),
    ("bed", "床"), ("dining table", "餐桌"), ("toilet", "马桶"), ("tv", "电视"),
    ("laptop", "笔记本电脑"), ("mouse", "鼠标"), ("remote", "遥控器"),
    ("keyboard", "键盘"), ("cell phone", "手机"), ("microwave", "微波炉"),
    ("oven", "烤箱"), ("toaster", "烤面包机"), ("sink", "水槽"),
    ("refrigerator", "冰箱"), ("book", "书"), ("clock", "时钟"), ("vase", "花瓶"),
    ("scissors", "剪刀"), ("teddy bear", "泰迪熊"), ("hair drier", "吹风机"),
    ("toothbrush", "牙刷"),
)

#: 常见别名：不同数据集里同一个东西叫法不一样，登记表里要落到同一个中文名
ALIASES: Dict[str, str] = {
    "mobile phone": "手机", "cellphone": "手机", "telephone": "手机",
    "tv monitor": "电视", "monitor": "显示器", "laptop computer": "笔记本电脑",
    "diningtable": "餐桌", "pottedplant": "盆栽", "sofa": "沙发",
    "motorbike": "摩托车", "aeroplane": "飞机", "pedestrian": "行人",
    "rubbish bin": "垃圾桶", "trash can": "垃圾桶", "package": "包裹",
    "box": "箱子", "helmet": "头盔", "safety helmet": "安全帽", "vest": "反光背心",
    "safety vest": "反光背心", "mask": "口罩", "face": "人脸", "head": "头",
    "hand": "手", "food": "食物", "drink": "饮品", "water bottle": "水瓶",
    "carton": "纸盒", "cans": "易拉罐", "candle": "蜡烛", "plant": "植物",
    "tree": "树", "flower": "花", "window": "窗户", "door": "门",
    "stairs": "楼梯", "elevator": "电梯", "shelf": "货架", "counter": "台面",
    "cash register": "收银机", "shopping cart": "购物车", "wheelchair": "轮椅",
    "stroller": "婴儿车", "pet": "宠物", "poultry": "家禽", "livestock": "牲畜",
}

ZH_INDEX: Dict[str, str] = {name.lower(): zh for name, zh in COCO_ZH}
ZH_INDEX.update({name.lower(): zh for name, zh in ALIASES.items()})


def zh_for(name: str) -> str:
    """英文类名 -> 中文显示名；查不到返回空串（调用方自己退回英文）。"""
    key = str(name or "").strip().lower()
    if not key:
        return ""
    if key in ZH_INDEX:
        return ZH_INDEX[key]
    return ZH_INDEX.get(key.replace("_", " "), "") or ZH_INDEX.get(key.replace("-", " "), "")


def labels_for(names: Sequence[str]) -> List[str]:
    """批量取中文名，缺的位置留空串。"""
    return [zh_for(name) for name in names]


def known_count() -> int:
    """对照表里一共有多少个类名（写文档/自检用）。"""
    return len(ZH_INDEX)


def source_of(name: str) -> str:
    """这个类名出自哪张对照表，界面「来源」一栏用。"""
    key = str(name or "").strip().lower()
    if key in {name.lower() for name, _ in COCO_ZH}:
        return "COCO-80"
    if key in ALIASES:
        return "常见别名"
    return ""


__all__ = ["ALIASES", "COCO_ZH", "ZH_INDEX", "labels_for", "known_count", "source_of", "zh_for"]
