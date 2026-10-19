"""试词：一张图上问清「这些词这个权重到底认不认、最高给了多少分」。

批量标注跑完只剩"0 个框"是这之前最难查的问题——权重压根没这一类、分被阈值挡了、
图里确实没这个目标，三种原因的下一步动作互不相同（换权重 / 降阈值 / 换图），
却只有一句"标注器没在这张图上检出目标"。这里守住试词报得准、且不碰任何数据：

* 每个词一行，分数、过没过线、下一步怎么做，分开说；
* 建议阈值只算**没过线**的那些词——已经有词出框了还建议降到 0.93 是添乱；
* 普通权重把"这一类权重根本不认"说明白，并给出该换成哪类权重；
* 全程只读：不写标签、不动索引、不建批次。

推理一律用替身：真跑开放词表权重要下几百 MB 的文本编码器。
"""
import os

import pytest

from autolabel import detect as dt
from autolabel.backends import Detection
from autolabel.dataset import adopt_from_pool, label_path_for, load_index
from autolabel.library import LIB_DETECT, LIB_POSE

from test_detect import MODEL_CLASSES, FakeBackend
from test_prompt_web import post


class VocabBackend:
    """开放词表后端替身：词表设在它身上，分数按词给。"""

    def __init__(self, scores, model="yolov8x-worldv2.pt"):
        self.model_name = model
        self.task = "detect"
        self.open_vocab = "YOLO-World"
        self.vocab = ()
        self.names = {}
        self.scores = dict(scores)
        self.vocabs = []

    def load(self):
        return self

    def describe(self):
        return {"model": os.path.basename(self.model_name), "task": self.task,
                "task_name": "目标检测", "class_count": len(self.names), "device": "cpu",
                "open_vocab": self.open_vocab, "vocab": list(self.vocab)}

    def class_names(self):
        return dict(self.names)

    def set_vocab(self, words):
        self.vocab = tuple(words)
        self.names = {index: word for index, word in enumerate(words)}
        self.vocabs.append(list(words))
        return dict(self.names)

    def predict(self, image, class_ids=(), conf=None, floor=None):
        """与真实后端同口径：给了 ``floor`` 就把没过 ``conf`` 的候选一起交回来。"""
        rows = []
        for class_id in (list(class_ids) if class_ids else sorted(self.names)):
            score = float(self.scores.get(self.names.get(class_id, ""), 0.0))
            if score <= 0:
                continue
            rows.append(Detection(class_id=class_id, class_name=self.names[class_id],
                                  score=score, cx=0.5, cy=0.5, w=0.2, h=0.2,
                                  near_miss=floor is not None and score < (conf or 0.0)))
        return rows


@pytest.fixture()
def detect_ds(dataset):
    """素材同时进目标检测库（与 test_detect 同一做法）。"""
    rels = [rel for rel, _ in load_index(dataset, LIB_POSE).ordered_items()]
    adopt_from_pool(dataset, rels, lib=LIB_DETECT)
    return dataset


def first_rel(root):
    return [rel for rel, _ in load_index(root, LIB_DETECT).ordered_items()][0]


def use_vocab(monkeypatch, scores, model="yolov8x-worldv2.pt"):
    backend = VocabBackend(scores, model=model)
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: backend)
    return backend


def test_probe_scores_each_word_and_suggests_for_missed_only(detect_ds, monkeypatch):
    backend = use_vocab(monkeypatch, {"person": 0.91, "shirt": 0.02, "t-shirt": 0.0})
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                         words=["person", "shirt", "t-shirt"], conf=0.25, imgsz=640)
    assert [row["word"] for row in out["words"]] == ["person", "shirt", "t-shirt"]
    person, shirt, tshirt = out["words"]
    assert person["over"] and person["boxes"] == 1 and "能用" in person["advice"]
    assert not shirt["over"] and shirt["best"] == pytest.approx(0.02)
    assert "降到" in shirt["advice"] and "1024" in shirt["advice"]
    assert tshirt["best"] == 0.0 and "一点分都没给" in tshirt["advice"]
    # 建议阈值只看过不了线的词：person 0.91 已经出框，不该拿它当"建议降到 0.909"
    assert out["suggested_conf"] == pytest.approx(0.019)
    assert out["missed_best"] == pytest.approx(0.02)
    assert out["best"] == pytest.approx(0.91)
    assert backend.vocabs == [["person", "shirt", "t-shirt"]]


def test_probe_without_any_over_box_suggests_threshold(detect_ds, monkeypatch):
    use_vocab(monkeypatch, {"shirt": 0.026, "top": 0.011})
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                         words=["shirt", "top"], conf=0.25)
    assert out["suggested_conf"] == pytest.approx(0.025)
    assert all(not row["over"] for row in out["words"])
    assert "衬衫" or "shirt" in out["words"][0]["word"]


def test_all_words_over_leaves_no_suggestion(detect_ds, monkeypatch):
    use_vocab(monkeypatch, {"person": 0.9, "dog": 0.7})
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                         words=["person", "dog"], conf=0.25)
    assert out["suggested_conf"] is None and out["missed_best"] is None


def test_plain_weight_says_this_class_is_not_in_the_weight(detect_ds, monkeypatch):
    backend = FakeBackend(names=MODEL_CLASSES)
    monkeypatch.setattr(dt, "_backend", lambda settings, dir_=None: backend)
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolo11n.pt",
                         words=["person", "t-shirt"], conf=0.25)
    rows = {row["word"]: row for row in out["words"]}
    assert rows["person"]["in_vocab"] and rows["person"]["over"]
    assert not rows["t-shirt"]["in_vocab"] and rows["t-shirt"]["model_id"] is None
    assert "开放词表" in rows["t-shirt"]["advice"]
    # 不存在的类别不该拿去问模型：类别号列表里只许有权重认识的那些
    assert backend.requests[0][0] == (0,)
    assert out["class_count"] == len(MODEL_CLASSES)
    assert "person" in out["model_classes"]


def test_words_split_on_chinese_commas_and_dedupe(detect_ds, monkeypatch):
    backend = use_vocab(monkeypatch, {"t-shirt": 0.3, "shirt": 0.2, "top": 0.1})
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                         words=["t-shirt，shirt\ntop", "shirt"], conf=0.25)
    assert backend.vocabs == [["t-shirt", "shirt", "top"]]
    assert [row["word"] for row in out["words"]] == ["t-shirt", "shirt", "top"]


def test_probe_accepts_one_whole_string(detect_ds, monkeypatch):
    """界面传的是输入框里那一整串：不能被拆成一个个字母。"""
    use_vocab(monkeypatch, {"t-shirt": 0.4, "shirt": 0.2})
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                         words="t-shirt,shirt", conf=0.25)
    assert sorted(row["word"] for row in out["words"]) == ["shirt", "t-shirt"]


def test_probe_refuses_empty_and_too_many_words(detect_ds, monkeypatch):
    use_vocab(monkeypatch, {"person": 0.5})
    rel = first_rel(detect_ds)
    with pytest.raises(ValueError) as bad:
        dt.probe_words(detect_ds, rel, words=["  ", "，"])
    assert "哪些词" in str(bad.value)
    with pytest.raises(ValueError) as many:
        dt.probe_words(detect_ds, rel, words=[f"w{i}" for i in range(13)])
    assert "最多试 12 个词" in str(many.value)


def test_probe_writes_nothing(detect_ds, monkeypatch):
    use_vocab(monkeypatch, {"person": 0.9})
    rel = first_rel(detect_ds)
    path = label_path_for(detect_ds, rel, LIB_DETECT)
    before = load_index(detect_ds, LIB_DETECT).get(rel)
    out = dt.probe_words(detect_ds, rel, model="yolov8x-worldv2.pt", words=["person"])
    assert out["words"][0]["boxes"] == 1
    assert not os.path.exists(path), "试词不该落标签文件"
    after = load_index(detect_ds, LIB_DETECT).get(rel)
    assert after.get("detect_objects") == before.get("detect_objects")
    assert after.get("detect_status") == before.get("detect_status")


def test_probe_missing_image_says_so(detect_ds, monkeypatch):
    use_vocab(monkeypatch, {"person": 0.9})
    with pytest.raises(FileNotFoundError) as exc:
        dt.probe_words(detect_ds, "ghost.jpg", words=["person"])
    # 只回一个文件名等于没说：图还在素材池、没进目标检测库是这里最常见的情况
    assert "目标检测库" in str(exc.value) and "ghost.jpg" in str(exc.value)


def test_probe_api_route_is_wired_and_answers_in_chinese(server):
    """接口挂上了没有：空请求体要回一句中文，而不是 404 或一栈英文。"""
    base, _store = server
    code, body, _ = post(base, "/api/detect/probe", {})
    assert code == 400 and "哪张图" in body["error"], body
    code, body, _ = post(base, "/api/detect/probe", {"image": "ghost.jpg", "words": "person"})
    # 图不在本库统一按 404 回（和读标注 / 存框 / 改状态一个口径），句子照样是中文
    assert code == 404 and "已不在" in body["error"], body


def test_probe_store_validates_before_touching_the_model(detect_ds, monkeypatch):
    """三种打错都在加载权重之前挡住：开放词表首次加载要几十秒，转半天才报错最坑。"""
    from autolabel.review import ReviewStore

    use_vocab(monkeypatch, {"person": 0.9})
    store = ReviewStore(detect_ds)
    rel = first_rel(detect_ds)
    with pytest.raises(ValueError) as no_image:
        store.probe_detect({"words": "person"})
    assert "哪张图" in str(no_image.value)
    with pytest.raises(ValueError) as no_words:
        store.probe_detect({"image": rel, "words": "， ,"})
    assert "哪些词" in str(no_words.value)
    with pytest.raises(ValueError) as bad_size:
        store.probe_detect({"image": rel, "words": "person", "imgsz": 99})
    assert "输入分辨率" in str(bad_size.value)
    out = store.probe_detect({"image": rel, "words": "person", "model": "yolov8x-worldv2.pt"})
    assert out["words"][0]["word"] == "person" and out["words"][0]["over"]


class PrintProbe:
    """``_print_probe`` 的替身：按图给一份试词结果，顺带记下被叫了几次。"""

    def __init__(self, results):
        self.results = dict(results)
        self.calls = []

    def probe_words(self, root, rel, **_kwargs):
        self.calls.append(rel)
        return self.results[rel]


def _probe_args(dataset, images):
    import argparse

    return argparse.Namespace(dataset=dataset, probe="person, dog", model="yolo11n.pt",
                              conf=None, imgsz=None, device="", images=images)


def _row(word, best, boxes):
    return {"word": word, "best": best, "boxes": boxes, "over": boxes > 0,
            "advice": f"{word} 的下一步"}


def _result(words, suggested=None, missed=None):
    return {"model": "yolo11n.pt", "task": "detect", "task_name": "目标检测",
            "device": "cpu", "imgsz": 640, "conf": 0.25, "open_vocab": "",
            "class_count": 80, "words": words, "best": max(w["best"] for w in words),
            "suggested_conf": suggested, "missed_best": missed,
            "model_classes": ["person", "bus"]}


def test_cli_probe_prints_every_image_then_the_next_step(detect_ds, capsys):
    """多张图试词要一张一张打完，最后给一条能照抄的命令。

    早先的汇总变量写错作用域，第二张图打完就 ``NameError``，
    推理全跑完了却在最后一步崩掉、连"下一步"都没打印出来。
    """
    from autolabel import cli

    rels = [rel for rel, _ in load_index(detect_ds, LIB_DETECT).ordered_items()][:2]
    assert len(rels) == 2, "样例库至少要有两张图，这条测的就是多张"
    detector = PrintProbe({
        rels[0]: _result([_row("person", 0.9, 3), _row("dog", 0.0, 0)]),
        rels[1]: _result([_row("person", 0.8, 2), _row("dog", 0.02, 0)],
                         suggested=0.019, missed=0.02),
    })
    assert cli._print_probe(_probe_args(detect_ds, rels), detector, rels) == 0
    out = capsys.readouterr().out
    assert out.count("下一步照抄") == 1, out
    assert "--add-classes person,dog" in out, out
    assert detector.calls == rels, "两张图都得真问到，不能只试第一张"


def cli_probe_limit():
    from autolabel.cli import PROBE_PRINT_IMAGES

    return PROBE_PRINT_IMAGES


def test_cli_probe_caps_images_and_says_so(detect_ds, capsys):
    """``--images`` 给了五张只打三张：悄悄丢掉剩下两张，比多打一行说明难查。"""
    from autolabel import cli

    rels = [rel for rel, _ in load_index(detect_ds, LIB_DETECT).ordered_items()]
    picked = rels[:3] + ["ghost_1.jpg", "ghost_2.jpg"]
    extra = len(picked) - cli_probe_limit()
    detector = PrintProbe({rel: _result([_row("person", 0.9, 1)]) for rel in picked})
    cli._print_probe(_probe_args(detect_ds, picked), detector, picked)
    out = capsys.readouterr().out
    assert f"多出的 {extra} 张这次不跑" in out, out
    assert detector.calls == picked[:3], detector.calls


def test_probe_stops_recommending_the_resolution_it_is_already_on(detect_ds, monkeypatch):
    """分辨率已经提到 1024，就别再说"同时把输入分辨率提到 1024"。

    实测踩出来的：`--probe --imgsz 1024` 跑完，那条建议读起来像没执行一样。
    """
    use_vocab(monkeypatch, {"person": 0.9, "shirt": 0.02})
    on_1024 = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                             words=["person", "shirt"], conf=0.25, imgsz=1024)
    assert on_1024["suggest_imgsz"] == 1280
    shirt = [row for row in on_1024["words"] if row["word"] == "shirt"][0]
    assert "提到 1280" in shirt["advice"] and "提到 1024" not in shirt["advice"], shirt["advice"]


def test_probe_at_the_top_resolution_says_so(detect_ds, monkeypatch):
    """最高一档没有"再提一档"可说，得明说没有，而不是不提这件事。"""
    use_vocab(monkeypatch, {"person": 0.9, "shirt": 0.02})
    out = dt.probe_words(detect_ds, first_rel(detect_ds), model="yolov8x-worldv2.pt",
                         words=["person", "shirt"], conf=0.25, imgsz=1280)
    assert out["suggest_imgsz"] is None
    shirt = [row for row in out["words"] if row["word"] == "shirt"][0]
    assert "没有更高一档" in shirt["advice"], shirt["advice"]
