"""业务口径与操作合理性：待复核只数「没定论」的帧、导出如实交代少了哪些张、撤销条认准那一帧。

第 6 轮（业务合理性）与第 7 轮（操作合理性）的回归测试，盯四件事：

1. 「等你复核」在命令行 / 门户 / 两个复核页是同一个数，正向标记不算待办；
2. 导出少收的帧（背景图超比例、状态被剔）必须在 manifest 里留痕；
3. 界面导出传 0 要真的生效（val_ratio=0 / seed=0 / max_empty_ratio=0）；
4. 前端"待确认"与撤销条的契约不被改回去。
"""

import json
import os
import urllib.error
import urllib.request

from _helpers import idx_names, write_label

from autolabel import qc
from autolabel.cli import main
from autolabel.dataset import load_index
from autolabel.export import export_dataset

SAMPLE17 = "0 0.500000 0.500000 0.400000 0.600000 " + " ".join(["0.500000 0.500000 2"] * 17)


def post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get(base, path):
    with urllib.request.urlopen(base + path) as response:
        return json.loads(response.read())


def set_pose(root, rel, **fields):
    index = load_index(root)
    index.upsert(rel, **fields)
    index.save()


def label_manifest(export_root):
    with open(os.path.join(export_root, "manifest.json"), encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# 一、「待复核」只数没定论的帧
# --------------------------------------------------------------------------- #

def test_rejected_frames_stop_occupying_the_queue(dataset):
    """已驳回的帧不该再挂优先复核清单最前面——它已经排除出训练集了。"""
    rels = sorted(idx_names(dataset))
    for rel in rels:
        set_pose(dataset, rel, status="unreviewed", objects=1, score=0.9, flags=["tiny_box"])
        write_label(dataset, rel, [SAMPLE17])
    rejected, edited = rels[0], rels[1]
    set_pose(dataset, rejected, status="rejected")
    set_pose(dataset, edited, status="edited")

    listed = {rel for rel, _item, _priority in qc.flagged_items(dataset)}
    assert rejected not in listed, "已驳回还占着待复核"
    assert edited in listed, "改过但没定论的帧要留在队列里（改完不等于看完）"

    everything = {rel for rel, _item, _priority in qc.flagged_items(dataset, include_reviewed=True)}
    assert rejected in everything, "include_reviewed 要能连已定论的一起复盘"


def test_positive_flag_is_not_an_issue(dataset):
    """「两个模型结论一致」是加分项，不能算进待确认。"""
    rel = sorted(idx_names(dataset))[0]
    write_label(dataset, rel, [SAMPLE17])
    set_pose(dataset, rel, objects=1, score=0.95, flags=[], object_flags=["multi_model_confirmed"])
    item = load_index(dataset).get(rel)
    assert qc.issue_flags(item) == []
    assert qc.needs_review(item) is False
    assert rel not in {row[0] for row in qc.flagged_items(dataset)}


def test_review_counts_match_the_page_queues(server):
    """门户「等你复核」= 两个库「没定论」之和，与两页默认筛选（pending）同口径。"""
    base, store = server
    payload = get(base, "/api/dataset")
    review = payload["review"]
    pose_rows = payload["items"]
    object_rows = payload["object_items"]

    def settled(row, key):
        return (row.get(key) or "unreviewed") in ("approved", "rejected")

    assert review["pose_pending"] == sum(1 for row in pose_rows if not settled(row, "status"))
    assert review["prompt_pending"] == sum(1 for row in object_rows if not settled(row, "prompt_status"))
    assert review["pending"] == review["pose_pending"] + review["prompt_pending"]
    # 正向标记不算问题、置信率阈值来自后端同一个常量
    assert payload["positive_flags"] == sorted(qc.POSITIVE_FLAGS)
    assert payload["low_score"] == qc.LOW_SCORE

    # 驳回一张，门户的数与队列同时掉一张（口径只有一份实现）
    rel = sorted(idx_names(store.root))[0]
    status, body = post(base, "/api/status", {"image": rel, "status": "rejected"})
    assert status == 200, body
    after = get(base, "/api/dataset")["review"]
    assert after["pose_pending"] == review["pose_pending"] - 1
    assert after["pending"] == review["pending"] - 1


# --------------------------------------------------------------------------- #
# 二、导出少收的帧必须留痕
# --------------------------------------------------------------------------- #

def test_background_images_dropped_by_ratio_are_recorded(dataset):
    """背景图被 --max-empty-ratio 压掉时，manifest 与命令行都要说清少了多少张。"""
    from autolabel.export import export_dataset

    rels = sorted(idx_names(dataset))
    write_label(dataset, rels[0], [SAMPLE17])
    set_pose(dataset, rels[0], objects=1, score=0.9)
    write_label(dataset, rels[1], [])
    set_pose(dataset, rels[1], objects=0, score=0.0, flags=["no_person"])

    out = os.path.join(dataset, "exports", "ratio")
    result = export_dataset(dataset, out=out, val_ratio=0.0, max_empty_ratio=0.0, mode="copy")
    assert result["empty"] == 0
    assert result["empty_dropped"] == 1
    assert "背景图超出比例上限" in result["skipped_note"]

    manifest = label_manifest(result["out"])
    reasons = {row["reason"] for row in manifest["skipped"]}
    assert any("背景图超出比例上限" in reason for reason in reasons), manifest["skipped"]
    assert manifest["counts"]["empty_dropped"] == 1
    assert manifest["max_empty_ratio"] == 0.0


def test_export_with_nothing_to_give_says_why(dataset, capsys):
    """全驳回还硬导，别丢一个空目录加一句「导出成功」。"""
    rels = sorted(idx_names(dataset))
    write_label(dataset, rels[0], [SAMPLE17])
    for rel in rels:
        set_pose(dataset, rel, status="rejected", objects=1, score=0.9)
    code = main(["export", dataset, "--only-approved"])
    err = capsys.readouterr().err
    assert code == 1
    assert "没有符合条件的样本可导出" in err and "status=rejected" in err


def test_detection_export_records_scope_and_dropped_background(dataset):
    """提示词库导出同样要说清：哪些空帧被比例压掉、哪些帧因状态被剔。"""
    from autolabel.prompt import export_detection_dataset

    rels = sorted(idx_names(dataset))
    index = load_index(dataset, "objects")
    write_label(dataset, rels[0], ["0 0.5 0.5 0.2 0.2"], lib="objects")
    index.upsert(rels[0], prompt_objects=1, prompt_status="approved")
    write_label(dataset, rels[1], [], lib="objects")          # 看过、确实没目标 = 空标签
    index.upsert(rels[1], prompt_objects=0, prompt_status="approved")
    index.save()

    first = export_detection_dataset(dataset, out=os.path.join(dataset, "exports", "d1"),
                                     val_ratio=0.0, max_empty_ratio=0.0)
    assert first["train"] == 1
    assert first["counts"]["empty_dropped"] == 1
    assert "背景图超出比例上限" in first["skipped_note"], first["skipped_note"]
    with open(os.path.join(first["out"], "prompt_meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["max_empty_ratio"] == 0.0
    assert meta["empty_dropped_images"] == [rels[1]]

    index.upsert(rels[1], prompt_status="rejected")
    index.save()
    second = export_detection_dataset(dataset, out=os.path.join(dataset, "exports", "d2"),
                                     val_ratio=0.0, exclude_status=("rejected",))
    assert "prompt_status=rejected" in second["skipped_note"], second["skipped_note"]


def test_all_broken_labels_do_not_export_an_empty_dataset(dataset):
    """选中帧的标签全读不通时，别丢一个 0 张的目录加一句「导出成功」。"""
    from autolabel.export import export_dataset

    rels = sorted(idx_names(dataset))
    for rel in rels[:2]:
        write_label(dataset, rel, ["0 0.5 0.5 0.6 0.7 0.5 0.5 2 0.5"])   # 字段数不是 3 的倍数
        set_pose(dataset, rel, objects=1, score=0.9)
    target = os.path.join(dataset, "exports", "allbad")
    try:
        export_dataset(dataset, out=target, val_ratio=0.0, mode="copy")
    except ValueError as exc:
        assert "标签全都读不通" in str(exc), exc
        assert not os.path.exists(target), "报错之后别留一个空目录"
        return
    raise AssertionError("全坏标签应该报错，而不是导出 0 张的训练集")


def test_one_broken_label_does_not_sink_the_whole_export(dataset):
    """一行坏标签只该少一张图，不该让两万张的导出整个失败。"""
    from autolabel.export import export_dataset

    rels = sorted(idx_names(dataset))
    write_label(dataset, rels[0], [SAMPLE17])
    set_pose(dataset, rels[0], objects=1, score=0.9)
    write_label(dataset, rels[1], ["0 0.5 0.5 0.6 0.7 0.5 0.5 2 0.5"])   # 字段数不是 3 的倍数
    set_pose(dataset, rels[1], objects=1, score=0.9)

    result = export_dataset(dataset, out=os.path.join(dataset, "exports", "bad"),
                            val_ratio=0.0, mode="copy")
    assert result["train"] == 1, "坏标签那张被跳过，好那张照常导出"
    assert "标签解析失败" in result["skipped_note"], result["skipped_note"]
    manifest = label_manifest(result["out"])
    broken = [row for row in manifest["skipped"] if row["image"] == rels[1]]
    assert broken and "标签解析失败" in broken[0]["reason"]

    stats = qc.dataset_stats(dataset)
    assert stats["bad_labels"] == 1, "QC 统计要给坏标签留一格，别静默少算"


def test_export_hint_names_the_switch_that_actually_helps(server):
    """整库没标注时的报错不能让人去点 --include-all：那个开关管不到缺标签的帧。"""
    base, store = server
    status, body = post(base, "/api/export", {"scope": "all", "val_ratio": 0.0})
    assert status == 400, body
    assert "还没有任何骨骼标签" in body["error"] and "自动标注" in body["error"], body
    assert "--include-all" not in body["error"], "提示里的开关换了也不会改变结果"

    # 命令行同一条路：只报"缺少标签"时不该出现 --include-all 的建议
    err = None
    try:
        export_dataset(store.root, out=os.path.join(store.root, "exports", "nolabel"),
                       val_ratio=0.0, mode="copy")
    except ValueError as exc:
        err = str(exc)
    assert err and "缺少标签" in err and "自动标注" in err, err
    assert "--include-all" not in err, err


def test_scope_filterd_empty_says_where_the_labels_are(server):
    """库里有标签、只是被导出范围滤空：提示要说清有多少张带标签，而不是让人去标注。"""
    base, store = server
    rel = sorted(idx_names(store.root))[0]
    write_label(store.root, rel, [SAMPLE17])
    set_pose(store.root, rel, objects=1, score=0.9, status="rejected")
    status, body = post(base, "/api/export", {"scope": "reviewed", "val_ratio": 0.0})
    assert status == 400, body
    assert "库里共 1 张带标签" in body["error"], body
    assert "换个导出范围" in body["error"], body


# --------------------------------------------------------------------------- #
# 三、界面导出传 0 要真的生效
# --------------------------------------------------------------------------- #

def test_web_export_honours_zero_parameters(server):
    """``val_ratio=0`` / ``seed=0`` / ``max_empty_ratio=0`` 是合法写法，不能被 or 吞成默认值。"""
    base, _store = server
    rels = sorted(idx_names(server[1].root))
    for rel in rels:
        set_pose(server[1].root, rel, objects=1, score=0.9)
        write_label(server[1].root, rel, [SAMPLE17])
    status, body = post(base, "/api/export", {"val_ratio": 0, "seed": 0, "max_empty_ratio": 0,
                                             "mode": "copy"})
    assert status == 200, body
    manifest = label_manifest(body["export"]["out"])
    assert manifest["val_ratio"] == 0.0
    assert manifest["seed"] == 0
    assert manifest["max_empty_ratio"] == 0.0
    assert manifest["counts"]["val"] == 0, "验证集比例 0 就把全部帧留作训练"


def test_web_export_validates_numbers(server):
    """范围错了当场说人话，别把 1.5 这种值悄悄拿去切分。"""
    base, store = server
    rel = sorted(idx_names(store.root))[0]
    write_label(store.root, rel, [SAMPLE17])
    set_pose(store.root, rel, objects=1, score=0.9)
    status, body = post(base, "/api/export", {"val_ratio": 1.5})
    assert status == 400 and "验证集比例不能大于" in body["error"]
    status, body = post(base, "/api/export", {"seed": "abc"})
    assert status == 400 and "随机种子要填数字" in body["error"]


def test_bad_parameter_beats_empty_library(server):
    """空库里填错参数时，先说参数错——"库里还没标签"会盖掉真正写错的那个框。"""
    base, store = server
    status, body = post(base, "/api/export", {"val_ratio": 1.5})
    assert status == 400
    assert "验证集比例不能大于" in body["error"], body
    status, body = post(base, "/api/prompt/export", {"scope": "nope", "seed": "x"})
    assert status == 400 and "未知的导出范围" in body["error"]


def test_detection_export_takes_scope(server):
    """界面上终于能选「只要人工确认过的」，与命令行 --only-approved 同口径。"""
    base, store = server
    rels = sorted(idx_names(store.root))
    index = load_index(store.root, "objects")
    for rel in rels:
        write_label(store.root, rel, ["0 0.5 0.5 0.2 0.2"], lib="objects")
        index.upsert(rel, prompt_objects=1, prompt_status="unreviewed")
    index.save()
    index.upsert(rels[0], prompt_status="approved")
    index.save()

    status, body = post(base, "/api/prompt/export", {"scope": "approved", "val_ratio": 0.0,
                                                    "mode": "copy"})
    assert status == 200, body
    assert body["export"]["scope"] == "approved"
    assert body["export"]["train"] == 1, "只有已接受的那帧该进训练集"
    status, body = post(base, "/api/prompt/export", {"scope": "nope"})
    assert status == 400 and "未知的导出范围" in body["error"]


# --------------------------------------------------------------------------- #
# 四、前端契约：待确认口径与撤销条
# --------------------------------------------------------------------------- #

def static_source(name):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, os.pardir, "autolabel", "static", name), encoding="utf-8") as handle:
        return handle.read()


def test_attention_rule_ignores_positive_flags_on_both_pages():
    """两个复核页的「待确认」都必须剔掉正向标记，且 F 键与它同口径。"""
    for name in ("app.js", "objects.js"):
        source = static_source(name)
        body = source.split("function needsAttention(it) {", 1)[1].split("\n}", 1)[0]
        assert "issueFlags(" in body, f"{name}: needsAttention 还在把正向标记当问题"
        jump = source.split("function jumpFlagged(dir) {", 1)[1].split("\n}", 1)[0]
        assert "needsAttention(" in jump, f"{name}: F 键跳转与顶栏「待确认」两套口径"


def test_undo_bar_targets_the_frame_that_just_changed():
    """撤销条必须带着"刚定论的那一帧"，否则 Q/E 自动翻页后会去回滚下一帧。"""
    for name in ("app.js", "objects.js"):
        source = static_source(name)
        call = source.split("showUndoBar(", 1)[1].split(");", 1)[0]
        assert call.count(",") >= 1, f"{name}: 撤销条只带了一句话，没带撤销动作"
        assert "btn-undo-bar" in source and "runPendingUndo()" in source
        key_case = [line for line in source.splitlines() if 'case "u"' in line]
        assert key_case and all("runPendingUndo()" in line for line in key_case), \
            f"{name}: U 键还在无条件回滚当前帧的坐标"
        assert "revertStatus(" in source, f"{name}: 缺状态回退函数"


def test_portal_uses_backend_review_numbers():
    """门户不再自己数一遍两个库（以前一个数全部、一个数带标记，拼出来的数没人信）。"""
    source = static_source("portal.js")
    assert "ds.review" in source or "review[key]" in source
    body = source.split("function buildStats(ds) {", 1)[1].split("\n}", 1)[0]
    assert "prompt_unreviewed" in body


def test_export_scope_options_match_the_backend(server):
    """界面能选的范围必须与后端 scope 表一致，别出现点了没反应的选项。"""
    source = static_source("common.js")
    block = source.split("const EXPORT_SCOPES = [", 1)[1].split("];", 1)[0]
    for scope in ("reviewed", "approved", "all"):
        assert '"%s"' % scope in block
    base, store = server
    rel = sorted(idx_names(store.root))[0]
    write_label(store.root, rel, [SAMPLE17])
    # 状态设成"已修改"：三档范围都得有东西可导，重点验"选项后端认不认"
    set_pose(store.root, rel, objects=1, score=0.9, status="edited")
    for scope in ("reviewed", "approved", "all"):
        status, body = post(base, "/api/export", {"scope": scope, "val_ratio": 0.0, "mode": "copy"})
        assert status == 200, (scope, body)
    status, body = post(base, "/api/export", {"scope": "wrong"})
    assert status == 400 and "未知的导出范围" in body["error"]
