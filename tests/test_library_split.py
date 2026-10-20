"""两个标注库的隔离契约：共享素材池、独立索引/批次、删除粒度与老布局迁移。"""

import os
import shutil

from _helpers import write_label
from test_upload import SAMPLE, jpeg_bytes

from autolabel.batches import (
    UNGROUPED_ID,
    attach_items,
    batch_file_rows,
    create_batch,
    delete_images,
    list_batches,
    load_batches,
    reconcile_with_disk,
    save_batches,
)
from autolabel.cli import main
from autolabel.dataset import (
    adopt_from_pool,
    images_dir,
    import_images,
    init_dataset,
    label_path_for,
    load_index,
    pool_items,
    resolve_image,
)
from autolabel.library import migrate
from autolabel.review import ReviewStore


def objects_items(root):
    return load_index(root, "objects").items


# --------------------------------------------------------------------------- //
# 素材池：同一帧只落盘一次，两个库各持引用
# --------------------------------------------------------------------------- //
def test_import_into_second_lib_reuses_pool_file(tmp_path):
    """同一个源目录导两次（分别进两个库）：文件只有一份，两个库各一条条目。"""
    root = str(tmp_path / "ds")
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "one.jpg").write_bytes(jpeg_bytes(3))
    init_dataset(root)

    first = import_images(root, [str(src)], dedupe="exact")
    assert first["imported"] == 1 and first["reused"] == 0
    rel = first["imported_files"][0][1]

    second = import_images(root, [str(src)], dedupe="exact", lib="objects")
    assert second["imported"] == 0 and second["reused"] == 1
    assert [r for _s, r in second["imported_files"]] == [rel]
    # 素材池里只有一份文件，且两个库都指向它
    assert sorted(os.listdir(images_dir(root))) == [rel]
    assert rel in load_index(root).items and rel in objects_items(root)

    rows = {row["image"]: row for row in pool_items(root)}
    assert rows[rel]["in_pose"] and rows[rel]["in_objects"]
    assert rows[rel]["bytes"] > 0


def test_adopt_from_pool_is_idempotent_and_does_not_leak_status(tmp_path):
    """从素材池添加只带素材元信息，不带对面的复核状态；重复添加不产生新条目。"""
    root = str(tmp_path / "ds")
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "a.jpg").write_bytes(jpeg_bytes(5))
    init_dataset(root)
    rel = import_images(root, [str(src)])["imported_files"][0][1]
    write_label(root, rel, [SAMPLE])
    pose = load_index(root)
    pose.upsert(rel, status="approved", objects=1)
    pose.save()

    stats = adopt_from_pool(root, [rel, "images/" + rel], lib="objects")
    assert stats["added"] == 1 and stats["already"] == 1 and stats["missing"] == []
    entry = objects_items(root)[rel]
    # 提示词库初始状态是自己的那一套，姿态库的 approved / objects 不能渗过来
    assert entry["prompt_status"] == "unreviewed" and entry["prompt_objects"] is None
    assert "status" not in entry and "objects" not in entry
    assert entry["width"] and entry["imported_at"]

    again = adopt_from_pool(root, [rel], lib="objects")
    assert again["added"] == 0 and again["already"] == 1
    assert len(objects_items(root)) == 1
    # 姿态库的标签与状态不受影响
    assert load_index(root).get(rel)["status"] == "approved"
    assert os.path.exists(label_path_for(root, rel))

    missing = adopt_from_pool(root, ["ghost.jpg"], lib="objects")
    assert missing["added"] == 0 and missing["missing"] == ["ghost.jpg"]


# --------------------------------------------------------------------------- //
# 删除粒度：lib 只摘本库引用，purge 才动素材文件
# --------------------------------------------------------------------------- //
def _shared_dataset(tmp_path):
    """一张素材、两个库都引用的最小数据集。"""
    root = str(tmp_path / "ds")
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "one.jpg").write_bytes(jpeg_bytes(7))
    (src / "two.jpg").write_bytes(jpeg_bytes(8))
    init_dataset(root)
    rels = [r for _s, r in import_images(root, [str(src)])["imported_files"]]
    for rel in rels:
        write_label(root, rel, [SAMPLE])
    # 提示词库引用同一批素材文件，并给第一张写好检测框标签
    adopt_from_pool(root, rels, lib="objects")
    write_label(root, rels[0], ["0 0.5 0.5 0.2 0.3"], lib="objects")
    pose = load_index(root)
    pose.upsert(rels[0], status="approved", objects=1)
    pose.save()
    obj = load_index(root, "objects")
    obj.upsert(rels[0], prompt_status="edited", prompt_objects=2)
    obj.save()
    return root, rels


def test_delete_lib_mode_keeps_file_and_other_library(tmp_path):
    root, (one, two) = _shared_dataset(tmp_path)

    result = delete_images(root, [one], lib="pose", regenerate=False)
    assert result["mode"] == "lib" and result["lib"] == "pose"
    assert result["items"] == 1 and result["removals"] == 1
    assert one not in load_index(root).items and one in objects_items(root)
    assert os.path.exists(resolve_image(root, one))
    # 姿态库自己的标签被清掉，提示词库的标签原封不动
    assert not os.path.exists(label_path_for(root, one))
    assert os.path.exists(label_path_for(root, one, "objects"))
    assert os.path.exists(result["snapshot"])


def test_delete_purge_drops_both_libraries_at_once(tmp_path):
    """purge = 两个库一起摘：一次调用带走两边条目、两套标签和素材文件。"""
    root, (one, two) = _shared_dataset(tmp_path)

    result = delete_images(root, [one], lib="pose", mode="purge", regenerate=False)
    assert result["removals"] == 2 and result["items"] == 1
    assert one not in load_index(root).items and one not in objects_items(root)
    assert not os.path.exists(label_path_for(root, one, "objects"))
    assert not os.path.exists(label_path_for(root, one))
    assert not os.path.exists(resolve_image(root, one))
    # 另一张没被勾选的素材与两个库的引用都完好
    assert two in load_index(root).items and two in objects_items(root)
    assert os.path.exists(resolve_image(root, two))

    # 已经没人引用了，再删一次要如实报错，而不是默默成功
    try:
        delete_images(root, [one], lib="objects", mode="purge", regenerate=False)
    except ValueError as exc:
        assert "不在索引里" in str(exc)
    else:
        raise AssertionError("重复删除应该报「不在索引里」")


def test_purge_keep_files_keeps_the_image(tmp_path):
    """keep_files=True：两个库的条目与标签都清掉，素材文件仍留在池里。"""
    root, (one, _two) = _shared_dataset(tmp_path / "keep")

    kept = delete_images(root, [one], lib="objects", mode="purge",
                         keep_files=True, regenerate=False)
    assert kept["keep_files"] is True
    # 两个库的标签都清了，素材本体留在池里
    assert not os.path.exists(label_path_for(root, one))
    assert not os.path.exists(label_path_for(root, one, "objects"))
    assert os.path.exists(resolve_image(root, one))
    assert one not in objects_items(root) and one not in load_index(root).items
    assert not os.path.exists(label_path_for(root, one, "objects"))


def test_batch_file_rows_marks_shared_files(tmp_path):
    """展开批次时要知道这张是不是两库共用，删除按钮才知道该说什么。"""
    root, (one, two) = _shared_dataset(tmp_path)
    batch = create_batch(root, name="共享批次", kind="image", source="cli", lib="objects")
    attach_items(root, batch["id"], [one, two], lib="objects")

    rows = {row["image"]: row for row in batch_file_rows(root, batch["id"])}
    assert rows[one]["shared"] is True and rows[one]["lib"] == "objects"
    # 跨库补来的姿态状态：提示词批次也要能显示骨骼标注进度
    assert rows[one]["status"] == "approved" and rows[one]["objects"] == 1

    delete_images(root, [one], lib="pose", regenerate=False)
    rows = {row["image"]: row for row in batch_file_rows(root, batch["id"])}
    assert rows[one]["shared"] is False


# --------------------------------------------------------------------------- //
# 老布局迁移
# --------------------------------------------------------------------------- //
def _make_legacy_dataset(root):
    """手工拼一个旧布局数据集：一本索引混放两套状态 + 一张批次表 + 老备份目录。"""
    src = os.path.join(root, "src")
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "one.jpg"), "wb") as fh:
        fh.write(jpeg_bytes(11))
    init_dataset(root)
    rel = import_images(root, [src])["imported_files"][0][1]

    # 旧布局只有一张批次表，提示词批次当时也写在姿态库的 batches.json 里
    prompt_batch = create_batch(root, name="老提示词批次", kind="prompt", source="cli", lib="pose")
    attach_items(root, prompt_batch["id"], [rel], lib="pose")

    idx = load_index(root)
    idx.upsert(rel, prompt_status="edited", prompt_objects=2, prompt_flags=["small_box"],
               batch_id=prompt_batch["id"])
    idx.save()
    # 批次表里那条 prompt 记录是旧写法：module=prompt（甚至可能没有 module）
    rows = load_batches(root)
    assert [row["id"] for row in rows] == [prompt_batch["id"]]
    for row in rows:
        row["module"] = "prompt"  # 旧写法：module=prompt
    save_batches(root, rows)
    # 提示词库自己的元数据全删掉，模拟"分库之前"的状态
    shutil.rmtree(os.path.join(root, ".autolabel", "objects"))
    legacy_originals = os.path.join(root, ".autolabel", "prompt_originals")
    os.makedirs(legacy_originals, exist_ok=True)
    with open(os.path.join(legacy_originals, "one.txt"), "w", encoding="utf-8") as fh:
        fh.write("legacy original")
    return root, rel, prompt_batch["id"]


def test_migrate_splits_index_batches_and_dirs(tmp_path):
    root, rel, batch_id = _make_legacy_dataset(str(tmp_path / "ds"))

    result = migrate(root)
    assert result["moved_items"] == 1 and result["moved_batches"] == 1
    assert result["moved_dirs"] and any("objects/originals" in line for line in result["moved_dirs"])

    # 姿态库只剩骨骼相关的键
    pose = load_index(root).get(rel)
    assert not any(key.startswith("prompt_") for key in pose)
    assert pose.get("batch_id") != batch_id

    # 提示词库拿到 prompt_* + 素材元信息 + 批次归属
    entry = objects_items(root)[rel]
    assert entry["prompt_status"] == "edited" and entry["prompt_objects"] == 2
    assert entry["prompt_flags"] == ["small_box"] and entry["batch_id"] == batch_id
    assert entry["width"] and entry["sha256"]

    assert {row["id"] for row in load_batches(root, "objects")} == {batch_id}
    assert batch_id not in {row["id"] for row in load_batches(root)}
    assert not os.path.isdir(os.path.join(root, ".autolabel", "prompt_originals"))
    assert os.path.exists(os.path.join(root, ".autolabel", "objects", "originals", "one.txt"))

    # 重复执行必须无副作用
    again = migrate(root)
    assert again["moved_items"] == 0 and again["moved_batches"] == 0 and again["moved_dirs"] == []


def test_cli_migrate_and_review_service_agree(tmp_path):
    """迁移完命令行和复核服务看到的必须是同一份数据。"""
    root, rel, batch_id = _make_legacy_dataset(str(tmp_path / "ds"))

    assert main(["migrate", root]) == 0
    out_lines = list_batches(root)
    modules = {row["id"]: row["module"] for row in out_lines}
    assert modules[batch_id] == "objects"
    assert modules[f"{UNGROUPED_ID}:pose"] == "pose"

    assert main(["migrate", root]) == 0
    # 幂等：第二次什么都没搬
    assert main(["migrate", root]) == 0
    assert rel in objects_items(root)


def test_migrate_gives_pose_only_dataset_an_objects_index(tmp_path, capsys):
    """纯姿态数据集（从没跑过提示词）迁移后要能直接打开复核界面。

    以前 ``migrate`` 只搬 ``prompt_*`` 条目，没有可搬的就什么都不做，
    提示词库索引始终不存在，而复核存储层要求两本索引都在 → 网页 500。
    """
    root = str(tmp_path / "ds")
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.jpg").write_bytes(jpeg_bytes(7))
    init_dataset(root)
    rel = import_images(root, [str(src)])["imported_files"][0][1]
    # 回到分库之前的状态：提示词库整块不存在，批次表只有一份
    shutil.rmtree(os.path.join(root, ".autolabel", "objects"))
    objects_index = os.path.join(root, ".autolabel", "objects", "index.json")
    assert not os.path.exists(objects_index)

    result = migrate(root)
    assert result["created"] == [os.path.join(".autolabel", "objects", "index.json")]
    assert os.path.exists(objects_index)
    # 空的提示词库：不掺水，也不丢名字
    assert objects_items(root) == {}
    assert load_index(root, "objects").data["name"] == load_index(root).data["name"]
    # 复核服务的存储层现在能构造，两个库各自看到自己的素材
    store = ReviewStore(root)
    assert rel in store.index.items and store.objects_index.items == {}

    # 命令行口径一致，且重复执行只报"已经迁移完"
    assert main(["migrate", root]) == 0
    assert "新建" not in capsys.readouterr().out
    assert main(["migrate", root]) == 0
    assert "已经是分库布局" in capsys.readouterr().out


def test_reconcile_claims_orphans_to_the_running_library(tmp_path):
    """对账时孤儿素材优先给"正在标注"的那个库，别把对方跑了一半的图抢走。"""
    root, (one, _two) = _shared_dataset(tmp_path)
    # 两库索引都丢掉引用，素材文件仍在 → 变孤儿
    for lib in ("pose", "objects"):
        idx = load_index(root, lib)
        idx.remove(one)
        idx.save()
    batch = create_batch(root, name="提示词在跑", kind="prompt", source="ui", lib="objects")
    attach_items(root, batch["id"], [], lib="objects")

    result = reconcile_with_disk(root)
    assert result["by_lib"]["objects"]["recovered"] == 1
    assert result["by_lib"]["pose"]["recovered"] == 0
    assert one in objects_items(root)
    assert one not in load_index(root).items
    assert objects_items(root)[one]["batch_id"] == batch["id"]


def test_dataset_status_reports_two_libraries(tmp_path, capsys):
    """status 要分开报两个库的规模，别再用一个数字糊弄。"""
    root, _rels = _shared_dataset(tmp_path)
    assert main(["status", root]) == 0
    out = capsys.readouterr().out
    assert "素材池" in out


def test_batch_rows_show_cross_library_progress(tmp_path):
    """批次行的骨骼/检测框两列要和展开的文件清单说的是同一件事。"""
    root, (one, two) = _shared_dataset(tmp_path)
    batch = create_batch(root, name="共享批次", kind="image", source="cli", lib="objects")
    attach_items(root, batch["id"], [one, two], lib="objects")

    rows = {row["id"]: row for row in list_batches(root)}
    shared = rows[batch["id"]]
    # 提示词库的批次也要看见姿态库标的 1 个人
    assert shared["pose"] == {"labeled": 1, "people": 1, "reviewed": 1,
                              "unreviewed": 1, "flagged": 0}
    assert shared["prompt"] == {"labeled": 1, "boxes": 2, "reviewed": 1,
                                "unreviewed": 1, "flagged": 0}
    # 两列的小计加起来不能超过这批素材的张数口径
    assert shared["items"] == 2


def test_cli_auto_into_objects_lib_skips_pose_stage(tmp_path, capsys):
    """`auto --lib objects`：素材进提示词库、批次归提示词库，且不碰 YOLO-pose 那三段。"""
    root = str(tmp_path / "ds")
    src = tmp_path / "src"
    src.mkdir(parents=True)
    (src / "a.jpg").write_bytes(jpeg_bytes(11))
    (src / "b.jpg").write_bytes(jpeg_bytes(12))

    assert main(["auto", root, str(src), "--lib", "objects"]) == 0
    out = capsys.readouterr().out
    assert "不跑 YOLO-pose" in out and "prompt" in out
    assert len(objects_items(root)) == 2
    assert load_index(root).items == {}          # 姿态库一条都不该有
    assert load_batches(root, "pose") == []      # 批次也不该记到姿态库
    assert [sorted(row["items"]) for row in load_batches(root, "objects")] == [["a.jpg", "b.jpg"]]


def test_dataset_status_tells_where_to_upload_empty_libs(tmp_path, capsys):
    """空库的 status 不再刷一屏 0，直接写清往哪个库灌素材。"""
    root = str(tmp_path / "ds")
    init_dataset(root)

    assert main(["status", root]) == 0
    out = capsys.readouterr().out
    assert "--lib pose" in out and "--lib objects" in out and "--lib detect" in out
    assert "每图人数" not in out and "框高度px" not in out
    # 空着的库只给一行指路，不许再刷一行全 0：一屏 0 加一句"还没有素材"，
    # 用户分不清是没数据还是软件读丢了
    assert "姿态库 图片 0" not in out


def test_status_of_detect_only_dataset_does_not_look_like_lost_data(tmp_path, capsys):
    """只在目标检测库干过活：姿态库那两行全 0 最容易让人以为标注丢了。"""
    root, rels = _shared_dataset(tmp_path)
    from autolabel.dataset import adopt_from_pool

    adopt_from_pool(root, list(rels), lib="detect")
    idx = load_index(root)
    for rel in list(idx.items):      # 清空姿态库，只留检测库
        idx.remove(rel)
    idx.save()

    assert main(["status", root]) == 0
    out = capsys.readouterr().out
    assert "姿态库还没有素材" in out and "姿态库 图片 0" not in out
    assert "目标检测 已标" in out


def test_split_legacy_index_merges_into_an_existing_objects_index(tmp_path):
    """提示词库索引**已经存在**时（先跑过一次 ``init``，或上次迁移只做了一半）
    必须把 ``prompt_*`` 并进去，不能整份跳过：下面照样会把它们从姿态库里剥掉，
    于是标注两边都没留，而命令行还报「已迁移 1 条」。
    """
    from autolabel.library import ensure_lib_index, split_legacy_index

    root, rel, _batch = _make_legacy_dataset(str(tmp_path / "ds"))
    assert ensure_lib_index(root, "objects") is True
    keep = load_index(root, "objects")
    keep.upsert("keep-me.jpg", prompt_status="approved", prompt_objects=3)
    keep.save()

    moved, _stripped = split_legacy_index(root)

    assert moved == 1
    entry = objects_items(root)[rel]
    assert entry["prompt_status"] == "edited" and entry["prompt_objects"] == 2
    assert entry["prompt_flags"] == ["small_box"]
    assert "keep-me.jpg" in objects_items(root), "并进来不能把库里原有的记录挤掉"
    # 姿态库这边照旧剥干净，否则同样的键在两库里各留一份、以后不知以谁为准
    assert not any(key.startswith("prompt_") for key in load_index(root).get(rel))


def test_reading_a_library_backfills_the_index_the_old_dataset_never_had(dataset, capsys):
    """老数据集没有新加的库，网页一开就顺手补一本空的，命令行以前没这一步：
    同一个数据集网页用得好好的，``autolabel validate`` 却来一句"先执行 init"，
    像让人把标好的数据重做一遍。两边得是同一套行为。"""
    from autolabel.library import index_path

    detect_index = index_path(dataset, "detect")
    # 老布局的数据集就是没有这本账本，这里手动摘掉来还原那个年代的状态
    os.path.exists(detect_index) and os.remove(detect_index)
    assert not os.path.exists(detect_index), "老布局本来就没有目标检测库"

    index = load_index(dataset, "detect")
    assert index.items == {} and os.path.exists(detect_index), "读一次就该把这本空账本补上"
    # 补出来的那本得是真能用的账本，不能是个空壳
    backfilled = load_index(dataset, "detect")
    backfilled.upsert("img0.jpg", detect_objects=1)
    backfilled.save()
    assert load_index(dataset, "detect").get("img0.jpg")["detect_objects"] == 1

    capsys.readouterr()
    assert main(["status", dataset]) == 0
    main(["validate", dataset])
    report = capsys.readouterr().out + capsys.readouterr().err
    assert "先执行 init" not in report, "同一个数据集网页开得了，命令行不该当它是废目录"


def test_a_directory_that_is_not_a_dataset_still_says_init(tmp_path):
    """补齐只救"老数据集缺本新账本"，不能把随便一个目录认成数据集。"""

    from autolabel.library import index_path

    blank = tmp_path / "不是数据集"
    blank.mkdir()
    try:
        load_index(str(blank), "detect")
    except FileNotFoundError as exc:
        assert "先执行 init" in str(exc)
        assert not os.path.exists(index_path(str(blank), "detect"))
    else:
        raise AssertionError("空目录被当成数据集补齐了账本")
