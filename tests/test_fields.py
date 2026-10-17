"""字段登记表（``.autolabel/fields.json``）的行为约束。"""
import json
import os

import pytest

from autolabel import fields as fd
from autolabel.dataset import label_path_for, prompt_label_path_for


def label(root, rel):
    return prompt_label_path_for(root, rel)


def test_upsert_assigns_class_id_and_persists(dataset):
    saved = fd.upsert_field(dataset, name="bottle", label="瓶子", prompt_hint="桌上或手里的水瓶")
    assert saved["class_id"] == 0
    assert saved["name"] == "bottle" and saved["label"] == "瓶子"
    assert saved["color"].startswith("#")
    on_disk = json.loads(open(fd.fields_path(dataset), encoding="utf-8").read())
    assert [f["name"] for f in on_disk["fields"]] == ["bottle"]
    assert on_disk["next_class_id"] == 1


def test_upsert_updates_same_name_keeps_class_id(dataset):
    first = fd.upsert_field(dataset, name="bottle", label="瓶子")
    second = fd.upsert_field(dataset, name="BOTTLE", label="水瓶", prompt_hint="换个描述")
    assert first["id"] == second["id"]
    assert second["class_id"] == first["class_id"]
    assert len(fd.list_fields(dataset)) == 1
    assert fd.find_field(dataset, "bottle")["label"] == "水瓶"


@pytest.mark.parametrize("raw", ["2bad", "有中文", "with space", "", "-", "a" * 33])
def test_normalize_name_rejects_bad_names(dataset, raw):
    with pytest.raises(ValueError):
        fd.upsert_field(dataset, name=raw)


def test_normalize_name_accepts_snake_and_dash(dataset):
    assert fd.normalize_name("  Water-Bottle_2 ") == "Water-Bottle_2"


def test_hint_length_is_rejected_not_truncated(dataset):
    saved = fd.upsert_field(dataset, name="bottle", prompt_hint="描述" * 300)
    assert len(saved["prompt_hint"]) == fd.MAX_HINT_LEN
    with pytest.raises(ValueError):
        fd.upsert_field(dataset, name="backpack", prompt_hint="描述" * 400)


def test_delete_field_without_boxes_is_allowed(dataset):
    fd.upsert_field(dataset, name="bottle")
    result = fd.delete_field(dataset, "bottle")
    assert result["name"] == "bottle"
    assert fd.list_fields(dataset) == []


def test_delete_field_protects_existing_boxes(dataset):
    fd.upsert_field(dataset, name="bottle")
    class_id = fd.list_fields(dataset)[0]["class_id"]
    fd.write_label_rows(label(dataset, "img0.jpg"), [[class_id, 0.5, 0.5, 0.2, 0.3]])
    with pytest.raises(ValueError) as exc:
        fd.delete_field(dataset, "bottle")
    assert "框" in str(exc.value)
    assert fd.class_id_usage(dataset) == {class_id: 1}
    dropped = fd.delete_field(dataset, "bottle", drop_boxes=True)
    assert dropped["boxes_removed"] == 1
    assert fd.class_id_usage(dataset) == {}


def test_class_id_is_never_reused(dataset):
    fd.upsert_field(dataset, name="bottle")
    fd.upsert_field(dataset, name="backpack")
    fd.delete_field(dataset, "bottle")
    third = fd.upsert_field(dataset, name="helmet")
    assert third["class_id"] == 2


def test_orphan_class_ids_and_payload_counts(dataset):
    fd.upsert_field(dataset, name="bottle")
    path = label(dataset, "img0.jpg")
    fd.write_label_rows(path, [[0, 0.5, 0.5, 0.2, 0.3], [0, 0.2, 0.2, 0.1, 0.1], [7, 0.1, 0.1, 0.1, 0.1]])
    payload = fd.fields_payload(dataset)
    assert payload["fields"][0]["boxes"] == 2
    assert payload["orphans"] == [7]


def test_label_rows_roundtrip_is_five_columns(dataset):
    rows = [[0, 0.5, 0.25, 0.123456789, 0.3], [3, 0.1, 0.2, 0.05, 0.06]]
    path = label(dataset, "img0.jpg")
    fd.write_label_rows(path, rows)
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert all(len(line.split()) == 5 for line in lines)
    back = fd.read_label_rows(path)
    assert back[0][0] == 0 and abs(back[0][4] - 0.3) < 1e-9
    assert abs(back[1][3] - 0.05) < 1e-6


def test_read_label_rows_skips_junk_lines(dataset):
    path = label(dataset, "img0.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("# 注释\n\n0 0.5 0.5 0.2\nnope nope nope nope nope\n1 0.1 0.1 0.1 0.1\n")
    assert fd.read_label_rows(path) == [(1, 0.1, 0.1, 0.1, 0.1)]


def test_resolve_fields_filters_and_errors(dataset):
    fd.upsert_field(dataset, name="bottle")
    fd.upsert_field(dataset, name="backpack")
    assert [f["name"] for f in fd.resolve_fields(dataset, ["backpack"])] == ["backpack"]
    assert len(fd.resolve_fields(dataset, [])) == 2
    with pytest.raises(ValueError):
        fd.resolve_fields(dataset, ["nonexistent"])


def test_prompt_labels_dir_is_separate_from_pose_labels(dataset):
    """提示词检测框与姿态骨骼各写各的目录，互不覆盖。"""
    prompt = label(dataset, "img0.jpg")
    pose = label_path_for(dataset, "img0.jpg")
    os.makedirs(os.path.dirname(pose), exist_ok=True)
    open(pose, "w", encoding="utf-8").write("0 0.5 0.5 0.2 0.3 0.1 0.1 2\n")
    fd.write_label_rows(prompt, [[0, 0.5, 0.5, 0.2, 0.3]])
    assert os.path.basename(prompt) == os.path.basename(pose) == "img0.txt"
    assert prompt != pose
    assert open(pose, encoding="utf-8").read().strip().split() == [
        "0", "0.5", "0.5", "0.2", "0.3", "0.1", "0.1", "2"]
    assert fd.read_label_rows(prompt) == [(0, 0.5, 0.5, 0.2, 0.3)]
