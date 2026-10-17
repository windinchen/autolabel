import math

import pytest

from autolabel import yolo_pose as yp


def _person(cx=0.5, cy=0.5, w=0.2, h=0.4, n=17, v=2):
    return Person(cx=cx, cy=cy, w=w, h=h,
                  keypoints=[yp.Keypoint(cx, cy, v, 0.9) for _ in range(n)], score=0.7)


Person = yp.Person


def test_format_parse_roundtrip():
    person = Person(cls=0, cx=0.5, cy=0.25, w=0.2, h=0.5, score=0.8,
                    keypoints=[yp.Keypoint(0.1 * i / 17 + 0.1, 0.2 + i / 100, i % 3, 0.5)
                               for i in range(17)])
    line = yp.format_person(person)
    assert len(line.split()) == 5 + 17 * 3
    parsed = yp.parse_person(line)
    assert parsed.cls == 0
    assert math.isclose(parsed.cx, person.cx, abs_tol=1e-6)
    assert [k.v for k in parsed.keypoints] == [k.v for k in person.keypoints]
    assert math.isclose(parsed.keypoints[5].x, person.keypoints[5].x, abs_tol=1e-6)


def test_write_empty_label_is_background(tmp_path):
    path = tmp_path / "labels" / "bg.txt"
    yp.write_label_file(str(path), [])
    assert path.exists() and path.read_text() == ""
    assert yp.parse_label_file(str(path)) == []


def test_bbox_xyxy_roundtrip():
    person = Person(cx=0.5, cy=0.5, w=0.25, h=0.5)
    box = person.xyxy(800, 600)
    assert box == pytest.approx((300, 150, 500, 450))
    back = Person.from_xyxy(box, 800, 600)
    assert back.cx == pytest.approx(0.5) and back.w == pytest.approx(0.25)


def test_mirror_swaps_left_right_and_x():
    kps = [yp.Keypoint(0.1, 0.1, 2, 0.9) for _ in range(17)]
    kps[1] = yp.Keypoint(0.30, 0.2, 2, 0.9)   # left_eye
    kps[2] = yp.Keypoint(0.35, 0.2, 2, 0.8)   # right_eye
    person = Person(cx=0.4, cy=0.5, w=0.2, h=0.5, keypoints=kps)
    mirrored = yp.mirror_person(person, 1000, 1000)
    assert mirrored.cx == pytest.approx(0.6)
    # 反映射回原方向后，左右眼应互换且坐标复原
    assert mirrored.keypoints[1].x == pytest.approx(1 - 0.35)
    restored = yp.mirror_person(mirrored, 1000, 1000)
    assert restored.keypoints[1].x == pytest.approx(0.30)


def test_merge_flips_weights_by_confidence():
    a = Person(keypoints=[yp.Keypoint(0.40, 0.5, 2, 0.9)], score=0.8)
    b = Person(keypoints=[yp.Keypoint(0.50, 0.5, 2, 0.1)], score=0.3)
    merged = yp.merge_person_flips(a, b)
    assert merged.keypoints[0].x == pytest.approx(0.41, abs=1e-6)
    assert merged.score == pytest.approx(0.8)


def test_expand_bbox_covers_keypoints():
    person = Person(cx=0.5, cy=0.5, w=0.1, h=0.1)
    person.keypoints = [yp.Keypoint(0.1, 0.1, 2, 0.9), yp.Keypoint(0.9, 0.9, 2, 0.9)]
    healed = yp.expand_bbox_with_keypoints(person, 1000, 1000, margin_ratio=0.0)
    x1, y1, x2, y2 = healed.xyxy(1000, 1000)
    assert x1 <= 100 and x2 >= 900 and y1 <= 100 and y2 >= 900


def test_validate_detects_range_and_count_errors(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 1.5 0.5 0.2 0.4 " + "0.1 0.1 2 " * 17 + "\n", encoding="utf-8")
    issues = yp.validate_label_file(str(path), 800, 600, num_keypoints=17, class_names={0: "person"})
    assert any(i.kind == "bbox_range" for i in issues)

    path.write_text("0 0.5 0.5 0.2 0.4 " + "0.1 0.1 2 " * 16 + "\n", encoding="utf-8")
    issues = yp.validate_label_file(str(path), 800, 600, num_keypoints=17)
    assert any(i.kind == "parse_error" for i in issues)

    path.write_text("0 0.5 0.5 0.2 0.4 " + "0.1 0.1 5 " * 17 + "\n", encoding="utf-8")
    issues = yp.validate_label_file(str(path), 800, 600, num_keypoints=17)
    assert any(i.kind == "bad_visibility" for i in issues)


def test_sanity_flags():
    tiny = Person(cx=0.5, cy=0.5, w=0.01, h=0.02,
                  keypoints=[yp.Keypoint(0.5, 0.5, 2, 0.9) for _ in range(17)])
    assert "tiny_box" in yp.person_sanity_flags(tiny, 1000, 1000)

    few = Person(cx=0.5, cy=0.5, w=0.2, h=0.4, keypoints=[yp.Keypoint(0.5, 0.5, 0, 0.1) for _ in range(17)])
    assert "few_keypoints" in yp.person_sanity_flags(few, 1000, 1000)

    kps = [yp.Keypoint(0.5, 0.5, 2, 0.9) for _ in range(17)]
    kps[5], kps[7] = yp.Keypoint(0.45, 0.40, 2, 0.9), yp.Keypoint(0.45, 0.60, 2, 0.9)
    kps[6], kps[8] = yp.Keypoint(0.55, 0.40, 2, 0.9), yp.Keypoint(0.55, 0.42, 2, 0.9)
    lopsided = Person(cx=0.5, cy=0.5, w=0.3, h=0.6, keypoints=kps)
    assert "limb_length_mismatch" in yp.person_sanity_flags(lopsided, 1000, 1000)


def test_iou_and_match():
    assert yp.iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert yp.iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    pairs = yp.match_by_iou([(0, 0, 10, 10), (50, 50, 60, 60)], [(1, 1, 11, 11)])
    assert [p[:2] for p in pairs] == [(0, 0)]
    # 交集 9x9，并集 119
    assert pairs[0][2] == pytest.approx(81 / 119)


def test_visibility_constants_follow_coco_semantics():
    assert (yp.VIS_UNLABELED, yp.VIS_OCCLUDED, yp.VIS_VISIBLE) == (0, 1, 2)
    assert len(yp.COCO17_KEYPOINTS) == 17
    assert sorted(yp.COCO17_FLIP_IDX) == list(range(17))
