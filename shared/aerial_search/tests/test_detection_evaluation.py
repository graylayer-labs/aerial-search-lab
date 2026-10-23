"""Detection scorer checked against answers worked out by hand.

Conventions used in the working below:

- Boxes are (x1, y1, x2, y2) in continuous pixel coordinates, so a box's
  width is x2 - x1 and IoU = intersection area / union area.
- AP is COCO's 101-point interpolated AP: the mean, over recall levels
  r = 0.00, 0.01, ..., 1.00, of the highest precision reached at any recall
  >= r (0 when recall r is never reached).
- Recall at a false-alarm rate f is the best recall over score cutoffs whose
  false alarms divided by the number of images is at most f.
- Detections with equal scores: within an image they are matched in box
  coordinate order (x1, y1, x2, y2); on the precision-recall curve false
  alarms enter before hits.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from aerial_search.evaluation.detection import (
    GroundTruth,
    Prediction,
    evaluate_detections,
)

A = (0.0, 0.0, 10.0, 10.0)  # a 10 x 10 person
B = (100.0, 100.0, 110.0, 110.0)  # another 10 x 10 person, far from A


def truth(image_id, boxes, modality="rgb", flight="f1"):
    return GroundTruth(image_id=image_id, boxes=boxes, modality=modality, flight=flight)


def test_perfect_predictions_score_one() -> None:
    truths = [truth("i1", [A]), truth("i2", [B])]
    preds = [Prediction("i1", [A], [0.9]), Prediction("i2", [B], [0.8])]

    m = evaluate_detections(truths, preds).overall

    # Every detection is a true positive: precision 1 at every recall level.
    assert m.ap_iou25 == pytest.approx(1.0)
    assert m.ap_iou50 == pytest.approx(1.0)
    assert m.ap_coco == pytest.approx(1.0)
    # No false alarms, so every budget reaches full recall.
    assert m.recall_at_fppi == {0.01: 1.0, 0.1: 1.0, 1.0: 1.0}
    assert (m.n_images, m.n_ground_truth, m.n_predictions) == (2, 2, 2)
    assert m.n_predictions_tied == 0


def test_no_predictions_score_zero() -> None:
    # The empty prediction list also covers pycocotools' loadRes, which
    # crashes on an empty list.
    report = evaluate_detections([truth("i1", [A])], [])

    m = report.overall
    assert m.ap_iou25 == 0.0
    assert m.ap_iou50 == 0.0
    assert m.recall_at_fppi == {0.01: 0.0, 0.1: 0.0, 1.0: 0.0}
    assert m.n_predictions == 0


def test_detections_on_image_without_people_are_false_alarms() -> None:
    truths = [truth("people", [A]), truth("empty", [])]
    # The false alarm on the empty image outscores the true detection.
    preds = [
        Prediction("empty", [A], [0.95]),
        Prediction("people", [A], [0.9]),
    ]

    m = evaluate_detections(truths, preds).overall

    # Ranked: FP (0.95), TP (0.9); 1 person.
    # recall = [0, 1], precision = [0, 1/2]; envelope = [1/2, 1/2].
    # Every recall level r in [0, 1] gets precision 1/2, so AP = 0.5.
    assert m.ap_iou25 == pytest.approx(0.5)
    # 2 images. Cutoffs: none -> (recall 0, FPPI 0); >=0.95 -> (0, 1/2);
    # >=0.9 -> (1, 1/2). Budgets 0.01 and 0.1 allow only the empty cutoff.
    assert m.recall_at_fppi == {0.01: 0.0, 0.1: 0.0, 1.0: 1.0}
    assert m.n_images_with_people == 1


def test_only_images_without_people_leave_ap_undefined() -> None:
    report = evaluate_detections(
        [truth("empty", [])], [Prediction("empty", [A], [0.5])]
    )

    m = report.overall
    # With no people, recall has no denominator: AP and recall are undefined.
    assert math.isnan(m.ap_iou25)
    assert math.isnan(m.ap_coco)
    assert all(math.isnan(v) for v in m.recall_at_fppi.values())
    assert (m.n_images, m.n_ground_truth, m.n_predictions) == (1, 0, 1)


def test_duplicate_detection_of_one_person_is_a_false_alarm() -> None:
    truths = [truth("i1", [A, B])]
    preds = [Prediction("i1", [A, A, B], [0.9, 0.8, 0.7])]

    m = evaluate_detections(truths, preds).overall

    # Ranked: A (0.9) TP, A again (0.8) FP because A is taken, B (0.7) TP.
    # recall = [1/2, 1/2, 1], precision = [1, 1/2, 2/3]
    # envelope = [1, 2/3, 2/3]
    # r = 0.00..0.50 (51 levels) -> 1; r = 0.51..1.00 (50 levels) -> 2/3
    # AP = (51 + 50 * 2/3) / 101 = 253 / 303
    assert m.ap_iou25 == pytest.approx(253 / 303)
    # 1 image. Cutoffs: 0.9 -> (1/2, 0); 0.8 -> (1/2, 1); 0.7 -> (1, 1).
    assert m.recall_at_fppi == {0.01: 0.5, 0.1: 0.5, 1.0: 1.0}


def test_primary_threshold_accepts_shift_that_iou_half_rejects() -> None:
    # A 12 x 12 person and a box shifted 3 px right and 3 px down.
    # Intersection 9 * 9 = 81; union 144 + 144 - 81 = 207; IoU = 0.391.
    truths = [truth("i1", [(0.0, 0.0, 12.0, 12.0)])]
    preds = [Prediction("i1", [(3.0, 3.0, 15.0, 15.0)], [0.9])]

    m = evaluate_detections(truths, preds).overall

    assert m.ap_iou25 == pytest.approx(1.0)  # 0.391 >= 0.25: a hit
    assert m.ap_iou50 == 0.0  # 0.391 < 0.5: a miss and a false alarm
    assert m.ap_coco == 0.0
    assert m.recall_at_fppi[1.0] == 1.0


def test_iou_equal_to_threshold_counts_as_match() -> None:
    # Box twice as tall: intersection 100, union 200, IoU exactly 0.5.
    truths = [truth("i1", [A])]
    preds = [Prediction("i1", [(0.0, 0.0, 10.0, 20.0)], [0.9])]

    m = evaluate_detections(truths, preds).overall

    assert m.ap_iou50 == pytest.approx(1.0)
    # COCO AP averages thresholds 0.50, 0.55, ..., 0.95; only 0.50 hits.
    assert m.ap_coco == pytest.approx(0.1)


def test_detection_matches_the_overlapping_person_it_fits_best() -> None:
    # Two overlapping 10 x 10 people, P at x 0-10 and Q at x 4-14.
    p, q = (0.0, 0.0, 10.0, 10.0), (4.0, 0.0, 14.0, 10.0)
    first = (3.0, 0.0, 13.0, 10.0)  # IoU with P = 70/130 = 0.54, Q = 90/110 = 0.82
    # `first` must take Q. If it took P, `second` (exactly P) would be left
    # with Q at IoU 60/140 = 0.43 < 0.5 and become a false alarm.
    preds = [Prediction("i1", [first, p], [0.9, 0.8])]

    m = evaluate_detections([truth("i1", [p, q])], preds).overall

    assert m.ap_iou50 == pytest.approx(1.0)


def test_small_multi_image_case_and_breakdowns() -> None:
    truths = [
        truth("i1", [A, B], modality="rgb", flight="fA"),
        truth("i2", [(20.0, 20.0, 30.0, 30.0)], modality="thermal", flight="fA"),
        truth("i3", [], modality="rgb", flight="fB"),
    ]
    preds = [
        Prediction("i1", [A, (200.0, 200.0, 210.0, 210.0)], [0.95, 0.6]),
        Prediction("i2", [(20.0, 20.0, 30.0, 30.0)], [0.8]),
        Prediction("i3", [(5.0, 5.0, 15.0, 15.0)], [0.85]),
    ]

    report = evaluate_detections(truths, preds)

    # Overall, 3 people. Ranked: 0.95 TP, 0.85 FP, 0.8 TP, 0.6 FP (B missed).
    # recall = [1/3, 1/3, 2/3, 2/3], precision = [1, 1/2, 2/3, 1/2]
    # envelope = [1, 2/3, 2/3, 1/2]
    # r = 0.00..0.33 (34 levels) -> 1; r = 0.34..0.66 (33 levels) -> 2/3;
    # r = 0.67..1.00 never reached -> 0.  AP = (34 + 22) / 101 = 56/101
    o = report.overall
    assert o.ap_iou25 == pytest.approx(56 / 101)
    # 3 images. Cutoffs: 0.95 -> (1/3, 0); 0.85 -> (1/3, 1/3);
    # 0.8 -> (2/3, 1/3); 0.6 -> (2/3, 2/3).
    assert o.recall_at_fppi == pytest.approx({0.01: 1 / 3, 0.1: 1 / 3, 1.0: 2 / 3})

    # RGB (i1, i3), 2 people. Ranked: 0.95 TP, 0.85 FP, 0.6 FP.
    # recall = [1/2, 1/2, 1/2]; envelope = [1, 1/2, 1/3]
    # r = 0.00..0.50 (51 levels) -> 1; above 0.5 never reached. AP = 51/101
    rgb = report.by_modality["rgb"]
    assert rgb.ap_iou25 == pytest.approx(51 / 101)
    # 2 images. Cutoffs: 0.95 -> (1/2, 0); 0.85 -> (1/2, 1/2); 0.6 -> (1/2, 1).
    assert rgb.recall_at_fppi == pytest.approx({0.01: 0.5, 0.1: 0.5, 1.0: 0.5})
    assert report.by_modality["thermal"].ap_iou25 == pytest.approx(1.0)

    # Flight fA (i1, i2), 3 people. Ranked: 0.95 TP, 0.8 TP, 0.6 FP.
    # recall = [1/3, 2/3, 2/3]; envelope = [1, 1, 2/3]
    # r = 0.00..0.66 (67 levels) -> 1; above 2/3 never reached. AP = 67/101
    fa = report.by_flight["fA"]
    assert fa.ap_iou25 == pytest.approx(67 / 101)
    # 2 images. Cutoffs: 0.95 -> (1/3, 0); 0.8 -> (2/3, 0); 0.6 -> (2/3, 1/2).
    assert fa.recall_at_fppi == pytest.approx({0.01: 2 / 3, 0.1: 2 / 3, 1.0: 2 / 3})
    # Flight fB has no people: AP undefined, but its detection is counted.
    fb = report.by_flight["fB"]
    assert math.isnan(fb.ap_iou25)
    assert (fb.n_images, fb.n_ground_truth, fb.n_predictions) == (1, 0, 1)


def test_recall_of_exactly_seven_tenths_reaches_level_070() -> None:
    people = [(20.0 * k, 0.0, 20.0 * k + 10.0, 10.0) for k in range(10)]
    preds = [Prediction("i1", people[:7], [0.9] * 7)]

    m = evaluate_detections([truth("i1", people)], preds).overall

    # 7 hits of 10, no false alarms: precision 1 up to recall 0.7.
    # Levels 0.00..0.70 (71 levels) -> 1; the rest are never reached.
    # AP = 71/101. (pycocotools' default grid puts level 0.70 one ulp above
    # 0.7 and would give 70/101.)
    assert m.ap_iou25 == pytest.approx(71 / 101)


def test_size_buckets_are_half_open_on_sqrt_area() -> None:
    eight = (0.0, 0.0, 8.0, 8.0)  # sqrt(area) = 8: in [8, 16), not [0, 8)
    six = (20.0, 20.0, 26.0, 26.0)  # sqrt(area) = 6: in [0, 8)
    truths = [truth("i1", [eight, six])]
    preds = [Prediction("i1", [eight], [0.9])]

    by_size = evaluate_detections(truths, preds).by_size

    assert by_size["tiny"].n_ground_truth == 1
    assert by_size["tiny"].ap_iou25 == pytest.approx(1.0)
    # The one very tiny person is missed; the detection belongs to the
    # 8 x 8 person, so it is neither a hit nor a false alarm here.
    assert by_size["very_tiny"].n_ground_truth == 1
    assert by_size["very_tiny"].ap_iou25 == 0.0
    assert by_size["very_tiny"].recall_at_fppi == {0.01: 0.0, 0.1: 0.0, 1.0: 0.0}
    assert by_size["small"].n_ground_truth == 0
    assert math.isnan(by_size["small"].ap_iou25)


def test_tied_scores_form_one_operating_point() -> None:
    # A hit and a false alarm with the same score cannot be separated by any
    # threshold, whichever order they arrive in.
    far = (50.0, 50.0, 60.0, 60.0)
    for boxes in ([A, far], [far, A]):
        preds = [Prediction("i1", boxes, [0.5, 0.5])]
        m = evaluate_detections([truth("i1", [A])], preds).overall
        # 1 image. The only non-empty cutoff is 0.5 -> (recall 1, FPPI 1).
        assert m.recall_at_fppi == {0.01: 0.0, 0.1: 0.0, 1.0: 1.0}


def test_size_bucket_recall_counts_false_alarms_of_every_size() -> None:
    # Reviewer's case: a 6 x 6 person found at 0.5, a 20 x 20 false alarm
    # at 0.9. A searcher sees every false alarm, whatever its size.
    tiny_person = (0.0, 0.0, 6.0, 6.0)
    big_false_alarm = (100.0, 100.0, 120.0, 120.0)
    preds = [Prediction("i1", [big_false_alarm, tiny_person], [0.9, 0.5])]

    report = evaluate_detections([truth("i1", [tiny_person])], preds)

    # 1 image. Cutoffs: 0.9 -> (recall 0, FPPI 1); 0.5 -> (1, 1).
    expected = {0.01: 0.0, 0.1: 0.0, 1.0: 1.0}
    assert report.overall.recall_at_fppi == expected
    very_tiny = report.by_size["very_tiny"]
    assert very_tiny.recall_at_fppi == expected
    # AP per size keeps the COCO rule: an unmatched detection of another size
    # is ignored in the bucket, so the bucket's AP is 1.
    assert very_tiny.ap_iou25 == pytest.approx(1.0)


def test_false_alarm_rate_divides_by_all_images() -> None:
    # One image with a person, nine without. Two false alarms at 0.9 on empty
    # images, one hit at 0.5.
    truths = [truth("p", [A])] + [truth(f"e{k}", []) for k in range(9)]
    preds = [
        Prediction("p", [A], [0.5]),
        Prediction("e0", [B], [0.9]),
        Prediction("e1", [B], [0.9]),
    ]

    m = evaluate_detections(truths, preds).overall

    # 10 images. Cutoffs: 0.9 -> (recall 0, FPPI 2/10); 0.5 -> (1, 2/10).
    # Dividing by images with people (1) would give FPPI 2 and recall 0 at 1.
    assert m.recall_at_fppi == {0.01: 0.0, 0.1: 0.0, 1.0: 1.0}


def test_one_detection_can_be_a_hit_in_two_size_buckets() -> None:
    # A 10 x 10 person with a 6 x 6 person inside, one detection equal to the
    # larger. IoU with the larger = 1; with the smaller = 36/100 = 0.36.
    big, small = (0.0, 0.0, 10.0, 10.0), (2.0, 2.0, 8.0, 8.0)
    preds = [Prediction("i1", [big], [0.9])]

    report = evaluate_detections([truth("i1", [big, small])], preds)

    # Overall the detection takes the larger person: recall 1/2, precision 1.
    # Levels 0.00..0.50 (51 levels) -> 1. AP = 51/101.
    assert report.overall.ap_iou25 == pytest.approx(51 / 101)
    # In each bucket the other person is ignored, so the same detection is
    # a hit in both (the COCO rule): per-size figures do not add up.
    assert report.by_size["tiny"].ap_iou25 == pytest.approx(1.0)
    assert report.by_size["very_tiny"].ap_iou25 == pytest.approx(1.0)


def test_tied_detections_in_one_image_match_in_coordinate_order() -> None:
    # Reviewer's within-image case, in this geometry: people P and Q, two
    # detections both at 0.7.
    p, q = (0.0, 0.0, 10.0, 10.0), (8.0, 0.0, 18.0, 10.0)
    near = (3.0, 0.0, 13.0, 10.0)  # IoU with P = 70/130 = 0.54, Q = 50/150 = 0.33
    # `p` (x1 = 0) is matched before `near` (x1 = 3), whatever the input
    # order: `p` takes P, then `near` takes Q. Two hits, no false alarm.
    # (Taking `near` first would give it P and leave `p` with Q at IoU
    # 20/180 = 0.11, a false alarm.)
    for boxes in ([near, p], [p, near]):
        preds = [Prediction("i1", boxes, [0.7, 0.7])]
        m = evaluate_detections([truth("i1", [p, q])], preds).overall
        assert m.ap_iou25 == pytest.approx(1.0)
        assert m.recall_at_fppi == {0.01: 1.0, 0.1: 1.0, 1.0: 1.0}
        assert m.n_predictions_tied == 2


def test_tied_hit_and_false_alarm_across_images_count_the_false_alarm_first() -> None:
    # Reviewer's across-image case: a hit on one image and a false alarm on
    # an empty image, both at score 1.0 (a saturated float32 sigmoid).
    for order in (["a", "b"], ["b", "a"]):
        images = {"a": truth("a", [A]), "b": truth("b", [])}
        preds = [Prediction("a", [A], [1.0]), Prediction("b", [A], [1.0])]

        m = evaluate_detections([images[k] for k in order], preds).overall

        # The lower-scoring order: FP then TP. recall = [0, 1],
        # precision = [0, 1/2], envelope = [1/2, 1/2]: AP = 0.5.
        # (Stock pycocotools gives 1.0 when image "a" is listed first.)
        assert m.ap_iou25 == pytest.approx(0.5)
        # 2 images, one cutoff at 1.0 -> (recall 1, FPPI 1/2).
        assert m.recall_at_fppi == {0.01: 0.0, 0.1: 0.0, 1.0: 1.0}
        assert m.n_predictions_tied == 2


def test_equal_overlap_with_two_people_does_not_depend_on_their_order() -> None:
    # People P1 = (0,0,10,10) and P2 = (6,0,16,10), each 10 x 10.
    # D1 = (3,0,13,10) at 0.9 overlaps each by 7 x 10 = 70; union 130; IoU 0.54.
    # D2 = (8,0,18,10) at 0.8: with P1 overlap 20, union 180, IoU 0.11 (a miss);
    #   with P2 overlap 80, union 120, IoU 0.67.
    # People are matched in coordinate order, and an equal overlap goes to the
    # later one, so D1 takes P2. D2 then has only P1 left and is a false alarm.
    # One of two people found with no false alarm ranked above: precision 1
    # up to recall 0.5, so AP = 51/101. Recall is 0.5 at every budget.
    p1, p2 = (0.0, 0.0, 10.0, 10.0), (6.0, 0.0, 16.0, 10.0)
    predictions = [
        Prediction("a", [(3.0, 0.0, 13.0, 10.0), (8.0, 0.0, 18.0, 10.0)], [0.9, 0.8])
    ]
    for people in ([p1, p2], [p2, p1]):
        report = evaluate_detections(
            [GroundTruth("a", people, "rgb", "f1")], predictions
        )
        assert report.overall.ap_iou25 == pytest.approx(51 / 101)
        assert report.overall.recall_at_fppi == {0.01: 0.5, 0.1: 0.5, 1.0: 0.5}


def test_tied_count_in_a_size_bucket_covers_only_that_bucket() -> None:
    # A 10 x 10 detection (bucket "tiny") and a 40 x 40 detection (bucket
    # "medium_plus") share the score 0.7. Overall, both are tied. Within each
    # bucket there is one detection, which has nothing to tie with.
    big = (100.0, 100.0, 140.0, 140.0)
    report = evaluate_detections(
        [GroundTruth("a", [A, big], "rgb", "f1")],
        [Prediction("a", [A, big], [0.7, 0.7])],
    )
    assert report.overall.n_predictions_tied == 2
    assert report.by_size["tiny"].n_predictions_tied == 0
    assert report.by_size["medium_plus"].n_predictions_tied == 0


def test_results_do_not_depend_on_input_order() -> None:
    p, q = (0.0, 0.0, 10.0, 10.0), (8.0, 0.0, 18.0, 10.0)
    near = (3.0, 0.0, 13.0, 10.0)
    truths = [
        truth("i1", [p, q, B], modality="rgb", flight="fA"),
        truth("i2", [A], modality="thermal", flight="fA"),
        truth("i3", [], modality="rgb", flight="fB"),
        truth("i4", [(0.0, 0.0, 6.0, 6.0), B], modality="thermal", flight="fB"),
    ]
    preds = [
        Prediction("i1", [near, p, B, (50.0, 50.0, 60.0, 60.0)], [1.0, 1.0, 0.6, 0.6]),
        Prediction("i2", [A, A], [1.0, 0.6]),
        Prediction("i3", [A, (30.0, 30.0, 50.0, 50.0)], [1.0, 0.6]),
        Prediction(
            "i4", [(1.0, 1.0, 7.0, 7.0), (100.0, 100.0, 120.0, 120.0)], [0.6, 1.0]
        ),
    ]
    baseline = evaluate_detections(truths, preds).to_dict()

    rng = random.Random(0)
    for _ in range(8):
        shuffled_truths = []
        for t in rng.sample(truths, len(truths)):
            boxes = np.asarray(t.boxes).tolist()
            shuffled = rng.sample(boxes, len(boxes))
            shuffled_truths.append(truth(t.image_id, shuffled, t.modality, t.flight))
        shuffled_preds = []
        for pr in rng.sample(preds, len(preds)):
            boxes, scores = np.asarray(pr.boxes), np.asarray(pr.scores)
            order = rng.sample(range(len(scores)), len(scores))
            shuffled_preds.append(Prediction(pr.image_id, boxes[order], scores[order]))
        assert (
            evaluate_detections(shuffled_truths, shuffled_preds).to_dict() == baseline
        )


def test_rejects_predictions_for_unknown_image() -> None:
    with pytest.raises(ValueError, match="not in the ground truth"):
        evaluate_detections([truth("i1", [A])], [Prediction("i9", [A], [0.5])])


def test_rejects_duplicate_image_ids() -> None:
    with pytest.raises(ValueError, match="more than once"):
        evaluate_detections([truth("i1", [A]), truth("i1", [B])], [])


@pytest.mark.parametrize(
    "box",
    [(5.0, 0.0, 5.0, 10.0), (10.0, 0.0, 0.0, 10.0), (0.0, 0.0, math.nan, 10.0)],
)
def test_rejects_degenerate_boxes(box) -> None:
    with pytest.raises(ValueError, match="box"):
        truth("i1", [box])


def test_rejects_score_count_mismatch() -> None:
    with pytest.raises(ValueError, match="scores"):
        Prediction("i1", [A, B], [0.5])


def test_report_writes_json_with_null_for_undefined(tmp_path: Path) -> None:
    truths = [truth("i1", [A], flight="fA"), truth("i2", [], flight="fB")]
    report = evaluate_detections(truths, [Prediction("i1", [A], [0.9])])

    path = report.write_json(tmp_path / "run-1")

    assert path == tmp_path / "run-1" / "detection_metrics.json"

    def reject(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant {constant}")

    data = json.loads(path.read_text(), parse_constant=reject)
    assert data["overall"]["ap_iou25"] == pytest.approx(1.0)
    assert data["by_flight"]["fB"]["ap_iou25"] is None
    assert data["overall"]["recall_at_fppi"]["0.1"] == 1.0
    assert data["config"]["primary_iou"] == 0.25
