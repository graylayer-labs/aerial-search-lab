"""Tiling (#96): the grid covers the frame, boxes round-trip, overlaps merge."""

import numpy as np
import pytest

from aerial_search.data import tiling


@pytest.mark.parametrize(
    "width,height,tile,overlap",
    [(1920, 1080, (672, 384), 160), (3840, 2160, (672, 384), 160)]
    + [
        (640, 512, (560, 448), 96),
        (700, 400, (672, 384), 0),
        (672, 384, (672, 384), 160),
    ],
)
def test_grid_covers_every_pixel_including_right_and_bottom_edges(
    width, height, tile, overlap
):
    tiles = tiling.tile_grid(width, height, tile, overlap)
    covered = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert (x1 - x0, y1 - y0) == tile
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    # Adjacent tiles overlap by at least `overlap` (the last one by more).
    xs = sorted({t[0] for t in tiles})
    assert all(b - a <= tile[0] - overlap for a, b in zip(xs, xs[1:], strict=False))


def test_grid_for_a_frame_smaller_than_a_tile_is_the_whole_frame():
    assert tiling.tile_grid(300, 200, (672, 384), 160) == [(0, 0, 300, 200)]


def test_overlap_must_be_smaller_than_the_tile():
    with pytest.raises(ValueError):
        tiling.tile_grid(1920, 1080, (672, 384), 384)


def test_tile_box_round_trip_to_source_pixels():
    tile = (512, 224, 1184, 608)
    source = np.array([[600.0, 300.0, 640.0, 380.0]])
    local, _ = tiling.boxes_in_tile(source, tile, min_visible=0.5)
    assert local.tolist() == [[88.0, 76.0, 128.0, 156.0]]
    assert tiling.to_source(local, tile).tolist() == source.tolist()


def test_box_cut_by_a_tile_edge_is_kept_if_enough_is_visible_and_clipped():
    tile = (0, 0, 100, 100)
    boxes = np.array(
        [
            [80.0, 10.0, 120.0, 50.0],  # 50% visible: kept, clipped to x 100
            [90.0, 10.0, 130.0, 50.0],  # 25% visible: dropped
            [10.0, 10.0, 30.0, 30.0],  # inside
            [150.0, 10.0, 170.0, 30.0],  # outside
        ]
    )
    local, kept = tiling.boxes_in_tile(boxes, tile, min_visible=0.5)
    assert kept.tolist() == [0, 2]
    assert local.tolist() == [[80.0, 10.0, 100.0, 50.0], [10.0, 10.0, 30.0, 30.0]]


def test_a_person_in_an_overlap_is_counted_once_after_merging():
    # The same person seen whole by two tiles (slightly different boxes), a
    # clipped sliver of them in a third, and a different person far away.
    boxes = np.array(
        [
            [600.0, 300.0, 640.0, 380.0],
            [601.0, 301.0, 641.0, 379.0],
            [600.0, 300.0, 630.0, 380.0],
            [100.0, 100.0, 140.0, 180.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.3, 0.7])
    merged, merged_scores = tiling.merge_detections(boxes, scores, iou=0.5)
    assert merged.shape == (2, 4)
    assert merged_scores.tolist() == [0.9, 0.7]


def test_two_adjacent_people_are_not_merged():
    boxes = np.array([[0.0, 0.0, 10.0, 20.0], [12.0, 0.0, 22.0, 20.0]])
    merged, _ = tiling.merge_detections(boxes, np.array([0.9, 0.8]), iou=0.5)
    assert len(merged) == 2


def test_merging_nothing_gives_nothing():
    merged, scores = tiling.merge_detections(np.zeros((0, 4)), np.zeros(0), iou=0.5)
    assert merged.shape == (0, 4) and scores.shape == (0,)


def test_empty_tiles_are_subsampled_to_a_ratio_of_the_positive_tiles():
    has_person = [True] * 10 + [False] * 90
    picks = tiling.pick_tiles(has_person, empty_ratio=0.5, seed=3)
    assert [i for i in picks if has_person[i]] == list(range(10))
    assert sum(1 for i in picks if not has_person[i]) == 5
    assert picks == tiling.pick_tiles(has_person, empty_ratio=0.5, seed=3)
    assert picks == sorted(picks)


def test_a_split_with_no_person_still_gets_some_empty_tiles():
    picks = tiling.pick_tiles([False] * 20, empty_ratio=0.5, seed=0)
    assert 0 < len(picks) <= 20
