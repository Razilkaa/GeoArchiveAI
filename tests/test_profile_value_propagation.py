import numpy as np
import pytest
from shapely.geometry import LineString

from services.map_digitizer.profile_value_propagation import (
    propagate_profile_values,
)


def vertical_contours(xs, height=200.0):
    return {
        index: LineString([(x, 0.0), (x, height)]) for index, x in enumerate(xs)
    }


def horizontal_profile(x_min=0.0, x_max=600.0, y=100.0):
    return {
        "anchor": np.asarray([x_min, y]),
        "direction": np.asarray([1.0, 0.0]),
        "t_min": 0.0,
        "t_max": x_max - x_min,
        "support_px": x_max - x_min,
    }


def test_pure_counting_between_two_seeds():
    geometries = vertical_contours([100, 200, 300, 400, 500])
    result = propagate_profile_values(
        geometries,
        {0: -3.0, 4: -3.8},
        [horizontal_profile()],
        interval=0.2,
    )
    assert result["propagated_values"] == {1: -3.2, 2: -3.4, 3: -3.6}
    assert result["conflicts"] == {}


def test_depth_marks_alone_anchor_the_bracket():
    geometries = vertical_contours([100, 200, 300, 400, 500])
    marks = np.asarray([[50.0, 100.0, -2.9], [550.0, 100.0, -3.9]])
    result = propagate_profile_values(
        geometries,
        {},
        [horizontal_profile()],
        interval=0.2,
        depth_marks=marks,
    )
    assert result["propagated_values"] == {
        0: -3.0,
        1: -3.2,
        2: -3.4,
        3: -3.6,
        4: -3.8,
    }
    assert result["mark_anchors"] == 2


def test_fragmented_contour_counts_once_and_shares_the_value():
    # Contour 1 is split into two fragments crossing the profile 10 px apart.
    geometries = vertical_contours([100, 300, 500])
    geometries[3] = LineString([(305, 0), (305, 200)])
    result = propagate_profile_values(
        geometries,
        {0: -3.0, 2: -3.4},
        [horizontal_profile()],
        interval=0.2,
    )
    assert result["propagated_values"] == {1: -3.2, 3: -3.2}


def test_count_mismatch_without_predictions_assigns_nothing():
    # Two seeds one interval apart with a stray crossing between them.
    geometries = vertical_contours([100, 300, 500])
    result = propagate_profile_values(
        geometries,
        {0: -3.0, 2: -3.2},
        [horizontal_profile()],
        interval=0.2,
    )
    assert result["propagated_values"] == {}


def test_prediction_fallback_respects_monotonicity():
    # Bracket -3.0 .. -3.8 with only two crossings: predictions decide levels.
    geometries = vertical_contours([100, 250, 400, 500])
    del geometries[2]

    def predict(x, y):
        return -3.0 - (x - 100.0) / 400.0 * 0.9

    result = propagate_profile_values(
        geometries,
        {0: -3.0, 3: -3.8},
        [horizontal_profile()],
        interval=0.2,
        predict=predict,
    )
    assert result["propagated_values"] == {1: -3.4}


def test_conflicting_votes_are_refused():
    geometries = vertical_contours([100, 200, 300])
    profiles = [
        horizontal_profile(y=50.0),
        horizontal_profile(y=150.0),
    ]
    result = propagate_profile_values(
        geometries,
        {0: -3.0, 2: -3.4},
        profiles,
        interval=0.2,
    )
    assert result["propagated_values"] == {1: -3.2}
    # Marks on the second profile bracket the same contour at a different
    # level; a 1:1 vote split must refuse the assignment.
    marks = np.asarray([[150.0, 150.0, -3.65], [250.0, 150.0, -3.85]])
    conflicted = propagate_profile_values(
        geometries,
        {0: -3.0, 2: -3.4},
        profiles,
        interval=0.2,
        depth_marks=marks,
    )
    assert 1 not in conflicted["propagated_values"]
    assert 1 in conflicted["conflicts"]
