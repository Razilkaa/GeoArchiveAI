from __future__ import annotations

import numpy as np
import pytest

from services.map_digitizer.georeference_grid import (
    fit_similarity,
    transform_xy,
)


def test_two_control_points_determine_reflected_similarity() -> None:
    controls = [
        {"pixel": [100.0, 200.0], "map": [500_000.0, 5_900_000.0]},
        {"pixel": [700.0, 900.0], "map": [504_200.0, 5_894_600.0]},
    ]

    matrix, quality = fit_similarity(controls)
    predicted = transform_xy(
        np.asarray([item["pixel"] for item in controls]),
        matrix,
    )

    assert predicted == pytest.approx(
        np.asarray([item["map"] for item in controls]),
        abs=1e-5,
    )
    assert np.linalg.det(matrix[:, :2]) < 0
    assert np.dot(matrix[:, 0], matrix[:, 1]) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(matrix[:, 0]) == pytest.approx(
        np.linalg.norm(matrix[:, 1])
    )
    assert quality["model"] == "two_point_similarity"
    assert quality["p95_m"] == pytest.approx(0.0, abs=1e-5)


def test_two_control_points_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        fit_similarity(
            [
                {"pixel": [10.0, 20.0], "map": [100.0, 200.0]},
                {"pixel": [10.0, 20.0], "map": [300.0, 400.0]},
            ]
        )
