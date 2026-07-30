import numpy as np

from services.map_digitizer.survey_profiles import (
    _profile_labels,
    profile_number_variants,
)


def test_volga_ural_profile_number_order_is_normalized() -> None:
    assert profile_number_variants("040887") == ("040887", "088704")

    labels = _profile_labels(
        {
            "lines": [
                {
                    "text": "04-0887",
                    "score": 0.99,
                    "polygon": [[0, 0], [20, 0], [20, 10], [0, 10]],
                }
            ]
        },
        {"088704"},
        1.0,
    )

    assert len(labels) == 1
    assert labels[0]["profile_id"] == "088704"
    assert np.allclose(labels[0]["center"], [10.0, 5.0])
