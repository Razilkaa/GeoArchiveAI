from __future__ import annotations

import pandas as pd

from scripts.build_efgi_requisition_queue import build_requisition_queue


def test_builds_one_old_candidate_per_inventory() -> None:
    frame = pd.DataFrame(
        {
            "n_uk_rosg": ["54", "54", "30", "99"],
            "in_n_rosg": ["344136", "344136", "344284", "500000"],
            "god_nach": ["1974", "1974", None, "2001"],
            "god_end": ["1975", "1975", "1975", "2002"],
            "name_otch": ["34413", "34413", "34428", "new"],
            "uk_id": ["14390011", "14390011", "13390134", "new"],
        }
    )

    queue = build_requisition_queue(frame)

    assert queue["inventory_id"].tolist() == [344136, 344284]
    assert queue["report_year"].tolist() == [1975, 1975]
    assert queue["source_features"].tolist() == [2, 1]


def test_uses_start_year_when_end_year_is_missing() -> None:
    frame = pd.DataFrame(
        {
            "n_uk_rosg": ["1"],
            "in_n_rosg": ["400001"],
            "god_nach": ["1988"],
            "god_end": [None],
            "name_otch": ["report"],
            "uk_id": ["1"],
        }
    )

    queue = build_requisition_queue(frame)

    assert queue.loc[0, "report_year"] == 1988
