from __future__ import annotations

import pandas as pd

from scripts.extract_efgi_candidates import extract_candidate_reports


def test_extracts_old_reports_and_merges_profile_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "N_RGF": [428786, 428786, 500000, 0],
            "INV_svod": ["", "", "", "15131\u0438\u0440"],
            "\u0413\u043e\u0434_\u0441": ["1986", "1986", "2008", "1992"],
            "\u041d\u0430\u0437\u0432\u0430": ["Old report", "Old report", "New report", "Fallback"],
            "\u0410\u0432\u0442\u043e\u0440": ["Author", "Author", "", ""],
            "\u041e\u0440\u0433\u0430\u043d": ["Org", "Org", "", ""],
            "\u041c\u0435\u0441\u0442\u043e": ["Fund", "Fund", "", ""],
            "\u041d\u0430\u043b\u0438\u0447": [
                "сканобразы (частично)",
                "сканобразы (частично)",
                "",
                "сканобразы каталожных карточек",
            ],
            "\u0414\u043e\u0441\u0442\u0443": [
                "https://efgi.ru/object/1",
                "https://efgi.ru/object/1",
                "",
                "https://efgi.ru/object/2",
            ],
            "link": ["", "", "", ""],
        }
    )

    result = extract_candidate_reports(frame)

    assert result["inventory_id"].tolist() == [428786, 15131]
    assert result.loc[0, "likely_downloadable"]
    assert not result.loc[1, "has_digital_copy"]


def test_rejects_missing_required_fields() -> None:
    frame = pd.DataFrame({"N_RGF": [1]})

    try:
        extract_candidate_reports(frame)
    except ValueError as error:
        assert "missing required fields" in str(error)
    else:
        raise AssertionError("Expected a missing-field ValueError")
