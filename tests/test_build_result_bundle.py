import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from build_result_bundle import enrich_georeference_packages, entity_groups


def test_entity_groups_derives_unique_horizons() -> None:
    agents = {
        "structures": {
            "result": {
                "items": [
                    {
                        "name": "Северная",
                        "horizons": ["К", "P2"],
                        "evidence": ["page:00021"],
                    }
                ]
            }
        },
        "maps": {
            "result": {
                "items": [
                    {
                        "title": "Структурная карта",
                        "horizons": "К",
                        "evidence": ["page:00190"],
                    }
                ]
            }
        },
    }

    groups = entity_groups(agents)

    assert [item["name"] for item in groups["horizons"]] == ["К", "P2"]
    assert groups["horizons"][0]["evidence"] == ["page:00021", "page:00190"]


def test_enrichment_removes_stale_unverified_delivery(tmp_path: Path) -> None:
    output_dir = tmp_path / "georef_exports"
    output_dir.mkdir()
    stale = output_dir / "page_00179_arcgis_petrel.zip"
    stale.write_bytes(b"unverified")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        """
        {
          "artifacts": [
            {
              "name": "georef_delivery",
              "path": "%s",
              "page_id": "page:00179"
            }
          ]
        }
        """
        % str(stale).replace("\\", "\\\\"),
        encoding="utf-8",
    )

    deliveries = enrich_georeference_packages(result_path, {})

    assert deliveries == []
    assert not stale.exists()
    assert '"artifacts": []' in result_path.read_text(encoding="utf-8")
