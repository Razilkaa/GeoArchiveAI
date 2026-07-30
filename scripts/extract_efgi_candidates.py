"""Build a deduplicated ЕФГИ report queue from the survey-profile shapefile."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyogrio


YEAR_COLUMN = "\u0413\u043e\u0434_\u0441"
TITLE_COLUMN = "\u041d\u0430\u0437\u0432\u0430"
AUTHOR_COLUMN = "\u0410\u0432\u0442\u043e\u0440"
ORGANIZATION_COLUMN = "\u041e\u0440\u0433\u0430\u043d"
AVAILABILITY_COLUMN = "\u041d\u0430\u043b\u0438\u0447"
OBJECT_URL_COLUMN = "\u0414\u043e\u0441\u0442\u0443"
LOCATION_COLUMN = "\u041c\u0435\u0441\u0442\u043e"


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("\u00a0", " ", regex=False).str.strip()


def _inventory_ids(frame: pd.DataFrame) -> pd.Series:
    primary = pd.to_numeric(frame["N_RGF"], errors="coerce").fillna(0).astype("int64")
    fallback = (
        _text(frame["INV_svod"])
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype("int64")
    )
    return primary.where(primary.gt(0), fallback)


def _join_values(series: pd.Series) -> str:
    values = [value for value in dict.fromkeys(_text(series)) if value]
    return " | ".join(values)


def _has_digital_copy(value: str) -> bool:
    normalized = value.casefold().replace("сканобразы каталожных карточек", "")
    return "сканобраз" in normalized or "электронная версия" in normalized


def extract_candidate_reports(
    frame: pd.DataFrame,
    *,
    start_year: int = 1980,
    end_year: int = 1999,
) -> pd.DataFrame:
    """Return one candidate per inventory/year/title without reading geometry."""
    required = {
        "N_RGF",
        "INV_svod",
        YEAR_COLUMN,
        TITLE_COLUMN,
        AUTHOR_COLUMN,
        ORGANIZATION_COLUMN,
        AVAILABILITY_COLUMN,
        OBJECT_URL_COLUMN,
        LOCATION_COLUMN,
        "link",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Shapefile is missing required fields: {', '.join(missing)}")

    years = pd.to_numeric(frame[YEAR_COLUMN], errors="coerce")
    inventory_ids = _inventory_ids(frame)
    selected = years.between(start_year, end_year) & inventory_ids.gt(0)

    candidates = pd.DataFrame(
        {
            "inventory_id": inventory_ids[selected],
            "year": years[selected].astype("int64"),
            "title": _text(frame.loc[selected, TITLE_COLUMN]),
            "author": _text(frame.loc[selected, AUTHOR_COLUMN]),
            "organization": _text(frame.loc[selected, ORGANIZATION_COLUMN]),
            "location": _text(frame.loc[selected, LOCATION_COLUMN]),
            "availability": _text(frame.loc[selected, AVAILABILITY_COLUMN]),
            "object_url": _text(frame.loc[selected, OBJECT_URL_COLUMN]),
            "archive_path": _text(frame.loc[selected, "link"]),
        }
    )

    grouped = (
        candidates.groupby(["inventory_id", "year", "title"], sort=True, dropna=False)
        .agg(
            {
                "author": _join_values,
                "organization": _join_values,
                "location": _join_values,
                "availability": _join_values,
                "object_url": _join_values,
                "archive_path": _join_values,
            }
        )
        .reset_index()
    )
    grouped["has_object_url"] = grouped["object_url"].ne("")
    grouped["has_digital_copy"] = grouped["availability"].map(_has_digital_copy)
    grouped["likely_downloadable"] = grouped["has_object_url"] & grouped["has_digital_copy"]
    return grouped.sort_values(
        ["likely_downloadable", "year", "inventory_id"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract old ЕФГИ/Rosgeolfond report candidates from a shapefile."
    )
    parser.add_argument("--shape", type=Path, default=Path("shapes/srr_all.shp"))
    parser.add_argument("--output", type=Path, default=Path("results/efgi_candidates_1980_1999.csv"))
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=1999)
    parser.add_argument(
        "--likely-downloadable-only",
        action="store_true",
        help="Keep only rows with an ЕФГИ object URL and a report scan/electronic-copy marker.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must not be greater than --end-year")

    frame = pyogrio.read_dataframe(args.shape, read_geometry=False)
    candidates = extract_candidate_reports(
        frame,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if args.likely_downloadable_only:
        candidates = candidates.loc[candidates["likely_downloadable"]].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.casefold() == ".json":
        args.output.write_text(
            json.dumps(candidates.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        candidates.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = {
        "shape": str(args.shape),
        "output": str(args.output),
        "reports": len(candidates),
        "inventories": int(candidates["inventory_id"].nunique()),
        "likely_downloadable": int(candidates["likely_downloadable"].sum()),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
