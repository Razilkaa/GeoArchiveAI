"""Build a deduplicated old-report queue from the Volga-Ural seismic shapefile."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyogrio


SHAPE_COLUMNS = [
    "n_uk_rosg",
    "in_n_rosg",
    "god_nach",
    "god_end",
    "name_otch",
    "uk_id",
]


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _join_values(series: pd.Series) -> str:
    values = [value for value in dict.fromkeys(_text(series)) if value]
    return " | ".join(values)


def build_requisition_queue(
    frame: pd.DataFrame,
    *,
    min_year: int = 1970,
    max_year: int = 1999,
) -> pd.DataFrame:
    """Return one old-report candidate per Росгеолфонд inventory number."""
    missing = sorted(set(SHAPE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Shapefile is missing required fields: {', '.join(missing)}")

    inventory = (
        _text(frame["in_n_rosg"])
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype("int64")
    )
    year_start = pd.to_numeric(frame["god_nach"], errors="coerce")
    year_end = pd.to_numeric(frame["god_end"], errors="coerce")
    report_year = year_end.where(year_end.notna(), year_start)
    selected = inventory.gt(0) & report_year.between(min_year, max_year)

    rows = pd.DataFrame(
        {
            "inventory_id": inventory[selected],
            "year_start": year_start[selected],
            "year_end": year_end[selected],
            "report_year": report_year[selected].astype("int64"),
            "survey_name": _text(frame.loc[selected, "name_otch"]),
            "survey_id": _text(frame.loc[selected, "n_uk_rosg"]),
            "uk_id": _text(frame.loc[selected, "uk_id"]),
        }
    )
    grouped = (
        rows.groupby("inventory_id", sort=True)
        .agg(
            year_start=("year_start", "min"),
            year_end=("year_end", "max"),
            report_year=("report_year", "max"),
            survey_name=("survey_name", _join_values),
            survey_id=("survey_id", _join_values),
            uk_id=("uk_id", _join_values),
            source_features=("inventory_id", "size"),
        )
        .reset_index()
    )
    return grouped.sort_values(["report_year", "inventory_id"], kind="stable").reset_index(drop=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an old-report ЕФГИ queue from Seismic_Rosnedra.shp."
    )
    parser.add_argument(
        "--shape",
        type=Path,
        default=Path("shapes/Seismic_Rosnedra.shp"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/efgi_volga_ural_queue.csv"),
    )
    parser.add_argument("--min-year", type=int, default=1970)
    parser.add_argument("--max-year", type=int, default=1999)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.min_year > args.max_year:
        raise SystemExit("--min-year must not be greater than --max-year")

    frame = pyogrio.read_dataframe(
        args.shape,
        columns=SHAPE_COLUMNS,
        read_geometry=False,
    )
    queue = build_requisition_queue(
        frame,
        min_year=args.min_year,
        max_year=args.max_year,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "shape": str(args.shape),
                "output": str(args.output),
                "reports": len(queue),
                "year_min": int(queue["report_year"].min()) if len(queue) else None,
                "year_max": int(queue["report_year"].max()) if len(queue) else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
