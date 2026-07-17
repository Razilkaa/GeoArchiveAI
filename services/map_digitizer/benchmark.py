"""Aggregate map-pipeline manifests into a reproducible quality report."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(float(ordered[index]), 3)


def summarize_results(results: list[dict]) -> dict:
    cases = []
    for payload in results:
        assignment = payload.get("stages", {}).get("assignment", {}).get("metrics", {})
        reconstruction = payload.get("stages", {}).get("reconstruction", {}).get("metrics", {})
        grid = reconstruction.get("grid_quality", {})
        topology = reconstruction.get("topology", {})
        cases.append(
            {
                "source": payload.get("source"),
                "status": payload.get("status"),
                "latency_s": payload.get("total_latency_s"),
                "interval_km": assignment.get("contour_interval_km"),
                "interval_support": (assignment.get("contour_interval_inference") or {}).get(
                    "selected_support"
                ),
                "mode": reconstruction.get("reconstruction_mode"),
                "profile_measurements": assignment.get("profile_measurements"),
                "accepted_traces": reconstruction.get(
                    "accepted_traces", reconstruction.get("final_constraints")
                ),
                "constraint_p95_m": grid.get("constraint_p95_abs_error_m"),
                "levels": topology.get("levels"),
                "closed_segments": topology.get("closed_segments"),
                "crossings": topology.get("crossing_pairs"),
                "reasons": (payload.get("quality") or {}).get("reasons", []),
            }
        )
    latencies = [float(case["latency_s"]) for case in cases if case["latency_s"] is not None]
    reconstructed = [case for case in cases if case["mode"]]
    return {
        "case_count": len(cases),
        "status_counts": dict(Counter(case["status"] for case in cases)),
        "mode_counts": dict(Counter(case["mode"] for case in reconstructed)),
        "latency_s": {
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "zero_crossing_rate": round(
            sum(case["crossings"] == 0 for case in reconstructed) / max(1, len(reconstructed)), 4
        ),
        "accepted_rate": round(
            sum(case["status"] == "accepted" for case in cases) / max(1, len(cases)), 4
        ),
        "cases": cases,
    }


def render_markdown(summary: dict) -> str:
    latency = summary["latency_s"]
    lines = [
        "# Map digitization benchmark",
        "",
        f"- Cases: **{summary['case_count']}**",
        f"- Statuses: **{summary['status_counts']}**",
        f"- Median / P95 latency: **{latency['median']} / {latency['p95']} s**",
        f"- Zero-crossing rate: **{summary['zero_crossing_rate']:.1%}**",
        f"- Auto-accepted rate: **{summary['accepted_rate']:.1%}**",
        "",
        "| Source | Status | Mode | Step, km | P95, m | Levels | Closed | Crossings | Time, s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        source = Path(case["source"]).name if case["source"] else "-"
        lines.append(
            f"| {source} | {case['status']} | {case['mode'] or '-'} | "
            f"{case['interval_km'] if case['interval_km'] is not None else '-'} | "
            f"{round(case['constraint_p95_m'], 2) if case['constraint_p95_m'] is not None else '-'} | "
            f"{case['levels'] if case['levels'] is not None else '-'} | "
            f"{case['closed_segments'] if case['closed_segments'] is not None else '-'} | "
            f"{case['crossings'] if case['crossings'] is not None else '-'} | "
            f"{case['latency_s'] if case['latency_s'] is not None else '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize map pipeline_result.json files")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for candidate in args.results:
        if candidate.is_dir():
            paths.extend(candidate.rglob("pipeline_result.json"))
        else:
            paths.append(candidate)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(set(paths))]
    summary = summarize_results(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
