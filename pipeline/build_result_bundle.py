from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factory_runner import load_manifest, set_stage


def source_index(manifest: dict[str, Any], agents: dict[str, Any]) -> list[dict[str, Any]]:
    page_by_id = {page["id"]: page for page in manifest["pages"]}
    referenced = {
        str(ref)
        for agent in agents.values()
        for item in agent.get("result", {}).get("items", [])
        for ref in (item.get("evidence") or [])
    }
    sources = []
    root = Path(manifest["source_root"])
    for ref in sorted(referenced):
        page = page_by_id.get(ref)
        if page:
            path = root / page["relative_path"]
            sources.append(
                {
                    "id": ref,
                    "path": str(path),
                    "relative_path": page["relative_path"],
                    "exists": path.exists(),
                    "role": page["role"],
                }
            )
        else:
            sources.append({"id": ref, "path": None, "relative_path": None, "exists": False, "role": "unknown"})
    return sources


def build_bundle(
    manifest_path: Path,
    ragflow_path: Path,
    agents_path: Path | None = None,
    map_result_path: Path | None = None,
) -> Path:
    manifest = load_manifest(manifest_path)
    ragflow = json.loads(ragflow_path.read_text(encoding="utf-8"))
    agent_payload = (
        json.loads(agents_path.read_text(encoding="utf-8"))
        if agents_path is not None and agents_path.exists()
        else {"agents": {}, "verification_queue": []}
    )
    agents = agent_payload.get("agents", {})
    failed = [name for name, value in agents.items() if value.get("status") != "completed"]
    review = [name for name, value in agents.items() if value.get("evidence_qc", {}).get("status") == "review"]
    output = manifest_path.parent / "result_bundle.json"
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_id": manifest["report_id"],
        "status": "failed" if failed else ("review" if review else "completed"),
        "strategy": manifest["strategy"],
        "summary": {
            **manifest["summary"],
            "rag_chunks": (ragflow.get("state") or {}).get("chunk_count"),
            "structures": len(agents.get("structures", {}).get("result", {}).get("items", [])),
            "wells": len(agents.get("wells", {}).get("result", {}).get("items", [])),
            "maps": len(agents.get("maps", {}).get("result", {}).get("items", [])),
            "key_results": len(agents.get("key_results", {}).get("result", {}).get("items", [])),
        },
        "stages": manifest["stages"],
        "entities": {name: value.get("result", {}).get("items", []) for name, value in agents.items()},
        "agent_qc": {name: value.get("evidence_qc", {}) for name, value in agents.items()},
        "verification_queue": agent_payload.get("verification_queue", []),
        "sources": source_index(manifest, agents),
        "artifacts": {
            "manifest": str(manifest_path),
            "ocr_markdown": str(manifest_path.parent / f"report_{manifest['report_id']}_fast_ocr.md"),
            "ragflow": str(ragflow_path),
            "agents": str(agents_path) if agents_path is not None and agents_path.exists() else None,
        },
    }
    if map_result_path and map_result_path.exists():
        map_result = json.loads(map_result_path.read_text(encoding="utf-8"))
        payload["map_digitization"] = {
            "summary": map_result.get("summary", {}),
            "result_path": str(map_result_path),
            "overlay_path": str(map_result_path.parents[1] / "overlay_v4.png"),
        }
        digitization_path = map_result_path.parent / "digitized" / "digitization_summary.json"
        if digitization_path.exists():
            payload["map_digitization"]["digitized"] = json.loads(
                digitization_path.read_text(encoding="utf-8")
            )
        tracing_path = map_result_path.parents[1] / "traced_isolines" / "tracing_summary.json"
        if tracing_path.exists():
            payload["map_digitization"]["tracing"] = json.loads(
                tracing_path.read_text(encoding="utf-8")
            )
        stitching_path = map_result_path.parents[1] / "stitched_isolines" / "stitching_summary.json"
        if stitching_path.exists():
            payload["map_digitization"]["stitching"] = json.loads(
                stitching_path.read_text(encoding="utf-8")
            )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    set_stage(
        manifest_path,
        "result_bundle",
        "completed",
        output=str(output),
        result_status=payload["status"],
        source_count=len(payload["sources"]),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frontend-ready report factory result")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ragflow", type=Path)
    parser.add_argument("agents", type=Path)
    parser.add_argument("--map-result", type=Path)
    args = parser.parse_args()
    print(build_bundle(args.manifest, args.ragflow, args.agents, args.map_result))


if __name__ == "__main__":
    main()
