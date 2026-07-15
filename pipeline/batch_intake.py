from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from report_factory import IMAGE_SUFFIXES, PDF_SUFFIXES, build_manifest, write_manifest


ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".rar", ".7z"}


def _safe_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"Unsafe archive path: {relative}")
    return destination


def _seven_zip() -> Path:
    candidates = (
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    )
    executable = next((path for path in candidates if path.exists()), None)
    if executable is None:
        raise ValueError("7-Zip is required for RAR/7z archives")
    return executable


def _seven_zip_entries(archive: Path) -> tuple[list[str], int]:
    completed = subprocess.run(
        [str(_seven_zip()), "l", "-slt", "-sccUTF-8", str(archive)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise ValueError(f"Cannot list archive: {archive.name}")
    records = completed.stdout.split("----------", 1)
    body = records[1] if len(records) == 2 else ""
    paths = []
    unpacked_size = 0
    for block in body.split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if " = " in line:
                key, value = line.split(" = ", 1)
                fields[key.strip()] = value.strip()
        relative = fields.get("Path")
        if relative:
            paths.append(relative)
        try:
            unpacked_size += int(fields.get("Size", "0"))
        except ValueError:
            pass
    if not paths:
        raise ValueError(f"Archive contains no entries: {archive.name}")
    return paths, unpacked_size


def _extract_with_seven_zip(archive: Path, destination: Path) -> None:
    entries, unpacked_size = _seven_zip_entries(archive)
    for relative in entries:
        _safe_destination(destination, relative)
    free_space = shutil.disk_usage(destination.parent).free
    if unpacked_size and free_space < int(unpacked_size * 1.1):
        raise ValueError(
            f"Not enough free space for {archive.name}: required {unpacked_size}, free {free_space}"
        )
    completed = subprocess.run(
        [
            str(_seven_zip()),
            "x",
            "-y",
            "-aoa",
            "-sccUTF-8",
            f"-o{destination}",
            str(archive),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=24 * 60 * 60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise ValueError(f"7-Zip extraction failed for {archive.name}: {completed.stderr[-500:]}")


def extract_archive(archive: Path, destination: Path) -> Path:
    marker = destination / ".extracted.json"
    upload_policy = archive.with_suffix(archive.suffix + ".policy.json")
    if marker.exists():
        if upload_policy.exists():
            (destination / ".upload_policy.json").write_text(
                upload_policy.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix.casefold() == ".zip":
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_destination(destination, member.filename)
            handle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                _safe_destination(destination, member.name)
            handle.extractall(destination, filter="data")
    elif archive.suffix.casefold() in {".rar", ".7z"}:
        _extract_with_seven_zip(archive, destination)
    else:
        raise ValueError(f"Unsupported archive: {archive.name}")
    marker.write_text(
        json.dumps(
            {"archive": str(archive), "size": archive.stat().st_size, "extracted_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if upload_policy.exists():
        (destination / ".upload_policy.json").write_text(
            upload_policy.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return destination


def has_images(path: Path) -> bool:
    return any(
        item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES | PDF_SUFFIXES
        for item in path.rglob("*")
    )


def report_roots(container: Path) -> list[Path]:
    children = [path for path in container.iterdir() if path.is_dir() and not path.name.startswith(".")]
    child_names = {path.name.casefold() for path in children}
    if has_images(container) and any("текст" in name or "график" in name for name in child_names):
        return [container]
    candidates = [path for path in children if has_images(path)]
    if candidates:
        return candidates
    return [container] if has_images(container) else []


def write_privacy_policy(run_dir: Path, source: Path) -> Path:
    path = run_dir / "privacy.json"
    upload_policy_path = next(
        (
            parent / ".upload_policy.json"
            for parent in (source, *source.parents)
            if (parent / ".upload_policy.json").exists()
        ),
        None,
    )
    upload_policy = {}
    if upload_policy_path is not None:
        try:
            upload_policy = json.loads(upload_policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            upload_policy = {}
    payload = {
        "classification": "accessible_archive",
        "external_processing": True,
        "source": str(source),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "External processing is enabled for the accessible report collection.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def graphics_contact_sheet(manifest: dict[str, Any], output: Path, limit: int = 40) -> Path | None:
    pages = [page for page in manifest["pages"] if page["role"] in {"graphic", "toc", "unknown"}][:limit]
    if not pages:
        return None
    from page_router import build_contact_sheet

    root = Path(manifest["source_root"])
    sheet = build_contact_sheet(pages, root, cell_size=(360, 290), columns=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=86, optimize=True)
    return output


def intake(inbox: Path, staging: Path, runs: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    containers = [path for path in inbox.iterdir() if path.is_dir()]
    unsupported = []
    for archive in (
        path
        for path in inbox.iterdir()
        if path.is_file() and not path.name.casefold().endswith(".policy.json")
    ):
        try:
            if archive.suffix.casefold() in ARCHIVE_SUFFIXES or tarfile.is_tarfile(archive):
                containers.append(extract_archive(archive, staging / archive.stem))
            else:
                unsupported.append(str(archive))
        except (ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
            unsupported.append(f"{archive}: {type(error).__name__}")

    registered = []
    used_ids = set()
    existing_by_root = {}
    for existing_manifest in runs.glob("*/job.json"):
        try:
            payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
            existing_by_root[str(Path(payload["source_root"]).resolve()).casefold()] = existing_manifest
            used_ids.add(str(payload["report_id"]))
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    for container in containers:
        for root in report_roots(container):
            root_key = str(root.resolve()).casefold()
            if root_key in existing_by_root:
                manifest_path = existing_by_root[root_key]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                policy = write_privacy_policy(manifest_path.parent, root)
                contact_path = manifest_path.parent / "graphics_contact_sheet.jpg"
                contact = contact_path if contact_path.exists() else graphics_contact_sheet(manifest, contact_path)
                registered.append(
                    {
                        "report_id": manifest["report_id"],
                        "root": str(root),
                        "manifest": str(manifest_path),
                        "privacy": str(policy),
                        "contact_sheet": str(contact) if contact else None,
                        "summary": manifest["summary"],
                        "existing": True,
                    }
                )
                continue
            report_id = root.name
            if report_id in used_ids or (runs / report_id / "job.json").exists():
                suffix = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
                report_id = f"{report_id}_{suffix}"
            used_ids.add(report_id)
            manifest = build_manifest(root, report_id)
            manifest_path = write_manifest(manifest, runs)
            policy = write_privacy_policy(manifest_path.parent, root)
            contact = graphics_contact_sheet(manifest, manifest_path.parent / "graphics_contact_sheet.jpg")
            registered.append(
                {
                    "report_id": report_id,
                    "root": str(root),
                    "manifest": str(manifest_path),
                    "privacy": str(policy),
                    "contact_sheet": str(contact) if contact else None,
                    "summary": manifest["summary"],
                }
            )
    output = runs / "intake_summary.json"
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": "local_inventory_only",
                "registered": registered,
                "unsupported": unsupported,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Register report archives without external processing")
    parser.add_argument("--inbox", type=Path, default=Path(r"C:\FINAM\Conference\reports_inbox"))
    parser.add_argument("--staging", type=Path, default=Path(r"C:\FINAM\Conference\reports_staging"))
    parser.add_argument("--runs", type=Path, default=Path(r"C:\FINAM\Conference\runs"))
    args = parser.parse_args()
    print(intake(args.inbox, args.staging, args.runs))


if __name__ == "__main__":
    main()
