"""Check ЕФГИ inventory numbers and optionally add requestable reports to a requisition.

The script intentionally does not submit the finished requisition. Review the
resulting list in ЕФГИ before using the green submit arrow.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright


SEARCH_URL = "https://efgi.ru/#/registry/search"
INVENTORY_HEADER = '.ag-header-cell[aria-colindex="4"] app-input'
RESULT_CELL = '.ag-cell[col-id="combinedInvGeo"]'
ACTION_CELL = '.ag-cell[col-id="0"]'


@dataclass
class CheckResult:
    inventory_id: int
    report_year: int
    status: str
    registry_id: str = ""
    title: str = ""
    place: str = ""
    carrier: str = ""
    added_to_request: bool = False
    error: str = ""


async def _request_count(page: Page) -> int | None:
    texts = await page.get_by_text(
        re.compile(r"^Перечень запрашиваемых документов \(\d+\)$")
    ).all_inner_texts()
    for text in texts:
        match = re.search(r"\((\d+)\)", text)
        if match:
            return int(match.group(1))
    return None


async def _ensure_authenticated(page: Page) -> None:
    await page.goto(SEARCH_URL, wait_until="domcontentloaded")
    try:
        await page.get_by_text("Перечень запрашиваемых документов", exact=False).wait_for(
            state="visible",
            timeout=15_000,
        )
    except PlaywrightTimeoutError:
        print("Войдите в ЕФГИ через Госуслуги в открытом окне Chrome.")
        await asyncio.to_thread(input, "После появления реестра нажмите Enter здесь: ")
        await page.goto(SEARCH_URL, wait_until="domcontentloaded")
        await page.get_by_text("Перечень запрашиваемых документов", exact=False).wait_for(
            state="visible",
            timeout=60_000,
        )


async def _search_inventory(page: Page, inventory_id: int) -> None:
    menu_input = page.get_by_role("textbox", name="Инвентарный номер", exact=True)
    if not await menu_input.is_visible():
        header = page.locator(f"{INVENTORY_HEADER}:visible")
        await header.click()
        await menu_input.wait_for(state="visible", timeout=10_000)
    await menu_input.fill(str(inventory_id))
    await menu_input.press("Enter")

    result = page.locator(RESULT_CELL, has_text=f"/ {inventory_id}")
    zero = page.get_by_text("Найдено объектов учета: 0", exact=True)
    await result.or_(zero).first.wait_for(state="visible", timeout=25_000)


async def _read_result(
    page: Page,
    *,
    inventory_id: int,
    report_year: int,
    add_to_request: bool,
) -> CheckResult:
    result_cell = page.locator(RESULT_CELL, has_text=f"/ {inventory_id}")
    result_count = await result_cell.count()
    if result_count == 0:
        return CheckResult(inventory_id, report_year, "not_found")
    if result_count != 1:
        return CheckResult(
            inventory_id,
            report_year,
            "ambiguous",
            error=f"ЕФГИ returned {result_count} matching rows",
        )

    data_row = result_cell.locator("xpath=ancestor::*[contains(@class,'ag-row')]")
    registry_id = (await data_row.locator('.ag-cell[col-id="id"]').inner_text()).strip()
    title = (await data_row.locator('.ag-cell[col-id="tableName"]').inner_text()).strip()
    place = (await data_row.locator('.ag-cell[col-id="place"]').inner_text()).strip()
    carrier = (
        await data_row.locator('.ag-cell[col-id="geoInfoCarrierTypesValue"]').inner_text()
    ).strip()

    visible_checkbox = page.locator(
        f"{ACTION_CELL} label.mat-checkbox-layout:visible"
    )
    if await visible_checkbox.count() != 1:
        return CheckResult(
            inventory_id,
            report_year,
            "not_requestable",
            registry_id,
            title,
            place,
            carrier,
        )

    added = False
    if add_to_request:
        before = await _request_count(page)
        await visible_checkbox.click()
        add_button = page.locator(".add-to-requisition-button")
        await add_button.wait_for(state="visible", timeout=10_000)
        await add_button.click()
        if before is not None:
            try:
                await page.get_by_text(
                    f"Перечень запрашиваемых документов ({before + 1})",
                    exact=True,
                ).wait_for(state="visible", timeout=15_000)
                added = True
            except PlaywrightTimeoutError:
                after = await _request_count(page)
                if after == before:
                    return CheckResult(
                        inventory_id,
                        report_year,
                        "requestable",
                        registry_id,
                        title,
                        place,
                        carrier,
                        False,
                        "Не добавлен: возможно, объект уже присутствует в текущей заявке",
                    )
        else:
            added = True

    return CheckResult(
        inventory_id,
        report_year,
        "requestable",
        registry_id,
        title,
        place,
        carrier,
        added,
    )


def _load_queue(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [
            (int(row["inventory_id"]), int(row["report_year"]))
            for row in csv.DictReader(stream)
        ]


def _load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            int(row["inventory_id"])
            for row in csv.DictReader(stream)
            if row.get("status") in {
                "requestable",
                "not_requestable",
                "not_found",
                "ambiguous",
            }
        }


def _append_result(path: Path, result: CheckResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(result)))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(result))


async def _run(args: argparse.Namespace) -> None:
    queue = _load_queue(args.queue)
    completed = _load_completed(args.output) if args.resume else set()
    remaining = [(inventory, year) for inventory, year in queue if inventory not in completed]
    if args.limit is not None:
        remaining = remaining[: args.limit]

    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            str(args.profile_dir.resolve()),
            channel="chrome",
            headless=False,
            viewport={"width": 1900, "height": 1000},
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await _ensure_authenticated(page)

        for index, (inventory_id, report_year) in enumerate(remaining, start=1):
            try:
                await _search_inventory(page, inventory_id)
                result = await _read_result(
                    page,
                    inventory_id=inventory_id,
                    report_year=report_year,
                    add_to_request=args.add_to_request,
                )
            except Exception as error:
                result = CheckResult(
                    inventory_id,
                    report_year,
                    "error",
                    error=f"{type(error).__name__}: {error}",
                )
            _append_result(args.output, result)
            print(
                f"[{index}/{len(remaining)}] {inventory_id}: {result.status}"
                + (" + added" if result.added_to_request else "")
            )
            await page.wait_for_timeout(args.delay_ms)

        print(f"Результаты: {args.output}")
        print("Заявка не отправлена: проверьте список в ЕФГИ и сформируйте её отдельно.")
        await context.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check old reports in ЕФГИ.")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("results/efgi_volga_ural_queue.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/efgi_volga_ural_checked.csv"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("runs/efgi_browser_profile"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--add-to-request",
        action="store_true",
        help="Check requestable rows and add them to the current ЕФГИ requisition.",
    )
    return parser


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
