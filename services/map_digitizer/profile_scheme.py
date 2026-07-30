"""Mask seismic profiles by borrowing the report's own profile scheme.

Structural sheets carry a dense grid of MOGT seismic profiles.  Traced blindly,
that grid digitises as false contours -- the "кракозябры" that clot the corners
of a sheet.  Every earlier attempt to erase them failed on the same missing
piece: no dependable registration of the profile network onto the map.

The profile *scheme* (a labelled appendix in every report) supplies it.  OCR
reads the SAME printed profile numbers on both the scheme and the structural
map, so a number found on both is a direct point correspondence -- the scheme's
label position against the map's label position.  A handful of shared numbers
pins an affine scheme->map transform; the whole (clean, complete) scheme network
then projects onto the map, and its corridor tells which traces are profile ink.

The transform's residual on its own anchors proves nothing (the lesson that
recurred all evening), so a caller must judge the fit two ways: enough shared
anchors, and a small residual measured in sheet pixels -- and, failing either,
skip the mask rather than gate the digitisation.  A trace is removed only when it
lies inside the corridor AND runs ALONG the nearest scheme line: an isoline that
merely crosses a profile corridor keeps its own heading and survives.
"""
from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .georeference import read_profile_number_labels
from .source_preserving_trace import _prune_spurs, _stitch, _walk_skeleton


# Acceptance gate on the scheme->map fit.  Both are in sheet pixels, never
# metres: metres silently change strictness with scale (georeference.py made the
# same move).  Below either bar the mask is skipped, the sheet digitises as-is.
MINIMUM_ANCHORS = 4
MAXIMUM_RESIDUAL_PX = 20.0

# Direction gate.  A cosine is scale-free; the corridor width is not, so it is
# derived from the fit residual rather than pinned to a constant.
ALONG_COSINE = 0.70
CORRIDOR_FRACTION = 0.60
CORRIDOR_FLOOR_PX = 20.0
CORRIDOR_RESIDUAL_FACTOR = 1.8
SAMPLE_STEP_PX = 8.0


# --------------------------------------------------------------------------- #
# 1. Locate the scheme sheet in a report
# --------------------------------------------------------------------------- #
def _safe_page_id(page_id: str) -> str:
    """Page id to job-directory name, mirroring report_batch._safe_page_id."""
    return "".join(ch if ch.isalnum() else "_" for ch in page_id)


def _reads_as_scheme(ocr_path: Path) -> bool:
    """OCR text of a MOGT profile scheme carries 'профил' with an ОГТ marker."""
    if not ocr_path.exists():
        return False
    text = json.loads(ocr_path.read_text(encoding="utf-8")).get("text", "").lower()
    return "профил" in text and ("могт" in text or "огт" in text)


def find_scheme_sheet(run_dir: Path) -> Path | None:
    """Find the job directory whose page reads as a MOGT profile scheme.

    The router's ``kind`` is not decisive here (a scheme lands under "chart" or
    "map", not a scheme-specific class), so the routing only narrows the search
    to non-seismic pages and the OCR title makes the call.  Reports assembled
    without a routing file (hand-built runs) fall back to scanning every job.
    Returns None when the report has no scheme, and the caller runs unchanged.
    """
    jobs_root = run_dir / "map_agent" / "jobs"
    if not jobs_root.exists():
        jobs_root = run_dir / "jobs"

    routing = run_dir / "page_routing.json"
    if routing.exists():
        payload = json.loads(routing.read_text(encoding="utf-8"))
        for entry in payload.get("pages") or []:
            if entry.get("kind") == "seismic_section":
                continue
            page_id = entry.get("page_id")
            if not page_id:
                continue
            job = jobs_root / _safe_page_id(str(page_id))
            if _reads_as_scheme(job / "ocr.json"):
                return job
        return None

    for job in sorted(jobs_root.glob("*")):
        if _reads_as_scheme(job / "ocr.json"):
            return job
    return None


# --------------------------------------------------------------------------- #
# 2. Read shared profile numbers and their positions
# --------------------------------------------------------------------------- #
def read_number_positions(ocr_path: Path) -> dict[str, np.ndarray]:
    """Map each profile number to a single label centre in that sheet's pixels.

    A number printed once is an unambiguous anchor; a number the OCR placed in
    two spots (a real repeat, or a misread) cannot say which end to trust, so it
    is dropped rather than averaged into a phantom midpoint.
    """
    grouped: dict[str, list[np.ndarray]] = {}
    for label in read_profile_number_labels(Path(ocr_path)):
        grouped.setdefault(label["number"], []).append(
            np.asarray([label["x"], label["y"]], dtype=float)
        )
    return {number: spots[0] for number, spots in grouped.items() if len(spots) == 1}


def match_common_numbers(
    scheme_ocr: Path, map_ocr: Path
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (numbers, scheme_points, map_points) for numbers read on both."""
    scheme = read_number_positions(scheme_ocr)
    sheet = read_number_positions(map_ocr)
    numbers = sorted(set(scheme) & set(sheet))
    if not numbers:
        return [], np.empty((0, 2)), np.empty((0, 2))
    src = np.asarray([scheme[n] for n in numbers])
    dst = np.asarray([sheet[n] for n in numbers])
    return numbers, src, dst


# --------------------------------------------------------------------------- #
# 3. Fit scheme -> map
# --------------------------------------------------------------------------- #
def _affine_from(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    design = np.column_stack([src, np.ones(len(src))])
    transform, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return transform  # (3, 2): [x, y, 1] @ T


def apply_affine(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ transform[:2] + transform[2]


def fit_scheme_to_map(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold_px: float = MAXIMUM_RESIDUAL_PX,
    iterations: int = 2000,
    seed: int = 0,
) -> dict | None:
    """RANSAC affine scheme->map, refit on inliers.

    Point correspondences are exact by construction (number to number), so the
    only outliers are OCR mispairs; a light RANSAC rejects them.  Returns the
    transform with its inlier residuals, or None when too few anchors to fit.
    """
    n = len(src)
    if n < 3:
        return None
    src_h = np.column_stack([src, np.ones(n)])
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    for _ in range(iterations):
        pick = rng.choice(n, 3, replace=False)
        try:
            trial = _affine_from(src[pick], dst[pick])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(src_h @ trial - dst, axis=1)
        inliers = residual <= threshold_px
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        best_inliers = np.ones(n, dtype=bool)
    transform = _affine_from(src[best_inliers], dst[best_inliers])
    residual = np.linalg.norm(src_h @ transform - dst, axis=1)
    inlier_residual = residual[best_inliers]
    return {
        "transform": transform,
        "inliers": best_inliers,
        "inlier_count": int(best_inliers.sum()),
        "anchor_count": n,
        "residual_median_px": float(np.median(inlier_residual)),
        "residual_p90_px": float(np.percentile(inlier_residual, 90)),
    }


# --------------------------------------------------------------------------- #
# 4. Trace the scheme network
# --------------------------------------------------------------------------- #
def trace_scheme_network(
    gray: np.ndarray, *, target_size: int = 2600, spur_length: int = 8
) -> list[np.ndarray]:
    """Trace the profile lines off a scheme scan, in full-resolution pixels.

    The scheme is clean linework, so a skeleton walk recovers it -- but NOT the
    isoline tracer (`trace_source_geometry`), which discards straight lines as
    profiles: here the straight lines ARE the target.  Small components (numbers,
    ticks) are dropped by span, and the frame and fold creases by their length
    against the sheet diagonal, before skeletonising.

    Profiles are drawn through chains of shot-point circles; the skeleton loops
    around every circle, so the raw walk fragments a profile at each one.  The
    stitch reconnects those fragments where their ends line up, restoring whole
    profiles -- without it dense clusters trace as short scraps and leave gaps
    the mask cannot cover.  The span filter is normalised by resolution so the
    same physical strokes survive whatever the working size is set to.
    """
    height, width = gray.shape
    scale = min(1.0, target_size / max(height, width))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Otsu picks the ink/paper split from this sheet's own histogram: an archival
    # scan can sit anywhere in tone (here the paper reads ~219, the faint profile
    # ink ~150), and a fixed 128 drops whole clusters of pale linework.
    threshold, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = np.uint8(small < threshold)

    small_h, small_w = ink.shape
    diagonal = float(np.hypot(small_h, small_w))
    resolution = max(small_h, small_w) / 2600.0
    min_span = round(40 * resolution)  # numbers and ticks fall below this
    gap = 22.0 * resolution            # bridge the loop a circle opens in a line

    # The frame rectangle hugs the sheet border, and once Otsu joins the profiles
    # into one network-wide component it can no longer be told from them by size.
    # Zeroing a thin border band removes it up front, so the interior network
    # survives whole instead of being discarded as an oversized component -- and
    # the size filter then only has to drop the small stuff (numbers, ticks).
    band = int(round(0.02 * min(small_h, small_w)))
    ink[:band] = 0
    ink[-band:] = 0
    ink[:, :band] = 0
    ink[:, -band:] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    retained = np.zeros_like(ink)
    for component in range(1, count):
        w, h = stats[component, cv2.CC_STAT_WIDTH], stats[component, cv2.CC_STAT_HEIGHT]
        if max(w, h) >= min_span:
            retained[labels == component] = 1

    skeleton = skeletonize(retained > 0)
    skeleton = _prune_spurs(skeleton, max_length=spur_length)
    walks: list[np.ndarray] = []
    for path in _walk_skeleton(skeleton):
        if len(path) < 5:
            continue
        walks.append(np.asarray(path, dtype=float)[:, ::-1])  # (row, col) -> (x, y)

    stitched = _stitch(walks, max_gap=gap, minimum_alignment=0.8)
    inverse = 1.0 / scale
    # Fold creases and any frame remnant cross the sheet as one near-straight
    # chord longer than half the diagonal; a real profile never spans that far.
    return [
        line * inverse
        for line in stitched
        if float(np.hypot(*(line[-1] - line[0]))) <= 0.5 * diagonal
    ]


# --------------------------------------------------------------------------- #
# 5. Build the mask
# --------------------------------------------------------------------------- #
def _densify(coords: np.ndarray, step: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords
    out = [coords[0]]
    for start, end in zip(coords[:-1], coords[1:]):
        segment = end - start
        length = float(np.hypot(*segment))
        if length < 1e-9:
            continue
        steps = max(1, int(length // step))
        for k in range(1, steps + 1):
            out.append(start + segment * (k / steps))
    return np.asarray(out)


def _network_segments(net_map: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Midpoints and unit headings of every scheme segment, in map pixels."""
    mids, dirs = [], []
    for line in net_map:
        line = np.asarray(line, dtype=float)
        if len(line) < 2:
            continue
        delta = np.diff(line, axis=0)
        length = np.hypot(delta[:, 0], delta[:, 1])
        keep = length > 1e-6
        if not keep.any():
            continue
        unit = delta[keep] / length[keep, None]
        mids.append(((line[:-1] + line[1:]) / 2.0)[keep])
        dirs.append(unit)
    if not mids:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(mids), np.vstack(dirs)


def build_profile_mask(
    net_map: list[np.ndarray],
    traces: list[np.ndarray],
    *,
    corridor_px: float,
    corridor_fraction: float = CORRIDOR_FRACTION,
    along_cosine: float = ALONG_COSINE,
    sample_step: float = SAMPLE_STEP_PX,
) -> list[dict]:
    """Decide which traces are profile ink under the projected scheme network.

    A trace is flagged when most of its length lies inside the corridor AND, over
    that in-corridor length, its heading agrees with the nearest scheme line
    (length-weighted mean |cos| high).  The second test is what spares an isoline
    that only crosses the corridor: it holds its own bearing across the profile,
    so its mean alignment stays low even while its coverage is high.
    """
    seg_mid, seg_dir = _network_segments(net_map)
    if len(seg_mid) == 0:
        return [{"index": i, "remove": False, "corridor_fraction": 0.0,
                 "alignment": 0.0} for i in range(len(traces))]
    tree = cKDTree(seg_mid)

    decisions: list[dict] = []
    for index, trace in enumerate(traces):
        points = _densify(trace, sample_step)
        corridor_fraction_value, alignment = 0.0, 0.0
        if len(points) >= 3:
            delta = np.diff(points, axis=0)
            seglen = np.hypot(delta[:, 0], delta[:, 1])
            heading = delta / np.where(seglen[:, None] > 1e-6, seglen[:, None], 1.0)
            mid = (points[:-1] + points[1:]) / 2.0
            total = float(seglen.sum())
            if total > 1e-6:
                distance, nearest = tree.query(mid)
                inside = distance <= corridor_px
                corridor_fraction_value = float(seglen[inside].sum() / total)
                if int(inside.sum()) >= 2:
                    scheme_heading = seg_dir[nearest[inside]]
                    cosine = np.abs(
                        heading[inside, 0] * scheme_heading[:, 0]
                        + heading[inside, 1] * scheme_heading[:, 1]
                    )
                    alignment = float(np.average(cosine, weights=seglen[inside]))
        remove = (
            corridor_fraction_value >= corridor_fraction
            and alignment >= along_cosine
        )
        decisions.append(
            {
                "index": index,
                "remove": remove,
                "corridor_fraction": corridor_fraction_value,
                "alignment": alignment,
            }
        )
    return decisions


# --------------------------------------------------------------------------- #
# 6. Orchestrate: fit, gate, mask
# --------------------------------------------------------------------------- #
def profile_scheme_mask(
    *,
    scheme_ocr: Path,
    map_ocr: Path,
    scheme_gray: np.ndarray,
    traces: list[np.ndarray],
    minimum_anchors: int = MINIMUM_ANCHORS,
    maximum_residual_px: float = MAXIMUM_RESIDUAL_PX,
) -> dict:
    """Full pass: match numbers, fit, gate, then mask traces.

    Always returns a report.  ``status`` is "applied" only when the fit clears
    the gate; otherwise ("no_common_numbers", "fit_failed", "fit_rejected") the
    caller leaves every trace in place -- the mask never blocks a good sheet to
    shave a handful of strokes.
    """
    numbers, src, dst = match_common_numbers(scheme_ocr, map_ocr)
    report: dict = {"status": None, "common_numbers": len(numbers), "removed": []}

    if len(numbers) < minimum_anchors:
        report["status"] = "no_common_numbers"
        return report

    fit = fit_scheme_to_map(src, dst, threshold_px=maximum_residual_px)
    if fit is None:
        report["status"] = "fit_failed"
        return report
    report["fit"] = {
        key: fit[key]
        for key in ("inlier_count", "anchor_count", "residual_median_px", "residual_p90_px")
    }

    if fit["inlier_count"] < minimum_anchors or fit["residual_p90_px"] > maximum_residual_px:
        report["status"] = "fit_rejected"
        return report

    net = trace_scheme_network(scheme_gray)
    net_map = [apply_affine(line, fit["transform"]) for line in net]
    # Corridor width follows the fit's own scatter, so the mask self-scales with
    # sheet resolution instead of trusting a pixel constant tuned on one report.
    corridor_px = max(CORRIDOR_FLOOR_PX, fit["residual_p90_px"] * CORRIDOR_RESIDUAL_FACTOR)
    report["corridor_px"] = corridor_px
    report["scheme_lines"] = len(net_map)

    decisions = build_profile_mask(net_map, traces, corridor_px=corridor_px)
    report["removed"] = [d["index"] for d in decisions if d["remove"]]
    report["decisions"] = decisions
    report["status"] = "applied"
    return report
