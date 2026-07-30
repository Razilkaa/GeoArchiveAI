"""Render the final digitized isoline layer over the source map raster.

QC view for the vector deliverable: dissolved isolines are coloured by level,
fragments without a trusted level stay grey, so gaps against the original
linework are visible at a glance.

Usage:
    python scripts/render_isoline_overlay.py runs/384092/map_agent/jobs/page_00184
        [--output overlay.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = json.loads((args.job_dir / "pipeline_result.json").read_text(encoding="utf-8"))
    surface_dir = args.job_dir / "surface"
    isolines = gpd.read_file(surface_dir / "digitized_isolines_by_level.geojson")
    preserved = gpd.read_file(surface_dir / "digitized_source_contours_pixels.geojson")
    unvalued = preserved[preserved["value_m"].isna()]

    with Image.open(result["source"]) as source:
        background = np.asarray(source.convert("L"))
    step = max(1, int(np.ceil(max(background.shape) / 4000.0)))

    fig, axis = plt.subplots(figsize=(16, 13), dpi=140)
    axis.imshow(
        background[::step, ::step],
        cmap="gray",
        alpha=0.55,
        extent=(0, background.shape[1], background.shape[0], 0),
    )
    norm = Normalize(
        vmin=float(isolines["value_m"].min()), vmax=float(isolines["value_m"].max())
    )
    colormap = plt.get_cmap("turbo")
    for row in isolines.itertuples():
        points = np.asarray(row.geometry.coords)
        axis.plot(
            points[:, 0], points[:, 1],
            color=colormap(norm(row.value_m)), lw=1.4, alpha=0.95,
        )
    for row in unvalued.itertuples():
        points = np.asarray(row.geometry.coords)
        axis.plot(points[:, 0], points[:, 1], color="#dd2266", lw=1.2, alpha=0.9)
    axis.set_title(
        f"Digitized isolines over source | {len(isolines)} isolines, "
        f"{len(unvalued)} fragments without level (magenta)"
    )
    axis.axis("off")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=axis, shrink=0.7, label="Depth, m",
    )
    output = args.output or (surface_dir / "digitized_isolines_overlay.png")
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
