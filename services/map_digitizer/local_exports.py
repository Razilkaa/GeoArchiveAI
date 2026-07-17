"""User-facing exports for a surface that is not georeferenced yet."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from PIL import Image
from pyproj import CRS

from services.map_digitizer.export_cps3_grid import Grid, write_cps3, write_xyz

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def materialize_local_exports(
    grid_path: str | Path,
    contours_path: str | Path,
    output_dir: str | Path,
    source_mask_path: str | Path | None = None,
) -> dict[str, str]:
    """Create clean preview and exchange files in raster-pixel coordinates."""
    grid_path = Path(grid_path)
    contours_path = Path(contours_path)
    source_mask_path = Path(source_mask_path) if source_mask_path else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(grid_path) as payload:
        grid = Grid(
            x=np.asarray(payload["x"], dtype=float),
            y=np.asarray(payload["y"], dtype=float),
            z=np.asarray(payload["z"], dtype=float),
            crs=CRS.from_epsg(3857),
        )
    contours = gpd.read_file(contours_path)

    cps3_path = output_dir / "surface_local_pixels.cps3"
    xyz_path = output_dir / "surface_local_pixels.xyz"
    preview_path = output_dir / "surface_clean_preview.png"
    metadata_path = output_dir / "surface_local_pixels.json"
    write_cps3(
        grid,
        cps3_path,
        "Digitized_surface_local_pixels",
        crs_label="LOCAL_PIXEL_COORDINATES (not georeferenced)",
    )
    write_xyz(grid, xyz_path)

    fig, axis = plt.subplots(figsize=(11, 9), dpi=140)
    fill = axis.pcolormesh(
        grid.x,
        grid.y,
        np.ma.masked_invalid(grid.z),
        shading="auto",
        cmap="viridis_r",
        alpha=0.42,
    )
    if len(contours):
        # Keep the raster-derived linework legible regardless of whether a
        # trustworthy value could be assigned to every fragment.
        contours.plot(ax=axis, color="#202020", linewidth=0.7, alpha=0.9)
        valued = contours[contours.get("value_m").notna()] if "value_m" in contours else contours
        unvalued = contours[contours.get("value_m").isna()] if "value_m" in contours else contours.iloc[0:0]
        if len(valued):
            finite_grid = grid.z[np.isfinite(grid.z)]
            norm = Normalize(
                vmin=float(np.min(finite_grid)), vmax=float(np.max(finite_grid))
            )
            valued.plot(
                ax=axis,
                column="value_m",
                cmap="viridis_r",
                norm=norm,
                linewidth=1.15,
            )
        if len(unvalued):
            unvalued.plot(ax=axis, color="#555555", linewidth=0.55, alpha=0.75)
    if source_mask_path and source_mask_path.is_file():
        with Image.open(source_mask_path) as source_mask:
            mask = np.asarray(source_mask.convert("L"))
        masked_lines = np.ma.masked_where(mask == 0, mask)
        x_step = float(np.median(np.diff(grid.x))) if len(grid.x) > 1 else 1.0
        y_step = float(np.median(np.diff(grid.y))) if len(grid.y) > 1 else 1.0
        axis.imshow(
            masked_lines,
            cmap="gray_r",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            extent=(
                float(grid.x[0]),
                float(grid.x[-1] + x_step),
                float(grid.y[-1] + y_step),
                float(grid.y[0]),
            ),
            alpha=0.92,
            zorder=5,
        )
    axis.set_ylim(float(grid.y[-1]), float(grid.y[0]))
    axis.set_aspect("equal")
    axis.set_title("Source-preserving digitized map | local pixel coordinates | REVIEW")
    axis.set_xlabel("Raster X, px")
    axis.set_ylabel("Raster Y, px")
    fig.colorbar(fill, ax=axis, shrink=0.8, label="Depth/elevation, m")
    fig.tight_layout()
    fig.savefig(preview_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata_path.write_text(
        json.dumps(
            {
                "coordinate_system": "local_pixel_coordinates",
                "georeferenced": False,
                "warning": "Assign control points before loading this grid into a spatial project.",
                "contour_geometry": "traced_from_source_raster",
                "source_mask": str(source_mask_path) if source_mask_path else None,
                "grid_role": "interpolation_between_traced_contours",
                "files": {
                    "cps3": str(cps3_path),
                    "xyz": str(xyz_path),
                    "preview": str(preview_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "surface_clean_preview": str(preview_path),
        "local_cps3": str(cps3_path),
        "local_xyz": str(xyz_path),
        "local_export_metadata": str(metadata_path),
    }
