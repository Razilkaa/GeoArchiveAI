"""User-facing exports for a surface that is not georeferenced yet."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
from pyproj import CRS

from services.map_digitizer.export_cps3_grid import Grid, write_cps3, write_xyz

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def materialize_local_exports(
    grid_path: str | Path,
    contours_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Create clean preview and exchange files in raster-pixel coordinates."""
    grid_path = Path(grid_path)
    contours_path = Path(contours_path)
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
    )
    if len(contours):
        contours.plot(ax=axis, color="#161616", linewidth=0.7)
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_title("Digitized surface | local pixel coordinates | REVIEW")
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
