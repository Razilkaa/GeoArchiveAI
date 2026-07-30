from pathlib import Path

import numpy as np

from services.map_digitizer.georeference_grid import georeference_grid_affine


def test_affine_grid_export_writes_projected_cps3(tmp_path: Path) -> None:
    source = tmp_path / "grid.npz"
    np.savez(
        source,
        x=np.asarray([0.0, 1.0, 2.0]),
        y=np.asarray([0.0, 1.0, 2.0]),
        z=np.asarray([
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
        ]),
    )
    matrix = np.asarray([
        [10.0, 0.0, 500_000.0],
        [0.0, -10.0, 7_000_000.0],
    ])

    result = georeference_grid_affine(
        source,
        matrix,
        tmp_path / "out",
        target_crs="EPSG:32639",
        validation={"residual_p90_m": 2.0},
    )

    assert result["status"] == "accepted"
    assert result["transform_model"] == "validated_affine"
    assert np.allclose(
        result["bounds"],
        [500000.0, 6999980.0, 500020.0, 7000000.0],
    )
    assert Path(result["files"]["cps3"]).is_file()
    assert Path(result["files"]["prj"]).is_file()
