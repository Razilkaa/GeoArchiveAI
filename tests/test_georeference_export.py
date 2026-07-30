import json
import zipfile
from pathlib import Path

from PIL import Image

from services.map_digitizer.georeference import export_raster_package


def test_export_raster_package_contains_arcgis_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "map.jpg"
    Image.new("RGB", (20, 10), "white").save(source)

    result = export_raster_package(
        source,
        tmp_path / "export",
        [[10.0, 0.0, 500_000.0], [0.0, -10.0, 6_000_000.0]],
        "EPSG:32639",
        {"method": "test", "quality_status": "review"},
    )

    world = Path(result["files"]["world_file"])
    assert world.read_text(encoding="ascii").splitlines() == [
        "10.000000000000",
        "0.000000000000",
        "0.000000000000",
        "-10.000000000000",
        "500000.000000000000",
        "6000000.000000000000",
    ]
    with zipfile.ZipFile(result["files"]["package"]) as archive:
        names = set(archive.namelist())
    assert {"map_georeferenced.jpg", "map_georeferenced.jgw", "map_georeferenced.prj"} <= names
    footprint = json.loads(Path(result["files"]["footprint"]).read_text(encoding="utf-8"))
    assert footprint["features"][0]["geometry"]["type"] == "Polygon"
