"""Export valued isolines of sheet 23 to GeoJSON (pixel coordinates for now;
georeferencing pending). Properties: value_km, verdict, source, closed."""
import json
from pathlib import Path

import numpy as np

TR = Path(r"C:\FINAM\Conference\pipeline\tracing\out\23")

iso_path = TR / ("isolines_clean.json" if (TR / "isolines_clean.json").exists() else "isolines.json")
iso = json.loads(iso_path.read_text(encoding="utf-8"))
va = json.loads((TR / "valued_isolines_23.json").read_text(encoding="utf-8"))
cc = json.loads((TR / "crosscheck_23.json").read_text(encoding="utf-8"))

labeled = {int(k): v for k, v in va["values"].items()}
from_surface = {int(k): v for k, v in cc["from_surface"].items()}
verdicts = {int(k): v for k, v in cc["verdicts"].items()}
leaks = set(va["profile_leaks"])

features = []
for i, pl in enumerate(iso["polylines_xy"]):
    if i in leaks:
        kind, value, source = "seismic_profile", None, "hough_straightness"
    elif i in labeled:
        kind, value = "isoline", labeled[i]
        source = "label"
    elif i in from_surface:
        kind, value = "isoline", from_surface[i]
        source = "surface_interpolation"
    else:
        kind, value, source = "isoline", None, "untraced_value"
    p = np.array(pl, float)
    end_gap = float(np.linalg.norm(p[0] - p[-1]))
    length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    features.append({
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[float(x), float(-y)] for x, y in pl]},
        "properties": {"id": i, "kind": kind, "value_km": value,
                       "verdict": verdicts.get(i), "source": source,
                       "length_px": round(length, 1),
                       "closed": bool(end_gap < 0.12 * length and length > 800)},
    })

gj = {"type": "FeatureCollection",
      "crs_note": "image pixels of 23.bmp, y inverted for GIS; NOT georeferenced",
      "contour_interval_km": 0.2,
      "source_sheet": "384092 / прил. 23 / Шеинская площадь / 1:100000 / 1978",
      "features": features}
(TR / "isolines_23.geojson").write_text(json.dumps(gj, ensure_ascii=False),
                                        encoding="utf-8")
n_val = sum(1 for f in features if f["properties"]["value_km"] is not None)
print(f"features: {len(features)}, valued: {n_val} -> isolines_23.geojson")
