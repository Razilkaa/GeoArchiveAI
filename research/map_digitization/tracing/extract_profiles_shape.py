"""Crop survey profiles from the master shapefile by inventory number (N_RGF)."""
import sys

import pyogrio

SHAPE = r"C:\FINAM\Conference\shapes\srr_all.shp"
INV = sys.argv[1] if len(sys.argv) > 1 else "384092"
OUT = rf"C:\FINAM\Conference\pipeline\tracing\out\profiles_{INV}.geojson"

df = None
for where in (f"N_RGF = '{INV}'", f"N_RGF = {INV}",
              f"INV_svod = '{INV}'", f"INV_svod = {INV}"):
    try:
        cand = pyogrio.read_dataframe(SHAPE, where=where, encoding="cp1251")
    except Exception as e:  # noqa: BLE001
        print(where, "->", str(e)[:90])
        continue
    print(where, "->", len(cand))
    if len(cand):
        df = cand
        break

if df is None or not len(df):
    sys.exit("no features found")

cols = [c for c in ("N_PROF", "Name", "N_RGF", "INV_svod", "DLINA_PROF",
                    "REGION", "NAME_DOC") if c in df.columns]
print(df[cols].head(25).to_string())
print("crs:", df.crs)
print("bounds:", df.total_bounds)
df.to_file(OUT, driver="GeoJSON")
print("saved:", OUT, len(df), "profiles")
