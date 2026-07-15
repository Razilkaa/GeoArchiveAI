"""Final overlay: values read from crops, positions from CV boxes."""
import csv
import numpy as np
import cv2

SCRATCH = r"C:\Users\Finam\AppData\Local\Temp\claude\C--FINAM-Conference\aa89716f-6567-4ed8-82fb-4ac47e43d16c\scratchpad"
boxes = np.load(SCRATCH + r"\boxes.npy")
OX, OY = 2700, 2700  # tile origin in full map

# box index -> reading; None = fragment/unreadable; strings keep multi-values
readings = {
    0: None, 1: "2.45|2.46", 2: "2.72", 3: "2.77", 4: "2.74|2.77|-2.8", 5: "2.94",
    6: "2.97", 7: "40412", 8: "2.07", 9: None, 10: "2.47|2.51", 11: "2.52",
    12: None, 13: "2.66", 14: "2.64|2.63", 15: "2.45", 16: "2.36", 17: "2.30",
    18: "2.56|2.53", 19: "2.54", 20: "2.52", 21: "2.47", 22: None, 23: "2.60",
    24: "2.64", 25: "2.72", 26: "2.67", 27: "2.65|2.6?", 28: "2.60", 29: "2.54",
    30: "2.47", 31: "2.55", 32: "2.50", 33: None, 34: "2.80|2.2?", 35: "2.75",
    36: "2.67", 37: "2.62", 38: "2.71", 39: "2.77", 40: "2.70", 41: "2.80",
    42: "2.82", 43: "2.96", 44: "2.62|2.57", 45: "2.62", 46: "-2.74", 47: "2.72|2.74|2.76|2.69",
    48: "2.68", 49: "2.70", 50: "2.70", 51: "2.66", 52: "-2.8|-2.9", 53: "2.91|2.91",
    54: "3.03", 55: "3.08|3.02|3.02", 56: "3.02|2.79|3.08", 57: "2.76", 58: "2.69",
    59: "2.87", 60: "2.83", 61: "2.84", 62: "40419", 63: "2.82", 64: "3.10",
    65: "3.13|3.17", 66: "3.07|3.06|3.03", 67: "2.94", 68: "2.85", 69: "2.86|2.78",
    70: "2.57", 71: "2.67|2.5?", 72: "3.11", 73: "3.00", 74: "3.07", 75: "3.11",
    76: "2.96", 77: "2.84", 78: "-3.2", 79: "3.08?", 80: "311182", 81: "2.68", 82: None,
}

img = cv2.imread(SCRATCH + r"\tile_center.png")
rows = []
ok = frag = 0
for i, (x, y, w, h) in enumerate(boxes):
    r = readings.get(i)
    if r is None:
        cv2.rectangle(img, (x, y), (x + w, y + h), (180, 180, 180), 1)
        frag += 1
        continue
    is_profile = r.replace("?", "").isdigit() and len(r) >= 5
    color = (255, 0, 0) if is_profile else (0, 0, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, r.split("|")[0], (x, max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    ok += 1
    rows.append([i, r, OX + x + w // 2, OY + y + h // 2, "profile_id" if is_profile else "depth_km"])

cv2.imwrite(SCRATCH + r"\tile_overlay_v2.png", img)
with open(SCRATCH + r"\points_v2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["box_id", "reading", "px_full", "py_full", "kind"])
    w.writerows(rows)
print(f"read: {ok}, fragments: {frag}, total: {len(boxes)}")
