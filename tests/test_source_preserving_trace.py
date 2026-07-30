import cv2
import numpy as np

from services.map_digitizer.source_preserving_trace import trace_source_geometry


def test_long_profile_is_removed_but_curved_contour_is_preserved():
    image = np.full((1200, 1200), 255, dtype=np.uint8)
    cv2.line(image, (100, 180), (1100, 180), 0, 3)
    cv2.ellipse(image, (600, 720), (330, 210), 0, 15, 345, 0, 3)

    geometry = trace_source_geometry(image)

    assert np.count_nonzero(geometry["profile_mask"][170:190]) > 0
    assert geometry["polylines"]
    assert any(
        np.asarray(line)[:, 1].max() > 850
        and np.asarray(line)[:, 1].min() < 600
        for line in geometry["polylines"]
    )
