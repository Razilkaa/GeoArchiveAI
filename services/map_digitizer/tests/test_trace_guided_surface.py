from __future__ import annotations

import unittest

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from services.map_digitizer.trace_guided_surface import assign_traced_values


class TraceGuidedSurfaceTest(unittest.TestCase):
    def test_short_fragment_is_not_used_as_surface_constraint(self):
        axis = np.linspace(0, 200, 21)
        values = np.full((21, 21), 2.0)
        interpolator = RegularGridInterpolator((axis, axis), values)
        short = np.asarray([[10.0, 10.0], [50.0, 20.0], [80.0, 30.0]])

        assignments = assign_traced_values(
            [short], interpolator, interval=0.1, minimum_length=100.0
        )

        self.assertFalse(assignments[0]["accepted"])
        self.assertLess(assignments[0]["length_px"], 100.0)
        self.assertEqual(assignments[0]["inferred_value_km"], -2.0)


if __name__ == "__main__":
    unittest.main()
