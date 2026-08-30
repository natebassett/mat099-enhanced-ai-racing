import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from racing_line.smoothing import (
    build_smoothed_racing_line,
    circular_moving_average,
    periodic_second_difference_smooth,
)


BASE_LINE = PROJECT_ROOT / "data" / "racing_lines" / "g-track-3.json"
IMPROVED_LINE = (
    PROJECT_ROOT
    / "data"
    / "racing_lines"
    / "g-track-3-agent7-smooth-v1.json"
)
TRACK_XML = (
    PROJECT_ROOT / "torcs" / "tracks" / "road" / "g-track-3" / "g-track-3.xml"
)


class Agent7RacingLineTests(unittest.TestCase):
    def test_periodic_regularizer_reduces_second_difference(self):
        values = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float64)
        smoothed = periodic_second_difference_smooth(values, 30.0)
        before = np.roll(values, -1) - 2.0 * values + np.roll(values, 1)
        after = np.roll(smoothed, -1) - 2.0 * smoothed + np.roll(smoothed, 1)

        self.assertLess(np.sqrt(np.mean(after**2)), np.sqrt(np.mean(before**2)))
        self.assertAlmostEqual(float(np.mean(smoothed)), float(np.mean(values)))

    def test_circular_filter_has_no_endpoint_special_case(self):
        values = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        filtered = circular_moving_average(values, 3)

        np.testing.assert_allclose(filtered, [1 / 3, 1 / 3, 0.0, 0.0, 1 / 3])

    def test_improved_artifact_is_smoother_bounded_and_traceable(self):
        base_bytes = BASE_LINE.read_bytes()
        line = build_smoothed_racing_line(BASE_LINE, TRACK_XML)
        metadata = line.payload["optimizer"]
        before = metadata["quality_before"]
        after = metadata["quality_after"]

        self.assertEqual(BASE_LINE.read_bytes(), base_bytes)
        self.assertEqual(len(line.waypoints), 947)
        self.assertLess(
            after["target_second_difference_rms"],
            before["target_second_difference_rms"] * 0.60,
        )
        self.assertLess(
            after["curvature_step_rms"],
            before["curvature_step_rms"] * 0.20,
        )
        self.assertLess(
            after["maximum_curvature_step"],
            before["maximum_curvature_step"] * 0.10,
        )
        self.assertLess(after["maximum_abs_target_track_pos"], 0.77)
        self.assertLess(after["maximum_target_deviation"], 0.10)
        self.assertEqual(metadata["base_line"], BASE_LINE.name)

    def test_checked_in_artifact_matches_reproducible_generator(self):
        expected = build_smoothed_racing_line(BASE_LINE, TRACK_XML).payload
        actual = json.loads(IMPROVED_LINE.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)

    def test_saving_to_a_new_path_does_not_touch_the_base_line(self):
        base_bytes = BASE_LINE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "smooth.json"
            build_smoothed_racing_line(BASE_LINE, TRACK_XML).save(destination)
            self.assertTrue(destination.is_file())

        self.assertEqual(BASE_LINE.read_bytes(), base_bytes)


if __name__ == "__main__":
    unittest.main()
