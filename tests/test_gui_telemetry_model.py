from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.telemetry_model import (  # noqa: E402
    TelemetryHistory,
    TelemetrySnapshot,
    explain_snapshot,
    explanation_event_key,
    format_explanation_entry,
)


class TelemetryModelTests(unittest.TestCase):
    def test_snapshot_normalizes_runner_sample(self):
        snapshot = TelemetrySnapshot.from_sample(
            {
                "step": "12",
                "speed_x": "87.5",
                "gear": "3",
                "accel": "0.42",
                "brake": "0.18",
                "steer": "-0.125",
                "track_pos": "0.33",
                "damage": "4",
                "reward": "1.5",
                "off_track": 0,
                "front_sensor": "30",
                "min_track_sensor": "12",
                "cur_lap_time": "14.25",
                "last_lap_time": "",
            }
        )

        self.assertEqual(snapshot.step, 12)
        self.assertEqual(snapshot.speed_kmh, 87.5)
        self.assertEqual(snapshot.gear, 3)
        self.assertEqual(snapshot.throttle_pct, 42.0)
        self.assertEqual(snapshot.brake_pct, 18.0)
        self.assertEqual(snapshot.steer, -0.125)
        self.assertIsNone(snapshot.last_lap_time)

    def test_history_keeps_bounded_chart_series(self):
        history = TelemetryHistory(max_points=2)
        history.append(TelemetrySnapshot.from_sample({"step": 1, "speed_x": 10}))
        history.append(TelemetrySnapshot.from_sample({"step": 2, "speed_x": 20}))
        history.append(TelemetrySnapshot.from_sample({"step": 3}))

        self.assertEqual(history.steps(), [2, 3])
        self.assertEqual(history.values("speed_kmh")[0], 20.0)
        self.assertTrue(math.isnan(history.values("speed_kmh")[1]))

    def test_explanation_calls_out_edge_and_braking(self):
        snapshot = TelemetrySnapshot.from_sample(
            {
                "step": 4,
                "speed_x": 55,
                "brake": 0.3,
                "steer": 0.2,
                "track_pos": 0.95,
                "front_sensor": 6,
            }
        )

        explanation = explain_snapshot(snapshot)

        self.assertIn("track edge", explanation)
        self.assertIn("Braking at 30%", explanation)
        self.assertIn("Steering right", explanation)

    def test_explanation_entry_is_timestamped_for_scannable_log(self):
        snapshot = TelemetrySnapshot.from_sample(
            {
                "step": 125,
                "speed_x": 101.25,
                "gear": 4,
                "track_pos": -0.22,
                "steer": -0.1,
                "accel": 0.5,
                "cur_lap_time": 18.5,
            }
        )

        entry = format_explanation_entry(snapshot)

        self.assertIn("[lap 18.50s | step 125]", entry)
        self.assertIn("101.2 km/h", entry)
        self.assertIn("trackPos -0.220", entry)

    def test_explanation_event_key_prioritizes_safety_events(self):
        snapshot = TelemetrySnapshot.from_sample(
            {
                "step": 8,
                "speed_x": 40,
                "accel": 0.6,
                "track_pos": 0.95,
                "front_sensor": 4,
            }
        )

        self.assertEqual(explanation_event_key(snapshot), "very_limited_road")


if __name__ == "__main__":
    unittest.main()
