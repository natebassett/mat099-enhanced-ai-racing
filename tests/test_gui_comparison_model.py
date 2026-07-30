from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.comparison_model import (  # noqa: E402
    compare_runs,
    moment_comparison_reasons,
    nearest_snapshot_index,
    snapshot_times,
)
from gui.telemetry_model import TelemetrySnapshot  # noqa: E402


class GuiComparisonModelTests(unittest.TestCase):
    def test_nearest_snapshot_index_uses_closest_time(self):
        self.assertEqual(nearest_snapshot_index([0.0, 1.0, 2.0], 1.6), 2)
        self.assertEqual(nearest_snapshot_index([0.0, 1.0, 2.0], 1.4), 1)

    def test_snapshot_times_stay_continuous_across_lap_reset(self):
        snapshots = [
            _snapshot(cur_lap_time=93.5),
            _snapshot(cur_lap_time=94.0),
            _snapshot(cur_lap_time=0.2),
            _snapshot(cur_lap_time=0.8),
        ]

        self.assertEqual(snapshot_times(snapshots), [93.5, 94.0, 94.2, 94.8])

    def test_compare_runs_explains_best_lap_delta(self):
        summary = compare_runs(
            _run(1, best_lap=91.0, avg_speed=2.0),
            _run(2, best_lap=94.5, avg_speed=1.8),
            [_snapshot(cur_lap_time=0.0)],
            [_snapshot(cur_lap_time=0.0)],
        )

        self.assertEqual(summary.headline, "Run A was 3.500s faster than Run B.")
        self.assertIn("Best lap: Run A 91.000s; Run B 94.500s.", summary.details)

    def test_moment_comparison_is_plain_english(self):
        reasons = moment_comparison_reasons(
            _snapshot(speed_x=120.0, accel=0.8, brake=0.0, track_pos=0.1),
            _snapshot(speed_x=90.0, accel=0.0, brake=0.5, track_pos=0.7),
        )

        self.assertIn("Run A is 30.0 km/h faster at this point.", reasons)
        self.assertIn("Run A is accelerating, while Run B is braking hard.", reasons)
        self.assertIn("Run B is closer to the edge of the road.", reasons)


def _run(run_id: int, *, best_lap: float, avg_speed: float) -> dict[str, object]:
    return {
        "id": run_id,
        "agent_name": f"Agent {run_id}",
        "track": "g-track-3",
        "best_lap_time_seconds": best_lap,
        "avg_speed": avg_speed,
        "termination_reason": "target_laps_completed",
    }


def _snapshot(
    *,
    cur_lap_time: float = 0.0,
    speed_x: float = 0.0,
    accel: float = 0.0,
    brake: float = 0.0,
    track_pos: float = 0.0,
) -> TelemetrySnapshot:
    return TelemetrySnapshot.from_sample(
        {
            "step": 1,
            "cur_lap_time": cur_lap_time,
            "speed_x": speed_x,
            "accel": accel,
            "brake": brake,
            "track_pos": track_pos,
        }
    )


if __name__ == "__main__":
    unittest.main()
