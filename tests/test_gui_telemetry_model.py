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
    build_explanation_frame,
    explain_snapshot,
    explanation_event_key,
    format_explanation_entry,
    summarize_run_explanation,
)


class TelemetryModelTests(unittest.TestCase):
    def test_snapshot_normalizes_runner_sample(self):
        snapshot = TelemetrySnapshot.from_sample(
            {
                "step": "12",
                "speed_x": "87.5",
                "speed_y": "1.25",
                "angle": "-0.05",
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
                "track_sensors": [str(value) for value in range(19)],
                "cur_lap_time": "14.25",
                "last_lap_time": "",
                "dyna_q_learning_enabled": "true",
                "dyna_q_finalised": 0,
                "dyna_q_state": "1,2,3",
                "dyna_q_action_label": "straight / maintain",
                "dyna_q_action_source": "greedy",
                "dyna_q_reward": "2.5",
                "dyna_q_td_error": "-0.125",
                "dyna_q_selected_q": "4.75",
                "dyna_q_epsilon": "0.012",
                "dyna_q_q_states": "123",
                "dyna_q_model_states": "122",
                "dyna_q_real_updates": "55",
                "dyna_q_planning_updates": "880",
            }
        )

        self.assertEqual(snapshot.step, 12)
        self.assertEqual(snapshot.speed_kmh, 87.5)
        self.assertEqual(snapshot.speed_y_kmh, 1.25)
        self.assertEqual(snapshot.angle, -0.05)
        self.assertEqual(snapshot.gear, 3)
        self.assertEqual(snapshot.throttle_pct, 42.0)
        self.assertEqual(snapshot.brake_pct, 18.0)
        self.assertEqual(snapshot.steer, -0.125)
        self.assertEqual(snapshot.track_sensors, tuple(float(value) for value in range(19)))
        self.assertIsNone(snapshot.last_lap_time)
        self.assertTrue(snapshot.dyna_q_learning_enabled)
        self.assertFalse(snapshot.dyna_q_finalised)
        self.assertEqual(snapshot.dyna_q_state, "1,2,3")
        self.assertEqual(snapshot.dyna_q_action_label, "straight / maintain")
        self.assertEqual(snapshot.dyna_q_action_source, "greedy")
        self.assertEqual(snapshot.dyna_q_reward, 2.5)
        self.assertEqual(snapshot.dyna_q_td_error, -0.125)
        self.assertEqual(snapshot.dyna_q_selected_q, 4.75)
        self.assertEqual(snapshot.dyna_q_epsilon, 0.012)
        self.assertEqual(snapshot.dyna_q_q_states, 123)
        self.assertEqual(snapshot.dyna_q_model_states, 122)
        self.assertEqual(snapshot.dyna_q_real_updates, 55)
        self.assertEqual(snapshot.dyna_q_planning_updates, 880)

    def test_history_keeps_bounded_chart_series(self):
        history = TelemetryHistory(max_points=2)
        history.append(TelemetrySnapshot.from_sample({"step": 1, "speed_x": 10}))
        history.append(TelemetrySnapshot.from_sample({"step": 2, "speed_x": 20}))
        history.append(TelemetrySnapshot.from_sample({"step": 3}))

        self.assertEqual(history.steps(), [2, 3])
        self.assertEqual(history.values("speed_kmh")[0], 20.0)
        self.assertTrue(math.isnan(history.values("speed_kmh")[1]))

    def test_history_uses_lap_time_for_rolling_chart_series(self):
        history = TelemetryHistory()
        history.append(
            TelemetrySnapshot.from_sample(
                {"step": 1, "speed_x": 50, "cur_lap_time": 5.0}
            )
        )
        history.append(
            TelemetrySnapshot.from_sample(
                {"step": 2, "speed_x": 80, "cur_lap_time": 20.0}
            )
        )
        history.append(
            TelemetrySnapshot.from_sample(
                {"step": 3, "speed_x": 100, "cur_lap_time": 95.0}
            )
        )

        x_values, y_values = history.windowed_series("speed_kmh", 80.0)

        self.assertEqual(x_values, [20.0, 95.0])
        self.assertEqual(y_values, [80.0, 100.0])

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

    def test_explanation_frame_marks_stuck_risk_after_launch(self):
        snapshot = TelemetrySnapshot.from_sample(
            {
                "step": 820,
                "speed_x": 1.5,
                "brake": 0.12,
                "track_pos": -0.1,
                "front_sensor": 18,
                "cur_lap_time": 16.0,
            }
        )

        frame = build_explanation_frame(snapshot)

        self.assertEqual(frame.severity, "Stuck")
        self.assertEqual(frame.event_key, "stuck")
        self.assertIn("near zero", frame.headline)

    def test_run_summary_calls_out_stable_completion(self):
        snapshots = [
            TelemetrySnapshot.from_sample(
                {
                    "step": 1,
                    "speed_x": 0,
                    "accel": 0.8,
                    "cur_lap_time": 0.0,
                }
            ),
            TelemetrySnapshot.from_sample(
                {
                    "step": 100,
                    "speed_x": 120,
                    "accel": 0.4,
                    "track_pos": 0.12,
                    "front_sensor": 80,
                    "cur_lap_time": 20.0,
                }
            ),
        ]

        summary = summarize_run_explanation(
            snapshots,
            {
                "termination_reason": "target_laps_completed",
                "laps_completed": 1,
                "best_lap_time_seconds": 92.564,
            },
            run_id=78,
        )

        self.assertIn("Completed lap in 92.564s", summary)
        self.assertIn("Mostly stable", summary)
        self.assertIn("Saved as run #78", summary)


if __name__ == "__main__":
    unittest.main()
