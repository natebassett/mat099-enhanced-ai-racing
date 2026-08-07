from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_dyna_q_policy.py"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

spec = importlib.util.spec_from_file_location("evaluate_dyna_q_policy", SCRIPT_PATH)
evaluate_dyna_q_policy = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["evaluate_dyna_q_policy"] = evaluate_dyna_q_policy
spec.loader.exec_module(evaluate_dyna_q_policy)


class EvaluateDynaQPolicyTests(unittest.TestCase):
    def test_evaluation_summary_reports_progress_and_final_state(self):
        summary = evaluate_dyna_q_policy._evaluation_summary(
            Path("policy.json"),
            "g-track-3",
            {
                "termination_reason": "out of bounds",
                "steps": 120,
                "laps_completed": 0,
                "best_lap_time_seconds": None,
                "total_score": 10.0,
                "max_speed": 1.2,
                "avg_speed": 0.7,
                "off_track": 3,
                "telemetry_samples": [
                    {"dist_raced": 5.0, "speed_x": 0.0},
                    {
                        "dist_raced": 255.0,
                        "speed_x": 30.0,
                        "track_pos": 1.1,
                        "angle": -0.3,
                        "front_sensor": 44.0,
                    },
                ],
            },
        )

        self.assertEqual(summary.reason, "out of bounds")
        self.assertEqual(summary.progress_m, 250.0)
        self.assertEqual(summary.max_speed_kmh, 60.0)
        self.assertEqual(summary.avg_speed_kmh, 35.0)
        self.assertEqual(summary.final_speed, 30.0)
        self.assertEqual(summary.final_track_pos, 1.1)

    def test_summary_text_includes_key_evaluation_fields(self):
        summary = evaluate_dyna_q_policy.EvaluationSummary(
            policy_path="policy.json",
            track="g-track-3",
            reason="stuck",
            steps=200,
            laps=0,
            best_lap=None,
            progress_m=400.0,
            progress_percent=14.1,
            total_score=20.0,
            max_speed_kmh=70.0,
            avg_speed_kmh=33.0,
            off_track=4,
            final_speed=0.0,
            final_track_pos=1.05,
            final_angle=-0.2,
            final_front=-1.0,
        )

        text = evaluate_dyna_q_policy._summary_text(summary)

        self.assertIn("Dyna-Q Policy Evaluation", text)
        self.assertIn("Progress: 400m (14.1%)", text)
        self.assertIn("Off track: 4", text)
        self.assertIn("trackPos= +1.05", text)

    def test_write_telemetry_files_outputs_json_and_csv(self):
        summary = evaluate_dyna_q_policy.EvaluationSummary(
            policy_path="policy.json",
            track="g-track-3",
            reason="stuck",
            steps=1,
            laps=0,
            best_lap=None,
            progress_m=10.0,
            progress_percent=0.4,
            total_score=1.0,
            max_speed_kmh=20.0,
            avg_speed_kmh=10.0,
            off_track=0,
            final_speed=10.0,
            final_track_pos=0.0,
            final_angle=0.0,
            final_front=100.0,
        )
        results = {
            "telemetry_samples": [
                {
                    "step": 1,
                    "dist_raced": 10.0,
                    "track_sensors": [1.0, 2.0, 3.0],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path = evaluate_dyna_q_policy._write_telemetry_files(
                Path(directory),
                "g-track-3",
                summary,
                results,
            )
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertEqual(payload["summary"]["progress_m"], 10.0)
        self.assertIn("track_sensors", csv_text)
        self.assertIn('"[1.0, 2.0, 3.0]"', csv_text)


if __name__ == "__main__":
    unittest.main()
