from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_dyna_q_policy.py"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

spec = importlib.util.spec_from_file_location("train_dyna_q_policy", SCRIPT_PATH)
train_dyna_q_policy = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["train_dyna_q_policy"] = train_dyna_q_policy
spec.loader.exec_module(train_dyna_q_policy)


class TrainDynaQPolicyTests(unittest.TestCase):
    def test_episode_progress_uses_dist_raced_delta(self):
        progress = train_dyna_q_policy._episode_progress_m(
            {
                "telemetry_samples": [
                    {"dist_raced": 10.0},
                    {"dist_raced": 40.0},
                    {"dist_raced": 95.0},
                ]
            },
            track_length_m=1000.0,
        )

        self.assertEqual(progress, 85.0)

    def test_episode_progress_wraps_dist_from_start(self):
        progress = train_dyna_q_policy._episode_progress_m(
            {
                "telemetry_samples": [
                    {"dist_from_start": 950.0},
                    {"dist_from_start": 980.0},
                    {"dist_from_start": 20.0},
                    {"dist_from_start": 80.0},
                ]
            },
            track_length_m=1000.0,
        )

        self.assertEqual(progress, 130.0)

    def test_dashboard_shows_progress_and_hides_old_rows(self):
        summaries = [
            train_dyna_q_policy.EpisodeSummary(
                episode=index,
                reason="stuck",
                steps=100 + index,
                laps=0,
                best_lap=None,
                progress_m=float(index * 10),
                progress_percent=float(index),
                q_states=index,
                epsilon=0.2,
            )
            for index in range(1, train_dyna_q_policy.VISIBLE_EPISODE_ROWS + 3)
        ]

        dashboard = train_dyna_q_policy._dashboard_text(
            summaries,
            total_episodes=30,
            policy_path=Path("policy.json"),
            status="training",
        )

        self.assertIn("Best progress", dashboard)
        self.assertIn("Progress", dashboard)
        self.assertIn("Lap %", dashboard)
        self.assertIn("earlier episode rows hidden", dashboard)
        self.assertIn(f"{summaries[-1].progress_m:7.0f}m", dashboard)

    def test_best_policy_checkpoint_only_updates_on_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"a": 1}}),
                encoding="utf-8",
            )

            first_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                250.0,
            )
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"b": 2}}),
                encoding="utf-8",
            )
            worse_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                200.0,
            )
            better_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                300.0,
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(first_update)
        self.assertFalse(worse_update)
        self.assertTrue(better_update)
        self.assertEqual(best_payload["best_progress_m"], 300.0)
        self.assertEqual(best_payload["q_values"], {"b": 2})

    def test_resume_best_policy_copies_best_and_saves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            previous_path = Path(directory) / "previous.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"old": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_progress_m": 2032.0,
                        "q_values": {"best": 2},
                    }
                ),
                encoding="utf-8",
            )

            previous_saved = train_dyna_q_policy._resume_best_policy(
                policy_path=policy_path,
                best_policy_path=best_path,
                previous_policy_path=previous_path,
                save_previous=True,
            )
            latest_payload = json.loads(policy_path.read_text(encoding="utf-8"))
            previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))

        self.assertTrue(previous_saved)
        self.assertEqual(latest_payload["q_values"], {"best": 2})
        self.assertEqual(latest_payload["checkpoint_type"], "latest")
        self.assertEqual(previous_payload["q_values"], {"old": 1})
        self.assertEqual(previous_payload["checkpoint_type"], "previous")


if __name__ == "__main__":
    unittest.main()
