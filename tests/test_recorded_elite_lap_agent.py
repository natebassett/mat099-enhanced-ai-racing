from __future__ import annotations

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

from agents.recorded_elite_lap_agent import RecordedEliteLapAgent


def telemetry(**overrides):
    value = {
        "speedX": 40.0,
        "trackPos": 0.0,
        "angle": 0.0,
        "damage": 0.0,
        "gear": 2,
        "rpm": 5_000.0,
    }
    value.update(overrides)
    return value


class RecordedEliteLapAgentTests(unittest.TestCase):
    def _artifacts(self, directory: str, actions: np.ndarray):
        root = Path(directory)
        actions_path = root / "actions.npz"
        metadata_path = root / "lap.json"
        np.savez_compressed(
            actions_path,
            actions=np.asarray(actions, dtype=np.float32),
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "format_version": "agent8_recorded_action_demo_v1",
                    "episode_transition_count": len(actions),
                    "episode": {
                        "steps": len(actions),
                        "laps_completed": 1,
                        "best_lap_time_seconds": 83.62,
                    },
                }
            ),
            encoding="utf-8",
        )
        return actions_path, metadata_path

    def test_replays_recorded_actions_in_order_and_stops_at_end(self):
        actions = np.asarray([[0.1, 0.8], [-0.2, -0.4]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            actions_path, metadata_path = self._artifacts(directory, actions)
            agent = RecordedEliteLapAgent(
                actions_path=actions_path,
                metadata_path=metadata_path,
            )

            first = agent.act(None, telemetry())
            second = agent.act(None, telemetry())
            exhausted = agent.act(None, telemetry())

        self.assertAlmostEqual(first["steer"], 0.1)
        self.assertAlmostEqual(first["accel"], 0.8)
        self.assertEqual(first["brake"], 0.0)
        self.assertAlmostEqual(second["steer"], -0.2)
        self.assertEqual(second["accel"], 0.0)
        self.assertAlmostEqual(second["brake"], 0.4)
        self.assertTrue(exhausted["terminate"])
        self.assertEqual(
            exhausted["termination_reason"],
            "recorded actions exhausted",
        )

    def test_bundled_recording_is_complete(self):
        agent = RecordedEliteLapAgent()

        self.assertEqual(agent.actions.shape, (4040, 2))
        self.assertTrue(np.all(np.isfinite(agent.actions)))
        self.assertEqual(agent.config["recorded_action_count"], 4040)
        self.assertEqual(agent.config["recorded_lap_time_seconds"], 83.62)

    def test_metadata_marks_playback_as_non_policy_demonstration(self):
        actions = np.asarray([[0.0, 0.5]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            actions_path, metadata_path = self._artifacts(directory, actions)
            agent = RecordedEliteLapAgent(
                actions_path=actions_path,
                metadata_path=metadata_path,
            )

            config = agent.config

        self.assertEqual(
            config["control_mode"],
            "open_loop_recorded_action_playback",
        )
        self.assertFalse(config["evaluation_eligible"])
        self.assertFalse(config["neural_policy_active"])
        self.assertEqual(config["recorded_lap_time_seconds"], 83.62)
        self.assertEqual(config["recorded_action_count"], 1)

    def test_reset_rewinds_the_recording(self):
        actions = np.asarray([[0.3, 0.6], [-0.4, 0.2]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            actions_path, metadata_path = self._artifacts(directory, actions)
            agent = RecordedEliteLapAgent(
                actions_path=actions_path,
                metadata_path=metadata_path,
            )
            agent.act(None, telemetry())
            agent.reset()

            replayed_first = agent.act(None, telemetry())

        self.assertAlmostEqual(replayed_first["steer"], 0.3)
        self.assertAlmostEqual(replayed_first["accel"], 0.6)


if __name__ == "__main__":
    unittest.main()
