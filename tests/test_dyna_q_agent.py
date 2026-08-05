from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.dyna_q_agent import (  # noqa: E402
    DynaQBaseAgent,
    DynaQFinalisedAgent,
    DynaQLearningAgent,
)


class DynaQAgentTests(unittest.TestCase):
    def test_real_transition_updates_q_value_and_model(self):
        agent = DynaQLearningAgent(seed=123, load_existing=False)
        agent.planning_steps = 0
        state = (0, 1, 2, 3, 4, 5, 2)
        next_state = (1, 1, 2, 3, 4, 5, 2)

        agent.learn_from_transition(state, 3, 10.0, next_state)

        self.assertGreater(agent.q_value(state, 3), 0.0)
        self.assertEqual(agent.real_updates, 1)
        self.assertEqual(agent.planning_updates, 0)
        self.assertIn("0,1,2,3,4,5,2|3", agent.model)

    def test_planning_replay_adds_extra_updates(self):
        agent = DynaQLearningAgent(seed=123, load_existing=False)
        agent.planning_steps = 4
        state = (0, 1, 2, 3, 4, 5, 2)
        next_state = (1, 1, 2, 3, 4, 5, 2)

        agent.learn_from_transition(state, 3, 10.0, next_state)

        self.assertEqual(agent.planning_updates, 4)
        self.assertGreater(agent.q_value(state, 3), agent.alpha * 10.0)

    def test_policy_save_and_finalised_load(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "policy.json"
            learning_agent = DynaQLearningAgent(
                seed=123,
                policy_path=policy_path,
                load_existing=False,
            )
            learning_agent.planning_steps = 0
            state = (0, 1, 2, 3, 4, 5, 2)
            learning_agent.learn_from_transition(state, 3, 8.0, state)
            learning_agent.save(policy_path)

            finalised_agent = DynaQFinalisedAgent(seed=999, policy_path=policy_path)

        self.assertEqual(finalised_agent.epsilon, 0.0)
        self.assertFalse(finalised_agent.learning_enabled)
        self.assertAlmostEqual(
            finalised_agent.q_value(state, 3),
            learning_agent.q_value(state, 3),
        )

    def test_finalised_agent_reports_missing_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                DynaQFinalisedAgent(policy_path=policy_path)

    def test_state_encoder_uses_distance_and_telemetry_bins(self):
        agent = DynaQBaseAgent(seed=1)
        state = agent.encode_state(
            {
                "distFromStart": 1421.0,
                "speedX": 121.0,
                "trackPos": -0.4,
                "angle": 0.2,
                "speedY": -5.0,
                "track": [80.0] * 8 + [110.0, 140.0, 90.0] + [70.0] * 8,
            }
        )

        self.assertEqual(len(state), 7)
        self.assertGreaterEqual(state[0], 20)
        self.assertLessEqual(state[0], 30)

    def test_reward_prefers_progress_over_stuck_state(self):
        agent = DynaQBaseAgent(seed=1)
        previous = _telemetry(distance=100.0, raced=100.0, speed=80.0)
        progressed = _telemetry(distance=112.0, raced=112.0, speed=92.0)
        stuck = _telemetry(distance=100.1, raced=100.1, speed=2.0)

        self.assertGreater(
            agent.calculate_reward(previous, progressed),
            agent.calculate_reward(previous, stuck),
        )


def _telemetry(*, distance: float, raced: float, speed: float) -> dict[str, object]:
    return {
        "distFromStart": distance,
        "distRaced": raced,
        "speedX": speed,
        "speedY": 0.0,
        "angle": 0.0,
        "trackPos": 0.0,
        "damage": 0.0,
        "track": [200.0] * 19,
    }


if __name__ == "__main__":
    unittest.main()
