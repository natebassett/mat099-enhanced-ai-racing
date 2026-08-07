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
    edge_recovery_zone,
    stable_open_corridor,
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

    def test_learning_agent_ignores_incompatible_policy_action_space(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "old_policy.json"
            policy_path.write_text(
                '{"algorithm":"dyna_q","action_labels":["old action"],"q_values":{"0|0":5}}',
                encoding="utf-8",
            )

            agent = DynaQLearningAgent(policy_path=policy_path, load_existing=True)

        self.assertFalse(agent.load_existing)
        self.assertEqual(agent.q_values, {})

    def test_finalised_agent_rejects_incompatible_policy_action_space(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "old_policy.json"
            policy_path.write_text(
                '{"algorithm":"dyna_q","action_labels":["old action"],"q_values":{"0|0":5}}',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
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

        self.assertEqual(len(state), 9)
        self.assertGreaterEqual(state[0], 20)
        self.assertLessEqual(state[0], 30)

    def test_state_encoder_tracks_corridor_opening_and_sensor_balance(self):
        agent = DynaQBaseAgent(seed=1)
        left_open = [40.0] * 5 + [180.0] * 4
        centre = [90.0]
        right_tight = [25.0] * 4 + [40.0] * 5
        state = agent.encode_state(
            {
                **_telemetry(distance=250.0, raced=250.0, speed=88.0),
                "track": left_open + centre + right_tight,
            }
        )

        self.assertGreaterEqual(state[7], 3)
        self.assertGreaterEqual(state[8], 3)

    def test_reward_prefers_progress_over_stuck_state(self):
        agent = DynaQBaseAgent(seed=1)
        previous = _telemetry(distance=100.0, raced=100.0, speed=80.0)
        progressed = _telemetry(distance=112.0, raced=112.0, speed=92.0)
        stuck = _telemetry(distance=100.1, raced=100.1, speed=2.0)

        self.assertGreater(
            agent.calculate_reward(previous, progressed),
            agent.calculate_reward(previous, stuck),
        )

    def test_reward_prefers_recovering_from_edge(self):
        agent = DynaQBaseAgent(seed=1)
        previous = {
            **_telemetry(distance=100.0, raced=100.0, speed=72.0),
            "trackPos": 0.82,
        }
        recovering = {
            **_telemetry(distance=104.0, raced=104.0, speed=74.0),
            "trackPos": 0.62,
        }
        drifting_wide = {
            **_telemetry(distance=104.0, raced=104.0, speed=74.0),
            "trackPos": 0.94,
        }

        self.assertGreater(
            agent.calculate_reward(previous, recovering),
            agent.calculate_reward(previous, drifting_wide),
        )

    def test_reward_penalises_outward_edge_drift(self):
        agent = DynaQBaseAgent(seed=1)
        previous = {
            **_telemetry(distance=100.0, raced=100.0, speed=78.0),
            "trackPos": 0.68,
        }
        outward = {
            **_telemetry(distance=103.0, raced=103.0, speed=78.0),
            "trackPos": 0.72,
            "speedY": 7.0,
        }
        inward = {
            **_telemetry(distance=103.0, raced=103.0, speed=78.0),
            "trackPos": 0.72,
            "speedY": -7.0,
        }

        self.assertLess(
            agent.calculate_reward(previous, outward),
            agent.calculate_reward(previous, inward),
        )

    def test_reward_heavily_penalises_crossing_outside_track(self):
        agent = DynaQBaseAgent(seed=1)
        previous = {
            **_telemetry(distance=100.0, raced=100.0, speed=52.0),
            "trackPos": 0.96,
        }
        outside = {
            **_telemetry(distance=101.0, raced=101.0, speed=12.0),
            "trackPos": 1.08,
            "track": [-1.0] * 19,
        }
        inside = {
            **_telemetry(distance=101.0, raced=101.0, speed=12.0),
            "trackPos": 0.84,
        }

        self.assertLess(
            agent.calculate_reward(previous, outside),
            agent.calculate_reward(previous, inside) - 80.0,
        )

    def test_launch_action_mask_prevents_brake_when_stable_and_slow(self):
        agent = DynaQBaseAgent(seed=1)
        state = (0, 0, 4, 4, 3, 7, 2)
        agent.epsilon_start = 0.0
        agent.epsilon_min = 0.0
        agent.q_values[agent._transition_key(state, 14)] = 100.0
        agent.q_values[agent._transition_key(state, 0)] = 1.0

        action = agent.choose_action(
            state,
            _telemetry(distance=0.0, raced=0.0, speed=4.0),
        )

        self.assertEqual(action, 0)

    def test_front_danger_mask_allows_brake_choices(self):
        agent = DynaQBaseAgent(seed=1)
        telemetry = _telemetry(distance=0.0, raced=0.0, speed=90.0)
        telemetry["track"] = [200.0] * 9 + [10.0] + [200.0] * 9

        self.assertIn(14, agent.available_actions(telemetry))
        self.assertIn(16, agent.available_actions(telemetry))
        self.assertNotIn(0, agent.available_actions(telemetry))

    def test_high_speed_mask_blocks_rotation_actions(self):
        agent = DynaQBaseAgent(seed=1)
        actions = agent.available_actions(
            _telemetry(distance=0.0, raced=0.0, speed=170.0)
        )

        self.assertNotIn(7, actions)
        self.assertNotIn(8, actions)
        self.assertIn(0, actions)

    def test_right_edge_recovery_mask_steers_back_left(self):
        agent = DynaQBaseAgent(seed=1)
        telemetry = _telemetry(distance=0.0, raced=0.0, speed=70.0)
        telemetry["trackPos"] = 0.95

        actions = agent.available_actions(telemetry)

        self.assertIn(7, actions)
        self.assertIn(10, actions)
        self.assertIn(13, actions)
        self.assertIn(14, actions)
        self.assertIn(17, actions)
        self.assertIn(19, actions)
        self.assertIn(21, actions)
        self.assertNotIn(0, actions)
        self.assertNotIn(8, actions)

    def test_low_speed_edge_uses_recovery_actions_before_launch_actions(self):
        agent = DynaQBaseAgent(seed=1)
        telemetry = _telemetry(distance=0.0, raced=0.0, speed=12.0)
        telemetry["trackPos"] = 0.70

        actions = agent.available_actions(telemetry)

        self.assertIn(17, actions)
        self.assertNotIn(0, actions)

    def test_left_edge_recovery_mask_steers_back_right(self):
        agent = DynaQBaseAgent(seed=1)
        telemetry = _telemetry(distance=0.0, raced=0.0, speed=70.0)
        telemetry["trackPos"] = -0.95

        actions = agent.available_actions(telemetry)

        self.assertIn(8, actions)
        self.assertIn(12, actions)
        self.assertIn(15, actions)
        self.assertIn(18, actions)
        self.assertIn(20, actions)
        self.assertIn(22, actions)
        self.assertNotIn(0, actions)
        self.assertNotIn(7, actions)

    def test_low_speed_controls_keep_launch_throttle_available(self):
        agent = DynaQBaseAgent(seed=1)
        controls = agent.action_to_controls(
            11,
            _telemetry(distance=0.0, raced=0.0, speed=3.0),
        )

        self.assertGreaterEqual(controls["accel"], 0.20)
        self.assertEqual(controls["brake"], 0.0)

    def test_edge_controls_bias_steering_inward(self):
        agent = DynaQBaseAgent(seed=1)
        controls = agent.action_to_controls(
            6,
            {
                **_telemetry(distance=0.0, raced=0.0, speed=82.0),
                "trackPos": 0.82,
            },
        )

        self.assertLess(controls["steer"], 0.34)
        self.assertLessEqual(controls["accel"], 0.42)

    def test_slow_edge_recovery_drive_keeps_throttle_available(self):
        agent = DynaQBaseAgent(seed=1)
        controls = agent.action_to_controls(
            17,
            {
                **_telemetry(distance=0.0, raced=0.0, speed=12.0),
                "trackPos": 0.95,
            },
        )

        self.assertLess(controls["steer"], 0.0)
        self.assertGreaterEqual(controls["accel"], 0.20)
        self.assertEqual(controls["brake"], 0.0)

    def test_hard_edge_recovery_overrides_low_speed_brake(self):
        agent = DynaQBaseAgent(seed=1)
        controls = agent.action_to_controls(
            14,
            {
                **_telemetry(distance=0.0, raced=0.0, speed=8.0),
                "trackPos": 1.14,
                "track": [-1.0] * 19,
            },
        )

        self.assertLess(controls["steer"], -0.10)
        self.assertGreaterEqual(controls["accel"], 0.20)
        self.assertEqual(controls["brake"], 0.0)

    def test_hard_left_edge_recovery_steers_right(self):
        agent = DynaQBaseAgent(seed=1)
        controls = agent.action_to_controls(
            14,
            {
                **_telemetry(distance=0.0, raced=0.0, speed=8.0),
                "trackPos": -1.14,
                "track": [-1.0] * 19,
            },
        )

        self.assertGreater(controls["steer"], 0.10)
        self.assertGreaterEqual(controls["accel"], 0.20)
        self.assertEqual(controls["brake"], 0.0)

    def test_reward_penalises_unneeded_stable_open_braking(self):
        agent = DynaQBaseAgent(seed=1)
        previous = _telemetry(distance=100.0, raced=100.0, speed=68.0)
        current = _telemetry(distance=104.0, raced=104.0, speed=70.0)

        maintain_reward = agent.calculate_reward(previous, current, 9)
        brake_reward = agent.calculate_reward(previous, current, 14)

        self.assertGreater(maintain_reward, brake_reward)

    def test_stable_open_corridor_requires_safe_alignment(self):
        self.assertTrue(
            stable_open_corridor(
                speed=70.0,
                track_position=0.2,
                angle=0.1,
                side_speed=1.0,
                front=150.0,
            )
        )
        self.assertFalse(
            stable_open_corridor(
                speed=70.0,
                track_position=0.7,
                angle=0.1,
                side_speed=1.0,
                front=150.0,
            )
        )

    def test_edge_recovery_zone_detects_mild_off_track_sensor_failure(self):
        self.assertTrue(edge_recovery_zone(1.04, 200.0))
        self.assertTrue(edge_recovery_zone(0.92, -1.0))
        self.assertFalse(edge_recovery_zone(0.80, -1.0))

    def test_reward_prefers_hard_edge_recovery_over_drifting_farther(self):
        agent = DynaQBaseAgent(seed=1)
        previous = {
            **_telemetry(distance=100.0, raced=100.0, speed=18.0),
            "trackPos": 1.12,
            "track": [-1.0] * 19,
        }
        recovering = {
            **_telemetry(distance=101.0, raced=101.0, speed=20.0),
            "trackPos": 1.04,
            "track": [-1.0] * 19,
        }
        drifting_farther = {
            **_telemetry(distance=101.0, raced=101.0, speed=20.0),
            "trackPos": 1.18,
            "track": [-1.0] * 19,
        }

        self.assertGreater(
            agent.calculate_reward(previous, recovering, 17),
            agent.calculate_reward(previous, drifting_farther, 14),
        )

    def test_stuck_shutdown_applies_terminal_learning_penalty(self):
        agent = DynaQLearningAgent(seed=123, load_existing=False)
        agent.planning_steps = 0
        previous = _telemetry(distance=100.0, raced=100.0, speed=0.0)
        current = _telemetry(distance=100.0, raced=100.0, speed=0.0)
        previous_state = agent.encode_state(previous)
        previous_action = 0
        agent.previous_state = previous_state
        agent.previous_action_index = previous_action
        agent.previous_telemetry = previous
        agent.stuck_counter = 260

        action = agent.act(None, current)

        self.assertTrue(action["terminate"])
        self.assertEqual(action["termination_reason"], "car stuck")
        self.assertLess(agent.q_value(previous_state, previous_action), 0.0)

    def test_hard_track_position_limit_shutdowns_as_out_of_bounds(self):
        agent = DynaQBaseAgent(seed=1)
        telemetry = {
            **_telemetry(distance=100.0, raced=100.0, speed=0.0),
            "trackPos": 1.33,
            "track": [-1.0] * 19,
        }

        shutdown, reason = agent.should_shutdown(telemetry)

        self.assertTrue(shutdown)
        self.assertEqual(reason, "out of bounds")

    def test_slow_edge_position_can_attempt_recovery_before_shutdown(self):
        agent = DynaQBaseAgent(seed=1)
        telemetry = {
            **_telemetry(distance=100.0, raced=100.0, speed=4.0),
            "trackPos": 1.25,
            "angle": 0.25,
            "track": [-1.0] * 19,
        }

        shutdown, reason = agent.should_shutdown(telemetry)

        self.assertFalse(shutdown)
        self.assertEqual(reason, "")

    def test_light_off_track_transition_can_learn_recovery(self):
        agent = DynaQBaseAgent(seed=1)
        previous = _telemetry(distance=100.0, raced=100.0, speed=20.0)
        current = {
            **_telemetry(distance=101.0, raced=101.0, speed=18.0),
            "trackPos": 1.10,
            "track": [-1.0] * 19,
        }

        self.assertFalse(agent.is_terminal_transition(previous, current))

    def test_hard_off_track_transition_remains_terminal(self):
        agent = DynaQBaseAgent(seed=1)
        previous = _telemetry(distance=100.0, raced=100.0, speed=20.0)
        current = {
            **_telemetry(distance=101.0, raced=101.0, speed=18.0),
            "trackPos": 1.25,
            "track": [-1.0] * 19,
        }

        self.assertTrue(agent.is_terminal_transition(previous, current))

    def test_no_progress_shutdown_applies_terminal_learning_penalty(self):
        agent = DynaQLearningAgent(seed=123, load_existing=False)
        agent.planning_steps = 0
        previous = _telemetry(distance=100.0, raced=100.0, speed=2.0)
        current = _telemetry(distance=100.0, raced=100.0, speed=2.0)
        previous_state = agent.encode_state(previous)
        previous_action = 0
        agent.previous_state = previous_state
        agent.previous_action_index = previous_action
        agent.previous_telemetry = previous
        agent.progress_stall_counter = 300

        action = agent.act(None, current)

        self.assertTrue(action["terminate"])
        self.assertEqual(action["termination_reason"], "no progress")
        self.assertLess(agent.q_value(previous_state, previous_action), 0.0)

    def test_exploration_rate_decays_with_training_steps(self):
        agent = DynaQBaseAgent(seed=1)
        agent.epsilon_start = 0.24
        agent.epsilon_min = 0.04
        agent.epsilon_decay_steps = 100

        initial = agent.exploration_rate()
        agent.steps_trained = 50
        middle = agent.exploration_rate()
        agent.steps_trained = 200
        final = agent.exploration_rate()

        self.assertGreater(initial, middle)
        self.assertGreater(middle, final)
        self.assertEqual(final, 0.04)

    def test_exploration_override_temporarily_controls_epsilon(self):
        agent = DynaQBaseAgent(seed=1)
        agent.steps_trained = 999999
        agent.exploration_override = 0.12

        self.assertEqual(agent.exploration_rate(), 0.12)


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
