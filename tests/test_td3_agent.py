from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_td3_agent.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

from agents.td3_agent import (  # noqa: E402
    AGENT6_ACTION_VERSION,
    AGENT6_MODEL_FAMILY,
    AGENT6_OBSERVATION_VERSION,
    FEATURE_NAMES,
    Td3ScratchAgent,
    build_td3_observation,
    decode_td3_action,
    episode_metadata_is_deterministic_probe,
    episode_metadata_is_policy_controlled,
    metadata_path_for_policy,
    policy_checkpoint_is_verified,
    policy_contract_mismatches,
    rate_limit_td3_action,
)

spec = importlib.util.spec_from_file_location("train_td3_agent", SCRIPT_PATH)
train_td3_agent = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["train_td3_agent"] = train_td3_agent
spec.loader.exec_module(train_td3_agent)


class DummyPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.seen_observation = None
        self.seen_deterministic = None

    def predict(self, observation, deterministic=True):
        self.seen_observation = np.asarray(observation)
        self.seen_deterministic = deterministic
        return self.action, None


class Td3ScratchAgentTests(unittest.TestCase):
    def test_observation_is_raw_telemetry_contract_without_teacher_features(self):
        observation = build_td3_observation(
            _telemetry(speed=123.0, track_pos=0.24),
            {"steer": 0.2, "accel": 0.4, "brake": 0.0},
        )

        self.assertEqual(len(observation), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in observation))
        self.assertTrue(all(-1.000001 <= value <= 1.000001 for value in observation))
        self.assertNotIn("teacher", " ".join(FEATURE_NAMES).casefold())
        self.assertNotIn("racing_line", " ".join(FEATURE_NAMES).casefold())

    def test_decode_action_maps_td3_outputs_to_exclusive_pedals(self):
        action = decode_td3_action([2.0, 0.8, 0.4])

        self.assertEqual(action["steer"], 0.90)
        self.assertEqual(action["accel"], 0.8)
        self.assertEqual(action["brake"], 0.0)

        braking = decode_td3_action([-2.0, 0.2, 0.7])
        self.assertEqual(braking["steer"], -0.90)
        self.assertEqual(braking["accel"], 0.0)
        self.assertEqual(braking["brake"], 0.7)

    def test_agent_config_declares_reward_only_learning(self):
        agent = Td3ScratchAgent(policy_path=Path("missing.zip"))
        config = agent.config

        self.assertEqual(config["algorithm"], "TD3")
        self.assertEqual(config["teacher"], None)
        self.assertFalse(config["behaviour_cloning"])
        self.assertFalse(config["racing_line_imitation"])
        self.assertTrue(config["automatic_gear_shift"])
        self.assertTrue(config["actuator_rate_limited"])
        self.assertFalse(agent.policy_loaded)

    def test_policy_prediction_drives_direct_controls(self):
        policy = DummyPolicy([0.5, 0.7, -0.2])
        agent = Td3ScratchAgent(policy=policy, deterministic=True)

        action = agent.act(None, _telemetry(speed=90.0))

        self.assertTrue(agent.policy_loaded)
        self.assertEqual(policy.seen_observation.shape, (len(FEATURE_NAMES),))
        self.assertTrue(policy.seen_deterministic)
        self.assertAlmostEqual(action["steer"], 0.09)
        self.assertAlmostEqual(action["accel"], 0.16)
        self.assertEqual(action["brake"], 0.0)
        self.assertIn(action["gear"], range(1, 7))

    def test_rate_limiter_smooths_actuator_changes(self):
        limited = rate_limit_td3_action(
            {"steer": 0.90, "accel": 1.0, "brake": 0.0},
            {"steer": 0.0, "accel": 0.0, "brake": 0.0},
            speed_kmh=160.0,
        )
        self.assertAlmostEqual(limited["steer"], 0.045)
        self.assertAlmostEqual(limited["accel"], 0.16)
        self.assertEqual(limited["brake"], 0.0)

        braking = rate_limit_td3_action(
            {"steer": 0.0, "accel": 0.0, "brake": 1.0},
            {"steer": 0.2, "accel": 0.8, "brake": 0.0},
            speed_kmh=120.0,
        )
        self.assertEqual(braking["accel"], 0.0)
        self.assertAlmostEqual(braking["brake"], 0.40)

    def test_policy_metadata_contract_detects_sac_or_old_models(self):
        mismatches = policy_contract_mismatches(
            {
                "model_family": "agent5_hybrid_sac",
                "observation_version": "agent5_observation",
                "action_version": "agent5_action",
                "feature_names": [],
                "action_shape": [5],
            }
        )

        self.assertIn("model_family", mismatches)
        self.assertIn("observation_version", mismatches)
        self.assertIn("action_shape", mismatches)

    def test_metadata_path_uses_sidecar_json(self):
        self.assertEqual(
            metadata_path_for_policy(Path("agent6.zip")),
            Path("agent6.metadata.json"),
        )

    def test_verified_checkpoint_requires_deterministic_probe_metadata(self):
        self.assertFalse(episode_metadata_is_policy_controlled({"distance_m": 182.3}))
        self.assertTrue(
            episode_metadata_is_policy_controlled(
                {"episode_summary": {"policy_controlled": True}}
            )
        )
        self.assertFalse(
            episode_metadata_is_deterministic_probe(
                {"episode_summary": {"policy_controlled": True}}
            )
        )
        self.assertTrue(
            episode_metadata_is_deterministic_probe(
                {
                    "episode_summary": {
                        "policy_controlled": True,
                        "deterministic_probe": True,
                    }
                }
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "policy.zip"
            policy_path.touch()
            metadata_path_for_policy(policy_path).write_text(
                (
                    '{"best_distance_episode": {"episode_summary": '
                    '{"policy_controlled": true, "deterministic_probe": true}}}'
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                policy_checkpoint_is_verified(
                    policy_path,
                    "best_distance_episode",
                )
            )


class Td3ScratchTrainingUtilityTests(unittest.TestCase):
    def test_reward_prefers_forward_progress_and_new_distance(self):
        previous = _telemetry(distance=100.0, raced=100.0, speed=60.0)
        progressed = _telemetry(distance=108.0, raced=108.0, speed=76.0)
        stuck = _telemetry(distance=100.1, raced=100.1, speed=2.0)
        stage = train_td3_agent.CURRICULUM[0]

        good_reward = train_td3_agent.calculate_td3_reward(
            progressed,
            {"accel": 0.5, "brake": 0.0},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=108.0,
            previous_furthest_distance_m=100.0,
        )
        stuck_reward = train_td3_agent.calculate_td3_reward(
            stuck,
            {"accel": 0.0, "brake": 0.0},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=100.1,
            previous_furthest_distance_m=100.0,
            stuck=True,
        )

        self.assertGreater(good_reward, stuck_reward)

    def test_reward_penalises_late_braking_into_sensor_danger(self):
        stage = train_td3_agent.CURRICULUM[1]
        previous = _telemetry(distance=300.0, raced=300.0, speed=140.0)
        danger = _telemetry(distance=306.0, raced=306.0, speed=142.0)
        danger["track"] = [200.0] * 9 + [20.0] + [200.0] * 9

        throttle_reward = train_td3_agent.calculate_td3_reward(
            danger,
            {"accel": 0.8, "brake": 0.0},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=306.0,
        )
        braking_reward = train_td3_agent.calculate_td3_reward(
            danger,
            {"accel": 0.0, "brake": 0.6},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=306.0,
        )

        self.assertLess(throttle_reward, braking_reward)

    def test_reward_heavily_penalises_off_track_failure(self):
        stage = train_td3_agent.CURRICULUM[0]
        previous = _telemetry(distance=20.0, raced=20.0, speed=40.0)
        off_track = _telemetry(distance=21.0, raced=21.0, speed=35.0, track_pos=1.20)
        off_track["track"] = [-1.0] * 19

        reward = train_td3_agent.calculate_td3_reward(
            off_track,
            {"accel": 0.5, "brake": 0.0},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=21.0,
            terminal_failure=True,
        )

        self.assertLess(reward, -120.0)

    def test_reward_penalises_high_speed_oversteer(self):
        stage = train_td3_agent.CURRICULUM[1]
        previous = _telemetry(distance=260.0, raced=260.0, speed=112.0)
        telemetry = _telemetry(distance=266.0, raced=266.0, speed=112.0)

        smooth_reward = train_td3_agent.calculate_td3_reward(
            telemetry,
            {"steer": 0.10, "accel": 0.4, "brake": 0.0},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=266.0,
            previous_furthest_distance_m=260.0,
        )
        oversteer_reward = train_td3_agent.calculate_td3_reward(
            telemetry,
            {"steer": 0.90, "accel": 0.4, "brake": 0.0},
            previous_telemetry=previous,
            stage=stage,
            episode_distance_m=266.0,
            previous_furthest_distance_m=260.0,
        )

        self.assertLess(oversteer_reward, smooth_reward - 4.0)

    def test_curriculum_stages_progress_from_launch_to_full_lap(self):
        stages = train_td3_agent.CURRICULUM

        self.assertEqual(stages[0].stage_id, "launch")
        self.assertEqual(stages[-1].stage_id, "full_lap")
        self.assertLess(stages[0].distance_target_m, stages[-1].distance_target_m)
        self.assertGreater(stages[-1].success_reward, stages[0].success_reward)

    def test_curriculum_requires_repeatable_launch_successes_in_recent_window(self):
        self.assertEqual(
            train_td3_agent.required_successes_for_stage(
                train_td3_agent.CURRICULUM[0]
            ),
            3,
        )
        self.assertEqual(train_td3_agent.required_successes_for_stage("first_corner"), 2)
        self.assertEqual(train_td3_agent.required_successes_for_stage("unknown"), 1)
        self.assertEqual(train_td3_agent.DEFAULT_STAGE_SUCCESS_WINDOW_SIZE, 10)

        launch = train_td3_agent.CURRICULUM[0]
        one_lucky_success = [False, False, True, False, False]
        spaced_successes = [True, False, False, True, False, False, True]

        self.assertFalse(
            train_td3_agent.curriculum_stage_can_advance(
                launch,
                one_lucky_success,
            )
        )
        self.assertTrue(
            train_td3_agent.curriculum_stage_can_advance(
                launch,
                spaced_successes,
            )
        )

    def test_recent_success_count_uses_last_ten_episodes(self):
        history = [True, True, False, False, False, False, False, False, False, False]
        history.extend([True, False, True, False, True])

        self.assertEqual(train_td3_agent.recent_success_count(history), 3)

    def test_warmup_episodes_cannot_advance_curriculum(self):
        launch = train_td3_agent.CURRICULUM[0]

        self.assertFalse(
            train_td3_agent.curriculum_stage_can_advance(
                launch,
                [True, True, True],
                curriculum_eligible=False,
            )
        )
        self.assertFalse(
            train_td3_agent.curriculum_stage_can_advance(
                launch,
                [True, True, True],
                completed_lap=120.0,
                curriculum_eligible=False,
            )
        )

    def test_policy_controlled_episode_must_start_after_warmup(self):
        self.assertFalse(
            train_td3_agent.episode_is_policy_controlled(19_999, 20_000)
        )
        self.assertTrue(
            train_td3_agent.episode_is_policy_controlled(20_000, 20_000)
        )

    def test_only_deterministic_policy_probes_are_curriculum_eligible(self):
        self.assertFalse(
            train_td3_agent.episode_is_curriculum_eligible(
                policy_controlled=False,
                deterministic_probe=True,
            )
        )
        self.assertFalse(
            train_td3_agent.episode_is_curriculum_eligible(
                policy_controlled=True,
                deterministic_probe=False,
            )
        )
        self.assertTrue(
            train_td3_agent.episode_is_curriculum_eligible(
                policy_controlled=True,
                deterministic_probe=True,
            )
        )

    def test_action_noise_decays_after_learning_starts(self):
        def sigma(timestep):
            return train_td3_agent.action_noise_sigma_at_timestep(
                timestep,
                learning_starts=20_000,
                initial_sigma=0.12,
                final_sigma=0.04,
                decay_steps=200_000,
            )

        self.assertAlmostEqual(sigma(0), 0.12)
        self.assertAlmostEqual(sigma(20_000), 0.12)
        self.assertAlmostEqual(sigma(120_000), 0.08)
        self.assertAlmostEqual(sigma(220_000), 0.04)
        self.assertAlmostEqual(sigma(300_000), 0.04)

    def test_learning_rate_schedule_decays_and_clamps_progress(self):
        schedule = train_td3_agent.linear_learning_rate_schedule(
            3e-4,
            1e-4,
            learning_starts=20_000,
            total_timesteps=220_000,
        )

        self.assertAlmostEqual(schedule(1.0), 3e-4)
        self.assertAlmostEqual(schedule(1.0 - (20_000 / 220_000)), 3e-4)
        self.assertAlmostEqual(schedule(1.0 - (120_000 / 220_000)), 2e-4)
        self.assertAlmostEqual(schedule(0.0), 1e-4)
        self.assertAlmostEqual(schedule(2.0), 3e-4)
        self.assertAlmostEqual(schedule(-1.0), 1e-4)

    def test_policy_probe_schedule_starts_promptly_and_then_spaces_probes(self):
        schedule = train_td3_agent.should_schedule_policy_probe

        self.assertFalse(
            schedule(
                timestep=19_999,
                learning_starts=20_000,
                probe_episodes_completed=0,
                policy_episodes_since_probe=0,
                probe_interval=10,
            )
        )
        self.assertTrue(
            schedule(
                timestep=20_000,
                learning_starts=20_000,
                probe_episodes_completed=0,
                policy_episodes_since_probe=0,
                probe_interval=10,
            )
        )
        self.assertFalse(
            schedule(
                timestep=30_000,
                learning_starts=20_000,
                probe_episodes_completed=1,
                policy_episodes_since_probe=9,
                probe_interval=10,
            )
        )
        self.assertTrue(
            schedule(
                timestep=31_000,
                learning_starts=20_000,
                probe_episodes_completed=1,
                policy_episodes_since_probe=10,
                probe_interval=10,
            )
        )

    def test_exploration_callback_toggles_probe_mode_at_episode_boundaries(self):
        class FakeBaseCallback:
            def __init__(self, verbose=0):
                self.verbose = verbose
                self.locals = {}
                self.model = types.SimpleNamespace(
                    action_noise=None,
                    gradient_steps=1,
                )
                self.num_timesteps = 0

        class FakeEnv:
            def __init__(self):
                self.modes = []

            def set_policy_mode(self, **mode):
                self.modes.append(mode)

        env = FakeEnv()
        Callback = train_td3_agent.make_exploration_schedule_callback_class(
            FakeBaseCallback
        )
        callback = Callback(
            env,
            lambda sigma: ("noise", sigma),
            learning_starts=20_000,
            initial_sigma=0.12,
            final_sigma=0.04,
            decay_steps=200_000,
            probe_interval=10,
            training_gradient_steps=1,
        )
        callback._on_training_start()

        self.assertEqual(callback.model.action_noise, ("noise", 0.12))
        self.assertEqual(callback.model.gradient_steps, 1)
        self.assertFalse(env.modes[-1]["deterministic_probe"])

        callback.num_timesteps = 20_100
        callback.locals = {
            "infos": [
                {
                    "episode_summary": {
                        "policy_controlled": False,
                        "deterministic_probe": False,
                    }
                }
            ]
        }
        callback._on_step()

        self.assertIsNone(callback.model.action_noise)
        self.assertEqual(callback.model.gradient_steps, 1)
        self.assertTrue(env.modes[-1]["deterministic_probe"])

        callback.num_timesteps = 20_101
        callback.locals = {"infos": [{}]}
        callback._on_step()

        self.assertEqual(callback.model.gradient_steps, 0)

        callback.num_timesteps = 20_500
        callback.locals = {
            "infos": [
                {
                    "episode_summary": {
                        "policy_controlled": True,
                        "deterministic_probe": True,
                    }
                }
            ]
        }
        callback._on_step()

        self.assertIsNotNone(callback.model.action_noise)
        self.assertEqual(callback.model.gradient_steps, 1)
        self.assertFalse(env.modes[-1]["deterministic_probe"])

    def test_default_training_args_use_stable_td3_warmup(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = train_td3_agent.parse_args()

        self.assertEqual(args.learning_starts, 20_000)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.learning_rate_final, 1e-4)
        self.assertEqual(args.checkpoint_freq, 25_000)
        self.assertEqual(args.action_noise_sigma, 0.12)
        self.assertEqual(args.action_noise_final_sigma, 0.04)
        self.assertEqual(args.action_noise_decay_steps, 200_000)
        self.assertEqual(args.policy_probe_interval, 10)
        self.assertEqual(args.warmup_steer_std, 0.18)

    def test_final_learning_rate_cannot_exceed_initial_rate(self):
        argv = [
            str(SCRIPT_PATH),
            "--learning-rate",
            "0.0001",
            "--learning-rate-final",
            "0.0003",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                train_td3_agent.parse_args()

    def test_best_model_callback_preserves_previous_global_bests(self):
        class FakeBaseCallback:
            def __init__(self, verbose=0):
                self.verbose = verbose

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            best_distance_path = directory_path / "best_distance.zip"
            best_reward_path = directory_path / "best_reward.zip"
            metadata_path_for_policy(best_distance_path).write_text(
                (
                    '{"best_distance_episode": {"distance_m": 171.2, '
                    '"episode_summary": {"policy_controlled": true, '
                    '"deterministic_probe": true}}}'
                ),
                encoding="utf-8",
            )
            metadata_path_for_policy(best_reward_path).write_text(
                (
                    '{"best_reward_episode": {"reward": 304.6, '
                    '"episode_summary": {"policy_controlled": true, '
                    '"deterministic_probe": true}}}'
                ),
                encoding="utf-8",
            )

            Callback = train_td3_agent.make_best_model_callback_class(FakeBaseCallback)
            callback = Callback(best_distance_path, best_reward_path, {})

        self.assertEqual(callback.best_distance_m, 171.2)
        self.assertEqual(callback.best_reward, 304.6)

    def test_best_model_callback_only_saves_deterministic_probes(self):
        class FakeBaseCallback:
            def __init__(self, verbose=0):
                self.verbose = verbose
                self.locals = {}
                self.model = mock.Mock()
                self.num_timesteps = 0

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            best_distance_path = directory_path / "best_distance.zip"
            best_reward_path = directory_path / "best_reward.zip"
            metadata_path_for_policy(best_distance_path).write_text(
                '{"best_distance_episode": {"distance_m": 182.3}}',
                encoding="utf-8",
            )
            metadata_path_for_policy(best_reward_path).write_text(
                '{"best_reward_episode": {"reward": 413.4}}',
                encoding="utf-8",
            )

            Callback = train_td3_agent.make_best_model_callback_class(
                FakeBaseCallback
            )
            callback = Callback(
                best_distance_path,
                best_reward_path,
                {},
                learning_starts=20_000,
            )

            self.assertIsNone(callback.best_distance_m)
            self.assertIsNone(callback.best_reward)

            callback.num_timesteps = 19_000
            callback.locals = {
                "infos": [
                    {
                        "episode_summary": {
                            "policy_controlled": True,
                            "deterministic_probe": False,
                            "curriculum_eligible": False,
                            "distance_m": 200.0,
                            "reward": 500.0,
                        }
                    }
                ]
            }
            callback._on_step()

            callback.model.save.assert_not_called()

            callback.num_timesteps = 21_000
            callback.locals = {
                "infos": [
                    {
                        "episode_summary": {
                            "policy_controlled": True,
                            "deterministic_probe": True,
                            "curriculum_eligible": True,
                            "distance_m": 80.0,
                            "reward": -100.0,
                        }
                    }
                ]
            }
            callback._on_step()

            self.assertEqual(callback.best_distance_m, 80.0)
            self.assertEqual(callback.best_reward, -100.0)
            self.assertEqual(callback.model.save.call_count, 2)

    def test_progress_line_reports_stage_and_best_distance(self):
        state = train_td3_agent.TrainingProgressState()
        state.record_episode(
            {
                "episodes_seen": 2,
                "stage_name": "First Corner Survival",
                "stage_id": "first_corner",
                "termination_reason": "off_track",
                "distance_m": 340.0,
                "reward": 12.0,
            }
        )
        state.record_best_distance(340.0)

        line = train_td3_agent.format_training_progress_line(
            50,
            100,
            10.0,
            state,
        )

        self.assertIn("stage=First Corner Survival", line)
        self.assertIn("best_dist=340m", line)

    def test_launch_biased_warmup_action_space_samples_bounded_actions(self):
        action_space = train_td3_agent.make_scratch_action_space(
            _FakeSpaces,
            warmup_action_mode="launch-biased",
            warmup_brake_probability=0.0,
        )
        action_space.seed(123)

        sample = action_space.sample()

        self.assertEqual(sample.shape, (3,))
        self.assertTrue(np.all(sample >= -1.0))
        self.assertTrue(np.all(sample <= 1.0))
        self.assertGreaterEqual(float(sample[1]), 0.25)
        self.assertEqual(float(sample[2]), 0.0)


class _FakeBox:
    def __init__(self, *, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.np_random = np.random.default_rng()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def sample(self, mask=None):
        return self.np_random.uniform(self.low, self.high, self.shape).astype(self.dtype)


class _FakeSpaces:
    Box = _FakeBox


def _telemetry(
    *,
    distance: float = 100.0,
    raced: float = 100.0,
    speed: float = 80.0,
    track_pos: float = 0.0,
) -> dict[str, object]:
    return {
        "distFromStart": distance,
        "distRaced": raced,
        "speedX": speed,
        "speedY": 0.0,
        "speedZ": 0.0,
        "angle": 0.0,
        "trackPos": track_pos,
        "damage": 0.0,
        "rpm": 5000.0,
        "gear": 3,
        "track": [200.0] * 19,
        "wheelSpinVel": [10.0] * 4,
        "curLapTime": 10.0,
        "lastLapTime": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
