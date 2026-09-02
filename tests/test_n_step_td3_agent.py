import importlib.util
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.n_step_td3_agent import NstepTd3Agent
from agents.sensor_n_step_td3_agent import SensorNstepTd3Agent
from n_step_td3.contracts import (
    ACTION_SIZE,
    BASE_OBSERVATION_NOISE_STD,
    BASE_OBSERVATION_SIZE,
    DEFAULT_PACE_STEERING_LIMIT,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    PACE_ACTION_VERSION,
    PACE_FAILURE_PENALTY,
    PACE_LAP_COMPLETION_BONUS,
    PACE_REWARD_VERSION,
    REFERENCE_OBSERVATION_NOISE_LEVEL,
    REWARD_VERSION,
    SENSOR_ONLY_MODEL_FAMILY,
    SENSOR_ONLY_OBSERVATION_VERSION,
    SENSOR_ONLY_REWARD_VERSION,
    STEERING_RATE_REWARD_VERSION,
    HistoryEncoder,
    apply_observation_noise,
    build_base_observation,
    calculate_progress_reward,
    calculate_reward,
    decode_action,
)
from n_step_td3.environment import (
    EpisodeSummary,
    NstepTorcsEnvironment,
    _legacy_torcs_argv,
)
from n_step_td3.learner import (
    NstepTd3Config,
    NstepTd3Learner,
    load_actor_checkpoint,
    load_checkpoint,
)
from n_step_td3.replay import (
    NstepReplayBuffer,
    NstepTransitionAccumulator,
    ReplayTransition,
)
from n_step_td3.racing_line import RacingLineFeatureMap


def telemetry(**overrides):
    value = {
        "speedX": 50.0,
        "speedY": 0.0,
        "speedZ": 0.0,
        "rpm": 5_000.0,
        "gear": 2,
        "angle": 0.0,
        "trackPos": 0.0,
        "track": [100.0] * 19,
        "wheelSpinVel": [20.0] * 4,
        "damage": 0.0,
        "distRaced": 0.0,
        "distFromStart": 0.0,
        "curLapTime": 1.0,
        "lastLapTime": 0.0,
    }
    value.update(overrides)
    return value


def load_training_script():
    path = PROJECT_ROOT / "scripts" / "train_n_step_td3_agent.py"
    spec = importlib.util.spec_from_file_location("train_n_step_td3_agent", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_evaluation_script():
    path = PROJECT_ROOT / "scripts" / "evaluate_n_step_td3_agent.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_n_step_td3_agent",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NstepTd3ContractTests(unittest.TestCase):
    def test_torcsrl_observation_contract_has_45_finite_features(self):
        observation = build_base_observation(telemetry())

        self.assertEqual(observation.shape, (BASE_OBSERVATION_SIZE,))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.all(observation <= 1.0))
        self.assertTrue(np.all(observation >= -1.0))

    def test_sensor_only_observation_masks_every_line_derived_feature(self):
        observation = build_base_observation(
            telemetry(trackPos=0.75, distFromStart=900.0),
            include_racing_line_features=False,
        )

        self.assertEqual(observation.shape, (BASE_OBSERVATION_SIZE,))
        self.assertAlmostEqual(observation[8], 0.75)
        np.testing.assert_array_equal(
            observation[-13:],
            np.zeros(13, dtype=np.float32),
        )

    def test_server_tita_fields_are_replaced_by_local_racing_line_map(self):
        first = telemetry(titaLineTrackPos=-0.8, titaSteer=1.0)
        second = telemetry(titaLineTrackPos=0.9, titaSteer=-1.0)
        racing_line = RacingLineFeatureMap(
            np.asarray([0.0, 100.0]),
            np.asarray([0.25, -0.25]),
            np.asarray([0.01, -0.01]),
            100.0,
        )

        np.testing.assert_array_equal(
            build_base_observation(first, racing_line=racing_line),
            build_base_observation(second, racing_line=racing_line),
        )

    def test_racing_line_adds_deviation_and_lookahead_curvature(self):
        racing_line = RacingLineFeatureMap(
            np.asarray([0.0, 50.0, 100.0]),
            np.asarray([0.2, -0.2, 0.2]),
            np.asarray([0.01, -0.02, 0.01]),
            100.0,
        )
        observation = build_base_observation(
            telemetry(distFromStart=25.0, trackPos=0.3),
            racing_line=racing_line,
        )

        self.assertEqual(observation.shape, (BASE_OBSERVATION_SIZE,))
        self.assertTrue(np.any(observation[-12:] != 0.0))

    def test_history_stacks_three_observations_and_actions(self):
        encoder = HistoryEncoder()
        first = np.full(BASE_OBSERVATION_SIZE, 0.1, dtype=np.float32)
        second = np.full(BASE_OBSERVATION_SIZE, 0.2, dtype=np.float32)
        action = np.asarray([0.3, 0.4], dtype=np.float32)

        encoder.reset(first)
        encoder.record_action(action)
        encoded = encoder.observe(second)

        self.assertEqual(encoded.shape, (OBSERVATION_SIZE,))
        observation_history = encoded[: BASE_OBSERVATION_SIZE * 3].reshape(3, -1)
        action_history = encoded[BASE_OBSERVATION_SIZE * 3 :].reshape(3, -1)
        np.testing.assert_array_equal(observation_history[0], np.zeros_like(first))
        np.testing.assert_array_equal(observation_history[1], first)
        np.testing.assert_array_equal(observation_history[2], second)
        np.testing.assert_array_equal(action_history[-1], action)

    def test_signed_longitudinal_action_never_conflicts_pedals(self):
        accelerate = decode_action(np.asarray([0.25, 0.8]), telemetry())
        brake = decode_action(np.asarray([-0.2, -0.6]), telemetry())

        self.assertAlmostEqual(accelerate["accel"], 0.8)
        self.assertEqual(accelerate["brake"], 0.0)
        self.assertEqual(brake["accel"], 0.0)
        self.assertAlmostEqual(brake["brake"], 0.6)

    def test_v5_action_scales_only_physical_steering(self):
        controls = decode_action(
            np.asarray([1.0, 0.75], dtype=np.float32),
            telemetry(),
            physical_steering_limit=DEFAULT_PACE_STEERING_LIMIT,
        )

        self.assertAlmostEqual(controls["steer"], 0.3)
        self.assertAlmostEqual(controls["accel"], 0.75)
        self.assertEqual(controls["brake"], 0.0)

    def test_v5_reward_contains_only_progress_and_terminal_events(self):
        progress = calculate_progress_reward(
            100.0,
            100.8,
            physical_failure=False,
            lap_completed=False,
        )
        failed = calculate_progress_reward(
            100.0,
            100.8,
            physical_failure=True,
            lap_completed=False,
        )
        completed = calculate_progress_reward(
            100.0,
            100.8,
            physical_failure=False,
            lap_completed=True,
        )

        self.assertAlmostEqual(progress.total, 0.8)
        self.assertAlmostEqual(
            failed.total,
            PACE_FAILURE_PENALTY + 0.8,
        )
        self.assertAlmostEqual(
            completed.total,
            PACE_LAP_COMPLETION_BONUS + 0.8,
        )
        self.assertEqual(progress.forward_velocity, 0.0)
        self.assertEqual(progress.racing_line_factor, 0.0)

    def test_reward_matches_torcsrl_heading_and_racing_line_formula(self):
        centered = calculate_reward(telemetry(speedX=125.0), physical_failure=False)
        corner = calculate_reward(
            telemetry(speedX=125.0, angle=0.2, trackPos=0.3),
            physical_failure=False,
            target_track_position=0.1,
        )
        failed = calculate_reward(telemetry(), physical_failure=True)

        expected_factor = math.cos(0.2) - abs(math.sin(0.2)) - 0.2
        self.assertAlmostEqual(corner.racing_line_factor, expected_factor)
        self.assertAlmostEqual(corner.total, 0.5 * expected_factor)
        self.assertGreater(centered.total, corner.total)
        self.assertEqual(failed.total, -10.0)

    def test_v4_reward_penalises_only_squared_steering_change(self):
        baseline = calculate_reward(
            telemetry(speedX=125.0),
            physical_failure=False,
            previous_steer=1.0,
            current_steer=-1.0,
        )
        v4 = calculate_reward(
            telemetry(speedX=125.0),
            physical_failure=False,
            previous_steer=1.0,
            current_steer=-1.0,
            steering_rate_cost_coefficient=0.0025,
        )

        self.assertEqual(baseline.steering_rate_penalty, 0.0)
        self.assertAlmostEqual(v4.steering_rate_penalty, -0.01)
        self.assertAlmostEqual(v4.total, baseline.total - 0.01)
        self.assertEqual(v4.forward_velocity, baseline.forward_velocity)
        self.assertEqual(v4.racing_line_factor, baseline.racing_line_factor)

    def test_reward_rejects_invalid_steering_rate_coefficients(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    calculate_reward(
                        telemetry(),
                        physical_failure=False,
                        steering_rate_cost_coefficient=value,
                    )

    def test_reverse_velocity_remains_negative_in_reward(self):
        result = calculate_reward(telemetry(speedX=-25.0), physical_failure=False)

        self.assertAlmostEqual(result.total, -0.1)

    def test_observation_noise_uses_feature_specific_reference_profile(self):
        profile = np.asarray(BASE_OBSERVATION_NOISE_STD)

        self.assertEqual(profile.shape, (BASE_OBSERVATION_SIZE,))
        np.testing.assert_array_equal(profile[:7], np.full(7, 0.0025))
        np.testing.assert_array_equal(profile[7:28], np.full(21, 0.00025))
        np.testing.assert_array_equal(profile[28:32], np.full(4, 0.0025))
        self.assertEqual(profile[32], 0.00025)
        np.testing.assert_array_equal(profile[33:], np.zeros(12))

        clean = np.zeros(BASE_OBSERVATION_SIZE, dtype=np.float32)
        noisy = apply_observation_noise(
            clean,
            rng=np.random.default_rng(19),
            level=REFERENCE_OBSERVATION_NOISE_LEVEL,
        )
        self.assertTrue(np.any(noisy[:33] != 0.0))
        np.testing.assert_array_equal(noisy[33:], clean[33:])

    def test_zero_observation_noise_returns_an_independent_clean_value(self):
        clean = np.linspace(-0.5, 0.5, BASE_OBSERVATION_SIZE, dtype=np.float32)
        result = apply_observation_noise(
            clean,
            rng=np.random.default_rng(23),
            level=0.0,
        )

        np.testing.assert_array_equal(result, clean)
        self.assertIsNot(result, clean)


class NstepReplayTests(unittest.TestCase):
    def test_accumulator_builds_discounted_three_step_return(self):
        accumulator = NstepTransitionAccumulator(3, 0.9)
        emitted = []
        for index, reward in enumerate((1.0, 2.0, 3.0)):
            emitted.extend(
                accumulator.append(
                    np.asarray([index], dtype=np.float32),
                    np.asarray([0.0], dtype=np.float32),
                    reward,
                    np.asarray([index + 1], dtype=np.float32),
                    terminated=False,
                    episode_end=False,
                )
            )

        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].discounted_return, 1 + 0.9 * 2 + 0.81 * 3)
        self.assertEqual(emitted[0].steps, 3)
        self.assertFalse(emitted[0].terminated)

    def test_timeout_flushes_with_bootstrap_mask_intact(self):
        accumulator = NstepTransitionAccumulator(3, 0.9)
        accumulator.append(
            np.asarray([0.0]),
            np.asarray([0.0]),
            1.0,
            np.asarray([1.0]),
            terminated=False,
            episode_end=False,
        )
        emitted = accumulator.append(
            np.asarray([1.0]),
            np.asarray([0.0]),
            2.0,
            np.asarray([2.0]),
            terminated=False,
            episode_end=True,
        )

        self.assertEqual([item.steps for item in emitted], [2, 1])
        self.assertTrue(all(not item.terminated for item in emitted))

    def test_physical_terminal_flushes_with_terminal_mask(self):
        accumulator = NstepTransitionAccumulator(3, 0.9)
        emitted = accumulator.append(
            np.asarray([0.0]),
            np.asarray([0.0]),
            -10.0,
            np.asarray([1.0]),
            terminated=True,
            episode_end=True,
        )

        self.assertTrue(emitted[0].terminated)

    def test_replay_round_trip_preserves_timeout_flag(self):
        replay = NstepReplayBuffer(2, 1, 8, seed=1)
        replay.add(
            ReplayTransition(
                observation=np.asarray([1.0, 2.0], dtype=np.float32),
                action=np.asarray([0.5], dtype=np.float32),
                discounted_return=3.0,
                next_observation=np.asarray([2.0, 3.0], dtype=np.float32),
                terminated=False,
                steps=2,
            )
        )
        restored = NstepReplayBuffer(2, 1, 8, seed=2)
        restored.load_state_dict(replay.state_dict())
        sample = restored.sample(1, torch.device("cpu"))

        self.assertEqual(len(restored), 1)
        self.assertEqual(float(sample[4].item()), 0.0)
        self.assertEqual(float(sample[5].item()), 2.0)


class NstepTd3LearnerTests(unittest.TestCase):
    def _learner(self):
        return NstepTd3Learner(
            NstepTd3Config(
                observation_size=4,
                action_size=2,
                hidden_size_1=8,
                hidden_size_2=8,
                replay_capacity=32,
                replay_start_size=8,
                batch_size=4,
                seed=7,
            ),
            device="cpu",
        )

    def _fill(self, learner):
        random = np.random.default_rng(3)
        for _ in range(8):
            observation = random.normal(size=4).astype(np.float32)
            learner.replay.add(
                ReplayTransition(
                    observation=observation,
                    action=random.uniform(-1, 1, size=2).astype(np.float32),
                    discounted_return=float(random.normal()),
                    next_observation=(observation + 0.1).astype(np.float32),
                    terminated=False,
                    steps=3,
                )
            )

    def test_policy_is_delayed_while_critic_updates_every_step(self):
        learner = self._learner()
        self._fill(learner)
        before = {
            name: value.detach().clone()
            for name, value in learner.actor.state_dict().items()
        }

        first = learner.train_for_interaction()[0]
        after_first = learner.actor.state_dict()
        self.assertIsNone(first.actor_loss)
        self.assertTrue(
            all(torch.equal(before[name], after_first[name]) for name in before)
        )

        second = learner.train_for_interaction()[0]
        after_second = learner.actor.state_dict()
        self.assertIsNotNone(second.actor_loss)
        self.assertTrue(
            any(not torch.equal(before[name], after_second[name]) for name in before)
        )
        self.assertEqual(learner.optimizer_steps, 2)
        self.assertEqual(learner.actor_updates, 1)

    def test_critic_only_warmup_preserves_actor_and_actor_target(self):
        learner = self._learner()
        self._fill(learner)
        actor_before = {
            name: value.detach().clone()
            for name, value in learner.actor.state_dict().items()
        }
        target_before = {
            name: value.detach().clone()
            for name, value in learner.actor_target.state_dict().items()
        }
        critic_before = {
            name: value.detach().clone()
            for name, value in learner.critic.state_dict().items()
        }

        metrics = learner.train_critic_only(4)

        self.assertEqual(len(metrics), 4)
        self.assertTrue(all(item.actor_loss is None for item in metrics))
        self.assertEqual(learner.optimizer_steps, 4)
        self.assertEqual(learner.actor_updates, 0)
        for name, value in learner.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])
        for name, value in learner.actor_target.state_dict().items():
            torch.testing.assert_close(value, target_before[name])
        self.assertTrue(
            any(
                not torch.equal(value, critic_before[name])
                for name, value in learner.critic.state_dict().items()
            )
        )

    def test_actor_updates_can_be_disabled_while_critic_keeps_learning(self):
        learner = self._learner()
        self._fill(learner)
        actor_before = {
            name: value.detach().clone()
            for name, value in learner.actor.state_dict().items()
        }

        for _ in range(4):
            metrics = learner.train_for_interaction(allow_actor_update=False)
            self.assertIsNone(metrics[0].actor_loss)

        self.assertEqual(learner.optimizer_steps, 4)
        self.assertEqual(learner.actor_updates, 0)
        for name, value in learner.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])

    def test_checkpoint_round_trip_preserves_actor_output(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            replay_capacity=16,
            replay_start_size=8,
            batch_size=4,
            seed=5,
        )
        learner = NstepTd3Learner(config, device="cpu")
        learner.best_evaluation_median_m = 123.0
        learner.best_evaluation_completion_rate = 1.0
        learner.best_evaluation_median_lap_time_seconds = 205.0
        learner.best_evaluation_mean_off_track_steps = 0.0
        learner.best_evaluation_mean_damage_delta = 0.0
        learner.best_lap_time_seconds = 204.0
        observation = np.linspace(-1, 1, OBSERVATION_SIZE, dtype=np.float32)
        expected = learner.select_action(observation, deterministic=True)
        learner.replay.add(
            ReplayTransition(
                observation=observation,
                action=np.zeros(ACTION_SIZE, dtype=np.float32),
                discounted_return=0.0,
                next_observation=observation,
                terminated=False,
                steps=3,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent7.pt"
            learner.save(path)
            restored = NstepTd3Learner.load(path, device="cpu")

        actual = restored.select_action(observation, deterministic=True)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertEqual(len(learner.replay), 1)
        self.assertEqual(len(restored.replay), 0)
        self.assertEqual(restored.best_evaluation_median_m, 123.0)
        self.assertEqual(restored.best_evaluation_completion_rate, 1.0)
        self.assertEqual(
            restored.best_evaluation_median_lap_time_seconds,
            205.0,
        )
        self.assertEqual(restored.best_lap_time_seconds, 204.0)

    def test_v4_checkpoint_round_trip_uses_reward_from_configuration(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            replay_capacity=16,
            replay_start_size=8,
            batch_size=4,
            steering_rate_cost_coefficient=0.0025,
        )
        learner = NstepTd3Learner(config, device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent7-v4.pt"
            learner.save(path)
            checkpoint = load_checkpoint(path)
            restored = NstepTd3Learner.load(path, device="cpu")

        self.assertEqual(checkpoint["reward_version"], STEERING_RATE_REWARD_VERSION)
        self.assertEqual(restored.reward_version, STEERING_RATE_REWARD_VERSION)
        self.assertEqual(restored.config.steering_rate_cost_coefficient, 0.0025)

    def test_v5_checkpoint_round_trip_preserves_contracts(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            replay_capacity=16,
            replay_start_size=8,
            batch_size=4,
            physical_steering_limit=DEFAULT_PACE_STEERING_LIMIT,
            progress_reward=True,
        )
        learner = NstepTd3Learner(config, device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent7-v5.pt"
            learner.save(path)
            checkpoint = load_checkpoint(path)
            restored = NstepTd3Learner.load(path, device="cpu")

        self.assertEqual(checkpoint["action_version"], PACE_ACTION_VERSION)
        self.assertEqual(checkpoint["reward_version"], PACE_REWARD_VERSION)
        self.assertEqual(restored.action_version, PACE_ACTION_VERSION)
        self.assertEqual(restored.reward_version, PACE_REWARD_VERSION)
        self.assertAlmostEqual(restored.config.physical_steering_limit, 0.3)
        self.assertTrue(restored.config.progress_reward)

    def test_sensor_only_checkpoint_has_an_isolated_contract(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            replay_capacity=16,
            replay_start_size=8,
            batch_size=4,
            racing_line_features=False,
        )
        learner = NstepTd3Learner(config, device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent8.pt"
            learner.save(path)
            checkpoint = load_checkpoint(path)
            restored = NstepTd3Learner.load(path, device="cpu")
            with self.assertRaises(ValueError):
                NstepTd3Agent(policy_path=path, require_policy=True)

        self.assertEqual(checkpoint["model_family"], SENSOR_ONLY_MODEL_FAMILY)
        self.assertEqual(
            checkpoint["observation_version"],
            SENSOR_ONLY_OBSERVATION_VERSION,
        )
        self.assertEqual(
            checkpoint["reward_version"],
            SENSOR_ONLY_REWARD_VERSION,
        )
        self.assertFalse(restored.config.racing_line_features)

    def test_actor_only_transfer_resets_all_learning_state(self):
        source_config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            replay_capacity=16,
            replay_start_size=8,
            batch_size=4,
            seed=5,
        )
        source = NstepTd3Learner(source_config, device="cpu")
        with torch.no_grad():
            for parameter in source.critic.parameters():
                parameter.add_(1.0)
        source.environment_steps = 123
        source.optimizer_steps = 45
        source.actor_updates = 6
        source.best_evaluation_completion_rate = 1.0
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        expected_action = source.select_action(observation, deterministic=True)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-v3.pt"
            source.save(path)
            target_config = replace(
                source_config,
                actor_learning_rate=1e-6,
                steering_rate_cost_coefficient=0.0025,
                seed=17,
            )
            transferred = NstepTd3Learner.from_actor_checkpoint(
                path,
                config=target_config,
                device="cpu",
            )

        actual_action = transferred.select_action(
            observation,
            deterministic=True,
        )
        np.testing.assert_allclose(actual_action, expected_action, atol=1e-7)
        self.assertEqual(transferred.environment_steps, 0)
        self.assertEqual(transferred.optimizer_steps, 0)
        self.assertEqual(transferred.actor_updates, 0)
        self.assertIsNone(transferred.best_evaluation_completion_rate)
        self.assertEqual(len(transferred.replay), 0)
        self.assertEqual(transferred.actor_optimizer.state_dict()["state"], {})
        self.assertEqual(transferred.reward_version, STEERING_RATE_REWARD_VERSION)
        self.assertTrue(
            any(
                not torch.equal(value, source.critic.state_dict()[name])
                for name, value in transferred.critic.state_dict().items()
            )
        )

    def test_legacy_actor_is_inference_compatible_but_not_resumable(self):
        learner = NstepTd3Learner(
            NstepTd3Config(
                hidden_size_1=8,
                hidden_size_2=8,
                replay_capacity=16,
                replay_start_size=8,
                batch_size=4,
            ),
            device="cpu",
        )
        legacy = learner.checkpoint()
        legacy["checkpoint_version"] = "agent7_n_step_td3_checkpoint_v1"
        legacy["observation_version"] = "agent7_torcsrl_history_v2"
        legacy["reward_version"] = "agent7_torcsrl_racing_line_velocity_v2"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(legacy, path)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                load_checkpoint(path)
            actor_checkpoint = load_actor_checkpoint(path)

        self.assertEqual(
            actor_checkpoint["observation_version"],
            "agent7_torcsrl_history_v2",
        )
        self.assertEqual(OBSERVATION_VERSION, "agent7_torcsrl_history_v3")
        self.assertEqual(REWARD_VERSION, "agent7_torcsrl_racing_line_velocity_v3")
        self.assertEqual(
            STEERING_RATE_REWARD_VERSION,
            "agent7_torcsrl_racing_line_velocity_steering_rate_v4",
        )

    def test_reseed_training_noise_is_reproducible_and_preserves_weights(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            seed=5,
        )
        source = NstepTd3Learner(config, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            source.save(path)
            first = NstepTd3Learner.load(path, device="cpu")
            second = NstepTd3Learner.load(path, device="cpu")
        actor_before = {
            name: value.detach().clone()
            for name, value in first.actor.state_dict().items()
        }
        critic_before = {
            name: value.detach().clone()
            for name, value in first.critic.state_dict().items()
        }

        first.reseed_training_noise(17)
        second.reseed_training_noise(17)
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        np.testing.assert_array_equal(
            first.select_action(observation, deterministic=False),
            second.select_action(observation, deterministic=False),
        )
        torch.testing.assert_close(
            first._normal_noise(torch.Size([4, ACTION_SIZE])),
            second._normal_noise(torch.Size([4, ACTION_SIZE])),
        )
        for name, value in first.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])
        for name, value in first.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[name])

    def test_different_resume_seeds_change_training_noise(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            seed=5,
        )
        first = NstepTd3Learner(config, device="cpu")
        second = NstepTd3Learner(config, device="cpu")
        first.reseed_training_noise(17)
        second.reseed_training_noise(18)
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

        self.assertFalse(
            np.array_equal(
                first.select_action(observation, deterministic=False),
                second.select_action(observation, deterministic=False),
            )
        )
        self.assertFalse(
            torch.equal(
                first._normal_noise(torch.Size([4, ACTION_SIZE])),
                second._normal_noise(torch.Size([4, ACTION_SIZE])),
            )
        )

    def test_fine_tuning_reduces_actor_update_scale_and_frequency(self):
        learner = self._learner()
        actor_before = {
            name: value.detach().clone()
            for name, value in learner.actor.state_dict().items()
        }
        critic_learning_rate = learner.config.critic_learning_rate

        learner.enable_fine_tuning()

        self.assertEqual(learner.config.actor_learning_rate, 3e-6)
        self.assertEqual(learner.config.critic_learning_rate, critic_learning_rate)
        self.assertEqual(learner.config.exploration_noise, 0.01)
        self.assertEqual(learner.config.longitudinal_exploration_noise, 0.01)
        self.assertEqual(learner.config.policy_delay, 4)
        self.assertEqual(learner.actor_optimizer.param_groups[0]["lr"], 3e-6)
        np.testing.assert_array_equal(
            learner._exploration_noise_scale,
            np.asarray([0.01, 0.01], dtype=np.float32),
        )
        for name, value in learner.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])

    def test_fine_tuning_accepts_a_custom_actor_learning_rate(self):
        learner = self._learner()

        learner.enable_fine_tuning(actor_learning_rate=7e-7)

        self.assertEqual(learner.config.actor_learning_rate, 7e-7)
        self.assertEqual(learner.actor_optimizer.param_groups[0]["lr"], 7e-7)

    def test_fine_tuning_rejects_invalid_actor_learning_rates(self):
        learner = self._learner()

        for value in (0.0, -1e-6, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    learner.enable_fine_tuning(actor_learning_rate=value)

    def test_longitudinal_exploration_can_escape_full_throttle_saturation(self):
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            exploration_noise=0.0,
            longitudinal_exploration_noise=0.5,
            seed=11,
        )
        learner = NstepTd3Learner(config, device="cpu")
        with torch.no_grad():
            for parameter in learner.actor.parameters():
                parameter.zero_()
            learner.actor.network[-2].bias[1] = 3.0

        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        deterministic_before = learner.select_action(
            observation,
            deterministic=True,
        )
        exploratory = np.asarray(
            [
                learner.select_action(observation, deterministic=False)
                for _ in range(2_000)
            ]
        )
        deterministic_after = learner.select_action(
            observation,
            deterministic=True,
        )

        self.assertTrue(np.all(exploratory[:, 0] == 0.0))
        self.assertTrue(np.any(exploratory[:, 1] < 0.0))
        np.testing.assert_array_equal(
            deterministic_after,
            deterministic_before,
        )

class NstepTd3RuntimeTests(unittest.TestCase):
    def test_agent7_is_a_separate_reward_only_runtime_agent(self):
        seen_shapes = []

        def policy(observation):
            seen_shapes.append(observation.shape)
            return np.asarray([0.2, 0.7], dtype=np.float32)

        agent = NstepTd3Agent(policy=policy)
        action = agent.act(None, telemetry())

        self.assertEqual(agent.agent_type, "n_step_td3")
        self.assertNotEqual(agent.agent_type, "td3_scratch")
        self.assertFalse(agent.config["racing_line_imitation"])
        self.assertTrue(agent.config["racing_line_reward_shaping"])
        self.assertEqual(
            agent.racing_line_path.name,
            "g-track-3-agent7-smooth-v1.json",
        )
        self.assertEqual(seen_shapes, [(OBSERVATION_SIZE,)])
        self.assertAlmostEqual(action["steer"], 0.2)
        self.assertAlmostEqual(action["accel"], 0.7)

    def test_agent8_runtime_uses_no_racing_line_features(self):
        seen_observations = []

        def policy(observation):
            seen_observations.append(observation.copy())
            return np.asarray([0.25, 0.8], dtype=np.float32)

        agent = SensorNstepTd3Agent(policy=policy)
        controls = agent.act(None, telemetry(trackPos=0.6))

        self.assertFalse(agent.requires_racing_line)
        self.assertIsNone(agent.racing_line_path)
        self.assertFalse(agent.config["racing_line_observation"])
        self.assertFalse(agent.config["racing_line_reward_shaping"])
        self.assertEqual(agent.config["model_family"], SENSOR_ONLY_MODEL_FAMILY)
        self.assertAlmostEqual(controls["steer"], 0.25)
        self.assertAlmostEqual(controls["accel"], 0.8)
        history = seen_observations[0][
            : BASE_OBSERVATION_SIZE * 3
        ].reshape(3, BASE_OBSERVATION_SIZE)
        np.testing.assert_array_equal(
            history[-1, -13:],
            np.zeros(13, dtype=np.float32),
        )

    def test_training_parser_uses_torcsrl_hyperparameter_recipe(self):
        script = load_training_script()
        args = script.parse_args(["--total-timesteps", "100"])
        config = script.config_from_args(args)

        self.assertEqual(config.n_steps, 3)
        self.assertEqual(config.gamma, 0.992)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.replay_start_size, 10_000)
        self.assertEqual(config.policy_delay, 2)
        self.assertEqual(config.exploration_noise, 0.1)
        self.assertEqual(config.longitudinal_exploration_noise, 0.1)
        self.assertEqual(config.replay_capacity, 1_000_000)
        self.assertEqual(args.evaluation_interval, 10_000)
        self.assertEqual(args.evaluation_repeats, 3)
        self.assertEqual(args.max_episode_steps, 25_000)
        self.assertEqual(args.observation_noise_std, 0.025)
        self.assertEqual(args.train_frequency, 4)
        self.assertEqual(config.steering_rate_cost_coefficient, 0.0)
        self.assertFalse(args.seed_was_explicit)
        self.assertEqual(args.seed, 0)

    def test_v3_training_outputs_are_isolated_from_legacy_models(self):
        script = load_training_script()

        self.assertEqual(script.MODEL_DIR.name, "agent7_n_step_td3_v3")
        self.assertEqual(script.RUNS_DIR.name, "agent7_n_step_td3_v3")
        self.assertEqual(
            script.STEERING_RATE_V4_MODEL_DIR.name,
            "agent7_n_step_td3_v4",
        )
        self.assertEqual(
            script.STEERING_RATE_V4_RUNS_DIR.name,
            "agent7_n_step_td3_v4",
        )
        self.assertEqual(script.PACE_V5_MODEL_DIR.name, "agent7_n_step_td3_v5")
        self.assertEqual(script.PACE_V5_RUNS_DIR.name, "agent7_n_step_td3_v5")
        self.assertEqual(
            script.SENSOR_ONLY_MODEL_DIR.name,
            "agent8_sensor_n_step_td3",
        )
        self.assertEqual(
            script.SENSOR_ONLY_RUNS_DIR.name,
            "agent8_sensor_n_step_td3",
        )
        self.assertEqual(
            script.SENSOR_STEERING_V2_MODEL_DIR.name,
            "agent8_sensor_n_step_td3_v2_steer070",
        )
        self.assertEqual(
            script.SENSOR_STEERING_V2_RUNS_DIR.name,
            "agent8_sensor_n_step_td3_v2_steer070",
        )
        self.assertEqual(
            script.SENSOR_CLEAN_OBSERVATION_V3_MODEL_DIR.name,
            "agent8_sensor_n_step_td3_v3_clean_observation",
        )
        self.assertEqual(
            script.SENSOR_CLEAN_OBSERVATION_V3_RUNS_DIR.name,
            "agent8_sensor_n_step_td3_v3_clean_observation",
        )
        self.assertEqual(
            script.SENSOR_V1_CONTINUATION_MODEL_DIR.name,
            "agent8_sensor_n_step_td3_v1_continuation",
        )
        self.assertEqual(
            script.SENSOR_V1_CONTINUATION_RUNS_DIR.name,
            "agent8_sensor_n_step_td3_v1_continuation",
        )
        self.assertEqual(
            script.SENSOR_V1_STABILITY_MODEL_DIR.name,
            "agent8_sensor_n_step_td3_v1_stability",
        )
        self.assertEqual(
            script.SENSOR_V1_STABILITY_RUNS_DIR.name,
            "agent8_sensor_n_step_td3_v1_stability",
        )
        self.assertEqual(
            script.SENSOR_V1_CLEAN_RELIABILITY_MODEL_DIR.name,
            "agent8_sensor_n_step_td3_v1_clean_reliability",
        )
        self.assertEqual(
            script.SENSOR_V1_CLEAN_RELIABILITY_RUNS_DIR.name,
            "agent8_sensor_n_step_td3_v1_clean_reliability",
        )
        self.assertEqual(
            script.SENSOR_V1_ROBUST_PACE_MODEL_DIR.name,
            "agent8_sensor_n_step_td3_v1_robust_pace",
        )
        self.assertEqual(
            script.SENSOR_V1_ROBUST_PACE_RUNS_DIR.name,
            "agent8_sensor_n_step_td3_v1_robust_pace",
        )

    def test_v5_parser_applies_fresh_minimal_pace_profile(self):
        script = load_training_script()
        args = script.parse_args(["--pace-v5", "--seed", "23"])
        config = script.config_from_args(args)

        self.assertEqual(args.total_timesteps, 1_000_000)
        self.assertEqual(args.evaluation_interval, 50_000)
        self.assertEqual(args.evaluation_repeats, 3)
        self.assertIsNone(args.resume)
        self.assertAlmostEqual(config.physical_steering_limit, 0.3)
        self.assertTrue(config.progress_reward)
        self.assertEqual(config.steering_rate_cost_coefficient, 0.0)

    def test_v5_rejects_resume_and_replay_imports(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with self.assertRaises(SystemExit):
                script.parse_args(
                    ["--pace-v5", "--resume", str(checkpoint)]
                )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    ["--pace-v5", "--replay-buffer-path", str(checkpoint)]
                )

    def test_sensor_only_parser_uses_a_separate_reward_only_profile(self):
        script = load_training_script()
        args = script.parse_args(["--sensor-only", "--seed", "29"])
        config = script.config_from_args(args)

        self.assertFalse(config.racing_line_features)
        self.assertFalse(config.progress_reward)
        self.assertEqual(config.steering_rate_cost_coefficient, 0.0)

    def test_sensor_only_parser_rejects_a_racing_line(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            racing_line = Path(directory) / "line.json"
            racing_line.touch()
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--sensor-only",
                        "--racing-line-path",
                        str(racing_line),
                    ]
                )

    def test_sensor_v1_pace_continuation_preserves_the_full_td3_checkpoint(self):
        source = NstepTd3Learner(
            NstepTd3Config(
                hidden_size_1=8,
                hidden_size_2=8,
                racing_line_features=False,
                seed=7,
            ),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            source.save(checkpoint)
            args = load_training_script().parse_args(
                [
                    "--sensor-only",
                    "--sensor-v1-pace-continuation",
                    "--resume",
                    str(checkpoint),
                    "--seed",
                    "31",
                ]
            )
            resumed = NstepTd3Learner.load(args.resume, device="cpu")

        self.assertEqual(args.observation_noise_std, 0.025)
        self.assertEqual(args.train_frequency, 4)
        self.assertFalse(args.fine_tune)
        self.assertEqual(resumed.config.actor_learning_rate, 3e-4)
        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(value, resumed.actor.state_dict()[name])
        for name, value in source.critic.state_dict().items():
            torch.testing.assert_close(value, resumed.critic.state_dict()[name])

    def test_sensor_v1_stability_continuation_refines_full_td3_state(self):
        source = NstepTd3Learner(
            NstepTd3Config(
                hidden_size_1=8,
                hidden_size_2=8,
                racing_line_features=False,
                seed=7,
            ),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            source.save(checkpoint)
            args = load_training_script().parse_args(
                [
                    "--sensor-only",
                    "--sensor-v1-stability-continuation",
                    "--resume",
                    str(checkpoint),
                    "--seed",
                    "41",
                ]
            )
            resumed = NstepTd3Learner.load(args.resume, device="cpu")
            resumed.enable_fine_tuning(
                actor_learning_rate=args.fine_tune_actor_learning_rate,
            )

        self.assertEqual(args.observation_noise_std, 0.025)
        self.assertEqual(args.train_frequency, 8)
        self.assertEqual(resumed.config.actor_learning_rate, 3e-6)
        self.assertEqual(resumed.config.exploration_noise, 0.01)
        self.assertEqual(resumed.config.policy_delay, 4)
        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(value, resumed.actor.state_dict()[name])
        for name, value in source.critic.state_dict().items():
            torch.testing.assert_close(value, resumed.critic.state_dict()[name])

    def test_sensor_steering_v2_requires_and_configures_a_sensor_checkpoint(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-steering-v2",
                    "--resume",
                    str(checkpoint),
                    "--seed",
                    "31",
                ]
            )

        self.assertEqual(args.resume, checkpoint)
        self.assertEqual(args.train_frequency, 8)
        self.assertEqual(args.fine_tune_actor_learning_rate, 1e-5)
        self.assertEqual(args.sensor_steering_limit, 0.7)
        with self.assertRaises(SystemExit):
            script.parse_args(["--sensor-only", "--sensor-steering-v2"])
        with self.assertRaises(SystemExit):
            script.parse_args(["--sensor-steering-v2"])

    def test_sensor_clean_observation_v3_matches_the_gui_contract(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-clean-observation-v3",
                    "--resume",
                    str(checkpoint),
                    "--observation-noise-std",
                    "0",
                    "--seed",
                    "37",
                ]
            )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--sensor-only",
                        "--sensor-clean-observation-v3",
                        "--resume",
                        str(checkpoint),
                        "--observation-noise-std",
                        "0.025",
                    ]
                )

        self.assertEqual(args.train_frequency, 8)
        self.assertEqual(args.fine_tune_actor_learning_rate, 1e-5)
        self.assertEqual(args.observation_noise_std, 0.0)

    def test_sensor_v1_clean_reliability_requires_clean_ten_run_evaluation(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-v1-clean-reliability-continuation",
                    "--resume",
                    str(checkpoint),
                    "--observation-noise-std",
                    "0",
                    "--seed",
                    "61",
                ]
            )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--sensor-only",
                        "--sensor-v1-clean-reliability-continuation",
                        "--resume",
                        str(checkpoint),
                        "--observation-noise-std",
                        "0.025",
                    ]
                )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--sensor-only",
                        "--sensor-v1-clean-reliability-continuation",
                        "--resume",
                        str(checkpoint),
                        "--observation-noise-std",
                        "0",
                        "--evaluation-repeats",
                        "9",
                    ]
                )

        self.assertEqual(args.observation_noise_std, 0.0)
        self.assertEqual(args.evaluation_repeats, 10)
        self.assertEqual(args.train_frequency, 8)
        self.assertEqual(args.fine_tune_actor_learning_rate, 3e-6)

    def test_sensor_v1_clean_reliability_transfers_only_the_actor(self):
        script = load_training_script()
        source = NstepTd3Learner(
            NstepTd3Config(
                hidden_size_1=8,
                hidden_size_2=8,
                racing_line_features=False,
                seed=7,
            ),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            source.save(checkpoint)
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-v1-clean-reliability-continuation",
                    "--resume",
                    str(checkpoint),
                    "--observation-noise-std",
                    "0",
                ]
            )
            transferred = script.initialize_sensor_v1_clean_reliability_learner(
                args
            )

        self.assertEqual(transferred.config.racing_line_features, False)
        self.assertEqual(transferred.config.physical_steering_limit, 1.0)
        self.assertEqual(transferred.config.actor_learning_rate, 3e-6)
        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(value, transferred.actor.state_dict()[name])
        self.assertFalse(
            torch.allclose(
                next(source.critic.parameters()),
                next(transferred.critic.parameters()),
            )
        )

    def test_clean_reliability_gate_requires_sub_ninety_eight_of_ten(self):
        script = load_training_script()
        gate = script.EvaluationPromotionGate(
            required_completion_rate=(
                script.SENSOR_V1_CLEAN_RELIABILITY_COMPLETION_RATE
            ),
            maximum_median_lap_time_seconds=(
                script.SENSOR_V1_CLEAN_RELIABILITY_MEDIAN_SECONDS
            ),
        )

        def score(completion_rate, median_lap_time):
            return script.EvaluationScore(
                completion_rate=completion_rate,
                median_lap_time_seconds=median_lap_time,
                fastest_lap_time_seconds=median_lap_time,
                median_distance_m=2943.0,
                mean_off_track_steps=0.0,
                mean_damage_delta=0.0,
            )

        self.assertFalse(gate.permits(score(0.7, 85.0)))
        self.assertFalse(gate.permits(score(0.8, 90.0)))
        self.assertTrue(gate.permits(score(0.8, 89.999)))

    def test_sensor_v1_robust_pace_requires_clean_ten_run_evaluation(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-v1-robust-pace-continuation",
                    "--resume",
                    str(checkpoint),
                    "--observation-noise-std",
                    "0",
                    "--seed",
                    "71",
                ]
            )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--sensor-only",
                        "--sensor-v1-robust-pace-continuation",
                        "--resume",
                        str(checkpoint),
                        "--observation-noise-std",
                        "0.025",
                    ]
                )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--sensor-only",
                        "--sensor-v1-robust-pace-continuation",
                        "--resume",
                        str(checkpoint),
                        "--observation-noise-std",
                        "0",
                        "--evaluation-repeats",
                        "9",
                    ]
                )

        self.assertEqual(args.observation_noise_std, 0.0)
        self.assertEqual(args.evaluation_repeats, 10)
        self.assertEqual(args.train_frequency, 8)
        self.assertEqual(args.fine_tune_actor_learning_rate, 3e-6)

    def test_sensor_v1_robust_pace_transfers_only_the_actor(self):
        script = load_training_script()
        source = NstepTd3Learner(
            NstepTd3Config(
                hidden_size_1=8,
                hidden_size_2=8,
                racing_line_features=False,
                seed=7,
            ),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            source.save(checkpoint)
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-v1-robust-pace-continuation",
                    "--resume",
                    str(checkpoint),
                    "--observation-noise-std",
                    "0",
                ]
            )
            transferred = script.initialize_sensor_v1_robust_pace_learner(
                args
            )

        self.assertEqual(transferred.config.racing_line_features, False)
        self.assertEqual(transferred.config.physical_steering_limit, 1.0)
        self.assertEqual(transferred.config.actor_learning_rate, 3e-6)
        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(value, transferred.actor.state_dict()[name])
        self.assertFalse(
            torch.allclose(
                next(source.critic.parameters()),
                next(transferred.critic.parameters()),
            )
        )

    def test_sensor_v1_robust_pace_noise_selector_has_two_profiles(self):
        script = load_training_script()

        class RandomSource:
            def __init__(self, value):
                self.value = value

            def random(self):
                return self.value

        self.assertEqual(
            script.select_sensor_v1_robust_pace_noise(RandomSource(0.0)),
            0.025,
        )
        self.assertEqual(
            script.select_sensor_v1_robust_pace_noise(RandomSource(0.25)),
            0.0,
        )

    def test_sensor_steering_v2_preserves_actor_and_resets_critic(self):
        script = load_training_script()
        source = NstepTd3Learner(
            NstepTd3Config(
                hidden_size_1=8,
                hidden_size_2=8,
                racing_line_features=False,
                seed=7,
            ),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            source.save(checkpoint)
            args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-steering-v2",
                    "--resume",
                    str(checkpoint),
                    "--seed",
                    "31",
                ]
            )
            transferred = script.initialize_sensor_steering_v2_learner(args)
            clean_args = script.parse_args(
                [
                    "--sensor-only",
                    "--sensor-clean-observation-v3",
                    "--resume",
                    str(checkpoint),
                    "--observation-noise-std",
                    "0",
                    "--seed",
                    "37",
                ]
            )
            clean_transferred = (
                script.initialize_sensor_clean_observation_v3_learner(clean_args)
            )

        self.assertEqual(transferred.config.physical_steering_limit, 0.7)
        self.assertEqual(transferred.config.racing_line_features, False)
        self.assertEqual(transferred.config.actor_learning_rate, 1e-5)
        self.assertEqual(
            clean_transferred.config.physical_steering_limit,
            1.0,
        )
        self.assertEqual(clean_transferred.config.racing_line_features, False)
        self.assertEqual(clean_transferred.config.actor_learning_rate, 1e-5)
        for name, value in source.actor.state_dict().items():
            torch.testing.assert_close(value, transferred.actor.state_dict()[name])
            torch.testing.assert_close(
                value,
                clean_transferred.actor.state_dict()[name],
            )
        self.assertFalse(
            torch.allclose(
                next(source.critic.parameters()),
                next(transferred.critic.parameters()),
            )
        )
        self.assertFalse(
            torch.allclose(
                next(source.critic.parameters()),
                next(clean_transferred.critic.parameters()),
            )
        )

    def test_evaluation_parser_accepts_a_safe_physical_steering_override(self):
        script = load_evaluation_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sensor_agent.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--policy-path",
                    str(checkpoint),
                    "--physical-steering-limit",
                    "0.7",
                ]
            )
            with self.assertRaises(SystemExit):
                script.parse_args(
                    [
                        "--policy-path",
                        str(checkpoint),
                        "--physical-steering-limit",
                        "0",
                    ]
                )

        self.assertEqual(args.physical_steering_limit, 0.7)

    def test_v4_parser_applies_the_bounded_actor_only_profile(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_evaluation.pt"
            checkpoint.touch()
            original = script.BEST_EVALUATION_PATH
            script.BEST_EVALUATION_PATH = checkpoint
            try:
                args = script.parse_args(
                    ["--resume-best", "--steering-rate-v4", "--seed", "19"]
                )
            finally:
                script.BEST_EVALUATION_PATH = original

        config = script.config_from_args(args)
        self.assertEqual(args.resume, checkpoint)
        self.assertEqual(args.total_timesteps, 1_280)
        self.assertEqual(args.train_frequency, 32)
        self.assertEqual(args.evaluation_repeats, 10)
        self.assertEqual(args.fine_tune_actor_learning_rate, 1e-6)
        self.assertEqual(args.fine_tune_replay_warmup_size, 30_000)
        self.assertEqual(args.fine_tune_critic_warmup_updates, 5_000)
        self.assertEqual(args.steering_rate_max_actor_updates, 10)
        self.assertEqual(args.steering_rate_promotion_median_seconds, 103.953)
        self.assertEqual(config.steering_rate_cost_coefficient, 0.0025)

    def test_v4_requires_the_protected_best_checkpoint_and_ten_evaluations(self):
        script = load_training_script()
        with self.assertRaises(SystemExit):
            script.parse_args(["--steering-rate-v4"])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_evaluation.pt"
            checkpoint.touch()
            original = script.BEST_EVALUATION_PATH
            script.BEST_EVALUATION_PATH = checkpoint
            try:
                with self.assertRaises(SystemExit):
                    script.parse_args(
                        [
                            "--resume-best",
                            "--steering-rate-v4",
                            "--evaluation-repeats",
                            "9",
                        ]
                    )
            finally:
                script.BEST_EVALUATION_PATH = original

    def test_v4_promotion_gate_is_strict_about_reliability_and_pace(self):
        script = load_training_script()
        gate = script.EvaluationPromotionGate(
            required_completion_rate=1.0,
            maximum_median_lap_time_seconds=103.953,
        )

        def score(completion_rate, median_lap_time):
            return script.EvaluationScore(
                completion_rate=completion_rate,
                median_lap_time_seconds=median_lap_time,
                fastest_lap_time_seconds=median_lap_time,
                median_distance_m=2943.0,
                mean_off_track_steps=0.0,
                mean_damage_delta=0.0,
            )

        self.assertFalse(gate.permits(score(0.9, 100.0)))
        self.assertFalse(gate.permits(score(1.0, 103.953)))
        self.assertTrue(gate.permits(score(1.0, 103.952)))

    def test_same_protocol_champion_is_protected_across_v4_seed_runs(self):
        script = load_training_script()
        config = NstepTd3Config(
            hidden_size_1=8,
            hidden_size_2=8,
            replay_capacity=16,
            replay_start_size=8,
            batch_size=4,
            steering_rate_cost_coefficient=0.0025,
        )
        incumbent = NstepTd3Learner(config, device="cpu")
        incumbent.best_evaluation_distance_m = 2943.2
        incumbent.best_evaluation_median_m = 2943.1
        incumbent.best_evaluation_completion_rate = 1.0
        incumbent.best_evaluation_median_lap_time_seconds = 102.5
        incumbent.best_evaluation_mean_off_track_steps = 0.0
        incumbent.best_evaluation_mean_damage_delta = 0.0

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            incumbent.save(model_dir / "best_evaluation.pt")
            fresh_candidate = NstepTd3Learner(config, device="cpu")
            script.synchronise_recorded_evaluation(fresh_candidate, model_dir)

        self.assertEqual(fresh_candidate.best_evaluation_completion_rate, 1.0)
        self.assertEqual(
            fresh_candidate.best_evaluation_median_lap_time_seconds,
            102.5,
        )
        self.assertEqual(fresh_candidate.best_evaluation_distance_m, 2943.2)

    def test_training_parser_tracks_an_explicit_resume_seed(self):
        script = load_training_script()
        args = script.parse_args(["--total-timesteps", "100", "--seed", "2"])

        self.assertTrue(args.seed_was_explicit)
        self.assertEqual(args.seed, 2)

    def test_fine_tune_flag_requires_an_existing_resume_checkpoint(self):
        script = load_training_script()
        with self.assertRaises(SystemExit):
            script.parse_args(["--total-timesteps", "100", "--fine-tune"])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent7.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--total-timesteps",
                    "100",
                    "--resume",
                    str(checkpoint),
                    "--fine-tune",
                ]
            )

        self.assertTrue(args.fine_tune)
        self.assertEqual(args.resume, checkpoint)
        self.assertEqual(args.train_frequency, 8)
        self.assertEqual(args.fine_tune_actor_learning_rate, 3e-6)
        self.assertEqual(args.fine_tune_replay_warmup_size, 30_000)
        self.assertEqual(args.fine_tune_critic_warmup_updates, 2_000)

    def test_fine_tune_actor_learning_rate_can_be_overridden(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent7.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--total-timesteps",
                    "100",
                    "--resume",
                    str(checkpoint),
                    "--fine-tune",
                    "--fine-tune-actor-learning-rate",
                    "8e-7",
                ]
            )

        self.assertEqual(args.fine_tune_actor_learning_rate, 8e-7)

    def test_fine_tune_train_frequency_can_be_overridden_explicitly(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "agent7.pt"
            checkpoint.touch()
            args = script.parse_args(
                [
                    "--total-timesteps",
                    "100",
                    "--resume",
                    str(checkpoint),
                    "--fine-tune",
                    "--train-frequency",
                    "6",
                ]
            )

        self.assertEqual(args.train_frequency, 6)

    def test_resume_pace_selects_the_independent_pace_checkpoint(self):
        script = load_training_script()
        with tempfile.TemporaryDirectory() as directory:
            pace = Path(directory) / "best_pace.pt"
            pace.touch()
            original = script.BEST_PACE_PATH
            script.BEST_PACE_PATH = pace
            try:
                args = script.parse_args(
                    ["--total-timesteps", "100", "--resume-pace", "--fine-tune"]
                )
            finally:
                script.BEST_PACE_PATH = original

        self.assertEqual(args.resume, pace)
        self.assertTrue(args.fine_tune)

    def test_evaluation_trace_flattens_existing_diagnostics(self):
        script = load_evaluation_script()
        curvature = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        info = {
            "episode": 2,
            "steps": 123,
            "distance_m": 350.0,
            "distance_from_start_m": 351.0,
            "speed_kmh": 120.0,
            "heading_angle_rad": 0.1,
            "track_position": 0.2,
            "target_track_position": 0.15,
            "racing_line_error": 0.05,
            "minimum_track_sensor_m": 12.0,
            "raw_steer": -0.25,
            "control_steer": -0.075,
            "steering_delta": -0.1,
            "raw_longitudinal": -0.4,
            "control_accel": 0.0,
            "control_brake": 0.4,
            "control_gear": 3,
            "reward_forward_velocity": 0.48,
            "reward_racing_line_factor": 0.975,
            "reward_steering_rate_penalty": -0.000025,
            "reward_terminal_penalty": 0.0,
            "reward_progress_m": 0.0,
            "reward_lap_completion_bonus": 0.0,
            "lookahead_curvature": curvature,
            "physical_failure": False,
            "termination_reason": "",
        }

        row = script.build_trace_row(1, 0.468, info)

        self.assertEqual(tuple(row), script.TRACE_COLUMNS)
        self.assertEqual(row["distance_m"], 350.0)
        self.assertEqual(row["raw_longitudinal"], -0.4)
        self.assertAlmostEqual(row["control_steer"], -0.075)
        self.assertEqual(row["control_brake"], 0.4)
        self.assertAlmostEqual(row["steering_delta"], -0.1)
        self.assertAlmostEqual(
            row["reward_steering_rate_penalty"],
            -0.000025,
        )
        self.assertAlmostEqual(row["lookahead_curvature_0m"], -1.0)
        self.assertAlmostEqual(row["lookahead_curvature_275m"], 1.0)

    def test_passive_evaluation_promotes_by_median_not_single_maximum(self):
        script = load_training_script()

        class FakeEnvironment:
            def __init__(self, distances, lap_times=None):
                self.distances = iter(distances)
                self.lap_times = iter(
                    lap_times if lap_times is not None else [None] * 3
                )
                self.last_summary = None

            def reset(self):
                distance = float(next(self.distances))
                lap_time = next(self.lap_times)
                self.last_summary = EpisodeSummary(
                    episode=1,
                    steps=1,
                    reward=distance,
                    distance_m=distance,
                    max_speed_kmh=10.0,
                    average_speed_kmh=10.0,
                    off_track_steps=0,
                    damage_delta=0.0,
                    laps_completed=0 if lap_time is None else 1,
                    best_lap_time_seconds=lap_time,
                    termination_reason=(
                        "time_limit" if lap_time is None else "lap_completed"
                    ),
                )
                return np.zeros(OBSERVATION_SIZE, dtype=np.float32), {}

            def step(self, _action):
                return (
                    np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                    0.0,
                    False,
                    True,
                    {},
                )

        class FakeLearner:
            def __init__(self):
                self.environment_steps = 100
                self.best_evaluation_distance_m = 344.0
                self.best_evaluation_median_m = 339.0
                self.best_evaluation_completion_rate = 0.0
                self.best_evaluation_median_lap_time_seconds = None
                self.best_evaluation_mean_off_track_steps = 0.0
                self.best_evaluation_mean_damage_delta = 0.0
                self.best_lap_time_seconds = None
                self.saved = []

            def select_action(self, _observation, *, deterministic):
                self.assert_deterministic = deterministic
                return np.zeros(ACTION_SIZE, dtype=np.float32)

            def save(self, path):
                self.saved.append(Path(path))

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        learner = FakeLearner()
        _evaluations, median = script.evaluate_and_record(
            FakeEnvironment([320.0, 400.0, 330.0]),
            learner,
            FakeLogger(),
            None,
            Path("models"),
            repeats=3,
        )
        self.assertEqual(median, 330.0)
        self.assertEqual(learner.best_evaluation_distance_m, 400.0)
        self.assertEqual(learner.best_evaluation_median_m, 339.0)
        self.assertEqual(learner.saved, [])

        _evaluations, median = script.evaluate_and_record(
            FakeEnvironment([340.0, 300.0, 350.0]),
            learner,
            FakeLogger(),
            None,
            Path("models"),
            repeats=3,
        )
        self.assertEqual(median, 340.0)
        self.assertEqual(learner.best_evaluation_median_m, 340.0)
        self.assertEqual(learner.saved, [Path("models/best_evaluation.pt")])

        evaluate_and_record_result = script.evaluate_and_record(
            FakeEnvironment(
                [2943.1, 2943.2, 2943.0],
                [205.0, 203.0, 204.0],
            ),
            learner,
            FakeLogger(),
            None,
            Path("models"),
            repeats=3,
        )
        self.assertEqual(evaluate_and_record_result[1], 2943.1)
        self.assertEqual(learner.best_evaluation_completion_rate, 1.0)
        self.assertEqual(
            learner.best_evaluation_median_lap_time_seconds,
            204.0,
        )
        self.assertEqual(
            learner.saved,
            [
                Path("models/best_evaluation.pt"),
                Path("models/best_evaluation.pt"),
                Path("models/best_pace.pt"),
                Path("models/best_lap.pt"),
            ],
        )

    def test_cold_start_evaluation_saves_an_immutable_candidate(self):
        script = load_training_script()

        class FakeEnvironment:
            def __init__(self):
                self.reset_calls = []
                self.last_summary = None

            def reset(self, *, force_relaunch=False):
                self.reset_calls.append(force_relaunch)
                self.last_summary = EpisodeSummary(
                    episode=len(self.reset_calls),
                    steps=1,
                    reward=1.0,
                    distance_m=900.0,
                    max_speed_kmh=100.0,
                    average_speed_kmh=80.0,
                    off_track_steps=1,
                    damage_delta=0.0,
                    laps_completed=0,
                    best_lap_time_seconds=None,
                    termination_reason="off_track",
                )
                return np.zeros(OBSERVATION_SIZE, dtype=np.float32), {}

            def step(self, _action):
                return (
                    np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                    0.0,
                    False,
                    True,
                    {},
                )

        class FakeLearner:
            environment_steps = 25000
            best_evaluation_distance_m = 0.0

            def __init__(self):
                self.saved = []

            def select_action(self, _observation, *, deterministic):
                return np.zeros(ACTION_SIZE, dtype=np.float32)

            def save(self, path):
                self.saved.append(Path(path))

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        environment = FakeEnvironment()
        learner = FakeLearner()
        with tempfile.TemporaryDirectory() as directory:
            candidate_dir = Path(directory) / "evaluation_candidates"
            script.evaluate_and_record(
                environment,
                learner,
                FakeLogger(),
                None,
                Path(directory),
                repeats=3,
                allow_promotion=False,
                cold_start_each_rollout=True,
                evaluation_candidate_dir=candidate_dir,
            )
            candidate_path = candidate_dir / "evaluation_step_0000025000.pt"
            metadata = json.loads(candidate_path.with_suffix(".json").read_text())

        self.assertEqual(environment.reset_calls, [True, True, True])
        self.assertEqual(learner.saved, [candidate_path])
        self.assertEqual(metadata["completion_rate"], 0.0)
        self.assertTrue(metadata["cold_start_each_rollout"])

    def test_faster_unreliable_evaluation_only_promotes_pace(self):
        script = load_training_script()

        class FakeEnvironment:
            def __init__(self):
                self.results = iter(
                    [
                        (2943.2, 165.0),
                        (2943.1, 164.0),
                        (900.0, None),
                    ]
                )
                self.last_summary = None

            def reset(self):
                distance, lap_time = next(self.results)
                self.last_summary = EpisodeSummary(
                    episode=1,
                    steps=1,
                    reward=distance,
                    distance_m=distance,
                    max_speed_kmh=100.0,
                    average_speed_kmh=80.0,
                    off_track_steps=0 if lap_time is not None else 1,
                    damage_delta=0.0,
                    laps_completed=0 if lap_time is None else 1,
                    best_lap_time_seconds=lap_time,
                    termination_reason=(
                        "lap_completed" if lap_time is not None else "off_track"
                    ),
                )
                return np.zeros(OBSERVATION_SIZE, dtype=np.float32), {}

            def step(self, _action):
                return (
                    np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                    0.0,
                    False,
                    True,
                    {},
                )

        class FakeLearner:
            environment_steps = 100
            best_evaluation_distance_m = 2943.3
            best_evaluation_median_m = 2943.2
            best_evaluation_completion_rate = 1.0
            best_evaluation_median_lap_time_seconds = 192.0
            best_evaluation_mean_off_track_steps = 0.0
            best_evaluation_mean_damage_delta = 0.0
            best_lap_time_seconds = 191.0

            def __init__(self):
                self.saved = []

            def select_action(self, _observation, *, deterministic):
                return np.zeros(ACTION_SIZE, dtype=np.float32)

            def save(self, path):
                self.saved.append(Path(path))

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        learner = FakeLearner()
        script.evaluate_and_record(
            FakeEnvironment(),
            learner,
            FakeLogger(),
            None,
            Path("models"),
            repeats=3,
        )

        self.assertEqual(learner.best_evaluation_completion_rate, 1.0)
        self.assertEqual(learner.best_lap_time_seconds, 164.0)
        self.assertEqual(
            learner.saved,
            [Path("models/best_pace.pt"), Path("models/best_lap.pt")],
        )

    def test_lap_aware_score_prioritises_reliability_then_median_pace(self):
        script = load_training_script()
        incumbent = script.EvaluationScore(
            completion_rate=1.0,
            median_lap_time_seconds=204.0,
            fastest_lap_time_seconds=203.0,
            median_distance_m=2943.2,
            mean_off_track_steps=0.0,
            mean_damage_delta=0.0,
        )
        faster = script.EvaluationScore(
            completion_rate=1.0,
            median_lap_time_seconds=198.0,
            fastest_lap_time_seconds=197.0,
            median_distance_m=2943.0,
            mean_off_track_steps=0.0,
            mean_damage_delta=0.0,
        )
        faster_but_unreliable = script.EvaluationScore(
            completion_rate=2.0 / 3.0,
            median_lap_time_seconds=180.0,
            fastest_lap_time_seconds=179.0,
            median_distance_m=2943.4,
            mean_off_track_steps=0.0,
            mean_damage_delta=0.0,
        )
        slower = script.EvaluationScore(
            completion_rate=1.0,
            median_lap_time_seconds=210.0,
            fastest_lap_time_seconds=209.0,
            median_distance_m=2943.4,
            mean_off_track_steps=0.0,
            mean_damage_delta=0.0,
        )

        self.assertTrue(script.evaluation_score_is_better(faster, incumbent))
        self.assertFalse(
            script.evaluation_score_is_better(
                faster_but_unreliable,
                incumbent,
            )
        )
        self.assertFalse(script.evaluation_score_is_better(slower, incumbent))

    def test_pace_score_ignores_completed_laps_with_safety_incidents(self):
        script = load_training_script()
        clean = EpisodeSummary(
            episode=1,
            steps=1,
            reward=1.0,
            distance_m=2943.0,
            max_speed_kmh=100.0,
            average_speed_kmh=80.0,
            off_track_steps=0,
            damage_delta=0.0,
            laps_completed=1,
            best_lap_time_seconds=180.0,
            termination_reason="lap_completed",
        )
        dirty = EpisodeSummary(
            episode=2,
            steps=1,
            reward=1.0,
            distance_m=2943.0,
            max_speed_kmh=120.0,
            average_speed_kmh=90.0,
            off_track_steps=1,
            damage_delta=0.0,
            laps_completed=1,
            best_lap_time_seconds=150.0,
            termination_reason="lap_completed",
        )

        score = script.summarise_evaluations([clean, dirty])

        self.assertEqual(score.median_lap_time_seconds, 165.0)
        self.assertEqual(score.fastest_lap_time_seconds, 180.0)

    def test_lap_aware_score_uses_distance_when_no_policy_finishes(self):
        script = load_training_script()
        incumbent = script.EvaluationScore(0.0, None, None, 350.0, 1.0, 0.0)
        farther = script.EvaluationScore(0.0, None, None, 400.0, 2.0, 0.0)
        safer_but_shorter = script.EvaluationScore(
            0.0,
            None,
            None,
            300.0,
            0.0,
            0.0,
        )

        self.assertTrue(script.evaluation_score_is_better(farther, incumbent))
        self.assertFalse(
            script.evaluation_score_is_better(safer_but_shorter, incumbent)
        )

    def test_training_cadence_runs_one_update_block_every_four_steps(self):
        script = load_training_script()

        self.assertFalse(script.should_run_gradient_updates(1, 4))
        self.assertFalse(script.should_run_gradient_updates(3, 4))
        self.assertTrue(script.should_run_gradient_updates(4, 4))
        self.assertTrue(script.should_run_gradient_updates(8, 4))
        with self.assertRaisesRegex(ValueError, "train_frequency"):
            script.should_run_gradient_updates(1, 0)

    def test_random_replay_warmup_precedes_learning_budget(self):
        script = load_training_script()
        learner = NstepTd3Learner(
            NstepTd3Config(
                observation_size=4,
                action_size=2,
                hidden_size_1=8,
                hidden_size_2=8,
                replay_capacity=16,
                replay_start_size=4,
                batch_size=2,
                n_steps=3,
                seed=9,
            ),
            device="cpu",
        )

        class FakeEnvironment:
            def __init__(self):
                self.episode_steps = 0
                self.episode = 0
                self.last_summary = None

            def reset(self):
                self.episode += 1
                self.episode_steps = 0
                return np.zeros(4, dtype=np.float32), {}

            def step(self, _action):
                self.episode_steps += 1
                truncated = self.episode_steps == 2
                observation = np.full(4, self.episode_steps, dtype=np.float32)
                if truncated:
                    self.last_summary = EpisodeSummary(
                        episode=self.episode,
                        steps=2,
                        reward=2.0,
                        distance_m=2.0,
                        max_speed_kmh=10.0,
                        average_speed_kmh=10.0,
                        off_track_steps=0,
                        damage_delta=0.0,
                        laps_completed=0,
                        best_lap_time_seconds=None,
                        termination_reason="time_limit",
                    )
                return observation, 1.0, False, truncated, {}

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        observation, interactions = script.collect_replay_warmup(
            FakeEnvironment(),
            learner,
            NstepTransitionAccumulator(3, learner.config.gamma),
            FakeLogger(),
            None,
            use_random_actions=True,
        )

        self.assertEqual(interactions, 4)
        self.assertTrue(learner.ready_to_train)
        self.assertEqual(learner.environment_steps, 0)
        self.assertEqual(observation.shape, (4,))

    def test_replay_warmup_can_collect_a_larger_fixed_policy_buffer(self):
        script = load_training_script()
        learner = NstepTd3Learner(
            NstepTd3Config(
                observation_size=4,
                action_size=2,
                hidden_size_1=8,
                hidden_size_2=8,
                replay_capacity=16,
                replay_start_size=4,
                batch_size=2,
                n_steps=3,
                seed=9,
            ),
            device="cpu",
        )

        class FakeEnvironment:
            def __init__(self):
                self.episode_steps = 0
                self.episode = 0
                self.last_summary = None

            def reset(self):
                self.episode += 1
                self.episode_steps = 0
                return np.zeros(4, dtype=np.float32), {}

            def step(self, _action):
                self.episode_steps += 1
                truncated = self.episode_steps == 2
                if truncated:
                    self.last_summary = EpisodeSummary(
                        episode=self.episode,
                        steps=2,
                        reward=2.0,
                        distance_m=2.0,
                        max_speed_kmh=10.0,
                        average_speed_kmh=10.0,
                        off_track_steps=0,
                        damage_delta=0.0,
                        laps_completed=0,
                        best_lap_time_seconds=None,
                        termination_reason="time_limit",
                    )
                return (
                    np.full(4, self.episode_steps, dtype=np.float32),
                    1.0,
                    False,
                    truncated,
                    {},
                )

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        deterministic_arguments = []
        select_action = learner.select_action

        def recording_select_action(observation, *, deterministic):
            deterministic_arguments.append(deterministic)
            return select_action(observation, deterministic=deterministic)

        learner.select_action = recording_select_action
        _observation, interactions = script.collect_replay_warmup(
            FakeEnvironment(),
            learner,
            NstepTransitionAccumulator(3, learner.config.gamma),
            FakeLogger(),
            None,
            use_random_actions=False,
            deterministic_policy_actions=True,
            target_replay_size=6,
        )

        self.assertEqual(interactions, 6)
        self.assertEqual(len(learner.replay), 6)
        self.assertEqual(deterministic_arguments, [True] * 6)

    def test_v5_warmup_holds_launch_biased_random_action_segments(self):
        script = load_training_script()
        learner = NstepTd3Learner(
            NstepTd3Config(
                observation_size=4,
                action_size=2,
                hidden_size_1=8,
                hidden_size_2=8,
                replay_capacity=16,
                replay_start_size=12,
                batch_size=2,
                n_steps=3,
                seed=9,
            ),
            device="cpu",
        )
        sampled_actions = iter(
            (
                np.asarray([0.5, 0.0], dtype=np.float32),
                np.asarray([-0.5, -0.8], dtype=np.float32),
            )
        )
        learner.random_action = lambda: next(sampled_actions)

        class FakeEnvironment:
            def __init__(self):
                self.actions = []
                self.last_summary = None

            def reset(self):
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                truncated = len(self.actions) == 12
                if truncated:
                    self.last_summary = EpisodeSummary(
                        episode=1,
                        steps=12,
                        reward=2.0,
                        distance_m=2.0,
                        max_speed_kmh=10.0,
                        average_speed_kmh=10.0,
                        off_track_steps=0,
                        damage_delta=0.0,
                        laps_completed=0,
                        best_lap_time_seconds=None,
                        termination_reason="time_limit",
                    )
                return np.zeros(4, dtype=np.float32), 0.0, False, truncated, {}

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        environment = FakeEnvironment()
        _observation, interactions = script.collect_replay_warmup(
            environment,
            learner,
            NstepTransitionAccumulator(3, learner.config.gamma),
            FakeLogger(),
            None,
            use_random_actions=True,
            random_action_hold_interactions=10,
            launch_capable_random_actions=True,
        )

        self.assertEqual(interactions, 12)
        self.assertTrue(all(
            np.array_equal(action, environment.actions[0])
            for action in environment.actions[:10]
        ))
        self.assertTrue(all(
            np.array_equal(action, environment.actions[10])
            for action in environment.actions[10:]
        ))
        self.assertGreater(environment.actions[0][1], 0.0)
        self.assertLess(environment.actions[10][1], 0.0)

    def test_sensor_only_warmup_limits_steering_without_changing_runtime_contract(self):
        script = load_training_script()
        learner = NstepTd3Learner(
            NstepTd3Config(
                observation_size=4,
                action_size=2,
                hidden_size_1=8,
                hidden_size_2=8,
                replay_capacity=16,
                replay_start_size=12,
                batch_size=2,
                n_steps=3,
                seed=9,
            ),
            device="cpu",
        )
        sampled_actions = iter(
            (
                np.asarray([0.9, 0.0], dtype=np.float32),
                np.asarray([-0.8, -0.8], dtype=np.float32),
            )
        )
        learner.random_action = lambda: next(sampled_actions)

        class FakeEnvironment:
            def __init__(self):
                self.actions = []
                self.last_summary = None

            def reset(self):
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                truncated = len(self.actions) == 12
                if truncated:
                    self.last_summary = EpisodeSummary(
                        episode=1,
                        steps=12,
                        reward=2.0,
                        distance_m=2.0,
                        max_speed_kmh=10.0,
                        average_speed_kmh=10.0,
                        off_track_steps=0,
                        damage_delta=0.0,
                        laps_completed=0,
                        best_lap_time_seconds=None,
                        termination_reason="time_limit",
                    )
                return np.zeros(4, dtype=np.float32), 0.0, False, truncated, {}

        class FakeLogger:
            def write_episode(self, *_args, **_kwargs):
                return None

        environment = FakeEnvironment()
        _observation, interactions = script.collect_replay_warmup(
            environment,
            learner,
            NstepTransitionAccumulator(3, learner.config.gamma),
            FakeLogger(),
            None,
            use_random_actions=True,
            random_action_hold_interactions=10,
            launch_capable_random_actions=True,
            random_action_steering_limit=0.3,
        )

        self.assertEqual(interactions, 12)
        self.assertTrue(all(
            np.array_equal(action, environment.actions[0])
            for action in environment.actions[:10]
        ))
        self.assertTrue(all(
            np.array_equal(action, environment.actions[10])
            for action in environment.actions[10:]
        ))
        self.assertAlmostEqual(float(environment.actions[0][0]), 0.3)
        self.assertAlmostEqual(float(environment.actions[10][0]), -0.3)
        self.assertGreater(environment.actions[0][1], 0.0)
        self.assertLess(environment.actions[10][1], 0.0)
        self.assertEqual(learner.config.physical_steering_limit, 1.0)


class _FakeClient:
    def __init__(self, value):
        self.S = type("Sensors", (), {"d": value})()

    def shutdown(self):
        return None


class _FakeTorcsEnv:
    def __init__(self, value):
        self.client = _FakeClient(value)
        self.reset_argv = None

    def reset(self, relaunch=False):
        self.reset_argv = list(sys.argv)
        return None

    def end(self):
        return None


class _FakeRunner:
    def __init__(self, initial, next_values):
        self.env = _FakeTorcsEnv(initial)
        self.next_values = list(next_values)
        self.controls = []

    def _step_full_control_agent(self, controls):
        self.controls.append(dict(controls))
        value = self.next_values.pop(0)
        self.env.client.S.d = value
        return value, 0.0, False, {}

    def shutdown(self):
        return None


class NstepTorcsEnvironmentTests(unittest.TestCase):
    def test_legacy_torcs_argv_restores_trainer_arguments_after_failure(self):
        original_argv = sys.argv
        trainer_argv = [
            "train_n_step_td3_agent.py",
            "--total-timesteps",
            "50000",
        ]
        sys.argv = trainer_argv
        try:
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with _legacy_torcs_argv():
                    self.assertEqual(sys.argv, ["train_n_step_td3_agent.py"])
                    raise RuntimeError("test failure")
            self.assertIs(sys.argv, trainer_argv)
        finally:
            sys.argv = original_argv

    def test_environment_hides_trainer_arguments_from_legacy_reset(self):
        runner = _FakeRunner(telemetry(), [])
        environment = NstepTorcsEnvironment(
            observation_noise_std=0.0,
            runner=runner,
        )
        original_argv = sys.argv
        trainer_argv = [
            "train_n_step_td3_agent.py",
            "--total-timesteps",
            "50000",
            "--no-tensorboard",
        ]
        sys.argv = trainer_argv
        try:
            environment.reset()
            self.assertEqual(
                runner.env.reset_argv,
                ["train_n_step_td3_agent.py"],
            )
            self.assertIs(sys.argv, trainer_argv)
        finally:
            sys.argv = original_argv

    def test_environment_marks_time_limit_as_truncation_not_failure(self):
        initial = telemetry()
        following = telemetry(distRaced=10.0, distFromStart=10.0)
        environment = NstepTorcsEnvironment(
            max_episode_steps=1,
            observation_noise_std=0.0,
            runner=_FakeRunner(initial, [following]),
        )
        observation, _ = environment.reset()
        next_observation, reward, terminated, truncated, info = environment.step(
            np.asarray([0.0, 0.5], dtype=np.float32)
        )

        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(next_observation.shape, (OBSERVATION_SIZE,))
        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["TimeLimit.truncated"])

    def test_environment_marks_off_track_as_physical_terminal(self):
        initial = telemetry()
        following = telemetry(trackPos=1.2, distRaced=2.0)
        environment = NstepTorcsEnvironment(
            observation_noise_std=0.0,
            runner=_FakeRunner(initial, [following]),
        )
        environment.reset()
        _observation, reward, terminated, truncated, info = environment.step(
            np.asarray([0.0, 0.5], dtype=np.float32)
        )

        self.assertEqual(reward, -10.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["physical_failure"])

    def test_environment_exposes_diagnostics_without_changing_reward(self):
        initial = telemetry(gear=3, rpm=4_000.0)
        following = telemetry(
            speedX=100.0,
            angle=0.1,
            trackPos=0.2,
            distRaced=350.0,
            distFromStart=350.0,
            gear=3,
            rpm=4_000.0,
        )
        environment = NstepTorcsEnvironment(
            observation_noise_std=0.0,
            runner=_FakeRunner(initial, [following]),
        )
        environment.reset()

        _observation, reward, terminated, truncated, info = environment.step(
            np.asarray([-0.25, -0.4], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["distance_from_start_m"], 350.0)
        self.assertEqual(info["heading_angle_rad"], 0.1)
        self.assertAlmostEqual(info["raw_steer"], -0.25)
        self.assertAlmostEqual(info["raw_longitudinal"], -0.4)
        self.assertEqual(info["control_accel"], 0.0)
        self.assertAlmostEqual(info["control_brake"], 0.4)
        self.assertEqual(info["control_gear"], 3)
        self.assertAlmostEqual(
            info["racing_line_error"],
            info["track_position"] - info["target_track_position"],
        )
        self.assertAlmostEqual(
            reward,
            info["reward_forward_velocity"]
            * info["reward_racing_line_factor"],
        )

    def test_environment_v4_penalises_action_change_without_filtering_steer(self):
        initial = telemetry()
        first = telemetry(speedX=100.0, distRaced=10.0, distFromStart=10.0)
        second = telemetry(speedX=100.0, distRaced=20.0, distFromStart=20.0)
        environment = NstepTorcsEnvironment(
            observation_noise_std=0.0,
            steering_rate_cost_coefficient=0.0025,
            runner=_FakeRunner(initial, [first, second]),
        )
        environment.reset()

        _observation, first_reward, *_rest, first_info = environment.step(
            np.asarray([1.0, 0.5], dtype=np.float32)
        )
        _observation, second_reward, *_rest, second_info = environment.step(
            np.asarray([-1.0, 0.5], dtype=np.float32)
        )

        self.assertEqual(first_info["raw_steer"], 1.0)
        self.assertEqual(second_info["raw_steer"], -1.0)
        self.assertEqual(first_info["steering_delta"], 1.0)
        self.assertEqual(second_info["steering_delta"], -2.0)
        self.assertAlmostEqual(
            first_info["reward_steering_rate_penalty"],
            -0.0025,
        )
        self.assertAlmostEqual(
            second_info["reward_steering_rate_penalty"],
            -0.01,
        )
        self.assertAlmostEqual(
            first_reward,
            first_info["reward_forward_velocity"]
            * first_info["reward_racing_line_factor"]
            - 0.0025,
        )
        self.assertAlmostEqual(
            second_reward,
            second_info["reward_forward_velocity"]
            * second_info["reward_racing_line_factor"]
            - 0.01,
        )

    def test_environment_v5_bounds_steering_and_uses_progress_reward(self):
        initial = telemetry(distRaced=100.0)
        following = telemetry(
            speedX=120.0,
            distRaced=100.8,
            distFromStart=100.8,
        )
        runner = _FakeRunner(initial, [following])
        environment = NstepTorcsEnvironment(
            max_episode_steps=1,
            observation_noise_std=0.0,
            physical_steering_limit=0.3,
            progress_reward=True,
            runner=runner,
        )
        environment.reset()

        _observation, reward, terminated, truncated, info = environment.step(
            np.asarray([1.0, 0.75], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, 0.8)
        self.assertAlmostEqual(runner.controls[0]["steer"], 0.3)
        self.assertEqual(info["raw_steer"], 1.0)
        self.assertAlmostEqual(info["control_steer"], 0.3)
        self.assertAlmostEqual(info["reward_progress_m"], 0.8)
        self.assertEqual(info["reward_racing_line_factor"], 0.0)

    def test_environment_sensor_only_uses_track_centre_without_a_map(self):
        initial = telemetry(distRaced=100.0, trackPos=0.0)
        following = telemetry(
            speedX=100.0,
            angle=0.0,
            trackPos=0.25,
            distRaced=100.5,
            distFromStart=100.5,
        )
        runner = _FakeRunner(initial, [following])
        environment = NstepTorcsEnvironment(
            max_episode_steps=1,
            observation_noise_std=0.025,
            racing_line_features=False,
            runner=runner,
        )
        observation, _ = environment.reset()

        next_observation, reward, terminated, truncated, info = environment.step(
            np.asarray([0.0, 0.75], dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, (100.0 / 250.0) * 0.75)
        self.assertIsNone(environment.racing_line_path)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(next_observation.shape, (OBSERVATION_SIZE,))
        for encoded in (observation, next_observation):
            history = encoded[
                : BASE_OBSERVATION_SIZE * 3
            ].reshape(3, BASE_OBSERVATION_SIZE)
            np.testing.assert_array_equal(
                history[-1, -13:],
                np.zeros(13, dtype=np.float32),
            )
        self.assertEqual(info["target_track_position"], 0.0)
        np.testing.assert_array_equal(
            info["lookahead_curvature"],
            np.zeros(12, dtype=np.float32),
        )

    def test_environment_applies_reproducible_observation_noise(self):
        first_environment = NstepTorcsEnvironment(
            observation_noise_std=0.025,
            seed=17,
            runner=_FakeRunner(telemetry(), []),
        )
        second_environment = NstepTorcsEnvironment(
            observation_noise_std=0.025,
            seed=17,
            runner=_FakeRunner(telemetry(), []),
        )

        first, _ = first_environment.reset()
        second, _ = second_environment.reset()

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.any(first != 0.0))

    def test_environment_can_switch_observation_noise_between_episodes(self):
        environment = NstepTorcsEnvironment(
            observation_noise_std=0.0,
            runner=_FakeRunner(telemetry(), []),
        )

        environment.set_observation_noise_std(0.025)
        self.assertEqual(environment.observation_noise_std, 0.025)
        environment.set_observation_noise_std(0.0)
        self.assertEqual(environment.observation_noise_std, 0.0)
        with self.assertRaises(ValueError):
            environment.set_observation_noise_std(-0.01)
        with self.assertRaises(ValueError):
            environment.set_observation_noise_std(float("nan"))


if __name__ == "__main__":
    unittest.main()
