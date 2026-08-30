import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.n_step_td3_agent import NstepTd3Agent
from n_step_td3.contracts import (
    ACTION_SIZE,
    BASE_OBSERVATION_NOISE_STD,
    BASE_OBSERVATION_SIZE,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    REFERENCE_OBSERVATION_NOISE_LEVEL,
    REWARD_VERSION,
    HistoryEncoder,
    apply_observation_noise,
    build_base_observation,
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

        self.assertEqual(learner.config.actor_learning_rate, 3e-5)
        self.assertEqual(learner.config.critic_learning_rate, critic_learning_rate)
        self.assertEqual(learner.config.exploration_noise, 0.01)
        self.assertEqual(learner.config.longitudinal_exploration_noise, 0.01)
        self.assertEqual(learner.config.policy_delay, 4)
        self.assertEqual(learner.actor_optimizer.param_groups[0]["lr"], 3e-5)
        np.testing.assert_array_equal(
            learner._exploration_noise_scale,
            np.asarray([0.01, 0.01], dtype=np.float32),
        )
        for name, value in learner.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name])

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
        self.assertFalse(args.seed_was_explicit)
        self.assertEqual(args.seed, 0)

    def test_v3_training_outputs_are_isolated_from_legacy_models(self):
        script = load_training_script()

        self.assertEqual(script.MODEL_DIR.name, "agent7_n_step_td3_v3")
        self.assertEqual(script.RUNS_DIR.name, "agent7_n_step_td3_v3")

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
        self.assertEqual(args.fine_tune_replay_warmup_size, 30_000)
        self.assertEqual(args.fine_tune_critic_warmup_updates, 2_000)

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
            "raw_longitudinal": -0.4,
            "control_accel": 0.0,
            "control_brake": 0.4,
            "control_gear": 3,
            "reward_forward_velocity": 0.48,
            "reward_racing_line_factor": 0.975,
            "reward_terminal_penalty": 0.0,
            "lookahead_curvature": curvature,
            "physical_failure": False,
            "termination_reason": "",
        }

        row = script.build_trace_row(1, 0.468, info)

        self.assertEqual(tuple(row), script.TRACE_COLUMNS)
        self.assertEqual(row["distance_m"], 350.0)
        self.assertEqual(row["raw_longitudinal"], -0.4)
        self.assertEqual(row["control_brake"], 0.4)
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

    def _step_full_control_agent(self, _controls):
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


if __name__ == "__main__":
    unittest.main()
