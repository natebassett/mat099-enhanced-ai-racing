from __future__ import annotations

import importlib.util
import json
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
    EXTERNAL_EVALUATION_SEED_MIN,
    FEATURE_NAMES,
    TRAINING_CHALLENGE_SEED_MAX,
    Td3ScratchAgent,
    build_rotating_training_seed_suite,
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

    def test_seeded_noise_is_applied_before_action_decode_and_rate_limit(self):
        class FixedNoise:
            steering_noise = 0.09

            def __call__(self):
                return np.asarray([0.1, 0.0, 0.0], dtype=np.float32)

            def reset(self):
                return None

        agent = Td3ScratchAgent(
            policy=DummyPolicy([0.0, 0.7, 0.0]),
            action_noise=FixedNoise(),
        )

        action = agent.act(None, _telemetry(speed=20.0))

        self.assertAlmostEqual(agent.last_policy_action[0], 0.0)
        self.assertAlmostEqual(agent.last_raw_action[0], 0.1)
        self.assertAlmostEqual(action["steer"], 0.09)
        self.assertAlmostEqual(
            agent.telemetry_debug()["td3_action_steering_noise"],
            0.09,
        )

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

    def test_actor_learning_rate_freezes_ramps_and_then_decays(self):
        schedule = train_td3_agent.gradual_actor_learning_rate_schedule(
            1e-5,
            3e-6,
            actor_update_start=40_000,
            unfreeze_steps=40_000,
            total_timesteps=300_000,
        )

        self.assertEqual(schedule(0), 0.0)
        self.assertEqual(schedule(40_000), 0.0)
        self.assertAlmostEqual(schedule(60_000), 5e-6)
        self.assertAlmostEqual(schedule(80_000), 1e-5)
        self.assertAlmostEqual(schedule(190_000), 6.5e-6)
        self.assertAlmostEqual(schedule(300_000), 3e-6)

    def test_continuation_phase_reports_each_update_stage(self):
        phase = train_td3_agent.continuation_training_phase
        kwargs = {
            "gradient_update_start": 20_000,
            "actor_update_start": 40_000,
            "actor_full_unfreeze": 80_000,
            "pre_update_phase": "replay_refill",
        }

        self.assertEqual(phase(19_999, **kwargs), "replay_refill")
        self.assertEqual(phase(20_000, **kwargs), "critic_warmup")
        self.assertEqual(phase(40_000, **kwargs), "actor_unfreeze")
        self.assertEqual(phase(80_000, **kwargs), "learning")

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

    def test_safe_actor_probe_decision_requires_paired_non_inferiority(self):
        accepted = train_td3_agent.safe_actor_probe_decision(
            [1430.0, 1300.0, 1420.0],
            [1442.0, 1310.0, 1440.0],
            minimum_improvement_m=5.0,
            maximum_paired_regression_m=100.0,
        )
        self.assertEqual(accepted.decision, "accepted")
        self.assertEqual(accepted.candidate_median_distance_m, 1440.0)
        self.assertEqual(accepted.candidate_minimum_distance_m, 1310.0)
        self.assertEqual(accepted.worst_paired_delta_m, 10.0)

        aggregate_improvement_with_seed_collapse = (
            train_td3_agent.safe_actor_probe_decision(
                [1400.0, 900.0, 500.0],
                [1200.0, 1500.0, 600.0],
                minimum_improvement_m=5.0,
                maximum_paired_regression_m=100.0,
            )
        )
        self.assertGreater(
            aggregate_improvement_with_seed_collapse.aggregate_median_gain_m,
            0.0,
        )
        self.assertGreater(
            aggregate_improvement_with_seed_collapse.minimum_gain_m,
            0.0,
        )
        self.assertEqual(
            aggregate_improvement_with_seed_collapse.decision,
            "rolled_back",
        )
        self.assertEqual(
            aggregate_improvement_with_seed_collapse.worst_paired_delta_m,
            -200.0,
        )

    def test_training_challenge_seed_suites_rotate_outside_evaluation_domain(self):
        first = build_rotating_training_seed_suite(
            repeats=5,
            base_seed=20260820,
            candidate_id=1,
        )
        repeated = build_rotating_training_seed_suite(
            repeats=5,
            base_seed=20260820,
            candidate_id=1,
        )
        second = build_rotating_training_seed_suite(
            repeats=5,
            base_seed=20260820,
            candidate_id=2,
        )

        self.assertEqual(first, repeated)
        self.assertTrue(set(first).isdisjoint(second))
        self.assertTrue(all(seed <= TRAINING_CHALLENGE_SEED_MAX for seed in first))
        self.assertTrue(all(seed < EXTERNAL_EVALUATION_SEED_MIN for seed in second))

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
        self.assertEqual(callback.model.gradient_steps, 0)
        self.assertFalse(env.modes[-1]["deterministic_probe"])
        self.assertEqual(env.modes[-1]["training_phase"], "warmup")

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
        self.assertEqual(env.modes[-1]["training_phase"], "deterministic_probe")

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
        self.assertEqual(env.modes[-1]["training_phase"], "learning")

    def test_continuation_callback_refills_replay_before_enabling_gradients(self):
        class FakeBaseCallback:
            def __init__(self, verbose=0):
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
            learning_starts=10_000,
            initial_sigma=0.04,
            final_sigma=0.02,
            decay_steps=200_000,
            probe_interval=10,
            training_gradient_steps=1,
            pre_update_training_phase="replay_refill",
            actor_update_start=30_000,
            actor_full_unfreeze=50_000,
        )

        callback._on_training_start()
        self.assertEqual(callback.model.gradient_steps, 0)
        self.assertEqual(env.modes[-1]["training_phase"], "replay_refill")

        callback.num_timesteps = 9_999
        callback.locals = {"infos": [{}]}
        callback._on_step()
        self.assertEqual(callback.model.gradient_steps, 0)

        callback.num_timesteps = 10_000
        callback._on_step()
        self.assertEqual(callback.model.gradient_steps, 1)
        self.assertEqual(env.modes[-1]["training_phase"], "critic_warmup")

        callback.num_timesteps = 30_000
        callback._on_step()
        self.assertEqual(env.modes[-1]["training_phase"], "actor_unfreeze")

        callback.num_timesteps = 50_000
        callback._on_step()
        self.assertEqual(env.modes[-1]["training_phase"], "learning")

    def test_safe_actor_candidate_forces_repeated_probes_then_rolls_back(self):
        class FakeBaseCallback:
            def __init__(self, verbose=0):
                self.locals = {}
                self.num_timesteps = 50_000

        class FakeModel:
            def __init__(self):
                self.action_noise = None
                self.gradient_steps = 1
                self.waiting = True
                self.probes = 0

            def safe_actor_probe_required(self):
                return self.waiting

            def safe_actor_probe_context(self):
                if not self.waiting:
                    return None
                return {
                    "mode": (
                        "accepted_reference" if self.probes == 0 else "candidate"
                    ),
                    "candidate_id": 1,
                    "probe_index": 1,
                    "probe_repeats": 1,
                    "seed": 100,
                }

            def record_safe_actor_probe(self, summary):
                self.probes += 1
                decision = (
                    "reference_recorded" if self.probes == 1 else "rolled_back"
                )
                if decision == "rolled_back":
                    self.waiting = False
                return {
                    "candidate_id": 1,
                    "probe_index": 1,
                    "probe_repeats": 1,
                    "decision": decision,
                    "mode": (
                        "accepted_reference" if self.probes == 1 else "candidate"
                    ),
                    "seed": 100,
                    "median_distance_m": (
                        None if decision == "reference_recorded" else 300.0
                    ),
                    "minimum_distance_m": (
                        None if decision == "reference_recorded" else 300.0
                    ),
                    "reference_median_distance_m": (
                        None if decision == "reference_recorded" else 500.0
                    ),
                    "reference_minimum_distance_m": (
                        None if decision == "reference_recorded" else 500.0
                    ),
                    "paired_median_delta_m": (
                        None if decision == "reference_recorded" else -200.0
                    ),
                    "worst_paired_delta_m": (
                        None if decision == "reference_recorded" else -200.0
                    ),
                    "paired_regressions": (
                        None if decision == "reference_recorded" else 1
                    ),
                    "accepted_distance_m": 0.0,
                    "accepted_minimum_distance_m": 0.0,
                }

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
            learning_starts=0,
            initial_sigma=0.02,
            final_sigma=0.006,
            decay_steps=100_000,
            probe_interval=10,
            training_gradient_steps=1,
        )
        callback.model = FakeModel()
        callback.current_episode_is_probe = True
        first_summary = {
            "policy_controlled": True,
            "deterministic_probe": True,
            "safe_actor_candidate_probe": True,
            "curriculum_eligible": False,
            "distance_m": 500.0,
        }
        callback.locals = {"infos": [{"episode_summary": first_summary}]}

        callback._on_step()

        self.assertEqual(
            first_summary["safe_actor_decision"],
            "reference_recorded",
        )
        self.assertTrue(env.modes[-1]["deterministic_probe"])
        self.assertTrue(env.modes[-1]["safe_actor_candidate_probe"])
        self.assertEqual(env.modes[-1]["safe_actor_probe_mode"], "candidate")
        self.assertEqual(env.modes[-1]["safe_actor_probe_seed"], 100)
        self.assertIsInstance(
            callback.model.action_noise,
            train_td3_agent.SeededSteeringActionNoise,
        )

        second_summary = {
            "policy_controlled": True,
            "deterministic_probe": True,
            "safe_actor_candidate_probe": True,
            "curriculum_eligible": False,
            "distance_m": 300.0,
        }
        callback.locals = {"infos": [{"episode_summary": second_summary}]}
        callback._on_step()

        self.assertEqual(second_summary["safe_actor_decision"], "rolled_back")
        self.assertFalse(env.modes[-1]["deterministic_probe"])
        self.assertFalse(env.modes[-1]["safe_actor_candidate_probe"])

    def test_accepted_safe_actor_probe_is_committed_to_curriculum(self):
        class FakeBaseCallback:
            def __init__(self, verbose=0):
                self.locals = {}
                self.num_timesteps = 50_000

        class FakeModel:
            action_noise = None
            gradient_steps = 1
            waiting = True

            def safe_actor_probe_required(self):
                return self.waiting

            def safe_actor_probe_context(self):
                if not self.waiting:
                    return None
                return {
                    "mode": "candidate",
                    "candidate_id": 2,
                    "probe_index": 3,
                    "probe_repeats": 3,
                    "seed": 102,
                }

            def record_safe_actor_probe(self, summary):
                self.waiting = False
                return {
                    "candidate_id": 2,
                    "probe_index": 3,
                    "probe_repeats": 3,
                    "decision": "accepted",
                    "mode": "candidate",
                    "seed": 102,
                    "median_distance_m": 1760.0,
                    "minimum_distance_m": 1500.0,
                    "reference_median_distance_m": 1740.0,
                    "reference_minimum_distance_m": 1490.0,
                    "paired_median_delta_m": 20.0,
                    "worst_paired_delta_m": 10.0,
                    "paired_regressions": 0,
                    "accepted_distance_m": 1760.0,
                    "accepted_minimum_distance_m": 1500.0,
                }

        class FakeEnv:
            def __init__(self):
                self.modes = []
                self.accepted = []

            def set_policy_mode(self, **mode):
                self.modes.append(mode)

            def accept_safe_actor_probe(self, summary, *, median_distance_m):
                summary["curriculum_eligible"] = True
                self.accepted.append(median_distance_m)

        env = FakeEnv()
        Callback = train_td3_agent.make_exploration_schedule_callback_class(
            FakeBaseCallback
        )
        callback = Callback(
            env,
            lambda sigma: ("noise", sigma),
            learning_starts=0,
            initial_sigma=0.02,
            final_sigma=0.006,
            decay_steps=100_000,
            probe_interval=10,
            training_gradient_steps=1,
        )
        callback.model = FakeModel()
        callback.current_episode_is_probe = True
        summary = {
            "policy_controlled": True,
            "deterministic_probe": True,
            "safe_actor_candidate_probe": True,
            "curriculum_eligible": False,
            "distance_m": 1750.0,
        }
        callback.locals = {"infos": [{"episode_summary": summary}]}

        callback._on_step()

        self.assertEqual(summary["safe_actor_decision"], "accepted")
        self.assertTrue(summary["curriculum_eligible"])
        self.assertEqual(env.accepted, [1760.0])

    def test_default_training_args_use_stable_td3_warmup(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = train_td3_agent.parse_args()

        self.assertEqual(args.learning_starts, 20_000)
        self.assertFalse(args.continuation)
        self.assertEqual(args.start_stage, "launch")
        self.assertEqual(args.replay_refill_steps, 0)
        self.assertEqual(args.critic_warmup_steps, 0)
        self.assertEqual(args.actor_unfreeze_steps, 0)
        self.assertEqual(args.buffer_size, 1_000_000)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.learning_rate_final, 1e-4)
        self.assertEqual(args.actor_learning_rate, 3e-4)
        self.assertEqual(args.actor_learning_rate_final, 1e-4)
        self.assertEqual(args.checkpoint_freq, 25_000)
        self.assertEqual(args.action_noise_sigma, 0.12)
        self.assertEqual(args.action_noise_final_sigma, 0.04)
        self.assertEqual(args.action_noise_decay_steps, 200_000)
        self.assertEqual(args.policy_probe_interval, 10)
        self.assertFalse(args.safe_actor_improvement)
        self.assertEqual(args.safe_actor_max_action_delta, 0.01)
        self.assertEqual(args.safe_actor_gradient_norm, 1.0)
        self.assertEqual(args.safe_actor_block_updates, 100)
        self.assertEqual(args.safe_actor_probe_repeats, 5)
        self.assertEqual(args.safe_actor_probe_base_seed, 20260820)
        self.assertEqual(args.safe_actor_probe_noise_std, 0.006)
        self.assertEqual(args.safe_actor_min_improvement_m, 5.0)
        self.assertEqual(args.safe_actor_max_paired_regression_m, 100.0)
        self.assertEqual(args.safe_actor_anchor_batch_size, 256)
        self.assertFalse(args.allow_non_champion_source)
        self.assertEqual(args.warmup_steer_std, 0.18)
        self.assertFalse(args.save_best_replay_buffer)

    def test_continuation_defaults_are_conservative_and_policy_controlled(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "verified.zip"
            checkpoint.write_bytes(b"checkpoint")
            metadata_path_for_policy(checkpoint).write_text(
                json.dumps(
                    {
                        "model_family": AGENT6_MODEL_FAMILY,
                        "observation_version": AGENT6_OBSERVATION_VERSION,
                        "action_version": AGENT6_ACTION_VERSION,
                        "feature_names": FEATURE_NAMES,
                        "action_shape": [3],
                        "track": "g-track-3",
                        "best_evaluation": {
                            "deterministic": True,
                            "repeats": 3,
                            "track": "g-track-3",
                            "median_progress_m": 1859.65,
                        },
                    }
                ),
                encoding="utf-8",
            )
            guarded_argv = [
                str(SCRIPT_PATH),
                "--resume-from",
                str(checkpoint),
                "--no-safe-actor-improvement",
                "--total-timesteps",
                "100000",
            ]
            with mock.patch.object(sys, "argv", guarded_argv):
                with self.assertRaises(SystemExit):
                    train_td3_agent.parse_args()
            argv = [
                str(SCRIPT_PATH),
                "--resume-from",
                str(checkpoint),
                "--allow-non-champion-source",
                "--total-timesteps",
                "100000",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = train_td3_agent.parse_args()

        self.assertTrue(args.continuation)
        self.assertEqual(args.start_stage, "sector_progression")
        self.assertEqual(args.replay_refill_steps, 20_000)
        self.assertEqual(args.critic_warmup_steps, 20_000)
        self.assertEqual(args.actor_unfreeze_steps, 40_000)
        self.assertEqual(args.buffer_size, 300_000)
        self.assertEqual(args.learning_rate, 3e-5)
        self.assertEqual(args.learning_rate_final, 1e-5)
        self.assertEqual(args.actor_learning_rate, 1e-5)
        self.assertEqual(args.actor_learning_rate_final, 3e-6)
        self.assertEqual(args.action_noise_sigma, 0.02)
        self.assertEqual(args.action_noise_final_sigma, 0.006)
        self.assertTrue(args.save_best_replay_buffer)
        self.assertTrue(args.safe_actor_improvement)
        self.assertEqual(args.safe_actor_max_action_delta, 0.01)
        self.assertEqual(args.safe_actor_gradient_norm, 1.0)
        self.assertEqual(args.safe_actor_block_updates, 100)
        self.assertEqual(args.safe_actor_probe_repeats, 5)
        self.assertEqual(args.safe_actor_probe_base_seed, 20260820)
        self.assertEqual(args.safe_actor_probe_noise_std, 0.006)
        self.assertEqual(args.safe_actor_min_improvement_m, 5.0)
        self.assertEqual(args.safe_actor_max_paired_regression_m, 100.0)
        self.assertTrue(args.allow_non_champion_source)
        self.assertEqual(train_td3_agent.policy_action_start_timestep(args), 0)
        self.assertEqual(
            train_td3_agent.gradient_update_start_timestep(args),
            20_000,
        )
        self.assertEqual(train_td3_agent.actor_update_start_timestep(args), 40_000)
        self.assertEqual(
            train_td3_agent.actor_full_unfreeze_timestep(args),
            80_000,
        )

    def test_continuation_model_loads_verified_weights_with_fresh_replay(self):
        class FakeTD3:
            load_call = None

            @classmethod
            def load(cls, path, **kwargs):
                cls.load_call = (path, kwargs)
                return types.SimpleNamespace(
                    num_timesteps=35_226,
                    policy=types.SimpleNamespace(
                        action_space="dynamic_action_space",
                        observation_space="dynamic_observation_space",
                    ),
                    replay_buffer=types.SimpleNamespace(
                        action_space="dynamic_action_space",
                        observation_space="dynamic_observation_space",
                    ),
                )

        args = types.SimpleNamespace(
            continuation=True,
            resume_from=Path("verified.zip"),
            verbose_training=False,
            buffer_size=300_000,
            learning_starts=20_000,
            replay_refill_steps=10_000,
            batch_size=256,
            gamma=0.995,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            seed=42,
            device="cpu",
        )
        schedule = lambda _: 1e-4
        env = types.SimpleNamespace(
            action_space="standard_action_space",
            observation_space="standard_observation_space",
        )
        model = train_td3_agent.create_td3_model(
            FakeTD3,
            args=args,
            env=env,
            tensorboard_log=None,
            action_noise="noise",
            learning_rate_schedule=schedule,
        )

        self.assertEqual(model.num_timesteps, 35_226)
        path, kwargs = FakeTD3.load_call
        self.assertEqual(path, "verified.zip")
        self.assertEqual(kwargs["learning_starts"], 0)
        self.assertEqual(kwargs["buffer_size"], 300_000)
        self.assertIs(kwargs["learning_rate"], schedule)
        self.assertEqual(model.action_space, "standard_action_space")
        self.assertEqual(model.policy.action_space, "standard_action_space")
        self.assertEqual(model.replay_buffer.action_space, "standard_action_space")

    def test_safe_actor_interleaves_matched_pairs_and_rotates_seed_suites(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import TD3

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
                dtype=np.float32,
            )
            action_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(3,),
                dtype=np.float32,
            )

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

        SafeTD3 = train_td3_agent.make_phased_continuation_td3_class(TD3)
        model = SafeTD3(
            "MlpPolicy",
            TinyEnv(),
            policy_kwargs={"net_arch": [8, 8]},
            learning_starts=0,
            buffer_size=32,
            batch_size=4,
            device="cpu",
        )
        model.configure_continuation_updates(
            actor_learning_rate_schedule=lambda _: 1e-4,
            actor_updates_start_at=0,
            safe_actor_improvement=True,
            safe_actor_probe_repeats=2,
            safe_actor_min_improvement_m=5.0,
            safe_actor_max_paired_regression_m=100.0,
            safe_actor_probe_base_seed=1234,
        )
        with torch.no_grad():
            next(model.actor.parameters()).add_(0.01)
        model._agent6_safe_actor_candidate_id = 1
        model._agent6_safe_actor_candidate_updates = 1
        model._agent6_safe_actor_waiting_for_probe = True

        reference_one = model.safe_actor_probe_context()
        first_suite = tuple(model.safe_actor_status()["challenge_seeds"])
        self.assertEqual(reference_one["mode"], "accepted_reference")
        self.assertEqual(reference_one["seed"], first_suite[0])
        self.assertEqual(
            model.record_safe_actor_probe({"distance_m": 100.0})["decision"],
            "reference_recorded",
        )
        candidate_one = model.safe_actor_probe_context()
        self.assertEqual(candidate_one["mode"], "candidate")
        self.assertEqual(candidate_one["seed"], reference_one["seed"])
        self.assertEqual(
            model.record_safe_actor_probe({"distance_m": 110.0})["decision"],
            "pair_complete",
        )

        reference_two = model.safe_actor_probe_context()
        self.assertEqual(reference_two["mode"], "accepted_reference")
        self.assertNotEqual(reference_two["seed"], reference_one["seed"])
        model.record_safe_actor_probe({"distance_m": 90.0})
        candidate_two = model.safe_actor_probe_context()
        self.assertEqual(candidate_two["seed"], reference_two["seed"])
        accepted = model.record_safe_actor_probe({"distance_m": 95.0})

        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["paired_deltas_m"], [10.0, 5.0])
        self.assertFalse(model.safe_actor_probe_required())
        self.assertEqual(model.safe_actor_status()["accepted_blocks"], 1)

        with torch.no_grad():
            next(model.actor.parameters()).add_(0.01)
        model._agent6_safe_actor_candidate_id = 2
        model._agent6_safe_actor_candidate_updates = 1
        model._agent6_safe_actor_waiting_for_probe = True
        model.safe_actor_probe_context()
        second_suite = tuple(model.safe_actor_status()["challenge_seeds"])

        self.assertTrue(set(first_suite).isdisjoint(second_suite))
        model.finalize_safe_actor_candidate()

    def test_safe_actor_model_rolls_back_degraded_candidate_exactly(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import TD3

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

        SafeTD3 = train_td3_agent.make_phased_continuation_td3_class(TD3)
        model = SafeTD3(
            "MlpPolicy",
            TinyEnv(),
            policy_kwargs={"net_arch": [8, 8]},
            learning_starts=0,
            buffer_size=32,
            batch_size=4,
            device="cpu",
        )
        model.configure_continuation_updates(
            actor_learning_rate_schedule=lambda _: 1e-4,
            actor_updates_start_at=0,
            safe_actor_improvement=True,
            safe_actor_block_updates=1,
            safe_actor_probe_repeats=2,
            safe_actor_min_improvement_m=5.0,
            safe_actor_max_paired_regression_m=100.0,
            safe_actor_probe_base_seed=4321,
        )
        accepted_actor = {
            name: value.detach().clone()
            for name, value in model.actor.state_dict().items()
        }
        accepted_target = {
            name: value.detach().clone()
            for name, value in model.actor_target.state_dict().items()
        }
        with torch.no_grad():
            next(model.actor.parameters()).add_(0.25)
            next(model.actor_target.parameters()).sub_(0.25)
        model._agent6_safe_actor_candidate_id = 1
        model._agent6_safe_actor_candidate_updates = 1
        model._agent6_safe_actor_waiting_for_probe = True
        candidate_actor = {
            name: value.detach().clone()
            for name, value in model.actor.state_dict().items()
        }
        reference_context = model.safe_actor_probe_context()
        self.assertEqual(reference_context["mode"], "accepted_reference")
        self.assertEqual(
            model.record_safe_actor_probe({"distance_m": 100.0})["decision"],
            "reference_recorded",
        )
        candidate_context = model.safe_actor_probe_context()
        self.assertEqual(candidate_context["mode"], "candidate")
        self.assertEqual(candidate_context["seed"], reference_context["seed"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate_checkpoint.zip"
            model.save(checkpoint)
            saved = TD3.load(checkpoint, device="cpu")
        for name, value in saved.actor.state_dict().items():
            self.assertTrue(torch.equal(value, accepted_actor[name]))
        for name, value in model.actor.state_dict().items():
            self.assertTrue(torch.equal(value, candidate_actor[name]))

        pending = model.record_safe_actor_probe({"distance_m": 102.0})
        model.safe_actor_probe_context()
        model.record_safe_actor_probe({"distance_m": 100.0})
        result = model.record_safe_actor_probe({"distance_m": 80.0})

        self.assertEqual(pending["decision"], "pair_complete")
        self.assertEqual(result["decision"], "rolled_back")
        self.assertEqual(result["paired_deltas_m"], [2.0, -20.0])
        self.assertFalse(model.safe_actor_probe_required())
        for name, value in model.actor.state_dict().items():
            self.assertTrue(torch.equal(value, accepted_actor[name]))
        for name, value in model.actor_target.state_dict().items():
            self.assertTrue(torch.equal(value, accepted_target[name]))

        with torch.no_grad():
            next(model.actor.parameters()).add_(0.01)
            next(model.actor_target.parameters()).add_(0.01)
        model._agent6_safe_actor_candidate_id = 2
        model._agent6_safe_actor_candidate_updates = 1
        model._agent6_safe_actor_waiting_for_probe = True
        model.safe_actor_probe_context()
        model.record_safe_actor_probe({"distance_m": 100.0})
        model.record_safe_actor_probe({"distance_m": 110.0})
        model.safe_actor_probe_context()
        model.record_safe_actor_probe({"distance_m": 100.0})
        accepted = model.record_safe_actor_probe({"distance_m": 112.0})
        accepted_actor = {
            name: value.detach().clone()
            for name, value in model.actor.state_dict().items()
        }

        self.assertEqual(accepted["decision"], "accepted")
        self.assertEqual(accepted["median_distance_m"], 111.0)
        with torch.no_grad():
            next(model.actor.parameters()).add_(0.5)
        model._agent6_safe_actor_candidate_id = 3
        model._agent6_safe_actor_candidate_updates = 1
        finalized = model.finalize_safe_actor_candidate()

        self.assertEqual(finalized["decision"], "rolled_back_incomplete")
        for name, value in model.actor.state_dict().items():
            self.assertTrue(torch.equal(value, accepted_actor[name]))
        status = model.safe_actor_status()
        self.assertEqual(status["accepted_blocks"], 1)
        self.assertEqual(status["rolled_back_blocks"], 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "safe_actor.zip"
            model.save(checkpoint)
            loaded = TD3.load(checkpoint, device="cpu")
        for name, value in loaded.actor.state_dict().items():
            self.assertTrue(torch.equal(value, accepted_actor[name]))

    def test_safe_actor_projection_enforces_action_space_limit(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import TD3

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

        SafeTD3 = train_td3_agent.make_phased_continuation_td3_class(TD3)
        model = SafeTD3(
            "MlpPolicy",
            TinyEnv(),
            policy_kwargs={"net_arch": [8, 8]},
            learning_starts=0,
            buffer_size=32,
            batch_size=4,
            device="cpu",
        )
        model.configure_continuation_updates(
            actor_learning_rate_schedule=lambda _: 1e-4,
            actor_updates_start_at=0,
            safe_actor_improvement=True,
            safe_actor_max_action_delta=0.01,
        )
        observations = torch.ones((8, 4), dtype=torch.float32)
        with torch.no_grad():
            for parameter in model.actor.parameters():
                parameter.add_(0.5)

        projected_delta = model._project_safe_actor(observations)

        self.assertLessEqual(projected_delta, 0.010001)

    def test_safe_actor_training_stops_at_candidate_block_boundary(self):
        import gymnasium as gym
        from stable_baselines3 import TD3

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(4, dtype=np.float32), 1.0, False, False, {}

        SafeTD3 = train_td3_agent.make_phased_continuation_td3_class(TD3)
        model = SafeTD3(
            "MlpPolicy",
            TinyEnv(),
            policy_kwargs={"net_arch": [8, 8]},
            learning_starts=0,
            buffer_size=64,
            batch_size=4,
            train_freq=1,
            gradient_steps=1,
            policy_delay=1,
            device="cpu",
        )
        model.configure_continuation_updates(
            actor_learning_rate_schedule=lambda _: 1e-4,
            actor_updates_start_at=0,
            safe_actor_improvement=True,
            safe_actor_block_updates=2,
            safe_actor_probe_repeats=3,
        )

        model.learn(total_timesteps=10)

        status = model.safe_actor_status()
        self.assertEqual(status["candidate_updates"], 2)
        self.assertTrue(status["waiting_for_probe"])
        self.assertEqual(status["accepted_blocks"], 0)
        self.assertEqual(status["rolled_back_blocks"], 0)

    def test_continuation_stage_requires_verified_prerequisite_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "verified.zip"
            checkpoint.write_bytes(b"checkpoint")
            metadata_path_for_policy(checkpoint).write_text(
                json.dumps(
                    {
                        "model_family": AGENT6_MODEL_FAMILY,
                        "observation_version": AGENT6_OBSERVATION_VERSION,
                        "action_version": AGENT6_ACTION_VERSION,
                        "feature_names": FEATURE_NAMES,
                        "action_shape": [3],
                        "best_evaluation": {
                            "deterministic": True,
                            "repeats": 3,
                            "track": "g-track-3",
                            "median_progress_m": 500.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            errors = train_td3_agent.continuation_source_errors(
                checkpoint,
                track="g-track-3",
                start_stage="sector_progression",
            )

        self.assertTrue(any("requires 520.0m" in error for error in errors))

    def test_replay_snapshot_failure_is_reported_without_partial_temp_file(self):
        model = mock.Mock()
        model.save_replay_buffer.side_effect = RuntimeError("disk unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.replay_buffer.pkl"

            error = train_td3_agent.save_replay_buffer_atomically(model, path)

            self.assertIn("RuntimeError: disk unavailable", error)
            self.assertFalse(path.is_file())
            self.assertFalse((Path(directory) / "best.replay_buffer.pkl.tmp").is_file())

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
                save_best_replay_buffer=True,
            )
            callback.model.save_replay_buffer.side_effect = (
                lambda path: Path(path).write_bytes(b"replay")
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

            callback.num_timesteps = 20_500
            callback.locals = {
                "infos": [
                    {
                        "episode_summary": {
                            "policy_controlled": True,
                            "deterministic_probe": True,
                            "safe_actor_decision": "pending",
                            "curriculum_eligible": False,
                            "distance_m": 5_000.0,
                            "reward": 10_000.0,
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
            distance_replay_path = train_td3_agent.replay_buffer_path_for_policy(
                best_distance_path
            )
            reward_replay_path = train_td3_agent.replay_buffer_path_for_policy(
                best_reward_path
            )
            self.assertEqual(callback.model.save_replay_buffer.call_count, 2)
            callback.model.save_replay_buffer.assert_has_calls(
                [
                    mock.call(
                        str(
                            distance_replay_path.with_name(
                                f"{distance_replay_path.name}.tmp"
                            )
                        )
                    ),
                    mock.call(
                        str(
                            reward_replay_path.with_name(
                                f"{reward_replay_path.name}.tmp"
                            )
                        )
                    ),
                ]
            )
            saved_metadata = json.loads(
                metadata_path_for_policy(best_distance_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved_metadata["best_distance_episode"]["replay_buffer_path"],
                str(distance_replay_path),
            )
            reward_metadata = json.loads(
                metadata_path_for_policy(best_reward_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                reward_metadata["best_reward_episode"]["replay_buffer_path"],
                str(reward_replay_path),
            )

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
