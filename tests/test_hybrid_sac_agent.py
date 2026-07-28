import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agents.hybrid_sac_agent import (  # noqa: E402
    DEFAULT_RACING_LINE_PATH,
    FEATURE_NAMES,
    MAX_ACCEL_RESIDUAL,
    MAX_BRAKE_RESIDUAL,
    MAX_STEER_RESIDUAL,
    HybridSacAgent,
    apply_hybrid_sac_residual,
    build_hybrid_sac_observation,
    calculate_sac_authority,
    decode_policy_residual,
    find_policy_contract_mismatches,
)
from racing_line import RacingLine  # noqa: E402
from train_hybrid_sac_agent import calculate_hybrid_sac_reward  # noqa: E402


def make_telemetry(**overrides):
    telemetry = {
        "speedX": 100.0,
        "speedY": 0.0,
        "speedZ": 0.0,
        "angle": 0.0,
        "trackPos": 0.0,
        "damage": 0.0,
        "rpm": 5000.0,
        "track": [200.0] * 19,
        "wheelSpinVel": [10.0] * 4,
        "distFromStart": 500.0,
        "distRaced": 500.0,
        "curLapTime": 10.0,
        "lastLapTime": 0.0,
    }
    telemetry.update(overrides)
    return telemetry


class DummySacPolicy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.seen_observation = None
        self.seen_deterministic = None

    def predict(self, observation, deterministic=True):
        self.seen_observation = np.asarray(observation)
        self.seen_deterministic = deterministic
        return self.action, None


class HybridSacObservationTests(unittest.TestCase):
    def test_observation_contract_is_normalized_and_stable(self):
        racing_line = RacingLine.load(DEFAULT_RACING_LINE_PATH)
        base_action = {"steer": 0.25, "accel": 0.7, "brake": 0.0, "gear": 4}

        observation = build_hybrid_sac_observation(
            make_telemetry(speedX=123.0, speedY=3.0),
            base_action,
            racing_line,
        )

        self.assertEqual(len(observation), len(FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in observation))
        self.assertTrue(all(-1.000001 <= value <= 1.000001 for value in observation))
        self.assertAlmostEqual(observation[25], base_action["steer"])
        self.assertAlmostEqual(observation[26], base_action["accel"])

    def test_policy_contract_mismatch_detects_wrong_metadata(self):
        mismatches = find_policy_contract_mismatches(
            {
                "model_family": "old_model",
                "observation_version": "old_observation",
                "action_version": "old_action",
                "feature_names": [],
                "action_shape": [2],
            }
        )

        self.assertIn("model_family", mismatches)
        self.assertIn("observation_version", mismatches)
        self.assertIn("action_shape", mismatches)


class HybridSacResidualTests(unittest.TestCase):
    def test_decode_policy_residual_scales_and_clips_actions(self):
        residual = decode_policy_residual([2.0, -2.0, float("nan")], residual_scale=0.5)

        self.assertEqual(
            residual,
            [
                MAX_STEER_RESIDUAL * 0.5,
                -MAX_ACCEL_RESIDUAL * 0.5,
                0.0,
            ],
        )

    def test_apply_residual_clamps_final_controls(self):
        action = apply_hybrid_sac_residual(
            {"steer": 0.98, "accel": 0.9, "brake": 0.0, "gear": 5},
            [1.0, 1.0, 1.0],
            make_telemetry(),
            residual_scale=1.0,
        )

        self.assertEqual(action["steer"], 1.0)
        self.assertEqual(action["accel"], 0.0)
        self.assertAlmostEqual(action["brake"], MAX_BRAKE_RESIDUAL)
        self.assertAlmostEqual(action["agent5_steer_residual"], MAX_STEER_RESIDUAL)

    def test_authority_gate_removes_edge_correction_before_shield(self):
        action = apply_hybrid_sac_residual(
            {"steer": 0.0, "accel": 0.2, "brake": 0.0, "gear": 4},
            [1.0, 1.0, -1.0],
            make_telemetry(trackPos=0.95),
            residual_scale=1.0,
        )

        self.assertEqual(action["agent5_sac_authority"], 0.0)
        self.assertTrue(action["agent5_authority_gate_active"])
        self.assertFalse(action["agent5_safety_shield_active"])
        self.assertEqual(action["agent5_steer_residual"], 0.0)
        self.assertEqual(action["accel"], 0.2)
        self.assertEqual(action["brake"], 0.0)

    def test_safety_shield_still_catches_forced_outward_edge_correction(self):
        action = apply_hybrid_sac_residual(
            {"steer": 0.0, "accel": 0.2, "brake": 0.0, "gear": 4},
            [1.0, 1.0, -1.0],
            make_telemetry(trackPos=0.95),
            residual_scale=1.0,
            sac_authority=1.0,
        )

        self.assertTrue(action["agent5_safety_shield_active"])
        self.assertEqual(action["agent5_steer_residual"], 0.0)
        self.assertEqual(action["accel"], 0.2)
        self.assertEqual(action["brake"], 0.0)

    def test_sac_authority_is_high_only_in_stable_windows(self):
        self.assertAlmostEqual(calculate_sac_authority(make_telemetry()), 1.0)
        self.assertEqual(
            calculate_sac_authority(make_telemetry(distRaced=20.0)),
            0.0,
        )
        self.assertEqual(
            calculate_sac_authority(make_telemetry(trackPos=0.95)),
            0.0,
        )


class HybridSacRuntimeTests(unittest.TestCase):
    def test_agent_runs_policy_residual_and_writes_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = DummySacPolicy([0.5, -0.5, 0.0])
            residual_telemetry_path = Path(directory) / "residual.csv"
            agent = HybridSacAgent(
                policy=policy,
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=residual_telemetry_path,
                residual_scale=1.0,
            )
            action = agent.act(None, make_telemetry())
            agent.close()

            with residual_telemetry_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertTrue(agent.policy_loaded)
        self.assertTrue(action["agent5_policy_loaded"])
        self.assertEqual(policy.seen_observation.shape, (len(FEATURE_NAMES),))
        self.assertTrue(policy.seen_deterministic)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(action["agent5_steer_residual"], MAX_STEER_RESIDUAL * 0.5)

    def test_missing_policy_falls_back_to_base_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = HybridSacAgent(
                policy_path=Path(directory) / "missing_model.zip",
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=Path(directory) / "residual.csv",
            )
            action = agent.act(None, make_telemetry())
            agent.close()

        self.assertFalse(agent.policy_loaded)
        self.assertFalse(action["agent5_policy_loaded"])
        self.assertEqual(action["agent5_raw_steer_residual"], 0.0)
        self.assertEqual(action["agent5_raw_accel_residual"], 0.0)
        self.assertEqual(action["agent5_raw_brake_residual"], 0.0)


class HybridSacRewardTests(unittest.TestCase):
    def test_reward_prefers_forward_progress(self):
        base_action = apply_hybrid_sac_residual(
            {"steer": 0.0, "accel": 0.6, "brake": 0.0, "gear": 4},
            [0.0, 0.0, 0.0],
            make_telemetry(distRaced=103.0, distFromStart=103.0, curLapTime=10.1),
            residual_scale=1.0,
        )

        reward = calculate_hybrid_sac_reward(
            make_telemetry(distRaced=103.0, distFromStart=103.0, curLapTime=10.1),
            base_action,
            previous_telemetry=make_telemetry(
                distRaced=100.0,
                distFromStart=100.0,
                curLapTime=10.0,
            ),
            racing_line=RacingLine.load(DEFAULT_RACING_LINE_PATH),
            episode_distance_m=103.0,
        )

        self.assertGreater(reward, 0.0)

    def test_off_track_failure_is_heavily_penalized(self):
        telemetry = make_telemetry(track=[-1.0] * 19, trackPos=1.10)
        action = apply_hybrid_sac_residual(
            {"steer": 0.0, "accel": 0.0, "brake": 0.2, "gear": 3},
            [0.0, 0.0, 0.0],
            telemetry,
            residual_scale=1.0,
        )

        reward = calculate_hybrid_sac_reward(
            telemetry,
            action,
            previous_telemetry=make_telemetry(),
            racing_line=RacingLine.load(DEFAULT_RACING_LINE_PATH),
            episode_distance_m=50.0,
        )

        self.assertLess(reward, -500.0)


if __name__ == "__main__":
    unittest.main()
