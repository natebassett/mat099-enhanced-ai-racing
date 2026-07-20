import sys
import tempfile
import unittest
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.hybrid_ppo_agent import (  # noqa: E402
    AGENT4_ACTION_VERSION,
    AGENT4_MODEL_FAMILY,
    AGENT4_OBSERVATION_VERSION,
    FEATURE_NAMES,
    HybridPpoAgent,
    MAX_ACCEL_RESIDUAL,
    MAX_STEER_RESIDUAL,
    apply_hybrid_residual,
    build_hybrid_ppo_observation,
    decode_policy_residual,
    finite_float,
)


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
        "distRaced": 0.0,
        "curLapTime": 10.0,
        "lastLapTime": 0.0,
    }
    telemetry.update(overrides)
    return telemetry


class FakePolicy:
    def __init__(self, residual):
        self.residual = residual

    def predict(self, _features, deterministic=True):
        return self.residual, None


class HybridPpoAgentTests(unittest.TestCase):
    def test_observation_matches_feature_contract(self):
        base_action = {"steer": 0.2, "accel": 0.7, "brake": 0.0, "gear": 4}

        observation = build_hybrid_ppo_observation(
            make_telemetry(speedY=4.0, trackPos=0.2),
            base_action,
            track_length=1000.0,
        )

        self.assertEqual(len(observation), len(FEATURE_NAMES))
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in observation))

    def test_observation_includes_smooth_lap_phase_features(self):
        base_action = {"steer": 0.0, "accel": 0.5, "brake": 0.0, "gear": 3}

        observation = build_hybrid_ppo_observation(
            make_telemetry(distFromStart=250.0),
            base_action,
            track_length=1000.0,
        )
        by_name = dict(zip(FEATURE_NAMES, observation))

        self.assertAlmostEqual(by_name["distance_phase"], 0.25)
        self.assertAlmostEqual(by_name["distance_phase_sin"], 1.0)
        self.assertAlmostEqual(by_name["distance_phase_cos"], 0.0, places=7)
        self.assertAlmostEqual(by_name["base_pedal_balance"], 0.5)

    def test_policy_residual_is_decoded_into_small_control_changes(self):
        steer_delta, accel_delta, brake_delta = decode_policy_residual([1.0, -1.0])

        self.assertEqual(steer_delta, MAX_STEER_RESIDUAL)
        self.assertEqual(accel_delta, -MAX_ACCEL_RESIDUAL)
        self.assertEqual(brake_delta, 0.0)

    def test_bad_policy_residual_values_are_sanitised(self):
        steer_delta, accel_delta, brake_delta = decode_policy_residual(
            [math.nan, math.inf, -math.inf]
        )

        self.assertEqual(steer_delta, 0.0)
        self.assertEqual(accel_delta, 0.0)
        self.assertEqual(brake_delta, 0.0)

    def test_bad_telemetry_values_still_produce_finite_observation(self):
        base_action = {
            "steer": math.nan,
            "accel": math.inf,
            "brake": -math.inf,
            "gear": 4,
        }
        observation = build_hybrid_ppo_observation(
            make_telemetry(
                speedX=math.nan,
                speedY=math.inf,
                angle=-math.inf,
                wheelSpinVel=[math.nan],
            ),
            base_action,
            track_length=1000.0,
        )

        self.assertTrue(all(math.isfinite(value) for value in observation))

    def test_safe_residual_can_adjust_base_action(self):
        base_action = {
            "steer": 0.0,
            "accel": 0.50,
            "brake": 0.0,
            "gear": 3,
            "terminate": False,
        }

        action = apply_hybrid_residual(
            base_action,
            [1.0, 1.0, 0.0],
            make_telemetry(),
        )

        self.assertAlmostEqual(action["steer"], MAX_STEER_RESIDUAL)
        self.assertAlmostEqual(action["accel"], 0.50 + MAX_ACCEL_RESIDUAL)
        self.assertEqual(action["brake"], 0.0)
        self.assertFalse(action["agent4_safety_shield_active"])

    def test_safety_shield_blocks_outward_steer_and_extra_throttle(self):
        base_action = {
            "steer": 0.0,
            "accel": 0.40,
            "brake": 0.20,
            "gear": 3,
            "terminate": False,
        }

        action = apply_hybrid_residual(
            base_action,
            [1.0, 1.0, -1.0],
            make_telemetry(trackPos=0.90),
        )

        self.assertEqual(action["agent4_steer_residual"], 0.0)
        self.assertEqual(action["agent4_accel_residual"], 0.0)
        self.assertEqual(action["agent4_brake_residual"], 0.0)
        self.assertEqual(action["brake"], 0.20)
        self.assertEqual(action["accel"], 0.0)
        self.assertTrue(action["agent4_safety_shield_active"])

    def test_untrained_agent_falls_back_to_zero_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = HybridPpoAgent(
                policy_path=Path(directory) / "missing_model.zip",
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=Path(directory) / "residual.csv",
            )
            try:
                action = agent.act(None, make_telemetry())
            finally:
                agent.close()

        self.assertFalse(action["agent4_policy_loaded"])
        self.assertEqual(action["agent4_steer_residual"], 0.0)
        self.assertEqual(action["agent4_accel_residual"], 0.0)
        self.assertEqual(action["agent4_brake_residual"], 0.0)
        self.assertLessEqual(abs(action["steer"]), 1.0)
        self.assertGreaterEqual(action["accel"], 0.0)
        self.assertLessEqual(action["accel"], 1.0)
        self.assertGreaterEqual(action["brake"], 0.0)
        self.assertLessEqual(action["brake"], 1.0)

    def test_injected_policy_marks_agent_as_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = HybridPpoAgent(
                policy=FakePolicy([0.5, 0.0, 0.0]),
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=Path(directory) / "residual.csv",
            )
            try:
                action = agent.act(None, make_telemetry())
            finally:
                agent.close()

        self.assertTrue(action["agent4_policy_loaded"])
        self.assertAlmostEqual(action["agent4_steer_residual"], MAX_STEER_RESIDUAL / 2)

    def test_agent_config_exposes_resume_compatibility_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = HybridPpoAgent(
                policy_path=Path(directory) / "missing_model.zip",
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=Path(directory) / "residual.csv",
            )
            try:
                config = agent.config
            finally:
                agent.close()

        self.assertEqual(config["model_family"], AGENT4_MODEL_FAMILY)
        self.assertEqual(config["observation_version"], AGENT4_OBSERVATION_VERSION)
        self.assertEqual(config["action_version"], AGENT4_ACTION_VERSION)
        self.assertEqual(config["observation_features"], FEATURE_NAMES)

    def test_finite_float_replaces_invalid_numbers(self):
        self.assertEqual(finite_float(math.nan, default=7.0), 7.0)
        self.assertEqual(finite_float("not-a-number", default=7.0), 7.0)


if __name__ == "__main__":
    unittest.main()
