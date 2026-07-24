import sys
import tempfile
import unittest
import math
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agents.hybrid_ppo_agent as hybrid_ppo_agent  # noqa: E402
from agents.hybrid_ppo_agent import (  # noqa: E402
    AGENT4_ACTION_VERSION,
    AGENT4_MODEL_FAMILY,
    AGENT4_OBSERVATION_VERSION,
    DEFAULT_RESIDUAL_SCALE,
    FEATURE_NAMES,
    HybridPpoAgent,
    MAX_ACCEL_RESIDUAL,
    MAX_BRAKE_RESIDUAL,
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

    def test_safety_shield_blocks_unsafe_residuals_away_from_track_edge(self):
        base_action = {
            "steer": 0.0,
            "accel": 0.40,
            "brake": 0.10,
            "gear": 3,
            "terminate": False,
        }

        action = apply_hybrid_residual(
            base_action,
            [1.0, 1.0, -1.0],
            make_telemetry(angle=0.45, trackPos=0.0),
        )

        self.assertAlmostEqual(
            action["agent4_steer_residual"],
            MAX_STEER_RESIDUAL * 0.35,
        )
        self.assertEqual(action["agent4_accel_residual"], 0.0)
        self.assertEqual(action["agent4_brake_residual"], 0.0)
        self.assertEqual(action["brake"], 0.10)
        self.assertGreater(action["agent4_unsafe_residual_pressure"], 0.0)
        self.assertTrue(action["agent4_safety_shield_active"])

    def test_safety_shield_allows_inward_edge_correction_without_intervention(self):
        base_action = {
            "steer": 0.0,
            "accel": 0.50,
            "brake": 0.0,
            "gear": 3,
            "terminate": False,
        }

        action = apply_hybrid_residual(
            base_action,
            [-1.0, 1.0, -1.0],
            make_telemetry(trackPos=0.95),
        )

        self.assertAlmostEqual(action["agent4_steer_residual"], -MAX_STEER_RESIDUAL)
        self.assertAlmostEqual(action["agent4_accel_residual"], MAX_ACCEL_RESIDUAL)
        self.assertLess(action["agent4_brake_residual"], 0.0)
        self.assertEqual(action["brake"], 0.0)
        self.assertGreater(action["agent4_risk_pressure"], 0.0)
        self.assertLess(action["agent4_risk_pressure"], 1.0)
        self.assertFalse(action["agent4_safety_shield_active"])

    def test_safety_shield_allows_low_speed_recovery_drive(self):
        base_action = {
            "steer": 0.0,
            "accel": 0.0,
            "brake": 0.20,
            "gear": 1,
            "terminate": False,
        }

        action = apply_hybrid_residual(
            base_action,
            [-1.0, 1.0, -1.0],
            make_telemetry(
                speedX=0.4,
                angle=0.05,
                trackPos=0.90,
                track=[4.0] * 19,
            ),
        )

        self.assertAlmostEqual(action["agent4_steer_residual"], -MAX_STEER_RESIDUAL)
        self.assertAlmostEqual(action["agent4_accel_residual"], MAX_ACCEL_RESIDUAL)
        self.assertAlmostEqual(action["agent4_brake_residual"], -MAX_BRAKE_RESIDUAL)
        self.assertGreater(action["accel"], 0.0)
        self.assertEqual(action["brake"], 0.0)
        self.assertTrue(action["agent4_low_speed_recovery_active"])
        self.assertFalse(action["agent4_safety_shield_active"])

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
                residual_scale=1.0,
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=Path(directory) / "residual.csv",
            )
            try:
                action = agent.act(None, make_telemetry())
            finally:
                agent.close()

        self.assertTrue(action["agent4_policy_loaded"])
        self.assertAlmostEqual(action["agent4_steer_residual"], MAX_STEER_RESIDUAL / 2)

    def test_agent_uses_residual_scale_from_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "agent4_hybrid_ppo.zip"
            policy_path.with_suffix(".metadata.json").write_text(
                '{"residual_scale": 0.2}',
                encoding="utf-8",
            )
            agent = HybridPpoAgent(
                policy_path=policy_path,
                base_telemetry_path=Path(directory) / "base.csv",
                residual_telemetry_path=Path(directory) / "residual.csv",
            )
            try:
                config = agent.config
            finally:
                agent.close()

        self.assertTrue(config["policy_metadata_loaded"])
        self.assertAlmostEqual(config["residual_scale"], 0.2)
        self.assertAlmostEqual(config["residual_limits"]["steer"], MAX_STEER_RESIDUAL * 0.2)

    def test_agent_uses_safe_default_residual_scale_without_metadata(self):
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

        self.assertAlmostEqual(config["residual_scale"], DEFAULT_RESIDUAL_SCALE)

    def test_default_policy_path_prefers_curriculum_winners_then_final_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_model_path = root / "agent4_hybrid_ppo.zip"
            best_progress_pace_model_path = (
                root / "agent4_hybrid_ppo_best_progress_pace.zip"
            )
            best_distance_model_path = root / "agent4_hybrid_ppo_best_distance.zip"
            best_clean_model_path = root / "agent4_hybrid_ppo_best.zip"

            patches = [
                patch.object(hybrid_ppo_agent, "DEFAULT_MODEL_PATH", final_model_path),
                patch.object(
                    hybrid_ppo_agent,
                    "DEFAULT_BEST_PROGRESS_PACE_MODEL_PATH",
                    best_progress_pace_model_path,
                ),
                patch.object(
                    hybrid_ppo_agent,
                    "DEFAULT_BEST_DISTANCE_MODEL_PATH",
                    best_distance_model_path,
                ),
                patch.object(
                    hybrid_ppo_agent,
                    "DEFAULT_BEST_MODEL_PATH",
                    best_clean_model_path,
                ),
            ]
            for patcher in patches:
                patcher.start()
                self.addCleanup(patcher.stop)

            final_model_path.write_text("final", encoding="utf-8")
            self.assertEqual(
                hybrid_ppo_agent.resolve_default_policy_path(),
                final_model_path,
            )

            best_distance_model_path.write_text("distance", encoding="utf-8")
            self.assertEqual(
                hybrid_ppo_agent.resolve_default_policy_path(),
                best_distance_model_path,
            )

            best_progress_pace_model_path.write_text(
                "progress pace",
                encoding="utf-8",
            )
            self.assertEqual(
                hybrid_ppo_agent.resolve_default_policy_path(),
                best_progress_pace_model_path,
            )

            best_clean_model_path.write_text("clean", encoding="utf-8")
            self.assertEqual(
                hybrid_ppo_agent.resolve_default_policy_path(),
                best_clean_model_path,
            )

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
