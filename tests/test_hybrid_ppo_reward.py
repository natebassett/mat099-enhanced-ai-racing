import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

spec = importlib.util.spec_from_file_location(
    "train_hybrid_ppo_agent",
    PROJECT_ROOT / "scripts" / "train_hybrid_ppo_agent.py",
)
train_hybrid_ppo_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_hybrid_ppo_agent)


class FakeRacingLine:
    track_length = 1000.0

    def lookup(self, _distance):
        return {
            "target_track_pos": 0.0,
            "target_speed_kmh": 144.0,
            "curvature": 0.004,
        }


class AmbitiousEdgeRacingLine:
    track_length = 1000.0

    def lookup(self, _distance):
        return {
            "target_track_pos": 0.58,
            "target_speed_kmh": 200.0,
            "curvature": 0.004,
        }


class BrakingZoneRacingLine:
    track_length = 1000.0

    def lookup(self, distance):
        return {
            "target_track_pos": 0.0,
            "target_speed_kmh": 200.0 if float(distance) < 150.0 else 120.0,
            "curvature": 0.004,
        }


def make_telemetry(**overrides):
    telemetry = {
        "speedX": 100.0,
        "speedY": 0.0,
        "angle": 0.0,
        "trackPos": 0.0,
        "damage": 0.0,
        "track": [200.0] * 19,
        "wheelSpinVel": [10.0] * 4,
        "distFromStart": 100.0,
        "distRaced": 100.0,
        "curLapTime": 1.0,
        "lastLapTime": 0.0,
    }
    telemetry.update(overrides)
    return telemetry


def make_action(**overrides):
    action = {
        "steer": 0.0,
        "accel": 0.5,
        "brake": 0.0,
        "agent4_steer_residual": 0.0,
        "agent4_accel_residual": 0.0,
        "agent4_brake_residual": 0.0,
        "agent4_safety_shield_active": False,
    }
    action.update(overrides)
    return action


class HybridPpoRewardTests(unittest.TestCase):
    def test_progress_line_summarises_training_status(self):
        state = train_hybrid_ppo_agent.TrainingProgressState()
        state.record_episode(
            {
                "episodes_seen": 3,
                "termination_reason": "off_track",
                "distance_m": 1200.0,
                "reward": -140.0,
                "laps_completed": 0,
            }
        )

        line = train_hybrid_ppo_agent.format_training_progress_line(
            50,
            100,
            10.0,
            state,
            width=10,
        )

        self.assertIn("50.0%", line)
        self.assertIn("50/100 steps", line)
        self.assertIn("ETA 00:00:10", line)
        self.assertIn("ep 3 off_track 1200m R=-140 laps=0", line)

    def test_episode_diagnostics_capture_failure_location_and_stability(self):
        telemetry = make_telemetry(
            speedX=132.0,
            speedY=7.5,
            angle=0.28,
            trackPos=1.08,
            track=[-1.0] + [200.0] * 18,
            distFromStart=540.0,
        )

        diagnostics = train_hybrid_ppo_agent.build_episode_diagnostics(
            telemetry,
            episode_distance_m=250.0,
            racing_line=FakeRacingLine(),
        )

        self.assertAlmostEqual(diagnostics["lap_completion_fraction"], 0.25)
        self.assertEqual(diagnostics["final_dist_from_start_m"], 540.0)
        self.assertEqual(diagnostics["final_track_pos"], 1.08)
        self.assertEqual(diagnostics["final_angle_rad"], 0.28)
        self.assertEqual(diagnostics["final_speed_x_kmh"], 132.0)
        self.assertEqual(diagnostics["final_speed_y_kmh"], 7.5)
        self.assertEqual(diagnostics["final_min_track_sensor_m"], -1.0)

    def test_ambitious_teacher_line_has_lower_reward_confidence(self):
        context = train_hybrid_ppo_agent.get_racing_line_context(
            make_telemetry(trackPos=0.2),
            AmbitiousEdgeRacingLine(),
        )

        self.assertLess(context["teacher_confidence"], 1.0)
        self.assertGreaterEqual(
            context["teacher_confidence"],
            train_hybrid_ppo_agent.TEACHER_CONFIDENCE_MIN,
        )
        self.assertLess(context["reward_target_speed"], context["target_speed"])
        self.assertGreater(context["teacher_lateral_accel_mps2"], 0.0)

    def test_reward_target_speed_uses_future_braking_zone(self):
        context = train_hybrid_ppo_agent.get_racing_line_context(
            make_telemetry(distFromStart=100.0),
            BrakingZoneRacingLine(),
        )

        self.assertEqual(context["target_speed"], 200.0)
        self.assertLess(context["lookahead_target_speed"], 150.0)
        self.assertLess(
            context["reward_target_speed"],
            context["lookahead_target_speed"],
        )

    def test_low_confidence_teacher_line_penalises_deviation_less(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        telemetry = make_telemetry(
            speedX=120.0,
            trackPos=0.2,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )

        normal_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            telemetry,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )
        ambitious_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            telemetry,
            make_action(),
            previous_telemetry=previous,
            racing_line=AmbitiousEdgeRacingLine(),
        )

        self.assertGreater(ambitious_reward, normal_reward - 0.05)

    def test_faster_progress_receives_higher_reward(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        slow = make_telemetry(
            speedX=36.0,
            distRaced=101.0,
            distFromStart=101.0,
            curLapTime=1.1,
        )
        fast = make_telemetry(
            speedX=180.0,
            distRaced=105.0,
            distFromStart=105.0,
            curLapTime=1.1,
        )

        slow_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            slow,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )
        fast_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            fast,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )

        self.assertGreater(fast_reward, slow_reward)

    def test_faster_safe_progress_can_outweigh_small_line_deviation(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        slow_on_line = make_telemetry(
            speedX=72.0,
            trackPos=0.0,
            distRaced=102.0,
            distFromStart=102.0,
            curLapTime=1.2,
        )
        faster_near_line = make_telemetry(
            speedX=126.0,
            trackPos=0.22,
            distRaced=105.0,
            distFromStart=105.0,
            curLapTime=1.2,
        )

        slow_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            slow_on_line,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )
        faster_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            faster_near_line,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )

        self.assertGreater(faster_reward, slow_reward)

    def test_completed_lap_bonus_still_rewards_finishing(self):
        previous = make_telemetry(distRaced=990.0, curLapTime=30.0)
        finished = make_telemetry(
            speedX=140.0,
            distRaced=1000.0,
            distFromStart=0.0,
            curLapTime=30.5,
        )

        reward_without_lap = train_hybrid_ppo_agent.calculate_hybrid_reward(
            finished,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )
        reward_with_lap = train_hybrid_ppo_agent.calculate_hybrid_reward(
            finished,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            completed_lap=30.5,
        )

        self.assertGreater(reward_with_lap, reward_without_lap)

    def test_off_track_failure_is_not_rewarded_like_fast_progress(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        safe_fast = make_telemetry(
            speedX=150.0,
            distRaced=108.0,
            distFromStart=108.0,
            curLapTime=1.2,
        )
        failed_fast = make_telemetry(
            speedX=150.0,
            trackPos=1.12,
            track=[-1.0] + [200.0] * 18,
            distRaced=108.0,
            distFromStart=108.0,
            curLapTime=1.2,
        )

        safe_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            safe_fast,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=8.0,
        )
        failed_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            failed_fast,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=8.0,
        )

        self.assertLess(failed_reward, safe_reward - 400.0)

    def test_mid_lap_off_track_failure_has_strong_terminal_penalty(self):
        previous = make_telemetry(distRaced=500.0, curLapTime=20.0)
        failed = make_telemetry(
            speedX=70.0,
            trackPos=1.08,
            track=[-1.0] + [200.0] * 18,
            distRaced=504.0,
            distFromStart=504.0,
            curLapTime=20.2,
        )

        reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            failed,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=500.0,
        )

        self.assertLess(reward, -450.0)

    def test_incomplete_lap_failure_penalty_shrinks_with_progress(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        failed = make_telemetry(
            speedX=120.0,
            trackPos=1.12,
            track=[-1.0] + [200.0] * 18,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )

        early_failure_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            failed,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=100.0,
        )
        later_failure_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            failed,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=800.0,
        )

        self.assertGreater(later_failure_reward, early_failure_reward)

    def test_clean_edge_use_is_not_penalised_by_track_position_alone(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        centre = make_telemetry(
            speedX=120.0,
            trackPos=0.0,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )
        stable_edge = make_telemetry(
            speedX=120.0,
            trackPos=0.98,
            speedY=0.0,
            angle=0.0,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )

        centre_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            centre,
            make_action(),
            previous_telemetry=previous,
        )
        stable_edge_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            stable_edge,
            make_action(),
            previous_telemetry=previous,
        )

        self.assertAlmostEqual(stable_edge_reward, centre_reward)

    def test_unstable_outward_edge_exit_is_penalised(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        stable_edge = make_telemetry(
            speedX=120.0,
            trackPos=0.98,
            speedY=0.0,
            angle=0.0,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )
        unstable_edge = make_telemetry(
            speedX=120.0,
            trackPos=0.98,
            speedY=8.5,
            angle=0.38,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )

        stable_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            stable_edge,
            make_action(steer=0.0, accel=0.4),
            previous_telemetry=previous,
        )
        unstable_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            unstable_edge,
            make_action(steer=0.4, accel=0.8),
            previous_telemetry=previous,
        )

        self.assertLess(unstable_reward, stable_reward)

    def test_unsafe_residual_pressure_reduces_reward(self):
        previous = make_telemetry(distRaced=100.0, curLapTime=1.0)
        telemetry = make_telemetry(
            speedX=120.0,
            distRaced=104.0,
            distFromStart=104.0,
            curLapTime=1.2,
        )

        safe_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            telemetry,
            make_action(agent4_unsafe_residual_pressure=0.0),
            previous_telemetry=previous,
        )
        unsafe_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            telemetry,
            make_action(agent4_unsafe_residual_pressure=1.0),
            previous_telemetry=previous,
        )

        self.assertLess(unsafe_reward, safe_reward)

    def test_stopped_car_time_accumulates_when_no_progress_is_made(self):
        previous = make_telemetry(speedX=0.0, distRaced=100.0, curLapTime=1.0)
        stopped = make_telemetry(speedX=0.0, distRaced=100.0, curLapTime=1.5)

        stopped_seconds = train_hybrid_ppo_agent.calculate_stopped_time_delta(
            previous,
            stopped,
            FakeRacingLine.track_length,
        )

        self.assertAlmostEqual(stopped_seconds, 0.5)

    def test_stopped_car_time_resets_when_progress_is_made(self):
        previous = make_telemetry(speedX=0.0, distRaced=100.0, curLapTime=1.0)
        moving = make_telemetry(speedX=12.0, distRaced=102.0, curLapTime=1.2)

        stopped_seconds = train_hybrid_ppo_agent.calculate_stopped_time_delta(
            previous,
            moving,
            FakeRacingLine.track_length,
        )

        self.assertEqual(stopped_seconds, 0.0)

    def test_stuck_state_receives_extra_penalty(self):
        previous = make_telemetry(speedX=0.0, distRaced=100.0, curLapTime=1.0)
        stopped = make_telemetry(speedX=0.0, distRaced=100.0, curLapTime=4.2)

        normal_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            stopped,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
        )
        stuck_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            stopped,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            stuck=True,
        )

        self.assertLess(stuck_reward, normal_reward)

    def test_mid_lap_stuck_failure_has_strong_failure_floor(self):
        previous = make_telemetry(speedX=0.0, distRaced=500.0, curLapTime=20.0)
        stopped = make_telemetry(
            speedX=0.0,
            distRaced=500.0,
            distFromStart=500.0,
            curLapTime=23.2,
        )

        normal_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            stopped,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=500.0,
        )
        stuck_reward = train_hybrid_ppo_agent.calculate_hybrid_reward(
            stopped,
            make_action(),
            previous_telemetry=previous,
            racing_line=FakeRacingLine(),
            episode_distance_m=500.0,
            stuck=True,
        )

        minimum_extra_penalty = (
            train_hybrid_ppo_agent.REWARD_WEIGHTS["stuck"]
            + train_hybrid_ppo_agent.REWARD_WEIGHTS["incomplete_lap_failure_floor"]
        )
        self.assertLess(stuck_reward, normal_reward - minimum_extra_penalty)


if __name__ == "__main__":
    unittest.main()
