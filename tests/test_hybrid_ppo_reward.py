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


if __name__ == "__main__":
    unittest.main()
