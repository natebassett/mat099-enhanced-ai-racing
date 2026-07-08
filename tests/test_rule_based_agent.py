import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.rule_based_agent import (  # noqa: E402
    calculate_steering,
    get_corner_throttle_cap,
    get_desired_track_pos,
    get_steering_limit,
    is_stability_mode,
    plan_next_curve,
)


def make_state(**overrides):
    state = {
        "speedX": 100.0,
        "speedY": 0.0,
        "speedZ": 0.0,
        "angle": 0.0,
        "trackPos": 0.0,
        "damage": 0.0,
        "rpm": 5000.0,
        "track": [200.0] * 19,
        "wheelSpinVel": [10.0] * 4,
    }
    state.update(overrides)
    return state


class RuleBasedAgentSafetyTests(unittest.TestCase):
    def test_racing_target_is_always_the_centreline(self):
        state = make_state(track=[35.0] * 19)
        plan = plan_next_curve(state)

        self.assertEqual(get_desired_track_pos(state, plan), 0.0)

    def test_steering_moves_car_back_toward_centre(self):
        right_of_centre = make_state(trackPos=0.5)
        left_of_centre = make_state(trackPos=-0.5)

        self.assertLess(calculate_steering(right_of_centre), 0.0)
        self.assertGreater(calculate_steering(left_of_centre), 0.0)

    def test_high_speed_steering_is_limited(self):
        state = make_state(speedX=180.0, angle=0.5)
        steer = calculate_steering(state)

        self.assertLessEqual(abs(steer), 0.38)
        self.assertEqual(get_steering_limit(180.0, 0.0, False), 0.38)

    def test_large_steering_input_cuts_throttle(self):
        self.assertEqual(get_corner_throttle_cap(0.8, 0.0), 0.10)
        self.assertEqual(get_corner_throttle_cap(0.65, 0.0), 0.20)
        self.assertEqual(get_corner_throttle_cap(0.2, 0.6), 0.30)

    def test_lateral_slide_activates_stability_mode_early(self):
        state = make_state(speedY=11.0, angle=0.13)

        self.assertTrue(is_stability_mode(state))

if __name__ == "__main__":
    unittest.main()
