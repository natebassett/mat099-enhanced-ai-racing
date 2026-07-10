import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.map_aware_agent import (  # noqa: E402
    MapAwareAgent,
    calculate_dynamic_target_speed,
    estimate_line_complexity,
    guard_racing_line_target,
    preview_track_position,
    choose_dynamic_control_phase,
)
from racing_line import RacingLine, TorcsTrackMap  # noqa: E402
from racing_line.optimizer import _CORKSCREW_REFERENCE_ANCHORS  # noqa: E402
from racing_line.controller import (  # noqa: E402
    DrivingMode,
    calculate_aggressive_speed_control,
    calculate_aggressive_steering,
    choose_driving_mode,
)


TRACK_PATH = PROJECT_ROOT / "torcs" / "tracks" / "road" / "corkscrew" / "corkscrew.xml"
LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "corkscrew.json"


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
        "curLapTime": 10.0,
        "lastLapTime": 0.0,
    }
    telemetry.update(overrides)
    return telemetry


class TrackMapTests(unittest.TestCase):
    def test_parses_corkscrew_geometry(self):
        track_map = TorcsTrackMap.from_xml(TRACK_PATH)

        self.assertEqual(track_map.name, "corkscrew")
        self.assertEqual(track_map.width, 12.0)
        self.assertEqual(len(track_map.segments), 66)
        self.assertAlmostEqual(
            sum(segment.turn_radians for segment in track_map.segments),
            2.0 * 3.141592653589793,
            places=4,
        )


class RacingLineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(LINE_PATH.read_text(encoding="utf-8"))
        cls.racing_line = RacingLine(cls.payload)

    def test_artifact_matches_source_track(self):
        expected_hash = hashlib.sha256(TRACK_PATH.read_bytes()).hexdigest()

        self.assertEqual(self.payload["source_sha256"], expected_hash)
        self.assertTrue(self.payload["optimizer"]["success"])

    def test_waypoints_stay_inside_safety_margin(self):
        positions = [
            waypoint["target_track_pos"]
            for waypoint in self.payload["waypoints"]
        ]
        speeds = [
            waypoint["target_speed_kmh"]
            for waypoint in self.payload["waypoints"]
        ]

        self.assertGreater(len(positions), 1000)
        self.assertLessEqual(max(map(abs, positions)), 0.75)
        self.assertGreater(max(map(abs, positions)), 0.30)
        self.assertGreater(min(speeds), 40.0)
        self.assertLessEqual(max(speeds), 216.0)

    def test_lookup_wraps_at_finish_line(self):
        before_finish = self.racing_line.lookup(self.racing_line.track_length - 0.1)
        after_finish = self.racing_line.lookup(self.racing_line.track_length + 0.1)

        self.assertLess(
            abs(
                before_finish["target_track_pos"]
                - after_finish["target_track_pos"]
            ),
            0.05,
        )

    def test_reference_profile_matches_corner_anchors(self):
        self.assertEqual(
            self.payload["optimizer"]["reference_profile"],
            "laguna_seca_corkscrew_v1",
        )

        for anchor in _CORKSCREW_REFERENCE_ANCHORS:
            waypoint = self.racing_line.lookup(anchor["distance"])

            self.assertLessEqual(
                abs(waypoint["target_track_pos"] - anchor["track_pos"]),
                0.05,
                anchor["name"],
            )
            self.assertLessEqual(
                abs(waypoint["target_speed_kmh"] - anchor["speed_kmh"]),
                5.0,
                anchor["name"],
            )

    def test_t1_entry_stays_on_launch_side_before_apex(self):
        for distance in [3598.0, 0.0, 50.0, 120.0]:
            waypoint = self.racing_line.lookup(distance)

            self.assertGreater(
                waypoint["target_track_pos"],
                0.30,
                f"T1 setup at {distance} m",
            )
            self.assertLess(
                waypoint["target_track_pos"],
                0.46,
                f"T1 setup at {distance} m",
            )

        self.assertLess(
            self.racing_line.lookup(200.0)["target_track_pos"],
            -0.30,
        )
        self.assertGreater(
            self.racing_line.lookup(545.0)["target_track_pos"],
            0.34,
        )
        self.assertLess(
            self.racing_line.lookup(545.0)["target_track_pos"],
            0.48,
        )

    def test_t3_setup_avoids_full_width_snap_after_t2(self):
        self.assertGreater(
            self.racing_line.lookup(585.0)["target_track_pos"],
            0.34,
        )
        self.assertLess(
            self.racing_line.lookup(585.0)["target_track_pos"],
            0.48,
        )
        self.assertGreater(
            self.racing_line.lookup(620.0)["target_track_pos"],
            -0.45,
        )
        self.assertLess(
            self.racing_line.lookup(620.0)["target_track_pos"],
            -0.20,
        )
        self.assertGreater(
            self.racing_line.lookup(775.0)["target_track_pos"],
            0.34,
        )
        self.assertLess(
            self.racing_line.lookup(775.0)["target_track_pos"],
            0.48,
        )

    def test_dynamic_speed_stays_fast_on_clear_straight(self):
        target = self.racing_line.lookup(90.0)
        lookahead = self.racing_line.lookup(116.0)
        far_lookahead = self.racing_line.lookup(138.0)
        very_far_lookahead = self.racing_line.lookup(175.0)

        target_speed = calculate_dynamic_target_speed(
            speed=80.0,
            target=target,
            lookahead=lookahead,
            far_lookahead=far_lookahead,
            very_far_lookahead=very_far_lookahead,
            live_sensor_speed_cap=216.0,
            line_complexity=0.0,
            line_error=0.02,
        )

        self.assertGreater(target_speed, 180.0)

    def test_dynamic_speed_anticipates_t2_curvature(self):
        target = self.racing_line.lookup(350.0)
        lookahead = self.racing_line.lookup(402.0)
        far_lookahead = self.racing_line.lookup(455.0)
        very_far_lookahead = self.racing_line.lookup(520.0)

        target_speed = calculate_dynamic_target_speed(
            speed=112.0,
            target=target,
            lookahead=lookahead,
            far_lookahead=far_lookahead,
            very_far_lookahead=very_far_lookahead,
            live_sensor_speed_cap=216.0,
            line_complexity=0.45,
            line_error=0.06,
        )

        self.assertLess(target_speed, 118.0)
        self.assertGreater(target_speed, 58.0)

    def test_dynamic_phase_brakes_from_target_delta(self):
        self.assertEqual(
            choose_dynamic_control_phase(
                speed=112.0,
                target_speed=92.0,
                curvature=0.002,
                front_sensor=120.0,
            ),
            "brake",
        )

    def test_line_has_no_abrupt_lateral_heading_changes(self):
        maximum_heading_offset = max(
            abs(waypoint["heading_offset"])
            for waypoint in self.payload["waypoints"]
        )

        # The reference profile deliberately cuts across the full track on
        # short straights, so its lateral heading is more aggressive than the
        # old generic centre-biased line.
        self.assertLess(maximum_heading_offset, 0.60)

    def test_line_contains_map_aware_driving_phases(self):
        phases = {
            waypoint["phase"]
            for waypoint in self.payload["waypoints"]
        }

        self.assertIn("brake", phases)
        self.assertIn("accelerate", phases)
        self.assertTrue(any("left_turn" in phase for phase in phases))
        self.assertTrue(any("right_turn" in phase for phase in phases))
        self.assertTrue(
            all("speed_delta_30m" in waypoint for waypoint in self.payload["waypoints"])
        )


class MapAwareAgentTests(unittest.TestCase):
    def test_action_is_bounded_and_uses_full_control(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = MapAwareAgent(
                racing_line_path=LINE_PATH,
                telemetry_path=Path(directory) / "telemetry.csv",
            )
            action = agent.act(None, make_telemetry())
            agent.close()

        self.assertTrue(agent.uses_full_control)
        self.assertLessEqual(abs(action["steer"]), 1.0)
        self.assertGreaterEqual(action["accel"], 0.0)
        self.assertLessEqual(action["accel"], 1.0)
        self.assertGreaterEqual(action["brake"], 0.0)
        self.assertLessEqual(action["brake"], 1.0)
        self.assertIn(action["gear"], range(1, 7))

    def test_startup_line_merge_does_not_reset_at_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            agent = MapAwareAgent(
                racing_line_path=LINE_PATH,
                telemetry_path=telemetry_path,
            )
            agent.act(
                None,
                make_telemetry(
                    distFromStart=3598.0,
                    distRaced=0.0,
                    trackPos=0.33,
                ),
            )
            agent.act(
                None,
                make_telemetry(
                    distFromStart=3598.0,
                    distRaced=3608.0,
                    trackPos=-0.70,
                ),
            )
            agent.close()

            with telemetry_path.open(newline="", encoding="utf-8") as handle:
                telemetry_rows = list(csv.DictReader(handle))

        self.assertEqual(float(telemetry_rows[-1]["lineBlend"]), 1.0)

    def test_launch_does_not_snap_to_racing_line_while_stationary(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            agent = MapAwareAgent(
                racing_line_path=LINE_PATH,
                telemetry_path=telemetry_path,
            )
            action = agent.act(
                None,
                make_telemetry(
                    speedX=0.0,
                    distFromStart=3598.45,
                    distRaced=0.0,
                    trackPos=0.33,
                ),
            )
            agent.close()

            with telemetry_path.open(newline="", encoding="utf-8") as handle:
                telemetry_rows = list(csv.DictReader(handle))

        self.assertEqual(float(telemetry_rows[-1]["lineBlend"]), 0.0)
        self.assertAlmostEqual(
            float(telemetry_rows[-1]["targetTrackPos"]),
            0.33,
        )
        self.assertLess(abs(action["steer"]), 0.08)
        self.assertGreaterEqual(action["accel"], 0.85)
        self.assertEqual(action["brake"], 0.0)

    def test_launch_target_does_not_creep_from_sensor_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            agent = MapAwareAgent(
                racing_line_path=LINE_PATH,
                telemetry_path=telemetry_path,
            )
            launch_track = [62.0] * 19
            launch_track[5] = 112.0
            for _step in range(12):
                agent.act(
                    None,
                    make_telemetry(
                        speedX=0.0,
                        distFromStart=3598.45,
                        distRaced=0.0,
                        trackPos=0.33,
                        track=launch_track,
                    ),
                )
            agent.close()

            with telemetry_path.open(newline="", encoding="utf-8") as handle:
                telemetry_rows = list(csv.DictReader(handle))

        self.assertAlmostEqual(
            float(telemetry_rows[-1]["targetTrackPos"]),
            0.33,
        )

    def test_launch_stuck_terminates_after_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = MapAwareAgent(
                racing_line_path=LINE_PATH,
                telemetry_path=Path(directory) / "telemetry.csv",
            )
            action = None
            for step in range(252):
                action = agent.act(
                    None,
                    make_telemetry(
                        speedX=0.0,
                        distFromStart=3598.45,
                        distRaced=0.0,
                        trackPos=0.33,
                        curLapTime=step / 50.0,
                    ),
                )
            agent.close()

        self.assertTrue(action["terminate"])
        self.assertEqual(action["termination_reason"], "launch_stuck")

    def test_preview_target_calms_tight_alternating_bends(self):
        target = {
            "target_track_pos": -0.20,
            "curvature": 0.015,
        }
        lookahead = {
            "target_track_pos": 0.18,
            "curvature": -0.014,
        }
        far_lookahead = {
            "target_track_pos": -0.10,
            "curvature": 0.012,
        }

        complexity = estimate_line_complexity(target, lookahead, far_lookahead)
        preview_position = preview_track_position(
            target,
            lookahead,
            far_lookahead,
            speed=120.0,
            line_complexity=complexity,
        )

        self.assertGreater(complexity, 0.90)
        self.assertGreater(preview_position, target["target_track_pos"])
        self.assertLess(preview_position, lookahead["target_track_pos"])

    def test_normal_corner_sensor_does_not_override_saved_line(self):
        track = [18.0] * 19
        guarded = guard_racing_line_target(
            track_position=0.02,
            target_position=0.72,
            speed=86.0,
            track_sensors=track,
            line_complexity=0.45,
            line_blend=1.0,
        )

        self.assertGreater(guarded, 0.68)

    def test_edge_hug_becomes_limit_mode_when_pointing_outward(self):
        mode = choose_driving_mode(
            track_pos=-0.35,
            angle=0.18,
            speed_y=0.2,
            line_error=0.05,
        )

        self.assertEqual(mode, DrivingMode.LIMIT)

    def test_edge_risk_countersteers_before_sliding_off(self):
        steer = calculate_aggressive_steering(
            speed=85.0,
            track_pos=-0.35,
            angle=0.18,
            speed_y=0.2,
            target_track_pos=-0.30,
            heading_offset=-0.06,
            curvature=0.001,
            previous_steer=0.0,
            mode=DrivingMode.LIMIT,
        )

        self.assertGreater(steer, 0.0)

    def test_braking_phase_accelerates_when_well_below_target(self):
        accel, brake, _adjusted_target = calculate_aggressive_speed_control(
            speed=82.0,
            target_speed=145.0,
            steer=0.04,
            track_pos=0.30,
            angle=0.02,
            speed_y=0.0,
            line_error=0.05,
            mode=DrivingMode.ATTACK,
            phase="brake",
        )

        self.assertGreater(accel, 0.5)
        self.assertEqual(brake, 0.0)

    def test_lateral_drift_cuts_throttle_before_full_spin(self):
        accel, brake, _adjusted_target = calculate_aggressive_speed_control(
            speed=84.0,
            target_speed=180.0,
            steer=0.03,
            track_pos=0.30,
            angle=-0.12,
            speed_y=-9.0,
            line_error=0.08,
            mode=DrivingMode.ATTACK,
            phase="accelerate_left_turn",
        )

        self.assertEqual(accel, 0.0)
        self.assertGreaterEqual(brake, 0.10)


if __name__ == "__main__":
    unittest.main()
