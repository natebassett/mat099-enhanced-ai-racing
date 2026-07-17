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
    DEFAULT_RACING_LINE_PATH,
    MapAwareAgent,
    RacingControlState,
    apply_edge_guard_crawl_assist,
    apply_practical_target_limit,
    apply_raceline_momentum_coast,
    cap_speed_for_line_join,
    cap_speed_for_control_state,
    calculate_path_tracking_steer,
    calculate_raceline_pursuit_steer,
    calculate_raceline_pedals,
    calculate_raceline_target_speed,
    calculate_dynamic_target_speed,
    choose_raceline_control_state,
    choose_control_state,
    confidence_speed_cap,
    estimate_line_complexity,
    get_racing_line_corridor,
    guard_racing_line_target,
    map_safety_speed_cap,
    optimizer_sweeper_speed_cap,
    preview_track_position,
    choose_dynamic_control_phase,
    soften_line_error,
    update_raceline_confidence_debt,
)
from racing_line import RacingLine, TorcsTrackMap  # noqa: E402
from racing_line.optimizer import _CORKSCREW_REFERENCE_ANCHORS  # noqa: E402
from racing_line.controller import (  # noqa: E402
    DrivingMode,
    calculate_aggressive_speed_control,
    calculate_aggressive_steering,
    choose_driving_mode,
)


CORKSCREW_TRACK_PATH = (
    PROJECT_ROOT / "torcs" / "tracks" / "road" / "corkscrew" / "corkscrew.xml"
)
CORKSCREW_LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "corkscrew.json"
GTRACK3_TRACK_PATH = (
    PROJECT_ROOT / "torcs" / "tracks" / "road" / "g-track-3" / "g-track-3.xml"
)
GTRACK3_LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "g-track-3.json"


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
        track_map = TorcsTrackMap.from_xml(CORKSCREW_TRACK_PATH)

        self.assertEqual(track_map.name, "corkscrew")
        self.assertEqual(track_map.width, 12.0)
        self.assertEqual(len(track_map.segments), 66)
        self.assertAlmostEqual(
            sum(segment.turn_radians for segment in track_map.segments),
            2.0 * 3.141592653589793,
            places=4,
        )

    def test_parses_g_track_3_geometry_with_default_objects_entity(self):
        track_map = TorcsTrackMap.from_xml(GTRACK3_TRACK_PATH)

        self.assertEqual(track_map.name, "CG SPEEDWAY 3")
        self.assertEqual(track_map.width, 10.0)
        self.assertEqual(len(track_map.segments), 39)
        self.assertAlmostEqual(
            sum(segment.turn_radians for segment in track_map.segments),
            2.0 * 3.141592653589793,
            places=4,
        )


class CorkscrewRacingLineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(CORKSCREW_LINE_PATH.read_text(encoding="utf-8"))
        cls.racing_line = RacingLine(cls.payload)

    def test_artifact_matches_source_track(self):
        expected_hash = hashlib.sha256(CORKSCREW_TRACK_PATH.read_bytes()).hexdigest()

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


class GTrack3RacingLineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(GTRACK3_LINE_PATH.read_text(encoding="utf-8"))
        cls.racing_line = RacingLine(cls.payload)

    def test_default_map_aware_line_is_g_track_3(self):
        self.assertEqual(DEFAULT_RACING_LINE_PATH, GTRACK3_LINE_PATH)

    def test_artifact_matches_source_track(self):
        expected_hash = hashlib.sha256(GTRACK3_TRACK_PATH.read_bytes()).hexdigest()

        self.assertEqual(self.payload["track_name"], "CG SPEEDWAY 3")
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

        self.assertGreater(len(positions), 900)
        maximum_offset = max(map(abs, positions))
        average_offset = sum(abs(position) for position in positions) / len(positions)
        self.assertGreater(maximum_offset, 0.56)
        self.assertLessEqual(maximum_offset, 0.65)
        self.assertGreater(average_offset, 0.26)
        self.assertLess(average_offset, 0.36)
        self.assertGreater(
            sum(abs(position) > 0.35 for position in positions),
            320,
        )
        self.assertGreater(
            sum(abs(position) > 0.50 for position in positions),
            90,
        )
        self.assertGreater(
            sum(0.10 <= abs(position) <= 0.22 for position in positions),
            80,
        )
        self.assertGreaterEqual(min(speeds), 82.0)
        self.assertLessEqual(max(speeds), 216.0)

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

    def test_speedway_line_cuts_inside_first_right_bend(self):
        self.assertGreater(
            self.racing_line.lookup(0.0)["target_track_pos"],
            0.15,
        )
        self.assertLess(
            abs(self.racing_line.lookup(20.0)["target_track_pos"]),
            0.05,
        )
        self.assertLess(
            self.racing_line.lookup(40.0)["target_track_pos"],
            -0.19,
        )
        self.assertLess(
            self.racing_line.lookup(50.0)["target_track_pos"],
            -0.44,
        )
        self.assertLess(
            self.racing_line.lookup(55.0)["target_track_pos"],
            -0.52,
        )
        self.assertLess(
            self.racing_line.lookup(60.0)["target_track_pos"],
            -0.52,
        )
        self.assertGreater(
            self.racing_line.lookup(80.0)["target_track_pos"],
            -0.16,
        )

    def test_speedway_line_cuts_inside_compound_bends(self):
        self.assertGreater(
            self.racing_line.lookup(315.0)["target_track_pos"],
            0.42,
        )
        self.assertLess(
            self.racing_line.lookup(840.0)["target_track_pos"],
            -0.38,
        )
        self.assertGreater(
            self.racing_line.lookup(1320.0)["target_track_pos"],
            0.42,
        )


class MapAwareAgentTests(unittest.TestCase):
    def test_action_is_bounded_and_uses_full_control(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = MapAwareAgent(
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
                telemetry_path=telemetry_path,
            )
            agent.act(
                None,
                make_telemetry(
                    distFromStart=2838.0,
                    distRaced=0.0,
                    trackPos=0.33,
                ),
            )
            agent.act(
                None,
                make_telemetry(
                    distFromStart=2838.0,
                    distRaced=2843.0,
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
                telemetry_path=telemetry_path,
            )
            action = agent.act(
                None,
                make_telemetry(
                    speedX=0.0,
                    distFromStart=2838.0,
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
                telemetry_path=telemetry_path,
            )
            launch_track = [62.0] * 19
            launch_track[5] = 112.0
            for _step in range(12):
                agent.act(
                    None,
                    make_telemetry(
                        speedX=0.0,
                        distFromStart=2838.0,
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

    def test_pre_finish_launch_gate_holds_spawn_lane(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            agent = MapAwareAgent(
                telemetry_path=telemetry_path,
            )
            agent.act(
                None,
                make_telemetry(
                    speedX=0.0,
                    distFromStart=2743.1,
                    distRaced=0.0,
                    trackPos=0.0,
                ),
            )
            action = agent.act(
                None,
                make_telemetry(
                    speedX=80.0,
                    speedY=-12.0,
                    angle=-0.27,
                    distFromStart=2761.5,
                    distRaced=18.4,
                    trackPos=0.08,
                ),
            )
            agent.close()

            with telemetry_path.open(newline="", encoding="utf-8") as handle:
                telemetry_rows = list(csv.DictReader(handle))

        latest = telemetry_rows[-1]
        self.assertLess(abs(float(latest["targetTrackPos"])), 0.05)
        self.assertGreater(float(latest["targetSpeed"]), 80.0)
        self.assertLess(action["steer"], 0.0)

    def test_pre_finish_launch_has_no_artificial_speed_leash_when_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            agent = MapAwareAgent(
                telemetry_path=telemetry_path,
            )
            agent.act(
                None,
                make_telemetry(
                    speedX=0.0,
                    distFromStart=2743.1,
                    distRaced=0.0,
                    trackPos=0.0,
                ),
            )
            agent.act(
                None,
                make_telemetry(
                    speedX=80.0,
                    speedY=0.0,
                    angle=0.0,
                    distFromStart=2761.5,
                    distRaced=18.4,
                    trackPos=0.02,
                ),
            )
            agent.close()

            with telemetry_path.open(newline="", encoding="utf-8") as handle:
                telemetry_rows = list(csv.DictReader(handle))

        latest = telemetry_rows[-1]
        self.assertLess(abs(float(latest["targetTrackPos"])), 0.05)
        self.assertGreater(float(latest["targetSpeed"]), 150.0)

    def test_launch_stuck_terminates_after_five_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = MapAwareAgent(
                telemetry_path=Path(directory) / "telemetry.csv",
            )
            action = None
            for step in range(252):
                action = agent.act(
                    None,
                    make_telemetry(
                        speedX=0.0,
                        distFromStart=2838.0,
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
            "turn_direction": 1.0,
        }
        lookahead = {
            "target_track_pos": 0.18,
            "curvature": -0.014,
            "turn_direction": -1.0,
        }
        far_lookahead = {
            "target_track_pos": -0.10,
            "curvature": 0.012,
            "turn_direction": 1.0,
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

    def test_preview_target_preserves_inside_apex_when_future_unwinds(self):
        target = {
            "target_track_pos": -0.48,
            "curvature": -0.025,
            "turn_direction": -1.0,
        }
        lookahead = {
            "target_track_pos": -0.12,
            "curvature": -0.004,
            "turn_direction": -1.0,
        }
        far_lookahead = {
            "target_track_pos": 0.02,
            "curvature": 0.000,
            "turn_direction": 0.0,
        }

        preview_position = preview_track_position(
            target,
            lookahead,
            far_lookahead,
            speed=95.0,
            line_complexity=0.35,
        )

        self.assertAlmostEqual(preview_position, -0.48)

    def test_agent_targets_locked_first_bend_apex_not_preview_unwind(self):
        with tempfile.TemporaryDirectory() as directory:
            telemetry_path = Path(directory) / "telemetry.csv"
            agent = MapAwareAgent(
                telemetry_path=telemetry_path,
            )
            distances = [
                (2743.0, 0.0),
                (2800.0, 57.0),
                (2840.0, 97.0),
                (0.0, 100.0),
                (20.0, 120.0),
                (30.0, 130.0),
                (40.0, 140.0),
                (50.0, 150.0),
                (55.0, 155.0),
                (60.0, 160.0),
            ]
            for dist_from_start, dist_raced in distances:
                agent.act(
                    None,
                    make_telemetry(
                        distFromStart=dist_from_start,
                        distRaced=dist_raced,
                        speedX=92.0,
                        trackPos=0.0,
                    ),
                )
            agent.close()

            with telemetry_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        by_distance = {
            round(float(row["distFromStart"]), 1): row
            for row in rows
        }
        self.assertLess(float(by_distance[30.0]["previewTrackPos"]), -0.42)
        self.assertLess(float(by_distance[30.0]["steer"]), -0.05)
        self.assertLess(float(by_distance[50.0]["rawRacingLinePos"]), -0.44)
        self.assertLess(float(by_distance[55.0]["rawRacingLinePos"]), -0.52)
        self.assertLess(float(by_distance[60.0]["rawRacingLinePos"]), -0.52)

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

    def test_line_error_retains_strong_pull_inside_corridor(self):
        corridor = get_racing_line_corridor(
            speed=140.0,
            curvature=0.010,
            line_complexity=0.20,
        )
        correction = soften_line_error(0.06, corridor)

        self.assertLess(corridor, 0.12)
        self.assertGreater(correction, 0.035)

    def test_large_line_join_error_caps_speed_progressively(self):
        capped = cap_speed_for_line_join(
            target_speed=190.0,
            track_position=0.08,
            preview_position=0.74,
        )

        self.assertEqual(capped, 98.0)

    def test_moderate_line_join_error_uses_gentler_speed_cap(self):
        capped = cap_speed_for_line_join(
            target_speed=190.0,
            track_position=0.38,
            preview_position=0.74,
        )

        self.assertEqual(capped, 130.0)

    def test_state_machine_starts_in_join_when_far_from_raceline(self):
        state = choose_control_state(
            RacingControlState.RACE,
            line_blend=1.0,
            track_position=0.05,
            target_position=0.52,
            speed=92.0,
            lateral_speed=0.4,
            angle=0.02,
            front_sensor=80.0,
            min_track_sensor=70.0,
            raw_sensor_speed_cap=216.0,
        )

        self.assertEqual(state, RacingControlState.JOIN_RACELINE)

    def test_state_machine_enters_edge_guard_for_short_sensor_when_unsettled(self):
        state = choose_control_state(
            RacingControlState.RACE,
            line_blend=1.0,
            track_position=0.44,
            target_position=0.62,
            speed=95.0,
            lateral_speed=-5.0,
            angle=-0.16,
            front_sensor=24.0,
            min_track_sensor=24.0,
            raw_sensor_speed_cap=105.0,
        )

        self.assertEqual(state, RacingControlState.EDGE_GUARD)

    def test_state_machine_ignores_moderate_front_sensor_when_stable(self):
        state = choose_control_state(
            RacingControlState.RACE,
            line_blend=1.0,
            track_position=0.53,
            target_position=0.56,
            speed=94.0,
            lateral_speed=0.1,
            angle=0.03,
            front_sensor=44.0,
            min_track_sensor=12.0,
            raw_sensor_speed_cap=128.0,
        )

        self.assertEqual(state, RacingControlState.RACE)

    def test_state_machine_ignores_short_front_sensor_when_stable_on_line(self):
        state = choose_control_state(
            RacingControlState.RACE,
            line_blend=1.0,
            track_position=0.25,
            target_position=0.28,
            speed=96.0,
            lateral_speed=0.2,
            angle=0.02,
            front_sensor=13.0,
            min_track_sensor=13.0,
            raw_sensor_speed_cap=60.0,
        )

        self.assertEqual(state, RacingControlState.RACE)

    def test_edge_guard_releases_to_join_when_stable_but_off_line(self):
        state = choose_control_state(
            RacingControlState.EDGE_GUARD,
            line_blend=1.0,
            track_position=-0.49,
            target_position=-0.10,
            speed=63.0,
            lateral_speed=0.2,
            angle=0.02,
            front_sensor=22.0,
            min_track_sensor=22.0,
            raw_sensor_speed_cap=82.0,
        )

        self.assertEqual(state, RacingControlState.JOIN_RACELINE)

    def test_practical_target_limit_keeps_join_state_off_extreme_edge(self):
        limited = apply_practical_target_limit(
            0.75,
            control_state=RacingControlState.JOIN_RACELINE,
            speed=85.0,
            lateral_speed=0.0,
            angle=0.0,
            line_blend=0.8,
        )

        self.assertGreater(limited, 0.66)
        self.assertLessEqual(limited, 0.70)

    def test_race_state_allows_full_width_only_when_stable(self):
        stable = apply_practical_target_limit(
            0.75,
            control_state=RacingControlState.RACE,
            speed=145.0,
            lateral_speed=0.3,
            angle=0.01,
            line_blend=1.0,
        )
        unstable = apply_practical_target_limit(
            0.75,
            control_state=RacingControlState.RACE,
            speed=145.0,
            lateral_speed=7.0,
            angle=0.18,
            line_blend=1.0,
        )

        self.assertGreater(stable, 0.70)
        self.assertLessEqual(unstable, 0.67)

    def test_control_state_caps_speed_when_joining_line(self):
        capped = cap_speed_for_control_state(
            190.0,
            control_state=RacingControlState.JOIN_RACELINE,
            line_join_error=0.52,
            line_error=-0.52,
            speed=105.0,
            lateral_speed=2.0,
            angle=0.02,
            front_sensor=80.0,
        )

        self.assertLessEqual(capped, 109.0)

    def test_path_tracker_steers_toward_target_raceline_gradually(self):
        steer = calculate_path_tracking_steer(
            speed=110.0,
            track_position=0.05,
            target_position=0.45,
            heading_offset=0.0,
            curvature=0.0,
            lateral_speed=0.0,
            angle=0.0,
            previous_steer=0.0,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertGreater(steer, 0.0)
        self.assertGreater(steer, 0.03)
        self.assertLess(steer, 0.08)

    def test_raceline_pursuit_countersteers_positive_drift(self):
        steer = calculate_raceline_pursuit_steer(
            speed=80.0,
            track_position=0.08,
            pursuit_target=0.0,
            heading_offset=0.0,
            curvature=0.0,
            lateral_speed=-12.0,
            angle=-0.20,
            previous_steer=0.0,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertLess(steer, 0.0)

    def test_stable_speedway_sweeper_gets_agent_two_speed_confidence(self):
        waypoint = {
            "curvature": 0.0105,
            "target_speed_kmh": 108.0,
        }

        target_speed = calculate_raceline_target_speed(
            speed=104.0,
            local=waypoint,
            near=waypoint,
            far=waypoint,
            very_far=waypoint,
            track_position=0.34,
            pursuit_target=0.30,
            lateral_speed=0.3,
            angle=0.02,
            sensor_cap=216.0,
            line_complexity=0.10,
        )

        self.assertGreaterEqual(target_speed, 150.0)

    def test_edge_apex_sweeper_respects_optimizer_speed_profile(self):
        waypoint = {
            "curvature": 0.0115,
            "target_speed_kmh": 119.0,
            "target_track_pos": 0.57,
            "turn_direction": 1.0,
        }

        target_speed = calculate_raceline_target_speed(
            speed=150.0,
            local=waypoint,
            near=waypoint,
            far=waypoint,
            very_far=waypoint,
            track_position=0.50,
            pursuit_target=0.57,
            lateral_speed=0.2,
            angle=0.02,
            sensor_cap=216.0,
            line_complexity=0.0,
        )

        self.assertGreaterEqual(target_speed, 119.0)
        self.assertLessEqual(target_speed, 126.0)

    def test_sweeper_cap_ramps_in_before_wide_apex(self):
        waypoint = {
            "curvature": 0.0115,
            "target_speed_kmh": 119.0,
            "turn_direction": 1.0,
        }

        early_cap = optimizer_sweeper_speed_cap(
            waypoint,
            waypoint,
            waypoint,
            pursuit_target=0.46,
            line_complexity=0.0,
        )
        apex_cap = optimizer_sweeper_speed_cap(
            waypoint,
            waypoint,
            waypoint,
            pursuit_target=0.57,
            line_complexity=0.0,
        )

        self.assertGreater(early_cap, apex_cap + 15.0)
        self.assertLessEqual(apex_cap, 126.0)

    def test_sweeper_cap_ignores_low_speed_corner_lookalike(self):
        local = {
            "curvature": 0.0115,
            "target_speed_kmh": 86.0,
            "target_track_pos": 0.50,
            "turn_direction": 1.0,
        }
        near = local | {"target_speed_kmh": 110.0}
        far = local | {"target_speed_kmh": 119.0}

        cap = optimizer_sweeper_speed_cap(
            local,
            near,
            far,
            pursuit_target=0.50,
            line_complexity=0.0,
        )

        self.assertEqual(cap, 216.0)

    def test_sweeper_cap_ignores_large_preview_lane_jump(self):
        waypoint = {
            "curvature": 0.0115,
            "target_speed_kmh": 119.0,
            "target_track_pos": 0.42,
            "turn_direction": 1.0,
        }

        cap = optimizer_sweeper_speed_cap(
            waypoint,
            waypoint,
            waypoint,
            pursuit_target=0.56,
            line_complexity=0.0,
        )

        self.assertEqual(cap, 216.0)

    def test_sweeper_cap_ignores_opposite_direction_sweeper(self):
        waypoint = {
            "curvature": -0.0115,
            "target_speed_kmh": 119.0,
            "target_track_pos": -0.57,
            "turn_direction": -1.0,
        }

        cap = optimizer_sweeper_speed_cap(
            waypoint,
            waypoint,
            waypoint,
            pursuit_target=-0.57,
            line_complexity=0.0,
        )

        self.assertEqual(cap, 216.0)

    def test_stable_sweeper_overspeed_coasts_instead_of_braking(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.36,
            focused_sweeper=True,
            speed=129.0,
            target_speed=108.5,
            pursuit_error=0.42,
            track_position=0.16,
            lateral_speed=-2.7,
            angle=-0.06,
            sensor_cap=216.0,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.0)

    def test_unstable_sweeper_keeps_braking_authority(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.36,
            focused_sweeper=True,
            speed=129.0,
            target_speed=108.5,
            pursuit_error=0.42,
            track_position=0.16,
            lateral_speed=-6.2,
            angle=-0.06,
            sensor_cap=216.0,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.36)

    def test_sweeper_sensor_danger_keeps_braking_authority(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.36,
            focused_sweeper=True,
            speed=129.0,
            target_speed=108.5,
            pursuit_error=0.42,
            track_position=0.16,
            lateral_speed=-2.7,
            angle=-0.06,
            sensor_cap=92.0,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.36)

    def test_stable_mapped_section_coasts_instead_of_mild_braking(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.26,
            focused_sweeper=False,
            speed=150.0,
            target_speed=134.0,
            pursuit_error=0.25,
            track_position=0.30,
            lateral_speed=2.0,
            angle=0.08,
            sensor_cap=216.0,
            control_state=RacingControlState.RACE,
            preview_curvature=0.012,
            line_complexity=0.20,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.0)

    def test_mapwide_coast_keeps_heavy_braking_authority(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.48,
            focused_sweeper=False,
            speed=150.0,
            target_speed=134.0,
            pursuit_error=0.25,
            track_position=0.30,
            lateral_speed=2.0,
            angle=0.08,
            sensor_cap=216.0,
            control_state=RacingControlState.RACE,
            preview_curvature=0.012,
            line_complexity=0.20,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.48)

    def test_mapwide_coast_keeps_linked_bend_braking_authority(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.26,
            focused_sweeper=False,
            speed=150.0,
            target_speed=134.0,
            pursuit_error=0.25,
            track_position=0.30,
            lateral_speed=2.0,
            angle=0.08,
            sensor_cap=216.0,
            control_state=RacingControlState.RACE,
            preview_curvature=0.012,
            line_complexity=0.70,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.26)

    def test_focused_sweeper_does_not_use_looser_mapwide_speed_limit(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.26,
            focused_sweeper=True,
            speed=150.0,
            target_speed=134.0,
            pursuit_error=0.25,
            track_position=0.30,
            lateral_speed=2.0,
            angle=0.08,
            sensor_cap=216.0,
            control_state=RacingControlState.RACE,
            preview_curvature=0.012,
            line_complexity=0.20,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.26)

    def test_gentle_lane_transition_coasts_despite_moderate_complexity(self):
        accel, brake = apply_raceline_momentum_coast(
            0.0,
            0.21,
            focused_sweeper=False,
            speed=132.0,
            target_speed=118.0,
            pursuit_error=0.20,
            track_position=0.25,
            lateral_speed=1.8,
            angle=0.07,
            sensor_cap=216.0,
            control_state=RacingControlState.JOIN_RACELINE,
            preview_curvature=0.006,
            line_complexity=0.70,
        )

        self.assertEqual(accel, 0.0)
        self.assertEqual(brake, 0.0)

    def test_moderate_line_miss_can_still_accelerate_when_under_target(self):
        accel, brake = calculate_raceline_pedals(
            speed=105.0,
            target_speed=132.0,
            steer=0.31,
            pursuit_error=0.28,
            track_position=0.32,
            lateral_speed=0.6,
            angle=0.03,
            previous_accel=None,
            previous_brake=None,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertEqual(brake, 0.0)
        self.assertGreaterEqual(accel, 0.50)

    def test_stable_on_line_releases_residual_brake_quickly(self):
        accel, brake = calculate_raceline_pedals(
            speed=195.0,
            target_speed=210.0,
            steer=0.02,
            pursuit_error=0.02,
            track_position=0.22,
            lateral_speed=0.05,
            angle=0.002,
            previous_accel=0.0,
            previous_brake=0.38,
            control_state=RacingControlState.RACE,
        )

        self.assertLessEqual(brake, 0.01)
        self.assertGreater(accel, 0.15)

    def test_on_line_transient_wobble_avoids_brake(self):
        accel, brake = calculate_raceline_pedals(
            speed=116.0,
            target_speed=145.0,
            steer=0.08,
            pursuit_error=0.08,
            track_position=0.24,
            lateral_speed=7.9,
            angle=0.06,
            previous_accel=None,
            previous_brake=None,
            control_state=RacingControlState.RACE,
        )

        self.assertEqual(brake, 0.0)
        self.assertGreater(accel, 0.0)
        self.assertLessEqual(accel, 0.36)

    def test_high_speed_raceline_steering_changes_like_agent_two(self):
        steer = calculate_raceline_pursuit_steer(
            speed=180.0,
            track_position=0.0,
            pursuit_target=0.45,
            heading_offset=0.0,
            curvature=0.0,
            lateral_speed=0.0,
            angle=0.0,
            previous_steer=0.0,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertGreater(steer, 0.0)
        self.assertLessEqual(abs(steer), 0.0341)

    def test_raceline_state_enters_edge_guard_before_run62_left_exit(self):
        state = choose_raceline_control_state(
            track_position=-0.37,
            local_target=-0.09,
            pursuit_target=-0.22,
            speed=107.0,
            lateral_speed=-0.5,
            angle=0.17,
            front_sensor=13.4,
            min_track_sensor=2.0,
        )

        self.assertEqual(state, RacingControlState.EDGE_GUARD)

    def test_raceline_confidence_debt_caps_speed_after_overshoot(self):
        debt = update_raceline_confidence_debt(
            previous_debt=0.0,
            previous_pursuit_error=0.44,
            pursuit_error=-0.27,
            lateral_speed=-2.6,
            angle=-0.05,
            track_position=0.23,
        )

        self.assertGreaterEqual(debt, 0.80)
        self.assertLessEqual(confidence_speed_cap(debt), 106.0)

    def test_slow_edge_guard_rejoin_does_not_snap_to_full_lock(self):
        steer = calculate_raceline_pursuit_steer(
            speed=70.0,
            track_position=-0.41,
            pursuit_target=0.58,
            heading_offset=0.0,
            curvature=0.011,
            lateral_speed=-0.1,
            angle=0.20,
            previous_steer=0.52,
            control_state=RacingControlState.EDGE_GUARD,
        )

        self.assertGreater(steer, 0.0)
        self.assertLessEqual(steer, 0.62)

    def test_slow_join_rejoin_does_not_snap_to_full_lock(self):
        steer = calculate_raceline_pursuit_steer(
            speed=49.0,
            track_position=-0.52,
            pursuit_target=0.59,
            heading_offset=0.0,
            curvature=0.011,
            lateral_speed=-0.1,
            angle=-0.29,
            previous_steer=0.62,
            control_state=RacingControlState.JOIN_RACELINE,
        )

        self.assertGreater(steer, 0.0)
        self.assertLessEqual(steer, 0.62)

    def test_edge_guard_crawl_assist_releases_on_track_stuck_brake(self):
        accel, brake = apply_edge_guard_crawl_assist(
            0.0,
            0.12,
            speed=0.2,
            target_speed=38.0,
            track_position=0.72,
            angle=-0.37,
            front_sensor=4.2,
            min_track_sensor=1.6,
            control_state=RacingControlState.EDGE_GUARD,
        )

        self.assertGreaterEqual(accel, 0.18)
        self.assertEqual(brake, 0.0)

    def test_normal_corner_sensor_does_not_cap_mapped_speed(self):
        track = [18.0] * 19
        track[7] = 92.0
        track[8] = 74.0
        track[9] = 10.0
        track[10] = 68.0
        track[11] = 88.0

        cap = map_safety_speed_cap(
            track,
            speed=126.0,
            previous_front=11.0,
            track_position=0.20,
            line_error=0.06,
        )

        self.assertEqual(cap, 216.0)

    def test_outward_line_join_sensor_cap_is_gradual(self):
        track = [60.0] * 19
        track[9] = 28.0

        cap = map_safety_speed_cap(
            track,
            speed=105.0,
            previous_front=34.0,
            track_position=0.10,
            line_error=-0.42,
            target_position=0.54,
            lateral_speed=-7.0,
            angle=-0.16,
        )

        self.assertLess(cap, 216.0)
        self.assertGreater(cap, 105.0)

    def test_edge_failure_sensor_caps_mapped_speed(self):
        track = [11.0] * 19
        track[9] = 6.0

        cap = map_safety_speed_cap(
            track,
            speed=80.0,
            previous_front=8.0,
            track_position=0.88,
            line_error=0.34,
        )

        self.assertLess(cap, 216.0)

    def test_guarded_target_biases_inward_progressively_near_wall(self):
        track = [22.0] * 19
        track[9] = 16.0

        guarded = guard_racing_line_target(
            track_position=0.22,
            target_position=0.74,
            speed=92.0,
            track_sensors=track,
            line_complexity=0.0,
            line_blend=1.0,
            lateral_speed=-7.0,
            angle=-0.18,
        )

        self.assertLess(guarded, 0.74)
        self.assertGreater(guarded, 0.20)

    def test_guarded_target_nudges_inward_without_centre_snap(self):
        track = [40.0] * 19
        track[9] = 28.0

        guarded = guard_racing_line_target(
            track_position=0.86,
            target_position=0.72,
            speed=84.0,
            track_sensors=track,
            line_complexity=0.0,
            line_blend=1.0,
            lateral_speed=0.2,
            angle=0.02,
        )

        self.assertGreater(guarded, 0.60)
        self.assertLess(guarded, 0.76)

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

    def test_attack_steering_commits_to_nearby_racing_line(self):
        steer = calculate_aggressive_steering(
            speed=130.0,
            track_pos=0.0,
            angle=0.0,
            speed_y=0.0,
            target_track_pos=0.12,
            heading_offset=0.0,
            curvature=0.0,
            previous_steer=0.0,
            mode=DrivingMode.ATTACK,
        )

        self.assertGreater(steer, 0.030)

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
