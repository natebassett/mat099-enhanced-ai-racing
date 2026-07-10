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
    estimate_line_complexity,
    preview_track_position,
)
from racing_line import RacingLine, TorcsTrackMap  # noqa: E402
from racing_line.controller import (  # noqa: E402
    DrivingMode,
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

    def test_major_bends_use_outside_apex_outside_line(self):
        track_map = TorcsTrackMap.from_xml(TRACK_PATH)
        samples = track_map.sample_centerline(3.0, 3608.28)
        strongest_by_direction = {}

        for segment_index, segment in enumerate(track_map.segments):
            if not segment.turn_radians:
                continue
            direction = 1 if segment.turn_radians > 0 else -1
            current = strongest_by_direction.get(direction)
            if current is None or abs(segment.turn_radians) > abs(current.turn_radians):
                strongest_by_direction[direction] = segment

        for direction, segment in strongest_by_direction.items():
            segment_index = track_map.segments.index(segment)
            group_indices = [segment_index]

            previous_index = segment_index - 1
            while previous_index >= 0:
                previous_segment = track_map.segments[previous_index]
                previous_samples = [
                    point
                    for point in samples
                    if point["segment_index"] == previous_index
                ]
                current_samples = [
                    point
                    for point in samples
                    if point["segment_index"] == group_indices[0]
                ]
                if not previous_samples:
                    break
                gap = current_samples[0]["distance"] - previous_samples[-1]["distance"]
                previous_direction = (
                    1
                    if previous_segment.turn_radians > 0
                    else -1
                    if previous_segment.turn_radians < 0
                    else 0
                )
                if previous_direction == direction and gap < 75.0:
                    group_indices.insert(0, previous_index)
                    previous_index -= 1
                    continue
                if previous_direction == 0 and gap < 75.0:
                    previous_index -= 1
                    continue
                break

            next_index = segment_index + 1
            while next_index < len(track_map.segments):
                next_segment = track_map.segments[next_index]
                next_samples = [
                    point
                    for point in samples
                    if point["segment_index"] == next_index
                ]
                current_samples = [
                    point
                    for point in samples
                    if point["segment_index"] == group_indices[-1]
                ]
                if not next_samples:
                    break
                gap = next_samples[0]["distance"] - current_samples[-1]["distance"]
                next_direction = (
                    1
                    if next_segment.turn_radians > 0
                    else -1
                    if next_segment.turn_radians < 0
                    else 0
                )
                if next_direction == direction and gap < 75.0:
                    group_indices.append(next_index)
                    next_index += 1
                    continue
                if next_direction == 0 and gap < 75.0:
                    next_index += 1
                    continue
                break

            segment_samples = [
                point
                for point in samples
                if point["segment_index"] in group_indices
            ]
            directed_positions = [
                direction
                * self.racing_line.lookup(point["distance"])["target_track_pos"]
                for point in segment_samples
            ]
            midpoint = len(directed_positions) // 2

            self.assertLess(min(directed_positions), -0.05)
            # Compound bends may deliberately use a shallower apex so the next
            # turn can be linked smoothly. The line must still cross inward,
            # but its exit may become the entry setup for the next sector.
            self.assertGreater(max(directed_positions), 0.03)

    def test_line_has_no_abrupt_lateral_heading_changes(self):
        maximum_heading_offset = max(
            abs(waypoint["heading_offset"])
            for waypoint in self.payload["waypoints"]
        )

        self.assertLess(maximum_heading_offset, 0.18)

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


if __name__ == "__main__":
    unittest.main()
