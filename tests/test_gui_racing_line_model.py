from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.racing_line_model import (  # noqa: E402
    build_racing_line_visual_profile,
    load_racing_line_profile,
    phase_family,
    point_at_distance,
    racing_line_point_text,
    racing_line_summary_lines,
    track_position_label,
)


class GuiRacingLineModelTests(unittest.TestCase):
    def test_loads_profile_from_racing_line_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")

            profile = load_racing_line_profile(path)

            self.assertEqual(profile.track_name, "demo-track")
            self.assertEqual(profile.track_length_m, 100.0)
            self.assertEqual(profile.track_width_m, 10.0)
            self.assertEqual(len(profile.points), 3)
            self.assertEqual(profile.max_speed_kmh, 120.0)

    def test_summary_and_current_point_are_plain_english(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            profile = load_racing_line_profile(path)

            summary = racing_line_summary_lines(profile)
            point = point_at_distance(profile, 52.0)
            point_text = racing_line_point_text(point)

            self.assertIn("Waypoints: 3", summary[2])
            self.assertIn("Dominant phases", summary[4])
            self.assertIn("Target speed: 80.0 km/h", point_text)
            self.assertIn("Phase: Brake Left Turn", point_text)

    def test_point_at_lap_end_uses_final_waypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            profile = load_racing_line_profile(path)

            point = point_at_distance(profile, profile.track_length_m)

            self.assertIsNotNone(point)
            assert point is not None
            self.assertEqual(point.distance, 99.0)

    def test_phase_and_track_position_labels(self):
        self.assertEqual(phase_family("brake_left_turn"), "Brake")
        self.assertEqual(phase_family("full_throttle"), "Full throttle")
        self.assertEqual(track_position_label(0.75), "wide right")
        self.assertEqual(track_position_label(-0.05), "near centre")

    def test_builds_visual_profile_from_real_track_geometry(self):
        profile = load_racing_line_profile(
            PROJECT_ROOT / "data" / "racing_lines" / "g-track-3.json"
        )

        visual = build_racing_line_visual_profile(
            profile,
            PROJECT_ROOT / "torcs" / "tracks" / "road" / "g-track-3" / "g-track-3.xml",
        )

        self.assertGreater(len(visual.centerline), 100)
        self.assertEqual(len(visual.line_points), len(profile.points))
        self.assertEqual(visual.line_points[0].source.distance, profile.points[0].distance)
        min_x, min_y, max_x, max_y = visual.bounds
        self.assertGreater(max_x - min_x, 100.0)
        self.assertGreater(max_y - min_y, 100.0)


def _payload() -> dict[str, object]:
    return {
        "track_name": "demo-track",
        "track_length_m": 100.0,
        "track_width_m": 10.0,
        "source_sha256": "abc123",
        "optimizer": {"message": "test racing line"},
        "waypoints": [
            {
                "distance": 0.0,
                "target_track_pos": 0.1,
                "target_speed_kmh": 120.0,
                "curvature": 0.0,
                "phase": "accelerate",
            },
            {
                "distance": 50.0,
                "target_track_pos": -0.4,
                "target_speed_kmh": 80.0,
                "curvature": 0.05,
                "phase": "brake_left_turn",
                "reference_section": "T1",
            },
            {
                "distance": 99.0,
                "target_track_pos": 0.2,
                "target_speed_kmh": 100.0,
                "curvature": -0.02,
                "phase": "full_throttle",
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
