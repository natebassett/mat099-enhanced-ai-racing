import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plot_raceline_tracking import render_svg  # noqa: E402
from replay_map_aware_telemetry import replay, reconstructed_track_sensors  # noqa: E402


class MapAwareToolTests(unittest.TestCase):
    def test_reconstructs_front_sensor_for_old_telemetry(self):
        row = {"sensorFront": "12.5", "sensorBestAngle": "4"}
        sensors = reconstructed_track_sensors(row)

        self.assertEqual(len(sensors), 19)
        self.assertEqual(sensors[9], 12.5)
        self.assertGreater(sensors[14], sensors[9])

    def test_plot_renders_available_tracking_series(self):
        rows = [
            {
                "distFromStart": "10",
                "trackPos": "0.0",
                "rawRacingLinePos": "0.5",
                "practicalTrackPos": "0.45",
                "targetTrackPos": "0.4",
            },
            {
                "distFromStart": "20",
                "trackPos": "0.1",
                "rawRacingLinePos": "0.55",
                "practicalTrackPos": "0.46",
                "targetTrackPos": "0.42",
            },
        ]

        svg = render_svg(rows)

        self.assertIn("<svg", svg)
        self.assertIn("actual", svg)
        self.assertIn("target", svg)

    def test_replay_writes_controller_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "output.csv"
            rows = [
                {
                    "step": "0",
                    "distFromStart": "2743.1",
                    "speedX": "0.0",
                    "speedY": "0.0",
                    "angle": "0.0",
                    "trackPos": "0.0",
                    "sensorFront": "160.0",
                    "sensorBestAngle": "0",
                    "curLapTime": "0.0",
                    "lastLapTime": "0.0",
                },
                {
                    "step": "1",
                    "distFromStart": "2758.0",
                    "speedX": "70.0",
                    "speedY": "-1.0",
                    "angle": "-0.05",
                    "trackPos": "0.05",
                    "sensorFront": "90.0",
                    "sensorBestAngle": "4",
                    "curLapTime": "0.5",
                    "lastLapTime": "0.0",
                },
            ]
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = replay(input_path, output_path)

            self.assertEqual(summary["rows"], 2)
            self.assertTrue(output_path.is_file())
            with output_path.open(newline="", encoding="utf-8") as handle:
                replay_rows = list(csv.DictReader(handle))
            self.assertIn("controlState", replay_rows[0])
            self.assertIn("targetSpeed", replay_rows[0])


if __name__ == "__main__":
    unittest.main()
