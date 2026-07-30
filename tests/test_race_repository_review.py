from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from storage import RaceRepository  # noqa: E402


class RaceRepositoryReviewTests(unittest.TestCase):
    def test_loads_run_telemetry_with_persisted_track_sensors(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = RaceRepository(Path(directory) / "results.db")
            agent_id = repository.register_agent(
                name="Map-Aware Racing-Line Agent",
                agent_type="map_aware",
                version="4.2",
            )
            run_id = repository.record_run(
                agent_id=agent_id,
                track="g-track-3",
                seed=None,
                results=_results(best_lap=92.5),
            )

            repository.record_run_telemetry(
                run_id,
                [
                    {
                        "step": 1,
                        "speed_x": 100.0,
                        "accel": 0.4,
                        "brake": 0.0,
                        "track_sensors": [str(value) for value in range(19)],
                        "cur_lap_time": 12.5,
                    }
                ],
            )

            run = repository.get_run(run_id)
            samples = repository.list_run_telemetry(run_id)

            self.assertIsNotNone(run)
            assert run is not None
            self.assertEqual(run["agent_type"], "map_aware")
            self.assertEqual(samples[0]["track_sensors"], [float(value) for value in range(19)])

    def test_finds_best_comparable_run_for_track_and_agent_type(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = RaceRepository(Path(directory) / "results.db")
            agent_id = repository.register_agent(
                name="Map-Aware Racing-Line Agent",
                agent_type="map_aware",
                version="4.2",
            )
            slower_run_id = repository.record_run(
                agent_id=agent_id,
                track="g-track-3",
                seed=None,
                results=_results(best_lap=94.0),
            )
            faster_run_id = repository.record_run(
                agent_id=agent_id,
                track="g-track-3",
                seed=None,
                results=_results(best_lap=91.0),
            )

            best_run = repository.best_run_for_track(
                "g-track-3",
                agent_type="map_aware",
            )

            self.assertNotEqual(slower_run_id, faster_run_id)
            self.assertIsNotNone(best_run)
            assert best_run is not None
            self.assertEqual(best_run["id"], faster_run_id)


def _results(best_lap: float) -> dict[str, object]:
    return {
        "started_at": "2026-07-30T00:00:00+00:00",
        "total_score": 1.0,
        "steps": 100,
        "max_speed": 4.0,
        "avg_speed": 2.0,
        "off_track": 0,
        "laps_completed": 1,
        "best_lap_time_seconds": best_lap,
        "average_lap_time_seconds": best_lap,
        "duration_seconds": best_lap + 2.0,
        "termination_reason": "target_laps_completed",
    }


if __name__ == "__main__":
    unittest.main()
