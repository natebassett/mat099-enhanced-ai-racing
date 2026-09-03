from __future__ import annotations

import sqlite3
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

    def test_prune_keeps_one_useful_run_per_agent_type_and_cascades(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "results.db"
            repository = RaceRepository(database_path)
            td3_v1 = repository.register_agent(
                name="Sensor TD3 v1",
                agent_type="sensor_n_step_td3",
                version="1.0",
            )
            td3_v2 = repository.register_agent(
                name="Sensor TD3 v2",
                agent_type="sensor_n_step_td3",
                version="2.0",
            )
            map_agent = repository.register_agent(
                name="Map-Aware",
                agent_type="map_aware",
                version="4.2",
            )

            old_td3 = repository.record_run(
                agent_id=td3_v1,
                track="g-track-3",
                seed=1,
                results=_results(best_lap=96.0),
                metrics={"distance": 2_500.0},
            )
            best_td3 = repository.record_run(
                agent_id=td3_v2,
                track="g-track-3",
                seed=2,
                results=_results(best_lap=88.0),
            )
            other_track_td3 = repository.record_run(
                agent_id=td3_v2,
                track="road-1",
                seed=3,
                results=_results(best_lap=70.0),
            )
            map_run = repository.record_run(
                agent_id=map_agent,
                track="g-track-3",
                seed=4,
                results=_results(best_lap=91.0),
            )
            repository.record_run_telemetry(
                old_td3,
                [{"step": 1, "speed_x": 10.0}],
            )

            result = repository.prune_runs_to_representatives(
                preferred_track="g-track-3"
            )

            self.assertEqual(result.original_count, 4)
            self.assertEqual(result.deleted_count, 2)
            self.assertEqual(repository.count_runs(), 2)
            self.assertEqual(
                result.retained_run_ids,
                tuple(sorted((best_td3, map_run))),
            )
            connection = sqlite3.connect(database_path)
            try:
                run_ids = {
                    row[0] for row in connection.execute("SELECT id FROM race_runs")
                }
                telemetry_count = connection.execute(
                    "SELECT COUNT(*) FROM run_telemetry WHERE run_id = ?",
                    (old_td3,),
                ).fetchone()[0]
                metric_count = connection.execute(
                    "SELECT COUNT(*) FROM run_metrics WHERE run_id = ?",
                    (old_td3,),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(run_ids, {best_td3, map_run})
            self.assertEqual(telemetry_count, 0)
            self.assertEqual(metric_count, 0)
            self.assertNotIn(other_track_td3, run_ids)


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
