import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from runner.lap_tracker import LapTracker  # noqa: E402
from storage import RaceRepository  # noqa: E402


class LapTrackerTests(unittest.TestCase):
    def test_records_each_completed_lap_once(self):
        tracker = LapTracker()

        self.assertIsNone(tracker.update({"lastLapTime": 0.0}))
        self.assertEqual(tracker.update({"lastLapTime": 91.25}), 91.25)
        self.assertIsNone(tracker.update({"lastLapTime": 91.25}))
        self.assertEqual(tracker.update({"lastLapTime": 89.75}), 89.75)
        self.assertEqual(tracker.laps_completed, 2)
        self.assertEqual(tracker.best_lap_time, 89.75)
        self.assertEqual(tracker.average_lap_time, 90.5)

    def test_ignores_a_stale_initial_lap_time(self):
        tracker = LapTracker(initial_last_lap_time=92.0)

        self.assertIsNone(tracker.update({"lastLapTime": 92.0}))
        self.assertEqual(tracker.laps_completed, 0)


class RaceRepositoryLapTimingTests(unittest.TestCase):
    def test_migrates_and_records_lap_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "results.db"
            repository = RaceRepository(database_path)
            agent_id = repository.register_agent(
                name="Test Agent",
                agent_type="test",
                version="1.0",
            )
            run_id = repository.record_run(
                agent_id=agent_id,
                track="test-track",
                seed=None,
                results={
                    "started_at": "2026-07-08T00:00:00+00:00",
                    "total_score": 1.0,
                    "steps": 100,
                    "max_speed": 2.0,
                    "avg_speed": 1.0,
                    "off_track": 0,
                    "laps_completed": 2,
                    "best_lap_time_seconds": 89.75,
                    "average_lap_time_seconds": 90.5,
                    "duration_seconds": 181.0,
                    "termination_reason": "target_laps_completed",
                },
            )

            connection = sqlite3.connect(database_path)
            row = connection.execute(
                """
                SELECT laps_completed, best_lap_time_seconds,
                       average_lap_time_seconds
                FROM race_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            connection.close()

            self.assertEqual(row, (2, 89.75, 90.5))


if __name__ == "__main__":
    unittest.main()
