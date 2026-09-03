from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.results_model import (  # noqa: E402
    EvaluationBatch,
    EvaluationEpisode,
    featured_evaluation,
    load_results_dataset,
)
from gui.results_view import ResultsView  # noqa: E402


class ResultsWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_loads_evaluation_training_and_saved_race_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture_project(root)

            dataset = load_results_dataset(root)

            self.assertEqual(len(dataset.evaluation_batches), 1)
            batch = dataset.evaluation_batches[0]
            self.assertEqual(batch.agent_family, "agent8")
            self.assertEqual(batch.trials, 2)
            self.assertEqual(batch.completed_laps, 1)
            self.assertEqual(batch.completion_rate, 0.5)
            self.assertEqual(batch.median_lap_time_seconds, 84.25)
            self.assertEqual(batch.minimum_distance_m, 2_100.0)

            self.assertEqual(len(dataset.training_runs), 1)
            training = dataset.training_runs[0]
            self.assertEqual(len(training.episodes), 2)
            self.assertEqual(training.total_interactions, 7_000)
            self.assertEqual(training.completed_laps, 1)
            self.assertEqual(training.fastest_lap_time_seconds, 87.5)

            self.assertEqual(len(dataset.race_summaries), 1)
            race = dataset.race_summaries[0]
            self.assertEqual(race.runs, 2)
            self.assertEqual(race.completed_runs, 1)
            self.assertEqual(race.completion_rate, 0.5)
            self.assertEqual(race.best_lap_time_seconds, 91.2)

    def test_results_view_uses_explicit_batch_and_run_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture_project(root)

            view = ResultsView(root)

            self.assertEqual(view.evaluation_batch_combo.count(), 1)
            self.assertEqual(view.training_run_combo.count(), 1)
            self.assertEqual(view.content_tabs.count(), 3)
            self.assertEqual(view.content_tabs.tabText(0), "Overview")
            self.assertEqual(view.content_tabs.tabText(1), "Learning Journey")
            self.assertEqual(view.overview_metrics["completion"].text(), "1 / 2")
            self.assertEqual(view.evaluation_metrics["completion"].text(), "50%")
            self.assertEqual(view.training_metrics["completed"].text(), "1")
            self.assertEqual(view.race_table.rowCount(), 1)
            self.assertIn("evaluation", view.evaluation_source_label.text())
            self.assertIsNotNone(view.technical_evidence_page)
            self.assertTrue(view.technical_evidence_page.isHidden())
            view.close()

    def test_loads_each_policy_from_agent6_multi_policy_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_dir = root / "data" / "evaluation" / "agent6_td3"
            evaluation_dir.mkdir(parents=True)
            policy_a = str(root / "models" / "candidate.zip")
            policy_b = str(root / "models" / "champion.zip")
            (evaluation_dir / "td3_evaluation_20260827-140701.json").write_text(
                json.dumps(
                    {
                        "episodes": {
                            policy_a: [
                                {"trial": 1, "progress_m": 500, "laps": 0},
                            ],
                            policy_b: [
                                {
                                    "trial": 1,
                                    "progress_m": 2943,
                                    "laps": 1,
                                    "best_lap_seconds": 110.0,
                                }
                            ],
                        },
                        "summaries": [
                            {"policy_path": policy_a, "track": "g-track-3"},
                            {"policy_path": policy_b, "track": "g-track-3"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = load_results_dataset(root)

            self.assertEqual(len(dataset.evaluation_batches), 2)
            self.assertTrue(
                all(batch.agent_family == "agent6" for batch in dataset.evaluation_batches)
            )
            self.assertEqual(
                sorted(batch.completion_rate for batch in dataset.evaluation_batches),
                [0.0, 1.0],
            )

    def test_overview_features_fastest_reliable_supported_policy(self) -> None:
        agent7 = _evaluation_batch("agent7", completed=10, trials=10, lap_time=104.0)
        agent8 = _evaluation_batch("agent8", completed=19, trials=20, lap_time=83.0)
        one_lucky_trial = _evaluation_batch(
            "other",
            completed=1,
            trials=1,
            lap_time=70.0,
        )

        featured = featured_evaluation((agent7, agent8, one_lucky_trial))

        self.assertIsNotNone(featured)
        self.assertEqual(featured.agent_family, "agent8")

    def test_main_window_exposes_results_in_primary_navigation(self) -> None:
        from gui.main_window import MainWindow

        window = MainWindow()

        self.assertEqual(window.tabs.tabText(window.results_tab_index), "Results")
        window.tabs.setCurrentIndex(window.results_tab_index)
        self.assertIs(window.results_view, window.tabs.currentWidget())
        window.close()


def _write_fixture_project(root: Path) -> None:
    evaluation_dir = root / "data" / "evaluation" / "agent8_sensor_n_step_td3"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "n_step_td3_evaluation_20260902-120000.json").write_text(
        json.dumps(
            {
                "policy_path": str(root / "models" / "agent8_champion.pt"),
                "track": "g-track-3",
                "model_family": "agent8_sensor_n_step_td3_racer",
                "observation_version": "sensor_v1",
                "action_version": "action_v1",
                "reward_version": "reward_v1",
                "deterministic": True,
                "episodes": [
                    {
                        "repeat": 1,
                        "distance_m": 2_943.2,
                        "laps_completed": 1,
                        "best_lap_time_seconds": 84.25,
                        "termination_reason": "lap_completed",
                    },
                    {
                        "repeat": 2,
                        "distance_m": 2_100.0,
                        "laps_completed": 0,
                        "best_lap_time_seconds": None,
                        "termination_reason": "off_track",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    training_dir = (
        root
        / "models"
        / "training_runs"
        / "agent8_sensor_n_step_td3"
        / "20260902-121000"
    )
    training_dir.mkdir(parents=True)
    (training_dir / "config.json").write_text(
        json.dumps(
            {
                "run_started": "20260902-121000",
                "track": "g-track-3",
                "run_seed": 8,
                "requested_training_timesteps": 10_000,
                "observation_version": "sensor_v1",
                "action_version": "action_v1",
                "reward_version": "reward_v1",
            }
        ),
        encoding="utf-8",
    )
    (training_dir / "episodes.csv").write_text(
        "mode,global_step,episode,steps,reward,distance_m,laps_completed,"
        "best_lap_time_seconds,termination_reason\n"
        "replay_warmup,0,1,3000,100,2100,0,,off_track\n"
        "training,4000,2,4000,200,2943.2,1,87.5,lap_completed\n"
        "evaluation,5000,3,4000,220,2943.2,1,86.0,lap_completed\n",
        encoding="utf-8",
    )

    (root / "latest_race_runs.csv").write_text(
        "id,started_at,agent_name,track,best_lap_time_seconds,avg_speed,"
        "off_track_count,termination_reason\n"
        "1,2026-09-02T10:00:00,Map-Aware Racing-Line Agent,g-track-3,"
        "91.2,118.0,0,lap_completed\n"
        "2,2026-09-02T11:00:00,Map-Aware Racing-Line Agent,g-track-3,,"
        "70.0,1,off_track\n",
        encoding="utf-8",
    )


def _evaluation_batch(
    family: str,
    *,
    completed: int,
    trials: int,
    lap_time: float,
) -> EvaluationBatch:
    episodes = tuple(
        EvaluationEpisode(
            trial=index + 1,
            distance_m=2_943.0 if index < completed else 1_500.0,
            lap_time_seconds=lap_time if index < completed else None,
            completed=index < completed,
            reward=None,
            max_speed_kmh=None,
            average_speed_kmh=None,
            termination_reason="lap_completed" if index < completed else "off_track",
        )
        for index in range(trials)
    )
    return EvaluationBatch(
        batch_id=family,
        agent_family=family,
        agent_label=family,
        policy_path=f"{family}.pt",
        policy_label=family,
        track="g-track-3",
        timestamp="20260902-120000",
        source_path=Path(f"{family}.json"),
        deterministic=True,
        episodes=episodes,
        observation_version="v1",
        action_version="v1",
        reward_version="v1",
        protocol="test",
    )


if __name__ == "__main__":
    unittest.main()
