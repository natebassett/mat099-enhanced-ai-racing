from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_td3_agent.py"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.td3_agent import evaluation_checkpoint_is_verified  # noqa: E402


spec = importlib.util.spec_from_file_location("evaluate_td3_agent", SCRIPT_PATH)
evaluate_td3_agent = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["evaluate_td3_agent"] = evaluate_td3_agent
spec.loader.exec_module(evaluate_td3_agent)


class Td3EvaluationTests(unittest.TestCase):
    def test_episode_progress_uses_furthest_forward_distance(self):
        results = {
            "telemetry_samples": [
                {"dist_raced": 50.0},
                {"dist_raced": 90.0},
                {"dist_raced": 82.0},
            ]
        }

        self.assertEqual(evaluate_td3_agent.episode_progress_m(results), 40.0)

    def test_aggregate_uses_median_progress_across_repeats(self):
        episodes = [
            _episode(1, progress=80.0, off_track=4),
            _episode(2, progress=140.0, off_track=2),
            _episode(3, progress=100.0, off_track=3),
        ]

        summary = evaluate_td3_agent.aggregate_evaluations(
            Path("candidate.zip"),
            "g-track-3",
            episodes,
        )

        self.assertTrue(summary.deterministic)
        self.assertEqual(summary.median_progress_m, 100.0)
        self.assertEqual(summary.minimum_progress_m, 80.0)
        self.assertEqual(summary.maximum_progress_m, 140.0)
        self.assertEqual(summary.median_off_track_steps, 3.0)

    def test_promoted_evaluation_checkpoint_is_verified_and_not_downgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.zip"
            best = root / "best_evaluation.zip"
            candidate.write_bytes(b"candidate-policy")
            episodes = [
                _episode(1, progress=120.0, off_track=2),
                _episode(2, progress=100.0, off_track=4),
                _episode(3, progress=110.0, off_track=3),
            ]
            summary = evaluate_td3_agent.aggregate_evaluations(
                candidate,
                "g-track-3",
                episodes,
            )

            promoted = evaluate_td3_agent.promote_best_evaluation(
                candidate,
                best,
                summary,
                episodes,
            )

            self.assertTrue(promoted)
            self.assertEqual(best.read_bytes(), b"candidate-policy")
            self.assertTrue(evaluation_checkpoint_is_verified(best))

            weaker = evaluate_td3_agent.aggregate_evaluations(
                candidate,
                "g-track-3",
                [
                    _episode(1, progress=90.0, off_track=1),
                    _episode(2, progress=80.0, off_track=1),
                    _episode(3, progress=70.0, off_track=1),
                ],
            )
            self.assertFalse(
                evaluate_td3_agent.promote_best_evaluation(
                    candidate,
                    best,
                    weaker,
                    episodes,
                )
            )

    def test_checkpoint_collection_is_numeric_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_100 = root / "agent6_td3_scratch_100_steps.zip"
            checkpoint_25 = root / "agent6_td3_scratch_25_steps.zip"
            checkpoint_100.touch()
            checkpoint_25.touch()

            policies = evaluate_td3_agent.collect_policy_paths(
                [checkpoint_100],
                root,
            )

            self.assertEqual(policies, [checkpoint_100.resolve(), checkpoint_25.resolve()])


def _episode(
    repeat: int,
    *,
    progress: float,
    off_track: int,
) -> evaluate_td3_agent.EvaluationEpisode:
    return evaluate_td3_agent.EvaluationEpisode(
        repeat=repeat,
        reason="out of bounds",
        steps=500,
        laps=0,
        best_lap_seconds=None,
        progress_m=progress,
        total_score=100.0,
        max_speed_kmh=90.0,
        average_speed_kmh=45.0,
        off_track_steps=off_track,
    )


if __name__ == "__main__":
    unittest.main()
