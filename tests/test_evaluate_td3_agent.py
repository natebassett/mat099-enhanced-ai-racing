from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


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
    def test_default_protocol_uses_four_balanced_trials_per_seed(self):
        with mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]):
            args = evaluate_td3_agent.parse_args()

        self.assertEqual(args.repeats, 5)
        self.assertEqual(args.trials_per_seed, 4)
        self.assertEqual(
            evaluate_td3_agent.EVALUATION_PROTOCOL,
            "agent6_order_stratified_robustness_v5",
        )

        with mock.patch.object(
            sys,
            "argv",
            [str(SCRIPT_PATH), "--trials-per-seed", "3"],
        ):
            with self.assertRaises(SystemExit):
                evaluate_td3_agent.parse_args()

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
        self.assertEqual(summary.raw_minimum_progress_m, 80.0)
        self.assertEqual(summary.median_off_track_steps, 3.0)
        self.assertEqual(summary.evaluation_seeds, (1, 2, 3))
        self.assertEqual(summary.trials_per_seed, 1)
        self.assertEqual(summary.total_trials, 3)

    def test_aggregate_reduces_within_order_position_before_combining(self):
        episodes = [
            _episode(1, seed=101, trial=1, order=1, progress=0.0, off_track=0),
            _episode(1, seed=101, trial=2, order=2, progress=101.0, off_track=0),
            _episode(1, seed=101, trial=3, order=1, progress=100.0, off_track=0),
            _episode(1, seed=101, trial=4, order=2, progress=102.0, off_track=0),
            _episode(2, seed=202, trial=1, order=2, progress=200.0, off_track=0),
            _episode(2, seed=202, trial=2, order=1, progress=180.0, off_track=0),
            _episode(2, seed=202, trial=3, order=2, progress=220.0, off_track=0),
            _episode(2, seed=202, trial=4, order=1, progress=200.0, off_track=0),
        ]

        summary = evaluate_td3_agent.aggregate_evaluations(
            Path("candidate.zip"),
            "g-track-3",
            episodes,
        )

        self.assertEqual(summary.seed_median_progress_m, (75.75, 200.0))
        self.assertEqual(summary.median_progress_m, 137.875)
        self.assertEqual(summary.minimum_progress_m, 75.75)
        self.assertEqual(summary.raw_minimum_progress_m, 0.0)
        self.assertEqual(summary.trials_per_seed, 4)
        self.assertEqual(summary.total_trials, 8)

    def test_lap_completion_requires_a_majority_within_the_seed(self):
        episodes = [
            _episode(1, seed=101, trial=1, order=1, progress=2800.0, off_track=0, laps=1),
            _episode(1, seed=101, trial=2, order=2, progress=1400.0, off_track=0, laps=0),
            _episode(1, seed=101, trial=3, order=1, progress=1400.0, off_track=0, laps=0),
            _episode(1, seed=101, trial=4, order=2, progress=1400.0, off_track=0, laps=0),
            _episode(2, seed=202, trial=1, order=1, progress=2800.0, off_track=0, laps=1),
            _episode(2, seed=202, trial=2, order=2, progress=2800.0, off_track=0, laps=1),
            _episode(2, seed=202, trial=3, order=1, progress=2800.0, off_track=0, laps=1),
            _episode(2, seed=202, trial=4, order=2, progress=1400.0, off_track=0, laps=0),
        ]

        summary = evaluate_td3_agent.aggregate_evaluations(
            Path("candidate.zip"),
            "g-track-3",
            episodes,
        )

        self.assertEqual(summary.completed_repeats, 1)
        self.assertEqual(summary.completed_laps, 1)

    def test_policy_schedule_rotates_order_across_trials_and_seeds(self):
        policies = [Path("candidate.zip"), Path("champion.zip")]

        self.assertEqual(
            evaluate_td3_agent.counterbalanced_policy_schedule(
                policies,
                seed_index=0,
                trial_index=0,
            ),
            (policies[0], policies[1]),
        )
        self.assertEqual(
            evaluate_td3_agent.counterbalanced_policy_schedule(
                policies,
                seed_index=0,
                trial_index=1,
            ),
            (policies[1], policies[0]),
        )
        self.assertEqual(
            evaluate_td3_agent.counterbalanced_policy_schedule(
                policies,
                seed_index=0,
                trial_index=2,
            ),
            (policies[0], policies[1]),
        )
        self.assertEqual(
            evaluate_td3_agent.counterbalanced_policy_schedule(
                policies,
                seed_index=0,
                trial_index=3,
            ),
            (policies[1], policies[0]),
        )
        self.assertEqual(
            evaluate_td3_agent.counterbalanced_policy_schedule(
                policies,
                seed_index=1,
                trial_index=0,
            ),
            (policies[1], policies[0]),
        )

    def test_identical_actors_are_grouped_without_comparing_critics(self):
        candidate = Path("candidate.zip").resolve()
        champion = Path("champion.zip").resolve()
        distinct = Path("distinct.zip").resolve()
        actor_states = {
            candidate: {"layer.weight": np.asarray([1.0, 2.0])},
            champion: {"layer.weight": np.asarray([1.0, 2.0])},
            distinct: {"layer.weight": np.asarray([1.0, 3.0])},
        }

        groups = evaluate_td3_agent.group_policy_actors(
            [candidate, champion, distinct],
            preferred_representative=champion,
            actor_state_loader=actor_states.__getitem__,
        )
        records = evaluate_td3_agent.policy_actor_equivalence_records(groups)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].representative_path, champion)
        self.assertEqual(groups[0].member_paths, (candidate, champion))
        candidate_record = next(
            record for record in records if record["policy_path"] == str(candidate)
        )
        self.assertEqual(candidate_record["classification"], "policy_equivalent")
        self.assertEqual(
            candidate_record["representative_policy_path"],
            str(champion),
        )

    def test_main_skips_torcs_when_all_requested_actors_are_equivalent(self):
        candidate = Path("candidate.zip").resolve()
        champion = Path("champion.zip").resolve()
        args = types.SimpleNamespace(
            policy_path=[candidate],
            checkpoint_dir=None,
            promote_best_if_improved=True,
            best_evaluation_model_path=champion,
            output_dir=Path("evaluation"),
        )
        actor_group = evaluate_td3_agent.PolicyActorGroup(
            representative_path=champion,
            member_paths=(candidate, champion),
        )

        with (
            mock.patch.object(evaluate_td3_agent, "parse_args", return_value=args),
            mock.patch.object(
                evaluate_td3_agent,
                "collect_policy_paths",
                return_value=[candidate],
            ),
            mock.patch.object(
                evaluate_td3_agent,
                "include_verified_baseline",
                return_value=[candidate, champion],
            ),
            mock.patch.object(
                evaluate_td3_agent,
                "group_policy_actors",
                return_value=[actor_group],
            ),
            mock.patch.object(
                evaluate_td3_agent,
                "write_evaluation_logs",
                return_value=(Path("result.json"), Path("result.csv")),
            ) as write_logs,
            mock.patch.object(evaluate_td3_agent, "run_evaluation") as run,
        ):
            self.assertEqual(evaluate_td3_agent.main(), 0)

        run.assert_not_called()
        write_logs.assert_called_once()

    def test_seed_suite_is_unique_and_reproducible(self):
        first = evaluate_td3_agent.build_evaluation_seeds(
            repeats=5,
            base_seed=1234,
            explicit_seeds=None,
        )
        second = evaluate_td3_agent.build_evaluation_seeds(
            repeats=5,
            base_seed=1234,
            explicit_seeds=None,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(
            all(
                evaluate_td3_agent.MIN_EVALUATION_SEED
                <= seed
                <= evaluate_td3_agent.MAX_EVALUATION_SEED
                for seed in first
            )
        )
        external_seeds = [
            evaluate_td3_agent.MIN_EVALUATION_SEED + offset
            for offset in (11, 22, 33)
        ]
        self.assertEqual(
            evaluate_td3_agent.build_evaluation_seeds(
                repeats=99,
                base_seed=0,
                explicit_seeds=external_seeds,
            ),
            tuple(external_seeds),
        )
        with self.assertRaises(ValueError):
            evaluate_td3_agent.build_evaluation_seeds(
                repeats=1,
                base_seed=0,
                explicit_seeds=[11],
            )

    def test_seeded_steering_disturbance_is_reproducible_and_steering_only(self):
        first = evaluate_td3_agent.SeededSteeringActionNoise(
            seed=42,
            steering_noise_std=0.2,
        )
        second = evaluate_td3_agent.SeededSteeringActionNoise(
            seed=42,
            steering_noise_std=0.2,
        )
        different = evaluate_td3_agent.SeededSteeringActionNoise(
            seed=43,
            steering_noise_std=0.2,
        )
        for agent in (first, second, different):
            agent.reset()

        first_actions = [first().tolist() for _ in range(20)]
        second_actions = [second().tolist() for _ in range(20)]
        different_actions = [different().tolist() for _ in range(20)]

        self.assertEqual(first_actions, second_actions)
        self.assertNotEqual(first_actions, different_actions)
        self.assertTrue(
            all(action[1:] == [0.0, 0.0] for action in first_actions)
        )
        first.reset()
        self.assertEqual(first_actions[0], first().tolist())

    def test_evaluation_quality_uses_worst_case_progress_after_median(self):
        common = {
            "completed_repeats": 0,
            "completed_laps": 0,
            "median_progress_m": 500.0,
            "median_off_track_steps": 2.0,
            "mean_total_score": 100.0,
        }
        stronger_worst_case = {**common, "minimum_progress_m": 450.0}
        weaker_worst_case = {**common, "minimum_progress_m": 300.0}

        self.assertGreater(
            evaluate_td3_agent.evaluation_quality(stronger_worst_case),
            evaluate_td3_agent.evaluation_quality(weaker_worst_case),
        )

    def test_promotion_guard_rejects_large_paired_seed_regression(self):
        reference = evaluate_td3_agent.aggregate_evaluations(
            Path("champion.zip"),
            "g-track-3",
            [
                _episode(1, progress=1400.0, off_track=0),
                _episode(2, progress=900.0, off_track=0),
                _episode(3, progress=500.0, off_track=0),
            ],
        )
        seed_regression = evaluate_td3_agent.aggregate_evaluations(
            Path("candidate.zip"),
            "g-track-3",
            [
                _episode(1, progress=1200.0, off_track=0),
                _episode(2, progress=1500.0, off_track=0),
                _episode(3, progress=600.0, off_track=0),
            ],
        )
        uniform_improvement = evaluate_td3_agent.aggregate_evaluations(
            Path("candidate.zip"),
            "g-track-3",
            [
                _episode(1, progress=1420.0, off_track=0),
                _episode(2, progress=920.0, off_track=0),
                _episode(3, progress=520.0, off_track=0),
            ],
        )

        self.assertGreater(
            seed_regression.median_progress_m,
            reference.median_progress_m,
        )
        self.assertGreater(
            seed_regression.minimum_progress_m,
            reference.minimum_progress_m,
        )
        self.assertFalse(
            evaluate_td3_agent.evaluation_candidate_is_noninferior(
                seed_regression,
                reference,
            )
        )
        self.assertTrue(
            evaluate_td3_agent.evaluation_candidate_is_noninferior(
                uniform_improvement,
                reference,
            )
        )

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

    def test_protocol_migration_requires_evaluated_baseline_authorisation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.zip"
            best = root / "best_evaluation.zip"
            candidate.write_bytes(b"candidate-policy")
            best.write_bytes(b"protected-policy")
            evaluate_td3_agent.metadata_path_for_policy(best).write_text(
                json.dumps(
                    {
                        "best_evaluation": {
                            "deterministic": True,
                            "repeats": 3,
                            "completed_repeats": 0,
                            "completed_laps": 0,
                            "median_progress_m": 1000.0,
                            "minimum_progress_m": 1000.0,
                            "median_off_track_steps": 1.0,
                            "mean_total_score": 100.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            episodes = [
                _episode(1, progress=800.0, off_track=2),
                _episode(2, progress=700.0, off_track=2),
                _episode(3, progress=600.0, off_track=2),
            ]
            summary = evaluate_td3_agent.aggregate_evaluations(
                candidate,
                "g-track-3",
                episodes,
            )

            self.assertFalse(
                evaluate_td3_agent.promote_best_evaluation(
                    candidate,
                    best,
                    summary,
                    episodes,
                )
            )
            self.assertEqual(best.read_bytes(), b"protected-policy")
            self.assertTrue(
                evaluate_td3_agent.promote_best_evaluation(
                    candidate,
                    best,
                    summary,
                    episodes,
                    allow_protocol_migration=True,
                )
            )
            self.assertEqual(best.read_bytes(), b"candidate-policy")

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

    def test_verified_baseline_is_added_once_for_fair_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.zip"
            baseline = root / "baseline.zip"
            candidate.touch()
            baseline.touch()

            policies = evaluate_td3_agent.include_verified_baseline(
                [candidate.resolve()],
                baseline,
            )
            deduplicated = evaluate_td3_agent.include_verified_baseline(
                policies,
                baseline,
            )

            self.assertEqual(policies, [candidate.resolve(), baseline.resolve()])
            self.assertEqual(deduplicated, policies)


def _episode(
    repeat: int,
    *,
    seed: int | None = None,
    trial: int = 1,
    order: int = 1,
    progress: float,
    off_track: int,
    laps: int = 0,
) -> evaluate_td3_agent.EvaluationEpisode:
    return evaluate_td3_agent.EvaluationEpisode(
        repeat=repeat,
        seed=repeat if seed is None else seed,
        trial=trial,
        order_position=order,
        reason="out of bounds",
        steps=500,
        laps=laps,
        best_lap_seconds=None,
        progress_m=progress,
        total_score=100.0,
        max_speed_kmh=90.0,
        average_speed_kmh=45.0,
        off_track_steps=off_track,
    )

if __name__ == "__main__":
    unittest.main()
