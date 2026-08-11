from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_dyna_q_policy.py"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

fake_gym_torcs = types.ModuleType("gym_torcs")


class FakeTorcsEnv:
    pass


fake_gym_torcs.TorcsEnv = FakeTorcsEnv
sys.modules.setdefault("gym_torcs", fake_gym_torcs)

spec = importlib.util.spec_from_file_location("train_dyna_q_policy", SCRIPT_PATH)
train_dyna_q_policy = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["train_dyna_q_policy"] = train_dyna_q_policy
spec.loader.exec_module(train_dyna_q_policy)


class TrainDynaQPolicyTests(unittest.TestCase):
    def test_episode_progress_uses_dist_raced_delta(self):
        progress = train_dyna_q_policy._episode_progress_m(
            {
                "telemetry_samples": [
                    {"dist_raced": 10.0},
                    {"dist_raced": 40.0},
                    {"dist_raced": 95.0},
                ]
            },
            track_length_m=1000.0,
        )

        self.assertEqual(progress, 85.0)

    def test_episode_progress_wraps_dist_from_start(self):
        progress = train_dyna_q_policy._episode_progress_m(
            {
                "telemetry_samples": [
                    {"dist_from_start": 950.0},
                    {"dist_from_start": 980.0},
                    {"dist_from_start": 20.0},
                    {"dist_from_start": 80.0},
                ]
            },
            track_length_m=1000.0,
        )

        self.assertEqual(progress, 130.0)

    def test_results_progress_uses_explicit_evaluation_progress(self):
        progress = train_dyna_q_policy._results_progress_m(
            {
                "evaluation_progress_m": 450.0,
                "telemetry_samples": [
                    {"dist_raced": 0.0},
                    {"dist_raced": 900.0},
                ],
            },
            track_length_m=1000.0,
        )

        self.assertEqual(progress, 450.0)

    def test_repeated_evaluation_aggregate_records_partial_completion(self):
        aggregate = train_dyna_q_policy._aggregate_evaluation_results(
            [
                {
                    "termination_reason": "target_laps_completed",
                    "steps": 1000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 330.0,
                    "off_track": 100,
                    "avg_speed": 0.70,
                    "progress_m": 1000.0,
                },
                {
                    "termination_reason": "out of bounds",
                    "steps": 500,
                    "laps_completed": 0,
                    "best_lap_time_seconds": None,
                    "off_track": 200,
                    "avg_speed": 0.55,
                    "progress_m": 620.0,
                },
            ],
            track_length_m=1000.0,
        )

        self.assertEqual(aggregate["laps_completed"], 1)
        self.assertEqual(aggregate["evaluation_repeats"], 2)
        self.assertEqual(aggregate["evaluation_completed_repeats"], 1)
        self.assertEqual(aggregate["evaluation_min_progress_m"], 620.0)
        self.assertEqual(aggregate["evaluation_max_progress_m"], 1000.0)
        self.assertEqual(aggregate["evaluation_progress_m"], 810.0)
        self.assertEqual(aggregate["best_lap_time_seconds"], 330.0)

    def test_repeated_evaluation_aggregate_uses_stable_medians(self):
        aggregate = train_dyna_q_policy._aggregate_evaluation_results(
            [
                {
                    "termination_reason": "target_laps_completed",
                    "steps": 1000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 340.0,
                    "off_track": 300,
                    "avg_speed": 0.60,
                    "progress_m": 1000.0,
                },
                {
                    "termination_reason": "target_laps_completed",
                    "steps": 900,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 320.0,
                    "off_track": 100,
                    "avg_speed": 0.70,
                    "progress_m": 1000.0,
                },
                {
                    "termination_reason": "target_laps_completed",
                    "steps": 950,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 330.0,
                    "off_track": 200,
                    "avg_speed": 0.65,
                    "progress_m": 1000.0,
                },
            ],
            track_length_m=1000.0,
        )

        self.assertEqual(aggregate["laps_completed"], 1)
        self.assertEqual(aggregate["best_lap_time_seconds"], 330.0)
        self.assertEqual(aggregate["off_track"], 200)
        self.assertEqual(aggregate["steps"], 950)
        self.assertEqual(aggregate["avg_speed"], 0.65)

    def test_dashboard_shows_progress_and_hides_old_rows(self):
        summaries = [
            train_dyna_q_policy.EpisodeSummary(
                episode=index,
                reason="stuck",
                steps=100 + index,
                laps=0,
                best_lap=None,
                progress_m=float(index * 10),
                progress_percent=float(index),
                q_states=index,
                epsilon=0.2,
            )
            for index in range(1, train_dyna_q_policy.VISIBLE_EPISODE_ROWS + 3)
        ]

        dashboard = train_dyna_q_policy._dashboard_text(
            summaries,
            total_episodes=30,
            policy_path=Path("policy.json"),
            status="training",
        )

        self.assertIn("Best progress", dashboard)
        self.assertIn("Progress", dashboard)
        self.assertIn("Lap %", dashboard)
        self.assertIn("earlier episode rows hidden", dashboard)
        self.assertIn(f"{summaries[-1].progress_m:7.0f}m", dashboard)

    def test_best_policy_checkpoint_only_updates_on_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"a": 1}}),
                encoding="utf-8",
            )

            first_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                250.0,
            )
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"b": 2}}),
                encoding="utf-8",
            )
            worse_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                200.0,
            )
            better_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                300.0,
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(first_update)
        self.assertFalse(worse_update)
        self.assertTrue(better_update)
        self.assertEqual(best_payload["best_progress_m"], 300.0)
        self.assertEqual(best_payload["q_values"], {"b": 2})

    def test_best_policy_checkpoint_can_be_gated_by_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"a": 1}}),
                encoding="utf-8",
            )

            first_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                1800.0,
                evaluation_progress_m=500.0,
                evaluation_results={
                    "termination_reason": "out of bounds",
                    "steps": 100,
                    "laps_completed": 0,
                    "best_lap_time_seconds": None,
                },
            )
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"b": 2}}),
                encoding="utf-8",
            )
            worse_greedy_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                2200.0,
                evaluation_progress_m=450.0,
            )
            better_greedy_update = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                1200.0,
                evaluation_progress_m=700.0,
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(first_update)
        self.assertFalse(worse_greedy_update)
        self.assertTrue(better_greedy_update)
        self.assertEqual(best_payload["checkpoint_score"], "finalised_evaluation_quality")
        self.assertEqual(best_payload["best_progress_m"], 700.0)
        self.assertEqual(best_payload["best_evaluation_progress_m"], 700.0)
        self.assertEqual(best_payload["candidate_training_progress_m"], 1200.0)
        self.assertEqual(best_payload["q_values"], {"b": 2})

    def test_evaluation_gate_respects_minimum_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"a": 1}}),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                1800.0,
                evaluation_progress_m=400.0,
                min_evaluation_progress_m=500.0,
            )

        self.assertFalse(updated)
        self.assertFalse(best_path.exists())

    def test_evaluation_gate_respects_maximum_off_track(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"messy": 1}}),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                2843.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 16000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 300.0,
                    "off_track": 220,
                    "avg_speed": 0.72,
                },
                max_evaluation_off_track=150,
            )

        self.assertFalse(updated)
        self.assertFalse(best_path.exists())

    def test_evaluation_gate_respects_minimum_completed_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"candidate": 1}}),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                2843.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "repeat_evaluations",
                    "steps": 16000,
                    "laps_completed": 0,
                    "best_lap_time_seconds": None,
                    "off_track": 80,
                    "avg_speed": 0.72,
                    "evaluation_repeats": 3,
                    "evaluation_completed_repeats": 0,
                },
                min_evaluation_completions=1,
            )

        self.assertFalse(updated)
        self.assertFalse(best_path.exists())

    def test_evaluation_gate_records_maximum_off_track_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"clean": 1}}),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                2843.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 16000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 300.0,
                    "off_track": 120,
                    "avg_speed": 0.72,
                },
                max_evaluation_off_track=150,
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(updated)
        self.assertEqual(best_payload["best_evaluation_off_track"], 120)
        self.assertEqual(best_payload["best_evaluation_max_off_track_gate"], 150)

    def test_evaluation_gate_preserves_legacy_training_best_until_it_is_beaten(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"new": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_progress_m": 2031.56,
                        "q_values": {"old": 2},
                    }
                ),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                760.0,
                evaluation_progress_m=1207.0,
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertFalse(updated)
        self.assertEqual(best_payload["best_progress_m"], 2031.56)
        self.assertEqual(best_payload["q_values"], {"old": 2})

    def test_evaluation_gate_prefers_completed_lap_over_progress_only(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"lap": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 0,
                        "q_values": {"progress": 2},
                    }
                ),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                150.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 18000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 360.0,
                    "off_track": 300,
                    "avg_speed": 0.6,
                },
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(updated)
        self.assertEqual(best_payload["best_evaluation_laps"], 1)
        self.assertEqual(best_payload["best_evaluation_lap_time"], 360.0)
        self.assertEqual(best_payload["q_values"], {"lap": 1})

    def test_evaluation_gate_prefers_faster_completed_lap(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"fast": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_progress_m": 2843.0,
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 1,
                        "best_evaluation_lap_time": 362.0,
                        "best_evaluation_off_track": 370,
                        "best_evaluation_avg_speed": 0.60,
                        "q_values": {"slow": 2},
                    }
                ),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                200.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 17000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 338.0,
                    "off_track": 450,
                    "avg_speed": 0.55,
                },
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(updated)
        self.assertEqual(best_payload["best_evaluation_lap_time"], 338.0)
        self.assertEqual(best_payload["q_values"], {"fast": 1})

    def test_evaluation_gate_rejects_slower_completed_lap(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"slow": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 1,
                        "best_evaluation_lap_time": 338.0,
                        "best_evaluation_off_track": 450,
                        "best_evaluation_avg_speed": 0.55,
                        "q_values": {"fast": 2},
                    }
                ),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                200.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 18000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 362.0,
                    "off_track": 100,
                    "avg_speed": 0.70,
                },
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertFalse(updated)
        self.assertEqual(best_payload["best_evaluation_lap_time"], 338.0)
        self.assertEqual(best_payload["q_values"], {"fast": 2})

    def test_evaluation_gate_uses_off_track_as_completed_lap_tiebreaker(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"clean": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 1,
                        "best_evaluation_lap_time": 340.0,
                        "best_evaluation_off_track": 300,
                        "best_evaluation_avg_speed": 0.62,
                        "q_values": {"messy": 2},
                    }
                ),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                200.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 17000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 340.0,
                    "off_track": 180,
                    "avg_speed": 0.58,
                },
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(updated)
        self.assertEqual(best_payload["best_evaluation_off_track"], 180)
        self.assertEqual(best_payload["q_values"], {"clean": 1})

    def test_evaluation_gate_records_repeated_evaluation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"stable": 1}}),
                encoding="utf-8",
            )

            updated = train_dyna_q_policy._maybe_update_best_policy(
                policy_path,
                best_path,
                2843.0,
                evaluation_progress_m=2843.0,
                evaluation_results={
                    "termination_reason": "target_laps_completed",
                    "steps": 16000,
                    "laps_completed": 1,
                    "best_lap_time_seconds": 320.0,
                    "off_track": 150,
                    "avg_speed": 0.66,
                    "evaluation_repeats": 3,
                    "evaluation_completed_repeats": 3,
                },
            )
            best_payload = json.loads(best_path.read_text(encoding="utf-8"))

        self.assertTrue(updated)
        self.assertEqual(best_payload["best_evaluation_repeats"], 3)
        self.assertEqual(best_payload["best_evaluation_completed_repeats"], 3)

    def test_gated_evaluation_relaunches_after_socket_reset(self):
        class FakeAgent:
            def __init__(self, *, policy_path):
                self.policy_path = policy_path

        class FakeConsole:
            def __init__(self):
                self.statuses = []

            def render(self, _summaries, status):
                self.statuses.append(status)

        class ResettingRunner:
            def __init__(self):
                self.shutdown_called = False

            def run(self, _agent, *, shutdown_on_finish):
                raise ConnectionError("socket reset")

            def shutdown(self):
                self.shutdown_called = True

        class WorkingRunner:
            def run(self, _agent, *, shutdown_on_finish):
                return {"telemetry_samples": [{"dist_raced": 0}, {"dist_raced": 10}]}

        original_agent = train_dyna_q_policy.DynaQFinalisedAgent
        original_start_runner = train_dyna_q_policy._start_runner
        first_runner = ResettingRunner()
        console = FakeConsole()

        try:
            train_dyna_q_policy.DynaQFinalisedAgent = FakeAgent
            train_dyna_q_policy._start_runner = (
                lambda _track, quiet=False: WorkingRunner()
            )
            runner, results = train_dyna_q_policy._run_gated_evaluation(
                first_runner,
                policy_path=Path("policy.json"),
                track="g-track-3",
                reconnect_attempts=1,
                console=console,
                summaries=[],
                episode=3,
            )
        finally:
            train_dyna_q_policy.DynaQFinalisedAgent = original_agent
            train_dyna_q_policy._start_runner = original_start_runner

        self.assertIsInstance(runner, WorkingRunner)
        self.assertTrue(first_runner.shutdown_called)
        self.assertEqual(results["telemetry_samples"][-1]["dist_raced"], 10)
        self.assertIn("relaunching TORCS", console.statuses[-1])

    def test_resume_best_policy_copies_best_and_saves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "latest.json"
            best_path = Path(directory) / "best.json"
            previous_path = Path(directory) / "previous.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "q_values": {"old": 1}}),
                encoding="utf-8",
            )
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "best_progress_m": 2032.0,
                        "q_values": {"best": 2},
                    }
                ),
                encoding="utf-8",
            )

            previous_saved = train_dyna_q_policy._resume_best_policy(
                policy_path=policy_path,
                best_policy_path=best_path,
                previous_policy_path=previous_path,
                save_previous=True,
            )
            latest_payload = json.loads(policy_path.read_text(encoding="utf-8"))
            previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))

        self.assertTrue(previous_saved)
        self.assertEqual(latest_payload["q_values"], {"best": 2})
        self.assertEqual(latest_payload["checkpoint_type"], "latest")
        self.assertEqual(previous_payload["q_values"], {"old": 1})
        self.assertEqual(previous_payload["checkpoint_type"], "previous")

    def test_policy_payload_write_replaces_file_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "policy.json"
            policy_path.write_text(
                json.dumps({"algorithm": "dyna_q", "old": True}),
                encoding="utf-8",
            )

            train_dyna_q_policy._write_policy_payload(
                policy_path,
                {"algorithm": "dyna_q", "new": True},
            )
            payload = json.loads(policy_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"algorithm": "dyna_q", "new": True})
        self.assertFalse(policy_path.with_name(f"{policy_path.name}.tmp").exists())

    def test_auto_promote_final_archives_old_final_and_syncs_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best_path = root / "best.json"
            final_path = root / "final.json"
            latest_path = root / "latest.json"
            archive_dir = root / "archive"
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "q_values": {"best": 1},
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 1,
                        "best_evaluation_lap_time": 268.0,
                        "best_evaluation_off_track": 24,
                        "best_evaluation_avg_speed": 0.80,
                        "best_evaluation_repeats": 3,
                        "best_evaluation_completed_repeats": 3,
                    }
                ),
                encoding="utf-8",
            )
            final_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "q_values": {"final": 2},
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 1,
                        "best_evaluation_lap_time": 316.0,
                        "best_evaluation_off_track": 194,
                        "best_evaluation_avg_speed": 0.68,
                    }
                ),
                encoding="utf-8",
            )

            decision = train_dyna_q_policy._auto_promote_final_policy(
                best_policy_path=best_path,
                final_policy_path=final_path,
                latest_policy_path=latest_path,
                archive_dir=archive_dir,
                archive_existing=True,
            )
            final_payload = json.loads(final_path.read_text(encoding="utf-8"))
            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            archives = list(archive_dir.glob("*.json"))

        self.assertTrue(decision.promoted)
        self.assertEqual(decision.confidence, "high")
        self.assertEqual(final_payload["q_values"], {"best": 1})
        self.assertEqual(final_payload["checkpoint_type"], "final")
        self.assertEqual(latest_payload["q_values"], {"best": 1})
        self.assertEqual(latest_payload["checkpoint_type"], "latest")
        self.assertEqual(len(archives), 1)

    def test_auto_promote_final_rejects_candidate_without_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best_path = root / "best.json"
            final_path = root / "final.json"
            latest_path = root / "latest.json"
            best_path.write_text(
                json.dumps(
                    {
                        "algorithm": "dyna_q",
                        "q_values": {"candidate": 1},
                        "best_evaluation_progress_m": 2843.0,
                        "best_evaluation_laps": 0,
                        "best_evaluation_off_track": 12,
                    }
                ),
                encoding="utf-8",
            )

            decision = train_dyna_q_policy._auto_promote_final_policy(
                best_policy_path=best_path,
                final_policy_path=final_path,
                latest_policy_path=latest_path,
                archive_dir=root / "archive",
                archive_existing=True,
            )

        self.assertFalse(decision.promoted)
        self.assertFalse(final_path.exists())
        self.assertFalse(latest_path.exists())


if __name__ == "__main__":
    unittest.main()
