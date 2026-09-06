import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "controlled-evaluation"


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, EVALUATION_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load_script("controlled_evaluation_runner", "run_evaluation.py")
analysis = load_script("controlled_evaluation_analysis", "analyse_results.py")


def minimal_protocol(repeats=2):
    return {
        "protocol_id": "test-protocol",
        "protocol_version": "1.0",
        "environment": {
            "track_id": "g-track-3",
            "track_category": "road",
            "car_id": "car1-ow1",
            "target_laps": 1,
            "max_steps": 100,
            "telemetry_interval_steps": 1,
        },
        "design": {
            "repeats_per_agent": repeats,
            "order_seed": 42,
            "base_agent_seed": 1000,
            "reverse_speed_threshold_kmh": -1.0,
            "bootstrap_resamples": 200,
        },
        "agents": [
            {
                "id": "agent_a",
                "label": "Agent A",
                "class": "example.AgentA",
                "seeded": False,
                "assets": [],
            },
            {
                "id": "agent_b",
                "label": "Agent B",
                "class": "example.AgentB",
                "seeded": False,
                "assets": [],
            },
        ],
    }


class ControlledEvaluationRunnerTests(unittest.TestCase):
    def test_schedule_is_balanced_and_reproducible(self):
        protocol = minimal_protocol(repeats=5)
        first = runner.build_schedule(protocol)
        second = runner.build_schedule(protocol)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        for block in range(1, 6):
            block_agents = {
                row["agent_id"] for row in first if row["block"] == block
            }
            self.assertEqual(block_agents, {"agent_a", "agent_b"})
        self.assertEqual(len({row["attempt_id"] for row in first}), 10)

    def test_legacy_argv_guard_restores_cli_arguments(self):
        original = ["tool.py", "run", "--session", "example"]
        with mock.patch.object(sys, "argv", original.copy()):
            with runner.legacy_argv_guard():
                self.assertEqual(sys.argv, ["tool.py"])
            self.assertEqual(sys.argv, original)

    def test_clean_completion_requires_no_recorded_safety_event(self):
        results = {
            "termination_reason": "target_laps_completed",
            "laps_completed": 1,
            "best_lap_time_seconds": 84.2,
            "steps": 100,
            "off_track": 0,
            "total_score": 5.0,
        }
        clean_samples = [
            {"speed_x": 40.0, "dist_raced": 10.0, "damage": 0.0, "off_track": False},
            {"speed_x": 80.0, "dist_raced": 2844.0, "damage": 0.0, "off_track": False},
        ]
        clean = runner.classify_attempt(
            results,
            clean_samples,
            reverse_threshold_kmh=-1.0,
            target_laps=1,
        )
        self.assertTrue(clean["clean_completion"])
        self.assertEqual(clean["failure_category"], "clean_completion")

        unsafe_samples = clean_samples + [
            {"speed_x": -2.0, "dist_raced": 2844.0, "damage": 0.0, "off_track": False}
        ]
        unsafe = runner.classify_attempt(
            results,
            unsafe_samples,
            reverse_threshold_kmh=-1.0,
            target_laps=1,
        )
        self.assertFalse(unsafe["clean_completion"])
        self.assertEqual(unsafe["failure_category"], "completed_unclean")
        self.assertEqual(unsafe["reverse_events"], 1)

    def test_prepare_refuses_to_overwrite_an_existing_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(minimal_protocol()), encoding="utf-8")

            def manifest(_protocol, snapshot_path):
                return {
                    "protocol_snapshot_sha256": hashlib.sha256(
                        snapshot_path.read_bytes()
                    ).hexdigest(),
                    "artifacts": {},
                }

            with mock.patch.object(runner, "validate_assets", return_value=[]), mock.patch.object(
                runner, "build_manifest", side_effect=manifest
            ):
                target = runner.prepare_session(
                    protocol_path=protocol_path,
                    results_root=root / "results",
                    session_id="test-session",
                )
                self.assertTrue((target / "schedule.csv").is_file())
                with self.assertRaises(FileExistsError):
                    runner.prepare_session(
                        protocol_path=protocol_path,
                        results_root=root / "results",
                        session_id="test-session",
                    )


class ControlledEvaluationAnalysisTests(unittest.TestCase):
    @staticmethod
    def _attempt(schedule_row, clean, lap_time, distance, failure):
        row = {field: "" for field in runner.ATTEMPT_FIELDS}
        row.update(schedule_row)
        row.update(
            {
                "valid_attempt": True,
                "clean_completion": clean,
                "failure_category": failure,
                "laps_completed": 1 if lap_time else 0,
                "lap_time_seconds": lap_time or "",
                "distance_m": distance,
                "steps": 100,
                "total_score": 10.0,
                "mean_speed_kmh": 70.0,
                "max_speed_kmh": 110.0,
                "off_track_steps": 0 if clean else 1,
                "off_track_events": 0 if clean else 1,
                "reverse_steps": 0,
                "reverse_events": 0,
                "stall_events": 0,
                "damage_delta": 0.0,
                "termination_reason": (
                    "target_laps_completed" if clean else "off_track"
                ),
            }
        )
        return row

    @staticmethod
    def _write_csv(path, fields, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_analysis_creates_reproducible_tables_and_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "synthetic"
            session.mkdir()
            protocol = minimal_protocol(repeats=2)
            protocol["session"] = {"session_id": "synthetic"}
            schedule = runner.build_schedule(protocol)
            attempts = []
            for row in schedule:
                if row["agent_id"] == "agent_a":
                    attempts.append(self._attempt(row, True, 80.0 + int(row["block"]), 2844, "clean_completion"))
                elif int(row["block"]) == 1:
                    attempts.append(self._attempt(row, True, 90.0, 2844, "clean_completion"))
                else:
                    attempts.append(self._attempt(row, False, None, 1400, "off_track"))

            (session / "protocol_snapshot.json").write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            self._write_csv(session / "schedule.csv", runner.SCHEDULE_FIELDS, schedule)
            self._write_csv(
                session / "raw_attempts.csv", runner.ATTEMPT_FIELDS, attempts
            )
            self._write_csv(
                session / "infrastructure_failures.csv",
                runner.INFRASTRUCTURE_FIELDS,
                [],
            )

            output = analysis.analyse_session(session)
            payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["summary"][0]["clean_completions"], 2)
            self.assertEqual(payload["summary"][1]["clean_completions"], 1)
            self.assertTrue((output / "summary_table.tex").is_file())
            figures = list((output / "figures").glob("*.png"))
            self.assertEqual(len(figures), 5)
            self.assertTrue(all(path.stat().st_size > 1000 for path in figures))


if __name__ == "__main__":
    unittest.main()
