from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.dyna_q_agent import (  # noqa: E402
    DEFAULT_BEST_POLICY_PATH,
    DEFAULT_FINAL_POLICY_PATH,
    DynaQFinalisedAgent,
    G_TRACK_3_LENGTH_METRES,
)
from gui.torcs_config import TorcsRaceSetup, TorcsRuntimeConfig  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_TRACK_ID = "g-track-3"
DEFAULT_TRACK_CATEGORY = "road"
DEFAULT_CAR_ID = "car1-ow1"


@dataclass(frozen=True)
class EvaluationSummary:
    policy_path: str
    track: str
    reason: str
    steps: int
    laps: int
    best_lap: float | None
    progress_m: float
    progress_percent: float
    total_score: float
    max_speed_kmh: float
    avg_speed_kmh: float
    off_track: int
    final_speed: float | None
    final_track_pos: float | None
    final_angle: float | None
    final_front: float | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved Dyna-Q policy as a finalised greedy TORCS driver. "
            "The policy is loaded read-only: learning, exploration, and saving are off."
        )
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_BEST_POLICY_PATH)
    parser.add_argument("--track", default=DEFAULT_TRACK_ID)
    parser.add_argument("--track-category", default=DEFAULT_TRACK_CATEGORY)
    parser.add_argument("--car", default=DEFAULT_CAR_ID)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional finalised-agent tie-break seed. By default the evaluator "
            "uses the seed stored in the policy file."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-save-telemetry",
        action="store_true",
        help="Skip writing telemetry JSON/CSV files.",
    )
    parser.add_argument(
        "--promote-best-if-improved",
        action="store_true",
        help=(
            "Copy the evaluated policy to the best checkpoint only if its "
            "finalised evaluation progress improves the current best evaluation."
        ),
    )
    parser.add_argument(
        "--best-policy-path",
        type=Path,
        default=DEFAULT_BEST_POLICY_PATH,
    )
    parser.add_argument(
        "--min-promotion-progress",
        type=float,
        default=0.0,
        help="Minimum finalised progress in metres required for best promotion.",
    )
    parser.add_argument(
        "--max-promotion-off-track",
        type=int,
        default=None,
        help="Optional maximum off-track sample count allowed for best promotion.",
    )
    parser.add_argument(
        "--promote-final",
        action="store_true",
        help="Copy the evaluated policy to the finalised policy path after the run.",
    )
    args = parser.parse_args()
    if args.min_promotion_progress < 0.0:
        parser.error("--min-promotion-progress cannot be negative")
    if args.max_promotion_off_track is not None and args.max_promotion_off_track < 0:
        parser.error("--max-promotion-off-track cannot be negative")

    # gym_torcs/snakeoil parses sys.argv internally. Hide evaluator-only flags.
    sys.argv = [sys.argv[0]]

    results = _run_evaluation(
        policy_path=args.policy_path,
        track=args.track,
        track_category=args.track_category,
        car=args.car,
        seed=args.seed,
    )
    summary = _evaluation_summary(args.policy_path, args.track, results)
    print(_summary_text(summary))

    if not args.no_save_telemetry:
        json_path, csv_path = _write_telemetry_files(
            args.output_dir,
            args.track,
            summary,
            results,
        )
        print(f"Telemetry JSON: {json_path}")
        print(f"Telemetry CSV: {csv_path}")

    if args.promote_best_if_improved:
        from train_dyna_q_policy import (  # noqa: PLC0415
            _best_policy_summary,
            _maybe_update_best_policy,
        )

        updated = _maybe_update_best_policy(
            args.policy_path,
            args.best_policy_path,
            summary.progress_m,
            evaluation_progress_m=summary.progress_m,
            evaluation_results=results,
            min_evaluation_progress_m=args.min_promotion_progress,
            max_evaluation_off_track=args.max_promotion_off_track,
        )
        if updated:
            print(
                "Promoted evaluated policy to best checkpoint: "
                f"{args.best_policy_path}"
            )
            print(f"Best checkpoint: {_best_policy_summary(args.best_policy_path)}")
        else:
            print("Best checkpoint unchanged; evaluation did not improve the gate.")

    if args.promote_final:
        DEFAULT_FINAL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.policy_path, DEFAULT_FINAL_POLICY_PATH)
        print(f"Promoted evaluated policy to finalised policy: {DEFAULT_FINAL_POLICY_PATH}")

    return 0


def _run_evaluation(
    *,
    policy_path: Path,
    track: str,
    track_category: str,
    car: str,
    seed: int | None = None,
) -> dict[str, Any]:
    from runner.torcs_runner import TorcsRunner  # noqa: PLC0415

    setup = TorcsRaceSetup(
        track_id=track,
        track_category=track_category,
        car_id=car,
    )
    with TorcsRuntimeConfig(setup):
        runner = TorcsRunner()
        try:
            runner.launch()
            runner.connect()
            runner.load_track(track)
            agent = DynaQFinalisedAgent(policy_path=policy_path, seed=seed)
            return runner.run(agent)
        finally:
            runner.shutdown()


def _evaluation_summary(
    policy_path: Path,
    track: str,
    results: dict[str, Any],
) -> EvaluationSummary:
    progress_m = _episode_progress_m(results)
    return EvaluationSummary(
        policy_path=str(policy_path),
        track=track,
        reason=str(results.get("termination_reason", "")),
        steps=int(results.get("steps", 0)),
        laps=int(results.get("laps_completed", 0)),
        best_lap=_optional_float(results.get("best_lap_time_seconds")),
        progress_m=progress_m,
        progress_percent=min(100.0, progress_m / G_TRACK_3_LENGTH_METRES * 100.0),
        total_score=float(results.get("total_score", 0.0)),
        max_speed_kmh=float(results.get("max_speed", 0.0)) * 50.0,
        avg_speed_kmh=float(results.get("avg_speed", 0.0)) * 50.0,
        off_track=int(results.get("off_track", 0)),
        final_speed=_last_sample_float(results, "speed_x"),
        final_track_pos=_last_sample_float(results, "track_pos"),
        final_angle=_last_sample_float(results, "angle"),
        final_front=_last_sample_float(results, "front_sensor"),
    )


def _episode_progress_m(
    results: dict[str, object],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> float:
    samples = results.get("telemetry_samples")
    if not isinstance(samples, list) or not samples:
        return 0.0

    dist_raced = [
        value
        for sample in samples
        if isinstance(sample, dict)
        for value in [_optional_float(sample.get("dist_raced"))]
        if value is not None
    ]
    if len(dist_raced) >= 2:
        return max(0.0, min(track_length_m, max(dist_raced) - dist_raced[0]))

    distances = [
        value % track_length_m
        for sample in samples
        if isinstance(sample, dict)
        for value in [_optional_float(sample.get("dist_from_start"))]
        if value is not None
    ]
    if len(distances) < 2:
        return 0.0
    start = distances[0]
    progress_values = [
        (distance - start) % track_length_m
        for distance in distances
    ]
    return max(0.0, min(track_length_m, max(progress_values)))


def _last_sample_float(results: dict[str, object], key: str) -> float | None:
    samples = results.get("telemetry_samples")
    if not isinstance(samples, list) or not samples:
        return None
    sample = samples[-1]
    if not isinstance(sample, dict):
        return None
    return _optional_float(sample.get(key))


def _summary_text(summary: EvaluationSummary) -> str:
    best_lap = "--" if summary.best_lap is None else f"{summary.best_lap:.3f}s"
    return "\n".join(
        [
            "Dyna-Q Policy Evaluation",
            f"Policy: {summary.policy_path}",
            f"Track: {summary.track}",
            f"Result: {summary.reason}",
            (
                f"Progress: {summary.progress_m:.0f}m "
                f"({summary.progress_percent:.1f}%) | "
                f"Steps: {summary.steps} | Laps: {summary.laps} | "
                f"Best lap: {best_lap}"
            ),
            (
                f"Speed: max={summary.max_speed_kmh:.1f} km/h | "
                f"avg={summary.avg_speed_kmh:.1f} km/h | "
                f"Off track: {summary.off_track}"
            ),
            (
                "Final: "
                f"speed={_format_optional_float(summary.final_speed, width=4, precision=0)} | "
                f"trackPos={_format_optional_float(summary.final_track_pos, width=6, precision=2, sign=True)} | "
                f"angle={_format_optional_float(summary.final_angle, width=6, precision=2, sign=True)} | "
                f"front={_format_optional_float(summary.final_front, width=5, precision=0)}"
            ),
        ]
    )


def _format_optional_float(
    value: float | None,
    *,
    width: int,
    precision: int,
    sign: bool = False,
) -> str:
    if value is None:
        return "--".rjust(width)
    sign_flag = "+" if sign else ""
    return f"{value:{sign_flag}{width}.{precision}f}"


def _write_telemetry_files(
    output_dir: Path,
    track: str,
    summary: EvaluationSummary,
    results: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    track_slug = track.replace("-", "_")
    json_path = output_dir / f"dyna_q_{track_slug}_evaluation.json"
    csv_path = output_dir / f"dyna_q_{track_slug}_evaluation.csv"
    samples = results.get("telemetry_samples", [])
    payload = {
        "summary": asdict(summary),
        "telemetry_samples": samples if isinstance(samples, list) else [],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_telemetry_csv(csv_path, payload["telemetry_samples"])
    return json_path, csv_path


def _write_telemetry_csv(path: Path, samples: list[Any]) -> None:
    fieldnames = [
        "step",
        "dist_from_start",
        "dist_raced",
        "speed_x",
        "speed_y",
        "angle",
        "track_pos",
        "damage",
        "steer",
        "accel",
        "brake",
        "gear",
        "reward",
        "off_track",
        "front_sensor",
        "min_track_sensor",
        "cur_lap_time",
        "last_lap_time",
        "track_sensors",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            row = {key: sample.get(key) for key in fieldnames}
            row["track_sensors"] = json.dumps(sample.get("track_sensors", []))
            writer.writerow(row)


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
