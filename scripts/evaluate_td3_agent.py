from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.td3_agent import (  # noqa: E402
    AGENT6_ACTION_VERSION,
    AGENT6_MODEL_FAMILY,
    AGENT6_OBSERVATION_VERSION,
    DEFAULT_BEST_EVALUATION_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    G_TRACK_3_LENGTH_METRES,
    Td3ScratchAgent,
    metadata_path_for_policy,
    read_policy_metadata,
)
from gui.torcs_config import TorcsRaceSetup, TorcsRuntimeConfig  # noqa: E402


DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "agent6_td3_scratch"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "agent6_td3"
DEFAULT_TRACK_ID = "g-track-3"
DEFAULT_TRACK_CATEGORY = "road"
DEFAULT_CAR_ID = "car1-ow1"
CHECKPOINT_GLOB = "agent6_td3_scratch_*_steps.zip"


@dataclass(frozen=True)
class EvaluationEpisode:
    repeat: int
    reason: str
    steps: int
    laps: int
    best_lap_seconds: float | None
    progress_m: float
    total_score: float
    max_speed_kmh: float
    average_speed_kmh: float
    off_track_steps: int


@dataclass(frozen=True)
class EvaluationSummary:
    policy_path: str
    track: str
    deterministic: bool
    repeats: int
    completed_repeats: int
    completed_laps: int
    median_progress_m: float
    minimum_progress_m: float
    maximum_progress_m: float
    median_off_track_steps: float
    mean_total_score: float
    best_lap_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically evaluate Agent 6 TD3 policies without training "
            "noise or replay-buffer writes."
        )
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Policy to evaluate. Repeat the flag to compare multiple policies. "
            "Defaults to the final Agent 6 model."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Also evaluate every periodic TD3 checkpoint in this directory.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--track", default=DEFAULT_TRACK_ID)
    parser.add_argument("--track-category", default=DEFAULT_TRACK_CATEGORY)
    parser.add_argument("--car", default=DEFAULT_CAR_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--promote-best-if-improved",
        action="store_true",
        help=(
            "Copy the strongest deterministic result to the verified Agent 6 "
            "evaluation checkpoint."
        ),
    )
    parser.add_argument(
        "--best-evaluation-model-path",
        type=Path,
        default=DEFAULT_BEST_EVALUATION_MODEL_PATH,
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    policies = collect_policy_paths(args.policy_path, args.checkpoint_dir)
    if not policies:
        raise SystemExit("No TD3 policy files were found to evaluate.")

    # gym_torcs/snakeoil reads sys.argv internally.
    sys.argv = [sys.argv[0]]
    summaries: list[EvaluationSummary] = []
    episodes_by_policy: dict[str, list[EvaluationEpisode]] = {}

    for policy_index, policy_path in enumerate(policies, start=1):
        print(f"\nEvaluating policy {policy_index}/{len(policies)}: {policy_path}")
        episodes = []
        for repeat in range(1, args.repeats + 1):
            print(f"Deterministic repeat {repeat}/{args.repeats}")
            results = run_evaluation(
                policy_path=policy_path,
                track=args.track,
                track_category=args.track_category,
                car=args.car,
            )
            episodes.append(evaluation_episode(repeat, results))

        summary = aggregate_evaluations(policy_path, args.track, episodes)
        summaries.append(summary)
        episodes_by_policy[str(policy_path)] = episodes
        print(summary_text(summary))

        if args.promote_best_if_improved:
            if promote_best_evaluation(
                policy_path,
                args.best_evaluation_model_path,
                summary,
                episodes,
            ):
                print(
                    "Promoted verified TD3 evaluation checkpoint: "
                    f"{args.best_evaluation_model_path}"
                )
            else:
                print("Verified TD3 evaluation checkpoint unchanged.")

    json_path, csv_path = write_evaluation_logs(
        args.output_dir,
        summaries,
        episodes_by_policy,
    )
    print(f"Evaluation JSON: {json_path}")
    print(f"Evaluation CSV: {csv_path}")
    return 0


def collect_policy_paths(
    explicit_paths: list[Path] | None,
    checkpoint_dir: Path | None,
) -> list[Path]:
    candidates = list(explicit_paths or [])
    if checkpoint_dir is not None and checkpoint_dir.is_dir():
        candidates.extend(sorted(checkpoint_dir.glob(CHECKPOINT_GLOB), key=_checkpoint_step))
    if not candidates:
        candidates.append(DEFAULT_MODEL_PATH)

    policies = []
    seen = set()
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        policies.append(path)
    return policies


def _checkpoint_step(path: Path) -> int:
    marker = path.stem.rsplit("_", 2)
    if len(marker) >= 2:
        try:
            return int(marker[-2])
        except ValueError:
            pass
    return sys.maxsize


def run_evaluation(
    *,
    policy_path: Path,
    track: str,
    track_category: str,
    car: str,
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
            agent = Td3ScratchAgent(
                policy_path=policy_path,
                require_policy=True,
                deterministic=True,
            )
            return runner.run(agent)
        finally:
            runner.shutdown()


def evaluation_episode(repeat: int, results: Mapping[str, Any]) -> EvaluationEpisode:
    return EvaluationEpisode(
        repeat=repeat,
        reason=str(results.get("termination_reason", "")),
        steps=int(results.get("steps", 0)),
        laps=int(results.get("laps_completed", 0)),
        best_lap_seconds=_optional_float(results.get("best_lap_time_seconds")),
        progress_m=episode_progress_m(results),
        total_score=float(results.get("total_score", 0.0)),
        max_speed_kmh=float(results.get("max_speed", 0.0)) * 50.0,
        average_speed_kmh=float(results.get("avg_speed", 0.0)) * 50.0,
        off_track_steps=int(results.get("off_track", 0)),
    )


def aggregate_evaluations(
    policy_path: Path,
    track: str,
    episodes: list[EvaluationEpisode],
) -> EvaluationSummary:
    if not episodes:
        raise ValueError("At least one deterministic evaluation is required")
    lap_times = [
        episode.best_lap_seconds
        for episode in episodes
        if episode.best_lap_seconds is not None
    ]
    return EvaluationSummary(
        policy_path=str(policy_path),
        track=track,
        deterministic=True,
        repeats=len(episodes),
        completed_repeats=sum(1 for episode in episodes if episode.laps > 0),
        completed_laps=sum(episode.laps for episode in episodes),
        median_progress_m=float(
            statistics.median(episode.progress_m for episode in episodes)
        ),
        minimum_progress_m=min(episode.progress_m for episode in episodes),
        maximum_progress_m=max(episode.progress_m for episode in episodes),
        median_off_track_steps=float(
            statistics.median(episode.off_track_steps for episode in episodes)
        ),
        mean_total_score=float(
            statistics.fmean(episode.total_score for episode in episodes)
        ),
        best_lap_seconds=min(lap_times) if lap_times else None,
    )


def episode_progress_m(
    results: Mapping[str, Any],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> float:
    samples = results.get("telemetry_samples")
    if not isinstance(samples, list) or not samples:
        return 0.0

    raced_values = [
        value
        for sample in samples
        if isinstance(sample, Mapping)
        for value in [_optional_float(sample.get("dist_raced"))]
        if value is not None
    ]
    if len(raced_values) >= 2:
        return max(0.0, min(track_length_m, max(raced_values) - raced_values[0]))

    distances = [
        value % track_length_m
        for sample in samples
        if isinstance(sample, Mapping)
        for value in [_optional_float(sample.get("dist_from_start"))]
        if value is not None
    ]
    if len(distances) < 2:
        return 0.0
    start = distances[0]
    return max(
        0.0,
        min(track_length_m, max((distance - start) % track_length_m for distance in distances)),
    )


def promote_best_evaluation(
    candidate_path: Path,
    best_path: Path,
    summary: EvaluationSummary,
    episodes: list[EvaluationEpisode],
) -> bool:
    existing = read_policy_metadata(best_path).get("best_evaluation")
    if isinstance(existing, Mapping) and evaluation_quality(existing) >= evaluation_quality(
        asdict(summary)
    ):
        return False

    best_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.resolve() != best_path.resolve():
        shutil.copy2(candidate_path, best_path)
    source_metadata = read_policy_metadata(candidate_path)
    metadata = {
        **source_metadata,
        "model_family": AGENT6_MODEL_FAMILY,
        "observation_version": AGENT6_OBSERVATION_VERSION,
        "action_version": AGENT6_ACTION_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
        "checkpoint_type": "best_deterministic_evaluation",
        "best_evaluation": {
            **asdict(summary),
            "source_policy_path": str(candidate_path),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "episodes": [asdict(episode) for episode in episodes],
        },
    }
    metadata_path_for_policy(best_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def evaluation_quality(evaluation: Mapping[str, Any]) -> tuple[int, float, float]:
    return (
        int(evaluation.get("completed_repeats", 0)),
        float(evaluation.get("median_progress_m", 0.0)),
        -float(evaluation.get("median_off_track_steps", 0.0)),
    )


def write_evaluation_logs(
    output_dir: Path,
    summaries: list[EvaluationSummary],
    episodes_by_policy: Mapping[str, list[EvaluationEpisode]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"td3_evaluation_{timestamp}.json"
    csv_path = output_dir / f"td3_evaluation_{timestamp}.csv"
    json_path.write_text(
        json.dumps(
            {
                "summaries": [asdict(summary) for summary in summaries],
                "episodes": {
                    policy: [asdict(episode) for episode in episodes]
                    for policy, episodes in episodes_by_policy.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["policy_path", *EvaluationEpisode.__dataclass_fields__]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for policy, episodes in episodes_by_policy.items():
            for episode in episodes:
                writer.writerow({"policy_path": policy, **asdict(episode)})
    return json_path, csv_path


def summary_text(summary: EvaluationSummary) -> str:
    best_lap = (
        "--"
        if summary.best_lap_seconds is None
        else f"{summary.best_lap_seconds:.3f}s"
    )
    return (
        f"Deterministic result: median={summary.median_progress_m:.1f}m | "
        f"range={summary.minimum_progress_m:.1f}-{summary.maximum_progress_m:.1f}m | "
        f"completed={summary.completed_repeats}/{summary.repeats} | "
        f"best_lap={best_lap}"
    )


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
