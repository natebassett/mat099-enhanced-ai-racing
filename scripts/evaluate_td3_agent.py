from __future__ import annotations

import argparse
import csv
import json
import math
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
    AGENT6_EXTERNAL_EVALUATION_BASE_SEED,
    AGENT6_ROBUSTNESS_PROTOCOL,
    AGENT6_ROBUSTNESS_REPEATS,
    AGENT6_ROBUSTNESS_STEERING_NOISE_STD,
    DEFAULT_BEST_EVALUATION_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    G_TRACK_3_LENGTH_METRES,
    EXTERNAL_EVALUATION_SEED_MAX,
    EXTERNAL_EVALUATION_SEED_MIN,
    SeededSteeringActionNoise,
    Td3ScratchAgent,
    build_reproducible_seed_suite,
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
EVALUATION_PROTOCOL = AGENT6_ROBUSTNESS_PROTOCOL
DEFAULT_BASE_SEED = AGENT6_EXTERNAL_EVALUATION_BASE_SEED
DEFAULT_EVALUATION_REPEATS = AGENT6_ROBUSTNESS_REPEATS
DEFAULT_STEERING_NOISE_STD = AGENT6_ROBUSTNESS_STEERING_NOISE_STD
MIN_EVALUATION_SEED = EXTERNAL_EVALUATION_SEED_MIN
MAX_EVALUATION_SEED = EXTERNAL_EVALUATION_SEED_MAX


@dataclass(frozen=True)
class EvaluationEpisode:
    repeat: int
    seed: int
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
    evaluation_protocol: str
    evaluation_seeds: tuple[int, ...]
    steering_noise_std: float
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
            "Evaluate deterministic Agent 6 TD3 policies under reproducible "
            "seeded steering disturbances, without training or replay writes."
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
    parser.add_argument("--repeats", type=int, default=DEFAULT_EVALUATION_REPEATS)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Generate a reproducible unique seed suite from this value.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit evaluation seeds. Overrides --repeats and --base-seed.",
    )
    parser.add_argument(
        "--steering-noise-std",
        type=float,
        default=DEFAULT_STEERING_NOISE_STD,
        help=(
            "Standard deviation of the small correlated steering disturbance "
            "used to test policy recovery. Set to 0 for an unperturbed baseline."
        ),
    )
    parser.add_argument("--track", default=DEFAULT_TRACK_ID)
    parser.add_argument("--track-category", default=DEFAULT_TRACK_CATEGORY)
    parser.add_argument("--car", default=DEFAULT_CAR_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--promote-best-if-improved",
        action="store_true",
        help=(
            "Copy the strongest seeded robustness result to the verified "
            "Agent 6 evaluation checkpoint."
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
    if not 0 <= args.base_seed <= MAX_EVALUATION_SEED:
        parser.error(
            f"--base-seed must be between 0 and {MAX_EVALUATION_SEED}"
        )
    if not math.isfinite(args.steering_noise_std) or args.steering_noise_std < 0.0:
        parser.error("--steering-noise-std must be finite and non-negative")
    if args.seeds is not None:
        if any(
            seed < MIN_EVALUATION_SEED or seed > MAX_EVALUATION_SEED
            for seed in args.seeds
        ):
            parser.error(
                "--seeds must use the external evaluation range "
                f"{MIN_EVALUATION_SEED}..{MAX_EVALUATION_SEED}"
            )
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must be unique")
    return args


def main() -> int:
    args = parse_args()
    policies = collect_policy_paths(args.policy_path, args.checkpoint_dir)
    if not policies:
        raise SystemExit("No TD3 policy files were found to evaluate.")
    if args.promote_best_if_improved:
        policies = include_verified_baseline(
            policies,
            args.best_evaluation_model_path,
        )
    seeds = build_evaluation_seeds(
        repeats=args.repeats,
        base_seed=args.base_seed,
        explicit_seeds=args.seeds,
    )
    print(f"Evaluation protocol: {EVALUATION_PROTOCOL}")
    print(f"Evaluation seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"Steering disturbance std: {args.steering_noise_std:.4f}")

    # gym_torcs/snakeoil reads sys.argv internally.
    sys.argv = [sys.argv[0]]
    summaries: list[EvaluationSummary] = []
    episodes_by_policy: dict[str, list[EvaluationEpisode]] = {}

    for policy_index, policy_path in enumerate(policies, start=1):
        print(f"\nEvaluating policy {policy_index}/{len(policies)}: {policy_path}")
        episodes = []
        for repeat, seed in enumerate(seeds, start=1):
            print(f"Robustness repeat {repeat}/{len(seeds)} | seed={seed}")
            results = run_evaluation(
                policy_path=policy_path,
                track=args.track,
                track_category=args.track_category,
                car=args.car,
                evaluation_seed=seed,
                steering_noise_std=args.steering_noise_std,
            )
            episodes.append(evaluation_episode(repeat, seed, results))

        summary = aggregate_evaluations(
            policy_path,
            args.track,
            episodes,
            steering_noise_std=args.steering_noise_std,
        )
        summaries.append(summary)
        episodes_by_policy[str(policy_path)] = episodes
        print(summary_text(summary))

    if args.promote_best_if_improved:
        best_summary = max(
            summaries,
            key=lambda candidate: evaluation_quality(asdict(candidate)),
        )
        best_episodes = episodes_by_policy[best_summary.policy_path]
        baseline_was_evaluated = (
            args.best_evaluation_model_path.resolve() in policies
        )
        if promote_best_evaluation(
            Path(best_summary.policy_path),
            args.best_evaluation_model_path,
            best_summary,
            best_episodes,
            allow_protocol_migration=baseline_was_evaluated,
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


def include_verified_baseline(policies: list[Path], best_path: Path) -> list[Path]:
    baseline = Path(best_path).resolve()
    if not baseline.is_file() or baseline in policies:
        return policies
    return [*policies, baseline]


def build_evaluation_seeds(
    *,
    repeats: int,
    base_seed: int,
    explicit_seeds: list[int] | None,
) -> tuple[int, ...]:
    if explicit_seeds is not None:
        seeds = tuple(int(seed) for seed in explicit_seeds)
        if not seeds:
            raise ValueError("explicit evaluation seeds cannot be empty")
        if any(
            seed < MIN_EVALUATION_SEED or seed > MAX_EVALUATION_SEED
            for seed in seeds
        ):
            raise ValueError(
                "evaluation seeds must use the held-out external range "
                f"{MIN_EVALUATION_SEED}..{MAX_EVALUATION_SEED}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("evaluation seeds must be unique")
        return seeds
    return build_reproducible_seed_suite(
        repeats=repeats,
        base_seed=base_seed,
        seed_min=MIN_EVALUATION_SEED,
        seed_max=MAX_EVALUATION_SEED,
    )


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
    evaluation_seed: int,
    steering_noise_std: float,
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
            policy_agent = Td3ScratchAgent(
                seed=evaluation_seed,
                policy_path=policy_path,
                require_policy=True,
                deterministic=True,
                action_noise=SeededSteeringActionNoise(
                    seed=evaluation_seed,
                    steering_noise_std=steering_noise_std,
                ),
            )
            return runner.run(policy_agent)
        finally:
            runner.shutdown()


def evaluation_episode(
    repeat: int,
    seed: int,
    results: Mapping[str, Any],
) -> EvaluationEpisode:
    return EvaluationEpisode(
        repeat=repeat,
        seed=seed,
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
    *,
    steering_noise_std: float = DEFAULT_STEERING_NOISE_STD,
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
        evaluation_protocol=EVALUATION_PROTOCOL,
        evaluation_seeds=tuple(episode.seed for episode in episodes),
        steering_noise_std=float(steering_noise_std),
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
    *,
    allow_protocol_migration: bool = False,
) -> bool:
    existing = read_policy_metadata(best_path).get("best_evaluation")
    candidate_evaluation = asdict(summary)
    if isinstance(existing, Mapping):
        same_protocol = evaluation_protocol_key(existing) == evaluation_protocol_key(
            candidate_evaluation
        )
        if not same_protocol and not allow_protocol_migration:
            return False
        if same_protocol and (
            evaluation_quality(existing) >= evaluation_quality(candidate_evaluation)
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
        "checkpoint_type": "best_seeded_robustness_evaluation",
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


def evaluation_protocol_key(
    evaluation: Mapping[str, Any],
) -> tuple[str, tuple[int, ...], float]:
    seeds = evaluation.get("evaluation_seeds", ())
    if not isinstance(seeds, (list, tuple)):
        seeds = ()
    return (
        str(evaluation.get("evaluation_protocol", "legacy_unperturbed_v0")),
        tuple(int(seed) for seed in seeds),
        float(evaluation.get("steering_noise_std", 0.0)),
    )


def evaluation_quality(
    evaluation: Mapping[str, Any],
) -> tuple[int, int, float, float, float, float]:
    return (
        int(evaluation.get("completed_repeats", 0)),
        int(evaluation.get("completed_laps", 0)),
        float(evaluation.get("median_progress_m", 0.0)),
        float(evaluation.get("minimum_progress_m", 0.0)),
        -float(evaluation.get("median_off_track_steps", 0.0)),
        float(evaluation.get("mean_total_score", 0.0)),
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
        f"Seeded robustness result: median={summary.median_progress_m:.1f}m | "
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
