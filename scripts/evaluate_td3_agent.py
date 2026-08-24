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
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.td3_agent import (  # noqa: E402
    AGENT6_ACTION_SHAPE,
    AGENT6_ACTION_VERSION,
    AGENT6_MODEL_FAMILY,
    AGENT6_OBSERVATION_VERSION,
    AGENT6_EXTERNAL_EVALUATION_BASE_SEED,
    AGENT6_ROBUSTNESS_MAX_PAIRED_REGRESSION_M,
    AGENT6_ROBUSTNESS_PROTOCOL,
    AGENT6_ROBUSTNESS_REPEATS,
    AGENT6_ROBUSTNESS_STEERING_NOISE_STD,
    AGENT6_ROBUSTNESS_TRIALS_PER_SEED,
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
    order_stratified_seed_aggregates,
    parameter_state_dicts_are_identical,
    policy_contract_mismatches,
    policy_uses_legacy_action_contract,
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
DEFAULT_TRIALS_PER_SEED = AGENT6_ROBUSTNESS_TRIALS_PER_SEED
DEFAULT_STEERING_NOISE_STD = AGENT6_ROBUSTNESS_STEERING_NOISE_STD
MIN_EVALUATION_SEED = EXTERNAL_EVALUATION_SEED_MIN
MAX_EVALUATION_SEED = EXTERNAL_EVALUATION_SEED_MAX
MAX_PAIRED_REGRESSION_M = AGENT6_ROBUSTNESS_MAX_PAIRED_REGRESSION_M


@dataclass(frozen=True)
class EvaluationEpisode:
    repeat: int
    seed: int
    trial: int
    order_position: int
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
    trials_per_seed: int
    total_trials: int
    completed_repeats: int
    completed_laps: int
    median_progress_m: float
    minimum_progress_m: float
    maximum_progress_m: float
    raw_minimum_progress_m: float
    seed_median_progress_m: tuple[float, ...]
    median_off_track_steps: float
    mean_total_score: float
    best_lap_seconds: float | None


@dataclass(frozen=True)
class PolicyActorGroup:
    representative_path: Path
    member_paths: tuple[Path, ...]


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
        "--trials-per-seed",
        type=int,
        default=DEFAULT_TRIALS_PER_SEED,
        help=(
            "Trials per policy and seed. Protocol v5 requires four so each "
            "policy runs twice in each order position."
        ),
    )
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
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if args.repeats % 2 == 0:
        parser.error("--repeats must be odd")
    if args.trials_per_seed != AGENT6_ROBUSTNESS_TRIALS_PER_SEED:
        parser.error(
            "--trials-per-seed must be exactly "
            f"{AGENT6_ROBUSTNESS_TRIALS_PER_SEED} for the balanced protocol"
        )
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
    actor_groups = group_policy_actors(
        policies,
        preferred_representative=(
            args.best_evaluation_model_path
            if args.promote_best_if_improved
            else None
        ),
    )
    actor_equivalence = policy_actor_equivalence_records(actor_groups)
    distinct_policies = [group.representative_path for group in actor_groups]
    for record in actor_equivalence:
        if record["classification"] == "policy_equivalent":
            print(
                f"Policy-equivalent actor: {record['policy_path']} == "
                f"{record['representative_policy_path']}"
            )
    if len(distinct_policies) > 2:
        raise SystemExit(
            "The balanced robustness protocol compares at most two distinct "
            "actors per run. Evaluate additional candidates pairwise."
        )
    if len(policies) > 1 and len(distinct_policies) == 1:
        print(
            "All requested policies have identical actor parameters; "
            "classified as policy-equivalent without launching TORCS."
        )
        json_path, csv_path = write_evaluation_logs(
            args.output_dir,
            [],
            {str(policy): [] for policy in policies},
            policy_actor_equivalence=actor_equivalence,
        )
        print(f"Evaluation JSON: {json_path}")
        print(f"Evaluation CSV: {csv_path}")
        return 0
    policies = distinct_policies
    seeds = build_evaluation_seeds(
        repeats=args.repeats,
        base_seed=args.base_seed,
        explicit_seeds=args.seeds,
    )
    print(f"Evaluation protocol: {EVALUATION_PROTOCOL}")
    print(f"Evaluation seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"Trials per policy-seed: {args.trials_per_seed}")
    print(f"Steering disturbance std: {args.steering_noise_std:.4f}")

    # gym_torcs/snakeoil reads sys.argv internally.
    sys.argv = [sys.argv[0]]
    summaries: list[EvaluationSummary] = []
    episodes_by_policy: dict[str, list[EvaluationEpisode]] = {
        str(policy): [] for policy in policies
    }
    total_trials = len(policies) * len(seeds) * args.trials_per_seed
    completed_trials = 0

    for seed_index, seed in enumerate(seeds):
        for trial_index in range(args.trials_per_seed):
            ordered_policies = counterbalanced_policy_schedule(
                policies,
                seed_index=seed_index,
                trial_index=trial_index,
            )
            for order_position, policy_path in enumerate(
                ordered_policies,
                start=1,
            ):
                completed_trials += 1
                print(
                    f"\nRobustness trial {completed_trials}/{total_trials} | "
                    f"seed={seed} ({seed_index + 1}/{len(seeds)}) | "
                    f"replicate={trial_index + 1}/{args.trials_per_seed} | "
                    f"order={order_position}/{len(policies)} | {policy_path}"
                )
                results = run_evaluation(
                    policy_path=policy_path,
                    track=args.track,
                    track_category=args.track_category,
                    car=args.car,
                    evaluation_seed=seed,
                    steering_noise_std=args.steering_noise_std,
                )
                episodes_by_policy[str(policy_path)].append(
                    evaluation_episode(
                        seed_index + 1,
                        seed,
                        results,
                        trial=trial_index + 1,
                        order_position=order_position,
                    )
                )

    for policy_path in policies:
        episodes = episodes_by_policy[str(policy_path)]
        summary = aggregate_evaluations(
            policy_path,
            args.track,
            episodes,
            steering_noise_std=args.steering_noise_std,
        )
        summaries.append(summary)
        print(f"\nPolicy summary: {policy_path}")
        print(summary_text(summary))

    if args.promote_best_if_improved:
        baseline_path = args.best_evaluation_model_path.resolve()
        baseline_summary = next(
            (
                summary
                for summary in summaries
                if Path(summary.policy_path).resolve() == baseline_path
            ),
            None,
        )
        eligible_summaries = summaries
        if baseline_summary is not None:
            eligible_summaries = [baseline_summary]
            for summary in summaries:
                if summary is baseline_summary:
                    continue
                if evaluation_candidate_is_noninferior(
                    summary,
                    baseline_summary,
                ):
                    eligible_summaries.append(summary)
                else:
                    print(
                        "Promotion guard rejected policy due to replicated "
                        f"paired-seed regression: {summary.policy_path}"
                    )
        best_summary = max(
            eligible_summaries,
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
        policy_actor_equivalence=actor_equivalence,
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


def load_policy_actor_state(policy_path: Path) -> dict[str, Any]:
    from stable_baselines3 import TD3  # noqa: PLC0415

    mismatches = policy_contract_mismatches(
        read_policy_metadata(policy_path),
        allow_legacy_action_contract=True,
    )
    if mismatches:
        raise ValueError(
            f"TD3 policy contract mismatch for {policy_path}: "
            + ", ".join(mismatches)
        )
    model = TD3.load(str(policy_path), device="cpu")
    return {
        name: value.detach().cpu().clone()
        for name, value in model.actor.state_dict().items()
    }


def group_policy_actors(
    policies: list[Path],
    *,
    preferred_representative: Path | None = None,
    actor_state_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> list[PolicyActorGroup]:
    """Group checkpoints by exact actor parameters, ignoring critic state."""
    loader = actor_state_loader or load_policy_actor_state
    preferred = (
        Path(preferred_representative).resolve()
        if preferred_representative is not None
        else None
    )
    groups: list[dict[str, Any]] = []
    for policy in policies:
        path = Path(policy).resolve()
        actor_state = loader(path)
        matching_group = next(
            (
                group
                for group in groups
                if parameter_state_dicts_are_identical(
                    group["actor_state"],
                    actor_state,
                )
            ),
            None,
        )
        if matching_group is None:
            groups.append({"actor_state": actor_state, "members": [path]})
        else:
            matching_group["members"].append(path)

    result: list[PolicyActorGroup] = []
    for group in groups:
        members = tuple(group["members"])
        representative = (
            preferred if preferred is not None and preferred in members else members[0]
        )
        result.append(
            PolicyActorGroup(
                representative_path=representative,
                member_paths=members,
            )
        )
    return result


def policy_actor_equivalence_records(
    groups: list[PolicyActorGroup],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in groups:
        equivalent_group = len(group.member_paths) > 1
        for policy in group.member_paths:
            is_representative = policy == group.representative_path
            records.append(
                {
                    "policy_path": str(policy),
                    "representative_policy_path": str(
                        group.representative_path
                    ),
                    "classification": (
                        "equivalence_representative"
                        if equivalent_group and is_representative
                        else (
                            "policy_equivalent"
                            if equivalent_group
                            else "distinct"
                        )
                    ),
                    "actor_parameters_identical": equivalent_group,
                }
            )
    return records


def counterbalanced_policy_schedule(
    policies: list[Path],
    *,
    seed_index: int,
    trial_index: int,
) -> tuple[Path, ...]:
    """Rotate policy order across seed trials to distribute temporal bias."""
    if not policies:
        raise ValueError("at least one policy is required")
    if int(seed_index) < 0 or int(trial_index) < 0:
        raise ValueError("seed_index and trial_index must be non-negative")
    rotation = (int(seed_index) + int(trial_index)) % len(policies)
    return tuple([*policies[rotation:], *policies[:rotation]])


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
    *,
    trial: int = 1,
    order_position: int = 1,
) -> EvaluationEpisode:
    return EvaluationEpisode(
        repeat=repeat,
        seed=seed,
        trial=int(trial),
        order_position=int(order_position),
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
    episodes_by_seed: dict[int, list[EvaluationEpisode]] = {}
    for episode in episodes:
        episodes_by_seed.setdefault(episode.seed, []).append(episode)
    trial_counts = {len(seed_episodes) for seed_episodes in episodes_by_seed.values()}
    if len(trial_counts) != 1:
        raise ValueError("every evaluation seed must have the same trial count")
    trials_per_seed = trial_counts.pop()
    if trials_per_seed < 1:
        raise ValueError("every evaluation seed must include at least one trial")
    for seed_episodes in episodes_by_seed.values():
        trial_ids = [episode.trial for episode in seed_episodes]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("evaluation trial identifiers must be unique per seed")
        order_positions = {episode.order_position for episode in seed_episodes}
        if not order_positions.issubset({1, 2}):
            raise ValueError("pairwise evaluation order positions must be 1 or 2")

    seed_median_progress = tuple(
        _order_stratified_episode_metric(seed_episodes, "progress_m")
        for seed_episodes in episodes_by_seed.values()
    )
    seed_median_off_track = tuple(
        _order_stratified_episode_metric(seed_episodes, "off_track_steps")
        for seed_episodes in episodes_by_seed.values()
    )
    seed_median_scores = tuple(
        _order_stratified_episode_metric(seed_episodes, "total_score")
        for seed_episodes in episodes_by_seed.values()
    )
    seed_lap_rates = tuple(
        _order_stratified_episode_metric(seed_episodes, "laps")
        for seed_episodes in episodes_by_seed.values()
    )
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
        evaluation_seeds=tuple(episodes_by_seed),
        steering_noise_std=float(steering_noise_std),
        repeats=len(episodes_by_seed),
        trials_per_seed=trials_per_seed,
        total_trials=len(episodes),
        completed_repeats=sum(1 for laps in seed_lap_rates if laps > 0.5),
        completed_laps=sum(1 for laps in seed_lap_rates if laps > 0.5),
        median_progress_m=float(statistics.median(seed_median_progress)),
        minimum_progress_m=min(seed_median_progress),
        maximum_progress_m=max(seed_median_progress),
        raw_minimum_progress_m=min(episode.progress_m for episode in episodes),
        seed_median_progress_m=seed_median_progress,
        median_off_track_steps=float(statistics.median(seed_median_off_track)),
        mean_total_score=float(statistics.fmean(seed_median_scores)),
        best_lap_seconds=min(lap_times) if lap_times else None,
    )


def _order_stratified_episode_metric(
    episodes: list[EvaluationEpisode],
    attribute: str,
) -> float:
    return order_stratified_seed_aggregates(
        [float(getattr(episode, attribute)) for episode in episodes],
        [episode.order_position for episode in episodes],
        seed_count=1,
        trials_per_seed=len(episodes),
    )[0]


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
    candidate_action_shape = source_metadata.get("action_shape", AGENT6_ACTION_SHAPE)
    candidate_action_version = source_metadata.get(
        "action_version",
        AGENT6_ACTION_VERSION,
    )
    candidate_observation_version = source_metadata.get(
        "observation_version",
        AGENT6_OBSERVATION_VERSION,
    )
    metadata = {
        **source_metadata,
        "model_family": AGENT6_MODEL_FAMILY,
        "observation_version": candidate_observation_version,
        "action_version": candidate_action_version,
        "feature_names": FEATURE_NAMES,
        "action_shape": candidate_action_shape,
        "legacy_action_adapter": policy_uses_legacy_action_contract(source_metadata),
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
) -> tuple[str, tuple[int, ...], int, float]:
    seeds = evaluation.get("evaluation_seeds", ())
    if not isinstance(seeds, (list, tuple)):
        seeds = ()
    return (
        str(evaluation.get("evaluation_protocol", "legacy_unperturbed_v0")),
        tuple(int(seed) for seed in seeds),
        int(evaluation.get("trials_per_seed", 1)),
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


def evaluation_candidate_is_noninferior(
    candidate: EvaluationSummary,
    reference: EvaluationSummary,
    *,
    maximum_paired_regression_m: float = MAX_PAIRED_REGRESSION_M,
) -> bool:
    """Require aggregate and paired non-inferiority before promotion."""
    if evaluation_protocol_key(asdict(candidate)) != evaluation_protocol_key(
        asdict(reference)
    ):
        return False
    candidate_medians = candidate.seed_median_progress_m
    reference_medians = reference.seed_median_progress_m
    if not candidate_medians or len(candidate_medians) != len(reference_medians):
        return False
    regression_limit = float(maximum_paired_regression_m)
    if not math.isfinite(regression_limit) or regression_limit < 0.0:
        raise ValueError("paired regression limit must be finite and non-negative")
    paired_deltas = tuple(
        candidate_value - reference_value
        for candidate_value, reference_value in zip(
            candidate_medians,
            reference_medians,
        )
    )
    return (
        evaluation_quality(asdict(candidate)) > evaluation_quality(asdict(reference))
        and candidate.median_progress_m >= reference.median_progress_m
        and candidate.minimum_progress_m >= reference.minimum_progress_m
        and statistics.median(paired_deltas) >= 0.0
        and min(paired_deltas) >= -regression_limit
    )


def write_evaluation_logs(
    output_dir: Path,
    summaries: list[EvaluationSummary],
    episodes_by_policy: Mapping[str, list[EvaluationEpisode]],
    *,
    policy_actor_equivalence: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"td3_evaluation_{timestamp}.json"
    csv_path = output_dir / f"td3_evaluation_{timestamp}.csv"
    json_path.write_text(
        json.dumps(
            {
                "summaries": [asdict(summary) for summary in summaries],
                "policy_actor_equivalence": list(
                    policy_actor_equivalence or []
                ),
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
        f"Order-stratified robustness: median={summary.median_progress_m:.1f}m | "
        f"range={summary.minimum_progress_m:.1f}-{summary.maximum_progress_m:.1f}m | "
        f"raw_min={summary.raw_minimum_progress_m:.1f}m | "
        f"completed={summary.completed_repeats}/{summary.repeats} seeds | "
        f"design={summary.repeats} seeds x {summary.trials_per_seed} | "
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
