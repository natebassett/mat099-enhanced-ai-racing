from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from n_step_td3.environment import EpisodeSummary, NstepTorcsEnvironment
from n_step_td3.contracts import (
    DEFAULT_PACE_STEERING_LIMIT,
    DEFAULT_STEERING_RATE_COST_COEFFICIENT,
    REWARD_VERSION,
)
from n_step_td3.learner import (
    FINE_TUNE_ACTOR_LEARNING_RATE,
    NstepTd3Config,
    NstepTd3Learner,
    UpdateMetrics,
    load_actor_checkpoint,
    load_checkpoint,
)
from n_step_td3.replay import NstepTransitionAccumulator


MODEL_DIR = PROJECT_ROOT / "models" / "agent7_n_step_td3_v3"
RUNS_DIR = PROJECT_ROOT / "models" / "training_runs" / "agent7_n_step_td3_v3"
BEST_EVALUATION_PATH = MODEL_DIR / "best_evaluation.pt"
BEST_PACE_PATH = MODEL_DIR / "best_pace.pt"
STEERING_RATE_V4_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent7_n_step_td3_v4"
)
STEERING_RATE_V4_RUNS_DIR = (
    PROJECT_ROOT / "models" / "training_runs" / "agent7_n_step_td3_v4"
)
PACE_V5_MODEL_DIR = PROJECT_ROOT / "models" / "agent7_n_step_td3_v5"
PACE_V5_RUNS_DIR = (
    PROJECT_ROOT / "models" / "training_runs" / "agent7_n_step_td3_v5"
)
SENSOR_ONLY_MODEL_DIR = PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3"
SENSOR_ONLY_RUNS_DIR = (
    PROJECT_ROOT / "models" / "training_runs" / "agent8_sensor_n_step_td3"
)
SENSOR_ONLY_BEST_EVALUATION_PATH = SENSOR_ONLY_MODEL_DIR / "best_evaluation.pt"
SENSOR_ONLY_BEST_PACE_PATH = SENSOR_ONLY_MODEL_DIR / "best_pace.pt"
SENSOR_STEERING_V2_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v2_steer070"
)
SENSOR_STEERING_V2_RUNS_DIR = (
    PROJECT_ROOT
    / "models"
    / "training_runs"
    / "agent8_sensor_n_step_td3_v2_steer070"
)
SENSOR_STEERING_V2_LIMIT = 0.7
SENSOR_STEERING_V2_ACTOR_LEARNING_RATE = 1e-5
SENSOR_CLEAN_OBSERVATION_V3_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v3_clean_observation"
)
SENSOR_CLEAN_OBSERVATION_V3_RUNS_DIR = (
    PROJECT_ROOT
    / "models"
    / "training_runs"
    / "agent8_sensor_n_step_td3_v3_clean_observation"
)
SENSOR_CLEAN_OBSERVATION_V3_ACTOR_LEARNING_RATE = 1e-5
SENSOR_V1_CONTINUATION_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_continuation"
)
SENSOR_V1_CONTINUATION_RUNS_DIR = (
    PROJECT_ROOT
    / "models"
    / "training_runs"
    / "agent8_sensor_n_step_td3_v1_continuation"
)
SENSOR_V1_STABILITY_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_stability"
)
SENSOR_V1_STABILITY_RUNS_DIR = (
    PROJECT_ROOT
    / "models"
    / "training_runs"
    / "agent8_sensor_n_step_td3_v1_stability"
)
SENSOR_V1_CLEAN_RELIABILITY_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_clean_reliability"
)
SENSOR_V1_CLEAN_RELIABILITY_RUNS_DIR = (
    PROJECT_ROOT
    / "models"
    / "training_runs"
    / "agent8_sensor_n_step_td3_v1_clean_reliability"
)
SENSOR_V1_CLEAN_RELIABILITY_ACTOR_LEARNING_RATE = 3e-6
SENSOR_V1_CLEAN_RELIABILITY_EVALUATION_REPEATS = 10
SENSOR_V1_CLEAN_RELIABILITY_COMPLETION_RATE = 0.8
SENSOR_V1_CLEAN_RELIABILITY_MEDIAN_SECONDS = 90.0
SENSOR_V1_ROBUST_PACE_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_robust_pace"
)
SENSOR_V1_ROBUST_PACE_RUNS_DIR = (
    PROJECT_ROOT
    / "models"
    / "training_runs"
    / "agent8_sensor_n_step_td3_v1_robust_pace"
)
SENSOR_V1_ROBUST_PACE_ACTOR_LEARNING_RATE = 3e-6
SENSOR_V1_ROBUST_PACE_EVALUATION_REPEATS = 10
SENSOR_V1_ROBUST_PACE_COMPLETION_RATE = 0.8
SENSOR_V1_ROBUST_PACE_MEDIAN_SECONDS = 90.0
SENSOR_V1_ROBUST_PACE_NOISY_EPISODE_FRACTION = 0.25
SENSOR_V1_ROBUST_PACE_NOISY_OBSERVATION_STD = 0.025
DEFAULT_TRAIN_FREQUENCY = 4
PACE_V5_TOTAL_TIMESTEPS = 1_000_000
PACE_V5_EVALUATION_INTERVAL = 50_000
PACE_V5_WARMUP_ACTION_HOLD_INTERACTIONS = 10
PACE_V5_WARMUP_THROTTLE_PROBABILITY = 0.8
PACE_V5_WARMUP_MIN_THROTTLE = 0.25
SENSOR_ONLY_WARMUP_ACTION_HOLD_INTERACTIONS = 10
SENSOR_ONLY_WARMUP_STEERING_LIMIT = 0.3
FINE_TUNE_TRAIN_FREQUENCY = 8
FINE_TUNE_REPLAY_WARMUP_SIZE = 30_000
FINE_TUNE_CRITIC_WARMUP_UPDATES = 2_000
STEERING_RATE_V4_TOTAL_TIMESTEPS = 1_280
STEERING_RATE_V4_TRAIN_FREQUENCY = 32
STEERING_RATE_V4_ACTOR_LEARNING_RATE = 1e-6
STEERING_RATE_V4_CRITIC_WARMUP_UPDATES = 5_000
STEERING_RATE_V4_MAX_ACTOR_UPDATES = 10
STEERING_RATE_V4_EVALUATION_REPEATS = 10
STEERING_RATE_V4_PROMOTION_MEDIAN_SECONDS = 103.953
EPISODE_COLUMNS = (
    "mode",
    "policy_controlled",
    "global_step",
    "episode",
    "steps",
    "reward",
    "distance_m",
    "max_speed_kmh",
    "average_speed_kmh",
    "off_track_steps",
    "damage_delta",
    "laps_completed",
    "best_lap_time_seconds",
    "termination_reason",
)
UPDATE_COLUMNS = (
    "global_step",
    "optimizer_steps",
    "actor_updates",
    "replay_size",
    "critic_loss",
    "actor_loss",
    "q1_mean",
    "q2_mean",
    "target_q_mean",
)


@dataclass(frozen=True)
class EvaluationScore:
    completion_rate: float
    median_lap_time_seconds: float | None
    fastest_lap_time_seconds: float | None
    median_distance_m: float
    mean_off_track_steps: float
    mean_damage_delta: float


@dataclass(frozen=True)
class EvaluationPromotionGate:
    required_completion_rate: float
    maximum_median_lap_time_seconds: float

    def permits(self, score: EvaluationScore) -> bool:
        lap_time = score.median_lap_time_seconds
        return (
            score.completion_rate >= self.required_completion_rate
            and lap_time is not None
            and lap_time < self.maximum_median_lap_time_seconds
        )


def summarise_evaluations(
    evaluations: Iterable[EpisodeSummary],
) -> EvaluationScore:
    values = list(evaluations)
    if not values:
        raise ValueError("at least one evaluation is required")
    completed = [summary for summary in values if summary.laps_completed > 0]
    completed_lap_times = [
        float(summary.best_lap_time_seconds)
        for summary in completed
        if summary.best_lap_time_seconds is not None
    ]
    clean_lap_times = [
        float(summary.best_lap_time_seconds)
        for summary in completed
        if summary.best_lap_time_seconds is not None
        and summary.off_track_steps == 0
        and summary.damage_delta <= 0.0
    ]
    return EvaluationScore(
        completion_rate=len(completed) / len(values),
        median_lap_time_seconds=(
            None
            if not completed_lap_times
            else float(statistics.median(completed_lap_times))
        ),
        fastest_lap_time_seconds=(
            None if not clean_lap_times else min(clean_lap_times)
        ),
        median_distance_m=float(
            statistics.median(summary.distance_m for summary in values)
        ),
        mean_off_track_steps=float(
            statistics.fmean(summary.off_track_steps for summary in values)
        ),
        mean_damage_delta=float(
            statistics.fmean(summary.damage_delta for summary in values)
        ),
    )


def current_best_evaluation_score(
    learner: NstepTd3Learner,
) -> EvaluationScore | None:
    completion_rate = getattr(
        learner,
        "best_evaluation_completion_rate",
        None,
    )
    if completion_rate is None:
        return None
    return EvaluationScore(
        completion_rate=float(completion_rate),
        median_lap_time_seconds=getattr(
            learner,
            "best_evaluation_median_lap_time_seconds",
            None,
        ),
        fastest_lap_time_seconds=getattr(
            learner,
            "best_lap_time_seconds",
            None,
        ),
        median_distance_m=float(learner.best_evaluation_median_m or 0.0),
        mean_off_track_steps=float(
            getattr(learner, "best_evaluation_mean_off_track_steps", 0.0)
            or 0.0
        ),
        mean_damage_delta=float(
            getattr(learner, "best_evaluation_mean_damage_delta", 0.0)
            or 0.0
        ),
    )


def evaluation_score_is_better(
    candidate: EvaluationScore,
    incumbent: EvaluationScore | None,
) -> bool:
    if incumbent is None:
        return True
    if not math.isclose(candidate.completion_rate, incumbent.completion_rate):
        return candidate.completion_rate > incumbent.completion_rate

    if candidate.completion_rate > 0.0:
        candidate_lap = candidate.median_lap_time_seconds
        incumbent_lap = incumbent.median_lap_time_seconds
        if candidate_lap is not None and incumbent_lap is None:
            return True
        if candidate_lap is None and incumbent_lap is not None:
            return False
        if (
            candidate_lap is not None
            and incumbent_lap is not None
            and not math.isclose(candidate_lap, incumbent_lap)
        ):
            return candidate_lap < incumbent_lap
    elif not math.isclose(
        candidate.median_distance_m,
        incumbent.median_distance_m,
    ):
        return candidate.median_distance_m > incumbent.median_distance_m

    if not math.isclose(
        candidate.mean_off_track_steps,
        incumbent.mean_off_track_steps,
    ):
        return candidate.mean_off_track_steps < incumbent.mean_off_track_steps
    if not math.isclose(
        candidate.mean_damage_delta,
        incumbent.mean_damage_delta,
    ):
        return candidate.mean_damage_delta < incumbent.mean_damage_delta
    return candidate.median_distance_m > incumbent.median_distance_m


def record_best_evaluation_score(
    learner: NstepTd3Learner,
    score: EvaluationScore,
) -> None:
    learner.best_evaluation_completion_rate = score.completion_rate
    learner.best_evaluation_median_lap_time_seconds = (
        score.median_lap_time_seconds
    )
    learner.best_evaluation_median_m = score.median_distance_m
    learner.best_evaluation_mean_off_track_steps = score.mean_off_track_steps
    learner.best_evaluation_mean_damage_delta = score.mean_damage_delta


def pace_lap_time_is_better(
    candidate_lap_time_seconds: float | None,
    incumbent_lap_time_seconds: float | None,
) -> bool:
    if (
        candidate_lap_time_seconds is None
        or not math.isfinite(candidate_lap_time_seconds)
        or candidate_lap_time_seconds <= 0.0
    ):
        return False
    return (
        incumbent_lap_time_seconds is None
        or candidate_lap_time_seconds < incumbent_lap_time_seconds
    )


def pace_score_is_better(
    candidate: EvaluationScore,
    incumbent_lap_time_seconds: float | None,
) -> bool:
    return pace_lap_time_is_better(
        candidate.fastest_lap_time_seconds,
        incumbent_lap_time_seconds,
    )


def synchronise_recorded_pace(
    learner: NstepTd3Learner,
    model_dir: Path,
) -> None:
    pace_path = model_dir / "best_pace.pt"
    if not pace_path.is_file():
        return
    checkpoint = load_checkpoint(pace_path, device="cpu")
    recorded = checkpoint.get("best_lap_time_seconds")
    if recorded is None:
        return
    recorded_lap = float(recorded)
    if pace_lap_time_is_better(
        recorded_lap,
        learner.best_lap_time_seconds,
    ):
        learner.best_lap_time_seconds = recorded_lap


def synchronise_recorded_evaluation(
    learner: NstepTd3Learner,
    model_dir: Path,
) -> None:
    """Protect an existing same-protocol champion across independent runs."""
    evaluation_path = model_dir / "best_evaluation.pt"
    if not evaluation_path.is_file():
        return
    checkpoint = load_checkpoint(evaluation_path, device="cpu")
    completion_rate = checkpoint.get("best_evaluation_completion_rate")
    if completion_rate is None:
        return
    median_lap_time = checkpoint.get(
        "best_evaluation_median_lap_time_seconds"
    )
    fastest_lap_time = checkpoint.get("best_lap_time_seconds")
    incumbent = EvaluationScore(
        completion_rate=float(completion_rate),
        median_lap_time_seconds=(
            None if median_lap_time is None else float(median_lap_time)
        ),
        fastest_lap_time_seconds=(
            None if fastest_lap_time is None else float(fastest_lap_time)
        ),
        median_distance_m=float(
            checkpoint.get("best_evaluation_median_m") or 0.0
        ),
        mean_off_track_steps=float(
            checkpoint.get("best_evaluation_mean_off_track_steps") or 0.0
        ),
        mean_damage_delta=float(
            checkpoint.get("best_evaluation_mean_damage_delta") or 0.0
        ),
    )
    if evaluation_score_is_better(
        incumbent,
        current_best_evaluation_score(learner),
    ):
        record_best_evaluation_score(learner, incumbent)
    learner.best_evaluation_distance_m = max(
        learner.best_evaluation_distance_m,
        float(checkpoint.get("best_evaluation_distance_m", 0.0)),
    )


class CsvRunLogger:
    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._episode_file = (run_dir / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._update_file = (run_dir / "updates.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._episode_writer = csv.DictWriter(
            self._episode_file,
            fieldnames=EPISODE_COLUMNS,
        )
        self._update_writer = csv.DictWriter(
            self._update_file,
            fieldnames=UPDATE_COLUMNS,
        )
        self._episode_writer.writeheader()
        self._update_writer.writeheader()

    def write_episode(
        self,
        summary: EpisodeSummary,
        *,
        mode: str,
        policy_controlled: bool,
        global_step: int,
    ) -> None:
        self._episode_writer.writerow(
            {
                "mode": mode,
                "policy_controlled": policy_controlled,
                "global_step": global_step,
                **summary.as_dict(),
            }
        )
        self._episode_file.flush()

    def write_updates(
        self,
        learner: NstepTd3Learner,
        metrics: list[UpdateMetrics],
    ) -> None:
        if not metrics:
            return
        actor_losses = [
            metric.actor_loss
            for metric in metrics
            if metric.actor_loss is not None
        ]
        self._update_writer.writerow(
            {
                "global_step": learner.environment_steps,
                "optimizer_steps": learner.optimizer_steps,
                "actor_updates": learner.actor_updates,
                "replay_size": len(learner.replay),
                "critic_loss": statistics.fmean(
                    metric.critic_loss for metric in metrics
                ),
                "actor_loss": (
                    statistics.fmean(actor_losses) if actor_losses else ""
                ),
                "q1_mean": statistics.fmean(metric.q1_mean for metric in metrics),
                "q2_mean": statistics.fmean(metric.q2_mean for metric in metrics),
                "target_q_mean": statistics.fmean(
                    metric.target_q_mean for metric in metrics
                ),
            }
        )
        self._update_file.flush()

    def close(self) -> None:
        self._episode_file.close()
        self._update_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an independent N-step TD3 racing policy."
    )
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--track", default="g-track-3")
    parser.add_argument("--racing-line-path", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run seed (default: 0). When explicitly supplied during a resume, "
            "starts fresh action and target-noise streams without changing weights."
        ),
    )
    parser.add_argument("--device", default="auto")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", type=Path)
    resume_group.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume the protected best-evaluation checkpoint with fresh replay.",
    )
    resume_group.add_argument(
        "--resume-pace",
        action="store_true",
        help="Resume the fastest clean evaluation-lap checkpoint with fresh replay.",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help=(
            "Refine a resumed policy with a conservative actor learning rate, "
            "exploration noise 0.01, and policy delay 4; requires a "
            "resume checkpoint."
        ),
    )
    parser.add_argument(
        "--steering-rate-v4",
        action="store_true",
        help=(
            "Initialize a v4 learner from --resume-best actor weights only, "
            "with a fresh critic/replay and steering-rate reward cost."
        ),
    )
    parser.add_argument(
        "--pace-v5",
        action="store_true",
        help=(
            "Train a fresh v5 pace policy with bounded physical steering and "
            "a minimal progress reward. Resume checkpoints are intentionally "
            "not accepted."
        ),
    )
    parser.add_argument(
        "--sensor-only",
        action="store_true",
        help=(
            "Train the separate Agent 8 policy from TORCS telemetry and track "
            "sensors without loading racing-line geometry."
        ),
    )
    parser.add_argument(
        "--sensor-steering-v2",
        action="store_true",
        help=(
            "Create an isolated Agent 8 v2 continuation from a sensor-only "
            "actor with a reduced physical steering scale and fresh critic/replay."
        ),
    )
    parser.add_argument(
        "--sensor-steering-limit",
        type=float,
        default=SENSOR_STEERING_V2_LIMIT,
        help=(
            "Absolute TORCS steering scale for --sensor-steering-v2 "
            f"(default: {SENSOR_STEERING_V2_LIMIT:g})."
        ),
    )
    parser.add_argument(
        "--sensor-clean-observation-v3",
        action="store_true",
        help=(
            "Create an isolated Agent 8 v3 continuation from a sensor-only "
            "actor with a fresh critic/replay and app-matched clean observations."
        ),
    )
    parser.add_argument(
        "--sensor-v1-pace-continuation",
        action="store_true",
        help=(
            "Resume a complete Agent 8 V1 pace checkpoint with its original "
            "sensor-only learning contract in isolated output directories."
        ),
    )
    parser.add_argument(
        "--sensor-v1-stability-continuation",
        action="store_true",
        help=(
            "Refine a complete Agent 8 pace checkpoint with small actor updates "
            "and passive completion-first evaluation in isolated directories."
        ),
    )
    parser.add_argument(
        "--sensor-v1-clean-reliability-continuation",
        action="store_true",
        help=(
            "Transfer an Agent 8 pace actor into an isolated, app-matched clean "
            "sensor run that promotes only reliable sub-90-second policies."
        ),
    )
    parser.add_argument(
        "--sensor-v1-robust-pace-continuation",
        action="store_true",
        help=(
            "Transfer an Agent 8 pace actor into a fresh critic/replay run "
            "with mostly clean training episodes, a small fixed noisy-episode "
            "mix, and clean-only promotion."
        ),
    )
    parser.add_argument(
        "--pace-steering-limit",
        type=float,
        default=DEFAULT_PACE_STEERING_LIMIT,
        help=(
            "Absolute TORCS steering limit used by --pace-v5 "
            f"(default: {DEFAULT_PACE_STEERING_LIMIT:g})."
        ),
    )
    parser.add_argument(
        "--fine-tune-actor-learning-rate",
        type=float,
        default=None,
        help=(
            "Actor learning rate used by conservative modes "
            f"(default: {FINE_TUNE_ACTOR_LEARNING_RATE:g}, "
            f"{SENSOR_STEERING_V2_ACTOR_LEARNING_RATE:g} for Agent 8 v2, or "
            f"{SENSOR_CLEAN_OBSERVATION_V3_ACTOR_LEARNING_RATE:g} for Agent 8 v3, or "
            f"{STEERING_RATE_V4_ACTOR_LEARNING_RATE:g} for v4)."
        ),
    )
    parser.add_argument(
        "--fine-tune-replay-warmup-size",
        type=int,
        default=FINE_TUNE_REPLAY_WARMUP_SIZE,
        help=(
            "Fresh replay transitions collected with a fixed resumed actor "
            "before pace fine-tuning (default: 30000)."
        ),
    )
    parser.add_argument(
        "--fine-tune-critic-warmup-updates",
        type=int,
        default=None,
        help=(
            "Critic-only gradient updates before pace actor updates "
            f"(default: {FINE_TUNE_CRITIC_WARMUP_UPDATES}, or "
            f"{STEERING_RATE_V4_CRITIC_WARMUP_UPDATES} for v4)."
        ),
    )
    parser.add_argument(
        "--steering-rate-cost-coefficient",
        type=float,
        default=DEFAULT_STEERING_RATE_COST_COEFFICIENT,
        help=(
            "Squared normalized steering-change cost used by v4 "
            f"(default: {DEFAULT_STEERING_RATE_COST_COEFFICIENT:g})."
        ),
    )
    parser.add_argument(
        "--steering-rate-max-actor-updates",
        type=int,
        default=STEERING_RATE_V4_MAX_ACTOR_UPDATES,
        help=(
            "Hard ceiling on v4 actor updates; critic updates continue after "
            f"the ceiling (default: {STEERING_RATE_V4_MAX_ACTOR_UPDATES})."
        ),
    )
    parser.add_argument(
        "--steering-rate-promotion-median-seconds",
        type=float,
        default=STEERING_RATE_V4_PROMOTION_MEDIAN_SECONDS,
        help=(
            "Strict v4 median-lap promotion threshold "
            f"(default: {STEERING_RATE_V4_PROMOTION_MEDIAN_SECONDS:g}s)."
        ),
    )
    parser.add_argument("--replay-buffer-path", type=Path)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=25_000)
    parser.add_argument(
        "--observation-noise-std",
        type=float,
        default=0.025,
        help=(
            "Observation-noise profile level: 0.025 selects torcsRL's "
            "feature-specific reference profile; 0 disables noise."
        ),
    )
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument(
        "--evaluation-repeats",
        type=int,
        default=None,
        help="Deterministic rollouts used for passive median model selection.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--replay-start-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.992)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument(
        "--longitudinal-exploration-noise",
        type=float,
        default=0.1,
        help=(
            "Training-only noise for the signed throttle/brake action "
            "(default: 0.1, matching steering and torcsRL)."
        ),
    )
    parser.add_argument("--target-policy-noise", type=float, default=0.5)
    parser.add_argument("--target-noise-clip", type=float, default=0.2)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=None,
        help=(
            "Run gradient batches every N simulator interactions "
            "(default: 4, 8 with --fine-tune, or 32 for v4)."
        ),
    )
    parser.add_argument("--manual-start", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_modes = sum(
        bool(value)
        for value in (
            args.fine_tune,
            args.steering_rate_v4,
            args.pace_v5,
        )
    )
    if selected_modes > 1 or (args.sensor_only and selected_modes > 0):
        parser.error(
            "--fine-tune, --steering-rate-v4, --pace-v5, and --sensor-only "
            "are mutually exclusive"
        )
    if args.sensor_steering_v2 and not args.sensor_only:
        parser.error("--sensor-steering-v2 requires --sensor-only")
    if args.sensor_clean_observation_v3 and not args.sensor_only:
        parser.error("--sensor-clean-observation-v3 requires --sensor-only")
    if args.sensor_v1_pace_continuation and not args.sensor_only:
        parser.error("--sensor-v1-pace-continuation requires --sensor-only")
    if args.sensor_v1_stability_continuation and not args.sensor_only:
        parser.error("--sensor-v1-stability-continuation requires --sensor-only")
    if args.sensor_v1_clean_reliability_continuation and not args.sensor_only:
        parser.error(
            "--sensor-v1-clean-reliability-continuation requires --sensor-only"
        )
    if args.sensor_v1_robust_pace_continuation and not args.sensor_only:
        parser.error(
            "--sensor-v1-robust-pace-continuation requires --sensor-only"
        )
    if args.sensor_steering_v2 and args.sensor_clean_observation_v3:
        parser.error(
            "--sensor-steering-v2 and --sensor-clean-observation-v3 "
            "are mutually exclusive"
        )
    if args.sensor_v1_pace_continuation and (
        args.sensor_steering_v2
        or args.sensor_clean_observation_v3
        or args.sensor_v1_robust_pace_continuation
    ):
        parser.error(
            "--sensor-v1-pace-continuation cannot be combined with Agent 8 "
            "v2 or v3 experiment modes"
        )
    if args.sensor_v1_stability_continuation and (
        args.sensor_steering_v2
        or args.sensor_clean_observation_v3
        or args.sensor_v1_pace_continuation
        or args.sensor_v1_clean_reliability_continuation
        or args.sensor_v1_robust_pace_continuation
    ):
        parser.error(
            "--sensor-v1-stability-continuation cannot be combined with other "
            "Agent 8 experiment modes"
        )
    if args.sensor_v1_clean_reliability_continuation and (
        args.sensor_steering_v2
        or args.sensor_clean_observation_v3
        or args.sensor_v1_pace_continuation
        or args.sensor_v1_stability_continuation
        or args.sensor_v1_robust_pace_continuation
    ):
        parser.error(
            "--sensor-v1-clean-reliability-continuation cannot be combined with "
            "other Agent 8 experiment modes"
        )
    if args.sensor_v1_robust_pace_continuation and (
        args.sensor_steering_v2
        or args.sensor_clean_observation_v3
        or args.sensor_v1_pace_continuation
        or args.sensor_v1_stability_continuation
        or args.sensor_v1_clean_reliability_continuation
    ):
        parser.error(
            "--sensor-v1-robust-pace-continuation cannot be combined with "
            "other Agent 8 experiment modes"
        )
    if args.total_timesteps is None:
        args.total_timesteps = (
            STEERING_RATE_V4_TOTAL_TIMESTEPS
            if args.steering_rate_v4
            else PACE_V5_TOTAL_TIMESTEPS
            if args.pace_v5
            else 500_000
        )
    if args.evaluation_interval is None:
        args.evaluation_interval = (
            PACE_V5_EVALUATION_INTERVAL if args.pace_v5 else 10_000
        )
    if args.evaluation_repeats is None:
        args.evaluation_repeats = (
            STEERING_RATE_V4_EVALUATION_REPEATS
            if args.steering_rate_v4
            else SENSOR_V1_CLEAN_RELIABILITY_EVALUATION_REPEATS
            if (
                args.sensor_v1_clean_reliability_continuation
                or args.sensor_v1_robust_pace_continuation
            )
            else 3
        )
    if args.fine_tune_actor_learning_rate is None:
        args.fine_tune_actor_learning_rate = (
            STEERING_RATE_V4_ACTOR_LEARNING_RATE
            if args.steering_rate_v4
            else SENSOR_STEERING_V2_ACTOR_LEARNING_RATE
            if args.sensor_steering_v2
            else SENSOR_CLEAN_OBSERVATION_V3_ACTOR_LEARNING_RATE
            if args.sensor_clean_observation_v3
            else SENSOR_V1_CLEAN_RELIABILITY_ACTOR_LEARNING_RATE
            if args.sensor_v1_clean_reliability_continuation
            else SENSOR_V1_ROBUST_PACE_ACTOR_LEARNING_RATE
            if args.sensor_v1_robust_pace_continuation
            else FINE_TUNE_ACTOR_LEARNING_RATE
        )
    if args.fine_tune_critic_warmup_updates is None:
        args.fine_tune_critic_warmup_updates = (
            STEERING_RATE_V4_CRITIC_WARMUP_UPDATES
            if args.steering_rate_v4
            else FINE_TUNE_CRITIC_WARMUP_UPDATES
        )
    args.seed_was_explicit = args.seed is not None
    if args.seed is None:
        args.seed = 0
    elif args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.train_frequency is None:
        if args.steering_rate_v4:
            args.train_frequency = STEERING_RATE_V4_TRAIN_FREQUENCY
        elif (
            args.fine_tune
            or args.sensor_steering_v2
            or args.sensor_clean_observation_v3
            or args.sensor_v1_stability_continuation
            or args.sensor_v1_clean_reliability_continuation
            or args.sensor_v1_robust_pace_continuation
        ):
            args.train_frequency = FINE_TUNE_TRAIN_FREQUENCY
        else:
            args.train_frequency = DEFAULT_TRAIN_FREQUENCY
    positive = (
        "total_timesteps",
        "max_episode_steps",
        "evaluation_interval",
        "evaluation_repeats",
        "checkpoint_interval",
        "replay_capacity",
        "replay_start_size",
        "batch_size",
        "n_steps",
        "policy_delay",
        "gradient_steps",
        "train_frequency",
        "steering_rate_max_actor_updates",
    )
    for name in positive:
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    non_negative = (
        "fine_tune_replay_warmup_size",
        "fine_tune_critic_warmup_updates",
    )
    for name in non_negative:
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if (
        not math.isfinite(args.fine_tune_actor_learning_rate)
        or args.fine_tune_actor_learning_rate <= 0.0
    ):
        parser.error(
            "--fine-tune-actor-learning-rate must be positive and finite"
        )
    if args.resume_best:
        args.resume = (
            SENSOR_ONLY_BEST_EVALUATION_PATH
            if args.sensor_only
            else BEST_EVALUATION_PATH
        )
    elif args.resume_pace:
        args.resume = (
            SENSOR_ONLY_BEST_PACE_PATH
            if args.sensor_only
            else BEST_PACE_PATH
        )
    if args.fine_tune and args.resume is None:
        parser.error(
            "--fine-tune requires --resume, --resume-best, or --resume-pace"
        )
    if args.steering_rate_v4 and not args.resume_best:
        parser.error("--steering-rate-v4 requires --resume-best")
    if args.steering_rate_v4 and args.evaluation_repeats < 10:
        parser.error("--steering-rate-v4 requires at least 10 evaluation repeats")
    if args.pace_v5 and args.resume is not None:
        parser.error("--pace-v5 starts from fresh weights and cannot resume")
    if args.pace_v5 and args.replay_buffer_path is not None:
        parser.error("--pace-v5 starts with fresh replay and cannot import replay")
    if args.sensor_only and args.racing_line_path is not None:
        parser.error("--sensor-only cannot accept --racing-line-path")
    if (
        not math.isfinite(args.pace_steering_limit)
        or not 0.0 < args.pace_steering_limit < 1.0
    ):
        parser.error("--pace-steering-limit must be finite and in (0, 1)")
    if (
        not math.isfinite(args.sensor_steering_limit)
        or not 0.0 < args.sensor_steering_limit < 1.0
    ):
        parser.error("--sensor-steering-limit must be finite and in (0, 1)")
    if (
        not math.isfinite(args.steering_rate_cost_coefficient)
        or args.steering_rate_cost_coefficient <= 0.0
    ):
        parser.error(
            "--steering-rate-cost-coefficient must be positive and finite"
        )
    if (
        not math.isfinite(args.steering_rate_promotion_median_seconds)
        or args.steering_rate_promotion_median_seconds <= 0.0
    ):
        parser.error(
            "--steering-rate-promotion-median-seconds must be positive and finite"
        )
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"--resume checkpoint does not exist: {args.resume}")
    if args.sensor_steering_v2 and args.resume is None:
        parser.error("--sensor-steering-v2 requires a --resume checkpoint")
    if args.sensor_clean_observation_v3 and args.resume is None:
        parser.error(
            "--sensor-clean-observation-v3 requires a --resume checkpoint"
        )
    if args.sensor_v1_pace_continuation and args.resume is None:
        parser.error("--sensor-v1-pace-continuation requires a --resume checkpoint")
    if args.sensor_v1_stability_continuation and args.resume is None:
        parser.error(
            "--sensor-v1-stability-continuation requires a --resume checkpoint"
        )
    if args.sensor_v1_clean_reliability_continuation and args.resume is None:
        parser.error(
            "--sensor-v1-clean-reliability-continuation requires a --resume "
            "checkpoint"
        )
    if args.sensor_v1_robust_pace_continuation and args.resume is None:
        parser.error(
            "--sensor-v1-robust-pace-continuation requires a --resume "
            "checkpoint"
        )
    if args.racing_line_path is not None and not args.racing_line_path.is_file():
        parser.error(
            f"--racing-line-path does not exist: {args.racing_line_path}"
        )
    if (
        not math.isfinite(args.observation_noise_std)
        or args.observation_noise_std < 0.0
    ):
        parser.error("--observation-noise-std must be finite and non-negative")
    if args.sensor_clean_observation_v3 and args.observation_noise_std != 0.0:
        parser.error(
            "--sensor-clean-observation-v3 requires "
            "--observation-noise-std 0"
        )
    if (
        args.sensor_v1_pace_continuation
        and args.observation_noise_std != 0.025
    ):
        parser.error(
            "--sensor-v1-pace-continuation requires "
            "--observation-noise-std 0.025"
        )
    if (
        args.sensor_v1_stability_continuation
        and args.observation_noise_std != 0.025
    ):
        parser.error(
            "--sensor-v1-stability-continuation requires "
            "--observation-noise-std 0.025"
        )
    if (
        args.sensor_v1_clean_reliability_continuation
        and args.observation_noise_std != 0.0
    ):
        parser.error(
            "--sensor-v1-clean-reliability-continuation requires "
            "--observation-noise-std 0"
        )
    if (
        args.sensor_v1_robust_pace_continuation
        and args.observation_noise_std != 0.0
    ):
        parser.error(
            "--sensor-v1-robust-pace-continuation requires "
            "--observation-noise-std 0"
        )
    if (
        args.sensor_v1_clean_reliability_continuation
        and args.evaluation_repeats
        < SENSOR_V1_CLEAN_RELIABILITY_EVALUATION_REPEATS
    ):
        parser.error(
            "--sensor-v1-clean-reliability-continuation requires at least "
            f"{SENSOR_V1_CLEAN_RELIABILITY_EVALUATION_REPEATS} evaluation "
            "repeats"
        )
    if (
        args.sensor_v1_robust_pace_continuation
        and args.evaluation_repeats
        < SENSOR_V1_ROBUST_PACE_EVALUATION_REPEATS
    ):
        parser.error(
            "--sensor-v1-robust-pace-continuation requires at least "
            f"{SENSOR_V1_ROBUST_PACE_EVALUATION_REPEATS} evaluation repeats"
        )
    if (
        args.replay_buffer_path is not None
        and not args.replay_buffer_path.is_file()
    ):
        parser.error(
            "--replay-buffer-path does not exist: "
            f"{args.replay_buffer_path}"
        )
    return args


def config_from_args(args: argparse.Namespace) -> NstepTd3Config:
    return NstepTd3Config(
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        n_steps=args.n_steps,
        gamma=args.gamma,
        tau=args.tau,
        replay_capacity=args.replay_capacity,
        replay_start_size=args.replay_start_size,
        batch_size=args.batch_size,
        exploration_noise=args.exploration_noise,
        longitudinal_exploration_noise=args.longitudinal_exploration_noise,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        policy_delay=args.policy_delay,
        gradient_steps_per_interaction=args.gradient_steps,
        steering_rate_cost_coefficient=(
            args.steering_rate_cost_coefficient
            if args.steering_rate_v4
            else 0.0
        ),
        physical_steering_limit=(
            args.pace_steering_limit if args.pace_v5 else 1.0
        ),
        progress_reward=bool(args.pace_v5),
        racing_line_features=not args.sensor_only,
        seed=args.seed,
    )


def initialize_steering_rate_v4_learner(
    args: argparse.Namespace,
) -> NstepTd3Learner:
    if not args.steering_rate_v4 or args.resume is None:
        raise ValueError("v4 actor transfer requires an explicit source checkpoint")
    source_checkpoint = load_actor_checkpoint(args.resume, device="cpu")
    if source_checkpoint.get("reward_version") != REWARD_VERSION:
        raise ValueError("v4 actor transfer requires a v3 reward checkpoint")

    source_config = NstepTd3Config(**source_checkpoint["config"])
    target_config = replace(
        source_config,
        actor_learning_rate=args.fine_tune_actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        exploration_noise=0.01,
        longitudinal_exploration_noise=0.01,
        policy_delay=4,
        gradient_steps_per_interaction=args.gradient_steps,
        steering_rate_cost_coefficient=(
            args.steering_rate_cost_coefficient
        ),
        seed=args.seed,
    )
    learner = NstepTd3Learner.from_actor_checkpoint(
        args.resume,
        config=target_config,
        device=args.device,
    )
    learner.enable_fine_tuning(
        actor_learning_rate=args.fine_tune_actor_learning_rate,
    )
    return learner


def initialize_sensor_actor_transfer_learner(
    args: argparse.Namespace,
    *,
    physical_steering_limit: float,
) -> NstepTd3Learner:
    """Transfer a sensor-policy actor while intentionally resetting TD3 state."""
    if args.resume is None:
        raise ValueError("sensor actor transfer requires a source checkpoint")
    source_checkpoint = load_actor_checkpoint(args.resume, device="cpu")
    source_config = NstepTd3Config(**source_checkpoint["config"])
    if source_config.racing_line_features:
        raise ValueError("sensor actor transfer requires a sensor-only checkpoint")

    target_config = replace(
        source_config,
        actor_learning_rate=args.fine_tune_actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        exploration_noise=min(source_config.exploration_noise, 0.05),
        longitudinal_exploration_noise=min(
            source_config.longitudinal_exploration_noise,
            0.05,
        ),
        physical_steering_limit=physical_steering_limit,
        seed=args.seed,
    )
    learner = NstepTd3Learner(target_config, device=args.device)
    learner.actor.load_state_dict(source_checkpoint["actor"])
    learner.actor_target.load_state_dict(source_checkpoint["actor"])
    return learner


def initialize_sensor_steering_v2_learner(
    args: argparse.Namespace,
) -> NstepTd3Learner:
    """Transfer only a sensor-policy actor into the scaled-action experiment."""
    if not args.sensor_steering_v2:
        raise ValueError("Agent 8 v2 transfer mode was not selected")
    return initialize_sensor_actor_transfer_learner(
        args,
        physical_steering_limit=args.sensor_steering_limit,
    )


def initialize_sensor_clean_observation_v3_learner(
    args: argparse.Namespace,
) -> NstepTd3Learner:
    """Transfer the actor into the GUI's unperturbed sensor contract."""
    if not args.sensor_clean_observation_v3:
        raise ValueError("Agent 8 clean-observation v3 mode was not selected")
    return initialize_sensor_actor_transfer_learner(
        args,
        physical_steering_limit=1.0,
    )


def initialize_sensor_v1_clean_reliability_learner(
    args: argparse.Namespace,
) -> NstepTd3Learner:
    """Reuse a pace actor while relearning value estimates on clean sensors."""
    if not args.sensor_v1_clean_reliability_continuation:
        raise ValueError("Agent 8 clean-reliability mode was not selected")
    return initialize_sensor_actor_transfer_learner(
        args,
        physical_steering_limit=1.0,
    )


def initialize_sensor_v1_robust_pace_learner(
    args: argparse.Namespace,
) -> NstepTd3Learner:
    """Reuse the pace actor while rebuilding TD3 state for robust deployment."""
    if not args.sensor_v1_robust_pace_continuation:
        raise ValueError("Agent 8 robust-pace mode was not selected")
    return initialize_sensor_actor_transfer_learner(
        args,
        physical_steering_limit=1.0,
    )


def evaluate_once(
    environment: NstepTorcsEnvironment,
    learner: NstepTd3Learner,
    *,
    cold_start: bool = False,
) -> EpisodeSummary:
    if cold_start:
        observation, _ = environment.reset(force_relaunch=True)
    else:
        observation, _ = environment.reset()
    done = False
    while not done:
        action = learner.select_action(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = environment.step(
            action
        )
        done = terminated or truncated
    summary = environment.last_summary
    if summary is None:
        raise RuntimeError(f"evaluation ended without a summary: {info}")
    return summary


def evaluate_and_record(
    environment: NstepTorcsEnvironment,
    learner: NstepTd3Learner,
    logger: CsvRunLogger,
    writer: Any,
    model_dir: Path,
    *,
    repeats: int,
    mode: str = "evaluation",
    allow_promotion: bool = True,
    promotion_gate: EvaluationPromotionGate | None = None,
    cold_start_each_rollout: bool = False,
    evaluation_candidate_dir: Path | None = None,
) -> tuple[list[EpisodeSummary], float]:
    if repeats < 1:
        raise ValueError("evaluation repeats must be positive")
    learner_config = getattr(learner, "config", None)
    has_racing_line = bool(
        getattr(learner_config, "racing_line_features", True)
    )
    agent_label = "Agent 7" if has_racing_line else "Agent 8"

    evaluations = [
        evaluate_once(
            environment,
            learner,
            cold_start=cold_start_each_rollout,
        )
        for _ in range(repeats)
    ]
    for evaluation in evaluations:
        logger.write_episode(
            evaluation,
            mode=mode,
            policy_controlled=True,
            global_step=learner.environment_steps,
        )

    distances = np.asarray(
        [evaluation.distance_m for evaluation in evaluations],
        dtype=np.float64,
    )
    median_distance = float(np.median(distances))
    maximum_distance = float(np.max(distances))
    minimum_distance = float(np.min(distances))
    score = summarise_evaluations(evaluations)
    completed_runs = sum(
        evaluation.laps_completed > 0 for evaluation in evaluations
    )
    learner.best_evaluation_distance_m = max(
        learner.best_evaluation_distance_m,
        maximum_distance,
    )
    if writer is not None:
        writer.add_scalar(
            f"{mode}/median_distance_m",
            median_distance,
            learner.environment_steps,
        )
        writer.add_scalar(
            f"{mode}/completion_rate",
            score.completion_rate,
            learner.environment_steps,
        )
        if score.median_lap_time_seconds is not None:
            writer.add_scalar(
                f"{mode}/median_lap_time_seconds",
                score.median_lap_time_seconds,
                learner.environment_steps,
            )
    lap_summary = (
        ""
        if score.median_lap_time_seconds is None
        else f" median_lap={score.median_lap_time_seconds:.2f}s"
    )
    print(
        f"Evaluation ({repeats} runs) | "
        f"median={median_distance:.1f}m "
        f"min={minimum_distance:.1f}m "
        f"max={maximum_distance:.1f}m "
        f"laps={completed_runs}/{repeats}{lap_summary}"
    )
    if evaluation_candidate_dir is not None:
        evaluation_candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = evaluation_candidate_dir / (
            f"{mode}_step_{learner.environment_steps:010d}.pt"
        )
        learner.save(candidate_path)
        _write_json(
            candidate_path.with_suffix(".json"),
            {
                "checkpoint": str(candidate_path),
                "global_step": learner.environment_steps,
                "mode": mode,
                "cold_start_each_rollout": cold_start_each_rollout,
                "completion_rate": score.completion_rate,
                "median_lap_time_seconds": score.median_lap_time_seconds,
                "fastest_lap_time_seconds": score.fastest_lap_time_seconds,
                "median_distance_m": score.median_distance_m,
                "minimum_distance_m": minimum_distance,
                "maximum_distance_m": maximum_distance,
            },
        )
        print(f"Saved evaluation candidate: {candidate_path}")
    promotion_permitted = (
        promotion_gate is None or promotion_gate.permits(score)
    )
    reliability_improved = (
        allow_promotion
        and promotion_permitted
        and evaluation_score_is_better(
            score,
            current_best_evaluation_score(learner),
        )
    )
    pace_improved = (
        allow_promotion
        and promotion_permitted
        and pace_score_is_better(
            score,
            learner.best_lap_time_seconds,
        )
    )
    if reliability_improved:
        record_best_evaluation_score(learner, score)
    if pace_improved:
        learner.best_lap_time_seconds = score.fastest_lap_time_seconds

    if reliability_improved:
        learner.save(model_dir / "best_evaluation.pt")
        pace_summary = (
            ""
            if score.median_lap_time_seconds is None
            else f", median_lap={score.median_lap_time_seconds:.2f}s"
        )
        print(
            f"New {agent_label} evaluation champion: "
            f"completion={score.completion_rate:.0%}{pace_summary}, "
            f"median_distance={median_distance:.1f}m"
        )
    if pace_improved:
        learner.save(model_dir / "best_pace.pt")
        learner.save(model_dir / "best_lap.pt")
        print(
            f"New {agent_label} pace champion: fastest clean evaluation lap="
            f"{score.fastest_lap_time_seconds:.3f}s"
        )
    elif allow_promotion and not promotion_permitted:
        print(
            "Evaluation candidate rejected by the strict promotion gate: "
            f"completion={score.completion_rate:.0%}, "
            "median_lap="
            f"{score.median_lap_time_seconds}, required completion="
            f"{promotion_gate.required_completion_rate:.0%}, median below "
            f"{promotion_gate.maximum_median_lap_time_seconds:.3f}s."
        )
    return evaluations, median_distance


def collect_replay_warmup(
    environment: NstepTorcsEnvironment,
    learner: NstepTd3Learner,
    accumulator: NstepTransitionAccumulator,
    logger: CsvRunLogger,
    writer: Any,
    *,
    use_random_actions: bool,
    deterministic_policy_actions: bool = False,
    target_replay_size: int | None = None,
    random_action_hold_interactions: int = 1,
    launch_capable_random_actions: bool = False,
    random_action_steering_limit: float | None = None,
    before_episode_reset: Callable[[], None] | None = None,
) -> tuple[np.ndarray, int]:
    if random_action_hold_interactions < 1:
        raise ValueError("random action hold interactions must be positive")
    if random_action_steering_limit is not None and (
        not np.isfinite(random_action_steering_limit)
        or not 0.0 < random_action_steering_limit <= 1.0
    ):
        raise ValueError("random action steering limit must be in (0, 1]")
    target_size = (
        learner.config.replay_start_size
        if target_replay_size is None
        else max(learner.config.replay_start_size, int(target_replay_size))
    )
    if target_size > learner.config.replay_capacity:
        raise ValueError("replay warm-up target exceeds replay capacity")
    if before_episode_reset is not None:
        before_episode_reset()
    observation, _ = environment.reset()
    accumulator.reset()
    interactions = 0
    held_random_action: np.ndarray | None = None
    held_random_action_steps = 0
    while len(learner.replay) < target_size:
        if use_random_actions:
            if held_random_action is None or held_random_action_steps == 0:
                held_random_action = learner.random_action()
                if launch_capable_random_actions:
                    held_random_action = pace_v5_warmup_action(
                        held_random_action
                    )
                if random_action_steering_limit is not None:
                    held_random_action[0] = np.clip(
                        held_random_action[0],
                        -random_action_steering_limit,
                        random_action_steering_limit,
                    )
                held_random_action_steps = random_action_hold_interactions
            action = held_random_action.copy()
            held_random_action_steps -= 1
        else:
            action = learner.select_action(
                observation,
                deterministic=deterministic_policy_actions,
            )
        next_observation, reward, terminated, truncated, _info = (
            environment.step(action)
        )
        episode_end = terminated or truncated
        learner.add_transitions(
            accumulator.append(
                observation,
                action,
                reward,
                next_observation,
                terminated=terminated,
                episode_end=episode_end,
            )
        )
        interactions += 1
        observation = next_observation
        if not episode_end:
            continue
        summary = environment.last_summary
        if summary is None:
            raise RuntimeError("replay warm-up ended without an episode summary")
        logger.write_episode(
            summary,
            mode="replay_warmup",
            policy_controlled=not use_random_actions,
            global_step=learner.environment_steps,
        )
        _record_tensorboard_episode(
            writer,
            summary,
            learner.environment_steps,
            "replay_warmup",
        )
        print(
            f"Replay warm-up {len(learner.replay):,}/"
            f"{target_size:,} | "
            f"{summary.termination_reason} {summary.distance_m:.1f}m"
        )
        if before_episode_reset is not None:
            before_episode_reset()
        observation, _ = environment.reset()
        accumulator.reset()
        held_random_action = None
        held_random_action_steps = 0
    return observation, interactions


def pace_v5_warmup_action(random_action: np.ndarray) -> np.ndarray:
    """Bias held random segments toward launch without prescribing a policy."""
    action = np.asarray(random_action, dtype=np.float32).reshape(-1).copy()
    if action.size < 2:
        raise ValueError("v5 warm-up requires steering and longitudinal actions")
    uniform_draw = (float(np.clip(action[1], -1.0, 1.0)) + 1.0) / 2.0
    brake_probability = 1.0 - PACE_V5_WARMUP_THROTTLE_PROBABILITY
    if uniform_draw < brake_probability:
        action[1] = -uniform_draw / brake_probability
    else:
        throttle_draw = (
            uniform_draw - brake_probability
        ) / PACE_V5_WARMUP_THROTTLE_PROBABILITY
        action[1] = PACE_V5_WARMUP_MIN_THROTTLE + throttle_draw * (
            1.0 - PACE_V5_WARMUP_MIN_THROTTLE
        )
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def select_sensor_v1_robust_pace_noise(
    random_source: np.random.Generator,
) -> float:
    """Choose the next training episode's observation-noise contract."""
    return (
        SENSOR_V1_ROBUST_PACE_NOISY_OBSERVATION_STD
        if random_source.random() < SENSOR_V1_ROBUST_PACE_NOISY_EPISODE_FRACTION
        else 0.0
    )


def _next_boundary(current: int, interval: int) -> int:
    return ((int(current) // int(interval)) + 1) * int(interval)


def should_run_gradient_updates(
    environment_steps: int,
    train_frequency: int,
) -> bool:
    """Return whether this interaction is scheduled to update TD3."""
    if train_frequency < 1:
        raise ValueError("train_frequency must be positive")
    return environment_steps > 0 and environment_steps % train_frequency == 0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _tensorboard_writer(run_dir: Path, disabled: bool):
    if disabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is unavailable; CSV logging remains enabled.")
        return None
    return SummaryWriter(log_dir=str(run_dir / "tensorboard"))


def _record_tensorboard_updates(writer: Any, metrics: Iterable[UpdateMetrics], step: int) -> None:
    if writer is None:
        return
    values = list(metrics)
    if not values:
        return
    writer.add_scalar(
        "train/critic_loss",
        statistics.fmean(item.critic_loss for item in values),
        step,
    )
    actor = [item.actor_loss for item in values if item.actor_loss is not None]
    if actor:
        writer.add_scalar("train/actor_loss", statistics.fmean(actor), step)
    writer.add_scalar(
        "train/q1_mean",
        statistics.fmean(item.q1_mean for item in values),
        step,
    )
    writer.add_scalar(
        "train/q2_mean",
        statistics.fmean(item.q2_mean for item in values),
        step,
    )


def _record_tensorboard_episode(writer: Any, summary: EpisodeSummary, step: int, mode: str) -> None:
    if writer is None:
        return
    writer.add_scalar(f"{mode}/reward", summary.reward, step)
    writer.add_scalar(f"{mode}/distance_m", summary.distance_m, step)
    writer.add_scalar(f"{mode}/average_speed_kmh", summary.average_speed_kmh, step)
    writer.add_scalar(f"{mode}/laps", summary.laps_completed, step)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    agent_label = (
        "Agent 8 v2"
        if args.sensor_steering_v2
        else "Agent 8 v3 clean observation"
        if args.sensor_clean_observation_v3
        else "Agent 8 v1 pace continuation"
        if args.sensor_v1_pace_continuation
        else "Agent 8 v1 stability continuation"
        if args.sensor_v1_stability_continuation
        else "Agent 8 v1 clean reliability continuation"
        if args.sensor_v1_clean_reliability_continuation
        else "Agent 8 v1 robust pace continuation"
        if args.sensor_v1_robust_pace_continuation
        else "Agent 8"
        if args.sensor_only
        else "Agent 7"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.sensor_steering_v2:
        run_root = SENSOR_STEERING_V2_RUNS_DIR
        model_dir = SENSOR_STEERING_V2_MODEL_DIR
    elif args.sensor_clean_observation_v3:
        run_root = SENSOR_CLEAN_OBSERVATION_V3_RUNS_DIR
        model_dir = SENSOR_CLEAN_OBSERVATION_V3_MODEL_DIR
    elif args.sensor_v1_pace_continuation:
        run_root = SENSOR_V1_CONTINUATION_RUNS_DIR
        model_dir = SENSOR_V1_CONTINUATION_MODEL_DIR
    elif args.sensor_v1_stability_continuation:
        run_root = SENSOR_V1_STABILITY_RUNS_DIR
        model_dir = SENSOR_V1_STABILITY_MODEL_DIR
    elif args.sensor_v1_clean_reliability_continuation:
        run_root = SENSOR_V1_CLEAN_RELIABILITY_RUNS_DIR
        model_dir = SENSOR_V1_CLEAN_RELIABILITY_MODEL_DIR
    elif args.sensor_v1_robust_pace_continuation:
        run_root = SENSOR_V1_ROBUST_PACE_RUNS_DIR
        model_dir = SENSOR_V1_ROBUST_PACE_MODEL_DIR
    elif args.sensor_only:
        run_root = SENSOR_ONLY_RUNS_DIR
        model_dir = SENSOR_ONLY_MODEL_DIR
    elif args.pace_v5:
        run_root = PACE_V5_RUNS_DIR
        model_dir = PACE_V5_MODEL_DIR
    elif args.steering_rate_v4:
        run_root = STEERING_RATE_V4_RUNS_DIR
        model_dir = STEERING_RATE_V4_MODEL_DIR
    else:
        run_root = RUNS_DIR
        model_dir = MODEL_DIR
    run_dir = run_root / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.sensor_steering_v2:
        learner = initialize_sensor_steering_v2_learner(args)
        print(
            "Initialized Agent 8 v2 from sensor-only actor parameters with "
            f"fresh critic/replay and physical steering limited to +/-"
            f"{learner.config.physical_steering_limit:g}: {args.resume}"
        )
    elif args.sensor_clean_observation_v3:
        learner = initialize_sensor_clean_observation_v3_learner(args)
        print(
            "Initialized Agent 8 v3 from sensor-only actor parameters with "
            f"fresh critic/replay and clean observations: {args.resume}"
        )
    elif args.sensor_v1_clean_reliability_continuation:
        learner = initialize_sensor_v1_clean_reliability_learner(args)
        print(
            "Initialized Agent 8 clean-reliability continuation from pace "
            "actor parameters only, with fresh critic/replay and app-matched "
            f"clean observations: {args.resume}"
        )
    elif args.sensor_v1_robust_pace_continuation:
        learner = initialize_sensor_v1_robust_pace_learner(args)
        print(
            "Initialized Agent 8 robust-pace continuation from pace actor "
            "parameters only, with fresh critic/replay, clean deployment "
            f"observations, and a fixed noisy-episode mix: {args.resume}"
        )
    elif args.steering_rate_v4:
        learner = initialize_steering_rate_v4_learner(args)
        print(
            "Initialized Agent 7 v4 from protected v3 actor parameters only: "
            f"{args.resume}. Critic, target critic, optimizers, replay, counters, "
            "and source evaluation records were not inherited."
        )
    elif args.resume is not None:
        learner = NstepTd3Learner.load(args.resume, device=args.device)
        print(f"Resuming {agent_label} from {args.resume}")
        if args.seed_was_explicit:
            learner.reseed_training_noise(args.seed)
            print(
                f"Reseeded Agent 7 training noise with seed {args.seed}; "
                "checkpoint weights and optimizer state are unchanged."
            )
        if args.fine_tune or args.sensor_v1_stability_continuation:
            learner.enable_fine_tuning(
                actor_learning_rate=args.fine_tune_actor_learning_rate,
            )
            print(
                f"Enabled {agent_label} conservative policy refinement: "
                "actor learning rate="
                f"{learner.config.actor_learning_rate:g}, action noise="
                f"{learner.config.exploration_noise:g}, policy delay="
                f"{learner.config.policy_delay}, train frequency="
                f"{args.train_frequency}."
            )
    else:
        learner = NstepTd3Learner(config_from_args(args), device=args.device)
        if args.sensor_only:
            print(
                "Starting Agent 8 sensor-only N-step TD3 from random network "
                "weights without a racing-line map."
            )
        elif args.pace_v5:
            print(
                "Starting Agent 7 v5 pace racer from fresh random network "
                "weights and fresh replay."
            )
        else:
            print("Starting Agent 7 N-step TD3 from random network weights.")
    synchronise_recorded_evaluation(learner, model_dir)
    synchronise_recorded_pace(learner, model_dir)
    if args.replay_buffer_path is not None:
        learner.load_replay(args.replay_buffer_path)
        print(
            f"Loaded {agent_label} replay buffer with "
            f"{len(learner.replay):,} transitions."
        )
    elif args.resume is not None:
        print(
            "Starting a fresh replay buffer for this continuation; "
            "the protected checkpoint remains unchanged unless evaluation improves."
        )

    conservative_mode = bool(
        args.fine_tune
        or args.steering_rate_v4
        or args.sensor_steering_v2
        or args.sensor_clean_observation_v3
        or args.sensor_v1_stability_continuation
        or args.sensor_v1_clean_reliability_continuation
        or args.sensor_v1_robust_pace_continuation
    )
    replay_warmup_size = learner.config.replay_start_size
    critic_warmup_updates = 0
    if conservative_mode:
        replay_warmup_size = max(
            learner.config.replay_start_size,
            args.fine_tune_replay_warmup_size,
        )
        critic_warmup_updates = args.fine_tune_critic_warmup_updates
    if (
        args.sensor_steering_v2
        or args.sensor_clean_observation_v3
        or conservative_mode
    ):
        replay_warmup_action_mode = "deterministic_policy"
    elif args.pace_v5 or args.sensor_only:
        replay_warmup_action_mode = "held_launch_biased_random"
    elif args.resume is None:
        replay_warmup_action_mode = "random"
    elif conservative_mode:
        replay_warmup_action_mode = "deterministic_policy"
    else:
        replay_warmup_action_mode = "noisy_policy"

    config_record = {
        "run_started": timestamp,
        "track": args.track,
        "requested_training_timesteps": args.total_timesteps,
        "resume": None if args.resume is None else str(args.resume),
        "run_seed": args.seed,
        "resume_training_noise_reseeded": bool(
            args.resume is not None and args.seed_was_explicit
        ),
        "fine_tune": bool(args.fine_tune),
        "steering_rate_v4": bool(args.steering_rate_v4),
        "pace_v5": bool(args.pace_v5),
        "sensor_only": bool(args.sensor_only),
        "sensor_steering_v2": bool(args.sensor_steering_v2),
        "sensor_steering_limit": (
            args.sensor_steering_limit if args.sensor_steering_v2 else None
        ),
        "sensor_clean_observation_v3": bool(
            args.sensor_clean_observation_v3
        ),
        "sensor_v1_pace_continuation": bool(args.sensor_v1_pace_continuation),
        "sensor_v1_stability_continuation": bool(
            args.sensor_v1_stability_continuation
        ),
        "sensor_v1_clean_reliability_continuation": bool(
            args.sensor_v1_clean_reliability_continuation
        ),
        "sensor_v1_robust_pace_continuation": bool(
            args.sensor_v1_robust_pace_continuation
        ),
        "actor_initialization": (
            "sensor_v2_actor_only"
            if args.sensor_steering_v2
            else "sensor_clean_observation_v3_actor_only"
            if args.sensor_clean_observation_v3
            else "sensor_v1_clean_reliability_actor_only"
            if args.sensor_v1_clean_reliability_continuation
            else "sensor_v1_robust_pace_actor_only"
            if args.sensor_v1_robust_pace_continuation
            else "v3_actor_only"
            if args.steering_rate_v4
            else "checkpoint_resume"
            if args.resume is not None
            else "random"
        ),
        "fine_tune_actor_learning_rate": (
            args.fine_tune_actor_learning_rate if conservative_mode else None
        ),
        "maximum_actor_updates": (
            args.steering_rate_max_actor_updates
            if args.steering_rate_v4
            else None
        ),
        "promotion_completion_rate": (
            1.0
            if args.steering_rate_v4
            else SENSOR_V1_CLEAN_RELIABILITY_COMPLETION_RATE
            if (
                args.sensor_v1_clean_reliability_continuation
                or args.sensor_v1_robust_pace_continuation
            )
            else None
        ),
        "promotion_median_lap_time_seconds": (
            args.steering_rate_promotion_median_seconds
            if args.steering_rate_v4
            else SENSOR_V1_CLEAN_RELIABILITY_MEDIAN_SECONDS
            if (
                args.sensor_v1_clean_reliability_continuation
                or args.sensor_v1_robust_pace_continuation
            )
            else None
        ),
        "replay_warmup_size": replay_warmup_size,
        "replay_warmup_action_mode": replay_warmup_action_mode,
        "replay_warmup_action_hold_interactions": (
            1
            if (
                args.sensor_steering_v2
                or args.sensor_clean_observation_v3
                or args.sensor_v1_clean_reliability_continuation
                or args.sensor_v1_robust_pace_continuation
            )
            else PACE_V5_WARMUP_ACTION_HOLD_INTERACTIONS
            if args.pace_v5
            else SENSOR_ONLY_WARMUP_ACTION_HOLD_INTERACTIONS
            if args.sensor_only
            else 1
        ),
        "replay_warmup_throttle_probability": (
            None
            if (
                args.sensor_steering_v2
                or args.sensor_clean_observation_v3
                or args.sensor_v1_clean_reliability_continuation
                or args.sensor_v1_robust_pace_continuation
            )
            else PACE_V5_WARMUP_THROTTLE_PROBABILITY
            if args.pace_v5 or args.sensor_only
            else None
        ),
        "replay_warmup_steering_limit": (
            None
            if (
                args.sensor_steering_v2
                or args.sensor_clean_observation_v3
                or args.sensor_v1_clean_reliability_continuation
                or args.sensor_v1_robust_pace_continuation
            )
            else SENSOR_ONLY_WARMUP_STEERING_LIMIT
            if args.sensor_only
            else None
        ),
        "critic_only_warmup_updates": critic_warmup_updates,
        "replay_source": (
            "fresh" if args.replay_buffer_path is None else str(args.replay_buffer_path)
        ),
        "device": str(learner.device),
        "algorithm": "N-step TD3",
        "model_family": learner.model_family,
        "observation_version": learner.observation_version,
        "action_version": learner.action_version,
        "reward_version": learner.reward_version,
        "source_design": "https://github.com/raphaelsenn/torcsRL",
        "source_revision_reviewed": "043e806e260507d1d5eef161004bc350f9d7f471",
        "source_code_copied": False,
        "racing_line_features": learner.config.racing_line_features,
        "racing_line_reward_shaping": bool(
            learner.config.racing_line_features
            and not learner.config.progress_reward
        ),
        "target_speed_features": False,
        "racing_line_path": (
            None
            if args.racing_line_path is None
            else str(args.racing_line_path)
        ),
        "teacher_actions": False,
        "curriculum": False,
        "actor_rollback": False,
        "internal_policy_probes": False,
        "observation_noise_std": args.observation_noise_std,
        "training_noisy_episode_fraction": (
            SENSOR_V1_ROBUST_PACE_NOISY_EPISODE_FRACTION
            if args.sensor_v1_robust_pace_continuation
            else 0.0
        ),
        "training_noisy_observation_std": (
            SENSOR_V1_ROBUST_PACE_NOISY_OBSERVATION_STD
            if args.sensor_v1_robust_pace_continuation
            else 0.0
        ),
        "evaluation_observation_noise_std": 0.0
        if args.sensor_v1_robust_pace_continuation
        else args.observation_noise_std,
        "cold_start_evaluation_rollouts": bool(
            args.sensor_v1_robust_pace_continuation
        ),
        "evaluation_repeats": args.evaluation_repeats,
        "train_frequency": args.train_frequency,
        "effective_update_to_data_ratio": (
            learner.config.gradient_steps_per_interaction / args.train_frequency
        ),
        "config": learner.config.__dict__,
    }
    _write_json(run_dir / "config.json", config_record)

    logger = CsvRunLogger(run_dir)
    writer = _tensorboard_writer(run_dir, args.no_tensorboard)
    environment = NstepTorcsEnvironment(
        track_name=args.track,
        max_episode_steps=args.max_episode_steps,
        manual_start=args.manual_start,
        observation_noise_std=args.observation_noise_std,
        seed=args.seed,
        racing_line_path=args.racing_line_path,
        steering_rate_cost_coefficient=(
            learner.config.steering_rate_cost_coefficient
        ),
        physical_steering_limit=learner.config.physical_steering_limit,
        progress_reward=learner.config.progress_reward,
        racing_line_features=learner.config.racing_line_features,
    )
    observation_noise_rng = np.random.default_rng(args.seed + 89_761)

    def prepare_training_episode() -> None:
        noise_std = args.observation_noise_std
        if args.sensor_v1_robust_pace_continuation:
            noise_std = select_sensor_v1_robust_pace_noise(
                observation_noise_rng
            )
        environment.set_observation_noise_std(noise_std)

    def prepare_evaluation() -> None:
        if args.sensor_v1_robust_pace_continuation:
            environment.set_observation_noise_std(0.0)
    accumulator = NstepTransitionAccumulator(
        learner.config.n_steps,
        learner.config.gamma,
    )
    next_evaluation = _next_boundary(
        learner.environment_steps,
        args.evaluation_interval,
    )
    next_checkpoint = _next_boundary(
        learner.environment_steps,
        args.checkpoint_interval,
    )
    run_steps = 0
    pending_metrics: list[UpdateMetrics] = []
    observation: Any = None
    last_evaluation_step: int | None = None
    actor_updates_at_start = learner.actor_updates
    actor_update_limit = (
        args.steering_rate_max_actor_updates
        if args.steering_rate_v4
        else None
    )
    cold_start_evaluations = bool(args.sensor_v1_robust_pace_continuation)
    promotion_gate = (
        EvaluationPromotionGate(
            required_completion_rate=1.0,
            maximum_median_lap_time_seconds=(
                args.steering_rate_promotion_median_seconds
            ),
        )
        if args.steering_rate_v4
        else EvaluationPromotionGate(
            required_completion_rate=SENSOR_V1_CLEAN_RELIABILITY_COMPLETION_RATE,
            maximum_median_lap_time_seconds=(
                SENSOR_V1_CLEAN_RELIABILITY_MEDIAN_SECONDS
            ),
        )
        if args.sensor_v1_clean_reliability_continuation
        or args.sensor_v1_robust_pace_continuation
        else None
    )
    observation_noise_summary = (
        "25% of training episodes use the original 0.025 sensor-noise "
        "profile; evaluation is clean-only, "
        if args.sensor_v1_robust_pace_continuation
        else ""
    )

    observation_summary = (
        "racing-line geometry observations"
        if learner.config.racing_line_features
        else "TORCS telemetry and track-sensor observations only"
    )
    print(
        f"{agent_label} protocol: torcsRL-aligned N-step TD3 core, full track, "
        f"{learner.config.n_steps}-step replay, "
        f"{replay_warmup_size:,}-transition replay warm-up, "
        f"replay actions={replay_warmup_action_mode}, "
        f"{observation_summary}, "
        f"observation noise profile level={args.observation_noise_std:g}, "
        f"{observation_noise_summary}"
        "training action noise sigma="
        f"(steer={learner.config.exploration_noise:g}, "
        "longitudinal="
        f"{learner.config.longitudinal_exploration_noise:g}), "
        f"{learner.config.gradient_steps_per_interaction} gradient batch(es) "
        f"every {args.train_frequency} interactions, no curriculum, "
        "no actor rollback, "
        f"reward={learner.reward_version}, steering-rate coefficient="
        f"{learner.config.steering_rate_cost_coefficient:g}, "
        "physical steering limit=+/-"
        f"{learner.config.physical_steering_limit:g}, "
        f"actor-update ceiling={actor_update_limit}, "
        f"critic-only warm-up={critic_warmup_updates:,} updates, "
        f"automatic {args.evaluation_repeats}-run median evaluation every "
        f"{args.evaluation_interval:,} steps; cold-start evaluations="
        f"{cold_start_evaluations}."
    )
    try:
        if (
            args.resume is not None
            and current_best_evaluation_score(learner) is None
            and not args.steering_rate_v4
        ):
            print(
                "Establishing a passive lap-aware evaluation baseline for this "
                f"checkpoint ({args.evaluation_repeats} deterministic runs)."
            )
            prepare_evaluation()
            baseline_evaluations, _baseline_median = evaluate_and_record(
                environment,
                learner,
                logger,
                writer,
                model_dir,
                repeats=args.evaluation_repeats,
                mode="evaluation_baseline",
                allow_promotion=False,
                cold_start_each_rollout=cold_start_evaluations,
            )
            baseline_score = summarise_evaluations(baseline_evaluations)
            record_best_evaluation_score(learner, baseline_score)
            if pace_score_is_better(
                baseline_score,
                learner.best_lap_time_seconds,
            ):
                learner.best_lap_time_seconds = (
                    baseline_score.fastest_lap_time_seconds
                )

        observation, warmup_interactions = collect_replay_warmup(
            environment,
            learner,
            accumulator,
            logger,
            writer,
            use_random_actions=args.resume is None,
            deterministic_policy_actions=conservative_mode,
            target_replay_size=replay_warmup_size,
            random_action_hold_interactions=(
                1
                if (
                    args.sensor_steering_v2
                    or args.sensor_clean_observation_v3
                    or args.sensor_v1_clean_reliability_continuation
                    or args.sensor_v1_robust_pace_continuation
                )
                else PACE_V5_WARMUP_ACTION_HOLD_INTERACTIONS
                if args.pace_v5
                else SENSOR_ONLY_WARMUP_ACTION_HOLD_INTERACTIONS
                if args.sensor_only
                else 1
            ),
            launch_capable_random_actions=args.pace_v5 or args.sensor_only,
            random_action_steering_limit=(
                SENSOR_ONLY_WARMUP_STEERING_LIMIT
                if args.sensor_only
                else None
            ),
            before_episode_reset=prepare_training_episode,
        )
        if warmup_interactions:
            print(
                f"Replay warm-up complete after {warmup_interactions:,} "
                "interactions; starting the requested learning budget."
            )
        if critic_warmup_updates:
            print(
                f"Adapting {agent_label} critic to fresh replay with the actor frozen: "
                f"{critic_warmup_updates:,} gradient updates."
            )
            critic_metrics = learner.train_critic_only(critic_warmup_updates)
            logger.write_updates(learner, critic_metrics)
            _record_tensorboard_updates(
                writer,
                critic_metrics,
                learner.environment_steps,
            )
            print(
                "Critic-only warm-up complete: mean critic loss="
                f"{statistics.fmean(item.critic_loss for item in critic_metrics):.4f}; "
                "pace actor weights remain unchanged."
            )
        while run_steps < args.total_timesteps:
            action = learner.select_action(
                observation,
                deterministic=False,
            )

            next_observation, reward, terminated, truncated, _info = (
                environment.step(action)
            )
            episode_end = terminated or truncated
            transitions = accumulator.append(
                observation,
                action,
                reward,
                next_observation,
                terminated=terminated,
                episode_end=episode_end,
            )
            learner.add_transitions(transitions)
            learner.environment_steps += 1
            run_steps += 1
            if should_run_gradient_updates(
                learner.environment_steps,
                args.train_frequency,
            ):
                actor_updates_used = (
                    learner.actor_updates - actor_updates_at_start
                )
                allow_actor_update = (
                    actor_update_limit is None
                    or actor_updates_used < actor_update_limit
                )
                pending_metrics.extend(
                    learner.train_for_interaction(
                        allow_actor_update=allow_actor_update,
                    )
                )
            observation = next_observation

            if learner.environment_steps % 1_000 == 0:
                logger.write_updates(learner, pending_metrics)
                _record_tensorboard_updates(
                    writer,
                    pending_metrics,
                    learner.environment_steps,
                )
                pending_metrics.clear()

            if not episode_end:
                continue

            learner.episodes += 1
            summary = environment.last_summary
            if summary is None:
                raise RuntimeError("training episode ended without a summary")
            logger.write_episode(
                summary,
                mode="training",
                policy_controlled=True,
                global_step=learner.environment_steps,
            )
            _record_tensorboard_episode(
                writer,
                summary,
                learner.environment_steps,
                "training",
            )
            if summary.distance_m > learner.best_training_distance_m:
                learner.best_training_distance_m = summary.distance_m
                learner.save(model_dir / "best_distance.pt")
                print(
                    f"New {agent_label} training frontier: {summary.distance_m:.1f}m"
                )
            print(
                f"Train {run_steps:,}/{args.total_timesteps:,} | "
                f"ep={learner.episodes} {summary.termination_reason} "
                f"distance={summary.distance_m:.1f}m "
                f"reward={summary.reward:.1f} "
                f"replay={len(learner.replay):,}"
            )

            if learner.environment_steps >= next_evaluation:
                prepare_evaluation()
                evaluate_and_record(
                    environment,
                    learner,
                    logger,
                    writer,
                    model_dir,
                    repeats=args.evaluation_repeats,
                    promotion_gate=promotion_gate,
                    cold_start_each_rollout=cold_start_evaluations,
                    evaluation_candidate_dir=(
                        run_dir / "evaluation_candidates"
                        if args.sensor_v1_robust_pace_continuation
                        else None
                    ),
                )
                last_evaluation_step = learner.environment_steps
                while learner.environment_steps >= next_evaluation:
                    next_evaluation += args.evaluation_interval

            if learner.environment_steps >= next_checkpoint:
                learner.save(model_dir / "latest.pt")
                learner.save_network_components(model_dir / "checkpoints")
                if args.save_replay_buffer:
                    learner.save_replay(model_dir / "latest.replay.pt")
                while learner.environment_steps >= next_checkpoint:
                    next_checkpoint += args.checkpoint_interval

            prepare_training_episode()
            observation, _ = environment.reset()
            accumulator.reset()

        if last_evaluation_step != learner.environment_steps:
            prepare_evaluation()
            evaluate_and_record(
                environment,
                learner,
                logger,
                writer,
                model_dir,
                repeats=args.evaluation_repeats,
                promotion_gate=promotion_gate,
                cold_start_each_rollout=cold_start_evaluations,
                evaluation_candidate_dir=(
                    run_dir / "evaluation_candidates"
                    if args.sensor_v1_robust_pace_continuation
                    else None
                ),
            )
            last_evaluation_step = learner.environment_steps

    except KeyboardInterrupt:
        interrupted_path = learner.save(model_dir / "interrupted.pt")
        print(f"\nTraining interrupted; saved {interrupted_path}")
    except Exception:
        learner.save(model_dir / "emergency.pt")
        print(
            f"Saved {agent_label} emergency checkpoint before propagating "
            "the error."
        )
        raise
    finally:
        if pending_metrics:
            logger.write_updates(learner, pending_metrics)
        learner.save(model_dir / "latest.pt")
        learner.save_network_components(model_dir / "checkpoints")
        if args.save_replay_buffer:
            learner.save_replay(model_dir / "latest.replay.pt")
        environment.close()
        logger.close()
        if writer is not None:
            writer.close()

    print(f"Saved {agent_label} model to {model_dir / 'latest.pt'}")
    print(f"Saved {agent_label} run logs to {run_dir}")


if __name__ == "__main__":
    main()
