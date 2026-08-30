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
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from n_step_td3.environment import EpisodeSummary, NstepTorcsEnvironment
from n_step_td3.contracts import (
    ACTION_VERSION,
    DEFAULT_STEERING_RATE_COST_COEFFICIENT,
    OBSERVATION_VERSION,
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
DEFAULT_TRAIN_FREQUENCY = 4
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
        description="Train the independent Agent 7 N-step TD3 racer."
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
        "--fine-tune-actor-learning-rate",
        type=float,
        default=None,
        help=(
            "Actor learning rate used by conservative modes "
            f"(default: {FINE_TUNE_ACTOR_LEARNING_RATE:g}, or "
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
    parser.add_argument("--evaluation-interval", type=int, default=10_000)
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
    if args.fine_tune and args.steering_rate_v4:
        parser.error("--fine-tune and --steering-rate-v4 are mutually exclusive")
    if args.total_timesteps is None:
        args.total_timesteps = (
            STEERING_RATE_V4_TOTAL_TIMESTEPS
            if args.steering_rate_v4
            else 500_000
        )
    if args.evaluation_repeats is None:
        args.evaluation_repeats = (
            STEERING_RATE_V4_EVALUATION_REPEATS
            if args.steering_rate_v4
            else 3
        )
    if args.fine_tune_actor_learning_rate is None:
        args.fine_tune_actor_learning_rate = (
            STEERING_RATE_V4_ACTOR_LEARNING_RATE
            if args.steering_rate_v4
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
        elif args.fine_tune:
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
        args.resume = BEST_EVALUATION_PATH
    elif args.resume_pace:
        args.resume = BEST_PACE_PATH
    if args.fine_tune and args.resume is None:
        parser.error(
            "--fine-tune requires --resume, --resume-best, or --resume-pace"
        )
    if args.steering_rate_v4 and not args.resume_best:
        parser.error("--steering-rate-v4 requires --resume-best")
    if args.steering_rate_v4 and args.evaluation_repeats < 10:
        parser.error("--steering-rate-v4 requires at least 10 evaluation repeats")
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
    if args.racing_line_path is not None and not args.racing_line_path.is_file():
        parser.error(
            f"--racing-line-path does not exist: {args.racing_line_path}"
        )
    if (
        not math.isfinite(args.observation_noise_std)
        or args.observation_noise_std < 0.0
    ):
        parser.error("--observation-noise-std must be finite and non-negative")
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


def evaluate_once(
    environment: NstepTorcsEnvironment,
    learner: NstepTd3Learner,
) -> EpisodeSummary:
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
) -> tuple[list[EpisodeSummary], float]:
    if repeats < 1:
        raise ValueError("evaluation repeats must be positive")

    evaluations = [evaluate_once(environment, learner) for _ in range(repeats)]
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
            "New Agent 7 evaluation champion: "
            f"completion={score.completion_rate:.0%}{pace_summary}, "
            f"median_distance={median_distance:.1f}m"
        )
    if pace_improved:
        learner.save(model_dir / "best_pace.pt")
        learner.save(model_dir / "best_lap.pt")
        print(
            "New Agent 7 pace champion: fastest clean evaluation lap="
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
) -> tuple[np.ndarray, int]:
    target_size = (
        learner.config.replay_start_size
        if target_replay_size is None
        else max(learner.config.replay_start_size, int(target_replay_size))
    )
    if target_size > learner.config.replay_capacity:
        raise ValueError("replay warm-up target exceeds replay capacity")
    observation, _ = environment.reset()
    accumulator.reset()
    interactions = 0
    while len(learner.replay) < target_size:
        action = (
            learner.random_action()
            if use_random_actions
            else learner.select_action(
                observation,
                deterministic=deterministic_policy_actions,
            )
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
        observation, _ = environment.reset()
        accumulator.reset()
    return observation, interactions


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
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        STEERING_RATE_V4_RUNS_DIR if args.steering_rate_v4 else RUNS_DIR
    ) / timestamp
    model_dir = (
        STEERING_RATE_V4_MODEL_DIR if args.steering_rate_v4 else MODEL_DIR
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.steering_rate_v4:
        learner = initialize_steering_rate_v4_learner(args)
        print(
            "Initialized Agent 7 v4 from protected v3 actor parameters only: "
            f"{args.resume}. Critic, target critic, optimizers, replay, counters, "
            "and source evaluation records were not inherited."
        )
    elif args.resume is not None:
        learner = NstepTd3Learner.load(args.resume, device=args.device)
        print(f"Resuming Agent 7 from {args.resume}")
        if args.seed_was_explicit:
            learner.reseed_training_noise(args.seed)
            print(
                f"Reseeded Agent 7 training noise with seed {args.seed}; "
                "checkpoint weights and optimizer state are unchanged."
            )
        if args.fine_tune:
            learner.enable_fine_tuning(
                actor_learning_rate=args.fine_tune_actor_learning_rate,
            )
            print(
                "Enabled Agent 7 pace fine-tuning: actor learning rate="
                f"{learner.config.actor_learning_rate:g}, action noise="
                f"{learner.config.exploration_noise:g}, policy delay="
                f"{learner.config.policy_delay}, train frequency="
                f"{args.train_frequency}."
            )
    else:
        learner = NstepTd3Learner(config_from_args(args), device=args.device)
        print("Starting Agent 7 N-step TD3 from random network weights.")
    synchronise_recorded_evaluation(learner, model_dir)
    synchronise_recorded_pace(learner, model_dir)
    if args.replay_buffer_path is not None:
        learner.load_replay(args.replay_buffer_path)
        print(
            f"Loaded Agent 7 replay buffer with {len(learner.replay):,} transitions."
        )
    elif args.resume is not None:
        print(
            "Starting a fresh replay buffer for this continuation; "
            "the protected checkpoint remains unchanged unless evaluation improves."
        )

    conservative_mode = bool(args.fine_tune or args.steering_rate_v4)
    replay_warmup_size = learner.config.replay_start_size
    critic_warmup_updates = 0
    if conservative_mode:
        replay_warmup_size = max(
            learner.config.replay_start_size,
            args.fine_tune_replay_warmup_size,
        )
        critic_warmup_updates = args.fine_tune_critic_warmup_updates
    if args.resume is None:
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
        "actor_initialization": (
            "v3_actor_only" if args.steering_rate_v4 else "checkpoint_resume"
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
            1.0 if args.steering_rate_v4 else None
        ),
        "promotion_median_lap_time_seconds": (
            args.steering_rate_promotion_median_seconds
            if args.steering_rate_v4
            else None
        ),
        "replay_warmup_size": replay_warmup_size,
        "replay_warmup_action_mode": replay_warmup_action_mode,
        "critic_only_warmup_updates": critic_warmup_updates,
        "replay_source": (
            "fresh" if args.replay_buffer_path is None else str(args.replay_buffer_path)
        ),
        "device": str(learner.device),
        "algorithm": "N-step TD3",
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "reward_version": learner.reward_version,
        "source_design": "https://github.com/raphaelsenn/torcsRL",
        "source_revision_reviewed": "043e806e260507d1d5eef161004bc350f9d7f471",
        "source_code_copied": False,
        "racing_line_features": True,
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
    )
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
    promotion_gate = (
        EvaluationPromotionGate(
            required_completion_rate=1.0,
            maximum_median_lap_time_seconds=(
                args.steering_rate_promotion_median_seconds
            ),
        )
        if args.steering_rate_v4
        else None
    )

    print(
        "Agent 7 protocol: torcsRL-aligned N-step TD3 core, full track, "
        f"{learner.config.n_steps}-step replay, "
        f"{replay_warmup_size:,}-transition replay warm-up, "
        f"replay actions={replay_warmup_action_mode}, "
        "racing-line state/reward shaping, "
        f"observation noise profile level={args.observation_noise_std:g}, "
        "training action noise sigma="
        f"(steer={learner.config.exploration_noise:g}, "
        "longitudinal="
        f"{learner.config.longitudinal_exploration_noise:g}), "
        f"{learner.config.gradient_steps_per_interaction} gradient batch(es) "
        f"every {args.train_frequency} interactions, no curriculum, "
        "no actor rollback, "
        f"reward={learner.reward_version}, steering-rate coefficient="
        f"{learner.config.steering_rate_cost_coefficient:g}, "
        f"actor-update ceiling={actor_update_limit}, "
        f"critic-only warm-up={critic_warmup_updates:,} updates, "
        f"automatic {args.evaluation_repeats}-run median evaluation every "
        f"{args.evaluation_interval:,} steps."
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
            baseline_evaluations, _baseline_median = evaluate_and_record(
                environment,
                learner,
                logger,
                writer,
                model_dir,
                repeats=args.evaluation_repeats,
                mode="evaluation_baseline",
                allow_promotion=False,
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
        )
        if warmup_interactions:
            print(
                f"Replay warm-up complete after {warmup_interactions:,} "
                "interactions; starting the requested learning budget."
            )
        if critic_warmup_updates:
            print(
                "Adapting Agent 7 critic to fresh replay with the actor frozen: "
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
                    f"New Agent 7 training frontier: {summary.distance_m:.1f}m"
                )
            print(
                f"Train {run_steps:,}/{args.total_timesteps:,} | "
                f"ep={learner.episodes} {summary.termination_reason} "
                f"distance={summary.distance_m:.1f}m "
                f"reward={summary.reward:.1f} "
                f"replay={len(learner.replay):,}"
            )

            if learner.environment_steps >= next_evaluation:
                evaluate_and_record(
                    environment,
                    learner,
                    logger,
                    writer,
                    model_dir,
                    repeats=args.evaluation_repeats,
                    promotion_gate=promotion_gate,
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

            observation, _ = environment.reset()
            accumulator.reset()

        if last_evaluation_step != learner.environment_steps:
            evaluate_and_record(
                environment,
                learner,
                logger,
                writer,
                model_dir,
                repeats=args.evaluation_repeats,
                promotion_gate=promotion_gate,
            )
            last_evaluation_step = learner.environment_steps

    except KeyboardInterrupt:
        interrupted_path = learner.save(model_dir / "interrupted.pt")
        print(f"\nTraining interrupted; saved {interrupted_path}")
    except Exception:
        learner.save(model_dir / "emergency.pt")
        print("Saved Agent 7 emergency checkpoint before propagating the error.")
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

    print(f"Saved Agent 7 model to {model_dir / 'latest.pt'}")
    print(f"Saved Agent 7 run logs to {run_dir}")


if __name__ == "__main__":
    main()
