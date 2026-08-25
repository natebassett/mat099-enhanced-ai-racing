from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import io
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ORIGINAL_COMMAND = " ".join(sys.argv)

from agents.td3_agent import (  # noqa: E402
    AGENT6_ACTION_SHAPE,
    AGENT6_ACTION_SIZE,
    AGENT6_ACTION_VERSION,
    AGENT6_LEGACY_ACTION_SHAPE,
    AGENT6_LEGACY_ACTION_SIZE,
    AGENT6_LEGACY_ACTION_VERSION,
    AGENT6_LEGACY_OBSERVATION_VERSION,
    AGENT6_MODEL_FAMILY,
    AGENT6_OBSERVATION_VERSION,
    AGENT6_ROBUSTNESS_PROTOCOL,
    AGENT6_ROBUSTNESS_MAX_PAIRED_REGRESSION_M,
    AGENT6_ROBUSTNESS_STEERING_NOISE_STD,
    AGENT6_TRAINING_CHALLENGE_BASE_SEED,
    DEFAULT_BEST_EVALUATION_MODEL_PATH,
    DEFAULT_BEST_DISTANCE_MODEL_PATH,
    DEFAULT_BEST_REWARD_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    EXTERNAL_EVALUATION_SEED_MAX,
    EXTERNAL_EVALUATION_SEED_MIN,
    FEATURE_NAMES,
    G_TRACK_3_LENGTH_METRES,
    MAX_REPRODUCIBLE_SEED,
    PEDAL_DEADZONE,
    SeededSteeringActionNoise,
    TRAINING_CHALLENGE_SEED_MAX,
    TRAINING_CHALLENGE_SEED_MIN,
    balanced_pair_order_positions,
    build_rotating_training_seed_suite,
    build_td3_observation,
    clamp,
    counterbalanced_pair_order,
    decode_legacy_td3_action,
    decode_td3_action,
    episode_metadata_is_deterministic_probe,
    finite_clamp,
    evaluation_checkpoint_is_verified,
    finite_float,
    metadata_path_for_policy,
    order_stratified_seed_aggregates,
    order_stratified_seed_position_medians,
    parameter_state_dicts_are_identical,
    policy_contract_mismatches,
    policy_uses_legacy_action_contract,
    rate_limit_td3_action,
    read_policy_metadata,
    shift_gears,
    track_sensors,
)
from runner.lap_tracker import LapTracker, practice_finish_is_plausible  # noqa: E402


DEFAULT_TRACK_NAME = "g-track-3"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "models" / "training_runs" / "agent6_td3_scratch"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "agent6_td3_scratch"
DEFAULT_TENSORBOARD_DIR = PROJECT_ROOT / "models" / "tensorboard" / "agent6_td3_scratch"
DEFAULT_REPLAY_BUFFER_PATH = PROJECT_ROOT / "models" / "replay_buffers" / "agent6_td3_scratch.pkl"
DEFAULT_CHECKPOINT_FREQ = 25_000
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0
ACTION_NOISE_UPDATE_INTERVAL_STEPS = 1_000
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_FINAL_LEARNING_RATE = 1e-4
DEFAULT_CONTINUATION_LEARNING_RATE = 3e-5
DEFAULT_CONTINUATION_FINAL_LEARNING_RATE = 1e-5
DEFAULT_CONTINUATION_ACTOR_LEARNING_RATE = 1e-5
DEFAULT_CONTINUATION_FINAL_ACTOR_LEARNING_RATE = 3e-6
DEFAULT_CONTINUATION_BUFFER_SIZE = 300_000
DEFAULT_CONTINUATION_REPLAY_REFILL_STEPS = 20_000
DEFAULT_CONTINUATION_CRITIC_WARMUP_STEPS = 20_000
DEFAULT_CONTINUATION_ACTOR_UNFREEZE_STEPS = 40_000
DEFAULT_CONTINUATION_ACTION_NOISE_SIGMA = 0.02
DEFAULT_CONTINUATION_FINAL_ACTION_NOISE_SIGMA = 0.006
DEFAULT_SAFE_ACTOR_MAX_ACTION_DELTA = 0.01
DEFAULT_SAFE_ACTOR_GRADIENT_NORM = 1.0
DEFAULT_SAFE_ACTOR_BLOCK_UPDATES = 100
DEFAULT_SAFE_ACTOR_BLOCKS_PER_PROBE = 25
DEFAULT_SAFE_ACTOR_PROBE_REPEATS = 3
DEFAULT_SAFE_ACTOR_TRIALS_PER_SEED = 2
DEFAULT_SAFE_ACTOR_SCREENING_TRIALS_PER_SEED = 2
DEFAULT_SAFE_ACTOR_SCREENING_MAX_EPISODE_STEPS = 1_200
DEFAULT_SAFE_ACTOR_PROBE_BASE_SEED = AGENT6_TRAINING_CHALLENGE_BASE_SEED
DEFAULT_SAFE_ACTOR_PROBE_NOISE_STD = AGENT6_ROBUSTNESS_STEERING_NOISE_STD
DEFAULT_SAFE_ACTOR_MIN_IMPROVEMENT_M = 5.0
DEFAULT_SAFE_ACTOR_MAX_PAIRED_REGRESSION_M = (
    AGENT6_ROBUSTNESS_MAX_PAIRED_REGRESSION_M
)
DEFAULT_SAFE_ACTOR_SCREENING_MAX_REGRESSION_M = 250.0
DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE = 256
DEFAULT_MAX_PROBE_FRACTION = 0.20
DEFAULT_PROBE_EVIDENCE_WINDOW = 8
DEFAULT_PROBE_MIN_EVIDENCE_EPISODES = 3
DEFAULT_PROBE_EVIDENCE_MAX_REGRESSION_M = 150.0
SAFE_ACTOR_ROBUSTNESS_PROTOCOL = (
    "agent6_ratio_budgeted_evidence_gated_internal_probe_v9"
)

REWARD_VERSION = "agent6_td3_reward_v2_clear_track_anti_stall"
OFF_TRACK_GRACE_STEPS = 4
HARD_TRACK_BOUNDARY = 1.18
SOFT_TRACK_BOUNDARY = 1.04
STUCK_SPEED_LIMIT_KMH = 5.0
STUCK_PROGRESS_LIMIT_M = 0.12
STUCK_SECONDS_LIMIT = 3.0
DEFAULT_STAGE_SUCCESS_WINDOW_SIZE = 10
DEFAULT_STAGE_SUCCESS_REQUIREMENTS = {
    "launch": 3,
    "first_corner": 2,
    "sector_progression": 1,
    "full_lap": 1,
}


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    name: str
    objective: str
    distance_target_m: float
    max_episode_steps: int
    progress_weight: float
    milestone_weight: float
    success_reward: float
    failure_penalty: float
    safe_front_speed_kmh: float


CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        "launch",
        "Launch and Stay On Track",
        "Learn throttle, basic steering, and stable forward motion.",
        140.0,
        2500,
        0.85,
        1.40,
        180.0,
        150.0,
        72.0,
    ),
    CurriculumStage(
        "first_corner",
        "First Corner Survival",
        "Reach and survive the first braking/turn-in sequence.",
        520.0,
        6500,
        0.95,
        1.65,
        260.0,
        220.0,
        92.0,
    ),
    CurriculumStage(
        "sector_progression",
        "Sector Progression",
        "Push the furthest distance while staying inside the circuit.",
        1750.0,
        11000,
        1.05,
        1.95,
        380.0,
        280.0,
        112.0,
    ),
    CurriculumStage(
        "full_lap",
        "Full Lap Attempt",
        "Train for complete-lap behaviour with a large finish reward.",
        G_TRACK_3_LENGTH_METRES,
        16000,
        1.15,
        2.20,
        850.0,
        340.0,
        130.0,
    ),
)
CURRICULUM_STAGE_IDS = tuple(stage.stage_id for stage in CURRICULUM)

EPISODE_COLUMNS = [
    "episodes_seen",
    "global_timestep",
    "training_budget_timestep",
    "probe_overhead_timestep",
    "probe_budget_available_steps",
    "probe_deferred_requests",
    "policy_controlled",
    "deterministic_probe",
    "safe_actor_candidate_probe",
    "safe_actor_probe_mode",
    "safe_actor_probe_seed",
    "safe_actor_candidate_id",
    "safe_actor_probe_index",
    "safe_actor_probe_trial",
    "safe_actor_trials_per_seed",
    "safe_actor_probe_stage",
    "safe_actor_probe_order_position",
    "safe_actor_decision",
    "safe_actor_training_evidence_decision",
    "safe_actor_probe_episodes_completed",
    "safe_actor_probe_episodes_maximum",
    "safe_actor_probe_episodes_skipped",
    "safe_actor_probe_median_distance_m",
    "safe_actor_probe_minimum_distance_m",
    "safe_actor_reference_median_distance_m",
    "safe_actor_reference_minimum_distance_m",
    "safe_actor_paired_median_delta_m",
    "safe_actor_worst_paired_delta_m",
    "safe_actor_paired_regressions",
    "safe_actor_worst_order_position_delta_m",
    "safe_actor_order_position_regressions",
    "safe_actor_order_position_deltas_m",
    "safe_actor_raw_paired_regressions",
    "safe_actor_reference_seed_medians_m",
    "safe_actor_candidate_seed_medians_m",
    "safe_actor_accepted_distance_m",
    "safe_actor_accepted_minimum_distance_m",
    "curriculum_eligible",
    "training_phase",
    "action_noise_sigma",
    "stage_id",
    "stage_name",
    "steps",
    "reward",
    "distance_m",
    "furthest_distance_m",
    "duration_seconds",
    "average_speed_kmh",
    "max_speed_kmh",
    "laps_completed",
    "best_lap_time_seconds",
    "lap_completion_fraction",
    "off_track_steps",
    "max_stopped_seconds",
    "stage_success_streak",
    "stage_required_successes",
    "stage_recent_successes",
    "stage_success_window_size",
    "stage_success_total",
    "termination_reason",
    "final_speed_kmh",
    "final_track_pos",
    "final_angle",
    "front_sensor",
    "min_track_sensor",
]

STEP_COLUMNS = [
    "episode",
    "stage_id",
    "episode_step",
    "global_step",
    "dist_from_start",
    "dist_raced",
    "speed_x",
    "speed_y",
    "angle",
    "track_pos",
    "damage",
    "front_sensor",
    "min_track_sensor",
    "raw_steer",
    "raw_longitudinal",
    "raw_accel_command",
    "raw_brake_command",
    "pedal_conflict",
    "steer",
    "accel",
    "brake",
    "gear",
    "reward",
    "reward_progress",
    "reward_aligned_speed",
    "reward_new_distance",
    "reward_stable_progress_bonus",
    "reward_survival_cost",
    "reward_lateral_position_penalty",
    "reward_angle_penalty",
    "reward_lateral_speed_penalty",
    "reward_speed_steer_penalty",
    "reward_spin_penalty",
    "reward_closing_danger_penalty",
    "reward_clear_track_anti_stall_penalty",
    "reward_stalled_penalty",
    "reward_off_track_penalty",
    "reward_crash_penalty",
    "reward_backwards_penalty",
    "reward_stuck_penalty",
    "reward_terminal_failure_penalty",
    "reward_stage_success_bonus",
    "reward_lap_completion_bonus",
    "terminated",
    "truncated",
    "termination_reason",
    "cur_lap_time",
    *[f"track_sensor_{index}" for index in range(19)],
]


def import_training_dependencies():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import TD3
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as exc:
        raise SystemExit(
            "Agent 6 TD3 training needs extra packages:\n"
            "  pip install stable-baselines3 gymnasium\n\n"
            "The normal race menu can still run the other agents without these packages."
        ) from exc

    return (
        gym,
        spaces,
        TD3,
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        Monitor,
        check_env,
        NormalActionNoise,
    )


def make_torcs_runner():
    with contextlib.redirect_stderr(io.StringIO()):
        from runner.torcs_runner import TorcsRunner

    return TorcsRunner()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS_DIR / timestamp


def curriculum_stage_index(stage_id: str) -> int:
    for index, stage in enumerate(CURRICULUM):
        if stage.stage_id == stage_id:
            return index
    raise ValueError(f"Unknown TD3 curriculum stage: {stage_id}")


def replay_buffer_path_for_policy(policy_path: Path) -> Path:
    return Path(policy_path).with_suffix(".replay_buffer.pkl")


def save_replay_buffer_atomically(model: Any, path: Path) -> str | None:
    target = Path(path)
    temporary = target.with_name(f"{target.name}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        temporary.unlink()
    try:
        model.save_replay_buffer(str(temporary))
        temporary.replace(target)
    except Exception as exc:  # Persistence must not terminate a training run.
        with contextlib.suppress(OSError):
            temporary.unlink()
        return f"{type(exc).__name__}: {exc}"
    return None


def continuation_source_errors(
    policy_path: Path,
    *,
    track: str,
    start_stage: str,
) -> list[str]:
    path = Path(policy_path)
    if not path.is_file():
        return [f"continuation checkpoint does not exist: {path}"]
    if not evaluation_checkpoint_is_verified(path):
        return ["continuation checkpoint has no verified deterministic evaluation"]

    metadata = read_policy_metadata(path)
    errors = [
        f"continuation checkpoint contract mismatch: {key}"
        for key in policy_contract_mismatches(
            metadata,
            allow_legacy_action_contract=True,
        )
    ]
    evaluation = metadata.get("best_evaluation")
    evaluation_repeats = (
        int(finite_float(evaluation.get("repeats")))
        if isinstance(evaluation, Mapping)
        else 0
    )
    if evaluation_repeats < 3:
        errors.append("continuation checkpoint requires at least 3 evaluation repeats")
    evaluation_track = (
        str(evaluation.get("track"))
        if isinstance(evaluation, Mapping) and evaluation.get("track")
        else str(metadata.get("track") or "")
    )
    if evaluation_track and evaluation_track != str(track):
        errors.append(
            "continuation checkpoint track mismatch: "
            f"expected {track}, found {evaluation_track}"
        )
    stage_index = curriculum_stage_index(start_stage)
    required_progress_m = (
        0.0
        if stage_index == 0
        else CURRICULUM[stage_index - 1].distance_target_m
    )
    evaluation_progress_m = (
        finite_float(evaluation.get("median_progress_m"))
        if isinstance(evaluation, Mapping)
        else 0.0
    )
    if evaluation_progress_m < required_progress_m:
        errors.append(
            f"continuation checkpoint reached {evaluation_progress_m:.1f}m, but "
            f"stage {start_stage} requires {required_progress_m:.1f}m"
        )
    return errors


def continuation_source_uses_legacy_action_contract(
    args: argparse.Namespace,
) -> bool:
    if not bool(getattr(args, "continuation", False)):
        return False
    return policy_uses_legacy_action_contract(
        read_policy_metadata(getattr(args, "resume_from", None))
    )


def training_uses_legacy_action_contract(args: argparse.Namespace) -> bool:
    """Preserve v1 end to end unless its lossy migration is explicitly requested."""
    return (
        continuation_source_uses_legacy_action_contract(args)
        and not bool(getattr(args, "allow_legacy_action_migration", False))
    )


def training_action_size(args: argparse.Namespace) -> int:
    return (
        AGENT6_LEGACY_ACTION_SIZE
        if training_uses_legacy_action_contract(args)
        else AGENT6_ACTION_SIZE
    )


def policy_action_start_timestep(args: argparse.Namespace) -> int:
    return 0 if args.continuation else max(0, int(args.learning_starts))


def gradient_update_start_timestep(args: argparse.Namespace) -> int:
    if args.continuation:
        return max(0, int(args.replay_refill_steps))
    return max(0, int(args.learning_starts))


def actor_update_start_timestep(args: argparse.Namespace) -> int:
    gradient_start = gradient_update_start_timestep(args)
    if not args.continuation:
        return gradient_start
    return gradient_start + max(0, int(args.critic_warmup_steps))


def actor_full_unfreeze_timestep(args: argparse.Namespace) -> int:
    return actor_update_start_timestep(args) + max(
        0,
        int(args.actor_unfreeze_steps),
    )


def continuation_training_phase(
    timestep: int,
    *,
    gradient_update_start: int,
    actor_update_start: int,
    actor_full_unfreeze: int,
    pre_update_phase: str,
) -> str:
    current = max(0, int(timestep))
    if current < max(0, int(gradient_update_start)):
        return str(pre_update_phase)
    if current < max(0, int(actor_update_start)):
        return "critic_warmup"
    if current < max(0, int(actor_full_unfreeze)):
        return "actor_unfreeze"
    return "learning"


def safe_actor_probe_episode_step_limit(
    stage: CurriculumStage,
    *,
    probe_stage: str | None,
    screening_max_episode_steps: int,
) -> int:
    stage_limit = max(1, int(stage.max_episode_steps))
    if probe_stage != "screening":
        return stage_limit
    return min(stage_limit, max(1, int(screening_max_episode_steps)))


def calculate_progress_delta(
    previous_telemetry: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> float:
    if previous_telemetry is None:
        return 0.0

    previous_raced = previous_telemetry.get("distRaced")
    current_raced = telemetry.get("distRaced")
    if previous_raced is not None and current_raced is not None:
        delta = finite_float(current_raced) - finite_float(previous_raced)
        return clamp(delta, -8.0, 22.0)

    previous_distance = finite_float(previous_telemetry.get("distFromStart"))
    current_distance = finite_float(telemetry.get("distFromStart"))
    delta = current_distance - previous_distance
    if delta < -track_length_m * 0.5:
        delta += track_length_m
    elif delta > track_length_m * 0.5:
        delta -= track_length_m
    return clamp(delta, -8.0, 22.0)


def calculate_step_time_delta(
    previous_telemetry: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
) -> float | None:
    if previous_telemetry is None:
        return None

    previous_time = finite_float(previous_telemetry.get("curLapTime"))
    current_time = finite_float(telemetry.get("curLapTime"))
    delta = current_time - previous_time
    if 0.0 < delta <= 1.0:
        return delta
    return None


def calculate_stopped_time_delta(
    previous_telemetry: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
) -> float:
    speed = abs(finite_float(telemetry.get("speedX")))
    if speed > STUCK_SPEED_LIMIT_KMH:
        return 0.0

    progress = calculate_progress_delta(previous_telemetry, telemetry)
    if abs(progress) > STUCK_PROGRESS_LIMIT_M:
        return 0.0

    delta_time = calculate_step_time_delta(previous_telemetry, telemetry)
    return 0.02 if delta_time is None else delta_time


def calculate_lap_completion_fraction(distance_m: float) -> float:
    return clamp(max(0.0, distance_m) / G_TRACK_3_LENGTH_METRES, 0.0, 1.0)


def required_successes_for_stage(stage: CurriculumStage | str) -> int:
    stage_id = stage.stage_id if isinstance(stage, CurriculumStage) else str(stage)
    return max(1, int(DEFAULT_STAGE_SUCCESS_REQUIREMENTS.get(stage_id, 1)))


def recent_success_count(success_history: list[bool]) -> int:
    return sum(1 for success in success_history[-DEFAULT_STAGE_SUCCESS_WINDOW_SIZE:] if success)


def curriculum_stage_can_advance(
    stage: CurriculumStage,
    success_history: list[bool],
    *,
    completed_lap: float | None = None,
    curriculum_eligible: bool = True,
) -> bool:
    if not curriculum_eligible:
        return False
    if completed_lap is not None:
        return True
    return recent_success_count(success_history) >= required_successes_for_stage(stage)


def episode_is_policy_controlled(
    episode_start_timestep: int,
    learning_starts: int,
) -> bool:
    """Return true only when every action in the episode came from the policy."""
    return int(episode_start_timestep) >= max(0, int(learning_starts))


def episode_is_curriculum_eligible(
    *,
    policy_controlled: bool,
    deterministic_probe: bool,
) -> bool:
    return bool(policy_controlled and deterministic_probe)


def action_noise_sigma_at_timestep(
    timestep: int,
    *,
    learning_starts: int,
    initial_sigma: float,
    final_sigma: float,
    decay_steps: int,
) -> float:
    initial = max(0.0, float(initial_sigma))
    final = max(0.0, float(final_sigma))
    elapsed = max(0, int(timestep) - max(0, int(learning_starts)))
    progress = clamp(elapsed / max(1, int(decay_steps)), 0.0, 1.0)
    return initial + (final - initial) * progress


def linear_learning_rate_schedule(
    initial_rate: float,
    final_rate: float,
    *,
    learning_starts: int = 0,
    total_timesteps: int = 1,
) -> Callable[[float], float]:
    """Reduce update size only after replay-buffer warm-up has completed."""
    initial = float(initial_rate)
    final = float(final_rate)
    total = max(1, int(total_timesteps))
    first_update = max(0, int(learning_starts))
    decay_steps = max(1, total - first_update)

    def schedule(progress_remaining: float) -> float:
        elapsed_fraction = 1.0 - clamp(float(progress_remaining), 0.0, 1.0)
        estimated_timestep = elapsed_fraction * total
        update_progress = clamp(
            (estimated_timestep - first_update) / decay_steps,
            0.0,
            1.0,
        )
        return initial + (final - initial) * update_progress

    return schedule


def gradual_actor_learning_rate_schedule(
    initial_rate: float,
    final_rate: float,
    *,
    actor_update_start: int,
    unfreeze_steps: int,
    total_timesteps: int,
) -> Callable[[int], float]:
    """Keep the actor fixed, ramp it in, then decay its learning rate."""
    initial = float(initial_rate)
    final = float(final_rate)
    start = max(0, int(actor_update_start))
    ramp_steps = max(1, int(unfreeze_steps))
    ramp_end = start + ramp_steps
    total = max(ramp_end + 1, int(total_timesteps))
    decay_steps = max(1, total - ramp_end)

    def schedule(timestep: int) -> float:
        current = max(0, int(timestep))
        if current <= start:
            return 0.0
        if current < ramp_end:
            ramp_progress = (current - start) / ramp_steps
            return initial * clamp(ramp_progress, 0.0, 1.0)
        decay_progress = clamp((current - ramp_end) / decay_steps, 0.0, 1.0)
        return initial + (final - initial) * decay_progress

    return schedule


def should_schedule_policy_probe(
    *,
    timestep: int,
    learning_starts: int,
    probe_episodes_completed: int,
    policy_episodes_since_probe: int,
    probe_interval: int,
) -> bool:
    if int(timestep) < max(0, int(learning_starts)):
        return False
    if int(probe_episodes_completed) == 0:
        return True
    return int(policy_episodes_since_probe) >= max(1, int(probe_interval))


@dataclass(frozen=True)
class SafeActorProbeComparison:
    decision: str
    reference_median_distance_m: float
    reference_minimum_distance_m: float
    candidate_median_distance_m: float
    candidate_minimum_distance_m: float
    aggregate_median_gain_m: float
    minimum_gain_m: float
    paired_median_delta_m: float
    worst_paired_delta_m: float
    paired_regressions: int
    paired_deltas_m: tuple[float, ...]
    worst_order_position_delta_m: float
    order_position_regressions: int
    order_position_deltas_m: tuple[float, ...]
    raw_paired_regressions: int
    reference_seed_medians_m: tuple[float, ...]
    candidate_seed_medians_m: tuple[float, ...]


def safe_actor_probe_decision(
    reference_distances_m: list[float],
    candidate_distances_m: list[float],
    *,
    seed_count: int,
    trials_per_seed: int,
    reference_order_positions: list[int] | tuple[int, ...] | None = None,
    candidate_order_positions: list[int] | tuple[int, ...] | None = None,
    minimum_improvement_m: float,
    maximum_paired_regression_m: float,
) -> SafeActorProbeComparison:
    """Compare policies using order-stratified seed metrics and safeguards."""
    if not reference_distances_m or len(reference_distances_m) != len(
        candidate_distances_m
    ):
        raise ValueError("safe actor comparison requires equal non-empty probe sets")
    reference = np.asarray(reference_distances_m, dtype=float)
    candidate = np.asarray(candidate_distances_m, dtype=float)
    if (
        not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(candidate))
        or np.any(reference < 0.0)
        or np.any(candidate < 0.0)
    ):
        raise ValueError("safe actor probe distances must be finite and non-negative")

    required_gain = float(minimum_improvement_m)
    regression_limit = float(maximum_paired_regression_m)
    if not math.isfinite(required_gain) or required_gain < 0.0:
        raise ValueError("minimum improvement must be finite and non-negative")
    if not math.isfinite(regression_limit) or regression_limit < 0.0:
        raise ValueError("paired regression limit must be finite and non-negative")
    if (reference_order_positions is None) != (
        candidate_order_positions is None
    ):
        raise ValueError("both policy order-position sets must be provided")
    if reference_order_positions is None:
        reference_order_positions, candidate_order_positions = (
            balanced_pair_order_positions(
                seed_count=seed_count,
                trials_per_seed=trials_per_seed,
            )
        )
    reference_seed_medians = np.asarray(
        order_stratified_seed_aggregates(
            reference_distances_m,
            reference_order_positions,
            seed_count=seed_count,
            trials_per_seed=trials_per_seed,
        ),
        dtype=float,
    )
    candidate_seed_medians = np.asarray(
        order_stratified_seed_aggregates(
            candidate_distances_m,
            candidate_order_positions,
            seed_count=seed_count,
            trials_per_seed=trials_per_seed,
        ),
        dtype=float,
    )
    reference_seed_positions = order_stratified_seed_position_medians(
        reference_distances_m,
        reference_order_positions,
        seed_count=seed_count,
        trials_per_seed=trials_per_seed,
    )
    candidate_seed_positions = order_stratified_seed_position_medians(
        candidate_distances_m,
        candidate_order_positions,
        seed_count=seed_count,
        trials_per_seed=trials_per_seed,
    )
    order_position_deltas: list[float] = []
    for reference_positions, candidate_positions in zip(
        reference_seed_positions,
        candidate_seed_positions,
    ):
        if set(reference_positions) != set(candidate_positions):
            raise ValueError(
                "reference and candidate must cover identical order positions"
            )
        order_position_deltas.extend(
            candidate_positions[position] - reference_positions[position]
            for position in sorted(reference_positions)
        )
    position_deltas = np.asarray(order_position_deltas, dtype=float)
    paired_deltas = candidate_seed_medians - reference_seed_medians
    raw_paired_deltas = candidate - reference
    reference_median = float(np.median(reference_seed_medians))
    candidate_median = float(np.median(candidate_seed_medians))
    reference_minimum = float(np.min(reference_seed_medians))
    candidate_minimum = float(np.min(candidate_seed_medians))
    aggregate_median_gain = candidate_median - reference_median
    minimum_gain = candidate_minimum - reference_minimum
    paired_median_delta = float(np.median(paired_deltas))
    worst_paired_delta = float(np.min(paired_deltas))
    paired_regressions = int(np.count_nonzero(paired_deltas < 0.0))
    worst_order_position_delta = float(np.min(position_deltas))
    order_position_regressions = int(
        np.count_nonzero(position_deltas < 0.0)
    )
    accepted = (
        aggregate_median_gain >= required_gain
        and minimum_gain >= 0.0
        and paired_median_delta >= required_gain
        and worst_paired_delta >= -regression_limit
        and worst_order_position_delta >= -regression_limit
    )
    return SafeActorProbeComparison(
        decision="accepted" if accepted else "rolled_back",
        reference_median_distance_m=reference_median,
        reference_minimum_distance_m=reference_minimum,
        candidate_median_distance_m=candidate_median,
        candidate_minimum_distance_m=candidate_minimum,
        aggregate_median_gain_m=aggregate_median_gain,
        minimum_gain_m=minimum_gain,
        paired_median_delta_m=paired_median_delta,
        worst_paired_delta_m=worst_paired_delta,
        paired_regressions=paired_regressions,
        paired_deltas_m=tuple(float(delta) for delta in paired_deltas),
        worst_order_position_delta_m=worst_order_position_delta,
        order_position_regressions=order_position_regressions,
        order_position_deltas_m=tuple(
            float(delta) for delta in position_deltas
        ),
        raw_paired_regressions=int(np.count_nonzero(raw_paired_deltas < 0.0)),
        reference_seed_medians_m=tuple(
            float(value) for value in reference_seed_medians
        ),
        candidate_seed_medians_m=tuple(
            float(value) for value in candidate_seed_medians
        ),
    )


def calculate_td3_reward_components(
    telemetry: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    previous_telemetry: Mapping[str, Any] | None = None,
    previous_damage: float = 0.0,
    stage: CurriculumStage | None = None,
    episode_distance_m: float = 0.0,
    previous_furthest_distance_m: float = 0.0,
    completed_lap: float | None = None,
    stage_success: bool = False,
    stuck: bool = False,
    terminal_failure: bool | None = None,
) -> dict[str, float]:
    stage = stage or CURRICULUM[0]
    sensors = track_sensors(telemetry)
    front_sensor = sensors[9] if len(sensors) > 9 else 200.0
    min_sensor = min(sensors) if sensors else 200.0
    speed = finite_float(telemetry.get("speedX"))
    angle = finite_float(telemetry.get("angle"))
    track_position = finite_float(telemetry.get("trackPos"))
    lateral_speed = finite_float(telemetry.get("speedY"))
    steer = abs(finite_float(action.get("steer")))
    damage = finite_float(telemetry.get("damage"), previous_damage)
    progress = calculate_progress_delta(previous_telemetry, telemetry)
    new_distance = max(0.0, episode_distance_m - previous_furthest_distance_m)
    aligned_speed = speed * max(0.0, math.cos(angle))
    off_track = min_sensor < 0.0 or abs(track_position) > SOFT_TRACK_BOUNDARY
    crashed = damage > previous_damage
    backwards = math.cos(angle) < 0.0
    if terminal_failure is None:
        terminal_failure = off_track or crashed or backwards or stuck

    components: dict[str, float] = {
        "progress": clamp(progress, -2.0, 10.0) * stage.progress_weight,
        "aligned_speed": clamp(aligned_speed / 190.0, 0.0, 1.0) * 0.55,
        "new_distance": clamp(new_distance, 0.0, 12.0) * stage.milestone_weight,
        "stable_progress_bonus": 0.0,
        "survival_cost": -0.015,
        "lateral_position_penalty": 0.0,
        "angle_penalty": 0.0,
        "lateral_speed_penalty": 0.0,
        "speed_steer_penalty": 0.0,
        "spin_penalty": 0.0,
        "closing_danger_penalty": 0.0,
        "clear_track_anti_stall_penalty": 0.0,
        "stalled_penalty": 0.0,
        "off_track_penalty": 0.0,
        "crash_penalty": 0.0,
        "backwards_penalty": 0.0,
        "stuck_penalty": 0.0,
        "terminal_failure_penalty": 0.0,
        "stage_success_bonus": 0.0,
        "lap_completion_bonus": 0.0,
    }
    if progress > 0.18 and abs(track_position) < 0.85 and abs(angle) < 0.55:
        components["stable_progress_bonus"] = 0.45

    components["lateral_position_penalty"] -= (
        max(0.0, abs(track_position) - 0.45) * 1.6
    )
    components["lateral_position_penalty"] -= (
        max(0.0, abs(track_position) - 0.76) * 5.8
    )
    components["lateral_position_penalty"] -= (
        max(0.0, abs(track_position) - 0.92) * 12.0
    )
    components["angle_penalty"] -= max(0.0, abs(angle) - 0.10) * 1.1
    components["angle_penalty"] -= max(0.0, abs(angle) - 0.45) * 4.5
    components["lateral_speed_penalty"] -= (
        max(0.0, abs(lateral_speed) - 4.0) * 0.10
    )
    components["lateral_speed_penalty"] -= (
        max(0.0, abs(lateral_speed) - 12.0) * 0.24
    )

    speed_pressure = clamp((speed - 40.0) / 115.0, 0.0, 1.0)
    components["speed_steer_penalty"] -= (
        max(0.0, steer - 0.28) * speed_pressure * 5.5
    )
    components["speed_steer_penalty"] -= (
        max(0.0, steer - 0.55) * speed_pressure * 12.0
    )
    components["speed_steer_penalty"] -= (
        max(0.0, abs(angle) - 0.35) * speed_pressure * 7.0
    )
    components["speed_steer_penalty"] -= (
        max(0.0, abs(lateral_speed) - 7.0) * speed_pressure * 0.28
    )
    if abs(angle) > 1.05 and speed > 18.0:
        spin_pressure = clamp((abs(angle) - 1.05) / 0.75, 0.0, 1.0)
        components["spin_penalty"] = -42.0 * spin_pressure

    closing_danger = front_sensor < 55.0 and speed > stage.safe_front_speed_kmh
    if closing_danger:
        brake = finite_float(action.get("brake"))
        accel = finite_float(action.get("accel"))
        overspeed_pressure = clamp(
            (speed - stage.safe_front_speed_kmh) / 80.0,
            0.0,
            1.0,
        )
        front_pressure = clamp((55.0 - front_sensor) / 55.0, 0.0, 1.0)
        components["closing_danger_penalty"] -= (
            front_pressure * overspeed_pressure * 3.2
        )
        components["closing_danger_penalty"] -= (
            front_pressure * overspeed_pressure * max(0.0, 0.25 - brake) * 4.0
        )
        components["closing_danger_penalty"] -= (
            front_pressure * overspeed_pressure * accel * 2.4
        )

    clear_track = (
        not off_track
        and not backwards
        and not crashed
        and min_sensor >= 2.0
        and front_sensor >= 24.0
        and abs(angle) < 0.65
        and episode_distance_m > 8.0
    )
    anti_stall_pressure = (
        clamp((42.0 - speed) / 42.0, 0.0, 1.0)
        * clamp((0.20 - progress) / 0.20, 0.0, 1.0)
    )
    if clear_track and anti_stall_pressure > 0.0:
        accel = finite_float(action.get("accel"))
        brake = finite_float(action.get("brake"))
        throttle_gap = max(0.0, 0.35 - accel)
        brake_drag = max(0.0, brake - 0.02)
        components["clear_track_anti_stall_penalty"] = -anti_stall_pressure * (
            2.2 + 2.4 * throttle_gap + 3.0 * brake_drag
        )

    if progress < 0.03 and speed < 8.0:
        components["stalled_penalty"] = -4.0
    if min_sensor < 0.0:
        components["off_track_penalty"] -= 28.0
    if abs(track_position) > 1.0:
        components["off_track_penalty"] -= 42.0
    if crashed:
        components["crash_penalty"] = (
            -65.0 - max(0.0, damage - previous_damage) * 0.15
        )
    if backwards:
        components["backwards_penalty"] = (
            -145.0 - clamp(abs(speed) / 120.0, 0.0, 1.0) * 55.0
        )
    if stuck:
        components["stuck_penalty"] = -80.0
    if terminal_failure and completed_lap is None and not stage_success:
        remaining = 1.0 - calculate_lap_completion_fraction(episode_distance_m)
        components["terminal_failure_penalty"] = (
            -stage.failure_penalty * (0.35 + 0.65 * remaining)
        )
    if stage_success:
        components["stage_success_bonus"] = stage.success_reward
    if completed_lap is not None:
        components["lap_completion_bonus"] = max(stage.success_reward, 850.0)

    unclipped_total = float(sum(components.values()))
    components["total_unclipped"] = unclipped_total
    components["total"] = float(clamp(unclipped_total, -450.0, 1000.0))
    return components


def calculate_td3_reward(
    telemetry: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    previous_telemetry: Mapping[str, Any] | None = None,
    previous_damage: float = 0.0,
    stage: CurriculumStage | None = None,
    episode_distance_m: float = 0.0,
    previous_furthest_distance_m: float = 0.0,
    completed_lap: float | None = None,
    stage_success: bool = False,
    stuck: bool = False,
    terminal_failure: bool | None = None,
) -> float:
    return calculate_td3_reward_components(
        telemetry,
        action,
        previous_telemetry=previous_telemetry,
        previous_damage=previous_damage,
        stage=stage,
        episode_distance_m=episode_distance_m,
        previous_furthest_distance_m=previous_furthest_distance_m,
        completed_lap=completed_lap,
        stage_success=stage_success,
        stuck=stuck,
        terminal_failure=terminal_failure,
    )["total"]


def make_training_metadata(args: argparse.Namespace) -> dict[str, Any]:
    source_metadata = read_policy_metadata(args.resume_from)
    source_evaluation = source_metadata.get("best_evaluation")
    source_uses_legacy_action_contract = policy_uses_legacy_action_contract(
        source_metadata
    )
    uses_legacy_action_contract = training_uses_legacy_action_contract(args)
    action_contract_migration_required = bool(
        source_uses_legacy_action_contract and not uses_legacy_action_contract
    )
    observation_version = (
        AGENT6_LEGACY_OBSERVATION_VERSION
        if uses_legacy_action_contract
        else AGENT6_OBSERVATION_VERSION
    )
    action_version = (
        AGENT6_LEGACY_ACTION_VERSION
        if uses_legacy_action_contract
        else AGENT6_ACTION_VERSION
    )
    action_shape = (
        AGENT6_LEGACY_ACTION_SHAPE
        if uses_legacy_action_contract
        else AGENT6_ACTION_SHAPE
    )
    source_progress_m = (
        finite_float(source_evaluation.get("median_progress_m"))
        if isinstance(source_evaluation, Mapping)
        else None
    )
    source_minimum_progress_m = (
        finite_float(source_evaluation.get("minimum_progress_m"))
        if isinstance(source_evaluation, Mapping)
        else None
    )
    return {
        "model_family": AGENT6_MODEL_FAMILY,
        "observation_version": observation_version,
        "action_version": action_version,
        "reward_version": REWARD_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": action_shape,
        "algorithm": "TD3",
        "agent_name": "TD3 Scratch Racer",
        "agent_number": 6,
        "track": args.track,
        "track_length_m": G_TRACK_3_LENGTH_METRES,
        "learning_source": "reward_only",
        "teacher": None,
        "behaviour_cloning": False,
        "racing_line_imitation": False,
        "automatic_gear_shift": True,
        "actuator_rate_limited": True,
        "action_contract": (
            {
                "outputs": ["steer", "throttle", "brake"],
                "pedal_resolution": (
                    "apply deadzones, then execute the stronger positive pedal"
                ),
                "exclusive_pedals_at_actuator": True,
                "legacy_contract_preserved": True,
            }
            if uses_legacy_action_contract
            else {
                "outputs": ["steer", "signed_longitudinal"],
                "signed_longitudinal": (
                    "positive values request throttle; negative values request brake"
                ),
                "exclusive_pedals_by_construction": True,
                "legacy_contract_preserved": False,
            }
        ),
        "reward_contract": {
            "version": REWARD_VERSION,
            "low_speed_creep_bonus": False,
            "clear_track_anti_stall_penalty": True,
            "logged_components": [
                column.removeprefix("reward_")
                for column in STEP_COLUMNS
                if column.startswith("reward_")
            ],
        },
        "telemetry_contract": {
            "step_log": "steps.csv",
            "all_track_sensors": [f"track_sensor_{index}" for index in range(19)],
            "pedal_conflict_logged": True,
            "reward_components_logged": True,
        },
        "stage_success_requirements": dict(DEFAULT_STAGE_SUCCESS_REQUIREMENTS),
        "stage_success_window_size": DEFAULT_STAGE_SUCCESS_WINDOW_SIZE,
        "curriculum_promotion": "deterministic_probe_successes_within_rolling_window",
        "curriculum_counts_warmup_episodes": False,
        "curriculum_counts_exploration_episodes": False,
        "policy_probe_interval_episodes": args.policy_probe_interval,
        "policy_probe_action_noise_sigma": 0.0,
        "policy_probe_freezes_gradient_updates": True,
        "policy_probe_writes_to_replay_buffer": True,
        "best_checkpoint_scope": "deterministic_policy_probe_episodes_only",
        "best_checkpoint_saves_replay_buffer": args.save_best_replay_buffer,
        "curriculum": [stage.__dict__ for stage in CURRICULUM],
        "curriculum_start_stage": args.start_stage,
        "continuation": {
            "enabled": args.continuation,
            "source_policy_path": (
                str(args.resume_from) if args.resume_from is not None else None
            ),
            "source_observation_version": source_metadata.get("observation_version"),
            "source_action_version": source_metadata.get("action_version"),
            "source_action_shape": source_metadata.get("action_shape"),
            "source_verified_evaluation_progress_m": source_progress_m,
            "source_verified_evaluation_minimum_progress_m": (
                source_minimum_progress_m
            ),
            "source_policy_contract_migration": {
                "required": action_contract_migration_required,
                "reason": (
                    "explicit_legacy_to_signed_longitudinal_experiment"
                    if action_contract_migration_required
                    else (
                        "legacy_contract_preserved_end_to_end"
                        if uses_legacy_action_contract
                        else "current_contract_or_no_source"
                    )
                ),
                "explicit_opt_in": bool(
                    getattr(args, "allow_legacy_action_migration", False)
                ),
                "target_observation_version": observation_version,
                "target_action_version": action_version,
                "target_action_shape": action_shape,
            },
            "compatibility_probe": {
                "enabled": bool(args.continuation_compatibility_probe),
                "minimum_source_median_fraction": (
                    args.continuation_compatibility_min_fraction
                ),
                "runs_before_replay_collection": True,
                "probe_only": bool(args.continuation_compatibility_probe_only),
            },
            "source_policy_guard": (
                "not_applicable"
                if not args.continuation
                else (
                    "explicit_override"
                    if args.allow_non_champion_source
                    else "verified_champion_only"
                )
            ),
            "fresh_replay_buffer": args.continuation,
            "replay_refill_steps": args.replay_refill_steps,
            "replay_refill_uses_loaded_policy": args.continuation,
            "replay_refill_freezes_gradient_updates": args.continuation,
            "critic_only_warmup_steps": args.critic_warmup_steps,
            "critic_only_warmup_updates_critic_target": args.continuation,
            "actor_frozen_until_timestep": actor_update_start_timestep(args),
            "actor_unfreeze_steps": args.actor_unfreeze_steps,
            "actor_fully_unfrozen_at_timestep": actor_full_unfreeze_timestep(args),
        },
        "safe_actor_improvement": {
            "enabled": args.safe_actor_improvement,
            "learning_source": "reward_only",
            "external_teacher": False,
            "behaviour_cloning": False,
            "racing_line_imitation": False,
            "constraint": "action_space_trust_region_to_last_accepted_actor",
            "max_normalized_action_delta": args.safe_actor_max_action_delta,
            "gradient_clip_norm": args.safe_actor_gradient_norm,
            "actor_updates_per_inner_block": args.safe_actor_block_updates,
            "inner_blocks_per_robustness_probe": (
                args.safe_actor_blocks_per_probe
            ),
            "actor_updates_per_candidate": (
                args.safe_actor_block_updates * args.safe_actor_blocks_per_probe
            ),
            "robustness_protocol": SAFE_ACTOR_ROBUSTNESS_PROTOCOL,
            "robustness_challenge_seeds": args.safe_actor_probe_repeats,
            "robustness_trials_per_seed": args.safe_actor_trials_per_seed,
            "screening_trials_per_seed": (
                args.safe_actor_screening_trials_per_seed
            ),
            "screening_maximum_regression_m": (
                args.safe_actor_screening_max_regression_m
            ),
            "screening_maximum_episode_steps": (
                args.safe_actor_screening_max_episode_steps
            ),
            "training_challenge_base_seed": args.safe_actor_probe_base_seed,
            "training_challenge_seed_range": [
                TRAINING_CHALLENGE_SEED_MIN,
                TRAINING_CHALLENGE_SEED_MAX,
            ],
            "external_evaluation_seed_range": [
                EXTERNAL_EVALUATION_SEED_MIN,
                EXTERNAL_EVALUATION_SEED_MAX,
            ],
            "challenge_seed_rotation": "unique_suite_per_candidate_block",
            "policy_equivalence_check": "exact_actor_parameter_equality",
            "policy_equivalent_action": "skip_matched_robustness_probes",
            "offline_screening": [
                "exact_actor_parameter_equivalence",
                "finite_actor_loss",
                "finite_clipped_actor_gradient_norm",
                "action_space_trust_region",
                "recent_training_progress_evidence",
            ],
            "evidence_window_episodes": args.probe_evidence_window,
            "minimum_evidence_episodes": args.probe_min_evidence_episodes,
            "evidence_maximum_median_regression_m": (
                args.probe_evidence_max_regression_m
            ),
            "probe_pairing": (
                "first_seed_short_ab_ba_collapse_screen_then_"
                "three_seed_balanced_confirmation"
            ),
            "early_rejection": (
                "relaxed_screen_then_irreversible_guard_after_each_seed"
            ),
            "probe_aggregation": (
                "median_within_order_position_then_equal_weight_across_positions"
            ),
            "probe_episodes_per_candidate": (
                2
                * args.safe_actor_probe_repeats
                * args.safe_actor_trials_per_seed
            ),
            "maximum_probe_episodes_per_candidate": (
                2
                * args.safe_actor_probe_repeats
                * args.safe_actor_trials_per_seed
            ),
            "minimum_probe_episodes_before_rejection": (
                2 * args.safe_actor_screening_trials_per_seed
            ),
            "robustness_probe_steering_noise_std": (
                args.safe_actor_probe_noise_std
            ),
            "reference_policy": "last_accepted_actor_on_same_challenge_seeds",
            "acceptance_metric": (
                "order_stratified_seed_metrics_with_seed_and_position_guards"
            ),
            "minimum_improvement_m": args.safe_actor_min_improvement_m,
            "maximum_paired_regression_m": (
                args.safe_actor_max_paired_regression_m
            ),
            "maximum_order_position_regression_m": (
                args.safe_actor_max_paired_regression_m
            ),
            "external_evaluation_remains_final_authority": True,
            "external_evaluation_during_candidate_blocks": False,
            "anchor_batch_size": args.safe_actor_anchor_batch_size,
            "rejected_candidate_restores": [
                "actor",
                "actor_target",
                "actor_optimizer",
            ],
            "critic_and_replay_survive_rollback": True,
            "candidate_probes_deferred_from_curriculum": True,
            "saved_checkpoint_actor": "last_accepted_actor",
        },
        "total_timesteps_requested": args.total_timesteps,
        "timestep_budget": {
            "semantics": "non_probe_environment_interactions",
            "training_steps_requested": args.total_timesteps,
            "probe_steps_excluded_from_training_budget": True,
            "maximum_probe_overhead_steps": args.max_probe_overhead_steps,
            "maximum_probe_fraction": args.max_probe_fraction,
            "minimum_training_fraction": 1.0 - args.max_probe_fraction,
            "effective_maximum_probe_steps": min(
                args.max_probe_overhead_steps,
                int(
                    math.floor(
                        args.total_timesteps
                        * args.max_probe_fraction
                        / (1.0 - args.max_probe_fraction)
                    )
                ),
            ),
            "maximum_environment_steps": (
                args.total_timesteps
                + min(
                    args.max_probe_overhead_steps,
                    int(
                        math.floor(
                            args.total_timesteps
                            * args.max_probe_fraction
                            / (1.0 - args.max_probe_fraction)
                        )
                    ),
                )
            ),
            "probe_exhaustion_action": (
                "defer_probe_freeze_actor_continue_critic_and_replay"
            ),
            "learning_rate_and_noise_schedules_use_training_steps": True,
            "checkpoints_use_training_steps": True,
        },
        "seed": args.seed,
        "manual_start": args.manual_start,
        "relaunch_frequency": args.relaunch_frequency,
        "reset_retries": args.reset_retries,
        "model_path": str(args.model_path),
        "best_distance_model_path": str(args.best_distance_model_path),
        "best_reward_model_path": str(args.best_reward_model_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "replay_buffer_path": str(args.replay_buffer_path),
        "monitor_path": str(args.run_dir / "monitor.csv") if args.run_dir else None,
        "check_env": args.check_env,
        "warmup_action_space": {
            "mode": args.warmup_action_mode,
            "steer_std": args.warmup_steer_std,
            "accel_min": args.warmup_accel_min,
            "accel_max": args.warmup_accel_max,
            "brake_probability": args.warmup_brake_probability,
        },
        "td3_hyperparameters": {
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "policy_action_starts_at_timestep": policy_action_start_timestep(args),
            "gradient_updates_start_at_timestep": gradient_update_start_timestep(args),
            "actor_updates_start_at_timestep": actor_update_start_timestep(args),
            "actor_fully_unfrozen_at_timestep": actor_full_unfreeze_timestep(args),
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "learning_rate": args.learning_rate,
            "learning_rate_final": args.learning_rate_final,
            "critic_learning_rate_schedule": "linear_after_replay_refill",
            "learning_rate_decay_starts_at_timestep": (
                gradient_update_start_timestep(args)
            ),
            "actor_learning_rate": args.actor_learning_rate,
            "actor_learning_rate_final": args.actor_learning_rate_final,
            "actor_learning_rate_schedule": "frozen_then_linear_ramp_and_decay",
            "critic_warmup_steps": args.critic_warmup_steps,
            "actor_unfreeze_steps": args.actor_unfreeze_steps,
            "safe_actor_improvement": args.safe_actor_improvement,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "policy_delay": args.policy_delay,
            "target_policy_noise": args.target_policy_noise,
            "target_noise_clip": args.target_noise_clip,
            "action_noise_initial_sigma": args.action_noise_sigma,
            "action_noise_final_sigma": args.action_noise_final_sigma,
            "action_noise_decay_steps": args.action_noise_decay_steps,
            "net_arch": args.net_arch,
            "device": args.device,
        },
        "command": ORIGINAL_COMMAND,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_scratch_action_space(
    spaces: Any,
    *,
    action_size: int = AGENT6_ACTION_SIZE,
    warmup_action_mode: str = "launch-biased",
    warmup_steer_std: float = 0.30,
    warmup_accel_min: float = 0.25,
    warmup_accel_max: float = 1.00,
    warmup_brake_probability: float = 0.12,
):
    class ScratchActionBox(spaces.Box):
        def __init__(self) -> None:
            if int(action_size) != AGENT6_ACTION_SIZE:
                raise ValueError(
                    "custom scratch warm-up supports the current two-head contract only"
                )
            super().__init__(
                low=-1.0,
                high=1.0,
                shape=(int(action_size),),
                dtype=np.float32,
            )
            self.warmup_action_mode = warmup_action_mode
            self.warmup_steer_std = max(0.0, float(warmup_steer_std))
            accel_min = clamp(float(warmup_accel_min), -1.0, 1.0)
            accel_max = clamp(float(warmup_accel_max), -1.0, 1.0)
            self.warmup_accel_min = min(accel_min, accel_max)
            self.warmup_accel_max = max(accel_min, accel_max)
            self.warmup_brake_probability = clamp(
                float(warmup_brake_probability),
                0.0,
                1.0,
            )

        def sample(self, mask: Any = None):
            if self.warmup_action_mode == "uniform":
                try:
                    return super().sample(mask=mask)
                except TypeError:
                    return super().sample()

            rng = getattr(self, "np_random", np.random.default_rng())
            steer = clamp(float(rng.normal(0.0, self.warmup_steer_std)), -1.0, 1.0)
            longitudinal = float(
                rng.uniform(self.warmup_accel_min, self.warmup_accel_max)
            )
            if float(rng.random()) < self.warmup_brake_probability:
                longitudinal = -float(rng.uniform(0.15, 0.85))
            return np.asarray(
                [
                    steer,
                    clamp(longitudinal, -1.0, 1.0),
                ],
                dtype=np.float32,
            )

    return ScratchActionBox()


def make_training_env_class(gym: Any, spaces: Any):
    class Td3ScratchTrainingEnv(gym.Env):
        metadata = {"render_modes": []}
        track_length_m = G_TRACK_3_LENGTH_METRES

        def __init__(
            self,
            *,
            track_name: str = DEFAULT_TRACK_NAME,
            manual_start: bool = False,
            relaunch_frequency: int = 0,
            reset_retries: int = 2,
            quiet_reset_log: bool = True,
            run_dir: Path | None = None,
            warmup_action_mode: str = "launch-biased",
            warmup_steer_std: float = 0.30,
            warmup_accel_min: float = 0.25,
            warmup_accel_max: float = 1.00,
            warmup_brake_probability: float = 0.12,
            learning_starts: int = 20_000,
            initial_stage_id: str = "launch",
            pre_update_training_phase: str = "warmup",
            use_custom_warmup_action_space: bool = True,
            use_legacy_action_contract: bool = False,
            training_budget_state: TrainingBudgetState | None = None,
            safe_actor_screening_max_episode_steps: int = (
                DEFAULT_SAFE_ACTOR_SCREENING_MAX_EPISODE_STEPS
            ),
        ) -> None:
            super().__init__()
            self.track_name = track_name
            self.manual_start = manual_start
            self.relaunch_frequency = max(0, int(relaunch_frequency))
            self.reset_retries = max(0, int(reset_retries))
            self.quiet_reset_log = quiet_reset_log
            self.run_dir = Path(run_dir) if run_dir is not None else None
            self.learning_starts = max(0, int(learning_starts))
            self.use_legacy_action_contract = bool(use_legacy_action_contract)
            self.action_size = (
                AGENT6_LEGACY_ACTION_SIZE
                if self.use_legacy_action_contract
                else AGENT6_ACTION_SIZE
            )
            self.training_budget_state = training_budget_state
            self.safe_actor_screening_max_episode_steps = max(
                1,
                int(safe_actor_screening_max_episode_steps),
            )
            self.runner = make_torcs_runner()
            if use_custom_warmup_action_space:
                self.action_space = make_scratch_action_space(
                    spaces,
                    action_size=self.action_size,
                    warmup_action_mode=warmup_action_mode,
                    warmup_steer_std=warmup_steer_std,
                    warmup_accel_min=warmup_accel_min,
                    warmup_accel_max=warmup_accel_max,
                    warmup_brake_probability=warmup_brake_probability,
                )
            else:
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.action_size,),
                    dtype=np.float32,
                )
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(FEATURE_NAMES),),
                dtype=np.float32,
            )
            self.stage_index = curriculum_stage_index(initial_stage_id)
            self.pre_update_training_phase = str(pre_update_training_phase)
            self.training_phase = self.pre_update_training_phase
            self.stage_success_streak = 0
            self.stage_success_totals = {
                stage.stage_id: 0
                for stage in CURRICULUM
            }
            self.stage_success_windows = {
                stage.stage_id: []
                for stage in CURRICULUM
            }
            self.episodes_started = 0
            self.total_steps = 0
            self.current_telemetry: dict[str, Any] | None = None
            self.previous_telemetry: dict[str, Any] | None = None
            self.initial_telemetry: dict[str, Any] | None = None
            self.previous_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
            self.previous_damage = 0.0
            self.lap_tracker: LapTracker | None = None
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = 0.0
            self.episode_start_time = 0.0
            self.episode_start_timestep = 0
            self.deterministic_probe = False
            self.safe_actor_candidate_probe = False
            self.safe_actor_probe_mode: str | None = None
            self.safe_actor_probe_seed: int | None = None
            self.safe_actor_probe_trial: int | None = None
            self.safe_actor_trials_per_seed: int | None = None
            self.safe_actor_probe_stage: str | None = None
            self.safe_actor_probe_order_position: int | None = None
            self.action_noise_sigma: float | None = None
            self.episode_max_speed = 0.0
            self.episode_off_track_steps = 0
            self.episode_furthest_distance = 0.0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.consecutive_off_track_steps = 0
            self._step_file = None
            self._step_writer = None
            self._open_step_telemetry()

        @property
        def current_stage(self) -> CurriculumStage:
            return CURRICULUM[self.stage_index]

        def set_policy_mode(
            self,
            *,
            deterministic_probe: bool,
            action_noise_sigma: float,
            training_phase: str,
            safe_actor_candidate_probe: bool = False,
            safe_actor_probe_mode: str | None = None,
            safe_actor_probe_seed: int | None = None,
            safe_actor_probe_trial: int | None = None,
            safe_actor_trials_per_seed: int | None = None,
            safe_actor_probe_stage: str | None = None,
            safe_actor_probe_order_position: int | None = None,
        ) -> None:
            self.deterministic_probe = bool(deterministic_probe)
            self.safe_actor_candidate_probe = bool(
                deterministic_probe and safe_actor_candidate_probe
            )
            self.safe_actor_probe_mode = (
                str(safe_actor_probe_mode)
                if self.safe_actor_candidate_probe and safe_actor_probe_mode
                else None
            )
            self.safe_actor_probe_seed = (
                int(safe_actor_probe_seed)
                if self.safe_actor_candidate_probe
                and safe_actor_probe_seed is not None
                else None
            )
            self.safe_actor_probe_trial = (
                int(safe_actor_probe_trial)
                if self.safe_actor_candidate_probe
                and safe_actor_probe_trial is not None
                else None
            )
            self.safe_actor_trials_per_seed = (
                int(safe_actor_trials_per_seed)
                if self.safe_actor_candidate_probe
                and safe_actor_trials_per_seed is not None
                else None
            )
            self.safe_actor_probe_stage = (
                str(safe_actor_probe_stage)
                if self.safe_actor_candidate_probe and safe_actor_probe_stage
                else None
            )
            self.safe_actor_probe_order_position = (
                int(safe_actor_probe_order_position)
                if self.safe_actor_candidate_probe
                and safe_actor_probe_order_position is not None
                else None
            )
            self.action_noise_sigma = max(0.0, float(action_noise_sigma))
            self.training_phase = str(training_phase)

        def accept_safe_actor_probe(
            self,
            summary: dict[str, Any],
            *,
            median_distance_m: float,
        ) -> None:
            """Commit one accepted probe result to the curriculum state."""
            if str(summary.get("stage_id")) != self.current_stage.stage_id:
                return
            completed_lap = int(summary.get("laps_completed") or 0) > 0
            stage_success = (
                float(median_distance_m) >= self.current_stage.distance_target_m
                or completed_lap
            )
            history = self.stage_success_windows[self.current_stage.stage_id]
            history.append(stage_success)
            if len(history) > DEFAULT_STAGE_SUCCESS_WINDOW_SIZE:
                del history[: len(history) - DEFAULT_STAGE_SUCCESS_WINDOW_SIZE]
            if stage_success:
                self.stage_success_streak += 1
                self.stage_success_totals[self.current_stage.stage_id] += 1
            else:
                self.stage_success_streak = 0

            summary["curriculum_eligible"] = True
            summary["stage_success_streak"] = self.stage_success_streak
            summary["stage_recent_successes"] = recent_success_count(history)
            summary["stage_success_total"] = self.stage_success_totals[
                self.current_stage.stage_id
            ]
            stage_completed = curriculum_stage_can_advance(
                self.current_stage,
                history,
                completed_lap=(1.0 if completed_lap else None),
                curriculum_eligible=True,
            )
            if stage_completed:
                summary["termination_reason"] = "curriculum_stage_completed"
                if self.stage_index < len(CURRICULUM) - 1:
                    self.stage_index += 1
                    self.stage_success_streak = 0

        def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
            super().reset(seed=seed)
            self._ensure_torcs()
            relaunch = bool(options.get("relaunch", False)) if options else False
            if (
                self.relaunch_frequency > 0
                and self.episodes_started > 0
                and self.episodes_started % self.relaunch_frequency == 0
            ):
                relaunch = True
            if self.manual_start:
                relaunch = False

            self._reset_torcs_env(relaunch=relaunch)
            self.episodes_started += 1
            assert self.runner.env is not None
            self.current_telemetry = dict(self.runner.env.client.S.d)
            self.initial_telemetry = dict(self.current_telemetry)
            self.previous_telemetry = None
            self.previous_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
            self.previous_damage = finite_float(self.current_telemetry.get("damage"))
            self.lap_tracker = LapTracker(self.current_telemetry.get("lastLapTime", 0.0))
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = finite_float(
                self.current_telemetry.get("distRaced")
            )
            self.episode_start_time = finite_float(self.current_telemetry.get("curLapTime"))
            self.episode_start_timestep = (
                self.training_budget_state.training_steps
                if self.training_budget_state is not None
                else self.total_steps
            )
            self.episode_max_speed = finite_float(self.current_telemetry.get("speedX"))
            self.episode_off_track_steps = 0
            self.episode_furthest_distance = 0.0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.consecutive_off_track_steps = 0
            return self._build_observation(), {}

        def step(self, raw_action: Any):
            assert self.current_telemetry is not None
            assert self.runner.env is not None
            assert self.lap_tracker is not None
            raw_values = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            action = (
                decode_legacy_td3_action(raw_values)
                if self.use_legacy_action_contract
                else decode_td3_action(raw_values)
            )
            current_speed = finite_float(self.current_telemetry.get("speedX"))
            action = rate_limit_td3_action(
                action,
                self.previous_action,
                speed_kmh=current_speed,
            )
            action["gear"] = shift_gears(current_speed)
            raw_observation, _runner_reward, done, _info = (
                self.runner._step_full_control_agent(action)
            )
            telemetry = dict(raw_observation)
            current_damage = finite_float(telemetry.get("damage"), self.previous_damage)
            sensors = track_sensors(telemetry)
            min_sensor = min(sensors) if sensors else 200.0
            track_position = finite_float(telemetry.get("trackPos"))
            speed = finite_float(telemetry.get("speedX"))
            angle = finite_float(telemetry.get("angle"))
            episode_distance_m = max(
                0.0,
                finite_float(telemetry.get("distRaced")) - self.episode_start_distance,
            )
            previous_furthest = self.episode_furthest_distance
            self.episode_furthest_distance = max(
                self.episode_furthest_distance,
                episode_distance_m,
            )
            completed_lap = self.lap_tracker.update(telemetry)
            if (
                completed_lap is None
                and done
                and practice_finish_is_plausible(
                    self.initial_telemetry,
                    telemetry,
                    self,
                )
            ):
                completed_lap = self.lap_tracker.record(telemetry.get("curLapTime", 0.0))

            stopped_time_delta = calculate_stopped_time_delta(
                self.previous_telemetry,
                telemetry,
            )
            if stopped_time_delta > 0.0:
                self.stopped_seconds += stopped_time_delta
            else:
                self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = max(
                self.episode_max_stopped_seconds,
                self.stopped_seconds,
            )
            stuck = self.stopped_seconds >= STUCK_SECONDS_LIMIT
            off_track = min_sensor < 0.0 or abs(track_position) > SOFT_TRACK_BOUNDARY
            self.consecutive_off_track_steps = (
                self.consecutive_off_track_steps + 1 if off_track else 0
            )
            left_track_bounds = abs(track_position) > HARD_TRACK_BOUNDARY
            off_track_terminal = (
                off_track and self.consecutive_off_track_steps >= OFF_TRACK_GRACE_STEPS
            )
            crashed = current_damage > self.previous_damage
            backwards = math.cos(angle) < 0.0
            stage_success = (
                episode_distance_m >= self.current_stage.distance_target_m
                or completed_lap is not None
            )
            terminal_failure = bool(
                done
                or left_track_bounds
                or off_track_terminal
                or crashed
                or backwards
                or stuck
            )
            reward_components = calculate_td3_reward_components(
                telemetry,
                action,
                previous_telemetry=self.previous_telemetry,
                previous_damage=self.previous_damage,
                stage=self.current_stage,
                episode_distance_m=episode_distance_m,
                previous_furthest_distance_m=previous_furthest,
                completed_lap=completed_lap,
                stage_success=stage_success,
                stuck=stuck,
                terminal_failure=terminal_failure,
            )
            reward = reward_components["total"]

            self.previous_damage = current_damage
            self.previous_telemetry = telemetry
            self.current_telemetry = telemetry
            self.previous_action = {
                "steer": action["steer"],
                "accel": action["accel"],
                "brake": action["brake"],
            }
            self.steps += 1
            self.total_steps += 1
            self.episode_reward += reward
            self.episode_max_speed = max(self.episode_max_speed, speed)
            if off_track:
                self.episode_off_track_steps += 1

            terminated = bool(stage_success or terminal_failure)
            episode_step_limit = safe_actor_probe_episode_step_limit(
                self.current_stage,
                probe_stage=(
                    self.safe_actor_probe_stage
                    if self.safe_actor_candidate_probe
                    else None
                ),
                screening_max_episode_steps=(
                    self.safe_actor_screening_max_episode_steps
                ),
            )
            screening_horizon_reached = bool(
                self.safe_actor_candidate_probe
                and self.safe_actor_probe_stage == "screening"
                and episode_step_limit < self.current_stage.max_episode_steps
                and self.steps >= episode_step_limit
            )
            truncated = self.steps >= episode_step_limit
            policy_controlled = episode_is_policy_controlled(
                self.episode_start_timestep,
                self.learning_starts,
            )
            curriculum_eligible = episode_is_curriculum_eligible(
                policy_controlled=policy_controlled,
                deterministic_probe=self.deterministic_probe,
            ) and not self.safe_actor_candidate_probe
            applied_action_noise_sigma = (
                None
                if not policy_controlled
                else self.action_noise_sigma
                if self.safe_actor_candidate_probe
                else 0.0
                if self.deterministic_probe
                else self.action_noise_sigma
            )
            stage_required_successes = required_successes_for_stage(self.current_stage)
            stage_success_total = self.stage_success_totals.get(
                self.current_stage.stage_id,
                0,
            )
            stage_success_history = self.stage_success_windows[
                self.current_stage.stage_id
            ]
            stage_recent_successes = recent_success_count(stage_success_history)
            stage_completed = False
            if terminated or truncated:
                if curriculum_eligible:
                    if stage_success:
                        self.stage_success_streak += 1
                        stage_success_total += 1
                        self.stage_success_totals[self.current_stage.stage_id] = (
                            stage_success_total
                        )
                    else:
                        self.stage_success_streak = 0
                    stage_success_history.append(bool(stage_success))
                    if len(stage_success_history) > DEFAULT_STAGE_SUCCESS_WINDOW_SIZE:
                        del stage_success_history[
                            : len(stage_success_history)
                            - DEFAULT_STAGE_SUCCESS_WINDOW_SIZE
                        ]
                    stage_recent_successes = recent_success_count(
                        stage_success_history
                    )
                    stage_completed = curriculum_stage_can_advance(
                        self.current_stage,
                        stage_success_history,
                        completed_lap=completed_lap,
                        curriculum_eligible=curriculum_eligible,
                    )
            reason = ""
            if terminated or truncated:
                reason = "truncated"
                if completed_lap is not None and not policy_controlled:
                    reason = "warmup_lap_completed"
                elif completed_lap is not None and not curriculum_eligible:
                    reason = "exploration_lap_completed"
                elif completed_lap is not None:
                    reason = "target_laps_completed"
                elif stage_success and not policy_controlled:
                    reason = "warmup_stage_success"
                elif stage_success and not curriculum_eligible:
                    reason = "exploration_stage_success"
                elif stage_completed:
                    reason = "curriculum_stage_completed"
                elif stage_success:
                    reason = "curriculum_stage_success"
                elif crashed:
                    reason = "crashed"
                elif off_track or left_track_bounds:
                    reason = "off_track"
                elif backwards:
                    reason = "backwards"
                elif stuck:
                    reason = "stuck"
                elif done:
                    reason = "torcs_done"
                elif screening_horizon_reached:
                    reason = "safe_actor_screening_horizon"

            self._write_step_telemetry(
                telemetry,
                raw_values,
                action,
                reward,
                reward_components,
                terminated,
                truncated,
                reason,
            )

            info: dict[str, Any] = {
                "policy_controlled": policy_controlled,
                "deterministic_probe": self.deterministic_probe,
                "safe_actor_candidate_probe": self.safe_actor_candidate_probe,
                "safe_actor_probe_mode": self.safe_actor_probe_mode,
                "safe_actor_probe_seed": self.safe_actor_probe_seed,
                "safe_actor_probe_trial": self.safe_actor_probe_trial,
                "safe_actor_trials_per_seed": self.safe_actor_trials_per_seed,
                "safe_actor_probe_stage": self.safe_actor_probe_stage,
                "safe_actor_probe_order_position": (
                    self.safe_actor_probe_order_position
                ),
                "curriculum_eligible": curriculum_eligible,
                "training_phase": self.training_phase,
                "action_noise_sigma": applied_action_noise_sigma,
                "stage_id": self.current_stage.stage_id,
                "stage_name": self.current_stage.name,
                "distance_m": self.episode_furthest_distance,
                "furthest_distance_m": self.episode_furthest_distance,
                "lap_completion_fraction": calculate_lap_completion_fraction(
                    self.episode_furthest_distance
                ),
                "termination_reason": reason,
            }
            if terminated or truncated:
                summary = self._episode_summary(
                    reason,
                    telemetry,
                    stage_success_streak=self.stage_success_streak,
                    stage_required_successes=stage_required_successes,
                    stage_recent_successes=stage_recent_successes,
                    stage_success_total=stage_success_total,
                    policy_controlled=policy_controlled,
                    deterministic_probe=self.deterministic_probe,
                    curriculum_eligible=curriculum_eligible,
                )
                info["episode_summary"] = summary
                if stage_completed and self.stage_index < len(CURRICULUM) - 1:
                    self.stage_index += 1
                    self.stage_success_streak = 0

            return self._build_observation(), reward, terminated, truncated, info

        def close(self) -> None:
            self._close_step_telemetry()
            self._close_torcs_env()
            self.runner.shutdown()

        def build_current_episode_summary(self, reason: str = "training_stopped"):
            if self.current_telemetry is None or self.steps <= 0:
                return None
            return self._episode_summary(reason, self.current_telemetry)

        def write_end_of_run_summary(self, reason: str = "training_stopped"):
            summary = self.build_current_episode_summary(reason)
            if summary is not None and self.run_dir is not None:
                write_json(self.run_dir / "end_of_run_summary.json", summary)
            return summary

        def _episode_summary(
            self,
            reason: str,
            telemetry: Mapping[str, Any],
            *,
            stage_success_streak: int | None = None,
            stage_required_successes: int | None = None,
            stage_recent_successes: int | None = None,
            stage_success_total: int | None = None,
            policy_controlled: bool | None = None,
            deterministic_probe: bool | None = None,
            curriculum_eligible: bool | None = None,
        ) -> dict[str, Any]:
            sensors = track_sensors(telemetry)
            distance_m = self.episode_furthest_distance
            stage_id = self.current_stage.stage_id
            duration_seconds = max(
                0.0,
                finite_float(telemetry.get("curLapTime")) - self.episode_start_time,
            )
            average_speed_kmh = (
                distance_m / duration_seconds * 3.6
                if duration_seconds > 0.0
                else 0.0
            )
            assert self.lap_tracker is not None
            if policy_controlled is None:
                policy_controlled = episode_is_policy_controlled(
                    self.episode_start_timestep,
                    self.learning_starts,
                )
            if deterministic_probe is None:
                deterministic_probe = self.deterministic_probe
            if curriculum_eligible is None:
                curriculum_eligible = episode_is_curriculum_eligible(
                    policy_controlled=policy_controlled,
                    deterministic_probe=deterministic_probe,
                ) and not self.safe_actor_candidate_probe
            applied_action_noise_sigma = (
                None
                if not policy_controlled
                else self.action_noise_sigma
                if self.safe_actor_candidate_probe
                else 0.0
                if deterministic_probe
                else self.action_noise_sigma
            )
            return {
                "policy_controlled": policy_controlled,
                "deterministic_probe": deterministic_probe,
                "safe_actor_candidate_probe": self.safe_actor_candidate_probe,
                "safe_actor_probe_mode": self.safe_actor_probe_mode,
                "safe_actor_probe_seed": self.safe_actor_probe_seed,
                "safe_actor_probe_trial": self.safe_actor_probe_trial,
                "safe_actor_trials_per_seed": self.safe_actor_trials_per_seed,
                "safe_actor_probe_stage": self.safe_actor_probe_stage,
                "safe_actor_probe_order_position": (
                    self.safe_actor_probe_order_position
                ),
                "curriculum_eligible": curriculum_eligible,
                "training_phase": self.training_phase,
                "action_noise_sigma": applied_action_noise_sigma,
                "stage_id": self.current_stage.stage_id,
                "stage_name": self.current_stage.name,
                "steps": self.steps,
                "reward": self.episode_reward,
                "distance_m": distance_m,
                "furthest_distance_m": distance_m,
                "duration_seconds": duration_seconds,
                "average_speed_kmh": average_speed_kmh,
                "max_speed_kmh": self.episode_max_speed,
                "laps_completed": self.lap_tracker.laps_completed,
                "best_lap_time_seconds": self.lap_tracker.best_lap_time,
                "lap_completion_fraction": calculate_lap_completion_fraction(distance_m),
                "off_track_steps": self.episode_off_track_steps,
                "max_stopped_seconds": self.episode_max_stopped_seconds,
                "stage_success_streak": (
                    self.stage_success_streak
                    if stage_success_streak is None
                    else stage_success_streak
                ),
                "stage_required_successes": (
                    required_successes_for_stage(self.current_stage)
                    if stage_required_successes is None
                    else stage_required_successes
                ),
                "stage_recent_successes": (
                    recent_success_count(self.stage_success_windows.get(stage_id, []))
                    if stage_recent_successes is None
                    else stage_recent_successes
                ),
                "stage_success_window_size": DEFAULT_STAGE_SUCCESS_WINDOW_SIZE,
                "stage_success_total": (
                    self.stage_success_totals.get(stage_id, 0)
                    if stage_success_total is None
                    else stage_success_total
                ),
                "termination_reason": reason,
                "final_speed_kmh": finite_float(telemetry.get("speedX")),
                "final_track_pos": finite_float(telemetry.get("trackPos")),
                "final_angle": finite_float(telemetry.get("angle")),
                "front_sensor": sensors[9] if len(sensors) > 9 else None,
                "min_track_sensor": min(sensors) if sensors else None,
            }

        def _ensure_torcs(self) -> None:
            if self.runner.env is not None:
                return
            self._launch_and_connect_torcs()

        def _reset_torcs_env(self, *, relaunch: bool = False):
            last_error: Exception | None = None
            for attempt in range(self.reset_retries + 1):
                try:
                    if relaunch and not self.manual_start:
                        self._restart_torcs()
                    assert self.runner.env is not None
                    with maybe_suppress_stdout(self.quiet_reset_log):
                        return self.runner.env.reset(relaunch=relaunch)
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.reset_retries:
                        raise RuntimeError(
                            "TORCS reset failed after "
                            f"{self.reset_retries + 1} attempts"
                        ) from last_error
                    if not self.manual_start:
                        self._restart_torcs()
                    else:
                        self._close_torcs_env()
                        self._ensure_torcs()
                    time.sleep(1.0)
            raise RuntimeError("TORCS reset failed") from last_error

        def _restart_torcs(self) -> None:
            self._close_torcs_env()
            self.runner.shutdown()
            self._launch_and_connect_torcs()

        def _launch_and_connect_torcs(self) -> None:
            if not self.manual_start:
                self.runner.launch()
            self.runner.connect()
            self.runner.load_track(self.track_name)

        def _close_torcs_env(self) -> None:
            if self.runner.env is None:
                return
            try:
                self.runner.env.end()
            except Exception:
                pass
            self.runner.env = None

        def _build_observation(self) -> np.ndarray:
            if self.current_telemetry is None:
                return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            return np.asarray(
                build_td3_observation(
                    self.current_telemetry,
                    self.previous_action,
                    track_length_m=self.track_length_m,
                ),
                dtype=np.float32,
            )

        def _open_step_telemetry(self) -> None:
            if self.run_dir is None:
                return
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._step_file = (self.run_dir / "steps.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self._step_writer = csv.writer(self._step_file)
            self._step_writer.writerow(STEP_COLUMNS)
            self._step_file.flush()

        def _close_step_telemetry(self) -> None:
            if self._step_file is not None:
                self._step_file.close()
            self._step_file = None
            self._step_writer = None

        def _write_step_telemetry(
            self,
            telemetry: Mapping[str, Any],
            raw_action: np.ndarray,
            action: Mapping[str, Any],
            reward: float,
            reward_components: Mapping[str, Any],
            terminated: bool,
            truncated: bool,
            reason: str,
        ) -> None:
            if self._step_writer is None:
                return
            sensors = track_sensors(telemetry)
            padded = np.zeros(self.action_size, dtype=float)
            padded[: min(self.action_size, len(raw_action))] = raw_action[
                : self.action_size
            ]
            if self.use_legacy_action_contract:
                raw_accel = max(0.0, finite_clamp(padded[1], -1.0, 1.0))
                raw_brake = max(0.0, finite_clamp(padded[2], -1.0, 1.0))
                raw_longitudinal = clamp(raw_accel - raw_brake, -1.0, 1.0)
                pedal_conflict = (
                    raw_accel >= PEDAL_DEADZONE and raw_brake >= PEDAL_DEADZONE
                )
            else:
                raw_longitudinal = finite_clamp(padded[1], -1.0, 1.0)
                raw_accel = max(0.0, raw_longitudinal)
                raw_brake = max(0.0, -raw_longitudinal)
                pedal_conflict = False
            self._step_writer.writerow(
                [
                    self.episodes_started,
                    self.current_stage.stage_id,
                    self.steps,
                    self.total_steps,
                    telemetry.get("distFromStart", 0),
                    telemetry.get("distRaced", 0),
                    telemetry.get("speedX", 0),
                    telemetry.get("speedY", 0),
                    telemetry.get("angle", 0),
                    telemetry.get("trackPos", 0),
                    telemetry.get("damage", 0),
                    sensors[9] if len(sensors) > 9 else 0,
                    min(sensors) if sensors else 0,
                    padded[0],
                    raw_longitudinal,
                    raw_accel,
                    raw_brake,
                    pedal_conflict,
                    action["steer"],
                    action["accel"],
                    action["brake"],
                    action["gear"],
                    reward,
                    reward_components.get("progress", 0.0),
                    reward_components.get("aligned_speed", 0.0),
                    reward_components.get("new_distance", 0.0),
                    reward_components.get("stable_progress_bonus", 0.0),
                    reward_components.get("survival_cost", 0.0),
                    reward_components.get("lateral_position_penalty", 0.0),
                    reward_components.get("angle_penalty", 0.0),
                    reward_components.get("lateral_speed_penalty", 0.0),
                    reward_components.get("speed_steer_penalty", 0.0),
                    reward_components.get("spin_penalty", 0.0),
                    reward_components.get("closing_danger_penalty", 0.0),
                    reward_components.get("clear_track_anti_stall_penalty", 0.0),
                    reward_components.get("stalled_penalty", 0.0),
                    reward_components.get("off_track_penalty", 0.0),
                    reward_components.get("crash_penalty", 0.0),
                    reward_components.get("backwards_penalty", 0.0),
                    reward_components.get("stuck_penalty", 0.0),
                    reward_components.get("terminal_failure_penalty", 0.0),
                    reward_components.get("stage_success_bonus", 0.0),
                    reward_components.get("lap_completion_bonus", 0.0),
                    terminated,
                    truncated,
                    reason,
                    telemetry.get("curLapTime", 0),
                    *[
                        sensors[index] if len(sensors) > index else 0.0
                        for index in range(19)
                    ],
                ]
            )
            self._step_file.flush()

    return Td3ScratchTrainingEnv


class TrainingProgressState:
    def __init__(self) -> None:
        self.episodes_seen = 0
        self.last_episode: dict[str, Any] | None = None
        self.best_distance_m: float | None = None
        self.best_reward: float | None = None
        self.current_stage = CURRICULUM[0].name

    def record_episode(self, row: Mapping[str, Any]) -> None:
        self.episodes_seen = int(row.get("episodes_seen", self.episodes_seen))
        self.last_episode = dict(row)
        self.current_stage = str(row.get("stage_name") or self.current_stage)

    def record_best_distance(self, distance_m: float) -> None:
        self.best_distance_m = float(distance_m)

    def record_best_reward(self, reward: float) -> None:
        self.best_reward = float(reward)


@dataclass
class ProbeBudgetController:
    """Admit probes without allowing them to displace training interactions."""

    target_training_steps: int
    maximum_probe_steps: int
    maximum_probe_fraction: float = DEFAULT_MAX_PROBE_FRACTION
    training_steps: int = 0
    probe_steps: int = 0
    environment_steps: int = 0
    deferred_probe_requests: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        self.target_training_steps = max(1, int(self.target_training_steps))
        self.maximum_probe_steps = max(0, int(self.maximum_probe_steps))
        self.maximum_probe_fraction = float(self.maximum_probe_fraction)
        if not 0.0 < self.maximum_probe_fraction <= DEFAULT_MAX_PROBE_FRACTION:
            raise ValueError(
                "maximum_probe_fraction must preserve at least 80% training "
                f"interactions: expected (0, {DEFAULT_MAX_PROBE_FRACTION:.2f}]"
            )

    @property
    def ratio_limited_probe_steps(self) -> int:
        training_fraction = 1.0 - self.maximum_probe_fraction
        return max(
            0,
            int(
                math.floor(
                    self.target_training_steps
                    * self.maximum_probe_fraction
                    / training_fraction
                )
            ),
        )

    @property
    def effective_maximum_probe_steps(self) -> int:
        return min(self.maximum_probe_steps, self.ratio_limited_probe_steps)

    @property
    def maximum_environment_steps(self) -> int:
        return self.target_training_steps + self.effective_maximum_probe_steps

    @property
    def available_probe_steps(self) -> int:
        training_fraction = 1.0 - self.maximum_probe_fraction
        earned_capacity = int(
            math.floor(
                self.training_steps
                * self.maximum_probe_fraction
                / training_fraction
            )
        )
        admitted_capacity = min(
            self.effective_maximum_probe_steps,
            earned_capacity,
        )
        return max(0, admitted_capacity - self.probe_steps)

    def can_start_probe(
        self,
        estimated_steps: int,
        *,
        record_deferral: bool = True,
    ) -> bool:
        required = max(1, int(estimated_steps))
        admitted = required <= self.available_probe_steps
        if not admitted and record_deferral:
            self.deferred_probe_requests += 1
        return admitted

    @property
    def progress_remaining(self) -> float:
        completed = min(self.training_steps, self.target_training_steps)
        return 1.0 - completed / self.target_training_steps

    def record_infos(self, infos: list[Any] | tuple[Any, ...]) -> None:
        for info in infos:
            self.environment_steps += 1
            deterministic_probe = bool(
                isinstance(info, Mapping) and info.get("deterministic_probe")
            )
            if deterministic_probe:
                self.probe_steps += 1
            else:
                self.training_steps += 1
        self._update_stop_reason()

    def _update_stop_reason(self) -> None:
        if self.training_steps >= self.target_training_steps:
            self.stop_reason = "training_budget_reached"
        else:
            self.stop_reason = None

    def should_stop(self) -> bool:
        return self.stop_reason is not None

    def as_dict(self) -> dict[str, Any]:
        total = max(1, self.environment_steps)
        return {
            "target_training_steps": self.target_training_steps,
            "maximum_probe_steps": self.maximum_probe_steps,
            "ratio_limited_probe_steps": self.ratio_limited_probe_steps,
            "effective_maximum_probe_steps": (
                self.effective_maximum_probe_steps
            ),
            "maximum_probe_fraction": self.maximum_probe_fraction,
            "minimum_training_fraction": 1.0 - self.maximum_probe_fraction,
            "maximum_environment_steps": self.maximum_environment_steps,
            "training_steps": self.training_steps,
            "probe_steps": self.probe_steps,
            "environment_steps": self.environment_steps,
            "probe_fraction": self.probe_steps / total,
            "training_fraction": self.training_steps / total,
            "available_probe_steps": self.available_probe_steps,
            "deferred_probe_requests": self.deferred_probe_requests,
            "stop_reason": self.stop_reason,
        }


# Kept as an API alias for existing analysis scripts and saved run tooling.
TrainingBudgetState = ProbeBudgetController


def training_progress_is_credible(
    distances_m: list[float] | tuple[float, ...],
    *,
    reference_distance_m: float,
    minimum_episodes: int,
    maximum_regression_m: float,
    minimum_improvement_m: float = 0.0,
    completed_lap: bool = False,
) -> bool:
    """Return whether noisy training rollouts justify an on-track comparison."""
    if completed_lap:
        return True
    values = [
        max(0.0, finite_float(value))
        for value in distances_m
        if math.isfinite(finite_float(value))
    ]
    if len(values) < max(1, int(minimum_episodes)):
        return False
    reference = max(0.0, finite_float(reference_distance_m))
    if reference <= 0.0:
        return max(values) > 0.0
    regression_limit = max(0.0, finite_float(maximum_regression_m))
    improvement = max(0.0, finite_float(minimum_improvement_m))
    return bool(
        max(values) >= reference + improvement
        and float(np.median(values)) >= reference - regression_limit
    )


def continuation_compatibility_minimum_distance_m(
    args: argparse.Namespace,
) -> float:
    metadata = read_policy_metadata(args.resume_from)
    evaluation = metadata.get("best_evaluation")
    source_median = (
        finite_float(evaluation.get("median_progress_m"))
        if isinstance(evaluation, Mapping)
        else 0.0
    )
    return source_median * float(args.continuation_compatibility_min_fraction)


def run_continuation_compatibility_probe(
    model: Any,
    raw_env: Any,
    *,
    minimum_distance_m: float,
    training_budget_state: TrainingBudgetState | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Verify the loaded policy in TORCS before replay collection or updates."""
    minimum_distance = max(0.0, finite_float(minimum_distance_m))
    raw_env.set_policy_mode(
        deterministic_probe=True,
        action_noise_sigma=0.0,
        training_phase="continuation_compatibility_probe",
    )
    try:
        observation, _reset_info = raw_env.reset(seed=seed)
        final_info: Mapping[str, Any] | None = None
        maximum_steps = max(1, int(raw_env.current_stage.max_episode_steps)) + 1
        budget_limited = False
        if training_budget_state is not None:
            remaining_allowance = max(
                0,
                training_budget_state.effective_maximum_probe_steps
                - training_budget_state.probe_steps,
            )
            if remaining_allowance < 1:
                raise RuntimeError(
                    "continuation compatibility probe has no probe-budget "
                    "allowance; increase --total-timesteps"
                )
            budget_limited = remaining_allowance < maximum_steps
            maximum_steps = min(maximum_steps, remaining_allowance)
        for _step in range(maximum_steps):
            prediction = model.predict(observation, deterministic=True)
            raw_action = prediction[0] if isinstance(prediction, tuple) else prediction
            observation, _reward, terminated, truncated, info = raw_env.step(
                raw_action
            )
            if training_budget_state is not None:
                training_budget_state.record_infos([info])
            if terminated or truncated:
                final_info = info if isinstance(info, Mapping) else {}
                break
        if final_info is None:
            if budget_limited:
                raise RuntimeError(
                    "continuation compatibility probe exhausted its bounded "
                    "interaction allowance before the episode ended; increase "
                    "--total-timesteps"
                )
            raise RuntimeError(
                "continuation compatibility probe exceeded the curriculum episode limit"
            )
        summary = final_info.get("episode_summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError(
                "continuation compatibility probe produced no episode summary"
            )
        distance_m = max(
            finite_float(summary.get("distance_m")),
            finite_float(summary.get("furthest_distance_m")),
        )
        result = {
            "passed": distance_m >= minimum_distance,
            "minimum_distance_m": minimum_distance,
            "distance_m": distance_m,
            "steps": int(finite_float(summary.get("steps"))),
            "termination_reason": str(summary.get("termination_reason") or ""),
            "deterministic": True,
            "seed": seed,
        }
        if not result["passed"]:
            raise RuntimeError(
                "continuation compatibility probe failed before training: "
                f"reached {distance_m:.1f}m, required {minimum_distance:.1f}m. "
                "The source policy/action contract was not preserved."
            )
        return result
    finally:
        raw_env.set_policy_mode(
            deterministic_probe=False,
            action_noise_sigma=0.0,
            training_phase="replay_refill",
        )


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_training_progress_line(
    completed_steps: int,
    total_steps: int,
    elapsed_seconds: float,
    progress_state: TrainingProgressState,
    *,
    width: int = 28,
    probe_steps: int = 0,
    maximum_probe_steps: int | None = None,
) -> str:
    total_steps = max(1, int(total_steps))
    completed_steps = max(0, min(int(completed_steps), total_steps))
    fraction = completed_steps / total_steps
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    percent = fraction * 100.0
    if completed_steps > 0 and elapsed_seconds > 0.0:
        rate_value = completed_steps / elapsed_seconds
        eta = format_duration((total_steps - completed_steps) / max(rate_value, 1e-9))
        rate = f"{rate_value:,.0f} step/s"
    else:
        eta = "--:--:--"
        rate = "-- step/s"

    if progress_state.last_episode is None:
        episode_text = "episodes=0"
    else:
        episode = progress_state.last_episode
        episode_text = (
            f"ep {progress_state.episodes_seen} "
            f"{episode.get('stage_id', 'stage')} "
            f"{episode.get('termination_reason', 'running')} "
            f"{float(episode.get('distance_m') or 0.0):.0f}m "
            f"R={float(episode.get('reward') or 0.0):.0f}"
        )

    best_parts = []
    if progress_state.best_distance_m is not None:
        best_parts.append(f"best_dist={progress_state.best_distance_m:.0f}m")
    if progress_state.best_reward is not None:
        best_parts.append(f"best_R={progress_state.best_reward:.0f}")
    best_text = f" | {' '.join(best_parts)}" if best_parts else ""
    probe_text = f" | probes={max(0, int(probe_steps)):,}"
    if maximum_probe_steps is not None:
        probe_text += f"/{max(0, int(maximum_probe_steps)):,}"
    return (
        f"Training [{bar}] {percent:5.1f}% | "
        f"{completed_steps:,}/{total_steps:,} steps | {rate} | "
        f"ETA {eta}{probe_text} | stage={progress_state.current_stage} | "
        f"{episode_text}{best_text}"
    )


def make_episode_summary_callback_class(BaseCallback: Any):
    class EpisodeSummaryCallback(BaseCallback):
        def __init__(self, run_dir: Path, progress_state: TrainingProgressState | None = None):
            super().__init__(verbose=0)
            self.run_dir = Path(run_dir)
            self.progress_state = progress_state
            self.episodes_path = self.run_dir / "episodes.csv"
            self.episodes_seen = 0
            self._file = None
            self._writer = None

        def _on_training_start(self):
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._file = self.episodes_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_COLUMNS)
            self._writer.writeheader()
            self._file.flush()

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                self.episodes_seen += 1
                row = {
                    column: summary.get(column)
                    for column in EPISODE_COLUMNS
                    if column not in {"episodes_seen", "global_timestep"}
                }
                row["episodes_seen"] = self.episodes_seen
                row["global_timestep"] = self.num_timesteps
                self._writer.writerow(row)
                self._file.flush()
                if self.progress_state is not None:
                    self.progress_state.record_episode(row)
            return True

        def _on_training_end(self):
            if self._file is not None:
                self._file.close()
            self._file = None
            self._writer = None

    return EpisodeSummaryCallback


def make_exploration_schedule_callback_class(BaseCallback: Any):
    class ExplorationScheduleCallback(BaseCallback):
        def __init__(
            self,
            raw_env: Any,
            noise_factory: Any,
            *,
            learning_starts: int,
            initial_sigma: float,
            final_sigma: float,
            decay_steps: int,
            probe_interval: int,
            training_gradient_steps: int,
            pre_update_training_phase: str = "warmup",
            actor_update_start: int | None = None,
            actor_full_unfreeze: int | None = None,
            safe_probe_noise_factory: Any = None,
            safe_probe_noise_std: float = DEFAULT_SAFE_ACTOR_PROBE_NOISE_STD,
            training_budget_state: TrainingBudgetState | None = None,
            probe_reference_distance_m: float = 0.0,
            probe_evidence_window: int = DEFAULT_PROBE_EVIDENCE_WINDOW,
            probe_min_evidence_episodes: int = (
                DEFAULT_PROBE_MIN_EVIDENCE_EPISODES
            ),
            probe_evidence_max_regression_m: float = (
                DEFAULT_PROBE_EVIDENCE_MAX_REGRESSION_M
            ),
        ) -> None:
            super().__init__(verbose=0)
            self.raw_env = raw_env
            self.noise_factory = noise_factory
            self.learning_starts = max(0, int(learning_starts))
            self.initial_sigma = max(0.0, float(initial_sigma))
            self.final_sigma = max(0.0, float(final_sigma))
            self.decay_steps = max(1, int(decay_steps))
            self.probe_interval = max(1, int(probe_interval))
            self.training_gradient_steps = int(training_gradient_steps)
            self.pre_update_training_phase = str(pre_update_training_phase)
            self.actor_update_start = (
                self.learning_starts
                if actor_update_start is None
                else max(self.learning_starts, int(actor_update_start))
            )
            self.actor_full_unfreeze = (
                self.actor_update_start
                if actor_full_unfreeze is None
                else max(self.actor_update_start, int(actor_full_unfreeze))
            )
            self.safe_probe_noise_factory = safe_probe_noise_factory
            self.safe_probe_noise_std = max(0.0, float(safe_probe_noise_std))
            self.training_budget_state = training_budget_state
            self.probe_reference_distance_m = max(
                0.0,
                finite_float(probe_reference_distance_m),
            )
            self.probe_evidence_window = max(1, int(probe_evidence_window))
            self.probe_min_evidence_episodes = max(
                1,
                int(probe_min_evidence_episodes),
            )
            self.probe_evidence_max_regression_m = max(
                0.0,
                finite_float(probe_evidence_max_regression_m),
            )
            self.recent_training_distances_m: list[float] = []
            self.recent_training_completed_lap = False
            self.probe_episodes_completed = 0
            self.policy_episodes_since_probe = 0
            self.current_episode_is_probe = False
            self.last_noise_update_timestep = -ACTION_NOISE_UPDATE_INTERVAL_STEPS

        def _on_training_start(self):
            self._apply_policy_mode(deterministic_probe=False, force=True)

        def _on_step(self):
            completed_episode = False
            infos = self.locals.get("infos", [])
            if self.training_budget_state is not None:
                self.training_budget_state.record_infos(infos)
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                if self.training_budget_state is not None:
                    summary["training_budget_timestep"] = (
                        self.training_budget_state.training_steps
                    )
                    summary["probe_overhead_timestep"] = (
                        self.training_budget_state.probe_steps
                    )
                    summary["probe_budget_available_steps"] = (
                        self.training_budget_state.available_probe_steps
                    )
                    summary["probe_deferred_requests"] = (
                        self.training_budget_state.deferred_probe_requests
                    )
                completed_episode = True
                if bool(summary.get("policy_controlled")):
                    if bool(summary.get("deterministic_probe")):
                        safe_result = self._record_safe_actor_probe(summary)
                        if safe_result is not None:
                            self._annotate_safe_actor_probe(summary, safe_result)
                        else:
                            self._record_regular_probe_result(summary)
                        self.probe_episodes_completed += 1
                        self.policy_episodes_since_probe = 0
                    else:
                        self.policy_episodes_since_probe += 1
                        self._record_training_evidence(summary)

            if completed_episode:
                next_episode_is_probe = self._next_episode_should_probe()
                self._apply_policy_mode(
                    deterministic_probe=next_episode_is_probe,
                    force=True,
                )
            elif self.current_episode_is_probe:
                # The first probe action is selected after the preceding rollout's
                # final update. Freeze from this point until the probe terminates.
                self.model.gradient_steps = 0
            elif (
                self._budget_timestep() >= self.learning_starts
                and self.model.gradient_steps != self.training_gradient_steps
            ):
                self._apply_policy_mode(deterministic_probe=False, force=True)
            elif (
                not self.current_episode_is_probe
                and self._budget_timestep() - self.last_noise_update_timestep
                >= ACTION_NOISE_UPDATE_INTERVAL_STEPS
            ):
                self._apply_policy_mode(deterministic_probe=False, force=False)
            return not (
                self.training_budget_state is not None
                and self.training_budget_state.should_stop()
            )

        def _on_training_end(self):
            finalize = getattr(self.model, "finalize_safe_actor_candidate", None)
            if callable(finalize):
                result = finalize()
                if result is not None:
                    print(
                        "\nSafe actor candidate "
                        f"{result['candidate_id']} rolled back at training end "
                        "because its matched comparison was incomplete."
                    )
            sigma = action_noise_sigma_at_timestep(
                self._budget_timestep(),
                learning_starts=self.learning_starts,
                initial_sigma=self.initial_sigma,
                final_sigma=self.final_sigma,
                decay_steps=self.decay_steps,
            )
            self.model.action_noise = self.noise_factory(sigma)
            self.model.gradient_steps = self.training_gradient_steps

        def _apply_policy_mode(
            self,
            *,
            deterministic_probe: bool,
            force: bool,
        ) -> None:
            sigma = action_noise_sigma_at_timestep(
                self._budget_timestep(),
                learning_starts=self.learning_starts,
                initial_sigma=self.initial_sigma,
                final_sigma=self.final_sigma,
                decay_steps=self.decay_steps,
            )
            safe_probe_requested = bool(
                deterministic_probe and self._safe_actor_probe_required()
            )
            safe_context = (
                self._safe_actor_probe_context() if deterministic_probe else None
            )
            if safe_probe_requested and safe_context is None:
                deterministic_probe = False
            safe_actor_probe = safe_context is not None
            mode_changed = deterministic_probe != self.current_episode_is_probe
            if force or mode_changed or not deterministic_probe:
                if safe_actor_probe:
                    if self.safe_probe_noise_factory is None:
                        self.model.action_noise = SeededSteeringActionNoise(
                            seed=int(safe_context["seed"]),
                            steering_noise_std=self.safe_probe_noise_std,
                        )
                    else:
                        self.model.action_noise = self.safe_probe_noise_factory(
                            int(safe_context["seed"]),
                            self.safe_probe_noise_std,
                        )
                else:
                    self.model.action_noise = (
                        None if deterministic_probe else self.noise_factory(sigma)
                    )
            self.model.gradient_steps = (
                0
                if self._budget_timestep() < self.learning_starts
                else self.training_gradient_steps
            )
            self.current_episode_is_probe = deterministic_probe
            self.last_noise_update_timestep = self._budget_timestep()
            if safe_actor_probe:
                training_phase = f"safe_actor_{safe_context['mode']}_probe"
            elif deterministic_probe:
                training_phase = "deterministic_probe"
            else:
                training_phase = continuation_training_phase(
                    self._budget_timestep(),
                    gradient_update_start=self.learning_starts,
                    actor_update_start=self.actor_update_start,
                    actor_full_unfreeze=self.actor_full_unfreeze,
                    pre_update_phase=self.pre_update_training_phase,
                )
            self.raw_env.set_policy_mode(
                deterministic_probe=deterministic_probe,
                action_noise_sigma=(
                    self.safe_probe_noise_std if safe_actor_probe else sigma
                ),
                training_phase=training_phase,
                safe_actor_candidate_probe=safe_actor_probe,
                safe_actor_probe_mode=(
                    safe_context.get("mode") if safe_context else None
                ),
                safe_actor_probe_seed=(
                    safe_context.get("seed") if safe_context else None
                ),
                safe_actor_probe_trial=(
                    safe_context.get("trial_index") if safe_context else None
                ),
                safe_actor_trials_per_seed=(
                    safe_context.get("trials_per_seed") if safe_context else None
                ),
                safe_actor_probe_stage=(
                    safe_context.get("probe_stage") if safe_context else None
                ),
                safe_actor_probe_order_position=(
                    safe_context.get("order_position") if safe_context else None
                ),
            )

        def _budget_timestep(self) -> int:
            if self.training_budget_state is None:
                return int(self.num_timesteps)
            return int(self.training_budget_state.training_steps)

        def _safe_actor_probe_required(self) -> bool:
            required = getattr(self.model, "safe_actor_probe_required", None)
            return bool(callable(required) and required())

        def _safe_actor_probe_eligible(self) -> bool:
            eligible = getattr(self.model, "safe_actor_probe_eligible", None)
            return bool(eligible()) if callable(eligible) else True

        def _safe_actor_next_probe_stage(self) -> str:
            stage = getattr(self.model, "safe_actor_next_probe_stage", None)
            return str(stage()) if callable(stage) else "screening"

        def _pause_safe_actor_probe(self) -> None:
            pause = getattr(self.model, "pause_safe_actor_probe", None)
            if callable(pause):
                pause()

        def _safe_actor_probe_context(self) -> dict[str, Any] | None:
            context = getattr(self.model, "safe_actor_probe_context", None)
            value = context() if callable(context) else None
            return dict(value) if isinstance(value, Mapping) else None

        def _record_training_evidence(self, summary: Mapping[str, Any]) -> None:
            distance_m = max(
                finite_float(summary.get("distance_m")),
                finite_float(summary.get("furthest_distance_m")),
            )
            self.recent_training_distances_m.append(max(0.0, distance_m))
            self.recent_training_distances_m = self.recent_training_distances_m[
                -self.probe_evidence_window :
            ]
            self.recent_training_completed_lap = bool(
                self.recent_training_completed_lap
                or int(finite_float(summary.get("laps_completed"))) > 0
            )
            record = getattr(self.model, "record_safe_actor_training_episode", None)
            result = record(summary) if callable(record) else None
            if not isinstance(result, Mapping):
                return
            decision = str(result.get("decision") or "")
            if isinstance(summary, dict):
                summary["safe_actor_training_evidence_decision"] = decision
            if decision == "rolled_back_no_evidence":
                self.recent_training_distances_m = []
                self.recent_training_completed_lap = False
                print(
                    "\nSafe actor candidate "
                    f"{result['candidate_id']} rolled back before on-track probes: "
                    "recent training episodes showed no credible progress signal."
                )

        def _record_regular_probe_result(self, summary: Mapping[str, Any]) -> None:
            distance_m = max(
                finite_float(summary.get("distance_m")),
                finite_float(summary.get("furthest_distance_m")),
            )
            self.probe_reference_distance_m = max(
                self.probe_reference_distance_m,
                distance_m,
            )
            self.recent_training_distances_m = []
            self.recent_training_completed_lap = False

        def _regular_probe_signal_is_credible(self) -> bool:
            return training_progress_is_credible(
                self.recent_training_distances_m,
                reference_distance_m=self.probe_reference_distance_m,
                minimum_episodes=self.probe_min_evidence_episodes,
                maximum_regression_m=self.probe_evidence_max_regression_m,
                completed_lap=self.recent_training_completed_lap,
            )

        def _probe_step_estimate(self, *, safe_actor_probe: bool) -> int:
            stage = getattr(self.raw_env, "current_stage", None)
            if stage is None:
                return 1
            probe_stage = (
                self._safe_actor_next_probe_stage()
                if safe_actor_probe
                else "confirmation"
            )
            screening_limit = int(
                getattr(
                    self.raw_env,
                    "safe_actor_screening_max_episode_steps",
                    DEFAULT_SAFE_ACTOR_SCREENING_MAX_EPISODE_STEPS,
                )
            )
            return safe_actor_probe_episode_step_limit(
                stage,
                probe_stage=probe_stage,
                screening_max_episode_steps=screening_limit,
            )

        def _probe_budget_allows(self, *, safe_actor_probe: bool) -> bool:
            if self.training_budget_state is None:
                return True
            allowed = self.training_budget_state.can_start_probe(
                self._probe_step_estimate(safe_actor_probe=safe_actor_probe)
            )
            if not allowed and safe_actor_probe:
                self._pause_safe_actor_probe()
            return allowed

        def _next_episode_should_probe(self) -> bool:
            if self._safe_actor_probe_required():
                return bool(
                    self._safe_actor_probe_eligible()
                    and self._probe_budget_allows(safe_actor_probe=True)
                )
            scheduled = should_schedule_policy_probe(
                timestep=self._budget_timestep(),
                learning_starts=self.learning_starts,
                probe_episodes_completed=self.probe_episodes_completed,
                policy_episodes_since_probe=self.policy_episodes_since_probe,
                probe_interval=self.probe_interval,
            )
            return bool(
                scheduled
                and self._regular_probe_signal_is_credible()
                and self._probe_budget_allows(safe_actor_probe=False)
            )

        def _record_safe_actor_probe(
            self,
            summary: Mapping[str, Any],
        ) -> dict[str, Any] | None:
            record = getattr(self.model, "record_safe_actor_probe", None)
            if (
                not bool(summary.get("safe_actor_candidate_probe"))
                or not callable(record)
                or not self._safe_actor_probe_required()
            ):
                return None
            return record(summary)

        def _annotate_safe_actor_probe(
            self,
            summary: dict[str, Any],
            result: Mapping[str, Any],
        ) -> None:
            summary["safe_actor_candidate_probe"] = True
            summary["safe_actor_probe_mode"] = result.get("mode")
            summary["safe_actor_probe_seed"] = result.get("seed")
            summary["safe_actor_candidate_id"] = result.get("candidate_id")
            summary["safe_actor_probe_index"] = result.get("probe_index")
            summary["safe_actor_probe_trial"] = result.get("trial_index")
            summary["safe_actor_trials_per_seed"] = result.get(
                "trials_per_seed"
            )
            summary["safe_actor_probe_stage"] = result.get("probe_stage")
            summary["safe_actor_probe_order_position"] = result.get(
                "order_position"
            )
            summary["safe_actor_decision"] = result.get("decision")
            summary["safe_actor_probe_episodes_completed"] = result.get(
                "probe_episodes_completed"
            )
            summary["safe_actor_probe_episodes_maximum"] = result.get(
                "probe_episodes_maximum"
            )
            summary["safe_actor_probe_episodes_skipped"] = result.get(
                "probe_episodes_skipped"
            )
            summary["safe_actor_probe_median_distance_m"] = result.get(
                "median_distance_m"
            )
            summary["safe_actor_probe_minimum_distance_m"] = result.get(
                "minimum_distance_m"
            )
            summary["safe_actor_reference_median_distance_m"] = result.get(
                "reference_median_distance_m"
            )
            summary["safe_actor_reference_minimum_distance_m"] = result.get(
                "reference_minimum_distance_m"
            )
            summary["safe_actor_paired_median_delta_m"] = result.get(
                "paired_median_delta_m"
            )
            summary["safe_actor_worst_paired_delta_m"] = result.get(
                "worst_paired_delta_m"
            )
            summary["safe_actor_paired_regressions"] = result.get(
                "paired_regressions"
            )
            summary["safe_actor_worst_order_position_delta_m"] = result.get(
                "worst_order_position_delta_m"
            )
            summary["safe_actor_order_position_regressions"] = result.get(
                "order_position_regressions"
            )
            summary["safe_actor_order_position_deltas_m"] = result.get(
                "order_position_deltas_m"
            )
            summary["safe_actor_raw_paired_regressions"] = result.get(
                "raw_paired_regressions"
            )
            summary["safe_actor_reference_seed_medians_m"] = result.get(
                "reference_seed_medians_m"
            )
            summary["safe_actor_candidate_seed_medians_m"] = result.get(
                "candidate_seed_medians_m"
            )
            summary["safe_actor_accepted_distance_m"] = result.get(
                "accepted_distance_m"
            )
            summary["safe_actor_accepted_minimum_distance_m"] = result.get(
                "accepted_minimum_distance_m"
            )
            decision = str(result.get("decision") or "")
            if decision == "accepted":
                self.raw_env.accept_safe_actor_probe(
                    summary,
                    median_distance_m=float(result["median_distance_m"]),
                )
            terminal_decisions = {
                "accepted",
                "rolled_back",
                "rolled_back_early",
                "rolled_back_screening",
            }
            if decision in terminal_decisions:
                self.recent_training_distances_m = []
                self.recent_training_completed_lap = False
                if decision == "accepted":
                    self.probe_reference_distance_m = max(
                        self.probe_reference_distance_m,
                        finite_float(result.get("accepted_distance_m")),
                    )
                worst_position_delta = result.get(
                    "worst_order_position_delta_m"
                )
                if worst_position_delta is None:
                    worst_position_delta = result["worst_paired_delta_m"]
                probe_episodes_completed = result.get(
                    "probe_episodes_completed"
                )
                probe_episodes_maximum = result.get(
                    "probe_episodes_maximum"
                )
                probe_suffix = "."
                if (
                    probe_episodes_completed is not None
                    and probe_episodes_maximum is not None
                ):
                    probe_suffix = (
                        f"; probes={probe_episodes_completed}/"
                        f"{probe_episodes_maximum}."
                    )
                print(
                    "\nSafe actor candidate "
                    f"{result['candidate_id']} {decision}: "
                    "reference median="
                    f"{float(result['reference_median_distance_m']):.1f}m, "
                    "candidate median="
                    f"{float(result['median_distance_m']):.1f}m, "
                    "paired seed median="
                    f"{float(result['paired_median_delta_m']):+.1f}m, "
                    "worst seed="
                    f"{float(result['worst_paired_delta_m']):+.1f}m, "
                    "worst seed-position="
                    f"{float(worst_position_delta):+.1f}m"
                    f"{probe_suffix}"
                )

    return ExplorationScheduleCallback


def make_best_model_callback_class(BaseCallback: Any):
    class BestModelCallback(BaseCallback):
        def __init__(
            self,
            best_distance_model_path: Path,
            best_reward_model_path: Path,
            metadata: Mapping[str, Any],
            progress_state: TrainingProgressState | None = None,
            learning_starts: int = 0,
            save_best_replay_buffer: bool = False,
            training_budget_state: TrainingBudgetState | None = None,
        ) -> None:
            super().__init__(verbose=0)
            self.best_distance_model_path = Path(best_distance_model_path)
            self.best_reward_model_path = Path(best_reward_model_path)
            self.metadata = dict(metadata)
            self.progress_state = progress_state
            self.learning_starts = max(0, int(learning_starts))
            self.save_best_replay_buffer = bool(save_best_replay_buffer)
            self.training_budget_state = training_budget_state
            self.best_distance_m = self._read_existing_best_distance()
            self.best_reward = self._read_existing_best_reward()
            if self.progress_state is not None:
                if self.best_distance_m is not None:
                    self.progress_state.record_best_distance(self.best_distance_m)
                if self.best_reward is not None:
                    self.progress_state.record_best_reward(self.best_reward)

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                if (
                    bool(summary.get("safe_actor_candidate_probe"))
                    and summary.get("safe_actor_decision") != "accepted"
                ):
                    continue
                if (
                    not bool(summary.get("curriculum_eligible"))
                    or self._budget_timestep() <= self.learning_starts
                ):
                    continue
                distance_m = float(summary.get("distance_m") or 0.0)
                reward = float(summary.get("reward") or 0.0)
                if self.best_distance_m is None or distance_m > self.best_distance_m:
                    self.best_distance_m = distance_m
                    if self.progress_state is not None:
                        self.progress_state.record_best_distance(distance_m)
                    self._save_best(
                        self.best_distance_model_path,
                        "best_distance_episode",
                        {
                            "selection_reason": (
                                "furthest distance reached by deterministic TD3 probe"
                            ),
                            "distance_m": distance_m,
                            "episode_summary": summary,
                        },
                        save_replay_buffer=self.save_best_replay_buffer,
                    )
                if self.best_reward is None or reward > self.best_reward:
                    self.best_reward = reward
                    if self.progress_state is not None:
                        self.progress_state.record_best_reward(reward)
                    self._save_best(
                        self.best_reward_model_path,
                        "best_reward_episode",
                        {
                            "selection_reason": (
                                "highest shaped reward reached by deterministic TD3 probe"
                            ),
                            "reward": reward,
                            "distance_m": distance_m,
                            "episode_summary": summary,
                        },
                        save_replay_buffer=self.save_best_replay_buffer,
                    )
            return True

        def _budget_timestep(self) -> int:
            if self.training_budget_state is None:
                return int(self.num_timesteps)
            return int(self.training_budget_state.training_steps)

        def _save_best(
            self,
            path: Path,
            metadata_key: str,
            payload: Mapping[str, Any],
            *,
            save_replay_buffer: bool = False,
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            saved_payload = dict(payload)
            if save_replay_buffer:
                replay_buffer_path = replay_buffer_path_for_policy(path)
                replay_error = save_replay_buffer_atomically(
                    self.model,
                    replay_buffer_path,
                )
                if replay_error is None:
                    saved_payload["replay_buffer_path"] = str(replay_buffer_path)
                else:
                    saved_payload["replay_buffer_save_error"] = replay_error
                    print(
                        "Warning: best TD3 model was saved, but its replay buffer "
                        f"was not: {replay_error}"
                    )
            write_json(
                metadata_path_for_policy(path),
                {
                    **self.metadata,
                    metadata_key: {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "global_timestep": self.num_timesteps,
                        **saved_payload,
                    },
                },
            )

        def _read_existing_best_distance(self) -> float | None:
            metadata = read_policy_metadata(self.best_distance_model_path)
            best_episode = metadata.get("best_distance_episode")
            if not self._is_verified_policy_episode(best_episode):
                return None
            distance_m = finite_float(best_episode.get("distance_m"), default=-1.0)
            return distance_m if distance_m >= 0.0 else None

        def _read_existing_best_reward(self) -> float | None:
            metadata = read_policy_metadata(self.best_reward_model_path)
            best_episode = metadata.get("best_reward_episode")
            if not self._is_verified_policy_episode(best_episode):
                return None
            reward = finite_float(best_episode.get("reward"), default=float("-inf"))
            return reward if math.isfinite(reward) else None

        @staticmethod
        def _is_verified_policy_episode(best_episode: Any) -> bool:
            return episode_metadata_is_deterministic_probe(best_episode)

    return BestModelCallback


def make_console_progress_callback_class(BaseCallback: Any):
    class ConsoleProgressCallback(BaseCallback):
        def __init__(
            self,
            total_timesteps: int,
            progress_state: TrainingProgressState,
            *,
            interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
            training_budget_state: TrainingBudgetState | None = None,
        ) -> None:
            super().__init__(verbose=0)
            self.total_timesteps = int(total_timesteps)
            self.progress_state = progress_state
            self.training_budget_state = training_budget_state
            self.interval_seconds = float(interval_seconds)
            self.start_timestep = 0
            self.start_time = 0.0
            self.last_render_time = 0.0
            self.previous_line_length = 0

        def _on_training_start(self):
            self.start_timestep = self.num_timesteps
            self.start_time = time.monotonic()
            self._render(force=True)

        def _on_step(self):
            now = time.monotonic()
            if now - self.last_render_time >= self.interval_seconds:
                self._render(now=now)
            return True

        def _on_training_end(self):
            self._render(force=True)
            if self.previous_line_length:
                print()

        def _render(self, *, now: float | None = None, force: bool = False):
            now = time.monotonic() if now is None else now
            if not force and now - self.last_render_time < self.interval_seconds:
                return
            completed_steps = (
                self.training_budget_state.training_steps
                if self.training_budget_state is not None
                else self.num_timesteps - self.start_timestep
            )
            line = format_training_progress_line(
                completed_steps,
                self.total_timesteps,
                now - self.start_time,
                self.progress_state,
                probe_steps=(
                    self.training_budget_state.probe_steps
                    if self.training_budget_state is not None
                    else 0
                ),
                maximum_probe_steps=(
                    self.training_budget_state.effective_maximum_probe_steps
                    if self.training_budget_state is not None
                    else None
                ),
            )
            terminal_width = shutil.get_terminal_size((120, 20)).columns
            max_width = max(60, terminal_width - 1)
            if len(line) > max_width:
                line = f"{line[: max_width - 3]}..."
            padding = " " * max(0, self.previous_line_length - len(line))
            print(f"\r{line}{padding}", end="", flush=True)
            self.previous_line_length = len(line)
            self.last_render_time = now

    return ConsoleProgressCallback


def make_checkpoint_callback(
    CheckpointCallback: Any,
    args: argparse.Namespace,
    *,
    BaseCallback: Any = None,
    training_budget_state: TrainingBudgetState | None = None,
):
    if training_budget_state is not None and BaseCallback is not None:
        class TrainingBudgetCheckpointCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.next_checkpoint_step = int(args.checkpoint_freq)

            def _on_step(self):
                training_steps = training_budget_state.training_steps
                if training_steps < self.next_checkpoint_step:
                    return True
                checkpoint_stem = (
                    Path(args.checkpoint_dir)
                    / f"agent6_td3_scratch_{training_steps}_steps"
                )
                self.model.save(str(checkpoint_stem))
                if args.save_checkpoint_replay_buffer:
                    self.model.save_replay_buffer(
                        str(
                            checkpoint_stem.with_name(
                                f"{checkpoint_stem.name}_replay_buffer.pkl"
                            )
                        )
                    )
                while self.next_checkpoint_step <= training_steps:
                    self.next_checkpoint_step += int(args.checkpoint_freq)
                return True

        return TrainingBudgetCheckpointCallback()

    checkpoint_kwargs = {
        "save_freq": args.checkpoint_freq,
        "save_path": str(args.checkpoint_dir),
        "name_prefix": "agent6_td3_scratch",
        "save_replay_buffer": args.save_checkpoint_replay_buffer,
        "save_vecnormalize": False,
    }
    try:
        return CheckpointCallback(**checkpoint_kwargs)
    except TypeError:
        checkpoint_kwargs.pop("save_replay_buffer", None)
        return CheckpointCallback(**checkpoint_kwargs)


def tensorboard_is_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("tensorboard") is not None


def resolve_tensorboard_log_dir(tensorboard_dir: Path, *, disable_tensorboard: bool):
    if disable_tensorboard:
        return None
    if tensorboard_is_available():
        return str(tensorboard_dir)
    print(
        "TensorBoard is not installed, so TD3 TensorBoard logging is disabled. "
        "Training will still write CSV run logs and metadata."
    )
    return None


def make_phased_continuation_td3_class(TD3: Any):
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3.common.utils import polyak_update, update_learning_rate

    class PhasedContinuationTD3(TD3):
        """TD3 continuation with phased and probe-gated actor improvement."""

        def _excluded_save_params(self) -> list[str]:
            return [
                *super()._excluded_save_params(),
                "_agent6_safe_actor_reference",
                "_agent6_safe_actor_state",
                "_agent6_safe_actor_target_state",
                "_agent6_safe_actor_optimizer_state",
                "_agent6_safe_actor_anchor_observations",
                "_agent6_safe_actor_candidate_state",
                "_agent6_safe_actor_candidate_target_state",
                "_agent6_safe_actor_candidate_optimizer_state",
                "_agent6_training_budget_state",
            ]

        def configure_training_budget(
            self,
            training_budget_state: TrainingBudgetState,
        ) -> None:
            self._agent6_training_budget_state = training_budget_state

        def _training_budget_timestep(self) -> int:
            state = getattr(self, "_agent6_training_budget_state", None)
            if state is None:
                return int(self.num_timesteps)
            return int(state.training_steps)

        def _apply_training_budget_progress(self) -> None:
            state = getattr(self, "_agent6_training_budget_state", None)
            if state is not None:
                self._current_progress_remaining = state.progress_remaining

        def save(
            self,
            path: Any,
            exclude: Any = None,
            include: Any = None,
        ) -> None:
            if (
                not getattr(self, "_agent6_safe_actor_enabled", False)
                or int(getattr(self, "_agent6_safe_actor_candidate_updates", 0))
                <= 0
            ):
                super().save(path, exclude=exclude, include=include)
                return

            runtime_actor = self._clone_module_state(self.actor)
            runtime_actor_target = self._clone_module_state(self.actor_target)
            runtime_optimizer = copy.deepcopy(self.actor.optimizer.state_dict())
            self._restore_safe_actor()
            try:
                super().save(path, exclude=exclude, include=include)
            finally:
                self.actor.load_state_dict(runtime_actor)
                self.actor_target.load_state_dict(runtime_actor_target)
                self.actor.optimizer.load_state_dict(runtime_optimizer)

        def configure_continuation_updates(
            self,
            *,
            actor_learning_rate_schedule: Callable[[int], float],
            actor_updates_start_at: int,
            safe_actor_improvement: bool = False,
            safe_actor_max_action_delta: float = DEFAULT_SAFE_ACTOR_MAX_ACTION_DELTA,
            safe_actor_gradient_norm: float = DEFAULT_SAFE_ACTOR_GRADIENT_NORM,
            safe_actor_block_updates: int = DEFAULT_SAFE_ACTOR_BLOCK_UPDATES,
            safe_actor_blocks_per_probe: int = (
                DEFAULT_SAFE_ACTOR_BLOCKS_PER_PROBE
            ),
            safe_actor_probe_repeats: int = DEFAULT_SAFE_ACTOR_PROBE_REPEATS,
            safe_actor_trials_per_seed: int = (
                DEFAULT_SAFE_ACTOR_TRIALS_PER_SEED
            ),
            safe_actor_screening_trials_per_seed: int = (
                DEFAULT_SAFE_ACTOR_SCREENING_TRIALS_PER_SEED
            ),
            safe_actor_min_improvement_m: float = (
                DEFAULT_SAFE_ACTOR_MIN_IMPROVEMENT_M
            ),
            safe_actor_max_paired_regression_m: float = (
                DEFAULT_SAFE_ACTOR_MAX_PAIRED_REGRESSION_M
            ),
            safe_actor_screening_max_regression_m: float = (
                DEFAULT_SAFE_ACTOR_SCREENING_MAX_REGRESSION_M
            ),
            safe_actor_anchor_batch_size: int = (
                DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE
            ),
            safe_actor_probe_base_seed: int = DEFAULT_SAFE_ACTOR_PROBE_BASE_SEED,
            safe_actor_evidence_window: int = DEFAULT_PROBE_EVIDENCE_WINDOW,
            safe_actor_min_evidence_episodes: int = 0,
            safe_actor_evidence_max_regression_m: float = (
                DEFAULT_PROBE_EVIDENCE_MAX_REGRESSION_M
            ),
            source_verified_distance_m: float = 0.0,
            source_verified_minimum_distance_m: float = 0.0,
        ) -> None:
            self._agent6_actor_learning_rate_schedule = actor_learning_rate_schedule
            self._agent6_actor_updates_start_at = max(
                0,
                int(actor_updates_start_at),
            )
            self._agent6_safe_actor_enabled = bool(safe_actor_improvement)
            self._agent6_safe_actor_max_action_delta = max(
                0.0,
                float(safe_actor_max_action_delta),
            )
            self._agent6_safe_actor_gradient_norm = max(
                0.0,
                float(safe_actor_gradient_norm),
            )
            self._agent6_safe_actor_block_updates = max(
                1,
                int(safe_actor_block_updates),
            )
            self._agent6_safe_actor_blocks_per_probe = max(
                1,
                int(safe_actor_blocks_per_probe),
            )
            self._agent6_safe_actor_probe_repeats = max(
                1,
                int(safe_actor_probe_repeats),
            )
            self._agent6_safe_actor_trials_per_seed = int(
                safe_actor_trials_per_seed
            )
            if self._agent6_safe_actor_enabled and (
                self._agent6_safe_actor_trials_per_seed < 2
                or self._agent6_safe_actor_trials_per_seed % 2 != 0
            ):
                raise ValueError(
                    "safe actor robustness probes require an even number of "
                    "trials per seed, with at least two"
                )
            self._agent6_safe_actor_screening_trials_per_seed = int(
                safe_actor_screening_trials_per_seed
            )
            if self._agent6_safe_actor_enabled and (
                self._agent6_safe_actor_screening_trials_per_seed < 2
                or self._agent6_safe_actor_screening_trials_per_seed % 2 != 0
                or self._agent6_safe_actor_screening_trials_per_seed
                > self._agent6_safe_actor_trials_per_seed
            ):
                raise ValueError(
                    "safe actor screening requires an even number of trials "
                    "between two and the full trials-per-seed count"
                )
            self._agent6_safe_actor_min_improvement_m = max(
                0.0,
                float(safe_actor_min_improvement_m),
            )
            self._agent6_safe_actor_max_paired_regression_m = max(
                0.0,
                float(safe_actor_max_paired_regression_m),
            )
            self._agent6_safe_actor_screening_max_regression_m = max(
                0.0,
                float(safe_actor_screening_max_regression_m),
            )
            if (
                self._agent6_safe_actor_enabled
                and self._agent6_safe_actor_screening_max_regression_m
                < self._agent6_safe_actor_max_paired_regression_m
            ):
                raise ValueError(
                    "safe actor screening regression limit must be no smaller "
                    "than the final paired regression limit"
                )
            self._agent6_safe_actor_anchor_batch_size = max(
                1,
                int(safe_actor_anchor_batch_size),
            )
            probe_base_seed = int(safe_actor_probe_base_seed)
            if not 0 <= probe_base_seed <= MAX_REPRODUCIBLE_SEED:
                raise ValueError(
                    "safe_actor_probe_base_seed must use the 32-bit unsigned range"
                )
            self._agent6_safe_actor_probe_base_seed = probe_base_seed
            self._agent6_safe_actor_evidence_window = max(
                1,
                int(safe_actor_evidence_window),
            )
            self._agent6_safe_actor_min_evidence_episodes = max(
                0,
                int(safe_actor_min_evidence_episodes),
            )
            self._agent6_safe_actor_evidence_max_regression_m = max(
                0.0,
                finite_float(safe_actor_evidence_max_regression_m),
            )
            self._agent6_safe_actor_accepted_distance_m = max(
                0.0,
                finite_float(source_verified_distance_m),
            )
            self._agent6_safe_actor_accepted_minimum_distance_m = max(
                0.0,
                finite_float(source_verified_minimum_distance_m),
            )
            self._agent6_safe_actor_candidate_id = 0
            self._agent6_safe_actor_candidate_updates = 0
            self._agent6_safe_actor_waiting_for_probe = False
            self._agent6_safe_actor_probe_phase = "idle"
            self._agent6_safe_actor_probe_seed_index = 0
            self._agent6_safe_actor_probe_trial_index = 0
            self._agent6_safe_actor_probe_order_position = 0
            self._agent6_safe_actor_probe_seeds: tuple[int, ...] = ()
            self._agent6_safe_actor_reference_distances_m: list[float] = []
            self._agent6_safe_actor_candidate_distances_m: list[float] = []
            self._agent6_safe_actor_reference_order_positions: list[int] = []
            self._agent6_safe_actor_candidate_order_positions: list[int] = []
            self._agent6_safe_actor_accepted_blocks = 0
            self._agent6_safe_actor_rolled_back_blocks = 0
            self._agent6_safe_actor_screened_out_blocks = 0
            self._agent6_safe_actor_early_rolled_back_blocks = 0
            self._agent6_safe_actor_equivalent_blocks = 0
            self._agent6_safe_actor_no_evidence_blocks = 0
            self._agent6_safe_actor_invalid_updates = 0
            self._agent6_safe_actor_probe_episodes_skipped = 0
            self._agent6_safe_actor_evidence_candidate_id = 0
            self._agent6_safe_actor_evidence_distances_m: list[float] = []
            self._agent6_safe_actor_evidence_completed_lap = False
            self._agent6_safe_actor_evidence_eligible = bool(
                self._agent6_safe_actor_min_evidence_episodes == 0
            )
            self._agent6_safe_actor_last_decision: dict[str, Any] | None = None
            self._agent6_safe_actor_anchor_observations = None
            self._clear_safe_actor_candidate_snapshot()
            if self._agent6_safe_actor_enabled:
                self._snapshot_safe_actor()

        @staticmethod
        def _clone_module_state(module: Any) -> dict[str, Any]:
            return {
                name: value.detach().clone()
                for name, value in module.state_dict().items()
            }

        def _snapshot_safe_actor(self) -> None:
            self._agent6_safe_actor_state = self._clone_module_state(self.actor)
            self._agent6_safe_actor_target_state = self._clone_module_state(
                self.actor_target
            )
            self._agent6_safe_actor_optimizer_state = copy.deepcopy(
                self.actor.optimizer.state_dict()
            )
            self._agent6_safe_actor_reference = copy.deepcopy(self.actor)
            self._agent6_safe_actor_reference.set_training_mode(False)
            for parameter in self._agent6_safe_actor_reference.parameters():
                parameter.requires_grad_(False)

        def _restore_safe_actor(self) -> None:
            self.actor.load_state_dict(self._agent6_safe_actor_state)
            self.actor_target.load_state_dict(self._agent6_safe_actor_target_state)
            self.actor.optimizer.load_state_dict(
                copy.deepcopy(self._agent6_safe_actor_optimizer_state)
            )

        def _snapshot_safe_actor_candidate(self) -> None:
            self._agent6_safe_actor_candidate_state = self._clone_module_state(
                self.actor
            )
            self._agent6_safe_actor_candidate_target_state = (
                self._clone_module_state(self.actor_target)
            )
            self._agent6_safe_actor_candidate_optimizer_state = copy.deepcopy(
                self.actor.optimizer.state_dict()
            )

        def _restore_safe_actor_candidate(self) -> None:
            self.actor.load_state_dict(self._agent6_safe_actor_candidate_state)
            self.actor_target.load_state_dict(
                self._agent6_safe_actor_candidate_target_state
            )
            self.actor.optimizer.load_state_dict(
                copy.deepcopy(
                    self._agent6_safe_actor_candidate_optimizer_state
                )
            )

        def _clear_safe_actor_candidate_snapshot(self) -> None:
            self._agent6_safe_actor_candidate_state = None
            self._agent6_safe_actor_candidate_target_state = None
            self._agent6_safe_actor_candidate_optimizer_state = None

        def _candidate_actor_is_policy_equivalent(self) -> bool:
            return parameter_state_dicts_are_identical(
                self._agent6_safe_actor_state,
                self.actor.state_dict(),
            )

        def _resolve_policy_equivalent_candidate(self) -> bool:
            if (
                not self._agent6_safe_actor_waiting_for_probe
                or self._agent6_safe_actor_probe_phase != "idle"
                or not self._candidate_actor_is_policy_equivalent()
            ):
                return False
            candidate_id = self._agent6_safe_actor_candidate_id
            self._restore_safe_actor()
            self._agent6_safe_actor_candidate_updates = 0
            self._agent6_safe_actor_waiting_for_probe = False
            self._agent6_safe_actor_equivalent_blocks += 1
            skipped_episodes = (
                2
                * self._agent6_safe_actor_probe_repeats
                * self._agent6_safe_actor_trials_per_seed
            )
            self._agent6_safe_actor_probe_episodes_skipped += skipped_episodes
            self._agent6_safe_actor_last_decision = {
                "candidate_id": candidate_id,
                "decision": "policy_equivalent",
                "probe_episodes_skipped": skipped_episodes,
            }
            self._clear_safe_actor_candidate_snapshot()
            print(
                "\nSafe actor candidate "
                f"{candidate_id} is policy-equivalent to the accepted actor; "
                "skipped matched robustness probes."
            )
            return True

        def _begin_safe_actor_comparison(self) -> None:
            if self._agent6_safe_actor_probe_phase != "idle":
                return
            self._snapshot_safe_actor_candidate()
            self._agent6_safe_actor_probe_seeds = (
                build_rotating_training_seed_suite(
                    repeats=self._agent6_safe_actor_probe_repeats,
                    base_seed=self._agent6_safe_actor_probe_base_seed,
                    candidate_id=self._agent6_safe_actor_candidate_id,
                )
            )
            self._agent6_safe_actor_probe_seed_index = 0
            self._agent6_safe_actor_probe_trial_index = 0
            self._agent6_safe_actor_probe_order_position = 0
            self._agent6_safe_actor_reference_distances_m = []
            self._agent6_safe_actor_candidate_distances_m = []
            self._agent6_safe_actor_reference_order_positions = []
            self._agent6_safe_actor_candidate_order_positions = []
            self._activate_safe_actor_probe_mode(
                self._current_safe_actor_probe_order()[0]
            )

        def _current_safe_actor_probe_order(self) -> tuple[str, str]:
            return counterbalanced_pair_order(
                seed_index=self._agent6_safe_actor_probe_seed_index,
                trial_index=self._agent6_safe_actor_probe_trial_index,
            )

        def _activate_safe_actor_probe_mode(self, mode: str) -> None:
            self._agent6_safe_actor_probe_phase = str(mode)
            if mode == "accepted_reference":
                self._restore_safe_actor()
                return
            if mode == "candidate":
                self._restore_safe_actor_candidate()
                return
            raise ValueError(f"unknown safe actor probe mode: {mode}")

        def safe_actor_probe_required(self) -> bool:
            return bool(
                getattr(self, "_agent6_safe_actor_enabled", False)
                and getattr(
                    self,
                    "_agent6_safe_actor_waiting_for_probe",
                    False,
                )
            )

        def safe_actor_probe_eligible(self) -> bool:
            return bool(
                self.safe_actor_probe_required()
                and getattr(
                    self,
                    "_agent6_safe_actor_evidence_eligible",
                    True,
                )
            )

        def safe_actor_next_probe_stage(self) -> str:
            first_seed = self._agent6_safe_actor_probe_seed_index == 0
            within_screen = (
                self._agent6_safe_actor_probe_trial_index
                < self._agent6_safe_actor_screening_trials_per_seed
            )
            return "screening" if first_seed and within_screen else "confirmation"

        def pause_safe_actor_probe(self) -> None:
            if (
                self.safe_actor_probe_required()
                and self._agent6_safe_actor_candidate_state is not None
            ):
                self._restore_safe_actor_candidate()

        def record_safe_actor_training_episode(
            self,
            summary: Mapping[str, Any],
        ) -> dict[str, Any] | None:
            if (
                not getattr(self, "_agent6_safe_actor_enabled", False)
                or self._agent6_safe_actor_candidate_updates <= 0
                or bool(summary.get("deterministic_probe"))
            ):
                return None
            candidate_id = self._agent6_safe_actor_candidate_id
            if self._agent6_safe_actor_evidence_candidate_id != candidate_id:
                self._agent6_safe_actor_evidence_candidate_id = candidate_id
                self._agent6_safe_actor_evidence_distances_m = []
                self._agent6_safe_actor_evidence_completed_lap = False
                self._agent6_safe_actor_evidence_eligible = bool(
                    self._agent6_safe_actor_min_evidence_episodes == 0
                )

            distance_m = max(
                finite_float(summary.get("distance_m")),
                finite_float(summary.get("furthest_distance_m")),
            )
            self._agent6_safe_actor_evidence_distances_m.append(
                max(0.0, distance_m)
            )
            self._agent6_safe_actor_evidence_distances_m = (
                self._agent6_safe_actor_evidence_distances_m[
                    -self._agent6_safe_actor_evidence_window :
                ]
            )
            self._agent6_safe_actor_evidence_completed_lap = bool(
                self._agent6_safe_actor_evidence_completed_lap
                or int(finite_float(summary.get("laps_completed"))) > 0
            )
            credible = training_progress_is_credible(
                self._agent6_safe_actor_evidence_distances_m,
                reference_distance_m=(
                    self._agent6_safe_actor_accepted_distance_m
                ),
                minimum_episodes=max(
                    1,
                    self._agent6_safe_actor_min_evidence_episodes,
                ),
                maximum_regression_m=(
                    self._agent6_safe_actor_evidence_max_regression_m
                ),
                minimum_improvement_m=(
                    self._agent6_safe_actor_min_improvement_m
                ),
                completed_lap=(
                    self._agent6_safe_actor_evidence_completed_lap
                ),
            )
            self._agent6_safe_actor_evidence_eligible = bool(
                credible or self._agent6_safe_actor_min_evidence_episodes == 0
            )
            if self._agent6_safe_actor_evidence_eligible:
                return {
                    "candidate_id": candidate_id,
                    "decision": "probe_eligible",
                    "evidence_episodes": len(
                        self._agent6_safe_actor_evidence_distances_m
                    ),
                }

            evidence_complete = (
                len(self._agent6_safe_actor_evidence_distances_m)
                >= self._agent6_safe_actor_evidence_window
            )
            if not (self.safe_actor_probe_required() and evidence_complete):
                return {
                    "candidate_id": candidate_id,
                    "decision": "evidence_pending",
                    "evidence_episodes": len(
                        self._agent6_safe_actor_evidence_distances_m
                    ),
                }

            skipped_episodes = (
                2
                * self._agent6_safe_actor_probe_repeats
                * self._agent6_safe_actor_trials_per_seed
            )
            self._restore_safe_actor()
            self._agent6_safe_actor_rolled_back_blocks += 1
            self._agent6_safe_actor_no_evidence_blocks += 1
            self._agent6_safe_actor_probe_episodes_skipped += skipped_episodes
            self._agent6_safe_actor_last_decision = {
                "candidate_id": candidate_id,
                "decision": "rolled_back_no_evidence",
                "probe_episodes_skipped": skipped_episodes,
            }
            self._reset_safe_actor_probe_runtime()
            return {
                "candidate_id": candidate_id,
                "decision": "rolled_back_no_evidence",
                "probe_episodes_skipped": skipped_episodes,
            }

        def safe_actor_probe_context(self) -> dict[str, Any] | None:
            if not self.safe_actor_probe_eligible():
                return None
            if self._resolve_policy_equivalent_candidate():
                return None
            self._begin_safe_actor_comparison()
            self._activate_safe_actor_probe_mode(
                self._agent6_safe_actor_probe_phase
            )
            probe_index = self._agent6_safe_actor_probe_seed_index + 1
            trial_index = self._agent6_safe_actor_probe_trial_index + 1
            return {
                "mode": self._agent6_safe_actor_probe_phase,
                "candidate_id": self._agent6_safe_actor_candidate_id,
                "probe_index": probe_index,
                "probe_repeats": self._agent6_safe_actor_probe_repeats,
                "trial_index": trial_index,
                "trials_per_seed": self._agent6_safe_actor_trials_per_seed,
                "probe_stage": self.safe_actor_next_probe_stage(),
                "order_position": (
                    self._agent6_safe_actor_probe_order_position + 1
                ),
                "seed": self._agent6_safe_actor_probe_seeds[probe_index - 1],
            }

        def safe_actor_status(self) -> dict[str, Any]:
            return {
                "enabled": bool(
                    getattr(self, "_agent6_safe_actor_enabled", False)
                ),
                "accepted_distance_m": float(
                    getattr(self, "_agent6_safe_actor_accepted_distance_m", 0.0)
                ),
                "accepted_minimum_distance_m": float(
                    getattr(
                        self,
                        "_agent6_safe_actor_accepted_minimum_distance_m",
                        0.0,
                    )
                ),
                "accepted_blocks": int(
                    getattr(self, "_agent6_safe_actor_accepted_blocks", 0)
                ),
                "rolled_back_blocks": int(
                    getattr(self, "_agent6_safe_actor_rolled_back_blocks", 0)
                ),
                "screened_out_blocks": int(
                    getattr(self, "_agent6_safe_actor_screened_out_blocks", 0)
                ),
                "early_rolled_back_blocks": int(
                    getattr(
                        self,
                        "_agent6_safe_actor_early_rolled_back_blocks",
                        0,
                    )
                ),
                "policy_equivalent_blocks": int(
                    getattr(self, "_agent6_safe_actor_equivalent_blocks", 0)
                ),
                "no_evidence_blocks": int(
                    getattr(self, "_agent6_safe_actor_no_evidence_blocks", 0)
                ),
                "invalid_actor_updates": int(
                    getattr(self, "_agent6_safe_actor_invalid_updates", 0)
                ),
                "probe_episodes_skipped": int(
                    getattr(
                        self,
                        "_agent6_safe_actor_probe_episodes_skipped",
                        0,
                    )
                ),
                "last_decision": copy.deepcopy(
                    getattr(self, "_agent6_safe_actor_last_decision", None)
                ),
                "candidate_id": int(
                    getattr(self, "_agent6_safe_actor_candidate_id", 0)
                ),
                "candidate_updates": int(
                    getattr(self, "_agent6_safe_actor_candidate_updates", 0)
                ),
                "candidate_inner_blocks": int(
                    getattr(self, "_agent6_safe_actor_candidate_updates", 0)
                    // max(
                        1,
                        getattr(self, "_agent6_safe_actor_block_updates", 1),
                    )
                ),
                "actor_updates_per_probe": int(
                    getattr(self, "_agent6_safe_actor_block_updates", 1)
                    * getattr(self, "_agent6_safe_actor_blocks_per_probe", 1)
                ),
                "waiting_for_probe": self.safe_actor_probe_required(),
                "probe_eligible": bool(
                    getattr(
                        self,
                        "_agent6_safe_actor_evidence_eligible",
                        True,
                    )
                ),
                "evidence_distances_m": list(
                    getattr(
                        self,
                        "_agent6_safe_actor_evidence_distances_m",
                        [],
                    )
                ),
                "probe_phase": str(
                    getattr(self, "_agent6_safe_actor_probe_phase", "idle")
                ),
                "probe_seed_index": int(
                    getattr(self, "_agent6_safe_actor_probe_seed_index", 0)
                ),
                "probe_trial_index": int(
                    getattr(self, "_agent6_safe_actor_probe_trial_index", 0)
                ),
                "probe_order_position": int(
                    getattr(self, "_agent6_safe_actor_probe_order_position", 0)
                ),
                "challenge_seeds": list(
                    getattr(self, "_agent6_safe_actor_probe_seeds", ())
                ),
                "completed_reference_probes": len(
                    getattr(
                        self,
                        "_agent6_safe_actor_reference_distances_m",
                        [],
                    )
                ),
                "completed_candidate_probes": len(
                    getattr(
                        self,
                        "_agent6_safe_actor_candidate_distances_m",
                        [],
                    )
                ),
            }

        def _compare_safe_actor_probes(
            self,
            *,
            seed_count: int,
            trials_per_seed: int,
            start: int = 0,
            stop: int | None = None,
            minimum_improvement_m: float | None = None,
            maximum_regression_m: float | None = None,
        ) -> SafeActorProbeComparison:
            probe_slice = slice(int(start), stop)
            return safe_actor_probe_decision(
                self._agent6_safe_actor_reference_distances_m[probe_slice],
                self._agent6_safe_actor_candidate_distances_m[probe_slice],
                seed_count=seed_count,
                trials_per_seed=trials_per_seed,
                reference_order_positions=(
                    self._agent6_safe_actor_reference_order_positions[
                        probe_slice
                    ]
                ),
                candidate_order_positions=(
                    self._agent6_safe_actor_candidate_order_positions[
                        probe_slice
                    ]
                ),
                minimum_improvement_m=(
                    self._agent6_safe_actor_min_improvement_m
                    if minimum_improvement_m is None
                    else minimum_improvement_m
                ),
                maximum_paired_regression_m=(
                    self._agent6_safe_actor_max_paired_regression_m
                    if maximum_regression_m is None
                    else maximum_regression_m
                ),
            )

        @staticmethod
        def _safe_actor_comparison_metrics(
            comparison: SafeActorProbeComparison,
        ) -> dict[str, Any]:
            return {
                "median_distance_m": comparison.candidate_median_distance_m,
                "minimum_distance_m": (
                    comparison.candidate_minimum_distance_m
                ),
                "reference_median_distance_m": (
                    comparison.reference_median_distance_m
                ),
                "reference_minimum_distance_m": (
                    comparison.reference_minimum_distance_m
                ),
                "aggregate_median_gain_m": comparison.aggregate_median_gain_m,
                "minimum_gain_m": comparison.minimum_gain_m,
                "paired_median_delta_m": comparison.paired_median_delta_m,
                "worst_paired_delta_m": comparison.worst_paired_delta_m,
                "paired_regressions": comparison.paired_regressions,
                "paired_deltas_m": list(comparison.paired_deltas_m),
                "worst_order_position_delta_m": (
                    comparison.worst_order_position_delta_m
                ),
                "order_position_regressions": (
                    comparison.order_position_regressions
                ),
                "order_position_deltas_m": list(
                    comparison.order_position_deltas_m
                ),
                "raw_paired_regressions": comparison.raw_paired_regressions,
                "reference_seed_medians_m": list(
                    comparison.reference_seed_medians_m
                ),
                "candidate_seed_medians_m": list(
                    comparison.candidate_seed_medians_m
                ),
            }

        def _reset_safe_actor_probe_runtime(self) -> None:
            self._agent6_safe_actor_candidate_updates = 0
            self._agent6_safe_actor_waiting_for_probe = False
            self._agent6_safe_actor_probe_phase = "idle"
            self._agent6_safe_actor_probe_seed_index = 0
            self._agent6_safe_actor_probe_trial_index = 0
            self._agent6_safe_actor_probe_order_position = 0
            self._agent6_safe_actor_probe_seeds = ()
            self._agent6_safe_actor_reference_distances_m = []
            self._agent6_safe_actor_candidate_distances_m = []
            self._agent6_safe_actor_reference_order_positions = []
            self._agent6_safe_actor_candidate_order_positions = []
            self._agent6_safe_actor_evidence_candidate_id = 0
            self._agent6_safe_actor_evidence_distances_m = []
            self._agent6_safe_actor_evidence_completed_lap = False
            self._agent6_safe_actor_evidence_eligible = bool(
                self._agent6_safe_actor_min_evidence_episodes == 0
            )
            self._clear_safe_actor_candidate_snapshot()

        def _resolve_safe_actor_comparison(
            self,
            result: dict[str, Any],
            comparison: SafeActorProbeComparison,
            *,
            decision: str,
        ) -> dict[str, Any]:
            probe_episodes_completed = (
                len(self._agent6_safe_actor_reference_distances_m)
                + len(self._agent6_safe_actor_candidate_distances_m)
            )
            probe_episodes_maximum = (
                2
                * self._agent6_safe_actor_probe_repeats
                * self._agent6_safe_actor_trials_per_seed
            )
            probe_episodes_skipped = (
                probe_episodes_maximum - probe_episodes_completed
            )
            if decision == "accepted":
                self._restore_safe_actor_candidate()
                self._agent6_safe_actor_accepted_distance_m = (
                    comparison.candidate_median_distance_m
                )
                self._agent6_safe_actor_accepted_minimum_distance_m = (
                    comparison.candidate_minimum_distance_m
                )
                self._agent6_safe_actor_accepted_blocks += 1
                self._snapshot_safe_actor()
            else:
                self._restore_safe_actor()
                self._agent6_safe_actor_rolled_back_blocks += 1
                if decision == "rolled_back_screening":
                    self._agent6_safe_actor_screened_out_blocks += 1
                elif decision == "rolled_back_early":
                    self._agent6_safe_actor_early_rolled_back_blocks += 1
            self._agent6_safe_actor_probe_episodes_skipped += (
                probe_episodes_skipped
            )

            self._agent6_safe_actor_last_decision = {
                "candidate_id": result["candidate_id"],
                "decision": decision,
                "probe_episodes_completed": probe_episodes_completed,
                "probe_episodes_skipped": probe_episodes_skipped,
            }
            metrics = self._safe_actor_comparison_metrics(comparison)
            self._reset_safe_actor_probe_runtime()
            result.update(metrics)
            result.update(
                {
                    "decision": decision,
                    "probe_episodes_completed": probe_episodes_completed,
                    "probe_episodes_maximum": probe_episodes_maximum,
                    "probe_episodes_skipped": probe_episodes_skipped,
                    "accepted_distance_m": (
                        self._agent6_safe_actor_accepted_distance_m
                    ),
                    "accepted_minimum_distance_m": (
                        self._agent6_safe_actor_accepted_minimum_distance_m
                    ),
                }
            )
            return result

        def record_safe_actor_probe(
            self,
            summary: Mapping[str, Any],
        ) -> dict[str, Any] | None:
            if not self.safe_actor_probe_required():
                return None
            context = self.safe_actor_probe_context()
            if context is None:
                return None
            distance_m = max(0.0, finite_float(summary.get("distance_m")))
            result = {
                "mode": context["mode"],
                "seed": context["seed"],
                "candidate_id": context["candidate_id"],
                "probe_index": context["probe_index"],
                "probe_repeats": self._agent6_safe_actor_probe_repeats,
                "trial_index": context["trial_index"],
                "trials_per_seed": self._agent6_safe_actor_trials_per_seed,
                "probe_stage": context["probe_stage"],
                "order_position": context["order_position"],
                "decision": "pending",
                "probe_episodes_completed": None,
                "probe_episodes_maximum": (
                    2
                    * self._agent6_safe_actor_probe_repeats
                    * self._agent6_safe_actor_trials_per_seed
                ),
                "probe_episodes_skipped": None,
                "median_distance_m": None,
                "minimum_distance_m": None,
                "reference_median_distance_m": None,
                "reference_minimum_distance_m": None,
                "paired_median_delta_m": None,
                "worst_paired_delta_m": None,
                "paired_regressions": None,
                "worst_order_position_delta_m": None,
                "order_position_regressions": None,
                "order_position_deltas_m": None,
                "accepted_distance_m": self._agent6_safe_actor_accepted_distance_m,
                "accepted_minimum_distance_m": (
                    self._agent6_safe_actor_accepted_minimum_distance_m
                ),
                "required_distance_m": self._agent6_safe_actor_min_improvement_m,
                "required_minimum_distance_m": 0.0,
                "maximum_paired_regression_m": (
                    self._agent6_safe_actor_max_paired_regression_m
                ),
            }
            if context["mode"] == "accepted_reference":
                self._agent6_safe_actor_reference_distances_m.append(distance_m)
                self._agent6_safe_actor_reference_order_positions.append(
                    int(context["order_position"])
                )
            else:
                self._agent6_safe_actor_candidate_distances_m.append(distance_m)
                self._agent6_safe_actor_candidate_order_positions.append(
                    int(context["order_position"])
                )
            result["probe_episodes_completed"] = (
                len(self._agent6_safe_actor_reference_distances_m)
                + len(self._agent6_safe_actor_candidate_distances_m)
            )

            order = self._current_safe_actor_probe_order()
            if self._agent6_safe_actor_probe_order_position == 0:
                self._agent6_safe_actor_probe_order_position = 1
                self._activate_safe_actor_probe_mode(order[1])
                result["decision"] = "first_policy_recorded"
                return result

            final_trial = (
                self._agent6_safe_actor_probe_trial_index + 1
                >= self._agent6_safe_actor_trials_per_seed
            )
            final_seed = (
                self._agent6_safe_actor_probe_seed_index + 1
                >= self._agent6_safe_actor_probe_repeats
            )
            completed_trial = self._agent6_safe_actor_probe_trial_index + 1

            if (
                self._agent6_safe_actor_probe_seed_index == 0
                and completed_trial
                == self._agent6_safe_actor_screening_trials_per_seed
            ):
                seed_start = (
                    self._agent6_safe_actor_probe_seed_index
                    * self._agent6_safe_actor_trials_per_seed
                )
                screening = self._compare_safe_actor_probes(
                    seed_count=1,
                    trials_per_seed=(
                        self._agent6_safe_actor_screening_trials_per_seed
                    ),
                    start=seed_start,
                    stop=(
                        seed_start
                        + self._agent6_safe_actor_screening_trials_per_seed
                    ),
                    minimum_improvement_m=0.0,
                    maximum_regression_m=(
                        self._agent6_safe_actor_screening_max_regression_m
                    ),
                )
                screening_limit = (
                    self._agent6_safe_actor_screening_max_regression_m
                )
                screening_failed = (
                    screening.worst_paired_delta_m < -screening_limit
                    or screening.worst_order_position_delta_m
                    < -screening_limit
                )
                if screening_failed:
                    return self._resolve_safe_actor_comparison(
                        result,
                        screening,
                        decision="rolled_back_screening",
                    )
                result.update(self._safe_actor_comparison_metrics(screening))
                result["decision"] = "screening_passed"
                if final_trial and final_seed:
                    comparison = self._compare_safe_actor_probes(
                        seed_count=1,
                        trials_per_seed=self._agent6_safe_actor_trials_per_seed,
                    )
                    return self._resolve_safe_actor_comparison(
                        result,
                        comparison,
                        decision=comparison.decision,
                    )
                if final_trial:
                    self._agent6_safe_actor_probe_seed_index += 1
                    self._agent6_safe_actor_probe_trial_index = 0
                else:
                    self._agent6_safe_actor_probe_trial_index += 1
                self._agent6_safe_actor_probe_order_position = 0
                self._activate_safe_actor_probe_mode(
                    self._current_safe_actor_probe_order()[0]
                )
                return result

            if final_trial and not final_seed:
                completed_seeds = self._agent6_safe_actor_probe_seed_index + 1
                completed_values = (
                    completed_seeds * self._agent6_safe_actor_trials_per_seed
                )
                confirmed = self._compare_safe_actor_probes(
                    seed_count=completed_seeds,
                    trials_per_seed=self._agent6_safe_actor_trials_per_seed,
                    stop=completed_values,
                )
                final_limit = self._agent6_safe_actor_max_paired_regression_m
                irreversibly_failed = (
                    confirmed.worst_paired_delta_m < -final_limit
                    or confirmed.worst_order_position_delta_m < -final_limit
                )
                if irreversibly_failed:
                    return self._resolve_safe_actor_comparison(
                        result,
                        confirmed,
                        decision="rolled_back_early",
                    )
                result.update(self._safe_actor_comparison_metrics(confirmed))
                result["decision"] = "seed_confirmed"
                self._agent6_safe_actor_probe_seed_index += 1
                self._agent6_safe_actor_probe_trial_index = 0
                self._agent6_safe_actor_probe_order_position = 0
                self._activate_safe_actor_probe_mode(
                    self._current_safe_actor_probe_order()[0]
                )
                return result

            if not (final_trial and final_seed):
                self._agent6_safe_actor_probe_trial_index += 1
                self._agent6_safe_actor_probe_order_position = 0
                self._activate_safe_actor_probe_mode(
                    self._current_safe_actor_probe_order()[0]
                )
                result["decision"] = "trial_complete"
                return result

            comparison = self._compare_safe_actor_probes(
                seed_count=self._agent6_safe_actor_probe_repeats,
                trials_per_seed=self._agent6_safe_actor_trials_per_seed,
            )
            return self._resolve_safe_actor_comparison(
                result,
                comparison,
                decision=comparison.decision,
            )

        def finalize_safe_actor_candidate(self) -> dict[str, Any] | None:
            candidate_updates = int(
                getattr(self, "_agent6_safe_actor_candidate_updates", 0)
            )
            if not getattr(self, "_agent6_safe_actor_enabled", False):
                return None
            if candidate_updates <= 0 and not self.safe_actor_probe_required():
                return None
            candidate_id = self._agent6_safe_actor_candidate_id
            self._restore_safe_actor()
            self._reset_safe_actor_probe_runtime()
            self._agent6_safe_actor_rolled_back_blocks += 1
            self._agent6_safe_actor_last_decision = {
                "candidate_id": candidate_id,
                "decision": "rolled_back_incomplete",
            }
            return {
                "candidate_id": candidate_id,
                "decision": "rolled_back_incomplete",
            }

        def _safe_actor_constraint_observations(self, observations: Any) -> Any:
            anchors = self._agent6_safe_actor_anchor_observations
            if anchors is None:
                anchor_count = min(
                    self._agent6_safe_actor_anchor_batch_size,
                    int(observations.shape[0]),
                )
                anchors = observations[:anchor_count].detach().clone()
                self._agent6_safe_actor_anchor_observations = anchors
            return th.cat((observations, anchors), dim=0)

        def _safe_actor_action_delta(self, observations: Any) -> float:
            with th.no_grad():
                candidate_actions = self.actor(observations)
                accepted_actions = self._agent6_safe_actor_reference(observations)
                return float(
                    (candidate_actions - accepted_actions).abs().max().item()
                )

        def _project_safe_actor(self, observations: Any) -> float:
            max_delta = self._agent6_safe_actor_max_action_delta
            candidate_parameters = [
                parameter.detach().clone()
                for parameter in self.actor.parameters()
            ]
            accepted_parameters = list(
                self._agent6_safe_actor_reference.parameters()
            )
            action_delta = self._safe_actor_action_delta(observations)
            if action_delta <= max_delta:
                return action_delta

            lower = 0.0
            upper = 1.0
            for _ in range(8):
                interpolation = (lower + upper) * 0.5
                with th.no_grad():
                    for parameter, accepted, candidate in zip(
                        self.actor.parameters(),
                        accepted_parameters,
                        candidate_parameters,
                    ):
                        parameter.copy_(
                            accepted + interpolation * (candidate - accepted)
                        )
                if self._safe_actor_action_delta(observations) <= max_delta:
                    lower = interpolation
                else:
                    upper = interpolation
            with th.no_grad():
                for parameter, accepted, candidate in zip(
                    self.actor.parameters(),
                    accepted_parameters,
                    candidate_parameters,
                ):
                    parameter.copy_(accepted + lower * (candidate - accepted))
            return self._safe_actor_action_delta(observations)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self._apply_training_budget_progress()
            actor_schedule = getattr(
                self,
                "_agent6_actor_learning_rate_schedule",
                None,
            )
            if actor_schedule is None:
                super().train(gradient_steps, batch_size)
                return

            self.policy.set_training_mode(True)
            budget_timestep = self._training_budget_timestep()
            critic_learning_rate = self.lr_schedule(
                self._current_progress_remaining
            )
            actor_learning_rate = float(actor_schedule(budget_timestep))
            update_learning_rate(self.critic.optimizer, critic_learning_rate)
            update_learning_rate(self.actor.optimizer, actor_learning_rate)
            self.logger.record("train/learning_rate", critic_learning_rate)
            self.logger.record("train/critic_learning_rate", critic_learning_rate)
            self.logger.record("train/actor_learning_rate", actor_learning_rate)

            actor_updates_enabled = (
                budget_timestep
                > int(getattr(self, "_agent6_actor_updates_start_at", 0))
                and actor_learning_rate > 0.0
                and not self.safe_actor_probe_required()
            )
            self.logger.record(
                "train/actor_updates_enabled",
                int(actor_updates_enabled),
            )

            actor_losses: list[float] = []
            actor_gradient_norms: list[float] = []
            actor_action_deltas: list[float] = []
            critic_losses: list[float] = []
            for _ in range(gradient_steps):
                self._n_updates += 1
                replay_data = self.replay_buffer.sample(
                    batch_size,
                    env=self._vec_normalize_env,
                )
                discounts = (
                    replay_data.discounts
                    if replay_data.discounts is not None
                    else self.gamma
                )

                with th.no_grad():
                    noise = replay_data.actions.clone().data.normal_(
                        0,
                        self.target_policy_noise,
                    )
                    noise = noise.clamp(
                        -self.target_noise_clip,
                        self.target_noise_clip,
                    )
                    next_actions = (
                        self.actor_target(replay_data.next_observations) + noise
                    ).clamp(-1, 1)
                    next_q_values = th.cat(
                        self.critic_target(
                            replay_data.next_observations,
                            next_actions,
                        ),
                        dim=1,
                    )
                    next_q_values, _ = th.min(
                        next_q_values,
                        dim=1,
                        keepdim=True,
                    )
                    target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * discounts * next_q_values
                    )

                current_q_values = self.critic(
                    replay_data.observations,
                    replay_data.actions,
                )
                critic_loss = sum(
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                )
                critic_losses.append(float(critic_loss.item()))
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

                if self._n_updates % self.policy_delay != 0:
                    continue

                candidate_block_complete = False
                if actor_updates_enabled and not self.safe_actor_probe_required():
                    actor_loss = -self.critic.q1_forward(
                        replay_data.observations,
                        self.actor(replay_data.observations),
                    ).mean()
                    actor_loss_value = float(actor_loss.item())
                    actor_update_valid = math.isfinite(actor_loss_value)
                    if actor_update_valid:
                        actor_losses.append(actor_loss_value)
                        self.actor.optimizer.zero_grad()
                        actor_loss.backward()
                    if (
                        actor_update_valid
                        and getattr(self, "_agent6_safe_actor_enabled", False)
                    ):
                        gradient_norm = th.nn.utils.clip_grad_norm_(
                            self.actor.parameters(),
                            self._agent6_safe_actor_gradient_norm,
                        )
                        gradient_norm_value = float(gradient_norm.item())
                        actor_update_valid = math.isfinite(gradient_norm_value)
                        if actor_update_valid:
                            actor_gradient_norms.append(gradient_norm_value)
                    if not actor_update_valid:
                        self.actor.optimizer.zero_grad()
                        if getattr(self, "_agent6_safe_actor_enabled", False):
                            self._agent6_safe_actor_invalid_updates += 1
                    else:
                        self.actor.optimizer.step()
                    if (
                        actor_update_valid
                        and getattr(self, "_agent6_safe_actor_enabled", False)
                    ):
                        if self._agent6_safe_actor_candidate_updates == 0:
                            self._agent6_safe_actor_candidate_id += 1
                            self._agent6_safe_actor_evidence_candidate_id = (
                                self._agent6_safe_actor_candidate_id
                            )
                            self._agent6_safe_actor_evidence_distances_m = []
                            self._agent6_safe_actor_evidence_completed_lap = False
                            self._agent6_safe_actor_evidence_eligible = bool(
                                self._agent6_safe_actor_min_evidence_episodes == 0
                            )
                        constraint_observations = (
                            self._safe_actor_constraint_observations(
                                replay_data.observations
                            )
                        )
                        actor_action_deltas.append(
                            self._project_safe_actor(constraint_observations)
                        )
                        self._agent6_safe_actor_candidate_updates += 1
                        updates_per_probe = (
                            self._agent6_safe_actor_block_updates
                            * self._agent6_safe_actor_blocks_per_probe
                        )
                        if (
                            self._agent6_safe_actor_candidate_updates
                            >= updates_per_probe
                        ):
                            self._agent6_safe_actor_waiting_for_probe = True
                            candidate_block_complete = True
                    if actor_update_valid:
                        polyak_update(
                            self.actor.parameters(),
                            self.actor_target.parameters(),
                            self.tau,
                        )
                        polyak_update(
                            self.actor_batch_norm_stats,
                            self.actor_batch_norm_stats_target,
                            1.0,
                        )
                    if candidate_block_complete:
                        self._resolve_policy_equivalent_candidate()

                # The critic target must continue adapting while the actor is frozen.
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )

            self.logger.record(
                "train/n_updates",
                self._n_updates,
                exclude="tensorboard",
            )
            if actor_losses:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
            if actor_gradient_norms:
                self.logger.record(
                    "train/actor_gradient_norm",
                    np.mean(actor_gradient_norms),
                )
            if actor_action_deltas:
                self.logger.record(
                    "train/actor_max_action_delta",
                    max(actor_action_deltas),
                )
            if critic_losses:
                self.logger.record("train/critic_loss", np.mean(critic_losses))
            budget_state = getattr(self, "_agent6_training_budget_state", None)
            if budget_state is not None:
                self.logger.record(
                    "time/training_budget_steps",
                    budget_state.training_steps,
                )
                self.logger.record(
                    "time/probe_overhead_steps",
                    budget_state.probe_steps,
                )
                self.logger.record(
                    "time/probe_available_steps",
                    budget_state.available_probe_steps,
                )
                self.logger.record(
                    "time/probe_fraction",
                    budget_state.as_dict()["probe_fraction"],
                )
                self.logger.record(
                    "time/probe_deferred_requests",
                    budget_state.deferred_probe_requests,
                )
            if getattr(self, "_agent6_safe_actor_enabled", False):
                status = self.safe_actor_status()
                self.logger.record(
                    "train/safe_actor_candidate_updates",
                    status["candidate_updates"],
                )
                self.logger.record(
                    "train/safe_actor_accepted_blocks",
                    status["accepted_blocks"],
                )
                self.logger.record(
                    "train/safe_actor_rolled_back_blocks",
                    status["rolled_back_blocks"],
                )
                self.logger.record(
                    "train/safe_actor_screened_out_blocks",
                    status["screened_out_blocks"],
                )
                self.logger.record(
                    "train/safe_actor_early_rolled_back_blocks",
                    status["early_rolled_back_blocks"],
                )
                self.logger.record(
                    "train/safe_actor_policy_equivalent_blocks",
                    status["policy_equivalent_blocks"],
                )
                self.logger.record(
                    "train/safe_actor_no_evidence_blocks",
                    status["no_evidence_blocks"],
                )
                self.logger.record(
                    "train/safe_actor_invalid_updates",
                    status["invalid_actor_updates"],
                )
                self.logger.record(
                    "train/safe_actor_probe_eligible",
                    int(status["probe_eligible"]),
                )
                self.logger.record(
                    "train/safe_actor_probe_episodes_skipped",
                    status["probe_episodes_skipped"],
                )

    return PhasedContinuationTD3


def create_td3_model(
    TD3: Any,
    *,
    args: argparse.Namespace,
    env: Any,
    tensorboard_log: str | None,
    action_noise: Any,
    learning_rate_schedule: Callable[[float], float],
) -> Any:
    model_kwargs = {
        "env": env,
        "verbose": int(args.verbose_training),
        "tensorboard_log": tensorboard_log,
        "buffer_size": args.buffer_size,
        "learning_starts": policy_action_start_timestep(args),
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "learning_rate": learning_rate_schedule,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "policy_delay": args.policy_delay,
        "target_policy_noise": args.target_policy_noise,
        "target_noise_clip": args.target_noise_clip,
        "action_noise": action_noise,
        "seed": args.seed,
        "device": args.device,
    }
    if args.continuation:
        if (
            continuation_source_uses_legacy_action_contract(args)
            and args.allow_legacy_action_migration
        ):
            legacy_model = TD3.load(str(args.resume_from), device=args.device)
            model = TD3(
                "MlpPolicy",
                policy_kwargs={"net_arch": args.net_arch},
                **model_kwargs,
            )
            migration = migrate_legacy_actor_to_signed_longitudinal(
                legacy_model,
                model,
            )
            migration["source_policy_path"] = str(args.resume_from)
            migration["source_model_num_timesteps"] = int(
                getattr(legacy_model, "num_timesteps", 0)
            )
            model.num_timesteps = int(getattr(legacy_model, "num_timesteps", 0))
            setattr(model, "_agent6_contract_migration", migration)
            return model
        model = TD3.load(str(args.resume_from), **model_kwargs)
        normalise_continuation_spaces(model, env)
        return model
    return TD3(
        "MlpPolicy",
        policy_kwargs={"net_arch": args.net_arch},
        **model_kwargs,
    )


def migrate_legacy_actor_to_signed_longitudinal(
    legacy_model: Any,
    model: Any,
) -> dict[str, Any]:
    """Transfer a v1 3-head actor into the v2 signed-longitudinal actor."""
    legacy_actor = getattr(legacy_model, "actor", None)
    actor = getattr(model, "actor", None)
    actor_target = getattr(model, "actor_target", None)
    if legacy_actor is None or actor is None:
        raise RuntimeError("TD3 actor migration requires source and target actors")

    legacy_state = legacy_actor.state_dict()
    target_state = actor.state_dict()
    copied_tensors = 0
    adapted_tensors = 0
    skipped_tensors: list[str] = []
    for name, target_value in list(target_state.items()):
        source_value = legacy_state.get(name)
        if source_value is None:
            skipped_tensors.append(name)
            continue
        if tuple(source_value.shape) == tuple(target_value.shape):
            target_state[name] = source_value.detach().clone()
            copied_tensors += 1
            continue
        source_shape = tuple(source_value.shape)
        target_shape = tuple(target_value.shape)
        if (
            len(source_shape) == len(target_shape)
            and len(source_shape) >= 1
            and source_shape[0] >= 3
            and target_shape[0] == AGENT6_ACTION_SIZE
            and source_shape[1:] == target_shape[1:]
        ):
            migrated_value = target_value.detach().clone()
            migrated_value[0].copy_(source_value[0])
            migrated_value[1].copy_((source_value[1] - source_value[2]) * 0.5)
            target_state[name] = migrated_value
            adapted_tensors += 1
            continue
        skipped_tensors.append(name)

    if adapted_tensors < 1:
        raise RuntimeError(
            "legacy TD3 actor migration found no 3-head output tensors to adapt"
        )
    actor.load_state_dict(target_state)
    if actor_target is not None:
        actor_target.load_state_dict(target_state)
    normalise_continuation_spaces(model, model.env)
    return {
        "required": True,
        "source_action_contract": "agent6_direct_steer_throttle_brake_v1",
        "target_action_contract": AGENT6_ACTION_VERSION,
        "method": "copy_shared_actor_layers_and_map_accel_minus_brake_to_signed_longitudinal",
        "copied_actor_tensors": copied_tensors,
        "adapted_actor_output_tensors": adapted_tensors,
        "skipped_actor_tensors": skipped_tensors,
        "critic_replay_source": "fresh_under_v2_contract",
        "learning_source": "reward_only",
    }


def normalise_continuation_spaces(model: Any, env: Any) -> None:
    """Remove dynamically defined warm-up spaces from replay serialization."""
    action_space = env.action_space
    observation_space = env.observation_space
    model.action_space = action_space
    model.observation_space = observation_space
    if getattr(model, "policy", None) is not None:
        model.policy.action_space = action_space
        model.policy.observation_space = observation_space
    replay_buffer = getattr(model, "replay_buffer", None)
    if replay_buffer is not None:
        replay_buffer.action_space = action_space
        replay_buffer.observation_space = observation_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Agent 6 from scratch or continue a verified reward-only TD3 "
            "checkpoint."
        ),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help=(
            "Non-probe environment interactions requested for training. "
            "Evaluation overhead is counted separately."
        ),
    )
    parser.add_argument(
        "--max-probe-overhead-steps",
        type=int,
        default=None,
        help=(
            "Absolute ceiling for probe interactions excluded from the "
            "training-step target. The ratio ceiling can make this lower."
        ),
    )
    parser.add_argument(
        "--max-probe-fraction",
        type=float,
        default=DEFAULT_MAX_PROBE_FRACTION,
        help=(
            "Maximum fraction of total interactions spent probing. Values above "
            "0.20 are rejected so at least 80%% remains training data."
        ),
    )
    parser.add_argument("--track", default=DEFAULT_TRACK_NAME)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Verified Agent 6 evaluation checkpoint to continue training.",
    )
    parser.add_argument(
        "--allow-non-champion-source",
        action="store_true",
        help=(
            "Permit continuation from a verified policy other than the "
            "canonical best-evaluation champion."
        ),
    )
    parser.add_argument(
        "--allow-legacy-action-migration",
        action="store_true",
        help=(
            "Experimentally migrate a verified v1 three-head actor into the v2 "
            "signed-longitudinal contract. By default, v1 is preserved exactly."
        ),
    )
    parser.add_argument(
        "--continuation-compatibility-probe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run one deterministic source-policy rollout before collecting replay. "
            "Enabled by default for continuation."
        ),
    )
    parser.add_argument(
        "--continuation-compatibility-min-fraction",
        type=float,
        default=0.90,
        help=(
            "Minimum fraction of the source checkpoint's verified median distance "
            "required by the pre-training compatibility rollout."
        ),
    )
    parser.add_argument(
        "--continuation-compatibility-probe-only",
        action="store_true",
        help=(
            "Run the pre-training compatibility rollout and exit without "
            "collecting replay, updating weights, or saving a model."
        ),
    )
    parser.add_argument(
        "--start-stage",
        choices=CURRICULUM_STAGE_IDS,
        default=None,
        help=(
            "Initial curriculum stage. Continuation defaults to sector_progression; "
            "fresh training defaults to launch."
        ),
    )
    parser.add_argument(
        "--replay-refill-steps",
        type=int,
        default=None,
        help=(
            "Policy-controlled steps collected into a fresh replay buffer with "
            "gradient updates frozen."
        ),
    )
    parser.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=None,
        help=(
            "Continuation steps with critic updates enabled while the actor "
            "and actor target remain frozen."
        ),
    )
    parser.add_argument(
        "--actor-unfreeze-steps",
        type=int,
        default=None,
        help=(
            "Continuation steps used to ramp the actor learning rate from zero."
        ),
    )
    parser.add_argument(
        "--safe-actor-improvement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Gate continuation actor updates through bounded update blocks and "
            "matched accepted-versus-candidate robustness probes."
        ),
    )
    parser.add_argument(
        "--safe-actor-max-action-delta",
        type=float,
        default=DEFAULT_SAFE_ACTOR_MAX_ACTION_DELTA,
        help="Maximum normalized action deviation from the accepted actor.",
    )
    parser.add_argument(
        "--safe-actor-gradient-norm",
        type=float,
        default=DEFAULT_SAFE_ACTOR_GRADIENT_NORM,
        help="Maximum actor gradient norm before each optimizer step.",
    )
    parser.add_argument(
        "--safe-actor-block-updates",
        type=int,
        default=DEFAULT_SAFE_ACTOR_BLOCK_UPDATES,
        help="Actor optimizer updates in each bounded trust-region block.",
    )
    parser.add_argument(
        "--safe-actor-blocks-per-probe",
        type=int,
        default=DEFAULT_SAFE_ACTOR_BLOCKS_PER_PROBE,
        help=(
            "Bounded actor-update blocks accumulated before an expensive "
            "on-track robustness comparison."
        ),
    )
    parser.add_argument(
        "--safe-actor-probe-repeats",
        type=int,
        default=DEFAULT_SAFE_ACTOR_PROBE_REPEATS,
        help="Fresh challenge seeds used to compare each actor candidate.",
    )
    parser.add_argument(
        "--safe-actor-trials-per-seed",
        type=int,
        default=DEFAULT_SAFE_ACTOR_TRIALS_PER_SEED,
        help=(
            "Balanced trials per policy and challenge seed. The internal "
            "protocol defaults to one run in each order position."
        ),
    )
    parser.add_argument(
        "--safe-actor-screening-trials-per-seed",
        type=int,
        default=DEFAULT_SAFE_ACTOR_SCREENING_TRIALS_PER_SEED,
        help=(
            "Balanced trials per policy used for the short collapse screen on "
            "the first challenge seed."
        ),
    )
    parser.add_argument(
        "--safe-actor-probe-base-seed",
        type=int,
        default=DEFAULT_SAFE_ACTOR_PROBE_BASE_SEED,
        help="Base seed used to rotate unique training challenge suites.",
    )
    parser.add_argument(
        "--safe-actor-probe-noise-std",
        type=float,
        default=DEFAULT_SAFE_ACTOR_PROBE_NOISE_STD,
        help="Correlated steering disturbance standard deviation for safe probes.",
    )
    parser.add_argument(
        "--safe-actor-min-improvement-m",
        type=float,
        default=DEFAULT_SAFE_ACTOR_MIN_IMPROVEMENT_M,
        help="Required median-distance gain before a candidate is accepted.",
    )
    parser.add_argument(
        "--safe-actor-max-paired-regression-m",
        type=float,
        default=DEFAULT_SAFE_ACTOR_MAX_PAIRED_REGRESSION_M,
        help="Largest permitted candidate loss on any matched challenge seed.",
    )
    parser.add_argument(
        "--safe-actor-screening-max-regression-m",
        type=float,
        default=DEFAULT_SAFE_ACTOR_SCREENING_MAX_REGRESSION_M,
        help=(
            "Relaxed seed or order-position loss that rejects a candidate "
            "after the two-trial screen."
        ),
    )
    parser.add_argument(
        "--safe-actor-screening-max-episode-steps",
        type=int,
        default=DEFAULT_SAFE_ACTOR_SCREENING_MAX_EPISODE_STEPS,
        help="Maximum TORCS steps for each first-stage screening rollout.",
    )
    parser.add_argument(
        "--safe-actor-anchor-batch-size",
        type=int,
        default=DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE,
        help="Replay observations retained for action-space trust-region checks.",
    )
    parser.add_argument(
        "--probe-evidence-window",
        type=int,
        default=DEFAULT_PROBE_EVIDENCE_WINDOW,
        help="Recent candidate-controlled episodes used before admitting probes.",
    )
    parser.add_argument(
        "--probe-min-evidence-episodes",
        type=int,
        default=DEFAULT_PROBE_MIN_EVIDENCE_EPISODES,
        help="Minimum recent training episodes required for a progress signal.",
    )
    parser.add_argument(
        "--probe-evidence-max-regression-m",
        type=float,
        default=DEFAULT_PROBE_EVIDENCE_MAX_REGRESSION_M,
        help="Largest median training regression allowed by the evidence gate.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--best-distance-model-path",
        type=Path,
        default=DEFAULT_BEST_DISTANCE_MODEL_PATH,
    )
    parser.add_argument(
        "--best-reward-model-path",
        type=Path,
        default=DEFAULT_BEST_REWARD_MODEL_PATH,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--tensorboard-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR)
    parser.add_argument("--replay-buffer-path", type=Path, default=DEFAULT_REPLAY_BUFFER_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--manual-start", action="store_true")
    parser.add_argument("--relaunch-frequency", type=int, default=0)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--show-torcs-reset-log", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run the SB3 environment checker before training. This starts TORCS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--learning-rate-final",
        type=float,
        default=None,
        help=(
            "Final TD3 optimizer learning rate. It is reached linearly at the "
            "end of the requested training run."
        ),
    )
    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=None,
        help="Peak actor learning rate after critic-only warm-up.",
    )
    parser.add_argument(
        "--actor-learning-rate-final",
        type=float,
        default=None,
        help="Final actor learning rate at the end of the training run.",
    )
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-policy-noise", type=float, default=0.20)
    parser.add_argument("--target-noise-clip", type=float, default=0.50)
    parser.add_argument("--action-noise-sigma", type=float, default=None)
    parser.add_argument("--action-noise-final-sigma", type=float, default=None)
    parser.add_argument("--action-noise-decay-steps", type=int, default=200_000)
    parser.add_argument("--policy-probe-interval", type=int, default=10)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[400, 300])
    parser.add_argument(
        "--warmup-action-mode",
        choices=("launch-biased", "uniform"),
        default="launch-biased",
    )
    parser.add_argument("--warmup-steer-std", type=float, default=0.18)
    parser.add_argument("--warmup-accel-min", type=float, default=0.25)
    parser.add_argument("--warmup-accel-max", type=float, default=1.00)
    parser.add_argument("--warmup-brake-probability", type=float, default=0.12)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-checkpoint-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-best-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save an aligned replay buffer beside each exact best-distance and "
            "best-reward model. Enabled by default for continuation training."
        ),
    )
    args = parser.parse_args()
    args.continuation = args.resume_from is not None
    if args.continuation_compatibility_probe_only:
        args.continuation_compatibility_probe = True
    if args.continuation_compatibility_probe is None:
        args.continuation_compatibility_probe = args.continuation
    if args.start_stage is None:
        args.start_stage = "sector_progression" if args.continuation else "launch"
    if args.replay_refill_steps is None:
        args.replay_refill_steps = (
            DEFAULT_CONTINUATION_REPLAY_REFILL_STEPS if args.continuation else 0
        )
    if args.critic_warmup_steps is None:
        args.critic_warmup_steps = (
            DEFAULT_CONTINUATION_CRITIC_WARMUP_STEPS if args.continuation else 0
        )
    if args.actor_unfreeze_steps is None:
        args.actor_unfreeze_steps = (
            DEFAULT_CONTINUATION_ACTOR_UNFREEZE_STEPS if args.continuation else 0
        )
    if args.buffer_size is None:
        args.buffer_size = (
            DEFAULT_CONTINUATION_BUFFER_SIZE if args.continuation else 1_000_000
        )
    if args.learning_rate is None:
        args.learning_rate = (
            DEFAULT_CONTINUATION_LEARNING_RATE
            if args.continuation
            else DEFAULT_LEARNING_RATE
        )
    if args.learning_rate_final is None:
        args.learning_rate_final = (
            DEFAULT_CONTINUATION_FINAL_LEARNING_RATE
            if args.continuation
            else DEFAULT_FINAL_LEARNING_RATE
        )
    if args.actor_learning_rate is None:
        args.actor_learning_rate = (
            DEFAULT_CONTINUATION_ACTOR_LEARNING_RATE
            if args.continuation
            else args.learning_rate
        )
    if args.actor_learning_rate_final is None:
        args.actor_learning_rate_final = (
            DEFAULT_CONTINUATION_FINAL_ACTOR_LEARNING_RATE
            if args.continuation
            else args.learning_rate_final
        )
    if args.action_noise_sigma is None:
        args.action_noise_sigma = (
            DEFAULT_CONTINUATION_ACTION_NOISE_SIGMA
            if args.continuation
            else 0.12
        )
    if args.action_noise_final_sigma is None:
        args.action_noise_final_sigma = (
            DEFAULT_CONTINUATION_FINAL_ACTION_NOISE_SIGMA
            if args.continuation
            else 0.04
        )
    if args.save_best_replay_buffer is None:
        args.save_best_replay_buffer = args.continuation
    if args.safe_actor_improvement is None:
        args.safe_actor_improvement = args.continuation
    if (
        not math.isfinite(args.max_probe_fraction)
        or not 0.0 < args.max_probe_fraction <= DEFAULT_MAX_PROBE_FRACTION
    ):
        parser.error(
            "--max-probe-fraction must be in (0, "
            f"{DEFAULT_MAX_PROBE_FRACTION:.2f}]"
        )
    if args.max_probe_overhead_steps is None:
        args.max_probe_overhead_steps = max(
            1,
            int(
                math.floor(
                    max(1, args.total_timesteps)
                    * args.max_probe_fraction
                    / (1.0 - args.max_probe_fraction)
                )
            ),
        )

    if not args.continuation and args.start_stage != "launch":
        parser.error("--start-stage requires --resume-from unless it is launch")
    if args.continuation_compatibility_probe and not args.continuation:
        parser.error("--continuation-compatibility-probe requires --resume-from")
    if args.continuation_compatibility_probe_only and not args.continuation:
        parser.error(
            "--continuation-compatibility-probe-only requires --resume-from"
        )
    if args.allow_legacy_action_migration and not args.continuation:
        parser.error("--allow-legacy-action-migration requires --resume-from")
    if args.continuation:
        source_errors = continuation_source_errors(
            args.resume_from,
            track=args.track,
            start_stage=args.start_stage,
        )
        if source_errors:
            parser.error(source_errors[0])
        if (
            args.allow_legacy_action_migration
            and not continuation_source_uses_legacy_action_contract(args)
        ):
            parser.error(
                "--allow-legacy-action-migration requires a verified legacy v1 source"
            )
        source_path = args.resume_from.resolve()
        if (
            not args.allow_non_champion_source
            and source_path != DEFAULT_BEST_EVALUATION_MODEL_PATH.resolve()
        ):
            parser.error(
                "continuation must restart from "
                f"{DEFAULT_BEST_EVALUATION_MODEL_PATH}; use "
                "--allow-non-champion-source only for deliberate experiments"
            )
        output_paths = (
            args.model_path.resolve(),
            args.best_distance_model_path.resolve(),
            args.best_reward_model_path.resolve(),
        )
        if source_path in output_paths:
            parser.error("continuation outputs must not overwrite --resume-from")
    if args.total_timesteps < 1:
        parser.error("--total-timesteps must be at least 1")
    if (
        not math.isfinite(args.continuation_compatibility_min_fraction)
        or not 0.0 < args.continuation_compatibility_min_fraction <= 1.0
    ):
        parser.error(
            "--continuation-compatibility-min-fraction must be in (0, 1]"
        )
    if args.max_probe_overhead_steps < 1:
        parser.error("--max-probe-overhead-steps must be at least 1")
    if args.replay_refill_steps < 0:
        parser.error("--replay-refill-steps cannot be negative")
    if args.critic_warmup_steps < 0:
        parser.error("--critic-warmup-steps cannot be negative")
    if args.actor_unfreeze_steps < 0:
        parser.error("--actor-unfreeze-steps cannot be negative")
    if args.safe_actor_improvement and not args.continuation:
        parser.error("--safe-actor-improvement requires --resume-from")
    if (
        not math.isfinite(args.safe_actor_max_action_delta)
        or args.safe_actor_max_action_delta <= 0.0
    ):
        parser.error(
            "--safe-actor-max-action-delta must be finite and greater than zero"
        )
    if (
        not math.isfinite(args.safe_actor_gradient_norm)
        or args.safe_actor_gradient_norm <= 0.0
    ):
        parser.error(
            "--safe-actor-gradient-norm must be finite and greater than zero"
        )
    if args.safe_actor_block_updates < 1:
        parser.error("--safe-actor-block-updates must be at least 1")
    if args.safe_actor_blocks_per_probe < 1:
        parser.error("--safe-actor-blocks-per-probe must be at least 1")
    if args.safe_actor_probe_repeats < 3:
        parser.error("--safe-actor-probe-repeats must be at least 3")
    if args.safe_actor_probe_repeats % 2 == 0:
        parser.error("--safe-actor-probe-repeats must be odd")
    if (
        args.safe_actor_trials_per_seed < 2
        or args.safe_actor_trials_per_seed % 2 != 0
    ):
        parser.error(
            "--safe-actor-trials-per-seed must be even and at least 2"
        )
    if (
        args.safe_actor_screening_trials_per_seed < 2
        or args.safe_actor_screening_trials_per_seed % 2 != 0
        or args.safe_actor_screening_trials_per_seed
        > args.safe_actor_trials_per_seed
    ):
        parser.error(
            "--safe-actor-screening-trials-per-seed must be even and between "
            "2 and --safe-actor-trials-per-seed"
        )
    if not 0 <= args.safe_actor_probe_base_seed <= MAX_REPRODUCIBLE_SEED:
        parser.error(
            "--safe-actor-probe-base-seed must be between 0 and "
            f"{MAX_REPRODUCIBLE_SEED}"
        )
    if (
        not math.isfinite(args.safe_actor_probe_noise_std)
        or args.safe_actor_probe_noise_std < 0.0
    ):
        parser.error("--safe-actor-probe-noise-std must be finite and non-negative")
    if (
        not math.isfinite(args.safe_actor_min_improvement_m)
        or args.safe_actor_min_improvement_m < 0.0
    ):
        parser.error(
            "--safe-actor-min-improvement-m must be finite and non-negative"
        )
    if (
        not math.isfinite(args.safe_actor_max_paired_regression_m)
        or args.safe_actor_max_paired_regression_m < 0.0
    ):
        parser.error(
            "--safe-actor-max-paired-regression-m must be finite and non-negative"
        )
    if (
        not math.isfinite(args.safe_actor_screening_max_regression_m)
        or args.safe_actor_screening_max_regression_m
        < args.safe_actor_max_paired_regression_m
    ):
        parser.error(
            "--safe-actor-screening-max-regression-m must be finite and no "
            "smaller than --safe-actor-max-paired-regression-m"
        )
    if args.safe_actor_screening_max_episode_steps < 1:
        parser.error(
            "--safe-actor-screening-max-episode-steps must be at least 1"
        )
    if args.safe_actor_anchor_batch_size < 1:
        parser.error("--safe-actor-anchor-batch-size must be at least 1")
    if args.probe_evidence_window < 1:
        parser.error("--probe-evidence-window must be at least 1")
    if not 1 <= args.probe_min_evidence_episodes <= args.probe_evidence_window:
        parser.error(
            "--probe-min-evidence-episodes must be between 1 and "
            "--probe-evidence-window"
        )
    if (
        not math.isfinite(args.probe_evidence_max_regression_m)
        or args.probe_evidence_max_regression_m < 0.0
    ):
        parser.error(
            "--probe-evidence-max-regression-m must be finite and non-negative"
        )
    if args.continuation and actor_update_start_timestep(args) >= args.total_timesteps:
        parser.error(
            "replay refill plus critic warm-up must be less than --total-timesteps"
        )
    if (
        args.continuation
        and actor_full_unfreeze_timestep(args) >= args.total_timesteps
    ):
        parser.error(
            "replay refill, critic warm-up, and actor unfreeze must leave "
            "at least one fully unfrozen training step"
        )
    if args.buffer_size < max(1, args.batch_size):
        parser.error("--buffer-size must be at least --batch-size")
    if args.checkpoint_freq < 1:
        parser.error("--checkpoint-freq must be at least 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be greater than zero")
    if not 0.0 < args.learning_rate_final <= args.learning_rate:
        parser.error(
            "--learning-rate-final must be greater than zero and no greater "
            "than --learning-rate"
        )
    if args.actor_learning_rate <= 0.0:
        parser.error("--actor-learning-rate must be greater than zero")
    if not 0.0 < args.actor_learning_rate_final <= args.actor_learning_rate:
        parser.error(
            "--actor-learning-rate-final must be greater than zero and no "
            "greater than --actor-learning-rate"
        )
    if args.action_noise_sigma < 0.0:
        parser.error("--action-noise-sigma cannot be negative")
    if not 0.0 <= args.action_noise_final_sigma <= args.action_noise_sigma:
        parser.error(
            "--action-noise-final-sigma must be between zero and "
            "--action-noise-sigma"
        )
    if args.action_noise_decay_steps < 1:
        parser.error("--action-noise-decay-steps must be at least 1")
    if args.policy_probe_interval < 1:
        parser.error("--policy-probe-interval must be at least 1")
    if args.train_freq != 1:
        parser.error("--train-freq must be 1 when deterministic probes are enabled")
    return args


def main() -> None:
    args = parse_args()
    sys.argv = [sys.argv[0]]
    random.seed(args.seed)
    np.random.seed(args.seed)

    (
        gym,
        spaces,
        TD3,
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        Monitor,
        check_env,
        NormalActionNoise,
    ) = import_training_dependencies()
    TrainingTD3 = make_phased_continuation_td3_class(TD3)

    run_dir = args.run_dir if args.run_dir is not None else make_default_run_dir()
    args.run_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_distance_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_reward_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    args.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    training_budget_state = ProbeBudgetController(
        target_training_steps=args.total_timesteps,
        maximum_probe_steps=args.max_probe_overhead_steps,
        maximum_probe_fraction=args.max_probe_fraction,
    )

    env_class = make_training_env_class(gym, spaces)
    raw_env = env_class(
        track_name=args.track,
        manual_start=args.manual_start,
        relaunch_frequency=args.relaunch_frequency,
        reset_retries=args.reset_retries,
        quiet_reset_log=not args.show_torcs_reset_log,
        run_dir=run_dir,
        warmup_action_mode=args.warmup_action_mode,
        warmup_steer_std=args.warmup_steer_std,
        warmup_accel_min=args.warmup_accel_min,
        warmup_accel_max=args.warmup_accel_max,
        warmup_brake_probability=args.warmup_brake_probability,
        learning_starts=policy_action_start_timestep(args),
        initial_stage_id=args.start_stage,
        pre_update_training_phase=(
            "replay_refill" if args.continuation else "warmup"
        ),
        use_custom_warmup_action_space=not args.continuation,
        use_legacy_action_contract=training_uses_legacy_action_contract(args),
        training_budget_state=training_budget_state,
        safe_actor_screening_max_episode_steps=(
            args.safe_actor_screening_max_episode_steps
        ),
    )
    raw_env.action_space.seed(args.seed)
    raw_env.observation_space.seed(args.seed)
    if args.check_env:
        try:
            check_env(raw_env, warn=True, skip_render_check=True)
        except TypeError:
            check_env(raw_env, warn=True)
    env = Monitor(
        raw_env,
        filename=str(run_dir / "monitor.csv"),
        info_keywords=(
            "policy_controlled",
            "deterministic_probe",
            "safe_actor_candidate_probe",
            "safe_actor_probe_mode",
            "safe_actor_probe_seed",
            "curriculum_eligible",
            "training_phase",
            "action_noise_sigma",
            "stage_id",
            "distance_m",
            "furthest_distance_m",
            "lap_completion_fraction",
            "termination_reason",
        ),
    )

    metadata = make_training_metadata(args)
    write_json(run_dir / "training_metadata.json", metadata)

    tensorboard_log = resolve_tensorboard_log_dir(
        args.tensorboard_dir,
        disable_tensorboard=args.no_tensorboard,
    )
    action_size = training_action_size(args)

    def make_action_noise(sigma: float):
        return NormalActionNoise(
            mean=np.zeros(action_size, dtype=np.float32),
            sigma=np.ones(action_size, dtype=np.float32)
            * max(0.0, float(sigma)),
        )

    def make_safe_probe_noise(seed: int, steering_noise_std: float):
        return SeededSteeringActionNoise(
            seed=seed,
            steering_noise_std=steering_noise_std,
            action_size=action_size,
        )

    action_noise = make_action_noise(args.action_noise_sigma)
    learning_rate_schedule = linear_learning_rate_schedule(
        args.learning_rate,
        args.learning_rate_final,
        learning_starts=gradient_update_start_timestep(args),
        total_timesteps=args.total_timesteps,
    )
    model = create_td3_model(
        TrainingTD3,
        args=args,
        env=env,
        tensorboard_log=tensorboard_log,
        action_noise=action_noise,
        learning_rate_schedule=learning_rate_schedule,
    )
    model.configure_training_budget(training_budget_state)
    if args.continuation:
        actor_learning_rate_schedule = gradual_actor_learning_rate_schedule(
            args.actor_learning_rate,
            args.actor_learning_rate_final,
            actor_update_start=actor_update_start_timestep(args),
            unfreeze_steps=args.actor_unfreeze_steps,
            total_timesteps=args.total_timesteps,
        )
        model.configure_continuation_updates(
            actor_learning_rate_schedule=actor_learning_rate_schedule,
            actor_updates_start_at=actor_update_start_timestep(args),
            safe_actor_improvement=args.safe_actor_improvement,
            safe_actor_max_action_delta=args.safe_actor_max_action_delta,
            safe_actor_gradient_norm=args.safe_actor_gradient_norm,
            safe_actor_block_updates=args.safe_actor_block_updates,
            safe_actor_blocks_per_probe=args.safe_actor_blocks_per_probe,
            safe_actor_probe_repeats=args.safe_actor_probe_repeats,
            safe_actor_trials_per_seed=args.safe_actor_trials_per_seed,
            safe_actor_screening_trials_per_seed=(
                args.safe_actor_screening_trials_per_seed
            ),
            safe_actor_min_improvement_m=args.safe_actor_min_improvement_m,
            safe_actor_max_paired_regression_m=(
                args.safe_actor_max_paired_regression_m
            ),
            safe_actor_screening_max_regression_m=(
                args.safe_actor_screening_max_regression_m
            ),
            safe_actor_anchor_batch_size=args.safe_actor_anchor_batch_size,
            safe_actor_probe_base_seed=args.safe_actor_probe_base_seed,
            safe_actor_evidence_window=args.probe_evidence_window,
            safe_actor_min_evidence_episodes=(
                args.probe_min_evidence_episodes
            ),
            safe_actor_evidence_max_regression_m=(
                args.probe_evidence_max_regression_m
            ),
            source_verified_distance_m=finite_float(
                metadata["continuation"].get(
                    "source_verified_evaluation_progress_m"
                )
            ),
            source_verified_minimum_distance_m=finite_float(
                metadata["continuation"].get(
                    "source_verified_evaluation_minimum_progress_m"
                )
            ),
        )
        metadata["continuation"]["source_model_num_timesteps"] = int(
            getattr(model, "num_timesteps", 0)
        )
        contract_migration = getattr(model, "_agent6_contract_migration", None)
        if isinstance(contract_migration, Mapping):
            metadata["continuation"][
                "source_policy_contract_migration"
            ] = dict(contract_migration)
        if args.continuation_compatibility_probe:
            minimum_distance_m = continuation_compatibility_minimum_distance_m(
                args
            )
            print(
                "Running pre-training continuation compatibility probe; "
                f"minimum distance={minimum_distance_m:.1f}m."
            )
            try:
                compatibility_result = run_continuation_compatibility_probe(
                    model,
                    raw_env,
                    minimum_distance_m=minimum_distance_m,
                    training_budget_state=training_budget_state,
                    seed=args.seed,
                )
            except Exception:
                env.close()
                raise
            metadata["continuation"]["compatibility_probe"][
                "result"
            ] = compatibility_result
            write_json(
                run_dir / "continuation_compatibility_probe.json",
                compatibility_result,
            )
            print(
                "Continuation compatibility probe passed: "
                f"{compatibility_result['distance_m']:.1f}m."
            )
        write_json(run_dir / "training_metadata.json", metadata)
        if args.continuation_compatibility_probe_only:
            print(
                "Compatibility probe-only run complete; no replay was collected, "
                "no weights were updated, and no model was saved."
            )
            env.close()
            return
        print(
            "Continuing verified Agent 6 policy from "
            f"{args.resume_from} at stage={args.start_stage}; "
            f"action contract={metadata['action_version']}; "
            f"replay refill={args.replay_refill_steps:,}, "
            f"critic warm-up={args.critic_warmup_steps:,}, "
            f"actor unfreeze={args.actor_unfreeze_steps:,} steps; "
            f"safe actor blocks={'on' if args.safe_actor_improvement else 'off'}; "
            f"actor updates per robustness probe="
            f"{args.safe_actor_block_updates * args.safe_actor_blocks_per_probe:,}; "
            f"robust probes={args.safe_actor_probe_repeats} seeds x "
            f"{args.safe_actor_trials_per_seed} trials x 2 policies "
            "per candidate, with a "
            f"{args.safe_actor_screening_trials_per_seed}-trial staged screen; "
            f"training budget={args.total_timesteps:,} non-probe steps, "
            "effective probe ceiling="
            f"{training_budget_state.effective_maximum_probe_steps:,} steps "
            f"({args.max_probe_fraction:.0%} of all interactions); "
            "probe exhaustion defers comparisons while critic learning continues."
        )

    progress_state = TrainingProgressState()
    ExplorationScheduleCallback = make_exploration_schedule_callback_class(
        BaseCallback
    )
    EpisodeSummaryCallback = make_episode_summary_callback_class(BaseCallback)
    BestModelCallback = make_best_model_callback_class(BaseCallback)
    callbacks = [
        ExplorationScheduleCallback(
            raw_env,
            make_action_noise,
            learning_starts=gradient_update_start_timestep(args),
            initial_sigma=args.action_noise_sigma,
            final_sigma=args.action_noise_final_sigma,
            decay_steps=args.action_noise_decay_steps,
            probe_interval=args.policy_probe_interval,
            training_gradient_steps=args.gradient_steps,
            pre_update_training_phase=(
                "replay_refill" if args.continuation else "warmup"
            ),
            actor_update_start=actor_update_start_timestep(args),
            actor_full_unfreeze=actor_full_unfreeze_timestep(args),
            safe_probe_noise_factory=make_safe_probe_noise,
            safe_probe_noise_std=args.safe_actor_probe_noise_std,
            training_budget_state=training_budget_state,
            probe_reference_distance_m=finite_float(
                metadata["continuation"].get(
                    "source_verified_evaluation_progress_m"
                )
            ),
            probe_evidence_window=args.probe_evidence_window,
            probe_min_evidence_episodes=args.probe_min_evidence_episodes,
            probe_evidence_max_regression_m=(
                args.probe_evidence_max_regression_m
            ),
        ),
        make_checkpoint_callback(
            CheckpointCallback,
            args,
            BaseCallback=BaseCallback,
            training_budget_state=training_budget_state,
        ),
        EpisodeSummaryCallback(run_dir, progress_state),
        BestModelCallback(
            args.best_distance_model_path,
            args.best_reward_model_path,
            metadata,
            progress_state,
            learning_starts=gradient_update_start_timestep(args),
            save_best_replay_buffer=args.save_best_replay_buffer,
            training_budget_state=training_budget_state,
        ),
    ]
    if not args.no_progress_bar and not args.verbose_training:
        ConsoleProgressCallback = make_console_progress_callback_class(BaseCallback)
        callbacks.append(
            ConsoleProgressCallback(
                args.total_timesteps,
                progress_state,
                training_budget_state=training_budget_state,
            )
        )
    callback = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=training_budget_state.maximum_environment_steps,
            callback=callback,
            reset_num_timesteps=True,
            log_interval=10,
        )
        end_summary = raw_env.write_end_of_run_summary(
            training_budget_state.stop_reason or "environment_step_limit_reached"
        )
        model.save(str(args.model_path))
        if args.save_replay_buffer:
            model.save_replay_buffer(str(args.replay_buffer_path))
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["timestep_budget"]["run_state"] = (
            training_budget_state.as_dict()
        )
        safe_actor_status = getattr(model, "safe_actor_status", None)
        if callable(safe_actor_status):
            metadata["safe_actor_improvement"]["run_state"] = safe_actor_status()
        if end_summary is not None:
            metadata["end_of_run_summary"] = end_summary
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        print(
            "Budget result: "
            f"training={training_budget_state.training_steps:,}/"
            f"{training_budget_state.target_training_steps:,}, "
            f"probes={training_budget_state.probe_steps:,}/"
            f"{training_budget_state.effective_maximum_probe_steps:,}, "
            f"environment={training_budget_state.environment_steps:,}, "
            "deferred probe requests="
            f"{training_budget_state.deferred_probe_requests:,}."
        )
        print(f"Saved Agent 6 TD3 model to {args.model_path}")
        print(f"Saved training run logs to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
