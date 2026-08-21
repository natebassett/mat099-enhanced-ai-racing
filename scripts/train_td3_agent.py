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
    AGENT6_ACTION_VERSION,
    AGENT6_MODEL_FAMILY,
    AGENT6_OBSERVATION_VERSION,
    AGENT6_ROBUSTNESS_PROTOCOL,
    AGENT6_ROBUSTNESS_REPEATS,
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
    SeededSteeringActionNoise,
    TRAINING_CHALLENGE_SEED_MAX,
    TRAINING_CHALLENGE_SEED_MIN,
    build_rotating_training_seed_suite,
    build_td3_observation,
    clamp,
    decode_td3_action,
    episode_metadata_is_deterministic_probe,
    evaluation_checkpoint_is_verified,
    finite_float,
    metadata_path_for_policy,
    policy_contract_mismatches,
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
DEFAULT_SAFE_ACTOR_PROBE_REPEATS = AGENT6_ROBUSTNESS_REPEATS
DEFAULT_SAFE_ACTOR_PROBE_BASE_SEED = AGENT6_TRAINING_CHALLENGE_BASE_SEED
DEFAULT_SAFE_ACTOR_PROBE_NOISE_STD = AGENT6_ROBUSTNESS_STEERING_NOISE_STD
DEFAULT_SAFE_ACTOR_MIN_IMPROVEMENT_M = 5.0
DEFAULT_SAFE_ACTOR_MAX_PAIRED_REGRESSION_M = 100.0
DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE = 256
SAFE_ACTOR_ROBUSTNESS_PROTOCOL = AGENT6_ROBUSTNESS_PROTOCOL

REWARD_VERSION = "agent6_td3_reward_v1_raw_progress_curriculum"
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
    "policy_controlled",
    "deterministic_probe",
    "safe_actor_candidate_probe",
    "safe_actor_probe_mode",
    "safe_actor_probe_seed",
    "safe_actor_candidate_id",
    "safe_actor_probe_index",
    "safe_actor_decision",
    "safe_actor_probe_median_distance_m",
    "safe_actor_probe_minimum_distance_m",
    "safe_actor_reference_median_distance_m",
    "safe_actor_reference_minimum_distance_m",
    "safe_actor_paired_median_delta_m",
    "safe_actor_worst_paired_delta_m",
    "safe_actor_paired_regressions",
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
    "raw_accel",
    "raw_brake",
    "steer",
    "accel",
    "brake",
    "gear",
    "reward",
    "terminated",
    "truncated",
    "termination_reason",
    "cur_lap_time",
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
        for key in policy_contract_mismatches(metadata)
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


def safe_actor_probe_decision(
    reference_distances_m: list[float],
    candidate_distances_m: list[float],
    *,
    minimum_improvement_m: float,
    maximum_paired_regression_m: float,
) -> SafeActorProbeComparison:
    """Compare policies on identical disturbances using paired non-inferiority."""
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
    paired_deltas = candidate - reference
    reference_median = float(np.median(reference))
    candidate_median = float(np.median(candidate))
    reference_minimum = float(np.min(reference))
    candidate_minimum = float(np.min(candidate))
    aggregate_median_gain = candidate_median - reference_median
    minimum_gain = candidate_minimum - reference_minimum
    paired_median_delta = float(np.median(paired_deltas))
    worst_paired_delta = float(np.min(paired_deltas))
    paired_regressions = int(np.count_nonzero(paired_deltas < 0.0))
    accepted = (
        aggregate_median_gain >= required_gain
        and minimum_gain >= 0.0
        and paired_median_delta >= required_gain
        and worst_paired_delta >= -regression_limit
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
    )


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

    reward = clamp(progress, -2.0, 10.0) * stage.progress_weight
    reward += clamp(aligned_speed / 190.0, 0.0, 1.0) * 0.55
    reward += clamp(new_distance, 0.0, 12.0) * stage.milestone_weight
    if progress > 0.18 and abs(track_position) < 0.85 and abs(angle) < 0.55:
        reward += 0.45
    if speed < 24.0 and progress > 0.08:
        reward += 0.35

    reward -= 0.015
    reward -= max(0.0, abs(track_position) - 0.45) * 1.6
    reward -= max(0.0, abs(track_position) - 0.76) * 5.8
    reward -= max(0.0, abs(track_position) - 0.92) * 12.0
    reward -= max(0.0, abs(angle) - 0.10) * 1.1
    reward -= max(0.0, abs(angle) - 0.45) * 4.5
    reward -= max(0.0, abs(lateral_speed) - 4.0) * 0.10
    reward -= max(0.0, abs(lateral_speed) - 12.0) * 0.24

    speed_pressure = clamp((speed - 40.0) / 115.0, 0.0, 1.0)
    reward -= max(0.0, steer - 0.28) * speed_pressure * 5.5
    reward -= max(0.0, steer - 0.55) * speed_pressure * 12.0
    reward -= max(0.0, abs(angle) - 0.35) * speed_pressure * 7.0
    reward -= max(0.0, abs(lateral_speed) - 7.0) * speed_pressure * 0.28
    if abs(angle) > 1.05 and speed > 18.0:
        spin_pressure = clamp((abs(angle) - 1.05) / 0.75, 0.0, 1.0)
        reward -= 42.0 * spin_pressure

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
        reward -= front_pressure * overspeed_pressure * 3.2
        reward -= front_pressure * overspeed_pressure * max(0.0, 0.25 - brake) * 4.0
        reward -= front_pressure * overspeed_pressure * accel * 2.4

    if progress < 0.03 and speed < 8.0:
        reward -= 4.0
    if min_sensor < 0.0:
        reward -= 28.0
    if abs(track_position) > 1.0:
        reward -= 42.0
    if crashed:
        reward -= 65.0 + max(0.0, damage - previous_damage) * 0.15
    if backwards:
        reward -= 145.0 + clamp(abs(speed) / 120.0, 0.0, 1.0) * 55.0
    if stuck:
        reward -= 80.0
    if terminal_failure and completed_lap is None and not stage_success:
        remaining = 1.0 - calculate_lap_completion_fraction(episode_distance_m)
        reward -= stage.failure_penalty * (0.35 + 0.65 * remaining)
    if stage_success:
        reward += stage.success_reward
    if completed_lap is not None:
        reward += max(stage.success_reward, 850.0)

    return float(clamp(reward, -450.0, 1000.0))


def make_training_metadata(args: argparse.Namespace) -> dict[str, Any]:
    source_metadata = read_policy_metadata(args.resume_from)
    source_evaluation = source_metadata.get("best_evaluation")
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
        "observation_version": AGENT6_OBSERVATION_VERSION,
        "action_version": AGENT6_ACTION_VERSION,
        "reward_version": REWARD_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
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
            "source_verified_evaluation_progress_m": source_progress_m,
            "source_verified_evaluation_minimum_progress_m": (
                source_minimum_progress_m
            ),
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
            "actor_updates_per_candidate_block": args.safe_actor_block_updates,
            "robustness_protocol": SAFE_ACTOR_ROBUSTNESS_PROTOCOL,
            "robustness_probe_repeats": args.safe_actor_probe_repeats,
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
            "probe_pairing": "interleaved_accepted_then_candidate_per_seed",
            "probe_episodes_per_candidate": 2 * args.safe_actor_probe_repeats,
            "robustness_probe_steering_noise_std": (
                args.safe_actor_probe_noise_std
            ),
            "reference_policy": "last_accepted_actor_on_same_challenge_seeds",
            "acceptance_metric": (
                "aggregate_and_paired_median_gain_with_minimum_and_paired_guards"
            ),
            "minimum_improvement_m": args.safe_actor_min_improvement_m,
            "maximum_paired_regression_m": (
                args.safe_actor_max_paired_regression_m
            ),
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
    warmup_action_mode: str = "launch-biased",
    warmup_steer_std: float = 0.30,
    warmup_accel_min: float = 0.25,
    warmup_accel_max: float = 1.00,
    warmup_brake_probability: float = 0.12,
):
    class ScratchActionBox(spaces.Box):
        def __init__(self) -> None:
            super().__init__(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
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
            accel = float(rng.uniform(self.warmup_accel_min, self.warmup_accel_max))
            brake = 0.0
            if float(rng.random()) < self.warmup_brake_probability:
                accel = float(rng.uniform(-0.25, 0.20))
                brake = float(rng.uniform(0.15, 0.85))
            return np.asarray(
                [
                    steer,
                    clamp(accel, -1.0, 1.0),
                    clamp(brake, -1.0, 1.0),
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
        ) -> None:
            super().__init__()
            self.track_name = track_name
            self.manual_start = manual_start
            self.relaunch_frequency = max(0, int(relaunch_frequency))
            self.reset_retries = max(0, int(reset_retries))
            self.quiet_reset_log = quiet_reset_log
            self.run_dir = Path(run_dir) if run_dir is not None else None
            self.learning_starts = max(0, int(learning_starts))
            self.runner = make_torcs_runner()
            if use_custom_warmup_action_space:
                self.action_space = make_scratch_action_space(
                    spaces,
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
                    shape=(3,),
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
            self.episode_start_timestep = self.total_steps
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
            action = decode_td3_action(raw_values)
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
            reward = calculate_td3_reward(
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
            truncated = self.steps >= self.current_stage.max_episode_steps
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

            self._write_step_telemetry(
                telemetry,
                raw_values,
                action,
                reward,
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
            terminated: bool,
            truncated: bool,
            reason: str,
        ) -> None:
            if self._step_writer is None:
                return
            sensors = track_sensors(telemetry)
            padded = np.zeros(3, dtype=float)
            padded[: min(3, len(raw_action))] = raw_action[:3]
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
                    padded[1],
                    padded[2],
                    action["steer"],
                    action["accel"],
                    action["brake"],
                    action["gear"],
                    reward,
                    terminated,
                    truncated,
                    reason,
                    telemetry.get("curLapTime", 0),
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
    return (
        f"Training [{bar}] {percent:5.1f}% | "
        f"{completed_steps:,}/{total_steps:,} steps | {rate} | "
        f"ETA {eta} | stage={progress_state.current_stage} | "
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
            self.probe_episodes_completed = 0
            self.policy_episodes_since_probe = 0
            self.current_episode_is_probe = False
            self.last_noise_update_timestep = -ACTION_NOISE_UPDATE_INTERVAL_STEPS

        def _on_training_start(self):
            self._apply_policy_mode(deterministic_probe=False, force=True)

        def _on_step(self):
            completed_episode = False
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                completed_episode = True
                if bool(summary.get("policy_controlled")):
                    if bool(summary.get("deterministic_probe")):
                        safe_result = self._record_safe_actor_probe(summary)
                        if safe_result is not None:
                            self._annotate_safe_actor_probe(summary, safe_result)
                        self.probe_episodes_completed += 1
                        self.policy_episodes_since_probe = 0
                    else:
                        self.policy_episodes_since_probe += 1

            if completed_episode:
                next_episode_is_probe = (
                    self._safe_actor_probe_required()
                    or should_schedule_policy_probe(
                        timestep=self.num_timesteps,
                        learning_starts=self.learning_starts,
                        probe_episodes_completed=self.probe_episodes_completed,
                        policy_episodes_since_probe=self.policy_episodes_since_probe,
                        probe_interval=self.probe_interval,
                    )
                )
                self._apply_policy_mode(
                    deterministic_probe=next_episode_is_probe,
                    force=True,
                )
            elif self.current_episode_is_probe:
                # The first probe action is selected after the preceding rollout's
                # final update. Freeze from this point until the probe terminates.
                self.model.gradient_steps = 0
            elif (
                self.num_timesteps >= self.learning_starts
                and self.model.gradient_steps != self.training_gradient_steps
            ):
                self._apply_policy_mode(deterministic_probe=False, force=True)
            elif (
                not self.current_episode_is_probe
                and self.num_timesteps - self.last_noise_update_timestep
                >= ACTION_NOISE_UPDATE_INTERVAL_STEPS
            ):
                self._apply_policy_mode(deterministic_probe=False, force=False)
            return True

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
                self.num_timesteps,
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
                self.num_timesteps,
                learning_starts=self.learning_starts,
                initial_sigma=self.initial_sigma,
                final_sigma=self.final_sigma,
                decay_steps=self.decay_steps,
            )
            safe_context = (
                self._safe_actor_probe_context() if deterministic_probe else None
            )
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
                if self.num_timesteps < self.learning_starts
                else self.training_gradient_steps
            )
            self.current_episode_is_probe = deterministic_probe
            self.last_noise_update_timestep = self.num_timesteps
            if safe_actor_probe:
                training_phase = f"safe_actor_{safe_context['mode']}_probe"
            elif deterministic_probe:
                training_phase = "deterministic_probe"
            else:
                training_phase = continuation_training_phase(
                    self.num_timesteps,
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
            )

        def _safe_actor_probe_required(self) -> bool:
            required = getattr(self.model, "safe_actor_probe_required", None)
            return bool(callable(required) and required())

        def _safe_actor_probe_context(self) -> dict[str, Any] | None:
            context = getattr(self.model, "safe_actor_probe_context", None)
            value = context() if callable(context) else None
            return dict(value) if isinstance(value, Mapping) else None

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
            summary["safe_actor_decision"] = result.get("decision")
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
            if decision in {"accepted", "rolled_back"}:
                print(
                    "\nSafe actor candidate "
                    f"{result['candidate_id']} {decision}: "
                    "reference median="
                    f"{float(result['reference_median_distance_m']):.1f}m, "
                    "candidate median="
                    f"{float(result['median_distance_m']):.1f}m, "
                    "paired median="
                    f"{float(result['paired_median_delta_m']):+.1f}m, "
                    "worst pair="
                    f"{float(result['worst_paired_delta_m']):+.1f}m."
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
        ) -> None:
            super().__init__(verbose=0)
            self.best_distance_model_path = Path(best_distance_model_path)
            self.best_reward_model_path = Path(best_reward_model_path)
            self.metadata = dict(metadata)
            self.progress_state = progress_state
            self.learning_starts = max(0, int(learning_starts))
            self.save_best_replay_buffer = bool(save_best_replay_buffer)
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
                    or self.num_timesteps <= self.learning_starts
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
        ) -> None:
            super().__init__(verbose=0)
            self.total_timesteps = int(total_timesteps)
            self.progress_state = progress_state
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
            completed_steps = self.num_timesteps - self.start_timestep
            line = format_training_progress_line(
                completed_steps,
                self.total_timesteps,
                now - self.start_time,
                self.progress_state,
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


def make_checkpoint_callback(CheckpointCallback: Any, args: argparse.Namespace):
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
            ]

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
            safe_actor_probe_repeats: int = DEFAULT_SAFE_ACTOR_PROBE_REPEATS,
            safe_actor_min_improvement_m: float = (
                DEFAULT_SAFE_ACTOR_MIN_IMPROVEMENT_M
            ),
            safe_actor_max_paired_regression_m: float = (
                DEFAULT_SAFE_ACTOR_MAX_PAIRED_REGRESSION_M
            ),
            safe_actor_anchor_batch_size: int = (
                DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE
            ),
            safe_actor_probe_base_seed: int = DEFAULT_SAFE_ACTOR_PROBE_BASE_SEED,
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
            self._agent6_safe_actor_probe_repeats = max(
                1,
                int(safe_actor_probe_repeats),
            )
            self._agent6_safe_actor_min_improvement_m = max(
                0.0,
                float(safe_actor_min_improvement_m),
            )
            self._agent6_safe_actor_max_paired_regression_m = max(
                0.0,
                float(safe_actor_max_paired_regression_m),
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
            self._agent6_safe_actor_accepted_distance_m = 0.0
            self._agent6_safe_actor_accepted_minimum_distance_m = 0.0
            self._agent6_safe_actor_candidate_id = 0
            self._agent6_safe_actor_candidate_updates = 0
            self._agent6_safe_actor_waiting_for_probe = False
            self._agent6_safe_actor_probe_phase = "idle"
            self._agent6_safe_actor_probe_pair_index = 0
            self._agent6_safe_actor_probe_seeds: tuple[int, ...] = ()
            self._agent6_safe_actor_reference_distances_m: list[float] = []
            self._agent6_safe_actor_candidate_distances_m: list[float] = []
            self._agent6_safe_actor_accepted_blocks = 0
            self._agent6_safe_actor_rolled_back_blocks = 0
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
            self._agent6_safe_actor_probe_pair_index = 0
            self._agent6_safe_actor_reference_distances_m = []
            self._agent6_safe_actor_candidate_distances_m = []
            self._agent6_safe_actor_probe_phase = "accepted_reference"
            self._restore_safe_actor()

        def safe_actor_probe_required(self) -> bool:
            return bool(
                getattr(self, "_agent6_safe_actor_enabled", False)
                and getattr(
                    self,
                    "_agent6_safe_actor_waiting_for_probe",
                    False,
                )
            )

        def safe_actor_probe_context(self) -> dict[str, Any] | None:
            if not self.safe_actor_probe_required():
                return None
            self._begin_safe_actor_comparison()
            probe_index = self._agent6_safe_actor_probe_pair_index + 1
            return {
                "mode": self._agent6_safe_actor_probe_phase,
                "candidate_id": self._agent6_safe_actor_candidate_id,
                "probe_index": probe_index,
                "probe_repeats": self._agent6_safe_actor_probe_repeats,
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
                "candidate_id": int(
                    getattr(self, "_agent6_safe_actor_candidate_id", 0)
                ),
                "candidate_updates": int(
                    getattr(self, "_agent6_safe_actor_candidate_updates", 0)
                ),
                "waiting_for_probe": self.safe_actor_probe_required(),
                "probe_phase": str(
                    getattr(self, "_agent6_safe_actor_probe_phase", "idle")
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
                "decision": "pending",
                "median_distance_m": None,
                "minimum_distance_m": None,
                "reference_median_distance_m": None,
                "reference_minimum_distance_m": None,
                "paired_median_delta_m": None,
                "worst_paired_delta_m": None,
                "paired_regressions": None,
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
                self._restore_safe_actor_candidate()
                self._agent6_safe_actor_probe_phase = "candidate"
                result["decision"] = "reference_recorded"
                return result

            self._agent6_safe_actor_candidate_distances_m.append(distance_m)
            pair_index = self._agent6_safe_actor_probe_pair_index
            if pair_index + 1 < self._agent6_safe_actor_probe_repeats:
                self._agent6_safe_actor_probe_pair_index += 1
                self._agent6_safe_actor_probe_phase = "accepted_reference"
                self._restore_safe_actor()
                result["decision"] = "pair_complete"
                return result

            comparison = safe_actor_probe_decision(
                self._agent6_safe_actor_reference_distances_m,
                self._agent6_safe_actor_candidate_distances_m,
                minimum_improvement_m=self._agent6_safe_actor_min_improvement_m,
                maximum_paired_regression_m=(
                    self._agent6_safe_actor_max_paired_regression_m
                ),
            )
            if comparison.decision == "accepted":
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
            self._agent6_safe_actor_candidate_updates = 0
            self._agent6_safe_actor_waiting_for_probe = False
            self._agent6_safe_actor_probe_phase = "idle"
            self._agent6_safe_actor_probe_pair_index = 0
            self._agent6_safe_actor_probe_seeds = ()
            self._agent6_safe_actor_reference_distances_m = []
            self._agent6_safe_actor_candidate_distances_m = []
            self._clear_safe_actor_candidate_snapshot()
            result.update(
                {
                    "decision": comparison.decision,
                    "median_distance_m": comparison.candidate_median_distance_m,
                    "minimum_distance_m": comparison.candidate_minimum_distance_m,
                    "reference_median_distance_m": (
                        comparison.reference_median_distance_m
                    ),
                    "reference_minimum_distance_m": (
                        comparison.reference_minimum_distance_m
                    ),
                    "aggregate_median_gain_m": (
                        comparison.aggregate_median_gain_m
                    ),
                    "minimum_gain_m": comparison.minimum_gain_m,
                    "paired_median_delta_m": comparison.paired_median_delta_m,
                    "worst_paired_delta_m": comparison.worst_paired_delta_m,
                    "paired_regressions": comparison.paired_regressions,
                    "paired_deltas_m": list(comparison.paired_deltas_m),
                    "accepted_distance_m": (
                        self._agent6_safe_actor_accepted_distance_m
                    ),
                    "accepted_minimum_distance_m": (
                        self._agent6_safe_actor_accepted_minimum_distance_m
                    ),
                }
            )
            return result

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
            self._agent6_safe_actor_candidate_updates = 0
            self._agent6_safe_actor_waiting_for_probe = False
            self._agent6_safe_actor_probe_phase = "idle"
            self._agent6_safe_actor_probe_pair_index = 0
            self._agent6_safe_actor_probe_seeds = ()
            self._agent6_safe_actor_reference_distances_m = []
            self._agent6_safe_actor_candidate_distances_m = []
            self._clear_safe_actor_candidate_snapshot()
            self._agent6_safe_actor_rolled_back_blocks += 1
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
            actor_schedule = getattr(
                self,
                "_agent6_actor_learning_rate_schedule",
                None,
            )
            if actor_schedule is None:
                super().train(gradient_steps, batch_size)
                return

            self.policy.set_training_mode(True)
            critic_learning_rate = self.lr_schedule(
                self._current_progress_remaining
            )
            actor_learning_rate = float(actor_schedule(self.num_timesteps))
            update_learning_rate(self.critic.optimizer, critic_learning_rate)
            update_learning_rate(self.actor.optimizer, actor_learning_rate)
            self.logger.record("train/learning_rate", critic_learning_rate)
            self.logger.record("train/critic_learning_rate", critic_learning_rate)
            self.logger.record("train/actor_learning_rate", actor_learning_rate)

            actor_updates_enabled = (
                self.num_timesteps
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

                if actor_updates_enabled and not self.safe_actor_probe_required():
                    actor_loss = -self.critic.q1_forward(
                        replay_data.observations,
                        self.actor(replay_data.observations),
                    ).mean()
                    actor_losses.append(float(actor_loss.item()))
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    if getattr(self, "_agent6_safe_actor_enabled", False):
                        gradient_norm = th.nn.utils.clip_grad_norm_(
                            self.actor.parameters(),
                            self._agent6_safe_actor_gradient_norm,
                        )
                        actor_gradient_norms.append(float(gradient_norm.item()))
                    self.actor.optimizer.step()
                    if getattr(self, "_agent6_safe_actor_enabled", False):
                        if self._agent6_safe_actor_candidate_updates == 0:
                            self._agent6_safe_actor_candidate_id += 1
                        constraint_observations = (
                            self._safe_actor_constraint_observations(
                                replay_data.observations
                            )
                        )
                        actor_action_deltas.append(
                            self._project_safe_actor(constraint_observations)
                        )
                        self._agent6_safe_actor_candidate_updates += 1
                        if (
                            self._agent6_safe_actor_candidate_updates
                            >= self._agent6_safe_actor_block_updates
                        ):
                            self._agent6_safe_actor_waiting_for_probe = True
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
        model = TD3.load(str(args.resume_from), **model_kwargs)
        normalise_continuation_spaces(model, env)
        return model
    return TD3(
        "MlpPolicy",
        policy_kwargs={"net_arch": args.net_arch},
        **model_kwargs,
    )


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
    parser.add_argument("--total-timesteps", type=int, default=100_000)
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
        help="Actor optimizer updates allowed before an acceptance probe set.",
    )
    parser.add_argument(
        "--safe-actor-probe-repeats",
        type=int,
        default=DEFAULT_SAFE_ACTOR_PROBE_REPEATS,
        help="Fresh matched seed pairs used to compare each actor candidate.",
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
        "--safe-actor-anchor-batch-size",
        type=int,
        default=DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE,
        help="Replay observations retained for action-space trust-region checks.",
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

    if not args.continuation and args.start_stage != "launch":
        parser.error("--start-stage requires --resume-from unless it is launch")
    if args.continuation:
        source_errors = continuation_source_errors(
            args.resume_from,
            track=args.track,
            start_stage=args.start_stage,
        )
        if source_errors:
            parser.error(source_errors[0])
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
    if args.safe_actor_probe_repeats < 1:
        parser.error("--safe-actor-probe-repeats must be at least 1")
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
    if args.safe_actor_anchor_batch_size < 1:
        parser.error("--safe-actor-anchor-batch-size must be at least 1")
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
    write_json(metadata_path_for_policy(args.model_path), metadata)

    tensorboard_log = resolve_tensorboard_log_dir(
        args.tensorboard_dir,
        disable_tensorboard=args.no_tensorboard,
    )
    def make_action_noise(sigma: float):
        return NormalActionNoise(
            mean=np.zeros(3, dtype=np.float32),
            sigma=np.ones(3, dtype=np.float32) * max(0.0, float(sigma)),
        )

    def make_safe_probe_noise(seed: int, steering_noise_std: float):
        return SeededSteeringActionNoise(
            seed=seed,
            steering_noise_std=steering_noise_std,
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
            safe_actor_probe_repeats=args.safe_actor_probe_repeats,
            safe_actor_min_improvement_m=args.safe_actor_min_improvement_m,
            safe_actor_max_paired_regression_m=(
                args.safe_actor_max_paired_regression_m
            ),
            safe_actor_anchor_batch_size=args.safe_actor_anchor_batch_size,
            safe_actor_probe_base_seed=args.safe_actor_probe_base_seed,
        )
        metadata["continuation"]["source_model_num_timesteps"] = int(
            getattr(model, "num_timesteps", 0)
        )
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        print(
            "Continuing verified Agent 6 policy from "
            f"{args.resume_from} at stage={args.start_stage}; "
            f"replay refill={args.replay_refill_steps:,}, "
            f"critic warm-up={args.critic_warmup_steps:,}, "
            f"actor unfreeze={args.actor_unfreeze_steps:,} steps; "
            f"safe actor blocks={'on' if args.safe_actor_improvement else 'off'}; "
            f"robust probes={args.safe_actor_probe_repeats} matched pairs "
            "per candidate."
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
        ),
        make_checkpoint_callback(CheckpointCallback, args),
        EpisodeSummaryCallback(run_dir, progress_state),
        BestModelCallback(
            args.best_distance_model_path,
            args.best_reward_model_path,
            metadata,
            progress_state,
            learning_starts=gradient_update_start_timestep(args),
            save_best_replay_buffer=args.save_best_replay_buffer,
        ),
    ]
    if not args.no_progress_bar and not args.verbose_training:
        ConsoleProgressCallback = make_console_progress_callback_class(BaseCallback)
        callbacks.append(ConsoleProgressCallback(args.total_timesteps, progress_state))
    callback = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            log_interval=10,
        )
        end_summary = raw_env.write_end_of_run_summary("total_timesteps_reached")
        model.save(str(args.model_path))
        if args.save_replay_buffer:
            model.save_replay_buffer(str(args.replay_buffer_path))
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        safe_actor_status = getattr(model, "safe_actor_status", None)
        if callable(safe_actor_status):
            metadata["safe_actor_improvement"]["run_state"] = safe_actor_status()
        if end_summary is not None:
            metadata["end_of_run_summary"] = end_summary
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        print(f"Saved Agent 6 TD3 model to {args.model_path}")
        print(f"Saved training run logs to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
