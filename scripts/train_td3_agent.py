from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
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
    AGENT6_ROBUSTNESS_STEERING_NOISE_STD,
    AGENT6_TRAINING_CHALLENGE_BASE_SEED,
    DEFAULT_BEST_EVALUATION_MODEL_PATH,
    DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH,
    DEFAULT_BEST_ROBUST_EVALUATION_MODEL_PATH,
    DEFAULT_BEST_DISTANCE_MODEL_PATH,
    DEFAULT_BEST_REWARD_MODEL_PATH,
    DEFAULT_FRONTIER_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    EXTERNAL_EVALUATION_SEED_MAX,
    EXTERNAL_EVALUATION_SEED_MIN,
    FEATURE_NAMES,
    G_TRACK_3_LENGTH_METRES,
    MAX_REPRODUCIBLE_SEED,
    PEDAL_DEADZONE,
    SeededGaussianActionNoise,
    SeededLegacyActionNoise,
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
from agents.td3_checkpointing import (  # noqa: E402
    atomic_save_policy,
    atomic_write_json,
    promote_best_completed_lap,
    read_json_mapping,
)
from runner.lap_tracker import LapTracker, practice_finish_is_plausible  # noqa: E402


DEFAULT_TRACK_NAME = "g-track-3"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "models" / "training_runs" / "agent6_td3_scratch"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "agent6_td3_scratch"
DEFAULT_TENSORBOARD_DIR = PROJECT_ROOT / "models" / "tensorboard" / "agent6_td3_scratch"
DEFAULT_REPLAY_BUFFER_PATH = PROJECT_ROOT / "models" / "replay_buffers" / "agent6_td3_scratch.pkl"
DEFAULT_LAP_CHECKPOINT_DIR = (
    PROJECT_ROOT / "models" / "checkpoints" / "agent6_td3_laps"
)
DEFAULT_FRONTIER_CHECKPOINT_DIR = (
    PROJECT_ROOT / "models" / "checkpoints" / "agent6_td3_frontier"
)
DEFAULT_FRONTIER_EVIDENCE_PATH = (
    PROJECT_ROOT / "data" / "policies" / "agent6_td3_frontier_evidence.json"
)
DEFAULT_FRONTIER_MILESTONES_M = (1_500.0, 1_700.0, 1_900.0)
DEFAULT_FRONTIER_MIN_IMPROVEMENT_M = 5.0
DEFAULT_FRONTIER_MAJOR_BREAKTHROUGH_M = 25.0
DEFAULT_FRONTIER_QUALIFICATION_DISTANCE_M = 1_400.0
DEFAULT_FRONTIER_QUALIFICATION_ATTEMPTS = 3
DEFAULT_FRONTIER_EXPLORATION_START_DISTANCE_M = 1_300.0
DEFAULT_DISCOVERY_COLLAPSE_EPISODES = 8
DEFAULT_DISCOVERY_COLLAPSE_DISTANCE_M = 5.0
DEFAULT_DISCOVERY_FRONTIER_PRESERVATION_FRACTION = 1.0
DEFAULT_DISCOVERY_RECOVERY_COOLDOWN_STEPS = 10_000
DEFAULT_DISCOVERY_MAX_RECOVERY_COOLDOWN_STEPS = 40_000
DEFAULT_DISCOVERY_ACTOR_LR_BACKOFF = 0.5
DEFAULT_DISCOVERY_MIN_ACTOR_LR_SCALE = 0.125
DEFAULT_DISCOVERY_MAX_POLICY_DELAY = 8
DEFAULT_EPISODE_GRADIENT_STEPS = 256
DEFAULT_FROZEN_POLICY_HEALTH_FRACTION = 0.65
DEFAULT_FROZEN_POLICY_HEALTH_FAILURES = 2
LEAN_FRONTIER_REPLAY_REFILL_STEPS = 5_000
LEAN_FRONTIER_CRITIC_WARMUP_STEPS = 10_000
LEAN_FRONTIER_CRITIC_LEARNING_RATE = 3e-4
LEAN_FRONTIER_ACTOR_LEARNING_RATE = 3e-5
LEAN_FRONTIER_ACTION_NOISE_SIGMA = 0.05
LEAN_FRONTIER_FINAL_ACTION_NOISE_SIGMA = 0.05
LEAN_FRONTIER_TARGET_POLICY_NOISE = 0.10
LEAN_FRONTIER_TARGET_NOISE_CLIP = 0.20
LEAN_FRONTIER_GRADIENT_STEPS = 1
LEAN_FRONTIER_GRADIENT_NORM = 1.0
LEAN_FRONTIER_HUBER_BETA = 1.0
DEFAULT_AUTOMATIC_EVALUATION_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "agent6_td3"
)
DEFAULT_EMERGENCY_CHECKPOINT_PATH = (
    DEFAULT_CHECKPOINT_DIR / "agent6_td3_emergency.zip"
)
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
DEFAULT_CONTINUATION_ACTION_NOISE_SIGMA = AGENT6_ROBUSTNESS_STEERING_NOISE_STD
DEFAULT_CONTINUATION_FINAL_ACTION_NOISE_SIGMA = 0.002
DEFAULT_SAFE_ACTOR_MAX_ACTION_DELTA = 0.01
DEFAULT_SAFE_ACTOR_GRADIENT_NORM = 1.0
DEFAULT_SAFE_ACTOR_BLOCK_UPDATES = 100
DEFAULT_SAFE_ACTOR_BLOCKS_PER_PROBE = 1
DEFAULT_SAFE_ACTOR_PROBE_REPEATS = 3
DEFAULT_SAFE_ACTOR_TRIALS_PER_SEED = 2
DEFAULT_SAFE_ACTOR_SCREENING_TRIALS_PER_SEED = 2
DEFAULT_SAFE_ACTOR_SCREENING_MAX_EPISODE_STEPS = 1_200
DEFAULT_SAFE_ACTOR_PROBE_BASE_SEED = AGENT6_TRAINING_CHALLENGE_BASE_SEED
DEFAULT_SAFE_ACTOR_PROBE_NOISE_STD = AGENT6_ROBUSTNESS_STEERING_NOISE_STD
DEFAULT_SAFE_ACTOR_MIN_IMPROVEMENT_M = 5.0
DEFAULT_SAFE_ACTOR_MEDIAN_REGRESSION_TOLERANCE_M = 25.0
DEFAULT_SAFE_ACTOR_MINIMUM_REGRESSION_TOLERANCE_M = 150.0
DEFAULT_SAFE_ACTOR_MAX_PAIRED_REGRESSION_M = 200.0
DEFAULT_SAFE_ACTOR_SCREENING_MAX_REGRESSION_M = 250.0
DEFAULT_SAFE_ACTOR_ANCHOR_BATCH_SIZE = 256
DEFAULT_MAX_PROBE_FRACTION = 0.20
DEFAULT_PROBE_EVIDENCE_WINDOW = 8
DEFAULT_PROBE_MIN_EVIDENCE_EPISODES = 3
DEFAULT_PROBE_EVIDENCE_MAX_REGRESSION_M = 150.0
DEFAULT_BREAKTHROUGH_IMPROVEMENT_M = 25.0
TRAINING_ACTION_NOISE_SEED_STRIDE = 0x9E3779B9
SAFE_ACTOR_ROBUSTNESS_PROTOCOL = (
    "agent6_breakthrough_aware_working_policy_internal_probe_v12"
)
FRONTIER_CONTINUATION_PROTOCOL = "agent6_frontier_trajectory_gate_v1"

REWARD_VERSION = "agent6_td3_reward_v4_streamlined_progress"
EXACT_TRAINING_STATE_VERSION = "agent6_td3_exact_training_state_v1"
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
SECTOR_PROGRESS_MILESTONE_REWARDS = {
    1450.0: 120.0,
    1550.0: 160.0,
    1750.0: 220.0,
}
SECTOR_PROGRESS_MILESTONES_M = tuple(SECTOR_PROGRESS_MILESTONE_REWARDS)


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
SAFE_ACTOR_PROGRESS_MILESTONES_M = tuple(
    sorted(
        {
            *(stage.distance_target_m for stage in CURRICULUM),
            *SECTOR_PROGRESS_MILESTONES_M,
        }
    )
)

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
    "safe_actor_training_evidence_reasons",
    "safe_actor_probe_episodes_completed",
    "safe_actor_probe_episodes_maximum",
    "safe_actor_probe_episodes_skipped",
    "safe_actor_probe_median_distance_m",
    "safe_actor_probe_minimum_distance_m",
    "safe_actor_probe_upper_quartile_distance_m",
    "safe_actor_probe_maximum_distance_m",
    "safe_actor_reference_median_distance_m",
    "safe_actor_reference_minimum_distance_m",
    "safe_actor_reference_upper_quartile_distance_m",
    "safe_actor_reference_maximum_distance_m",
    "safe_actor_aggregate_median_gain_m",
    "safe_actor_minimum_gain_m",
    "safe_actor_upper_quartile_gain_m",
    "safe_actor_maximum_gain_m",
    "safe_actor_paired_median_delta_m",
    "safe_actor_worst_paired_delta_m",
    "safe_actor_paired_regressions",
    "safe_actor_worst_order_position_delta_m",
    "safe_actor_order_position_regressions",
    "safe_actor_order_position_deltas_m",
    "safe_actor_raw_paired_regressions",
    "safe_actor_reference_seed_medians_m",
    "safe_actor_candidate_seed_medians_m",
    "safe_actor_reference_lap_completions",
    "safe_actor_candidate_lap_completions",
    "safe_actor_new_progress_milestones_m",
    "safe_actor_acceptance_reasons",
    "safe_actor_safety_failures",
    "safe_actor_accepted_distance_m",
    "safe_actor_accepted_minimum_distance_m",
    "discovery_launch_failure_streak",
    "discovery_recovery_triggered",
    "discovery_recovery_count",
    "discovery_recovery_cooldown_until",
    "discovery_actor_learning_rate_scale",
    "discovery_effective_policy_delay",
    "discovery_actor_updates_since_anchor",
    "discovery_frontier_gate_open",
    "discovery_frontier_gate_distance_m",
    "frozen_policy_health_threshold_m",
    "frozen_policy_health_failure_streak",
    "frozen_policy_health_decision",
    "curriculum_eligible",
    "training_phase",
    "action_noise_sigma",
    "stage_id",
    "stage_name",
    "steps",
    "reward",
    "distance_m",
    "furthest_distance_m",
    "sector_milestones_reached_m",
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
    "reward_forward_speed",
    "reward_track_edge_penalty",
    "reward_terminal_failure",
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
    )


def make_torcs_runner():
    with contextlib.redirect_stderr(io.StringIO()):
        from runner.torcs_runner import TorcsRunner

    return TorcsRunner()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


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


def training_state_path_for_policy(policy_path: Path) -> Path:
    return Path(policy_path).with_suffix(".training_state.pt")


def sector_milestones_crossed(
    previous_furthest_distance_m: float,
    furthest_distance_m: float,
    awarded_milestones_m: set[float] | tuple[float, ...] = (),
) -> tuple[float, ...]:
    """Return newly crossed sector gates in deterministic distance order."""
    previous = max(0.0, finite_float(previous_furthest_distance_m))
    current = max(previous, finite_float(furthest_distance_m))
    awarded = {float(value) for value in awarded_milestones_m}
    return tuple(
        milestone
        for milestone in SECTOR_PROGRESS_MILESTONES_M
        if milestone not in awarded and previous < milestone <= current
    )


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


def save_torch_payload_atomically(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    target = Path(path)
    temporary = target.with_name(f"{target.name}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        temporary.unlink()
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as file:
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def load_exact_training_state(policy_path: Path, *, device: Any = "cpu") -> dict[str, Any]:
    import torch

    state_path = training_state_path_for_policy(policy_path)
    if not state_path.is_file():
        raise FileNotFoundError(f"exact training state does not exist: {state_path}")
    try:
        payload = torch.load(state_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(state_path, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid exact training state payload: {state_path}")
    if payload.get("version") != EXACT_TRAINING_STATE_VERSION:
        raise RuntimeError(
            "exact training state version mismatch: expected "
            f"{EXACT_TRAINING_STATE_VERSION}, found {payload.get('version')}"
        )
    return payload


def save_exact_training_checkpoint(
    model: Any,
    raw_env: Any,
    exploration_callback: Any,
    training_budget_state: "TrainingBudgetState",
    policy_path: Path,
    metadata: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    target = Path(policy_path)
    replay_path = replay_buffer_path_for_policy(target)
    state_path = training_state_path_for_policy(target)
    save_runtime = getattr(model, "save_runtime_policy", None)
    save_policy = save_runtime if callable(save_runtime) else model.save
    atomic_save_policy(lambda path: save_policy(str(path)), target)
    replay_error = save_replay_buffer_atomically(model, replay_path)
    if replay_error is not None:
        raise RuntimeError(f"failed to save exact-resume replay: {replay_error}")
    checkpoint_record = {
        "version": EXACT_TRAINING_STATE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason),
        "policy_path": str(target),
        "replay_buffer_path": str(replay_path),
        "training_state_path": str(state_path),
        "episode_restart_required": bool(
            getattr(raw_env, "current_telemetry", None) is not None
            and int(getattr(raw_env, "steps", 0)) > 0
        ),
        "training_budget": training_budget_state.as_dict(),
        "curriculum_stage": raw_env.current_stage.stage_id,
        "sector_milestone_history": copy.deepcopy(
            raw_env.sector_milestone_history
        ),
    }
    payload = {
        "version": EXACT_TRAINING_STATE_VERSION,
        "checkpoint": checkpoint_record,
        "model": model.exact_training_state_dict(),
        "environment": raw_env.training_state_dict(),
        "exploration_callback": exploration_callback.training_state_dict(),
        "training_budget": training_budget_state.as_dict(),
    }
    save_torch_payload_atomically(payload, state_path)
    checkpoint_metadata = copy.deepcopy(dict(metadata))
    checkpoint_metadata["exact_resume_checkpoint"] = checkpoint_record
    checkpoint_metadata["timestep_budget"]["run_state"] = (
        training_budget_state.as_dict()
    )
    checkpoint_metadata["milestone_history"] = copy.deepcopy(
        raw_env.sector_milestone_history
    )
    atomic_write_json(metadata_path_for_policy(target), checkpoint_metadata)
    return checkpoint_record


def restore_exact_training_checkpoint(
    model: Any,
    raw_env: Any,
    exploration_callback: Any,
    training_budget_state: "TrainingBudgetState",
    policy_path: Path,
    *,
    allow_budget_extension: bool = False,
) -> dict[str, Any]:
    state = load_exact_training_state(
        policy_path,
        device=getattr(model, "device", "cpu"),
    )
    model.load_replay_buffer(str(replay_buffer_path_for_policy(policy_path)))
    model.restore_exact_training_state(state["model"])
    raw_env.restore_training_state(state["environment"])
    training_budget_state.restore(
        state["training_budget"],
        allow_budget_extension=allow_budget_extension,
    )
    exploration_callback.restore_training_state(
        state["exploration_callback"]
    )
    checkpoint = state.get("checkpoint")
    return dict(checkpoint) if isinstance(checkpoint, Mapping) else {}


def exact_training_checkpoint_errors(policy_path: Path) -> list[str]:
    path = Path(policy_path)
    errors: list[str] = []
    if not path.is_file():
        errors.append(f"exact-resume policy does not exist: {path}")
    state_path = training_state_path_for_policy(path)
    if not state_path.is_file():
        errors.append(f"exact-resume state does not exist: {state_path}")
    replay_path = replay_buffer_path_for_policy(path)
    if not replay_path.is_file():
        errors.append(f"exact-resume replay buffer does not exist: {replay_path}")
    metadata = read_policy_metadata(path)
    checkpoint = metadata.get("exact_resume_checkpoint")
    if not isinstance(checkpoint, Mapping):
        errors.append("exact-resume metadata is missing exact_resume_checkpoint")
    elif checkpoint.get("version") != EXACT_TRAINING_STATE_VERSION:
        errors.append("exact-resume metadata uses an unsupported state version")
    errors.extend(
        f"exact-resume checkpoint contract mismatch: {key}"
        for key in policy_contract_mismatches(
            metadata,
            allow_legacy_action_contract=True,
        )
    )
    return errors


def continuation_source_path(args: argparse.Namespace) -> Path | None:
    return getattr(args, "resume_exact_from", None) or getattr(
        args,
        "resume_from",
        None,
    )


def continuation_training_protocol_enabled(args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "continuation", False)):
        return False
    if not bool(getattr(args, "exact_resume", False)):
        return True
    return bool(getattr(args, "exact_source_was_continuation", True))


def lean_frontier_training_enabled(args: argparse.Namespace) -> bool:
    """Use live per-step stable fine-tuning for frontier discovery."""
    return bool(getattr(args, "lean_frontier_training", False))


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


def frontier_evidence_distance_m(
    metadata: Mapping[str, Any],
    external_evidence: Mapping[str, Any] | None = None,
) -> float:
    """Return the strongest recorded distance evidence without claiming robustness."""
    candidates: list[float] = []
    if isinstance(external_evidence, Mapping):
        candidates.append(finite_float(external_evidence.get("distance_m")))
    frontier = metadata.get("frontier_champion")
    if isinstance(frontier, Mapping):
        candidates.append(finite_float(frontier.get("distance_m")))
    evaluation = metadata.get("best_evaluation")
    if isinstance(evaluation, Mapping):
        candidates.extend(
            finite_float(evaluation.get(key))
            for key in ("maximum_progress_m", "median_progress_m")
        )
    best_distance = metadata.get("best_distance_episode")
    if isinstance(best_distance, Mapping):
        candidates.append(finite_float(best_distance.get("distance_m")))
    continuation = metadata.get("continuation")
    if isinstance(continuation, Mapping):
        candidates.extend(
            finite_float(continuation.get(key))
            for key in (
                "source_verified_evaluation_maximum_progress_m",
                "source_verified_evaluation_progress_m",
            )
        )
    discovery = metadata.get("discovery")
    if isinstance(discovery, Mapping):
        candidates.append(
            finite_float(discovery.get("source_frontier_evidence_distance_m"))
        )
        run_state = discovery.get("run_state")
        if isinstance(run_state, Mapping):
            candidates.append(finite_float(run_state.get("best_distance_m")))
    return max((value for value in candidates if value >= 0.0), default=0.0)


def load_verified_frontier_evidence(
    policy_path: Path,
    evidence_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Load external distance evidence only when it binds to the exact policy."""
    evidence_file = Path(evidence_path)
    if not evidence_file.is_file():
        return {}, []
    evidence = read_json_mapping(evidence_file)
    configured_policy = evidence.get("policy_path")
    configured_hash = str(evidence.get("policy_sha256") or "").casefold()
    if not configured_policy or not configured_hash:
        return {}, [
            f"frontier evidence is missing policy_path or policy_sha256: {evidence_file}"
        ]
    configured_path = Path(str(configured_policy))
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    source_path = Path(policy_path)
    if configured_path.resolve() != source_path.resolve():
        return {}, []
    try:
        with source_path.open("rb") as policy_file:
            actual_hash = hashlib.file_digest(policy_file, "sha256").hexdigest()
    except OSError as exc:
        return {}, [f"could not hash frontier evidence policy: {exc}"]
    if actual_hash.casefold() != configured_hash:
        return {}, [
            "frontier evidence policy hash mismatch: expected "
            f"{configured_hash}, found {actual_hash.casefold()}"
        ]
    distance_m = finite_float(evidence.get("distance_m"), -1.0)
    if distance_m <= 0.0:
        return {}, ["frontier evidence distance_m must be positive"]
    return dict(evidence), []


def frontier_continuation_source_errors(
    policy_path: Path,
    *,
    track: str,
    start_stage: str,
    external_evidence: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate a reward-only frontier without requiring robust-evaluation status."""
    path = Path(policy_path)
    if not path.is_file():
        return [f"frontier checkpoint does not exist: {path}"]
    metadata = read_policy_metadata(path)
    errors = [
        f"frontier checkpoint contract mismatch: {key}"
        for key in policy_contract_mismatches(
            metadata,
            allow_legacy_action_contract=True,
        )
    ]
    if str(metadata.get("learning_source") or "") != "reward_only":
        errors.append("frontier checkpoint must be trained from reward only")
    source_track = str(metadata.get("track") or "")
    if source_track and source_track != str(track):
        errors.append(
            "frontier checkpoint track mismatch: "
            f"expected {track}, found {source_track}"
        )
    stage_index = curriculum_stage_index(start_stage)
    required_progress_m = (
        0.0
        if stage_index == 0
        else CURRICULUM[stage_index - 1].distance_target_m
    )
    evidence_distance_m = frontier_evidence_distance_m(
        metadata,
        external_evidence,
    )
    if evidence_distance_m < required_progress_m:
        errors.append(
            f"frontier checkpoint has {evidence_distance_m:.1f}m evidence, but "
            f"stage {start_stage} requires {required_progress_m:.1f}m"
        )
    return errors


def continuation_source_uses_legacy_action_contract(
    args: argparse.Namespace,
) -> bool:
    if not bool(getattr(args, "continuation", False)):
        return False
    return policy_uses_legacy_action_contract(
        read_policy_metadata(continuation_source_path(args))
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
    if lean_frontier_training_enabled(args):
        # A transferred actor starts collecting immediately, but the fresh
        # critic must not optimize against a tiny, highly correlated buffer.
        return max(0, int(args.replay_refill_steps))
    return (
        0
        if continuation_training_protocol_enabled(args)
        else max(0, int(args.learning_starts))
    )


def gradient_update_start_timestep(args: argparse.Namespace) -> int:
    if continuation_training_protocol_enabled(args):
        return max(0, int(args.replay_refill_steps))
    return max(0, int(args.learning_starts))


def actor_update_start_timestep(args: argparse.Namespace) -> int:
    gradient_start = gradient_update_start_timestep(args)
    if not continuation_training_protocol_enabled(args):
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


def training_action_noise_seed(base_seed: int, generation: int) -> int:
    """Derive a reproducible stream seed without using NumPy's global RNG."""
    seed = int(base_seed)
    index = int(generation)
    if not 0 <= seed <= MAX_REPRODUCIBLE_SEED:
        raise ValueError(
            f"base_seed must be between 0 and {MAX_REPRODUCIBLE_SEED}"
        )
    if index < 0:
        raise ValueError("generation must be non-negative")
    return int(
        (seed + TRAINING_ACTION_NOISE_SEED_STRIDE * (index + 1))
        & MAX_REPRODUCIBLE_SEED
    )


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
    reference_upper_quartile_distance_m: float
    reference_maximum_distance_m: float
    candidate_median_distance_m: float
    candidate_minimum_distance_m: float
    candidate_upper_quartile_distance_m: float
    candidate_maximum_distance_m: float
    aggregate_median_gain_m: float
    minimum_gain_m: float
    upper_quartile_gain_m: float
    maximum_gain_m: float
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
    reference_lap_completions: int
    candidate_lap_completions: int
    new_progress_milestones_m: tuple[float, ...]
    acceptance_reasons: tuple[str, ...]
    safety_failures: tuple[str, ...]


def safe_actor_probe_decision(
    reference_distances_m: list[float],
    candidate_distances_m: list[float],
    *,
    seed_count: int,
    trials_per_seed: int,
    reference_order_positions: list[int] | tuple[int, ...] | None = None,
    candidate_order_positions: list[int] | tuple[int, ...] | None = None,
    reference_laps_completed: list[int] | tuple[int, ...] | None = None,
    candidate_laps_completed: list[int] | tuple[int, ...] | None = None,
    minimum_improvement_m: float,
    maximum_paired_regression_m: float,
    median_regression_tolerance_m: float = (
        DEFAULT_SAFE_ACTOR_MEDIAN_REGRESSION_TOLERANCE_M
    ),
    minimum_regression_tolerance_m: float = (
        DEFAULT_SAFE_ACTOR_MINIMUM_REGRESSION_TOLERANCE_M
    ),
    progress_milestones_m: tuple[float, ...] = (
        SAFE_ACTOR_PROGRESS_MILESTONES_M
    ),
) -> SafeActorProbeComparison:
    """Compare working-policy candidates using progress and safety guards."""
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
    median_tolerance = float(median_regression_tolerance_m)
    minimum_tolerance = float(minimum_regression_tolerance_m)
    if not math.isfinite(required_gain) or required_gain < 0.0:
        raise ValueError("minimum improvement must be finite and non-negative")
    if not math.isfinite(regression_limit) or regression_limit < 0.0:
        raise ValueError("paired regression limit must be finite and non-negative")
    if not math.isfinite(median_tolerance) or median_tolerance < 0.0:
        raise ValueError("median regression tolerance must be finite and non-negative")
    if not math.isfinite(minimum_tolerance) or minimum_tolerance < 0.0:
        raise ValueError("minimum regression tolerance must be finite and non-negative")
    milestones = tuple(sorted({float(value) for value in progress_milestones_m}))
    if any(not math.isfinite(value) or value < 0.0 for value in milestones):
        raise ValueError("progress milestones must be finite and non-negative")
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
    expected_trials = len(reference_distances_m)
    if reference_laps_completed is None:
        reference_laps_completed = (0,) * expected_trials
    if candidate_laps_completed is None:
        candidate_laps_completed = (0,) * expected_trials
    if (
        len(reference_laps_completed) != expected_trials
        or len(candidate_laps_completed) != expected_trials
    ):
        raise ValueError("lap-completion sets must match the probe distances")
    reference_laps = np.asarray(reference_laps_completed, dtype=float)
    candidate_laps = np.asarray(candidate_laps_completed, dtype=float)
    if (
        not np.all(np.isfinite(reference_laps))
        or not np.all(np.isfinite(candidate_laps))
        or np.any(reference_laps < 0.0)
        or np.any(candidate_laps < 0.0)
    ):
        raise ValueError("lap completions must be finite and non-negative")
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
    reference_upper_quartile = float(
        np.percentile(reference_seed_medians, 75.0)
    )
    candidate_upper_quartile = float(
        np.percentile(candidate_seed_medians, 75.0)
    )
    reference_maximum = float(np.max(reference))
    candidate_maximum = float(np.max(candidate))
    aggregate_median_gain = candidate_median - reference_median
    minimum_gain = candidate_minimum - reference_minimum
    upper_quartile_gain = candidate_upper_quartile - reference_upper_quartile
    maximum_gain = candidate_maximum - reference_maximum
    paired_median_delta = float(np.median(paired_deltas))
    worst_paired_delta = float(np.min(paired_deltas))
    paired_regressions = int(np.count_nonzero(paired_deltas < 0.0))
    worst_order_position_delta = float(np.min(position_deltas))
    order_position_regressions = int(
        np.count_nonzero(position_deltas < 0.0)
    )
    reference_lap_completions = int(np.sum(reference_laps))
    candidate_lap_completions = int(np.sum(candidate_laps))
    new_milestones = tuple(
        milestone
        for milestone in milestones
        if reference_maximum < milestone <= candidate_maximum
    )
    progress_reasons: list[str] = []
    if aggregate_median_gain >= required_gain:
        progress_reasons.append("median_distance")
    if upper_quartile_gain >= required_gain:
        progress_reasons.append("upper_quartile_distance")
    if maximum_gain >= required_gain:
        progress_reasons.append("furthest_distance")
    if new_milestones:
        progress_reasons.append("new_sector_milestone")
    if candidate_lap_completions > reference_lap_completions:
        progress_reasons.append("lap_completion")

    safety_failures: list[str] = []
    if aggregate_median_gain < -median_tolerance:
        safety_failures.append("median_regression")
    if minimum_gain < -minimum_tolerance:
        safety_failures.append("minimum_regression")
    if worst_paired_delta < -regression_limit:
        safety_failures.append("paired_seed_regression")
    if worst_order_position_delta < -regression_limit:
        safety_failures.append("order_position_regression")
    accepted = bool(progress_reasons) and not safety_failures
    return SafeActorProbeComparison(
        decision="accepted" if accepted else "rolled_back",
        reference_median_distance_m=reference_median,
        reference_minimum_distance_m=reference_minimum,
        reference_upper_quartile_distance_m=reference_upper_quartile,
        reference_maximum_distance_m=reference_maximum,
        candidate_median_distance_m=candidate_median,
        candidate_minimum_distance_m=candidate_minimum,
        candidate_upper_quartile_distance_m=candidate_upper_quartile,
        candidate_maximum_distance_m=candidate_maximum,
        aggregate_median_gain_m=aggregate_median_gain,
        minimum_gain_m=minimum_gain,
        upper_quartile_gain_m=upper_quartile_gain,
        maximum_gain_m=maximum_gain,
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
        reference_lap_completions=reference_lap_completions,
        candidate_lap_completions=candidate_lap_completions,
        new_progress_milestones_m=new_milestones,
        acceptance_reasons=tuple(progress_reasons),
        safety_failures=tuple(safety_failures),
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
    newly_achieved_sector_milestones_m: tuple[float, ...] = (),
    completed_lap: float | None = None,
    stage_success: bool = False,
    stuck: bool = False,
    terminal_failure: bool | None = None,
) -> dict[str, float]:
    # TORCS reports speedX in km/h and trackPos as a normalized lateral
    # coordinate where +/-1 is the track edge. Keep this reward deliberately
    # independent of curriculum stage and actuator choices so the critic sees
    # one stable objective throughout continuation training.
    _ = (
        action,
        previous_telemetry,
        episode_distance_m,
        previous_furthest_distance_m,
        newly_achieved_sector_milestones_m,
        stage,
    )
    sensors = track_sensors(telemetry)
    min_sensor = min(sensors) if sensors else 200.0
    speed_mps = finite_float(telemetry.get("speedX")) / 3.6
    angle = finite_float(telemetry.get("angle"))
    track_position = finite_float(telemetry.get("trackPos"))
    damage = finite_float(telemetry.get("damage"), previous_damage)
    off_track = min_sensor < 0.0 or abs(track_position) > SOFT_TRACK_BOUNDARY
    crashed = damage > previous_damage
    backwards = math.cos(angle) < 0.0
    if terminal_failure is None:
        terminal_failure = off_track or crashed or backwards or stuck
    physical_failure = bool(
        terminal_failure and completed_lap is None and not stage_success
    )
    if physical_failure:
        return {
            "forward_speed": 0.0,
            "track_edge_penalty": 0.0,
            "terminal_failure": -25.0,
            "total_unclipped": -25.0,
            "total": -25.0,
        }

    forward_speed = speed_mps * max(0.0, math.cos(angle)) * 0.10
    normalized_offset = abs(track_position)
    track_edge_penalty = 0.0
    if normalized_offset > 0.8:
        track_edge_penalty = -1.5 * (
            (normalized_offset - 0.8) / 0.2
        ) ** 2
    unclipped_total = forward_speed + track_edge_penalty
    return {
        "forward_speed": float(forward_speed),
        "track_edge_penalty": float(track_edge_penalty),
        "terminal_failure": 0.0,
        "total_unclipped": float(unclipped_total),
        "total": float(clamp(unclipped_total, -10.0, 5.0)),
    }


def calculate_td3_reward(
    telemetry: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    previous_telemetry: Mapping[str, Any] | None = None,
    previous_damage: float = 0.0,
    stage: CurriculumStage | None = None,
    episode_distance_m: float = 0.0,
    previous_furthest_distance_m: float = 0.0,
    newly_achieved_sector_milestones_m: tuple[float, ...] = (),
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
        newly_achieved_sector_milestones_m=(
            newly_achieved_sector_milestones_m
        ),
        completed_lap=completed_lap,
        stage_success=stage_success,
        stuck=stuck,
        terminal_failure=terminal_failure,
    )["total"]


def make_training_metadata(args: argparse.Namespace) -> dict[str, Any]:
    source_path = continuation_source_path(args)
    source_metadata = read_policy_metadata(source_path)
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
    source_maximum_progress_m = (
        finite_float(source_evaluation.get("maximum_progress_m"))
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
        "training_objective": (
            "frontier_distance_discovery"
            if args.discovery_mode
            else "completed_lap_frontier_robustification"
            if args.robustify_frontier
            else "robust_policy_improvement"
        ),
        "discovery": {
            "enabled": bool(args.discovery_mode),
            "source_frontier_evidence_distance_m": (
                frontier_evidence_distance_m(
                    source_metadata,
                    args.frontier_verified_evidence,
                )
                if args.discovery_mode
                else None
            ),
            "source_frontier_evidence": (
                copy.deepcopy(args.frontier_verified_evidence)
                if args.discovery_mode
                else None
            ),
            "frontier_evidence_path": str(args.frontier_evidence_path),
            "frontier_model_path": str(args.frontier_model_path),
            "frontier_checkpoint_dir": str(args.frontier_checkpoint_dir),
            "milestones_m": list(args.frontier_milestones_m),
            "minimum_record_improvement_m": args.frontier_min_improvement_m,
            "major_breakthrough_improvement_m": (
                args.frontier_major_breakthrough_m
            ),
            "routine_internal_robustness_probes": not args.discovery_mode,
            "safe_actor_rollback": False,
            "trajectory_gate": {
                "protocol": FRONTIER_CONTINUATION_PROTOCOL,
                "enabled": bool(
                    args.discovery_mode
                    and not lean_frontier_training_enabled(args)
                ),
                "qualification_enabled": bool(
                    args.frontier_qualification_enabled
                ),
                "qualification_distance_m": (
                    args.frontier_qualification_distance_m
                ),
                "qualification_attempts": args.frontier_qualification_attempts,
                "qualification_result": None,
                "exploration_start_distance_m": (
                    args.frontier_exploration_start_distance_m
                ),
                "source_actor_frozen_during_qualification": True,
                "replay_refill_action_noise_sigma": (
                    args.action_noise_sigma
                    if lean_frontier_training_enabled(args)
                    else 0.0
                ),
                "critic_warmup_action_noise_sigma": (
                    args.action_noise_sigma
                    if lean_frontier_training_enabled(args)
                    else 0.0
                ),
                "critic_may_learn_before_frontier": True,
                "actor_updates_before_frontier": lean_frontier_training_enabled(args),
                "exploration_before_frontier": lean_frontier_training_enabled(args),
                "optimisation_boundary": (
                    "environment_step"
                    if lean_frontier_training_enabled(args)
                    else "completed_episode"
                ),
                "frozen_policy_health_fraction": (
                    DEFAULT_FROZEN_POLICY_HEALTH_FRACTION
                ),
                "frozen_policy_health_failures": (
                    DEFAULT_FROZEN_POLICY_HEALTH_FAILURES
                ),
                "learning_source": "reward_only",
                "behaviour_cloning": False,
                "trajectory_imitation": False,
            },
            "collapse_recovery": {
                "enabled": bool(
                    args.discovery_mode
                    and not lean_frontier_training_enabled(args)
                ),
                "consecutive_failure_episodes": (
                    args.discovery_collapse_episodes
                ),
                "failure_distance_m": args.discovery_collapse_distance_m,
                "frontier_preservation_fraction": (
                    args.discovery_frontier_preservation_fraction
                ),
                "effective_failure_distance_m": max(
                    args.discovery_collapse_distance_m,
                    args.frontier_exploration_start_distance_m
                    * args.discovery_frontier_preservation_fraction,
                ),
                "base_actor_cooldown_steps": (
                    args.discovery_recovery_cooldown_steps
                ),
                "maximum_actor_cooldown_steps": (
                    args.discovery_max_recovery_cooldown_steps
                ),
                "actor_learning_rate_backoff": (
                    args.discovery_actor_lr_backoff
                ),
                "minimum_actor_learning_rate_scale": (
                    args.discovery_min_actor_lr_scale
                ),
                "maximum_effective_policy_delay": (
                    args.discovery_max_policy_delay
                ),
                "restore_scope": "actor_target_and_actor_optimizer_only",
                "critic_and_replay_preserved": True,
            },
            "legacy_exploration_noise": (
                "correlated_steer_and_antisymmetric_longitudinal"
                if training_uses_legacy_action_contract(args)
                else None
            ),
            "pre_actor_exploration_noise": (
                "none_before_frontier_gate"
                if args.discovery_mode
                else "seeded_correlated_steering_only"
                if args.continuation
                else None
            ),
            "lightweight_screen": (
                "completed_policy_controlled_training_episode_with_forward_progress"
            ),
            "strict_evaluation_trigger": (
                "new_distance_milestone_configured_frontier_gain_or_completed_lap"
            ),
            "strict_evaluation_timing": "after_training_torcs_shutdown",
            "robust_champion_is_independent": True,
            "frontier_is_not_a_robustness_claim": True,
        },
        "robustify_frontier": bool(args.robustify_frontier),
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
            "speed_input_unit": "km/h converted to m/s before reward",
            "track_position_input": "normalized TORCS trackPos; +/-1 is edge",
            "forward_speed_scale": 0.10,
            "soft_track_edge_start": 0.8,
            "nonterminal_clip": [-10.0, 5.0],
            "physical_failure_reward": -25.0,
            "curriculum_changes_reward": False,
            "sector_milestones_affect_reward": False,
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
        "sector_progression_milestones_m": list(
            SECTOR_PROGRESS_MILESTONES_M
        ),
        "curriculum_start_stage": args.start_stage,
        "continuation": {
            "enabled": args.continuation,
            "training_protocol_enabled": (
                continuation_training_protocol_enabled(args)
            ),
            "source_policy_path": (
                str(source_path) if source_path is not None else None
            ),
            "exact_resume": bool(args.resume_exact_from),
            "exact_resume_budget_extension": copy.deepcopy(
                getattr(args, "exact_resume_budget_extension", None)
            ),
            "source_observation_version": source_metadata.get("observation_version"),
            "source_action_version": source_metadata.get("action_version"),
            "source_action_shape": source_metadata.get("action_shape"),
            "source_verified_evaluation_progress_m": source_progress_m,
            "source_verified_evaluation_minimum_progress_m": (
                source_minimum_progress_m
            ),
            "source_verified_evaluation_maximum_progress_m": (
                source_maximum_progress_m
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
            "fresh_replay_buffer": bool(
                args.continuation and not args.resume_exact_from
            ),
            "actor_only_transfer": bool(
                args.continuation and not args.resume_exact_from
            ),
            "fresh_critic": bool(
                args.continuation and not args.resume_exact_from
            ),
            "fresh_critic_target": bool(
                args.continuation and not args.resume_exact_from
            ),
            "fresh_optimizers": bool(
                args.continuation and not args.resume_exact_from
            ),
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
            "robust_champion": {
                "role": "immutable_source_policy",
                "path": str(source_path) if args.continuation else None,
                "may_be_replaced_by_internal_probe": False,
            },
            "working_policy": {
                "role": "last_locally_accepted_continuation_actor",
                "path": str(args.model_path),
                "may_accept_bounded_temporary_regressions": True,
            },
            "qualified_candidate_validation": {
                "automatic": True,
                "completed_lap_freezes_actor_immediately": True,
                "breakthrough_freezes_actor_immediately": True,
                "completed_candidate_blocks_receive_seeded_screen": True,
                "noisy_training_evidence_is_advisory": True,
                "uses_rotating_internal_training_seeds": True,
                "uses_external_evaluation_seeds": False,
            },
            "constraint": "action_space_trust_region_to_working_policy",
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
            "breakthrough_improvement_m": args.breakthrough_improvement_m,
            "breakthrough_milestones_m": list(
                SAFE_ACTOR_PROGRESS_MILESTONES_M
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
            "reference_policy": "working_policy_on_same_challenge_seeds",
            "acceptance_metric": (
                "progress_in_median_upper_quartile_furthest_milestone_or_lap_"
                "with_bounded_seed_and_position_regressions"
            ),
            "minimum_improvement_m": args.safe_actor_min_improvement_m,
            "median_regression_tolerance_m": (
                args.safe_actor_median_regression_tolerance_m
            ),
            "minimum_regression_tolerance_m": (
                args.safe_actor_minimum_regression_tolerance_m
            ),
            "progress_milestones_m": list(SAFE_ACTOR_PROGRESS_MILESTONES_M),
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
            "saved_checkpoint_actor": "working_policy",
            "held_out_evaluation_seeds_used_during_training": False,
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
            "end_of_budget_probe_drain": (
                "finish_pending_safe_actor_comparison_only_when_each_next_"
                "trial_fits_earned_probe_capacity"
            ),
            "training_target_episode_boundary": "truncate_without_curriculum_credit",
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
        "best_robust_evaluation_model_path": str(
            args.best_robust_evaluation_model_path
        ),
        "best_completed_lap_model_path": str(
            args.best_completed_lap_model_path
        ),
        "frontier_model_path": str(args.frontier_model_path),
        "frontier_checkpoint_dir": str(args.frontier_checkpoint_dir),
        "lap_checkpoint_dir": str(args.lap_checkpoint_dir),
        "automatic_evaluation": {
            "enabled": bool(args.end_of_run_evaluation),
            "timing": "after_training_and_after_training_torcs_shutdown",
            "external_evaluation_during_candidate_blocks": False,
            "output_dir": str(args.automatic_evaluation_output_dir),
            "robust_champion_path": str(
                args.best_robust_evaluation_model_path
            ),
            "completed_lap_champion_path": str(
                args.best_completed_lap_model_path
            ),
            "promotion": "atomic_model_and_metadata_staging",
        },
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
            "optimizer_protocol": (
                "stable_frontier_td3_v1"
                if lean_frontier_training_enabled(args)
                else "legacy_phased_td3"
            ),
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
            "critic_loss": (
                "smooth_l1_beta_1.0"
                if lean_frontier_training_enabled(args)
                else "smooth_l1"
            ),
            "critic_gradient_clip_norm": LEAN_FRONTIER_GRADIENT_NORM,
            "learning_rate_decay_starts_at_timestep": (
                gradient_update_start_timestep(args)
            ),
            "actor_learning_rate": args.actor_learning_rate,
            "actor_learning_rate_final": args.actor_learning_rate_final,
            "actor_learning_rate_schedule": (
                "frozen_then_constant"
                if lean_frontier_training_enabled(args)
                else "frozen_then_linear_ramp_and_decay"
            ),
            "actor_gradient_clip_norm": args.safe_actor_gradient_norm,
            "critic_warmup_steps": args.critic_warmup_steps,
            "critic_warmup_unit": (
                "successful_critic_optimizer_updates"
                if lean_frontier_training_enabled(args)
                else "environment_steps"
            ),
            "actor_unfreeze_steps": args.actor_unfreeze_steps,
            "safe_actor_improvement": args.safe_actor_improvement,
            "train_freq": args.train_freq,
            "train_freq_unit": (
                "step" if lean_frontier_training_enabled(args) else "episode"
            ),
            "gradient_steps": args.gradient_steps,
            "live_episode_gradient_updates": lean_frontier_training_enabled(args),
            "episode_atomic_policy": not lean_frontier_training_enabled(args),
            "offline_optimizer_torcs_suspension": bool(
                args.continuation and not lean_frontier_training_enabled(args)
            ),
            "time_limit_bootstrapping": (
                "bootstrap_on_truncation_stop_on_physical_failure"
            ),
            "policy_delay": args.policy_delay,
            "target_policy_noise": args.target_policy_noise,
            "target_noise_clip": args.target_noise_clip,
            "action_noise_initial_sigma": args.action_noise_sigma,
            "action_noise_final_sigma": args.action_noise_final_sigma,
            "action_noise_decay_steps": args.action_noise_decay_steps,
            "action_noise_rng": "isolated_seeded_numpy_generator",
            "action_noise_seed_schedule": "base_seed_plus_golden_ratio_stride",
            "action_noise_independent_from_replay_sampling": True,
            "frontier_conditioned_exploration": bool(
                args.discovery_mode and not lean_frontier_training_enabled(args)
            ),
            "frontier_exploration_start_distance_m": (
                args.frontier_exploration_start_distance_m
                if args.discovery_mode and not lean_frontier_training_enabled(args)
                else None
            ),
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
            self.discovery_frontier_gate_open = False
            self.discovery_frontier_gate_distance_m = 0.0
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
            self.episode_sector_milestones_awarded_m: set[float] = set()
            self.sector_milestone_history: list[dict[str, Any]] = []
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.consecutive_off_track_steps = 0
            self._step_file = None
            self._step_writer = None
            self._open_step_telemetry()

        @property
        def current_stage(self) -> CurriculumStage:
            return CURRICULUM[self.stage_index]

        def training_state_dict(self) -> dict[str, Any]:
            return {
                "stage_id": self.current_stage.stage_id,
                "stage_success_streak": self.stage_success_streak,
                "stage_success_totals": dict(self.stage_success_totals),
                "stage_success_windows": copy.deepcopy(
                    self.stage_success_windows
                ),
                "episodes_started": self.episodes_started,
                "total_steps": self.total_steps,
                "learning_starts": self.learning_starts,
                "pre_update_training_phase": self.pre_update_training_phase,
                "safe_actor_screening_max_episode_steps": (
                    self.safe_actor_screening_max_episode_steps
                ),
                "sector_milestone_history": copy.deepcopy(
                    self.sector_milestone_history
                ),
            }

        def restore_training_state(self, state: Mapping[str, Any]) -> None:
            stage_id = str(state.get("stage_id") or self.current_stage.stage_id)
            self.stage_index = curriculum_stage_index(stage_id)
            self.stage_success_streak = max(
                0,
                int(finite_float(state.get("stage_success_streak"))),
            )
            totals = state.get("stage_success_totals")
            if isinstance(totals, Mapping):
                self.stage_success_totals = {
                    stage.stage_id: max(
                        0,
                        int(finite_float(totals.get(stage.stage_id))),
                    )
                    for stage in CURRICULUM
                }
            windows = state.get("stage_success_windows")
            if isinstance(windows, Mapping):
                self.stage_success_windows = {
                    stage.stage_id: [
                        bool(value)
                        for value in list(windows.get(stage.stage_id, []))[
                            -DEFAULT_STAGE_SUCCESS_WINDOW_SIZE:
                        ]
                    ]
                    for stage in CURRICULUM
                }
            self.episodes_started = max(
                0,
                int(finite_float(state.get("episodes_started"))),
            )
            self.total_steps = max(
                0,
                int(finite_float(state.get("total_steps"))),
            )
            self.learning_starts = max(
                0,
                int(finite_float(state.get("learning_starts"), self.learning_starts)),
            )
            self.pre_update_training_phase = str(
                state.get("pre_update_training_phase")
                or self.pre_update_training_phase
            )
            self.safe_actor_screening_max_episode_steps = max(
                1,
                int(
                    finite_float(
                        state.get("safe_actor_screening_max_episode_steps"),
                        self.safe_actor_screening_max_episode_steps,
                    )
                ),
            )
            history = state.get("sector_milestone_history")
            self.sector_milestone_history = (
                [dict(item) for item in history if isinstance(item, Mapping)]
                if isinstance(history, (list, tuple))
                else []
            )
            self.episode_sector_milestones_awarded_m = set()

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
            discovery_frontier_gate_open: bool = False,
            discovery_frontier_gate_distance_m: float = 0.0,
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
            self.discovery_frontier_gate_open = bool(
                discovery_frontier_gate_open
            )
            self.discovery_frontier_gate_distance_m = max(
                0.0,
                finite_float(discovery_frontier_gate_distance_m),
            )

        def accept_safe_actor_probe(
            self,
            summary: dict[str, Any],
            *,
            median_distance_m: float,
        ) -> None:
            """Commit one accepted probe result to the curriculum state."""
            if str(summary.get("stage_id")) != self.current_stage.stage_id:
                return
            completed_lap = bool(
                int(summary.get("laps_completed") or 0) > 0
                or int(
                    summary.get("safe_actor_candidate_lap_completions") or 0
                )
                > 0
            )
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
            self.episode_sector_milestones_awarded_m = set()
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
            new_sector_milestones = sector_milestones_crossed(
                previous_furthest,
                self.episode_furthest_distance,
                self.episode_sector_milestones_awarded_m,
            )
            if new_sector_milestones:
                self.episode_sector_milestones_awarded_m.update(
                    new_sector_milestones
                )
                for milestone in new_sector_milestones:
                    self.sector_milestone_history.append(
                        {
                            "milestone_m": milestone,
                            "episode": self.episodes_started,
                            "global_step": self.total_steps + 1,
                            "stage_id": self.current_stage.stage_id,
                            "deterministic_probe": self.deterministic_probe,
                        }
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
            physical_failure = bool(
                terminal_failure
                and completed_lap is None
                and not stage_success
            )
            reward_components = calculate_td3_reward_components(
                telemetry,
                action,
                previous_telemetry=self.previous_telemetry,
                previous_damage=self.previous_damage,
                stage=self.current_stage,
                episode_distance_m=episode_distance_m,
                previous_furthest_distance_m=previous_furthest,
                newly_achieved_sector_milestones_m=new_sector_milestones,
                completed_lap=completed_lap,
                stage_success=stage_success,
                stuck=stuck,
                terminal_failure=physical_failure,
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
            training_budget_boundary_reached = bool(
                self.training_budget_state is not None
                and not self.deterministic_probe
                and self.training_budget_state.reaches_training_boundary()
            )
            truncated = bool(
                self.steps >= episode_step_limit
                or training_budget_boundary_reached
            )
            policy_controlled = episode_is_policy_controlled(
                self.episode_start_timestep,
                self.learning_starts,
            )
            curriculum_eligible = episode_is_curriculum_eligible(
                policy_controlled=policy_controlled,
                deterministic_probe=self.deterministic_probe,
            ) and not (
                self.safe_actor_candidate_probe
                or training_budget_boundary_reached
            )
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
                elif training_budget_boundary_reached:
                    reason = "training_budget_reached"
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
                "discovery_frontier_gate_open": (
                    self.discovery_frontier_gate_open
                ),
                "discovery_frontier_gate_distance_m": (
                    self.discovery_frontier_gate_distance_m
                ),
                "stage_id": self.current_stage.stage_id,
                "stage_name": self.current_stage.name,
                "distance_m": self.episode_furthest_distance,
                "furthest_distance_m": self.episode_furthest_distance,
                "sector_milestones_reached_m": sorted(
                    self.episode_sector_milestones_awarded_m
                ),
                "lap_completion_fraction": calculate_lap_completion_fraction(
                    self.episode_furthest_distance
                ),
                "training_budget_boundary_reached": (
                    training_budget_boundary_reached
                ),
                "physical_failure": physical_failure,
                "TimeLimit.truncated": bool(truncated and not terminated),
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
                "discovery_frontier_gate_open": (
                    self.discovery_frontier_gate_open
                ),
                "discovery_frontier_gate_distance_m": (
                    self.discovery_frontier_gate_distance_m
                ),
                "stage_id": self.current_stage.stage_id,
                "stage_name": self.current_stage.name,
                "steps": self.steps,
                "reward": self.episode_reward,
                "distance_m": distance_m,
                "furthest_distance_m": distance_m,
                "sector_milestones_reached_m": sorted(
                    self.episode_sector_milestones_awarded_m
                ),
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
                        # Process ownership belongs to TorcsRunner. Passing the
                        # relaunch request into gym_torcs invokes its legacy
                        # shell-based path as well as the managed restart above.
                        return self.runner.env.reset(relaunch=False)
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
            last_error: Exception | None = None
            for attempt in range(self.reset_retries + 1):
                try:
                    if not self.manual_start:
                        self.runner.launch()
                    self.runner.connect()
                    self.runner.load_track(self.track_name)
                    return
                except Exception as exc:
                    last_error = exc
                    self._close_torcs_env()
                    if not self.manual_start:
                        self.runner.shutdown()
                    if attempt >= self.reset_retries:
                        break
                    print(
                        "TORCS connection failed; restarting simulator "
                        f"[{attempt + 2}/{self.reset_retries + 1}]..."
                    )
                    time.sleep(1.0)
            raise RuntimeError(
                "TORCS launch/connect failed after "
                f"{self.reset_retries + 1} attempts"
            ) from last_error

        def _close_torcs_env(self) -> None:
            torcs_env = self.runner.env
            if torcs_env is None:
                return
            self.runner.env = None
            client = getattr(torcs_env, "client", None)
            shutdown_client = getattr(client, "shutdown", None)
            if callable(shutdown_client):
                with contextlib.suppress(Exception):
                    shutdown_client()
            with contextlib.suppress(Exception):
                torcs_env.end()

        def suspend_torcs_for_offline_updates(self) -> None:
            """Release the live simulator before an episode-boundary replay batch."""
            self._close_torcs_env()
            if not self.manual_start:
                self.runner.shutdown()

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
                    reward_components.get("forward_speed", 0.0),
                    reward_components.get("track_edge_penalty", 0.0),
                    reward_components.get("terminal_failure", 0.0),
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
        candidate = float(distance_m)
        if self.best_distance_m is None or candidate > self.best_distance_m:
            self.best_distance_m = candidate

    def record_best_reward(self, reward: float) -> None:
        candidate = float(reward)
        if self.best_reward is None or candidate > self.best_reward:
            self.best_reward = candidate


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

    @property
    def training_budget_reached(self) -> bool:
        return self.training_steps >= self.target_training_steps

    def reaches_training_boundary(self, additional_steps: int = 1) -> bool:
        """Return whether a non-probe transition reaches the training target."""
        return (
            self.training_steps + max(1, int(additional_steps))
            >= self.target_training_steps
        )

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
        if self.training_budget_reached:
            self.stop_reason = "training_budget_reached"
        else:
            self.stop_reason = None

    def should_stop(self) -> bool:
        return self.stop_reason is not None

    def restore(
        self,
        state: Mapping[str, Any],
        *,
        allow_budget_extension: bool = False,
    ) -> None:
        saved_target = max(
            1,
            int(finite_float(state.get("target_training_steps"), 1.0)),
        )
        saved_maximum_probe_steps = max(
            0,
            int(finite_float(state.get("maximum_probe_steps"))),
        )
        saved_maximum_probe_fraction = finite_float(
            state.get("maximum_probe_fraction"),
            DEFAULT_MAX_PROBE_FRACTION,
        )
        target_matches = saved_target == self.target_training_steps
        probe_ceiling_matches = (
            saved_maximum_probe_steps == self.maximum_probe_steps
        )
        fraction_matches = math.isclose(
            saved_maximum_probe_fraction,
            self.maximum_probe_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        valid_extension = bool(
            allow_budget_extension
            and self.target_training_steps >= saved_target
            and self.maximum_probe_steps >= saved_maximum_probe_steps
            and fraction_matches
        )
        if not (
            fraction_matches
            and ((target_matches and probe_ceiling_matches) or valid_extension)
        ):
            raise ValueError(
                "exact-resume budget mismatch: current target/probe/fraction "
                f"{self.target_training_steps}/{self.maximum_probe_steps}/"
                f"{self.maximum_probe_fraction:.6f}, saved "
                f"{saved_target}/{saved_maximum_probe_steps}/"
                f"{saved_maximum_probe_fraction:.6f}"
            )
        self.training_steps = max(
            0,
            int(finite_float(state.get("training_steps"))),
        )
        self.probe_steps = max(
            0,
            int(finite_float(state.get("probe_steps"))),
        )
        self.environment_steps = max(
            self.training_steps + self.probe_steps,
            int(finite_float(state.get("environment_steps"))),
        )
        self.deferred_probe_requests = max(
            0,
            int(finite_float(state.get("deferred_probe_requests"))),
        )
        self._update_stop_reason()

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


def resolve_exact_resume_budget(
    state: Mapping[str, Any],
    *,
    additional_training_steps: int,
    requested_maximum_probe_steps: int | None,
) -> dict[str, int | float]:
    """Extend a checkpoint budget monotonically while preserving its ratio cap."""
    extension = int(additional_training_steps)
    if extension < 0:
        raise ValueError("additional training steps cannot be negative")
    saved_target = max(
        1,
        int(finite_float(state.get("target_training_steps"), 1.0)),
    )
    saved_probe_ceiling = max(
        0,
        int(finite_float(state.get("maximum_probe_steps"))),
    )
    saved_probe_fraction = finite_float(
        state.get("maximum_probe_fraction"),
        DEFAULT_MAX_PROBE_FRACTION,
    )
    if not 0.0 < saved_probe_fraction <= DEFAULT_MAX_PROBE_FRACTION:
        raise ValueError("saved exact-resume probe fraction is invalid")

    added_probe_capacity = int(
        math.floor(
            extension
            * saved_probe_fraction
            / (1.0 - saved_probe_fraction)
        )
    )
    if requested_maximum_probe_steps is None:
        probe_ceiling = saved_probe_ceiling + added_probe_capacity
    else:
        probe_ceiling = int(requested_maximum_probe_steps)
        if probe_ceiling < saved_probe_ceiling:
            raise ValueError(
                "exact-resume probe ceiling cannot be lower than the saved "
                f"ceiling ({saved_probe_ceiling:,})"
            )

    return {
        "saved_training_target_steps": saved_target,
        "additional_training_steps": extension,
        "target_training_steps": saved_target + extension,
        "saved_maximum_probe_steps": saved_probe_ceiling,
        "added_probe_capacity_steps": added_probe_capacity,
        "maximum_probe_steps": probe_ceiling,
        "maximum_probe_fraction": saved_probe_fraction,
    }


def training_breakthrough_reasons(
    distance_m: float,
    *,
    reference_maximum_distance_m: float,
    minimum_improvement_m: float = DEFAULT_BREAKTHROUGH_IMPROVEMENT_M,
    progress_milestones_m: tuple[float, ...] = (
        SAFE_ACTOR_PROGRESS_MILESTONES_M
    ),
    completed_lap: bool = False,
) -> tuple[str, ...]:
    """Classify a frontier result that merits immediate matched validation."""
    if completed_lap:
        return ("lap_completion",)
    distance = max(0.0, finite_float(distance_m))
    reference = max(0.0, finite_float(reference_maximum_distance_m))
    required_gain = max(0.0, finite_float(minimum_improvement_m))
    reasons: list[str] = []
    if distance >= reference + required_gain:
        reasons.append("furthest_distance")
    new_milestones = tuple(
        sorted(
            {
                float(milestone)
                for milestone in progress_milestones_m
                if reference < float(milestone) <= distance
            }
        )
    )
    if new_milestones:
        reasons.append("new_sector_milestone")
    return tuple(reasons)


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
    metadata = read_policy_metadata(continuation_source_path(args))
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
    raise_on_failure: bool = True,
    training_phase: str = "continuation_compatibility_probe",
) -> dict[str, Any]:
    """Verify the loaded policy in TORCS before replay collection or updates."""
    minimum_distance = max(0.0, finite_float(minimum_distance_m))
    raw_env.set_policy_mode(
        deterministic_probe=True,
        action_noise_sigma=0.0,
        training_phase=training_phase,
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
        if not result["passed"] and raise_on_failure:
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


def run_frontier_source_qualification(
    model: Any,
    raw_env: Any,
    *,
    minimum_distance_m: float,
    maximum_attempts: int,
    training_budget_state: TrainingBudgetState | None = None,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Admit discovery only after the frozen source reproduces its frontier."""
    required_distance_m = max(0.0, finite_float(minimum_distance_m))
    attempts_requested = max(1, int(maximum_attempts))
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(attempts_requested):
        seed = (int(base_seed) + attempt_index) % (MAX_REPRODUCIBLE_SEED + 1)
        result = run_continuation_compatibility_probe(
            model,
            raw_env,
            minimum_distance_m=required_distance_m,
            training_budget_state=training_budget_state,
            seed=seed,
            raise_on_failure=False,
            training_phase="frontier_source_qualification",
        )
        result["attempt_index"] = attempt_index + 1
        attempts.append(result)
        if bool(result["passed"]):
            return {
                "protocol": FRONTIER_CONTINUATION_PROTOCOL,
                "passed": True,
                "minimum_distance_m": required_distance_m,
                "best_distance_m": max(
                    finite_float(item.get("distance_m")) for item in attempts
                ),
                "attempts_requested": attempts_requested,
                "attempts_completed": len(attempts),
                "successful_attempt": attempt_index + 1,
                "actor_frozen": True,
                "action_noise_sigma": 0.0,
                "gradient_updates": 0,
                "attempts": attempts,
            }

    distances = ", ".join(
        f"{finite_float(item.get('distance_m')):.1f}m" for item in attempts
    )
    raise RuntimeError(
        "frontier source qualification failed before training: the frozen "
        f"actor reached [{distances}], but must reproduce "
        f"{required_distance_m:.1f}m. No replay or gradient training was started."
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
            pre_actor_noise_factory: Any = None,
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
            regular_probes_enabled: bool = True,
            frontier_exploration_start_distance_m: float = 0.0,
            frozen_policy_baseline_distance_m: float = 0.0,
            frozen_policy_health_fraction: float = (
                DEFAULT_FROZEN_POLICY_HEALTH_FRACTION
            ),
            frozen_policy_health_failures: int = (
                DEFAULT_FROZEN_POLICY_HEALTH_FAILURES
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
            self.pre_actor_noise_factory = pre_actor_noise_factory
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
            self.regular_probes_enabled = bool(regular_probes_enabled)
            self.frontier_exploration_start_distance_m = max(
                0.0,
                finite_float(frontier_exploration_start_distance_m),
            )
            self.frontier_gate_enabled = bool(
                self.frontier_exploration_start_distance_m > 0.0
            )
            self.frontier_gate_open = False
            self.frozen_policy_baseline_distance_m = max(
                0.0,
                finite_float(frozen_policy_baseline_distance_m),
            )
            self.frozen_policy_health_fraction = clamp(
                finite_float(frozen_policy_health_fraction),
                0.01,
                1.0,
            )
            self.frozen_policy_health_failures = max(
                1,
                int(frozen_policy_health_failures),
            )
            self.frozen_policy_health_failure_streak = 0
            self.frozen_policy_health_failure_reason: str | None = None
            self.recent_training_distances_m: list[float] = []
            self.recent_training_completed_lap = False
            self.probe_episodes_completed = 0
            self.policy_episodes_since_probe = 0
            self.current_episode_is_probe = False
            self.last_noise_update_timestep = -ACTION_NOISE_UPDATE_INTERVAL_STEPS
            self._restored_state = False

        def _on_training_start(self):
            # TORCS resumes at a new episode, so exact-resume state cannot carry
            # an open within-episode frontier gate across process boundaries.
            self._set_frontier_gate_open(False)
            self._set_model_episode_actor_eligible(False)
            self._apply_policy_mode(
                deterministic_probe=(
                    self.current_episode_is_probe if self._restored_state else False
                ),
                force=True,
            )
            self._restored_state = False

        def training_state_dict(self) -> dict[str, Any]:
            return {
                "learning_starts": self.learning_starts,
                "initial_sigma": self.initial_sigma,
                "final_sigma": self.final_sigma,
                "decay_steps": self.decay_steps,
                "probe_interval": self.probe_interval,
                "training_gradient_steps": self.training_gradient_steps,
                "pre_update_training_phase": self.pre_update_training_phase,
                "actor_update_start": self.actor_update_start,
                "actor_full_unfreeze": self.actor_full_unfreeze,
                "safe_probe_noise_std": self.safe_probe_noise_std,
                "probe_evidence_window": self.probe_evidence_window,
                "probe_min_evidence_episodes": self.probe_min_evidence_episodes,
                "probe_evidence_max_regression_m": (
                    self.probe_evidence_max_regression_m
                ),
                "regular_probes_enabled": self.regular_probes_enabled,
                "frontier_exploration_start_distance_m": (
                    self.frontier_exploration_start_distance_m
                ),
                "frontier_gate_enabled": self.frontier_gate_enabled,
                "frontier_gate_open": self.frontier_gate_open,
                "frozen_policy_baseline_distance_m": (
                    self.frozen_policy_baseline_distance_m
                ),
                "frozen_policy_health_fraction": self.frozen_policy_health_fraction,
                "frozen_policy_health_failures": self.frozen_policy_health_failures,
                "frozen_policy_health_failure_streak": (
                    self.frozen_policy_health_failure_streak
                ),
                "frozen_policy_health_failure_reason": (
                    self.frozen_policy_health_failure_reason
                ),
                "probe_reference_distance_m": self.probe_reference_distance_m,
                "recent_training_distances_m": list(
                    self.recent_training_distances_m
                ),
                "recent_training_completed_lap": (
                    self.recent_training_completed_lap
                ),
                "probe_episodes_completed": self.probe_episodes_completed,
                "policy_episodes_since_probe": self.policy_episodes_since_probe,
                "current_episode_is_probe": self.current_episode_is_probe,
                "last_noise_update_timestep": self.last_noise_update_timestep,
            }

        def restore_training_state(self, state: Mapping[str, Any]) -> None:
            integer_fields = (
                "learning_starts",
                "decay_steps",
                "probe_interval",
                "training_gradient_steps",
                "actor_update_start",
                "actor_full_unfreeze",
                "probe_evidence_window",
                "probe_min_evidence_episodes",
                "frozen_policy_health_failures",
            )
            for name in integer_fields:
                if state.get(name) is not None:
                    setattr(self, name, int(finite_float(state[name])))
            float_fields = (
                "initial_sigma",
                "final_sigma",
                "safe_probe_noise_std",
                "probe_evidence_max_regression_m",
                "frontier_exploration_start_distance_m",
                "frozen_policy_baseline_distance_m",
                "frozen_policy_health_fraction",
            )
            for name in float_fields:
                if state.get(name) is not None:
                    setattr(self, name, finite_float(state[name]))
            if state.get("pre_update_training_phase") is not None:
                self.pre_update_training_phase = str(
                    state["pre_update_training_phase"]
                )
            if state.get("regular_probes_enabled") is not None:
                self.regular_probes_enabled = bool(
                    state["regular_probes_enabled"]
                )
            if state.get("frontier_gate_enabled") is not None:
                self.frontier_gate_enabled = bool(state["frontier_gate_enabled"])
            self.frontier_gate_open = bool(state.get("frontier_gate_open", False))
            self.frozen_policy_health_failure_streak = max(
                0,
                int(
                    finite_float(
                        state.get("frozen_policy_health_failure_streak"),
                    )
                ),
            )
            restored_health_reason = state.get("frozen_policy_health_failure_reason")
            self.frozen_policy_health_failure_reason = (
                str(restored_health_reason) if restored_health_reason else None
            )
            self.probe_reference_distance_m = max(
                0.0,
                finite_float(state.get("probe_reference_distance_m")),
            )
            distances = state.get("recent_training_distances_m")
            self.recent_training_distances_m = (
                [max(0.0, finite_float(value)) for value in distances][
                    -self.probe_evidence_window:
                ]
                if isinstance(distances, (list, tuple))
                else []
            )
            self.recent_training_completed_lap = bool(
                state.get("recent_training_completed_lap")
            )
            self.probe_episodes_completed = max(
                0,
                int(finite_float(state.get("probe_episodes_completed"))),
            )
            self.policy_episodes_since_probe = max(
                0,
                int(finite_float(state.get("policy_episodes_since_probe"))),
            )
            self.current_episode_is_probe = bool(
                state.get("current_episode_is_probe")
            )
            self.last_noise_update_timestep = int(
                finite_float(
                    state.get("last_noise_update_timestep"),
                    -ACTION_NOISE_UPDATE_INTERVAL_STEPS,
                )
            )
            self._restored_state = True

        def _on_step(self):
            completed_episode = False
            health_abort_requested = False
            infos = self.locals.get("infos", [])
            if self.training_budget_state is not None:
                self.training_budget_state.record_infos(infos)
            self._update_frontier_gate(infos)
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
                health_abort_requested = bool(
                    self._record_frozen_policy_health(summary)
                    or health_abort_requested
                )
                if bool(summary.get("policy_controlled")):
                    if bool(summary.get("deterministic_probe")):
                        safe_result = self._record_safe_actor_probe(summary)
                        if safe_result is not None:
                            self._annotate_safe_actor_probe(summary, safe_result)
                        else:
                            self._record_regular_probe_result(summary)
                        self.probe_episodes_completed += 1
                        self.policy_episodes_since_probe = 0
                    elif not bool(
                        summary.get("training_budget_boundary_reached")
                    ):
                        self.policy_episodes_since_probe += 1
                        self._record_discovery_recovery(summary)
                        self._record_training_evidence(summary)

            if completed_episode:
                completed_episode_was_probe = self.current_episode_is_probe
                self._set_model_episode_actor_eligible(
                    bool(
                        self.frontier_gate_open
                        and not completed_episode_was_probe
                    )
                )
                self._set_frontier_gate_open(False)
                next_episode_is_probe = self._next_episode_should_probe()
                self._apply_policy_mode(
                    deterministic_probe=next_episode_is_probe,
                    force=True,
                )
                if completed_episode_was_probe:
                    self.model.gradient_steps = 0
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
            if health_abort_requested:
                self.model.gradient_steps = 0
                return False
            if self.training_budget_state is None:
                return True
            if not self.training_budget_state.should_stop():
                return True
            return bool(
                self.current_episode_is_probe
                and self._safe_actor_probe_required()
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
            self._set_frontier_gate_open(False)
            self.model.action_noise = None if self.frontier_gate_enabled else (
                self._training_noise(sigma)
            )
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
            training_noise_enabled = bool(
                not self.frontier_gate_enabled or self.frontier_gate_open
            )
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
                        None
                        if deterministic_probe or not training_noise_enabled
                        else self._training_noise(sigma)
                    )
            self._set_model_frontier_gate(
                bool(training_noise_enabled and not deterministic_probe)
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
                    self.safe_probe_noise_std
                    if safe_actor_probe
                    else sigma
                    if training_noise_enabled and not deterministic_probe
                    else 0.0
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
                discovery_frontier_gate_open=(
                    self.frontier_gate_open
                    if self.frontier_gate_enabled
                    else False
                ),
                discovery_frontier_gate_distance_m=(
                    self.frontier_exploration_start_distance_m
                    if self.frontier_gate_enabled
                    else 0.0
                ),
            )

        def _update_frontier_gate(self, infos: Any) -> None:
            if (
                not self.frontier_gate_enabled
                or self.current_episode_is_probe
                or self.frontier_gate_open
                or self._budget_timestep() < self.actor_update_start
            ):
                return
            furthest_distance_m = 0.0
            for info in infos:
                if not isinstance(info, Mapping):
                    continue
                if bool(info.get("deterministic_probe")):
                    continue
                furthest_distance_m = max(
                    furthest_distance_m,
                    finite_float(info.get("distance_m")),
                    finite_float(info.get("furthest_distance_m")),
                )
            if (
                furthest_distance_m
                < self.frontier_exploration_start_distance_m
            ):
                return
            self._set_frontier_gate_open(True)
            self._apply_policy_mode(deterministic_probe=False, force=True)

        def _set_frontier_gate_open(self, value: bool) -> None:
            self.frontier_gate_open = bool(
                self.frontier_gate_enabled and value
            )
            self._set_model_frontier_gate(self.frontier_gate_open)

        def _set_model_frontier_gate(self, value: bool) -> None:
            setter = getattr(
                self.model,
                "set_discovery_frontier_gate_open",
                None,
            )
            if callable(setter):
                setter(bool(value))

        def _set_model_episode_actor_eligible(self, value: bool) -> None:
            setter = getattr(
                self.model,
                "set_discovery_episode_actor_eligible",
                None,
            )
            if callable(setter):
                setter(bool(value))

        def _record_frozen_policy_health(self, summary: dict[str, Any]) -> bool:
            threshold_m = (
                self.frozen_policy_baseline_distance_m
                * self.frozen_policy_health_fraction
            )
            monitoring = bool(
                threshold_m > 0.0
                and self._budget_timestep() < self.actor_update_start
                and bool(summary.get("policy_controlled"))
                and not bool(summary.get("deterministic_probe"))
                and finite_float(summary.get("action_noise_sigma")) <= 0.0
            )
            if not monitoring:
                return False

            distance_m = max(
                finite_float(summary.get("distance_m")),
                finite_float(summary.get("furthest_distance_m")),
            )
            if distance_m >= threshold_m:
                self.frozen_policy_health_failure_streak = 0
                decision = "healthy"
            else:
                self.frozen_policy_health_failure_streak += 1
                decision = "degraded"

            abort_requested = bool(
                self.frozen_policy_health_failure_streak
                >= self.frozen_policy_health_failures
            )
            if abort_requested:
                decision = "abort_before_actor_updates"
                self.frozen_policy_health_failure_reason = (
                    "frozen policy became unhealthy before actor updates: "
                    f"{self.frozen_policy_health_failure_streak} consecutive "
                    f"episodes below {threshold_m:.1f}m"
                )
            summary["frozen_policy_health_threshold_m"] = threshold_m
            summary["frozen_policy_health_failure_streak"] = (
                self.frozen_policy_health_failure_streak
            )
            summary["frozen_policy_health_decision"] = decision
            return abort_requested

        def _training_noise(self, sigma: float) -> Any:
            if (
                self.pre_actor_noise_factory is not None
                and self._budget_timestep() < self.actor_update_start
            ):
                return self.pre_actor_noise_factory(sigma)
            return self.noise_factory(sigma)

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
                summary["safe_actor_training_evidence_reasons"] = result.get(
                    "evidence_reasons"
                )
            if decision == "probe_eligible_breakthrough" and bool(
                result.get("new_breakthrough")
            ):
                print(
                    "\nSafe actor candidate "
                    f"{result['candidate_id']} reached a new frontier; "
                    "actor updates are frozen pending matched validation "
                    f"({','.join(result.get('evidence_reasons') or [])})."
                )
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
            if (
                self.training_budget_state is not None
                and self.training_budget_state.training_budget_reached
            ):
                return False
            if not self.regular_probes_enabled:
                return False
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

        def _record_discovery_recovery(
            self,
            summary: dict[str, Any],
        ) -> None:
            record = getattr(
                self.model,
                "record_discovery_training_episode",
                None,
            )
            if not callable(record):
                return
            result = record(summary)
            if not isinstance(result, Mapping):
                return
            summary["discovery_launch_failure_streak"] = result.get(
                "launch_failure_streak"
            )
            summary["discovery_recovery_triggered"] = bool(
                result.get("triggered")
            )
            summary["discovery_recovery_count"] = result.get(
                "recovery_count"
            )
            summary["discovery_recovery_cooldown_until"] = result.get(
                "cooldown_until"
            )
            summary["discovery_actor_learning_rate_scale"] = result.get(
                "actor_learning_rate_scale"
            )
            summary["discovery_effective_policy_delay"] = result.get(
                "effective_policy_delay"
            )
            summary["discovery_actor_updates_since_anchor"] = result.get(
                "actor_updates_since_anchor"
            )
            if bool(result.get("triggered")):
                print(
                    "\nDiscovery frontier regression recovered: restored the "
                    f"{finite_float(result.get('anchor_distance_m')):.1f}m "
                    "frontier actor; recovery="
                    f"{int(finite_float(result.get('recovery_count')))}, "
                    "actor cooldown until training step "
                    f"{int(finite_float(result.get('cooldown_until'))):,}; "
                    "critic and replay retained."
                )

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
            summary["safe_actor_probe_upper_quartile_distance_m"] = result.get(
                "upper_quartile_distance_m"
            )
            summary["safe_actor_probe_maximum_distance_m"] = result.get(
                "maximum_distance_m"
            )
            summary["safe_actor_reference_median_distance_m"] = result.get(
                "reference_median_distance_m"
            )
            summary["safe_actor_reference_minimum_distance_m"] = result.get(
                "reference_minimum_distance_m"
            )
            summary["safe_actor_reference_upper_quartile_distance_m"] = (
                result.get("reference_upper_quartile_distance_m")
            )
            summary["safe_actor_reference_maximum_distance_m"] = result.get(
                "reference_maximum_distance_m"
            )
            summary["safe_actor_aggregate_median_gain_m"] = result.get(
                "aggregate_median_gain_m"
            )
            summary["safe_actor_minimum_gain_m"] = result.get(
                "minimum_gain_m"
            )
            summary["safe_actor_upper_quartile_gain_m"] = result.get(
                "upper_quartile_gain_m"
            )
            summary["safe_actor_maximum_gain_m"] = result.get(
                "maximum_gain_m"
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
            summary["safe_actor_reference_lap_completions"] = result.get(
                "reference_lap_completions"
            )
            summary["safe_actor_candidate_lap_completions"] = result.get(
                "candidate_lap_completions"
            )
            summary["safe_actor_new_progress_milestones_m"] = result.get(
                "new_progress_milestones_m"
            )
            summary["safe_actor_acceptance_reasons"] = result.get(
                "acceptance_reasons"
            )
            summary["safe_actor_safety_failures"] = result.get(
                "safety_failures"
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
                    "; progress="
                    f"{','.join(result.get('acceptance_reasons') or ['none'])}"
                    "; safety="
                    f"{','.join(result.get('safety_failures') or ['passed'])}"
                    f"{probe_suffix}"
                )

    return ExplorationScheduleCallback


def policy_lap_is_valid(summary: Mapping[str, Any]) -> bool:
    if not bool(summary.get("policy_controlled")):
        return False
    laps_completed = int(finite_float(summary.get("laps_completed")))
    lap_time = finite_float(
        summary.get("best_lap_time_seconds"),
        default=float("nan"),
    )
    distance_m = max(
        finite_float(summary.get("distance_m")),
        finite_float(summary.get("furthest_distance_m")),
    )
    return bool(
        laps_completed > 0
        and math.isfinite(lap_time)
        and lap_time > 0.0
        and distance_m >= G_TRACK_3_LENGTH_METRES * 0.90
    )


def lap_checkpoint_record(
    summary: Mapping[str, Any],
    *,
    source_policy_path: Path,
    global_timestep: int,
    track: str = DEFAULT_TRACK_NAME,
) -> dict[str, Any]:
    laps_completed = max(
        1,
        int(finite_float(summary.get("laps_completed"))),
    )
    return {
        "source": "training_valid_lap",
        "source_policy_path": str(source_policy_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "global_timestep": max(0, int(global_timestep)),
        "track": str(track),
        "deterministic": bool(summary.get("deterministic_probe")),
        "completed_laps": laps_completed,
        "validation_trials": 1,
        "reliability": 1.0,
        "best_lap_seconds": finite_float(
            summary.get("best_lap_time_seconds")
        ),
        "distance_m": max(
            finite_float(summary.get("distance_m")),
            finite_float(summary.get("furthest_distance_m")),
        ),
        "policy_role": (
            "candidate"
            if summary.get("safe_actor_probe_mode") == "candidate"
            else "working_policy"
        ),
        "episode_summary": dict(summary),
    }


def make_lap_checkpoint_callback_class(BaseCallback: Any):
    class LapCheckpointCallback(BaseCallback):
        def __init__(
            self,
            lap_checkpoint_dir: Path,
            best_completed_lap_model_path: Path,
            metadata: Mapping[str, Any],
        ) -> None:
            super().__init__(verbose=0)
            self.lap_checkpoint_dir = Path(lap_checkpoint_dir)
            self.best_completed_lap_model_path = Path(
                best_completed_lap_model_path
            )
            self.metadata = dict(metadata)
            self.saved_laps = 0

        def _on_step(self):
            for info in self.locals.get("infos", []):
                summary = (
                    info.get("episode_summary")
                    if isinstance(info, Mapping)
                    else None
                )
                if not isinstance(summary, Mapping) or not policy_lap_is_valid(
                    summary
                ):
                    continue
                self._archive_lap(summary)
            return True

        def _archive_lap(self, summary: Mapping[str, Any]) -> None:
            self.saved_laps += 1
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%d-%H%M%S-%f"
            )
            archive_path = self.lap_checkpoint_dir / (
                f"agent6_td3_lap_{int(self.num_timesteps):09d}_{timestamp}.zip"
            )
            save_runtime = getattr(self.model, "save_runtime_policy", None)

            def save_policy(path: Path) -> None:
                if callable(save_runtime):
                    save_runtime(path)
                else:
                    self.model.save(str(path))

            atomic_save_policy(save_policy, archive_path)
            lap_record = lap_checkpoint_record(
                summary,
                source_policy_path=archive_path,
                global_timestep=self.num_timesteps,
                track=str(self.metadata.get("track") or DEFAULT_TRACK_NAME),
            )
            archive_metadata = {
                **self.metadata,
                "checkpoint_type": "valid_completed_lap_archive",
                "valid_completed_lap": lap_record,
            }
            write_json(
                metadata_path_for_policy(archive_path),
                archive_metadata,
            )
            promoted = promote_best_completed_lap(
                archive_path,
                self.best_completed_lap_model_path,
                archive_metadata,
                lap_record,
            )
            print(
                "\nSaved valid Agent 6 lap checkpoint: "
                f"{archive_path}"
            )
            if promoted:
                print(
                    "Promoted completed-lap champion: "
                    f"{self.best_completed_lap_model_path}"
                )

    return LapCheckpointCallback


def make_frontier_checkpoint_callback_class(BaseCallback: Any):
    class FrontierCheckpointCallback(BaseCallback):
        """Protect distance discoveries independently from the robust champion."""

        def __init__(
            self,
            frontier_model_path: Path,
            frontier_checkpoint_dir: Path,
            metadata: Mapping[str, Any],
            *,
            source_policy_path: Path | None,
            source_evidence: Mapping[str, Any] | None,
            milestones_m: tuple[float, ...],
            minimum_improvement_m: float,
            major_breakthrough_improvement_m: float,
            progress_state: TrainingProgressState | None = None,
        ) -> None:
            super().__init__(verbose=0)
            self.frontier_model_path = Path(frontier_model_path)
            self.frontier_checkpoint_dir = Path(frontier_checkpoint_dir)
            self.metadata = dict(metadata)
            self.source_policy_path = (
                Path(source_policy_path) if source_policy_path is not None else None
            )
            self.source_evidence = dict(source_evidence or {})
            self.milestones_m = tuple(
                sorted({max(0.0, finite_float(value)) for value in milestones_m})
            )
            self.minimum_improvement_m = max(
                0.0,
                finite_float(minimum_improvement_m),
            )
            self.major_breakthrough_improvement_m = max(
                self.minimum_improvement_m,
                finite_float(major_breakthrough_improvement_m),
            )
            self.progress_state = progress_state
            existing = read_policy_metadata(self.frontier_model_path)
            existing_frontier = existing.get("frontier_champion")
            self.best_distance_m = (
                max(0.0, finite_float(existing_frontier.get("distance_m")))
                if isinstance(existing_frontier, Mapping)
                else None
            )
            self.completed_lap = bool(
                isinstance(existing_frontier, Mapping)
                and existing_frontier.get("completed_lap")
            )
            self.best_lap_time_seconds = (
                finite_float(
                    existing_frontier.get("best_lap_time_seconds"),
                    default=float("inf"),
                )
                if isinstance(existing_frontier, Mapping)
                else float("inf")
            )
            history = existing.get("frontier_history")
            self.history = (
                [dict(item) for item in history if isinstance(item, Mapping)][-24:]
                if isinstance(history, list)
                else []
            )
            self.saved_milestones_m = {
                milestone
                for milestone in self.milestones_m
                if self.best_distance_m is not None
                and self.best_distance_m >= milestone
            }
            self.major_breakthrough_this_run = False
            self.completed_lap_this_run = False
            self.records_saved_this_run = 0
            self.strict_evaluation_reference_distance_m = (
                self.best_distance_m or 0.0
            )
            if self.progress_state is not None and self.best_distance_m is not None:
                self.progress_state.record_best_distance(self.best_distance_m)

        def _on_training_start(self) -> None:
            if self.best_distance_m is not None and self.frontier_model_path.is_file():
                return
            source_metadata = read_policy_metadata(self.source_policy_path)
            initial_distance_m = frontier_evidence_distance_m(
                source_metadata,
                self.source_evidence,
            )
            self._save_frontier(
                distance_m=initial_distance_m,
                summary=None,
                completed_lap=False,
                selection_reason="bootstrap_from_historical_frontier_evidence",
                count_as_run_breakthrough=False,
            )
            self.strict_evaluation_reference_distance_m = (
                self.best_distance_m or 0.0
            )

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                summary = (
                    info.get("episode_summary")
                    if isinstance(info, Mapping)
                    else None
                )
                if not isinstance(summary, Mapping) or not bool(
                    summary.get("policy_controlled")
                ):
                    continue
                distance_m = max(
                    0.0,
                    finite_float(summary.get("distance_m")),
                    finite_float(summary.get("furthest_distance_m")),
                )
                lap_completed = policy_lap_is_valid(summary)
                lap_time_seconds = finite_float(
                    summary.get("best_lap_time_seconds"),
                    default=float("inf"),
                )
                if self.completed_lap:
                    if not lap_completed:
                        continue
                    if lap_time_seconds >= self.best_lap_time_seconds:
                        continue
                previous_distance_m = self.best_distance_m or 0.0
                crossed_milestone = any(
                    previous_distance_m < milestone <= distance_m
                    for milestone in self.milestones_m
                )
                improved = distance_m >= (
                    previous_distance_m + self.minimum_improvement_m
                )
                if not (lap_completed or crossed_milestone or improved):
                    continue
                self._save_frontier(
                    distance_m=distance_m,
                    summary=summary,
                    completed_lap=lap_completed,
                    selection_reason=(
                        "completed_lap"
                        if lap_completed
                        else "distance_milestone"
                        if crossed_milestone
                        else "new_furthest_policy_controlled_episode"
                    ),
                    count_as_run_breakthrough=True,
                )
            return True

        @property
        def requires_strict_evaluation(self) -> bool:
            return bool(
                self.major_breakthrough_this_run or self.completed_lap_this_run
            )

        def status(self) -> dict[str, Any]:
            return {
                "frontier_model_path": str(self.frontier_model_path),
                "best_distance_m": self.best_distance_m,
                "completed_lap": self.completed_lap,
                "best_lap_time_seconds": (
                    self.best_lap_time_seconds
                    if math.isfinite(self.best_lap_time_seconds)
                    else None
                ),
                "saved_milestones_m": sorted(self.saved_milestones_m),
                "records_saved_this_run": self.records_saved_this_run,
                "strict_evaluation_reference_distance_m": (
                    self.strict_evaluation_reference_distance_m
                ),
                "major_breakthrough_this_run": self.major_breakthrough_this_run,
                "completed_lap_this_run": self.completed_lap_this_run,
                "requires_strict_evaluation": self.requires_strict_evaluation,
            }

        def _save_frontier(
            self,
            *,
            distance_m: float,
            summary: Mapping[str, Any] | None,
            completed_lap: bool,
            selection_reason: str,
            count_as_run_breakthrough: bool,
        ) -> None:
            previous_distance_m = self.best_distance_m or 0.0
            recorded_distance_m = max(previous_distance_m, float(distance_m))
            crossed = tuple(
                milestone
                for milestone in self.milestones_m
                if previous_distance_m < milestone <= recorded_distance_m
            )
            significant_gain = recorded_distance_m >= (
                self.strict_evaluation_reference_distance_m
                + self.major_breakthrough_improvement_m
            )
            best_lap_time_seconds = (
                finite_float(
                    summary.get("best_lap_time_seconds"),
                    default=float("inf"),
                )
                if completed_lap and summary is not None
                else self.best_lap_time_seconds
            )
            record = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "global_timestep": int(self.num_timesteps),
                "distance_m": recorded_distance_m,
                "episode_distance_m": float(distance_m),
                "completed_lap": bool(completed_lap),
                "best_lap_time_seconds": (
                    best_lap_time_seconds
                    if math.isfinite(best_lap_time_seconds)
                    else None
                ),
                "selection_reason": str(selection_reason),
                "policy_controlled": (
                    bool(summary.get("policy_controlled"))
                    if summary is not None
                    else None
                ),
                "evidence_source": (
                    "observed_training_episode"
                    if summary is not None
                    else "source_checkpoint_metadata"
                ),
                "deterministic": bool(
                    summary is not None and summary.get("deterministic_probe")
                ),
                "robustness_verified": False,
                "strict_evaluation_required": bool(
                    count_as_run_breakthrough
                    and (crossed or completed_lap or significant_gain)
                ),
                "milestones_crossed_m": list(crossed),
                "source_policy_path": (
                    str(self.source_policy_path)
                    if self.source_policy_path is not None
                    else None
                ),
                "source_external_evidence": (
                    dict(self.source_evidence) if self.source_evidence else None
                ),
                "episode_summary": dict(summary) if summary is not None else None,
            }
            self.history.append(
                {
                    key: value
                    for key, value in record.items()
                    if key != "episode_summary"
                }
            )
            self.history = self.history[-25:]
            payload = {
                **self.metadata,
                "checkpoint_type": "distance_frontier_champion",
                "frontier_champion": record,
                "frontier_history": list(self.history),
            }
            self._save_policy_bundle(self.frontier_model_path, payload)
            for milestone in crossed:
                milestone_path = self.frontier_checkpoint_dir / (
                    f"agent6_td3_frontier_{int(round(milestone)):04d}m.zip"
                )
                milestone_payload = {
                    **payload,
                    "checkpoint_type": "distance_frontier_milestone",
                    "frontier_milestone": {
                        "milestone_m": milestone,
                        "source_frontier_path": str(self.frontier_model_path),
                        "record": record,
                    },
                }
                self._save_policy_bundle(milestone_path, milestone_payload)
                self.saved_milestones_m.add(milestone)
                print(
                    "\nSaved Agent 6 frontier milestone: "
                    f"{milestone:.0f}m -> {milestone_path}"
                )
            self.best_distance_m = recorded_distance_m
            self.completed_lap = bool(self.completed_lap or completed_lap)
            if completed_lap:
                self.best_lap_time_seconds = best_lap_time_seconds
            snapshot_recovery_anchor = getattr(
                self.model,
                "snapshot_discovery_recovery_anchor",
                None,
            )
            if callable(snapshot_recovery_anchor):
                snapshot_recovery_anchor(self.best_distance_m)
            if self.progress_state is not None:
                self.progress_state.record_best_distance(self.best_distance_m)
            if count_as_run_breakthrough:
                self.records_saved_this_run += 1
                self.major_breakthrough_this_run = bool(
                    self.major_breakthrough_this_run
                    or crossed
                    or significant_gain
                )
                self.completed_lap_this_run = bool(
                    self.completed_lap_this_run or completed_lap
                )
                if crossed or significant_gain or completed_lap:
                    self.strict_evaluation_reference_distance_m = (
                        recorded_distance_m
                    )
                print(
                    "\nUpdated Agent 6 frontier champion: "
                    f"{self.best_distance_m:.1f}m -> {self.frontier_model_path}"
                )

        def _save_policy_bundle(
            self,
            path: Path,
            payload: Mapping[str, Any],
        ) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_runtime = getattr(self.model, "save_runtime_policy", None)

            def save_policy(target: Path) -> None:
                if callable(save_runtime):
                    save_runtime(target)
                else:
                    self.model.save(str(target))

            atomic_save_policy(save_policy, path)
            write_json(metadata_path_for_policy(path), payload)

    return FrontierCheckpointCallback


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
    checkpoint_saver: Callable[[Path, str], Any] | None = None,
):
    if training_budget_state is not None and BaseCallback is not None:
        class TrainingBudgetCheckpointCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.next_checkpoint_step = int(args.checkpoint_freq)

            def _on_training_start(self) -> None:
                completed = training_budget_state.training_steps
                frequency = int(args.checkpoint_freq)
                self.next_checkpoint_step = (
                    completed // frequency + 1
                ) * frequency

            def _on_step(self):
                training_steps = training_budget_state.training_steps
                if training_steps < self.next_checkpoint_step:
                    return True
                checkpoint_stem = (
                    Path(args.checkpoint_dir)
                    / f"agent6_td3_scratch_{training_steps}_steps"
                )
                if checkpoint_saver is not None:
                    checkpoint_saver(
                        checkpoint_stem.with_suffix(".zip"),
                        "periodic_checkpoint",
                    )
                else:
                    self.model.save(str(checkpoint_stem))
                if args.save_checkpoint_replay_buffer and checkpoint_saver is None:
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


def make_stable_frontier_td3_class(TD3: Any):
    """Build the compact actor-preserving TD3 continuation used by Agent 6."""
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3.common.utils import polyak_update, update_learning_rate

    class StableFrontierTD3(TD3):
        def _excluded_save_params(self) -> list[str]:
            return [
                *super()._excluded_save_params(),
                "_agent6_training_budget_state",
                "_agent6_actor_learning_rate_schedule",
            ]

        @staticmethod
        def _set_module_requires_grad(module: Any, enabled: bool) -> None:
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

        def configure_training_budget(
            self,
            training_budget_state: TrainingBudgetState,
        ) -> None:
            self._agent6_training_budget_state = training_budget_state

        def configure_continuation_updates(
            self,
            *,
            actor_learning_rate_schedule: Callable[[int], float],
            actor_updates_start_at: int,
            critic_warmup_updates: int = LEAN_FRONTIER_CRITIC_WARMUP_STEPS,
            critic_gradient_norm: float = LEAN_FRONTIER_GRADIENT_NORM,
            critic_huber_beta: float = LEAN_FRONTIER_HUBER_BETA,
            freeze_actor_before_updates: bool = True,
            **_legacy_options: Any,
        ) -> None:
            self._agent6_actor_learning_rate_schedule = (
                actor_learning_rate_schedule
            )
            self._agent6_actor_updates_start_at = max(
                0,
                int(actor_updates_start_at),
            )
            self._agent6_critic_warmup_updates = max(
                0,
                int(critic_warmup_updates),
            )
            self._agent6_critic_updates_since_transfer = 0
            self._agent6_critic_gradient_norm = max(
                0.0,
                float(critic_gradient_norm),
            )
            self._agent6_critic_huber_beta = max(
                np.finfo(np.float32).eps,
                float(critic_huber_beta),
            )
            self._agent6_actor_gradient_norm = LEAN_FRONTIER_GRADIENT_NORM
            self._agent6_actor_parameters_frozen = bool(
                freeze_actor_before_updates
                and self._agent6_critic_warmup_updates > 0
            )
            self._set_module_requires_grad(
                self.actor,
                not self._agent6_actor_parameters_frozen,
            )

        def _training_budget_timestep(self) -> int:
            state = getattr(self, "_agent6_training_budget_state", None)
            return int(self.num_timesteps) if state is None else int(
                state.training_steps
            )

        def _apply_training_budget_progress(self) -> None:
            state = getattr(self, "_agent6_training_budget_state", None)
            if state is not None:
                self._current_progress_remaining = state.progress_remaining

        def save_runtime_policy(self, path: Any) -> None:
            super().save(path)

        def exact_training_state_dict(self) -> dict[str, Any]:
            return {
                "frontier_finetuning": {
                    "critic_updates_since_transfer": int(
                        getattr(
                            self,
                            "_agent6_critic_updates_since_transfer",
                            0,
                        )
                    ),
                    "critic_warmup_updates": int(
                        getattr(self, "_agent6_critic_warmup_updates", 0)
                    ),
                    "actor_updates_start_at": int(
                        getattr(self, "_agent6_actor_updates_start_at", 0)
                    ),
                    "actor_parameters_frozen": bool(
                        getattr(
                            self,
                            "_agent6_actor_parameters_frozen",
                            False,
                        )
                    ),
                    "critic_gradient_norm": float(
                        getattr(
                            self,
                            "_agent6_critic_gradient_norm",
                            LEAN_FRONTIER_GRADIENT_NORM,
                        )
                    ),
                    "critic_huber_beta": float(
                        getattr(
                            self,
                            "_agent6_critic_huber_beta",
                            LEAN_FRONTIER_HUBER_BETA,
                        )
                    ),
                },
                "n_updates": int(getattr(self, "_n_updates", 0)),
                "num_timesteps": int(getattr(self, "num_timesteps", 0)),
            }

        def restore_exact_training_state(self, state: Mapping[str, Any]) -> None:
            protocol = state.get("frontier_finetuning")
            if not isinstance(protocol, Mapping):
                raise RuntimeError(
                    "exact-resume state is missing frontier fine-tuning data"
                )
            self._agent6_critic_updates_since_transfer = max(
                0,
                int(
                    finite_float(
                        protocol.get("critic_updates_since_transfer")
                    )
                ),
            )
            self._agent6_critic_warmup_updates = max(
                0,
                int(finite_float(protocol.get("critic_warmup_updates"))),
            )
            self._agent6_actor_updates_start_at = max(
                0,
                int(finite_float(protocol.get("actor_updates_start_at"))),
            )
            self._agent6_critic_gradient_norm = max(
                0.0,
                finite_float(
                    protocol.get("critic_gradient_norm"),
                    LEAN_FRONTIER_GRADIENT_NORM,
                ),
            )
            self._agent6_critic_huber_beta = max(
                np.finfo(np.float32).eps,
                finite_float(
                    protocol.get("critic_huber_beta"),
                    LEAN_FRONTIER_HUBER_BETA,
                ),
            )
            self._agent6_actor_parameters_frozen = bool(
                protocol.get("actor_parameters_frozen")
            )
            self._set_module_requires_grad(
                self.actor,
                not self._agent6_actor_parameters_frozen,
            )
            self._n_updates = max(0, int(finite_float(state.get("n_updates"))))
            self.num_timesteps = max(
                0,
                int(finite_float(state.get("num_timesteps"))),
            )

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self._apply_training_budget_progress()
            self.policy.set_training_mode(True)
            budget_timestep = self._training_budget_timestep()
            critic_learning_rate = self.lr_schedule(
                self._current_progress_remaining
            )
            actor_learning_rate = float(
                self._agent6_actor_learning_rate_schedule(budget_timestep)
            )
            update_learning_rate(self.critic.optimizer, critic_learning_rate)
            update_learning_rate(self.actor.optimizer, actor_learning_rate)

            actor_losses: list[float] = []
            actor_gradient_norms: list[float] = []
            critic_losses: list[float] = []
            critic_gradient_norms: list[float] = []
            q1_means: list[float] = []
            q2_means: list[float] = []
            target_q_means: list[float] = []
            td_error_means: list[float] = []
            td_error_maxima: list[float] = []

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
                    noise = (
                        th.randn_like(replay_data.actions)
                        * self.target_policy_noise
                    ).clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (
                        self.actor_target(replay_data.next_observations) + noise
                    ).clamp(-1.0, 1.0)
                    target_q_values = th.cat(
                        self.critic_target(
                            replay_data.next_observations,
                            next_actions,
                        ),
                        dim=1,
                    ).min(dim=1, keepdim=True).values
                    target_q_values = (
                        replay_data.rewards
                        + (1.0 - replay_data.dones)
                        * discounts
                        * target_q_values
                    )

                current_q_values = self.critic(
                    replay_data.observations,
                    replay_data.actions,
                )
                critic_loss = sum(
                    F.smooth_l1_loss(
                        current_q,
                        target_q_values,
                        beta=self._agent6_critic_huber_beta,
                    )
                    for current_q in current_q_values
                )
                critic_loss_value = float(critic_loss.item())
                self.critic.optimizer.zero_grad()
                critic_update_valid = math.isfinite(critic_loss_value)
                if critic_update_valid:
                    critic_loss.backward()
                    critic_gradient_norm = th.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self._agent6_critic_gradient_norm,
                    )
                    critic_gradient_norm_value = float(
                        critic_gradient_norm.item()
                    )
                    critic_update_valid = math.isfinite(
                        critic_gradient_norm_value
                    )
                else:
                    critic_gradient_norm_value = math.nan
                if not critic_update_valid:
                    self.critic.optimizer.zero_grad()
                    continue

                self.critic.optimizer.step()
                self._agent6_critic_updates_since_transfer += 1
                critic_losses.append(critic_loss_value)
                critic_gradient_norms.append(critic_gradient_norm_value)
                q1_means.append(float(current_q_values[0].mean().item()))
                q2_means.append(float(current_q_values[1].mean().item()))
                target_q_means.append(float(target_q_values.mean().item()))
                absolute_td_errors = th.cat(
                    [
                        (current_q.detach() - target_q_values).abs()
                        for current_q in current_q_values
                    ],
                    dim=1,
                )
                td_error_means.append(float(absolute_td_errors.mean().item()))
                td_error_maxima.append(float(absolute_td_errors.max().item()))
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

                actor_ready = bool(
                    budget_timestep > self._agent6_actor_updates_start_at
                    and self._agent6_critic_updates_since_transfer
                    > self._agent6_critic_warmup_updates
                    and actor_learning_rate > 0.0
                )
                if actor_ready and self._agent6_actor_parameters_frozen:
                    self._set_module_requires_grad(self.actor, True)
                    self._agent6_actor_parameters_frozen = False
                if not actor_ready or self._n_updates % self.policy_delay != 0:
                    continue

                actor_loss = -self.critic.q1_forward(
                    replay_data.observations,
                    self.actor(replay_data.observations),
                ).mean()
                actor_loss_value = float(actor_loss.item())
                self.actor.optimizer.zero_grad()
                if not math.isfinite(actor_loss_value):
                    continue
                actor_loss.backward()
                actor_gradient_norm = th.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self._agent6_actor_gradient_norm,
                )
                actor_gradient_norm_value = float(actor_gradient_norm.item())
                if not math.isfinite(actor_gradient_norm_value):
                    self.actor.optimizer.zero_grad()
                    continue
                self.actor.optimizer.step()
                actor_losses.append(actor_loss_value)
                actor_gradient_norms.append(actor_gradient_norm_value)
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

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/learning_rate", critic_learning_rate)
            self.logger.record("train/critic_learning_rate", critic_learning_rate)
            self.logger.record("train/actor_learning_rate", actor_learning_rate)
            self.logger.record(
                "train/critic_warmup_updates_completed",
                self._agent6_critic_updates_since_transfer,
            )
            self.logger.record(
                "train/actor_parameters_frozen",
                int(self._agent6_actor_parameters_frozen),
            )
            self.logger.record(
                "train/actor_updates_enabled",
                int(not self._agent6_actor_parameters_frozen),
            )
            if critic_losses:
                self.logger.record("train/critic_loss", np.mean(critic_losses))
                self.logger.record(
                    "train/critic_gradient_norm",
                    np.mean(critic_gradient_norms),
                )
                self.logger.record("train/q1_mean", np.mean(q1_means))
                self.logger.record("train/q2_mean", np.mean(q2_means))
                self.logger.record(
                    "train/target_q_mean",
                    np.mean(target_q_means),
                )
                self.logger.record(
                    "train/td_error_abs_mean",
                    np.mean(td_error_means),
                )
                self.logger.record(
                    "train/td_error_abs_max",
                    np.max(td_error_maxima),
                )
            if actor_losses:
                self.logger.record("train/actor_loss", np.mean(actor_losses))
                self.logger.record(
                    "train/actor_gradient_norm",
                    np.mean(actor_gradient_norms),
                )

    return StableFrontierTD3


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
                "_agent6_safe_actor_robust_state",
                "_agent6_safe_actor_robust_target_state",
                "_agent6_safe_actor_state",
                "_agent6_safe_actor_target_state",
                "_agent6_safe_actor_optimizer_state",
                "_agent6_safe_actor_anchor_observations",
                "_agent6_safe_actor_candidate_state",
                "_agent6_safe_actor_candidate_target_state",
                "_agent6_safe_actor_candidate_optimizer_state",
                "_agent6_training_budget_state",
                "_agent6_episode_atomic_raw_env",
                "_agent6_discovery_anchor_state",
                "_agent6_discovery_anchor_target_state",
                "_agent6_discovery_anchor_optimizer_state",
            ]

        def configure_training_budget(
            self,
            training_budget_state: TrainingBudgetState,
        ) -> None:
            self._agent6_training_budget_state = training_budget_state

        def configure_standard_td3_updates(self) -> None:
            """Use SB3's ordinary per-step TD3 optimizer without policy gates."""
            self._agent6_standard_td3_updates = True

        @staticmethod
        def _set_module_requires_grad(module: Any, enabled: bool) -> None:
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)

        def configure_episode_atomic_updates(self, raw_env: Any) -> None:
            """Run post-episode replay updates without a live TORCS session."""
            self._agent6_episode_atomic_raw_env = raw_env
            self._agent6_episode_atomic_batches = 0

        def _reset_after_episode_atomic_update(self) -> None:
            if self.env is None:
                raise RuntimeError(
                    "cannot resume TORCS after replay updates without an environment"
                )
            fresh_observation = self.env.reset()
            self._last_obs = fresh_observation
            if self._vec_normalize_env is not None:
                self._last_original_obs = (
                    self._vec_normalize_env.get_original_obs()
                )
            self._last_episode_starts = np.ones(
                (self.env.num_envs,),
                dtype=bool,
            )
            if self.action_noise is not None:
                self.action_noise.reset()

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

        def save_runtime_policy(self, path: Any) -> None:
            """Save the actor currently driving TORCS, including a live candidate."""
            super().save(path)

        def exact_training_state_dict(self) -> dict[str, Any]:
            """Serialize the continuation state excluded from ordinary SB3 saves."""
            state: dict[str, Any] = {}
            for name, value in self.__dict__.items():
                if not name.startswith("_agent6_safe_actor_"):
                    continue
                if name == "_agent6_safe_actor_reference" or callable(value):
                    continue
                state[name] = copy.deepcopy(value)
            return {
                "safe_actor": state,
                "discovery_recovery": {
                    name: copy.deepcopy(value)
                    for name, value in self.__dict__.items()
                    if name.startswith("_agent6_discovery_")
                    and not callable(value)
                },
                "frontier_finetuning": {
                    "critic_updates_since_transfer": int(
                        getattr(
                            self,
                            "_agent6_critic_updates_since_transfer",
                            0,
                        )
                    ),
                    "critic_warmup_updates": int(
                        getattr(self, "_agent6_critic_warmup_updates", 0)
                    ),
                    "actor_parameters_frozen": bool(
                        getattr(
                            self,
                            "_agent6_actor_parameters_frozen",
                            False,
                        )
                    ),
                    "critic_gradient_norm": float(
                        getattr(
                            self,
                            "_agent6_critic_gradient_norm",
                            LEAN_FRONTIER_GRADIENT_NORM,
                        )
                    ),
                    "critic_huber_beta": float(
                        getattr(
                            self,
                            "_agent6_critic_huber_beta",
                            LEAN_FRONTIER_HUBER_BETA,
                        )
                    ),
                },
                "n_updates": int(getattr(self, "_n_updates", 0)),
                "num_timesteps": int(getattr(self, "num_timesteps", 0)),
                "phased_continuation_configured": hasattr(
                    self,
                    "_agent6_actor_learning_rate_schedule",
                ),
            }

        def restore_exact_training_state(self, state: Mapping[str, Any]) -> None:
            safe_actor = state.get("safe_actor")
            if not isinstance(safe_actor, Mapping):
                raise RuntimeError("exact-resume state is missing safe_actor data")
            for name, value in safe_actor.items():
                if str(name).startswith("_agent6_safe_actor_"):
                    setattr(self, str(name), copy.deepcopy(value))
            discovery_recovery = state.get("discovery_recovery")
            if isinstance(discovery_recovery, Mapping):
                for name, value in discovery_recovery.items():
                    if str(name).startswith("_agent6_discovery_"):
                        setattr(self, str(name), copy.deepcopy(value))
            frontier_finetuning = state.get("frontier_finetuning")
            if isinstance(frontier_finetuning, Mapping):
                self._agent6_critic_updates_since_transfer = max(
                    0,
                    int(
                        finite_float(
                            frontier_finetuning.get(
                                "critic_updates_since_transfer"
                            )
                        )
                    ),
                )
                self._agent6_critic_warmup_updates = max(
                    0,
                    int(
                        finite_float(
                            frontier_finetuning.get("critic_warmup_updates")
                        )
                    ),
                )
                self._agent6_critic_gradient_norm = max(
                    0.0,
                    finite_float(
                        frontier_finetuning.get("critic_gradient_norm"),
                        LEAN_FRONTIER_GRADIENT_NORM,
                    ),
                )
                self._agent6_critic_huber_beta = max(
                    np.finfo(np.float32).eps,
                    finite_float(
                        frontier_finetuning.get("critic_huber_beta"),
                        LEAN_FRONTIER_HUBER_BETA,
                    ),
                )
                self._agent6_actor_parameters_frozen = bool(
                    frontier_finetuning.get("actor_parameters_frozen")
                )
                self._set_module_requires_grad(
                    self.actor,
                    not self._agent6_actor_parameters_frozen,
                )
            if getattr(self, "_agent6_safe_actor_enabled", False):
                reference = copy.deepcopy(self.actor)
                accepted_state = getattr(
                    self,
                    "_agent6_safe_actor_state",
                    None,
                )
                if not isinstance(accepted_state, Mapping):
                    raise RuntimeError(
                        "exact-resume state is missing the accepted actor snapshot"
                    )
                reference.load_state_dict(accepted_state)
                reference.set_training_mode(False)
                for parameter in reference.parameters():
                    parameter.requires_grad_(False)
                self._agent6_safe_actor_reference = reference
                candidate_snapshot = (
                    getattr(self, "_agent6_safe_actor_candidate_state", None),
                    getattr(
                        self,
                        "_agent6_safe_actor_candidate_target_state",
                        None,
                    ),
                    getattr(
                        self,
                        "_agent6_safe_actor_candidate_optimizer_state",
                        None,
                    ),
                )
                waiting_for_probe = bool(
                    getattr(self, "_agent6_safe_actor_waiting_for_probe", False)
                )
                candidate_snapshot_complete = all(
                    isinstance(value, Mapping) for value in candidate_snapshot
                )
                if waiting_for_probe and candidate_snapshot_complete:
                    self._restore_safe_actor_candidate()
                elif waiting_for_probe:
                    self._restore_safe_actor()
                    self._agent6_safe_actor_rolled_back_blocks += 1
                    self._agent6_safe_actor_last_decision = {
                        "candidate_id": self._agent6_safe_actor_candidate_id,
                        "decision": "rolled_back_resume_missing_candidate",
                    }
                    self._reset_safe_actor_probe_runtime()
                else:
                    self._restore_safe_actor()
            self._n_updates = max(
                0,
                int(finite_float(state.get("n_updates"))),
            )
            self.num_timesteps = max(
                0,
                int(finite_float(state.get("num_timesteps"))),
            )
            if not bool(state.get("phased_continuation_configured", True)):
                with contextlib.suppress(AttributeError):
                    del self._agent6_actor_learning_rate_schedule
                with contextlib.suppress(AttributeError):
                    del self._agent6_actor_updates_start_at

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
            safe_actor_median_regression_tolerance_m: float = (
                DEFAULT_SAFE_ACTOR_MEDIAN_REGRESSION_TOLERANCE_M
            ),
            safe_actor_minimum_regression_tolerance_m: float = (
                DEFAULT_SAFE_ACTOR_MINIMUM_REGRESSION_TOLERANCE_M
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
            safe_actor_breakthrough_improvement_m: float = (
                DEFAULT_BREAKTHROUGH_IMPROVEMENT_M
            ),
            source_verified_distance_m: float = 0.0,
            source_verified_minimum_distance_m: float = 0.0,
            source_verified_maximum_distance_m: float = 0.0,
            legacy_action_contract: bool = False,
            discovery_recovery_enabled: bool = False,
            discovery_collapse_episodes: int = (
                DEFAULT_DISCOVERY_COLLAPSE_EPISODES
            ),
            discovery_collapse_distance_m: float = (
                DEFAULT_DISCOVERY_COLLAPSE_DISTANCE_M
            ),
            discovery_recovery_cooldown_steps: int = (
                DEFAULT_DISCOVERY_RECOVERY_COOLDOWN_STEPS
            ),
            discovery_max_recovery_cooldown_steps: int = (
                DEFAULT_DISCOVERY_MAX_RECOVERY_COOLDOWN_STEPS
            ),
            discovery_actor_lr_backoff: float = (
                DEFAULT_DISCOVERY_ACTOR_LR_BACKOFF
            ),
            discovery_min_actor_lr_scale: float = (
                DEFAULT_DISCOVERY_MIN_ACTOR_LR_SCALE
            ),
            discovery_max_policy_delay: int = (
                DEFAULT_DISCOVERY_MAX_POLICY_DELAY
            ),
            discovery_frontier_gate_enabled: bool = False,
            discovery_frontier_gate_distance_m: float = 0.0,
            discovery_frontier_preservation_fraction: float = (
                DEFAULT_DISCOVERY_FRONTIER_PRESERVATION_FRACTION
            ),
            critic_warmup_updates: int = 0,
            critic_gradient_norm: float = LEAN_FRONTIER_GRADIENT_NORM,
            critic_huber_beta: float = LEAN_FRONTIER_HUBER_BETA,
            freeze_actor_before_updates: bool = True,
        ) -> None:
            self._agent6_actor_learning_rate_schedule = actor_learning_rate_schedule
            self._agent6_actor_updates_start_at = max(
                0,
                int(actor_updates_start_at),
            )
            self._agent6_critic_warmup_updates = max(
                0,
                int(critic_warmup_updates),
            )
            self._agent6_critic_updates_since_transfer = 0
            self._agent6_critic_gradient_norm = max(
                0.0,
                float(critic_gradient_norm),
            )
            self._agent6_critic_huber_beta = max(
                np.finfo(np.float32).eps,
                float(critic_huber_beta),
            )
            self._agent6_actor_parameters_frozen = bool(
                freeze_actor_before_updates
                and self._agent6_critic_warmup_updates > 0
            )
            self._set_module_requires_grad(
                self.actor,
                not self._agent6_actor_parameters_frozen,
            )
            self._agent6_safe_actor_enabled = bool(safe_actor_improvement)
            self._agent6_legacy_action_contract = bool(legacy_action_contract)
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
            self._agent6_safe_actor_median_regression_tolerance_m = max(
                0.0,
                float(safe_actor_median_regression_tolerance_m),
            )
            self._agent6_safe_actor_minimum_regression_tolerance_m = max(
                0.0,
                float(safe_actor_minimum_regression_tolerance_m),
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
            self._agent6_safe_actor_breakthrough_improvement_m = max(
                0.0,
                finite_float(safe_actor_breakthrough_improvement_m),
            )
            self._agent6_safe_actor_accepted_distance_m = max(
                0.0,
                finite_float(source_verified_distance_m),
            )
            self._agent6_safe_actor_accepted_minimum_distance_m = max(
                0.0,
                finite_float(source_verified_minimum_distance_m),
            )
            self._agent6_safe_actor_accepted_upper_quartile_distance_m = (
                self._agent6_safe_actor_accepted_distance_m
            )
            self._agent6_safe_actor_accepted_maximum_distance_m = max(
                self._agent6_safe_actor_accepted_distance_m,
                finite_float(source_verified_maximum_distance_m),
            )
            self._agent6_safe_actor_accepted_lap_completions = 0
            self._agent6_safe_actor_accepted_milestones_m: tuple[float, ...] = ()
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
            self._agent6_safe_actor_reference_laps_completed: list[int] = []
            self._agent6_safe_actor_candidate_laps_completed: list[int] = []
            self._agent6_safe_actor_accepted_blocks = 0
            self._agent6_safe_actor_rolled_back_blocks = 0
            self._agent6_safe_actor_screened_out_blocks = 0
            self._agent6_safe_actor_early_rolled_back_blocks = 0
            self._agent6_safe_actor_equivalent_blocks = 0
            self._agent6_safe_actor_no_evidence_blocks = 0
            self._agent6_safe_actor_breakthrough_candidates = 0
            self._agent6_safe_actor_invalid_updates = 0
            self._agent6_safe_actor_probe_episodes_skipped = 0
            self._agent6_safe_actor_evidence_candidate_id = 0
            self._agent6_safe_actor_evidence_distances_m: list[float] = []
            self._agent6_safe_actor_evidence_completed_lap = False
            self._agent6_safe_actor_evidence_breakthrough_reasons: tuple[
                str, ...
            ] = ()
            self._agent6_safe_actor_evidence_eligible = bool(
                self._agent6_safe_actor_min_evidence_episodes == 0
            )
            self._agent6_safe_actor_last_decision: dict[str, Any] | None = None
            self._agent6_safe_actor_anchor_observations = None
            self._clear_safe_actor_candidate_snapshot()
            if self._agent6_safe_actor_enabled:
                if not hasattr(self, "_agent6_safe_actor_robust_state"):
                    self._snapshot_safe_actor_robust_champion()
                self._snapshot_safe_actor()
            self._configure_discovery_recovery(
                enabled=discovery_recovery_enabled,
                collapse_episodes=discovery_collapse_episodes,
                collapse_distance_m=discovery_collapse_distance_m,
                cooldown_steps=discovery_recovery_cooldown_steps,
                maximum_cooldown_steps=(
                    discovery_max_recovery_cooldown_steps
                ),
                actor_learning_rate_backoff=discovery_actor_lr_backoff,
                minimum_actor_learning_rate_scale=(
                    discovery_min_actor_lr_scale
                ),
                maximum_policy_delay=discovery_max_policy_delay,
                source_distance_m=source_verified_maximum_distance_m,
            )
            self._agent6_discovery_frontier_gate_enabled = bool(
                discovery_frontier_gate_enabled
            )
            self._agent6_discovery_frontier_gate_distance_m = max(
                0.0,
                finite_float(discovery_frontier_gate_distance_m),
            )
            self._agent6_discovery_frontier_gate_open = not bool(
                self._agent6_discovery_frontier_gate_enabled
            )
            self._agent6_discovery_episode_actor_eligible = not bool(
                self._agent6_discovery_frontier_gate_enabled
            )
            self._agent6_discovery_frontier_preservation_fraction = clamp(
                finite_float(discovery_frontier_preservation_fraction),
                0.01,
                1.0,
            )

        def _configure_discovery_recovery(
            self,
            *,
            enabled: bool,
            collapse_episodes: int,
            collapse_distance_m: float,
            cooldown_steps: int,
            maximum_cooldown_steps: int,
            actor_learning_rate_backoff: float,
            minimum_actor_learning_rate_scale: float,
            maximum_policy_delay: int,
            source_distance_m: float,
        ) -> None:
            self._agent6_discovery_recovery_enabled = bool(enabled)
            self._agent6_discovery_collapse_episodes = max(
                2,
                int(collapse_episodes),
            )
            self._agent6_discovery_collapse_distance_m = max(
                0.0,
                finite_float(collapse_distance_m),
            )
            self._agent6_discovery_recovery_cooldown_steps = max(
                1,
                int(cooldown_steps),
            )
            self._agent6_discovery_max_recovery_cooldown_steps = max(
                self._agent6_discovery_recovery_cooldown_steps,
                int(maximum_cooldown_steps),
            )
            self._agent6_discovery_actor_lr_backoff = clamp(
                finite_float(actor_learning_rate_backoff),
                0.01,
                1.0,
            )
            self._agent6_discovery_min_actor_lr_scale = clamp(
                finite_float(minimum_actor_learning_rate_scale),
                0.001,
                1.0,
            )
            self._agent6_discovery_max_policy_delay = max(
                int(self.policy_delay),
                int(maximum_policy_delay),
            )
            self._agent6_discovery_launch_failure_streak = 0
            self._agent6_discovery_recovery_count = 0
            self._agent6_discovery_cooldown_until = 0
            self._agent6_discovery_actor_learning_rate_scale = 1.0
            self._agent6_discovery_effective_policy_delay = int(
                self.policy_delay
            )
            self._agent6_discovery_actor_updates_since_anchor = 0
            self._agent6_discovery_anchor_distance_m = max(
                0.0,
                finite_float(source_distance_m),
            )
            self._agent6_discovery_last_event: dict[str, Any] | None = None
            if self._agent6_discovery_recovery_enabled:
                self.snapshot_discovery_recovery_anchor(
                    self._agent6_discovery_anchor_distance_m
                )

        def snapshot_discovery_recovery_anchor(
            self,
            distance_m: float,
        ) -> None:
            if not getattr(
                self,
                "_agent6_discovery_recovery_enabled",
                False,
            ):
                return
            self._agent6_discovery_anchor_state = self._clone_module_state(
                self.actor
            )
            self._agent6_discovery_anchor_target_state = (
                self._clone_module_state(self.actor_target)
            )
            self._agent6_discovery_anchor_optimizer_state = copy.deepcopy(
                self.actor.optimizer.state_dict()
            )
            self._agent6_discovery_anchor_distance_m = max(
                self._agent6_discovery_anchor_distance_m,
                finite_float(distance_m),
            )
            self._agent6_discovery_launch_failure_streak = 0
            self._agent6_discovery_actor_updates_since_anchor = 0

        def _restore_discovery_recovery_anchor(self) -> None:
            self.actor.load_state_dict(self._agent6_discovery_anchor_state)
            self.actor_target.load_state_dict(
                self._agent6_discovery_anchor_target_state
            )
            self.actor.optimizer.load_state_dict(
                copy.deepcopy(
                    self._agent6_discovery_anchor_optimizer_state
                )
            )
            self._agent6_discovery_actor_updates_since_anchor = 0

        def record_discovery_training_episode(
            self,
            summary: Mapping[str, Any],
        ) -> dict[str, Any] | None:
            if not getattr(
                self,
                "_agent6_discovery_recovery_enabled",
                False,
            ):
                return None
            if bool(summary.get("deterministic_probe")) or not bool(
                summary.get("policy_controlled")
            ):
                return None
            timestep = self._training_budget_timestep()
            if timestep <= int(self._agent6_actor_updates_start_at):
                return None
            if (
                self._discovery_actor_in_cooldown(timestep)
                or self._agent6_discovery_actor_updates_since_anchor < 1
            ):
                self._agent6_discovery_launch_failure_streak = 0
                return {
                    "triggered": False,
                    **self.discovery_recovery_status(),
                }
            distance_m = max(
                0.0,
                finite_float(summary.get("distance_m")),
                finite_float(summary.get("furthest_distance_m")),
            )
            effective_failure_distance_m = max(
                self._agent6_discovery_collapse_distance_m,
                self._agent6_discovery_frontier_gate_distance_m
                * self._agent6_discovery_frontier_preservation_fraction,
            )
            launch_failed = distance_m < effective_failure_distance_m
            if launch_failed:
                self._agent6_discovery_launch_failure_streak += 1
            else:
                self._agent6_discovery_launch_failure_streak = 0

            triggered = bool(
                launch_failed
                and self._agent6_discovery_launch_failure_streak
                >= self._agent6_discovery_collapse_episodes
            )
            if triggered:
                self._restore_discovery_recovery_anchor()
                self._agent6_discovery_recovery_count += 1
                recovery_index = self._agent6_discovery_recovery_count - 1
                cooldown = min(
                    self._agent6_discovery_max_recovery_cooldown_steps,
                    self._agent6_discovery_recovery_cooldown_steps
                    * (2 ** min(recovery_index, 16)),
                )
                self._agent6_discovery_cooldown_until = timestep + cooldown
                self._agent6_discovery_actor_learning_rate_scale = max(
                    self._agent6_discovery_min_actor_lr_scale,
                    self._agent6_discovery_actor_lr_backoff
                    ** self._agent6_discovery_recovery_count,
                )
                self._agent6_discovery_effective_policy_delay = min(
                    self._agent6_discovery_max_policy_delay,
                    int(self.policy_delay)
                    * (2 ** self._agent6_discovery_recovery_count),
                )
                self._agent6_discovery_launch_failure_streak = 0
                self._agent6_discovery_last_event = {
                    "triggered_at_timestep": timestep,
                    "recovery_count": self._agent6_discovery_recovery_count,
                    "failed_episode_distance_m": distance_m,
                    "effective_failure_distance_m": (
                        effective_failure_distance_m
                    ),
                    "anchor_distance_m": (
                        self._agent6_discovery_anchor_distance_m
                    ),
                    "cooldown_steps": cooldown,
                    "cooldown_until": self._agent6_discovery_cooldown_until,
                    "actor_learning_rate_scale": (
                        self._agent6_discovery_actor_learning_rate_scale
                    ),
                    "effective_policy_delay": (
                        self._agent6_discovery_effective_policy_delay
                    ),
                    "critic_and_replay_preserved": True,
                }
            return {
                "triggered": triggered,
                "launch_failure_streak": (
                    self._agent6_discovery_launch_failure_streak
                ),
                **self.discovery_recovery_status(),
            }

        def discovery_recovery_status(self) -> dict[str, Any]:
            return {
                "enabled": bool(
                    getattr(
                        self,
                        "_agent6_discovery_recovery_enabled",
                        False,
                    )
                ),
                "recovery_count": int(
                    getattr(self, "_agent6_discovery_recovery_count", 0)
                ),
                "launch_failure_streak": int(
                    getattr(
                        self,
                        "_agent6_discovery_launch_failure_streak",
                        0,
                    )
                ),
                "cooldown_until": int(
                    getattr(self, "_agent6_discovery_cooldown_until", 0)
                ),
                "actor_learning_rate_scale": float(
                    getattr(
                        self,
                        "_agent6_discovery_actor_learning_rate_scale",
                        1.0,
                    )
                ),
                "effective_policy_delay": int(
                    getattr(
                        self,
                        "_agent6_discovery_effective_policy_delay",
                        self.policy_delay,
                    )
                ),
                "actor_updates_since_anchor": int(
                    getattr(
                        self,
                        "_agent6_discovery_actor_updates_since_anchor",
                        0,
                    )
                ),
                "anchor_distance_m": float(
                    getattr(self, "_agent6_discovery_anchor_distance_m", 0.0)
                ),
                "frontier_gate_enabled": bool(
                    getattr(
                        self,
                        "_agent6_discovery_frontier_gate_enabled",
                        False,
                    )
                ),
                "frontier_gate_open": bool(
                    getattr(
                        self,
                        "_agent6_discovery_frontier_gate_open",
                        True,
                    )
                ),
                "episode_actor_eligible": bool(
                    getattr(
                        self,
                        "_agent6_discovery_episode_actor_eligible",
                        True,
                    )
                ),
                "frontier_gate_distance_m": float(
                    getattr(
                        self,
                        "_agent6_discovery_frontier_gate_distance_m",
                        0.0,
                    )
                ),
                "frontier_preservation_fraction": float(
                    getattr(
                        self,
                        "_agent6_discovery_frontier_preservation_fraction",
                        DEFAULT_DISCOVERY_FRONTIER_PRESERVATION_FRACTION,
                    )
                ),
                "effective_failure_distance_m": max(
                    float(
                        getattr(
                            self,
                            "_agent6_discovery_collapse_distance_m",
                            0.0,
                        )
                    ),
                    float(
                        getattr(
                            self,
                            "_agent6_discovery_frontier_gate_distance_m",
                            0.0,
                        )
                    )
                    * float(
                        getattr(
                            self,
                            "_agent6_discovery_frontier_preservation_fraction",
                            DEFAULT_DISCOVERY_FRONTIER_PRESERVATION_FRACTION,
                        )
                    ),
                ),
                "last_event": copy.deepcopy(
                    getattr(self, "_agent6_discovery_last_event", None)
                ),
            }

        def set_discovery_frontier_gate_open(self, value: bool) -> None:
            enabled = bool(
                getattr(
                    self,
                    "_agent6_discovery_frontier_gate_enabled",
                    False,
                )
            )
            self._agent6_discovery_frontier_gate_open = bool(
                not enabled or value
            )

        def set_discovery_episode_actor_eligible(self, value: bool) -> None:
            enabled = bool(
                getattr(
                    self,
                    "_agent6_discovery_frontier_gate_enabled",
                    False,
                )
            )
            self._agent6_discovery_episode_actor_eligible = bool(
                not enabled or value
            )

        def _discovery_actor_in_cooldown(self, timestep: int) -> bool:
            return bool(
                getattr(
                    self,
                    "_agent6_discovery_recovery_enabled",
                    False,
                )
                and int(timestep)
                < int(getattr(self, "_agent6_discovery_cooldown_until", 0))
            )

        def _target_policy_smoothing_noise(self, actions: Any) -> Any:
            noise = actions.clone().data.normal_(
                0,
                self.target_policy_noise,
            )
            if (
                getattr(self, "_agent6_legacy_action_contract", False)
                and int(noise.shape[1]) >= AGENT6_LEGACY_ACTION_SIZE
            ):
                longitudinal_noise = noise[:, 1].clone()
                noise[:, 1] = 0.5 * longitudinal_noise
                noise[:, 2] = -0.5 * longitudinal_noise
            return noise.clamp(
                -self.target_noise_clip,
                self.target_noise_clip,
            )

        @staticmethod
        def _clone_module_state(module: Any) -> dict[str, Any]:
            return {
                name: value.detach().clone()
                for name, value in module.state_dict().items()
            }

        def _snapshot_safe_actor_robust_champion(self) -> None:
            """Capture the source champion once; working-policy accepts never replace it."""
            self._agent6_safe_actor_robust_state = self._clone_module_state(
                self.actor
            )
            self._agent6_safe_actor_robust_target_state = (
                self._clone_module_state(self.actor_target)
            )

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
                f"{candidate_id} is policy-equivalent to the working policy; "
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
            self._agent6_safe_actor_reference_laps_completed = []
            self._agent6_safe_actor_candidate_laps_completed = []
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
            updates_per_probe = max(
                1,
                int(getattr(self, "_agent6_safe_actor_block_updates", 1))
                * int(getattr(self, "_agent6_safe_actor_blocks_per_probe", 1)),
            )
            completed_candidate_block = bool(
                int(getattr(self, "_agent6_safe_actor_candidate_updates", 0))
                >= updates_per_probe
            )
            return bool(
                self.safe_actor_probe_required()
                and (
                    completed_candidate_block
                    or getattr(
                        self,
                        "_agent6_safe_actor_evidence_eligible",
                        True,
                    )
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
                self._agent6_safe_actor_evidence_breakthrough_reasons = ()
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
            breakthrough_reasons = training_breakthrough_reasons(
                distance_m,
                reference_maximum_distance_m=(
                    self._agent6_safe_actor_accepted_maximum_distance_m
                ),
                minimum_improvement_m=(
                    self._agent6_safe_actor_breakthrough_improvement_m
                ),
                completed_lap=(
                    self._agent6_safe_actor_evidence_completed_lap
                ),
            )
            new_breakthrough = bool(
                breakthrough_reasons
                and not self._agent6_safe_actor_evidence_breakthrough_reasons
            )
            if breakthrough_reasons:
                if new_breakthrough:
                    self._agent6_safe_actor_breakthrough_candidates += 1
                self._agent6_safe_actor_evidence_breakthrough_reasons = tuple(
                    sorted(
                        {
                            *self._agent6_safe_actor_evidence_breakthrough_reasons,
                            *breakthrough_reasons,
                        }
                    )
                )
                # Preserve the exact frontier-producing actor until its matched
                # internal validation can run within the probe allowance.
                self._agent6_safe_actor_waiting_for_probe = True
                if self._agent6_safe_actor_candidate_state is None:
                    self._snapshot_safe_actor_candidate()
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
                # This gate asks whether the candidate is stable enough to
                # compare. The matched probe still requires real improvement.
                minimum_improvement_m=0.0,
                completed_lap=(
                    self._agent6_safe_actor_evidence_completed_lap
                ),
            )
            updates_per_probe = max(
                1,
                self._agent6_safe_actor_block_updates
                * self._agent6_safe_actor_blocks_per_probe,
            )
            completed_candidate_block = bool(
                self._agent6_safe_actor_candidate_updates >= updates_per_probe
            )
            self._agent6_safe_actor_evidence_eligible = bool(
                self._agent6_safe_actor_evidence_breakthrough_reasons
                or credible
                or completed_candidate_block
                or self._agent6_safe_actor_min_evidence_episodes == 0
            )
            if self._agent6_safe_actor_evidence_eligible:
                evidence_reasons = (
                    self._agent6_safe_actor_evidence_breakthrough_reasons
                    or (("repeatable_progress",) if credible else ())
                    or (
                        ("completed_candidate_block",)
                        if completed_candidate_block
                        else ()
                    )
                )
                return {
                    "candidate_id": candidate_id,
                    "decision": (
                        "probe_eligible_breakthrough"
                        if self._agent6_safe_actor_evidence_breakthrough_reasons
                        else (
                            "probe_eligible"
                            if credible
                            else "probe_eligible_candidate_block"
                        )
                    ),
                    "evidence_reasons": list(evidence_reasons),
                    "new_breakthrough": new_breakthrough,
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
            robust_state = getattr(
                self,
                "_agent6_safe_actor_robust_state",
                {},
            )
            working_state = getattr(self, "_agent6_safe_actor_state", {})
            return {
                "enabled": bool(
                    getattr(self, "_agent6_safe_actor_enabled", False)
                ),
                "robust_champion_role": "immutable_source_actor",
                "working_policy_role": "last_locally_accepted_actor",
                "working_policy_differs_from_robust_champion": bool(
                    robust_state
                    and working_state
                    and not parameter_state_dicts_are_identical(
                        robust_state,
                        working_state,
                    )
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
                "accepted_upper_quartile_distance_m": float(
                    getattr(
                        self,
                        "_agent6_safe_actor_accepted_upper_quartile_distance_m",
                        0.0,
                    )
                ),
                "accepted_maximum_distance_m": float(
                    getattr(
                        self,
                        "_agent6_safe_actor_accepted_maximum_distance_m",
                        0.0,
                    )
                ),
                "accepted_lap_completions": int(
                    getattr(
                        self,
                        "_agent6_safe_actor_accepted_lap_completions",
                        0,
                    )
                ),
                "accepted_milestones_m": list(
                    getattr(
                        self,
                        "_agent6_safe_actor_accepted_milestones_m",
                        (),
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
                "breakthrough_candidates": int(
                    getattr(
                        self,
                        "_agent6_safe_actor_breakthrough_candidates",
                        0,
                    )
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
                reference_laps_completed=(
                    self._agent6_safe_actor_reference_laps_completed[
                        probe_slice
                    ]
                ),
                candidate_laps_completed=(
                    self._agent6_safe_actor_candidate_laps_completed[
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
                median_regression_tolerance_m=(
                    self._agent6_safe_actor_median_regression_tolerance_m
                ),
                minimum_regression_tolerance_m=(
                    self._agent6_safe_actor_minimum_regression_tolerance_m
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
                "upper_quartile_distance_m": (
                    comparison.candidate_upper_quartile_distance_m
                ),
                "maximum_distance_m": (
                    comparison.candidate_maximum_distance_m
                ),
                "reference_median_distance_m": (
                    comparison.reference_median_distance_m
                ),
                "reference_minimum_distance_m": (
                    comparison.reference_minimum_distance_m
                ),
                "reference_upper_quartile_distance_m": (
                    comparison.reference_upper_quartile_distance_m
                ),
                "reference_maximum_distance_m": (
                    comparison.reference_maximum_distance_m
                ),
                "aggregate_median_gain_m": comparison.aggregate_median_gain_m,
                "minimum_gain_m": comparison.minimum_gain_m,
                "upper_quartile_gain_m": comparison.upper_quartile_gain_m,
                "maximum_gain_m": comparison.maximum_gain_m,
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
                "reference_lap_completions": (
                    comparison.reference_lap_completions
                ),
                "candidate_lap_completions": (
                    comparison.candidate_lap_completions
                ),
                "new_progress_milestones_m": list(
                    comparison.new_progress_milestones_m
                ),
                "acceptance_reasons": list(comparison.acceptance_reasons),
                "safety_failures": list(comparison.safety_failures),
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
            self._agent6_safe_actor_reference_laps_completed = []
            self._agent6_safe_actor_candidate_laps_completed = []
            self._agent6_safe_actor_evidence_candidate_id = 0
            self._agent6_safe_actor_evidence_distances_m = []
            self._agent6_safe_actor_evidence_completed_lap = False
            self._agent6_safe_actor_evidence_breakthrough_reasons = ()
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
                self._agent6_safe_actor_accepted_upper_quartile_distance_m = (
                    comparison.candidate_upper_quartile_distance_m
                )
                self._agent6_safe_actor_accepted_maximum_distance_m = max(
                    self._agent6_safe_actor_accepted_maximum_distance_m,
                    comparison.candidate_maximum_distance_m,
                )
                self._agent6_safe_actor_accepted_lap_completions = (
                    comparison.candidate_lap_completions
                )
                self._agent6_safe_actor_accepted_milestones_m = tuple(
                    sorted(
                        {
                            *self._agent6_safe_actor_accepted_milestones_m,
                            *comparison.new_progress_milestones_m,
                        }
                    )
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
                "acceptance_reasons": list(comparison.acceptance_reasons),
                "safety_failures": list(comparison.safety_failures),
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
            laps_completed = max(
                0,
                int(finite_float(summary.get("laps_completed"))),
            )
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
                "upper_quartile_distance_m": None,
                "maximum_distance_m": None,
                "reference_median_distance_m": None,
                "reference_minimum_distance_m": None,
                "reference_upper_quartile_distance_m": None,
                "reference_maximum_distance_m": None,
                "aggregate_median_gain_m": None,
                "minimum_gain_m": None,
                "upper_quartile_gain_m": None,
                "maximum_gain_m": None,
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
                "reference_lap_completions": None,
                "candidate_lap_completions": None,
                "new_progress_milestones_m": None,
                "acceptance_reasons": None,
                "safety_failures": None,
                "required_distance_m": self._agent6_safe_actor_min_improvement_m,
                "median_regression_tolerance_m": (
                    self._agent6_safe_actor_median_regression_tolerance_m
                ),
                "minimum_regression_tolerance_m": (
                    self._agent6_safe_actor_minimum_regression_tolerance_m
                ),
                "maximum_paired_regression_m": (
                    self._agent6_safe_actor_max_paired_regression_m
                ),
            }
            if context["mode"] == "accepted_reference":
                self._agent6_safe_actor_reference_distances_m.append(distance_m)
                self._agent6_safe_actor_reference_order_positions.append(
                    int(context["order_position"])
                )
                self._agent6_safe_actor_reference_laps_completed.append(
                    laps_completed
                )
            else:
                self._agent6_safe_actor_candidate_distances_m.append(distance_m)
                self._agent6_safe_actor_candidate_order_positions.append(
                    int(context["order_position"])
                )
                self._agent6_safe_actor_candidate_laps_completed.append(
                    laps_completed
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
            if getattr(self, "_agent6_standard_td3_updates", False):
                self._apply_training_budget_progress()
                super().train(gradient_steps, batch_size)
                return

            raw_env = getattr(self, "_agent6_episode_atomic_raw_env", None)
            if raw_env is None:
                self._train_from_replay(gradient_steps, batch_size)
                return

            raw_env.suspend_torcs_for_offline_updates()
            self._train_from_replay(gradient_steps, batch_size)
            self._agent6_episode_atomic_batches += 1
            self._reset_after_episode_atomic_update()

        def _train_from_replay(
            self,
            gradient_steps: int,
            batch_size: int = 100,
        ) -> None:
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
            actor_learning_rate = float(actor_schedule(budget_timestep)) * float(
                getattr(
                    self,
                    "_agent6_discovery_actor_learning_rate_scale",
                    1.0,
                )
            )
            update_learning_rate(self.critic.optimizer, critic_learning_rate)
            update_learning_rate(self.actor.optimizer, actor_learning_rate)
            self.logger.record("train/learning_rate", critic_learning_rate)
            self.logger.record("train/critic_learning_rate", critic_learning_rate)
            self.logger.record("train/actor_learning_rate", actor_learning_rate)

            episode_actor_eligible = bool(
                getattr(
                    self,
                    "_agent6_discovery_episode_actor_eligible",
                    True,
                )
            )
            # Eligibility is consumed by one post-episode optimisation batch.
            self._agent6_discovery_episode_actor_eligible = not bool(
                getattr(
                    self,
                    "_agent6_discovery_frontier_gate_enabled",
                    False,
                )
            )

            critic_updates_since_transfer = int(
                getattr(self, "_agent6_critic_updates_since_transfer", 0)
            )
            critic_warmup_updates = int(
                getattr(self, "_agent6_critic_warmup_updates", 0)
            )
            critic_warmup_complete = bool(
                critic_updates_since_transfer >= critic_warmup_updates
            )
            actor_updates_enabled = (
                budget_timestep
                > int(getattr(self, "_agent6_actor_updates_start_at", 0))
                and critic_warmup_complete
                and actor_learning_rate > 0.0
                and not self.safe_actor_probe_required()
                and not self._discovery_actor_in_cooldown(budget_timestep)
                and episode_actor_eligible
            )
            if actor_updates_enabled and getattr(
                self,
                "_agent6_actor_parameters_frozen",
                False,
            ):
                self._set_module_requires_grad(self.actor, True)
                self._agent6_actor_parameters_frozen = False
            effective_policy_delay = int(
                getattr(
                    self,
                    "_agent6_discovery_effective_policy_delay",
                    self.policy_delay,
                )
            )
            self.logger.record(
                "train/actor_updates_enabled",
                int(actor_updates_enabled),
            )
            self.logger.record(
                "train/actor_parameters_frozen",
                int(
                    getattr(
                        self,
                        "_agent6_actor_parameters_frozen",
                        False,
                    )
                ),
            )
            self.logger.record(
                "train/critic_warmup_updates_completed",
                critic_updates_since_transfer,
            )
            self.logger.record(
                "train/effective_policy_delay",
                effective_policy_delay,
            )
            self.logger.record(
                "train/discovery_actor_cooldown",
                int(self._discovery_actor_in_cooldown(budget_timestep)),
            )
            self.logger.record(
                "train/discovery_frontier_gate_open",
                int(episode_actor_eligible),
            )

            actor_losses: list[float] = []
            actor_gradient_norms: list[float] = []
            actor_action_deltas: list[float] = []
            critic_losses: list[float] = []
            critic_gradient_norms: list[float] = []
            q1_means: list[float] = []
            q2_means: list[float] = []
            target_q_means: list[float] = []
            td_error_means: list[float] = []
            td_error_maxima: list[float] = []
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
                    noise = self._target_policy_smoothing_noise(
                        replay_data.actions
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
                huber_beta = float(
                    getattr(
                        self,
                        "_agent6_critic_huber_beta",
                        LEAN_FRONTIER_HUBER_BETA,
                    )
                )
                critic_loss = sum(
                    F.smooth_l1_loss(
                        current_q,
                        target_q_values,
                        beta=huber_beta,
                    )
                    for current_q in current_q_values
                )
                critic_loss_value = float(critic_loss.item())
                critic_update_valid = math.isfinite(critic_loss_value)
                self.critic.optimizer.zero_grad()
                if critic_update_valid:
                    critic_loss.backward()
                    critic_gradient_norm = th.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        float(
                            getattr(
                                self,
                                "_agent6_critic_gradient_norm",
                                LEAN_FRONTIER_GRADIENT_NORM,
                            )
                        ),
                    )
                    critic_gradient_norm_value = float(
                        critic_gradient_norm.item()
                    )
                    critic_update_valid = math.isfinite(
                        critic_gradient_norm_value
                    )
                else:
                    critic_gradient_norm_value = math.nan
                if critic_update_valid:
                    self.critic.optimizer.step()
                    self._agent6_critic_updates_since_transfer = int(
                        getattr(
                            self,
                            "_agent6_critic_updates_since_transfer",
                            0,
                        )
                    ) + 1
                    critic_losses.append(critic_loss_value)
                    critic_gradient_norms.append(critic_gradient_norm_value)
                    q1_means.append(float(current_q_values[0].mean().item()))
                    q2_means.append(float(current_q_values[1].mean().item()))
                    target_q_means.append(float(target_q_values.mean().item()))
                    absolute_td_errors = th.cat(
                        [
                            (current_q.detach() - target_q_values).abs()
                            for current_q in current_q_values
                        ],
                        dim=1,
                    )
                    td_error_means.append(
                        float(absolute_td_errors.mean().item())
                    )
                    td_error_maxima.append(
                        float(absolute_td_errors.max().item())
                    )

                    # Keep target-Q estimates current throughout critic-only
                    # warm-up; the target actor remains the frozen frontier.
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
                else:
                    self.critic.optimizer.zero_grad()

                if self._n_updates % self.policy_delay != 0:
                    continue

                candidate_block_complete = False
                actor_update_due = (
                    self._n_updates % effective_policy_delay == 0
                )
                if (
                    actor_updates_enabled
                    and critic_update_valid
                    and actor_update_due
                    and not self.safe_actor_probe_required()
                ):
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
                    if actor_update_valid:
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
                        if getattr(
                            self,
                            "_agent6_discovery_recovery_enabled",
                            False,
                        ):
                            self._agent6_discovery_actor_updates_since_anchor += 1
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
                            self._agent6_safe_actor_evidence_breakthrough_reasons = ()
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
                            self._snapshot_safe_actor_candidate()
                            self._agent6_safe_actor_waiting_for_probe = True
                            self._agent6_safe_actor_evidence_eligible = True
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
            if critic_gradient_norms:
                self.logger.record(
                    "train/critic_gradient_norm",
                    np.mean(critic_gradient_norms),
                )
            if q1_means:
                self.logger.record("train/q1_mean", np.mean(q1_means))
                self.logger.record("train/q2_mean", np.mean(q2_means))
                self.logger.record(
                    "train/target_q_mean",
                    np.mean(target_q_means),
                )
                self.logger.record(
                    "train/td_error_abs_mean",
                    np.mean(td_error_means),
                )
                self.logger.record(
                    "train/td_error_abs_max",
                    np.max(td_error_maxima),
                )
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
        "train_freq": (
            args.train_freq
            if lean_frontier_training_enabled(args)
            else (args.train_freq, "episode")
        ),
        "gradient_steps": args.gradient_steps,
        "policy_delay": args.policy_delay,
        "target_policy_noise": args.target_policy_noise,
        "target_noise_clip": args.target_noise_clip,
        "action_noise": action_noise,
        "seed": args.seed,
        "device": args.device,
    }
    if args.continuation:
        source_path = continuation_source_path(args)
        if bool(getattr(args, "exact_resume", False)):
            model = TD3.load(str(source_path), **model_kwargs)
            normalise_continuation_spaces(model, env)
            return model
        if (
            continuation_source_uses_legacy_action_contract(args)
            and args.allow_legacy_action_migration
        ):
            legacy_model = TD3.load(str(source_path), device=args.device)
            model = TD3(
                "MlpPolicy",
                policy_kwargs={"net_arch": args.net_arch},
                **model_kwargs,
            )
            migration = migrate_legacy_actor_to_signed_longitudinal(
                legacy_model,
                model,
            )
            migration["source_policy_path"] = str(source_path)
            migration["source_model_num_timesteps"] = int(
                getattr(legacy_model, "num_timesteps", 0)
            )
            model.num_timesteps = 0
            setattr(model, "_agent6_contract_migration", migration)
            setattr(
                model,
                "_agent6_actor_only_transfer",
                {
                    "source_policy_path": str(source_path),
                    "source_model_num_timesteps": migration[
                        "source_model_num_timesteps"
                    ],
                    "fresh_critic": True,
                    "fresh_critic_target": True,
                    "fresh_actor_optimizer": True,
                    "fresh_critic_optimizer": True,
                    "fresh_replay_buffer": True,
                },
            )
            return model
        source_model = TD3.load(str(source_path), device=args.device)
        source_policy_kwargs = copy.deepcopy(
            getattr(source_model, "policy_kwargs", None)
        ) or {"net_arch": getattr(args, "net_arch", [400, 300])}
        model = TD3(
            "MlpPolicy",
            policy_kwargs=source_policy_kwargs,
            **model_kwargs,
        )
        transfer_frontier_actor(source_model, model)
        model.num_timesteps = 0
        setattr(
            model,
            "_agent6_actor_only_transfer",
            {
                "source_policy_path": str(source_path),
                "source_model_num_timesteps": int(
                    getattr(source_model, "num_timesteps", 0)
                ),
                "fresh_critic": True,
                "fresh_critic_target": True,
                "fresh_actor_optimizer": True,
                "fresh_critic_optimizer": True,
                "fresh_replay_buffer": True,
            },
        )
        return model
    return TD3(
        "MlpPolicy",
        policy_kwargs={"net_arch": args.net_arch},
        **model_kwargs,
    )


def transfer_frontier_actor(source_model: Any, model: Any) -> None:
    """Copy only the learned policy into a TD3 model with fresh value networks."""
    source_actor = getattr(source_model, "actor", None)
    actor = getattr(model, "actor", None)
    actor_target = getattr(model, "actor_target", None)
    if source_actor is None or actor is None or actor_target is None:
        raise RuntimeError("frontier continuation requires actor networks")
    source_state = source_actor.state_dict()
    try:
        actor.load_state_dict(source_state, strict=True)
        actor_target.load_state_dict(source_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "frontier actor architecture does not match the continuation model"
        ) from exc


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


def automatic_evaluation_command(
    args: argparse.Namespace,
    *,
    policy_path: Path | None = None,
) -> list[str]:
    evaluation_policy_path = Path(policy_path or args.model_path)
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_td3_agent.py"),
        "--policy-path",
        str(evaluation_policy_path),
        "--track",
        str(args.track),
        "--output-dir",
        str(args.automatic_evaluation_output_dir),
        "--promote-best-if-improved",
        "--best-robust-evaluation-model-path",
        str(args.best_robust_evaluation_model_path),
        "--best-completed-lap-model-path",
        str(args.best_completed_lap_model_path),
    ]


def run_automatic_end_evaluation(
    args: argparse.Namespace,
    *,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    command = automatic_evaluation_command(args, policy_path=policy_path)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )
        return_code = int(completed.returncode)
        error = None
    except OSError as exc:
        return_code = -1
        error = f"{type(exc).__name__}: {exc}"
    return {
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code,
        "succeeded": return_code == 0,
        "error": error,
    }


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
        "--resume-exact-from",
        type=Path,
        default=None,
        help=(
            "Resume an interrupted Patch 5 checkpoint with its replay, "
            "curriculum, probe, optimizer, and RNG state."
        ),
    )
    parser.add_argument(
        "--discovery-mode",
        action="store_true",
        help=(
            "Optimise for new distance frontiers with unrestricted reward-only "
            "TD3 updates; robust promotion remains a separate held-out decision."
        ),
    )
    parser.add_argument(
        "--resume-frontier",
        action="store_true",
        help=(
            "Continue discovery from the canonical frontier, or bootstrap it "
            "from the robust champion when no frontier exists yet."
        ),
    )
    parser.add_argument(
        "--robustify-frontier",
        action="store_true",
        help=(
            "Continue from a completed-lap frontier under the strict safe-actor "
            "protocol to improve repeatability and pace."
        ),
    )
    parser.add_argument(
        "--extend-training-steps",
        type=int,
        default=0,
        help=(
            "Additional non-probe interactions to append to an exact-resume "
            "checkpoint's saved target. Its 80/20 probe ratio is preserved."
        ),
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
        help="Maximum normalized action deviation from the working policy.",
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
        help=(
            "Required gain in at least one progress metric before a safe "
            "candidate can become the working policy."
        ),
    )
    parser.add_argument(
        "--breakthrough-improvement-m",
        type=float,
        default=DEFAULT_BREAKTHROUGH_IMPROVEMENT_M,
        help=(
            "Single-episode frontier gain that freezes a candidate immediately "
            "for matched validation. Crossing a curriculum milestone also "
            "qualifies."
        ),
    )
    parser.add_argument(
        "--safe-actor-median-regression-tolerance-m",
        type=float,
        default=DEFAULT_SAFE_ACTOR_MEDIAN_REGRESSION_TOLERANCE_M,
        help=(
            "Small median-distance regression allowed when another progress "
            "metric improves."
        ),
    )
    parser.add_argument(
        "--safe-actor-minimum-regression-tolerance-m",
        type=float,
        default=DEFAULT_SAFE_ACTOR_MINIMUM_REGRESSION_TOLERANCE_M,
        help=(
            "Minimum-distance regression allowed before a candidate is "
            "classified as unsafe."
        ),
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
    parser.add_argument(
        "--best-robust-evaluation-model-path",
        type=Path,
        default=DEFAULT_BEST_ROBUST_EVALUATION_MODEL_PATH,
        help="Canonical held-out robustness champion.",
    )
    parser.add_argument(
        "--best-completed-lap-model-path",
        type=Path,
        default=DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH,
        help="Best valid completed-lap policy found during training or evaluation.",
    )
    parser.add_argument(
        "--frontier-model-path",
        type=Path,
        default=DEFAULT_FRONTIER_MODEL_PATH,
        help="Canonical greatest-distance policy, independent of robustness status.",
    )
    parser.add_argument(
        "--frontier-checkpoint-dir",
        type=Path,
        default=DEFAULT_FRONTIER_CHECKPOINT_DIR,
        help="Milestone snapshots for frontier discovery.",
    )
    parser.add_argument(
        "--frontier-evidence-path",
        type=Path,
        default=DEFAULT_FRONTIER_EVIDENCE_PATH,
        help="Audited distance evidence bound to an exact historical policy hash.",
    )
    parser.add_argument(
        "--frontier-milestones-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRONTIER_MILESTONES_M),
        help="Distances that trigger preserved frontier snapshots and evaluation.",
    )
    parser.add_argument(
        "--frontier-min-improvement-m",
        type=float,
        default=DEFAULT_FRONTIER_MIN_IMPROVEMENT_M,
        help="Minimum non-milestone distance gain needed to replace the frontier.",
    )
    parser.add_argument(
        "--frontier-major-breakthrough-m",
        type=float,
        default=DEFAULT_FRONTIER_MAJOR_BREAKTHROUGH_M,
        help="Distance gain that triggers strict held-out evaluation after discovery.",
    )
    parser.add_argument(
        "--frontier-qualification-distance-m",
        type=float,
        default=DEFAULT_FRONTIER_QUALIFICATION_DISTANCE_M,
        help=(
            "Distance the frozen source actor must reproduce deterministically "
            "before a fresh frontier-discovery continuation may train."
        ),
    )
    parser.add_argument(
        "--frontier-qualification-attempts",
        type=int,
        default=DEFAULT_FRONTIER_QUALIFICATION_ATTEMPTS,
        help=(
            "Maximum frozen deterministic source-policy attempts used by the "
            "frontier admission gate."
        ),
    )
    parser.add_argument(
        "--frontier-exploration-start-distance-m",
        type=float,
        default=DEFAULT_FRONTIER_EXPLORATION_START_DISTANCE_M,
        help=(
            "Per-episode distance before discovery exploration and actor updates "
            "are enabled; the critic may still learn from the preserved approach."
        ),
    )
    parser.add_argument(
        "--discovery-collapse-episodes",
        type=int,
        default=DEFAULT_DISCOVERY_COLLAPSE_EPISODES,
        help="Consecutive failed launches before restoring the frontier actor.",
    )
    parser.add_argument(
        "--discovery-collapse-distance-m",
        type=float,
        default=DEFAULT_DISCOVERY_COLLAPSE_DISTANCE_M,
        help="Episodes below this distance count as failed discovery launches.",
    )
    parser.add_argument(
        "--discovery-frontier-preservation-fraction",
        type=float,
        default=DEFAULT_DISCOVERY_FRONTIER_PRESERVATION_FRACTION,
        help=(
            "Fraction of the frontier gate that updated actors must preserve "
            "across the collapse window before the source actor is restored."
        ),
    )
    parser.add_argument(
        "--discovery-recovery-cooldown-steps",
        type=int,
        default=DEFAULT_DISCOVERY_RECOVERY_COOLDOWN_STEPS,
        help="Initial critic-only recovery window after launch collapse.",
    )
    parser.add_argument(
        "--discovery-max-recovery-cooldown-steps",
        type=int,
        default=DEFAULT_DISCOVERY_MAX_RECOVERY_COOLDOWN_STEPS,
        help="Maximum exponentially backed-off recovery window.",
    )
    parser.add_argument(
        "--discovery-actor-lr-backoff",
        type=float,
        default=DEFAULT_DISCOVERY_ACTOR_LR_BACKOFF,
        help="Actor learning-rate multiplier applied after each recovery.",
    )
    parser.add_argument(
        "--discovery-min-actor-lr-scale",
        type=float,
        default=DEFAULT_DISCOVERY_MIN_ACTOR_LR_SCALE,
        help="Minimum cumulative actor learning-rate scale after recoveries.",
    )
    parser.add_argument(
        "--discovery-max-policy-delay",
        type=int,
        default=DEFAULT_DISCOVERY_MAX_POLICY_DELAY,
        help="Largest delayed-policy-update interval after repeated collapse.",
    )
    parser.add_argument(
        "--lap-checkpoint-dir",
        type=Path,
        default=DEFAULT_LAP_CHECKPOINT_DIR,
        help="Immutable archive directory for every valid policy-controlled lap.",
    )
    parser.add_argument(
        "--end-of-run-evaluation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the strict held-out evaluator once after training and promote "
            "eligible robust/lap champions. Enabled by default for continuation."
        ),
    )
    parser.add_argument(
        "--automatic-evaluation-output-dir",
        type=Path,
        default=DEFAULT_AUTOMATIC_EVALUATION_OUTPUT_DIR,
        help="JSON and CSV destination for automatic held-out evaluation.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument(
        "--emergency-checkpoint-path",
        type=Path,
        default=DEFAULT_EMERGENCY_CHECKPOINT_PATH,
        help="Exact-resume checkpoint written on Ctrl+C or a TORCS failure.",
    )
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
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=DEFAULT_EPISODE_GRADIENT_STEPS,
        help="Replay updates performed after each completed training episode.",
    )
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
    end_evaluation_was_explicit = args.end_of_run_evaluation is not None
    compatibility_probe_was_explicit = (
        args.continuation_compatibility_probe is not None
    )
    safe_actor_was_explicit = args.safe_actor_improvement is not None
    args.lean_frontier_training = bool(args.resume_frontier)
    args.stable_frontier_exact_resume = False
    if args.resume_frontier and args.robustify_frontier:
        parser.error(
            "--resume-frontier and --robustify-frontier are mutually exclusive"
        )
    if args.resume_frontier or args.robustify_frontier:
        if args.resume_from is not None or args.resume_exact_from is not None:
            parser.error(
                "frontier continuation flags are mutually exclusive with "
                "--resume-from and --resume-exact-from"
            )
        if args.robustify_frontier:
            if not args.frontier_model_path.is_file():
                parser.error(
                    "--robustify-frontier requires an existing frontier checkpoint"
                )
            frontier_metadata = read_policy_metadata(args.frontier_model_path)
            frontier_champion = frontier_metadata.get("frontier_champion")
            if not (
                isinstance(frontier_champion, Mapping)
                and frontier_champion.get("completed_lap")
            ):
                parser.error(
                    "--robustify-frontier requires a completed-lap frontier"
                )
            args.resume_from = args.frontier_model_path
        else:
            args.discovery_mode = True
            args.resume_from = (
                args.frontier_model_path
                if args.frontier_model_path.is_file()
                else args.best_robust_evaluation_model_path
            )
    if args.resume_from is not None and args.resume_exact_from is not None:
        parser.error("--resume-from and --resume-exact-from are mutually exclusive")
    args.exact_resume = args.resume_exact_from is not None
    args.continuation = bool(args.resume_from or args.resume_exact_from)
    args.end_of_run_evaluation_was_explicit = end_evaluation_was_explicit
    args.frontier_bootstrap_source = bool(
        args.resume_frontier
        and Path(args.resume_from).resolve()
        != args.frontier_model_path.resolve()
    )
    args.frontier_training_source = bool(
        args.resume_frontier or args.robustify_frontier
    )
    args.exact_source_was_continuation = False
    args.exact_resume_budget_extension = None
    args.frontier_verified_evidence = {}
    if args.extend_training_steps < 0:
        parser.error("--extend-training-steps cannot be negative")
    if args.extend_training_steps and not args.exact_resume:
        parser.error("--extend-training-steps requires --resume-exact-from")
    if args.exact_resume:
        exact_errors = exact_training_checkpoint_errors(args.resume_exact_from)
        if exact_errors:
            parser.error(exact_errors[0])
        try:
            exact_state = load_exact_training_state(args.resume_exact_from)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        budget_state = exact_state.get("training_budget")
        if not isinstance(budget_state, Mapping):
            parser.error("exact-resume state is missing its training budget")
        try:
            extended_budget = resolve_exact_resume_budget(
                budget_state,
                additional_training_steps=args.extend_training_steps,
                requested_maximum_probe_steps=args.max_probe_overhead_steps,
            )
        except ValueError as exc:
            parser.error(str(exc))
        args.exact_resume_budget_extension = extended_budget
        args.total_timesteps = int(extended_budget["target_training_steps"])
        args.max_probe_overhead_steps = int(
            extended_budget["maximum_probe_steps"]
        )
        args.max_probe_fraction = float(
            extended_budget["maximum_probe_fraction"]
        )
        environment_state = exact_state.get("environment")
        if isinstance(environment_state, Mapping):
            args.start_stage = str(
                environment_state.get("stage_id") or "sector_progression"
            )
        source_metadata = read_policy_metadata(args.resume_exact_from)
        source_hyperparameters = source_metadata.get("td3_hyperparameters")
        args.stable_frontier_exact_resume = bool(
            isinstance(source_hyperparameters, Mapping)
            and source_hyperparameters.get("optimizer_protocol")
            == "stable_frontier_td3_v1"
        )
        saved_discovery = source_metadata.get("discovery")
        saved_discovery_mode = bool(
            isinstance(saved_discovery, Mapping)
            and saved_discovery.get("enabled")
        )
        if args.discovery_mode and not saved_discovery_mode:
            parser.error(
                "--discovery-mode exact resume requires a discovery checkpoint"
            )
        args.discovery_mode = saved_discovery_mode
        if saved_discovery_mode and not args.stable_frontier_exact_resume:
            model_state = exact_state.get("model")
            recovery_state = (
                model_state.get("discovery_recovery")
                if isinstance(model_state, Mapping)
                else None
            )
            required_recovery_fields = {
                "_agent6_discovery_anchor_state",
                "_agent6_discovery_anchor_target_state",
                "_agent6_discovery_anchor_optimizer_state",
                "_agent6_discovery_recovery_enabled",
                "_agent6_discovery_frontier_gate_enabled",
            }
            if not (
                isinstance(recovery_state, Mapping)
                and required_recovery_fields.issubset(recovery_state)
            ):
                parser.error(
                    "discovery exact-resume checkpoint predates the frontier "
                    "trajectory gate or collapse recovery; "
                    "restart safely with --resume-frontier"
                )
            args.frontier_training_source = True
            trajectory_gate = saved_discovery.get("trajectory_gate")
            if isinstance(trajectory_gate, Mapping):
                args.frontier_qualification_distance_m = finite_float(
                    trajectory_gate.get("qualification_distance_m"),
                    args.frontier_qualification_distance_m,
                )
                args.frontier_exploration_start_distance_m = finite_float(
                    trajectory_gate.get("exploration_start_distance_m"),
                    args.frontier_exploration_start_distance_m,
                )
        elif saved_discovery_mode:
            args.frontier_training_source = True
        hyperparameters = source_metadata.get("td3_hyperparameters")
        if isinstance(hyperparameters, Mapping):
            exact_argument_keys = {
                "buffer_size": "buffer_size",
                "batch_size": "batch_size",
                "gamma": "gamma",
                "tau": "tau",
                "learning_rate": "learning_rate",
                "learning_rate_final": "learning_rate_final",
                "actor_learning_rate": "actor_learning_rate",
                "actor_learning_rate_final": "actor_learning_rate_final",
                "critic_warmup_steps": "critic_warmup_steps",
                "actor_unfreeze_steps": "actor_unfreeze_steps",
                "safe_actor_improvement": "safe_actor_improvement",
                "train_freq": "train_freq",
                "gradient_steps": "gradient_steps",
                "policy_delay": "policy_delay",
                "target_policy_noise": "target_policy_noise",
                "target_noise_clip": "target_noise_clip",
                "action_noise_initial_sigma": "action_noise_sigma",
                "action_noise_final_sigma": "action_noise_final_sigma",
                "action_noise_decay_steps": "action_noise_decay_steps",
                "net_arch": "net_arch",
            }
            for metadata_key, argument_key in exact_argument_keys.items():
                if hyperparameters.get(metadata_key) is not None:
                    setattr(args, argument_key, hyperparameters[metadata_key])
        continuation = source_metadata.get("continuation")
        if isinstance(continuation, Mapping):
            args.exact_source_was_continuation = bool(
                continuation.get(
                    "training_protocol_enabled",
                    continuation.get("enabled"),
                )
            )
            args.replay_refill_steps = int(
                finite_float(continuation.get("replay_refill_steps"))
            )
        safe_actor = source_metadata.get("safe_actor_improvement")
        if isinstance(safe_actor, Mapping):
            safe_argument_keys = {
                "max_normalized_action_delta": "safe_actor_max_action_delta",
                "gradient_clip_norm": "safe_actor_gradient_norm",
                "actor_updates_per_inner_block": "safe_actor_block_updates",
                "inner_blocks_per_robustness_probe": (
                    "safe_actor_blocks_per_probe"
                ),
                "robustness_challenge_seeds": "safe_actor_probe_repeats",
                "robustness_trials_per_seed": "safe_actor_trials_per_seed",
                "screening_trials_per_seed": (
                    "safe_actor_screening_trials_per_seed"
                ),
                "screening_maximum_regression_m": (
                    "safe_actor_screening_max_regression_m"
                ),
                "screening_maximum_episode_steps": (
                    "safe_actor_screening_max_episode_steps"
                ),
                "training_challenge_base_seed": "safe_actor_probe_base_seed",
                "evidence_window_episodes": "probe_evidence_window",
                "minimum_evidence_episodes": "probe_min_evidence_episodes",
                "evidence_maximum_median_regression_m": (
                    "probe_evidence_max_regression_m"
                ),
                "robustness_probe_steering_noise_std": (
                    "safe_actor_probe_noise_std"
                ),
                "minimum_improvement_m": "safe_actor_min_improvement_m",
                "breakthrough_improvement_m": (
                    "breakthrough_improvement_m"
                ),
                "median_regression_tolerance_m": (
                    "safe_actor_median_regression_tolerance_m"
                ),
                "minimum_regression_tolerance_m": (
                    "safe_actor_minimum_regression_tolerance_m"
                ),
                "maximum_paired_regression_m": (
                    "safe_actor_max_paired_regression_m"
                ),
                "anchor_batch_size": "safe_actor_anchor_batch_size",
            }
            for metadata_key, argument_key in safe_argument_keys.items():
                if safe_actor.get(metadata_key) is not None:
                    setattr(args, argument_key, safe_actor[metadata_key])
        args.seed = int(finite_float(source_metadata.get("seed"), args.seed))
        args.policy_probe_interval = int(
            finite_float(
                source_metadata.get("policy_probe_interval_episodes"),
                args.policy_probe_interval,
            )
        )
        args.continuation_compatibility_probe = False
    if args.continuation_compatibility_probe_only:
        args.continuation_compatibility_probe = True
    if args.continuation_compatibility_probe is None:
        args.continuation_compatibility_probe = bool(
            args.continuation and not args.discovery_mode
        )
    args.frontier_qualification_enabled = bool(
        args.discovery_mode
        and not args.exact_resume
        and not lean_frontier_training_enabled(args)
    )
    if args.end_of_run_evaluation is None:
        args.end_of_run_evaluation = bool(
            args.continuation
            and not args.continuation_compatibility_probe_only
        )
    if args.start_stage is None:
        args.start_stage = (
            "full_lap"
            if args.discovery_mode or args.robustify_frontier
            else "sector_progression"
            if args.continuation
            else "launch"
        )
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
    if lean_frontier_training_enabled(args):
        # Preserve the frontier policy while rebuilding the value side under the
        # current reward contract before any actor gradient is allowed through.
        args.replay_refill_steps = LEAN_FRONTIER_REPLAY_REFILL_STEPS
        args.critic_warmup_steps = LEAN_FRONTIER_CRITIC_WARMUP_STEPS
        args.actor_unfreeze_steps = 0
        args.learning_rate = LEAN_FRONTIER_CRITIC_LEARNING_RATE
        args.learning_rate_final = LEAN_FRONTIER_CRITIC_LEARNING_RATE
        args.actor_learning_rate = LEAN_FRONTIER_ACTOR_LEARNING_RATE
        args.actor_learning_rate_final = LEAN_FRONTIER_ACTOR_LEARNING_RATE
        args.action_noise_sigma = LEAN_FRONTIER_ACTION_NOISE_SIGMA
        args.action_noise_final_sigma = LEAN_FRONTIER_FINAL_ACTION_NOISE_SIGMA
        args.target_policy_noise = LEAN_FRONTIER_TARGET_POLICY_NOISE
        args.target_noise_clip = LEAN_FRONTIER_TARGET_NOISE_CLIP
        args.gradient_steps = LEAN_FRONTIER_GRADIENT_STEPS
    elif args.stable_frontier_exact_resume:
        # Exact resume restores these values and optimizer counters from the
        # checkpoint; enable the compact class without resetting its phases.
        args.lean_frontier_training = True
    if args.save_best_replay_buffer is None:
        args.save_best_replay_buffer = args.continuation
    if args.safe_actor_improvement is None:
        args.safe_actor_improvement = args.continuation
    if args.discovery_mode:
        if safe_actor_was_explicit and args.safe_actor_improvement:
            parser.error(
                "--discovery-mode cannot use --safe-actor-improvement; frontier "
                "checkpoints protect breakthroughs without rolling actors back"
            )
        args.safe_actor_improvement = False
    if args.discovery_mode and not args.continuation:
        parser.error("--discovery-mode requires --resume-frontier or --resume-from")
    if (
        not math.isfinite(args.frontier_min_improvement_m)
        or args.frontier_min_improvement_m < 0.0
    ):
        parser.error("--frontier-min-improvement-m must be finite and non-negative")
    if (
        not math.isfinite(args.frontier_major_breakthrough_m)
        or args.frontier_major_breakthrough_m < args.frontier_min_improvement_m
    ):
        parser.error(
            "--frontier-major-breakthrough-m must be finite and no smaller than "
            "--frontier-min-improvement-m"
        )
    if (
        not math.isfinite(args.frontier_qualification_distance_m)
        or args.frontier_qualification_distance_m <= 0.0
    ):
        parser.error(
            "--frontier-qualification-distance-m must be positive and finite"
        )
    if args.frontier_qualification_attempts < 1:
        parser.error("--frontier-qualification-attempts must be at least 1")
    if (
        not math.isfinite(args.frontier_exploration_start_distance_m)
        or args.frontier_exploration_start_distance_m <= 0.0
    ):
        parser.error(
            "--frontier-exploration-start-distance-m must be positive and finite"
        )
    if (
        args.frontier_exploration_start_distance_m
        > args.frontier_qualification_distance_m
    ):
        parser.error(
            "--frontier-exploration-start-distance-m cannot exceed the "
            "qualification distance"
        )
    if args.discovery_collapse_episodes < 2:
        parser.error("--discovery-collapse-episodes must be at least 2")
    if (
        not math.isfinite(args.discovery_collapse_distance_m)
        or args.discovery_collapse_distance_m <= 0.0
    ):
        parser.error(
            "--discovery-collapse-distance-m must be positive and finite"
        )
    if not 0.0 < args.discovery_frontier_preservation_fraction <= 1.0:
        parser.error(
            "--discovery-frontier-preservation-fraction must be in (0, 1]"
        )
    if args.discovery_recovery_cooldown_steps < 1:
        parser.error(
            "--discovery-recovery-cooldown-steps must be at least 1"
        )
    if (
        args.discovery_max_recovery_cooldown_steps
        < args.discovery_recovery_cooldown_steps
    ):
        parser.error(
            "--discovery-max-recovery-cooldown-steps cannot be smaller than "
            "--discovery-recovery-cooldown-steps"
        )
    if not 0.0 < args.discovery_actor_lr_backoff <= 1.0:
        parser.error("--discovery-actor-lr-backoff must be in (0, 1]")
    if not 0.0 < args.discovery_min_actor_lr_scale <= 1.0:
        parser.error("--discovery-min-actor-lr-scale must be in (0, 1]")
    if args.policy_delay < 1:
        parser.error("--policy-delay must be at least 1")
    if args.discovery_max_policy_delay < args.policy_delay:
        parser.error(
            "--discovery-max-policy-delay cannot be smaller than --policy-delay"
        )
    if args.discovery_max_policy_delay % args.policy_delay != 0:
        parser.error(
            "--discovery-max-policy-delay must be a multiple of --policy-delay"
        )
    if not args.frontier_milestones_m or any(
        not math.isfinite(value) or value <= 0.0
        for value in args.frontier_milestones_m
    ):
        parser.error("--frontier-milestones-m must contain positive finite values")
    args.frontier_milestones_m = tuple(
        sorted({float(value) for value in args.frontier_milestones_m})
    )
    if compatibility_probe_was_explicit and args.discovery_mode:
        args.continuation_compatibility_probe = bool(
            args.continuation_compatibility_probe
        )
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
        parser.error("--start-stage requires continuation unless it is launch")
    if args.continuation_compatibility_probe and not args.continuation:
        parser.error("--continuation-compatibility-probe requires --resume-from")
    if args.continuation_compatibility_probe_only and not args.continuation:
        parser.error(
            "--continuation-compatibility-probe-only requires --resume-from"
        )
    if args.allow_legacy_action_migration and not args.continuation:
        parser.error("--allow-legacy-action-migration requires --resume-from")
    managed_policy_paths = (
        args.model_path.resolve(),
        args.best_distance_model_path.resolve(),
        args.best_reward_model_path.resolve(),
        args.best_robust_evaluation_model_path.resolve(),
        args.best_completed_lap_model_path.resolve(),
        args.frontier_model_path.resolve(),
        args.emergency_checkpoint_path.resolve(),
    )
    if len(set(managed_policy_paths)) != len(managed_policy_paths):
        parser.error(
            "working, distance, reward, robust, completed-lap, frontier, and emergency "
            "policy "
            "paths must be distinct"
        )
    if args.continuation:
        if not args.exact_resume:
            if args.frontier_training_source:
                (
                    args.frontier_verified_evidence,
                    frontier_evidence_errors,
                ) = load_verified_frontier_evidence(
                    args.resume_from,
                    args.frontier_evidence_path,
                )
                if frontier_evidence_errors:
                    parser.error(frontier_evidence_errors[0])
            source_errors = (
                frontier_continuation_source_errors(
                    args.resume_from,
                    track=args.track,
                    start_stage=args.start_stage,
                    external_evidence=args.frontier_verified_evidence,
                )
                if args.frontier_training_source
                else continuation_source_errors(
                    args.resume_from,
                    track=args.track,
                    start_stage=args.start_stage,
                )
            )
            if source_errors:
                parser.error(source_errors[0])
            if args.frontier_qualification_enabled:
                source_frontier_distance_m = frontier_evidence_distance_m(
                    read_policy_metadata(args.resume_from),
                    args.frontier_verified_evidence,
                )
                if (
                    source_frontier_distance_m
                    < args.frontier_qualification_distance_m
                ):
                    parser.error(
                        "frontier qualification requires source evidence of at "
                        f"least {args.frontier_qualification_distance_m:.1f}m; "
                        f"the exact policy has {source_frontier_distance_m:.1f}m"
                    )
        if (
            args.allow_legacy_action_migration
            and not continuation_source_uses_legacy_action_contract(args)
        ):
            parser.error(
                "--allow-legacy-action-migration requires a verified legacy v1 source"
            )
        source_path = continuation_source_path(args).resolve()
        if (
            not args.exact_resume
            and not args.frontier_training_source
            and not args.allow_non_champion_source
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
            args.best_completed_lap_model_path.resolve(),
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
        not math.isfinite(args.breakthrough_improvement_m)
        or args.breakthrough_improvement_m < 0.0
    ):
        parser.error(
            "--breakthrough-improvement-m must be finite and non-negative"
        )
    if (
        not math.isfinite(args.safe_actor_median_regression_tolerance_m)
        or args.safe_actor_median_regression_tolerance_m < 0.0
    ):
        parser.error(
            "--safe-actor-median-regression-tolerance-m must be finite and "
            "non-negative"
        )
    if (
        not math.isfinite(args.safe_actor_minimum_regression_tolerance_m)
        or args.safe_actor_minimum_regression_tolerance_m < 0.0
    ):
        parser.error(
            "--safe-actor-minimum-regression-tolerance-m must be finite and "
            "non-negative"
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
        parser.error(
            "--train-freq must be 1 so optimisation remains episode-atomic"
        )
    if args.gradient_steps < 1:
        parser.error("--gradient-steps must be at least 1")
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
    ) = import_training_dependencies()
    TrainingTD3 = (
        make_stable_frontier_td3_class(TD3)
        if lean_frontier_training_enabled(args)
        else make_phased_continuation_td3_class(TD3)
    )

    run_dir = args.run_dir if args.run_dir is not None else make_default_run_dir()
    args.run_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_distance_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_reward_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_robust_evaluation_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.best_completed_lap_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.frontier_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.frontier_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.lap_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.automatic_evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.emergency_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
            "learning"
            if args.exact_resume
            else "replay_refill"
            if args.continuation
            else "warmup"
        ),
        use_custom_warmup_action_space=(
            not continuation_training_protocol_enabled(args)
        ),
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
            "discovery_frontier_gate_open",
            "discovery_frontier_gate_distance_m",
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
    uses_legacy_action_contract = training_uses_legacy_action_contract(args)

    action_noise_generation = 0

    def next_action_noise_seed() -> int:
        nonlocal action_noise_generation
        seed = training_action_noise_seed(args.seed, action_noise_generation)
        action_noise_generation += 1
        return seed

    def make_action_noise(sigma: float):
        seed = next_action_noise_seed()
        if uses_legacy_action_contract:
            return SeededLegacyActionNoise(
                seed=seed,
                standard_deviation=max(0.0, float(sigma)),
            )
        return SeededGaussianActionNoise(
            seed=seed,
            standard_deviation=max(0.0, float(sigma)),
            action_size=action_size,
        )

    def make_pre_actor_noise(sigma: float):
        return SeededSteeringActionNoise(
            seed=next_action_noise_seed(),
            steering_noise_std=max(0.0, float(sigma)),
            action_size=action_size,
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
        if not lean_frontier_training_enabled(args):
            model.configure_episode_atomic_updates(raw_env)
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
            safe_actor_median_regression_tolerance_m=(
                args.safe_actor_median_regression_tolerance_m
            ),
            safe_actor_minimum_regression_tolerance_m=(
                args.safe_actor_minimum_regression_tolerance_m
            ),
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
            safe_actor_breakthrough_improvement_m=(
                args.breakthrough_improvement_m
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
            source_verified_maximum_distance_m=finite_float(
                max(
                    finite_float(
                        metadata["continuation"].get(
                            "source_verified_evaluation_maximum_progress_m"
                        )
                    ),
                    finite_float(
                        metadata["discovery"].get(
                            "source_frontier_evidence_distance_m"
                        )
                    ),
                )
            ),
            legacy_action_contract=uses_legacy_action_contract,
            discovery_recovery_enabled=(
                args.discovery_mode
                and not lean_frontier_training_enabled(args)
            ),
            discovery_collapse_episodes=args.discovery_collapse_episodes,
            discovery_collapse_distance_m=args.discovery_collapse_distance_m,
            discovery_recovery_cooldown_steps=(
                args.discovery_recovery_cooldown_steps
            ),
            discovery_max_recovery_cooldown_steps=(
                args.discovery_max_recovery_cooldown_steps
            ),
            discovery_actor_lr_backoff=args.discovery_actor_lr_backoff,
            discovery_min_actor_lr_scale=args.discovery_min_actor_lr_scale,
            discovery_max_policy_delay=args.discovery_max_policy_delay,
            discovery_frontier_gate_enabled=(
                args.discovery_mode
                and not lean_frontier_training_enabled(args)
            ),
            discovery_frontier_gate_distance_m=(
                args.frontier_exploration_start_distance_m
            ),
            discovery_frontier_preservation_fraction=(
                args.discovery_frontier_preservation_fraction
            ),
            critic_warmup_updates=(
                args.critic_warmup_steps
                if lean_frontier_training_enabled(args)
                else 0
            ),
            critic_gradient_norm=LEAN_FRONTIER_GRADIENT_NORM,
            critic_huber_beta=LEAN_FRONTIER_HUBER_BETA,
            freeze_actor_before_updates=bool(
                lean_frontier_training_enabled(args)
            ),
        )
        actor_transfer = getattr(model, "_agent6_actor_only_transfer", None)
        metadata["continuation"]["actor_only_transfer_result"] = (
            dict(actor_transfer) if isinstance(actor_transfer, Mapping) else None
        )
        metadata["continuation"]["source_model_num_timesteps"] = int(
            finite_float(
                actor_transfer.get("source_model_num_timesteps")
                if isinstance(actor_transfer, Mapping)
                else getattr(model, "num_timesteps", 0)
            )
        )
        contract_migration = getattr(model, "_agent6_contract_migration", None)
        if isinstance(contract_migration, Mapping):
            metadata["continuation"][
                "source_policy_contract_migration"
            ] = dict(contract_migration)
        if args.frontier_qualification_enabled:
            print(
                "Qualifying frozen frontier actor before training; "
                f"required distance={args.frontier_qualification_distance_m:.1f}m, "
                f"attempts={args.frontier_qualification_attempts}."
            )
            try:
                qualification_result = run_frontier_source_qualification(
                    model,
                    raw_env,
                    minimum_distance_m=(
                        args.frontier_qualification_distance_m
                    ),
                    maximum_attempts=args.frontier_qualification_attempts,
                    training_budget_state=training_budget_state,
                    base_seed=args.seed,
                )
            except Exception:
                env.close()
                raise
            metadata["discovery"]["trajectory_gate"][
                "qualification_result"
            ] = qualification_result
            write_json(
                run_dir / "frontier_source_qualification.json",
                qualification_result,
            )
            print(
                "Frozen frontier actor qualified: "
                f"{qualification_result['best_distance_m']:.1f}m; "
                "replay and gradient training may now begin."
            )
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
        protocol_description = (
            "stable frontier fine-tuning=on; actor-only checkpoint transfer; "
            "fresh critics and replay; Smooth L1 critic loss; clipped actor/critic "
            "gradients; actor frozen through critic warm-up; qualification, "
            "trajectory gate, probes, rollback, and recovery controller=off; "
            f"frontier={args.frontier_model_path}; milestones="
            + ",".join(f"{value:.0f}m" for value in args.frontier_milestones_m)
            if lean_frontier_training_enabled(args)
            else "discovery mode=on; frozen source qualification=on; "
            "noise/actor gate="
            f"{args.frontier_exploration_start_distance_m:.0f}m; "
            "routine robustness probes=off; actor rollback=off; "
            f"frontier={args.frontier_model_path}; milestones="
            + ",".join(f"{value:.0f}m" for value in args.frontier_milestones_m)
            if args.discovery_mode
            else (
                f"safe actor blocks={'on' if args.safe_actor_improvement else 'off'}; "
                "actor updates per robustness probe="
                f"{args.safe_actor_block_updates * args.safe_actor_blocks_per_probe:,}; "
                f"robust probes={args.safe_actor_probe_repeats} seeds x "
                f"{args.safe_actor_trials_per_seed} trials x 2 policies per candidate, "
                f"with a {args.safe_actor_screening_trials_per_seed}-trial staged screen"
            )
        )
        update_description = (
            f"live phased updates={args.gradient_steps:,} gradient/environment step"
            if lean_frontier_training_enabled(args)
            else "offline episode-atomic updates="
            f"{args.gradient_steps:,} gradients/episode"
        )
        print(
            (
                "Resuming exact Agent 6 training state from "
                if args.exact_resume
                else "Continuing Agent 6 frontier discovery from "
                if args.discovery_mode
                else "Robustifying Agent 6 completed-lap frontier from "
                if args.robustify_frontier
                else "Continuing verified Agent 6 policy from "
            )
            + f"{continuation_source_path(args)} at stage={args.start_stage}; "
            f"action contract={metadata['action_version']}; "
            f"replay refill={args.replay_refill_steps:,}, "
            f"critic warm-up={args.critic_warmup_steps:,}, "
            f"actor unfreeze={args.actor_unfreeze_steps:,} steps; "
            f"{update_description}; "
            f"{protocol_description}; "
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
    LapCheckpointCallback = make_lap_checkpoint_callback_class(BaseCallback)
    FrontierCheckpointCallback = make_frontier_checkpoint_callback_class(
        BaseCallback
    )
    EpisodeSummaryCallback = make_episode_summary_callback_class(BaseCallback)
    BestModelCallback = make_best_model_callback_class(BaseCallback)
    qualification_metadata = metadata["discovery"]["trajectory_gate"].get(
        "qualification_result"
    )
    frozen_policy_baseline_distance_m = (
        finite_float(qualification_metadata.get("best_distance_m"))
        if isinstance(qualification_metadata, Mapping)
        else 0.0
    )
    exploration_callback = ExplorationScheduleCallback(
        raw_env,
        make_action_noise,
        learning_starts=gradient_update_start_timestep(args),
        initial_sigma=args.action_noise_sigma,
        final_sigma=args.action_noise_final_sigma,
        decay_steps=args.action_noise_decay_steps,
        probe_interval=args.policy_probe_interval,
        training_gradient_steps=args.gradient_steps,
        pre_update_training_phase=(
            "learning"
            if args.exact_resume
            else "replay_refill"
            if args.continuation
            else "warmup"
        ),
        actor_update_start=actor_update_start_timestep(args),
        actor_full_unfreeze=actor_full_unfreeze_timestep(args),
        pre_actor_noise_factory=(
            make_pre_actor_noise
            if args.continuation and not lean_frontier_training_enabled(args)
            else None
        ),
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
        regular_probes_enabled=(
            not lean_frontier_training_enabled(args) and not args.discovery_mode
        ),
        frontier_exploration_start_distance_m=(
            args.frontier_exploration_start_distance_m
            if args.discovery_mode and not lean_frontier_training_enabled(args)
            else 0.0
        ),
        frozen_policy_baseline_distance_m=(
            frozen_policy_baseline_distance_m
            if args.discovery_mode and not lean_frontier_training_enabled(args)
            else 0.0
        ),
    )

    def save_training_checkpoint(path: Path, reason: str) -> dict[str, Any]:
        return save_exact_training_checkpoint(
            model,
            raw_env,
            exploration_callback,
            training_budget_state,
            path,
            metadata,
            reason=reason,
        )

    callbacks = [
        # Capture the actor that drove the lap before the exploration callback
        # switches between candidate and working-policy probe roles.
        LapCheckpointCallback(
            args.lap_checkpoint_dir,
            args.best_completed_lap_model_path,
            metadata,
        ),
    ]
    frontier_callback = None
    if args.discovery_mode:
        frontier_callback = FrontierCheckpointCallback(
            args.frontier_model_path,
            args.frontier_checkpoint_dir,
            metadata,
            source_policy_path=continuation_source_path(args),
            source_evidence=args.frontier_verified_evidence,
            milestones_m=args.frontier_milestones_m,
            minimum_improvement_m=args.frontier_min_improvement_m,
            major_breakthrough_improvement_m=(
                args.frontier_major_breakthrough_m
            ),
            progress_state=progress_state,
        )
        callbacks.append(frontier_callback)
    callbacks.extend(
        [
        exploration_callback,
        make_checkpoint_callback(
            CheckpointCallback,
            args,
            BaseCallback=BaseCallback,
            training_budget_state=training_budget_state,
            checkpoint_saver=save_training_checkpoint,
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
    )
    if args.exact_resume:
        budget_extension = args.exact_resume_budget_extension or {}
        allow_budget_extension = bool(
            int(budget_extension.get("additional_training_steps", 0)) > 0
            or int(budget_extension.get("maximum_probe_steps", 0))
            > int(budget_extension.get("saved_maximum_probe_steps", 0))
        )
        restored_checkpoint = restore_exact_training_checkpoint(
            model,
            raw_env,
            exploration_callback,
            training_budget_state,
            args.resume_exact_from,
            allow_budget_extension=allow_budget_extension,
        )
        metadata["continuation"]["exact_resume_checkpoint"] = (
            restored_checkpoint
        )
        metadata["continuation"]["restored_training_budget"] = (
            training_budget_state.as_dict()
        )
        metadata["continuation"]["restored_curriculum_stage"] = (
            raw_env.current_stage.stage_id
        )
        metadata["milestone_history"] = copy.deepcopy(
            raw_env.sector_milestone_history
        )
        write_json(run_dir / "training_metadata.json", metadata)
        print(
            "Exact state restored: training="
            f"{training_budget_state.training_steps:,}/"
            f"{training_budget_state.target_training_steps:,}, probes="
            f"{training_budget_state.probe_steps:,}, stage="
            f"{raw_env.current_stage.stage_id}, milestones="
            f"{len(raw_env.sector_milestone_history):,}."
        )
        if allow_budget_extension:
            print(
                "Exact-resume budget extended: +"
                f"{int(budget_extension['additional_training_steps']):,} training "
                "steps; new target="
                f"{training_budget_state.target_training_steps:,}, probe ceiling="
                f"{training_budget_state.effective_maximum_probe_steps:,}."
            )
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

    def record_frontier_run_state() -> None:
        if frontier_callback is not None:
            metadata["discovery"]["run_state"] = frontier_callback.status()

    def record_learning_controller_state() -> None:
        safe_actor_status = getattr(model, "safe_actor_status", None)
        if callable(safe_actor_status):
            metadata["safe_actor_improvement"]["run_state"] = (
                safe_actor_status()
            )
        discovery_recovery_status = getattr(
            model,
            "discovery_recovery_status",
            None,
        )
        if callable(discovery_recovery_status):
            metadata["discovery"]["recovery_run_state"] = (
                discovery_recovery_status()
            )
        metadata["discovery"]["trajectory_gate"]["health_run_state"] = {
            "baseline_distance_m": (
                exploration_callback.frozen_policy_baseline_distance_m
            ),
            "threshold_m": (
                exploration_callback.frozen_policy_baseline_distance_m
                * exploration_callback.frozen_policy_health_fraction
            ),
            "failure_streak": (
                exploration_callback.frozen_policy_health_failure_streak
            ),
            "failure_reason": (
                exploration_callback.frozen_policy_health_failure_reason
            ),
        }

    training_completed = False
    training_interrupted = False
    try:
        remaining_environment_steps = (
            max(
                0,
                training_budget_state.target_training_steps
                - training_budget_state.training_steps,
            )
            + max(
                0,
                training_budget_state.effective_maximum_probe_steps
                - training_budget_state.probe_steps,
            )
        )
        if remaining_environment_steps < 1:
            raise RuntimeError(
                "exact-resume checkpoint has no remaining training budget"
            )
        model.learn(
            total_timesteps=remaining_environment_steps,
            callback=callback,
            reset_num_timesteps=not args.exact_resume,
            log_interval=10,
        )
        if exploration_callback.frozen_policy_health_failure_reason is not None:
            raise RuntimeError(
                exploration_callback.frozen_policy_health_failure_reason
            )
        end_summary = raw_env.write_end_of_run_summary(
            training_budget_state.stop_reason or "environment_step_limit_reached"
        )
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["timestep_budget"]["run_state"] = (
            training_budget_state.as_dict()
        )
        metadata["milestone_history"] = copy.deepcopy(
            raw_env.sector_milestone_history
        )
        record_learning_controller_state()
        record_frontier_run_state()
        if end_summary is not None:
            metadata["end_of_run_summary"] = end_summary
        final_checkpoint = save_training_checkpoint(
            args.model_path,
            "training_completed",
        )
        metadata["exact_resume_checkpoint"] = final_checkpoint
        if args.save_replay_buffer:
            replay_error = save_replay_buffer_atomically(
                model,
                args.replay_buffer_path,
            )
            if replay_error is not None:
                raise RuntimeError(replay_error)
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
        training_completed = True
    except KeyboardInterrupt:
        training_interrupted = True
        end_summary = raw_env.write_end_of_run_summary("keyboard_interrupt")
        metadata["interrupted_at"] = datetime.now(timezone.utc).isoformat()
        metadata["interruption_reason"] = "keyboard_interrupt"
        metadata["timestep_budget"]["run_state"] = (
            training_budget_state.as_dict()
        )
        metadata["milestone_history"] = copy.deepcopy(
            raw_env.sector_milestone_history
        )
        record_learning_controller_state()
        record_frontier_run_state()
        if end_summary is not None:
            metadata["end_of_run_summary"] = end_summary
        emergency = save_training_checkpoint(
            args.emergency_checkpoint_path,
            "keyboard_interrupt",
        )
        metadata["exact_resume_checkpoint"] = emergency
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(
            metadata_path_for_policy(args.emergency_checkpoint_path),
            metadata,
        )
        print(
            "\nTraining interrupted safely. Exact resume checkpoint: "
            f"{args.emergency_checkpoint_path}"
        )
    except Exception as exc:
        metadata["interrupted_at"] = datetime.now(timezone.utc).isoformat()
        metadata["interruption_reason"] = f"{type(exc).__name__}: {exc}"
        metadata["timestep_budget"]["run_state"] = (
            training_budget_state.as_dict()
        )
        metadata["milestone_history"] = copy.deepcopy(
            raw_env.sector_milestone_history
        )
        record_learning_controller_state()
        record_frontier_run_state()
        try:
            emergency = save_training_checkpoint(
                args.emergency_checkpoint_path,
                f"runtime_failure:{type(exc).__name__}",
            )
            metadata["exact_resume_checkpoint"] = emergency
            write_json(run_dir / "training_metadata.json", metadata)
            write_json(
                metadata_path_for_policy(args.emergency_checkpoint_path),
                metadata,
            )
            print(
                "Saved exact emergency checkpoint after runtime failure: "
                f"{args.emergency_checkpoint_path}",
                file=sys.stderr,
            )
        except Exception as checkpoint_exc:
            print(
                "Emergency checkpoint also failed: "
                f"{type(checkpoint_exc).__name__}: {checkpoint_exc}",
                file=sys.stderr,
            )
        raise
    finally:
        env.close()

    if training_interrupted:
        return

    frontier_requires_evaluation = bool(
        frontier_callback is not None
        and frontier_callback.requires_strict_evaluation
    )
    forced_end_evaluation = bool(
        args.end_of_run_evaluation_was_explicit
        and args.end_of_run_evaluation
    )
    should_run_end_evaluation = bool(
        training_completed
        and args.end_of_run_evaluation
        and (
            not args.discovery_mode
            or forced_end_evaluation
            or frontier_requires_evaluation
        )
    )
    if should_run_end_evaluation:
        evaluation_policy_path = (
            args.frontier_model_path if args.discovery_mode else args.model_path
        )
        print(
            "Starting one held-out end-of-run evaluation after closing the "
            f"training TORCS instance for {evaluation_policy_path}..."
        )
        evaluation_result = run_automatic_end_evaluation(
            args,
            policy_path=evaluation_policy_path,
        )
        metadata["automatic_evaluation"]["run_state"] = evaluation_result
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        if evaluation_result["succeeded"]:
            print("Automatic held-out evaluation completed successfully.")
        else:
            print(
                "Warning: training completed, but automatic held-out evaluation "
                f"failed with exit code {evaluation_result['return_code']}."
            )
    elif training_completed and args.discovery_mode:
        metadata["automatic_evaluation"]["run_state"] = {
            "skipped": True,
            "reason": "no_new_frontier_milestone_or_completed_lap",
            "frontier_model_path": str(args.frontier_model_path),
        }
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        print(
            "Skipped strict held-out evaluation: discovery produced no new "
            "distance milestone or completed lap."
        )


if __name__ == "__main__":
    main()
