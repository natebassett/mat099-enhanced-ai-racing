import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
ORIGINAL_COMMAND = " ".join(sys.argv)

from agents.hybrid_sac_agent import (  # noqa: E402
    AGENT5_ACTION_VERSION,
    AGENT5_MODEL_FAMILY,
    AGENT5_OBSERVATION_VERSION,
    DEFAULT_BEST_DISTANCE_MODEL_PATH,
    DEFAULT_BEST_MODEL_PATH,
    DEFAULT_BEST_PROGRESS_PACE_MODEL_PATH,
    DEFAULT_BEST_REWARD_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPLAY_BUFFER_PATH,
    DEFAULT_RESIDUAL_SCALE,
    FEATURE_NAMES,
    HIGH_RISK_PRESSURE,
    LOW_SPEED_RECOVERY_ANGLE_LIMIT,
    LOW_SPEED_RECOVERY_SLIDE_LIMIT,
    LOW_SPEED_RECOVERY_SPEED_LIMIT,
    LOW_SPEED_RECOVERY_TRACK_LIMIT,
    MAX_ACCEL_RESIDUAL,
    MAX_BRAKE_RESIDUAL,
    MAX_STEER_RESIDUAL,
    RISK_ANGLE_LIMIT,
    RISK_FRONT_SENSOR_LIMIT,
    RISK_SLIDE_LIMIT,
    RISK_TRACK_LIMIT,
    apply_hybrid_sac_residual,
    build_hybrid_sac_observation,
    find_policy_contract_mismatches,
    get_racing_line_snapshot,
    get_sac_authority_config,
    metadata_path_for_policy,
)
from agents.map_aware_agent import DEFAULT_RACING_LINE_PATH, MapAwareAgent, clamp  # noqa: E402
from agents.map_aware_agent import get_track_sensors  # noqa: E402
from runner.lap_tracker import LapTracker, practice_finish_is_plausible  # noqa: E402


DEFAULT_TRACK_NAME = "g-track-3"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "agent5_hybrid_sac"
DEFAULT_TENSORBOARD_DIR = PROJECT_ROOT / "models" / "tensorboard" / "agent5_hybrid_sac"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "models" / "training_runs" / "agent5_hybrid_sac"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0
DEFAULT_CHECKPOINT_FREQ = 100_000
RESUME_SOURCE_CHOICES = (
    "auto",
    "final",
    "best-clean",
    "best-progress-pace",
    "best-distance",
    "best-reward",
)
REWARD_VERSION = "agent5_reward_v2_supervised_residual_authority"
REWARD_WEIGHTS = {
    "progress": 0.42,
    "pace": 0.18,
    "slow_pace": 0.18,
    "time_step": 0.004,
    "aligned_speed": 0.06,
    "racing_line_error": 0.11,
    "large_racing_line_error": 0.24,
    "overspeed": 0.34,
    "lateral_slide": 0.12,
    "residual_size": 0.05,
    "raw_residual_size": 0.025,
    "low_authority_residual": 0.20,
    "authority_gate": 0.04,
    "safety_shield": 0.12,
    "unsafe_edge_exit": 0.26,
    "unsafe_residual": 0.38,
    "off_track": 340.0,
    "crash": 320.0,
    "backwards": 260.0,
    "stuck": 300.0,
    "incomplete_lap_failure_floor": 180.0,
    "incomplete_lap_failure": 260.0,
    "lap_completed": 450.0,
}
STUCK_SPEED_LIMIT_KMH = 5.0
STUCK_PROGRESS_LIMIT_M = 0.20
STUCK_SECONDS_LIMIT = 3.0
LINE_GUIDE_FREE_ERROR = 0.18
LINE_GUIDE_LARGE_ERROR = 0.50
TEACHER_CONFIDENCE_MIN = 0.35
TEACHER_EDGE_START = 0.42
TEACHER_EDGE_SPAN = 0.24
TEACHER_LATERAL_ACCEL_START = 9.5
TEACHER_LATERAL_ACCEL_SPAN = 4.0
TEACHER_SPEED_RELAXATION_KMH = 45.0
TEACHER_LOOKAHEAD_SPEED_RELAXATION_KMH = 25.0
SPEED_LOOKAHEAD_DISTANCES_M = (35.0, 70.0, 105.0)
SPEED_LOOKAHEAD_ALLOWANCES_KMH = (4.0, 14.0, 28.0)

EPISODE_COLUMNS = [
    "episodes_seen",
    "global_timestep",
    "steps",
    "reward",
    "distance_m",
    "duration_seconds",
    "average_speed_kmh",
    "max_speed_kmh",
    "laps_completed",
    "best_lap_time_seconds",
    "lap_completion_fraction",
    "off_track_steps",
    "safety_shield_steps",
    "unsafe_residual_steps",
    "authority_gate_steps",
    "low_authority_residual_steps",
    "mean_sac_authority",
    "max_unsafe_residual_pressure",
    "max_stopped_seconds",
    "termination_reason",
    "final_dist_from_start",
    "final_speed_kmh",
    "final_track_pos",
    "final_line_error",
    "final_reward_target_speed_kmh",
]

EPISODE_START_COLUMNS = [
    "episode",
    "distFromStart",
    "distRaced",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
    "frontSensor",
    "minTrackSensor",
    "curLapTime",
    "lastLapTime",
    "baseSteer",
    "baseAccel",
    "baseBrake",
    "baseGear",
]

STEP_TELEMETRY_COLUMNS = [
    "episode",
    "episodeStep",
    "globalStep",
    "distFromStart",
    "distRaced",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
    "damage",
    "frontSensor",
    "minTrackSensor",
    *[f"trackSensor{index}" for index in range(19)],
    "racingLineTargetPos",
    "racingLineError",
    "racingLineTargetSpeed",
    "racingLineCurvature",
    "baseSteer",
    "baseAccel",
    "baseBrake",
    "rawSteerResidual",
    "rawAccelResidual",
    "rawBrakeResidual",
    "steerResidual",
    "accelResidual",
    "brakeResidual",
    "sacAuthority",
    "teacherAuthority",
    "authorityGateActive",
    "finalSteer",
    "finalAccel",
    "finalBrake",
    "gear",
    "reward",
    "safetyShieldActive",
    "riskPressure",
    "unsafeResidualPressure",
    "lowSpeedRecoveryActive",
    "terminated",
    "truncated",
    "terminationReason",
    "curLapTime",
    "lastLapTime",
]


def import_training_dependencies():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
    except ImportError as exc:
        raise SystemExit(
            "Agent 5 training needs extra packages:\n"
            "  pip install stable-baselines3 gymnasium\n\n"
            "The normal race menu does not need these packages unless you want "
            "to train or load a SAC model."
        ) from exc

    return gym, spaces, SAC, BaseCallback, CallbackList, CheckpointCallback


def make_torcs_runner():
    with contextlib.redirect_stderr(io.StringIO()):
        from runner.torcs_runner import TorcsRunner

    return TorcsRunner()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_model_metadata(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        return {}

    metadata_path = metadata_path_for_policy(model_path)
    if not metadata_path.is_file():
        return {}

    try:
        return read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return {}


def finite_summary_value(summary, key, default=0.0):
    try:
        value = float(summary.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def make_default_run_dir():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS_DIR / timestamp


def format_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"

    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def tensorboard_is_available():
    return importlib.util.find_spec("tensorboard") is not None


def resolve_tensorboard_log_dir(tensorboard_dir, *, disable_tensorboard=False):
    if disable_tensorboard:
        return None

    if tensorboard_is_available():
        return str(tensorboard_dir)

    print(
        "TensorBoard is not installed, so SAC TensorBoard logging is disabled. "
        "Training will still run and write episodes.csv/training_metadata.json."
    )
    return None


def validate_resume_metadata(model_path, *, force_resume=False):
    metadata = read_model_metadata(model_path)
    if not metadata:
        message = (
            "No metadata file was found beside the SAC model. The weights can "
            "probably be loaded, but the observation/action contract cannot be checked."
        )
        if force_resume:
            print(f"WARNING: {message}")
            return None
        raise SystemExit(f"{message}\nUse --force-resume if you still want to continue.")

    mismatches = find_policy_contract_mismatches(metadata)
    if mismatches and not force_resume:
        formatted = ", ".join(mismatches)
        raise SystemExit(
            "The saved SAC model was trained with an incompatible Agent 5 "
            f"contract: {formatted}.\n"
            "Start a fresh model, or use --force-resume only if you know the "
            "change is compatible."
        )

    if metadata.get("reward_version") != REWARD_VERSION:
        print(
            "WARNING: reward version changed from "
            f"{metadata.get('reward_version')} to {REWARD_VERSION}. "
            "Continuing will optimize a different objective."
        )

    return metadata


def resolve_resume_model_path(args):
    source_paths = {
        "final": [args.model_path],
        "best-clean": [args.best_model_path],
        "best-progress-pace": [args.best_progress_pace_model_path],
        "best-distance": [args.best_distance_model_path],
        "best-reward": [args.best_reward_model_path],
        "auto": [
            args.best_model_path,
            args.best_progress_pace_model_path,
            args.best_distance_model_path,
            args.best_reward_model_path,
            args.model_path,
        ],
    }
    for path in source_paths[args.resume_source]:
        path = Path(path)
        if path.is_file():
            return path
    return Path(args.model_path)


def calculate_progress_pace_score(summary, *, target_laps=1):
    distance_m = max(0.0, finite_summary_value(summary, "distance_m"))
    average_speed_kmh = max(0.0, finite_summary_value(summary, "average_speed_kmh"))
    lap_fraction = clamp(
        finite_summary_value(summary, "lap_completion_fraction"),
        0.0,
        1.0,
    )
    laps_completed = int(max(0.0, finite_summary_value(summary, "laps_completed")))
    target_laps = max(1, int(target_laps))
    termination_reason = str(summary.get("termination_reason", ""))

    off_track_steps = max(0.0, finite_summary_value(summary, "off_track_steps"))
    safety_shield_steps = max(0.0, finite_summary_value(summary, "safety_shield_steps"))
    unsafe_residual_steps = max(
        0.0,
        finite_summary_value(summary, "unsafe_residual_steps"),
    )
    stopped_seconds = max(0.0, finite_summary_value(summary, "max_stopped_seconds"))
    stability_penalty = (
        off_track_steps * 1.5
        + safety_shield_steps * 0.04
        + unsafe_residual_steps * 0.12
        + stopped_seconds * 45.0
    )

    completed_target = (
        laps_completed >= target_laps
        or termination_reason == "target_laps_completed"
    )
    if completed_target:
        lap_time = finite_summary_value(summary, "best_lap_time_seconds", default=0.0)
        if lap_time <= 0.0:
            lap_time = finite_summary_value(summary, "duration_seconds", default=0.0)
        if lap_time <= 0.0:
            lap_time = 9999.0

        return (
            1_000_000.0
            + laps_completed * 50_000.0
            - lap_time * 120.0
            + average_speed_kmh * 15.0
            - stability_penalty
        )

    failure_penalty = {
        "stuck": 300.0,
        "backwards": 450.0,
        "crashed": 550.0,
    }.get(termination_reason, 0.0)

    return (
        distance_m * 10.0
        + average_speed_kmh * 2.0
        + lap_fraction * 500.0
        - stability_penalty
        - failure_penalty
    )


def calculate_step_time_delta(previous_telemetry, telemetry):
    if previous_telemetry is None:
        return None

    previous_time = finite_summary_value(previous_telemetry, "curLapTime")
    current_time = finite_summary_value(telemetry, "curLapTime")
    delta = current_time - previous_time
    if 0.0 < delta <= 1.0:
        return delta
    return None


def calculate_progress_delta(previous_telemetry, telemetry, track_length=None):
    if previous_telemetry is None:
        return 0.0

    if "distRaced" in telemetry and "distRaced" in previous_telemetry:
        delta = finite_summary_value(telemetry, "distRaced") - finite_summary_value(
            previous_telemetry,
            "distRaced",
        )
        return clamp(delta, -8.0, 20.0)

    if not track_length:
        return 0.0

    previous_distance = finite_summary_value(previous_telemetry, "distFromStart")
    current_distance = finite_summary_value(telemetry, "distFromStart")
    delta = current_distance - previous_distance
    if delta < -track_length * 0.5:
        delta += track_length
    elif delta > track_length * 0.5:
        delta -= track_length
    return clamp(delta, -8.0, 20.0)


def calculate_stopped_time_delta(previous_telemetry, telemetry, track_length=None):
    speed = abs(finite_summary_value(telemetry, "speedX"))
    if speed > STUCK_SPEED_LIMIT_KMH:
        return 0.0

    progress = calculate_progress_delta(previous_telemetry, telemetry, track_length)
    if abs(progress) > STUCK_PROGRESS_LIMIT_M:
        return 0.0

    delta_time = calculate_step_time_delta(previous_telemetry, telemetry)
    return 0.02 if delta_time is None else delta_time


def calculate_teacher_confidence(telemetry, line_error):
    track_position = finite_summary_value(telemetry, "trackPos")
    lateral_speed = finite_summary_value(telemetry, "speedY")
    angle = finite_summary_value(telemetry, "angle")
    edge_pressure = clamp(
        (abs(track_position) - TEACHER_EDGE_START) / TEACHER_EDGE_SPAN,
        0.0,
        1.0,
    )
    lateral_pressure = clamp(
        (abs(lateral_speed) - TEACHER_LATERAL_ACCEL_START)
        / TEACHER_LATERAL_ACCEL_SPAN,
        0.0,
        1.0,
    )
    heading_pressure = clamp((abs(angle) - 0.24) / 0.36, 0.0, 1.0)
    line_pressure = clamp((abs(line_error) - 0.38) / 0.35, 0.0, 1.0)
    return clamp(
        1.0
        - 0.65
        * max(edge_pressure, lateral_pressure, heading_pressure, line_pressure),
        TEACHER_CONFIDENCE_MIN,
        1.0,
    )


def get_reward_line_context(telemetry, racing_line):
    if racing_line is None:
        return {
            "line_error": 0.0,
            "teacher_confidence": TEACHER_CONFIDENCE_MIN,
            "reward_target_speed": 216.0,
            "curvature": 0.0,
        }

    speed = finite_summary_value(telemetry, "speedX")
    distance = finite_summary_value(telemetry, "distFromStart")
    track_position = finite_summary_value(telemetry, "trackPos")
    lookahead_distance = clamp(24.0 + speed * 0.16, 24.0, 56.0)
    target = racing_line.lookup(distance)
    lookaheads = [
        racing_line.lookup(distance + lookahead)
        for lookahead in SPEED_LOOKAHEAD_DISTANCES_M
    ]
    line_error = track_position - float(target.get("target_track_pos", 0.0))
    teacher_confidence = calculate_teacher_confidence(telemetry, line_error)
    target_speed = float(target.get("target_speed_kmh", 216.0))
    reward_target_speed = min(
        target_speed + TEACHER_SPEED_RELAXATION_KMH * (1.0 - teacher_confidence),
        *[
            float(waypoint.get("target_speed_kmh", 216.0)) + allowance
            for waypoint, allowance in zip(
                lookaheads,
                SPEED_LOOKAHEAD_ALLOWANCES_KMH,
            )
        ],
    )
    close_lookahead = racing_line.lookup(distance + lookahead_distance)
    curvature = max(
        abs(float(target.get("curvature", 0.0))),
        abs(float(close_lookahead.get("curvature", 0.0))),
    )
    if curvature < 0.0035:
        reward_target_speed = max(reward_target_speed, 188.0)
    elif curvature < 0.006:
        reward_target_speed = max(reward_target_speed, 162.0)
    reward_target_speed += TEACHER_LOOKAHEAD_SPEED_RELAXATION_KMH * (
        1.0 - teacher_confidence
    )

    return {
        "line_error": line_error,
        "teacher_confidence": teacher_confidence,
        "reward_target_speed": clamp(reward_target_speed, 42.0, 216.0),
        "curvature": curvature,
    }


def calculate_lap_completion_fraction(episode_distance_m, racing_line):
    track_length = float(getattr(racing_line, "track_length", 0.0) or 0.0)
    if track_length <= 0.0:
        return 0.0
    return clamp(max(0.0, episode_distance_m or 0.0) / track_length, 0.0, 1.0)


def calculate_incomplete_lap_failure_penalty(lap_fraction):
    remaining = 1.0 - clamp(lap_fraction, 0.0, 1.0)
    return (
        REWARD_WEIGHTS["incomplete_lap_failure_floor"]
        + REWARD_WEIGHTS["incomplete_lap_failure"] * remaining
    )


def calculate_unsafe_edge_exit_pressure(track_position, lateral_speed, angle, action):
    edge_pressure = clamp((abs(track_position) - RISK_TRACK_LIMIT) / 0.24, 0.0, 1.0)
    if edge_pressure <= 0.0:
        return 0.0

    steer = float(action.get("steer", 0.0))
    accel = float(action.get("accel", 0.0))
    outward_steer = steer * track_position > 0.0
    outward_slide = lateral_speed * track_position > 1.5
    outward_heading = angle * track_position < -0.05
    throttle_pressure = clamp(accel, 0.0, 1.0) if outward_slide or outward_heading else 0.0
    steer_pressure = clamp(abs(steer), 0.0, 1.0) if outward_steer else 0.0
    return edge_pressure * max(throttle_pressure, steer_pressure)


def calculate_hybrid_sac_reward(
    telemetry,
    action,
    *,
    previous_telemetry=None,
    previous_damage=0.0,
    racing_line=None,
    completed_lap=None,
    episode_distance_m=None,
    stuck=False,
):
    track_sensors = get_track_sensors(telemetry)
    speed = finite_summary_value(telemetry, "speedX")
    angle = finite_summary_value(telemetry, "angle")
    track_position = finite_summary_value(telemetry, "trackPos")
    lateral_speed = finite_summary_value(telemetry, "speedY")
    damage = finite_summary_value(telemetry, "damage", default=previous_damage)
    line_context = get_reward_line_context(telemetry, racing_line)
    line_error = abs(line_context["line_error"])
    teacher_confidence = float(line_context["teacher_confidence"])
    target_speed = max(40.0, line_context["reward_target_speed"])
    curvature_abs = abs(line_context["curvature"])
    progress_delta = calculate_progress_delta(
        previous_telemetry,
        telemetry,
        getattr(racing_line, "track_length", None),
    )
    delta_time = calculate_step_time_delta(previous_telemetry, telemetry)
    aligned_speed = speed * math.cos(angle)
    overspeed_margin = 12.0 if curvature_abs < 0.008 else 5.0
    overspeed = max(0.0, speed - target_speed - overspeed_margin)
    target_mps = max(target_speed / 3.6, 8.0)
    if delta_time is None:
        pace_ratio = max(0.0, aligned_speed / 3.6) / target_mps
    else:
        pace_ratio = (progress_delta / delta_time) / target_mps
    residual_pressure = (
        abs(float(action.get("agent5_steer_residual", 0.0))) / MAX_STEER_RESIDUAL
        + abs(float(action.get("agent5_accel_residual", 0.0))) / MAX_ACCEL_RESIDUAL
        + abs(float(action.get("agent5_brake_residual", 0.0))) / MAX_BRAKE_RESIDUAL
    ) / 3.0
    raw_residual_pressure = (
        abs(clamp(float(action.get("agent5_raw_steer_residual", 0.0)), -1.0, 1.0))
        + abs(clamp(float(action.get("agent5_raw_accel_residual", 0.0)), -1.0, 1.0))
        + abs(clamp(float(action.get("agent5_raw_brake_residual", 0.0)), -1.0, 1.0))
    ) / 3.0
    sac_authority = clamp(float(action.get("agent5_sac_authority", 1.0)), 0.0, 1.0)
    low_authority_residual_pressure = raw_residual_pressure * (1.0 - sac_authority)

    reward = clamp(progress_delta, -3.0, 8.0) * REWARD_WEIGHTS["progress"]
    reward += clamp(pace_ratio, -0.5, 1.25) * REWARD_WEIGHTS["pace"]
    reward -= max(0.0, 0.62 - pace_ratio) * REWARD_WEIGHTS["slow_pace"]
    reward -= REWARD_WEIGHTS["time_step"]
    reward += (
        clamp(aligned_speed, 0.0, target_speed + 15.0)
        / (target_speed + 15.0)
        * REWARD_WEIGHTS["aligned_speed"]
    )
    free_line_error = LINE_GUIDE_FREE_ERROR + (1.0 - teacher_confidence) * 0.28
    large_line_error = LINE_GUIDE_LARGE_ERROR + (1.0 - teacher_confidence) * 0.35
    reward -= (
        max(0.0, line_error - free_line_error)
        * REWARD_WEIGHTS["racing_line_error"]
        * teacher_confidence
    )
    reward -= (
        max(0.0, line_error - large_line_error)
        * REWARD_WEIGHTS["large_racing_line_error"]
        * teacher_confidence
    )
    reward -= (clamp(overspeed / 70.0, 0.0, 1.0) ** 2) * REWARD_WEIGHTS["overspeed"]
    reward -= clamp(abs(lateral_speed) / 22.0, 0.0, 1.0) * REWARD_WEIGHTS[
        "lateral_slide"
    ]
    reward -= clamp(residual_pressure, 0.0, 1.0) * REWARD_WEIGHTS["residual_size"]
    reward -= (
        clamp(raw_residual_pressure, 0.0, 1.0)
        * REWARD_WEIGHTS["raw_residual_size"]
    )
    reward -= (
        clamp(low_authority_residual_pressure, 0.0, 1.0)
        * REWARD_WEIGHTS["low_authority_residual"]
    )
    reward -= (
        calculate_unsafe_edge_exit_pressure(
            track_position,
            lateral_speed,
            angle,
            action,
        )
        * REWARD_WEIGHTS["unsafe_edge_exit"]
    )
    reward -= (
        clamp(float(action.get("agent5_unsafe_residual_pressure", 0.0)), 0.0, 1.0)
        * REWARD_WEIGHTS["unsafe_residual"]
    )

    if action.get("agent5_authority_gate_active", False):
        reward -= (1.0 - sac_authority) * REWARD_WEIGHTS["authority_gate"]
    if action.get("agent5_safety_shield_active", False):
        reward -= REWARD_WEIGHTS["safety_shield"]

    off_track = min(track_sensors) < 0.0 or abs(track_position) > 1.05
    crashed = damage > previous_damage
    backwards = math.cos(angle) < 0.0
    failed_episode = off_track or crashed or backwards or stuck

    if off_track:
        reward -= REWARD_WEIGHTS["off_track"]
    if crashed:
        reward -= REWARD_WEIGHTS["crash"]
    if backwards:
        reward -= REWARD_WEIGHTS["backwards"]
    if stuck:
        reward -= REWARD_WEIGHTS["stuck"]
    if failed_episode and completed_lap is None:
        lap_fraction = calculate_lap_completion_fraction(
            episode_distance_m,
            racing_line,
        )
        reward -= calculate_incomplete_lap_failure_penalty(lap_fraction)
    if completed_lap is not None:
        reward += REWARD_WEIGHTS["lap_completed"]

    return float(reward)


def build_episode_diagnostics(telemetry, episode_distance_m, racing_line):
    line_context = get_reward_line_context(telemetry, racing_line)
    return {
        "lap_completion_fraction": calculate_lap_completion_fraction(
            episode_distance_m,
            racing_line,
        ),
        "final_dist_from_start": finite_summary_value(telemetry, "distFromStart"),
        "final_speed_kmh": finite_summary_value(telemetry, "speedX"),
        "final_track_pos": finite_summary_value(telemetry, "trackPos"),
        "final_line_error": line_context["line_error"],
        "final_reward_target_speed_kmh": line_context["reward_target_speed"],
    }


@contextlib.contextmanager
def maybe_suppress_stdout(enabled):
    if not enabled:
        yield
        return

    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_training_metadata(args, *, resumed_from=None):
    return {
        "model_family": AGENT5_MODEL_FAMILY,
        "observation_version": AGENT5_OBSERVATION_VERSION,
        "action_version": AGENT5_ACTION_VERSION,
        "reward_version": REWARD_VERSION,
        "control_profile": get_sac_authority_config()["profile"],
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
        "base_agent": "Map-Aware Racing-Line Agent",
        "teacher_policy": {
            "name": "Map-Aware Racing-Line Agent",
            "role": "stable controller and teacher prior, not the final optimum",
            "learning_goal": (
                "learn residual racecraft around the controller: turn-in, "
                "braking release, throttle timing, and controlled line deviations"
            ),
        },
        "track": args.track,
        "residual_scale": args.residual_scale,
        "sac_authority": get_sac_authority_config(),
        "max_episode_steps": args.max_episode_steps,
        "target_laps": args.target_laps,
        "manual_start": args.manual_start,
        "relaunch_frequency": args.relaunch_frequency,
        "reset_retries": args.reset_retries,
        "resume_source": args.resume_source,
        "total_timesteps_requested": args.total_timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "tensorboard_requested": not args.no_tensorboard,
        "tensorboard_available": tensorboard_is_available(),
        "progress_bar_enabled": not args.no_progress_bar,
        "verbose_training": args.verbose_training,
        "show_torcs_reset_log": args.show_torcs_reset_log,
        "seed": args.seed,
        "model_path": str(args.model_path),
        "best_model_path": str(args.best_model_path),
        "best_progress_pace_model_path": str(args.best_progress_pace_model_path),
        "best_distance_model_path": str(args.best_distance_model_path),
        "best_reward_model_path": str(args.best_reward_model_path),
        "replay_buffer_path": str(args.replay_buffer_path),
        "save_replay_buffer": args.save_replay_buffer,
        "save_checkpoint_replay_buffer": args.save_checkpoint_replay_buffer,
        "load_replay_buffer": args.load_replay_buffer,
        "long_run_model_preservation": {
            "checkpoint_freq": args.checkpoint_freq,
            "checkpoint_dir": str(args.checkpoint_dir),
            "best_clean_lap_model": str(args.best_model_path),
            "best_progress_pace_model": str(args.best_progress_pace_model_path),
            "best_distance_model": str(args.best_distance_model_path),
            "best_reward_model": str(args.best_reward_model_path),
            "runtime_default_order": [
                "best_clean_lap_model",
                "best_progress_pace_model",
                "best_distance_model",
                "best_reward_model",
                "final_model",
            ],
        },
        "sac_hyperparameters": {
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "learning_rate": args.learning_rate,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "ent_coef": args.ent_coef,
            "target_update_interval": args.target_update_interval,
            "device": args.device,
        },
        "curriculum_recommendation": [
            {
                "phase": 1,
                "residual_scale": 0.05,
                "goal": "verify SAC can match Agent 4-style map-aware behavior without instability",
            },
            {
                "phase": 2,
                "residual_scale": 0.10,
                "goal": "learn harmless timing corrections and build replay diversity",
            },
            {
                "phase": 3,
                "residual_scale": 0.20,
                "goal": "improve corner entry and exit pace once clean laps are reliable",
            },
            {
                "phase": 4,
                "residual_scale": 0.30,
                "goal": "widen authority only after safety-shield intervention becomes rare",
            },
        ],
        "stuck_detection": {
            "speed_limit_kmh": STUCK_SPEED_LIMIT_KMH,
            "progress_limit_m": STUCK_PROGRESS_LIMIT_M,
            "seconds_limit": STUCK_SECONDS_LIMIT,
        },
        "global_safety": {
            "risk_track_limit": RISK_TRACK_LIMIT,
            "risk_angle_limit_rad": RISK_ANGLE_LIMIT,
            "risk_slide_limit_kmh": RISK_SLIDE_LIMIT,
            "risk_front_sensor_limit_m": RISK_FRONT_SENSOR_LIMIT,
            "high_risk_pressure": HIGH_RISK_PRESSURE,
            "low_speed_recovery_speed_limit_kmh": LOW_SPEED_RECOVERY_SPEED_LIMIT,
            "low_speed_recovery_track_limit": LOW_SPEED_RECOVERY_TRACK_LIMIT,
            "low_speed_recovery_angle_limit_rad": LOW_SPEED_RECOVERY_ANGLE_LIMIT,
            "low_speed_recovery_slide_limit_kmh": LOW_SPEED_RECOVERY_SLIDE_LIMIT,
        },
        "reward_weights": REWARD_WEIGHTS,
        "command": ORIGINAL_COMMAND,
        "python_version": sys.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resumed_from": resumed_from,
    }


def make_training_env_class(gym, spaces):
    class HybridSacTrainingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            *,
            track_name=DEFAULT_TRACK_NAME,
            racing_line_path=DEFAULT_RACING_LINE_PATH,
            residual_scale=DEFAULT_RESIDUAL_SCALE,
            max_episode_steps=12000,
            target_laps=1,
            manual_start=False,
            relaunch_frequency=0,
            reset_retries=2,
            quiet_reset_log=True,
            run_dir=None,
        ):
            super().__init__()
            self.track_name = track_name
            self.residual_scale = residual_scale
            self.max_episode_steps = max_episode_steps
            self.target_laps = target_laps
            self.manual_start = manual_start
            self.relaunch_frequency = relaunch_frequency
            self.reset_retries = reset_retries
            self.quiet_reset_log = quiet_reset_log
            self.run_dir = Path(run_dir) if run_dir is not None else None
            self.runner = make_torcs_runner()
            self.base_agent = MapAwareAgent(racing_line_path=racing_line_path)
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
            self.current_observation = None
            self.current_telemetry = None
            self.current_base_action = None
            self.initial_telemetry = None
            self.previous_telemetry = None
            self.previous_damage = 0.0
            self.lap_tracker = None
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = 0.0
            self.episode_start_time = 0.0
            self.episode_max_speed = 0.0
            self.episode_off_track_steps = 0
            self.episode_shield_steps = 0
            self.episode_unsafe_residual_steps = 0
            self.episode_authority_gate_steps = 0
            self.episode_low_authority_residual_steps = 0
            self.episode_sac_authority_sum = 0.0
            self.episode_max_unsafe_residual_pressure = 0.0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.episodes_started = 0
            self.total_steps = 0
            self._step_telemetry_file = None
            self._step_telemetry_writer = None
            self._episode_start_file = None
            self._episode_start_writer = None
            self._open_training_telemetry()

        def reset(self, *, seed=None, options=None):
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

            self.current_observation = self._reset_torcs_env(relaunch=relaunch)
            self.episodes_started += 1
            self.base_agent.reset()
            self.current_telemetry = dict(self.runner.env.client.S.d)
            self.initial_telemetry = dict(self.current_telemetry)
            self.previous_telemetry = None
            self.previous_damage = float(self.current_telemetry.get("damage", 0.0))
            self.lap_tracker = LapTracker(self.current_telemetry.get("lastLapTime", 0.0))
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = float(
                self.current_telemetry.get("distRaced", 0.0)
            )
            self.episode_start_time = float(
                self.current_telemetry.get("curLapTime", 0.0)
            )
            self.episode_max_speed = float(self.current_telemetry.get("speedX", 0.0))
            self.episode_off_track_steps = 0
            self.episode_shield_steps = 0
            self.episode_unsafe_residual_steps = 0
            self.episode_authority_gate_steps = 0
            self.episode_low_authority_residual_steps = 0
            self.episode_sac_authority_sum = 0.0
            self.episode_max_unsafe_residual_pressure = 0.0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.current_base_action = self.base_agent.act(
                self.current_observation,
                self.current_telemetry,
            )
            self._write_episode_start()

            return self._build_observation(), {}

        def step(self, residual_action):
            residual_action = np.asarray(residual_action, dtype=np.float32)
            action = apply_hybrid_sac_residual(
                self.current_base_action,
                residual_action,
                self.current_telemetry,
                self.residual_scale,
                self.base_agent.racing_line,
            )
            raw_observation, _runner_reward, done, _info = (
                self.runner._step_full_control_agent(action)
            )

            telemetry = dict(raw_observation)
            current_damage = float(telemetry.get("damage", self.previous_damage))
            crashed = current_damage > self.previous_damage
            completed_lap = self.lap_tracker.update(telemetry)
            if (
                completed_lap is None
                and done
                and practice_finish_is_plausible(
                    self.initial_telemetry,
                    telemetry,
                    self.base_agent,
                )
            ):
                completed_lap = self.lap_tracker.record(telemetry.get("curLapTime", 0.0))

            stopped_time_delta = calculate_stopped_time_delta(
                self.previous_telemetry,
                telemetry,
                self.base_agent.racing_line.track_length,
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
            episode_distance_m = max(
                0.0,
                float(telemetry.get("distRaced", 0.0))
                - self.episode_start_distance,
            )

            reward = calculate_hybrid_sac_reward(
                telemetry,
                action,
                previous_telemetry=self.previous_telemetry,
                previous_damage=self.previous_damage,
                racing_line=self.base_agent.racing_line,
                completed_lap=completed_lap,
                episode_distance_m=episode_distance_m,
                stuck=stuck,
            )
            self.previous_damage = current_damage
            self.previous_telemetry = telemetry
            self.current_observation = raw_observation
            self.current_telemetry = telemetry
            self.steps += 1
            self.episode_reward += reward
            self.episode_max_speed = max(
                self.episode_max_speed,
                float(telemetry.get("speedX", 0.0)),
            )

            off_track = min(get_track_sensors(telemetry)) < 0.0
            if off_track:
                self.episode_off_track_steps += 1
            if action.get("agent5_safety_shield_active", False):
                self.episode_shield_steps += 1
            unsafe_residual_pressure = float(
                action.get("agent5_unsafe_residual_pressure", 0.0)
            )
            if unsafe_residual_pressure > 0.01:
                self.episode_unsafe_residual_steps += 1
            self.episode_max_unsafe_residual_pressure = max(
                self.episode_max_unsafe_residual_pressure,
                unsafe_residual_pressure,
            )
            sac_authority = clamp(
                float(action.get("agent5_sac_authority", 1.0)),
                0.0,
                1.0,
            )
            self.episode_sac_authority_sum += sac_authority
            raw_residual_pressure = (
                abs(clamp(float(action.get("agent5_raw_steer_residual", 0.0)), -1.0, 1.0))
                + abs(clamp(float(action.get("agent5_raw_accel_residual", 0.0)), -1.0, 1.0))
                + abs(clamp(float(action.get("agent5_raw_brake_residual", 0.0)), -1.0, 1.0))
            ) / 3.0
            if action.get("agent5_authority_gate_active", False):
                self.episode_authority_gate_steps += 1
            if raw_residual_pressure * (1.0 - sac_authority) > 0.05:
                self.episode_low_authority_residual_steps += 1

            terminated = bool(done)
            terminated = terminated or off_track
            left_track_bounds = abs(float(telemetry.get("trackPos", 0.0))) > 1.08
            terminated = terminated or left_track_bounds
            terminated = terminated or crashed
            backwards = math.cos(float(telemetry.get("angle", 0.0))) < 0.0
            terminated = terminated or backwards
            terminated = terminated or stuck
            if (
                completed_lap is not None
                and self.lap_tracker.laps_completed >= self.target_laps
            ):
                terminated = True

            truncated = self.steps >= self.max_episode_steps
            info = {}
            reason = ""
            if terminated or truncated:
                reason = "truncated"
                if completed_lap is not None:
                    reason = "target_laps_completed"
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

                distance_m = episode_distance_m
                duration_seconds = max(
                    0.0,
                    float(telemetry.get("curLapTime", 0.0))
                    - self.episode_start_time,
                )
                average_speed_kmh = (
                    distance_m / duration_seconds * 3.6
                    if duration_seconds > 0.0
                    else 0.0
                )
                diagnostics = build_episode_diagnostics(
                    telemetry,
                    distance_m,
                    self.base_agent.racing_line,
                )
                info["episode_summary"] = {
                    "steps": self.steps,
                    "reward": self.episode_reward,
                    "distance_m": distance_m,
                    "duration_seconds": duration_seconds,
                    "average_speed_kmh": average_speed_kmh,
                    "max_speed_kmh": self.episode_max_speed,
                    "laps_completed": self.lap_tracker.laps_completed,
                    "best_lap_time_seconds": self.lap_tracker.best_lap_time,
                    "off_track_steps": self.episode_off_track_steps,
                    "safety_shield_steps": self.episode_shield_steps,
                    "unsafe_residual_steps": self.episode_unsafe_residual_steps,
                    "authority_gate_steps": self.episode_authority_gate_steps,
                    "low_authority_residual_steps": (
                        self.episode_low_authority_residual_steps
                    ),
                    "mean_sac_authority": (
                        self.episode_sac_authority_sum / max(1, self.steps)
                    ),
                    "max_unsafe_residual_pressure": (
                        self.episode_max_unsafe_residual_pressure
                    ),
                    "max_stopped_seconds": self.episode_max_stopped_seconds,
                    "termination_reason": reason,
                    **diagnostics,
                }

            self.total_steps += 1
            self._write_step_telemetry(
                telemetry,
                action,
                reward,
                terminated,
                truncated,
                reason,
            )

            if not terminated and not truncated:
                self.current_base_action = self.base_agent.act(
                    self.current_observation,
                    self.current_telemetry,
                )

            return self._build_observation(), reward, terminated, truncated, info

        def close(self):
            self.base_agent.close()
            self._close_training_telemetry()
            self._close_torcs_env()
            self.runner.shutdown()

        def build_current_episode_summary(self, reason="training_stopped"):
            if self.current_telemetry is None or self.steps <= 0:
                return None

            telemetry = self.current_telemetry
            distance_m = max(
                0.0,
                float(telemetry.get("distRaced", 0.0)) - self.episode_start_distance,
            )
            duration_seconds = max(
                0.0,
                float(telemetry.get("curLapTime", 0.0)) - self.episode_start_time,
            )
            average_speed_kmh = (
                distance_m / duration_seconds * 3.6
                if duration_seconds > 0.0
                else 0.0
            )
            diagnostics = build_episode_diagnostics(
                telemetry,
                distance_m,
                self.base_agent.racing_line,
            )
            return {
                "partial": True,
                "steps": self.steps,
                "reward": self.episode_reward,
                "distance_m": distance_m,
                "duration_seconds": duration_seconds,
                "average_speed_kmh": average_speed_kmh,
                "max_speed_kmh": self.episode_max_speed,
                "laps_completed": self.lap_tracker.laps_completed,
                "best_lap_time_seconds": self.lap_tracker.best_lap_time,
                "off_track_steps": self.episode_off_track_steps,
                "safety_shield_steps": self.episode_shield_steps,
                "unsafe_residual_steps": self.episode_unsafe_residual_steps,
                "authority_gate_steps": self.episode_authority_gate_steps,
                "low_authority_residual_steps": (
                    self.episode_low_authority_residual_steps
                ),
                "mean_sac_authority": (
                    self.episode_sac_authority_sum / max(1, self.steps)
                ),
                "max_unsafe_residual_pressure": (
                    self.episode_max_unsafe_residual_pressure
                ),
                "max_stopped_seconds": self.episode_max_stopped_seconds,
                "termination_reason": reason,
                **diagnostics,
            }

        def write_end_of_run_summary(self, reason="training_stopped"):
            summary = self.build_current_episode_summary(reason)
            if summary is None or self.run_dir is None:
                return summary

            write_json(self.run_dir / "end_of_run_summary.json", summary)
            return summary

        def _open_training_telemetry(self):
            if self.run_dir is None:
                return

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._step_telemetry_file = (self.run_dir / "steps.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self._step_telemetry_writer = csv.writer(self._step_telemetry_file)
            self._step_telemetry_writer.writerow(STEP_TELEMETRY_COLUMNS)
            self._step_telemetry_file.flush()

            self._episode_start_file = (self.run_dir / "episode_starts.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self._episode_start_writer = csv.writer(self._episode_start_file)
            self._episode_start_writer.writerow(EPISODE_START_COLUMNS)
            self._episode_start_file.flush()

        def _close_training_telemetry(self):
            if self._step_telemetry_file is not None:
                self._step_telemetry_file.close()
            if self._episode_start_file is not None:
                self._episode_start_file.close()
            self._step_telemetry_file = None
            self._step_telemetry_writer = None
            self._episode_start_file = None
            self._episode_start_writer = None

        def _write_episode_start(self):
            if self._episode_start_writer is None:
                return

            track_sensors = get_track_sensors(self.current_telemetry)
            self._episode_start_writer.writerow(
                [
                    self.episodes_started,
                    self.current_telemetry.get("distFromStart", 0),
                    self.current_telemetry.get("distRaced", 0),
                    self.current_telemetry.get("speedX", 0),
                    self.current_telemetry.get("speedY", 0),
                    self.current_telemetry.get("angle", 0),
                    self.current_telemetry.get("trackPos", 0),
                    track_sensors[9],
                    min(track_sensors),
                    self.current_telemetry.get("curLapTime", 0),
                    self.current_telemetry.get("lastLapTime", 0),
                    self.current_base_action.get("steer", 0),
                    self.current_base_action.get("accel", 0),
                    self.current_base_action.get("brake", 0),
                    self.current_base_action.get("gear", 0),
                ]
            )
            self._episode_start_file.flush()

        def _write_step_telemetry(
            self,
            telemetry,
            action,
            reward,
            terminated,
            truncated,
            reason,
        ):
            if self._step_telemetry_writer is None:
                return

            track_sensors = get_track_sensors(telemetry)
            line = get_racing_line_snapshot(telemetry, self.base_agent.racing_line)
            self._step_telemetry_writer.writerow(
                [
                    self.episodes_started,
                    self.steps,
                    self.total_steps,
                    telemetry.get("distFromStart", 0),
                    telemetry.get("distRaced", 0),
                    telemetry.get("speedX", 0),
                    telemetry.get("speedY", 0),
                    telemetry.get("angle", 0),
                    telemetry.get("trackPos", 0),
                    telemetry.get("damage", 0),
                    track_sensors[9],
                    min(track_sensors),
                    *track_sensors,
                    line["target_track_pos"],
                    line["line_error"],
                    line["target_speed_kmh"],
                    line["curvature"],
                    action["agent5_base_steer"],
                    action["agent5_base_accel"],
                    action["agent5_base_brake"],
                    action["agent5_raw_steer_residual"],
                    action["agent5_raw_accel_residual"],
                    action["agent5_raw_brake_residual"],
                    action["agent5_steer_residual"],
                    action["agent5_accel_residual"],
                    action["agent5_brake_residual"],
                    action["agent5_sac_authority"],
                    action["agent5_teacher_authority"],
                    action["agent5_authority_gate_active"],
                    action["steer"],
                    action["accel"],
                    action["brake"],
                    action["gear"],
                    reward,
                    action["agent5_safety_shield_active"],
                    action["agent5_risk_pressure"],
                    action["agent5_unsafe_residual_pressure"],
                    action["agent5_low_speed_recovery_active"],
                    terminated,
                    truncated,
                    reason,
                    telemetry.get("curLapTime", 0),
                    telemetry.get("lastLapTime", 0),
                ]
            )
            self._step_telemetry_file.flush()

        def _ensure_torcs(self):
            if self.runner.env is not None:
                return

            self._launch_and_connect_torcs()

        def _reset_torcs_env(self, *, relaunch=False):
            last_error = None
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
                        try:
                            self._restart_torcs()
                        except Exception as restart_exc:
                            last_error = restart_exc
                            print(
                                "TORCS relaunch failed while recovering from reset "
                                f"attempt {attempt + 1}/{self.reset_retries + 1}: "
                                f"{restart_exc}"
                            )
                            time.sleep(2.0)
                    else:
                        self._close_torcs_env()
                        try:
                            self._ensure_torcs()
                        except Exception as reconnect_exc:
                            last_error = reconnect_exc
                            print(
                                "TORCS reconnect failed while recovering from reset "
                                f"attempt {attempt + 1}/{self.reset_retries + 1}: "
                                f"{reconnect_exc}"
                            )
                            time.sleep(2.0)

            raise RuntimeError("TORCS reset failed") from last_error

        def _restart_torcs(self):
            self._close_torcs_env()
            self.runner.shutdown()
            self._launch_and_connect_torcs()

        def _launch_and_connect_torcs(self):
            if not self.manual_start:
                self.runner.launch()
            self.runner.connect()
            self.runner.load_track(self.track_name)

        def _close_torcs_env(self):
            if self.runner.env is None:
                return
            try:
                self.runner.env.end()
            except Exception:
                pass
            self.runner.env = None

        def _build_observation(self):
            if self.current_telemetry is None or self.current_base_action is None:
                return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

            return np.asarray(
                build_hybrid_sac_observation(
                    self.current_telemetry,
                    self.current_base_action,
                    self.base_agent.racing_line,
                ),
                dtype=np.float32,
            )

    return HybridSacTrainingEnv


class TrainingProgressState:
    def __init__(self):
        self.episodes_seen = 0
        self.last_episode = None
        self.best_clean_lap_time = None
        self.best_progress_pace = None
        self.best_distance_m = None
        self.best_reward = None

    def record_episode(self, row):
        self.episodes_seen = int(row.get("episodes_seen", self.episodes_seen))
        self.last_episode = dict(row)

    def record_best_clean_lap(self, lap_time):
        self.best_clean_lap_time = float(lap_time)

    def record_best_progress_pace(
        self,
        score,
        distance_m,
        average_speed_kmh,
        laps_completed,
        lap_time_seconds=None,
    ):
        self.best_progress_pace = {
            "score": float(score),
            "distance_m": float(distance_m),
            "average_speed_kmh": float(average_speed_kmh),
            "laps_completed": int(laps_completed),
            "lap_time_seconds": (
                float(lap_time_seconds) if lap_time_seconds is not None else None
            ),
        }

    def record_best_distance(self, distance_m):
        self.best_distance_m = float(distance_m)

    def record_best_reward(self, reward):
        self.best_reward = float(reward)


def format_training_progress_line(
    completed_steps,
    total_steps,
    elapsed_seconds,
    progress_state,
    *,
    width=28,
):
    total_steps = max(1, int(total_steps))
    completed_steps = max(0, min(int(completed_steps), total_steps))
    fraction = completed_steps / total_steps
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    percent = fraction * 100.0

    if completed_steps > 0 and elapsed_seconds > 0.0:
        steps_per_second = completed_steps / elapsed_seconds
        eta_seconds = (total_steps - completed_steps) / max(steps_per_second, 1e-9)
        eta = format_duration(eta_seconds)
        rate = f"{steps_per_second:,.0f} step/s"
    else:
        eta = "--:--:--"
        rate = "-- step/s"

    last_episode = progress_state.last_episode
    if last_episode is None:
        episode_text = "episodes=0"
    else:
        mean_authority = last_episode.get("mean_sac_authority")
        authority_text = (
            ""
            if mean_authority in (None, "")
            else f" auth={float(mean_authority):.2f}"
        )
        episode_text = (
            f"ep {progress_state.episodes_seen} "
            f"{last_episode.get('termination_reason', 'running')} "
            f"{float(last_episode.get('distance_m', 0.0)):.0f}m "
            f"R={float(last_episode.get('reward', 0.0)):.0f} "
            f"laps={int(last_episode.get('laps_completed', 0))}"
            f"{authority_text}"
        )

    best_parts = []
    if progress_state.best_progress_pace is not None:
        best_progress_pace = progress_state.best_progress_pace
        if int(best_progress_pace.get("laps_completed", 0)) > 0:
            lap_time_seconds = best_progress_pace.get("lap_time_seconds")
            if lap_time_seconds is None:
                lap_time_seconds = 0.0
            best_parts.append(
                "best_pp="
                f"{float(lap_time_seconds):.1f}s/"
                f"{float(best_progress_pace.get('average_speed_kmh', 0.0)):.0f}kph"
            )
        else:
            best_parts.append(
                "best_pp="
                f"{float(best_progress_pace.get('distance_m', 0.0)):.0f}m/"
                f"{float(best_progress_pace.get('average_speed_kmh', 0.0)):.0f}kph"
            )
    if progress_state.best_distance_m is not None:
        best_parts.append(f"best_dist={progress_state.best_distance_m:.0f}m")
    if progress_state.best_reward is not None:
        best_parts.append(f"best_R={progress_state.best_reward:.0f}")
    if progress_state.best_clean_lap_time is not None:
        best_parts.append(f"best_lap={progress_state.best_clean_lap_time:.3f}s")

    best_text = f" | {' '.join(best_parts)}" if best_parts else ""

    return (
        f"Training [{bar}] {percent:5.1f}% | "
        f"{completed_steps:,}/{total_steps:,} steps | {rate} | "
        f"ETA {eta} | {episode_text}{best_text}"
    )


def make_episode_summary_callback_class(BaseCallback):
    class EpisodeSummaryCallback(BaseCallback):
        def __init__(self, run_dir, progress_state=None, verbose=0):
            super().__init__(verbose=verbose)
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
                if self.verbose:
                    print(
                        "episode="
                        f"{self.episodes_seen} "
                        f"reason={row['termination_reason']} "
                        f"distance={float(row['distance_m'] or 0.0):.1f}m "
                        f"reward={float(row['reward'] or 0.0):.2f}"
                    )
            return True

        def _on_training_end(self):
            if self._file is not None:
                self._file.close()
            self._file = None
            self._writer = None

    return EpisodeSummaryCallback


def make_best_episode_model_callback_class(BaseCallback):
    class BestEpisodeModelCallback(BaseCallback):
        def __init__(
            self,
            best_progress_pace_model_path,
            best_distance_model_path,
            best_reward_model_path,
            metadata,
            *,
            target_laps=1,
            progress_state=None,
            verbose=0,
        ):
            super().__init__(verbose=verbose)
            self.best_progress_pace_model_path = Path(best_progress_pace_model_path)
            self.best_distance_model_path = Path(best_distance_model_path)
            self.best_reward_model_path = Path(best_reward_model_path)
            self.metadata = dict(metadata)
            self.target_laps = target_laps
            self.progress_state = progress_state
            self.best_progress_pace_score = None
            self.best_distance_m = None
            self.best_reward = None
            self._load_existing_bests()

        def _load_existing_bests(self):
            progress_pace = read_model_metadata(
                self.best_progress_pace_model_path
            ).get("best_progress_pace_episode")
            if isinstance(progress_pace, dict):
                score = finite_summary_value(progress_pace, "score", default=math.nan)
                if math.isfinite(score):
                    self.best_progress_pace_score = score
            distance = read_model_metadata(self.best_distance_model_path).get(
                "best_distance_episode"
            )
            if isinstance(distance, dict):
                best_distance = finite_summary_value(distance, "distance_m", default=math.nan)
                if math.isfinite(best_distance):
                    self.best_distance_m = best_distance
                    if self.progress_state is not None:
                        self.progress_state.record_best_distance(best_distance)
            reward = read_model_metadata(self.best_reward_model_path).get(
                "best_reward_episode"
            )
            if isinstance(reward, dict):
                best_reward = finite_summary_value(reward, "reward", default=math.nan)
                if math.isfinite(best_reward):
                    self.best_reward = best_reward
                    if self.progress_state is not None:
                        self.progress_state.record_best_reward(best_reward)

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue

                distance_m = finite_summary_value(summary, "distance_m")
                reward = finite_summary_value(summary, "reward")
                average_speed_kmh = finite_summary_value(summary, "average_speed_kmh")
                laps_completed = int(finite_summary_value(summary, "laps_completed"))
                lap_time = summary.get("best_lap_time_seconds")
                progress_pace_score = calculate_progress_pace_score(
                    summary,
                    target_laps=self.target_laps,
                )

                if (
                    self.best_progress_pace_score is None
                    or progress_pace_score > self.best_progress_pace_score
                ):
                    self.best_progress_pace_score = progress_pace_score
                    if self.progress_state is not None:
                        self.progress_state.record_best_progress_pace(
                            progress_pace_score,
                            distance_m,
                            average_speed_kmh,
                            laps_completed,
                            lap_time,
                        )
                    self._save_best_model(
                        self.best_progress_pace_model_path,
                        "best_progress_pace_episode",
                        {
                            "selection_reason": "progress-first, pace-aware curriculum selector",
                            "score": progress_pace_score,
                            "distance_m": distance_m,
                            "average_speed_kmh": average_speed_kmh,
                            "laps_completed": laps_completed,
                            "lap_time_seconds": lap_time,
                            "lap_completion_fraction": finite_summary_value(
                                summary,
                                "lap_completion_fraction",
                            ),
                            "termination_reason": summary.get("termination_reason"),
                            "episode_summary": summary,
                        },
                    )

                if self.best_distance_m is None or distance_m > self.best_distance_m:
                    self.best_distance_m = distance_m
                    if self.progress_state is not None:
                        self.progress_state.record_best_distance(distance_m)
                    self._save_best_model(
                        self.best_distance_model_path,
                        "best_distance_episode",
                        {
                            "selection_reason": "furthest distance reached in training",
                            "distance_m": distance_m,
                            "lap_completion_fraction": finite_summary_value(
                                summary,
                                "lap_completion_fraction",
                            ),
                            "termination_reason": summary.get("termination_reason"),
                            "episode_summary": summary,
                        },
                    )

                if self.best_reward is None or reward > self.best_reward:
                    self.best_reward = reward
                    if self.progress_state is not None:
                        self.progress_state.record_best_reward(reward)
                    self._save_best_model(
                        self.best_reward_model_path,
                        "best_reward_episode",
                        {
                            "selection_reason": "highest shaped episode reward",
                            "reward": reward,
                            "distance_m": distance_m,
                            "lap_completion_fraction": finite_summary_value(
                                summary,
                                "lap_completion_fraction",
                            ),
                            "termination_reason": summary.get("termination_reason"),
                            "episode_summary": summary,
                        },
                    )
            return True

        def _save_best_model(self, path, metadata_key, payload):
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            write_json(
                metadata_path_for_policy(path),
                {
                    **self.metadata,
                    metadata_key: {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "global_timestep": self.num_timesteps,
                        **payload,
                    },
                },
            )
            if self.verbose:
                print(f"Saved new best Agent 5 SAC model to {path}")

    return BestEpisodeModelCallback


def make_best_clean_lap_callback_class(BaseCallback):
    class BestCleanLapCallback(BaseCallback):
        def __init__(self, best_model_path, metadata, progress_state=None, verbose=0):
            super().__init__(verbose=verbose)
            self.best_model_path = Path(best_model_path)
            self.metadata = dict(metadata)
            self.progress_state = progress_state
            self.best_lap_time = self._load_existing_best_lap_time()
            if self.best_lap_time is not None and self.progress_state is not None:
                self.progress_state.record_best_clean_lap(self.best_lap_time)

        def _load_existing_best_lap_time(self):
            best_clean_lap = read_model_metadata(self.best_model_path).get(
                "best_clean_lap"
            )
            if not isinstance(best_clean_lap, dict):
                return None

            lap_time = finite_summary_value(
                best_clean_lap,
                "lap_time_seconds",
                default=math.nan,
            )
            return lap_time if math.isfinite(lap_time) else None

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                if summary.get("termination_reason") != "target_laps_completed":
                    continue
                if int(summary.get("off_track_steps", 0)) > 0:
                    continue
                if float(summary.get("max_stopped_seconds", 0.0)) >= 0.5:
                    continue

                lap_time = summary.get("best_lap_time_seconds")
                if lap_time is None:
                    lap_time = summary.get("duration_seconds")
                lap_time = float(lap_time)

                if self.best_lap_time is not None and lap_time >= self.best_lap_time:
                    continue

                self.best_lap_time = lap_time
                if self.progress_state is not None:
                    self.progress_state.record_best_clean_lap(lap_time)
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
                write_json(
                    metadata_path_for_policy(self.best_model_path),
                    {
                        **self.metadata,
                        "best_clean_lap": {
                            "saved_at": datetime.now(timezone.utc).isoformat(),
                            "global_timestep": self.num_timesteps,
                            "lap_time_seconds": lap_time,
                            "episode_summary": summary,
                        },
                    },
                )

                if self.verbose:
                    print(
                        "Saved new best clean Agent 5 SAC model "
                        f"({lap_time:.3f}s) to {self.best_model_path}"
                    )
            return True

    return BestCleanLapCallback


def make_console_progress_callback_class(BaseCallback):
    class ConsoleProgressCallback(BaseCallback):
        def __init__(
            self,
            total_timesteps,
            progress_state,
            *,
            interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
            verbose=0,
        ):
            super().__init__(verbose=verbose)
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
            self.last_render_time = 0.0
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

        def _render(self, *, now=None, force=False):
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


def make_checkpoint_callback(CheckpointCallback, args):
    checkpoint_kwargs = {
        "save_freq": args.checkpoint_freq,
        "save_path": str(args.checkpoint_dir),
        "name_prefix": "agent5_hybrid_sac",
        "save_replay_buffer": args.save_checkpoint_replay_buffer,
        "save_vecnormalize": False,
    }
    try:
        return CheckpointCallback(**checkpoint_kwargs)
    except TypeError:
        checkpoint_kwargs.pop("save_replay_buffer", None)
        return CheckpointCallback(**checkpoint_kwargs)


def load_replay_buffer_if_requested(model, args):
    if not args.load_replay_buffer:
        return False
    replay_buffer_path = Path(args.replay_buffer_path)
    if not replay_buffer_path.is_file():
        return False
    model.load_replay_buffer(str(replay_buffer_path))
    return True


def save_replay_buffer_if_requested(model, args):
    if not args.save_replay_buffer:
        return
    replay_buffer_path = Path(args.replay_buffer_path)
    replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_replay_buffer(str(replay_buffer_path))
    print(f"Saved Agent 5 replay buffer to {replay_buffer_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent 5 as a SAC residual on top of Agent 4.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--best-model-path", type=Path, default=DEFAULT_BEST_MODEL_PATH)
    parser.add_argument(
        "--best-progress-pace-model-path",
        type=Path,
        default=DEFAULT_BEST_PROGRESS_PACE_MODEL_PATH,
    )
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
    parser.add_argument("--replay-buffer-path", type=Path, default=DEFAULT_REPLAY_BUFFER_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tensorboard-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR)
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help=(
            "Disable Stable-Baselines3 TensorBoard logging. TensorBoard is "
            "enabled by default when the package is installed."
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the clean one-line console progress display.",
    )
    parser.add_argument(
        "--verbose-training",
        action="store_true",
        help="Show Stable-Baselines3's detailed SAC training tables.",
    )
    parser.add_argument(
        "--show-torcs-reset-log",
        action="store_true",
        help="Show TORCS reset countdown/socket logs instead of hiding them.",
    )
    parser.add_argument("--track", default=DEFAULT_TRACK_NAME)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-source",
        choices=RESUME_SOURCE_CHOICES,
        default="auto",
        help=(
            "Which saved policy to continue from when --resume is used. Auto "
            "prefers clean-lap best, then progress+pace best, distance, reward, final."
        ),
    )
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--max-episode-steps", type=int, default=12_000)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--manual-start",
        action="store_true",
        help=(
            "Do not launch or close TORCS. Connect to an already-running "
            "Practice/SCR server and reset it between learning episodes."
        ),
    )
    parser.add_argument(
        "--relaunch-frequency",
        type=int,
        default=0,
        help=(
            "In automatic mode, relaunch TORCS every N episodes. Use 0 to "
            "disable periodic relaunch."
        ),
    )
    parser.add_argument(
        "--reset-retries",
        type=int,
        default=2,
        help="Retry failed TORCS resets this many times before stopping training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--target-update-interval", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save SAC replay buffer beside model artifacts.",
    )
    parser.add_argument(
        "--save-checkpoint-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also save replay buffers at every checkpoint. Disabled by default "
            "because each checkpoint buffer can be large."
        ),
    )
    parser.add_argument(
        "--load-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, load --replay-buffer-path if it exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # gym_torcs/snakeoil parses sys.argv when it opens the TORCS socket.
    # Keep this training script's CLI flags away from that legacy parser.
    sys.argv = [sys.argv[0]]

    random.seed(args.seed)
    np.random.seed(args.seed)

    (
        gym,
        spaces,
        SAC,
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    ) = import_training_dependencies()
    env_class = make_training_env_class(gym, spaces)
    run_dir = args.run_dir if args.run_dir is not None else make_default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = env_class(
        track_name=args.track,
        residual_scale=args.residual_scale,
        max_episode_steps=args.max_episode_steps,
        target_laps=args.target_laps,
        manual_start=args.manual_start,
        relaunch_frequency=args.relaunch_frequency,
        reset_retries=args.reset_retries,
        quiet_reset_log=not args.show_torcs_reset_log,
        run_dir=run_dir,
    )
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_progress_pace_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_distance_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_reward_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = resolve_tensorboard_log_dir(
        args.tensorboard_dir,
        disable_tensorboard=args.no_tensorboard,
    )
    resumed_metadata = None

    resume_model_path = resolve_resume_model_path(args)
    if args.resume and resume_model_path.is_file():
        resume_model_metadata = validate_resume_metadata(
            resume_model_path,
            force_resume=args.force_resume,
        )
        resumed_metadata = {
            "resume_source": args.resume_source,
            "model_path": str(resume_model_path),
            "metadata": resume_model_metadata,
        }
        print(f"Resuming Agent 5 SAC from {resume_model_path}")
        model = SAC.load(
            str(resume_model_path),
            env=env,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
        model.verbose = int(args.verbose_training)
        loaded_replay = load_replay_buffer_if_requested(model, args)
        if loaded_replay:
            print(f"Loaded Agent 5 replay buffer from {args.replay_buffer_path}")
        reset_num_timesteps = False
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=int(args.verbose_training),
            tensorboard_log=tensorboard_log,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            learning_rate=args.learning_rate,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=args.ent_coef,
            target_update_interval=args.target_update_interval,
            policy_kwargs={"net_arch": [256, 256]},
            seed=args.seed,
            device=args.device,
        )
        reset_num_timesteps = True

    metadata = make_training_metadata(
        args,
        resumed_from=resumed_metadata,
    )
    write_json(run_dir / "training_metadata.json", metadata)
    write_json(metadata_path_for_policy(args.model_path), metadata)

    progress_state = TrainingProgressState()
    checkpoint_callback = make_checkpoint_callback(CheckpointCallback, args)
    EpisodeSummaryCallback = make_episode_summary_callback_class(BaseCallback)
    BestEpisodeModelCallback = make_best_episode_model_callback_class(BaseCallback)
    BestCleanLapCallback = make_best_clean_lap_callback_class(BaseCallback)
    callbacks = [
        checkpoint_callback,
        EpisodeSummaryCallback(
            run_dir,
            progress_state=progress_state,
            verbose=int(args.verbose_training),
        ),
        BestEpisodeModelCallback(
            args.best_progress_pace_model_path,
            args.best_distance_model_path,
            args.best_reward_model_path,
            metadata,
            target_laps=args.target_laps,
            progress_state=progress_state,
            verbose=int(args.verbose_training),
        ),
        BestCleanLapCallback(
            args.best_model_path,
            metadata,
            progress_state=progress_state,
            verbose=int(args.verbose_training),
        ),
    ]
    if not args.no_progress_bar and not args.verbose_training:
        ConsoleProgressCallback = make_console_progress_callback_class(BaseCallback)
        callbacks.append(
            ConsoleProgressCallback(
                args.total_timesteps,
                progress_state,
            )
        )
    callback = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            log_interval=10,
        )
        end_summary = env.write_end_of_run_summary("total_timesteps_reached")
        model.save(str(args.model_path))
        save_replay_buffer_if_requested(model, args)
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        if end_summary is not None:
            metadata["end_of_run_summary"] = end_summary
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        print(f"Saved Agent 5 SAC model to {args.model_path}")
        print(f"Saved training run logs to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
