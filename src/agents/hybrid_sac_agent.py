import csv
import json
import math
from pathlib import Path

import numpy as np

from agents.map_aware_agent import (
    DEFAULT_RACING_LINE_PATH,
    MapAwareAgent,
    PROJECT_ROOT,
    clamp,
    get_track_sensors,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "agent5_hybrid_sac.zip"
DEFAULT_BEST_MODEL_PATH = PROJECT_ROOT / "models" / "agent5_hybrid_sac_best.zip"
DEFAULT_BEST_PROGRESS_PACE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent5_hybrid_sac_best_progress_pace.zip"
)
DEFAULT_BEST_DISTANCE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent5_hybrid_sac_best_distance.zip"
)
DEFAULT_BEST_REWARD_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent5_hybrid_sac_best_reward.zip"
)
DEFAULT_REPLAY_BUFFER_PATH = (
    PROJECT_ROOT / "models" / "replay_buffers" / "agent5_hybrid_sac.pkl"
)
DEFAULT_BASE_TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "generated" / "hybrid_sac_base_telemetry.csv"
)
DEFAULT_RESIDUAL_TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "generated" / "hybrid_sac_residual_telemetry.csv"
)

AGENT5_MODEL_FAMILY = "agent5_hybrid_sac_residual"
AGENT5_OBSERVATION_VERSION = "agent5_observation_v1_map_phase_teacher"
AGENT5_ACTION_VERSION = "agent5_residual_action_v1"
SAC_AUTHORITY_PROFILE = "agent5_supervised_residual_authority_v1"

MAX_STEER_RESIDUAL = 0.080
MAX_ACCEL_RESIDUAL = 0.180
MAX_BRAKE_RESIDUAL = 0.220
DEFAULT_RESIDUAL_SCALE = 0.10
FINITE_DEFAULT = 0.0

RISK_TRACK_LIMIT = 0.84
RISK_ANGLE_LIMIT = 0.38
RISK_SLIDE_LIMIT = 8.0
RISK_FRONT_SENSOR_LIMIT = 22.0
HIGH_RISK_PRESSURE = 0.55
CRITICAL_RISK_PRESSURE = 0.75
LOW_SPEED_RECOVERY_SPEED_LIMIT = 8.0
LOW_SPEED_RECOVERY_TRACK_LIMIT = 0.98
LOW_SPEED_RECOVERY_ANGLE_LIMIT = 0.70
LOW_SPEED_RECOVERY_SLIDE_LIMIT = 4.0
SAC_AUTHORITY_LAUNCH_FREE_DISTANCE_M = 80.0
SAC_AUTHORITY_LAUNCH_FULL_DISTANCE_M = 220.0
SAC_AUTHORITY_MIN_SPEED_KMH = 25.0
SAC_AUTHORITY_FULL_SPEED_KMH = 75.0
SAC_AUTHORITY_STABLE_TRACK_LIMIT = 0.42
SAC_AUTHORITY_EDGE_TRACK_LIMIT = 0.82
SAC_AUTHORITY_STABLE_LINE_ERROR = 0.24
SAC_AUTHORITY_LINE_ERROR_LIMIT = 0.62
SAC_AUTHORITY_STABLE_ANGLE_RAD = 0.14
SAC_AUTHORITY_ANGLE_LIMIT_RAD = RISK_ANGLE_LIMIT
SAC_AUTHORITY_STABLE_SLIDE_KMH = 3.0
SAC_AUTHORITY_SLIDE_LIMIT_KMH = RISK_SLIDE_LIMIT
SAC_AUTHORITY_FRONT_SENSOR_MIN_M = 26.0
SAC_AUTHORITY_FRONT_SENSOR_FULL_M = 70.0

FEATURE_NAMES = [
    "speed_x",
    "speed_y",
    "angle",
    "track_pos",
    "front_sensor",
    "front_left_sensor",
    "front_right_sensor",
    "far_left_sensor",
    "far_right_sensor",
    "min_track_sensor",
    "wheel_spin_balance",
    "damage",
    "lap_time",
    "distance_phase",
    "distance_phase_sin",
    "distance_phase_cos",
    "racing_line_target_pos",
    "racing_line_error",
    "racing_line_target_speed",
    "racing_line_curvature",
    "racing_line_heading_offset",
    "racing_line_speed_delta_30m",
    "racing_line_turn_direction",
    "racing_line_reference_throttle",
    "racing_line_reference_brake",
    "base_steer",
    "base_accel",
    "base_brake",
    "base_gear",
    "base_pedal_balance",
]

RESIDUAL_TELEMETRY_COLUMNS = [
    "step",
    "distFromStart",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
    "frontSensor",
    "minTrackSensor",
    "racingLineTargetPos",
    "racingLineError",
    "racingLineTargetSpeed",
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
    "safetyShieldActive",
    "riskPressure",
    "unsafeResidualPressure",
    "lowSpeedRecoveryActive",
    "policyLoaded",
]


def finite_float(value, default=FINITE_DEFAULT):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(number):
        return number
    return default


def finite_clamp(value, lower, upper, default=FINITE_DEFAULT):
    return clamp(finite_float(value, default), lower, upper)


def normalise_sensor(value):
    return finite_clamp(finite_float(value) / 200.0, -1.0, 1.0)


def normalise_signed(value, scale):
    return finite_clamp(finite_float(value) / scale, -1.0, 1.0)


def ramp_score(value, lower, upper):
    if upper <= lower:
        return 1.0 if value >= upper else 0.0
    return clamp((value - lower) / (upper - lower), 0.0, 1.0)


def stable_window_score(abs_value, stable_limit, unsafe_limit):
    if unsafe_limit <= stable_limit:
        return 1.0 if abs_value <= stable_limit else 0.0
    return 1.0 - clamp(
        (abs_value - stable_limit) / (unsafe_limit - stable_limit),
        0.0,
        1.0,
    )


def get_sac_authority_config():
    return {
        "profile": SAC_AUTHORITY_PROFILE,
        "launch_free_distance_m": SAC_AUTHORITY_LAUNCH_FREE_DISTANCE_M,
        "launch_full_distance_m": SAC_AUTHORITY_LAUNCH_FULL_DISTANCE_M,
        "min_speed_kmh": SAC_AUTHORITY_MIN_SPEED_KMH,
        "full_speed_kmh": SAC_AUTHORITY_FULL_SPEED_KMH,
        "stable_track_limit": SAC_AUTHORITY_STABLE_TRACK_LIMIT,
        "edge_track_limit": SAC_AUTHORITY_EDGE_TRACK_LIMIT,
        "stable_line_error": SAC_AUTHORITY_STABLE_LINE_ERROR,
        "line_error_limit": SAC_AUTHORITY_LINE_ERROR_LIMIT,
        "stable_angle_rad": SAC_AUTHORITY_STABLE_ANGLE_RAD,
        "angle_limit_rad": SAC_AUTHORITY_ANGLE_LIMIT_RAD,
        "stable_slide_kmh": SAC_AUTHORITY_STABLE_SLIDE_KMH,
        "slide_limit_kmh": SAC_AUTHORITY_SLIDE_LIMIT_KMH,
        "front_sensor_min_m": SAC_AUTHORITY_FRONT_SENSOR_MIN_M,
        "front_sensor_full_m": SAC_AUTHORITY_FRONT_SENSOR_FULL_M,
    }


def get_racing_line_snapshot(telemetry, racing_line=None):
    if racing_line is None:
        return {
            "target_track_pos": 0.0,
            "line_error": 0.0,
            "target_speed_kmh": 0.0,
            "curvature": 0.0,
            "heading_offset": 0.0,
            "speed_delta_30m": 0.0,
            "turn_direction": 0.0,
            "reference_throttle": 0.0,
            "reference_brake": 0.0,
        }

    distance = finite_float(telemetry.get("distFromStart", 0.0))
    target = racing_line.lookup(distance)
    target_position = finite_float(target.get("target_track_pos", 0.0))
    track_position = finite_float(telemetry.get("trackPos", 0.0))
    return {
        "target_track_pos": target_position,
        "line_error": track_position - target_position,
        "target_speed_kmh": finite_float(target.get("target_speed_kmh", 0.0)),
        "curvature": finite_float(target.get("curvature", 0.0)),
        "heading_offset": finite_float(target.get("heading_offset", 0.0)),
        "speed_delta_30m": finite_float(target.get("speed_delta_30m", 0.0)),
        "turn_direction": finite_float(target.get("turn_direction", 0.0)),
        "reference_throttle": finite_float(target.get("reference_throttle", 0.0)),
        "reference_brake": finite_float(target.get("reference_brake", 0.0)),
    }


def build_hybrid_sac_observation(
    telemetry,
    base_action,
    racing_line=None,
    track_length=None,
):
    """Build the stable, normalized observation used by the residual SAC policy."""

    if track_length is None and racing_line is not None:
        track_length = getattr(racing_line, "track_length", None)

    track_sensors = get_track_sensors(telemetry)
    front_sensor = track_sensors[9]
    min_track_sensor = min(track_sensors) if track_sensors else front_sensor
    wheel_spin = telemetry.get("wheelSpinVel") or [0.0, 0.0, 0.0, 0.0]
    wheel_spin_values = [finite_float(value) for value in list(wheel_spin)[:4]]
    while len(wheel_spin_values) < 4:
        wheel_spin_values.append(0.0)
    front_wheel_spin = (wheel_spin_values[0] + wheel_spin_values[1]) / 2.0
    rear_wheel_spin = (wheel_spin_values[2] + wheel_spin_values[3]) / 2.0
    distance = finite_float(telemetry.get("distFromStart", 0.0))

    if track_length:
        distance_phase = (distance % track_length) / track_length
    else:
        distance_phase = 0.0
    distance_phase_angle = 2.0 * math.pi * distance_phase

    line = get_racing_line_snapshot(telemetry, racing_line)

    return [
        normalise_signed(telemetry.get("speedX", 0.0), 240.0),
        normalise_signed(telemetry.get("speedY", 0.0), 40.0),
        normalise_signed(telemetry.get("angle", 0.0), 1.0),
        normalise_signed(telemetry.get("trackPos", 0.0), 1.4),
        normalise_sensor(front_sensor),
        normalise_sensor(track_sensors[7]),
        normalise_sensor(track_sensors[11]),
        normalise_sensor(track_sensors[0]),
        normalise_sensor(track_sensors[18]),
        normalise_sensor(min_track_sensor),
        normalise_signed(rear_wheel_spin - front_wheel_spin, 70.0),
        normalise_signed(telemetry.get("damage", 0.0), 10000.0),
        normalise_signed(telemetry.get("curLapTime", 0.0), 120.0),
        finite_clamp(distance_phase, 0.0, 1.0),
        math.sin(distance_phase_angle),
        math.cos(distance_phase_angle),
        normalise_signed(line["target_track_pos"], 1.0),
        normalise_signed(line["line_error"], 1.4),
        normalise_signed(line["target_speed_kmh"], 240.0),
        normalise_signed(line["curvature"], 0.05),
        normalise_signed(line["heading_offset"], 0.8),
        normalise_signed(line["speed_delta_30m"], 90.0),
        normalise_signed(line["turn_direction"], 1.0),
        finite_clamp(line["reference_throttle"], 0.0, 1.0),
        finite_clamp(line["reference_brake"], 0.0, 1.0),
        finite_clamp(base_action.get("steer", 0.0), -1.0, 1.0),
        finite_clamp(base_action.get("accel", 0.0), 0.0, 1.0),
        finite_clamp(base_action.get("brake", 0.0), 0.0, 1.0),
        finite_clamp(finite_float(base_action.get("gear", 1)) / 6.0, 0.0, 1.0),
        finite_clamp(
            finite_float(base_action.get("accel", 0.0))
            - finite_float(base_action.get("brake", 0.0)),
            -1.0,
            1.0,
        ),
    ]


def resolve_default_policy_path():
    if DEFAULT_BEST_MODEL_PATH.is_file():
        return DEFAULT_BEST_MODEL_PATH
    if DEFAULT_BEST_PROGRESS_PACE_MODEL_PATH.is_file():
        return DEFAULT_BEST_PROGRESS_PACE_MODEL_PATH
    if DEFAULT_BEST_DISTANCE_MODEL_PATH.is_file():
        return DEFAULT_BEST_DISTANCE_MODEL_PATH
    if DEFAULT_BEST_REWARD_MODEL_PATH.is_file():
        return DEFAULT_BEST_REWARD_MODEL_PATH
    return DEFAULT_MODEL_PATH


def metadata_path_for_policy(policy_path):
    return Path(policy_path).with_suffix(".metadata.json")


def read_policy_metadata(policy_path):
    if policy_path is None:
        return {}

    metadata_path = metadata_path_for_policy(policy_path)
    if not metadata_path.is_file():
        return {}

    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return {}

    return metadata if isinstance(metadata, dict) else {}


def find_policy_contract_mismatches(metadata):
    if not metadata:
        return []

    expected = {
        "model_family": AGENT5_MODEL_FAMILY,
        "observation_version": AGENT5_OBSERVATION_VERSION,
        "action_version": AGENT5_ACTION_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
    }
    return [
        key
        for key, value in expected.items()
        if metadata.get(key) != value
    ]


def resolve_residual_scale(policy_path, residual_scale=None):
    if residual_scale is not None:
        return max(0.0, finite_float(residual_scale, DEFAULT_RESIDUAL_SCALE))

    metadata = read_policy_metadata(policy_path)
    return max(
        0.0,
        finite_float(metadata.get("residual_scale"), DEFAULT_RESIDUAL_SCALE),
    )


def get_residual_limits(residual_scale=1.0):
    scale = max(0.0, float(residual_scale))
    return [
        MAX_STEER_RESIDUAL * scale,
        MAX_ACCEL_RESIDUAL * scale,
        MAX_BRAKE_RESIDUAL * scale,
    ]


def coerce_policy_residual(raw_residual):
    if raw_residual is None:
        values = np.asarray([], dtype=float)
    else:
        try:
            values = np.asarray(raw_residual, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)

    padded = np.zeros(3, dtype=float)
    padded[: min(3, len(values))] = values[:3]
    return [finite_float(padded[0]), finite_float(padded[1]), finite_float(padded[2])]


def decode_policy_residual(raw_residual, residual_scale=1.0):
    """Convert a SAC action in [-1, 1] into bounded control deltas."""

    values = coerce_policy_residual(raw_residual)
    limits = get_residual_limits(residual_scale)
    return [
        finite_clamp(values[0], -1.0, 1.0) * limits[0],
        finite_clamp(values[1], -1.0, 1.0) * limits[1],
        finite_clamp(values[2], -1.0, 1.0) * limits[2],
    ]


def calculate_risk_pressure(telemetry):
    track_sensors = get_track_sensors(telemetry)
    front_sensor = track_sensors[9]
    min_track_sensor = min(track_sensors) if track_sensors else front_sensor
    track_position = finite_float(telemetry.get("trackPos", 0.0))
    angle = finite_float(telemetry.get("angle", 0.0))
    lateral_speed = finite_float(telemetry.get("speedY", 0.0))

    return max(
        1.0 if min_track_sensor < 0.0 else 0.0,
        clamp((abs(track_position) - RISK_TRACK_LIMIT) / 0.24, 0.0, 1.0),
        clamp((abs(angle) - RISK_ANGLE_LIMIT) / 0.28, 0.0, 1.0),
        clamp((abs(lateral_speed) - RISK_SLIDE_LIMIT) / 12.0, 0.0, 1.0),
        clamp(
            (RISK_FRONT_SENSOR_LIMIT - front_sensor) / RISK_FRONT_SENSOR_LIMIT,
            0.0,
            1.0,
        ),
    )


def calculate_sac_authority(telemetry, racing_line=None):
    """Return how much freedom SAC has over the teacher controller this step."""

    track_sensors = get_track_sensors(telemetry)
    front_sensor = track_sensors[9]
    min_track_sensor = min(track_sensors) if track_sensors else front_sensor
    if min_track_sensor < 0.0:
        return 0.0

    speed = finite_float(telemetry.get("speedX", 0.0))
    track_position = finite_float(telemetry.get("trackPos", 0.0))
    angle = finite_float(telemetry.get("angle", 0.0))
    lateral_speed = finite_float(telemetry.get("speedY", 0.0))
    dist_raced = max(0.0, finite_float(telemetry.get("distRaced", 0.0)))
    line = get_racing_line_snapshot(telemetry, racing_line)
    line_error = finite_float(line.get("line_error", 0.0))
    risk_score = 1.0 - clamp(calculate_risk_pressure(telemetry), 0.0, 1.0)

    authority = min(
        ramp_score(
            dist_raced,
            SAC_AUTHORITY_LAUNCH_FREE_DISTANCE_M,
            SAC_AUTHORITY_LAUNCH_FULL_DISTANCE_M,
        ),
        ramp_score(
            speed,
            SAC_AUTHORITY_MIN_SPEED_KMH,
            SAC_AUTHORITY_FULL_SPEED_KMH,
        ),
        stable_window_score(
            abs(track_position),
            SAC_AUTHORITY_STABLE_TRACK_LIMIT,
            SAC_AUTHORITY_EDGE_TRACK_LIMIT,
        ),
        stable_window_score(
            abs(line_error),
            SAC_AUTHORITY_STABLE_LINE_ERROR,
            SAC_AUTHORITY_LINE_ERROR_LIMIT,
        ),
        stable_window_score(
            abs(angle),
            SAC_AUTHORITY_STABLE_ANGLE_RAD,
            SAC_AUTHORITY_ANGLE_LIMIT_RAD,
        ),
        stable_window_score(
            abs(lateral_speed),
            SAC_AUTHORITY_STABLE_SLIDE_KMH,
            SAC_AUTHORITY_SLIDE_LIMIT_KMH,
        ),
        ramp_score(
            front_sensor,
            SAC_AUTHORITY_FRONT_SENSOR_MIN_M,
            SAC_AUTHORITY_FRONT_SENSOR_FULL_M,
        ),
        risk_score,
    )
    if low_speed_recovery_is_active(telemetry):
        authority = min(authority, 0.15)
    return clamp(authority, 0.0, 1.0)


def low_speed_recovery_is_active(telemetry):
    track_sensors = get_track_sensors(telemetry)
    min_track_sensor = min(track_sensors) if track_sensors else 0.0
    return (
        abs(finite_float(telemetry.get("speedX", 0.0)))
        < LOW_SPEED_RECOVERY_SPEED_LIMIT
        and min_track_sensor >= 0.0
        and abs(finite_float(telemetry.get("trackPos", 0.0)))
        < LOW_SPEED_RECOVERY_TRACK_LIMIT
        and abs(finite_float(telemetry.get("angle", 0.0)))
        < LOW_SPEED_RECOVERY_ANGLE_LIMIT
        and abs(finite_float(telemetry.get("speedY", 0.0)))
        < LOW_SPEED_RECOVERY_SLIDE_LIMIT
    )


def residual_worsens_recovery_path(residual, telemetry):
    steer_delta = finite_float(residual[0])
    track_position = finite_float(telemetry.get("trackPos", 0.0))
    angle = finite_float(telemetry.get("angle", 0.0))

    if abs(track_position) > RISK_TRACK_LIMIT and steer_delta * track_position > 0.0:
        return True
    if abs(angle) > RISK_ANGLE_LIMIT and steer_delta * angle > 0.0:
        return True
    return False


def calculate_unsafe_residual_pressure(base_action, residual, telemetry):
    steer_delta, accel_delta, brake_delta = residual
    track_position = finite_float(telemetry.get("trackPos", 0.0))
    angle = finite_float(telemetry.get("angle", 0.0))
    lateral_speed = finite_float(telemetry.get("speedY", 0.0))
    risk_pressure = calculate_risk_pressure(telemetry)
    base_brake = finite_float(base_action.get("brake", 0.0))
    recovery_drive_allowed = (
        low_speed_recovery_is_active(telemetry)
        and not residual_worsens_recovery_path(residual, telemetry)
    )

    outward_edge_pressure = 0.0
    if abs(track_position) > RISK_TRACK_LIMIT and steer_delta * track_position > 0.0:
        outward_edge_pressure = clamp(
            abs(steer_delta) / MAX_STEER_RESIDUAL,
            0.0,
            1.0,
        )

    heading_pressure = 0.0
    if abs(angle) > RISK_ANGLE_LIMIT and steer_delta * angle > 0.0:
        heading_pressure = clamp(abs(steer_delta) / MAX_STEER_RESIDUAL, 0.0, 1.0)

    throttle_pressure = 0.0
    if (
        risk_pressure > HIGH_RISK_PRESSURE
        and accel_delta > 0.0
        and not recovery_drive_allowed
    ):
        throttle_pressure = clamp(accel_delta / MAX_ACCEL_RESIDUAL, 0.0, 1.0)

    brake_release_pressure = 0.0
    if (
        risk_pressure > HIGH_RISK_PRESSURE
        or base_brake > 0.08
    ) and brake_delta < 0.0 and not recovery_drive_allowed:
        brake_release_pressure = clamp(abs(brake_delta) / MAX_BRAKE_RESIDUAL, 0.0, 1.0)

    slide_pressure = 0.0
    if abs(lateral_speed) > RISK_SLIDE_LIMIT and accel_delta > 0.0:
        slide_pressure = clamp(accel_delta / MAX_ACCEL_RESIDUAL, 0.0, 1.0)

    return max(
        outward_edge_pressure,
        heading_pressure,
        throttle_pressure,
        brake_release_pressure,
        slide_pressure,
    )


def shield_residual_for_safety(base_action, residual, telemetry):
    """Prevent SAC corrections from fighting the map-aware controller."""

    original_residual = list(residual)
    steer_delta, accel_delta, brake_delta = residual
    risk_pressure = calculate_risk_pressure(telemetry)
    unsafe_pressure = calculate_unsafe_residual_pressure(
        base_action,
        residual,
        telemetry,
    )
    recovery_drive_allowed = (
        low_speed_recovery_is_active(telemetry)
        and not residual_worsens_recovery_path(residual, telemetry)
    )
    if risk_pressure <= 0.0 and unsafe_pressure <= 0.0:
        return residual, False, risk_pressure, unsafe_pressure, recovery_drive_allowed

    track_position = finite_float(telemetry.get("trackPos", 0.0))
    angle = finite_float(telemetry.get("angle", 0.0))

    if abs(track_position) > RISK_TRACK_LIMIT and steer_delta * track_position > 0.0:
        steer_delta = 0.0
    elif abs(angle) > RISK_ANGLE_LIMIT and steer_delta * angle > 0.0:
        steer_delta *= 0.35
    else:
        steer_delta *= clamp(1.0 - 0.45 * risk_pressure, 0.55, 1.0)

    if (
        risk_pressure > HIGH_RISK_PRESSURE or unsafe_pressure > 0.0
    ) and not recovery_drive_allowed:
        accel_delta = min(accel_delta, 0.0)
    if (
        finite_float(base_action.get("brake", 0.0)) > 0.08
        or risk_pressure > CRITICAL_RISK_PRESSURE
    ) and not recovery_drive_allowed:
        brake_delta = max(brake_delta, 0.0)

    residual = [steer_delta, accel_delta, brake_delta]
    shield_active = any(
        abs(current - original) > 1e-12
        for current, original in zip(residual, original_residual)
    )
    return residual, shield_active, risk_pressure, unsafe_pressure, recovery_drive_allowed


def apply_hybrid_sac_residual(
    base_action,
    raw_residual,
    telemetry,
    residual_scale=1.0,
    racing_line=None,
    sac_authority=None,
):
    raw_values = coerce_policy_residual(raw_residual)
    residual = decode_policy_residual(raw_residual, residual_scale)
    if sac_authority is None:
        sac_authority = calculate_sac_authority(telemetry, racing_line)
    sac_authority = clamp(finite_float(sac_authority, 0.0), 0.0, 1.0)
    ungated_residual = list(residual)
    residual = [value * sac_authority for value in residual]
    (
        residual,
        shield_active,
        risk_pressure,
        unsafe_pressure,
        recovery_active,
    ) = shield_residual_for_safety(base_action, residual, telemetry)

    action = dict(base_action)
    action["steer"] = finite_clamp(
        finite_float(base_action.get("steer", 0.0)) + residual[0],
        -1.0,
        1.0,
    )
    action["accel"] = finite_clamp(
        finite_float(base_action.get("accel", 0.0)) + residual[1],
        0.0,
        1.0,
    )
    action["brake"] = finite_clamp(
        finite_float(base_action.get("brake", 0.0)) + residual[2],
        0.0,
        1.0,
    )

    if action["brake"] > 0.03:
        action["accel"] = 0.0
    elif action["accel"] > 0.03:
        action["brake"] = 0.0

    action["agent5_base_steer"] = finite_float(base_action.get("steer", 0.0))
    action["agent5_base_accel"] = finite_float(base_action.get("accel", 0.0))
    action["agent5_base_brake"] = finite_float(base_action.get("brake", 0.0))
    action["agent5_raw_steer_residual"] = raw_values[0]
    action["agent5_raw_accel_residual"] = raw_values[1]
    action["agent5_raw_brake_residual"] = raw_values[2]
    action["agent5_ungated_steer_residual"] = ungated_residual[0]
    action["agent5_ungated_accel_residual"] = ungated_residual[1]
    action["agent5_ungated_brake_residual"] = ungated_residual[2]
    action["agent5_steer_residual"] = residual[0]
    action["agent5_accel_residual"] = residual[1]
    action["agent5_brake_residual"] = residual[2]
    action["agent5_sac_authority"] = sac_authority
    action["agent5_teacher_authority"] = 1.0 - sac_authority
    action["agent5_authority_gate_active"] = (
        sac_authority < 0.999
        and any(abs(value) > 1e-9 for value in ungated_residual)
    )
    action["agent5_safety_shield_active"] = shield_active
    action["agent5_risk_pressure"] = risk_pressure
    action["agent5_unsafe_residual_pressure"] = unsafe_pressure
    action["agent5_low_speed_recovery_active"] = recovery_active

    return action


class HybridSacAgent:
    name = "Hybrid SAC Residual Agent"
    agent_type = "hybrid_sac"
    version = "0.1"
    seed = None
    uses_full_control = True
    max_steps = MapAwareAgent.max_steps
    target_laps = MapAwareAgent.target_laps

    def __init__(
        self,
        racing_line_path=DEFAULT_RACING_LINE_PATH,
        policy_path=None,
        policy=None,
        residual_scale=None,
        base_telemetry_path=DEFAULT_BASE_TELEMETRY_PATH,
        residual_telemetry_path=DEFAULT_RESIDUAL_TELEMETRY_PATH,
        require_policy=False,
        deterministic=True,
    ):
        if policy_path is None:
            policy_path = resolve_default_policy_path()
        self.policy_path = Path(policy_path) if policy_path is not None else None
        self.policy = policy
        self.policy_loaded = policy is not None
        self.policy_metadata = read_policy_metadata(self.policy_path)
        self.residual_scale = resolve_residual_scale(self.policy_path, residual_scale)
        self.base_agent = MapAwareAgent(
            racing_line_path=racing_line_path,
            telemetry_path=base_telemetry_path,
        )
        self.residual_telemetry_path = Path(residual_telemetry_path)
        self.deterministic = bool(deterministic)
        self._residual_telemetry_file = None
        self._residual_telemetry_writer = None
        self.step = 0

        mismatches = find_policy_contract_mismatches(self.policy_metadata)
        if mismatches:
            formatted = ", ".join(mismatches)
            raise ValueError(
                "Hybrid SAC policy metadata is incompatible with this runtime: "
                f"{formatted}"
            )

        if self.policy is None:
            self.policy = self._load_policy(require_policy=require_policy)
            self.policy_loaded = self.policy is not None

        self.reset()

    @property
    def config(self):
        policy_path = None
        if self.policy_path is not None:
            try:
                policy_path = str(self.policy_path.relative_to(PROJECT_ROOT))
            except ValueError:
                policy_path = str(self.policy_path)

        return {
            "model_family": AGENT5_MODEL_FAMILY,
            "observation_version": AGENT5_OBSERVATION_VERSION,
            "action_version": AGENT5_ACTION_VERSION,
            "control_profile": SAC_AUTHORITY_PROFILE,
            "base_agent": self.base_agent.name,
            "base_agent_version": self.base_agent.version,
            "base_agent_type": self.base_agent.agent_type,
            "policy_path": policy_path,
            "policy_loaded": self.policy_loaded,
            "policy_metadata_loaded": bool(self.policy_metadata),
            "deterministic": self.deterministic,
            "residual_scale": self.residual_scale,
            "residual_limits": {
                "steer": MAX_STEER_RESIDUAL * self.residual_scale,
                "accel": MAX_ACCEL_RESIDUAL * self.residual_scale,
                "brake": MAX_BRAKE_RESIDUAL * self.residual_scale,
            },
            "sac_authority": get_sac_authority_config(),
            "observation_features": FEATURE_NAMES,
            "replay_buffer_path": str(
                DEFAULT_REPLAY_BUFFER_PATH.relative_to(PROJECT_ROOT)
            ),
            "safety_shield": {
                "risk_track_limit": RISK_TRACK_LIMIT,
                "risk_angle_limit": RISK_ANGLE_LIMIT,
                "risk_slide_limit": RISK_SLIDE_LIMIT,
                "risk_front_sensor_limit": RISK_FRONT_SENSOR_LIMIT,
                "high_risk_pressure": HIGH_RISK_PRESSURE,
                "critical_risk_pressure": CRITICAL_RISK_PRESSURE,
                "low_speed_recovery_speed_limit": LOW_SPEED_RECOVERY_SPEED_LIMIT,
                "low_speed_recovery_track_limit": LOW_SPEED_RECOVERY_TRACK_LIMIT,
                "low_speed_recovery_angle_limit": LOW_SPEED_RECOVERY_ANGLE_LIMIT,
                "low_speed_recovery_slide_limit": LOW_SPEED_RECOVERY_SLIDE_LIMIT,
            },
        }

    @property
    def racing_line(self):
        return self.base_agent.racing_line

    def reset(self):
        self.base_agent.reset()
        self.step = 0
        self._open_residual_telemetry()

    def close(self):
        self.base_agent.close()
        if self._residual_telemetry_file is not None:
            self._residual_telemetry_file.close()
        self._residual_telemetry_file = None
        self._residual_telemetry_writer = None

    def act(self, observation, telemetry=None):
        if telemetry is None:
            raise ValueError("HybridSacAgent requires raw TORCS telemetry")

        base_action = self.base_agent.act(observation, telemetry)
        if base_action.get("terminate", False):
            return base_action

        features = build_hybrid_sac_observation(
            telemetry,
            base_action,
            self.base_agent.racing_line,
        )
        raw_residual = self._predict_residual(features)
        action = apply_hybrid_sac_residual(
            base_action,
            raw_residual,
            telemetry,
            self.residual_scale,
            self.base_agent.racing_line,
        )
        action["agent5_policy_loaded"] = self.policy_loaded

        self._write_residual_telemetry(telemetry, action)
        self.step += 1
        return action

    def _load_policy(self, require_policy=False):
        if self.policy_path is None or not self.policy_path.is_file():
            if require_policy:
                raise FileNotFoundError(
                    "Hybrid SAC model not found. Train Agent 5 first or run "
                    f"without require_policy: {self.policy_path}"
                )
            return None

        if require_policy and not self.policy_metadata:
            raise FileNotFoundError(
                "Hybrid SAC model metadata not found. Refusing strict deployment "
                f"load for {self.policy_path}"
            )

        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            if require_policy:
                raise ImportError(
                    "stable-baselines3 is required to load the Hybrid SAC model"
                ) from exc
            return None

        return SAC.load(str(self.policy_path))

    def _predict_residual(self, features):
        if self.policy is None:
            return [0.0, 0.0, 0.0]

        policy_observation = np.asarray(features, dtype=np.float32)
        prediction = self.policy.predict(
            policy_observation,
            deterministic=self.deterministic,
        )
        raw_residual = prediction[0] if isinstance(prediction, tuple) else prediction
        return list(raw_residual)

    def _open_residual_telemetry(self):
        if self._residual_telemetry_file is not None:
            self._residual_telemetry_file.close()

        self.residual_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self._residual_telemetry_file = self.residual_telemetry_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._residual_telemetry_writer = csv.writer(self._residual_telemetry_file)
        self._residual_telemetry_writer.writerow(RESIDUAL_TELEMETRY_COLUMNS)
        self._residual_telemetry_file.flush()

    def _write_residual_telemetry(self, telemetry, action):
        if self._residual_telemetry_writer is None:
            return

        track_sensors = get_track_sensors(telemetry)
        line = get_racing_line_snapshot(telemetry, self.base_agent.racing_line)
        self._residual_telemetry_writer.writerow(
            [
                self.step,
                telemetry.get("distFromStart", 0),
                telemetry.get("speedX", 0),
                telemetry.get("speedY", 0),
                telemetry.get("angle", 0),
                telemetry.get("trackPos", 0),
                track_sensors[9],
                min(track_sensors),
                line["target_track_pos"],
                line["line_error"],
                line["target_speed_kmh"],
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
                action["agent5_safety_shield_active"],
                action["agent5_risk_pressure"],
                action["agent5_unsafe_residual_pressure"],
                action["agent5_low_speed_recovery_active"],
                action["agent5_policy_loaded"],
            ]
        )
        self._residual_telemetry_file.flush()
