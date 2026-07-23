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


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "agent4_hybrid_ppo.zip"
DEFAULT_BEST_MODEL_PATH = PROJECT_ROOT / "models" / "agent4_hybrid_ppo_best.zip"
DEFAULT_BASE_TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "generated" / "hybrid_ppo_base_telemetry.csv"
)
DEFAULT_RESIDUAL_TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "generated" / "hybrid_ppo_residual_telemetry.csv"
)

AGENT4_MODEL_FAMILY = "agent4_hybrid_ppo_residual"
AGENT4_OBSERVATION_VERSION = "agent4_observation_v2_track_phase_teacher"
AGENT4_ACTION_VERSION = "agent4_residual_action_v1"

MAX_STEER_RESIDUAL = 0.080
MAX_ACCEL_RESIDUAL = 0.180
MAX_BRAKE_RESIDUAL = 0.220
DEFAULT_RESIDUAL_SCALE = 0.1
SAFE_TRACK_LIMIT = 0.84
HARD_TRACK_LIMIT = 0.96
FINITE_DEFAULT = 0.0

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
    "baseSteer",
    "baseAccel",
    "baseBrake",
    "rawSteerResidual",
    "rawAccelResidual",
    "rawBrakeResidual",
    "steerResidual",
    "accelResidual",
    "brakeResidual",
    "finalSteer",
    "finalAccel",
    "finalBrake",
    "safetyShieldActive",
    "policyLoaded",
]


def normalise_sensor(value):
    return finite_clamp(finite_float(value) / 200.0, -1.0, 1.0)


def normalise_signed(value, scale):
    return finite_clamp(finite_float(value) / scale, -1.0, 1.0)


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


def build_hybrid_ppo_observation(telemetry, base_action, track_length=None):
    """Build the compact observation Agent 4 gives to its residual PPO policy."""

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
        distance_phase,
        math.sin(distance_phase_angle),
        math.cos(distance_phase_angle),
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
    """Convert a PPO action in [-1, 1] into bounded control deltas."""

    values = coerce_policy_residual(raw_residual)
    limits = get_residual_limits(residual_scale)
    return [
        finite_clamp(values[0], -1.0, 1.0) * limits[0],
        finite_clamp(values[1], -1.0, 1.0) * limits[1],
        finite_clamp(values[2], -1.0, 1.0) * limits[2],
    ]


def safety_shield_is_active(telemetry):
    track_sensors = get_track_sensors(telemetry)
    front_sensor = track_sensors[9]
    min_track_sensor = min(track_sensors) if track_sensors else front_sensor
    track_position = finite_float(telemetry.get("trackPos", 0.0))

    return (
        min_track_sensor < 0.0
        or front_sensor < 22.0
        or abs(track_position) > SAFE_TRACK_LIMIT
        or abs(finite_float(telemetry.get("angle", 0.0))) > 0.52
        or abs(finite_float(telemetry.get("speedY", 0.0))) > 13.0
    )


def shield_residual_for_safety(base_action, residual, telemetry):
    """Prevent PPO corrections from fighting Agent 3 in risky states."""

    steer_delta, accel_delta, brake_delta = residual
    if not safety_shield_is_active(telemetry):
        return residual, False

    track_position = finite_float(telemetry.get("trackPos", 0.0))

    if abs(track_position) > SAFE_TRACK_LIMIT and steer_delta * track_position > 0.0:
        steer_delta = 0.0
    else:
        steer_delta *= 0.35

    accel_delta = min(accel_delta, 0.0)
    if finite_float(base_action.get("brake", 0.0)) > 0.0:
        brake_delta = max(brake_delta, 0.0)

    return [steer_delta, accel_delta, brake_delta], True


def apply_hybrid_residual(base_action, raw_residual, telemetry, residual_scale=1.0):
    raw_values = coerce_policy_residual(raw_residual)
    residual = decode_policy_residual(raw_residual, residual_scale)
    residual, shield_active = shield_residual_for_safety(
        base_action,
        residual,
        telemetry,
    )

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

    action["agent4_base_steer"] = finite_float(base_action.get("steer", 0.0))
    action["agent4_base_accel"] = finite_float(base_action.get("accel", 0.0))
    action["agent4_base_brake"] = finite_float(base_action.get("brake", 0.0))
    action["agent4_raw_steer_residual"] = raw_values[0]
    action["agent4_raw_accel_residual"] = raw_values[1]
    action["agent4_raw_brake_residual"] = raw_values[2]
    action["agent4_steer_residual"] = residual[0]
    action["agent4_accel_residual"] = residual[1]
    action["agent4_brake_residual"] = residual[2]
    action["agent4_safety_shield_active"] = shield_active

    return action


class HybridPpoAgent:
    name = "Hybrid PPO Agent"
    agent_type = "hybrid_ppo"
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
        self._residual_telemetry_file = None
        self._residual_telemetry_writer = None
        self.step = 0

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
            "model_family": AGENT4_MODEL_FAMILY,
            "observation_version": AGENT4_OBSERVATION_VERSION,
            "action_version": AGENT4_ACTION_VERSION,
            "base_agent": self.base_agent.name,
            "base_agent_version": self.base_agent.version,
            "base_agent_type": self.base_agent.agent_type,
            "policy_path": policy_path,
            "policy_loaded": self.policy_loaded,
            "policy_metadata_loaded": bool(self.policy_metadata),
            "residual_scale": self.residual_scale,
            "residual_limits": {
                "steer": MAX_STEER_RESIDUAL * self.residual_scale,
                "accel": MAX_ACCEL_RESIDUAL * self.residual_scale,
                "brake": MAX_BRAKE_RESIDUAL * self.residual_scale,
            },
            "observation_features": FEATURE_NAMES,
            "safety_shield": {
                "safe_track_limit": SAFE_TRACK_LIMIT,
                "hard_track_limit": HARD_TRACK_LIMIT,
                "front_sensor_limit": 22.0,
            },
        }

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
            raise ValueError("HybridPpoAgent requires raw TORCS telemetry")

        base_action = self.base_agent.act(observation, telemetry)
        if base_action.get("terminate", False):
            return base_action

        features = build_hybrid_ppo_observation(
            telemetry,
            base_action,
            self.base_agent.racing_line.track_length,
        )
        raw_residual = self._predict_residual(features)
        action = apply_hybrid_residual(
            base_action,
            raw_residual,
            telemetry,
            self.residual_scale,
        )
        action["agent4_policy_loaded"] = self.policy_loaded

        self._write_residual_telemetry(telemetry, action)
        self.step += 1
        return action

    def _load_policy(self, require_policy=False):
        if self.policy_path is None or not self.policy_path.is_file():
            if require_policy:
                raise FileNotFoundError(
                    "Hybrid PPO model not found. Train Agent 4 first or run "
                    f"without require_policy: {self.policy_path}"
                )
            return None

        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            if require_policy:
                raise ImportError(
                    "stable-baselines3 is required to load the Hybrid PPO model"
                ) from exc
            return None

        return PPO.load(str(self.policy_path))

    def _predict_residual(self, features):
        if self.policy is None:
            return [0.0, 0.0, 0.0]

        policy_observation = np.asarray(features, dtype=np.float32)
        prediction = self.policy.predict(policy_observation, deterministic=True)
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
                action["agent4_base_steer"],
                action["agent4_base_accel"],
                action["agent4_base_brake"],
                action["agent4_raw_steer_residual"],
                action["agent4_raw_accel_residual"],
                action["agent4_raw_brake_residual"],
                action["agent4_steer_residual"],
                action["agent4_accel_residual"],
                action["agent4_brake_residual"],
                action["steer"],
                action["accel"],
                action["brake"],
                action["agent4_safety_shield_active"],
                action["agent4_policy_loaded"],
            ]
        )
        self._residual_telemetry_file.flush()
