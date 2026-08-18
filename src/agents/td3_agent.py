from __future__ import annotations

import json
import math
import secrets
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
G_TRACK_3_LENGTH_METRES = 2843.0934

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "agent6_td3_scratch_racer.zip"
DEFAULT_BEST_DISTANCE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent6_td3_scratch_racer_best_distance.zip"
)
DEFAULT_BEST_REWARD_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent6_td3_scratch_racer_best_reward.zip"
)
DEFAULT_BEST_EVALUATION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent6_td3_scratch_racer_best_evaluation.zip"
)

AGENT6_MODEL_FAMILY = "agent6_td3_scratch_racer"
AGENT6_OBSERVATION_VERSION = "agent6_td3_raw_telemetry_v1"
AGENT6_ACTION_VERSION = "agent6_direct_steer_throttle_brake_v1"

GEAR_SPEEDS = [0.0, 45.0, 85.0, 125.0, 165.0, 205.0]
MAX_STEER = 0.90
PEDAL_DEADZONE = 0.04
ACCEL_RAMP_UP_PER_STEP = 0.16
ACCEL_RELEASE_PER_STEP = 0.55
BRAKE_RAMP_UP_PER_STEP = 0.40
BRAKE_RELEASE_PER_STEP = 0.50
STUCK_SPEED_KMH = 5.0
STUCK_STEP_LIMIT = 240

FEATURE_NAMES = [
    "speed_x",
    "speed_y",
    "speed_z",
    "angle",
    "track_pos",
    "rpm",
    "gear",
    "front_sensor",
    "front_left_sensor",
    "front_right_sensor",
    "near_left_sensor",
    "near_right_sensor",
    "far_left_sensor",
    "far_right_sensor",
    "min_track_sensor",
    "sensor_balance",
    "best_sensor_direction",
    "wheel_spin_balance",
    "damage",
    "lap_time",
    "distance_phase",
    "distance_phase_sin",
    "distance_phase_cos",
    "previous_steer",
    "previous_accel",
    "previous_brake",
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(number):
        return number
    return default


def finite_clamp(value: Any, lower: float, upper: float, default: float = 0.0) -> float:
    return clamp(finite_float(value, default), lower, upper)


def normalise_signed(value: Any, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return finite_clamp(finite_float(value) / scale, -1.0, 1.0)


def normalise_sensor(value: Any) -> float:
    return finite_clamp(finite_float(value) / 200.0, -1.0, 1.0)


def track_sensors(telemetry: Mapping[str, Any]) -> list[float]:
    track = telemetry.get("track")
    if track is not None:
        try:
            values = [finite_float(value, 200.0) for value in list(track)[:19]]
        except TypeError:
            values = []
        if len(values) >= 19:
            return values

    return [
        finite_float(telemetry.get(f"track_{index}", 200.0), 200.0)
        for index in range(19)
    ]


def shift_gears(speed_kmh: float) -> int:
    gear = 1
    for index, threshold in enumerate(GEAR_SPEEDS):
        if speed_kmh > threshold:
            gear = index + 1
    return int(clamp(float(gear), 1.0, 6.0))


def build_td3_observation(
    telemetry: Mapping[str, Any],
    previous_action: Mapping[str, Any] | None = None,
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> list[float]:
    sensors = track_sensors(telemetry)
    front_sensor = sensors[9]
    min_sensor = min(sensors) if sensors else front_sensor
    left_opening = sum(sensors[5:9]) / 4.0
    right_opening = sum(sensors[10:14]) / 4.0
    best_index = max(range(len(sensors)), key=lambda index: sensors[index])
    wheel_spin = telemetry.get("wheelSpinVel") or [0.0, 0.0, 0.0, 0.0]
    wheel_values = [finite_float(value) for value in list(wheel_spin)[:4]]
    while len(wheel_values) < 4:
        wheel_values.append(0.0)
    front_spin = (wheel_values[0] + wheel_values[1]) / 2.0
    rear_spin = (wheel_values[2] + wheel_values[3]) / 2.0
    distance = finite_float(telemetry.get("distFromStart", 0.0))
    if track_length_m > 0.0:
        distance_phase = (distance % track_length_m) / track_length_m
    else:
        distance_phase = 0.0
    distance_angle = 2.0 * math.pi * distance_phase
    previous_action = previous_action or {}

    return [
        normalise_signed(telemetry.get("speedX", 0.0), 240.0),
        normalise_signed(telemetry.get("speedY", 0.0), 45.0),
        normalise_signed(telemetry.get("speedZ", 0.0), 20.0),
        normalise_signed(telemetry.get("angle", 0.0), 1.0),
        normalise_signed(telemetry.get("trackPos", 0.0), 1.4),
        normalise_signed(telemetry.get("rpm", 0.0), 10000.0),
        finite_clamp(finite_float(telemetry.get("gear", 1.0)) / 6.0, 0.0, 1.0),
        normalise_sensor(front_sensor),
        normalise_sensor(sensors[7]),
        normalise_sensor(sensors[11]),
        normalise_sensor(sensors[5]),
        normalise_sensor(sensors[13]),
        normalise_sensor(sensors[0]),
        normalise_sensor(sensors[18]),
        normalise_sensor(min_sensor),
        normalise_signed(left_opening - right_opening, 200.0),
        normalise_signed(best_index - 9, 9.0),
        normalise_signed(rear_spin - front_spin, 80.0),
        normalise_signed(telemetry.get("damage", 0.0), 10000.0),
        normalise_signed(telemetry.get("curLapTime", 0.0), 180.0),
        finite_clamp(distance_phase, 0.0, 1.0),
        math.sin(distance_angle),
        math.cos(distance_angle),
        finite_clamp(previous_action.get("steer", 0.0), -1.0, 1.0),
        finite_clamp(previous_action.get("accel", 0.0), 0.0, 1.0),
        finite_clamp(previous_action.get("brake", 0.0), 0.0, 1.0),
    ]


def decode_td3_action(raw_action: Any) -> dict[str, float]:
    try:
        values = np.asarray(raw_action, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=float)

    padded = np.zeros(3, dtype=float)
    padded[: min(3, len(values))] = values[:3]
    steer = finite_clamp(padded[0], -1.0, 1.0) * MAX_STEER
    accel = max(0.0, finite_clamp(padded[1], -1.0, 1.0))
    brake = max(0.0, finite_clamp(padded[2], -1.0, 1.0))

    if accel < PEDAL_DEADZONE:
        accel = 0.0
    if brake < PEDAL_DEADZONE:
        brake = 0.0
    if brake > 0.0 and brake >= accel:
        accel = 0.0
    elif accel > 0.0:
        brake = 0.0

    return {
        "steer": clamp(steer, -MAX_STEER, MAX_STEER),
        "accel": clamp(accel, 0.0, 1.0),
        "brake": clamp(brake, 0.0, 1.0),
    }


def steering_delta_limit(speed_kmh: Any) -> float:
    speed = abs(finite_float(speed_kmh))
    if speed >= 150.0:
        return 0.045
    if speed >= 105.0:
        return 0.065
    if speed >= 70.0:
        return 0.090
    return 0.140


def _rate_limit_axis(
    target: float,
    previous: float,
    *,
    max_decrease: float,
    max_increase: float,
) -> float:
    delta = clamp(target - previous, -max_decrease, max_increase)
    return previous + delta


def rate_limit_td3_action(
    action: Mapping[str, Any],
    previous_action: Mapping[str, Any] | None = None,
    *,
    speed_kmh: Any = 0.0,
) -> dict[str, float]:
    previous_action = previous_action or {}
    previous_steer = finite_clamp(previous_action.get("steer", 0.0), -MAX_STEER, MAX_STEER)
    previous_accel = finite_clamp(previous_action.get("accel", 0.0), 0.0, 1.0)
    previous_brake = finite_clamp(previous_action.get("brake", 0.0), 0.0, 1.0)

    target_steer = finite_clamp(action.get("steer", 0.0), -MAX_STEER, MAX_STEER)
    target_accel = finite_clamp(action.get("accel", 0.0), 0.0, 1.0)
    target_brake = finite_clamp(action.get("brake", 0.0), 0.0, 1.0)
    target_braking = target_brake > 0.0

    steer_limit = steering_delta_limit(speed_kmh)
    steer = _rate_limit_axis(
        target_steer,
        previous_steer,
        max_decrease=steer_limit,
        max_increase=steer_limit,
    )
    accel = _rate_limit_axis(
        target_accel,
        previous_accel,
        max_decrease=ACCEL_RELEASE_PER_STEP,
        max_increase=ACCEL_RAMP_UP_PER_STEP,
    )
    brake = _rate_limit_axis(
        target_brake,
        previous_brake,
        max_decrease=BRAKE_RELEASE_PER_STEP,
        max_increase=BRAKE_RAMP_UP_PER_STEP,
    )

    if accel < PEDAL_DEADZONE:
        accel = 0.0
    if brake < PEDAL_DEADZONE:
        brake = 0.0
    if target_braking and brake > 0.0:
        accel = 0.0
    elif accel > 0.0:
        brake = 0.0

    return {
        "steer": clamp(steer, -MAX_STEER, MAX_STEER),
        "accel": clamp(accel, 0.0, 1.0),
        "brake": clamp(brake, 0.0, 1.0),
    }


def metadata_path_for_policy(policy_path: Path) -> Path:
    return Path(policy_path).with_suffix(".metadata.json")


def read_policy_metadata(policy_path: Path | None) -> dict[str, Any]:
    if policy_path is None:
        return {}
    metadata_path = metadata_path_for_policy(Path(policy_path))
    if not metadata_path.is_file():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def episode_metadata_is_policy_controlled(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    summary = value.get("episode_summary")
    return isinstance(summary, Mapping) and bool(summary.get("policy_controlled"))


def episode_metadata_is_deterministic_probe(value: Any) -> bool:
    if not episode_metadata_is_policy_controlled(value):
        return False
    summary = value.get("episode_summary")
    return isinstance(summary, Mapping) and bool(summary.get("deterministic_probe"))


def policy_checkpoint_is_verified(policy_path: Path, metadata_key: str) -> bool:
    if not policy_path.is_file():
        return False
    metadata = read_policy_metadata(policy_path)
    return episode_metadata_is_deterministic_probe(metadata.get(metadata_key))


def evaluation_checkpoint_is_verified(policy_path: Path) -> bool:
    if not policy_path.is_file():
        return False
    evaluation = read_policy_metadata(policy_path).get("best_evaluation")
    return (
        isinstance(evaluation, Mapping)
        and evaluation.get("deterministic") is True
        and finite_float(evaluation.get("repeats")) >= 1.0
    )


def policy_contract_mismatches(metadata: Mapping[str, Any]) -> list[str]:
    if not metadata:
        return []
    expected = {
        "model_family": AGENT6_MODEL_FAMILY,
        "observation_version": AGENT6_OBSERVATION_VERSION,
        "action_version": AGENT6_ACTION_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
    }
    return [key for key, value in expected.items() if metadata.get(key) != value]


class Td3ScratchAgent:
    name = "TD3 Scratch Racer"
    agent_type = "td3_scratch"
    version = "0.1"
    uses_full_control = True
    requires_racing_line = False
    track_length_m = G_TRACK_3_LENGTH_METRES
    max_steps = 150000
    target_laps = 1

    def __init__(
        self,
        *,
        seed: int | None = None,
        policy_path: str | Path | None = None,
        policy: Any = None,
        require_policy: bool = False,
        deterministic: bool = True,
    ) -> None:
        self.seed = seed if seed is not None else secrets.randbits(32)
        self.policy_path = Path(policy_path or _default_policy_path())
        self.policy = policy
        self.policy_loaded = policy is not None
        self.deterministic = bool(deterministic)
        self.policy_metadata = read_policy_metadata(self.policy_path)
        mismatches = policy_contract_mismatches(self.policy_metadata)
        if mismatches:
            formatted = ", ".join(mismatches)
            raise ValueError(
                "TD3 Scratch policy metadata is incompatible with this runtime: "
                f"{formatted}"
            )
        if self.policy is None:
            self.policy = self._load_policy(require_policy=require_policy)
            self.policy_loaded = self.policy is not None
        self.reset()

    @property
    def config(self) -> dict[str, Any]:
        try:
            policy_path = str(self.policy_path.relative_to(PROJECT_ROOT))
        except ValueError:
            policy_path = str(self.policy_path)
        return {
            "algorithm": "TD3",
            "model_family": AGENT6_MODEL_FAMILY,
            "observation_version": AGENT6_OBSERVATION_VERSION,
            "action_version": AGENT6_ACTION_VERSION,
            "feature_names": FEATURE_NAMES,
            "action_shape": [3],
            "policy_path": policy_path,
            "policy_loaded": self.policy_loaded,
            "learning_source": "reward_only",
            "teacher": None,
            "behaviour_cloning": False,
            "racing_line_imitation": False,
            "automatic_gear_shift": True,
            "track_length_m": self.track_length_m,
            "actuator_rate_limited": True,
            "deterministic": self.deterministic,
        }

    def reset(self) -> None:
        self.previous_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
        self.previous_gear = 1
        self.stuck_counter = 0
        self.last_raw_action = [0.0, 0.0, 0.0]

    def act(
        self,
        _observation: Any,
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if telemetry is None:
            raise ValueError("Td3ScratchAgent requires raw TORCS telemetry")

        shutdown, reason = self.should_shutdown(telemetry)
        if shutdown:
            return {
                "steer": self.previous_action["steer"],
                "accel": 0.0,
                "brake": 1.0,
                "gear": self.previous_gear,
                "terminate": True,
                "termination_reason": reason,
            }

        features = build_td3_observation(
            telemetry,
            self.previous_action,
            track_length_m=self.track_length_m,
        )
        raw_action = self._predict_action(features)
        action = decode_td3_action(raw_action)
        speed = finite_float(telemetry.get("speedX", 0.0))
        action = rate_limit_td3_action(
            action,
            self.previous_action,
            speed_kmh=speed,
        )
        action.update(
            {
                "gear": shift_gears(speed),
                "terminate": False,
                "termination_reason": "",
            }
        )
        self.previous_action = {
            "steer": action["steer"],
            "accel": action["accel"],
            "brake": action["brake"],
        }
        self.previous_gear = action["gear"]
        self.last_raw_action = list(raw_action)
        return action

    def should_shutdown(self, telemetry: Mapping[str, Any]) -> tuple[bool, str]:
        speed = finite_float(telemetry.get("speedX", 0.0))
        track_position = abs(finite_float(telemetry.get("trackPos", 0.0)))
        angle = abs(finite_float(telemetry.get("angle", 0.0)))
        damage = finite_float(telemetry.get("damage", 0.0))
        sensors = track_sensors(telemetry)
        min_sensor = min(sensors) if sensors else 200.0

        self.stuck_counter = self.stuck_counter + 1 if speed < STUCK_SPEED_KMH else 0
        if damage > 650.0:
            return True, "damage limit exceeded"
        if track_position > 1.35:
            return True, "out of bounds"
        if track_position > 1.22 and min_sensor < -0.1 and speed > 28.0:
            return True, "out of bounds"
        if angle > 2.25:
            return True, "wrong direction"
        if self.stuck_counter > STUCK_STEP_LIMIT:
            return True, "car stuck"
        return False, ""

    def telemetry_debug(self) -> dict[str, Any]:
        return {
            "td3_policy_loaded": self.policy_loaded,
            "td3_raw_steer": self.last_raw_action[0],
            "td3_raw_accel": self.last_raw_action[1],
            "td3_raw_brake": self.last_raw_action[2],
        }

    def close(self) -> None:
        return None

    def _load_policy(self, *, require_policy: bool) -> Any:
        if not self.policy_path.is_file():
            if require_policy:
                raise FileNotFoundError(
                    "TD3 Scratch model not found. Train Agent 6 first: "
                    f"{self.policy_path}"
                )
            return None

        try:
            from stable_baselines3 import TD3
        except ImportError as exc:
            if require_policy:
                raise ImportError(
                    "stable-baselines3 is required to load the TD3 Scratch model"
                ) from exc
            return None

        return TD3.load(str(self.policy_path))

    def _predict_action(self, features: list[float]) -> list[float]:
        if self.policy is None:
            return [0.0, 0.0, 0.0]

        policy_observation = np.asarray(features, dtype=np.float32)
        prediction = self.policy.predict(
            policy_observation,
            deterministic=self.deterministic,
        )
        raw_action = prediction[0] if isinstance(prediction, tuple) else prediction
        return list(np.asarray(raw_action, dtype=float).reshape(-1)[:3])


def _default_policy_path() -> Path:
    if evaluation_checkpoint_is_verified(DEFAULT_BEST_EVALUATION_MODEL_PATH):
        return DEFAULT_BEST_EVALUATION_MODEL_PATH
    if policy_checkpoint_is_verified(
        DEFAULT_BEST_DISTANCE_MODEL_PATH,
        "best_distance_episode",
    ):
        return DEFAULT_BEST_DISTANCE_MODEL_PATH
    if policy_checkpoint_is_verified(
        DEFAULT_BEST_REWARD_MODEL_PATH,
        "best_reward_episode",
    ):
        return DEFAULT_BEST_REWARD_MODEL_PATH
    return DEFAULT_MODEL_PATH
