from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .racing_line import LOOKAHEAD_DISTANCES_M, RacingLineFeatureMap


MODEL_FAMILY = "agent7_n_step_td3_racer"
OBSERVATION_VERSION = "agent7_torcsrl_history_v3"
ACTION_VERSION = "agent7_signed_steer_longitudinal_v1"
REWARD_VERSION = "agent7_torcsrl_racing_line_velocity_v3"
SENSOR_ONLY_MODEL_FAMILY = "agent8_sensor_n_step_td3_racer"
SENSOR_ONLY_OBSERVATION_VERSION = "agent8_torcs_sensor_history_v1"
SENSOR_ONLY_REWARD_VERSION = "agent8_centerline_velocity_v1"
STEERING_RATE_REWARD_VERSION = (
    "agent7_torcsrl_racing_line_velocity_steering_rate_v4"
)
PACE_ACTION_VERSION = "agent7_bounded_steer_longitudinal_v5"
PACE_REWARD_VERSION = "agent7_progress_per_interaction_v5"
DEFAULT_STEERING_RATE_COST_COEFFICIENT = 0.0025
DEFAULT_PACE_STEERING_LIMIT = 0.3
PACE_FAILURE_PENALTY = -10.0
PACE_LAP_COMPLETION_BONUS = 1_000.0
MAX_ABSOLUTE_PROGRESS_PER_INTERACTION_M = 5.0

MAX_SPEED_KMH = 250.0
MAX_ACCELERATION_MPS2 = 50.0
MAX_RPM = 10_000.0
MAX_TRACK_SENSOR_M = 200.0
MAX_WHEEL_SPIN = 100.0

BASE_OBSERVATION_SIZE = 45
ACTION_SIZE = 2
HISTORY_OBSERVATIONS = 3
HISTORY_ACTIONS = 3
OBSERVATION_SIZE = (
    BASE_OBSERVATION_SIZE * HISTORY_OBSERVATIONS
    + ACTION_SIZE * HISTORY_ACTIONS
)

# The public CLI keeps its historical 0.025 default as a profile level. The
# individual standard deviations mirror torcsRL's feature-specific sensor noise.
REFERENCE_OBSERVATION_NOISE_LEVEL = 0.025
BASE_OBSERVATION_NOISE_STD = (
    *(0.0025 for _ in range(3)),  # speed
    *(0.0025 for _ in range(3)),  # locally derived acceleration
    0.0025,  # rpm
    0.00025,  # angle
    0.00025,  # track position
    *(0.00025 for _ in range(19)),  # track sensors
    *(0.0025 for _ in range(4)),  # wheel spin
    0.00025,  # racing-line difference
    *(0.0 for _ in LOOKAHEAD_DISTANCES_M),  # look-ahead curvature
)
if len(BASE_OBSERVATION_NOISE_STD) != BASE_OBSERVATION_SIZE:
    raise RuntimeError("Agent 7 observation-noise profile has an invalid size")

BASE_FEATURE_NAMES = (
    "speed_x",
    "speed_y",
    "speed_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "rpm",
    "angle",
    "track_pos",
    *(f"track_{index}" for index in range(19)),
    *(f"wheel_spin_{index}" for index in range(4)),
    "racing_line_difference",
    *(f"curvature_{int(distance)}m" for distance in LOOKAHEAD_DISTANCES_M),
)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def track_sensors(telemetry: Mapping[str, Any]) -> np.ndarray:
    raw = telemetry.get("track")
    try:
        values = list(raw)[:19] if raw is not None else []
    except TypeError:
        values = []
    values.extend([MAX_TRACK_SENSOR_M] * (19 - len(values)))
    return np.asarray(
        [finite_float(value, MAX_TRACK_SENSOR_M) for value in values[:19]],
        dtype=np.float32,
    )


def build_base_observation(
    telemetry: Mapping[str, Any],
    *,
    previous_telemetry: Mapping[str, Any] | None = None,
    racing_line: RacingLineFeatureMap | None = None,
    include_racing_line_features: bool = True,
) -> np.ndarray:
    speed = np.asarray(
        [
            finite_float(telemetry.get("speedX")) / MAX_SPEED_KMH,
            finite_float(telemetry.get("speedY")) / MAX_SPEED_KMH,
            finite_float(telemetry.get("speedZ")) / MAX_SPEED_KMH,
        ],
        dtype=np.float32,
    )
    acceleration = _acceleration_observation(telemetry, previous_telemetry)
    rpm = np.asarray(
        [(finite_float(telemetry.get("rpm")) / MAX_RPM) * 2.0 - 1.0],
        dtype=np.float32,
    )
    angle = np.asarray(
        [finite_float(telemetry.get("angle")) / math.pi],
        dtype=np.float32,
    )
    track_position = np.asarray(
        [finite_float(telemetry.get("trackPos"))],
        dtype=np.float32,
    )
    road = (track_sensors(telemetry) / MAX_TRACK_SENSOR_M) * 2.0 - 1.0

    raw_wheel_spin = telemetry.get("wheelSpinVel")
    try:
        wheel_values = list(raw_wheel_spin)[:4] if raw_wheel_spin is not None else []
    except TypeError:
        wheel_values = []
    wheel_values.extend([0.0] * (4 - len(wheel_values)))
    wheel_spin = (
        np.asarray(
            [finite_float(value) for value in wheel_values[:4]],
            dtype=np.float32,
        )
        / MAX_WHEEL_SPIN
    ) * 2.0 - 1.0

    if not include_racing_line_features:
        if racing_line is not None:
            raise ValueError(
                "sensor-only observations cannot accept a racing-line map"
            )
        racing_line_difference = np.zeros(1, dtype=np.float32)
        lookahead_curvature = np.zeros(
            LOOKAHEAD_DISTANCES_M.shape,
            dtype=np.float32,
        )
    elif racing_line is None:
        target_track_position = 0.0
        lookahead_curvature = np.zeros(
            LOOKAHEAD_DISTANCES_M.shape,
            dtype=np.float32,
        )
        racing_line_difference = np.asarray(
            [finite_float(telemetry.get("trackPos")) / 2.0],
            dtype=np.float32,
        )
    else:
        line_sample = racing_line.sample(
            finite_float(telemetry.get("distFromStart"))
        )
        target_track_position = line_sample.target_track_position
        lookahead_curvature = line_sample.lookahead_curvature
        racing_line_difference = np.asarray(
            [
                (
                    finite_float(telemetry.get("trackPos"))
                    - target_track_position
                )
                / 2.0
            ],
            dtype=np.float32,
        )

    observation = np.concatenate(
        (
            speed,
            acceleration,
            rpm,
            angle,
            track_position,
            road,
            wheel_spin,
            racing_line_difference,
            lookahead_curvature,
        )
    ).astype(np.float32)
    if observation.shape != (BASE_OBSERVATION_SIZE,):
        raise RuntimeError(
            "Agent 7 observation contract produced an invalid shape: "
            f"{observation.shape}"
        )
    return np.clip(observation, -1.0, 1.0)


def apply_observation_noise(
    observation: np.ndarray,
    *,
    rng: np.random.Generator,
    level: float = REFERENCE_OBSERVATION_NOISE_LEVEL,
) -> np.ndarray:
    """Apply torcsRL-style feature noise to one normalized observation."""
    value = np.asarray(observation, dtype=np.float32).reshape(-1)
    if value.shape != (BASE_OBSERVATION_SIZE,):
        raise ValueError(
            f"expected base observation shape {(BASE_OBSERVATION_SIZE,)}, "
            f"got {value.shape}"
        )
    if not math.isfinite(level) or level < 0.0:
        raise ValueError("observation noise level must be finite and non-negative")
    if level == 0.0:
        return value.copy()

    scale = float(level) / REFERENCE_OBSERVATION_NOISE_LEVEL
    standard_deviation = np.asarray(
        BASE_OBSERVATION_NOISE_STD,
        dtype=np.float32,
    ) * scale
    noise = rng.normal(0.0, standard_deviation).astype(np.float32)
    return np.clip(value + noise, -1.0, 1.0).astype(np.float32)


def _acceleration_observation(
    telemetry: Mapping[str, Any],
    previous_telemetry: Mapping[str, Any] | None,
) -> np.ndarray:
    raw_acceleration = [telemetry.get(f"accel{axis}") for axis in "XYZ"]
    if all(value is not None for value in raw_acceleration):
        values = [finite_float(value) for value in raw_acceleration]
    elif previous_telemetry is not None:
        delta_time = (
            finite_float(telemetry.get("curLapTime"))
            - finite_float(previous_telemetry.get("curLapTime"))
        )
        if delta_time > 1e-6:
            values = [
                (
                    finite_float(telemetry.get(f"speed{axis}"))
                    - finite_float(previous_telemetry.get(f"speed{axis}"))
                )
                / 3.6
                / delta_time
                for axis in "XYZ"
            ]
        else:
            values = [0.0, 0.0, 0.0]
    else:
        values = [0.0, 0.0, 0.0]
    return np.clip(
        np.asarray(values, dtype=np.float32) / MAX_ACCELERATION_MPS2,
        -1.0,
        1.0,
    )


class HistoryEncoder:
    """Stack recent telemetry observations and actions as in torcsRL."""

    def __init__(
        self,
        observation_horizon: int = HISTORY_OBSERVATIONS,
        action_horizon: int = HISTORY_ACTIONS,
    ) -> None:
        if observation_horizon < 1 or action_horizon < 0:
            raise ValueError("history horizons must be non-negative")
        self.observation_horizon = int(observation_horizon)
        self.action_horizon = int(action_horizon)
        self._observations: deque[np.ndarray] = deque(
            maxlen=self.observation_horizon
        )
        self._actions: deque[np.ndarray] = deque(maxlen=self.action_horizon)
        self.reset()

    @property
    def size(self) -> int:
        return (
            BASE_OBSERVATION_SIZE * self.observation_horizon
            + ACTION_SIZE * self.action_horizon
        )

    def reset(self, initial_observation: np.ndarray | None = None) -> np.ndarray:
        self._observations.clear()
        self._actions.clear()
        for _ in range(self.observation_horizon):
            self._observations.append(
                np.zeros(BASE_OBSERVATION_SIZE, dtype=np.float32)
            )
        for _ in range(self.action_horizon):
            self._actions.append(np.zeros(ACTION_SIZE, dtype=np.float32))
        if initial_observation is not None:
            self._observations[-1] = self._validate_observation(
                initial_observation
            )
        return self.encoded()

    def observe(self, observation: np.ndarray) -> np.ndarray:
        self._observations.append(self._validate_observation(observation))
        return self.encoded()

    def record_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32).reshape(-1)
        if value.shape != (ACTION_SIZE,):
            raise ValueError(f"expected action shape {(ACTION_SIZE,)}, got {value.shape}")
        if self.action_horizon:
            self._actions.append(np.clip(value, -1.0, 1.0))
        return self.encoded()

    def encoded(self) -> np.ndarray:
        parts = [*self._observations, *self._actions]
        encoded = np.concatenate(parts).astype(np.float32)
        if encoded.shape != (self.size,):
            raise RuntimeError(f"invalid history observation shape: {encoded.shape}")
        return encoded

    @staticmethod
    def _validate_observation(observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(-1)
        if value.shape != (BASE_OBSERVATION_SIZE,):
            raise ValueError(
                f"expected base observation shape {(BASE_OBSERVATION_SIZE,)}, "
                f"got {value.shape}"
            )
        return np.clip(value, -1.0, 1.0)


def automatic_gear(gear: Any, rpm: Any) -> int:
    current = int(np.clip(round(finite_float(gear, 1.0)), 1, 6))
    engine_rpm = finite_float(rpm)
    if engine_rpm > 8_000.0 and current < 6:
        return current + 1
    if engine_rpm < 3_500.0 and current > 1:
        return current - 1
    return current


def decode_action(
    raw_action: np.ndarray,
    telemetry: Mapping[str, Any],
    *,
    physical_steering_limit: float = 1.0,
) -> dict[str, float | int]:
    steering_limit = float(physical_steering_limit)
    if not math.isfinite(steering_limit) or not 0.0 < steering_limit <= 1.0:
        raise ValueError("physical steering limit must be finite and in (0, 1]")
    values = np.zeros(ACTION_SIZE, dtype=np.float32)
    supplied = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    values[: min(ACTION_SIZE, supplied.size)] = supplied[:ACTION_SIZE]
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    values = np.clip(values, -1.0, 1.0)
    longitudinal = float(values[1])
    return {
        "steer": float(values[0]) * steering_limit,
        "accel": max(0.0, longitudinal),
        "brake": max(0.0, -longitudinal),
        "gear": automatic_gear(
            telemetry.get("gear", 1),
            telemetry.get("rpm", 0.0),
        ),
    }


@dataclass(frozen=True)
class RewardResult:
    total: float
    forward_velocity: float
    racing_line_factor: float
    steering_rate_penalty: float
    terminal_penalty: float
    progress_m: float = 0.0
    lap_completion_bonus: float = 0.0


def calculate_reward(
    telemetry: Mapping[str, Any],
    *,
    physical_failure: bool,
    target_track_position: float = 0.0,
    previous_steer: float = 0.0,
    current_steer: float = 0.0,
    steering_rate_cost_coefficient: float = 0.0,
) -> RewardResult:
    coefficient = float(steering_rate_cost_coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(
            "steering-rate cost coefficient must be finite and non-negative"
        )
    if physical_failure:
        return RewardResult(-10.0, 0.0, 0.0, 0.0, -10.0)

    speed = finite_float(telemetry.get("speedX")) / MAX_SPEED_KMH
    angle = finite_float(telemetry.get("angle"))
    racing_line_difference = abs(
        finite_float(telemetry.get("trackPos"))
        - finite_float(target_track_position)
    )
    racing_line_factor = (
        math.cos(angle) - abs(math.sin(angle)) - racing_line_difference
    )
    forward_velocity = speed
    steering_delta = finite_float(current_steer) - finite_float(previous_steer)
    steering_rate_penalty = -coefficient * steering_delta * steering_delta
    reward = forward_velocity * racing_line_factor + steering_rate_penalty
    return RewardResult(
        total=float(reward),
        forward_velocity=float(forward_velocity),
        racing_line_factor=float(racing_line_factor),
        steering_rate_penalty=float(steering_rate_penalty),
        terminal_penalty=0.0,
    )


def calculate_progress_reward(
    previous_distance_raced_m: Any,
    current_distance_raced_m: Any,
    *,
    physical_failure: bool,
    lap_completed: bool,
) -> RewardResult:
    """Return the minimal v5 pace objective in metres per interaction."""
    progress_m = float(
        np.clip(
            finite_float(current_distance_raced_m)
            - finite_float(previous_distance_raced_m),
            -MAX_ABSOLUTE_PROGRESS_PER_INTERACTION_M,
            MAX_ABSOLUTE_PROGRESS_PER_INTERACTION_M,
        )
    )
    terminal_penalty = PACE_FAILURE_PENALTY if physical_failure else 0.0
    lap_bonus = PACE_LAP_COMPLETION_BONUS if lap_completed else 0.0
    return RewardResult(
        total=progress_m + terminal_penalty + lap_bonus,
        forward_velocity=0.0,
        racing_line_factor=0.0,
        steering_rate_penalty=0.0,
        terminal_penalty=terminal_penalty,
        progress_m=progress_m,
        lap_completion_bonus=lap_bonus,
    )
