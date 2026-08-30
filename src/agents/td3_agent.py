from __future__ import annotations

import json
import math
import random
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
DEFAULT_BEST_ROBUST_EVALUATION_MODEL_PATH = (
    DEFAULT_BEST_EVALUATION_MODEL_PATH
)
DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent6_td3_scratch_racer_best_completed_lap.zip"
)
DEFAULT_FRONTIER_MODEL_PATH = (
    PROJECT_ROOT / "models" / "agent6_td3_scratch_racer_frontier.zip"
)

AGENT6_MODEL_FAMILY = "agent6_td3_scratch_racer"
AGENT6_OBSERVATION_VERSION = "agent6_td3_raw_telemetry_v2"
AGENT6_ACTION_VERSION = "agent6_signed_steer_longitudinal_v2"
AGENT6_ACTION_SHAPE = [2]
AGENT6_ACTION_SIZE = 2
AGENT6_LEGACY_OBSERVATION_VERSION = "agent6_td3_raw_telemetry_v1"
AGENT6_LEGACY_ACTION_VERSION = "agent6_direct_steer_throttle_brake_v1"
AGENT6_LEGACY_ACTION_SHAPE = [3]
AGENT6_LEGACY_ACTION_SIZE = 3
AGENT6_ROBUSTNESS_PROTOCOL = "agent6_order_stratified_robustness_v5"
AGENT6_ROBUSTNESS_REPEATS = 5
AGENT6_ROBUSTNESS_TRIALS_PER_SEED = 4
AGENT6_ROBUSTNESS_MAX_PAIRED_REGRESSION_M = 100.0
AGENT6_TRAINING_CHALLENGE_BASE_SEED = 20260820
AGENT6_EXTERNAL_EVALUATION_BASE_SEED = 20260821
AGENT6_ROBUSTNESS_STEERING_NOISE_STD = 0.006

GEAR_SPEEDS = [0.0, 45.0, 85.0, 125.0, 165.0, 205.0]
MAX_STEER = 0.90
PEDAL_DEADZONE = 0.04
ACCEL_RAMP_UP_PER_STEP = 0.16
ACCEL_RELEASE_PER_STEP = 0.55
BRAKE_RAMP_UP_PER_STEP = 0.40
BRAKE_RELEASE_PER_STEP = 0.50
STUCK_SPEED_KMH = 5.0
STUCK_STEP_LIMIT = 240
MAX_REPRODUCIBLE_SEED = (2**32) - 1
TRAINING_CHALLENGE_SEED_MIN = 0
TRAINING_CHALLENGE_SEED_MAX = (2**31) - 1
EXTERNAL_EVALUATION_SEED_MIN = 2**31
EXTERNAL_EVALUATION_SEED_MAX = MAX_REPRODUCIBLE_SEED
STEERING_ACTION_NOISE_CORRELATION = 0.985

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

    padded = np.zeros(AGENT6_ACTION_SIZE, dtype=float)
    padded[: min(AGENT6_ACTION_SIZE, len(values))] = values[
        :AGENT6_ACTION_SIZE
    ]
    steer = finite_clamp(padded[0], -1.0, 1.0) * MAX_STEER
    longitudinal = finite_clamp(padded[1], -1.0, 1.0)
    accel = max(0.0, longitudinal)
    brake = max(0.0, -longitudinal)

    if accel < PEDAL_DEADZONE:
        accel = 0.0
    if brake < PEDAL_DEADZONE:
        brake = 0.0

    return {
        "steer": clamp(steer, -MAX_STEER, MAX_STEER),
        "accel": clamp(accel, 0.0, 1.0),
        "brake": clamp(brake, 0.0, 1.0),
    }


def decode_legacy_td3_action(raw_action: Any) -> dict[str, float]:
    """Decode the original Agent 6 steer/throttle/brake action contract exactly."""
    try:
        values = np.asarray(raw_action, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=float)

    padded = np.zeros(AGENT6_LEGACY_ACTION_SIZE, dtype=float)
    padded[: min(AGENT6_LEGACY_ACTION_SIZE, len(values))] = values[
        :AGENT6_LEGACY_ACTION_SIZE
    ]
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


def build_reproducible_seed_suite(
    *,
    repeats: int,
    base_seed: int,
    seed_min: int = 0,
    seed_max: int = MAX_REPRODUCIBLE_SEED,
) -> tuple[int, ...]:
    repeat_count = int(repeats)
    seed = int(base_seed)
    lower = int(seed_min)
    upper = int(seed_max)
    if repeat_count < 1:
        raise ValueError("repeats must be at least 1")
    if lower < 0 or upper > MAX_REPRODUCIBLE_SEED or lower > upper:
        raise ValueError("seed range must be within the 32-bit unsigned range")
    if repeat_count > upper - lower + 1:
        raise ValueError("repeats exceeds the reproducible seed population")
    if not 0 <= seed <= MAX_REPRODUCIBLE_SEED:
        raise ValueError(
            f"base_seed must be between 0 and {MAX_REPRODUCIBLE_SEED}"
        )
    generator = random.Random(seed)
    return tuple(
        generator.sample(range(lower, upper + 1), repeat_count)
    )


def build_rotating_training_seed_suite(
    *,
    repeats: int,
    base_seed: int,
    candidate_id: int,
) -> tuple[int, ...]:
    """Return one deterministic, non-overlapping challenge suite per candidate."""
    repeat_count = int(repeats)
    block = int(candidate_id)
    generator_seed = int(base_seed)
    if repeat_count < 1:
        raise ValueError("repeats must be at least 1")
    if block < 1:
        raise ValueError("candidate_id must be at least 1")
    if not 0 <= generator_seed <= MAX_REPRODUCIBLE_SEED:
        raise ValueError(
            f"base_seed must be between 0 and {MAX_REPRODUCIBLE_SEED}"
        )
    required_count = repeat_count * block
    available_count = (
        TRAINING_CHALLENGE_SEED_MAX - TRAINING_CHALLENGE_SEED_MIN + 1
    )
    if required_count > available_count:
        raise ValueError(
            "requested challenge suites exceed the training seed domain"
        )

    generator = random.Random(generator_seed)
    generated: list[int] = []
    seen: set[int] = set()
    while len(generated) < required_count:
        seed = generator.randint(
            TRAINING_CHALLENGE_SEED_MIN,
            TRAINING_CHALLENGE_SEED_MAX,
        )
        if seed not in seen:
            seen.add(seed)
            generated.append(seed)
    start = repeat_count * (block - 1)
    return tuple(generated[start : start + repeat_count])


def counterbalanced_pair_order(
    *,
    seed_index: int,
    trial_index: int,
) -> tuple[str, str]:
    """Alternate pair order across seeds and trials to reduce temporal bias."""
    if int(seed_index) < 0 or int(trial_index) < 0:
        raise ValueError("seed_index and trial_index must be non-negative")
    accepted = "accepted_reference"
    candidate = "candidate"
    if (int(seed_index) + int(trial_index)) % 2 == 0:
        return accepted, candidate
    return candidate, accepted


def balanced_pair_order_positions(
    *,
    seed_count: int,
    trials_per_seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return seed-major positions for a fully counterbalanced policy pair."""
    seeds = int(seed_count)
    trials = int(trials_per_seed)
    if seeds < 1:
        raise ValueError("seed_count must be at least 1")
    if trials < 2 or trials % 2 != 0:
        raise ValueError("trials_per_seed must be an even value of at least 2")

    reference_positions: list[int] = []
    candidate_positions: list[int] = []
    for seed_index in range(seeds):
        for trial_index in range(trials):
            order = counterbalanced_pair_order(
                seed_index=seed_index,
                trial_index=trial_index,
            )
            reference_positions.append(order.index("accepted_reference") + 1)
            candidate_positions.append(order.index("candidate") + 1)
    return tuple(reference_positions), tuple(candidate_positions)


def order_stratified_seed_aggregates(
    values: list[float] | tuple[float, ...],
    order_positions: list[int] | tuple[int, ...],
    *,
    seed_count: int,
    trials_per_seed: int,
) -> tuple[float, ...]:
    """Aggregate each order position first, then weight positions equally."""
    position_medians = order_stratified_seed_position_medians(
        values,
        order_positions,
        seed_count=seed_count,
        trials_per_seed=trials_per_seed,
    )
    return tuple(
        float(np.mean(tuple(seed_positions.values())))
        for seed_positions in position_medians
    )


def order_stratified_seed_position_medians(
    values: list[float] | tuple[float, ...],
    order_positions: list[int] | tuple[int, ...],
    *,
    seed_count: int,
    trials_per_seed: int,
) -> tuple[dict[int, float], ...]:
    """Return each seed's median metric keyed by execution-order position."""
    seeds = int(seed_count)
    trials = int(trials_per_seed)
    if seeds < 1 or trials < 1:
        raise ValueError("seed_count and trials_per_seed must be at least 1")
    expected = seeds * trials
    if len(values) != expected or len(order_positions) != expected:
        raise ValueError(
            f"expected {expected} values and order positions, received "
            f"{len(values)} and {len(order_positions)}"
        )
    array = np.asarray(values, dtype=float)
    positions = np.asarray(order_positions, dtype=int)
    if not np.all(np.isfinite(array)):
        raise ValueError("order-stratified values must be finite")
    if np.any(positions < 1):
        raise ValueError("order positions must be positive integers")

    value_rows = array.reshape(seeds, trials)
    position_rows = positions.reshape(seeds, trials)
    aggregates: list[dict[int, float]] = []
    for seed_values, seed_positions in zip(value_rows, position_rows):
        unique_positions = sorted(set(int(value) for value in seed_positions))
        position_counts = [
            int(np.count_nonzero(seed_positions == position))
            for position in unique_positions
        ]
        if len(set(position_counts)) != 1:
            raise ValueError(
                "each order position must occur equally often within every seed"
            )
        position_medians = {
            position: float(
                np.median(seed_values[seed_positions == position])
            )
            for position in unique_positions
        }
        aggregates.append(position_medians)
    return tuple(aggregates)


def parameter_state_dicts_are_identical(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Return whether two neural-network state dictionaries match exactly."""
    if set(reference) != set(candidate):
        return False
    for name in sorted(reference):
        reference_value = reference[name]
        candidate_value = candidate[name]
        if hasattr(reference_value, "detach"):
            reference_value = reference_value.detach().cpu().numpy()
        if hasattr(candidate_value, "detach"):
            candidate_value = candidate_value.detach().cpu().numpy()
        reference_array = np.asarray(reference_value)
        candidate_array = np.asarray(candidate_value)
        if (
            reference_array.shape != candidate_array.shape
            or reference_array.dtype != candidate_array.dtype
            or not np.array_equal(reference_array, candidate_array)
        ):
            return False
    return True


def signed_longitudinal_from_legacy_outputs(raw_action: Any) -> float:
    """Map legacy pedal heads for an explicitly requested experimental migration."""
    try:
        values = np.asarray(raw_action, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=float)
    padded = np.zeros(AGENT6_LEGACY_ACTION_SIZE, dtype=float)
    padded[: min(AGENT6_LEGACY_ACTION_SIZE, len(values))] = values[
        :AGENT6_LEGACY_ACTION_SIZE
    ]
    accel = max(0.0, finite_clamp(padded[1], -1.0, 1.0))
    brake = max(0.0, finite_clamp(padded[2], -1.0, 1.0))
    if accel < PEDAL_DEADZONE:
        accel = 0.0
    if brake < PEDAL_DEADZONE:
        brake = 0.0
    return clamp(accel - brake, -1.0, 1.0)


def replicated_seed_medians(
    values: list[float] | tuple[float, ...],
    *,
    seed_count: int,
    trials_per_seed: int,
) -> tuple[float, ...]:
    """Collapse repeated seed-major trials with a median robust to one outlier."""
    seeds = int(seed_count)
    trials = int(trials_per_seed)
    if seeds < 1 or trials < 1:
        raise ValueError("seed_count and trials_per_seed must be at least 1")
    expected = seeds * trials
    if len(values) != expected:
        raise ValueError(
            f"expected {expected} replicated values, received {len(values)}"
        )
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("replicated values must be finite")
    reshaped = array.reshape(seeds, trials)
    return tuple(float(value) for value in np.median(reshaped, axis=1))


class SeededSteeringActionNoise:
    """Reproducible correlated steering noise in normalized TD3 action space."""

    def __init__(
        self,
        *,
        seed: int,
        steering_noise_std: float,
        correlation: float = STEERING_ACTION_NOISE_CORRELATION,
        action_size: int = AGENT6_ACTION_SIZE,
    ) -> None:
        if not 0 <= int(seed) <= MAX_REPRODUCIBLE_SEED:
            raise ValueError(
                f"seed must be between 0 and {MAX_REPRODUCIBLE_SEED}"
            )
        if not math.isfinite(steering_noise_std) or steering_noise_std < 0.0:
            raise ValueError("steering_noise_std must be finite and non-negative")
        if not math.isfinite(correlation) or not 0.0 <= correlation < 1.0:
            raise ValueError("correlation must be finite and in [0, 1)")
        if int(action_size) < 1:
            raise ValueError("action_size must be at least 1")
        self.seed = int(seed)
        self.steering_noise_std = float(steering_noise_std)
        self.correlation = float(correlation)
        self.action_size = int(action_size)
        self._random = random.Random(self.seed)
        self._steering_noise = 0.0

    @property
    def steering_noise(self) -> float:
        return self._steering_noise

    def __call__(self) -> np.ndarray:
        innovation_scale = self.steering_noise_std * math.sqrt(
            1.0 - (self.correlation**2)
        )
        self._steering_noise = (
            self.correlation * self._steering_noise
            + self._random.gauss(0.0, innovation_scale)
        )
        noise = np.zeros(self.action_size, dtype=np.float32)
        noise[0] = self._steering_noise / MAX_STEER
        return noise

    def reset(self) -> None:
        self._random.seed(self.seed)
        self._steering_noise = 0.0


class SeededGaussianActionNoise:
    """Gaussian exploration backed by an RNG isolated from replay sampling."""

    def __init__(
        self,
        *,
        seed: int,
        standard_deviation: float,
        action_size: int = AGENT6_ACTION_SIZE,
    ) -> None:
        if not 0 <= int(seed) <= MAX_REPRODUCIBLE_SEED:
            raise ValueError(
                f"seed must be between 0 and {MAX_REPRODUCIBLE_SEED}"
            )
        if (
            not math.isfinite(standard_deviation)
            or standard_deviation < 0.0
        ):
            raise ValueError(
                "standard_deviation must be finite and non-negative"
            )
        if int(action_size) < 1:
            raise ValueError("action_size must be at least 1")
        self.seed = int(seed)
        self.standard_deviation = float(standard_deviation)
        self.action_size = int(action_size)
        self._random = np.random.default_rng(self.seed)

    def __call__(self) -> np.ndarray:
        return self._random.normal(
            loc=0.0,
            scale=self.standard_deviation,
            size=self.action_size,
        ).astype(np.float32)

    def reset(self) -> None:
        # Keep one independent stream across episodes. Rewinding here would
        # repeat the same exploratory actions after every TORCS reset.
        return None


class SeededLegacyActionNoise:
    """Correlated exploration that respects the legacy competing-pedal contract."""

    def __init__(
        self,
        *,
        seed: int,
        standard_deviation: float,
        correlation: float = STEERING_ACTION_NOISE_CORRELATION,
    ) -> None:
        if not 0 <= int(seed) <= MAX_REPRODUCIBLE_SEED:
            raise ValueError(
                f"seed must be between 0 and {MAX_REPRODUCIBLE_SEED}"
            )
        if (
            not math.isfinite(standard_deviation)
            or standard_deviation < 0.0
        ):
            raise ValueError(
                "standard_deviation must be finite and non-negative"
            )
        if not math.isfinite(correlation) or not 0.0 <= correlation < 1.0:
            raise ValueError("correlation must be finite and in [0, 1)")
        self.seed = int(seed)
        self.standard_deviation = float(standard_deviation)
        self.correlation = float(correlation)
        self.action_size = AGENT6_LEGACY_ACTION_SIZE
        self._random = np.random.default_rng(self.seed)
        self._state = np.zeros(2, dtype=np.float64)

    def __call__(self) -> np.ndarray:
        innovation_scale = self.standard_deviation * math.sqrt(
            1.0 - (self.correlation**2)
        )
        innovation = self._random.normal(
            loc=0.0,
            scale=innovation_scale,
            size=2,
        )
        self._state = self.correlation * self._state + innovation
        steer_noise, longitudinal_noise = self._state
        return np.asarray(
            [
                steer_noise,
                0.5 * longitudinal_noise,
                -0.5 * longitudinal_noise,
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        return None


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


def completed_lap_checkpoint_is_verified(policy_path: Path) -> bool:
    if not policy_path.is_file():
        return False
    completed_lap = read_policy_metadata(policy_path).get(
        "best_completed_lap"
    )
    return bool(
        isinstance(completed_lap, Mapping)
        and finite_float(completed_lap.get("completed_laps")) >= 1.0
        and finite_float(completed_lap.get("best_lap_seconds")) > 0.0
    )


def policy_contract_mismatches(
    metadata: Mapping[str, Any],
    *,
    allow_legacy_action_contract: bool = False,
) -> list[str]:
    return _policy_contract_mismatches(
        metadata,
        allow_legacy_action_contract=allow_legacy_action_contract,
    )


def _policy_contract_mismatches(
    metadata: Mapping[str, Any],
    *,
    allow_legacy_action_contract: bool,
) -> list[str]:
    if not metadata:
        return []
    expected = {
        "model_family": AGENT6_MODEL_FAMILY,
        "observation_version": AGENT6_OBSERVATION_VERSION,
        "action_version": AGENT6_ACTION_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": AGENT6_ACTION_SHAPE,
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if (
        mismatches
        and allow_legacy_action_contract
        and policy_uses_legacy_action_contract(metadata)
    ):
        return []
    return mismatches


def policy_uses_legacy_action_contract(metadata: Mapping[str, Any]) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("model_family") == AGENT6_MODEL_FAMILY
        and metadata.get("observation_version") == AGENT6_LEGACY_OBSERVATION_VERSION
        and metadata.get("action_version") == AGENT6_LEGACY_ACTION_VERSION
        and metadata.get("feature_names") == FEATURE_NAMES
        and metadata.get("action_shape") == AGENT6_LEGACY_ACTION_SHAPE
    )


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
        action_noise: Any = None,
    ) -> None:
        self.seed = seed if seed is not None else secrets.randbits(32)
        self.policy_path = Path(policy_path or _default_policy_path())
        self.policy = policy
        self.policy_loaded = policy is not None
        self.deterministic = bool(deterministic)
        self.action_noise = action_noise
        self.policy_metadata = read_policy_metadata(self.policy_path)
        self.uses_legacy_action_contract = policy_uses_legacy_action_contract(
            self.policy_metadata
        )
        self.action_size = (
            AGENT6_LEGACY_ACTION_SIZE
            if self.uses_legacy_action_contract
            else AGENT6_ACTION_SIZE
        )
        mismatches = _policy_contract_mismatches(
            self.policy_metadata,
            allow_legacy_action_contract=True,
        )
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
        observation_version = (
            AGENT6_LEGACY_OBSERVATION_VERSION
            if self.uses_legacy_action_contract
            else AGENT6_OBSERVATION_VERSION
        )
        action_version = (
            AGENT6_LEGACY_ACTION_VERSION
            if self.uses_legacy_action_contract
            else AGENT6_ACTION_VERSION
        )
        action_shape = (
            AGENT6_LEGACY_ACTION_SHAPE
            if self.uses_legacy_action_contract
            else AGENT6_ACTION_SHAPE
        )
        return {
            "algorithm": "TD3",
            "model_family": AGENT6_MODEL_FAMILY,
            "observation_version": observation_version,
            "action_version": action_version,
            "feature_names": FEATURE_NAMES,
            "action_shape": action_shape,
            "legacy_action_adapter": False,
            "legacy_action_decoder": self.uses_legacy_action_contract,
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
        reset_action_noise = getattr(self.action_noise, "reset", None)
        if callable(reset_action_noise):
            reset_action_noise()
        self.previous_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
        self.previous_gear = 1
        self.stuck_counter = 0
        self.last_raw_action = [0.0] * self.action_size
        self.last_policy_action = [0.0] * self.action_size

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
        predicted_action = np.asarray(
            self._predict_action(features),
            dtype=np.float32,
        ).reshape(-1)
        policy_action = np.zeros(self.action_size, dtype=np.float32)
        policy_action[: min(self.action_size, len(predicted_action))] = (
            predicted_action[: self.action_size]
        )
        raw_action = policy_action.copy()
        if self.action_noise is not None:
            noise = np.asarray(self.action_noise(), dtype=np.float32).reshape(-1)
            raw_action[: min(self.action_size, len(noise))] += noise[
                : self.action_size
            ]
        raw_action = np.clip(raw_action, -1.0, 1.0)
        action = (
            decode_legacy_td3_action(raw_action)
            if self.uses_legacy_action_contract
            else decode_td3_action(raw_action)
        )
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
        self.last_policy_action = list(policy_action)
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
        if self.uses_legacy_action_contract:
            raw_accel = max(0.0, finite_float(self.last_raw_action[1]))
            raw_brake = max(0.0, finite_float(self.last_raw_action[2]))
            raw_longitudinal = clamp(raw_accel - raw_brake, -1.0, 1.0)
            policy_longitudinal = signed_longitudinal_from_legacy_outputs(
                self.last_policy_action
            )
        else:
            raw_longitudinal = finite_float(self.last_raw_action[1])
            raw_accel = max(0.0, raw_longitudinal)
            raw_brake = max(0.0, -raw_longitudinal)
            policy_longitudinal = finite_float(self.last_policy_action[1])
        return {
            "td3_policy_loaded": self.policy_loaded,
            "td3_raw_steer": self.last_raw_action[0],
            "td3_raw_longitudinal": raw_longitudinal,
            "td3_raw_accel": raw_accel,
            "td3_raw_brake": raw_brake,
            "td3_policy_raw_steer": self.last_policy_action[0],
            "td3_policy_raw_longitudinal": policy_longitudinal,
            "td3_legacy_action_adapter": False,
            "td3_legacy_action_decoder": self.uses_legacy_action_contract,
            "td3_action_steering_noise": finite_float(
                getattr(self.action_noise, "steering_noise", 0.0)
            ),
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
            return [0.0, 0.0]

        policy_observation = np.asarray(features, dtype=np.float32)
        prediction = self.policy.predict(
            policy_observation,
            deterministic=self.deterministic,
        )
        raw_action = prediction[0] if isinstance(prediction, tuple) else prediction
        limit = (
            AGENT6_LEGACY_ACTION_SIZE
            if self.uses_legacy_action_contract
            else AGENT6_ACTION_SIZE
        )
        return list(np.asarray(raw_action, dtype=float).reshape(-1)[:limit])


def _default_policy_path() -> Path:
    if evaluation_checkpoint_is_verified(DEFAULT_BEST_EVALUATION_MODEL_PATH):
        return DEFAULT_BEST_EVALUATION_MODEL_PATH
    if completed_lap_checkpoint_is_verified(
        DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH
    ):
        return DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH
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
