from __future__ import annotations

import json
import math
import random
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
DEFAULT_LATEST_POLICY_PATH = POLICY_DIR / "dyna_q_g_track_3_latest.json"
DEFAULT_FINAL_POLICY_PATH = POLICY_DIR / "dyna_q_g_track_3_final.json"
DEFAULT_BEST_POLICY_PATH = POLICY_DIR / "dyna_q_g_track_3_best.json"
G_TRACK_3_LENGTH_METRES = 2843.0934

GEAR_SPEEDS = [0, 45, 85, 125, 165, 205]
POLICY_FORMAT_VERSION = 1
TRAINING_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class DynaQAction:
    label: str
    steer: float
    accel: float
    brake: float


@dataclass(frozen=True)
class DynaQTransition:
    next_state: tuple[int, ...]
    reward: float
    terminal: bool


ACTION_SPACE: tuple[DynaQAction, ...] = (
    DynaQAction("straight / launch attack", 0.00, 1.00, 0.00),
    DynaQAction("slight left / full attack", -0.10, 0.92, 0.00),
    DynaQAction("slight right / full attack", 0.10, 0.92, 0.00),
    DynaQAction("left arc / attack", -0.22, 0.74, 0.00),
    DynaQAction("right arc / attack", 0.22, 0.74, 0.00),
    DynaQAction("left arc / balanced", -0.34, 0.54, 0.00),
    DynaQAction("right arc / balanced", 0.34, 0.54, 0.00),
    DynaQAction("hard left / rotate", -0.58, 0.18, 0.00),
    DynaQAction("hard right / rotate", 0.58, 0.18, 0.00),
    DynaQAction("straight / maintain", 0.00, 0.48, 0.00),
    DynaQAction("slight left / coast", -0.18, 0.00, 0.00),
    DynaQAction("straight / coast", 0.00, 0.00, 0.00),
    DynaQAction("slight right / coast", 0.18, 0.00, 0.00),
    DynaQAction("left / trail brake", -0.22, 0.00, 0.32),
    DynaQAction("straight / firm brake", 0.00, 0.00, 0.55),
    DynaQAction("right / trail brake", 0.22, 0.00, 0.32),
    DynaQAction("straight / emergency brake", 0.00, 0.00, 0.85),
    DynaQAction("left recovery / drive", -0.46, 0.34, 0.00),
    DynaQAction("right recovery / drive", 0.46, 0.34, 0.00),
    DynaQAction("left recovery / coast", -0.48, 0.00, 0.00),
    DynaQAction("right recovery / coast", 0.48, 0.00, 0.00),
    DynaQAction("left recovery / brake", -0.40, 0.00, 0.24),
    DynaQAction("right recovery / brake", 0.40, 0.00, 0.24),
    DynaQAction("slight left / maintain", -0.12, 0.52, 0.00),
    DynaQAction("slight right / maintain", 0.12, 0.52, 0.00),
    DynaQAction("straight / gentle brake", 0.00, 0.00, 0.24),
)

LAUNCH_ACTIONS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 23, 24)
FRONT_DANGER_ACTIONS = (10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 25)
HIGH_SPEED_ACTIONS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 23, 24, 25)
RECOVERY_ACTIONS = (5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25)
RIGHT_EDGE_RECOVERY_ACTIONS = (5, 7, 10, 13, 14, 16, 17, 19, 21, 25)
LEFT_EDGE_RECOVERY_ACTIONS = (6, 8, 12, 14, 15, 16, 18, 20, 22, 25)
ALL_ACTIONS = tuple(range(len(ACTION_SPACE)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def shift_gears(speed: float) -> int:
    gear = 1
    for index, threshold in enumerate(GEAR_SPEEDS):
        if speed > threshold:
            gear = index + 1
    return int(clamp(float(gear), 1.0, 6.0))


def stable_open_corridor(
    *,
    speed: float,
    track_position: float,
    angle: float,
    side_speed: float,
    front: float,
) -> bool:
    return (
        speed < 110.0
        and track_position < 0.48
        and angle < 0.22
        and side_speed < 5.5
        and front > 90.0
    )


def edge_recovery_zone(track_position: float, min_track_sensor: float) -> bool:
    return track_position > 1.0 or (
        track_position > 0.88 and min_track_sensor < -0.1
    )


class DynaQBaseAgent:
    name = "Dyna-Q Agent"
    agent_type = "dyna_q"
    version = "0.1"
    uses_full_control = True
    requires_racing_line = False
    max_steps = 150000
    target_laps = 1

    alpha = 0.18
    gamma = 0.96
    epsilon_start = 0.24
    epsilon_min = 0.035
    epsilon_decay_steps = 80000
    epsilon = epsilon_start
    planning_steps = 16
    track_length_m = G_TRACK_3_LENGTH_METRES
    section_count = 48

    def __init__(
        self,
        *,
        seed: int | None = None,
        policy_path: str | Path | None = None,
        load_existing: bool = False,
        save_policy: bool = False,
        learning_enabled: bool = True,
        finalised: bool = False,
    ) -> None:
        self.seed = seed if seed is not None else secrets.randbits(32)
        self._explicit_seed = seed is not None
        self.policy_path = Path(policy_path or DEFAULT_LATEST_POLICY_PATH)
        self.load_existing = False
        self.save_policy = save_policy
        self.learning_enabled = learning_enabled
        self.finalised = finalised
        self._random = random.Random(self.seed)
        self.q_values: dict[str, float] = {}
        self.model: dict[str, DynaQTransition] = {}
        self.steps_trained = 0
        self.real_updates = 0
        self.planning_updates = 0
        self.episode_count = 0
        self.last_reward = 0.0
        self.last_td_error = 0.0
        self.last_action_label = ""
        self.last_state: tuple[int, ...] | None = None
        self.last_action_index: int | None = None
        self.last_action_was_exploratory = False
        self.last_selected_q_value = 0.0
        self.exploration_override: float | None = None
        self.previous_state: tuple[int, ...] | None = None
        self.previous_action_index: int | None = None
        self.previous_telemetry: dict[str, Any] | None = None
        self.previous_steer = 0.0
        self.previous_accel = 0.0
        self.previous_brake = 0.0
        self.previous_gear = 1
        self.stuck_counter = 0
        self.progress_stall_counter = 0

        if load_existing:
            self.load_existing = self._load_policy(
                self.policy_path,
                strict=finalised,
            )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "algorithm": "Dyna-Q",
            "track": "g-track-3",
            "track_length_m": self.track_length_m,
            "state_encoding": [
                "lap section",
                "speed band",
                "track position band",
                "heading angle band",
                "side speed band",
                "front sensor danger",
                "best sensor direction",
                "forward corridor opening",
                "sensor opening balance",
            ],
            "actions": [action.label for action in ACTION_SPACE],
            "alpha": 0.0 if not self.learning_enabled else self.alpha,
            "gamma": self.gamma,
            "epsilon": self.exploration_rate(),
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "planning_steps": 0 if not self.learning_enabled else self.planning_steps,
            "policy_path": str(self.policy_path.relative_to(PROJECT_ROOT)),
            "loads_existing_policy": self.load_existing,
            "saves_policy": self.save_policy,
            "learning_enabled": self.learning_enabled,
        }

    def reset(self) -> None:
        self._random.seed(self.seed)
        self.previous_state = None
        self.previous_action_index = None
        self.previous_telemetry = None
        self.previous_steer = 0.0
        self.previous_accel = 0.0
        self.previous_brake = 0.0
        self.previous_gear = 1
        self.stuck_counter = 0
        self.progress_stall_counter = 0
        self.last_state = None
        self.last_action_index = None
        self.last_action_was_exploratory = False
        self.last_selected_q_value = 0.0
        self.episode_count += 1

    def act(
        self,
        _observation: Any,
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if telemetry is None:
            raise ValueError("Dyna-Q agents require raw TORCS telemetry")

        current_telemetry = dict(telemetry)
        current_state = self.encode_state(current_telemetry)
        self._update_progress_stall(current_telemetry)
        shutdown, reason = self.should_shutdown(current_telemetry)
        if (
            self.learning_enabled
            and self.previous_state is not None
            and self.previous_action_index is not None
            and self.previous_telemetry is not None
        ):
            reward = self.calculate_reward(
                self.previous_telemetry,
                current_telemetry,
                self.previous_action_index,
            )
            terminal = shutdown or self.is_terminal_transition(
                self.previous_telemetry,
                current_telemetry,
            )
            if shutdown:
                reward += self.terminal_reward(reason)
            self.learn_from_transition(
                self.previous_state,
                self.previous_action_index,
                reward,
                current_state,
                terminal,
            )
            self.last_reward = reward

        if shutdown:
            return {
                "steer": self.previous_steer,
                "accel": 0.0,
                "brake": 1.0,
                "gear": self.previous_gear,
                "terminate": True,
                "termination_reason": reason,
            }

        action_index = self.choose_action(current_state, current_telemetry)
        action = self.action_to_controls(action_index, current_telemetry)

        self.previous_state = current_state
        self.previous_action_index = action_index
        self.previous_telemetry = current_telemetry
        self.previous_steer = action["steer"]
        self.previous_accel = action["accel"]
        self.previous_brake = action["brake"]
        self.previous_gear = action["gear"]
        self.last_state = current_state
        self.last_action_index = action_index
        self.last_action_label = ACTION_SPACE[action_index].label
        return action

    def encode_state(self, telemetry: Mapping[str, Any]) -> tuple[int, ...]:
        track = track_sensors(telemetry)
        section = self._section_bin(float_value(telemetry.get("distFromStart")))
        speed = _bucket(
            float_value(telemetry.get("speedX")),
            (20.0, 45.0, 70.0, 95.0, 120.0, 145.0, 170.0, 195.0),
        )
        track_position = _bucket(
            float_value(telemetry.get("trackPos")),
            (-0.86, -0.58, -0.32, -0.12, 0.12, 0.32, 0.58, 0.86),
        )
        angle = _bucket(
            float_value(telemetry.get("angle")),
            (-0.52, -0.30, -0.16, -0.055, 0.055, 0.16, 0.30, 0.52),
        )
        side_speed = _bucket(
            float_value(telemetry.get("speedY")),
            (-16.0, -8.0, -3.0, 3.0, 8.0, 16.0),
        )
        front = _bucket(
            track[9] if len(track) > 9 else 200.0,
            (10.0, 20.0, 34.0, 54.0, 82.0, 120.0, 165.0),
        )
        best_direction = self._best_sensor_direction(track)
        corridor_opening = _bucket(
            max(track[7:12]) if len(track) >= 12 else 200.0,
            (18.0, 32.0, 52.0, 78.0, 112.0, 155.0),
        )
        sensor_balance = self._sensor_opening_balance(track)
        return (
            section,
            speed,
            track_position,
            angle,
            side_speed,
            front,
            best_direction,
            corridor_opening,
            sensor_balance,
        )

    def choose_action(
        self,
        state: tuple[int, ...],
        telemetry: Mapping[str, Any] | None = None,
    ) -> int:
        available_actions = self.available_actions(telemetry)
        if (
            self.learning_enabled
            and not self.finalised
            and self._random.random() < self.exploration_rate()
        ):
            action_index = self._random.choice(available_actions)
            self.last_action_was_exploratory = True
            self.last_selected_q_value = self.q_value(state, action_index)
            return action_index

        values = [
            self.q_value(state, action_index)
            for action_index in available_actions
        ]
        best_value = max(values)
        best_actions = [
            action_index
            for action_index, value in zip(available_actions, values)
            if math.isclose(value, best_value, abs_tol=1e-12)
        ]
        action_index = self._random.choice(best_actions)
        self.last_action_was_exploratory = False
        self.last_selected_q_value = self.q_value(state, action_index)
        return action_index

    def telemetry_debug(self) -> dict[str, Any]:
        return {
            "dyna_q_learning_enabled": self.learning_enabled,
            "dyna_q_finalised": self.finalised,
            "dyna_q_state": _format_state(self.last_state),
            "dyna_q_action_index": self.last_action_index,
            "dyna_q_action_label": self.last_action_label,
            "dyna_q_action_source": (
                "explore" if self.last_action_was_exploratory else "greedy"
            ),
            "dyna_q_reward": self.last_reward,
            "dyna_q_td_error": self.last_td_error,
            "dyna_q_selected_q": self.last_selected_q_value,
            "dyna_q_epsilon": self.exploration_rate(),
            "dyna_q_q_states": len(self.q_values),
            "dyna_q_model_states": len(self.model),
            "dyna_q_real_updates": self.real_updates,
            "dyna_q_planning_updates": self.planning_updates,
        }

    def available_actions(self, telemetry: Mapping[str, Any] | None) -> tuple[int, ...]:
        if telemetry is None:
            return ALL_ACTIONS

        speed = float_value(telemetry.get("speedX"))
        signed_track_pos = float_value(telemetry.get("trackPos"))
        track_pos = abs(signed_track_pos)
        angle = abs(float_value(telemetry.get("angle")))
        signed_side_speed = float_value(telemetry.get("speedY"))
        side_speed = abs(signed_side_speed)
        track = track_sensors(telemetry)
        front = track[9] if len(track) > 9 else 200.0
        front_danger = front < 24.0 and speed > 48.0
        high_speed = speed > 145.0
        edge_danger = track_pos > 0.62
        outward_drift = signed_track_pos * signed_side_speed > 1.2
        unstable = (
            track_pos > 0.82
            or angle > 0.48
            or side_speed > 10.0
            or front < 12.0
            or (track_pos > 0.48 and outward_drift)
        )

        if edge_danger:
            return (
                RIGHT_EDGE_RECOVERY_ACTIONS
                if signed_track_pos > 0.0
                else LEFT_EDGE_RECOVERY_ACTIONS
            )
        if speed < 34.0 and not unstable:
            return LAUNCH_ACTIONS
        if front_danger:
            return FRONT_DANGER_ACTIONS
        if unstable:
            return RECOVERY_ACTIONS
        if high_speed:
            return HIGH_SPEED_ACTIONS
        return ALL_ACTIONS

    def exploration_rate(self) -> float:
        if self.finalised or not self.learning_enabled:
            return 0.0
        if self.exploration_override is not None:
            return clamp(float(self.exploration_override), 0.0, 1.0)
        progress = clamp(
            self.steps_trained / max(1.0, float(self.epsilon_decay_steps)),
            0.0,
            1.0,
        )
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            1.0 - progress
        )

    def action_to_controls(
        self,
        action_index: int,
        telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = ACTION_SPACE[action_index]
        speed = float_value(telemetry.get("speedX"))
        track_pos = float_value(telemetry.get("trackPos"))
        angle = float_value(telemetry.get("angle"))
        track = track_sensors(telemetry)
        min_track_sensor = min(track) if track else 200.0
        edge_recovery = edge_recovery_zone(abs(track_pos), min_track_sensor)
        recovery_drive = (
            request.accel > 0.20
            and request.brake < 0.05
            and abs(request.steer) >= 0.40
        )

        steer_limit = 0.78
        max_delta = 0.085
        if speed > 165.0:
            steer_limit = 0.42
            max_delta = 0.038
        elif speed > 120.0:
            steer_limit = 0.52
            max_delta = 0.050
        elif speed > 80.0:
            steer_limit = 0.64
            max_delta = 0.066

        target_steer = clamp(request.steer, -steer_limit, steer_limit)
        if abs(track_pos) > 0.68:
            target_steer = clamp(target_steer - track_pos * 0.28, -0.88, 0.88)
            max_delta = max(max_delta, 0.085)
        if abs(track_pos) > 0.92:
            target_steer = clamp(target_steer - track_pos * 0.42, -0.90, 0.90)
            max_delta = max(max_delta, 0.12)
        if edge_recovery:
            target_steer = clamp(target_steer - track_pos * 0.74, -0.90, 0.90)
            max_delta = max(max_delta, 0.20 if speed > 18.0 else 0.26)
        if speed < 45.0 and abs(track_pos) > 0.62:
            max_delta = max(max_delta, 0.14)
        if abs(angle) > 0.62:
            target_steer = clamp(target_steer + angle * 0.20, -0.88, 0.88)

        steer = self.previous_steer + clamp(
            target_steer - self.previous_steer,
            -max_delta,
            max_delta,
        )

        accel = request.accel
        brake = request.brake
        front = track[9] if len(track) > 9 else 200.0
        if brake > 0.02:
            accel = 0.0
        if abs(track_pos) > 0.90:
            if speed < 45.0 and recovery_drive:
                accel = min(max(request.accel, 0.32), 0.48)
                brake = 0.0
            else:
                accel = min(accel, 0.18)
                if speed > 45.0:
                    brake = max(brake, 0.10)
        elif abs(track_pos) > 0.82 or abs(angle) > 0.42:
            if speed < 38.0 and recovery_drive:
                accel = min(max(request.accel, 0.30), 0.48)
            else:
                accel = min(accel, 0.28)
        elif abs(track_pos) > 0.68:
            accel = min(accel, 0.42)
        if front < 22.0 and speed > 55.0:
            accel = min(accel, 0.08)
            brake = max(brake, 0.22)
        if abs(float_value(telemetry.get("speedY"))) > 10.0:
            accel = min(accel, 0.12)
            brake = max(brake, 0.14)
        if speed < 18.0 and abs(track_pos) < 1.02 and abs(angle) < 0.70:
            accel = max(accel, 0.52 if abs(track_pos) > 0.82 else 0.58)
            brake = 0.0
        elif speed < 34.0 and brake < 0.05 and abs(track_pos) < 0.82:
            accel = max(accel, 0.38)
        if edge_recovery:
            if speed < 34.0 and abs(angle) < 0.90:
                accel = max(accel, 0.32 if abs(track_pos) > 1.12 else 0.40)
                brake = 0.0
            elif speed < 18.0:
                accel = max(accel, 0.24)
                brake = 0.0
            if speed > 58.0 or abs(angle) > 1.05:
                accel = min(accel, 0.10)
                brake = max(brake, 0.18)

        accel = self.previous_accel + clamp(accel - self.previous_accel, -0.30, 0.22)
        brake_up = 0.55 if request.brake > 0.70 else 0.34
        brake = self.previous_brake + clamp(
            brake - self.previous_brake,
            -0.25,
            brake_up,
        )
        if brake > 0.05:
            accel = 0.0

        return {
            "steer": clamp(steer, -0.90, 0.90),
            "accel": clamp(accel, 0.0, 1.0),
            "brake": clamp(brake, 0.0, 1.0),
            "gear": shift_gears(speed),
            "terminate": False,
            "termination_reason": "",
        }

    def calculate_reward(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        action_index: int | None = None,
    ) -> float:
        progress = self._progress_delta(previous, current)
        speed = max(0.0, float_value(current.get("speedX")))
        signed_angle = float_value(current.get("angle"))
        angle = abs(signed_angle)
        signed_track_position = float_value(current.get("trackPos"))
        track_position = abs(signed_track_position)
        previous_track_position = abs(float_value(previous.get("trackPos")))
        signed_side_speed = float_value(current.get("speedY"))
        side_speed = abs(signed_side_speed)
        track = track_sensors(current)
        front = track[9] if len(track) > 9 else 200.0
        invalid_track_sensor = bool(track and min(track) < 0.0)
        damage_delta = max(
            0.0,
            float_value(current.get("damage")) - float_value(previous.get("damage")),
        )

        alignment = max(0.0, math.cos(angle))
        outward_motion = signed_track_position * signed_side_speed
        outward_heading = signed_track_position * signed_angle < -0.045
        stable_open = stable_open_corridor(
            speed=speed,
            track_position=track_position,
            angle=angle,
            side_speed=side_speed,
            front=front,
        )
        reward = progress * 0.62
        if speed < 45.0:
            reward += progress * 0.32
        if progress > 0.18:
            reward += 0.55
        if speed < 24.0 and progress > 0.12:
            reward += 0.8
        if track_position < 0.84 and angle < 0.55:
            reward += min(speed, 220.0) * alignment * 0.006
        if (
            progress > 0.08
            and not invalid_track_sensor
            and track_position < 0.58
            and angle < 0.38
            and side_speed < 4.0
        ):
            reward += 0.45 + progress * 0.36
        if progress > 0.12 and track_position < 0.42 and angle < 0.24:
            reward += 0.30
        reward -= max(0.0, track_position - 0.35) * 2.2
        reward -= max(0.0, track_position - 0.66) * 8.5
        reward -= max(0.0, track_position - 0.78) * 11.0
        reward -= max(0.0, track_position - 0.88) * 18.0
        if track_position > 0.92 and speed > 28.0:
            reward -= min(24.0, (track_position - 0.92) * 42.0)
            reward -= min(18.0, max(0.0, speed - 28.0) * 0.16)
        if previous_track_position > 0.48:
            edge_delta = previous_track_position - track_position
            reward += edge_delta * 10.0
            if previous_track_position > 0.92:
                reward += edge_delta * 14.0
        if track_position > 0.50 and outward_motion > 1.2:
            reward -= min(7.0, outward_motion * 0.45)
        if track_position > 0.50 and outward_motion < -1.2:
            reward += min(2.5, abs(outward_motion) * 0.18)
        if track_position > 0.52 and outward_heading:
            reward -= min(6.0, track_position * angle * 8.0)
        reward -= max(0.0, angle - 0.06) * 2.8
        reward -= max(0.0, angle - 0.28) * 5.0
        reward -= max(0.0, side_speed - 3.5) * 0.12

        if progress < 0.04 and speed < 8.0:
            reward -= 6.0
        elif progress < 0.04 and speed < 25.0:
            reward -= 2.8
        elif progress < 0.04 and speed > 35.0:
            reward -= 2.2
        if action_index is not None:
            action = ACTION_SPACE[action_index]
            if speed < 24.0 and action.brake > 0.05 and track_position < 0.90:
                reward -= 8.0
            if speed < 18.0 and action.accel < 0.20 and track_position < 0.90:
                reward -= 5.0
            if speed > 105.0 and abs(action.steer) > 0.45:
                reward -= 2.0
            if stable_open and action.brake > 0.05:
                reward -= 3.0
            if stable_open and abs(action.steer) > 0.30:
                reward -= min(2.5, (abs(action.steer) - 0.30) * 6.0)
            if stable_open and action.accel > 0.30 and action.brake < 0.05:
                reward += 0.35
            if track_position > 0.94 and speed < 35.0 and action.brake > 0.05:
                reward -= 4.0
            if (
                track_position > 0.94
                and speed < 34.0
                and action.accel > 0.20
                and action.brake < 0.05
                and signed_track_position * action.steer < 0.0
            ):
                reward += 1.0
        if front < 26.0 and speed > 65.0:
            reward -= (65.0 - front) * 0.06
        if invalid_track_sensor:
            reward -= 42.0
            reward -= min(30.0, max(0.0, speed - 18.0) * 0.20)
            if track_position > 0.92:
                reward -= min(22.0, (track_position - 0.92) * 55.0)
        if track_position > 1.0:
            reward -= 56.0
        if damage_delta > 0.0:
            reward -= 60.0 + damage_delta * 0.20

        return clamp(reward, -120.0, 120.0)

    @staticmethod
    def terminal_reward(reason: str) -> float:
        if reason in {"car stuck", "stuck", "no progress"}:
            return -90.0
        if reason in {"out of bounds", "wrong direction"}:
            return -110.0
        if reason == "damage limit exceeded":
            return -120.0
        return -70.0

    def learn_from_transition(
        self,
        state: tuple[int, ...],
        action_index: int,
        reward: float,
        next_state: tuple[int, ...],
        terminal: bool = False,
    ) -> None:
        self._q_update(state, action_index, reward, next_state, terminal)
        self.model[self._transition_key(state, action_index)] = DynaQTransition(
            next_state=next_state,
            reward=reward,
            terminal=terminal,
        )
        self.steps_trained += 1
        self.real_updates += 1

        if self.model and self.planning_steps > 0:
            model_keys = list(self.model)
            for _ in range(self.planning_steps):
                key = self._random.choice(model_keys)
                planned_state, planned_action = self._parse_transition_key(key)
                transition = self.model[key]
                self._q_update(
                    planned_state,
                    planned_action,
                    transition.reward,
                    transition.next_state,
                    transition.terminal,
                )
                self.planning_updates += 1

    def q_value(self, state: tuple[int, ...], action_index: int) -> float:
        return self.q_values.get(self._transition_key(state, action_index), 0.0)

    def should_shutdown(self, telemetry: Mapping[str, Any]) -> tuple[bool, str]:
        speed = float_value(telemetry.get("speedX"))
        track_position = abs(float_value(telemetry.get("trackPos")))
        angle = abs(float_value(telemetry.get("angle")))
        damage = float_value(telemetry.get("damage"))

        self.stuck_counter = self.stuck_counter + 1 if speed < 5.0 else 0
        if damage > 650.0:
            return True, "damage limit exceeded"
        if track_position > 1.32:
            return True, "out of bounds"
        if track_position > 1.24 and (speed > 35.0 or angle > 1.20):
            return True, "out of bounds"
        if angle > 2.25:
            return True, "wrong direction"
        if self.progress_stall_counter > 300:
            return True, "no progress"
        if self.stuck_counter > 220:
            return True, "car stuck"
        return False, ""

    def is_terminal_transition(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> bool:
        damage_delta = float_value(current.get("damage")) - float_value(
            previous.get("damage")
        )
        track = track_sensors(current)
        track_position = abs(float_value(current.get("trackPos")))
        return (
            damage_delta > 25.0
            or track_position > 1.24
            or abs(float_value(current.get("angle"))) > 1.45
            or bool(track and min(track) < -0.1 and track_position > 1.18)
        )

    def close(self) -> None:
        if self.save_policy and self.q_values:
            self.save(self.policy_path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": POLICY_FORMAT_VERSION,
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "algorithm": "dyna_q",
            "track_id": "g-track-3",
            "track_length_m": self.track_length_m,
            "agent_version": self.version,
            "seed": self.seed,
            "steps_trained": self.steps_trained,
            "real_updates": self.real_updates,
            "planning_updates": self.planning_updates,
            "epsilon": self.exploration_rate(),
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "action_labels": [action.label for action in ACTION_SPACE],
            "q_values": self.q_values,
            "model": {
                key: {
                    "next_state": list(transition.next_state),
                    "reward": transition.reward,
                    "terminal": transition.terminal,
                }
                for key, transition in self.model.items()
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_policy(self, path: Path, *, strict: bool = True) -> bool:
        if not path.is_file():
            raise FileNotFoundError(
                f"No Dyna-Q policy found at {path}. Train a learning agent first "
                "or provide a final policy file."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "dyna_q":
            raise ValueError(f"{path} is not a Dyna-Q policy file")

        current_action_labels = [action.label for action in ACTION_SPACE]
        saved_action_labels = payload.get("action_labels")
        schema_version = payload.get("training_schema_version")
        if (
            saved_action_labels != current_action_labels
            or schema_version != TRAINING_SCHEMA_VERSION
        ):
            message = (
                f"{path} was trained with a different Dyna-Q learning schema. "
                "Train a new policy from scratch before loading it."
            )
            if strict:
                raise ValueError(message)
            return False

        self.q_values = {
            str(key): float(value)
            for key, value in dict(payload.get("q_values", {})).items()
        }
        if self.finalised and not self._explicit_seed:
            saved_seed = payload.get("seed")
            if isinstance(saved_seed, int):
                self.seed = saved_seed
                self._random.seed(self.seed)
        self.model = {}
        for key, value in dict(payload.get("model", {})).items():
            if not isinstance(value, Mapping):
                continue
            self.model[str(key)] = DynaQTransition(
                next_state=tuple(int(part) for part in value.get("next_state", [])),
                reward=float(value.get("reward", 0.0)),
                terminal=bool(value.get("terminal", False)),
            )
        self.steps_trained = int(payload.get("steps_trained", 0))
        self.real_updates = int(payload.get("real_updates", 0))
        self.planning_updates = int(payload.get("planning_updates", 0))
        return True

    def _q_update(
        self,
        state: tuple[int, ...],
        action_index: int,
        reward: float,
        next_state: tuple[int, ...],
        terminal: bool,
    ) -> None:
        key = self._transition_key(state, action_index)
        current = self.q_values.get(key, 0.0)
        next_value = 0.0 if terminal else max(
            self.q_value(next_state, next_action)
            for next_action in range(len(ACTION_SPACE))
        )
        target = reward + self.gamma * next_value
        td_error = target - current
        self.q_values[key] = current + self.alpha * td_error
        self.last_td_error = td_error

    def _progress_delta(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> float:
        previous_raced = previous.get("distRaced")
        current_raced = current.get("distRaced")
        if previous_raced is not None and current_raced is not None:
            delta = float_value(current_raced) - float_value(previous_raced)
            if -5.0 <= delta <= 160.0:
                return max(0.0, delta)

        previous_distance = (
            float_value(previous.get("distFromStart")) % self.track_length_m
        )
        current_distance = (
            float_value(current.get("distFromStart")) % self.track_length_m
        )
        delta = (current_distance - previous_distance) % self.track_length_m
        if delta > self.track_length_m * 0.50:
            return 0.0
        return delta

    def _update_progress_stall(self, current: Mapping[str, Any]) -> None:
        if self.previous_telemetry is None:
            self.progress_stall_counter = 0
            return

        progress = self._progress_delta(self.previous_telemetry, current)
        speed = max(0.0, float_value(current.get("speedX")))
        track_position = abs(float_value(current.get("trackPos")))
        barely_moving = progress < 0.025 and speed < 10.0
        creeping = progress < 0.04 and speed < 16.0

        if track_position < 1.05 and (barely_moving or creeping):
            self.progress_stall_counter += 1
        else:
            self.progress_stall_counter = max(0, self.progress_stall_counter - 4)

    def _section_bin(self, distance: float) -> int:
        wrapped = distance % self.track_length_m
        section = int(wrapped / self.track_length_m * self.section_count)
        return int(clamp(float(section), 0.0, float(self.section_count - 1)))

    def _best_sensor_direction(self, track: list[float]) -> int:
        if len(track) < 19:
            return 2
        window = range(4, 15)
        best_index = max(window, key=lambda index: track[index])
        if best_index <= 5:
            return 0
        if best_index <= 8:
            return 1
        if best_index == 9:
            return 2
        if best_index <= 12:
            return 3
        return 4

    def _sensor_opening_balance(self, track: list[float]) -> int:
        if len(track) < 14:
            return 2
        left_opening = sum(track[5:9]) / 4.0
        right_opening = sum(track[10:14]) / 4.0
        difference = left_opening - right_opening
        if difference < -55.0:
            return 0
        if difference < -18.0:
            return 1
        if difference > 55.0:
            return 4
        if difference > 18.0:
            return 3
        return 2

    @staticmethod
    def _transition_key(state: tuple[int, ...], action_index: int) -> str:
        return f"{','.join(str(part) for part in state)}|{action_index}"

    @staticmethod
    def _parse_transition_key(key: str) -> tuple[tuple[int, ...], int]:
        state_text, action_text = key.split("|", 1)
        return tuple(int(part) for part in state_text.split(",")), int(action_text)


class DynaQLearningAgent(DynaQBaseAgent):
    name = "Dyna-Q Learning Agent"
    agent_type = "dyna_q_learning"
    version = "0.1"

    def __init__(
        self,
        *,
        seed: int | None = None,
        policy_path: str | Path | None = None,
        load_existing: bool = True,
    ) -> None:
        path = Path(policy_path or DEFAULT_LATEST_POLICY_PATH)
        super().__init__(
            seed=seed,
            policy_path=path,
            load_existing=load_existing and path.is_file(),
            save_policy=True,
            learning_enabled=True,
            finalised=False,
        )


class DynaQFinalisedAgent(DynaQBaseAgent):
    name = "Dyna-Q Finalised Agent"
    agent_type = "dyna_q_finalised"
    version = "0.1"

    def __init__(
        self,
        *,
        seed: int | None = None,
        policy_path: str | Path | None = None,
    ) -> None:
        selected_path = (
            Path(policy_path)
            if policy_path is not None
            else _default_final_policy_path()
        )
        super().__init__(
            seed=seed,
            policy_path=selected_path,
            load_existing=True,
            save_policy=False,
            learning_enabled=False,
            finalised=True,
        )
        self.alpha = 0.0
        self.epsilon = 0.0
        self.planning_steps = 0


def _default_final_policy_path() -> Path:
    if DEFAULT_FINAL_POLICY_PATH.is_file():
        return DEFAULT_FINAL_POLICY_PATH
    return DEFAULT_LATEST_POLICY_PATH


def track_sensors(telemetry: Mapping[str, Any]) -> list[float]:
    track = telemetry.get("track")
    if track is not None:
        try:
            values = [float(value) for value in list(track)[:19]]
        except (TypeError, ValueError):
            values = []
        if len(values) >= 19:
            return values

    values = []
    for index in range(19):
        values.append(float_value(telemetry.get(f"track_{index}", 200.0)))
    return values


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_state(state: tuple[int, ...] | None) -> str:
    if state is None:
        return ""
    return ",".join(str(part) for part in state)


def _bucket(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)
