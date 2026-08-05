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
G_TRACK_3_LENGTH_METRES = 2843.0934

GEAR_SPEEDS = [0, 45, 85, 125, 165, 205]
POLICY_FORMAT_VERSION = 1


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
    DynaQAction("hard left / cautious throttle", -0.62, 0.22, 0.00),
    DynaQAction("left / balanced throttle", -0.38, 0.44, 0.00),
    DynaQAction("slight left / attack", -0.18, 0.76, 0.00),
    DynaQAction("straight / full attack", 0.00, 1.00, 0.00),
    DynaQAction("slight right / attack", 0.18, 0.76, 0.00),
    DynaQAction("right / balanced throttle", 0.38, 0.44, 0.00),
    DynaQAction("hard right / cautious throttle", 0.62, 0.22, 0.00),
    DynaQAction("left / coast", -0.24, 0.00, 0.00),
    DynaQAction("straight / coast", 0.00, 0.00, 0.00),
    DynaQAction("right / coast", 0.24, 0.00, 0.00),
    DynaQAction("left / brake", -0.24, 0.00, 0.44),
    DynaQAction("straight / brake", 0.00, 0.00, 0.56),
    DynaQAction("right / brake", 0.24, 0.00, 0.44),
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def shift_gears(speed: float) -> int:
    gear = 1
    for index, threshold in enumerate(GEAR_SPEEDS):
        if speed > threshold:
            gear = index + 1
    return int(clamp(float(gear), 1.0, 6.0))


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
    epsilon = 0.14
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
        self.policy_path = Path(policy_path or DEFAULT_LATEST_POLICY_PATH)
        self.load_existing = load_existing
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
        self.previous_state: tuple[int, ...] | None = None
        self.previous_action_index: int | None = None
        self.previous_telemetry: dict[str, Any] | None = None
        self.previous_steer = 0.0
        self.previous_accel = 0.0
        self.previous_brake = 0.0
        self.previous_gear = 1
        self.stuck_counter = 0

        if load_existing:
            self._load_policy(self.policy_path)

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
            ],
            "actions": [action.label for action in ACTION_SPACE],
            "alpha": 0.0 if not self.learning_enabled else self.alpha,
            "gamma": self.gamma,
            "epsilon": 0.0 if self.finalised else self.epsilon,
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

        if (
            self.learning_enabled
            and self.previous_state is not None
            and self.previous_action_index is not None
            and self.previous_telemetry is not None
        ):
            reward = self.calculate_reward(self.previous_telemetry, current_telemetry)
            terminal = self.is_terminal_transition(
                self.previous_telemetry,
                current_telemetry,
            )
            self.learn_from_transition(
                self.previous_state,
                self.previous_action_index,
                reward,
                current_state,
                terminal,
            )
            self.last_reward = reward

        shutdown, reason = self.should_shutdown(current_telemetry)
        if shutdown:
            return {
                "steer": self.previous_steer,
                "accel": 0.0,
                "brake": 1.0,
                "gear": self.previous_gear,
                "terminate": True,
                "termination_reason": reason,
            }

        action_index = self.choose_action(current_state)
        action = self.action_to_controls(action_index, current_telemetry)

        self.previous_state = current_state
        self.previous_action_index = action_index
        self.previous_telemetry = current_telemetry
        self.previous_steer = action["steer"]
        self.previous_accel = action["accel"]
        self.previous_brake = action["brake"]
        self.previous_gear = action["gear"]
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
        return (
            section,
            speed,
            track_position,
            angle,
            side_speed,
            front,
            best_direction,
        )

    def choose_action(self, state: tuple[int, ...]) -> int:
        if (
            self.learning_enabled
            and not self.finalised
            and self._random.random() < self.epsilon
        ):
            return self._random.randrange(len(ACTION_SPACE))

        values = [
            self.q_value(state, action_index)
            for action_index in range(len(ACTION_SPACE))
        ]
        best_value = max(values)
        best_actions = [
            action_index
            for action_index, value in enumerate(values)
            if math.isclose(value, best_value, abs_tol=1e-12)
        ]
        return self._random.choice(best_actions)

    def action_to_controls(
        self,
        action_index: int,
        telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = ACTION_SPACE[action_index]
        speed = float_value(telemetry.get("speedX"))
        track_pos = float_value(telemetry.get("trackPos"))
        angle = float_value(telemetry.get("angle"))

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
        if abs(track_pos) > 0.92:
            target_steer = clamp(target_steer - track_pos * 0.34, -0.88, 0.88)
            max_delta = max(max_delta, 0.10)
        if abs(angle) > 0.62:
            target_steer = clamp(target_steer + angle * 0.20, -0.88, 0.88)

        steer = self.previous_steer + clamp(
            target_steer - self.previous_steer,
            -max_delta,
            max_delta,
        )

        accel = request.accel
        brake = request.brake
        if brake > 0.02:
            accel = 0.0
        if abs(track_pos) > 0.82 or abs(angle) > 0.42:
            accel = min(accel, 0.28)
        if abs(float_value(telemetry.get("speedY"))) > 10.0:
            accel = min(accel, 0.12)
            brake = max(brake, 0.14)

        accel = self.previous_accel + clamp(accel - self.previous_accel, -0.30, 0.22)
        brake = self.previous_brake + clamp(brake - self.previous_brake, -0.25, 0.34)
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
    ) -> float:
        progress = self._progress_delta(previous, current)
        speed = max(0.0, float_value(current.get("speedX")))
        angle = abs(float_value(current.get("angle")))
        track_position = abs(float_value(current.get("trackPos")))
        side_speed = abs(float_value(current.get("speedY")))
        track = track_sensors(current)
        front = track[9] if len(track) > 9 else 200.0
        damage_delta = max(
            0.0,
            float_value(current.get("damage")) - float_value(previous.get("damage")),
        )

        alignment = max(0.0, math.cos(angle))
        reward = progress * 0.10
        reward += min(speed, 220.0) * alignment * 0.012
        reward -= max(0.0, track_position - 0.35) * 1.8
        reward -= max(0.0, angle - 0.08) * 2.2
        reward -= max(0.0, side_speed - 4.0) * 0.09

        if progress < 0.08 and speed < 8.0:
            reward -= 2.5
        if front < 18.0 and speed > 80.0:
            reward -= (80.0 - front) * 0.035
        if track and min(track) < 0.0:
            reward -= 35.0
        if track_position > 1.0:
            reward -= 45.0
        if damage_delta > 0.0:
            reward -= 60.0 + damage_delta * 0.20

        return clamp(reward, -120.0, 120.0)

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
        if track_position > 1.22:
            return True, "out of bounds"
        if angle > 2.25:
            return True, "wrong direction"
        if self.stuck_counter > 260:
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
        return (
            damage_delta > 25.0
            or abs(float_value(current.get("trackPos"))) > 1.08
            or abs(float_value(current.get("angle"))) > 1.45
            or bool(track and min(track) < -0.1)
        )

    def close(self) -> None:
        if self.save_policy and self.q_values:
            self.save(self.policy_path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": POLICY_FORMAT_VERSION,
            "algorithm": "dyna_q",
            "track_id": "g-track-3",
            "track_length_m": self.track_length_m,
            "agent_version": self.version,
            "seed": self.seed,
            "steps_trained": self.steps_trained,
            "real_updates": self.real_updates,
            "planning_updates": self.planning_updates,
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

    def _load_policy(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"No Dyna-Q policy found at {path}. Train a learning agent first "
                "or provide a final policy file."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "dyna_q":
            raise ValueError(f"{path} is not a Dyna-Q policy file")

        self.q_values = {
            str(key): float(value)
            for key, value in dict(payload.get("q_values", {})).items()
        }
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


def _bucket(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return index
    return len(thresholds)
