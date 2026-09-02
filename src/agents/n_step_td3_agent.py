from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from n_step_td3.contracts import (
    ACTION_SIZE,
    ACTION_VERSION,
    OBSERVATION_SIZE,
    REWARD_VERSION,
    HistoryEncoder,
    build_base_observation,
    decode_action,
    finite_float,
)
from n_step_td3.learner import (
    NstepTd3Config,
    load_actor_checkpoint,
    model_family_for_config,
    observation_version_for_config,
    reward_version_for_config,
)
from n_step_td3.networks import Actor
from n_step_td3.racing_line import RacingLineFeatureMap, default_racing_line_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "agent7_n_step_td3_v3"
LEGACY_MODEL_DIR = PROJECT_ROOT / "models" / "agent7_n_step_td3"
DEFAULT_RACING_LINE_PATH = default_racing_line_path("g-track-3")
DEFAULT_MODEL_CANDIDATES = (
    MODEL_DIR / "best_evaluation.pt",
    MODEL_DIR / "best_pace.pt",
    MODEL_DIR / "best_lap.pt",
    MODEL_DIR / "best_distance.pt",
    MODEL_DIR / "latest.pt",
    LEGACY_MODEL_DIR / "best_evaluation.pt",
    LEGACY_MODEL_DIR / "best_lap.pt",
    LEGACY_MODEL_DIR / "best_distance.pt",
    LEGACY_MODEL_DIR / "latest.pt",
)


class NstepTd3Agent:
    name = "torcsRL N-Step TD3 Racer"
    agent_type = "n_step_td3"
    version = "0.2"
    uses_full_control = True
    requires_racing_line = True
    track_length_m = 2843.0934
    max_steps = 15_000
    target_laps = 1
    default_model_candidates = DEFAULT_MODEL_CANDIDATES

    def __init__(
        self,
        *,
        seed: int = 0,
        policy_path: str | Path | None = None,
        policy: Any = None,
        require_policy: bool = False,
        device: str = "cpu",
        racing_line_path: str | Path | None = None,
    ) -> None:
        self.seed = int(seed)
        self.policy_path = (
            Path(policy_path) if policy_path else self.default_model_path()
        )
        self.device = torch.device(device)
        if self.requires_racing_line:
            self.racing_line_path = Path(
                racing_line_path or DEFAULT_RACING_LINE_PATH
            )
            if not self.racing_line_path.is_file():
                raise FileNotFoundError(
                    f"{self.name} requires racing-line features: "
                    f"{self.racing_line_path}"
                )
            self.racing_line = RacingLineFeatureMap.from_json(
                self.racing_line_path
            )
            self.track_length_m = self.racing_line.track_length_m
        else:
            if racing_line_path is not None:
                raise ValueError(
                    f"{self.name} cannot accept a racing-line path"
                )
            self.racing_line_path = None
            self.racing_line = None
        self.policy_config = NstepTd3Config(
            racing_line_features=self.requires_racing_line
        )
        self.model_family = model_family_for_config(self.policy_config)
        self.observation_version = observation_version_for_config(
            self.policy_config
        )
        self.action_version = ACTION_VERSION
        self.reward_version = reward_version_for_config(self.policy_config)
        self.policy = policy
        self.policy_loaded = policy is not None
        if self.policy is None:
            self.policy = self._load_policy(require_policy=require_policy)
            self.policy_loaded = self.policy is not None
        self.history = HistoryEncoder()
        self.reset()

    @property
    def config(self) -> dict[str, Any]:
        try:
            policy_path = str(self.policy_path.relative_to(PROJECT_ROOT))
        except ValueError:
            policy_path = str(self.policy_path)
        return {
            "algorithm": "N-step TD3",
            "model_family": self.model_family,
            "observation_version": self.observation_version,
            "action_version": self.action_version,
            "reward_version": self.reward_version,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "n_steps": 3,
            "policy_path": policy_path,
            "policy_loaded": self.policy_loaded,
            "learning_source": "reward_only",
            "teacher": None,
            "behaviour_cloning": False,
            "racing_line_imitation": False,
            "racing_line_observation": self.requires_racing_line,
            "racing_line_reward_shaping": bool(
                self.requires_racing_line
                and not self.policy_config.progress_reward
            ),
            "racing_line_path": (
                None
                if self.racing_line_path is None
                else str(self.racing_line_path)
            ),
            "automatic_gear_shift": True,
            "implementation": "independent_torcsrl_informed",
        }

    def reset(self) -> None:
        self.history.reset()
        self.stalled_steps = 0
        self.previous_telemetry: Mapping[str, Any] | None = None
        self.last_raw_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.last_controls = {
            "steer": 0.0,
            "accel": 0.0,
            "brake": 0.0,
            "gear": 1,
        }

    def act(
        self,
        _observation: Any,
        telemetry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if telemetry is None:
            raise ValueError("NstepTd3Agent requires raw TORCS telemetry")
        should_stop, reason = self.should_shutdown(telemetry)
        if should_stop:
            return {
                **self.last_controls,
                "accel": 0.0,
                "brake": 1.0,
                "terminate": True,
                "termination_reason": reason,
            }

        observation = self.history.observe(
            build_base_observation(
                telemetry,
                previous_telemetry=self.previous_telemetry,
                racing_line=self.racing_line,
                include_racing_line_features=self.requires_racing_line,
            )
        )
        raw_action = np.clip(self._predict(observation), -1.0, 1.0)
        controls = decode_action(
            raw_action,
            telemetry,
            physical_steering_limit=(
                self.policy_config.physical_steering_limit
            ),
        )
        self.history.record_action(raw_action)
        self.last_raw_action = raw_action.copy()
        self.last_controls = dict(controls)
        self.previous_telemetry = dict(telemetry)
        return {
            **controls,
            "terminate": False,
            "termination_reason": "",
        }

    def should_shutdown(self, telemetry: Mapping[str, Any]) -> tuple[bool, str]:
        speed = finite_float(telemetry.get("speedX"))
        track_position = abs(finite_float(telemetry.get("trackPos")))
        angle = finite_float(telemetry.get("angle"))
        damage = finite_float(telemetry.get("damage"))
        self.stalled_steps = self.stalled_steps + 1 if speed < 5.0 else 0
        if damage > 650.0:
            return True, "damage limit exceeded"
        if track_position > 1.05:
            return True, "off track"
        if np.cos(angle) < 0.0:
            return True, "wrong direction"
        if self.stalled_steps > 120:
            return True, "car stalled"
        return False, ""

    def telemetry_debug(self) -> dict[str, Any]:
        return {
            "nstep_td3_policy_loaded": self.policy_loaded,
            "nstep_td3_raw_steer": float(self.last_raw_action[0]),
            "nstep_td3_control_steer": float(self.last_controls["steer"]),
            "nstep_td3_raw_longitudinal": float(self.last_raw_action[1]),
        }

    def close(self) -> None:
        return None

    def _load_policy(self, *, require_policy: bool) -> Actor | None:
        if not self.policy_path.is_file():
            if require_policy:
                raise FileNotFoundError(
                    f"{self.name} model not found. Train it first: "
                    f"{self.policy_path}"
                )
            return None
        checkpoint = load_actor_checkpoint(self.policy_path, device="cpu")
        config = NstepTd3Config(**checkpoint["config"])
        if config.racing_line_features != self.requires_racing_line:
            raise ValueError(
                f"{self.name} checkpoint uses a different observation contract"
            )
        self.policy_config = config
        self.model_family = str(checkpoint["model_family"])
        self.observation_version = str(checkpoint["observation_version"])
        self.action_version = str(checkpoint["action_version"])
        self.reward_version = str(checkpoint.get("reward_version", REWARD_VERSION))
        if config.observation_size != OBSERVATION_SIZE:
            raise ValueError(f"{self.name} model uses a different observation size")
        actor = Actor(
            config.observation_size,
            config.action_size,
            (config.hidden_size_1, config.hidden_size_2),
        ).to(self.device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        return actor

    def _predict(self, observation: np.ndarray) -> np.ndarray:
        if self.policy is None:
            return np.zeros(ACTION_SIZE, dtype=np.float32)
        if isinstance(self.policy, torch.nn.Module):
            with torch.no_grad():
                tensor = torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(1, -1)
                prediction = self.policy(tensor).squeeze(0).cpu().numpy()
        else:
            prediction = self.policy(observation)
        supplied = np.asarray(prediction, dtype=np.float32).reshape(-1)
        action = np.zeros(ACTION_SIZE, dtype=np.float32)
        action[: min(ACTION_SIZE, supplied.size)] = supplied[:ACTION_SIZE]
        return np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)

    @classmethod
    def default_model_path(cls) -> Path:
        return next(
            (path for path in cls.default_model_candidates if path.is_file()),
            cls.default_model_candidates[0],
        )


def default_model_path() -> Path:
    return NstepTd3Agent.default_model_path()
