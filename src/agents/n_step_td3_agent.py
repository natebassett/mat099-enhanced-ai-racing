from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from n_step_td3.contracts import (
    ACTION_SIZE,
    ACTION_VERSION,
    MODEL_FAMILY,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    REWARD_VERSION,
    HistoryEncoder,
    build_base_observation,
    decode_action,
    finite_float,
)
from n_step_td3.learner import NstepTd3Config, load_actor_checkpoint
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
        self.policy_path = Path(policy_path) if policy_path else default_model_path()
        self.device = torch.device(device)
        self.racing_line_path = Path(
            racing_line_path or DEFAULT_RACING_LINE_PATH
        )
        if not self.racing_line_path.is_file():
            raise FileNotFoundError(
                "Agent 7 requires the torcsRL-style racing-line features: "
                f"{self.racing_line_path}"
            )
        self.racing_line = RacingLineFeatureMap.from_json(
            self.racing_line_path
        )
        self.track_length_m = self.racing_line.track_length_m
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
            "model_family": MODEL_FAMILY,
            "observation_version": OBSERVATION_VERSION,
            "action_version": ACTION_VERSION,
            "reward_version": REWARD_VERSION,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "n_steps": 3,
            "policy_path": policy_path,
            "policy_loaded": self.policy_loaded,
            "learning_source": "reward_only",
            "teacher": None,
            "behaviour_cloning": False,
            "racing_line_imitation": False,
            "racing_line_observation": True,
            "racing_line_reward_shaping": True,
            "racing_line_path": str(self.racing_line_path),
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
            )
        )
        raw_action = np.clip(self._predict(observation), -1.0, 1.0)
        controls = decode_action(raw_action, telemetry)
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
            "nstep_td3_raw_longitudinal": float(self.last_raw_action[1]),
        }

    def close(self) -> None:
        return None

    def _load_policy(self, *, require_policy: bool) -> Actor | None:
        if not self.policy_path.is_file():
            if require_policy:
                raise FileNotFoundError(
                    "Agent 7 model not found. Train N-Step TD3 first: "
                    f"{self.policy_path}"
                )
            return None
        checkpoint = load_actor_checkpoint(self.policy_path, device="cpu")
        config = NstepTd3Config(**checkpoint["config"])
        if config.observation_size != OBSERVATION_SIZE:
            raise ValueError("Agent 7 model uses a different observation size")
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


def default_model_path() -> Path:
    return next(
        (path for path in DEFAULT_MODEL_CANDIDATES if path.is_file()),
        DEFAULT_MODEL_CANDIDATES[0],
    )
