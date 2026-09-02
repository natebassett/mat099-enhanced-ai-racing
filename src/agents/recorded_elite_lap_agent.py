from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from agents.sensor_n_step_td3_agent import SensorNstepTd3Agent
from n_step_td3.contracts import ACTION_SIZE, decode_action


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDED_LAP_DIR = PROJECT_ROOT / "data" / "recorded_laps"
DEFAULT_ACTIONS_PATH = RECORDED_LAP_DIR / "agent8_elite_83_620s_actions.npz"
DEFAULT_METADATA_PATH = (
    RECORDED_LAP_DIR / "agent8_elite_83_620s.json"
)
RECORDED_LAP_VERSION = "agent8_recorded_action_demo_v1"


class RecordedEliteLapAgent(SensorNstepTd3Agent):
    """Replay one self-generated Agent 8 lap as an open-loop demonstration."""

    name = "Agent 8 Recorded 83.620s Demonstration"
    agent_type = "agent8_recorded_elite_lap"
    version = "1.0"
    uses_full_control = True
    requires_racing_line = False
    target_laps = 1

    def __init__(
        self,
        *,
        actions_path: str | Path = DEFAULT_ACTIONS_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        seed: int = 0,
    ) -> None:
        self.seed = int(seed)
        self.actions_path = Path(actions_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()
        self.actions = self._load_actions()
        self.max_steps = len(self.actions) + 1
        self.reset()

    @property
    def config(self) -> dict[str, Any]:
        episode = self.metadata["episode"]
        return {
            "control_mode": "open_loop_recorded_action_playback",
            "evaluation_eligible": False,
            "neural_policy_active": False,
            "source_agent": "Agent 8 sensor-only N-step TD3",
            "source_learning": "reward_only",
            "teacher": None,
            "behaviour_cloning": False,
            "racing_line": False,
            "recorded_lap_time_seconds": float(
                episode["best_lap_time_seconds"]
            ),
            "recorded_action_count": len(self.actions),
            "actions_path": self._display_path(self.actions_path),
            "metadata_path": self._display_path(self.metadata_path),
            "automatic_gear_shift": True,
        }

    def reset(self) -> None:
        self.action_index = 0
        self.stalled_steps = 0
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
            raise ValueError(
                "RecordedEliteLapAgent requires raw TORCS telemetry"
            )
        should_stop, reason = self.should_shutdown(telemetry)
        if should_stop:
            return self._termination_controls(reason)
        if self.action_index >= len(self.actions):
            return self._termination_controls("recorded actions exhausted")

        raw_action = self.actions[self.action_index]
        self.action_index += 1
        controls = decode_action(raw_action, telemetry)
        self.last_raw_action = raw_action.copy()
        self.last_controls = dict(controls)
        return {
            **controls,
            "terminate": False,
            "termination_reason": "",
        }

    def telemetry_debug(self) -> dict[str, Any]:
        return {
            "recorded_demo": True,
            "recorded_action_index": self.action_index,
            "recorded_action_count": len(self.actions),
            "recorded_raw_steer": float(self.last_raw_action[0]),
            "recorded_raw_longitudinal": float(self.last_raw_action[1]),
        }

    def _termination_controls(self, reason: str) -> dict[str, Any]:
        return {
            **self.last_controls,
            "accel": 0.0,
            "brake": 1.0,
            "terminate": True,
            "termination_reason": reason,
        }

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                "Agent 8 recorded-lap metadata was not found: "
                f"{self.metadata_path}"
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != RECORDED_LAP_VERSION:
            raise ValueError("recorded-lap metadata version differs")
        episode = metadata.get("episode")
        if not isinstance(episode, dict):
            raise ValueError("recorded-lap metadata is missing its episode")
        transition_count = int(metadata.get("episode_transition_count", 0))
        if transition_count < 1:
            raise ValueError("recorded-lap metadata has an invalid action count")
        if int(episode.get("laps_completed", 0)) < 1:
            raise ValueError("recorded demonstration did not complete a lap")
        if int(episode.get("steps", 0)) != transition_count:
            raise ValueError(
                "recorded-lap transition count differs from episode steps"
            )
        return metadata

    def _load_actions(self) -> np.ndarray:
        if not self.actions_path.is_file():
            raise FileNotFoundError(
                "Agent 8 recorded actions were not found: "
                f"{self.actions_path}"
            )
        transition_count = int(self.metadata["episode_transition_count"])
        with np.load(self.actions_path, allow_pickle=False) as payload:
            if "actions" not in payload:
                raise ValueError("recorded-lap artifact is missing actions")
            actions = np.asarray(payload["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1:] != (ACTION_SIZE,):
            raise ValueError("recorded-lap artifact has an invalid action shape")
        if actions.shape != (transition_count, ACTION_SIZE):
            raise ValueError("Agent 8 recorded action sequence is incomplete")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Agent 8 recorded actions contain non-finite values")
        return np.clip(actions, -1.0, 1.0)

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)
