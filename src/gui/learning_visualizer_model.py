from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class LearningStep:
    title: str
    explanation: str
    active_group: str
    equation: str


@dataclass(frozen=True)
class LearningVisualizationProfile:
    kind: str
    title: str
    status: str
    input_labels: tuple[str, ...]
    output_labels: tuple[str, ...]
    steps: tuple[LearningStep, ...]


@dataclass(frozen=True)
class LayerStatistics:
    name: str
    shape: tuple[int, ...]
    parameter_count: int
    mean_absolute_weight: float
    standard_deviation: float
    maximum_absolute_weight: float


@dataclass(frozen=True)
class CheckpointStatistics:
    path: Path
    model_family: str
    parameter_count: int
    tensor_count: int
    mean_absolute_weight: float
    standard_deviation: float
    maximum_absolute_weight: float
    environment_steps: int | None
    actor_updates: int | None
    layers: tuple[LayerStatistics, ...]


def build_learning_visualization_profile(
    agent_type: str | None,
) -> LearningVisualizationProfile:
    if agent_type in {"td3_scratch", "n_step_td3", "sensor_n_step_td3"}:
        return _td3_profile(agent_type)
    if agent_type in {"dyna_q", "dyna_q_learning", "dyna_q_finalised"}:
        return _dyna_q_profile(agent_type)
    return _non_learning_profile(agent_type)


def default_checkpoint_for_agent(agent_type: str | None) -> Path | None:
    if agent_type == "n_step_td3":
        from agents.n_step_td3_agent import NstepTd3Agent

        return _existing_path(NstepTd3Agent.default_model_candidates)
    if agent_type == "sensor_n_step_td3":
        from agents.sensor_n_step_td3_agent import SensorNstepTd3Agent

        return _existing_path(SensorNstepTd3Agent.default_model_candidates)
    if agent_type == "td3_scratch":
        from agents.td3_agent import (
            DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH,
            DEFAULT_BEST_DISTANCE_MODEL_PATH,
            DEFAULT_BEST_EVALUATION_MODEL_PATH,
            DEFAULT_BEST_REWARD_MODEL_PATH,
            DEFAULT_MODEL_PATH,
        )

        return _existing_path(
            (
                DEFAULT_BEST_EVALUATION_MODEL_PATH,
                DEFAULT_BEST_COMPLETED_LAP_MODEL_PATH,
                DEFAULT_BEST_DISTANCE_MODEL_PATH,
                DEFAULT_BEST_REWARD_MODEL_PATH,
                DEFAULT_MODEL_PATH,
            )
        )
    return None


def inspect_td3_checkpoint(path: str | Path) -> CheckpointStatistics:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TD3 checkpoint not found: {checkpoint_path}")

    payload = _load_checkpoint_payload(checkpoint_path)
    actor_state, metadata = _actor_state_and_metadata(payload)
    tensors = [
        (name, value.detach().cpu())
        for name, value in actor_state.items()
        if _is_tensor(value) and value.numel() > 0
    ]
    if not tensors:
        raise ValueError("TD3 checkpoint does not contain actor parameters")

    parameter_values = np.concatenate(
        [value.numpy().astype(np.float64, copy=False).reshape(-1) for _, value in tensors]
    )
    if not np.all(np.isfinite(parameter_values)):
        raise ValueError("TD3 checkpoint actor contains non-finite parameters")

    weight_tensors = [
        (name, value)
        for name, value in tensors
        if name.endswith("weight") and value.ndim >= 2
    ]
    if not weight_tensors:
        weight_tensors = tensors
    weight_values = np.concatenate(
        [value.numpy().astype(np.float64, copy=False).reshape(-1) for _, value in weight_tensors]
    )
    layers = tuple(
        _layer_statistics(name, value)
        for name, value in weight_tensors
    )
    model_family = str(metadata.get("model_family") or _model_family_from_path(checkpoint_path))
    return CheckpointStatistics(
        path=checkpoint_path,
        model_family=model_family,
        parameter_count=int(parameter_values.size),
        tensor_count=len(tensors),
        mean_absolute_weight=float(np.mean(np.abs(weight_values))),
        standard_deviation=float(np.std(weight_values)),
        maximum_absolute_weight=float(np.max(np.abs(weight_values))),
        environment_steps=_optional_int(metadata.get("environment_steps")),
        actor_updates=_optional_int(metadata.get("actor_updates")),
        layers=layers,
    )


def _td3_profile(agent_type: str) -> LearningVisualizationProfile:
    if agent_type == "sensor_n_step_td3":
        title = "Sensor-only N-step TD3 update"
        inputs = ("Motion", "Road sensors", "Wheel spin", "History", "Track position")
    elif agent_type == "n_step_td3":
        title = "Racing-line-informed N-step TD3 update"
        inputs = ("Motion", "Road sensors", "Wheel spin", "History", "Line geometry")
    else:
        title = "Scratch TD3 update"
        inputs = ("Motion", "Road sensors", "Wheel spin", "Previous action", "Lap phase")
    return LearningVisualizationProfile(
        kind="td3",
        title=title,
        status="Conceptual update with actual saved actor measurements",
        input_labels=inputs,
        output_labels=("Steering", "Throttle / brake"),
        steps=(
            LearningStep(
                "Sample experience",
                "A replay batch supplies states, actions, rewards, and later states.",
                "replay",
                "(s, a, r, s') from replay",
            ),
            LearningStep(
                "Actor predicts controls",
                "The actor converts each state into continuous steering and pedal intent.",
                "actor_forward",
                "a = mu(s)",
            ),
            LearningStep(
                "Twin critics estimate value",
                "Two critics independently score how useful the sampled controls are.",
                "critics",
                "Q1(s, a), Q2(s, a)",
            ),
            LearningStep(
                "Build the TD target",
                "Observed reward is combined with the smaller smoothed target value.",
                "target",
                "y = R(n) + gamma^n min(Q1', Q2')",
            ),
            LearningStep(
                "Measure prediction error",
                "The critic loss measures the gap between predicted value and the TD target.",
                "loss",
                "Lcritic = (Q1-y)^2 + (Q2-y)^2",
            ),
            LearningStep(
                "Backpropagate gradients",
                "Gradients move backward through active networks and identify useful parameter changes.",
                "backward",
                "gradient = dL / dweight",
            ),
            LearningStep(
                "Apply a small update",
                "Critic weights change every update; the actor changes less often and targets follow slowly.",
                "update",
                "weight <- weight - learning_rate * gradient",
            ),
        ),
    )


def _dyna_q_profile(agent_type: str) -> LearningVisualizationProfile:
    finalised = agent_type == "dyna_q_finalised"
    status = (
        "Saved Q-table policy: learning disabled"
        if finalised
        else "Conceptual live Q-table and planning update"
    )
    return LearningVisualizationProfile(
        kind="dyna_q",
        title="Dyna-Q value update",
        status=status,
        input_labels=("Lap section", "Speed", "Road position", "Heading", "Sensor danger"),
        output_labels=("Discrete action",),
        steps=(
            LearningStep(
                "Encode the state",
                "Continuous telemetry is grouped into a compact table key.",
                "state",
                "telemetry -> state bins",
            ),
            LearningStep(
                "Choose an action",
                "Learning mode sometimes explores; finalised mode chooses the highest value.",
                "choice",
                "a = epsilon-greedy(Q(s, a))",
            ),
            LearningStep(
                "Observe reward",
                "The next TORCS frame reports progress, stability, and failures.",
                "reward",
                "transition = (s, a, r, s')",
            ),
            LearningStep(
                "Calculate TD error",
                "The new target is compared with the stored state-action value.",
                "td_error",
                "delta = r + gamma max Q(s', a') - Q(s, a)",
            ),
            LearningStep(
                "Update one table cell",
                "Only the chosen state-action value moves toward the new target.",
                "q_update",
                "Q(s, a) <- Q(s, a) + alpha * delta",
            ),
            LearningStep(
                "Replay learned transitions",
                "Dyna-Q samples its model and repeats the same update without another race frame.",
                "planning",
                "model(s, a) -> (r, s')",
            ),
        ),
    )


def _non_learning_profile(agent_type: str | None) -> LearningVisualizationProfile:
    if agent_type == "map_aware":
        title = "Map-aware decision path"
        middle = "Racing-line planner"
        status = "Engineered controller: no neural weights are trained"
    elif agent_type == "rule_based":
        title = "Rule-based decision path"
        middle = "Rules and safety limits"
        status = "Engineered controller: no neural weights are trained"
    elif agent_type == "random":
        title = "Random baseline path"
        middle = "Bounded random sample"
        status = "Baseline driver: no learning state is retained"
    else:
        title = "Agent decision path"
        middle = "Agent logic"
        status = "No learning visualisation is defined for this agent"
    return LearningVisualizationProfile(
        kind="static",
        title=title,
        status=status,
        input_labels=("TORCS telemetry",),
        output_labels=("Driving controls",),
        steps=(
            LearningStep("Observe", "Read the current simulator state.", "observe", "telemetry"),
            LearningStep("Decide", f"Apply {middle.lower()} to the observation.", "decide", middle),
            LearningStep("Act", "Send steering and pedal controls to TORCS.", "act", "controls"),
            LearningStep(
                "No weight update",
                "This agent does not train a neural network or update a Q-table.",
                "no_update",
                "parameters remain unchanged",
            ),
        ),
    )


def _existing_path(paths: tuple[Path, ...]) -> Path | None:
    return next((Path(path) for path in paths if Path(path).is_file()), None)


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    import torch

    if path.suffix.casefold() == ".zip":
        with ZipFile(path) as archive:
            try:
                policy_bytes = archive.read("policy.pth")
            except KeyError as exc:
                raise ValueError("SB3 checkpoint is missing policy.pth") from exc
        payload = torch.load(
            BytesIO(policy_bytes),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("SB3 policy payload is invalid")
        return payload

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("TD3 checkpoint payload is invalid")
    return payload


def _actor_state_and_metadata(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    actor = payload.get("actor")
    if isinstance(actor, Mapping):
        return actor, payload

    sb3_actor = {
        name.removeprefix("actor."): value
        for name, value in payload.items()
        if name.startswith("actor.") and not name.startswith("actor_target.")
    }
    if not sb3_actor:
        raise ValueError("checkpoint does not contain a recognised TD3 actor")
    return sb3_actor, {"model_family": "stable_baselines3_td3"}


def _layer_statistics(name: str, value: Any) -> LayerStatistics:
    values = value.numpy().astype(np.float64, copy=False)
    return LayerStatistics(
        name=name,
        shape=tuple(int(size) for size in value.shape),
        parameter_count=int(value.numel()),
        mean_absolute_weight=float(np.mean(np.abs(values))),
        standard_deviation=float(np.std(values)),
        maximum_absolute_weight=float(np.max(np.abs(values))),
    )


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _model_family_from_path(path: Path) -> str:
    return "TD3 actor"
