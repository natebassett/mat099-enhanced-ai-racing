from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayTransition:
    observation: np.ndarray
    action: np.ndarray
    discounted_return: float
    next_observation: np.ndarray
    terminated: bool
    steps: int


@dataclass(frozen=True)
class _PendingStep:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool


class NstepTransitionAccumulator:
    """Turn one-step interactions into terminal-aware N-step transitions."""

    def __init__(self, n_steps: int, gamma: float) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self._pending: deque[_PendingStep] = deque()

    def reset(self) -> None:
        self._pending.clear()

    def append(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminated: bool,
        episode_end: bool,
    ) -> list[ReplayTransition]:
        self._pending.append(
            _PendingStep(
                np.asarray(observation, dtype=np.float32).copy(),
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                bool(terminated),
            )
        )
        emitted: list[ReplayTransition] = []
        if len(self._pending) >= self.n_steps:
            emitted.append(self._emit_oldest(self.n_steps))
        if episode_end:
            while self._pending:
                emitted.append(self._emit_oldest(len(self._pending)))
        return emitted

    def _emit_oldest(self, steps: int) -> ReplayTransition:
        window = list(self._pending)[:steps]
        discounted_return = sum(
            item.reward * (self.gamma**index)
            for index, item in enumerate(window)
        )
        transition = ReplayTransition(
            observation=window[0].observation,
            action=window[0].action,
            discounted_return=float(discounted_return),
            next_observation=window[-1].next_observation,
            terminated=any(item.terminated for item in window),
            steps=len(window),
        )
        self._pending.popleft()
        return transition


class NstepReplayBuffer:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        capacity: int,
        *,
        seed: int,
    ) -> None:
        if observation_size < 1 or action_size < 1 or capacity < 1:
            raise ValueError("replay dimensions and capacity must be positive")
        self.observation_size = int(observation_size)
        self.action_size = int(action_size)
        self.capacity = int(capacity)
        self.observations = np.empty(
            (capacity, observation_size), dtype=np.float32
        )
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.returns = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, observation_size), dtype=np.float32
        )
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.steps = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self._random = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return self.size

    def extend(self, transitions: Iterable[ReplayTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def add(self, transition: ReplayTransition) -> None:
        index = self.position
        observation = np.asarray(transition.observation, dtype=np.float32)
        action = np.asarray(transition.action, dtype=np.float32)
        next_observation = np.asarray(
            transition.next_observation,
            dtype=np.float32,
        )
        if observation.shape != (self.observation_size,):
            raise ValueError(f"invalid replay observation shape: {observation.shape}")
        if next_observation.shape != (self.observation_size,):
            raise ValueError(
                f"invalid replay next-observation shape: {next_observation.shape}"
            )
        if action.shape != (self.action_size,):
            raise ValueError(f"invalid replay action shape: {action.shape}")

        self.observations[index] = observation
        self.actions[index] = action
        self.returns[index, 0] = float(transition.discounted_return)
        self.next_observations[index] = next_observation
        self.terminated[index, 0] = float(transition.terminated)
        self.steps[index, 0] = float(transition.steps)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        if batch_size < 1 or self.size < batch_size:
            raise ValueError(
                f"cannot sample batch {batch_size} from replay size {self.size}"
            )
        indices = self._random.integers(0, self.size, size=int(batch_size))
        arrays = (
            self.observations[indices],
            self.actions[indices],
            self.returns[indices],
            self.next_observations[indices],
            self.terminated[indices],
            self.steps[indices],
        )
        return tuple(
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in arrays
        )

    def state_dict(self) -> dict[str, object]:
        indices = self._ordered_indices()
        return {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "capacity": self.capacity,
            "size": self.size,
            "observations": self.observations[indices].copy(),
            "actions": self.actions[indices].copy(),
            "returns": self.returns[indices].copy(),
            "next_observations": self.next_observations[indices].copy(),
            "terminated": self.terminated[indices].copy(),
            "steps": self.steps[indices].copy(),
            "random_state": self._random.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError("saved replay exceeds configured capacity")
        if int(state["observation_size"]) != self.observation_size:
            raise ValueError("saved replay observation contract differs")
        if int(state["action_size"]) != self.action_size:
            raise ValueError("saved replay action contract differs")
        for target, name in (
            (self.observations, "observations"),
            (self.actions, "actions"),
            (self.returns, "returns"),
            (self.next_observations, "next_observations"),
            (self.terminated, "terminated"),
            (self.steps, "steps"),
        ):
            value = np.asarray(state[name], dtype=np.float32)
            target[:size] = value[:size]
        self.size = size
        self.position = size % self.capacity
        random_state = state.get("random_state")
        if isinstance(random_state, dict):
            self._random.bit_generator.state = random_state

    def _ordered_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size)
        return np.concatenate(
            (
                np.arange(self.position, self.capacity),
                np.arange(0, self.position),
            )
        )
