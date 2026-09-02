from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional

from .contracts import (
    ACTION_SIZE,
    ACTION_VERSION,
    MODEL_FAMILY,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    PACE_ACTION_VERSION,
    PACE_REWARD_VERSION,
    REWARD_VERSION,
    SENSOR_ONLY_MODEL_FAMILY,
    SENSOR_ONLY_OBSERVATION_VERSION,
    SENSOR_ONLY_REWARD_VERSION,
    STEERING_RATE_REWARD_VERSION,
)
from .networks import Actor, TwinCritic
from .replay import NstepReplayBuffer, ReplayTransition


CHECKPOINT_VERSION = "agent7_n_step_td3_checkpoint_v2"
FINE_TUNE_ACTOR_LEARNING_RATE = 3e-6
FINE_TUNE_EXPLORATION_NOISE = 0.01
FINE_TUNE_POLICY_DELAY = 4
ACTOR_COMPATIBLE_OBSERVATION_VERSIONS = frozenset(
    {
        "agent7_torcsrl_history_v2",
        OBSERVATION_VERSION,
    }
)


@dataclass(frozen=True)
class NstepTd3Config:
    observation_size: int = OBSERVATION_SIZE
    action_size: int = ACTION_SIZE
    hidden_size_1: int = 256
    hidden_size_2: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    n_steps: int = 3
    gamma: float = 0.992
    tau: float = 0.005
    replay_capacity: int = 1_000_000
    replay_start_size: int = 10_000
    batch_size: int = 256
    exploration_noise: float = 0.1
    longitudinal_exploration_noise: float = 0.1
    target_policy_noise: float = 0.5
    target_noise_clip: float = 0.2
    policy_delay: int = 2
    gradient_steps_per_interaction: int = 1
    steering_rate_cost_coefficient: float = 0.0
    physical_steering_limit: float = 1.0
    progress_reward: bool = False
    racing_line_features: bool = True
    seed: int = 0

    def validate(self) -> None:
        if self.observation_size < 1 or self.action_size < 1:
            raise ValueError("network dimensions must be positive")
        if self.hidden_size_1 < 1 or self.hidden_size_2 < 1:
            raise ValueError("hidden dimensions must be positive")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.n_steps < 1 or self.policy_delay < 1:
            raise ValueError("n_steps and policy_delay must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")
        if self.replay_capacity < self.replay_start_size:
            raise ValueError("replay capacity must cover replay start size")
        if self.replay_start_size < self.batch_size:
            raise ValueError("replay start size must cover one batch")
        if self.batch_size < 1 or self.gradient_steps_per_interaction < 1:
            raise ValueError("batch and gradient-step counts must be positive")
        if min(
            self.exploration_noise,
            self.longitudinal_exploration_noise,
            self.target_policy_noise,
            self.target_noise_clip,
        ) < 0.0:
            raise ValueError("noise values must be non-negative")
        if (
            not math.isfinite(self.steering_rate_cost_coefficient)
            or self.steering_rate_cost_coefficient < 0.0
        ):
            raise ValueError(
                "steering-rate cost coefficient must be finite and non-negative"
            )
        if (
            not math.isfinite(self.physical_steering_limit)
            or not 0.0 < self.physical_steering_limit <= 1.0
        ):
            raise ValueError(
                "physical steering limit must be finite and in (0, 1]"
            )
        if self.progress_reward and self.steering_rate_cost_coefficient > 0.0:
            raise ValueError(
                "progress reward cannot be combined with steering-rate shaping"
            )
        if self.progress_reward and self.physical_steering_limit >= 1.0:
            raise ValueError("progress reward requires bounded physical steering")
        if not self.racing_line_features and self.progress_reward:
            raise ValueError(
                "sensor-only training uses the centreline velocity reward"
            )
        if (
            not self.racing_line_features
            and self.steering_rate_cost_coefficient > 0.0
        ):
            raise ValueError(
                "sensor-only training does not use racing-line fine-tuning modes"
            )


def model_family_for_config(config: NstepTd3Config) -> str:
    return MODEL_FAMILY if config.racing_line_features else SENSOR_ONLY_MODEL_FAMILY


def observation_version_for_config(config: NstepTd3Config) -> str:
    return (
        OBSERVATION_VERSION
        if config.racing_line_features
        else SENSOR_ONLY_OBSERVATION_VERSION
    )


def action_version_for_config(config: NstepTd3Config) -> str:
    return (
        PACE_ACTION_VERSION
        if config.physical_steering_limit < 1.0
        else ACTION_VERSION
    )


def reward_version_for_config(config: NstepTd3Config) -> str:
    if not config.racing_line_features:
        return SENSOR_ONLY_REWARD_VERSION
    if config.progress_reward:
        return PACE_REWARD_VERSION
    if config.steering_rate_cost_coefficient > 0.0:
        return STEERING_RATE_REWARD_VERSION
    return REWARD_VERSION


@dataclass(frozen=True)
class UpdateMetrics:
    critic_loss: float
    actor_loss: float | None
    q1_mean: float
    q2_mean: float
    target_q_mean: float


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class NstepTd3Learner:
    """Compact N-step TD3 learner with no policy gates or rollback logic."""

    def __init__(
        self,
        config: NstepTd3Config,
        *,
        device: str | torch.device = "auto",
    ) -> None:
        config.validate()
        self.config = config
        self.model_family = model_family_for_config(config)
        self.observation_version = observation_version_for_config(config)
        self.action_version = action_version_for_config(config)
        self.reward_version = reward_version_for_config(config)
        self.device = (
            resolve_device(device) if isinstance(device, str) else device
        )
        self._seed_everything(config.seed)
        hidden_sizes = (config.hidden_size_1, config.hidden_size_2)
        self.actor = Actor(
            config.observation_size,
            config.action_size,
            hidden_sizes,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = TwinCritic(
            config.observation_size,
            config.action_size,
            hidden_sizes,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for module in (self.actor_target, self.critic_target):
            module.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_learning_rate,
        )
        self.replay = NstepReplayBuffer(
            config.observation_size,
            config.action_size,
            config.replay_capacity,
            seed=config.seed + 1,
        )
        self.environment_steps = 0
        self.optimizer_steps = 0
        self.actor_updates = 0
        self.episodes = 0
        self.best_training_distance_m = 0.0
        self.best_evaluation_distance_m = 0.0
        self.best_evaluation_median_m: float | None = None
        self.best_evaluation_completion_rate: float | None = None
        self.best_evaluation_median_lap_time_seconds: float | None = None
        self.best_evaluation_mean_off_track_steps: float | None = None
        self.best_evaluation_mean_damage_delta: float | None = None
        self.best_lap_time_seconds: float | None = None
        self._action_random = np.random.default_rng(config.seed + 2)
        self._exploration_noise_scale = np.full(
            config.action_size,
            config.exploration_noise,
            dtype=np.float32,
        )
        if config.action_size > 1:
            self._exploration_noise_scale[1] = (
                config.longitudinal_exploration_noise
            )
        generator_device = self.device.type if self.device.type != "mps" else "cpu"
        self._torch_random = torch.Generator(device=generator_device)
        self._torch_random.manual_seed(config.seed + 3)

    @property
    def ready_to_train(self) -> bool:
        return len(self.replay) >= self.config.replay_start_size

    def reseed_training_noise(self, seed: int) -> None:
        """Start independent action and target-noise streams after a resume."""
        value = int(seed)
        if value < 0:
            raise ValueError("training noise seed must be non-negative")
        self._action_random = np.random.default_rng(value + 2)
        self._torch_random.manual_seed(value + 3)

    def enable_fine_tuning(
        self,
        *,
        actor_learning_rate: float = FINE_TUNE_ACTOR_LEARNING_RATE,
    ) -> None:
        """Use conservative actor updates and exploration for pace refinement."""
        learning_rate = float(actor_learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "fine-tuning actor learning rate must be positive and finite"
            )
        self.config = replace(
            self.config,
            actor_learning_rate=learning_rate,
            exploration_noise=FINE_TUNE_EXPLORATION_NOISE,
            longitudinal_exploration_noise=FINE_TUNE_EXPLORATION_NOISE,
            policy_delay=FINE_TUNE_POLICY_DELAY,
        )
        self.config.validate()
        for parameter_group in self.actor_optimizer.param_groups:
            parameter_group["lr"] = self.config.actor_learning_rate
        self._exploration_noise_scale.fill(self.config.exploration_noise)
        if self.config.action_size > 1:
            self._exploration_noise_scale[1] = (
                self.config.longitudinal_exploration_noise
            )

    def random_action(self) -> np.ndarray:
        return self._action_random.uniform(
            -1.0,
            1.0,
            size=self.config.action_size,
        ).astype(np.float32)

    @torch.no_grad()
    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> np.ndarray:
        value = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).reshape(1, -1)
        action = self.actor(value).squeeze(0).cpu().numpy()
        if not deterministic:
            action = action + self._action_random.normal(
                0.0,
                self._exploration_noise_scale,
                size=self.config.action_size,
            )
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def add_transitions(self, transitions: list[ReplayTransition]) -> None:
        self.replay.extend(transitions)

    def train_for_interaction(
        self,
        *,
        allow_actor_update: bool = True,
    ) -> list[UpdateMetrics]:
        if not self.ready_to_train:
            return []
        return [
            self._gradient_step(allow_actor_update=allow_actor_update)
            for _ in range(self.config.gradient_steps_per_interaction)
        ]

    def train_critic_only(self, gradient_steps: int) -> list[UpdateMetrics]:
        """Adapt the critic to fresh replay without changing the actor."""
        steps = int(gradient_steps)
        if steps < 0:
            raise ValueError("critic-only gradient steps must be non-negative")
        if steps > 0 and not self.ready_to_train:
            raise RuntimeError("critic-only training requires a ready replay buffer")
        return [self._gradient_step(allow_actor_update=False) for _ in range(steps)]

    def _gradient_step(
        self,
        *,
        allow_actor_update: bool = True,
    ) -> UpdateMetrics:
        (
            observations,
            actions,
            returns,
            next_observations,
            terminated,
            n_steps,
        ) = self.replay.sample(self.config.batch_size, self.device)

        with torch.no_grad():
            target_noise = self._normal_noise(actions.shape)
            target_noise = target_noise.mul(self.config.target_policy_noise).clamp(
                -self.config.target_noise_clip,
                self.config.target_noise_clip,
            )
            next_actions = (self.actor_target(next_observations) + target_noise).clamp(
                -1.0,
                1.0,
            )
            target_q1, target_q2 = self.critic_target(
                next_observations,
                next_actions,
            )
            target_q = returns + (
                (1.0 - terminated)
                * torch.pow(self.config.gamma, n_steps)
                * torch.minimum(target_q1, target_q2)
            )

        q1, q2 = self.critic(observations, actions)
        critic_loss = functional.mse_loss(q1, target_q) + functional.mse_loss(
            q2,
            target_q,
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.optimizer_steps += 1
        actor_loss_value: float | None = None
        delayed_update = self.optimizer_steps % self.config.policy_delay == 0
        if allow_actor_update and delayed_update:
            self.critic.requires_grad_(False)
            actor_loss = -self.critic.first(
                observations,
                self.actor(observations),
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            self.actor_updates += 1
            actor_loss_value = float(actor_loss.detach().item())
            self._soft_update(self.actor, self.actor_target)
        if delayed_update:
            self._soft_update(self.critic, self.critic_target)

        return UpdateMetrics(
            critic_loss=float(critic_loss.detach().item()),
            actor_loss=actor_loss_value,
            q1_mean=float(q1.detach().mean().item()),
            q2_mean=float(q2.detach().mean().item()),
            target_q_mean=float(target_q.detach().mean().item()),
        )

    @torch.no_grad()
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for source_parameter, target_parameter in zip(
            source.parameters(),
            target.parameters(),
        ):
            target_parameter.lerp_(source_parameter, self.config.tau)

    def _normal_noise(self, shape: torch.Size) -> torch.Tensor:
        if self.device.type == "mps":
            return torch.randn(shape, generator=self._torch_random).to(self.device)
        return torch.randn(
            shape,
            generator=self._torch_random,
            device=self.device,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_family": self.model_family,
            "observation_version": self.observation_version,
            "action_version": self.action_version,
            "reward_version": self.reward_version,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "optimizer_steps": self.optimizer_steps,
            "actor_updates": self.actor_updates,
            "episodes": self.episodes,
            "best_training_distance_m": self.best_training_distance_m,
            "best_evaluation_distance_m": self.best_evaluation_distance_m,
            "best_evaluation_median_m": self.best_evaluation_median_m,
            "best_evaluation_completion_rate": (
                self.best_evaluation_completion_rate
            ),
            "best_evaluation_median_lap_time_seconds": (
                self.best_evaluation_median_lap_time_seconds
            ),
            "best_evaluation_mean_off_track_steps": (
                self.best_evaluation_mean_off_track_steps
            ),
            "best_evaluation_mean_damage_delta": (
                self.best_evaluation_mean_damage_delta
            ),
            "best_lap_time_seconds": self.best_lap_time_seconds,
            "numpy_action_rng": self._action_random.bit_generator.state,
            "torch_rng": self._torch_random.get_state(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        torch.save(self.checkpoint(), temporary)
        os.replace(temporary, target)
        return target

    def save_replay(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        torch.save(self.replay.state_dict(), temporary)
        os.replace(temporary, target)
        return target

    def save_network_components(
        self,
        directory: str | Path,
        *,
        step: int | None = None,
    ) -> tuple[Path, Path]:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_step = self.environment_steps if step is None else int(step)
        actor_path = target_dir / f"actor-n_step_td3-{checkpoint_step}.pt"
        critic_path = target_dir / f"critic-n_step_td3-{checkpoint_step}.pt"
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        return actor_path, critic_path

    def load_replay(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError("invalid Agent 7 replay checkpoint")
        self.replay.load_state_dict(state)

    @classmethod
    def from_actor_checkpoint(
        cls,
        path: str | Path,
        *,
        config: NstepTd3Config,
        device: str | torch.device = "auto",
    ) -> "NstepTd3Learner":
        """Initialize a fresh learner from actor parameters only."""
        checkpoint = load_actor_checkpoint(path, device="cpu")
        source_config = NstepTd3Config(**checkpoint["config"])
        if checkpoint.get("model_family") != model_family_for_config(config):
            raise ValueError("actor transfer requires the same model family")
        if (
            checkpoint.get("observation_version")
            != observation_version_for_config(config)
        ):
            raise ValueError("actor transfer requires the same observation contract")
        if checkpoint.get("action_version") != action_version_for_config(config):
            raise ValueError("actor transfer requires the same action contract")
        source_architecture = (
            source_config.observation_size,
            source_config.action_size,
            source_config.hidden_size_1,
            source_config.hidden_size_2,
        )
        target_architecture = (
            config.observation_size,
            config.action_size,
            config.hidden_size_1,
            config.hidden_size_2,
        )
        if source_architecture != target_architecture:
            raise ValueError(
                "actor transfer requires an identical network architecture"
            )

        learner = cls(config, device=device)
        learner.actor.load_state_dict(checkpoint["actor"])
        learner.actor_target.load_state_dict(checkpoint["actor"])
        return learner

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "auto",
        load_optimizers: bool = True,
    ) -> "NstepTd3Learner":
        checkpoint = load_checkpoint(path, device="cpu")
        config = NstepTd3Config(**checkpoint["config"])
        learner = cls(config, device=device)
        learner.actor.load_state_dict(checkpoint["actor"])
        learner.actor_target.load_state_dict(checkpoint["actor_target"])
        learner.critic.load_state_dict(checkpoint["critic"])
        learner.critic_target.load_state_dict(checkpoint["critic_target"])
        if load_optimizers:
            learner.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer"]
            )
            learner.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer"]
            )
        learner.environment_steps = int(checkpoint.get("environment_steps", 0))
        learner.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        learner.actor_updates = int(checkpoint.get("actor_updates", 0))
        learner.episodes = int(checkpoint.get("episodes", 0))
        learner.best_training_distance_m = float(
            checkpoint.get("best_training_distance_m", 0.0)
        )
        learner.best_evaluation_distance_m = float(
            checkpoint.get("best_evaluation_distance_m", 0.0)
        )
        evaluation_median = checkpoint.get("best_evaluation_median_m")
        learner.best_evaluation_median_m = (
            None if evaluation_median is None else float(evaluation_median)
        )
        completion_rate = checkpoint.get("best_evaluation_completion_rate")
        learner.best_evaluation_completion_rate = (
            None if completion_rate is None else float(completion_rate)
        )
        median_lap_time = checkpoint.get(
            "best_evaluation_median_lap_time_seconds"
        )
        learner.best_evaluation_median_lap_time_seconds = (
            None if median_lap_time is None else float(median_lap_time)
        )
        mean_off_track = checkpoint.get("best_evaluation_mean_off_track_steps")
        learner.best_evaluation_mean_off_track_steps = (
            None if mean_off_track is None else float(mean_off_track)
        )
        mean_damage = checkpoint.get("best_evaluation_mean_damage_delta")
        learner.best_evaluation_mean_damage_delta = (
            None if mean_damage is None else float(mean_damage)
        )
        lap_time = checkpoint.get("best_lap_time_seconds")
        learner.best_lap_time_seconds = (
            None if lap_time is None else float(lap_time)
        )
        numpy_state = checkpoint.get("numpy_action_rng")
        if isinstance(numpy_state, dict):
            learner._action_random.bit_generator.state = numpy_state
        torch_state = checkpoint.get("torch_rng")
        if isinstance(torch_state, torch.Tensor):
            learner._torch_random.set_state(torch_state.cpu())
        return learner

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("invalid Agent 7 checkpoint payload")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Agent 7 checkpoint is missing its configuration")
    config = NstepTd3Config(**raw_config)
    config.validate()
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_family": model_family_for_config(config),
        "observation_version": observation_version_for_config(config),
        "action_version": action_version_for_config(config),
        "reward_version": reward_version_for_config(config),
    }
    mismatches = [
        name for name, value in expected.items() if checkpoint.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "Agent 7 checkpoint contract mismatch: " + ", ".join(mismatches)
        )
    return checkpoint


def load_actor_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load an inference-compatible actor without accepting it for training."""
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("invalid Agent 7 checkpoint payload")

    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Agent 7 checkpoint is missing its configuration")
    config = NstepTd3Config(**raw_config)
    config.validate()

    mismatches: list[str] = []
    if checkpoint.get("model_family") != model_family_for_config(config):
        mismatches.append("model_family")
    if checkpoint.get("action_version") != action_version_for_config(config):
        mismatches.append("action_version")
    compatible_observation_versions = (
        ACTOR_COMPATIBLE_OBSERVATION_VERSIONS
        if config.racing_line_features
        else frozenset({SENSOR_ONLY_OBSERVATION_VERSION})
    )
    if checkpoint.get("observation_version") not in compatible_observation_versions:
        mismatches.append("observation_version")
    if mismatches:
        raise ValueError(
            "Agent 7 actor contract mismatch: " + ", ".join(mismatches)
        )

    if int(raw_config.get("observation_size", -1)) != OBSERVATION_SIZE:
        raise ValueError("Agent 7 actor uses a different observation size")
    if int(raw_config.get("action_size", -1)) != ACTION_SIZE:
        raise ValueError("Agent 7 actor uses a different action size")
    if not isinstance(checkpoint.get("actor"), Mapping):
        raise ValueError("Agent 7 checkpoint is missing actor parameters")
    return checkpoint
