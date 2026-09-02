from __future__ import annotations

import contextlib
import math
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from runner.lap_tracker import LapTracker, practice_finish_is_plausible

if TYPE_CHECKING:
    from runner.torcs_runner import TorcsRunner

from .contracts import (
    DEFAULT_SENSOR_STABILITY_CENTERING_COEFFICIENT,
    SENSOR_STABILITY_HEADING_LIMIT_RAD,
    HistoryEncoder,
    apply_observation_noise,
    build_base_observation,
    calculate_progress_reward,
    calculate_reward,
    calculate_sensor_stability_reward,
    decode_action,
    finite_float,
    track_sensors,
)
from .racing_line import (
    LOOKAHEAD_DISTANCES_M,
    RacingLineFeatureMap,
    RacingLineSample,
    default_racing_line_path,
)


G_TRACK_3_LENGTH_METRES = 2843.0934
@contextlib.contextmanager
def _legacy_torcs_argv() -> Iterator[None]:
    """Hide trainer flags from snakeoil3's process-global CLI parser."""
    original_argv = sys.argv
    sys.argv = original_argv[:1]
    try:
        yield
    finally:
        sys.argv = original_argv


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    steps: int
    reward: float
    distance_m: float
    max_speed_kmh: float
    average_speed_kmh: float
    off_track_steps: int
    damage_delta: float
    laps_completed: int
    best_lap_time_seconds: float | None
    termination_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class NstepTorcsEnvironment:
    """Small TORCS boundary for the N-step TD3 racing policies."""

    track_length_m = G_TRACK_3_LENGTH_METRES
    target_laps = 1

    def __init__(
        self,
        *,
        track_name: str = "g-track-3",
        max_episode_steps: int = 25_000,
        manual_start: bool = False,
        reset_retries: int = 2,
        observation_noise_std: float = 0.025,
        seed: int = 0,
        racing_line_path: str | Path | None = None,
        racing_line: RacingLineFeatureMap | None = None,
        steering_rate_cost_coefficient: float = 0.0,
        physical_steering_limit: float = 1.0,
        progress_reward: bool = False,
        stability_reward: bool = False,
        stability_centering_coefficient: float = (
            DEFAULT_SENSOR_STABILITY_CENTERING_COEFFICIENT
        ),
        racing_line_features: bool = True,
        runner: TorcsRunner | None = None,
    ) -> None:
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not math.isfinite(observation_noise_std) or observation_noise_std < 0.0:
            raise ValueError("observation_noise_std must be finite and non-negative")
        if (
            not math.isfinite(steering_rate_cost_coefficient)
            or steering_rate_cost_coefficient < 0.0
        ):
            raise ValueError(
                "steering-rate cost coefficient must be finite and non-negative"
            )
        if (
            not math.isfinite(physical_steering_limit)
            or not 0.0 < physical_steering_limit <= 1.0
        ):
            raise ValueError(
                "physical steering limit must be finite and in (0, 1]"
            )
        if progress_reward and steering_rate_cost_coefficient > 0.0:
            raise ValueError(
                "progress reward cannot be combined with steering-rate shaping"
            )
        if progress_reward and stability_reward:
            raise ValueError("progress and stability rewards are mutually exclusive")
        if stability_reward and racing_line_features:
            raise ValueError("stability reward is reserved for sensor-only Agent 8")
        if (
            not math.isfinite(stability_centering_coefficient)
            or stability_centering_coefficient < 0.0
        ):
            raise ValueError(
                "stability centering coefficient must be finite and non-negative"
            )
        self.track_name = str(track_name)
        self.max_episode_steps = int(max_episode_steps)
        self.manual_start = bool(manual_start)
        self.reset_retries = max(0, int(reset_retries))
        self.observation_noise_std = float(observation_noise_std)
        self.steering_rate_cost_coefficient = float(
            steering_rate_cost_coefficient
        )
        self.physical_steering_limit = float(physical_steering_limit)
        self.progress_reward = bool(progress_reward)
        self.stability_reward = bool(stability_reward)
        self.stability_centering_coefficient = float(
            stability_centering_coefficient
        )
        self.racing_line_features = bool(racing_line_features)
        self._observation_random = np.random.default_rng(seed)
        if runner is None:
            from runner.torcs_runner import TorcsRunner

            runner = TorcsRunner()
        self.runner = runner
        if not self.racing_line_features:
            if racing_line is not None or racing_line_path is not None:
                raise ValueError(
                    "sensor-only TD3 cannot load a racing-line map"
                )
            self.racing_line_path = None
            self.racing_line = None
            self.track_length_m = G_TRACK_3_LENGTH_METRES
        elif racing_line is None:
            source = (
                Path(racing_line_path)
                if racing_line_path is not None
                else default_racing_line_path(self.track_name)
            )
            if not source.is_file():
                raise FileNotFoundError(
                    "line-aware N-step TD3 requires a racing-line file: "
                    f"{source}"
                )
            racing_line = RacingLineFeatureMap.from_json(source)
            self.racing_line_path = source
            self.racing_line = racing_line
            self.track_length_m = racing_line.track_length_m
        else:
            self.racing_line_path = (
                None if racing_line_path is None else Path(racing_line_path)
            )
            self.racing_line = racing_line
            self.track_length_m = racing_line.track_length_m
        self.history = HistoryEncoder()
        self.episode = 0
        self.current_telemetry: dict[str, Any] | None = None
        self.initial_telemetry: dict[str, Any] | None = None
        self.previous_damage = 0.0
        self.initial_damage = 0.0
        self.start_distance = 0.0
        self.furthest_distance = 0.0
        self.steps = 0
        self.reward = 0.0
        self.speed_sum = 0.0
        self.max_speed = 0.0
        self.off_track_steps = 0
        self.previous_steer = 0.0
        self.lap_tracker: LapTracker | None = None
        self.last_summary: EpisodeSummary | None = None

    def reset(self, *, force_relaunch: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        self._ensure_torcs()
        if force_relaunch and not self.manual_start:
            self._restart_torcs()
        self._reset_client()
        if self.runner.env is None:
            raise RuntimeError("TORCS environment was not connected")

        telemetry = dict(self.runner.env.client.S.d)
        self.episode += 1
        self.current_telemetry = telemetry
        self.initial_telemetry = dict(telemetry)
        self.previous_damage = finite_float(telemetry.get("damage"))
        self.initial_damage = self.previous_damage
        self.start_distance = finite_float(telemetry.get("distRaced"))
        self.furthest_distance = 0.0
        self.steps = 0
        self.reward = 0.0
        self.speed_sum = 0.0
        self.max_speed = finite_float(telemetry.get("speedX"))
        self.off_track_steps = 0
        self.previous_steer = 0.0
        self.lap_tracker = LapTracker(telemetry.get("lastLapTime", 0.0))
        self.last_summary = None
        observation = self.history.reset(self._base_observation(telemetry))
        return observation, {"episode": self.episode, "track": self.track_name}

    def set_observation_noise_std(self, value: float) -> None:
        """Select the normalized-observation noise profile for the next reset."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "observation noise level must be finite and non-negative"
            )
        self.observation_noise_std = float(value)

    def step(
        self,
        raw_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_telemetry is None or self.runner.env is None:
            raise RuntimeError("reset the N-step TD3 environment before stepping")
        if self.lap_tracker is None or self.initial_telemetry is None:
            raise RuntimeError("episode tracking was not initialised")

        action_values = np.asarray(raw_action, dtype=np.float32).reshape(-1)[:2]
        if action_values.shape != (2,):
            raise ValueError("N-step TD3 expects [steer, signed longitudinal]")
        action_values = np.clip(action_values, -1.0, 1.0)
        controls = decode_action(
            action_values,
            self.current_telemetry,
            physical_steering_limit=self.physical_steering_limit,
        )
        raw_observation, _runner_reward, server_stopped, _ = (
            self.runner._step_full_control_agent(controls)
        )
        telemetry = dict(raw_observation)
        self.steps += 1

        road = track_sensors(telemetry)
        track_position = finite_float(telemetry.get("trackPos"))
        speed = finite_float(telemetry.get("speedX"))
        angle = finite_float(telemetry.get("angle"))
        damage = finite_float(telemetry.get("damage"), self.previous_damage)
        damage_delta = damage - self.previous_damage
        off_track = bool(abs(track_position) > 1.0 or float(road.min()) < 0.0)
        crashed = bool(damage_delta > 0.0)
        backwards = bool(math.cos(angle) < 0.0)
        unsafe_heading = bool(
            self.stability_reward
            and abs(angle) > SENSOR_STABILITY_HEADING_LIMIT_RAD
        )
        stalled = bool(self.steps > 100 and speed < 5.0)

        completed_lap = self.lap_tracker.update(telemetry)
        if (
            completed_lap is None
            and server_stopped
            and practice_finish_is_plausible(
                self.initial_telemetry,
                telemetry,
                self,
            )
        ):
            completed_lap = self.lap_tracker.record(
                telemetry.get("curLapTime", 0.0)
            )

        physical_failure = (
            off_track or crashed or backwards or unsafe_heading or stalled
        )
        goal_terminal = bool(
            self.stability_reward and completed_lap is not None
        )
        terminated = bool(physical_failure or goal_terminal)
        truncated = bool(
            not goal_terminal
            and (
                completed_lap is not None
                or self.steps >= self.max_episode_steps
                or (server_stopped and not physical_failure)
            )
        )
        line_sample = (
            self.racing_line.sample(
                finite_float(telemetry.get("distFromStart"))
            )
            if self.racing_line is not None
            else RacingLineSample(
                target_track_position=0.0,
                lookahead_curvature=np.zeros(
                    LOOKAHEAD_DISTANCES_M.shape,
                    dtype=np.float32,
                ),
            )
        )
        if self.stability_reward:
            reward_result = calculate_sensor_stability_reward(
                telemetry,
                physical_failure=physical_failure,
                lap_completed=completed_lap is not None,
                previous_steer=self.previous_steer,
                current_steer=float(action_values[0]),
                centering_coefficient=(
                    self.stability_centering_coefficient
                ),
                steering_rate_cost_coefficient=(
                    self.steering_rate_cost_coefficient
                ),
            )
        elif self.progress_reward:
            reward_result = calculate_progress_reward(
                self.current_telemetry.get("distRaced"),
                telemetry.get("distRaced"),
                physical_failure=physical_failure,
                lap_completed=completed_lap is not None,
            )
        else:
            reward_result = calculate_reward(
                telemetry,
                physical_failure=physical_failure,
                target_track_position=line_sample.target_track_position,
                previous_steer=self.previous_steer,
                current_steer=float(action_values[0]),
                steering_rate_cost_coefficient=(
                    self.steering_rate_cost_coefficient
                ),
            )
        steering_delta = float(action_values[0]) - self.previous_steer

        self.history.record_action(action_values)
        self.previous_steer = float(action_values[0])
        observation = self.history.observe(
            self._base_observation(
                telemetry,
                previous_telemetry=self.current_telemetry,
            )
        )
        distance = max(
            0.0,
            finite_float(telemetry.get("distRaced")) - self.start_distance,
        )
        self.furthest_distance = max(self.furthest_distance, distance)
        self.reward += reward_result.total
        self.speed_sum += speed
        self.max_speed = max(self.max_speed, speed)
        if off_track:
            self.off_track_steps += 1
        self.previous_damage = damage
        self.current_telemetry = telemetry

        reason = ""
        if completed_lap is not None:
            reason = "lap_completed"
        elif off_track:
            reason = "off_track"
        elif crashed:
            reason = "crashed"
        elif backwards:
            reason = "backwards"
        elif unsafe_heading:
            reason = "unsafe_heading"
        elif stalled:
            reason = "stalled"
        elif server_stopped:
            reason = "torcs_stopped"
        elif self.steps >= self.max_episode_steps:
            reason = "time_limit"

        info = {
            "episode": self.episode,
            "steps": self.steps,
            "distance_m": self.furthest_distance,
            "distance_from_start_m": finite_float(
                telemetry.get("distFromStart")
            ),
            "speed_kmh": speed,
            "heading_angle_rad": angle,
            "track_position": track_position,
            "minimum_track_sensor_m": float(road.min()),
            "raw_steer": float(action_values[0]),
            "control_steer": float(controls["steer"]),
            "raw_longitudinal": float(action_values[1]),
            "control_accel": float(controls["accel"]),
            "control_brake": float(controls["brake"]),
            "control_gear": int(controls["gear"]),
            "off_track": off_track,
            "crashed": crashed,
            "backwards": backwards,
            "unsafe_heading": unsafe_heading,
            "stalled": stalled,
            "physical_failure": physical_failure,
            "laps_completed": self.lap_tracker.laps_completed,
            "best_lap_time_seconds": self.lap_tracker.best_lap_time,
            "termination_reason": reason,
            "reward_forward_velocity": reward_result.forward_velocity,
            "reward_racing_line_factor": reward_result.racing_line_factor,
            "reward_steering_rate_penalty": (
                reward_result.steering_rate_penalty
            ),
            "reward_terminal_penalty": reward_result.terminal_penalty,
            "reward_progress_m": reward_result.progress_m,
            "reward_lap_completion_bonus": (
                reward_result.lap_completion_bonus
            ),
            "reward_centering_penalty": reward_result.centering_penalty,
            "steering_delta": steering_delta,
            "target_track_position": line_sample.target_track_position,
            "racing_line_error": (
                track_position - line_sample.target_track_position
            ),
            "lookahead_curvature": line_sample.lookahead_curvature.copy(),
            "TimeLimit.truncated": bool(truncated and not terminated),
        }
        if terminated or truncated:
            self.last_summary = EpisodeSummary(
                episode=self.episode,
                steps=self.steps,
                reward=self.reward,
                distance_m=self.furthest_distance,
                max_speed_kmh=self.max_speed,
                average_speed_kmh=self.speed_sum / max(1, self.steps),
                off_track_steps=self.off_track_steps,
                damage_delta=max(0.0, damage - self.initial_damage),
                laps_completed=self.lap_tracker.laps_completed,
                best_lap_time_seconds=self.lap_tracker.best_lap_time,
                termination_reason=reason,
            )
            info["episode_summary"] = self.last_summary.as_dict()
        return observation, reward_result.total, terminated, truncated, info

    def _base_observation(
        self,
        telemetry: dict[str, Any],
        *,
        previous_telemetry: dict[str, Any] | None = None,
    ) -> np.ndarray:
        observation = build_base_observation(
            telemetry,
            previous_telemetry=previous_telemetry,
            racing_line=self.racing_line,
            include_racing_line_features=self.racing_line_features,
        )
        noisy_observation = apply_observation_noise(
            observation,
            rng=self._observation_random,
            level=self.observation_noise_std,
        )
        if not self.racing_line_features:
            masked_feature_count = 1 + LOOKAHEAD_DISTANCES_M.size
            noisy_observation[-masked_feature_count:] = 0.0
        return noisy_observation

    def close(self) -> None:
        self._close_client()
        self.runner.shutdown()

    def _ensure_torcs(self) -> None:
        if self.runner.env is not None:
            return
        self._launch_and_connect()

    def _launch_and_connect(self) -> None:
        last_error: Exception | None = None
        for attempt in range(self.reset_retries + 1):
            try:
                with _legacy_torcs_argv():
                    if not self.manual_start:
                        self.runner.launch()
                    self.runner.connect()
                    self.runner.load_track(self.track_name)
                return
            except Exception as error:
                last_error = error
                self._close_client()
                if not self.manual_start:
                    self.runner.shutdown()
                if attempt < self.reset_retries:
                    print(
                        "N-step TD3 TORCS connection failed; retrying "
                        f"[{attempt + 2}/{self.reset_retries + 1}]..."
                    )
                    time.sleep(1.0)
        raise RuntimeError("N-step TD3 could not launch/connect to TORCS") from last_error

    def _reset_client(self) -> None:
        last_error: Exception | None = None
        for attempt in range(self.reset_retries + 1):
            try:
                if self.runner.env is None:
                    self._launch_and_connect()
                assert self.runner.env is not None
                with _legacy_torcs_argv():
                    return self.runner.env.reset(relaunch=False)
            except Exception as error:
                last_error = error
                if self.manual_start:
                    self._close_client()
                else:
                    self._restart_torcs()
                if attempt < self.reset_retries:
                    time.sleep(1.0)
        raise RuntimeError("N-step TD3 could not reset TORCS") from last_error

    def _restart_torcs(self) -> None:
        self._close_client()
        self.runner.shutdown()
        self._launch_and_connect()

    def _close_client(self) -> None:
        torcs_env = self.runner.env
        if torcs_env is None:
            return
        self.runner.env = None
        client = getattr(torcs_env, "client", None)
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            with contextlib.suppress(Exception):
                shutdown()
        with contextlib.suppress(Exception):
            torcs_env.end()
