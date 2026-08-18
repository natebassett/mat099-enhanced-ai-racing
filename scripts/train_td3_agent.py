from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ORIGINAL_COMMAND = " ".join(sys.argv)

from agents.td3_agent import (  # noqa: E402
    AGENT6_ACTION_VERSION,
    AGENT6_MODEL_FAMILY,
    AGENT6_OBSERVATION_VERSION,
    DEFAULT_BEST_DISTANCE_MODEL_PATH,
    DEFAULT_BEST_REWARD_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    G_TRACK_3_LENGTH_METRES,
    build_td3_observation,
    clamp,
    decode_td3_action,
    episode_metadata_is_policy_controlled,
    finite_float,
    metadata_path_for_policy,
    rate_limit_td3_action,
    read_policy_metadata,
    shift_gears,
    track_sensors,
)
from runner.lap_tracker import LapTracker, practice_finish_is_plausible  # noqa: E402


DEFAULT_TRACK_NAME = "g-track-3"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "models" / "training_runs" / "agent6_td3_scratch"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "agent6_td3_scratch"
DEFAULT_TENSORBOARD_DIR = PROJECT_ROOT / "models" / "tensorboard" / "agent6_td3_scratch"
DEFAULT_REPLAY_BUFFER_PATH = PROJECT_ROOT / "models" / "replay_buffers" / "agent6_td3_scratch.pkl"
DEFAULT_CHECKPOINT_FREQ = 100_000
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0

REWARD_VERSION = "agent6_td3_reward_v1_raw_progress_curriculum"
OFF_TRACK_GRACE_STEPS = 4
HARD_TRACK_BOUNDARY = 1.18
SOFT_TRACK_BOUNDARY = 1.04
STUCK_SPEED_LIMIT_KMH = 5.0
STUCK_PROGRESS_LIMIT_M = 0.12
STUCK_SECONDS_LIMIT = 3.0
DEFAULT_STAGE_SUCCESS_WINDOW_SIZE = 10
DEFAULT_STAGE_SUCCESS_REQUIREMENTS = {
    "launch": 3,
    "first_corner": 2,
    "sector_progression": 1,
    "full_lap": 1,
}


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    name: str
    objective: str
    distance_target_m: float
    max_episode_steps: int
    progress_weight: float
    milestone_weight: float
    success_reward: float
    failure_penalty: float
    safe_front_speed_kmh: float


CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        "launch",
        "Launch and Stay On Track",
        "Learn throttle, basic steering, and stable forward motion.",
        140.0,
        2500,
        0.85,
        1.40,
        180.0,
        150.0,
        72.0,
    ),
    CurriculumStage(
        "first_corner",
        "First Corner Survival",
        "Reach and survive the first braking/turn-in sequence.",
        520.0,
        6500,
        0.95,
        1.65,
        260.0,
        220.0,
        92.0,
    ),
    CurriculumStage(
        "sector_progression",
        "Sector Progression",
        "Push the furthest distance while staying inside the circuit.",
        1750.0,
        11000,
        1.05,
        1.95,
        380.0,
        280.0,
        112.0,
    ),
    CurriculumStage(
        "full_lap",
        "Full Lap Attempt",
        "Train for complete-lap behaviour with a large finish reward.",
        G_TRACK_3_LENGTH_METRES,
        16000,
        1.15,
        2.20,
        850.0,
        340.0,
        130.0,
    ),
)

EPISODE_COLUMNS = [
    "episodes_seen",
    "global_timestep",
    "policy_controlled",
    "stage_id",
    "stage_name",
    "steps",
    "reward",
    "distance_m",
    "furthest_distance_m",
    "duration_seconds",
    "average_speed_kmh",
    "max_speed_kmh",
    "laps_completed",
    "best_lap_time_seconds",
    "lap_completion_fraction",
    "off_track_steps",
    "max_stopped_seconds",
    "stage_success_streak",
    "stage_required_successes",
    "stage_recent_successes",
    "stage_success_window_size",
    "stage_success_total",
    "termination_reason",
    "final_speed_kmh",
    "final_track_pos",
    "final_angle",
    "front_sensor",
    "min_track_sensor",
]

STEP_COLUMNS = [
    "episode",
    "stage_id",
    "episode_step",
    "global_step",
    "dist_from_start",
    "dist_raced",
    "speed_x",
    "speed_y",
    "angle",
    "track_pos",
    "damage",
    "front_sensor",
    "min_track_sensor",
    "raw_steer",
    "raw_accel",
    "raw_brake",
    "steer",
    "accel",
    "brake",
    "gear",
    "reward",
    "terminated",
    "truncated",
    "termination_reason",
    "cur_lap_time",
]


def import_training_dependencies():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import TD3
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.noise import NormalActionNoise
    except ImportError as exc:
        raise SystemExit(
            "Agent 6 TD3 training needs extra packages:\n"
            "  pip install stable-baselines3 gymnasium\n\n"
            "The normal race menu can still run the other agents without these packages."
        ) from exc

    return (
        gym,
        spaces,
        TD3,
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        Monitor,
        check_env,
        NormalActionNoise,
    )


def make_torcs_runner():
    with contextlib.redirect_stderr(io.StringIO()):
        from runner.torcs_runner import TorcsRunner

    return TorcsRunner()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS_DIR / timestamp


def calculate_progress_delta(
    previous_telemetry: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> float:
    if previous_telemetry is None:
        return 0.0

    previous_raced = previous_telemetry.get("distRaced")
    current_raced = telemetry.get("distRaced")
    if previous_raced is not None and current_raced is not None:
        delta = finite_float(current_raced) - finite_float(previous_raced)
        return clamp(delta, -8.0, 22.0)

    previous_distance = finite_float(previous_telemetry.get("distFromStart"))
    current_distance = finite_float(telemetry.get("distFromStart"))
    delta = current_distance - previous_distance
    if delta < -track_length_m * 0.5:
        delta += track_length_m
    elif delta > track_length_m * 0.5:
        delta -= track_length_m
    return clamp(delta, -8.0, 22.0)


def calculate_step_time_delta(
    previous_telemetry: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
) -> float | None:
    if previous_telemetry is None:
        return None

    previous_time = finite_float(previous_telemetry.get("curLapTime"))
    current_time = finite_float(telemetry.get("curLapTime"))
    delta = current_time - previous_time
    if 0.0 < delta <= 1.0:
        return delta
    return None


def calculate_stopped_time_delta(
    previous_telemetry: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any],
) -> float:
    speed = abs(finite_float(telemetry.get("speedX")))
    if speed > STUCK_SPEED_LIMIT_KMH:
        return 0.0

    progress = calculate_progress_delta(previous_telemetry, telemetry)
    if abs(progress) > STUCK_PROGRESS_LIMIT_M:
        return 0.0

    delta_time = calculate_step_time_delta(previous_telemetry, telemetry)
    return 0.02 if delta_time is None else delta_time


def calculate_lap_completion_fraction(distance_m: float) -> float:
    return clamp(max(0.0, distance_m) / G_TRACK_3_LENGTH_METRES, 0.0, 1.0)


def required_successes_for_stage(stage: CurriculumStage | str) -> int:
    stage_id = stage.stage_id if isinstance(stage, CurriculumStage) else str(stage)
    return max(1, int(DEFAULT_STAGE_SUCCESS_REQUIREMENTS.get(stage_id, 1)))


def recent_success_count(success_history: list[bool]) -> int:
    return sum(1 for success in success_history[-DEFAULT_STAGE_SUCCESS_WINDOW_SIZE:] if success)


def curriculum_stage_can_advance(
    stage: CurriculumStage,
    success_history: list[bool],
    *,
    completed_lap: float | None = None,
    policy_controlled: bool = True,
) -> bool:
    if not policy_controlled:
        return False
    if completed_lap is not None:
        return True
    return recent_success_count(success_history) >= required_successes_for_stage(stage)


def episode_is_policy_controlled(
    episode_start_timestep: int,
    learning_starts: int,
) -> bool:
    """Return true only when every action in the episode came from the policy."""
    return int(episode_start_timestep) >= max(0, int(learning_starts))


def calculate_td3_reward(
    telemetry: Mapping[str, Any],
    action: Mapping[str, Any],
    *,
    previous_telemetry: Mapping[str, Any] | None = None,
    previous_damage: float = 0.0,
    stage: CurriculumStage | None = None,
    episode_distance_m: float = 0.0,
    previous_furthest_distance_m: float = 0.0,
    completed_lap: float | None = None,
    stage_success: bool = False,
    stuck: bool = False,
    terminal_failure: bool | None = None,
) -> float:
    stage = stage or CURRICULUM[0]
    sensors = track_sensors(telemetry)
    front_sensor = sensors[9] if len(sensors) > 9 else 200.0
    min_sensor = min(sensors) if sensors else 200.0
    speed = finite_float(telemetry.get("speedX"))
    angle = finite_float(telemetry.get("angle"))
    track_position = finite_float(telemetry.get("trackPos"))
    lateral_speed = finite_float(telemetry.get("speedY"))
    steer = abs(finite_float(action.get("steer")))
    damage = finite_float(telemetry.get("damage"), previous_damage)
    progress = calculate_progress_delta(previous_telemetry, telemetry)
    new_distance = max(0.0, episode_distance_m - previous_furthest_distance_m)
    aligned_speed = speed * max(0.0, math.cos(angle))
    off_track = min_sensor < 0.0 or abs(track_position) > SOFT_TRACK_BOUNDARY
    crashed = damage > previous_damage
    backwards = math.cos(angle) < 0.0
    if terminal_failure is None:
        terminal_failure = off_track or crashed or backwards or stuck

    reward = clamp(progress, -2.0, 10.0) * stage.progress_weight
    reward += clamp(aligned_speed / 190.0, 0.0, 1.0) * 0.55
    reward += clamp(new_distance, 0.0, 12.0) * stage.milestone_weight
    if progress > 0.18 and abs(track_position) < 0.85 and abs(angle) < 0.55:
        reward += 0.45
    if speed < 24.0 and progress > 0.08:
        reward += 0.35

    reward -= 0.015
    reward -= max(0.0, abs(track_position) - 0.45) * 1.6
    reward -= max(0.0, abs(track_position) - 0.76) * 5.8
    reward -= max(0.0, abs(track_position) - 0.92) * 12.0
    reward -= max(0.0, abs(angle) - 0.10) * 1.1
    reward -= max(0.0, abs(angle) - 0.45) * 4.5
    reward -= max(0.0, abs(lateral_speed) - 4.0) * 0.10
    reward -= max(0.0, abs(lateral_speed) - 12.0) * 0.24

    speed_pressure = clamp((speed - 40.0) / 115.0, 0.0, 1.0)
    reward -= max(0.0, steer - 0.28) * speed_pressure * 5.5
    reward -= max(0.0, steer - 0.55) * speed_pressure * 12.0
    reward -= max(0.0, abs(angle) - 0.35) * speed_pressure * 7.0
    reward -= max(0.0, abs(lateral_speed) - 7.0) * speed_pressure * 0.28
    if abs(angle) > 1.05 and speed > 18.0:
        spin_pressure = clamp((abs(angle) - 1.05) / 0.75, 0.0, 1.0)
        reward -= 42.0 * spin_pressure

    closing_danger = front_sensor < 55.0 and speed > stage.safe_front_speed_kmh
    if closing_danger:
        brake = finite_float(action.get("brake"))
        accel = finite_float(action.get("accel"))
        overspeed_pressure = clamp(
            (speed - stage.safe_front_speed_kmh) / 80.0,
            0.0,
            1.0,
        )
        front_pressure = clamp((55.0 - front_sensor) / 55.0, 0.0, 1.0)
        reward -= front_pressure * overspeed_pressure * 3.2
        reward -= front_pressure * overspeed_pressure * max(0.0, 0.25 - brake) * 4.0
        reward -= front_pressure * overspeed_pressure * accel * 2.4

    if progress < 0.03 and speed < 8.0:
        reward -= 4.0
    if min_sensor < 0.0:
        reward -= 28.0
    if abs(track_position) > 1.0:
        reward -= 42.0
    if crashed:
        reward -= 65.0 + max(0.0, damage - previous_damage) * 0.15
    if backwards:
        reward -= 145.0 + clamp(abs(speed) / 120.0, 0.0, 1.0) * 55.0
    if stuck:
        reward -= 80.0
    if terminal_failure and completed_lap is None and not stage_success:
        remaining = 1.0 - calculate_lap_completion_fraction(episode_distance_m)
        reward -= stage.failure_penalty * (0.35 + 0.65 * remaining)
    if stage_success:
        reward += stage.success_reward
    if completed_lap is not None:
        reward += max(stage.success_reward, 850.0)

    return float(clamp(reward, -450.0, 1000.0))


def make_training_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_family": AGENT6_MODEL_FAMILY,
        "observation_version": AGENT6_OBSERVATION_VERSION,
        "action_version": AGENT6_ACTION_VERSION,
        "reward_version": REWARD_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
        "algorithm": "TD3",
        "agent_name": "TD3 Scratch Racer",
        "agent_number": 6,
        "track": args.track,
        "track_length_m": G_TRACK_3_LENGTH_METRES,
        "learning_source": "reward_only",
        "teacher": None,
        "behaviour_cloning": False,
        "racing_line_imitation": False,
        "automatic_gear_shift": True,
        "actuator_rate_limited": True,
        "stage_success_requirements": dict(DEFAULT_STAGE_SUCCESS_REQUIREMENTS),
        "stage_success_window_size": DEFAULT_STAGE_SUCCESS_WINDOW_SIZE,
        "curriculum_promotion": "successes_within_rolling_window",
        "curriculum_counts_warmup_episodes": False,
        "best_checkpoint_scope": "policy_controlled_training_episodes_only",
        "curriculum": [stage.__dict__ for stage in CURRICULUM],
        "total_timesteps_requested": args.total_timesteps,
        "seed": args.seed,
        "manual_start": args.manual_start,
        "relaunch_frequency": args.relaunch_frequency,
        "reset_retries": args.reset_retries,
        "model_path": str(args.model_path),
        "best_distance_model_path": str(args.best_distance_model_path),
        "best_reward_model_path": str(args.best_reward_model_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "replay_buffer_path": str(args.replay_buffer_path),
        "monitor_path": str(args.run_dir / "monitor.csv") if args.run_dir else None,
        "check_env": args.check_env,
        "warmup_action_space": {
            "mode": args.warmup_action_mode,
            "steer_std": args.warmup_steer_std,
            "accel_min": args.warmup_accel_min,
            "accel_max": args.warmup_accel_max,
            "brake_probability": args.warmup_brake_probability,
        },
        "td3_hyperparameters": {
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "learning_rate": args.learning_rate,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "policy_delay": args.policy_delay,
            "target_policy_noise": args.target_policy_noise,
            "target_noise_clip": args.target_noise_clip,
            "action_noise_sigma": args.action_noise_sigma,
            "net_arch": args.net_arch,
            "device": args.device,
        },
        "command": ORIGINAL_COMMAND,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_scratch_action_space(
    spaces: Any,
    *,
    warmup_action_mode: str = "launch-biased",
    warmup_steer_std: float = 0.30,
    warmup_accel_min: float = 0.25,
    warmup_accel_max: float = 1.00,
    warmup_brake_probability: float = 0.12,
):
    class ScratchActionBox(spaces.Box):
        def __init__(self) -> None:
            super().__init__(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            self.warmup_action_mode = warmup_action_mode
            self.warmup_steer_std = max(0.0, float(warmup_steer_std))
            accel_min = clamp(float(warmup_accel_min), -1.0, 1.0)
            accel_max = clamp(float(warmup_accel_max), -1.0, 1.0)
            self.warmup_accel_min = min(accel_min, accel_max)
            self.warmup_accel_max = max(accel_min, accel_max)
            self.warmup_brake_probability = clamp(
                float(warmup_brake_probability),
                0.0,
                1.0,
            )

        def sample(self, mask: Any = None):
            if self.warmup_action_mode == "uniform":
                try:
                    return super().sample(mask=mask)
                except TypeError:
                    return super().sample()

            rng = getattr(self, "np_random", np.random.default_rng())
            steer = clamp(float(rng.normal(0.0, self.warmup_steer_std)), -1.0, 1.0)
            accel = float(rng.uniform(self.warmup_accel_min, self.warmup_accel_max))
            brake = 0.0
            if float(rng.random()) < self.warmup_brake_probability:
                accel = float(rng.uniform(-0.25, 0.20))
                brake = float(rng.uniform(0.15, 0.85))
            return np.asarray(
                [
                    steer,
                    clamp(accel, -1.0, 1.0),
                    clamp(brake, -1.0, 1.0),
                ],
                dtype=np.float32,
            )

    return ScratchActionBox()


def make_training_env_class(gym: Any, spaces: Any):
    class Td3ScratchTrainingEnv(gym.Env):
        metadata = {"render_modes": []}
        track_length_m = G_TRACK_3_LENGTH_METRES

        def __init__(
            self,
            *,
            track_name: str = DEFAULT_TRACK_NAME,
            manual_start: bool = False,
            relaunch_frequency: int = 0,
            reset_retries: int = 2,
            quiet_reset_log: bool = True,
            run_dir: Path | None = None,
            warmup_action_mode: str = "launch-biased",
            warmup_steer_std: float = 0.30,
            warmup_accel_min: float = 0.25,
            warmup_accel_max: float = 1.00,
            warmup_brake_probability: float = 0.12,
            learning_starts: int = 20_000,
        ) -> None:
            super().__init__()
            self.track_name = track_name
            self.manual_start = manual_start
            self.relaunch_frequency = max(0, int(relaunch_frequency))
            self.reset_retries = max(0, int(reset_retries))
            self.quiet_reset_log = quiet_reset_log
            self.run_dir = Path(run_dir) if run_dir is not None else None
            self.learning_starts = max(0, int(learning_starts))
            self.runner = make_torcs_runner()
            self.action_space = make_scratch_action_space(
                spaces,
                warmup_action_mode=warmup_action_mode,
                warmup_steer_std=warmup_steer_std,
                warmup_accel_min=warmup_accel_min,
                warmup_accel_max=warmup_accel_max,
                warmup_brake_probability=warmup_brake_probability,
            )
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(FEATURE_NAMES),),
                dtype=np.float32,
            )
            self.stage_index = 0
            self.stage_success_streak = 0
            self.stage_success_totals = {
                stage.stage_id: 0
                for stage in CURRICULUM
            }
            self.stage_success_windows = {
                stage.stage_id: []
                for stage in CURRICULUM
            }
            self.episodes_started = 0
            self.total_steps = 0
            self.current_telemetry: dict[str, Any] | None = None
            self.previous_telemetry: dict[str, Any] | None = None
            self.initial_telemetry: dict[str, Any] | None = None
            self.previous_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
            self.previous_damage = 0.0
            self.lap_tracker: LapTracker | None = None
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = 0.0
            self.episode_start_time = 0.0
            self.episode_start_timestep = 0
            self.episode_max_speed = 0.0
            self.episode_off_track_steps = 0
            self.episode_furthest_distance = 0.0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.consecutive_off_track_steps = 0
            self._step_file = None
            self._step_writer = None
            self._open_step_telemetry()

        @property
        def current_stage(self) -> CurriculumStage:
            return CURRICULUM[self.stage_index]

        def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
            super().reset(seed=seed)
            self._ensure_torcs()
            relaunch = bool(options.get("relaunch", False)) if options else False
            if (
                self.relaunch_frequency > 0
                and self.episodes_started > 0
                and self.episodes_started % self.relaunch_frequency == 0
            ):
                relaunch = True
            if self.manual_start:
                relaunch = False

            self._reset_torcs_env(relaunch=relaunch)
            self.episodes_started += 1
            assert self.runner.env is not None
            self.current_telemetry = dict(self.runner.env.client.S.d)
            self.initial_telemetry = dict(self.current_telemetry)
            self.previous_telemetry = None
            self.previous_action = {"steer": 0.0, "accel": 0.0, "brake": 0.0}
            self.previous_damage = finite_float(self.current_telemetry.get("damage"))
            self.lap_tracker = LapTracker(self.current_telemetry.get("lastLapTime", 0.0))
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = finite_float(
                self.current_telemetry.get("distRaced")
            )
            self.episode_start_time = finite_float(self.current_telemetry.get("curLapTime"))
            self.episode_start_timestep = self.total_steps
            self.episode_max_speed = finite_float(self.current_telemetry.get("speedX"))
            self.episode_off_track_steps = 0
            self.episode_furthest_distance = 0.0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.consecutive_off_track_steps = 0
            return self._build_observation(), {}

        def step(self, raw_action: Any):
            assert self.current_telemetry is not None
            assert self.runner.env is not None
            assert self.lap_tracker is not None
            raw_values = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            action = decode_td3_action(raw_values)
            current_speed = finite_float(self.current_telemetry.get("speedX"))
            action = rate_limit_td3_action(
                action,
                self.previous_action,
                speed_kmh=current_speed,
            )
            action["gear"] = shift_gears(current_speed)
            raw_observation, _runner_reward, done, _info = (
                self.runner._step_full_control_agent(action)
            )
            telemetry = dict(raw_observation)
            current_damage = finite_float(telemetry.get("damage"), self.previous_damage)
            sensors = track_sensors(telemetry)
            min_sensor = min(sensors) if sensors else 200.0
            track_position = finite_float(telemetry.get("trackPos"))
            speed = finite_float(telemetry.get("speedX"))
            angle = finite_float(telemetry.get("angle"))
            episode_distance_m = max(
                0.0,
                finite_float(telemetry.get("distRaced")) - self.episode_start_distance,
            )
            previous_furthest = self.episode_furthest_distance
            self.episode_furthest_distance = max(
                self.episode_furthest_distance,
                episode_distance_m,
            )
            completed_lap = self.lap_tracker.update(telemetry)
            if (
                completed_lap is None
                and done
                and practice_finish_is_plausible(
                    self.initial_telemetry,
                    telemetry,
                    self,
                )
            ):
                completed_lap = self.lap_tracker.record(telemetry.get("curLapTime", 0.0))

            stopped_time_delta = calculate_stopped_time_delta(
                self.previous_telemetry,
                telemetry,
            )
            if stopped_time_delta > 0.0:
                self.stopped_seconds += stopped_time_delta
            else:
                self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = max(
                self.episode_max_stopped_seconds,
                self.stopped_seconds,
            )
            stuck = self.stopped_seconds >= STUCK_SECONDS_LIMIT
            off_track = min_sensor < 0.0 or abs(track_position) > SOFT_TRACK_BOUNDARY
            self.consecutive_off_track_steps = (
                self.consecutive_off_track_steps + 1 if off_track else 0
            )
            left_track_bounds = abs(track_position) > HARD_TRACK_BOUNDARY
            off_track_terminal = (
                off_track and self.consecutive_off_track_steps >= OFF_TRACK_GRACE_STEPS
            )
            crashed = current_damage > self.previous_damage
            backwards = math.cos(angle) < 0.0
            stage_success = (
                episode_distance_m >= self.current_stage.distance_target_m
                or completed_lap is not None
            )
            terminal_failure = bool(
                done
                or left_track_bounds
                or off_track_terminal
                or crashed
                or backwards
                or stuck
            )
            reward = calculate_td3_reward(
                telemetry,
                action,
                previous_telemetry=self.previous_telemetry,
                previous_damage=self.previous_damage,
                stage=self.current_stage,
                episode_distance_m=episode_distance_m,
                previous_furthest_distance_m=previous_furthest,
                completed_lap=completed_lap,
                stage_success=stage_success,
                stuck=stuck,
                terminal_failure=terminal_failure,
            )

            self.previous_damage = current_damage
            self.previous_telemetry = telemetry
            self.current_telemetry = telemetry
            self.previous_action = {
                "steer": action["steer"],
                "accel": action["accel"],
                "brake": action["brake"],
            }
            self.steps += 1
            self.total_steps += 1
            self.episode_reward += reward
            self.episode_max_speed = max(self.episode_max_speed, speed)
            if off_track:
                self.episode_off_track_steps += 1

            terminated = bool(stage_success or terminal_failure)
            truncated = self.steps >= self.current_stage.max_episode_steps
            policy_controlled = episode_is_policy_controlled(
                self.episode_start_timestep,
                self.learning_starts,
            )
            stage_required_successes = required_successes_for_stage(self.current_stage)
            stage_success_total = self.stage_success_totals.get(
                self.current_stage.stage_id,
                0,
            )
            stage_success_history = self.stage_success_windows[
                self.current_stage.stage_id
            ]
            stage_recent_successes = recent_success_count(stage_success_history)
            stage_completed = False
            if terminated or truncated:
                if policy_controlled:
                    if stage_success:
                        self.stage_success_streak += 1
                        stage_success_total += 1
                        self.stage_success_totals[self.current_stage.stage_id] = (
                            stage_success_total
                        )
                    else:
                        self.stage_success_streak = 0
                    stage_success_history.append(bool(stage_success))
                    if len(stage_success_history) > DEFAULT_STAGE_SUCCESS_WINDOW_SIZE:
                        del stage_success_history[
                            : len(stage_success_history)
                            - DEFAULT_STAGE_SUCCESS_WINDOW_SIZE
                        ]
                    stage_recent_successes = recent_success_count(
                        stage_success_history
                    )
                    stage_completed = curriculum_stage_can_advance(
                        self.current_stage,
                        stage_success_history,
                        completed_lap=completed_lap,
                        policy_controlled=policy_controlled,
                    )
            reason = ""
            if terminated or truncated:
                reason = "truncated"
                if completed_lap is not None and not policy_controlled:
                    reason = "warmup_lap_completed"
                elif completed_lap is not None:
                    reason = "target_laps_completed"
                elif stage_success and not policy_controlled:
                    reason = "warmup_stage_success"
                elif stage_completed:
                    reason = "curriculum_stage_completed"
                elif stage_success:
                    reason = "curriculum_stage_success"
                elif crashed:
                    reason = "crashed"
                elif off_track or left_track_bounds:
                    reason = "off_track"
                elif backwards:
                    reason = "backwards"
                elif stuck:
                    reason = "stuck"
                elif done:
                    reason = "torcs_done"

            self._write_step_telemetry(
                telemetry,
                raw_values,
                action,
                reward,
                terminated,
                truncated,
                reason,
            )

            info: dict[str, Any] = {
                "policy_controlled": policy_controlled,
                "stage_id": self.current_stage.stage_id,
                "stage_name": self.current_stage.name,
                "distance_m": self.episode_furthest_distance,
                "furthest_distance_m": self.episode_furthest_distance,
                "lap_completion_fraction": calculate_lap_completion_fraction(
                    self.episode_furthest_distance
                ),
                "termination_reason": reason,
            }
            if terminated or truncated:
                summary = self._episode_summary(
                    reason,
                    telemetry,
                    stage_success_streak=self.stage_success_streak,
                    stage_required_successes=stage_required_successes,
                    stage_recent_successes=stage_recent_successes,
                    stage_success_total=stage_success_total,
                    policy_controlled=policy_controlled,
                )
                info["episode_summary"] = summary
                if stage_completed and self.stage_index < len(CURRICULUM) - 1:
                    self.stage_index += 1
                    self.stage_success_streak = 0

            return self._build_observation(), reward, terminated, truncated, info

        def close(self) -> None:
            self._close_step_telemetry()
            self._close_torcs_env()
            self.runner.shutdown()

        def build_current_episode_summary(self, reason: str = "training_stopped"):
            if self.current_telemetry is None or self.steps <= 0:
                return None
            return self._episode_summary(reason, self.current_telemetry)

        def write_end_of_run_summary(self, reason: str = "training_stopped"):
            summary = self.build_current_episode_summary(reason)
            if summary is not None and self.run_dir is not None:
                write_json(self.run_dir / "end_of_run_summary.json", summary)
            return summary

        def _episode_summary(
            self,
            reason: str,
            telemetry: Mapping[str, Any],
            *,
            stage_success_streak: int | None = None,
            stage_required_successes: int | None = None,
            stage_recent_successes: int | None = None,
            stage_success_total: int | None = None,
            policy_controlled: bool | None = None,
        ) -> dict[str, Any]:
            sensors = track_sensors(telemetry)
            distance_m = self.episode_furthest_distance
            stage_id = self.current_stage.stage_id
            duration_seconds = max(
                0.0,
                finite_float(telemetry.get("curLapTime")) - self.episode_start_time,
            )
            average_speed_kmh = (
                distance_m / duration_seconds * 3.6
                if duration_seconds > 0.0
                else 0.0
            )
            assert self.lap_tracker is not None
            if policy_controlled is None:
                policy_controlled = episode_is_policy_controlled(
                    self.episode_start_timestep,
                    self.learning_starts,
                )
            return {
                "policy_controlled": policy_controlled,
                "stage_id": self.current_stage.stage_id,
                "stage_name": self.current_stage.name,
                "steps": self.steps,
                "reward": self.episode_reward,
                "distance_m": distance_m,
                "furthest_distance_m": distance_m,
                "duration_seconds": duration_seconds,
                "average_speed_kmh": average_speed_kmh,
                "max_speed_kmh": self.episode_max_speed,
                "laps_completed": self.lap_tracker.laps_completed,
                "best_lap_time_seconds": self.lap_tracker.best_lap_time,
                "lap_completion_fraction": calculate_lap_completion_fraction(distance_m),
                "off_track_steps": self.episode_off_track_steps,
                "max_stopped_seconds": self.episode_max_stopped_seconds,
                "stage_success_streak": (
                    self.stage_success_streak
                    if stage_success_streak is None
                    else stage_success_streak
                ),
                "stage_required_successes": (
                    required_successes_for_stage(self.current_stage)
                    if stage_required_successes is None
                    else stage_required_successes
                ),
                "stage_recent_successes": (
                    recent_success_count(self.stage_success_windows.get(stage_id, []))
                    if stage_recent_successes is None
                    else stage_recent_successes
                ),
                "stage_success_window_size": DEFAULT_STAGE_SUCCESS_WINDOW_SIZE,
                "stage_success_total": (
                    self.stage_success_totals.get(stage_id, 0)
                    if stage_success_total is None
                    else stage_success_total
                ),
                "termination_reason": reason,
                "final_speed_kmh": finite_float(telemetry.get("speedX")),
                "final_track_pos": finite_float(telemetry.get("trackPos")),
                "final_angle": finite_float(telemetry.get("angle")),
                "front_sensor": sensors[9] if len(sensors) > 9 else None,
                "min_track_sensor": min(sensors) if sensors else None,
            }

        def _ensure_torcs(self) -> None:
            if self.runner.env is not None:
                return
            self._launch_and_connect_torcs()

        def _reset_torcs_env(self, *, relaunch: bool = False):
            last_error: Exception | None = None
            for attempt in range(self.reset_retries + 1):
                try:
                    if relaunch and not self.manual_start:
                        self._restart_torcs()
                    assert self.runner.env is not None
                    with maybe_suppress_stdout(self.quiet_reset_log):
                        return self.runner.env.reset(relaunch=relaunch)
                except Exception as exc:
                    last_error = exc
                    if attempt >= self.reset_retries:
                        raise RuntimeError(
                            "TORCS reset failed after "
                            f"{self.reset_retries + 1} attempts"
                        ) from last_error
                    if not self.manual_start:
                        self._restart_torcs()
                    else:
                        self._close_torcs_env()
                        self._ensure_torcs()
                    time.sleep(1.0)
            raise RuntimeError("TORCS reset failed") from last_error

        def _restart_torcs(self) -> None:
            self._close_torcs_env()
            self.runner.shutdown()
            self._launch_and_connect_torcs()

        def _launch_and_connect_torcs(self) -> None:
            if not self.manual_start:
                self.runner.launch()
            self.runner.connect()
            self.runner.load_track(self.track_name)

        def _close_torcs_env(self) -> None:
            if self.runner.env is None:
                return
            try:
                self.runner.env.end()
            except Exception:
                pass
            self.runner.env = None

        def _build_observation(self) -> np.ndarray:
            if self.current_telemetry is None:
                return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            return np.asarray(
                build_td3_observation(
                    self.current_telemetry,
                    self.previous_action,
                    track_length_m=self.track_length_m,
                ),
                dtype=np.float32,
            )

        def _open_step_telemetry(self) -> None:
            if self.run_dir is None:
                return
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._step_file = (self.run_dir / "steps.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self._step_writer = csv.writer(self._step_file)
            self._step_writer.writerow(STEP_COLUMNS)
            self._step_file.flush()

        def _close_step_telemetry(self) -> None:
            if self._step_file is not None:
                self._step_file.close()
            self._step_file = None
            self._step_writer = None

        def _write_step_telemetry(
            self,
            telemetry: Mapping[str, Any],
            raw_action: np.ndarray,
            action: Mapping[str, Any],
            reward: float,
            terminated: bool,
            truncated: bool,
            reason: str,
        ) -> None:
            if self._step_writer is None:
                return
            sensors = track_sensors(telemetry)
            padded = np.zeros(3, dtype=float)
            padded[: min(3, len(raw_action))] = raw_action[:3]
            self._step_writer.writerow(
                [
                    self.episodes_started,
                    self.current_stage.stage_id,
                    self.steps,
                    self.total_steps,
                    telemetry.get("distFromStart", 0),
                    telemetry.get("distRaced", 0),
                    telemetry.get("speedX", 0),
                    telemetry.get("speedY", 0),
                    telemetry.get("angle", 0),
                    telemetry.get("trackPos", 0),
                    telemetry.get("damage", 0),
                    sensors[9] if len(sensors) > 9 else 0,
                    min(sensors) if sensors else 0,
                    padded[0],
                    padded[1],
                    padded[2],
                    action["steer"],
                    action["accel"],
                    action["brake"],
                    action["gear"],
                    reward,
                    terminated,
                    truncated,
                    reason,
                    telemetry.get("curLapTime", 0),
                ]
            )
            self._step_file.flush()

    return Td3ScratchTrainingEnv


class TrainingProgressState:
    def __init__(self) -> None:
        self.episodes_seen = 0
        self.last_episode: dict[str, Any] | None = None
        self.best_distance_m: float | None = None
        self.best_reward: float | None = None
        self.current_stage = CURRICULUM[0].name

    def record_episode(self, row: Mapping[str, Any]) -> None:
        self.episodes_seen = int(row.get("episodes_seen", self.episodes_seen))
        self.last_episode = dict(row)
        self.current_stage = str(row.get("stage_name") or self.current_stage)

    def record_best_distance(self, distance_m: float) -> None:
        self.best_distance_m = float(distance_m)

    def record_best_reward(self, reward: float) -> None:
        self.best_reward = float(reward)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_training_progress_line(
    completed_steps: int,
    total_steps: int,
    elapsed_seconds: float,
    progress_state: TrainingProgressState,
    *,
    width: int = 28,
) -> str:
    total_steps = max(1, int(total_steps))
    completed_steps = max(0, min(int(completed_steps), total_steps))
    fraction = completed_steps / total_steps
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    percent = fraction * 100.0
    if completed_steps > 0 and elapsed_seconds > 0.0:
        rate_value = completed_steps / elapsed_seconds
        eta = format_duration((total_steps - completed_steps) / max(rate_value, 1e-9))
        rate = f"{rate_value:,.0f} step/s"
    else:
        eta = "--:--:--"
        rate = "-- step/s"

    if progress_state.last_episode is None:
        episode_text = "episodes=0"
    else:
        episode = progress_state.last_episode
        episode_text = (
            f"ep {progress_state.episodes_seen} "
            f"{episode.get('stage_id', 'stage')} "
            f"{episode.get('termination_reason', 'running')} "
            f"{float(episode.get('distance_m') or 0.0):.0f}m "
            f"R={float(episode.get('reward') or 0.0):.0f}"
        )

    best_parts = []
    if progress_state.best_distance_m is not None:
        best_parts.append(f"best_dist={progress_state.best_distance_m:.0f}m")
    if progress_state.best_reward is not None:
        best_parts.append(f"best_R={progress_state.best_reward:.0f}")
    best_text = f" | {' '.join(best_parts)}" if best_parts else ""
    return (
        f"Training [{bar}] {percent:5.1f}% | "
        f"{completed_steps:,}/{total_steps:,} steps | {rate} | "
        f"ETA {eta} | stage={progress_state.current_stage} | "
        f"{episode_text}{best_text}"
    )


def make_episode_summary_callback_class(BaseCallback: Any):
    class EpisodeSummaryCallback(BaseCallback):
        def __init__(self, run_dir: Path, progress_state: TrainingProgressState | None = None):
            super().__init__(verbose=0)
            self.run_dir = Path(run_dir)
            self.progress_state = progress_state
            self.episodes_path = self.run_dir / "episodes.csv"
            self.episodes_seen = 0
            self._file = None
            self._writer = None

        def _on_training_start(self):
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._file = self.episodes_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_COLUMNS)
            self._writer.writeheader()
            self._file.flush()

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                self.episodes_seen += 1
                row = {
                    column: summary.get(column)
                    for column in EPISODE_COLUMNS
                    if column not in {"episodes_seen", "global_timestep"}
                }
                row["episodes_seen"] = self.episodes_seen
                row["global_timestep"] = self.num_timesteps
                self._writer.writerow(row)
                self._file.flush()
                if self.progress_state is not None:
                    self.progress_state.record_episode(row)
            return True

        def _on_training_end(self):
            if self._file is not None:
                self._file.close()
            self._file = None
            self._writer = None

    return EpisodeSummaryCallback


def make_best_model_callback_class(BaseCallback: Any):
    class BestModelCallback(BaseCallback):
        def __init__(
            self,
            best_distance_model_path: Path,
            best_reward_model_path: Path,
            metadata: Mapping[str, Any],
            progress_state: TrainingProgressState | None = None,
            learning_starts: int = 0,
        ) -> None:
            super().__init__(verbose=0)
            self.best_distance_model_path = Path(best_distance_model_path)
            self.best_reward_model_path = Path(best_reward_model_path)
            self.metadata = dict(metadata)
            self.progress_state = progress_state
            self.learning_starts = max(0, int(learning_starts))
            self.best_distance_m = self._read_existing_best_distance()
            self.best_reward = self._read_existing_best_reward()
            if self.progress_state is not None:
                if self.best_distance_m is not None:
                    self.progress_state.record_best_distance(self.best_distance_m)
                if self.best_reward is not None:
                    self.progress_state.record_best_reward(self.best_reward)

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                if (
                    not bool(summary.get("policy_controlled"))
                    or self.num_timesteps <= self.learning_starts
                ):
                    continue
                distance_m = float(summary.get("distance_m") or 0.0)
                reward = float(summary.get("reward") or 0.0)
                if self.best_distance_m is None or distance_m > self.best_distance_m:
                    self.best_distance_m = distance_m
                    if self.progress_state is not None:
                        self.progress_state.record_best_distance(distance_m)
                    self._save_best(
                        self.best_distance_model_path,
                        "best_distance_episode",
                        {
                            "selection_reason": "furthest distance reached by scratch TD3",
                            "distance_m": distance_m,
                            "episode_summary": summary,
                        },
                    )
                if self.best_reward is None or reward > self.best_reward:
                    self.best_reward = reward
                    if self.progress_state is not None:
                        self.progress_state.record_best_reward(reward)
                    self._save_best(
                        self.best_reward_model_path,
                        "best_reward_episode",
                        {
                            "selection_reason": "highest shaped reward reached by scratch TD3",
                            "reward": reward,
                            "distance_m": distance_m,
                            "episode_summary": summary,
                        },
                    )
            return True

        def _save_best(self, path: Path, metadata_key: str, payload: Mapping[str, Any]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(path))
            write_json(
                metadata_path_for_policy(path),
                {
                    **self.metadata,
                    metadata_key: {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "global_timestep": self.num_timesteps,
                        **payload,
                    },
                },
            )

        def _read_existing_best_distance(self) -> float | None:
            metadata = read_policy_metadata(self.best_distance_model_path)
            best_episode = metadata.get("best_distance_episode")
            if not self._is_verified_policy_episode(best_episode):
                return None
            distance_m = finite_float(best_episode.get("distance_m"), default=-1.0)
            return distance_m if distance_m >= 0.0 else None

        def _read_existing_best_reward(self) -> float | None:
            metadata = read_policy_metadata(self.best_reward_model_path)
            best_episode = metadata.get("best_reward_episode")
            if not self._is_verified_policy_episode(best_episode):
                return None
            reward = finite_float(best_episode.get("reward"), default=float("-inf"))
            return reward if math.isfinite(reward) else None

        @staticmethod
        def _is_verified_policy_episode(best_episode: Any) -> bool:
            return episode_metadata_is_policy_controlled(best_episode)

    return BestModelCallback


def make_console_progress_callback_class(BaseCallback: Any):
    class ConsoleProgressCallback(BaseCallback):
        def __init__(
            self,
            total_timesteps: int,
            progress_state: TrainingProgressState,
            *,
            interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
        ) -> None:
            super().__init__(verbose=0)
            self.total_timesteps = int(total_timesteps)
            self.progress_state = progress_state
            self.interval_seconds = float(interval_seconds)
            self.start_timestep = 0
            self.start_time = 0.0
            self.last_render_time = 0.0
            self.previous_line_length = 0

        def _on_training_start(self):
            self.start_timestep = self.num_timesteps
            self.start_time = time.monotonic()
            self._render(force=True)

        def _on_step(self):
            now = time.monotonic()
            if now - self.last_render_time >= self.interval_seconds:
                self._render(now=now)
            return True

        def _on_training_end(self):
            self._render(force=True)
            if self.previous_line_length:
                print()

        def _render(self, *, now: float | None = None, force: bool = False):
            now = time.monotonic() if now is None else now
            if not force and now - self.last_render_time < self.interval_seconds:
                return
            completed_steps = self.num_timesteps - self.start_timestep
            line = format_training_progress_line(
                completed_steps,
                self.total_timesteps,
                now - self.start_time,
                self.progress_state,
            )
            terminal_width = shutil.get_terminal_size((120, 20)).columns
            max_width = max(60, terminal_width - 1)
            if len(line) > max_width:
                line = f"{line[: max_width - 3]}..."
            padding = " " * max(0, self.previous_line_length - len(line))
            print(f"\r{line}{padding}", end="", flush=True)
            self.previous_line_length = len(line)
            self.last_render_time = now

    return ConsoleProgressCallback


def make_checkpoint_callback(CheckpointCallback: Any, args: argparse.Namespace):
    checkpoint_kwargs = {
        "save_freq": args.checkpoint_freq,
        "save_path": str(args.checkpoint_dir),
        "name_prefix": "agent6_td3_scratch",
        "save_replay_buffer": args.save_checkpoint_replay_buffer,
        "save_vecnormalize": False,
    }
    try:
        return CheckpointCallback(**checkpoint_kwargs)
    except TypeError:
        checkpoint_kwargs.pop("save_replay_buffer", None)
        return CheckpointCallback(**checkpoint_kwargs)


def tensorboard_is_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("tensorboard") is not None


def resolve_tensorboard_log_dir(tensorboard_dir: Path, *, disable_tensorboard: bool):
    if disable_tensorboard:
        return None
    if tensorboard_is_available():
        return str(tensorboard_dir)
    print(
        "TensorBoard is not installed, so TD3 TensorBoard logging is disabled. "
        "Training will still write CSV run logs and metadata."
    )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Agent 6 as a from-scratch TD3 continuous-control racer.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--track", default=DEFAULT_TRACK_NAME)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--best-distance-model-path",
        type=Path,
        default=DEFAULT_BEST_DISTANCE_MODEL_PATH,
    )
    parser.add_argument(
        "--best-reward-model-path",
        type=Path,
        default=DEFAULT_BEST_REWARD_MODEL_PATH,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--tensorboard-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR)
    parser.add_argument("--replay-buffer-path", type=Path, default=DEFAULT_REPLAY_BUFFER_PATH)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--manual-start", action="store_true")
    parser.add_argument("--relaunch-frequency", type=int, default=0)
    parser.add_argument("--reset-retries", type=int, default=2)
    parser.add_argument("--show-torcs-reset-log", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run the SB3 environment checker before training. This starts TORCS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-policy-noise", type=float, default=0.20)
    parser.add_argument("--target-noise-clip", type=float, default=0.50)
    parser.add_argument("--action-noise-sigma", type=float, default=0.12)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[400, 300])
    parser.add_argument(
        "--warmup-action-mode",
        choices=("launch-biased", "uniform"),
        default="launch-biased",
    )
    parser.add_argument("--warmup-steer-std", type=float, default=0.18)
    parser.add_argument("--warmup-accel-min", type=float, default=0.25)
    parser.add_argument("--warmup-accel-max", type=float, default=1.00)
    parser.add_argument("--warmup-brake-probability", type=float, default=0.12)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-checkpoint-replay-buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.argv = [sys.argv[0]]
    random.seed(args.seed)
    np.random.seed(args.seed)

    (
        gym,
        spaces,
        TD3,
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        Monitor,
        check_env,
        NormalActionNoise,
    ) = import_training_dependencies()

    run_dir = args.run_dir if args.run_dir is not None else make_default_run_dir()
    args.run_dir = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_distance_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_reward_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    args.replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)

    env_class = make_training_env_class(gym, spaces)
    raw_env = env_class(
        track_name=args.track,
        manual_start=args.manual_start,
        relaunch_frequency=args.relaunch_frequency,
        reset_retries=args.reset_retries,
        quiet_reset_log=not args.show_torcs_reset_log,
        run_dir=run_dir,
        warmup_action_mode=args.warmup_action_mode,
        warmup_steer_std=args.warmup_steer_std,
        warmup_accel_min=args.warmup_accel_min,
        warmup_accel_max=args.warmup_accel_max,
        warmup_brake_probability=args.warmup_brake_probability,
        learning_starts=args.learning_starts,
    )
    raw_env.action_space.seed(args.seed)
    raw_env.observation_space.seed(args.seed)
    if args.check_env:
        try:
            check_env(raw_env, warn=True, skip_render_check=True)
        except TypeError:
            check_env(raw_env, warn=True)
    env = Monitor(
        raw_env,
        filename=str(run_dir / "monitor.csv"),
        info_keywords=(
            "policy_controlled",
            "stage_id",
            "distance_m",
            "furthest_distance_m",
            "lap_completion_fraction",
            "termination_reason",
        ),
    )

    metadata = make_training_metadata(args)
    write_json(run_dir / "training_metadata.json", metadata)
    write_json(metadata_path_for_policy(args.model_path), metadata)

    tensorboard_log = resolve_tensorboard_log_dir(
        args.tensorboard_dir,
        disable_tensorboard=args.no_tensorboard,
    )
    action_noise = NormalActionNoise(
        mean=np.zeros(3, dtype=np.float32),
        sigma=np.ones(3, dtype=np.float32) * args.action_noise_sigma,
    )
    model = TD3(
        "MlpPolicy",
        env,
        verbose=int(args.verbose_training),
        tensorboard_log=tensorboard_log,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_rate=args.learning_rate,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_delay=args.policy_delay,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        action_noise=action_noise,
        policy_kwargs={"net_arch": args.net_arch},
        seed=args.seed,
        device=args.device,
    )

    progress_state = TrainingProgressState()
    EpisodeSummaryCallback = make_episode_summary_callback_class(BaseCallback)
    BestModelCallback = make_best_model_callback_class(BaseCallback)
    callbacks = [
        make_checkpoint_callback(CheckpointCallback, args),
        EpisodeSummaryCallback(run_dir, progress_state),
        BestModelCallback(
            args.best_distance_model_path,
            args.best_reward_model_path,
            metadata,
            progress_state,
            learning_starts=args.learning_starts,
        ),
    ]
    if not args.no_progress_bar and not args.verbose_training:
        ConsoleProgressCallback = make_console_progress_callback_class(BaseCallback)
        callbacks.append(ConsoleProgressCallback(args.total_timesteps, progress_state))
    callback = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=True,
            log_interval=10,
        )
        end_summary = raw_env.write_end_of_run_summary("total_timesteps_reached")
        model.save(str(args.model_path))
        if args.save_replay_buffer:
            model.save_replay_buffer(str(args.replay_buffer_path))
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        if end_summary is not None:
            metadata["end_of_run_summary"] = end_summary
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_policy(args.model_path), metadata)
        print(f"Saved Agent 6 TD3 model to {args.model_path}")
        print(f"Saved training run logs to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
