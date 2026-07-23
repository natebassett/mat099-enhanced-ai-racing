import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
ORIGINAL_COMMAND = " ".join(sys.argv)

from agents.hybrid_ppo_agent import (  # noqa: E402
    AGENT4_ACTION_VERSION,
    AGENT4_MODEL_FAMILY,
    AGENT4_OBSERVATION_VERSION,
    DEFAULT_RESIDUAL_SCALE,
    DEFAULT_BEST_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    MAX_ACCEL_RESIDUAL,
    MAX_BRAKE_RESIDUAL,
    MAX_STEER_RESIDUAL,
    apply_hybrid_residual,
    build_hybrid_ppo_observation,
    get_track_sensors,
)
from agents.map_aware_agent import DEFAULT_RACING_LINE_PATH, MapAwareAgent  # noqa: E402
from runner.lap_tracker import LapTracker, practice_finish_is_plausible  # noqa: E402
with contextlib.redirect_stderr(io.StringIO()):
    from runner.torcs_runner import TorcsRunner  # noqa: E402


DEFAULT_TRACK_NAME = "g-track-3"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "agent4_hybrid_ppo"
DEFAULT_TENSORBOARD_DIR = PROJECT_ROOT / "models" / "tensorboard" / "agent4_hybrid_ppo"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "models" / "training_runs" / "agent4_hybrid_ppo"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 1.0
DEFAULT_LOG_STD_INIT = -1.2
REWARD_VERSION = "agent4_reward_v7_track_width_safe_exit"
REWARD_WEIGHTS = {
    "progress": 0.35,
    "pace": 0.14,
    "slow_pace": 0.16,
    "time_step": 0.004,
    "aligned_speed": 0.06,
    "racing_line_error": 0.12,
    "large_racing_line_error": 0.25,
    "overspeed": 0.32,
    "lateral_slide": 0.12,
    "residual_size": 0.05,
    "safety_shield": 0.12,
    "unsafe_edge_exit": 0.26,
    "off_track": 260.0,
    "crash": 320.0,
    "backwards": 260.0,
    "stuck": 180.0,
    "incomplete_lap_failure": 220.0,
    "lap_completed": 450.0,
}
STUCK_SPEED_LIMIT_KMH = 5.0
STUCK_PROGRESS_LIMIT_M = 0.20
STUCK_SECONDS_LIMIT = 3.0
LINE_GUIDE_FREE_ERROR = 0.18
LINE_GUIDE_LARGE_ERROR = 0.50


def import_training_dependencies():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
        )
    except ImportError as exc:
        raise SystemExit(
            "Agent 4 training needs extra packages:\n"
            "  pip install stable-baselines3 gymnasium\n\n"
            "The normal race menu does not need these packages unless you want "
            "to load/train a PPO model."
        ) from exc

    return gym, spaces, PPO, BaseCallback, CallbackList, CheckpointCallback


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def metadata_path_for_model(model_path):
    return Path(model_path).with_suffix(".metadata.json")


def make_default_run_dir():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUNS_DIR / timestamp


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


@contextlib.contextmanager
def maybe_suppress_stdout(enabled):
    if not enabled:
        yield
        return

    with contextlib.redirect_stdout(io.StringIO()):
        yield


def format_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"

    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgressState:
    def __init__(self):
        self.episodes_seen = 0
        self.last_episode = None
        self.best_clean_lap_time = None

    def record_episode(self, row):
        self.episodes_seen = int(row.get("episodes_seen", self.episodes_seen))
        self.last_episode = dict(row)

    def record_best_clean_lap(self, lap_time):
        self.best_clean_lap_time = float(lap_time)


def format_training_progress_line(
    completed_steps,
    total_steps,
    elapsed_seconds,
    progress_state,
    *,
    width=28,
):
    total_steps = max(1, int(total_steps))
    completed_steps = max(0, min(int(completed_steps), total_steps))
    fraction = completed_steps / total_steps
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    percent = fraction * 100.0

    if completed_steps > 0 and elapsed_seconds > 0.0:
        steps_per_second = completed_steps / elapsed_seconds
        eta_seconds = (total_steps - completed_steps) / max(steps_per_second, 1e-9)
        eta = format_duration(eta_seconds)
        rate = f"{steps_per_second:,.0f} step/s"
    else:
        eta = "--:--:--"
        rate = "-- step/s"

    last_episode = progress_state.last_episode
    if last_episode is None:
        episode_text = "episodes=0"
    else:
        episode_text = (
            f"ep {progress_state.episodes_seen} "
            f"{last_episode.get('termination_reason', 'running')} "
            f"{float(last_episode.get('distance_m', 0.0)):.0f}m "
            f"R={float(last_episode.get('reward', 0.0)):.0f} "
            f"laps={int(last_episode.get('laps_completed', 0))}"
        )

    best_text = ""
    if progress_state.best_clean_lap_time is not None:
        best_text = f" | best={progress_state.best_clean_lap_time:.3f}s"

    return (
        f"Training [{bar}] {percent:5.1f}% | "
        f"{completed_steps:,}/{total_steps:,} steps | {rate} | "
        f"ETA {eta} | {episode_text}{best_text}"
    )


def tensorboard_is_available():
    return importlib.util.find_spec("tensorboard") is not None


def resolve_tensorboard_log_dir(tensorboard_dir, *, disable_tensorboard=False):
    if disable_tensorboard:
        return None

    if tensorboard_is_available():
        return str(tensorboard_dir)

    print(
        "TensorBoard is not installed, so PPO TensorBoard logging is disabled. "
        "Training will still run and write episodes.csv/training_metadata.json."
    )
    return None


def build_training_metadata(args, *, resumed_from=None):
    return {
        "model_family": AGENT4_MODEL_FAMILY,
        "observation_version": AGENT4_OBSERVATION_VERSION,
        "action_version": AGENT4_ACTION_VERSION,
        "reward_version": REWARD_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
        "base_agent": "Map-Aware Racing-Line Agent",
        "teacher_policy": {
            "name": "Map-Aware Racing-Line Agent",
            "role": "fastest_current_baseline_not_ground_truth",
            "learning_goal": (
                "learn racecraft corrections around the teacher: braking, "
                "coasting, throttle timing, and turn-in timing"
            ),
        },
        "track": args.track,
        "residual_scale": args.residual_scale,
        "max_episode_steps": args.max_episode_steps,
        "target_laps": args.target_laps,
        "manual_start": args.manual_start,
        "relaunch_frequency": args.relaunch_frequency,
        "reset_retries": args.reset_retries,
        "total_timesteps_requested": args.total_timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "tensorboard_requested": not args.no_tensorboard,
        "tensorboard_available": tensorboard_is_available(),
        "progress_bar_enabled": not args.no_progress_bar,
        "verbose_training": args.verbose_training,
        "show_torcs_reset_log": args.show_torcs_reset_log,
        "seed": args.seed,
        "model_path": str(args.model_path),
        "best_model_path": str(args.best_model_path),
        "ppo_hyperparameters": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "log_std_init": args.log_std_init,
            "device": args.device,
        },
        "curriculum_recommendation": [
            {
                "phase": 1,
                "residual_scale": 0.1,
                "goal": "learn to survive and reach farther than Agent 3 baseline disruption",
            },
            {
                "phase": 2,
                "residual_scale": 0.2,
                "goal": "resume once distance and failure location stabilize",
            },
            {
                "phase": 3,
                "residual_scale": 0.3,
                "goal": "widen useful racecraft corrections after clean-lap reliability improves",
            },
        ],
        "stuck_detection": {
            "speed_limit_kmh": STUCK_SPEED_LIMIT_KMH,
            "progress_limit_m": STUCK_PROGRESS_LIMIT_M,
            "seconds_limit": STUCK_SECONDS_LIMIT,
        },
        "line_guidance": {
            "role": "soft_teacher_prior_not_optimal_real_world_racing_line",
            "free_error": LINE_GUIDE_FREE_ERROR,
            "large_error": LINE_GUIDE_LARGE_ERROR,
        },
        "reward_weights": REWARD_WEIGHTS,
        "command": ORIGINAL_COMMAND,
        "python_version": sys.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resumed_from": resumed_from,
    }


def validate_resume_metadata(model_path, *, force_resume=False):
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.is_file():
        message = (
            "No metadata file was found beside the model. The weights can "
            "probably be loaded, but compatibility cannot be checked."
        )
        if force_resume:
            print(f"WARNING: {message}")
            return None
        raise SystemExit(f"{message}\nUse --force-resume if you still want to continue.")

    metadata = read_json(metadata_path)
    expected = {
        "model_family": AGENT4_MODEL_FAMILY,
        "observation_version": AGENT4_OBSERVATION_VERSION,
        "action_version": AGENT4_ACTION_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_shape": [3],
    }
    mismatches = [
        key
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches and not force_resume:
        formatted = ", ".join(mismatches)
        raise SystemExit(
            "The saved PPO model was trained with an incompatible Agent 4 "
            f"contract: {formatted}.\n"
            "Start a fresh model, or use --force-resume only if you know the "
            "change is compatible."
        )

    if metadata.get("reward_version") != REWARD_VERSION:
        print(
            "WARNING: reward version changed from "
            f"{metadata.get('reward_version')} to {REWARD_VERSION}. "
            "The model can resume, but compare evaluation laps carefully."
        )

    return metadata


def calculate_progress_delta(previous_telemetry, telemetry, track_length=None):
    if previous_telemetry is None:
        return 0.0

    if "distRaced" in previous_telemetry and "distRaced" in telemetry:
        return float(telemetry["distRaced"]) - float(previous_telemetry["distRaced"])

    if not track_length:
        return 0.0

    previous_distance = float(previous_telemetry.get("distFromStart", 0.0))
    current_distance = float(telemetry.get("distFromStart", previous_distance))
    delta = (current_distance % track_length) - (previous_distance % track_length)
    if delta < -track_length / 2.0:
        delta += track_length
    elif delta > track_length / 2.0:
        delta -= track_length
    return delta


def calculate_step_time_delta(previous_telemetry, telemetry):
    if previous_telemetry is None:
        return None

    previous_time = float(previous_telemetry.get("curLapTime", 0.0))
    current_time = float(telemetry.get("curLapTime", previous_time))
    delta_time = current_time - previous_time

    # During reset/lap transitions TORCS can report repeated or wrapped times.
    # Ignore those for the pace term; progress reward still handles the step.
    if delta_time <= 0.0 or delta_time > 2.0:
        return None
    return delta_time


def calculate_stopped_time_delta(previous_telemetry, telemetry, track_length=None):
    delta_time = calculate_step_time_delta(previous_telemetry, telemetry)
    if delta_time is None:
        return 0.0

    speed = abs(float(telemetry.get("speedX", 0.0)))
    progress_delta = abs(
        calculate_progress_delta(previous_telemetry, telemetry, track_length)
    )
    if speed < STUCK_SPEED_LIMIT_KMH and progress_delta < STUCK_PROGRESS_LIMIT_M:
        return delta_time
    return 0.0


def get_racing_line_context(telemetry, racing_line=None):
    if racing_line is None:
        return {
            "line_error": 0.0,
            "target_speed": 216.0,
            "curvature": 0.0,
        }

    waypoint = racing_line.lookup(float(telemetry.get("distFromStart", 0.0)))
    return {
        "line_error": float(telemetry.get("trackPos", 0.0))
        - float(waypoint.get("target_track_pos", 0.0)),
        "target_speed": float(waypoint.get("target_speed_kmh", 216.0)),
        "curvature": float(waypoint.get("curvature", 0.0)),
    }


def calculate_lap_completion_fraction(episode_distance_m, racing_line=None):
    if episode_distance_m is None:
        return 0.0

    track_length = getattr(racing_line, "track_length", None)
    if not track_length:
        return 0.0

    return clamp(float(episode_distance_m) / float(track_length), 0.0, 1.0)


def build_episode_diagnostics(telemetry, episode_distance_m, racing_line=None):
    track_sensors = get_track_sensors(telemetry)
    min_track_sensor = min(track_sensors) if track_sensors else 0.0
    return {
        "lap_completion_fraction": calculate_lap_completion_fraction(
            episode_distance_m,
            racing_line,
        ),
        "final_dist_from_start_m": float(telemetry.get("distFromStart", 0.0)),
        "final_track_pos": float(telemetry.get("trackPos", 0.0)),
        "final_angle_rad": float(telemetry.get("angle", 0.0)),
        "final_speed_x_kmh": float(telemetry.get("speedX", 0.0)),
        "final_speed_y_kmh": float(telemetry.get("speedY", 0.0)),
        "final_min_track_sensor_m": float(min_track_sensor),
    }


def calculate_unsafe_edge_exit_pressure(track_position, lateral_speed, angle, action):
    edge_amount = clamp((abs(track_position) - 0.92) / 0.16, 0.0, 1.0)
    if edge_amount <= 0.0:
        return 0.0

    steering_outward = float(action.get("steer", 0.0)) * track_position > 0.03
    unstable_at_edge = abs(lateral_speed) > 7.0 or abs(angle) > 0.34
    if not steering_outward and not unstable_at_edge:
        return 0.0

    throttle_pressure = 1.25 if float(action.get("accel", 0.0)) > 0.70 else 1.0
    return edge_amount * throttle_pressure


def calculate_hybrid_reward(
    telemetry,
    action,
    *,
    previous_telemetry=None,
    previous_damage=0.0,
    racing_line=None,
    completed_lap=None,
    episode_distance_m=None,
    stuck=False,
):
    track_sensors = get_track_sensors(telemetry)
    speed = float(telemetry.get("speedX", 0.0))
    angle = float(telemetry.get("angle", 0.0))
    track_position = float(telemetry.get("trackPos", 0.0))
    lateral_speed = float(telemetry.get("speedY", 0.0))
    damage = float(telemetry.get("damage", previous_damage))
    line_context = get_racing_line_context(telemetry, racing_line)
    line_error = abs(line_context["line_error"])
    target_speed = max(40.0, line_context["target_speed"])
    curvature_abs = abs(line_context["curvature"])
    progress_delta = calculate_progress_delta(
        previous_telemetry,
        telemetry,
        getattr(racing_line, "track_length", None),
    )
    delta_time = calculate_step_time_delta(previous_telemetry, telemetry)
    aligned_speed = speed * math.cos(angle)
    overspeed_margin = 12.0 if curvature_abs < 0.008 else 5.0
    overspeed = max(0.0, speed - target_speed - overspeed_margin)
    target_mps = max(target_speed / 3.6, 8.0)
    if delta_time is None:
        pace_ratio = max(0.0, aligned_speed / 3.6) / target_mps
    else:
        pace_ratio = (progress_delta / delta_time) / target_mps
    residual_pressure = (
        abs(float(action.get("agent4_steer_residual", 0.0))) / MAX_STEER_RESIDUAL
        + abs(float(action.get("agent4_accel_residual", 0.0))) / MAX_ACCEL_RESIDUAL
        + abs(float(action.get("agent4_brake_residual", 0.0))) / MAX_BRAKE_RESIDUAL
    ) / 3.0

    reward = clamp(progress_delta, -3.0, 8.0) * REWARD_WEIGHTS["progress"]
    reward += clamp(pace_ratio, -0.5, 1.25) * REWARD_WEIGHTS["pace"]
    reward -= max(0.0, 0.62 - pace_ratio) * REWARD_WEIGHTS["slow_pace"]
    reward -= REWARD_WEIGHTS["time_step"]
    reward += (
        clamp(aligned_speed, 0.0, target_speed + 15.0)
        / (target_speed + 15.0)
        * REWARD_WEIGHTS["aligned_speed"]
    )
    reward -= (
        max(0.0, line_error - LINE_GUIDE_FREE_ERROR)
        * REWARD_WEIGHTS["racing_line_error"]
    )
    reward -= (
        max(0.0, line_error - LINE_GUIDE_LARGE_ERROR)
        * REWARD_WEIGHTS["large_racing_line_error"]
    )
    reward -= (clamp(overspeed / 70.0, 0.0, 1.0) ** 2) * REWARD_WEIGHTS["overspeed"]
    reward -= clamp(abs(lateral_speed) / 22.0, 0.0, 1.0) * REWARD_WEIGHTS["lateral_slide"]
    reward -= clamp(residual_pressure, 0.0, 1.0) * REWARD_WEIGHTS["residual_size"]
    reward -= (
        calculate_unsafe_edge_exit_pressure(
            track_position,
            lateral_speed,
            angle,
            action,
        )
        * REWARD_WEIGHTS["unsafe_edge_exit"]
    )

    if action.get("agent4_safety_shield_active", False):
        reward -= REWARD_WEIGHTS["safety_shield"]

    off_track = min(track_sensors) < 0.0 or abs(track_position) > 1.05
    crashed = damage > previous_damage
    backwards = math.cos(angle) < 0.0
    failed_episode = off_track or crashed or backwards or stuck

    if off_track:
        reward -= REWARD_WEIGHTS["off_track"]
    if crashed:
        reward -= REWARD_WEIGHTS["crash"]
    if backwards:
        reward -= REWARD_WEIGHTS["backwards"]
    if stuck:
        reward -= REWARD_WEIGHTS["stuck"]
    if failed_episode and completed_lap is None:
        lap_fraction = calculate_lap_completion_fraction(
            episode_distance_m,
            racing_line,
        )
        reward -= (1.0 - lap_fraction) * REWARD_WEIGHTS["incomplete_lap_failure"]
    if completed_lap is not None:
        reward += REWARD_WEIGHTS["lap_completed"]

    return float(reward)


def make_training_env_class(gym, spaces):
    class HybridPpoTrainingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(
            self,
            *,
            track_name=DEFAULT_TRACK_NAME,
            racing_line_path=DEFAULT_RACING_LINE_PATH,
            residual_scale=DEFAULT_RESIDUAL_SCALE,
            max_episode_steps=12000,
            target_laps=1,
            manual_start=False,
            relaunch_frequency=0,
            reset_retries=2,
            quiet_reset_log=True,
        ):
            super().__init__()
            self.track_name = track_name
            self.residual_scale = residual_scale
            self.max_episode_steps = max_episode_steps
            self.target_laps = target_laps
            self.manual_start = manual_start
            self.relaunch_frequency = relaunch_frequency
            self.reset_retries = reset_retries
            self.quiet_reset_log = quiet_reset_log
            self.runner = TorcsRunner()
            self.base_agent = MapAwareAgent(racing_line_path=racing_line_path)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(FEATURE_NAMES),),
                dtype=np.float32,
            )
            self.current_observation = None
            self.current_telemetry = None
            self.current_base_action = None
            self.initial_telemetry = None
            self.previous_telemetry = None
            self.previous_damage = 0.0
            self.lap_tracker = None
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = 0.0
            self.episode_start_time = 0.0
            self.episode_max_speed = 0.0
            self.episode_off_track_steps = 0
            self.episode_shield_steps = 0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.episodes_started = 0

        def reset(self, *, seed=None, options=None):
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

            self.current_observation = self._reset_torcs_env(relaunch=relaunch)
            self.episodes_started += 1
            self.base_agent.reset()
            self.current_telemetry = dict(self.runner.env.client.S.d)
            self.initial_telemetry = dict(self.current_telemetry)
            self.previous_telemetry = None
            self.previous_damage = float(self.current_telemetry.get("damage", 0.0))
            self.lap_tracker = LapTracker(self.current_telemetry.get("lastLapTime", 0.0))
            self.steps = 0
            self.episode_reward = 0.0
            self.episode_start_distance = float(
                self.current_telemetry.get("distRaced", 0.0)
            )
            self.episode_start_time = float(
                self.current_telemetry.get("curLapTime", 0.0)
            )
            self.episode_max_speed = float(self.current_telemetry.get("speedX", 0.0))
            self.episode_off_track_steps = 0
            self.episode_shield_steps = 0
            self.stopped_seconds = 0.0
            self.episode_max_stopped_seconds = 0.0
            self.current_base_action = self.base_agent.act(
                self.current_observation,
                self.current_telemetry,
            )

            return self._build_observation(), {}

        def step(self, residual_action):
            residual_action = np.asarray(residual_action, dtype=np.float32)
            action = apply_hybrid_residual(
                self.current_base_action,
                residual_action,
                self.current_telemetry,
                self.residual_scale,
            )
            raw_observation, _runner_reward, done, _info = (
                self.runner._step_full_control_agent(action)
            )

            telemetry = dict(raw_observation)
            current_damage = float(telemetry.get("damage", self.previous_damage))
            crashed = current_damage > self.previous_damage
            completed_lap = self.lap_tracker.update(telemetry)
            if (
                completed_lap is None
                and done
                and practice_finish_is_plausible(
                    self.initial_telemetry,
                    telemetry,
                    self.base_agent,
                )
            ):
                completed_lap = self.lap_tracker.record(telemetry.get("curLapTime", 0.0))

            stopped_time_delta = calculate_stopped_time_delta(
                self.previous_telemetry,
                telemetry,
                self.base_agent.racing_line.track_length,
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
            episode_distance_m = max(
                0.0,
                float(telemetry.get("distRaced", 0.0))
                - self.episode_start_distance,
            )

            reward = calculate_hybrid_reward(
                telemetry,
                action,
                previous_telemetry=self.previous_telemetry,
                previous_damage=self.previous_damage,
                racing_line=self.base_agent.racing_line,
                completed_lap=completed_lap,
                episode_distance_m=episode_distance_m,
                stuck=stuck,
            )
            self.previous_damage = current_damage
            self.previous_telemetry = telemetry
            self.current_telemetry = telemetry
            self.steps += 1
            self.episode_reward += reward
            self.episode_max_speed = max(
                self.episode_max_speed,
                float(telemetry.get("speedX", 0.0)),
            )

            off_track = min(get_track_sensors(telemetry)) < 0.0
            if off_track:
                self.episode_off_track_steps += 1
            if action.get("agent4_safety_shield_active", False):
                self.episode_shield_steps += 1

            terminated = bool(done)
            terminated = terminated or off_track
            left_track_bounds = abs(float(telemetry.get("trackPos", 0.0))) > 1.08
            terminated = terminated or left_track_bounds
            terminated = terminated or crashed
            backwards = math.cos(float(telemetry.get("angle", 0.0))) < 0.0
            terminated = terminated or backwards
            terminated = terminated or stuck
            if completed_lap is not None and self.lap_tracker.laps_completed >= self.target_laps:
                terminated = True

            truncated = self.steps >= self.max_episode_steps
            info = {}
            if terminated or truncated:
                reason = "truncated"
                if completed_lap is not None:
                    reason = "target_laps_completed"
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

                distance_m = episode_distance_m
                duration_seconds = max(
                    0.0,
                    float(telemetry.get("curLapTime", 0.0))
                    - self.episode_start_time,
                )
                average_speed_kmh = (
                    distance_m / duration_seconds * 3.6
                    if duration_seconds > 0.0
                    else 0.0
                )
                diagnostics = build_episode_diagnostics(
                    telemetry,
                    distance_m,
                    self.base_agent.racing_line,
                )
                info["episode_summary"] = {
                    "steps": self.steps,
                    "reward": self.episode_reward,
                    "distance_m": distance_m,
                    "duration_seconds": duration_seconds,
                    "average_speed_kmh": average_speed_kmh,
                    "max_speed_kmh": self.episode_max_speed,
                    "laps_completed": self.lap_tracker.laps_completed,
                    "best_lap_time_seconds": self.lap_tracker.best_lap_time,
                    "off_track_steps": self.episode_off_track_steps,
                    "safety_shield_steps": self.episode_shield_steps,
                    "max_stopped_seconds": self.episode_max_stopped_seconds,
                    "termination_reason": reason,
                    **diagnostics,
                }

            if not terminated and not truncated:
                self.current_base_action = self.base_agent.act(
                    self.current_observation,
                    self.current_telemetry,
                )

            return self._build_observation(), reward, terminated, truncated, info

        def close(self):
            self.base_agent.close()
            if self.manual_start:
                if self.runner.env is not None:
                    self.runner.env.end()
                return

            self.runner.shutdown()

        def _ensure_torcs(self):
            if self.runner.env is not None:
                return

            if not self.manual_start:
                self.runner.launch()
            else:
                print(
                    "Manual TORCS mode enabled. Start TORCS yourself, open "
                    "Race > Practice > New Race, and leave the SCR server "
                    "running on port 3001. Python will connect, reset between "
                    "episodes, and keep training the same PPO model."
                )
            self.runner.connect()
            self.runner.load_track(self.track_name)

        def _reset_torcs_env(self, relaunch=False):
            last_error = None
            for attempt in range(self.reset_retries + 1):
                try:
                    assert self.runner.env is not None
                    with maybe_suppress_stdout(self.quiet_reset_log):
                        return self.runner.env.reset(relaunch=relaunch)
                except Exception as error:
                    last_error = error
                    print(
                        "[RESET WARNING] TORCS reset failed "
                        f"(attempt {attempt + 1}/{self.reset_retries + 1}): {error}"
                    )
                    time.sleep(3.0)

                    if self.manual_start:
                        continue

                    try:
                        self.runner.shutdown()
                    except Exception as shutdown_error:
                        print(f"[RESET WARNING] TORCS shutdown failed: {shutdown_error}")

                    self.runner.env = None
                    self.runner.launch()
                    self.runner.connect()
                    self.runner.load_track(self.track_name)
                    relaunch = False

            raise RuntimeError(
                "TORCS reset failed repeatedly; training cannot continue"
            ) from last_error

        def _build_observation(self):
            return np.asarray(
                build_hybrid_ppo_observation(
                    self.current_telemetry,
                    self.current_base_action,
                    self.base_agent.racing_line.track_length,
                ),
                dtype=np.float32,
            )

    return HybridPpoTrainingEnv


def make_episode_summary_callback_class(BaseCallback):
    class EpisodeSummaryCallback(BaseCallback):
        def __init__(self, run_dir, progress_state=None, verbose=0):
            super().__init__(verbose=verbose)
            self.run_dir = Path(run_dir)
            self.progress_state = progress_state
            self.csv_file = None
            self.writer = None
            self.episodes_seen = 0

        def _on_training_start(self):
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.csv_file = (self.run_dir / "episodes.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "global_timestep",
                    "episodes_seen",
                    "steps",
                    "reward",
                    "distance_m",
                    "duration_seconds",
                    "average_speed_kmh",
                    "max_speed_kmh",
                    "laps_completed",
                    "best_lap_time_seconds",
                    "off_track_steps",
                    "safety_shield_steps",
                    "max_stopped_seconds",
                    "termination_reason",
                    "lap_completion_fraction",
                    "final_dist_from_start_m",
                    "final_track_pos",
                    "final_angle_rad",
                    "final_speed_x_kmh",
                    "final_speed_y_kmh",
                    "final_min_track_sensor_m",
                ],
            )
            self.writer.writeheader()
            self.csv_file.flush()

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue

                self.episodes_seen += 1
                row = {
                    "global_timestep": self.num_timesteps,
                    "episodes_seen": self.episodes_seen,
                    **summary,
                }
                self.writer.writerow(row)
                self.csv_file.flush()
                if self.progress_state is not None:
                    self.progress_state.record_episode(row)

                if self.verbose:
                    print(
                        "Episode "
                        f"{row['episodes_seen']} | "
                        f"steps={row['steps']} | "
                        f"reward={row['reward']:.2f} | "
                        f"laps={row['laps_completed']} | "
                        f"reason={row['termination_reason']}"
                    )
            return True

        def _on_training_end(self):
            if self.csv_file is not None:
                self.csv_file.close()
            self.csv_file = None
            self.writer = None

    return EpisodeSummaryCallback


def make_best_clean_lap_callback_class(BaseCallback):
    class BestCleanLapCallback(BaseCallback):
        def __init__(self, best_model_path, metadata, progress_state=None, verbose=0):
            super().__init__(verbose=verbose)
            self.best_model_path = Path(best_model_path)
            self.metadata = dict(metadata)
            self.progress_state = progress_state
            self.best_lap_time = None

        def _on_step(self):
            infos = self.locals.get("infos", [])
            for info in infos:
                summary = info.get("episode_summary") if isinstance(info, dict) else None
                if summary is None:
                    continue
                if summary.get("termination_reason") != "target_laps_completed":
                    continue
                if int(summary.get("off_track_steps", 0)) > 0:
                    continue
                if float(summary.get("max_stopped_seconds", 0.0)) >= 0.5:
                    continue

                lap_time = summary.get("best_lap_time_seconds")
                if lap_time is None:
                    lap_time = summary.get("duration_seconds")
                lap_time = float(lap_time)

                if self.best_lap_time is not None and lap_time >= self.best_lap_time:
                    continue

                self.best_lap_time = lap_time
                if self.progress_state is not None:
                    self.progress_state.record_best_clean_lap(lap_time)
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
                best_metadata = {
                    **self.metadata,
                    "best_clean_lap": {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "global_timestep": self.num_timesteps,
                        "lap_time_seconds": lap_time,
                        "episode_summary": summary,
                    },
                }
                write_json(metadata_path_for_model(self.best_model_path), best_metadata)

                if self.verbose:
                    print(
                        "Saved new best clean Agent 4 model "
                        f"({lap_time:.3f}s) to {self.best_model_path}"
                    )
            return True

    return BestCleanLapCallback


def make_console_progress_callback_class(BaseCallback):
    class ConsoleProgressCallback(BaseCallback):
        def __init__(
            self,
            total_timesteps,
            progress_state,
            *,
            interval_seconds=DEFAULT_PROGRESS_INTERVAL_SECONDS,
            verbose=0,
        ):
            super().__init__(verbose=verbose)
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
            self.last_render_time = 0.0
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

        def _render(self, *, now=None, force=False):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent 4 as a PPO residual on top of Agent 3.",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--best-model-path", type=Path, default=DEFAULT_BEST_MODEL_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tensorboard-dir", type=Path, default=DEFAULT_TENSORBOARD_DIR)
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help=(
            "Disable Stable-Baselines3 TensorBoard logging. TensorBoard is "
            "enabled by default when the package is installed."
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the clean one-line console progress display.",
    )
    parser.add_argument(
        "--verbose-training",
        action="store_true",
        help="Show Stable-Baselines3's detailed PPO training tables.",
    )
    parser.add_argument(
        "--show-torcs-reset-log",
        action="store_true",
        help="Show TORCS reset countdown/socket logs instead of hiding them.",
    )
    parser.add_argument("--track", default=DEFAULT_TRACK_NAME)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--max-episode-steps", type=int, default=12_000)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--manual-start",
        action="store_true",
        help=(
            "Do not launch or close TORCS. Connect to an already-running "
            "Practice/SCR server and reset it between learning episodes."
        ),
    )
    parser.add_argument(
        "--relaunch-frequency",
        type=int,
        default=0,
        help=(
            "In automatic mode, relaunch TORCS every N episodes. Use 0 to "
            "disable periodic relaunch."
        ),
    )
    parser.add_argument(
        "--reset-retries",
        type=int,
        default=2,
        help="Retry failed TORCS resets this many times before stopping training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=DEFAULT_LOG_STD_INIT,
        help=(
            "Initial PPO policy log standard deviation for new models. Lower "
            "values make early residual exploration gentler."
        ),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    # gym_torcs/snakeoil parses sys.argv when it opens the TORCS socket.
    # Keep this training script's CLI flags away from that legacy parser.
    sys.argv = [sys.argv[0]]

    if args.batch_size > args.n_steps:
        raise SystemExit("--batch-size must be less than or equal to --n-steps")
    if args.n_steps % args.batch_size != 0:
        print(
            "WARNING: --n-steps is not divisible by --batch-size; PPO will use "
            "a truncated final minibatch."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    (
        gym,
        spaces,
        PPO,
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    ) = import_training_dependencies()
    env_class = make_training_env_class(gym, spaces)
    run_dir = args.run_dir if args.run_dir is not None else make_default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = env_class(
        track_name=args.track,
        residual_scale=args.residual_scale,
        max_episode_steps=args.max_episode_steps,
        target_laps=args.target_laps,
        manual_start=args.manual_start,
        relaunch_frequency=args.relaunch_frequency,
        reset_retries=args.reset_retries,
        quiet_reset_log=not args.show_torcs_reset_log,
    )
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = resolve_tensorboard_log_dir(
        args.tensorboard_dir,
        disable_tensorboard=args.no_tensorboard,
    )
    resumed_metadata = None

    if args.resume and args.model_path.is_file():
        resumed_metadata = validate_resume_metadata(
            args.model_path,
            force_resume=args.force_resume,
        )
        model = PPO.load(
            str(args.model_path),
            env=env,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
        model.verbose = int(args.verbose_training)
        model.ent_coef = args.ent_coef
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=int(args.verbose_training),
            tensorboard_log=tensorboard_log,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            policy_kwargs={"log_std_init": args.log_std_init},
            seed=args.seed,
            device=args.device,
        )
        reset_num_timesteps = True

    metadata = build_training_metadata(
        args,
        resumed_from=resumed_metadata,
    )
    write_json(run_dir / "training_metadata.json", metadata)
    write_json(metadata_path_for_model(args.model_path), metadata)

    progress_state = TrainingProgressState()
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.checkpoint_dir),
        name_prefix="agent4_hybrid_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    EpisodeSummaryCallback = make_episode_summary_callback_class(BaseCallback)
    BestCleanLapCallback = make_best_clean_lap_callback_class(BaseCallback)
    callbacks = [
        checkpoint_callback,
        EpisodeSummaryCallback(
            run_dir,
            progress_state=progress_state,
            verbose=int(args.verbose_training),
        ),
        BestCleanLapCallback(
            args.best_model_path,
            metadata,
            progress_state=progress_state,
            verbose=int(args.verbose_training),
        ),
    ]
    if not args.no_progress_bar and not args.verbose_training:
        ConsoleProgressCallback = make_console_progress_callback_class(BaseCallback)
        callbacks.append(
            ConsoleProgressCallback(
                args.total_timesteps,
                progress_state,
            )
        )
    callback = CallbackList(callbacks)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(str(args.model_path))
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "training_metadata.json", metadata)
        write_json(metadata_path_for_model(args.model_path), metadata)
        print(f"Saved Agent 4 model to {args.model_path}")
        print(f"Saved training run logs to {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
