from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.n_step_td3_agent import default_model_path
from n_step_td3.contracts import (
    ACTION_VERSION,
    REWARD_VERSION,
)
from n_step_td3.environment import NstepTorcsEnvironment
from n_step_td3.learner import (
    NstepTd3Config,
    load_actor_checkpoint,
    resolve_device,
)
from n_step_td3.networks import Actor
from n_step_td3.racing_line import LOOKAHEAD_DISTANCES_M


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "agent7_n_step_td3_v3"
STEERING_RATE_V4_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "agent7_n_step_td3_v4"
)
PACE_V5_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "agent7_n_step_td3_v5"
)
SENSOR_ONLY_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "agent8_sensor_n_step_td3"
)
TRACE_CURVATURE_COLUMNS = tuple(
    f"lookahead_curvature_{int(distance)}m"
    for distance in LOOKAHEAD_DISTANCES_M
)
TRACE_COLUMNS = (
    "repeat",
    "episode",
    "step",
    "distance_m",
    "distance_from_start_m",
    "speed_kmh",
    "heading_angle_rad",
    "track_position",
    "target_track_position",
    "racing_line_error",
    "minimum_track_sensor_m",
    "raw_steer",
    "control_steer",
    "steering_delta",
    "raw_longitudinal",
    "control_accel",
    "control_brake",
    "control_gear",
    "reward",
    "reward_forward_velocity",
    "reward_racing_line_factor",
    "reward_steering_rate_penalty",
    "reward_terminal_penalty",
    "reward_progress_m",
    "reward_lap_completion_bonus",
    *TRACE_CURVATURE_COLUMNS,
    "physical_failure",
    "termination_reason",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an N-step TD3 racing checkpoint."
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=default_model_path(),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--track", default="g-track-3")
    parser.add_argument("--racing-line-path", type=Path)
    parser.add_argument("--max-episode-steps", type=int, default=25_000)
    parser.add_argument(
        "--observation-noise-std",
        type=float,
        default=0.025,
        help=(
            "Observation-noise profile level: 0.025 selects torcsRL's "
            "feature-specific reference profile; 0 disables noise."
        ),
    )
    parser.add_argument(
        "--physical-steering-limit",
        type=float,
        help=(
            "Override the checkpoint's physical TORCS steering scale for this "
            "evaluation only. The actor weights, reward, and observations are "
            "unchanged."
        ),
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manual-start", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--trace-distance-range",
        type=float,
        nargs=2,
        metavar=("START_M", "END_M"),
        help="Write per-step diagnostic telemetry within this distance range.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.policy_path.is_file():
        parser.error(f"policy checkpoint does not exist: {args.policy_path}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if (
        not math.isfinite(args.observation_noise_std)
        or args.observation_noise_std < 0.0
    ):
        parser.error("--observation-noise-std must be finite and non-negative")
    if args.physical_steering_limit is not None and (
        not math.isfinite(args.physical_steering_limit)
        or not 0.0 < args.physical_steering_limit <= 1.0
    ):
        parser.error(
            "--physical-steering-limit must be finite and in (0, 1]"
        )
    if args.racing_line_path is not None and not args.racing_line_path.is_file():
        parser.error(
            f"--racing-line-path does not exist: {args.racing_line_path}"
        )
    if args.trace_distance_range is not None:
        start, end = args.trace_distance_range
        if not math.isfinite(start) or not math.isfinite(end):
            parser.error("--trace-distance-range values must be finite")
        if start < 0.0 or end <= start:
            parser.error(
                "--trace-distance-range requires 0 <= START_M < END_M"
            )
    return args


def build_trace_row(
    repeat: int,
    reward: float,
    info: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "repeat": int(repeat),
        "episode": int(info["episode"]),
        "step": int(info["steps"]),
        "distance_m": float(info["distance_m"]),
        "distance_from_start_m": float(info["distance_from_start_m"]),
        "speed_kmh": float(info["speed_kmh"]),
        "heading_angle_rad": float(info["heading_angle_rad"]),
        "track_position": float(info["track_position"]),
        "target_track_position": float(info["target_track_position"]),
        "racing_line_error": float(info["racing_line_error"]),
        "minimum_track_sensor_m": float(info["minimum_track_sensor_m"]),
        "raw_steer": float(info["raw_steer"]),
        "control_steer": float(info["control_steer"]),
        "steering_delta": float(info["steering_delta"]),
        "raw_longitudinal": float(info["raw_longitudinal"]),
        "control_accel": float(info["control_accel"]),
        "control_brake": float(info["control_brake"]),
        "control_gear": int(info["control_gear"]),
        "reward": float(reward),
        "reward_forward_velocity": float(info["reward_forward_velocity"]),
        "reward_racing_line_factor": float(
            info["reward_racing_line_factor"]
        ),
        "reward_steering_rate_penalty": float(
            info["reward_steering_rate_penalty"]
        ),
        "reward_terminal_penalty": float(info["reward_terminal_penalty"]),
        "reward_progress_m": float(info["reward_progress_m"]),
        "reward_lap_completion_bonus": float(
            info["reward_lap_completion_bonus"]
        ),
    }
    curvature = np.asarray(info["lookahead_curvature"], dtype=np.float32)
    if curvature.shape != (len(TRACE_CURVATURE_COLUMNS),):
        raise ValueError("invalid look-ahead curvature diagnostic shape")
    row.update(
        {
            name: float(value)
            for name, value in zip(TRACE_CURVATURE_COLUMNS, curvature)
        }
    )
    row["physical_failure"] = bool(info["physical_failure"])
    row["termination_reason"] = str(info["termination_reason"])
    return row


class DeterministicCheckpointPolicy:
    def __init__(self, path: Path, device: str) -> None:
        checkpoint = load_actor_checkpoint(path, device="cpu")
        config = NstepTd3Config(**checkpoint["config"])
        self.config = config
        self.model_family = str(checkpoint["model_family"])
        self.observation_version = str(checkpoint["observation_version"])
        self.action_version = str(checkpoint.get("action_version", ACTION_VERSION))
        self.reward_version = str(checkpoint.get("reward_version", REWARD_VERSION))
        self.device = resolve_device(device)
        self.actor = Actor(
            config.observation_size,
            config.action_size,
            (config.hidden_size_1, config.hidden_size_2),
        ).to(self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        value = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)
        return self.actor(value).squeeze(0).cpu().numpy().astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    policy = DeterministicCheckpointPolicy(args.policy_path, args.device)
    physical_steering_limit = (
        policy.config.physical_steering_limit
        if args.physical_steering_limit is None
        else args.physical_steering_limit
    )
    if args.output_dir == DEFAULT_OUTPUT_DIR and not policy.config.racing_line_features:
        args.output_dir = SENSOR_ONLY_OUTPUT_DIR
    elif (
        args.output_dir == DEFAULT_OUTPUT_DIR
        and policy.config.progress_reward
    ):
        args.output_dir = PACE_V5_OUTPUT_DIR
    elif (
        args.output_dir == DEFAULT_OUTPUT_DIR
        and policy.config.steering_rate_cost_coefficient > 0.0
    ):
        args.output_dir = STEERING_RATE_V4_OUTPUT_DIR
    environment = NstepTorcsEnvironment(
        track_name=args.track,
        max_episode_steps=args.max_episode_steps,
        manual_start=args.manual_start,
        observation_noise_std=args.observation_noise_std,
        seed=args.seed,
        racing_line_path=args.racing_line_path,
        steering_rate_cost_coefficient=(
            policy.config.steering_rate_cost_coefficient
        ),
        physical_steering_limit=physical_steering_limit,
        progress_reward=policy.config.progress_reward,
        racing_line_features=policy.config.racing_line_features,
    )
    rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    try:
        for repeat in range(1, args.repeats + 1):
            observation, _ = environment.reset()
            done = False
            while not done:
                action = policy.act(observation)
                observation, reward, terminated, truncated, info = environment.step(
                    action
                )
                if args.trace_distance_range is not None:
                    start, end = args.trace_distance_range
                    distance = float(info["distance_m"])
                    if start <= distance <= end:
                        trace_rows.append(build_trace_row(repeat, reward, info))
                done = terminated or truncated
            summary = environment.last_summary
            if summary is None:
                raise RuntimeError("N-step TD3 evaluation ended without a summary")
            row = {"repeat": repeat, **summary.as_dict()}
            rows.append(row)
            print(
                f"Evaluation {repeat}/{args.repeats}: "
                f"{summary.distance_m:.1f}m, reward={summary.reward:.1f}, "
                f"laps={summary.laps_completed}, "
                f"ended={summary.termination_reason}"
            )
    finally:
        environment.close()

    distances = [float(row["distance_m"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    lap_times = [
        float(row["best_lap_time_seconds"])
        for row in rows
        if int(row["laps_completed"]) > 0
        and row["best_lap_time_seconds"] is not None
    ]
    summary_record = {
        "policy_path": str(args.policy_path.resolve()),
        "track": args.track,
        "model_family": policy.model_family,
        "observation_version": policy.observation_version,
        "action_version": policy.action_version,
        "reward_version": policy.reward_version,
        "steering_rate_cost_coefficient": (
            policy.config.steering_rate_cost_coefficient
        ),
        "physical_steering_limit": physical_steering_limit,
        "checkpoint_physical_steering_limit": (
            policy.config.physical_steering_limit
        ),
        "progress_reward": policy.config.progress_reward,
        "racing_line_features": policy.config.racing_line_features,
        "racing_line_path": (
            None
            if environment.racing_line_path is None
            else str(environment.racing_line_path)
        ),
        "deterministic": True,
        "repeats": args.repeats,
        "median_distance_m": statistics.median(distances),
        "minimum_distance_m": min(distances),
        "maximum_distance_m": max(distances),
        "mean_reward": statistics.fmean(rewards),
        "completed_laps": sum(int(row["laps_completed"]) for row in rows),
        "completion_rate": len(lap_times) / len(rows),
        "median_lap_time_seconds": (
            None if not lap_times else statistics.median(lap_times)
        ),
        "fastest_lap_time_seconds": None if not lap_times else min(lap_times),
        "episodes": rows,
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"n_step_td3_evaluation_{timestamp}.json"
    csv_path = args.output_dir / f"n_step_td3_evaluation_{timestamp}.csv"
    trace_path: Path | None = None
    if args.trace_distance_range is not None:
        trace_path = args.output_dir / f"n_step_td3_trace_{timestamp}.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=TRACE_COLUMNS)
            writer.writeheader()
            writer.writerows(trace_rows)
        summary_record["trace_distance_range_m"] = args.trace_distance_range
        summary_record["trace_rows"] = len(trace_rows)
        summary_record["trace_csv"] = str(trace_path.resolve())
    json_path.write_text(json.dumps(summary_record, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    median_lap_time = summary_record["median_lap_time_seconds"]
    lap_time_summary = (
        "n/a" if median_lap_time is None else f"{float(median_lap_time):.3f}s"
    )
    print(
        f"{'Agent 7' if policy.config.racing_line_features else 'Agent 8'} verdict: "
        f"median={summary_record['median_distance_m']:.1f}m, "
        f"minimum={summary_record['minimum_distance_m']:.1f}m, "
        f"maximum={summary_record['maximum_distance_m']:.1f}m, "
        f"laps={summary_record['completed_laps']}/{args.repeats}, "
        f"median_lap={lap_time_summary}"
    )
    print(f"Evaluation JSON: {json_path}")
    print(f"Evaluation CSV: {csv_path}")
    if trace_path is not None:
        print(f"Diagnostic trace CSV: {trace_path}")


if __name__ == "__main__":
    main()
