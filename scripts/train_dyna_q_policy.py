from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.dyna_q_agent import (  # noqa: E402
    DEFAULT_BEST_POLICY_PATH,
    DEFAULT_FINAL_POLICY_PATH,
    DEFAULT_LATEST_POLICY_PATH,
    DynaQLearningAgent,
    G_TRACK_3_LENGTH_METRES,
)
from gui.torcs_config import TorcsRaceSetup, TorcsRuntimeConfig  # noqa: E402
from runner.torcs_runner import TorcsRunner  # noqa: E402


DEFAULT_TRACK_ID = "g-track-3"
DEFAULT_TRACK_CATEGORY = "road"
DEFAULT_CAR_ID = "car1-ow1"
VISIBLE_EPISODE_ROWS = 14
DEFAULT_PREVIOUS_POLICY_PATH = DEFAULT_LATEST_POLICY_PATH.with_name(
    "dyna_q_g_track_3_previous.json"
)


@dataclass(frozen=True)
class EpisodeSummary:
    episode: int
    reason: str
    steps: int
    laps: int
    best_lap: float | None
    progress_m: float
    progress_percent: float
    q_states: int
    epsilon: float
    final_speed: float | None = None
    final_track_pos: float | None = None
    final_angle: float | None = None
    final_front: float | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the from-scratch Dyna-Q policy in TORCS and save the latest "
            "Q-table for the finalised agent."
        )
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--track", default=DEFAULT_TRACK_ID)
    parser.add_argument("--track-category", default=DEFAULT_TRACK_CATEGORY)
    parser.add_argument("--car", default=DEFAULT_CAR_ID)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_LATEST_POLICY_PATH)
    parser.add_argument("--best-policy-path", type=Path, default=DEFAULT_BEST_POLICY_PATH)
    parser.add_argument(
        "--previous-policy-path",
        type=Path,
        default=DEFAULT_PREVIOUS_POLICY_PATH,
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete the selected policy before episode 1, then train from empty Q-values.",
    )
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Copy the best checkpoint to the latest policy, then continue training.",
    )
    parser.add_argument(
        "--exploration-boost",
        type=float,
        default=None,
        help="Temporarily override epsilon during this run, for example 0.12.",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=2,
        help="How many times to relaunch TORCS if the SCR socket does not reset.",
    )
    parser.add_argument(
        "--no-dynamic-console",
        action="store_true",
        help="Print one row per episode instead of redrawing a compact dashboard.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable automatic previous/best policy checkpoints.",
    )
    parser.add_argument("--promote-final", action="store_true")
    args = parser.parse_args()
    if args.from_scratch and args.resume_best:
        parser.error("--from-scratch and --resume-best cannot be used together")
    if args.exploration_boost is not None and not 0.0 <= args.exploration_boost <= 1.0:
        parser.error("--exploration-boost must be between 0.0 and 1.0")

    # gym_torcs/snakeoil reads sys.argv internally and rejects script-level
    # flags such as --from-scratch. Keep only the program name before TORCS
    # creates its legacy client.
    sys.argv = [sys.argv[0]]

    if args.from_scratch and args.policy_path.is_file():
        args.policy_path.unlink()
        print(f"Deleted existing policy before from-scratch training: {args.policy_path}")
        if not args.no_checkpoints and args.best_policy_path.is_file():
            args.best_policy_path.unlink()
            print(f"Deleted existing best checkpoint: {args.best_policy_path}")
    elif args.resume_best:
        previous_saved = _resume_best_policy(
            policy_path=args.policy_path,
            best_policy_path=args.best_policy_path,
            previous_policy_path=args.previous_policy_path,
            save_previous=not args.no_checkpoints,
        )
        if previous_saved:
            print(f"Saved pre-run policy checkpoint: {args.previous_policy_path}")
        print(f"Resumed latest policy from best checkpoint: {args.best_policy_path}")
    elif not args.no_checkpoints and args.policy_path.is_file():
        args.previous_policy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.policy_path, args.previous_policy_path)
        _annotate_policy(args.previous_policy_path, checkpoint_type="previous")
        print(f"Saved pre-run policy checkpoint: {args.previous_policy_path}")
    if args.exploration_boost is not None:
        print(f"Temporary exploration override: epsilon={args.exploration_boost:.3f}")

    best_lap = None
    summaries: list[EpisodeSummary] = []
    setup = TorcsRaceSetup(
        track_id=args.track,
        track_category=args.track_category,
        car_id=args.car,
    )
    episode_total = max(1, args.episodes)
    console = TrainingConsole(
        total_episodes=episode_total,
        policy_path=args.policy_path,
        dynamic=not args.no_dynamic_console,
    )
    console.render(summaries, "Starting TORCS...")

    with TorcsRuntimeConfig(setup):
        runner = None
        try:
            console.render(summaries, "Launching TORCS...")
            runner = _start_runner(args.track)
            episode = 1
            retry_count = 0
            while episode <= episode_total:
                console.render(summaries, f"Running episode {episode}/{episode_total}...")
                agent = DynaQLearningAgent(
                    policy_path=args.policy_path,
                    load_existing=True,
                )
                if args.exploration_boost is not None:
                    agent.exploration_override = args.exploration_boost
                try:
                    results = runner.run(agent, shutdown_on_finish=False)
                except (ConnectionError, OSError) as error:
                    if retry_count >= max(0, args.reconnect_attempts):
                        raise
                    retry_count += 1
                    console.render(
                        summaries,
                        (
                            f"SCR socket reset before episode {episode}; "
                            f"relaunching TORCS ({retry_count}/"
                            f"{max(0, args.reconnect_attempts)})..."
                        ),
                    )
                    runner.shutdown()
                    runner = _start_runner(args.track)
                    continue

                retry_count = 0

                lap_time = results.get("best_lap_time_seconds")
                if isinstance(lap_time, (int, float)):
                    best_lap = (
                        lap_time
                        if best_lap is None
                        else min(best_lap, float(lap_time))
                    )
                summary = _episode_summary(episode, results, agent)
                summaries.append(summary)
                if not args.no_checkpoints:
                    _maybe_update_best_policy(
                        args.policy_path,
                        args.best_policy_path,
                        summary.progress_m,
                    )
                console.render(
                    summaries,
                    f"Completed episode {episode}/{episode_total}.",
                )
                episode += 1
        finally:
            if runner is not None:
                runner.shutdown()

    console.render(
        summaries,
        f"Finished {len(summaries)}/{episode_total} episodes.",
        final=True,
    )
    print()
    print(f"Latest policy: {args.policy_path} | episodes={episode_total}")
    if args.policy_path.is_file():
        print(f"Learning totals: {_policy_summary(args.policy_path)}")
    if not args.no_checkpoints and args.best_policy_path.is_file():
        print(f"Best checkpoint: {args.best_policy_path} | {_best_policy_summary(args.best_policy_path)}")

    if args.promote_final and args.policy_path.is_file():
        DEFAULT_FINAL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.policy_path, DEFAULT_FINAL_POLICY_PATH)
        print(f"Promoted latest policy to {DEFAULT_FINAL_POLICY_PATH}")

    if best_lap is not None:
        print(f"Best completed lap during this training run: {best_lap:.3f}s")
    return 0


class TrainingConsole:
    def __init__(
        self,
        *,
        total_episodes: int,
        policy_path: Path,
        dynamic: bool,
    ) -> None:
        self.total_episodes = total_episodes
        self.policy_path = policy_path
        self.dynamic = dynamic and sys.stdout.isatty()
        self._printed_header = False
        self._printed_rows = 0

    def render(
        self,
        summaries: list[EpisodeSummary],
        status: str,
        *,
        final: bool = False,
    ) -> None:
        dashboard = _dashboard_text(
            summaries,
            total_episodes=self.total_episodes,
            policy_path=self.policy_path,
            status=status,
        )
        if self.dynamic:
            print("\033[2J\033[H" + dashboard, end="", flush=True)
            if final:
                print()
            return

        if not self._printed_header:
            print(_static_table_header(self.total_episodes, self.policy_path))
            self._printed_header = True
        for summary in summaries[self._printed_rows :]:
            print(_episode_row(summary))
        self._printed_rows = len(summaries)


def _start_runner(track: str) -> TorcsRunner:
    runner = TorcsRunner()
    runner.launch()
    runner.connect()
    runner.load_track(track)
    return runner


def _episode_summary(
    episode: int,
    results: dict[str, object],
    agent: DynaQLearningAgent,
) -> EpisodeSummary:
    progress_m = _episode_progress_m(results)
    return EpisodeSummary(
        episode=episode,
        reason=str(results.get("termination_reason", ""))[:19],
        steps=int(results.get("steps", 0)),
        laps=int(results.get("laps_completed", 0)),
        best_lap=_optional_float(results.get("best_lap_time_seconds")),
        progress_m=progress_m,
        progress_percent=min(100.0, progress_m / G_TRACK_3_LENGTH_METRES * 100.0),
        q_states=len(agent.q_values),
        epsilon=agent.exploration_rate(),
        final_speed=_last_sample_float(results, "speed_x"),
        final_track_pos=_last_sample_float(results, "track_pos"),
        final_angle=_last_sample_float(results, "angle"),
        final_front=_last_sample_float(results, "front_sensor"),
    )


def _dashboard_text(
    summaries: list[EpisodeSummary],
    *,
    total_episodes: int,
    policy_path: Path,
    status: str,
) -> str:
    best_progress = max((summary.progress_m for summary in summaries), default=0.0)
    best_percent = min(100.0, best_progress / G_TRACK_3_LENGTH_METRES * 100.0)
    best_lap = min(
        (
            summary.best_lap
            for summary in summaries
            if summary.best_lap is not None
        ),
        default=None,
    )
    lap_text = "--" if best_lap is None else f"{best_lap:.3f}s"
    hidden_rows = max(0, len(summaries) - VISIBLE_EPISODE_ROWS)
    visible = summaries[-VISIBLE_EPISODE_ROWS:]
    lines = [
        "Dyna-Q TORCS Training",
        f"Status: {status}",
        f"Policy: {policy_path}",
        (
            f"Episodes: {len(summaries)}/{total_episodes} | "
            f"Best progress: {best_progress:.0f}m ({best_percent:.1f}%) | "
            f"Best lap: {lap_text}"
        ),
        "",
        (
            "Episode | Result              | Steps | Progress | Lap % | Laps | EndV | "
            "EndPos | EndAng | Front | Q states | Epsilon"
        ),
        (
            "--------+---------------------+-------+----------+-------+------+------+"
            "--------+--------+-------+----------+--------"
        ),
    ]
    if hidden_rows:
        lines.append(f"... {hidden_rows} earlier episode rows hidden ...")
    lines.extend(_episode_row(summary) for summary in visible)
    if not visible:
        lines.append(
            "      -- | waiting             |    -- |       -- |    -- |   -- |   -- |"
            "     -- |     -- |    -- |       -- | --"
        )
    return "\n".join(lines) + "\n"


def _episode_row(summary: EpisodeSummary) -> str:
    lap_percent = min(100.0, summary.progress_percent)
    return (
        f"{summary.episode:7d} | "
        f"{summary.reason:<19} | "
        f"{summary.steps:5d} | "
        f"{summary.progress_m:7.0f}m | "
        f"{lap_percent:5.1f} | "
        f"{summary.laps:4d} | "
        f"{_format_optional_float(summary.final_speed, width=4, precision=0)} | "
        f"{_format_optional_float(summary.final_track_pos, width=6, precision=2, sign=True)} | "
        f"{_format_optional_float(summary.final_angle, width=6, precision=2, sign=True)} | "
        f"{_format_optional_float(summary.final_front, width=5, precision=0)} | "
        f"{summary.q_states:8d} | "
        f"{summary.epsilon:.3f}"
    )


def _static_table_header(total_episodes: int, policy_path: Path) -> str:
    return "\n".join(
        [
            "Dyna-Q TORCS Training",
            f"Policy: {policy_path}",
            f"Episodes: 0/{total_episodes}",
            "",
            (
                "Episode | Result              | Steps | Progress | Lap % | Laps | EndV | "
                "EndPos | EndAng | Front | Q states | Epsilon"
            ),
            (
                "--------+---------------------+-------+----------+-------+------+------+"
                "--------+--------+-------+----------+--------"
            ),
        ]
    )


def _episode_progress_m(
    results: dict[str, object],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> float:
    samples = results.get("telemetry_samples")
    if not isinstance(samples, list) or not samples:
        return 0.0

    dist_raced = [
        value
        for sample in samples
        if isinstance(sample, dict)
        for value in [_optional_float(sample.get("dist_raced"))]
        if value is not None
    ]
    if len(dist_raced) >= 2:
        return max(0.0, min(track_length_m, max(dist_raced) - dist_raced[0]))

    distances = [
        value % track_length_m
        for sample in samples
        if isinstance(sample, dict)
        for value in [_optional_float(sample.get("dist_from_start"))]
        if value is not None
    ]
    if len(distances) < 2:
        return 0.0
    start = distances[0]
    progress_values = [
        (distance - start) % track_length_m
        for distance in distances
    ]
    return max(0.0, min(track_length_m, max(progress_values)))


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _last_sample_float(results: dict[str, object], key: str) -> float | None:
    samples = results.get("telemetry_samples")
    if not isinstance(samples, list) or not samples:
        return None
    sample = samples[-1]
    if not isinstance(sample, dict):
        return None
    return _optional_float(sample.get(key))


def _format_optional_float(
    value: float | None,
    *,
    width: int,
    precision: int,
    sign: bool = False,
) -> str:
    if value is None:
        return "--".rjust(width)
    sign_flag = "+" if sign else ""
    return f"{value:{sign_flag}{width}.{precision}f}"


def _policy_summary(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        f"q_values={len(payload.get('q_values', {}))} | "
        f"model={len(payload.get('model', {}))} | "
        f"trained_steps={payload.get('steps_trained')} | "
        f"epsilon={float(payload.get('epsilon', 0.0)):.3f} | "
        f"planning_updates={payload.get('planning_updates')}"
    )


def _maybe_update_best_policy(
    policy_path: Path,
    best_policy_path: Path,
    progress_m: float,
) -> bool:
    if not policy_path.is_file():
        return False

    previous_best = _policy_best_progress(best_policy_path)
    if previous_best is not None and progress_m <= previous_best:
        return False

    best_policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(policy_path, best_policy_path)
    _annotate_policy(
        best_policy_path,
        checkpoint_type="best",
        best_progress_m=progress_m,
        best_progress_percent=min(100.0, progress_m / G_TRACK_3_LENGTH_METRES * 100.0),
    )
    return True


def _resume_best_policy(
    *,
    policy_path: Path,
    best_policy_path: Path,
    previous_policy_path: Path,
    save_previous: bool,
) -> bool:
    if not best_policy_path.is_file():
        raise FileNotFoundError(
            f"No best Dyna-Q checkpoint found at {best_policy_path}"
        )

    previous_saved = False
    if save_previous and policy_path.is_file():
        previous_policy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(policy_path, previous_policy_path)
        _annotate_policy(previous_policy_path, checkpoint_type="previous")
        previous_saved = True

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_policy_path, policy_path)
    _annotate_policy(
        policy_path,
        checkpoint_type="latest",
        resumed_from_best=str(best_policy_path),
    )
    return previous_saved


def _annotate_policy(path: Path, **metadata: object) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(metadata)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _policy_best_progress(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _optional_float(payload.get("best_progress_m"))


def _best_policy_summary(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_progress = _optional_float(payload.get("best_progress_m")) or 0.0
    best_percent = min(100.0, best_progress / G_TRACK_3_LENGTH_METRES * 100.0)
    return (
        f"best_progress={best_progress:.0f}m ({best_percent:.1f}%) | "
        f"q_values={len(payload.get('q_values', {}))} | "
        f"trained_steps={payload.get('steps_trained')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
