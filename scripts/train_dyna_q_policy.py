from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from io import StringIO
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.dyna_q_agent import (  # noqa: E402
    DEFAULT_BEST_POLICY_PATH,
    DEFAULT_FINAL_POLICY_PATH,
    DEFAULT_LATEST_POLICY_PATH,
    DynaQFinalisedAgent,
    DynaQLearningAgent,
    G_TRACK_3_LENGTH_METRES,
)
from gui.torcs_config import TorcsRaceSetup, TorcsRuntimeConfig  # noqa: E402


DEFAULT_TRACK_ID = "g-track-3"
DEFAULT_TRACK_CATEGORY = "road"
DEFAULT_CAR_ID = "car1-ow1"
VISIBLE_EPISODE_ROWS = 8
DEFAULT_PREVIOUS_POLICY_PATH = DEFAULT_LATEST_POLICY_PATH.with_name(
    "dyna_q_g_track_3_previous.json"
)
DEFAULT_POLICY_ARCHIVE_DIR = DEFAULT_LATEST_POLICY_PATH.parent / "archive"


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
    off_track: int | None = None
    avg_speed: float | None = None


@dataclass(frozen=True)
class PolicyQuality:
    completed_laps: int
    best_lap: float | None
    off_track: int | None
    avg_speed: float | None
    progress_m: float
    evaluation_repeats: int | None = None
    completed_repeats: int | None = None


@dataclass(frozen=True)
class FinalPromotionDecision:
    promoted: bool
    confidence: str
    reason: str
    previous_final: PolicyQuality | None
    candidate: PolicyQuality | None
    archive_path: Path | None = None


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
    parser.add_argument(
        "--gate-best-with-evaluation",
        action="store_true",
        help=(
            "Before updating the best checkpoint, run the saved policy once as "
            "a finalised greedy agent and score the checkpoint using that result."
        ),
    )
    parser.add_argument(
        "--best-evaluation-frequency",
        type=int,
        default=1,
        help="When evaluation gating is enabled, evaluate every N episodes.",
    )
    parser.add_argument(
        "--best-evaluation-repeats",
        type=int,
        default=1,
        help=(
            "When evaluation gating is enabled, run N greedy evaluations and "
            "promote only from the repeated aggregate result."
        ),
    )
    parser.add_argument(
        "--best-evaluation-min-progress",
        type=float,
        default=0.0,
        help=(
            "Minimum greedy evaluation progress in metres required before an "
            "evaluation-gated best checkpoint can be promoted."
        ),
    )
    parser.add_argument(
        "--best-evaluation-max-off-track",
        type=int,
        default=None,
        help=(
            "Optional maximum off-track sample count allowed before an "
            "evaluation-gated best checkpoint can be promoted."
        ),
    )
    parser.add_argument(
        "--best-evaluation-min-completions",
        type=int,
        default=1,
        help=(
            "Minimum completed greedy evaluation laps required before an "
            "evaluation-gated best checkpoint can be promoted. Use 1 to allow "
            "breakthrough candidates while still rejecting policies that never "
            "finish the map."
        ),
    )
    parser.add_argument(
        "--candidate-min-training-laps",
        type=int,
        default=0,
        help=(
            "When evaluation gating is enabled, skip the expensive greedy "
            "evaluation unless the learning episode completed at least this "
            "many laps. Use 1 when improving an already lap-capable policy."
        ),
    )
    parser.add_argument(
        "--auto-promote-final",
        action="store_true",
        help=(
            "After training, compare best against the current final policy. If "
            "best is better, archive final locally, copy best to final, and sync "
            "latest to best. Git is never touched."
        ),
    )
    parser.add_argument(
        "--restore-latest-after-rejected-evaluation",
        action="store_true",
        help=(
            "When evaluation gating rejects a candidate, copy the protected "
            "best checkpoint back to the latest policy before the next episode."
        ),
    )
    parser.add_argument(
        "--final-policy-path",
        type=Path,
        default=DEFAULT_FINAL_POLICY_PATH,
        help="Policy file used by --auto-promote-final.",
    )
    parser.add_argument(
        "--policy-archive-dir",
        type=Path,
        default=DEFAULT_POLICY_ARCHIVE_DIR,
        help="Archive directory used before replacing a final policy.",
    )
    parser.add_argument(
        "--no-final-archive",
        action="store_true",
        help="Disable archiving the old final policy during auto-promotion.",
    )
    parser.add_argument("--promote-final", action="store_true")
    args = parser.parse_args()
    if args.from_scratch and args.resume_best:
        parser.error("--from-scratch and --resume-best cannot be used together")
    if args.exploration_boost is not None and not 0.0 <= args.exploration_boost <= 1.0:
        parser.error("--exploration-boost must be between 0.0 and 1.0")
    if args.best_evaluation_frequency < 1:
        parser.error("--best-evaluation-frequency must be at least 1")
    if args.best_evaluation_repeats < 1:
        parser.error("--best-evaluation-repeats must be at least 1")
    if args.best_evaluation_min_progress < 0.0:
        parser.error("--best-evaluation-min-progress cannot be negative")
    if (
        args.best_evaluation_max_off_track is not None
        and args.best_evaluation_max_off_track < 0
    ):
        parser.error("--best-evaluation-max-off-track cannot be negative")
    if args.best_evaluation_min_completions < 0:
        parser.error("--best-evaluation-min-completions cannot be negative")
    if args.candidate_min_training_laps < 0:
        parser.error("--candidate-min-training-laps cannot be negative")
    if args.candidate_min_training_laps and not args.gate_best_with_evaluation:
        parser.error("--candidate-min-training-laps requires --gate-best-with-evaluation")
    if args.restore_latest_after_rejected_evaluation and args.no_checkpoints:
        parser.error(
            "--restore-latest-after-rejected-evaluation requires checkpoints"
        )
    if (
        args.restore_latest_after_rejected_evaluation
        and not args.gate_best_with_evaluation
    ):
        parser.error(
            "--restore-latest-after-rejected-evaluation requires "
            "--gate-best-with-evaluation"
        )

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
            runner = _start_runner(args.track, quiet=console.dynamic)
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
                    with _quiet_stdout(console.dynamic):
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
                    with _quiet_stdout(console.dynamic):
                        runner.shutdown()
                    runner = _start_runner(args.track, quiet=console.dynamic)
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
                    evaluation_results = None
                    evaluation_progress_m = None
                    should_update_best = not args.gate_best_with_evaluation
                    if (
                        args.gate_best_with_evaluation
                        and episode % args.best_evaluation_frequency == 0
                    ):
                        if _candidate_training_gate_passed(
                            summary,
                            min_training_laps=args.candidate_min_training_laps,
                        ):
                            console.render(
                                summaries,
                                (
                                    "Evaluating saved policy greedily before best "
                                    f"checkpoint update after episode {episode}..."
                                ),
                            )
                            runner, evaluation_results = _run_gated_evaluations(
                                runner,
                                policy_path=args.policy_path,
                                track=args.track,
                                reconnect_attempts=args.reconnect_attempts,
                                console=console,
                                summaries=summaries,
                                episode=episode,
                                repeats=args.best_evaluation_repeats,
                            )
                            evaluation_progress_m = _results_progress_m(
                                evaluation_results
                            )
                            should_update_best = True
                        else:
                            restored = False
                            if args.restore_latest_after_rejected_evaluation:
                                restored = _restore_latest_from_best_policy(
                                    policy_path=args.policy_path,
                                    best_policy_path=args.best_policy_path,
                                    episode=episode,
                                    reason="skipped_incomplete_training_lap",
                                )
                            message = (
                                "Skipped greedy evaluation because the learning "
                                f"episode completed {summary.laps} lap(s)."
                            )
                            if restored:
                                message += " Restored latest policy from best."
                            console.render(summaries, message)
                    if should_update_best:
                        updated_best = _maybe_update_best_policy(
                            args.policy_path,
                            args.best_policy_path,
                            summary.progress_m,
                            evaluation_progress_m=evaluation_progress_m,
                            evaluation_results=evaluation_results,
                            min_evaluation_progress_m=(
                                args.best_evaluation_min_progress
                            ),
                            max_evaluation_off_track=(
                                args.best_evaluation_max_off_track
                            ),
                            min_evaluation_completions=(
                                args.best_evaluation_min_completions
                            ),
                        )
                        if (
                            args.restore_latest_after_rejected_evaluation
                            and evaluation_results is not None
                            and not updated_best
                        ):
                            restored = _restore_latest_from_best_policy(
                                policy_path=args.policy_path,
                                best_policy_path=args.best_policy_path,
                                episode=episode,
                            )
                            if restored:
                                console.render(
                                    summaries,
                                    (
                                        "Candidate rejected; restored latest policy "
                                        f"from best after episode {episode}."
                                    ),
                                )
                console.render(
                    summaries,
                    f"Completed episode {episode}/{episode_total}.",
                )
                episode += 1
        finally:
            if runner is not None:
                with _quiet_stdout(console.dynamic):
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

    if args.auto_promote_final:
        decision = _auto_promote_final_policy(
            best_policy_path=args.best_policy_path,
            final_policy_path=args.final_policy_path,
            latest_policy_path=args.policy_path,
            archive_dir=args.policy_archive_dir,
            archive_existing=not args.no_final_archive,
        )
        print()
        print(_final_promotion_summary(decision))

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
        self._dynamic_line_count = 0
        self._started_at = time.monotonic()

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
            elapsed_seconds=time.monotonic() - self._started_at,
            final=final,
        )
        if self.dynamic:
            prefix = (
                f"\033[{self._dynamic_line_count}F\033[J"
                if self._dynamic_line_count
                else ""
            )
            print(prefix + dashboard, end="", flush=True)
            self._dynamic_line_count = dashboard.count("\n")
            if final:
                print()
            return

        if not self._printed_header:
            print(_static_table_header(self.total_episodes, self.policy_path))
            self._printed_header = True
        for summary in summaries[self._printed_rows :]:
            print(_episode_row(summary))
        self._printed_rows = len(summaries)


def _start_runner(track: str, *, quiet: bool = False) -> TorcsRunner:
    from runner.torcs_runner import TorcsRunner  # noqa: PLC0415

    runner = TorcsRunner()
    with _quiet_stdout(quiet):
        runner.launch()
        runner.connect()
        runner.load_track(track)
    return runner


def _run_gated_evaluation(
    runner: object,
    *,
    policy_path: Path,
    track: str,
    reconnect_attempts: int,
    console: TrainingConsole,
    summaries: list[EpisodeSummary],
    episode: int,
) -> tuple[object, dict[str, object]]:
    retry_count = 0
    max_reconnects = max(0, reconnect_attempts)
    quiet = bool(getattr(console, "dynamic", False))
    while True:
        evaluation_agent = DynaQFinalisedAgent(policy_path=policy_path)
        try:
            with _quiet_stdout(quiet):
                results = runner.run(evaluation_agent, shutdown_on_finish=False)
            return runner, results
        except (ConnectionError, OSError):
            if retry_count >= max_reconnects:
                raise
            retry_count += 1
            console.render(
                summaries,
                (
                    "SCR socket reset before gated evaluation after episode "
                    f"{episode}; relaunching TORCS ({retry_count}/"
                    f"{max_reconnects})..."
                ),
            )
            with _quiet_stdout(quiet):
                runner.shutdown()
            runner = _start_runner(track, quiet=quiet)


def _run_gated_evaluations(
    runner: object,
    *,
    policy_path: Path,
    track: str,
    reconnect_attempts: int,
    console: TrainingConsole,
    summaries: list[EpisodeSummary],
    episode: int,
    repeats: int,
) -> tuple[object, dict[str, object]]:
    evaluation_results = []
    repeat_total = max(1, repeats)
    for repeat in range(1, repeat_total + 1):
        console.render(
            summaries,
            (
                f"Greedy best-check evaluation {repeat}/{repeat_total} "
                f"after episode {episode}..."
            ),
        )
        runner, results = _run_gated_evaluation(
            runner,
            policy_path=policy_path,
            track=track,
            reconnect_attempts=reconnect_attempts,
            console=console,
            summaries=summaries,
            episode=episode,
        )
        evaluation_results.append(results)
    if len(evaluation_results) == 1:
        return runner, evaluation_results[0]
    return runner, _aggregate_evaluation_results(evaluation_results)


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
        off_track=_optional_int(results.get("off_track")),
        avg_speed=_optional_float(results.get("avg_speed")),
    )


def _dashboard_text(
    summaries: list[EpisodeSummary],
    *,
    total_episodes: int,
    policy_path: Path,
    status: str,
    elapsed_seconds: float | None = None,
    final: bool = False,
) -> str:
    best_progress_summary = _best_progress_episode(summaries)
    best_lap_summary = _best_lap_episode(summaries)
    latest = summaries[-1] if summaries else None
    hidden_rows = max(0, len(summaries) - VISIBLE_EPISODE_ROWS)
    visible = summaries[-VISIBLE_EPISODE_ROWS:]
    lines = [
        "Dyna-Q TORCS Training",
        f"Status: {status}",
        f"Policy: {_display_path(policy_path)}",
        (
            f"Episodes: {len(summaries)}/{total_episodes} | "
            f"Best progress: {_best_progress_text(best_progress_summary)} | "
            f"Best lap: {_best_lap_text(best_lap_summary)}"
        ),
        _time_status_text(
            completed_episodes=len(summaries),
            total_episodes=total_episodes,
            elapsed_seconds=elapsed_seconds,
            final=final,
        ),
        f"Latest: {_episode_detail_text(latest)}",
        f"Best progress run: {_episode_detail_text(best_progress_summary)}",
        f"Best lap run: {_episode_detail_text(best_lap_summary)}",
        "",
        "Recent episodes",
        (
            "Ep | Result              | Steps | Progress | Lap % | Lap Time | Off | "
            "Avg km/h | Q states | Eps"
        ),
        (
            "---+---------------------+-------+----------+-------+----------+-----+"
            "---------+----------+------"
        ),
    ]
    if hidden_rows:
        lines.append(f"... {hidden_rows} earlier episode rows hidden ...")
    lines.extend(_episode_row(summary) for summary in visible)
    if not visible:
        lines.append(
            "-- | waiting             |    -- |       -- |    -- |       -- |  -- |"
            "      -- |       -- | --"
        )
    return "\n".join(lines) + "\n"


def _episode_row(summary: EpisodeSummary) -> str:
    lap_percent = min(100.0, summary.progress_percent)
    return (
        f"{summary.episode:2d} | "
        f"{summary.reason:<19} | "
        f"{summary.steps:5d} | "
        f"{summary.progress_m:7.0f}m | "
        f"{lap_percent:5.1f} | "
        f"{_format_lap_time(summary.best_lap):>8} | "
        f"{_format_optional_int(summary.off_track, width=3)} | "
        f"{_format_avg_speed(summary.avg_speed):>7} | "
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
                "Ep | Result              | Steps | Progress | Lap % | Lap Time | Off | "
                "Avg km/h | Q states | Eps"
            ),
            (
                "---+---------------------+-------+----------+-------+----------+-----+"
                "---------+----------+------"
            ),
        ]
    )


def _best_progress_episode(
    summaries: list[EpisodeSummary],
) -> EpisodeSummary | None:
    if not summaries:
        return None
    return max(
        summaries,
        key=lambda summary: (
            summary.progress_m,
            summary.laps,
            -(summary.best_lap or float("inf")),
        ),
    )


def _best_lap_episode(
    summaries: list[EpisodeSummary],
) -> EpisodeSummary | None:
    completed = [summary for summary in summaries if summary.best_lap is not None]
    if not completed:
        return None
    return min(completed, key=lambda summary: summary.best_lap or float("inf"))


def _best_progress_text(summary: EpisodeSummary | None) -> str:
    if summary is None:
        return "--"
    return (
        f"ep {summary.episode} "
        f"{summary.progress_m:.0f}m ({summary.progress_percent:.1f}%)"
    )


def _best_lap_text(summary: EpisodeSummary | None) -> str:
    if summary is None or summary.best_lap is None:
        return "--"
    return f"ep {summary.episode} {summary.best_lap:.3f}s"


def _episode_detail_text(summary: EpisodeSummary | None) -> str:
    if summary is None:
        return "--"
    return (
        f"ep {summary.episode} | {summary.reason} | "
        f"{summary.progress_m:.0f}m ({summary.progress_percent:.1f}%) | "
        f"lap={_format_lap_time(summary.best_lap)} | "
        f"off={_format_optional_int(summary.off_track, width=1)} | "
        f"avg={_format_avg_speed(summary.avg_speed)}km/h | "
        f"q={summary.q_states} | eps={summary.epsilon:.3f}"
    )


def _time_status_text(
    *,
    completed_episodes: int,
    total_episodes: int,
    elapsed_seconds: float | None,
    final: bool,
) -> str:
    if elapsed_seconds is None:
        return "Time: elapsed -- | ETA -- | avg/episode --"
    elapsed_text = _format_duration(elapsed_seconds)
    if final or completed_episodes >= total_episodes:
        average_text = _average_episode_time_text(
            completed_episodes,
            elapsed_seconds,
        )
        return (
            f"Time: elapsed {elapsed_text} | ETA done | "
            f"avg/episode {average_text}"
        )
    if completed_episodes <= 0:
        return f"Time: elapsed {elapsed_text} | ETA calculating | avg/episode --"
    remaining = max(0, total_episodes - completed_episodes)
    average_seconds = elapsed_seconds / completed_episodes
    eta_seconds = average_seconds * remaining
    return (
        f"Time: elapsed {elapsed_text} | ETA {_format_duration(eta_seconds)} | "
        f"avg/episode {_format_duration(average_seconds)}"
    )


def _average_episode_time_text(completed_episodes: int, elapsed_seconds: float) -> str:
    if completed_episodes <= 0:
        return "--"
    return _format_duration(elapsed_seconds / completed_episodes)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def _results_progress_m(
    results: dict[str, object],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> float:
    explicit_progress = _optional_float(results.get("evaluation_progress_m"))
    if explicit_progress is None:
        explicit_progress = _optional_float(results.get("progress_m"))
    if explicit_progress is not None:
        return max(0.0, min(track_length_m, explicit_progress))
    return _episode_progress_m(results, track_length_m=track_length_m)


def _aggregate_evaluation_results(
    results: list[dict[str, object]],
    *,
    track_length_m: float = G_TRACK_3_LENGTH_METRES,
) -> dict[str, object]:
    if not results:
        raise ValueError("At least one evaluation result is required")

    progresses = [
        _results_progress_m(result, track_length_m=track_length_m)
        for result in results
    ]
    laps = [
        max(0, _optional_int(result.get("laps_completed")) or 0)
        for result in results
    ]
    completed_results = [
        result
        for result in results
        if max(0, _optional_int(result.get("laps_completed")) or 0) > 0
    ]
    completed_lap_times = [
        value
        for result in completed_results
        for value in [_optional_float(result.get("best_lap_time_seconds"))]
        if value is not None
    ]
    completed_off_track_values = [
        value
        for result in completed_results
        for value in [_optional_int(result.get("off_track"))]
        if value is not None
    ]
    completed_avg_speed_values = [
        value
        for result in completed_results
        for value in [_optional_float(result.get("avg_speed"))]
        if value is not None
    ]
    completed_step_values = [
        value
        for result in completed_results
        for value in [_optional_int(result.get("steps"))]
        if value is not None
    ]
    off_track_values = [
        value
        for result in results
        for value in [_optional_int(result.get("off_track"))]
        if value is not None
    ]
    avg_speed_values = [
        value
        for result in results
        for value in [_optional_float(result.get("avg_speed"))]
        if value is not None
    ]
    step_values = [
        value
        for result in results
        for value in [_optional_int(result.get("steps"))]
        if value is not None
    ]
    reasons = [
        str(result.get("termination_reason", ""))
        for result in results
        if result.get("termination_reason") is not None
    ]

    repeat_count = len(results)
    completed_repeats = sum(1 for lap_count in laps if lap_count > 0)
    repeated_laps = 1 if completed_repeats > 0 else 0
    lap_time = _median_float(completed_lap_times)
    progress_m = _median_float(progresses) or 0.0
    reason = (
        reasons[0]
        if reasons and all(reason == reasons[0] for reason in reasons)
        else "repeat_evaluations"
    )

    return {
        "termination_reason": reason,
        "steps": _median_int(completed_step_values or step_values),
        "laps_completed": repeated_laps,
        "best_lap_time_seconds": lap_time,
        "off_track": _median_int(completed_off_track_values or off_track_values),
        "avg_speed": _median_float(completed_avg_speed_values or avg_speed_values),
        "evaluation_progress_m": progress_m,
        "evaluation_min_progress_m": min(progresses) if progresses else 0.0,
        "evaluation_max_progress_m": max(progresses) if progresses else 0.0,
        "evaluation_median_progress_m": progress_m,
        "evaluation_repeats": repeat_count,
        "evaluation_completed_repeats": completed_repeats,
    }


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


def _format_optional_int(value: int | None, *, width: int) -> str:
    if value is None:
        return "--".rjust(width)
    return f"{value:{width}d}"


def _format_lap_time(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}s"


def _format_avg_speed(value: float | None) -> str:
    speed = _scaled_speed_kmh(value)
    if speed is None:
        return "--"
    return f"{speed:.1f}"


@contextmanager
def _quiet_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with redirect_stdout(StringIO()):
        yield


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
    *,
    evaluation_progress_m: float | None = None,
    evaluation_results: dict[str, object] | None = None,
    min_evaluation_progress_m: float = 0.0,
    max_evaluation_off_track: int | None = None,
    min_evaluation_completions: int = 0,
) -> bool:
    if not policy_path.is_file():
        return False

    score_m = progress_m
    previous_best = _policy_best_progress(best_policy_path)
    metadata = {
        "checkpoint_type": "best",
        "checkpoint_score": "training_progress",
        "best_progress_m": score_m,
        "best_progress_percent": min(100.0, score_m / G_TRACK_3_LENGTH_METRES * 100.0),
    }

    if evaluation_progress_m is not None:
        if evaluation_progress_m < min_evaluation_progress_m:
            return False
        score_m = evaluation_progress_m
        candidate_quality = _evaluation_quality(
            evaluation_progress_m,
            evaluation_results,
        )
        if (
            candidate_quality.completed_repeats is not None
            and candidate_quality.completed_repeats < min_evaluation_completions
        ):
            return False
        if (
            candidate_quality.completed_repeats is None
            and candidate_quality.completed_laps < min_evaluation_completions
        ):
            return False
        if max_evaluation_off_track is not None and (
            candidate_quality.off_track is None
            or candidate_quality.off_track > max_evaluation_off_track
        ):
            return False
        previous_quality = _policy_best_quality(best_policy_path)
        if previous_quality is not None and not _quality_is_better(
            candidate_quality,
            previous_quality,
        ):
            return False
        metadata.update(
            {
                "checkpoint_score": "finalised_evaluation_quality",
                "best_progress_m": score_m,
                "best_progress_percent": min(
                    100.0,
                    score_m / G_TRACK_3_LENGTH_METRES * 100.0,
                ),
                "best_evaluation_progress_m": evaluation_progress_m,
                "best_evaluation_progress_percent": min(
                    100.0,
                    evaluation_progress_m / G_TRACK_3_LENGTH_METRES * 100.0,
                ),
                "candidate_training_progress_m": progress_m,
                "candidate_training_progress_percent": min(
                    100.0,
                    progress_m / G_TRACK_3_LENGTH_METRES * 100.0,
                ),
                "best_evaluation_laps": candidate_quality.completed_laps,
                "best_evaluation_lap_time": candidate_quality.best_lap,
                "best_evaluation_off_track": candidate_quality.off_track,
                "best_evaluation_avg_speed": candidate_quality.avg_speed,
            }
        )
        if max_evaluation_off_track is not None:
            metadata["best_evaluation_max_off_track_gate"] = (
                max_evaluation_off_track
            )
        if min_evaluation_completions:
            metadata["best_evaluation_min_completions_gate"] = (
                min_evaluation_completions
            )
        if candidate_quality.avg_speed is not None:
            metadata["best_evaluation_avg_speed_kmh"] = (
                candidate_quality.avg_speed * 50.0
            )
        if evaluation_results is not None:
            metadata.update(
                {
                    "best_evaluation_reason": str(
                        evaluation_results.get("termination_reason", "")
                    ),
                    "best_evaluation_steps": int(evaluation_results.get("steps", 0)),
                    "best_evaluation_laps": int(
                        evaluation_results.get("laps_completed", 0)
                    ),
                    "best_evaluation_lap_time": evaluation_results.get(
                        "best_lap_time_seconds"
                    ),
                    "best_evaluation_off_track": int(
                        evaluation_results.get("off_track", 0)
                    ),
                    "best_evaluation_avg_speed": evaluation_results.get("avg_speed"),
                    "best_evaluation_avg_speed_kmh": _scaled_speed_kmh(
                        evaluation_results.get("avg_speed")
                    ),
                }
            )
            evaluation_repeats = _optional_int(
                evaluation_results.get("evaluation_repeats")
            )
            completed_repeats = _optional_int(
                evaluation_results.get("evaluation_completed_repeats")
            )
            if evaluation_repeats is not None:
                metadata["best_evaluation_repeats"] = evaluation_repeats
            if completed_repeats is not None:
                metadata["best_evaluation_completed_repeats"] = completed_repeats
            min_progress = _optional_float(
                evaluation_results.get("evaluation_min_progress_m")
            )
            max_progress = _optional_float(
                evaluation_results.get("evaluation_max_progress_m")
            )
            median_progress = _optional_float(
                evaluation_results.get("evaluation_median_progress_m")
            )
            if min_progress is not None:
                metadata["best_evaluation_min_progress_m"] = min_progress
            if max_progress is not None:
                metadata["best_evaluation_max_progress_m"] = max_progress
            if median_progress is not None:
                metadata["best_evaluation_median_progress_m"] = median_progress

    if evaluation_progress_m is None and previous_best is not None and score_m <= previous_best:
        return False

    best_policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(policy_path, best_policy_path)
    _annotate_policy(best_policy_path, **metadata)
    return True


def _auto_promote_final_policy(
    *,
    best_policy_path: Path,
    final_policy_path: Path,
    latest_policy_path: Path,
    archive_dir: Path,
    archive_existing: bool,
) -> FinalPromotionDecision:
    candidate = _policy_best_quality(best_policy_path)
    previous_final = _policy_best_quality(final_policy_path)
    if candidate is None:
        return FinalPromotionDecision(
            promoted=False,
            confidence="none",
            reason=f"No readable best checkpoint found at {best_policy_path}.",
            previous_final=previous_final,
            candidate=None,
        )
    if candidate.completed_laps <= 0:
        return FinalPromotionDecision(
            promoted=False,
            confidence="rejected",
            reason="Best checkpoint has no completed finalised evaluation lap.",
            previous_final=previous_final,
            candidate=candidate,
        )
    completed_repeats = candidate.completed_repeats
    if completed_repeats is not None and completed_repeats <= 0:
        return FinalPromotionDecision(
            promoted=False,
            confidence="rejected",
            reason="Best checkpoint completed 0 evaluation repeats.",
            previous_final=previous_final,
            candidate=candidate,
        )

    if previous_final is not None and not _quality_is_better(
        candidate,
        previous_final,
    ):
        return FinalPromotionDecision(
            promoted=False,
            confidence="rejected",
            reason="Best checkpoint does not beat the current final policy.",
            previous_final=previous_final,
            candidate=candidate,
        )

    archive_path = None
    if archive_existing and final_policy_path.is_file():
        archive_path = _archive_policy(final_policy_path, archive_dir)

    final_policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_policy_path, final_policy_path)
    _annotate_policy(
        final_policy_path,
        checkpoint_type="final",
        promoted_from_best=str(best_policy_path),
    )

    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_policy_path, latest_policy_path)
    _annotate_policy(
        latest_policy_path,
        checkpoint_type="latest",
        synced_from_best_after_final_promotion=str(best_policy_path),
    )

    return FinalPromotionDecision(
        promoted=True,
        confidence=_promotion_confidence(candidate),
        reason="Best checkpoint beats the current final policy.",
        previous_final=previous_final,
        candidate=candidate,
        archive_path=archive_path,
    )


def _archive_policy(path: Path, archive_dir: Path) -> Path:
    quality = _policy_best_quality(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if quality is None:
        metrics = "unknown"
    else:
        lap = "nolap" if quality.best_lap is None else f"{quality.best_lap:.0f}s"
        off = "offna" if quality.off_track is None else f"{quality.off_track}off"
        metrics = f"{lap}_{off}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{path.stem}_{metrics}_{timestamp}{path.suffix}"
    shutil.copyfile(path, archive_path)
    _annotate_policy(
        archive_path,
        checkpoint_type="archived_final",
        archived_from=str(path),
    )
    return archive_path


def _promotion_confidence(candidate: PolicyQuality) -> str:
    repeats = candidate.evaluation_repeats
    completed = candidate.completed_repeats
    if repeats is None or completed is None:
        return "single-evaluation"
    if completed >= repeats:
        return "high"
    if completed >= max(1, repeats - 1):
        return "medium"
    return "breakthrough"


def _final_promotion_summary(decision: FinalPromotionDecision) -> str:
    lines = [
        "Final policy decision:",
        f"Previous final: {_quality_summary(decision.previous_final)}",
        f"Candidate best: {_quality_summary(decision.candidate)}",
        f"Decision: {'promoted' if decision.promoted else 'rejected'}",
        f"Confidence: {decision.confidence}",
        f"Reason: {decision.reason}",
    ]
    if decision.archive_path is not None:
        lines.append(f"Archived previous final: {decision.archive_path}")
    lines.append("Git: unchanged; commit/push manually when you are ready.")
    return "\n".join(lines)


def _quality_summary(quality: PolicyQuality | None) -> str:
    if quality is None:
        return "--"
    repeats = ""
    if quality.evaluation_repeats is not None:
        repeats = (
            f" | evals={quality.completed_repeats or 0}/"
            f"{quality.evaluation_repeats}"
        )
    avg_speed = _scaled_speed_kmh(quality.avg_speed)
    avg_text = "" if avg_speed is None else f" | avg={avg_speed:.1f}km/h"
    lap_text = "--" if quality.best_lap is None else f"{quality.best_lap:.3f}s"
    off_text = "--" if quality.off_track is None else str(quality.off_track)
    return (
        f"lap={lap_text} | off_track={off_text} | "
        f"progress={quality.progress_m:.0f}m{avg_text}{repeats}"
    )


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


def _candidate_training_gate_passed(
    summary: EpisodeSummary,
    *,
    min_training_laps: int,
) -> bool:
    return summary.laps >= min_training_laps


def _restore_latest_from_best_policy(
    *,
    policy_path: Path,
    best_policy_path: Path,
    episode: int,
    reason: str = "rejected_evaluation",
) -> bool:
    if not best_policy_path.is_file():
        return False

    policy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_policy_path, policy_path)
    metadata: dict[str, object] = {
        "checkpoint_type": "latest",
        "restored_from_best_reason": reason,
        "restored_from_best_episode": episode,
    }
    if reason == "rejected_evaluation":
        metadata.update(
            {
                "restored_from_best_after_rejected_evaluation": str(
                    best_policy_path
                ),
                "restored_after_rejected_evaluation_episode": episode,
            }
        )
    elif reason == "skipped_incomplete_training_lap":
        metadata.update(
            {
                "restored_from_best_after_skipped_evaluation": str(
                    best_policy_path
                ),
                "restored_after_skipped_evaluation_episode": episode,
            }
        )
    _annotate_policy(policy_path, **metadata)
    return True


def _annotate_policy(path: Path, **metadata: object) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(metadata)
    _write_policy_payload(path, payload)


def _write_policy_payload(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _policy_best_progress(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _optional_float(payload.get("best_progress_m"))


def _policy_best_evaluation_progress(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _optional_float(payload.get("best_evaluation_progress_m"))


def _evaluation_quality(
    progress_m: float,
    results: dict[str, object] | None,
) -> PolicyQuality:
    if results is None:
        return PolicyQuality(
            completed_laps=0,
            best_lap=None,
            off_track=None,
            avg_speed=None,
            progress_m=progress_m,
        )
    evaluation_repeats = _optional_int(results.get("evaluation_repeats"))
    completed_repeats = _optional_int(results.get("evaluation_completed_repeats"))
    return PolicyQuality(
        completed_laps=max(0, _optional_int(results.get("laps_completed")) or 0),
        best_lap=_optional_float(results.get("best_lap_time_seconds")),
        off_track=_optional_int(results.get("off_track")),
        avg_speed=_optional_float(results.get("avg_speed")),
        progress_m=progress_m,
        evaluation_repeats=evaluation_repeats,
        completed_repeats=completed_repeats,
    )


def _policy_best_quality(path: Path) -> PolicyQuality | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    progress_m = _optional_float(payload.get("best_evaluation_progress_m"))
    if progress_m is None:
        progress_m = _optional_float(payload.get("best_progress_m"))
    if progress_m is None:
        return None

    completed_laps = _optional_int(payload.get("best_evaluation_laps"))
    if completed_laps is None:
        completed_laps = 1 if progress_m >= G_TRACK_3_LENGTH_METRES * 0.999 else 0
    return PolicyQuality(
        completed_laps=max(0, completed_laps),
        best_lap=_optional_float(payload.get("best_evaluation_lap_time")),
        off_track=_optional_int(payload.get("best_evaluation_off_track")),
        avg_speed=_optional_float(payload.get("best_evaluation_avg_speed")),
        progress_m=progress_m,
        evaluation_repeats=_optional_int(payload.get("best_evaluation_repeats")),
        completed_repeats=_optional_int(
            payload.get("best_evaluation_completed_repeats")
        ),
    )


def _quality_is_better(candidate: PolicyQuality, current: PolicyQuality) -> bool:
    if candidate.completed_laps != current.completed_laps:
        return candidate.completed_laps > current.completed_laps

    if candidate.completed_laps > 0:
        lap_comparison = _compare_lower_is_better(
            candidate.best_lap,
            current.best_lap,
        )
        if lap_comparison != 0:
            return lap_comparison < 0

        off_track_comparison = _compare_lower_is_better(
            candidate.off_track,
            current.off_track,
        )
        if off_track_comparison != 0:
            return off_track_comparison < 0

        speed_comparison = _compare_higher_is_better(
            candidate.avg_speed,
            current.avg_speed,
        )
        if speed_comparison != 0:
            return speed_comparison > 0

        return candidate.progress_m > current.progress_m + 1e-6

    progress_delta = candidate.progress_m - current.progress_m
    if abs(progress_delta) > 1e-6:
        return progress_delta > 0.0

    off_track_comparison = _compare_lower_is_better(
        candidate.off_track,
        current.off_track,
    )
    if off_track_comparison != 0:
        return off_track_comparison < 0

    speed_comparison = _compare_higher_is_better(
        candidate.avg_speed,
        current.avg_speed,
    )
    return speed_comparison > 0


def _compare_lower_is_better(
    candidate: float | int | None,
    current: float | int | None,
) -> int:
    if candidate is None and current is None:
        return 0
    if candidate is None:
        return 1
    if current is None:
        return -1
    if math_isclose(float(candidate), float(current)):
        return 0
    return -1 if candidate < current else 1


def _compare_higher_is_better(
    candidate: float | int | None,
    current: float | int | None,
) -> int:
    if candidate is None and current is None:
        return 0
    if candidate is None:
        return -1
    if current is None:
        return 1
    if math_isclose(float(candidate), float(current)):
        return 0
    return 1 if candidate > current else -1


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _median_float(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(round(float(median(values))))


def _scaled_speed_kmh(value: object) -> float | None:
    speed = _optional_float(value)
    if speed is None:
        return None
    return speed * 50.0


def _best_policy_summary(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    best_progress = _optional_float(payload.get("best_progress_m")) or 0.0
    best_percent = min(100.0, best_progress / G_TRACK_3_LENGTH_METRES * 100.0)
    evaluation_progress = _optional_float(payload.get("best_evaluation_progress_m"))
    if evaluation_progress is not None:
        evaluation_percent = min(
            100.0,
            evaluation_progress / G_TRACK_3_LENGTH_METRES * 100.0,
        )
        training_progress = _optional_float(
            payload.get("candidate_training_progress_m")
        )
        training_text = (
            ""
            if training_progress is None
            else f" | train_progress={training_progress:.0f}m"
        )
        laps = _optional_int(payload.get("best_evaluation_laps"))
        lap_time = _optional_float(payload.get("best_evaluation_lap_time"))
        off_track = _optional_int(payload.get("best_evaluation_off_track"))
        avg_speed = _optional_float(payload.get("best_evaluation_avg_speed_kmh"))
        repeats = _optional_int(payload.get("best_evaluation_repeats"))
        completed_repeats = _optional_int(
            payload.get("best_evaluation_completed_repeats")
        )
        lap_text = "" if laps is None else f" | laps={laps}"
        lap_time_text = "" if lap_time is None else f" | lap={lap_time:.3f}s"
        off_track_text = "" if off_track is None else f" | off_track={off_track}"
        avg_speed_text = "" if avg_speed is None else f" | avg={avg_speed:.1f}km/h"
        repeat_text = (
            ""
            if repeats is None
            else f" | evals={completed_repeats or 0}/{repeats}"
        )
        return (
            f"best_eval_progress={evaluation_progress:.0f}m "
            f"({evaluation_percent:.1f}%){lap_text}{lap_time_text}"
            f"{off_track_text}{avg_speed_text}{repeat_text}{training_text} | "
            f"q_values={len(payload.get('q_values', {}))} | "
            f"trained_steps={payload.get('steps_trained')}"
        )
    return (
        f"best_progress={best_progress:.0f}m ({best_percent:.1f}%) | "
        f"q_values={len(payload.get('q_values', {}))} | "
        f"trained_steps={payload.get('steps_trained')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
