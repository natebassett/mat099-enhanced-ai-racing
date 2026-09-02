from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .project_discovery import RunSummary, discover_run_history
except ImportError:
    from project_discovery import RunSummary, discover_run_history


@dataclass(frozen=True)
class EvaluationEpisode:
    trial: int
    distance_m: float
    lap_time_seconds: float | None
    completed: bool
    reward: float | None
    max_speed_kmh: float | None
    average_speed_kmh: float | None
    termination_reason: str


@dataclass(frozen=True)
class EvaluationBatch:
    batch_id: str
    agent_family: str
    agent_label: str
    policy_path: str
    policy_label: str
    track: str
    timestamp: str
    source_path: Path
    deterministic: bool
    episodes: tuple[EvaluationEpisode, ...]
    observation_version: str
    action_version: str
    reward_version: str
    protocol: str

    @property
    def trials(self) -> int:
        return len(self.episodes)

    @property
    def completed_laps(self) -> int:
        return sum(episode.completed for episode in self.episodes)

    @property
    def completion_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return self.completed_laps / len(self.episodes)

    @property
    def completed_lap_times(self) -> tuple[float, ...]:
        return tuple(
            episode.lap_time_seconds
            for episode in self.episodes
            if episode.completed and episode.lap_time_seconds is not None
        )

    @property
    def median_lap_time_seconds(self) -> float | None:
        times = self.completed_lap_times
        return statistics.median(times) if times else None

    @property
    def fastest_lap_time_seconds(self) -> float | None:
        times = self.completed_lap_times
        return min(times) if times else None

    @property
    def median_distance_m(self) -> float | None:
        if not self.episodes:
            return None
        return statistics.median(episode.distance_m for episode in self.episodes)

    @property
    def minimum_distance_m(self) -> float | None:
        if not self.episodes:
            return None
        return min(episode.distance_m for episode in self.episodes)

    @property
    def label(self) -> str:
        return f"{self.agent_label} | {self.policy_label} | {self.timestamp}"


@dataclass(frozen=True)
class TrainingEpisode:
    episode: int
    interactions: int
    distance_m: float
    lap_time_seconds: float | None
    completed: bool
    reward: float | None
    mode: str
    termination_reason: str


@dataclass(frozen=True)
class TrainingRun:
    run_id: str
    agent_family: str
    agent_label: str
    timestamp: str
    track: str
    seed: int | None
    source_path: Path
    config_path: Path | None
    episodes: tuple[TrainingEpisode, ...]
    requested_timesteps: int | None
    observation_version: str
    action_version: str
    reward_version: str

    @property
    def total_interactions(self) -> int:
        return self.episodes[-1].interactions if self.episodes else 0

    @property
    def completed_laps(self) -> int:
        return sum(episode.completed for episode in self.episodes)

    @property
    def farthest_distance_m(self) -> float | None:
        if not self.episodes:
            return None
        return max(episode.distance_m for episode in self.episodes)

    @property
    def fastest_lap_time_seconds(self) -> float | None:
        times = [
            episode.lap_time_seconds
            for episode in self.episodes
            if episode.completed and episode.lap_time_seconds is not None
        ]
        return min(times) if times else None

    @property
    def label(self) -> str:
        seed = f"seed {self.seed}" if self.seed is not None else "seed unknown"
        return f"{self.agent_label} | {self.timestamp} | {seed}"


@dataclass(frozen=True)
class RaceAgentSummary:
    agent_name: str
    track: str
    runs: int
    completed_runs: int
    completion_rate: float
    best_lap_time_seconds: float | None
    median_lap_time_seconds: float | None
    median_average_speed_kmh: float | None
    source_path: str


@dataclass(frozen=True)
class ResultsDataset:
    evaluation_batches: tuple[EvaluationBatch, ...]
    training_runs: tuple[TrainingRun, ...]
    race_summaries: tuple[RaceAgentSummary, ...]
    run_history: tuple[RunSummary, ...]
    warnings: tuple[str, ...]

    @property
    def tracks(self) -> tuple[str, ...]:
        values = {
            batch.track for batch in self.evaluation_batches if batch.track
        } | {run.track for run in self.training_runs if run.track} | {
            summary.track for summary in self.race_summaries if summary.track
        }
        return tuple(sorted(values))


def load_results_dataset(project_root: Path) -> ResultsDataset:
    warnings: list[str] = []
    evaluations = _load_evaluation_batches(project_root, warnings)
    training_runs = _load_training_runs(project_root, warnings)

    try:
        run_history, run_source = discover_run_history(project_root, limit=10_000)
    except (OSError, ValueError) as error:
        warnings.append(f"Could not read saved race history: {error}")
        run_history = []
        run_source = "No saved run history found"

    race_summaries = summarize_race_runs(run_history, run_source)
    return ResultsDataset(
        evaluation_batches=tuple(evaluations),
        training_runs=tuple(training_runs),
        race_summaries=tuple(race_summaries),
        run_history=tuple(run_history),
        warnings=tuple(warnings),
    )


def summarize_race_runs(
    runs: Iterable[RunSummary],
    source_path: str,
) -> list[RaceAgentSummary]:
    grouped: dict[tuple[str, str], list[RunSummary]] = {}
    for run in runs:
        grouped.setdefault((run.agent_name, run.track), []).append(run)

    summaries: list[RaceAgentSummary] = []
    for (agent_name, track), group in grouped.items():
        completed_times = [
            run.best_lap_time_seconds
            for run in group
            if run.best_lap_time_seconds is not None
            and run.best_lap_time_seconds > 0.0
        ]
        speeds = [run.avg_speed for run in group if run.avg_speed is not None]
        completed = len(completed_times)
        summaries.append(
            RaceAgentSummary(
                agent_name=agent_name,
                track=track,
                runs=len(group),
                completed_runs=completed,
                completion_rate=completed / len(group),
                best_lap_time_seconds=min(completed_times)
                if completed_times
                else None,
                median_lap_time_seconds=statistics.median(completed_times)
                if completed_times
                else None,
                median_average_speed_kmh=statistics.median(speeds) if speeds else None,
                source_path=source_path,
            )
        )
    return sorted(summaries, key=lambda item: (item.track, item.agent_name.casefold()))


def filter_evaluations(
    batches: Iterable[EvaluationBatch],
    *,
    track: str = "All tracks",
    agent_family: str = "All agents",
) -> list[EvaluationBatch]:
    return [
        batch
        for batch in batches
        if (track == "All tracks" or batch.track == track)
        and (agent_family == "All agents" or batch.agent_family == agent_family)
    ]


def representative_evaluations(
    batches: Iterable[EvaluationBatch],
    *,
    minimum_trials: int = 10,
) -> list[EvaluationBatch]:
    grouped: dict[str, list[EvaluationBatch]] = {}
    for batch in batches:
        grouped.setdefault(batch.agent_family, []).append(batch)

    representatives: list[EvaluationBatch] = []
    for family, family_batches in grouped.items():
        supported = [
            batch for batch in family_batches if batch.trials >= minimum_trials
        ]
        candidates = supported or family_batches
        representatives.append(max(candidates, key=_representative_score))
    return sorted(representatives, key=lambda batch: batch.agent_family)


def featured_evaluation(
    batches: Iterable[EvaluationBatch],
    *,
    minimum_trials: int = 10,
    minimum_completion_rate: float = 0.8,
) -> EvaluationBatch | None:
    representatives = representative_evaluations(
        batches,
        minimum_trials=minimum_trials,
    )
    reliable = [
        batch
        for batch in representatives
        if batch.trials >= minimum_trials
        and batch.completion_rate >= minimum_completion_rate
        and batch.median_lap_time_seconds is not None
    ]
    if reliable:
        return min(
            reliable,
            key=lambda batch: (
                batch.median_lap_time_seconds or math.inf,
                -batch.completion_rate,
            ),
        )
    return max(representatives, key=_representative_score, default=None)


def _load_evaluation_batches(
    project_root: Path,
    warnings: list[str],
) -> list[EvaluationBatch]:
    root = project_root / "data" / "evaluation"
    if not root.is_dir():
        return []

    batches: list[EvaluationBatch] = []
    for path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                continue
            batches.extend(_parse_evaluation_document(path, document))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            warnings.append(f"Skipped evaluation {path.name}: {error}")
    return sorted(batches, key=lambda batch: (batch.timestamp, batch.batch_id), reverse=True)


def _parse_evaluation_document(
    source_path: Path,
    document: dict[str, Any],
) -> list[EvaluationBatch]:
    summaries = document.get("summaries")
    episode_groups = document.get("episodes")
    if isinstance(summaries, list) and isinstance(episode_groups, dict):
        parsed: list[EvaluationBatch] = []
        for index, summary in enumerate(summaries):
            if not isinstance(summary, dict):
                continue
            policy_path = str(summary.get("policy_path") or "")
            episodes = episode_groups.get(policy_path, [])
            if not isinstance(episodes, list):
                episodes = []
            batch = _make_evaluation_batch(
                source_path,
                summary,
                episodes,
                suffix=f"-{index + 1}",
            )
            if batch.episodes:
                parsed.append(batch)
        return parsed

    episodes = document.get("episodes")
    if not isinstance(episodes, list):
        return []
    batch = _make_evaluation_batch(source_path, document, episodes)
    return [batch] if batch.episodes else []


def _make_evaluation_batch(
    source_path: Path,
    metadata: dict[str, Any],
    raw_episodes: list[Any],
    *,
    suffix: str = "",
) -> EvaluationBatch:
    episodes = tuple(
        episode
        for index, raw in enumerate(raw_episodes, start=1)
        if isinstance(raw, dict)
        and (episode := _parse_evaluation_episode(raw, index)) is not None
    )
    policy_path = str(metadata.get("policy_path") or "Unknown policy")
    agent_family, agent_label = _agent_identity(source_path, policy_path, metadata)
    timestamp = _timestamp_from_path(source_path)
    policy_label = Path(policy_path).stem if policy_path else "Unknown policy"
    return EvaluationBatch(
        batch_id=f"{source_path.as_posix()}{suffix}",
        agent_family=agent_family,
        agent_label=agent_label,
        policy_path=policy_path,
        policy_label=policy_label,
        track=str(metadata.get("track") or "unknown"),
        timestamp=timestamp,
        source_path=source_path,
        deterministic=bool(metadata.get("deterministic", True)),
        episodes=episodes,
        observation_version=str(metadata.get("observation_version") or "not recorded"),
        action_version=str(metadata.get("action_version") or "not recorded"),
        reward_version=str(metadata.get("reward_version") or "not recorded"),
        protocol=str(metadata.get("evaluation_protocol") or "deterministic repeats"),
    )


def _parse_evaluation_episode(
    raw: dict[str, Any],
    fallback_trial: int,
) -> EvaluationEpisode | None:
    distance = _optional_float(raw.get("distance_m", raw.get("progress_m")))
    if distance is None:
        return None
    lap_time = _optional_float(
        raw.get("best_lap_time_seconds", raw.get("best_lap_seconds"))
    )
    laps = _optional_int(raw.get("laps_completed", raw.get("laps"))) or 0
    return EvaluationEpisode(
        trial=_optional_int(raw.get("trial", raw.get("repeat"))) or fallback_trial,
        distance_m=distance,
        lap_time_seconds=lap_time,
        completed=laps > 0 or lap_time is not None,
        reward=_optional_float(raw.get("reward", raw.get("total_score"))),
        max_speed_kmh=_optional_float(raw.get("max_speed_kmh")),
        average_speed_kmh=_optional_float(raw.get("average_speed_kmh")),
        termination_reason=str(
            raw.get("termination_reason", raw.get("reason", "unknown"))
        ),
    )


def _load_training_runs(
    project_root: Path,
    warnings: list[str],
) -> list[TrainingRun]:
    root = project_root / "models" / "training_runs"
    if not root.is_dir():
        return []

    runs: list[TrainingRun] = []
    for episodes_path in sorted(root.rglob("episodes.csv")):
        family_name = episodes_path.parent.parent.name
        if not family_name.startswith(("agent7_", "agent8_")):
            continue
        try:
            run = _parse_training_run(episodes_path)
            if run.episodes:
                runs.append(run)
        except (OSError, ValueError, csv.Error, json.JSONDecodeError) as error:
            warnings.append(f"Skipped training run {episodes_path.parent.name}: {error}")
    return sorted(runs, key=lambda run: (run.timestamp, run.run_id), reverse=True)


def _parse_training_run(episodes_path: Path) -> TrainingRun:
    config_path = episodes_path.with_name("config.json")
    config: dict[str, Any] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded

    cumulative_interactions = 0
    episodes: list[TrainingEpisode] = []
    with episodes_path.open(newline="", encoding="utf-8") as handle:
        for fallback_episode, row in enumerate(csv.DictReader(handle), start=1):
            mode = str(row.get("mode") or "training")
            if mode.startswith("evaluation"):
                continue
            steps = max(0, _optional_int(row.get("steps")) or 0)
            cumulative_interactions += steps
            distance = _optional_float(row.get("distance_m"))
            if distance is None:
                continue
            lap_time = _optional_float(row.get("best_lap_time_seconds"))
            laps = _optional_int(row.get("laps_completed")) or 0
            episodes.append(
                TrainingEpisode(
                    episode=_optional_int(row.get("episode")) or fallback_episode,
                    interactions=cumulative_interactions,
                    distance_m=distance,
                    lap_time_seconds=lap_time,
                    completed=laps > 0 or lap_time is not None,
                    reward=_optional_float(row.get("reward")),
                    mode=mode,
                    termination_reason=str(row.get("termination_reason") or "unknown"),
                )
            )

    family_name = episodes_path.parent.parent.name
    agent_family, agent_label = _agent_identity(
        episodes_path,
        str(config.get("resume") or family_name),
        config,
    )
    timestamp = str(config.get("run_started") or episodes_path.parent.name)
    return TrainingRun(
        run_id=episodes_path.parent.as_posix(),
        agent_family=agent_family,
        agent_label=agent_label,
        timestamp=timestamp,
        track=str(config.get("track") or "g-track-3"),
        seed=_optional_int(config.get("run_seed", config.get("seed"))),
        source_path=episodes_path,
        config_path=config_path if config_path.is_file() else None,
        episodes=tuple(episodes),
        requested_timesteps=_optional_int(config.get("requested_training_timesteps")),
        observation_version=str(config.get("observation_version") or "not recorded"),
        action_version=str(config.get("action_version") or "not recorded"),
        reward_version=str(config.get("reward_version") or "not recorded"),
    )


def _agent_identity(
    source_path: Path,
    policy_path: str,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    evidence = " ".join(
        (
            source_path.as_posix(),
            policy_path,
            str(metadata.get("model_family") or ""),
        )
    ).casefold()
    if "agent8" in evidence or "sensor_n_step_td3" in evidence:
        return "agent8", "Agent 8 - Sensor-only N-step TD3"
    if "agent7" in evidence or "n_step_td3" in evidence:
        return "agent7", "Agent 7 - Racing-line N-step TD3"
    if "agent6" in evidence or "scratch_racer" in evidence:
        return "agent6", "Agent 6 - TD3 Scratch Racer"
    return "other", "Other evaluation"


def _timestamp_from_path(path: Path) -> str:
    stem = path.stem
    for token in reversed(stem.split("_")):
        if len(token) == 15 and token[8] == "-":
            return token
    return path.parent.name


def _representative_score(batch: EvaluationBatch) -> tuple[Any, ...]:
    if batch.median_lap_time_seconds is not None:
        return (
            1,
            batch.completion_rate,
            -batch.median_lap_time_seconds,
            batch.trials,
            batch.timestamp,
        )
    return (
        0,
        batch.completion_rate,
        batch.median_distance_m or 0.0,
        batch.minimum_distance_m or 0.0,
        batch.trials,
        batch.timestamp,
    )


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
