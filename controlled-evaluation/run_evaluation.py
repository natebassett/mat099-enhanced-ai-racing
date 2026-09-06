#!/usr/bin/env python3
"""Run the pre-declared MAT099 agent comparison without modifying app code."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol_v1.json")
DEFAULT_RESULTS_ROOT = Path(__file__).with_name("results")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SCHEDULE_FIELDS = [
    "attempt_id",
    "block",
    "position",
    "agent_trial",
    "agent_id",
    "agent_label",
    "seed",
]

ATTEMPT_FIELDS = [
    "attempt_id",
    "block",
    "position",
    "agent_trial",
    "agent_id",
    "agent_label",
    "seed",
    "started_at",
    "finished_at",
    "valid_attempt",
    "clean_completion",
    "failure_category",
    "laps_completed",
    "lap_time_seconds",
    "distance_m",
    "steps",
    "total_score",
    "mean_speed_kmh",
    "max_speed_kmh",
    "off_track_steps",
    "off_track_events",
    "reverse_steps",
    "reverse_events",
    "stall_events",
    "damage_delta",
    "termination_reason",
    "telemetry_path",
]

INFRASTRUCTURE_FIELDS = [
    "attempt_id",
    "block",
    "agent_id",
    "recorded_at",
    "retry_number",
    "exception_type",
    "message",
    "traceback_path",
    "partial_telemetry_path",
]

CORE_TELEMETRY_FIELDS = [
    "step",
    "dist_from_start",
    "dist_raced",
    "speed_x",
    "speed_y",
    "angle",
    "track_pos",
    "damage",
    "steer",
    "accel",
    "brake",
    "gear",
    "reward",
    "off_track",
    "front_sensor",
    "min_track_sensor",
    "track_sensors",
    "cur_lap_time",
    "last_lap_time",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    row: Mapping[str, Any],
) -> None:
    rows: list[Mapping[str, Any]] = read_csv_rows(path)
    rows.append(row)
    atomic_write_csv(path, fieldnames, rows)


def validate_session_id(session_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", session_id):
        raise ValueError(
            "Session IDs must be 1-80 characters using letters, numbers, '.', '_' "
            "or '-', and must start with a letter or number."
        )
    return session_id


def session_dir(results_root: Path, session_id: str) -> Path:
    return results_root.resolve() / validate_session_id(session_id)


def protocol_agents(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    agents = protocol.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Protocol must define at least one agent")
    identifiers = [str(agent.get("id", "")) for agent in agents]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every protocol agent requires a non-empty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Protocol agent ids must be unique")
    return agents


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    environment = protocol.get("environment", {})
    design = protocol.get("design", {})
    required_environment = ("track_id", "track_category", "car_id", "target_laps")
    missing = [key for key in required_environment if not environment.get(key)]
    if missing:
        raise ValueError(f"Protocol environment is missing: {', '.join(missing)}")
    if int(environment.get("max_steps", 0)) <= 0:
        raise ValueError("environment.max_steps must be positive")
    if int(design.get("repeats_per_agent", 0)) <= 0:
        raise ValueError("design.repeats_per_agent must be positive")
    protocol_agents(protocol)


def select_protocol(
    source: Mapping[str, Any],
    *,
    selected_agent_ids: Sequence[str] | None,
    repeats: int | None,
    session_id: str,
    purpose: str,
) -> dict[str, Any]:
    protocol = json.loads(json.dumps(source))
    agents = protocol_agents(protocol)
    if selected_agent_ids:
        requested = list(dict.fromkeys(selected_agent_ids))
        available = {agent["id"]: agent for agent in agents}
        unknown = [identifier for identifier in requested if identifier not in available]
        if unknown:
            raise ValueError(f"Unknown agent id(s): {', '.join(unknown)}")
        protocol["agents"] = [available[identifier] for identifier in requested]
    if repeats is not None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        protocol["design"]["repeats_per_agent"] = int(repeats)
    protocol["session"] = {
        "session_id": session_id,
        "purpose": purpose,
        "prepared_at": utc_now(),
    }
    validate_protocol(protocol)
    return protocol


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_assets(protocol: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for agent in protocol_agents(protocol):
        for value in agent.get("assets", []):
            path = resolve_project_path(value)
            if not path.is_file():
                raise FileNotFoundError(f"Required asset for {agent['id']} is missing: {path}")
            paths.append(path)
    torcs_path = PROJECT_ROOT / "torcs" / "wtorcs.exe"
    if not torcs_path.is_file():
        raise FileNotFoundError(f"TORCS executable is missing: {torcs_path}")
    paths.append(torcs_path)
    return paths


def build_schedule(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    agents = protocol_agents(protocol)
    repeats = int(protocol["design"]["repeats_per_agent"])
    order_seed = int(protocol["design"]["order_seed"])
    base_seed = int(protocol["design"]["base_agent_seed"])
    randomiser = random.Random(order_seed)
    canonical_index = {agent["id"]: index for index, agent in enumerate(agents)}
    schedule: list[dict[str, Any]] = []

    for block in range(1, repeats + 1):
        ordered_agents = list(agents)
        randomiser.shuffle(ordered_agents)
        for position, agent in enumerate(ordered_agents, start=1):
            index = canonical_index[agent["id"]]
            seed = base_seed + block * 100 + index
            schedule.append(
                {
                    "attempt_id": f"b{block:03d}-p{position:02d}-{agent['id']}",
                    "block": block,
                    "position": position,
                    "agent_trial": block,
                    "agent_id": agent["id"],
                    "agent_label": agent["label"],
                    "seed": seed,
                }
            )
    return schedule


def _run_git(*arguments: str, cwd: Path = PROJECT_ROOT) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("gym", "matplotlib", "numpy", "scipy", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_artifacts(protocol_path: Path) -> list[Path]:
    candidates = [
        protocol_path,
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "runner" / "torcs_runner.py",
        PROJECT_ROOT / "src" / "gui" / "torcs_config.py",
        PROJECT_ROOT / "src" / "agents" / "rule_based_agent.py",
        PROJECT_ROOT / "src" / "agents" / "map_aware_agent.py",
        PROJECT_ROOT / "src" / "agents" / "dyna_q_agent.py",
        PROJECT_ROOT / "src" / "agents" / "n_step_td3_agent.py",
        PROJECT_ROOT / "src" / "agents" / "sensor_n_step_td3_agent.py",
        PROJECT_ROOT / "torcs-wrapper" / "gym_torcs" / "gym_torcs.py",
        PROJECT_ROOT / "torcs-wrapper" / "gym_torcs" / "snakeoil3_gym.py",
    ]
    return [path for path in candidates if path.is_file()]


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_manifest(
    protocol: Mapping[str, Any],
    protocol_snapshot_path: Path,
) -> dict[str, Any]:
    artifact_paths = _source_artifacts(DEFAULT_PROTOCOL.resolve())
    artifact_paths.extend(validate_assets(protocol))
    unique_paths = list(dict.fromkeys(path.resolve() for path in artifact_paths))
    nested_root = PROJECT_ROOT / "torcs-wrapper" / "gym_torcs"
    return {
        "created_at": utc_now(),
        "protocol_snapshot_sha256": sha256_file(protocol_snapshot_path),
        "git": {
            "revision": _run_git("rev-parse", "HEAD"),
            "branch": _run_git("branch", "--show-current"),
            "status_porcelain": _run_git("status", "--porcelain"),
            "nested_revision": _run_git("rev-parse", "HEAD", cwd=nested_root),
            "nested_status_porcelain": _run_git("status", "--porcelain", cwd=nested_root),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "dependencies": _dependency_versions(),
        },
        "artifacts": {
            _relative_or_absolute(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in unique_paths
        },
    }


def prepare_session(
    *,
    protocol_path: Path,
    results_root: Path,
    session_id: str,
    selected_agent_ids: Sequence[str] | None = None,
    repeats: int | None = None,
    purpose: str = "final",
) -> Path:
    validate_session_id(session_id)
    base_protocol = load_json(protocol_path)
    validate_protocol(base_protocol)
    protocol = select_protocol(
        base_protocol,
        selected_agent_ids=selected_agent_ids,
        repeats=repeats,
        session_id=session_id,
        purpose=purpose,
    )
    validate_assets(protocol)

    target = session_dir(results_root, session_id)
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Session already exists: {target}. Resume it with 'run --session "
            f"{session_id}', or prepare a new session id."
        ) from error

    for directory in ("telemetry", "infrastructure", "agent-internal", "analysis"):
        (target / directory).mkdir()

    snapshot_path = target / "protocol_snapshot.json"
    atomic_write_json(snapshot_path, protocol)
    atomic_write_csv(target / "schedule.csv", SCHEDULE_FIELDS, build_schedule(protocol))
    atomic_write_csv(target / "raw_attempts.csv", ATTEMPT_FIELDS, [])
    atomic_write_csv(
        target / "infrastructure_failures.csv", INFRASTRUCTURE_FIELDS, []
    )
    atomic_write_json(target / "manifest.json", build_manifest(protocol, snapshot_path))
    return target


def verify_frozen_session(target: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = target / "protocol_snapshot.json"
    manifest_path = target / "manifest.json"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Session is incomplete: {target}")
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    actual_protocol_hash = sha256_file(protocol_path)
    expected_protocol_hash = manifest.get("protocol_snapshot_sha256")
    if actual_protocol_hash != expected_protocol_hash:
        raise RuntimeError("The prepared protocol snapshot has changed; create a new session")

    mismatches = []
    for recorded_path, metadata in manifest.get("artifacts", {}).items():
        path = resolve_project_path(recorded_path)
        if not path.is_file():
            mismatches.append(f"missing: {recorded_path}")
        elif sha256_file(path) != metadata.get("sha256"):
            mismatches.append(f"changed: {recorded_path}")
    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Frozen evaluation artifacts no longer match the manifest:\n  - " + detail
        )
    return protocol, manifest


@contextlib.contextmanager
def legacy_argv_guard() -> Iterable[None]:
    """Hide this tool's CLI flags from the legacy snakeoil option parser."""

    original = list(sys.argv)
    sys.argv[:] = original[:1]
    try:
        yield
    finally:
        sys.argv[:] = original


def _agent_definition(protocol: Mapping[str, Any], agent_id: str) -> dict[str, Any]:
    for definition in protocol_agents(protocol):
        if definition["id"] == agent_id:
            return definition
    raise KeyError(f"Agent is not present in protocol: {agent_id}")


def _seed_runtime(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def instantiate_agent(
    definition: Mapping[str, Any],
    *,
    seed: int,
    internal_telemetry_path: Path,
    max_steps: int,
    target_laps: int,
) -> Any:
    module_name, class_name = str(definition["class"]).rsplit(".", 1)
    agent_class = getattr(importlib.import_module(module_name), class_name)
    kwargs = dict(definition.get("constructor", {}))
    for key, value in list(kwargs.items()):
        if key.endswith("_path") and isinstance(value, str):
            kwargs[key] = resolve_project_path(value)
    signature = inspect.signature(agent_class)
    if definition.get("seeded") and "seed" in signature.parameters:
        kwargs["seed"] = seed
    if "telemetry_path" in signature.parameters:
        kwargs["telemetry_path"] = internal_telemetry_path
    agent = agent_class(**kwargs)
    agent.max_steps = int(max_steps)
    agent.target_laps = int(target_laps)
    return agent


def finite_values(samples: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = []
    for sample in samples:
        try:
            value = float(sample.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def count_true_events(flags: Sequence[bool]) -> int:
    count = 0
    previous = False
    for flag in flags:
        current = bool(flag)
        if current and not previous:
            count += 1
        previous = current
    return count


def _serialise_telemetry_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def write_telemetry(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    extras = sorted(
        {key for sample in samples for key in sample if key not in CORE_TELEMETRY_FIELDS}
    )
    fields = CORE_TELEMETRY_FIELDS + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {key: _serialise_telemetry_value(value) for key, value in sample.items()}
            )
    temporary.replace(path)


def classify_attempt(
    results: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    reverse_threshold_kmh: float,
    target_laps: int,
) -> dict[str, Any]:
    speed_values = finite_values(samples, "speed_x")
    distance_values = finite_values(samples, "dist_raced")
    damage_values = finite_values(samples, "damage")
    off_track_flags = [bool(sample.get("off_track")) for sample in samples]
    reverse_flags = [
        math.isfinite(speed) and speed < reverse_threshold_kmh
        for speed in [
            _finite_or_nan(sample.get("speed_x"))
            for sample in samples
        ]
    ]
    termination_reason = str(results.get("termination_reason") or "unknown")
    laps_completed = int(results.get("laps_completed") or 0)
    off_track_steps = max(int(results.get("off_track") or 0), sum(off_track_flags))
    reverse_steps = sum(reverse_flags)
    stall_events = int(termination_reason == "stuck")
    lap_time = _finite_or_none(results.get("best_lap_time_seconds"))
    clean_completion = bool(
        laps_completed >= target_laps
        and lap_time is not None
        and termination_reason == "target_laps_completed"
        and off_track_steps == 0
        and reverse_steps == 0
        and stall_events == 0
    )

    if clean_completion:
        failure_category = "clean_completion"
    elif laps_completed >= target_laps:
        failure_category = "completed_unclean"
    elif off_track_steps:
        failure_category = "off_track"
    elif reverse_steps:
        failure_category = "reverse"
    elif stall_events:
        failure_category = "stall"
    elif termination_reason == "crashed":
        failure_category = "crash"
    elif termination_reason == "max_steps":
        failure_category = "max_steps"
    else:
        failure_category = termination_reason

    return {
        "valid_attempt": True,
        "clean_completion": clean_completion,
        "failure_category": failure_category,
        "laps_completed": laps_completed,
        "lap_time_seconds": lap_time,
        "distance_m": max(distance_values) if distance_values else None,
        "steps": int(results.get("steps") or 0),
        "total_score": _finite_or_none(results.get("total_score")),
        "mean_speed_kmh": (
            sum(speed_values) / len(speed_values) if speed_values else None
        ),
        "max_speed_kmh": max(speed_values) if speed_values else None,
        "off_track_steps": off_track_steps,
        "off_track_events": count_true_events(off_track_flags),
        "reverse_steps": reverse_steps,
        "reverse_events": count_true_events(reverse_flags),
        "stall_events": stall_events,
        "damage_delta": (
            max(damage_values) - damage_values[0] if damage_values else None
        ),
        "termination_reason": termination_reason,
    }


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite_or_none(value: Any) -> float | None:
    number = _finite_or_nan(value)
    return number if math.isfinite(number) else None


def execute_attempt(
    protocol: Mapping[str, Any],
    target: Path,
    schedule_row: Mapping[str, Any],
) -> dict[str, Any]:
    from gui.torcs_config import TorcsRaceSetup, TorcsRuntimeConfig
    from runner.torcs_runner import TorcsRunner

    environment = protocol["environment"]
    design = protocol["design"]
    attempt_id = str(schedule_row["attempt_id"])
    agent_id = str(schedule_row["agent_id"])
    seed = int(schedule_row["seed"])
    definition = _agent_definition(protocol, agent_id)
    telemetry_path = target / "telemetry" / f"{attempt_id}.csv.gz"
    internal_path = target / "agent-internal" / f"{attempt_id}.csv"
    setup = TorcsRaceSetup(
        track_id=str(environment["track_id"]),
        track_category=str(environment["track_category"]),
        car_id=str(environment["car_id"]),
    )
    samples: list[dict[str, Any]] = []
    runner = TorcsRunner()
    _seed_runtime(seed)
    started_at = utc_now()

    try:
        agent = instantiate_agent(
            definition,
            seed=seed,
            internal_telemetry_path=internal_path,
            max_steps=int(environment["max_steps"]),
            target_laps=int(environment["target_laps"]),
        )
        with TorcsRuntimeConfig(setup, project_root=PROJECT_ROOT):
            with legacy_argv_guard():
                runner.launch()
                runner.connect()
                runner.load_track(environment["track_id"])
                results = runner.run(
                    agent,
                    telemetry_callback=lambda sample: samples.append(dict(sample)),
                    telemetry_interval_steps=int(
                        environment.get("telemetry_interval_steps", 1)
                    ),
                    shutdown_on_finish=True,
                )
    except BaseException as error:
        setattr(error, "evaluation_samples", samples)
        raise
    finally:
        runner.shutdown()

    if not samples:
        samples = [dict(sample) for sample in results.get("telemetry_samples", [])]
    write_telemetry(telemetry_path, samples)
    metrics = classify_attempt(
        results,
        samples,
        reverse_threshold_kmh=float(design["reverse_speed_threshold_kmh"]),
        target_laps=int(environment["target_laps"]),
    )
    return {
        **{field: schedule_row.get(field) for field in SCHEDULE_FIELDS},
        "started_at": results.get("started_at") or started_at,
        "finished_at": utc_now(),
        **metrics,
        "telemetry_path": telemetry_path.relative_to(target).as_posix(),
    }


def record_infrastructure_failure(
    target: Path,
    schedule_row: Mapping[str, Any],
    error: BaseException,
) -> None:
    existing = read_csv_rows(target / "infrastructure_failures.csv")
    attempt_id = str(schedule_row["attempt_id"])
    retry_number = 1 + sum(row.get("attempt_id") == attempt_id for row in existing)
    stem = f"{attempt_id}-retry-{retry_number:02d}"
    traceback_path = target / "infrastructure" / f"{stem}.txt"
    traceback_path.write_text(
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        encoding="utf-8",
    )
    samples = getattr(error, "evaluation_samples", [])
    partial_path: Path | None = None
    if samples:
        partial_path = target / "infrastructure" / f"{stem}-partial.csv.gz"
        write_telemetry(partial_path, samples)
    append_csv_atomic(
        target / "infrastructure_failures.csv",
        INFRASTRUCTURE_FIELDS,
        {
            "attempt_id": attempt_id,
            "block": schedule_row["block"],
            "agent_id": schedule_row["agent_id"],
            "recorded_at": utc_now(),
            "retry_number": retry_number,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback_path": traceback_path.relative_to(target).as_posix(),
            "partial_telemetry_path": (
                partial_path.relative_to(target).as_posix() if partial_path else ""
            ),
        },
    )


def run_session(
    target: Path,
    *,
    max_attempts: int | None = None,
) -> int:
    protocol, _manifest = verify_frozen_session(target)
    schedule = read_csv_rows(target / "schedule.csv")
    completed_rows = read_csv_rows(target / "raw_attempts.csv")
    completed = {row["attempt_id"] for row in completed_rows}
    pending = [row for row in schedule if row["attempt_id"] not in completed]
    if max_attempts is not None:
        pending = pending[:max_attempts]

    if not pending:
        print("No pending attempts. The selected session is complete.")
        print_status(target)
        return 0

    print(
        f"Running {len(pending)} pending attempt(s) for session "
        f"'{target.name}'."
    )
    for index, schedule_row in enumerate(pending, start=1):
        print(
            f"\n[{index}/{len(pending)}] Block {schedule_row['block']}, "
            f"position {schedule_row['position']}: {schedule_row['agent_label']}"
        )
        try:
            attempt = execute_attempt(protocol, target, schedule_row)
        except KeyboardInterrupt:
            print("\nEvaluation interrupted. The current attempt remains pending.")
            return 130
        except BaseException as error:
            record_infrastructure_failure(target, schedule_row, error)
            print(
                "\nInfrastructure failure recorded; this attempt was not counted. "
                "Fix the cause and run the same command to resume."
            )
            print(f"{type(error).__name__}: {error}")
            return 2
        append_csv_atomic(target / "raw_attempts.csv", ATTEMPT_FIELDS, attempt)
        outcome = "clean completion" if attempt["clean_completion"] else attempt["failure_category"]
        lap = attempt.get("lap_time_seconds")
        lap_text = f", lap {float(lap):.3f} s" if lap not in (None, "") else ""
        print(f"Recorded: {outcome}{lap_text}")

    print_status(target)
    return 0


def print_status(target: Path) -> None:
    protocol, _manifest = verify_frozen_session(target)
    schedule = read_csv_rows(target / "schedule.csv")
    attempts = read_csv_rows(target / "raw_attempts.csv")
    infrastructure = read_csv_rows(target / "infrastructure_failures.csv")
    completed_ids = {row["attempt_id"] for row in attempts}
    print(f"Session: {target.name}")
    print(f"Valid attempts: {len(attempts)}/{len(schedule)}")
    print(f"Infrastructure failure records: {len(infrastructure)}")
    for agent in protocol_agents(protocol):
        planned = sum(row["agent_id"] == agent["id"] for row in schedule)
        rows = [row for row in attempts if row["agent_id"] == agent["id"]]
        clean = sum(_csv_bool(row.get("clean_completion")) for row in rows)
        print(f"  {agent['label']}: {len(rows)}/{planned} valid, {clean} clean")
    pending = len(schedule) - len(completed_ids)
    if pending:
        print(f"Pending attempts: {pending}")
    else:
        print("All scheduled attempts are complete.")


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled, resumable TORCS evaluation for the MAT099 dissertation."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Session output root (default: controlled-evaluation/results).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze a new final session.")
    prepare.add_argument("--session-id", required=True)
    prepare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)

    run = subparsers.add_parser("run", help="Run or resume a prepared session.")
    run.add_argument("--session", required=True)
    run.add_argument(
        "--max-attempts",
        type=int,
        help="Optional debugging limit; omit for the final experiment.",
    )

    status = subparsers.add_parser("status", help="Show session progress.")
    status.add_argument("--session", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="Prepare and run a four-attempt Agent 7/8 pipeline check."
    )
    smoke.add_argument("--session-id", required=True)
    smoke.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            target = prepare_session(
                protocol_path=args.protocol.resolve(),
                results_root=args.results_root,
                session_id=args.session_id,
            )
            print(f"Prepared session: {target}")
            print_status(target)
            return 0
        if args.command == "run":
            if args.max_attempts is not None and args.max_attempts <= 0:
                raise ValueError("--max-attempts must be positive")
            return run_session(
                session_dir(args.results_root, args.session),
                max_attempts=args.max_attempts,
            )
        if args.command == "status":
            print_status(session_dir(args.results_root, args.session))
            return 0
        if args.command == "smoke":
            target = prepare_session(
                protocol_path=args.protocol.resolve(),
                results_root=args.results_root,
                session_id=args.session_id,
                selected_agent_ids=("agent7_v4", "agent8_final"),
                repeats=2,
                purpose="pipeline-smoke-test-not-for-reporting",
            )
            print(f"Prepared smoke session: {target}")
            return run_session(target)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
