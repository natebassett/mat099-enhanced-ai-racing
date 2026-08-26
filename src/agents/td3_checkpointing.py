from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping


MIN_LAP_RELIABILITY_TRIALS = 5


def metadata_path_for_policy(policy_path: Path) -> Path:
    return Path(policy_path).with_suffix(".metadata.json")


def read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    temporary = _temporary_sibling(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def atomic_save_policy(
    save_policy: Callable[[Path], None],
    target_path: Path,
) -> None:
    target = Path(target_path)
    temporary = _temporary_policy_sibling(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_policy(temporary)
        if not temporary.is_file():
            raise RuntimeError(
                f"policy saver did not create the staged checkpoint: {temporary}"
            )
        _flush_file(temporary)
        os.replace(temporary, target)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def promote_policy_bundle_atomically(
    candidate_path: Path,
    champion_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Stage complete artifacts before replacing either champion file."""
    source = Path(candidate_path)
    target = Path(champion_path)
    if not source.is_file():
        raise FileNotFoundError(f"candidate policy does not exist: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target_metadata = metadata_path_for_policy(target)
    staged_model = _temporary_policy_sibling(target)
    staged_metadata = _temporary_sibling(target_metadata)
    replace_model = source.resolve() != target.resolve()
    try:
        if replace_model:
            shutil.copy2(source, staged_model)
            _flush_file(staged_model)
        with staged_metadata.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if replace_model:
            os.replace(staged_model, target)
        os.replace(staged_metadata, target_metadata)
    finally:
        for staged_path in (staged_model, staged_metadata):
            with contextlib.suppress(OSError):
                staged_path.unlink()


def completed_lap_quality(lap_record: Mapping[str, Any]) -> tuple[int, float, float]:
    completed = max(0, int(_finite_float(lap_record.get("completed_laps"))))
    trials = max(1, int(_finite_float(lap_record.get("validation_trials"), 1.0)))
    reliability = _finite_float(
        lap_record.get("reliability"),
        completed / trials,
    )
    reliability = min(1.0, max(0.0, reliability))
    reliability *= min(1.0, trials / MIN_LAP_RELIABILITY_TRIALS)
    lap_time = _finite_float(
        lap_record.get("best_lap_seconds"),
        float("inf"),
    )
    if lap_time <= 0.0:
        lap_time = float("inf")
    return completed, reliability, -lap_time


def promote_best_completed_lap(
    candidate_path: Path,
    champion_path: Path,
    source_metadata: Mapping[str, Any],
    lap_record: Mapping[str, Any],
) -> bool:
    completed_laps = int(_finite_float(lap_record.get("completed_laps")))
    best_lap_seconds = _finite_float(
        lap_record.get("best_lap_seconds"),
        float("nan"),
    )
    if (
        completed_laps < 1
        or not math.isfinite(best_lap_seconds)
        or best_lap_seconds <= 0.0
    ):
        return False
    existing_metadata = read_json_mapping(
        metadata_path_for_policy(champion_path)
    )
    existing = existing_metadata.get("best_completed_lap")
    if isinstance(existing, Mapping) and completed_lap_quality(
        existing
    ) >= completed_lap_quality(lap_record):
        return False

    metadata = {
        **dict(source_metadata),
        "checkpoint_type": "best_completed_lap",
        "best_completed_lap": dict(lap_record),
    }
    promote_policy_bundle_atomically(
        candidate_path,
        champion_path,
        metadata,
    )
    return True


def _temporary_sibling(path: Path) -> Path:
    target = Path(path)
    return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")


def _temporary_policy_sibling(path: Path) -> Path:
    target = Path(path)
    suffix = target.suffix if target.suffix else ".zip"
    return target.with_name(
        f".{target.stem}.{uuid.uuid4().hex}.tmp{suffix}"
    )


def _flush_file(path: Path) -> None:
    with Path(path).open("r+b") as file:
        file.flush()
        os.fsync(file.fileno())


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)
