from __future__ import annotations

import csv
import sqlite3
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_RUN_LIMIT = 100

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


@dataclass(frozen=True)
class AgentOption:
    name: str
    agent_type: str
    version: str
    class_path: str
    uses_full_control: bool
    requires_racing_line: bool
    max_steps: int | None
    target_laps: int | None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.agent_type} v{self.version})"


@dataclass(frozen=True)
class TrackOption:
    track_id: str
    display_name: str
    category: str
    path: Path
    racing_line_path: Path | None

    @property
    def label(self) -> str:
        if self.display_name and self.display_name != self.track_id:
            return f"{self.display_name} ({self.track_id})"
        return self.track_id

    @property
    def has_racing_line(self) -> bool:
        return self.racing_line_path is not None


@dataclass(frozen=True)
class CarOption:
    car_id: str
    display_name: str
    category: str
    path: Path

    @property
    def label(self) -> str:
        if self.display_name and self.display_name != self.car_id:
            return f"{self.display_name} ({self.car_id})"
        return self.car_id


@dataclass(frozen=True)
class RunSummary:
    run_id: int | str
    started_at: str
    agent_name: str
    track: str
    best_lap_time_seconds: float | None
    avg_speed: float | None
    off_track_count: int | None
    termination_reason: str
    source: str


@dataclass(frozen=True)
class ProjectOptions:
    agents: list[AgentOption]
    tracks: list[TrackOption]
    cars: list[CarOption]
    runs: list[RunSummary]
    run_history_source: str


def load_project_options(
    project_root: Path = PROJECT_ROOT,
    *,
    run_limit: int = DEFAULT_RUN_LIMIT,
) -> ProjectOptions:
    runs, run_source = discover_run_history(project_root, limit=run_limit)
    return ProjectOptions(
        agents=discover_agents(),
        tracks=discover_tracks(project_root),
        cars=discover_cars(project_root),
        runs=runs,
        run_history_source=run_source,
    )


def discover_agents() -> list[AgentOption]:
    from agents.map_aware_agent import MapAwareAgent
    from agents.random_agent import RandomAgent
    from agents.rule_based_agent import RuleBasedAgent

    return [
        _agent_option(MapAwareAgent),
        _agent_option(RuleBasedAgent),
        _agent_option(RandomAgent),
    ]


def discover_tracks(project_root: Path = PROJECT_ROOT) -> list[TrackOption]:
    tracks_root = project_root / "torcs" / "tracks"
    racing_lines_root = project_root / "data" / "racing_lines"
    tracks: list[TrackOption] = []

    for category_dir in sorted(path for path in tracks_root.iterdir() if path.is_dir()):
        category = category_dir.name
        for track_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            xml_path = track_dir / f"{track_dir.name}.xml"
            if not xml_path.is_file():
                continue

            metadata = _read_track_metadata(xml_path)
            tracks.append(
                TrackOption(
                    track_id=track_dir.name,
                    display_name=metadata.get("display_name") or track_dir.name,
                    category=metadata.get("category") or category,
                    path=xml_path,
                    racing_line_path=_existing_racing_line_path(
                        racing_lines_root,
                        track_dir.name,
                    ),
                )
            )

    return sorted(tracks, key=lambda track: (track.category, track.track_id))


def discover_cars(project_root: Path = PROJECT_ROOT) -> list[CarOption]:
    cars_root = project_root / "torcs" / "cars"
    cars: list[CarOption] = []

    for car_dir in sorted(path for path in cars_root.iterdir() if path.is_dir()):
        xml_path = car_dir / f"{car_dir.name}.xml"
        if not xml_path.is_file():
            continue

        metadata = _read_car_metadata(xml_path)
        cars.append(
            CarOption(
                car_id=car_dir.name,
                display_name=metadata.get("display_name") or car_dir.name,
                category=metadata.get("category") or "unknown",
                path=xml_path,
            )
        )

    return sorted(cars, key=lambda car: (car.category.casefold(), car.car_id.casefold()))


def discover_run_history(
    project_root: Path = PROJECT_ROOT,
    *,
    limit: int = DEFAULT_RUN_LIMIT,
) -> tuple[list[RunSummary], str]:
    database_path = project_root / "data" / "generated" / "race_results.db"
    if database_path.is_file():
        return _read_runs_from_database(database_path, limit), str(database_path)

    csv_path = project_root / "latest_race_runs.csv"
    if csv_path.is_file():
        return _read_runs_from_csv(csv_path, limit), str(csv_path)

    return [], "No saved run history found"


def compatible_tracks_for_agent(
    agent: AgentOption,
    tracks: list[TrackOption],
) -> list[TrackOption]:
    if agent.requires_racing_line:
        return [track for track in tracks if track.has_racing_line]
    return tracks


def _agent_option(agent_class: type) -> AgentOption:
    agent_type = str(getattr(agent_class, "agent_type", agent_class.__name__))
    return AgentOption(
        name=str(getattr(agent_class, "name", agent_class.__name__)),
        agent_type=agent_type,
        version=str(getattr(agent_class, "version", "unknown")),
        class_path=f"{agent_class.__module__}.{agent_class.__name__}",
        uses_full_control=bool(getattr(agent_class, "uses_full_control", False)),
        requires_racing_line=bool(
            getattr(agent_class, "requires_racing_line", agent_type == "map_aware")
        ),
        max_steps=_optional_int(getattr(agent_class, "max_steps", None)),
        target_laps=_optional_int(getattr(agent_class, "target_laps", None)),
    )


def _existing_racing_line_path(racing_lines_root: Path, track_id: str) -> Path | None:
    path = racing_lines_root / f"{track_id}.json"
    return path if path.is_file() else None


def _read_track_metadata(path: Path) -> dict[str, str]:
    root = _read_torcs_xml(path)
    metadata = {"display_name": root.get("name", path.stem)}
    header = _find_named_section(root, "Header")
    if header is not None:
        attributes = _direct_attributes(header)
        metadata["display_name"] = attributes.get(
            "name",
            metadata["display_name"],
        )
        metadata["category"] = attributes.get("category", "")
    return metadata


def _read_car_metadata(path: Path) -> dict[str, str]:
    root = _read_torcs_xml(path)
    metadata = {"display_name": root.get("name", path.stem)}
    car_section = _find_named_section(root, "Car")
    if car_section is not None:
        metadata["category"] = _direct_attributes(car_section).get("category", "")
    return metadata


def _read_torcs_xml(path: Path) -> ET.Element:
    xml_text = (
        path.read_text(encoding="utf-8", errors="replace")
        .replace("&default-surfaces;", "")
        .replace("&default-objects;", "")
    )
    return ET.fromstring(xml_text)


def _find_named_section(root: ET.Element, name: str) -> ET.Element | None:
    return next(
        (
            section
            for section in root.findall("section")
            if section.get("name") == name
        ),
        None,
    )


def _direct_attributes(section: ET.Element) -> dict[str, str]:
    return {
        child.get("name", ""): child.get("val", "")
        for child in section
        if child.tag in {"attnum", "attstr"} and child.get("name")
    }


def _read_runs_from_database(path: Path, limit: int) -> list[RunSummary]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                race_runs.id,
                race_runs.started_at,
                race_runs.track,
                race_runs.best_lap_time_seconds,
                race_runs.avg_speed,
                race_runs.off_track_count,
                race_runs.termination_reason,
                agents.name AS agent_name
            FROM race_runs
            JOIN agents ON agents.id = race_runs.agent_id
            ORDER BY race_runs.started_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()

    return [
        RunSummary(
            run_id=row["id"],
            started_at=row["started_at"] or "",
            agent_name=row["agent_name"] or "",
            track=row["track"] or "",
            best_lap_time_seconds=_optional_float(row["best_lap_time_seconds"]),
            avg_speed=_optional_float(row["avg_speed"]),
            off_track_count=_optional_int(row["off_track_count"]),
            termination_reason=row["termination_reason"] or "",
            source="sqlite",
        )
        for row in rows
    ]


def _read_runs_from_csv(path: Path, limit: int) -> list[RunSummary]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rows.sort(key=lambda row: row.get("started_at", ""), reverse=True)
    return [
        RunSummary(
            run_id=row.get("id", ""),
            started_at=row.get("started_at", ""),
            agent_name=row.get("agent_name") or f"Agent #{row.get('agent_id', '')}",
            track=row.get("track", ""),
            best_lap_time_seconds=_optional_float(row.get("best_lap_time_seconds")),
            avg_speed=_optional_float(row.get("avg_speed")),
            off_track_count=_optional_int(row.get("off_track_count")),
            termination_reason=row.get("termination_reason", ""),
            source="csv",
        )
        for row in rows[: max(1, limit)]
    ]


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
        return int(value)
    except (TypeError, ValueError):
        return None
