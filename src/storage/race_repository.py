import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "generated" / "race_results.db"
MIGRATIONS_PATH = Path(__file__).resolve().parent / "migrations"


class RaceRepository:
    """Stores compact race summaries and agent metadata in SQLite."""

    def __init__(self, database_path: Optional[Path] = None):
        self.database_path = Path(database_path or DEFAULT_DATABASE_PATH)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._apply_migrations()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _apply_migrations(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )
                )
                """
            )
            applied_versions = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }

            for migration_path in sorted(MIGRATIONS_PATH.glob("*.sql")):
                version = int(migration_path.stem.split("_", 1)[0])
                if version in applied_versions:
                    continue

                migration_name = migration_path.name.replace("'", "''")
                migration_sql = migration_path.read_text(encoding="utf-8")
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    {migration_sql}
                    INSERT INTO schema_migrations(version, name)
                    VALUES ({version}, '{migration_name}');
                    COMMIT;
                    """
                )

    def register_agent(
        self,
        *,
        name: str,
        agent_type: str,
        version: str,
        config: Optional[Mapping[str, Any]] = None,
        code_commit: Optional[str] = None,
    ) -> int:
        config_json = json.dumps(
            dict(config or {}),
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id
                FROM agents
                WHERE name = ? AND version = ? AND config_json = ?
                """,
                (name, version, config_json),
            ).fetchone()

            if existing is not None:
                return int(existing["id"])

            cursor = connection.execute(
                """
                INSERT INTO agents(name, agent_type, version, config_json, code_commit)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, agent_type, version, config_json, code_commit),
            )
            return int(cursor.lastrowid)

    def create_experiment(self, name: str, description: Optional[str] = None) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO experiments(name, description) VALUES (?, ?)",
                (name, description),
            )
            return int(cursor.lastrowid)

    def record_run(
        self,
        *,
        agent_id: int,
        track: str,
        seed: Optional[int],
        results: Mapping[str, Any],
        experiment_id: Optional[int] = None,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO race_runs(
                    experiment_id,
                    agent_id,
                    started_at,
                    track,
                    seed,
                    total_score,
                    steps,
                    max_speed,
                    avg_speed,
                    off_track_count,
                    laps_completed,
                    best_lap_time_seconds,
                    average_lap_time_seconds,
                    duration_seconds,
                    termination_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    agent_id,
                    results["started_at"],
                    track,
                    seed,
                    results["total_score"],
                    results["steps"],
                    results["max_speed"],
                    results["avg_speed"],
                    results["off_track"],
                    results.get("laps_completed", 0),
                    results.get("best_lap_time_seconds"),
                    results.get("average_lap_time_seconds"),
                    results["duration_seconds"],
                    results["termination_reason"],
                ),
            )
            run_id = int(cursor.lastrowid)

            if metrics:
                connection.executemany(
                    """
                    INSERT INTO run_metrics(run_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                    """,
                    ((run_id, name, value) for name, value in metrics.items()),
                )

            return run_id

    def record_run_telemetry(
        self,
        run_id: int,
        samples: list[Mapping[str, Any]],
    ) -> None:
        if not samples:
            return

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO run_telemetry(
                    run_id,
                    step,
                    dist_from_start,
                    dist_raced,
                    speed_x,
                    speed_y,
                    angle,
                    track_pos,
                    damage,
                    steer,
                    accel,
                    brake,
                    gear,
                    reward,
                    off_track,
                    front_sensor,
                    min_track_sensor,
                    track_sensors_json,
                    cur_lap_time,
                    last_lap_time
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        run_id,
                        sample["step"],
                        sample.get("dist_from_start"),
                        sample.get("dist_raced"),
                        sample.get("speed_x"),
                        sample.get("speed_y"),
                        sample.get("angle"),
                        sample.get("track_pos"),
                        sample.get("damage"),
                        sample.get("steer"),
                        sample.get("accel"),
                        sample.get("brake"),
                        sample.get("gear"),
                        sample.get("reward"),
                        1 if sample.get("off_track") else 0,
                        sample.get("front_sensor"),
                        sample.get("min_track_sensor"),
                        _track_sensors_json(sample),
                        sample.get("cur_lap_time"),
                        sample.get("last_lap_time"),
                    )
                    for sample in samples
                ),
            )

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    race_runs.*,
                    agents.name AS agent_name,
                    agents.agent_type,
                    agents.version AS agent_version
                FROM race_runs
                JOIN agents ON agents.id = race_runs.agent_id
                WHERE race_runs.id = ?
                """,
                (run_id,),
            ).fetchone()

        return None if row is None else dict(row)

    def list_run_telemetry(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    step,
                    dist_from_start,
                    dist_raced,
                    speed_x,
                    speed_y,
                    angle,
                    track_pos,
                    damage,
                    steer,
                    accel,
                    brake,
                    gear,
                    reward,
                    off_track,
                    front_sensor,
                    min_track_sensor,
                    track_sensors_json,
                    cur_lap_time,
                    last_lap_time
                FROM run_telemetry
                WHERE run_id = ?
                ORDER BY step ASC
                """,
                (run_id,),
            ).fetchall()

        telemetry = []
        for row in rows:
            sample = dict(row)
            sample["track_sensors"] = _decode_track_sensors(
                sample.pop("track_sensors_json")
            )
            telemetry.append(sample)
        return telemetry

    def best_run_for_track(
        self,
        track: str,
        *,
        agent_type: Optional[str] = None,
    ) -> dict[str, Any] | None:
        where_clause = [
            "race_runs.track = ?",
            "race_runs.best_lap_time_seconds IS NOT NULL",
        ]
        parameters: list[Any] = [track]
        if agent_type:
            where_clause.append("agents.agent_type = ?")
            parameters.append(agent_type)

        query = f"""
            SELECT
                race_runs.*,
                agents.name AS agent_name,
                agents.agent_type,
                agents.version AS agent_version
            FROM race_runs
            JOIN agents ON agents.id = race_runs.agent_id
            WHERE {" AND ".join(where_clause)}
            ORDER BY race_runs.best_lap_time_seconds ASC
            LIMIT 1
        """

        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()

        return None if row is None else dict(row)

    def list_runs(
        self,
        *,
        sort_by: str = "started_at",
        descending: bool = True,
        limit: int = 100,
        agent_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        sortable_columns = {
            "started_at": "race_runs.started_at",
            "total_score": "race_runs.total_score",
            "steps": "race_runs.steps",
            "duration_seconds": "race_runs.duration_seconds",
            "best_lap_time": "race_runs.best_lap_time_seconds",
        }
        if sort_by not in sortable_columns:
            raise ValueError(f"Unsupported run sort column: {sort_by}")

        where_clause = "WHERE agents.agent_type = ?" if agent_type else ""
        parameters: list[Any] = [agent_type] if agent_type else []
        parameters.append(max(1, limit))
        direction = "DESC" if descending else "ASC"

        query = f"""
            SELECT
                race_runs.*,
                agents.name AS agent_name,
                agents.agent_type,
                agents.version AS agent_version
            FROM race_runs
            JOIN agents ON agents.id = race_runs.agent_id
            {where_clause}
            ORDER BY {sortable_columns[sort_by]} {direction}
            LIMIT ?
        """

        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]


def _track_sensors_json(sample: Mapping[str, Any]) -> str | None:
    values = sample.get("track_sensors")
    if not values:
        return None
    try:
        return json.dumps([float(value) for value in values], separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _decode_track_sensors(value: Any) -> list[float]:
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(decoded, list):
        return []

    sensors = []
    for item in decoded:
        try:
            sensors.append(float(item))
        except (TypeError, ValueError):
            continue
    return sensors
