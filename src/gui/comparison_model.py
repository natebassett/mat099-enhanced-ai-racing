from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from .telemetry_model import TelemetrySnapshot, build_explanation_frame, format_value
except ImportError:
    from telemetry_model import TelemetrySnapshot, build_explanation_frame, format_value


@dataclass(frozen=True)
class ComparisonSummary:
    headline: str
    details: tuple[str, ...]
    finish_time: float


def run_combo_label(run: Mapping[str, Any]) -> str:
    run_id = run.get("id", "--")
    agent = str(run.get("agent_name") or "Unknown agent")
    best_lap = _format_lap_time(run.get("best_lap_time_seconds"))
    result = str(run.get("termination_reason") or "unknown")
    return f"#{run_id} | {agent} | {best_lap} | {result}"


def run_title(run: Mapping[str, Any], fallback: str) -> str:
    run_id = run.get("id")
    agent = str(run.get("agent_name") or fallback)
    if run_id is None:
        return agent
    return f"Run #{run_id} - {agent}"


def snapshot_times(snapshots: list[TelemetrySnapshot]) -> list[float]:
    times: list[float] = []
    offset = 0.0
    previous_raw_time: float | None = None

    for index, snapshot in enumerate(snapshots):
        raw_time = snapshot_replay_time(snapshot, index)
        if previous_raw_time is not None and raw_time < previous_raw_time - 1.0:
            offset += previous_raw_time

        replay_time = raw_time + offset
        if times and replay_time < times[-1]:
            replay_time = times[-1]
        times.append(replay_time)
        previous_raw_time = raw_time

    return times


def snapshot_replay_time(snapshot: TelemetrySnapshot, fallback_index: int) -> float:
    if snapshot.cur_lap_time is not None and snapshot.cur_lap_time >= 0.0:
        return snapshot.cur_lap_time
    return float(fallback_index)


def replay_finish_time(snapshots: list[TelemetrySnapshot]) -> float:
    if not snapshots:
        return 0.0
    return max(snapshot_times(snapshots))


def comparison_finish_time(
    snapshots_a: list[TelemetrySnapshot],
    snapshots_b: list[TelemetrySnapshot],
) -> float:
    return max(replay_finish_time(snapshots_a), replay_finish_time(snapshots_b))


def nearest_snapshot_index(times: list[float], target_time: float) -> int:
    if not times:
        return 0

    insertion = bisect_left(times, target_time)
    if insertion <= 0:
        return 0
    if insertion >= len(times):
        return len(times) - 1

    before = insertion - 1
    after = insertion
    if abs(times[after] - target_time) < abs(target_time - times[before]):
        return after
    return before


def compare_runs(
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    snapshots_a: list[TelemetrySnapshot],
    snapshots_b: list[TelemetrySnapshot],
) -> ComparisonSummary:
    finish_time = comparison_finish_time(snapshots_a, snapshots_b)
    track = run_a.get("track") or run_b.get("track") or "this track"
    lap_delta = _numeric_delta(
        run_a.get("best_lap_time_seconds"),
        run_b.get("best_lap_time_seconds"),
    )
    speed_delta = _numeric_delta(run_a.get("avg_speed"), run_b.get("avg_speed"))

    if lap_delta is not None:
        if abs(lap_delta) < 0.001:
            headline = "Both runs saved the same best lap time."
        elif lap_delta < 0:
            headline = f"Run A was {abs(lap_delta):.3f}s faster than Run B."
        else:
            headline = f"Run B was {lap_delta:.3f}s faster than Run A."
    elif speed_delta is not None:
        speed_delta_kmh = abs(speed_delta) * 50.0
        if abs(speed_delta_kmh) < 0.1:
            headline = "Both runs had almost the same average speed."
        elif speed_delta > 0:
            headline = f"Run A averaged {speed_delta_kmh:.1f} km/h more than Run B."
        else:
            headline = f"Run B averaged {speed_delta_kmh:.1f} km/h more than Run A."
    else:
        headline = "Comparison loaded for the same track."

    details = (
        f"Track: {track}.",
        _lap_detail(run_a, run_b),
        _speed_detail(run_a, run_b),
        _result_detail(run_a, run_b),
        (
            f"Replay samples: Run A has {len(snapshots_a)}, "
            f"Run B has {len(snapshots_b)}."
        ),
        f"Synced replay span: {finish_time:.1f}s.",
    )
    return ComparisonSummary(headline=headline, details=details, finish_time=finish_time)


def moment_comparison_reasons(
    snapshot_a: TelemetrySnapshot | None,
    snapshot_b: TelemetrySnapshot | None,
) -> tuple[str, ...]:
    if snapshot_a is None or snapshot_b is None:
        return ("Load two runs with saved telemetry to compare a moment.",)

    reasons = []
    speed_reason = _speed_moment_reason(snapshot_a, snapshot_b)
    if speed_reason:
        reasons.append(speed_reason)

    pedal_reason = _pedal_moment_reason(snapshot_a, snapshot_b)
    if pedal_reason:
        reasons.append(pedal_reason)

    road_reason = _road_moment_reason(snapshot_a, snapshot_b)
    if road_reason:
        reasons.append(road_reason)

    if not reasons:
        reasons.append("At this moment the two runs look broadly similar.")
    return tuple(reasons[:3])


def moment_panel_text(snapshot: TelemetrySnapshot | None) -> tuple[str, str, str]:
    if snapshot is None:
        return (
            "Idle",
            "No telemetry sample selected.",
            "- No comparison sample.",
        )

    frame = build_explanation_frame(snapshot)
    values = (
        f"Speed: {format_value(snapshot.speed_kmh, '.1f')} km/h\n"
        f"Throttle: {format_value(snapshot.throttle_pct, '.0f')}%\n"
        f"Brake: {format_value(snapshot.brake_pct, '.0f')}%\n"
        f"Track position: {format_value(snapshot.track_pos, '.3f')}"
    )
    return frame.severity, frame.headline, values


def _format_lap_time(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}s"
    return "no lap"


def _format_run_avg_speed(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 50.0:.1f} km/h"
    return "--"


def _numeric_delta(value_a: object, value_b: object) -> float | None:
    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return None
    return float(value_a) - float(value_b)


def _lap_detail(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> str:
    return (
        "Best lap: "
        f"Run A {_format_lap_time(run_a.get('best_lap_time_seconds'))}; "
        f"Run B {_format_lap_time(run_b.get('best_lap_time_seconds'))}."
    )


def _speed_detail(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> str:
    return (
        "Average speed: "
        f"Run A {_format_run_avg_speed(run_a.get('avg_speed'))}; "
        f"Run B {_format_run_avg_speed(run_b.get('avg_speed'))}."
    )


def _result_detail(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> str:
    return (
        "Race result: "
        f"Run A {run_a.get('termination_reason') or 'unknown'}; "
        f"Run B {run_b.get('termination_reason') or 'unknown'}."
    )


def _speed_moment_reason(
    snapshot_a: TelemetrySnapshot,
    snapshot_b: TelemetrySnapshot,
) -> str | None:
    if snapshot_a.speed_kmh is None or snapshot_b.speed_kmh is None:
        return None

    delta = snapshot_a.speed_kmh - snapshot_b.speed_kmh
    if abs(delta) < 5.0:
        return "The cars are at a similar speed."
    if delta > 0:
        return f"Run A is {delta:.1f} km/h faster at this point."
    return f"Run B is {abs(delta):.1f} km/h faster at this point."


def _pedal_moment_reason(
    snapshot_a: TelemetrySnapshot,
    snapshot_b: TelemetrySnapshot,
) -> str | None:
    state_a = _pedal_state(snapshot_a)
    state_b = _pedal_state(snapshot_b)
    if state_a == state_b:
        return f"Both cars are {state_a}."
    return f"Run A is {state_a}, while Run B is {state_b}."


def _pedal_state(snapshot: TelemetrySnapshot) -> str:
    brake = snapshot.brake_pct or 0.0
    throttle = snapshot.throttle_pct or 0.0
    if brake >= 35.0:
        return "braking hard"
    if brake >= 8.0:
        return "braking"
    if throttle >= 35.0:
        return "accelerating"
    if throttle >= 8.0:
        return "gently accelerating"
    return "coasting"


def _road_moment_reason(
    snapshot_a: TelemetrySnapshot,
    snapshot_b: TelemetrySnapshot,
) -> str | None:
    if snapshot_a.track_pos is None or snapshot_b.track_pos is None:
        return None

    edge_a = abs(snapshot_a.track_pos)
    edge_b = abs(snapshot_b.track_pos)
    if edge_a > edge_b + 0.22:
        return "Run A is closer to the edge of the road."
    if edge_b > edge_a + 0.22:
        return "Run B is closer to the edge of the road."
    if edge_a < 0.35 and edge_b < 0.35:
        return "Both cars are staying near the middle of the road."
    return "Both cars are using a similar part of the road."
