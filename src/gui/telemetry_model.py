from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TelemetrySnapshot:
    step: int
    speed_kmh: float | None
    gear: int | None
    throttle_pct: float | None
    brake_pct: float | None
    steer: float | None
    track_pos: float | None
    damage: float | None
    reward: float | None
    off_track: bool
    front_sensor: float | None
    min_track_sensor: float | None
    cur_lap_time: float | None
    last_lap_time: float | None

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any]) -> "TelemetrySnapshot":
        return cls(
            step=_to_int(sample.get("step")) or 0,
            speed_kmh=_to_float(sample.get("speed_x")),
            gear=_to_int(sample.get("gear")),
            throttle_pct=_scale_percent(sample.get("accel")),
            brake_pct=_scale_percent(sample.get("brake")),
            steer=_to_float(sample.get("steer")),
            track_pos=_to_float(sample.get("track_pos")),
            damage=_to_float(sample.get("damage")),
            reward=_to_float(sample.get("reward")),
            off_track=bool(sample.get("off_track", False)),
            front_sensor=_to_float(sample.get("front_sensor")),
            min_track_sensor=_to_float(sample.get("min_track_sensor")),
            cur_lap_time=_to_float(sample.get("cur_lap_time")),
            last_lap_time=_to_float(sample.get("last_lap_time")),
        )


@dataclass(frozen=True)
class ExplanationFrame:
    severity: str
    event_key: str
    headline: str
    reasons: tuple[str, ...]
    event: str


class TelemetryHistory:
    def __init__(self, max_points: int = 300) -> None:
        self._snapshots: deque[TelemetrySnapshot] = deque(maxlen=max_points)

    def append(self, snapshot: TelemetrySnapshot) -> None:
        self._snapshots.append(snapshot)

    def clear(self) -> None:
        self._snapshots.clear()

    def steps(self) -> list[int]:
        return [snapshot.step for snapshot in self._snapshots]

    def values(self, attribute: str) -> list[float]:
        values: list[float] = []
        for snapshot in self._snapshots:
            value = getattr(snapshot, attribute)
            values.append(float(value) if value is not None else math.nan)
        return values

    def snapshots(self) -> list[TelemetrySnapshot]:
        return list(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)


def explain_snapshot(snapshot: TelemetrySnapshot) -> str:
    frame = build_explanation_frame(snapshot)
    messages = [frame.headline, *frame.reasons]

    return "\n".join(messages)


def build_explanation_frame(snapshot: TelemetrySnapshot) -> ExplanationFrame:
    event_key = explanation_event_key(snapshot)
    severity = _severity_for_event(event_key)
    headline = _headline_for_event(snapshot, event_key)
    reasons = tuple(_reason_breakdown(snapshot))
    event = _event_message(snapshot, event_key)
    return ExplanationFrame(
        severity=severity,
        event_key=event_key,
        headline=headline,
        reasons=reasons,
        event=event,
    )


def summarize_run_explanation(
    snapshots: list[TelemetrySnapshot],
    results: Mapping[str, Any],
    run_id: int | None = None,
) -> str:
    termination = str(results.get("termination_reason", "finished"))
    laps = int(results.get("laps_completed", 0) or 0)
    best_lap = _to_float(results.get("best_lap_time_seconds"))
    saved_text = f" Saved as run #{run_id}." if run_id is not None else ""

    if best_lap is not None:
        headline = f"Completed lap in {best_lap:.3f}s.{saved_text}"
    else:
        headline = f"Race ended by {termination}. Laps completed: {laps}.{saved_text}"

    sample_count = len(snapshots)
    damage = max((snapshot.damage or 0.0 for snapshot in snapshots), default=0.0)
    at_risk_count = sum(
        1
        for snapshot in snapshots
        if build_explanation_frame(snapshot).severity in {"At Risk", "Stuck"}
    )
    caution_count = sum(
        1
        for snapshot in snapshots
        if build_explanation_frame(snapshot).severity == "Caution"
    )
    low_speed_count = sum(
        1
        for snapshot in snapshots
        if (snapshot.cur_lap_time or 0.0) >= 8.0
        and (snapshot.speed_kmh is not None and abs(snapshot.speed_kmh) < 4.0)
    )

    if termination in {"stuck", "launch_stuck"}:
        verdict = "Stuck condition detected, so the run needs driver recovery work."
    elif termination in {"crashed", "off_track"}:
        verdict = f"Run ended by {termination}; treat this as an unstable lap."
    elif damage > 0.0:
        verdict = f"Completed with damage reaching {damage:.0f}."
    elif at_risk_count == 0 and low_speed_count == 0:
        verdict = "Mostly stable: no damage, no stuck samples, and no at-risk samples."
    elif sample_count and at_risk_count / sample_count < 0.08:
        verdict = "Mostly stable with brief correction moments."
    else:
        verdict = "Completed, but there were several caution or at-risk moments."

    return (
        f"{headline}\n"
        f"{verdict}\n"
        f"Telemetry samples reviewed: {sample_count}; "
        f"caution samples: {caution_count}; at-risk samples: {at_risk_count}; "
        f"low-speed samples after launch: {low_speed_count}."
    )


def explanation_event_key(snapshot: TelemetrySnapshot) -> str:
    if snapshot.off_track:
        return "off_track"
    if snapshot.damage is not None and snapshot.damage > 0.0:
        return "damage"
    if _is_low_speed_after_launch(snapshot):
        return "stuck"
    if snapshot.front_sensor is not None and snapshot.front_sensor < 8.0:
        return "very_limited_road"
    if snapshot.track_pos is not None and abs(snapshot.track_pos) > 0.90:
        return "track_edge"
    if snapshot.brake_pct is not None and snapshot.brake_pct > 40.0:
        return "heavy_braking"
    if snapshot.brake_pct is not None and snapshot.brake_pct > 8.0:
        return "braking"
    if snapshot.front_sensor is not None and snapshot.front_sensor < 24.0:
        return "limited_road"
    if snapshot.track_pos is not None and abs(snapshot.track_pos) > 0.62:
        return "running_wide"
    if snapshot.steer is not None and abs(snapshot.steer) > 0.35:
        return "steering_correction"
    if snapshot.throttle_pct is not None and snapshot.throttle_pct > 8.0:
        return "accelerating"
    return "settled"


def format_explanation_entry(
    snapshot: TelemetrySnapshot,
    frame: ExplanationFrame | None = None,
) -> str:
    explanation = frame or build_explanation_frame(snapshot)
    return (
        f"{_timestamp(snapshot)} {explanation.severity}: {explanation.event}\n"
        f"{explanation.headline}\n"
        f"{_snapshot_headline(snapshot)}"
    )


def _reason_breakdown(snapshot: TelemetrySnapshot) -> list[str]:
    messages = []

    if snapshot.off_track:
        messages.append("The car is off track, so recovery is the priority.")
    elif snapshot.track_pos is not None:
        edge = abs(snapshot.track_pos)
        if edge > 0.90:
            messages.append("The car is right on the track edge and needs to come back in.")
        elif edge > 0.62:
            messages.append("The car is running wide, so the controller should be cautious.")
        elif edge < 0.18:
            messages.append("The car is close to the centre of the track.")

    if snapshot.front_sensor is not None:
        if snapshot.front_sensor < 8.0:
            messages.append("There is very little road ahead, so braking or recovery is expected.")
        elif snapshot.front_sensor < 24.0:
            messages.append("Forward road space is limited, so speed should be managed.")

    if snapshot.brake_pct is not None and snapshot.brake_pct > 8.0:
        messages.append(f"Braking at {snapshot.brake_pct:.0f}%.")
    elif snapshot.throttle_pct is not None and snapshot.throttle_pct > 8.0:
        messages.append(f"Applying throttle at {snapshot.throttle_pct:.0f}%.")

    if snapshot.steer is not None:
        if snapshot.steer > 0.08:
            messages.append("Steering right.")
        elif snapshot.steer < -0.08:
            messages.append("Steering left.")
        else:
            messages.append("Steering is almost centred.")

    if snapshot.damage is not None and snapshot.damage > 0.0:
        messages.append(f"Damage is {snapshot.damage:.0f}.")

    if not messages:
        messages.append("Telemetry is steady; no caution reason is active.")

    return messages


def format_value(value: float | int | None, precision: str = ".2f") -> str:
    if value is None:
        return "--"
    return format(value, precision)


def _severity_for_event(event_key: str) -> str:
    if event_key in {"off_track", "damage", "stuck", "very_limited_road", "track_edge"}:
        return "At Risk" if event_key != "stuck" else "Stuck"
    if event_key in {"heavy_braking", "limited_road", "running_wide"}:
        return "Caution"
    if event_key in {"braking", "steering_correction"}:
        return "Correcting"
    return "Normal"


def _headline_for_event(snapshot: TelemetrySnapshot, event_key: str) -> str:
    direction = _steer_direction(snapshot.steer)
    if event_key == "off_track":
        return "The car is off track and prioritising recovery."
    if event_key == "damage":
        return "Damage has been detected; the run is no longer clean."
    if event_key == "stuck":
        return "The car may be stuck: speed is near zero after launch."
    if event_key == "very_limited_road":
        return "Very little road is visible ahead, so the car is managing risk."
    if event_key == "track_edge":
        return "The car is close to the track edge and needs to move back in."
    if event_key == "heavy_braking":
        return f"Heavy braking while steering {direction}."
    if event_key == "braking":
        return f"Braking while steering {direction}."
    if event_key == "limited_road":
        return "Forward road space is limited, so speed is being managed."
    if event_key == "running_wide":
        return "The car is running wide but still inside the track limits."
    if event_key == "steering_correction":
        return f"Correcting heading by steering {direction}."
    if event_key == "accelerating":
        return f"Accelerating while steering {direction}."
    return "Holding a stable line."


def _event_message(snapshot: TelemetrySnapshot, event_key: str) -> str:
    if event_key == "off_track":
        return "Off-track recovery active"
    if event_key == "damage":
        return f"Damage detected at {format_value(snapshot.damage, '.0f')}"
    if event_key == "stuck":
        return "Low-speed stuck risk"
    if event_key == "very_limited_road":
        return f"Very limited road ahead ({format_value(snapshot.front_sensor, '.1f')}m)"
    if event_key == "track_edge":
        return f"Near track edge (trackPos {_format_signed(snapshot.track_pos)})"
    if event_key == "heavy_braking":
        return f"Heavy braking at {format_value(snapshot.brake_pct, '.0f')}%"
    if event_key == "braking":
        return f"Braking at {format_value(snapshot.brake_pct, '.0f')}%"
    if event_key == "limited_road":
        return f"Limited road ahead ({format_value(snapshot.front_sensor, '.1f')}m)"
    if event_key == "running_wide":
        return f"Running wide (trackPos {_format_signed(snapshot.track_pos)})"
    if event_key == "steering_correction":
        return f"Steering correction {_format_signed(snapshot.steer)}"
    if event_key == "accelerating":
        return f"Accelerating at {format_value(snapshot.throttle_pct, '.0f')}%"
    return "Stable line"


def _snapshot_headline(snapshot: TelemetrySnapshot) -> str:
    parts = [
        f"{format_value(snapshot.speed_kmh, '.1f')} km/h",
        f"gear {_format_optional(snapshot.gear)}",
        f"trackPos {_format_signed(snapshot.track_pos)}",
    ]
    if snapshot.steer is not None:
        parts.append(f"steer {_format_signed(snapshot.steer)}")
    return ", ".join(parts) + "."


def _timestamp(snapshot: TelemetrySnapshot) -> str:
    if snapshot.cur_lap_time is not None:
        return f"[lap {snapshot.cur_lap_time:.2f}s | step {snapshot.step}]"
    return f"[step {snapshot.step}]"


def _is_low_speed_after_launch(snapshot: TelemetrySnapshot) -> bool:
    return (
        snapshot.cur_lap_time is not None
        and snapshot.cur_lap_time >= 8.0
        and snapshot.speed_kmh is not None
        and abs(snapshot.speed_kmh) < 4.0
    )


def _steer_direction(steer: float | None) -> str:
    if steer is None or abs(steer) < 0.08:
        return "straight"
    return "right" if steer > 0 else "left"


def _completion_severity(termination: str) -> str:
    if termination in {"stuck", "launch_stuck"}:
        return "Stuck"
    if termination in {"crashed", "off_track", "torcs_closed"}:
        return "At Risk"
    if termination == "stopped":
        return "Caution"
    return "Normal"


def _format_optional(value: object) -> str:
    return "--" if value is None else str(value)


def _format_signed(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.3f}"


def _scale_percent(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return max(0.0, min(100.0, numeric * 100.0))


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    return None if numeric is None else int(numeric)
