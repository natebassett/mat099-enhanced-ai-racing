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

    def __len__(self) -> int:
        return len(self._snapshots)


def explain_snapshot(snapshot: TelemetrySnapshot) -> str:
    messages = [_snapshot_headline(snapshot)]

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

    return "\n".join(messages)


def explanation_event_key(snapshot: TelemetrySnapshot) -> str:
    if snapshot.off_track:
        return "off_track"
    if snapshot.damage is not None and snapshot.damage > 0.0:
        return "damage"
    if snapshot.front_sensor is not None and snapshot.front_sensor < 8.0:
        return "very_limited_road"
    if snapshot.track_pos is not None and abs(snapshot.track_pos) > 0.90:
        return "track_edge"
    if snapshot.brake_pct is not None and snapshot.brake_pct > 8.0:
        return "braking"
    if snapshot.track_pos is not None and abs(snapshot.track_pos) > 0.62:
        return "running_wide"
    if snapshot.throttle_pct is not None and snapshot.throttle_pct > 8.0:
        return "accelerating"
    return "settled"


def format_explanation_entry(snapshot: TelemetrySnapshot) -> str:
    return f"{_timestamp(snapshot)}\n{explain_snapshot(snapshot)}"


def format_value(value: float | int | None, precision: str = ".2f") -> str:
    if value is None:
        return "--"
    return format(value, precision)


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
