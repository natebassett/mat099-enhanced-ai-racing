from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from racing_line.track_map import TorcsTrackMap


Coordinate = tuple[float, float]


@dataclass(frozen=True)
class RacingLinePoint:
    distance: float
    target_track_pos: float
    target_speed_kmh: float
    curvature: float
    phase: str
    reference_section: str | None = None


@dataclass(frozen=True)
class RacingLineProfile:
    path: Path
    track_name: str
    track_length_m: float
    track_width_m: float
    optimizer_message: str
    source_sha256: str
    points: tuple[RacingLinePoint, ...]

    @property
    def min_speed_kmh(self) -> float:
        return min((point.target_speed_kmh for point in self.points), default=0.0)

    @property
    def max_speed_kmh(self) -> float:
        return max((point.target_speed_kmh for point in self.points), default=0.0)

    @property
    def min_track_pos(self) -> float:
        return min((point.target_track_pos for point in self.points), default=0.0)

    @property
    def max_track_pos(self) -> float:
        return max((point.target_track_pos for point in self.points), default=0.0)


@dataclass(frozen=True)
class TrackMapPoint:
    distance: float
    x: float
    y: float
    normal_x: float
    normal_y: float


@dataclass(frozen=True)
class RacingLineVisualPoint:
    x: float
    y: float
    source: RacingLinePoint


@dataclass(frozen=True)
class RacingLineVisualProfile:
    racing_line: RacingLineProfile
    centerline: tuple[Coordinate, ...]
    left_edge: tuple[Coordinate, ...]
    right_edge: tuple[Coordinate, ...]
    line_points: tuple[RacingLineVisualPoint, ...]
    bounds: tuple[float, float, float, float]


def load_racing_line_profile(path: Path) -> RacingLineProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError(f"Racing line has no waypoints: {path}")

    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        optimizer = {}

    points = tuple(
        _point_from_waypoint(waypoint)
        for waypoint in waypoints
        if isinstance(waypoint, Mapping)
    )
    if not points:
        raise ValueError(f"Racing line has no readable waypoints: {path}")

    return RacingLineProfile(
        path=Path(path),
        track_name=str(payload.get("track_name") or Path(path).stem),
        track_length_m=_to_float(payload.get("track_length_m")),
        track_width_m=_to_float(payload.get("track_width_m")),
        optimizer_message=str(optimizer.get("message") or "racing-line profile"),
        source_sha256=str(payload.get("source_sha256") or ""),
        points=points,
    )


def build_racing_line_visual_profile(
    profile: RacingLineProfile,
    track_xml_path: Path,
) -> RacingLineVisualProfile:
    track_map = TorcsTrackMap.from_xml(track_xml_path)
    center_samples = tuple(
        _track_map_point(sample)
        for sample in track_map.sample_centerline(
            spacing=3.0,
            measured_length=profile.track_length_m,
        )
    )
    if not center_samples:
        raise ValueError(f"Track geometry has no samples: {track_xml_path}")

    track_width = profile.track_width_m or track_map.width
    half_width = track_width / 2.0
    centerline = tuple((point.x, point.y) for point in center_samples)
    left_edge = tuple(
        (
            point.x + point.normal_x * half_width,
            point.y + point.normal_y * half_width,
        )
        for point in center_samples
    )
    right_edge = tuple(
        (
            point.x - point.normal_x * half_width,
            point.y - point.normal_y * half_width,
        )
        for point in center_samples
    )
    line_points = tuple(
        _visual_point(profile_point, center_samples, profile.track_length_m, half_width)
        for profile_point in profile.points
    )

    return RacingLineVisualProfile(
        racing_line=profile,
        centerline=centerline,
        left_edge=left_edge,
        right_edge=right_edge,
        line_points=line_points,
        bounds=_bounds((*left_edge, *right_edge, *((point.x, point.y) for point in line_points))),
    )


def racing_line_summary_lines(profile: RacingLineProfile) -> tuple[str, ...]:
    return (
        f"Track: {profile.track_name}.",
        f"Lap length: {profile.track_length_m:.1f}m; width: {profile.track_width_m:.1f}m.",
        (
            f"Waypoints: {len(profile.points)}; target speed range: "
            f"{profile.min_speed_kmh:.0f}-{profile.max_speed_kmh:.0f} km/h."
        ),
        (
            f"Road-position range: {profile.min_track_pos:.2f} to "
            f"{profile.max_track_pos:.2f}."
        ),
        f"Dominant phases: {_dominant_phase_text(profile)}.",
        f"Optimizer note: {profile.optimizer_message}.",
    )


def racing_line_point_text(point: RacingLinePoint | None) -> str:
    if point is None:
        return "No racing-line point selected."

    lines = [
        f"Distance: {point.distance:.1f}m",
        f"Target road position: {point.target_track_pos:.2f} ({track_position_label(point.target_track_pos)})",
        f"Target speed: {point.target_speed_kmh:.1f} km/h",
        f"Phase: {phase_label(point.phase)}",
    ]
    if point.reference_section:
        lines.append(f"Reference section: {point.reference_section}")
    return "\n".join(lines)


def point_at_distance(
    profile: RacingLineProfile | None,
    distance: float,
) -> RacingLinePoint | None:
    if profile is None or not profile.points:
        return None

    if distance >= profile.track_length_m:
        return profile.points[-1]

    wrapped_distance = distance % max(profile.track_length_m, 1.0)
    best_point = profile.points[0]
    best_delta = abs(best_point.distance - wrapped_distance)
    for point in profile.points[1:]:
        delta = abs(point.distance - wrapped_distance)
        if delta < best_delta:
            best_point = point
            best_delta = delta
    return best_point


def visual_point_at_distance(
    profile: RacingLineVisualProfile | None,
    distance: float,
) -> RacingLineVisualPoint | None:
    if profile is None or not profile.line_points:
        return None
    if distance >= profile.racing_line.track_length_m:
        return profile.line_points[-1]

    wrapped_distance = distance % max(profile.racing_line.track_length_m, 1.0)
    best_point = profile.line_points[0]
    best_delta = abs(best_point.source.distance - wrapped_distance)
    for point in profile.line_points[1:]:
        delta = abs(point.source.distance - wrapped_distance)
        if delta < best_delta:
            best_point = point
            best_delta = delta
    return best_point


def phase_family(phase: str) -> str:
    normalized = phase.replace("_", " ").casefold()
    if "brake" in normalized:
        return "Brake"
    if "full throttle" in normalized:
        return "Full throttle"
    if "accelerate" in normalized:
        return "Accelerate"
    if "turn" in normalized:
        return "Turn"
    return "Settle"


def phase_label(phase: str) -> str:
    return phase.replace("_", " ").strip().title() or "Settle"


def track_position_label(track_position: float) -> str:
    side = "right" if track_position > 0 else "left"
    edge = abs(track_position)
    if edge > 0.90:
        return f"near {side} edge"
    if edge > 0.62:
        return f"wide {side}"
    if edge < 0.18:
        return "near centre"
    return f"slightly {side}"


def _point_from_waypoint(waypoint: Mapping[str, Any]) -> RacingLinePoint:
    return RacingLinePoint(
        distance=_to_float(waypoint.get("distance")),
        target_track_pos=_to_float(waypoint.get("target_track_pos")),
        target_speed_kmh=_to_float(waypoint.get("target_speed_kmh")),
        curvature=_to_float(waypoint.get("curvature")),
        phase=str(waypoint.get("phase") or "settle"),
        reference_section=(
            str(waypoint["reference_section"])
            if waypoint.get("reference_section")
            else None
        ),
    )


def _track_map_point(sample: Mapping[str, Any]) -> TrackMapPoint:
    return TrackMapPoint(
        distance=_to_float(sample.get("distance")),
        x=_to_float(sample.get("x")),
        y=_to_float(sample.get("y")),
        normal_x=_to_float(sample.get("normal_x")),
        normal_y=_to_float(sample.get("normal_y")),
    )


def _visual_point(
    point: RacingLinePoint,
    center_samples: tuple[TrackMapPoint, ...],
    track_length: float,
    half_width: float,
) -> RacingLineVisualPoint:
    center = _centerline_at_distance(center_samples, point.distance, track_length)
    offset = point.target_track_pos * half_width
    return RacingLineVisualPoint(
        x=center.x + center.normal_x * offset,
        y=center.y + center.normal_y * offset,
        source=point,
    )


def _centerline_at_distance(
    center_samples: tuple[TrackMapPoint, ...],
    distance: float,
    track_length: float,
) -> TrackMapPoint:
    if distance >= track_length:
        return center_samples[-1]

    distances = [point.distance for point in center_samples]
    wrapped_distance = distance % max(track_length, 1.0)
    right_index = bisect_right(distances, wrapped_distance)
    if right_index <= 0:
        left = center_samples[-1]
        right = center_samples[0]
        left_distance = left.distance - track_length
        right_distance = right.distance
    elif right_index >= len(center_samples):
        left = center_samples[-1]
        right = center_samples[0]
        left_distance = left.distance
        right_distance = right.distance + track_length
    else:
        left = center_samples[right_index - 1]
        right = center_samples[right_index]
        left_distance = left.distance
        right_distance = right.distance

    span = max(0.001, right_distance - left_distance)
    fraction = (wrapped_distance - left_distance) / span
    normal_x = _interpolate(left.normal_x, right.normal_x, fraction)
    normal_y = _interpolate(left.normal_y, right.normal_y, fraction)
    normal_length = (normal_x * normal_x + normal_y * normal_y) ** 0.5 or 1.0
    return TrackMapPoint(
        distance=wrapped_distance,
        x=_interpolate(left.x, right.x, fraction),
        y=_interpolate(left.y, right.y, fraction),
        normal_x=normal_x / normal_length,
        normal_y=normal_y / normal_length,
    )


def _interpolate(left: float, right: float, fraction: float) -> float:
    return left + (right - left) * max(0.0, min(1.0, fraction))


def _bounds(points: tuple[Coordinate, ...]) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 1.0, 1.0)
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return (min(x_values), min(y_values), max(x_values), max(y_values))


def _dominant_phase_text(profile: RacingLineProfile) -> str:
    counts = Counter(phase_family(point.phase) for point in profile.points)
    if not counts:
        return "none"
    return ", ".join(
        f"{phase.lower()} ({count})"
        for phase, count in counts.most_common(3)
    )


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
