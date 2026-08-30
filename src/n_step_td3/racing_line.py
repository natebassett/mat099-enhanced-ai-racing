from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOOKAHEAD_DISTANCES_M = np.arange(0.0, 300.0, 25.0, dtype=np.float32)
CURVATURE_SCALE = 0.1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACING_LINE_DIR = PROJECT_ROOT / "data" / "racing_lines"


def default_racing_line_path(track_name: str) -> Path:
    """Resolve the shared Agent 7 line, preferring its smoothed g-track-3 map."""
    standard = DEFAULT_RACING_LINE_DIR / f"{track_name}.json"
    if track_name != "g-track-3":
        return standard
    smooth = DEFAULT_RACING_LINE_DIR / "g-track-3-agent7-smooth-v1.json"
    return smooth if smooth.is_file() else standard


@dataclass(frozen=True)
class RacingLineSample:
    target_track_position: float
    lookahead_curvature: np.ndarray


class RacingLineFeatureMap:
    """Interpolate the local racing-line JSON into torcsRL-style features."""

    def __init__(
        self,
        distances: np.ndarray,
        target_track_positions: np.ndarray,
        curvatures: np.ndarray,
        track_length_m: float,
    ) -> None:
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        target_track_positions = np.asarray(
            target_track_positions,
            dtype=np.float64,
        ).reshape(-1)
        curvatures = np.asarray(curvatures, dtype=np.float64).reshape(-1)
        if distances.size < 2:
            raise ValueError("racing line requires at least two waypoints")
        if not (
            distances.shape
            == target_track_positions.shape
            == curvatures.shape
        ):
            raise ValueError("racing-line waypoint arrays must have equal sizes")
        if not np.all(np.isfinite(distances)) or np.any(np.diff(distances) < 0):
            raise ValueError("racing-line distances must be finite and ordered")
        if not np.all(np.isfinite(target_track_positions)):
            raise ValueError("racing-line target positions must be finite")
        if not np.all(np.isfinite(curvatures)):
            raise ValueError("racing-line curvatures must be finite")
        if not np.isfinite(track_length_m) or track_length_m <= 0.0:
            raise ValueError("track length must be positive")
        self.distances = distances
        self.target_track_positions = target_track_positions
        self.curvatures = curvatures
        self.track_length_m = float(track_length_m)

    @classmethod
    def from_json(cls, path: str | Path) -> "RacingLineFeatureMap":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        waypoints = payload.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError(f"racing line has no waypoints: {source}")
        return cls(
            distances=np.asarray(
                [waypoint["distance"] for waypoint in waypoints],
                dtype=np.float64,
            ),
            target_track_positions=np.asarray(
                [waypoint["target_track_pos"] for waypoint in waypoints],
                dtype=np.float64,
            ),
            curvatures=np.asarray(
                [waypoint["curvature"] for waypoint in waypoints],
                dtype=np.float64,
            ),
            track_length_m=float(payload["track_length_m"]),
        )

    def sample(self, distance_from_start_m: float) -> RacingLineSample:
        distance = float(distance_from_start_m) % self.track_length_m
        target = float(
            np.interp(
                distance,
                self.distances,
                self.target_track_positions,
                period=self.track_length_m,
            )
        )
        lookahead_distances = (
            distance + LOOKAHEAD_DISTANCES_M
        ) % self.track_length_m
        curvature = np.interp(
            lookahead_distances,
            self.distances,
            self.curvatures,
            period=self.track_length_m,
        )
        normalised_curvature = np.clip(
            curvature / CURVATURE_SCALE,
            -1.0,
            1.0,
        ).astype(np.float32)
        return RacingLineSample(
            target_track_position=float(np.clip(target, -1.0, 1.0)),
            lookahead_curvature=normalised_curvature,
        )
