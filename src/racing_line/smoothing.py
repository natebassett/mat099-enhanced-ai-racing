from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .optimizer import (
    RacingLine,
    RacingLineOptimizer,
    _closed_path_curvature,
    _heading_offsets,
    _phase,
    _speed_deltas,
    _turn_directions,
)
from .track_map import TorcsTrackMap


def build_smoothed_racing_line(
    base_line_path: str | Path,
    track_xml_path: str | Path,
    *,
    regularization: float = 30.0,
    curvature_window: int = 21,
) -> RacingLine:
    """Create a smoother, bounded derivative of an existing racing line."""

    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    if curvature_window < 3 or curvature_window % 2 == 0:
        raise ValueError("curvature_window must be an odd integer of at least 3")

    base_path = Path(base_line_path)
    track_path = Path(track_xml_path)
    base_line = RacingLine.load(base_path)
    track_map = TorcsTrackMap.from_xml(track_path)
    metadata = base_line.payload.get("optimizer", {})
    spacing = float(metadata.get("spacing_m", 3.0))
    safety_margin = float(metadata.get("safety_margin_m", 1.15))
    samples = track_map.sample_centerline(spacing, base_line.track_length)
    if len(samples) != len(base_line.waypoints):
        raise ValueError(
            "base racing line does not align with the sampled track geometry: "
            f"{len(base_line.waypoints)} waypoints != {len(samples)} samples"
        )

    sample_distances = np.asarray(
        [sample["distance"] for sample in samples],
        dtype=np.float64,
    )
    base_distances = np.asarray(
        [waypoint["distance"] for waypoint in base_line.waypoints],
        dtype=np.float64,
    )
    if not np.allclose(sample_distances, base_distances, atol=1e-3, rtol=0.0):
        raise ValueError("base racing-line distances do not match the track map")

    base_positions = np.asarray(
        [waypoint["target_track_pos"] for waypoint in base_line.waypoints],
        dtype=np.float64,
    )
    target_positions = periodic_second_difference_smooth(
        base_positions,
        regularization,
    )
    half_width = track_map.width / 2.0
    maximum_track_position = max(
        0.0,
        (half_width - safety_margin) / max(half_width, 1e-9),
    )
    target_positions = np.clip(
        target_positions,
        -maximum_track_position,
        maximum_track_position,
    )

    centre = np.asarray(
        [(sample["x"], sample["y"]) for sample in samples],
        dtype=np.float64,
    )
    normals = np.asarray(
        [(sample["normal_x"], sample["normal_y"]) for sample in samples],
        dtype=np.float64,
    )
    offsets_m = target_positions * half_width
    path = centre + normals * offsets_m[:, None]
    _raw_curvature, path_lengths = _closed_path_curvature(path)
    curvature_path = circular_moving_average(path, curvature_window)
    curvatures, _curvature_lengths = _closed_path_curvature(curvature_path)

    optimizer = RacingLineOptimizer(
        track_map,
        measured_length=base_line.track_length,
        spacing=spacing,
        safety_margin=safety_margin,
        maximum_speed_kmh=max(
            float(waypoint.get("target_speed_kmh", 216.0))
            for waypoint in base_line.waypoints
        ),
        minimum_speed_kmh=float(metadata.get("minimum_speed_kmh", 82.0)),
        lateral_acceleration=float(
            metadata.get("lateral_acceleration_mps2", 12.5)
        ),
        acceleration=float(metadata.get("acceleration_mps2", 8.5)),
        braking=float(metadata.get("braking_mps2", 10.0)),
    )
    speeds = optimizer._speed_profile(curvatures, path_lengths)
    heading_offsets = _heading_offsets(offsets_m, samples)
    speed_deltas = _speed_deltas(speeds, spacing, window_m=30.0)
    turn_directions = _turn_directions(curvatures, spacing, window_m=30.0)

    waypoints = []
    for index, sample in enumerate(samples):
        speed_kmh = float(speeds[index] * 3.6)
        speed_delta_kmh = float(speed_deltas[index] * 3.6)
        turn_direction = float(turn_directions[index])
        curvature = float(curvatures[index])
        waypoints.append(
            {
                "distance": round(float(sample["distance"]), 4),
                "target_track_pos": round(float(target_positions[index]), 6),
                "target_speed_kmh": round(speed_kmh, 3),
                "curvature": round(curvature, 7),
                "heading_offset": round(float(heading_offsets[index]), 6),
                "speed_delta_30m": round(speed_delta_kmh, 3),
                "turn_direction": round(turn_direction, 6),
                "phase": _phase(
                    speed_kmh,
                    speed_delta_kmh,
                    turn_direction,
                    curvature,
                    optimizer.maximum_speed_kmh,
                ),
            }
        )

    before_metrics = racing_line_quality_metrics(base_positions, np.asarray(
        [waypoint["curvature"] for waypoint in base_line.waypoints],
        dtype=np.float64,
    ))
    after_metrics = racing_line_quality_metrics(target_positions, curvatures)
    after_metrics.update(
        {
            "maximum_target_deviation": float(
                np.max(np.abs(target_positions - base_positions))
            ),
            "rms_target_deviation": float(
                np.sqrt(np.mean((target_positions - base_positions) ** 2))
            ),
            "path_length_m": float(np.sum(path_lengths)),
        }
    )

    payload = {
        "format_version": 1,
        "track_name": base_line.track_name,
        "track_length_m": round(float(base_line.track_length), 4),
        "track_width_m": track_map.width,
        "source_sha256": hashlib.sha256(track_path.read_bytes()).hexdigest(),
        "optimizer": {
            "success": True,
            "message": "periodic regularized Agent 7 racing line",
            "objective": round(float(after_metrics["curvature_rms"]), 9),
            "spacing_m": spacing,
            "safety_margin_m": safety_margin,
            "lateral_acceleration_mps2": optimizer.lateral_acceleration,
            "minimum_speed_kmh": optimizer.minimum_speed_kmh,
            "acceleration_mps2": optimizer.acceleration,
            "braking_mps2": optimizer.braking,
            "base_line": base_path.name,
            "base_line_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "smoothing_method": "cyclic_second_difference_tikhonov",
            "smoothing_regularization": regularization,
            "curvature_geometry_filter": "periodic_moving_average",
            "curvature_window_waypoints": curvature_window,
            "quality_before": before_metrics,
            "quality_after": after_metrics,
        },
        "waypoints": waypoints,
    }
    return RacingLine(payload)


def periodic_second_difference_smooth(
    values: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Solve a cyclic second-difference Tikhonov smoothing problem by FFT."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("values must contain at least three finite samples")
    spectrum = np.fft.rfft(values)
    angular_frequency = 2.0 * np.pi * np.fft.rfftfreq(values.size)
    eigenvalues = (2.0 - 2.0 * np.cos(angular_frequency)) ** 2
    smoothed = np.fft.irfft(
        spectrum / (1.0 + regularization * eigenvalues),
        n=values.size,
    )
    return smoothed.astype(np.float64)


def circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in {1, 2}:
        raise ValueError("values must be a vector or matrix")
    if window < 1 or window % 2 == 0 or window > len(values):
        raise ValueError("window must be odd and no larger than the sample count")
    pad = window // 2
    padded = np.concatenate((values[-pad:], values, values[:pad]), axis=0)
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    if values.ndim == 1:
        return np.convolve(padded, kernel, mode="valid")
    return np.column_stack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])]
    )


def racing_line_quality_metrics(
    target_positions: np.ndarray,
    curvatures: np.ndarray,
) -> dict[str, float]:
    target_positions = np.asarray(target_positions, dtype=np.float64).reshape(-1)
    curvatures = np.asarray(curvatures, dtype=np.float64).reshape(-1)
    if target_positions.shape != curvatures.shape:
        raise ValueError("target-position and curvature arrays must align")
    target_steps = np.roll(target_positions, -1) - target_positions
    target_second_difference = (
        np.roll(target_positions, -1)
        - 2.0 * target_positions
        + np.roll(target_positions, 1)
    )
    curvature_steps = np.roll(curvatures, -1) - curvatures
    return {
        "maximum_abs_target_track_pos": float(np.max(np.abs(target_positions))),
        "maximum_target_step": float(np.max(np.abs(target_steps))),
        "target_step_rms": float(np.sqrt(np.mean(target_steps**2))),
        "target_second_difference_rms": float(
            np.sqrt(np.mean(target_second_difference**2))
        ),
        "maximum_abs_curvature": float(np.max(np.abs(curvatures))),
        "curvature_rms": float(np.sqrt(np.mean(curvatures**2))),
        "curvature_step_rms": float(np.sqrt(np.mean(curvature_steps**2))),
        "maximum_curvature_step": float(np.max(np.abs(curvature_steps))),
    }
