import bisect
import hashlib
import json
import math
from pathlib import Path


class RacingLine:
    def __init__(self, payload):
        self.payload = payload
        self.track_name = payload["track_name"]
        self.track_length = float(payload["track_length_m"])
        self.waypoints = payload["waypoints"]
        self._distances = [point["distance"] for point in self.waypoints]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def lookup(self, distance):
        distance = float(distance) % self.track_length
        right_index = bisect.bisect_right(self._distances, distance) % len(
            self.waypoints
        )
        left_index = (right_index - 1) % len(self.waypoints)
        left = self.waypoints[left_index]
        right = self.waypoints[right_index]
        span = (right["distance"] - left["distance"]) % self.track_length
        travelled = (distance - left["distance"]) % self.track_length
        fraction = travelled / span if span else 0.0

        return {
            key: _interpolate(left[key], right[key], fraction)
            for key in (
                "target_track_pos",
                "target_speed_kmh",
                "curvature",
                "heading_offset",
            )
        } | {"phase": left["phase"]}


class RacingLineOptimizer:
    """Bounded minimum-curvature line plus friction-limited speed profile."""

    def __init__(
        self,
        track_map,
        *,
        measured_length=None,
        spacing=3.0,
        safety_margin=1.5,
        maximum_speed_kmh=216.0,
        minimum_speed_kmh=72.0,
        lateral_acceleration=11.5,
        acceleration=6.5,
        braking=10.0,
    ):
        self.track_map = track_map
        self.measured_length = measured_length
        self.spacing = spacing
        self.safety_margin = safety_margin
        self.maximum_speed_kmh = maximum_speed_kmh
        self.minimum_speed_kmh = minimum_speed_kmh
        self.lateral_acceleration = lateral_acceleration
        self.acceleration = acceleration
        self.braking = braking

    def optimize(self):
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        from scipy.optimize import minimize

        samples = self.track_map.sample_centerline(
            self.spacing,
            self.measured_length,
        )
        centre = np.array([(point["x"], point["y"]) for point in samples])
        normals = np.array(
            [(point["normal_x"], point["normal_y"]) for point in samples]
        )
        sample_distances = np.array([point["distance"] for point in samples])
        control_indices = []
        for segment_index, segment in enumerate(self.track_map.segments):
            sample_indices = [
                index
                for index, point in enumerate(samples)
                if point["segment_index"] == segment_index
            ]
            control_indices.append(sample_indices[0])
            if segment.turn_radians and len(sample_indices) > 2:
                control_indices.append(sample_indices[len(sample_indices) // 2])
        control_indices = sorted(set(control_indices))
        control_distances = sample_distances[control_indices]
        control_count = len(control_indices)
        lookup_length = self.measured_length or self.track_map.geometry_length
        usable_half_width = self.track_map.width / 2.0 - self.safety_margin

        def expanded_offsets(controls):
            periodic_distances = np.concatenate(
                (
                    [control_distances[-1] - lookup_length],
                    control_distances,
                    [control_distances[0] + lookup_length],
                    [control_distances[1] + lookup_length],
                )
            )
            periodic_controls = np.concatenate(
                ([controls[-1]], controls, [controls[0]], [controls[1]])
            )
            spline = PchipInterpolator(
                periodic_distances,
                periodic_controls,
            )
            return spline(sample_distances)

        initial_values = []
        for sample_index in control_indices:
            sample = samples[sample_index]
            segment_index = sample["segment_index"]
            segment = self.track_map.segments[segment_index]
            turn_sign = math.copysign(1.0, segment.turn_radians) if segment.turn_radians else 0.0
            if segment.turn_radians and sample["segment_fraction"] > 0.25:
                # Inside at the apex.
                value = turn_sign * usable_half_width * 0.82
            elif segment.turn_radians:
                # Outside on corner entry.
                value = -turn_sign * usable_half_width * 0.90
            else:
                previous_segment = self.track_map.segments[segment_index - 1]
                if previous_segment.turn_radians:
                    previous_sign = math.copysign(1.0, previous_segment.turn_radians)
                    value = -previous_sign * usable_half_width * 0.90
                else:
                    value = 0.0
            initial_values.append(value)
        initial = np.array(initial_values)

        def objective(controls):
            offsets = expanded_offsets(controls)
            path = centre + normals * offsets[:, None]
            vectors = np.roll(path, -1, axis=0) - path
            lengths = np.linalg.norm(vectors, axis=1)
            units = vectors / np.maximum(lengths[:, None], 1e-8)
            previous_units = np.roll(units, 1, axis=0)
            turns = np.arctan2(
                previous_units[:, 0] * units[:, 1]
                - previous_units[:, 1] * units[:, 0],
                np.sum(previous_units * units, axis=1),
            )
            curvature_cost = np.sum(
                turns**2 / np.maximum((lengths + np.roll(lengths, 1)) / 2.0, 0.1)
            )
            control_second_difference = (
                np.roll(controls, -1) - 2.0 * controls + np.roll(controls, 1)
            )
            smoothness_cost = 0.0008 * np.sum(control_second_difference**2)
            length_cost = 0.00004 * np.sum(lengths)
            # Keep the numerical solution recognisably on a racing line. Pure
            # curvature minimisation tends to settle near the centre on linked
            # bends, even though that sacrifices the outside-apex-outside line
            # and the exit speed it enables.
            classical_line_cost = 0.02 * np.sum((controls - initial) ** 2)
            return curvature_cost + smoothness_cost + length_cost + classical_line_cost
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-usable_half_width, usable_half_width)] * control_count,
            options={
                "maxiter": 400,
                "maxfun": 60000,
                "ftol": 1e-10,
                "maxls": 40,
            },
        )
        offsets = expanded_offsets(result.x)
        path = centre + normals * offsets[:, None]
        curvatures, path_lengths = _closed_path_curvature(path)
        speeds = self._speed_profile(curvatures, path_lengths)
        heading_offsets = _heading_offsets(offsets, samples)

        waypoints = []
        for index, sample in enumerate(samples):
            speed_kmh = float(speeds[index] * 3.6)
            waypoints.append(
                {
                    "distance": round(float(sample["distance"]), 4),
                    "target_track_pos": round(
                        float(offsets[index] / (self.track_map.width / 2.0)),
                        6,
                    ),
                    "target_speed_kmh": round(speed_kmh, 3),
                    "curvature": round(float(curvatures[index]), 7),
                    "heading_offset": round(float(heading_offsets[index]), 6),
                    "phase": _phase(speed_kmh, self.maximum_speed_kmh),
                }
            )

        source_hash = hashlib.sha256(
            self.track_map.source_path.read_bytes()
        ).hexdigest()
        return RacingLine(
            {
                "format_version": 1,
                "track_name": self.track_map.name,
                "track_length_m": round(
                    float(self.measured_length or self.track_map.geometry_length),
                    4,
                ),
                "track_width_m": self.track_map.width,
                "source_sha256": source_hash,
                "optimizer": {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "objective": float(result.fun),
                    "spacing_m": self.spacing,
                    "safety_margin_m": self.safety_margin,
                    "lateral_acceleration_mps2": self.lateral_acceleration,
                    "minimum_speed_kmh": self.minimum_speed_kmh,
                    "acceleration_mps2": self.acceleration,
                    "braking_mps2": self.braking,
                },
                "waypoints": waypoints,
            }
        )

    def _speed_profile(self, curvatures, path_lengths):
        import numpy as np

        maximum_speed = self.maximum_speed_kmh / 3.6
        curvature_magnitude = np.maximum(np.abs(curvatures), 1e-6)
        speeds = np.minimum(
            maximum_speed,
            np.sqrt(self.lateral_acceleration / curvature_magnitude),
        )
        speeds = np.maximum(speeds, self.minimum_speed_kmh / 3.6)

        for _ in range(8):
            for index in range(len(speeds)):
                following = (index + 1) % len(speeds)
                speeds[following] = min(
                    speeds[following],
                    math.sqrt(
                        speeds[index] ** 2
                        + 2.0 * self.acceleration * path_lengths[index]
                    ),
                )
            for index in range(len(speeds) - 1, -1, -1):
                following = (index + 1) % len(speeds)
                speeds[index] = min(
                    speeds[index],
                    math.sqrt(
                        speeds[following] ** 2
                        + 2.0 * self.braking * path_lengths[index]
                    ),
                )
        return speeds


def _closed_path_curvature(path):
    import numpy as np

    vectors = np.roll(path, -1, axis=0) - path
    lengths = np.linalg.norm(vectors, axis=1)
    units = vectors / np.maximum(lengths[:, None], 1e-8)
    previous_units = np.roll(units, 1, axis=0)
    turns = np.arctan2(
        previous_units[:, 0] * units[:, 1] - previous_units[:, 1] * units[:, 0],
        np.sum(previous_units * units, axis=1),
    )
    local_lengths = np.maximum((lengths + np.roll(lengths, 1)) / 2.0, 0.1)
    return turns / local_lengths, lengths


def _heading_offsets(offsets, samples):
    offsets_before = [offsets[index - 1] for index in range(len(offsets))]
    offsets_after = [offsets[(index + 1) % len(offsets)] for index in range(len(offsets))]
    headings = []
    track_length = samples[-1]["distance"] + (
        samples[-1]["distance"] - samples[-2]["distance"]
    )
    for index, sample in enumerate(samples):
        previous = samples[index - 1]
        following = samples[(index + 1) % len(samples)]
        span = (following["distance"] - previous["distance"]) % track_length
        derivative = (offsets_after[index] - offsets_before[index]) / max(span, 0.1)
        headings.append(math.atan(derivative))
    return headings


def _phase(speed_kmh, maximum_speed_kmh):
    ratio = speed_kmh / maximum_speed_kmh
    if ratio > 0.92:
        return "full_throttle"
    if ratio > 0.72:
        return "fast_corner"
    if ratio > 0.50:
        return "braking_or_medium_corner"
    return "slow_corner"


def _interpolate(left, right, fraction):
    return left + (right - left) * fraction
