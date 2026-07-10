import bisect
import hashlib
import json
import math
from pathlib import Path


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


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

        numeric_keys = (
            "target_track_pos",
            "target_speed_kmh",
            "curvature",
            "heading_offset",
            "speed_delta_30m",
            "turn_direction",
        )
        interpolated = {}
        for key in numeric_keys:
            if key in left and key in right:
                interpolated[key] = _interpolate(left[key], right[key], fraction)
            else:
                interpolated[key] = 0.0
        return interpolated | {"phase": left["phase"]}


class RacingLineOptimizer:
    """Bounded minimum-curvature line plus friction-limited speed profile."""

    def __init__(
        self,
        track_map,
        *,
        measured_length=None,
        spacing=3.0,
        safety_margin=1.15,
        maximum_speed_kmh=216.0,
        minimum_speed_kmh=82.0,
        lateral_acceleration=12.5,
        acceleration=8.5,
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

        samples = self.track_map.sample_centerline(
            self.spacing,
            self.measured_length,
        )
        centre = np.array([(point["x"], point["y"]) for point in samples])
        normals = np.array(
            [(point["normal_x"], point["normal_y"]) for point in samples]
        )
        sample_distances = np.array([point["distance"] for point in samples])
        desired_controls = {}

        def add_control(index, value, weight=1.0):
            values = desired_controls.setdefault(index % len(samples), [])
            values.append((value, weight))

        def nearest_sample_index(distance):
            distance = distance % lookup_length
            index = int(np.searchsorted(sample_distances, distance))
            before = (index - 1) % len(samples)
            after = index % len(samples)
            before_error = abs(
                (sample_distances[before] - distance + lookup_length / 2.0)
                % lookup_length
                - lookup_length / 2.0
            )
            after_error = abs(
                (sample_distances[after] - distance + lookup_length / 2.0)
                % lookup_length
                - lookup_length / 2.0
            )
            return before if before_error <= after_error else after

        lookup_length = self.measured_length or self.track_map.geometry_length
        usable_half_width = self.track_map.width / 2.0 - self.safety_margin

        segment_sample_indices = []
        for segment_index, _segment in enumerate(self.track_map.segments):
            segment_sample_indices.append(
                [
                    index
                    for index, point in enumerate(samples)
                    if point["segment_index"] == segment_index
                ]
            )

        def neighbouring_turn(segment_index, step):
            index = (segment_index + step) % len(self.track_map.segments)
            while index != segment_index:
                segment = self.track_map.segments[index]
                if segment.turn_radians:
                    return index, math.copysign(1.0, segment.turn_radians)
                index = (index + step) % len(self.track_map.segments)
            return None, 0.0

        def segment_start_distance(segment_index):
            return sample_distances[segment_sample_indices[segment_index][0]]

        def segment_end_distance(segment_index):
            return (
                sample_distances[segment_sample_indices[segment_index][-1]]
                + self.spacing
            ) % lookup_length

        def cyclic_gap(start, end):
            return (end - start) % lookup_length

        turn_groups = []
        current_group = None

        for segment_index, segment in enumerate(self.track_map.segments):
            sample_indices = segment_sample_indices[segment_index]
            if not sample_indices:
                continue

            if not segment.turn_radians or len(sample_indices) <= 2:
                continue

            turn_sign = math.copysign(1.0, segment.turn_radians)
            corner_start = sample_distances[sample_indices[0]]
            corner_end = (
                sample_distances[sample_indices[-1]]
                + self.spacing
            ) % lookup_length

            if (
                current_group is not None
                and current_group["sign"] == turn_sign
                and cyclic_gap(current_group["end"], corner_start) < 75.0
            ):
                current_group["end"] = corner_end
                current_group["segments"].append(segment_index)
                current_group["turn_radians"] += segment.turn_radians
            else:
                current_group = {
                    "sign": turn_sign,
                    "start": corner_start,
                    "end": corner_end,
                    "segments": [segment_index],
                    "turn_radians": segment.turn_radians,
                }
                turn_groups.append(current_group)

        for index, group in enumerate(turn_groups):
            turn_sign = group["sign"]
            corner_start = group["start"]
            corner_end = group["end"]
            span = max(cyclic_gap(corner_start, corner_end), self.spacing)
            apex_distance = (corner_start + span * 0.68) % lookup_length
            previous_group = turn_groups[index - 1]
            next_group = turn_groups[(index + 1) % len(turn_groups)]
            approach_gap = cyclic_gap(previous_group["end"], corner_start)
            exit_gap = cyclic_gap(corner_end, next_group["start"])
            technical_complex = _is_tight_alternating_complex(
                previous_group,
                group,
                next_group,
                lookup_length,
            )
            approach_length = clamp(approach_gap * 0.55, 45.0, 125.0)
            exit_length = clamp(exit_gap * 0.48, 35.0, 115.0)
            next_approach_length = clamp(exit_gap * 0.55, 45.0, 125.0)
            exit_entry_overlap = exit_length + next_approach_length > exit_gap
            group["approach_start"] = (corner_start - approach_length) % lookup_length
            group["apex"] = apex_distance
            group["exit_end"] = (corner_end + exit_length) % lookup_length
            group["technical_complex"] = technical_complex

            outside_scale, apex_scale, exit_scale = _line_scales(technical_complex)
            next_technical_complex = _is_tight_alternating_complex(
                group,
                next_group,
                turn_groups[(index + 2) % len(turn_groups)],
                lookup_length,
            )
            next_entry_scale, _next_apex_scale, _next_exit_scale = _line_scales(
                next_technical_complex
            )
            outside = -turn_sign * usable_half_width * outside_scale
            apex = turn_sign * usable_half_width * apex_scale
            next_entry_outside = -next_group["sign"] * usable_half_width * next_entry_scale
            natural_exit_outside = -turn_sign * usable_half_width * exit_scale
            if exit_entry_overlap:
                # The exit of this bend is the entry setup for the next bend.
                # Do not open to this corner's outside in the same metres
                # where the following sector is trying to set up its entry.
                corner_exit = (
                    next_entry_outside if exit_gap < 75.0 else natural_exit_outside
                )
                mid_exit = (corner_exit + next_entry_outside) / 2.0
                final_exit = next_entry_outside
            else:
                corner_exit = natural_exit_outside
                mid_exit = natural_exit_outside
                final_exit = natural_exit_outside

            # A usable racing line starts before the whole corner group,
            # crosses once toward the apex, then opens once toward the exit.
            # Treating every TORCS curve segment as its own corner caused the
            # first compound left to ask for multiple different turns.
            add_control(
                nearest_sample_index(group["approach_start"]),
                outside,
                weight=3.50,
            )
            add_control(
                nearest_sample_index(corner_start - approach_length * 0.35),
                outside,
                weight=4.00,
            )
            add_control(nearest_sample_index(corner_start), outside, weight=5.00)
            add_control(
                nearest_sample_index(corner_start + span * 0.34),
                outside * 0.60 + apex * 0.40,
                weight=5.50,
            )
            add_control(nearest_sample_index(apex_distance), apex, weight=8.00)
            add_control(nearest_sample_index(corner_end), corner_exit, weight=5.00)
            add_control(
                nearest_sample_index(corner_end + exit_length * 0.45),
                mid_exit,
                weight=4.00,
            )
            add_control(
                nearest_sample_index(group["exit_end"]),
                final_exit,
                weight=3.00,
            )

        control_indices = sorted(desired_controls)
        control_distances = sample_distances[control_indices]
        control_count = len(control_indices)

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
            return np.interp(sample_distances, periodic_distances, periodic_controls)

        initial_values = [
            (
                sum(value * weight for value, weight in desired_controls[sample_index])
                / sum(weight for _value, weight in desired_controls[sample_index])
            )
            for sample_index in control_indices
        ]
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
            smoothness_cost = 0.0040 * np.sum(control_second_difference**2)
            lateral_slope = (
                np.roll(offsets, -1) - np.roll(offsets, 1)
            ) / (2.0 * self.spacing)
            maximum_lateral_slope = math.tan(math.radians(9.0))
            slope_excess = np.maximum(
                np.abs(lateral_slope) - maximum_lateral_slope,
                0.0,
            )
            line_heading_cost = 30.0 * np.sum(slope_excess**2)
            length_cost = 0.00004 * np.sum(lengths)
            # Keep the numerical solution recognisably on a racing line. Pure
            # curvature minimisation tends to settle near the centre on linked
            # bends, even though that sacrifices the outside-apex-outside line
            # and the exit speed it enables.
            classical_line_cost = 0.055 * np.sum((controls - initial) ** 2)
            return (
                curvature_cost
                + smoothness_cost
                + line_heading_cost
                + length_cost
                + classical_line_cost
            )
        optimized_controls = initial
        optimizer_success = True
        optimizer_message = "deterministic sector-based racing line"
        optimizer_objective = float(objective(initial))

        offsets = expanded_offsets(optimized_controls)
        offsets = _enforce_classical_corner_shape(
            offsets,
            samples,
            turn_groups,
            lookup_length,
            usable_half_width,
        )
        offsets = _smooth_cyclic_values(offsets, passes=1)
        offsets = _limit_lateral_slope(
            offsets,
            max_delta=self.spacing * math.tan(math.radians(5.0)),
        )
        offsets = _smooth_cyclic_values(offsets, passes=1)
        offsets = np.clip(offsets, -usable_half_width, usable_half_width)
        path = centre + normals * offsets[:, None]
        curvatures, path_lengths = _closed_path_curvature(path)
        speeds = self._speed_profile(curvatures, path_lengths)
        heading_offsets = _heading_offsets(offsets, samples)
        speed_deltas = _speed_deltas(speeds, self.spacing, window_m=30.0)
        turn_directions = _turn_directions(curvatures, self.spacing, window_m=30.0)

        waypoints = []
        for index, sample in enumerate(samples):
            speed_kmh = float(speeds[index] * 3.6)
            speed_delta_kmh = float(speed_deltas[index] * 3.6)
            turn_direction = float(turn_directions[index])
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
                    "speed_delta_30m": round(speed_delta_kmh, 3),
                    "turn_direction": round(turn_direction, 6),
                    "phase": _phase(
                        speed_kmh,
                        speed_delta_kmh,
                        turn_direction,
                        float(curvatures[index]),
                        self.maximum_speed_kmh,
                    ),
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
                    "success": optimizer_success,
                    "message": optimizer_message,
                    "objective": optimizer_objective,
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


def _speed_deltas(speeds, spacing, window_m=30.0):
    import numpy as np

    step = max(1, round(window_m / spacing))
    return np.roll(speeds, -step) - speeds


def _turn_directions(curvatures, spacing, window_m=30.0):
    import numpy as np

    radius = max(1, round(window_m / spacing))
    directions = []
    for index in range(len(curvatures)):
        window = [
            curvatures[(index + offset) % len(curvatures)]
            for offset in range(-radius // 2, radius + 1)
        ]
        mean_curvature = float(np.mean(window))
        if abs(mean_curvature) < 0.0025:
            directions.append(0.0)
        else:
            directions.append(math.copysign(1.0, mean_curvature))
    return directions


def _smooth_cyclic_values(values, passes=1):
    import numpy as np

    smoothed = np.asarray(values, dtype=float).copy()
    for _ in range(passes):
        smoothed = (
            np.roll(smoothed, 2) * 0.06
            + np.roll(smoothed, 1) * 0.24
            + smoothed * 0.40
            + np.roll(smoothed, -1) * 0.24
            + np.roll(smoothed, -2) * 0.06
        )
    return smoothed


def _limit_lateral_slope(values, max_delta):
    import numpy as np

    limited = np.asarray(values, dtype=float).copy()
    count = len(limited)
    for _ in range(6):
        for index in range(count):
            previous = (index - 1) % count
            delta = limited[index] - limited[previous]
            if delta > max_delta:
                limited[index] = limited[previous] + max_delta
            elif delta < -max_delta:
                limited[index] = limited[previous] - max_delta

        for index in range(count - 1, -1, -1):
            following = (index + 1) % count
            delta = limited[index] - limited[following]
            if delta > max_delta:
                limited[index] = limited[following] + max_delta
            elif delta < -max_delta:
                limited[index] = limited[following] - max_delta
    return limited


def _is_tight_alternating_complex(previous_group, group, next_group, track_length):
    """True when a bend is part of a close left-right or right-left complex.

    In these linked corners, the fastest drivable line is not a full
    outside-apex-outside swing for each individual bend. Asking for the full
    swing makes the controller clip the edge, recover, and lose far more time
    than the shorter geometric path could ever gain.
    """

    approach_gap = (group["start"] - previous_group["end"]) % track_length
    exit_gap = (next_group["start"] - group["end"]) % track_length
    tight_opposite_entry = (
        previous_group["sign"] != group["sign"] and approach_gap < 135.0
    )
    tight_opposite_exit = next_group["sign"] != group["sign"] and exit_gap < 135.0
    return tight_opposite_entry or tight_opposite_exit


def _line_scales(technical_complex):
    if technical_complex:
        # TORCS' visible rubbered-in groove is smoother than a textbook
        # outside-apex-outside line in linked chicanes. Keep the car on a
        # recognisable racing path, but avoid asking for full-width lane swaps
        # where the next bend immediately reverses direction.
        return 0.24, 0.12, 0.20
    return 0.56, 0.34, 0.52


def _enforce_classical_corner_shape(
    offsets,
    samples,
    turn_groups,
    track_length,
    usable_half_width,
):
    import numpy as np

    shaped = np.asarray(offsets, dtype=float).copy()
    group_by_sample = {}
    for group_index, group in enumerate(turn_groups):
        group["index"] = group_index
        span = (group["end"] - group["start"]) % track_length
        if span <= 0.0:
            continue
        for index, sample in enumerate(samples):
            travelled = (sample["distance"] - group["start"]) % track_length
            if travelled <= span:
                group_by_sample[index] = group

    for index, sample in enumerate(samples):
        group = group_by_sample.get(index)
        if group is None:
            continue

        turn_sign = group["sign"]
        outside_scale, apex_scale, exit_scale = _line_scales(
            group.get("technical_complex", False)
        )
        outside = -turn_sign * usable_half_width * outside_scale
        apex = turn_sign * usable_half_width * apex_scale
        next_group = turn_groups[(group["index"] + 1) % len(turn_groups)]
        exit_gap = (next_group["start"] - group["end"]) % track_length
        exit_length = (group["exit_end"] - group["end"]) % track_length
        next_approach_length = clamp(exit_gap * 0.55, 45.0, 125.0)
        if exit_length + next_approach_length > exit_gap:
            next_entry_scale, _next_apex_scale, _next_exit_scale = _line_scales(
                next_group.get("technical_complex", False)
            )
            exit_outside = -next_group["sign"] * usable_half_width * next_entry_scale
        else:
            exit_outside = -turn_sign * usable_half_width * exit_scale
        span = max((group["end"] - group["start"]) % track_length, 0.1)
        fraction = ((sample["distance"] - group["start"]) % track_length) / span
        if fraction <= 0.5:
            blend = fraction / 0.5
            desired = outside * (1.0 - blend) + apex * blend
        else:
            blend = (fraction - 0.5) / 0.5
            desired = apex * (1.0 - blend) + exit_outside * blend

        apex_weight = 1.0 - min(abs(fraction - 0.5) / 0.5, 1.0)
        weight = 0.45 + 0.35 * apex_weight
        shaped[index] = shaped[index] * (1.0 - weight) + desired * weight
    return shaped


def _phase(speed_kmh, speed_delta_kmh, turn_direction, curvature, maximum_speed_kmh):
    if turn_direction > 0.0:
        turn = "left_turn"
    elif turn_direction < 0.0:
        turn = "right_turn"
    else:
        turn = ""

    if speed_delta_kmh < -11.0:
        return f"brake_{turn}" if turn else "brake"
    if speed_delta_kmh > 8.0:
        return f"accelerate_{turn}" if turn else "accelerate"
    if turn:
        return turn

    ratio = speed_kmh / maximum_speed_kmh
    if ratio > 0.92:
        return "full_throttle"
    if abs(curvature) > 0.004:
        return "turn"
    return "accelerate"


def _interpolate(left, right, fraction):
    return left + (right - left) * fraction
