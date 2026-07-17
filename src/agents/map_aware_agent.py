import csv
import hashlib
import math
from enum import Enum
from pathlib import Path

from racing_line import (
    RacingLine,
    DrivingMode,
    choose_driving_mode,
    smooth_value,
    limit_change,
    calculate_aggressive_speed_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACING_LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "g-track-3.json"
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "data" / "map_aware_telemetry.csv"

GEAR_SPEEDS = [0, 45, 85, 125, 165, 205]
TRACK_SENSOR_ANGLES = [
    -45,
    -19,
    -12,
    -7,
    -4,
    -2.5,
    -1.7,
    -1,
    -0.5,
    0,
    0.5,
    1,
    1.7,
    2.5,
    4,
    7,
    12,
    19,
    45,
]
INSIDE_CORRIDOR_CORRECTION_GAIN = 0.58
CORRIDOR_EDGE_CORRECTION_GAIN = 0.98
OUTSIDE_CORRIDOR_CORRECTION_GAIN = 1.30
MAX_STARTUP_LINE_BLEND_FLOOR = 0.65
SAFE_TRACK_LIMIT = 0.82
HARD_TRACK_LIMIT = 0.96
TELEMETRY_COLUMNS = [
    "step",
    "distFromStart",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
    "controlState",
    "rawRacingLinePos",
    "practicalTrackPos",
    "targetTrackPos",
    "previewTrackPos",
    "steeringTrackPos",
    "lineError",
    "effectiveLineError",
    "corridorHalfWidth",
    "lineComplexity",
    "targetSpeed",
    "targetCurvature",
    "headingOffset",
    "speedDelta30m",
    "turnDirection",
    "sensorFront",
    "sensorBestAngle",
    "sensorRawSpeedCap",
    "sensorSafetyCap",
    *[f"trackSensor{index}" for index in range(19)],
    "lookaheadDistance",
    "lineBlend",
    "referenceThrottle",
    "referenceBrake",
    "referenceSection",
    "phase",
    "steer",
    "accel",
    "brake",
    "gear",
    "curLapTime",
    "lastLapTime",
]


class RacingControlState(str, Enum):
    JOIN_RACELINE = "JOIN_RACELINE"
    RACE = "RACE"
    EDGE_GUARD = "EDGE_GUARD"
    RECOVERY = "RECOVERY"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def shift_gears(speed):
    gear = 1
    for index, threshold in enumerate(GEAR_SPEEDS):
        if speed > threshold:
            gear = index + 1
    return int(clamp(gear, 1, 6))


def get_racing_line_corridor(speed, curvature, line_complexity=0.0):
    """Return a normalized lateral leeway around the saved racing line.

    TORCS trackPos is normalized around the centreline where roughly -1 and +1
    are the track edges. The saved racing line is still the ideal point, but
    the car should not snap toward it while already inside a useful racing
    corridor. Faster sections get a little more leeway so steering stays calm;
    tight corners keep a narrower corridor so the apex still matters.
    """
    curvature_abs = abs(curvature)
    if curvature_abs > 0.028:
        corridor = 0.052
    elif curvature_abs > 0.016:
        corridor = 0.070
    elif curvature_abs > 0.008:
        corridor = 0.090
    else:
        corridor = 0.112

    if speed > 170:
        corridor += 0.014
    elif speed > 125:
        corridor += 0.008
    elif speed < 70:
        corridor -= 0.008

    # Linked bends need a little more lateral freedom. If the target line
    # moves left-right-left quickly, snapping to the exact ideal point causes
    # oscillation and costs more time than a slightly wider, calmer corridor.
    corridor += 0.014 * line_complexity

    return clamp(corridor, 0.040, 0.135)


def soften_line_error(line_error, corridor_half_width):
    """Keep a gentle pull to the line inside the corridor, not a hard snap."""
    error_abs = abs(line_error)
    if error_abs < 1e-9:
        return 0.0

    direction = 1.0 if line_error > 0.0 else -1.0
    if error_abs <= corridor_half_width:
        corridor_fraction = error_abs / corridor_half_width
        gain = (
            INSIDE_CORRIDOR_CORRECTION_GAIN
            + (CORRIDOR_EDGE_CORRECTION_GAIN - INSIDE_CORRIDOR_CORRECTION_GAIN)
            * corridor_fraction
            * corridor_fraction
        )
        return line_error * gain

    outside_error = error_abs - corridor_half_width
    softened = (
        corridor_half_width * CORRIDOR_EDGE_CORRECTION_GAIN
        + outside_error * OUTSIDE_CORRIDOR_CORRECTION_GAIN
    )
    return direction * softened


def cap_speed_for_line_join(target_speed, track_position, preview_position):
    """Keep line joining progressive when the car is far from the saved path."""

    join_error = abs(track_position - preview_position)
    if join_error > 0.58:
        return min(target_speed, 98.0)
    if join_error > 0.46:
        return min(target_speed, 112.0)
    if join_error > 0.34:
        return min(target_speed, 130.0)
    if join_error > 0.24:
        return min(target_speed, 150.0)
    return target_speed


def target_limit_for_state(control_state, speed, lateral_speed, angle, line_blend):
    """Return the practical raceline edge limit for the current control state."""

    stability = 1.0 - clamp(
        max(abs(lateral_speed) / 5.5, abs(angle) / 0.12),
        0.0,
        1.0,
    )
    if control_state == RacingControlState.RECOVERY:
        return 0.48
    if control_state == RacingControlState.EDGE_GUARD:
        return 0.64
    if control_state == RacingControlState.JOIN_RACELINE:
        return 0.64 + 0.06 * stability * line_blend

    speed_authority = clamp((speed - 65.0) / 80.0, 0.0, 1.0)
    return 0.66 + 0.09 * stability * speed_authority


def apply_practical_target_limit(
    target_position,
    *,
    control_state,
    speed,
    lateral_speed,
    angle,
    line_blend,
):
    limit = target_limit_for_state(
        control_state,
        speed,
        lateral_speed,
        angle,
        line_blend,
    )
    return clamp(target_position, -limit, limit)


def choose_control_state(
    previous_state,
    *,
    line_blend,
    track_position,
    target_position,
    speed,
    lateral_speed,
    angle,
    front_sensor,
    min_track_sensor,
    raw_sensor_speed_cap,
):
    """Select the high-level racing controller state with simple hysteresis."""

    edge_abs = abs(track_position)
    line_error_abs = abs(track_position - target_position)
    outward_motion = track_position * lateral_speed > 2.2
    outward_heading = track_position * angle < -0.05 and edge_abs > 0.30
    off_track = front_sensor < 0.0 or min_track_sensor < 0.0 or edge_abs > 1.02

    recovery_trigger = (
        off_track
        or edge_abs > 0.96
        or abs(angle) > 0.60
        or abs(lateral_speed) > 16.0
        or (edge_abs > 0.74 and outward_motion)
    )
    if previous_state == RacingControlState.RECOVERY and not (
        edge_abs < 0.76
        and abs(angle) < 0.34
        and abs(lateral_speed) < 8.0
        and front_sensor > 18.0
    ):
        recovery_trigger = True
    if recovery_trigger:
        return RacingControlState.RECOVERY

    close_front_risk = (
        front_sensor < 8.0
        or (
            front_sensor < 16.0
            and (
                edge_abs > 0.40
                or abs(lateral_speed) > 2.8
                or abs(angle) > 0.10
                or line_error_abs > 0.26
            )
        )
    )
    sensor_edge_risk = (
        raw_sensor_speed_cap < 216.0
        and (
            close_front_risk
            or edge_abs > 0.72
            or (front_sensor < 32.0 and edge_abs > 0.38 and abs(lateral_speed) > 4.0)
            or (front_sensor < 32.0 and edge_abs > 0.38 and abs(angle) > 0.14)
            or (
                edge_abs > 0.54
                and (outward_heading or outward_motion or abs(lateral_speed) > 3.2)
            )
            or (line_error_abs > 0.34 and abs(lateral_speed) > 2.5)
        )
    )
    edge_trigger = edge_abs > 0.86 or front_sensor < 6.0 or sensor_edge_risk
    if previous_state == RacingControlState.EDGE_GUARD:
        edge_guard_can_release = (
            edge_abs < 0.70
            and front_sensor > 20.0
            and abs(lateral_speed) < 5.0
            and abs(angle) < 0.22
        ) or (
            edge_abs < 0.80
            and front_sensor > 32.0
            and abs(lateral_speed) < 3.2
            and abs(angle) < 0.14
        )
        if not edge_guard_can_release:
            edge_trigger = True
    if edge_trigger:
        return RacingControlState.EDGE_GUARD

    join_trigger = line_blend < 0.995 or line_error_abs > 0.18
    if previous_state == RacingControlState.JOIN_RACELINE and not (
        line_blend > 0.995
        and line_error_abs < 0.10
        and abs(lateral_speed) < 4.0
        and abs(angle) < 0.12
    ):
        join_trigger = True
    if join_trigger:
        return RacingControlState.JOIN_RACELINE

    return RacingControlState.RACE


def cap_speed_for_control_state(
    target_speed,
    *,
    control_state,
    line_join_error,
    line_error,
    speed,
    lateral_speed,
    angle,
    front_sensor,
):
    """Make speed reflect how well the car can actually follow the path."""

    capped = target_speed
    if control_state == RacingControlState.JOIN_RACELINE:
        if line_join_error > 0.50:
            capped = min(capped, 102.0)
        elif line_join_error > 0.36:
            capped = min(capped, 122.0)
        elif line_join_error > 0.24:
            capped = min(capped, 145.0)
    elif control_state == RacingControlState.EDGE_GUARD:
        capped = min(capped, 118.0)
        if front_sensor < 24.0:
            capped = min(capped, 96.0)
        if abs(lateral_speed) > 7.0 or abs(angle) > 0.22:
            capped = min(capped, 86.0)
    elif control_state == RacingControlState.RECOVERY:
        capped = min(capped, 62.0)

    if abs(line_error) > 0.42:
        capped = min(capped, max(72.0, speed + 4.0))
    if abs(lateral_speed) > 10.0:
        capped = min(capped, 82.0)
    if abs(angle) > 0.36:
        capped = min(capped, 76.0)
    return clamp(capped, 42.0, 216.0)


def calculate_path_tracking_steer(
    *,
    speed,
    track_position,
    target_position,
    heading_offset,
    curvature,
    lateral_speed,
    angle,
    previous_steer,
    control_state,
):
    """Stanley-style path tracker with curvature feed-forward."""

    line_error = track_position - target_position
    speed_ms = max(speed / 3.6, 4.0)

    if control_state == RacingControlState.RECOVERY:
        raw_steer = (
            -track_position * 1.50
            + (angle + heading_offset) * 2.1
            - lateral_speed * 0.040
        )
        limit = 0.90
        max_delta = 0.12 if speed > 35.0 else 0.18
        smooth_alpha = 0.48
    else:
        if control_state == RacingControlState.JOIN_RACELINE:
            cross_gain = 4.2
            heading_gain = 0.82
            curvature_gain = 1.45
            lateral_gain = 0.024
            cross_component_gain = 2.10
            limit = 0.62 if speed > 80.0 else 0.70
            max_delta = 0.046 if speed > 80.0 else 0.060
            smooth_alpha = 0.34
        elif control_state == RacingControlState.EDGE_GUARD:
            cross_gain = 3.8
            heading_gain = 0.86
            curvature_gain = 1.10
            lateral_gain = 0.040
            cross_component_gain = 1.85
            limit = 0.74
            max_delta = 0.066 if speed > 80.0 else 0.082
            smooth_alpha = 0.38
        else:
            cross_gain = 5.0
            heading_gain = 1.00
            curvature_gain = 2.45
            lateral_gain = 0.024
            cross_component_gain = 2.15
            limit = 0.70 if speed < 150.0 else 0.58
            max_delta = 0.052 if speed > 110.0 else 0.066
            smooth_alpha = 0.38

        heading_error = angle + heading_offset
        cross_track_component = -math.atan2(cross_gain * line_error, speed_ms + 6.0)
        raw_steer = (
            heading_error * heading_gain
            + cross_track_component * cross_component_gain
            + curvature * curvature_gain
            - lateral_speed * lateral_gain
        )

    raw_steer = clamp(raw_steer, -limit, limit)
    smoothed = previous_steer * (1.0 - smooth_alpha) + raw_steer * smooth_alpha
    steer = limit_change(previous_steer, smoothed, max_delta)
    return clamp(steer, -limit, limit)


def estimate_line_complexity(target, lookahead, far_lookahead):
    lateral_swing = max(
        abs(lookahead["target_track_pos"] - target["target_track_pos"]),
        abs(far_lookahead["target_track_pos"] - lookahead["target_track_pos"]),
    )
    curvature_values = [
        target["curvature"],
        lookahead["curvature"],
        far_lookahead["curvature"],
    ]
    strong_curvatures = [
        value
        for value in curvature_values
        if abs(value) > 0.006
    ]
    has_direction_change = any(
        left * right < 0.0
        for left, right in zip(strong_curvatures, strong_curvatures[1:])
    )
    complexity = clamp((lateral_swing - 0.10) / 0.22, 0.0, 1.0)
    if has_direction_change:
        complexity += 0.28
    return clamp(complexity, 0.0, 1.0)


def preview_track_position(target, lookahead, far_lookahead, speed, line_complexity):
    """Aim through the racing line instead of chasing one instantaneous point."""

    current_position = target["target_track_pos"]
    future_position = (
        lookahead["target_track_pos"] * 0.55
        + far_lookahead["target_track_pos"] * 0.45
    )
    active_corner = (
        abs(target["curvature"]) > 0.006
        or abs(lookahead["curvature"]) > 0.006
        or abs(target.get("turn_direction", 0.0)) > 0.4
    )
    unwinding_from_apex = (
        active_corner
        and abs(current_position) > 0.26
        and (
            current_position * future_position <= 0.0
            or abs(future_position) < abs(current_position) - 0.08
        )
    )
    if unwinding_from_apex:
        return current_position

    preview_weight = clamp(0.30 + speed / 420.0 + line_complexity * 0.12, 0.30, 0.72)
    if active_corner:
        preview_weight = min(preview_weight, 0.32)
    maximum_preview_shift = 0.28 + line_complexity * 0.16
    preview_shift = clamp(
        future_position - current_position,
        -maximum_preview_shift,
        maximum_preview_shift,
    )
    return current_position + preview_shift * preview_weight


def get_track_sensors(telemetry):
    track = telemetry.get("track", [])
    if len(track) >= 19:
        return [float(value) for value in track[:19]]
    return [float(telemetry.get(f"track_{index}", 200.0)) for index in range(19)]


def get_best_sensor(track):
    front = track[9]
    candidate_indices = range(4, 15) if front < 65.0 else range(6, 13)
    best_index = 9
    best_score = front * 1.12

    for index in candidate_indices:
        offset = abs(index - 9)
        centre_penalty = 1.0 - offset * (0.035 if front < 65.0 else 0.10)
        score = track[index] * centre_penalty
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def sensor_speed_cap(track, speed, previous_front):
    front = track[9]
    front_delta = 0.0 if previous_front is None else front - previous_front
    closing_fast = front_delta < -0.75

    if front < 8.0:
        return 42.0
    if front < 14.0:
        return 60.0
    if front < 22.0:
        return 82.0
    if front < 32.0:
        return 105.0
    if front < 45.0:
        return 128.0
    if closing_fast and speed > 155.0:
        if front < 76.0:
            return 150.0
        if front < 102.0:
            return 172.0
    return 216.0


def map_safety_speed_cap(
    track,
    speed,
    previous_front,
    track_position,
    line_error,
    target_position=None,
    lateral_speed=0.0,
    angle=0.0,
):
    """Use range sensors as an emergency veto, not as the normal speed plan.

    On sweeping TORCS corners, the straight-ahead sensor often points at the
    outside edge even though the mapped racing line continues safely through the
    bend. The map-aware agent should therefore drive from its saved path and
    only let sensors cap speed when the car is plausibly leaving the drivable
    corridor or has almost no forward road left.
    """

    raw_cap = sensor_speed_cap(track, speed, previous_front)
    if raw_cap >= 216.0:
        return 216.0

    front = track[9]
    edge_abs = abs(track_position)
    minimum_sensor = min(track) if track else front
    best_index = get_best_sensor(track)
    best_distance = track[best_index]
    best_angle = abs(TRACK_SENSOR_ANGLES[best_index])
    off_track = front < 0.0 or minimum_sensor < 0.0 or edge_abs > 1.02
    almost_no_road = front < 7.0 and best_distance < 18.0
    wrong_side_edge = edge_abs > 0.78 and abs(line_error) > 0.22 and front < 24.0
    no_escape_lane = front < 12.0 and best_distance < max(front + 8.0, 20.0)
    outward_target = (
        target_position is not None
        and abs(target_position) > abs(track_position) + 0.08
        and target_position * (track_position if abs(track_position) > 0.04 else target_position) > 0.0
    )
    unstable_join = (
        abs(lateral_speed) > 4.5
        or abs(angle) > 0.14
        or abs(line_error) > 0.24
    )

    if off_track:
        return min(raw_cap, 70.0)
    if almost_no_road or no_escape_lane:
        return raw_cap
    if outward_target and unstable_join and front < 45.0:
        margin = 36.0
        if front < 32.0:
            margin = 28.0
        if front < 24.0:
            margin = 20.0
        return min(216.0, raw_cap + margin)
    if front < 24.0 and abs(line_error) > 0.32:
        return min(raw_cap + 20.0, 150.0)
    if wrong_side_edge and best_angle > 7.0:
        return min(raw_cap + 18.0, 150.0)
    return 216.0


def curvature_limited_speed(
    curvature,
    *,
    lateral_acceleration=8.6,
    minimum_speed=58.0,
    maximum_speed=216.0,
):
    curvature_abs = abs(curvature)
    if curvature_abs < 0.0012:
        return maximum_speed

    speed = math.sqrt(lateral_acceleration / curvature_abs) * 3.6
    return clamp(speed, minimum_speed, maximum_speed)


def calculate_dynamic_target_speed(
    *,
    speed,
    target,
    lookahead,
    far_lookahead,
    very_far_lookahead,
    live_sensor_speed_cap,
    line_complexity,
    line_error,
):
    """Calculate speed from the saved raceline, with map-based preview taps."""

    saved_speed_limits = [
        target["target_speed_kmh"] + 8.0,
        lookahead["target_speed_kmh"] + 5.0,
        far_lookahead["target_speed_kmh"] + 12.0,
        very_far_lookahead["target_speed_kmh"] + 22.0,
    ]
    curvature_limits = [
        curvature_limited_speed(target["curvature"], lateral_acceleration=11.2) + 8.0,
        curvature_limited_speed(lookahead["curvature"], lateral_acceleration=10.8)
        + 12.0,
        curvature_limited_speed(far_lookahead["curvature"], lateral_acceleration=10.0)
        + 26.0,
        curvature_limited_speed(
            very_far_lookahead["curvature"],
            lateral_acceleration=9.5,
        )
        + 42.0,
    ]
    target_speed = min(min(saved_speed_limits), min(curvature_limits))

    max_preview_curvature = max(
        abs(target["curvature"]),
        abs(lookahead["curvature"]),
        abs(far_lookahead["curvature"]),
        abs(very_far_lookahead["curvature"]),
    )
    if max_preview_curvature < 0.0035 and live_sensor_speed_cap > 170.0:
        target_speed = max(target_speed, 188.0)
    elif max_preview_curvature < 0.006 and live_sensor_speed_cap > 150.0:
        target_speed = max(target_speed, 162.0)

    if line_complexity > 0.72:
        target_speed -= 3.0

    if abs(line_error) > 0.35:
        target_speed -= 8.0
    if abs(line_error) > 0.55:
        target_speed -= 12.0

    if live_sensor_speed_cap < 216.0:
        target_speed = min(target_speed, live_sensor_speed_cap)

    # Avoid asking for a sudden over-brake from one frame of lookahead noise.
    if target_speed < speed - 65.0 and live_sensor_speed_cap > 82.0:
        target_speed = speed - 65.0

    return clamp(target_speed, 58.0, 216.0)


def choose_dynamic_control_phase(speed, target_speed, curvature, front_sensor):
    if target_speed < speed - 7.0:
        return "brake"
    if abs(curvature) > 0.006 or front_sensor < 7.0:
        return "turn"
    if target_speed > speed + 8.0:
        return "accelerate"
    return "accelerate"


def pursuit_lookahead_distance(speed, line_complexity=0.0):
    """Short lookahead for lateral tracking, long enough to start turn-in."""

    base = 14.0 + speed * 0.080
    if line_complexity > 0.45:
        base -= 2.5
    return clamp(base, 12.0, 24.0)


def choose_pursuit_track_position(local, near, far, speed, line_complexity):
    """Return the trackPos the steering should chase right now.

    The local raceline is the truth for telemetry. Steering, however, needs to
    chase a point slightly ahead on that same raceline so it starts turn-in
    before the car reaches the apex.
    """

    local_pos = local["target_track_pos"]
    near_pos = near["target_track_pos"]
    far_pos = far["target_track_pos"]
    active_corner = (
        abs(local["curvature"]) > 0.006
        or abs(near["curvature"]) > 0.006
        or abs(local.get("turn_direction", 0.0)) > 0.4
        or abs(near.get("turn_direction", 0.0)) > 0.4
    )

    if active_corner:
        if abs(near_pos) > abs(local_pos) + 0.08:
            return near_pos
        if abs(far_pos) > abs(local_pos) + 0.16 and speed > 70.0:
            return near_pos * 0.70 + far_pos * 0.30
        return local_pos

    preview_weight = clamp(0.25 + speed / 520.0 + line_complexity * 0.08, 0.25, 0.55)
    return local_pos * (1.0 - preview_weight) + near_pos * preview_weight


def choose_raceline_control_state(
    *,
    track_position,
    local_target,
    pursuit_target,
    speed,
    lateral_speed,
    angle,
    front_sensor,
    min_track_sensor,
):
    """Small state machine for the raceline-first controller."""

    edge_abs = abs(track_position)
    line_error_abs = max(
        abs(track_position - local_target),
        abs(track_position - pursuit_target),
    )
    outward_motion = track_position * lateral_speed > 1.4
    outward_heading = track_position * angle < -0.045 and edge_abs > 0.25
    off_track = min_track_sensor < 0.0 or front_sensor < 0.0 or edge_abs > 1.03
    if (
        off_track
        or abs(angle) > 0.62
        or abs(lateral_speed) > 17.0
        or (edge_abs > 0.94 and front_sensor < 12.0)
    ):
        return RacingControlState.RECOVERY

    near_edge_with_short_road = (
        front_sensor < 18.0
        and (
            edge_abs > 0.36
            or line_error_abs > 0.22
            or outward_motion
            or outward_heading
        )
    )
    if (
        edge_abs > 0.88
        or front_sensor < 5.0
        or (edge_abs > 0.68 and (outward_motion or outward_heading))
        or near_edge_with_short_road
    ):
        return RacingControlState.EDGE_GUARD

    if (
        abs(track_position - local_target) > 0.12
        or abs(track_position - pursuit_target) > 0.18
    ):
        return RacingControlState.JOIN_RACELINE

    return RacingControlState.RACE


def sensor_emergency_speed_cap(
    track_sensors,
    track_position,
    speed,
    previous_front,
    lateral_speed=0.0,
    angle=0.0,
):
    """Use TORCS range sensors as emergency limits, not navigation."""

    if not track_sensors:
        return 216.0

    front = track_sensors[9]
    minimum = min(track_sensors)
    raw_cap = sensor_speed_cap(track_sensors, speed, previous_front)
    edge_abs = abs(track_position)
    outward_motion = track_position * lateral_speed > 1.4
    outward_heading = track_position * angle < -0.045 and edge_abs > 0.25
    if front < 0.0 or minimum < 0.0:
        return 45.0
    if front < 5.0:
        return 38.0
    if front < 8.0:
        return 55.0
    if front < 18.0 and edge_abs > 0.34 and (
        outward_motion or outward_heading or abs(lateral_speed) > 2.8
    ):
        return min(raw_cap + 20.0, 92.0)
    if front < 24.0 and edge_abs > 0.52 and (outward_motion or outward_heading):
        return min(raw_cap + 28.0, 112.0)
    if edge_abs > 0.88 and front < 18.0:
        return min(raw_cap, 70.0)
    if edge_abs > 0.72 and front < 14.0:
        return min(raw_cap + 10.0, 90.0)
    return 216.0


def speedway_confidence_speed(preview_curvature, line_complexity=0.0):
    """Return an aggressive-but-plausible speed from map curvature."""

    curvature_abs = abs(preview_curvature)
    if curvature_abs < 0.0035:
        speed = 210.0
    elif curvature_abs < 0.006:
        speed = 198.0
    elif curvature_abs < 0.009:
        speed = 176.0
    elif curvature_abs < 0.012:
        speed = 154.0
    elif curvature_abs < 0.018:
        speed = 132.0
    elif curvature_abs < 0.026:
        speed = 112.0
    elif curvature_abs < 0.034:
        speed = 98.0
    else:
        speed = 90.0

    speed -= 8.0 * clamp(line_complexity, 0.0, 1.0)
    return clamp(speed, 88.0, 216.0)


def optimizer_sweeper_speed_cap(
    local,
    near,
    far,
    *,
    pursuit_target,
    line_complexity,
):
    """Respect the saved speed profile in long, edge-apex sweepers.

    The broad speedway confidence buckets are useful on open bends, but they
    can exceed the grip-limited speed when a sustained corner also asks the
    car to hold a wide lateral offset. Ramp this cap in with the apex offset so
    braking stays progressive on corner entry.
    """

    preview_curvature = max(
        abs(local["curvature"]),
        abs(near["curvature"]),
        abs(far["curvature"]),
    )
    directions = [
        float(local.get("turn_direction", 0.0)),
        float(near.get("turn_direction", 0.0)),
        float(far.get("turn_direction", 0.0)),
    ]
    saved_speeds = [
        float(local["target_speed_kmh"]),
        float(near["target_speed_kmh"]),
        float(far["target_speed_kmh"]),
    ]
    same_sustained_turn = (
        all(abs(direction) > 0.4 for direction in directions)
        and directions[0] * directions[1] > 0.0
        and directions[1] * directions[2] > 0.0
    )
    local_target = float(local.get("target_track_pos", pursuit_target))
    if not (
        same_sustained_turn
        and line_complexity < 0.18
        and 0.009 <= preview_curvature <= 0.015
        and pursuit_target > 0.44
        and min(saved_speeds) > 112.0
        and abs(pursuit_target - local_target) < 0.08
    ):
        return 216.0

    saved_profile_cap = min(
        saved_speeds[0] + 12.0,
        saved_speeds[1] + 16.0,
        saved_speeds[2] + 26.0,
    )
    grip_cap = curvature_limited_speed(
        preview_curvature,
        lateral_acceleration=10.8,
    ) + 14.0
    full_cap = min(saved_profile_cap, grip_cap)
    apex_commitment = clamp((abs(pursuit_target) - 0.44) / 0.08, 0.0, 1.0)
    open_sweeper_speed = speedway_confidence_speed(
        preview_curvature,
        line_complexity,
    )
    return (
        open_sweeper_speed * (1.0 - apex_commitment)
        + full_cap * apex_commitment
    )


def raceline_corner_throttle_cap(
    steer,
    track_position,
    pursuit_error,
    lateral_speed=0.0,
    angle=0.0,
):
    """Borrow agent 2's corner throttle discipline without centreline steering."""

    steer_abs = abs(steer)
    if steer_abs >= 0.75:
        cap = 0.10
    elif steer_abs >= 0.60:
        cap = 0.20
    elif steer_abs >= 0.45:
        cap = 0.35
    elif steer_abs >= 0.30:
        cap = 0.55
    else:
        cap = 1.0

    if pursuit_error > 0.58:
        cap = min(cap, 0.18)
    elif pursuit_error > 0.44:
        cap = min(cap, 0.32)
    elif pursuit_error > 0.30:
        cap = min(cap, 0.52)

    settled_on_wide_line = (
        pursuit_error < 0.12
        and steer_abs < 0.22
        and abs(lateral_speed) < 2.0
        and abs(angle) < 0.08
    )
    if abs(track_position) > 0.72:
        cap = min(cap, 0.42 if settled_on_wide_line else 0.26)
    elif abs(track_position) > 0.58:
        cap = min(cap, 0.72 if settled_on_wide_line else 0.42)
    return cap


def apply_raceline_traction_control(wheel_spin, accel):
    if wheel_spin is None or len(wheel_spin) < 4:
        return accel

    rear_spin = float(wheel_spin[2]) + float(wheel_spin[3])
    front_spin = float(wheel_spin[0]) + float(wheel_spin[1])
    if rear_spin - front_spin > 3.2:
        accel -= 0.10
    return clamp(accel, 0.0, 1.0)


def update_raceline_confidence_debt(
    previous_debt,
    previous_pursuit_error,
    pursuit_error,
    lateral_speed,
    angle,
    track_position,
):
    """Track whether the car has earned full-speed raceline confidence."""

    debt = previous_debt * 0.975
    error_abs = abs(pursuit_error)

    if (
        previous_pursuit_error is not None
        and previous_pursuit_error * pursuit_error < 0.0
        and max(abs(previous_pursuit_error), error_abs) > 0.16
        and error_abs > 0.10
    ):
        debt = max(debt, 0.82)

    if error_abs > 0.58:
        debt = max(debt, 1.0)
    elif error_abs > 0.42:
        debt = max(debt, 0.84)
    elif error_abs > 0.30:
        debt = max(debt, 0.62)
    elif error_abs > 0.24:
        debt = max(debt, 0.44)
    elif error_abs > 0.16:
        debt = max(debt, 0.28)

    if abs(lateral_speed) > 5.8 or abs(angle) > 0.18:
        debt = max(debt, 0.78)
    elif abs(lateral_speed) > 3.4 or abs(angle) > 0.12:
        debt = max(debt, 0.52)
    elif abs(lateral_speed) > 2.4 or abs(angle) > 0.09:
        debt = max(debt, 0.36)

    if abs(track_position) > 0.58 and track_position * lateral_speed > 1.0:
        debt = max(debt, 0.78)

    if error_abs < 0.10 and abs(lateral_speed) < 1.0 and abs(angle) < 0.045:
        debt *= 0.72
    elif error_abs < 0.16 and abs(lateral_speed) < 1.8 and abs(angle) < 0.075:
        debt *= 0.86

    return clamp(debt, 0.0, 1.0)


def confidence_speed_cap(confidence_debt):
    if confidence_debt <= 0.05:
        return 216.0
    return clamp(168.0 - confidence_debt * 96.0, 82.0, 216.0)


def calculate_raceline_target_speed(
    *,
    speed,
    local,
    near,
    far,
    very_far,
    track_position,
    pursuit_target,
    lateral_speed,
    angle,
    sensor_cap,
    line_complexity=0.0,
):
    """Speed plan from the saved raceline plus ability to reach it."""

    preview_curvature = max(
        abs(local["curvature"]),
        abs(near["curvature"]),
        abs(far["curvature"]),
    )
    curvature_cap = curvature_limited_speed(
        preview_curvature,
        lateral_acceleration=10.8,
        minimum_speed=58.0,
        maximum_speed=216.0,
    )
    target_speed = min(
        local["target_speed_kmh"] + 10.0,
        near["target_speed_kmh"] + 16.0,
        far["target_speed_kmh"] + 30.0,
        very_far["target_speed_kmh"] + 48.0,
        curvature_cap + 18.0,
        sensor_cap,
    )

    pursuit_error = abs(track_position - pursuit_target)
    settled_body = (
        abs(lateral_speed) < 4.8
        and abs(angle) < 0.12
        and abs(track_position) < 0.78
    )
    confidence_speed = speedway_confidence_speed(preview_curvature, line_complexity)

    if sensor_cap > 150.0:
        if settled_body and pursuit_error < 0.20:
            target_speed = max(target_speed, confidence_speed)
        elif settled_body and pursuit_error < 0.34:
            target_speed = max(target_speed, confidence_speed - 18.0)
        elif (
            pursuit_error < 0.48
            and abs(lateral_speed) < 3.2
            and abs(angle) < 0.08
        ):
            target_speed = max(target_speed, confidence_speed - 32.0)

    target_speed = min(
        target_speed,
        optimizer_sweeper_speed_cap(
            local,
            near,
            far,
            pursuit_target=pursuit_target,
            line_complexity=line_complexity,
        ),
    )
    target_speed = min(target_speed, sensor_cap)

    if pursuit_error > 0.82:
        target_speed = min(target_speed, 52.0)
    elif pursuit_error > 0.64:
        target_speed = min(target_speed, 68.0)
    elif pursuit_error > 0.48:
        target_speed = min(target_speed, 92.0)
    elif pursuit_error > 0.34:
        target_speed = min(target_speed, 118.0)
    elif pursuit_error > 0.22 and not settled_body:
        target_speed = min(target_speed, 138.0)

    if abs(lateral_speed) > 7.0:
        target_speed = min(target_speed, 118.0)
    if abs(lateral_speed) > 10.0:
        target_speed = min(target_speed, 88.0)
    if abs(angle) > 0.26:
        target_speed = min(target_speed, 96.0)
    if abs(angle) > 0.38:
        target_speed = min(target_speed, 72.0)

    # If the car is already too fast for the line, allow a strong speed drop.
    if target_speed < speed - 70.0 and sensor_cap > 70.0:
        target_speed = speed - 70.0
    return clamp(target_speed, 36.0, 216.0)


def calculate_raceline_pursuit_steer(
    *,
    speed,
    track_position,
    pursuit_target,
    heading_offset,
    curvature,
    lateral_speed,
    angle,
    previous_steer,
    control_state,
):
    """Direct Stanley/pure-pursuit style steering for the saved raceline."""

    if control_state == RacingControlState.RECOVERY:
        raw_steer = -track_position * 1.45 + angle * 1.90 - lateral_speed * 0.045
        limit = 0.90
        max_delta = 0.18 if speed < 60.0 else 0.13
        smooth_alpha = 0.62
    else:
        line_error = track_position - pursuit_target
        speed_ms = max(speed / 3.6, 4.0)
        heading_error = angle + heading_offset
        cross_track = -math.atan2(6.4 * line_error, speed_ms + 4.0)
        raw_steer = (
            cross_track * 2.55
            + heading_error * 0.72
            + curvature * 2.65
            + lateral_speed * 0.034
        )
        limit = 0.84 if speed < 120.0 else 0.72
        if control_state == RacingControlState.EDGE_GUARD:
            limit = 0.90
        if speed > 170.0:
            limit = min(limit, 0.42)
            max_delta = 0.034
            smooth_alpha = 0.22
        elif speed > 140.0:
            limit = min(limit, 0.52)
            max_delta = 0.046
            smooth_alpha = 0.28
        elif speed > 105.0:
            limit = min(limit, 0.62)
            max_delta = 0.060
            smooth_alpha = 0.36
        elif speed > 75.0:
            max_delta = 0.082
            smooth_alpha = 0.46
        else:
            max_delta = 0.108
            smooth_alpha = 0.54
        if (
            control_state in (
                RacingControlState.EDGE_GUARD,
                RacingControlState.JOIN_RACELINE,
            )
            and speed < 92.0
            and abs(line_error) > 0.62
        ):
            limit = min(limit, 0.62)
            max_delta = min(max_delta, 0.056 if speed > 45.0 else 0.070)
            smooth_alpha = min(smooth_alpha, 0.34)

    raw_steer = clamp(raw_steer, -limit, limit)
    if (
        previous_steer * raw_steer < 0.0
        and abs(previous_steer) > 0.22
        and abs(raw_steer) > 0.22
        and speed > 55.0
        and control_state != RacingControlState.RECOVERY
    ):
        raw_steer *= 0.55
    smoothed = previous_steer * (1.0 - smooth_alpha) + raw_steer * smooth_alpha
    return clamp(limit_change(previous_steer, smoothed, max_delta), -limit, limit)


def calculate_raceline_pedals(
    *,
    speed,
    target_speed,
    steer,
    pursuit_error,
    track_position,
    lateral_speed,
    angle,
    previous_accel,
    previous_brake,
    control_state,
):
    """Simple speed controller for the raceline follower."""

    speed_error = target_speed - speed
    stable_on_line = (
        pursuit_error < 0.18
        and abs(track_position) < 0.76
        and abs(angle) < 0.18
        and abs(lateral_speed) < 4.2
    )
    late_brake_ready = (
        control_state
        not in (
            RacingControlState.EDGE_GUARD,
            RacingControlState.RECOVERY,
        )
        and pursuit_error < 0.16
        and abs(track_position) < 0.72
        and abs(angle) < 0.10
        and abs(lateral_speed) < 2.8
    )
    if control_state == RacingControlState.RECOVERY:
        if speed < 8.0:
            accel, brake = 0.24, 0.0
        else:
            accel, brake = 0.0, 0.38
    else:
        strong_brake_error = -20.0 if late_brake_ready else -18.0
        mild_brake_error = -7.0 if late_brake_ready else -5.0
        if speed_error < strong_brake_error:
            accel = 0.0
            brake_offset = 10.0 if late_brake_ready else 8.0
            brake = clamp(
                (-speed_error - brake_offset) / 34.0,
                0.12,
                0.82,
            )
        elif speed_error < mild_brake_error:
            accel = 0.0
            brake = clamp(
                (-speed_error + mild_brake_error) / 42.0,
                0.0,
                0.36,
            )
        elif late_brake_ready and speed_error < -5.0:
            accel = 0.0
            brake = 0.0
        elif speed_error < 2.0 and (abs(steer) > 0.42 or pursuit_error > 0.34):
            accel = 0.0
            brake = clamp((2.0 - speed_error) / 34.0, 0.0, 0.14)
        else:
            brake = 0.0
            accel = clamp(0.46 + speed_error / 26.0, 0.20, 1.0)

        accel = min(
            accel,
            raceline_corner_throttle_cap(
                steer,
                track_position,
                pursuit_error,
                lateral_speed,
                angle,
            ),
        )

        transient_wobble = (
            (abs(lateral_speed) > 7.5 or abs(angle) > 0.20)
            and abs(lateral_speed) < 8.8
            and speed_error > 8.0
            and pursuit_error < 0.20
            and abs(track_position) < 0.76
            and abs(angle) < 0.26
        )
        if (abs(lateral_speed) > 7.5 or abs(angle) > 0.20) and not transient_wobble:
            accel = min(accel, 0.12)
            brake = max(brake, 0.12)
        elif transient_wobble:
            accel = min(accel, 0.36)
        if abs(lateral_speed) > 11.0:
            accel = 0.0
            brake = max(brake, 0.26)

    if previous_accel is not None and previous_brake is not None:
        accel_rise = 0.22
        if late_brake_ready and speed_error > 12.0:
            accel_rise = 0.34
        elif stable_on_line and speed_error > 6.0:
            accel_rise = 0.28
        accel = previous_accel + clamp(
            accel - previous_accel,
            -0.42,
            accel_rise,
        )
        brake_release = 0.30
        if speed_error > 6.0 and stable_on_line:
            brake_release = 0.80
        elif speed_error > 0.0 and stable_on_line:
            brake_release = 0.42
        brake = previous_brake + clamp(brake - previous_brake, -brake_release, 0.34)
        if brake > 0.02:
            accel = 0.0

    return clamp(accel, 0.0, 1.0), clamp(brake, 0.0, 1.0)


def apply_raceline_momentum_coast(
    accel,
    brake,
    *,
    focused_sweeper,
    speed,
    target_speed,
    pursuit_error,
    track_position,
    lateral_speed,
    angle,
    sensor_cap,
    control_state,
):
    """Trade mild braking for a lift in the validated long sweeper."""

    safe_control_state = control_state not in (
        RacingControlState.EDGE_GUARD,
        RacingControlState.RECOVERY,
    )
    focused_safe = (
        focused_sweeper
        and safe_control_state
        and sensor_cap > 150.0
        and speed < 145.0
        and speed - target_speed < 28.0
        and pursuit_error < 0.58
        and abs(track_position) < 0.78
        and abs(lateral_speed) < 5.8
        and abs(angle) < 0.16
    )
    if focused_safe and brake > 0.02:
        return 0.0, 0.0
    return accel, brake


def apply_edge_guard_crawl_assist(
    accel,
    brake,
    *,
    speed,
    target_speed,
    track_position,
    angle,
    front_sensor,
    min_track_sensor,
    control_state,
):
    """Let an on-track EDGE_GUARD car crawl out instead of sitting on brake."""

    if (
        control_state == RacingControlState.EDGE_GUARD
        and speed < 12.0
        and target_speed > speed + 10.0
        and abs(track_position) < 0.86
        and abs(angle) < 0.55
        and front_sensor > 3.0
        and min_track_sensor > 1.0
    ):
        return max(accel, 0.18), 0.0
    return accel, brake


def guard_racing_line_target(
    *,
    track_position,
    target_position,
    speed,
    track_sensors,
    line_complexity,
    line_blend=1.0,
    lateral_speed=0.0,
    angle=0.0,
):
    """Keep the saved racing line drivable when the live road edge disagrees.

    The map line is still the ideal. This guard only vetoes it when the car is
    already close to an edge or the road sensors say the chosen lane is about to
    disappear. That lets the agent use the track width aggressively without
    obeying a target that is physically unrecoverable in the current state.
    """

    guarded = target_position
    edge_abs = abs(track_position)
    front = track_sensors[9]
    minimum_sensor = min(track_sensors) if track_sensors else front
    best_index = get_best_sensor(track_sensors)
    best_angle = TRACK_SENSOR_ANGLES[best_index]
    best_distance = track_sensors[best_index]

    if edge_abs > HARD_TRACK_LIMIT:
        guarded = clamp(guarded, -0.34, 0.34)
    elif edge_abs > SAFE_TRACK_LIMIT:
        guarded = clamp(guarded, -0.62, 0.62)

    # Do not keep asking the car to move farther toward the same edge once it is
    # already using most of the track. Hold a small leeway, then recover inward.
    inward_margin = 0.10 + 0.12 * clamp(
        (edge_abs - SAFE_TRACK_LIMIT) / max(1.0 - SAFE_TRACK_LIMIT, 0.01),
        0.0,
        1.0,
    )
    if track_position < -SAFE_TRACK_LIMIT:
        guarded = max(guarded, track_position + inward_margin)
    elif track_position > SAFE_TRACK_LIMIT:
        guarded = min(guarded, track_position - inward_margin)

    # A normal corner often has a short straight-ahead sensor; that must not
    # veto the map line. Only bias away from the line when the car is already
    # near/off the edge or the road immediately ahead has effectively vanished.
    off_track = front < 0.0 or minimum_sensor < 0.0 or edge_abs > 1.02
    immediate_wall = front < 7.0 and speed > 18.0
    edge_escape = (
        edge_abs > SAFE_TRACK_LIMIT
        and front < 34.0
        and best_distance > max(front * 1.55, front + 12.0)
    )
    outward_target = (
        abs(guarded) > abs(track_position) + 0.08
        and guarded * (track_position if abs(track_position) > 0.04 else guarded) > 0.0
    )
    unsettled_join = (
        abs(lateral_speed) > 4.0
        or abs(angle) > 0.12
    )
    progressive_edge_approach = (
        front < 34.0
        and outward_target
        and speed > 55.0
        and (unsettled_join or front < 12.0)
    )
    if line_blend > 0.18 and (
        off_track
        or immediate_wall
        or edge_escape
        or progressive_edge_approach
    ):
        sensor_target = clamp(best_angle / 45.0, -0.55, 0.55)
        sensor_weight = clamp((34.0 - front) / 32.0, 0.0, 0.40)
        if progressive_edge_approach and not off_track:
            direction = 1.0 if (track_position or guarded) > 0.0 else -1.0
            inward_step = 0.08 + 0.08 * clamp((34.0 - front) / 28.0, 0.0, 1.0)
            sensor_target = track_position - direction * inward_step
            sensor_weight = max(
                sensor_weight,
                clamp((34.0 - front) / 52.0, 0.08, 0.34),
            )
            if unsettled_join:
                sensor_weight = min(0.42, sensor_weight + 0.06)
        if speed > 125.0:
            sensor_weight = min(0.50, sensor_weight + 0.08)
        if line_complexity > 0.45:
            sensor_weight = min(0.55, sensor_weight + 0.07)
        if off_track:
            sensor_weight = max(sensor_weight, 0.62)
        guarded = guarded * (1.0 - sensor_weight) + sensor_target * sensor_weight

    return clamp(guarded, -0.76, 0.76)


class MapAwareAgent:
    name = "Map-Aware Racing-Line Agent"
    agent_type = "map_aware"
    version = "4.2"
    seed = None
    uses_full_control = True
    max_steps = 150000
    target_laps = 1
    line_merge_distance = 28.0

    def __init__(
        self,
        racing_line_path=DEFAULT_RACING_LINE_PATH,
        telemetry_path=DEFAULT_TELEMETRY_PATH,
    ):
        self.racing_line_path = Path(racing_line_path)
        self.telemetry_path = Path(telemetry_path)
        if not self.racing_line_path.is_file():
            raise FileNotFoundError(
                "Saved racing line not found. Generate it first: "
                f"{self.racing_line_path}"
            )
        self.racing_line = RacingLine.load(self.racing_line_path)
        self._telemetry_file = None
        self._telemetry_writer = None
        self.reset()

    @property
    def config(self):
        metadata = self.racing_line.payload
        return {
            "track": self.racing_line.track_name,
            "racing_line_file": str(self.racing_line_path.relative_to(PROJECT_ROOT)),
            "racing_line_source_sha256": metadata["source_sha256"],
            "racing_line_sha256": hashlib.sha256(
                self.racing_line_path.read_bytes()
            ).hexdigest(),
            "optimizer_format": metadata["format_version"],
            "target_laps": self.target_laps,
            "max_steps": self.max_steps,
            "line_merge_distance": self.line_merge_distance,
            "control_profile": "raceline_pursuit_with_late_brake_fast_exit",
            "speed_profile": "speedway_confidence_curvature_buckets",
        }

    def reset(self):
        self.close()
        self.previous_steer = 0.0
        self.previous_accel = None
        self.previous_brake = None
        self.previous_target_speed = None
        self.previous_target_position = None
        self.previous_pursuit_error = None
        self.line_confidence_debt = 0.0
        self.start_distance = None
        self.start_distance_raced = None
        self.start_track_position = None
        self.control_state = RacingControlState.JOIN_RACELINE
        self.state_steps = 0
        self.launch_stuck_steps = 0
        self.previous_front = None
        self.step = 0
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry_file = self.telemetry_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._telemetry_writer = csv.writer(self._telemetry_file)
        self._telemetry_writer.writerow(TELEMETRY_COLUMNS)
        self._telemetry_file.flush()

    def _act_raceline_pursuit(self, telemetry):
        """Raceline-first controller.

        This is intentionally simpler than the older layered controller:
        the saved map line defines where the car should be, a short lookahead
        point defines where steering should aim, and sensors only veto unsafe
        speed or recovery situations.
        """

        speed = float(telemetry["speedX"])
        lateral_speed = float(telemetry["speedY"])
        angle = float(telemetry["angle"])
        track_position = float(telemetry["trackPos"])
        distance = float(telemetry.get("distFromStart", 0.0))
        track_sensors = get_track_sensors(telemetry)
        front_sensor = track_sensors[9]
        min_track_sensor = min(track_sensors) if track_sensors else front_sensor
        best_sensor_index = get_best_sensor(track_sensors)
        best_sensor_angle = TRACK_SENSOR_ANGLES[best_sensor_index]
        raw_sensor_speed_cap = sensor_speed_cap(
            track_sensors,
            speed,
            self.previous_front,
        )

        if self.start_distance is None:
            self.start_distance = distance
            self.start_distance_raced = float(telemetry.get("distRaced", 0.0))
            self.start_track_position = track_position

        if "distRaced" in telemetry:
            distance_travelled = max(
                0.0,
                float(telemetry["distRaced"]) - self.start_distance_raced,
            )
        else:
            distance_travelled = (
                distance - self.start_distance
            ) % self.racing_line.track_length

        merge_distance = 45.0 if distance_travelled < 120.0 else self.line_merge_distance
        merge_fraction = clamp(distance_travelled / merge_distance, 0.0, 1.0)
        line_blend = merge_fraction * merge_fraction * (3.0 - 2.0 * merge_fraction)

        target = self.racing_line.lookup(distance)
        first_lookahead = pursuit_lookahead_distance(speed)
        near = self.racing_line.lookup(distance + first_lookahead)
        far = self.racing_line.lookup(distance + first_lookahead * 2.0)
        line_complexity = estimate_line_complexity(target, near, far)
        lookahead_distance = pursuit_lookahead_distance(speed, line_complexity)
        if abs(lookahead_distance - first_lookahead) > 0.5:
            near = self.racing_line.lookup(distance + lookahead_distance)
            far = self.racing_line.lookup(distance + lookahead_distance * 2.0)
            line_complexity = estimate_line_complexity(target, near, far)
        very_far = self.racing_line.lookup(distance + lookahead_distance * 3.25)

        raw_local_target = target["target_track_pos"]
        raw_pursuit_target = choose_pursuit_track_position(
            target,
            near,
            far,
            speed,
            line_complexity,
        )
        local_target = (
            self.start_track_position * (1.0 - line_blend)
            + raw_local_target * line_blend
        )
        pursuit_target = (
            self.start_track_position * (1.0 - line_blend)
            + raw_pursuit_target * line_blend
        )
        pre_finish_launch = (
            self.start_distance is not None
            and self.start_distance > self.racing_line.track_length - 180.0
            and distance_travelled < 108.0
        )
        if pre_finish_launch:
            # TORCS starts this practice setup just before the timing line.
            # Chasing the final-straight setup lane immediately made run 59
            # slide into the outside wall before it ever reached turn 1.
            # Hold the spawn lane until the car is settled, then release the
            # normal map-aware pursuit after the line.
            launch_fraction = clamp(distance_travelled / 108.0, 0.0, 1.0)
            allowed_shift = 0.025 + 0.075 * launch_fraction
            local_target = clamp(
                local_target,
                self.start_track_position - allowed_shift,
                self.start_track_position + allowed_shift,
            )
            pursuit_target = clamp(
                pursuit_target,
                self.start_track_position - allowed_shift,
                self.start_track_position + allowed_shift,
            )

        sensor_cap = sensor_emergency_speed_cap(
            track_sensors,
            track_position,
            speed,
            self.previous_front,
            lateral_speed=lateral_speed,
            angle=angle,
        )
        control_state = choose_raceline_control_state(
            track_position=track_position,
            local_target=local_target,
            pursuit_target=pursuit_target,
            speed=speed,
            lateral_speed=lateral_speed,
            angle=angle,
            front_sensor=front_sensor,
            min_track_sensor=min_track_sensor,
        )

        pursuit_error = track_position - pursuit_target
        line_error = track_position - local_target
        self.line_confidence_debt = update_raceline_confidence_debt(
            self.line_confidence_debt,
            self.previous_pursuit_error,
            pursuit_error,
            lateral_speed,
            angle,
            track_position,
        )
        target_speed = calculate_raceline_target_speed(
            speed=speed,
            local=target,
            near=near,
            far=far,
            very_far=very_far,
            track_position=track_position,
            pursuit_target=pursuit_target,
            lateral_speed=lateral_speed,
            angle=angle,
            sensor_cap=sensor_cap,
            line_complexity=line_complexity,
        )
        focused_sweeper = optimizer_sweeper_speed_cap(
            target,
            near,
            far,
            pursuit_target=pursuit_target,
            line_complexity=line_complexity,
        ) < 216.0
        if self.line_confidence_debt > 0.05:
            target_speed = min(
                target_speed,
                confidence_speed_cap(self.line_confidence_debt),
            )
        if control_state == RacingControlState.EDGE_GUARD:
            if front_sensor < 18.0:
                target_speed = min(target_speed, 78.0)
            elif track_position * lateral_speed > 1.0 or track_position * angle < -0.045:
                target_speed = min(target_speed, 104.0)
        desired_heading = (
            target["heading_offset"] * 0.30
            + near["heading_offset"] * 0.50
            + far["heading_offset"] * 0.20
        ) * line_blend
        target_curvature = (
            target["curvature"] * 0.25
            + near["curvature"] * 0.55
            + far["curvature"] * 0.20
        ) * line_blend
        steer = calculate_raceline_pursuit_steer(
            speed=speed,
            track_position=track_position,
            pursuit_target=pursuit_target,
            heading_offset=desired_heading,
            curvature=target_curvature,
            lateral_speed=lateral_speed,
            angle=angle,
            previous_steer=self.previous_steer,
            control_state=control_state,
        )

        if line_blend < 0.20 and speed < 35.0:
            steer = limit_change(self.previous_steer, steer, 0.035)
            steer = clamp(steer, -0.20, 0.20)
        if pre_finish_launch:
            launch_steer_limit = 0.12 + 0.10 * clamp(distance_travelled / 108.0, 0.0, 1.0)
            steer = clamp(steer, -launch_steer_limit, launch_steer_limit)
            steer = limit_change(
                self.previous_steer,
                steer,
                0.045 if speed < 55.0 else 0.060,
            )

        accel, brake = calculate_raceline_pedals(
            speed=speed,
            target_speed=target_speed,
            steer=steer,
            pursuit_error=abs(pursuit_error),
            track_position=track_position,
            lateral_speed=lateral_speed,
            angle=angle,
            previous_accel=self.previous_accel,
            previous_brake=self.previous_brake,
            control_state=control_state,
        )
        accel = apply_raceline_traction_control(
            telemetry.get("wheelSpinVel"),
            accel,
        )
        requested_brake = brake
        accel, brake = apply_raceline_momentum_coast(
            accel,
            brake,
            focused_sweeper=focused_sweeper,
            speed=speed,
            target_speed=target_speed,
            pursuit_error=abs(pursuit_error),
            track_position=track_position,
            lateral_speed=lateral_speed,
            angle=angle,
            sensor_cap=sensor_cap,
            control_state=control_state,
        )
        momentum_coasting = requested_brake > 0.02 and brake <= 0.02
        accel, brake = apply_edge_guard_crawl_assist(
            accel,
            brake,
            speed=speed,
            target_speed=target_speed,
            track_position=track_position,
            angle=angle,
            front_sensor=front_sensor,
            min_track_sensor=min_track_sensor,
            control_state=control_state,
        )

        is_launching = line_blend < 0.05 and speed < 18.0
        if is_launching:
            accel = max(accel, 0.85)
            brake = 0.0

        control_phase = choose_dynamic_control_phase(
            speed,
            target_speed,
            target_curvature,
            front_sensor,
        )
        if brake > 0.05:
            control_phase = "brake"
        elif momentum_coasting:
            control_phase = "coast"
        elif is_launching:
            control_phase = "launch"

        action = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": shift_gears(speed),
            "terminate": False,
            "termination_reason": "",
        }

        if (
            distance_travelled < 2.0
            and abs(speed) < 1.0
            and float(telemetry.get("curLapTime", 0.0)) >= 0.0
        ):
            self.launch_stuck_steps += 1
        else:
            self.launch_stuck_steps = 0

        if self.launch_stuck_steps > 250:
            action["terminate"] = True
            action["termination_reason"] = "launch_stuck"

        corridor_half_width = get_racing_line_corridor(
            speed,
            max(abs(target["curvature"]), abs(near["curvature"])),
            line_complexity,
        )
        self._write_telemetry(
            telemetry,
            action,
            control_state,
            raw_local_target,
            local_target,
            local_target,
            pursuit_target,
            pursuit_target,
            line_error,
            pursuit_error,
            corridor_half_width,
            line_complexity,
            target_speed,
            target,
            front_sensor,
            best_sensor_angle,
            raw_sensor_speed_cap,
            sensor_cap,
            lookahead_distance,
            line_blend,
            target.get("reference_throttle", 0.0),
            target.get("reference_brake", 0.0),
            target.get("reference_section", ""),
            control_phase,
        )

        self.previous_steer = steer
        self.previous_accel = accel
        self.previous_brake = brake
        self.previous_target_speed = target_speed
        self.previous_target_position = local_target
        self.previous_pursuit_error = pursuit_error
        self.previous_front = front_sensor
        if control_state == self.control_state:
            self.state_steps += 1
        else:
            self.state_steps = 0
        self.control_state = control_state
        self.step += 1
        return action

    def act(self, _observation, telemetry=None):
        if telemetry is None:
            raise ValueError("MapAwareAgent requires raw TORCS telemetry")

        return self._act_raceline_pursuit(telemetry)

        speed = float(telemetry["speedX"])
        track_position = float(telemetry["trackPos"])
        distance = float(telemetry.get("distFromStart", 0.0))
        track_sensors = get_track_sensors(telemetry)
        front_sensor = track_sensors[9]
        best_sensor_index = get_best_sensor(track_sensors)
        best_sensor_angle = TRACK_SENSOR_ANGLES[best_sensor_index]
        raw_sensor_speed_cap = sensor_speed_cap(
            track_sensors,
            speed,
            self.previous_front,
        )
        if self.start_distance is None:
            self.start_distance = distance
            self.start_distance_raced = float(telemetry.get("distRaced", 0.0))
            self.start_track_position = track_position
        if "distRaced" in telemetry:
            distance_travelled = max(
                0.0,
                float(telemetry["distRaced"]) - self.start_distance_raced,
            )
        else:
            distance_travelled = (
                distance - self.start_distance
            ) % self.racing_line.track_length
        merge_distance = self.line_merge_distance
        if distance_travelled < 120.0:
            merge_distance = 45.0
        merge_fraction = clamp(
            distance_travelled / merge_distance,
            0.0,
            1.0,
        )
        line_blend = merge_fraction * merge_fraction * (3.0 - 2.0 * merge_fraction)
        lookahead_distance = clamp(24.0 + speed * 0.16, 24.0, 56.0)
        target = self.racing_line.lookup(distance)
        lookahead = self.racing_line.lookup(distance + lookahead_distance)
        far_lookahead = self.racing_line.lookup(
            distance + lookahead_distance * 1.85
        )
        very_far_lookahead = self.racing_line.lookup(
            distance + lookahead_distance * 3.25
        )
        line_complexity = estimate_line_complexity(
            target,
            lookahead,
            far_lookahead,
        )
        lateral_speed = float(telemetry["speedY"])
        angle = float(telemetry["angle"])
        initial_line_error = abs(self.start_track_position - target["target_track_pos"])
        launch_authority = clamp(
            (distance_travelled - 6.0) / 32.0,
            0.0,
            1.0,
        ) * clamp(speed / 34.0, 0.0, 1.0)
        minimum_line_blend = clamp(
            (initial_line_error - 0.12) / 0.55,
            0.0,
            MAX_STARTUP_LINE_BLEND_FLOOR,
        ) * launch_authority
        line_blend = max(line_blend, minimum_line_blend)

        # Cross-track error mostly belongs to the line underneath the car, but
        # a small preview window helps linked bends become one flowing target
        # rather than a sequence of urgent left-right corrections.
        preview_position = preview_track_position(
            target,
            lookahead,
            far_lookahead,
            speed,
            line_complexity,
        )
        target_position = preview_position
        target_position = (
            self.start_track_position * (1.0 - line_blend)
            + target_position * line_blend
        )
        if distance_travelled < 70.0:
            launch_fraction = distance_travelled / 70.0
            maximum_launch_shift = 0.12 + 0.50 * launch_fraction
            target_position = clamp(
                target_position,
                self.start_track_position - maximum_launch_shift,
                self.start_track_position + maximum_launch_shift,
            )

        raw_racing_line_position = target_position
        control_state = choose_control_state(
            self.control_state,
            line_blend=line_blend,
            track_position=track_position,
            target_position=raw_racing_line_position,
            speed=speed,
            lateral_speed=lateral_speed,
            angle=angle,
            front_sensor=front_sensor,
            min_track_sensor=min(track_sensors) if track_sensors else front_sensor,
            raw_sensor_speed_cap=raw_sensor_speed_cap,
        )
        practical_target_position = apply_practical_target_limit(
            raw_racing_line_position,
            control_state=control_state,
            speed=speed,
            lateral_speed=lateral_speed,
            angle=angle,
            line_blend=line_blend,
        )
        target_position = practical_target_position

        # Smooth the target so the car does not twitch when the racing line changes.
        target_position = smooth_value(
            self.previous_target_position,
            target_position,
            alpha=0.36 - line_complexity * 0.08,
        )
        if self.previous_target_position is not None:
            max_target_delta = 0.080
            if speed > 150:
                max_target_delta = 0.056
            elif speed > 100:
                max_target_delta = 0.066
            max_target_delta *= 1.0 - line_complexity * 0.14
            target_position = limit_change(
                self.previous_target_position,
                target_position,
                max_target_delta,
            )
        target_position = apply_practical_target_limit(
            target_position,
            control_state=control_state,
            speed=speed,
            lateral_speed=lateral_speed,
            angle=angle,
            line_blend=line_blend,
        )
        practical_target_position = target_position

        target_position = guard_racing_line_target(
            track_position=track_position,
            target_position=target_position,
            speed=speed,
            track_sensors=track_sensors,
            line_complexity=line_complexity,
            line_blend=line_blend,
            lateral_speed=lateral_speed,
            angle=angle,
        )
        if self.previous_target_position is not None:
            guarded_delta = 0.060
            if line_complexity > 0.42 or speed > 120.0:
                guarded_delta = 0.050
            if (
                raw_sensor_speed_cap < 216.0
                and (front_sensor < 18.0 or abs(track_position) > 0.68)
            ):
                guarded_delta = min(guarded_delta, 0.044)
            if front_sensor < 0.0 or abs(track_position) > 0.95:
                guarded_delta = max(guarded_delta, 0.088)
            target_position = limit_change(
                self.previous_target_position,
                target_position,
                guarded_delta,
            )

        raw_line_error = track_position - target_position
        target_speed = calculate_dynamic_target_speed(
            speed=speed,
            target=target,
            lookahead=lookahead,
            far_lookahead=far_lookahead,
            very_far_lookahead=very_far_lookahead,
            live_sensor_speed_cap=216.0,
            line_complexity=line_complexity,
            line_error=raw_line_error,
        )
        target_speed = cap_speed_for_line_join(
            target_speed,
            track_position,
            preview_position,
        )
        line_join_error = abs(track_position - practical_target_position)
        target_speed = cap_speed_for_control_state(
            target_speed,
            control_state=control_state,
            line_join_error=line_join_error,
            line_error=raw_line_error,
            speed=speed,
            lateral_speed=lateral_speed,
            angle=angle,
            front_sensor=front_sensor,
        )
        safety_sensor_speed_cap = map_safety_speed_cap(
            track_sensors,
            speed,
            self.previous_front,
            track_position,
            raw_line_error,
            target_position=target_position,
            lateral_speed=lateral_speed,
            angle=angle,
        )
        target_speed = min(target_speed, safety_sensor_speed_cap)

        speed_alpha = 0.24
        if (
            self.previous_target_speed is not None
            and target_speed < self.previous_target_speed - 8.0
        ):
            speed_alpha = 0.42
        elif target_speed > speed + 20.0:
            speed_alpha = 0.28
        target_speed = smooth_value(
            self.previous_target_speed,
            target_speed,
            alpha=speed_alpha,
        )
        if (
            safety_sensor_speed_cap < 216.0
            and (front_sensor < 7.0 or abs(track_position) > 0.90)
        ):
            target_speed = min(target_speed, safety_sensor_speed_cap)

        corridor_half_width = get_racing_line_corridor(
            speed,
            max(abs(target["curvature"]), abs(lookahead["curvature"])),
            line_complexity,
        )
        effective_line_error = soften_line_error(
            raw_line_error,
            corridor_half_width,
        )
        steering_target_position = track_position - effective_line_error
        mode_line_error = max(
            0.0,
            abs(raw_line_error) - corridor_half_width * 0.65,
        )

        mode = choose_driving_mode(
            track_pos=track_position,
            angle=angle,
            speed_y=lateral_speed,
            line_error=mode_line_error,
        )
        if control_state == RacingControlState.RECOVERY:
            mode = DrivingMode.RECOVERY
        elif control_state == RacingControlState.EDGE_GUARD and mode == DrivingMode.ATTACK:
            mode = DrivingMode.LIMIT
        elif (
            control_state == RacingControlState.JOIN_RACELINE
            and abs(raw_line_error) > 0.24
            and mode == DrivingMode.ATTACK
        ):
            mode = DrivingMode.LIMIT

        desired_heading = (
            target["heading_offset"] * 0.45
            + lookahead["heading_offset"] * 0.40
            + far_lookahead["heading_offset"] * 0.15
        )

        # Use the lookahead curvature because it prepares the car for the corner.
        target_curvature = (
            target["curvature"] * 0.25
            + lookahead["curvature"] * 0.50
            + far_lookahead["curvature"] * 0.25
        )
        # TORCS can spawn close to the timing line, then immediately wrap from
        # distFromStart ~= track_length to 0. During the startup merge, the
        # lateral target is blended from the car's actual position; heading and
        # curvature need the same ramp-in or they can command a full racing-line
        # corner before the car has physically joined it.
        line_authority = line_blend * line_blend
        desired_heading *= line_authority
        target_curvature *= line_authority

        steer = calculate_path_tracking_steer(
            speed=speed,
            track_position=track_position,
            angle=angle,
            lateral_speed=lateral_speed,
            target_position=steering_target_position,
            heading_offset=desired_heading,
            curvature=target_curvature,
            previous_steer=self.previous_steer,
            control_state=control_state,
        )
        if (
            distance_travelled < 70.0
            and speed < 120.0
            and abs(track_position) < 0.62
        ):
            launch_fraction = distance_travelled / 70.0
            launch_steer_limit = 0.20 + 0.10 * launch_fraction
            steer = clamp(steer, -launch_steer_limit, launch_steer_limit)
            steer = limit_change(
                self.previous_steer,
                steer,
                0.035 if speed > 45.0 else 0.045,
            )
        if line_complexity > 0.42 and speed > 35.0:
            complex_limit = 0.72 if speed < 95.0 else 0.62
            if abs(track_position) > 0.62:
                complex_limit += 0.10
            steer = clamp(steer, -complex_limit, complex_limit)
            steer = limit_change(
                self.previous_steer,
                steer,
                0.048 if speed > 75.0 else 0.060,
            )

        control_phase = choose_dynamic_control_phase(
            speed,
            target_speed,
            target_curvature,
            front_sensor,
        )
        is_launching = line_blend < 0.05 and speed < 18.0
        if is_launching:
            control_phase = "launch"

        accel, brake, adjusted_target_speed = calculate_aggressive_speed_control(
            speed=speed,
            target_speed=target_speed,
            steer=steer,
            track_pos=track_position,
            angle=angle,
            speed_y=lateral_speed,
            line_error=mode_line_error,
            mode=mode,
            phase=control_phase,
            wheel_spin=telemetry.get("wheelSpinVel"),
        )
        if is_launching:
            accel = max(accel, 0.85)
            brake = 0.0
        else:
            accel, brake = self._smooth_pedals(
                accel,
                brake,
                line_complexity,
                mode,
            )
            if (
                abs(track_position) > 0.46
                and abs(raw_line_error) > 0.20
                and speed > 45.0
            ):
                accel = min(accel, 0.34)
            if (
                abs(lateral_speed) > 4.0
                and abs(angle) > 0.05
                and speed > 65.0
            ):
                accel = min(accel, 0.18)
            line_join_error = abs(track_position - preview_position)
            if raw_sensor_speed_cap < 216.0 and line_join_error > 0.34:
                accel = min(accel, 0.32)
                if front_sensor < 24.0 or line_join_error > 0.50:
                    accel = min(accel, 0.14)
            if abs(lateral_speed) > 7.5 and speed > 55.0:
                accel = min(accel, 0.06)
                if abs(angle) > 0.08 or abs(track_position) > 0.48:
                    brake = max(brake, 0.10)
            if abs(lateral_speed) > 12.0 and speed > 45.0:
                accel = 0.0
                brake = max(brake, 0.18)
            if abs(lateral_speed) > 16.0 and speed > 35.0:
                accel = 0.0
                brake = max(brake, 0.30)
            if (
                control_state == RacingControlState.RECOVERY
                and speed < 4.0
                and abs(track_position) > 0.92
                and front_sensor < 4.0
                and abs(angle) < 0.45
            ):
                accel = max(accel, 0.20)
                brake = 0.0
                steer = clamp(-track_position * 1.15, -0.90, 0.90)
            if distance_travelled < 45.0 and abs(lateral_speed) > 4.0:
                accel = min(accel, 0.08)
            if (
                speed < 30.0
                and brake < 0.05
                and target_speed > speed + 35.0
                and front_sensor > 24.0
            ):
                accel = max(accel, 0.42)

        action = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": shift_gears(speed),
            "terminate": False,
            "termination_reason": "",
        }
        if (
            distance_travelled < 2.0
            and abs(speed) < 1.0
            and float(telemetry.get("curLapTime", 0.0)) >= 0.0
        ):
            self.launch_stuck_steps += 1
        else:
            self.launch_stuck_steps = 0

        if self.launch_stuck_steps > 250:
            action["terminate"] = True
            action["termination_reason"] = "launch_stuck"

        self._write_telemetry(
            telemetry,
            action,
            control_state,
            raw_racing_line_position,
            practical_target_position,
            target_position,
            preview_position,
            steering_target_position,
            raw_line_error,
            effective_line_error,
            corridor_half_width,
            line_complexity,
            adjusted_target_speed,
            target,
            front_sensor,
            best_sensor_angle,
            raw_sensor_speed_cap,
            safety_sensor_speed_cap,
            lookahead_distance,
            line_blend,
            target.get("reference_throttle", 0.0),
            target.get("reference_brake", 0.0),
            target.get("reference_section", ""),
            control_phase,
        )
        self.previous_steer = steer
        self.previous_accel = accel
        self.previous_brake = brake
        self.previous_target_speed = target_speed
        self.previous_target_position = target_position
        self.previous_front = front_sensor
        if control_state == self.control_state:
            self.state_steps += 1
        else:
            self.state_steps = 0
        self.control_state = control_state
        self.step += 1
        return action

    def _smooth_pedals(self, accel, brake, line_complexity, mode):
        if self.previous_accel is None or self.previous_brake is None:
            return accel, brake

        if brake > 0.0:
            accel = 0.0
        elif self.previous_brake > 0.0 and line_complexity > 0.35:
            accel = min(accel, 0.34)

        accel_up = 0.18 if line_complexity > 0.42 else 0.25
        accel_down = 0.34
        if mode.value == "RECOVERY":
            accel_up = 0.12
            accel_down = 0.40
        accel = self.previous_accel + clamp(
            accel - self.previous_accel,
            -accel_down,
            accel_up,
        )

        brake_up = 0.30
        brake_down = 0.20 if line_complexity > 0.42 else 0.28
        brake = self.previous_brake + clamp(
            brake - self.previous_brake,
            -brake_down,
            brake_up,
        )
        if brake > 0.02:
            accel = 0.0
        return accel, brake

    def _write_telemetry(
        self,
        telemetry,
        action,
        control_state,
        raw_racing_line_position,
        practical_target_position,
        target_position,
        preview_position,
        steering_target_position,
        raw_line_error,
        effective_line_error,
        corridor_half_width,
        line_complexity,
        target_speed,
        target,
        front_sensor,
        best_sensor_angle,
        raw_sensor_speed_cap,
        safety_sensor_speed_cap,
        lookahead_distance,
        line_blend,
        reference_throttle,
        reference_brake,
        reference_section,
        control_phase,
    ):
        if self._telemetry_writer is None:
            return
        self._telemetry_writer.writerow(
            [
                self.step,
                telemetry.get("distFromStart", 0),
                telemetry.get("speedX", 0),
                telemetry.get("speedY", 0),
                telemetry.get("angle", 0),
                telemetry.get("trackPos", 0),
                control_state.value,
                raw_racing_line_position,
                practical_target_position,
                target_position,
                preview_position,
                steering_target_position,
                raw_line_error,
                effective_line_error,
                corridor_half_width,
                line_complexity,
                target_speed,
                target["curvature"],
                target["heading_offset"],
                target.get("speed_delta_30m", 0.0),
                target.get("turn_direction", 0.0),
                front_sensor,
                best_sensor_angle,
                raw_sensor_speed_cap,
                safety_sensor_speed_cap,
                *get_track_sensors(telemetry),
                lookahead_distance,
                line_blend,
                reference_throttle,
                reference_brake,
                reference_section,
                control_phase,
                action["steer"],
                action["accel"],
                action["brake"],
                action["gear"],
                telemetry.get("curLapTime", 0),
                telemetry.get("lastLapTime", 0),
            ]
        )

    def close(self):
        if self._telemetry_file is not None:
            self._telemetry_file.close()
        self._telemetry_file = None
        self._telemetry_writer = None
