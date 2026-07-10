import math
from enum import Enum


class DrivingMode(str, Enum):
    ATTACK = "ATTACK"
    LIMIT = "LIMIT"
    RECOVERY = "RECOVERY"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def limit_change(previous, target, max_delta):
    return previous + clamp(target - previous, -max_delta, max_delta)


def smooth_value(previous, target, alpha):
    if previous is None:
        return target
    return previous * (1.0 - alpha) + target * alpha


def choose_driving_mode(track_pos, angle, speed_y, line_error):
    abs_track = abs(track_pos)
    abs_angle = abs(angle)
    abs_lateral = abs(speed_y)
    abs_line_error = abs(line_error)
    drifting_outward = track_pos * speed_y > 2.4
    pointing_outward = track_pos * angle < -0.040

    if (
        abs_track > 0.96
        or abs_angle > 0.60
        or abs_lateral > 15.0
        or (abs_angle > 0.42 and abs_lateral > 9.0)
        or (abs_track > 0.70 and drifting_outward)
        or (abs_track > 0.52 and pointing_outward)
    ):
        return DrivingMode.RECOVERY

    if (
        abs_track > 0.82
        or abs_angle > 0.42
        or abs_lateral > 9.0
        or abs_line_error > 0.55
        or (abs_track > 0.45 and drifting_outward)
        or (abs_track > 0.28 and pointing_outward)
    ):
        return DrivingMode.LIMIT

    return DrivingMode.ATTACK


def steering_limit(speed, track_pos, mode):
    if mode == DrivingMode.RECOVERY:
        return 0.90

    if mode == DrivingMode.LIMIT:
        if speed < 100:
            return 0.78
        if speed < 150:
            return 0.62
        return 0.48

    # ATTACK mode
    if abs(track_pos) > 0.78:
        return 0.68
    if speed < 90:
        return 0.82
    if speed < 140:
        return 0.66
    if speed < 180:
        return 0.52
    return 0.42


def calculate_aggressive_steering(
    *,
    speed,
    track_pos,
    angle,
    speed_y,
    target_track_pos,
    heading_offset,
    curvature,
    previous_steer,
    mode,
):
    line_error = track_pos - target_track_pos
    pointing_outward = abs(track_pos) > 0.26 and track_pos * angle < -0.035
    drifting_outward = abs(track_pos) > 0.36 and track_pos * speed_y > 1.0
    edge_risk = pointing_outward or drifting_outward

    if mode == DrivingMode.RECOVERY:
        # Save the car first.
        if edge_risk:
            raw_steer = (
                -track_pos * 1.35
                - angle * 7.0 / math.pi
                - speed_y * 0.045
            )
        else:
            raw_steer = angle * 7.5 / math.pi - track_pos * 1.10 - speed_y * 0.035

    elif mode == DrivingMode.LIMIT:
        # Still follow the line, but prioritise stability.
        heading_component = (angle + heading_offset) * 7.4 / math.pi
        line_component = -line_error * 0.90
        lateral_component = -speed_y * 0.030
        curvature_component = curvature * 2.2

        if abs(line_error) > 0.14:
            heading_component *= 0.55
        base_steer = heading_component + line_component + lateral_component
        if base_steer * curvature_component < 0.0 and abs(line_error) > 0.08:
            curvature_component *= 0.15

        raw_steer = base_steer + curvature_component
        if edge_risk:
            raw_steer = raw_steer * 0.45 - track_pos * 0.80

    else:
        # ATTACK: commit harder to the racing line.
        heading_component = (angle + heading_offset) * 7.8 / math.pi
        line_component = -line_error * 1.06
        lateral_component = -speed_y * 0.024
        curvature_component = curvature * 2.7

        if abs(line_error) > 0.14:
            heading_component *= 0.55
        base_steer = heading_component + line_component + lateral_component
        if base_steer * curvature_component < 0.0 and abs(line_error) > 0.08:
            curvature_component *= 0.15

        raw_steer = base_steer + curvature_component
        if edge_risk:
            raw_steer = raw_steer * 0.65 - track_pos * 0.55

    limit = steering_limit(speed, track_pos, mode)
    raw_steer = clamp(raw_steer, -limit, limit)

    # If the car is already steering one way, do not instantly ask for a large
    # opposite lock unless it is a genuine low-speed save. This is the main
    # oscillation killer: the racing line can move side to side, but the car's
    # tyres cannot teleport to that next instruction.
    if (
        previous_steer * raw_steer < 0.0
        and abs(previous_steer) > 0.22
        and abs(raw_steer) > 0.22
        and speed > 55.0
    ):
        raw_steer *= 0.55

    # Aggressive but not twitchy.
    if speed > 170:
        max_delta = 0.032
        smooth_alpha = 0.18
    elif speed > 120:
        max_delta = 0.042
        smooth_alpha = 0.22
    else:
        max_delta = 0.060
        smooth_alpha = 0.30

    if mode == DrivingMode.RECOVERY:
        max_delta *= 1.90
        smooth_alpha = 0.46

    smoothed = previous_steer * (1.0 - smooth_alpha) + raw_steer * smooth_alpha
    steer = limit_change(previous_steer, smoothed, max_delta)

    return clamp(steer, -limit, limit)


def calculate_aggressive_speed_control(
    *,
    speed,
    target_speed,
    steer,
    track_pos,
    angle,
    speed_y,
    line_error,
    mode,
    phase="",
    wheel_spin=None,
):
    adjusted_target = target_speed
    abs_line_error = abs(line_error)
    drifting_outward = track_pos * speed_y > 1.2
    is_braking_phase = phase.startswith("brake")
    is_accel_phase = phase.startswith("accelerate") or phase == "full_throttle"
    is_turn_phase = "turn" in phase

    if mode == DrivingMode.ATTACK:
        adjusted_target *= 1.05

    elif mode == DrivingMode.LIMIT:
        adjusted_target *= 0.88

    else:
        adjusted_target *= 0.62

    # Map-aware phase control, generated from the racing-line speed profile.
    # This mirrors the reference style: brake before turn-in, coast/rotate
    # around the apex, accelerate on exit, and full throttle on clear straights.
    if is_braking_phase:
        adjusted_target *= 0.92
    elif is_accel_phase and mode == DrivingMode.ATTACK:
        adjusted_target *= 1.05
    elif is_turn_phase and not is_accel_phase:
        adjusted_target *= 0.97

    # Dynamic guardrails. These only bite when the car is actually in trouble.
    if abs_line_error > 0.42:
        adjusted_target *= 0.90
    if abs_line_error > 0.62:
        adjusted_target *= 0.82
    if abs(track_pos) > 0.82:
        adjusted_target *= 0.78
    if abs(angle) > 0.45:
        adjusted_target *= 0.84
    if abs(speed_y) > 10.0:
        adjusted_target *= 0.82
    if abs(speed_y) > 14.0:
        adjusted_target *= 0.70
    if abs(speed_y) > 18.0:
        adjusted_target *= 0.58
    if abs(track_pos) > 0.45 and drifting_outward:
        adjusted_target *= 0.82
    if abs(track_pos) > 0.62 and drifting_outward:
        adjusted_target *= 0.72

    # Steering-based throttle discipline.
    if abs(steer) > 0.68:
        adjusted_target *= 0.78
    elif abs(steer) > 0.52:
        adjusted_target *= 0.88
    elif abs(steer) > 0.36:
        adjusted_target *= 0.94

    speed_error = adjusted_target - speed

    if mode == DrivingMode.RECOVERY:
        if speed < 8.0 and abs(track_pos) < 0.95:
            # Do not sit on the brake at full lock. At near-zero speed the
            # agent needs a controlled push to rotate and rejoin the line.
            accel = 0.22 if abs(angle) > 0.60 else 0.34
            brake = 0.0
        elif speed < 18.0 and abs(track_pos) < 0.90 and abs(speed_y) < 8.0:
            accel = 0.18
            brake = 0.0
        else:
            accel = (
                0.06
                if abs(track_pos) < 0.85
                and abs(speed_y) < 10.0
                and abs(angle) < 0.50
                else 0.0
            )
            if abs(speed_y) > 18.0 or abs(angle) > 0.65:
                brake = 0.42
            elif speed > adjusted_target:
                brake = 0.35
            else:
                brake = 0.12
    elif is_braking_phase and speed > 40.0:
        accel = 0.0
        brake = clamp((6.0 - speed_error) / 22.0, 0.10, 0.62)
    elif abs(speed_y) > 18.0 and speed > 35.0:
        accel = 0.0
        brake = 0.36
    elif abs(speed_y) > 14.0 and speed > 45.0:
        accel = 0.0
        brake = 0.24
    elif speed_error < -4.0:
        accel = 0.0
        brake = clamp((-speed_error - 2.0) / 26.0, 0.0, 1.0)
    else:
        brake = 0.0
        accel = clamp(0.46 + speed_error / 20.0, 0.20, 1.0)

    if wheel_spin is not None and len(wheel_spin) >= 4:
        slip = wheel_spin[2] + wheel_spin[3] - wheel_spin[0] - wheel_spin[1]
        if slip > 3.2:
            accel = max(0.0, accel - 0.12)

    if abs(angle) > 0.38 or abs(speed_y) > 10.0:
        accel = min(accel, 0.12)
    elif is_braking_phase:
        accel = 0.0
    elif abs(track_pos) > 0.45 and drifting_outward:
        accel = min(accel, 0.18)
    elif is_turn_phase and not is_accel_phase:
        accel = min(accel, 0.28)
    elif abs(steer) > 0.52:
        accel = min(accel, 0.18)
    elif abs(steer) > 0.38:
        accel = min(accel, 0.34)
    elif abs(steer) > 0.28:
        accel = min(accel, 0.55)

    return accel, brake, adjusted_target
