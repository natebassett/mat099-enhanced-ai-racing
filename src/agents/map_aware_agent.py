import csv
import hashlib
from pathlib import Path

from racing_line import (
    RacingLine,
    choose_driving_mode,
    smooth_value,
    limit_change,
    calculate_aggressive_steering,
    calculate_aggressive_speed_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACING_LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "corkscrew.json"
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
INSIDE_CORRIDOR_CORRECTION_GAIN = 0.18
CORRIDOR_EDGE_CORRECTION_GAIN = 0.42
OUTSIDE_CORRIDOR_CORRECTION_GAIN = 1.08
MAX_STARTUP_LINE_BLEND_FLOOR = 0.65
SAFE_TRACK_LIMIT = 0.72
HARD_TRACK_LIMIT = 0.90
TELEMETRY_COLUMNS = [
    "step",
    "distFromStart",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
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
    "sensorSpeedCap",
    "lookaheadDistance",
    "lineBlend",
    "phase",
    "steer",
    "accel",
    "brake",
    "gear",
    "curLapTime",
    "lastLapTime",
]


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
        corridor = 0.115
    elif curvature_abs > 0.016:
        corridor = 0.145
    elif curvature_abs > 0.008:
        corridor = 0.175
    else:
        corridor = 0.215

    if speed > 170:
        corridor += 0.035
    elif speed > 125:
        corridor += 0.020
    elif speed < 70:
        corridor -= 0.020

    # Linked bends need a little more lateral freedom. If the target line
    # moves left-right-left quickly, snapping to the exact ideal point causes
    # oscillation and costs more time than a slightly wider, calmer corridor.
    corridor += 0.040 * line_complexity

    return clamp(corridor, 0.10, 0.25)


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
        lookahead["target_track_pos"] * 0.68
        + far_lookahead["target_track_pos"] * 0.32
    )
    preview_weight = clamp(0.18 + speed / 520.0 + line_complexity * 0.10, 0.18, 0.48)
    maximum_preview_shift = 0.17 - line_complexity * 0.055
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


def guard_racing_line_target(
    *,
    track_position,
    target_position,
    speed,
    track_sensors,
    line_complexity,
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
    best_index = get_best_sensor(track_sensors)
    best_angle = TRACK_SENSOR_ANGLES[best_index]
    best_distance = track_sensors[best_index]

    if edge_abs > HARD_TRACK_LIMIT:
        guarded = clamp(guarded, -0.22, 0.22)
    elif edge_abs > SAFE_TRACK_LIMIT:
        guarded = clamp(guarded, -0.42, 0.42)

    # Do not keep asking the car to move farther toward the same edge once it is
    # already using most of the track. Hold a small leeway, then recover inward.
    if track_position < -SAFE_TRACK_LIMIT:
        guarded = max(guarded, track_position + 0.20)
    elif track_position > SAFE_TRACK_LIMIT:
        guarded = min(guarded, track_position - 0.20)

    # If the forward road is short, bias toward the sensor with the most road.
    # This matters most in the Corkscrew/last hairpin where the generated line
    # may be valid geometrically but the car arrives rotated or late.
    if front < 52.0 or best_distance > front * 1.35:
        sensor_target = clamp(best_angle / 45.0, -0.55, 0.55)
        sensor_weight = clamp((70.0 - front) / 55.0, 0.0, 0.55)
        if speed > 125.0:
            sensor_weight = min(0.68, sensor_weight + 0.12)
        if line_complexity > 0.45:
            sensor_weight = min(0.74, sensor_weight + 0.12)
        guarded = guarded * (1.0 - sensor_weight) + sensor_target * sensor_weight

    return clamp(guarded, -0.62, 0.62)


class MapAwareAgent:
    name = "Map-Aware Racing-Line Agent"
    agent_type = "map_aware"
    version = "1.7"
    seed = None
    uses_full_control = True
    max_steps = 150000
    target_laps = 1
    line_merge_distance = 45.0

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
        }

    def reset(self):
        self.close()
        self.previous_steer = 0.0
        self.previous_accel = None
        self.previous_brake = None
        self.previous_target_speed = None
        self.previous_target_position = None
        self.start_distance = None
        self.start_distance_raced = None
        self.start_track_position = None
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

    def act(self, _observation, telemetry=None):
        if telemetry is None:
            raise ValueError("MapAwareAgent requires raw TORCS telemetry")

        speed = float(telemetry["speedX"])
        track_position = float(telemetry["trackPos"])
        distance = float(telemetry.get("distFromStart", 0.0))
        track_sensors = get_track_sensors(telemetry)
        front_sensor = track_sensors[9]
        best_sensor_index = get_best_sensor(track_sensors)
        best_sensor_angle = TRACK_SENSOR_ANGLES[best_sensor_index]
        live_sensor_speed_cap = sensor_speed_cap(
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
            merge_distance = 95.0
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
        line_complexity = estimate_line_complexity(
            target,
            lookahead,
            far_lookahead,
        )
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
        target_speed = min(
            target["target_speed_kmh"],
            lookahead["target_speed_kmh"] + 8.0,
            far_lookahead["target_speed_kmh"] + 15.0,
            live_sensor_speed_cap,
        )

        lateral_speed = float(telemetry["speedY"])
        angle = float(telemetry["angle"])

        # Smooth the target so the car does not twitch when the racing line changes.
        target_position = smooth_value(
            self.previous_target_position,
            target_position,
            alpha=0.22 - line_complexity * 0.06,
        )
        if self.previous_target_position is not None:
            max_target_delta = 0.026
            if speed > 150:
                max_target_delta = 0.018
            elif speed > 100:
                max_target_delta = 0.022
            max_target_delta *= 1.0 - line_complexity * 0.28
            target_position = limit_change(
                self.previous_target_position,
                target_position,
                max_target_delta,
            )

        target_position = guard_racing_line_target(
            track_position=track_position,
            target_position=target_position,
            speed=speed,
            track_sensors=track_sensors,
            line_complexity=line_complexity,
        )
        if self.previous_target_position is not None:
            guarded_delta = 0.030
            if line_complexity > 0.42 or speed > 120.0:
                guarded_delta = 0.022
            target_position = limit_change(
                self.previous_target_position,
                target_position,
                guarded_delta,
            )

        speed_alpha = 0.18
        if target["phase"].startswith("brake"):
            speed_alpha = 0.36
        elif target["phase"].startswith("accelerate"):
            speed_alpha = 0.26
        target_speed = smooth_value(
            self.previous_target_speed,
            target_speed,
            alpha=speed_alpha,
        )

        raw_line_error = track_position - target_position
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
        # The car spawns close to the Corkscrew timing line, then immediately
        # wraps from distFromStart ~= track_length to 0. During the startup
        # merge, the lateral target is blended from the car's actual position;
        # heading and curvature need the same ramp-in or they can command a
        # full racing-line corner before the car has physically joined it.
        line_authority = line_blend * line_blend
        desired_heading *= line_authority
        target_curvature *= line_authority

        steer = calculate_aggressive_steering(
            speed=speed,
            track_pos=track_position,
            angle=angle,
            speed_y=lateral_speed,
            target_track_pos=steering_target_position,
            heading_offset=desired_heading,
            curvature=target_curvature,
            previous_steer=self.previous_steer,
            mode=mode,
        )
        if line_complexity > 0.42 and speed > 35.0:
            complex_limit = 0.56 if speed < 95.0 else 0.46
            if abs(track_position) > 0.62:
                complex_limit += 0.10
            steer = clamp(steer, -complex_limit, complex_limit)
            steer = limit_change(
                self.previous_steer,
                steer,
                0.040 if speed > 75.0 else 0.052,
            )

        control_phase = target["phase"]
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
                accel = min(accel, 0.28)

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
            live_sensor_speed_cap,
            lookahead_distance,
            line_blend,
        )
        self.previous_steer = steer
        self.previous_accel = accel
        self.previous_brake = brake
        self.previous_target_speed = target_speed
        self.previous_target_position = target_position
        self.previous_front = front_sensor
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
        live_sensor_speed_cap,
        lookahead_distance,
        line_blend,
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
                live_sensor_speed_cap,
                lookahead_distance,
                line_blend,
                target["phase"],
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
