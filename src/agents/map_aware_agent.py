import csv
import hashlib
from pathlib import Path

from racing_line import (
    RacingLine,
    choose_driving_mode,
    smooth_value,
    calculate_aggressive_steering,
    calculate_aggressive_speed_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACING_LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "corkscrew.json"
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "data" / "map_aware_telemetry.csv"

GEAR_SPEEDS = [0, 45, 85, 125, 165, 205]
INSIDE_CORRIDOR_CORRECTION_GAIN = 0.18
CORRIDOR_EDGE_CORRECTION_GAIN = 0.42
OUTSIDE_CORRIDOR_CORRECTION_GAIN = 1.08
MAX_STARTUP_LINE_BLEND_FLOOR = 0.65
TELEMETRY_COLUMNS = [
    "step",
    "distFromStart",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
    "targetTrackPos",
    "steeringTrackPos",
    "lineError",
    "effectiveLineError",
    "corridorHalfWidth",
    "targetSpeed",
    "targetCurvature",
    "headingOffset",
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


def get_racing_line_corridor(speed, curvature):
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


class MapAwareAgent:
    name = "Map-Aware Racing-Line Agent"
    agent_type = "map_aware"
    version = "1.5"
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
        self.previous_target_speed = None
        self.previous_target_position = None
        self.start_distance = None
        self.start_distance_raced = None
        self.start_track_position = None
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
        merge_fraction = clamp(
            distance_travelled / self.line_merge_distance,
            0.0,
            1.0,
        )
        line_blend = merge_fraction * merge_fraction * (3.0 - 2.0 * merge_fraction)
        lookahead_distance = clamp(22.0 + speed * 0.15, 22.0, 46.0)
        target = self.racing_line.lookup(distance)
        lookahead = self.racing_line.lookup(distance + lookahead_distance)
        initial_line_error = abs(self.start_track_position - target["target_track_pos"])
        minimum_line_blend = clamp(
            (initial_line_error - 0.12) / 0.55,
            0.0,
            MAX_STARTUP_LINE_BLEND_FLOOR,
        )
        line_blend = max(line_blend, minimum_line_blend)

        # Cross-track error belongs to the line underneath the car. Using a
        # future lateral position here made the controller cut toward the exit
        # before reaching the apex and then reverse its steering to recover.
        target_position = target["target_track_pos"]
        target_position = (
            self.start_track_position * (1.0 - line_blend)
            + target_position * line_blend
        )
        target_speed = min(
            target["target_speed_kmh"],
            lookahead["target_speed_kmh"] + 4.0,
        )

        lateral_speed = float(telemetry["speedY"])
        angle = float(telemetry["angle"])

        # Smooth the target so the car does not twitch when the racing line changes.
        target_position = smooth_value(
            self.previous_target_position,
            target_position,
            alpha=0.22,
        )

        target_speed = smooth_value(
            self.previous_target_speed,
            target_speed,
            alpha=0.18,
        )

        raw_line_error = track_position - target_position
        corridor_half_width = get_racing_line_corridor(
            speed,
            max(abs(target["curvature"]), abs(lookahead["curvature"])),
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
            target["heading_offset"] * 0.60
            + lookahead["heading_offset"] * 0.40
        )

        # Use the lookahead curvature because it prepares the car for the corner.
        target_curvature = (
            target["curvature"] * 0.35
            + lookahead["curvature"] * 0.65
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

        accel, brake, adjusted_target_speed = calculate_aggressive_speed_control(
            speed=speed,
            target_speed=target_speed,
            steer=steer,
            track_pos=track_position,
            angle=angle,
            speed_y=lateral_speed,
            line_error=mode_line_error,
            mode=mode,
            wheel_spin=telemetry.get("wheelSpinVel"),
        )

        action = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": shift_gears(speed),
            "terminate": False,
            "termination_reason": "",
        }
        self._write_telemetry(
            telemetry,
            action,
            target_position,
            steering_target_position,
            raw_line_error,
            effective_line_error,
            corridor_half_width,
            adjusted_target_speed,
            lookahead,
            lookahead_distance,
            line_blend,
        )
        self.previous_steer = steer
        self.previous_target_speed = target_speed
        self.previous_target_position = target_position
        self.step += 1
        return action

    def _write_telemetry(
        self,
        telemetry,
        action,
        target_position,
        steering_target_position,
        raw_line_error,
        effective_line_error,
        corridor_half_width,
        target_speed,
        target,
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
                steering_target_position,
                raw_line_error,
                effective_line_error,
                corridor_half_width,
                target_speed,
                target["curvature"],
                target["heading_offset"],
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
