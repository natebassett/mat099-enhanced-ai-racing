import csv
import hashlib
import math
from pathlib import Path

from racing_line import RacingLine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACING_LINE_PATH = PROJECT_ROOT / "data" / "racing_lines" / "corkscrew.json"
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "data" / "map_aware_telemetry.csv"

GEAR_SPEEDS = [0, 45, 85, 125, 165, 205]
TELEMETRY_COLUMNS = [
    "step",
    "distFromStart",
    "speedX",
    "speedY",
    "angle",
    "trackPos",
    "targetTrackPos",
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


def steering_limit(speed, track_position):
    if abs(track_position) > 0.78:
        return 0.82
    if speed < 70:
        return 0.78
    if speed < 110:
        return 0.66
    if speed < 150:
        return 0.50
    return 0.36


def shift_gears(speed):
    gear = 1
    for index, threshold in enumerate(GEAR_SPEEDS):
        if speed > threshold:
            gear = index + 1
    return int(clamp(gear, 1, 6))


class MapAwareAgent:
    name = "Map-Aware Racing-Line Agent"
    agent_type = "map_aware"
    version = "1.1"
    seed = None
    uses_full_control = True
    max_steps = 150000
    target_laps = 1
    line_merge_distance = 140.0

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
        lookahead_distance = clamp(12.0 + speed * 0.12, 12.0, 38.0)
        target = self.racing_line.lookup(distance)
        lookahead = self.racing_line.lookup(distance + lookahead_distance)

        target_position = (
            target["target_track_pos"] * 0.35
            + lookahead["target_track_pos"] * 0.65
        )
        target_position = (
            self.start_track_position * (1.0 - line_blend)
            + target_position * line_blend
        )
        target_speed = min(
            target["target_speed_kmh"],
            lookahead["target_speed_kmh"] + 4.0,
        )
        position_error = track_position - target_position
        heading_error = float(telemetry["angle"]) + target["heading_offset"]
        lateral_speed = float(telemetry["speedY"])

        stability_mode = abs(lateral_speed) > 12.0 or abs(telemetry["angle"]) > 0.70
        if stability_mode:
            raw_steer = (
                float(telemetry["angle"]) * 7.0 / math.pi
                - track_position * 0.85
                - lateral_speed * 0.030
            )
        else:
            raw_steer = (
                heading_error * 10.0 / math.pi
                - position_error * 0.72
                + lookahead["curvature"] * 6.0
                - lateral_speed * 0.018
            )

        limit = steering_limit(speed, track_position)
        raw_steer = clamp(raw_steer, -limit, limit)
        smoothed_steer = 0.82 * self.previous_steer + 0.18 * raw_steer
        maximum_change = 0.075 if abs(track_position) > 0.72 else 0.055
        steer = self.previous_steer + clamp(
            smoothed_steer - self.previous_steer,
            -maximum_change,
            maximum_change,
        )
        steer = clamp(steer, -limit, limit)

        speed_error = target_speed - speed
        if stability_mode:
            accel = 0.0
            brake = 0.30 if abs(lateral_speed) < 20 else 0.45
        elif speed_error < -4.0:
            accel = 0.0
            brake = clamp((-speed_error - 2.0) / 24.0, 0.0, 1.0)
        else:
            brake = 0.0
            accel = clamp(0.16 + speed_error / 32.0, 0.08, 1.0)

        steering_magnitude = abs(steer)
        if steering_magnitude > 0.65:
            accel = min(accel, 0.16)
        elif steering_magnitude > 0.48:
            accel = min(accel, 0.32)
        elif steering_magnitude > 0.32:
            accel = min(accel, 0.55)
        if abs(track_position) > 0.72:
            accel = min(accel, 0.20)

        wheel_spin = telemetry["wheelSpinVel"]
        slip = wheel_spin[2] + wheel_spin[3] - wheel_spin[0] - wheel_spin[1]
        if slip > 3.2:
            accel = max(0.0, accel - 0.10)

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
            target_speed,
            lookahead,
            lookahead_distance,
            line_blend,
        )
        self.previous_steer = steer
        self.step += 1
        return action

    def _write_telemetry(
        self,
        telemetry,
        action,
        target_position,
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
