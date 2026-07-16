"""Replay recorded map-aware telemetry through the current controller.

This is an offline evaluator: it does not move TORCS. It feeds recorded state
rows back into MapAwareAgent and writes the controller's current target/speed/
steer decisions for the same distances and car states.
"""

import argparse
import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.map_aware_agent import DEFAULT_RACING_LINE_PATH, MapAwareAgent  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "map_aware_telemetry.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "map_aware_replay_eval.csv"


def parse_float(row, name, default=0.0):
    value = row.get(name)
    if value in (None, ""):
        return default
    return float(value)


def reconstructed_track_sensors(row):
    if all(row.get(f"trackSensor{index}") not in (None, "") for index in range(19)):
        return [parse_float(row, f"trackSensor{index}", 200.0) for index in range(19)]

    front = parse_float(row, "sensorFront", 200.0)
    if front < 0.0:
        return [-1.0] * 19

    sensors = [200.0] * 19
    sensors[9] = front
    best_angle = parse_float(row, "sensorBestAngle", 0.0)
    angle_to_index = {
        -45.0: 0,
        -19.0: 1,
        -12.0: 2,
        -7.0: 3,
        -4.0: 4,
        -2.5: 5,
        -1.7: 6,
        -1.0: 7,
        -0.5: 8,
        0.0: 9,
        0.5: 10,
        1.0: 11,
        1.7: 12,
        2.5: 13,
        4.0: 14,
        7.0: 15,
        12.0: 16,
        19.0: 17,
        45.0: 18,
    }
    best_index = angle_to_index.get(best_angle, 9)
    sensors[best_index] = max(front + 20.0, 40.0)
    return sensors


def telemetry_from_row(row):
    return {
        "speedX": parse_float(row, "speedX"),
        "speedY": parse_float(row, "speedY"),
        "speedZ": 0.0,
        "angle": parse_float(row, "angle"),
        "trackPos": parse_float(row, "trackPos"),
        "damage": 0.0,
        "rpm": 5000.0,
        "track": reconstructed_track_sensors(row),
        "wheelSpinVel": [10.0] * 4,
        "distFromStart": parse_float(row, "distFromStart"),
        "curLapTime": parse_float(row, "curLapTime"),
        "lastLapTime": parse_float(row, "lastLapTime"),
    }


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def replay(input_path, output_path, racing_line_path=DEFAULT_RACING_LINE_PATH):
    input_rows = load_rows(input_path)
    if not input_rows:
        raise ValueError(f"No telemetry rows found in {input_path}")

    with tempfile.TemporaryDirectory() as directory:
        controller_log = Path(directory) / "controller.csv"
        agent = MapAwareAgent(
            racing_line_path=racing_line_path,
            telemetry_path=controller_log,
        )
        actions = []
        try:
            for row in input_rows:
                actions.append(agent.act(None, telemetry_from_row(row)))
        finally:
            agent.close()

        replay_rows = load_rows(controller_log)

    output_rows = []
    for source, replayed, action in zip(input_rows, replay_rows, actions):
        output_rows.append(
            {
                "step": replayed["step"],
                "distFromStart": source.get("distFromStart", replayed["distFromStart"]),
                "actualTrackPos": source.get("trackPos", ""),
                "controlState": replayed.get("controlState", ""),
                "rawRacingLinePos": replayed.get("rawRacingLinePos", ""),
                "practicalTrackPos": replayed.get("practicalTrackPos", ""),
                "targetTrackPos": replayed.get("targetTrackPos", ""),
                "previewTrackPos": replayed.get("previewTrackPos", ""),
                "steeringTrackPos": replayed.get("steeringTrackPos", ""),
                "lineError": replayed.get("lineError", ""),
                "effectiveLineError": replayed.get("effectiveLineError", ""),
                "targetSpeed": replayed.get("targetSpeed", ""),
                "sensorFront": source.get("sensorFront", replayed.get("sensorFront", "")),
                "steer": action["steer"],
                "accel": action["accel"],
                "brake": action["brake"],
            }
        )

    write_rows(output_path, output_rows)
    return summarize(output_rows, output_path)


def summarize(rows, output_path):
    state_counts = Counter(row["controlState"] for row in rows)
    line_errors = [abs(parse_float(row, "lineError")) for row in rows]
    target_speeds = [parse_float(row, "targetSpeed") for row in rows]
    front_values = [parse_float(row, "sensorFront", 200.0) for row in rows]
    off_track_like = sum(
        1
        for row in rows
        if abs(parse_float(row, "actualTrackPos")) > 1.0
        or parse_float(row, "sensorFront", 200.0) < 0.0
    )

    return {
        "rows": len(rows),
        "output": str(output_path),
        "state_counts": dict(state_counts),
        "mean_abs_line_error": sum(line_errors) / len(line_errors),
        "max_abs_line_error": max(line_errors),
        "min_target_speed": min(target_speeds),
        "max_target_speed": max(target_speeds),
        "front_under_20": sum(front < 20.0 for front in front_values),
        "off_track_like_rows": off_track_like,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--racing-line", type=Path, default=DEFAULT_RACING_LINE_PATH)
    args = parser.parse_args()

    summary = replay(args.input, args.output, args.racing_line)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
