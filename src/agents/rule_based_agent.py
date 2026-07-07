import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEMETRY_PATH = PROJECT_ROOT / "data" / "rule_based_telemetry.csv"

TARGET_SPEED = 216
FAST_BEND_SPEED = 202
SHALLOW_BEND_SPEED = 180
MEDIUM_CORNER_SPEED = 138
TIGHT_CORNER_SPEED = 106
HAIRPIN_SPEED = 82

ANGLE_STEER_GAIN = 13.5
CENTERING_GAIN = 0.65
SENSOR_STEER_GAIN = 0.060
STEER_SMOOTHING = 0.845
MAX_STEER_CHANGE = 0.050
STEER_DEADZONE = 0.010
SAFE_TRACK_LIMIT = 0.76
HARD_TRACK_LIMIT = 0.92
LATERAL_SPEED_LIMIT = 10.0
HIGH_LATERAL_SPEED_LIMIT = 20.0
LATERAL_STEER_DAMPING = 0.025
ENABLE_TRACTION_CONTROL = True
ENABLE_DEBUG_OUTPUT = False
LOG_EVERY_N_STEPS = 1
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

TELEMETRY_COLUMNS = [
    "step",
    "speedX",
    "speedY",
    "speedZ",
    "angle",
    "trackPos",
    "damage",
    "rpm",
    "gear",
    "accel",
    "brake",
    "steer",
    "targetSpeed",
    "section",
    "phase",
    "front",
    "front_delta",
    "best_i",
    "best_distance",
    "anglePlan",
    "severity",
    "desired_track_pos",
    "emergency_front_brake",
    "stability_mode",
    "stuck_counter",
] + [f"track_{index}" for index in range(19)]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def get_best_sensor(state):
    track = state["track"]
    front = track[9]
    candidate_indices = range(4, 15) if front < 65 else range(6, 13)
    best_index = 9
    best_score = front * 1.12

    for index in candidate_indices:
        offset = abs(index - 9)
        centre_penalty = 1.0 - offset * (0.035 if front < 65 else 0.10)
        score = track[index] * centre_penalty
        if score > best_score:
            best_score = score
            best_index = index

    return best_index


def is_controlled_low_speed_bend(state):
    return (
        state["speedX"] < 92
        and abs(state["trackPos"]) < 0.56
        and abs(state["angle"]) < 0.46
        and abs(state["speedY"]) < 5.2
        and not is_stability_mode(state)
    )


def is_slow_tight_rotation_zone(state, plan):
    return (
        plan["section"] in ["TIGHT", "HAIRPIN"]
        and plan["front"] < 42
        and state["speedX"] < 105
        and abs(state["trackPos"]) < 0.74
        and abs(state["speedY"]) < 7.2
        and abs(state["angle"]) < 0.70
        and not is_stability_mode(state)
    )


def get_emergency_front_brake(state):
    speed = state["speedX"]
    front = state["track"][9]
    controlled_bend = is_controlled_low_speed_bend(state)

    if front < 8 and speed > 30:
        return 0.42 if controlled_bend and speed < 72 else 1.00
    if front < 14 and speed > 55:
        return 0.28 if controlled_bend and speed < 84 else 1.00
    if front < 22 and speed > 85:
        return 0.75 if controlled_bend else 1.00
    if front < 32 and speed > 110:
        return 0.90
    if front < 45 and speed > 138:
        return 0.70
    return 0.0


def classify_section(front, planned_angle_abs, best_distance):
    closing = max(0.0, (145.0 - front) / 145.0)
    angle_factor = planned_angle_abs / 19.0
    distance_factor = max(0.0, (125.0 - best_distance) / 125.0)
    severity = 0.40 * angle_factor + 0.45 * closing + 0.15 * distance_factor

    if front > 155 and planned_angle_abs <= 7:
        section = "STRAIGHT"
    elif front > 125 and planned_angle_abs <= 10:
        section = "FAST_BEND"
    elif severity < 0.28:
        section = "FAST_BEND"
    elif severity < 0.46:
        section = "SHALLOW"
    elif severity < 0.66:
        section = "MEDIUM"
    elif severity < 0.88:
        section = "TIGHT"
    else:
        section = "HAIRPIN"

    if front < 18:
        section = "HAIRPIN"
    elif front < 34 and planned_angle_abs >= 4:
        section = "TIGHT"
    elif (
        front < 55
        and planned_angle_abs >= 7
        and section in ["STRAIGHT", "FAST_BEND", "SHALLOW"]
    ):
        section = "MEDIUM"
    elif front < 55 and planned_angle_abs < 7 and section == "MEDIUM":
        section = "SHALLOW"

    if front > 145 and section in ["TIGHT", "HAIRPIN"]:
        section = "MEDIUM"
    return section, severity


def get_section_speed(section):
    return {
        "STRAIGHT": TARGET_SPEED,
        "FAST_BEND": FAST_BEND_SPEED,
        "SHALLOW": SHALLOW_BEND_SPEED,
        "MEDIUM": MEDIUM_CORNER_SPEED,
        "TIGHT": TIGHT_CORNER_SPEED,
        "HAIRPIN": HAIRPIN_SPEED,
    }[section]


def get_corner_phase(state, section, front, front_delta):
    speed = state["speedX"]
    angle_abs = abs(state["angle"])
    if section in ["STRAIGHT", "FAST_BEND"]:
        return "STRAIGHT"
    if front_delta < -1.0 and speed > get_section_speed(section) + 8:
        return "APPROACH"
    if front_delta > 0.35 and angle_abs < 0.38 and front > 14:
        return "EXIT"
    if front < 38 or angle_abs > 0.44:
        return "APEX"
    if front_delta > 0.2 and angle_abs < 0.42:
        return "EXIT"
    if speed > get_section_speed(section) + 14:
        return "APPROACH"
    return "APEX"


def plan_next_curve(state, previous_front=None):
    front = state["track"][9]
    front_delta = 0.0 if previous_front is None else front - previous_front
    best_index = get_best_sensor(state)
    best_distance = state["track"][best_index]
    best_angle = TRACK_SENSOR_ANGLES[best_index]
    section, severity = classify_section(front, abs(best_angle), best_distance)
    return {
        "front": front,
        "front_delta": front_delta,
        "best_i": best_index,
        "best_distance": best_distance,
        "best_angle": best_angle,
        "planned_angle_abs": abs(best_angle),
        "severity": severity,
        "section": section,
        "phase": get_corner_phase(state, section, front, front_delta),
    }


def is_stability_mode(state):
    lateral_speed = abs(state["speedY"])
    forward_speed = max(abs(state["speedX"]), 1.0)
    angle_abs = abs(state["angle"])
    return (
        angle_abs > 0.72
        or lateral_speed > HIGH_LATERAL_SPEED_LIMIT
        or (lateral_speed > LATERAL_SPEED_LIMIT and angle_abs > 0.12)
        or (lateral_speed > forward_speed * 0.32 and forward_speed > 45)
    )


def get_predictive_front_speed_cap(state, previous_front=None):
    front = state["track"][9]
    speed = state["speedX"]
    front_delta = 0.0 if previous_front is None else front - previous_front
    closing_fast = front_delta < -0.85
    closing_near_fast = front_delta < -0.58 and front < 78 and speed > 158
    if not (closing_fast or closing_near_fast) or speed < 125:
        return TARGET_SPEED

    for distance, cap in [
        (22, 64),
        (32, 88),
        (44, 114),
        (58, 138),
        (76, 158),
        (98, 176),
        (122, 192),
    ]:
        if front < distance:
            return cap
    return TARGET_SPEED


def get_high_speed_entry_risk(state, previous_front=None):
    front = state["track"][9]
    front_delta = 0.0 if previous_front is None else front - previous_front
    return state["speedX"] > 155 and front < 102 and front_delta < -0.72


def get_dynamic_target_speed(state, previous_front=None):
    plan = plan_next_curve(state, previous_front)
    front = plan["front"]
    if is_stability_mode(state):
        return 45
    if abs(state["trackPos"]) > HARD_TRACK_LIMIT:
        return 50
    if abs(state["trackPos"]) > SAFE_TRACK_LIMIT:
        return 82

    controlled_bend = is_controlled_low_speed_bend(state)
    if front < 10:
        return 64 if controlled_bend else 42
    if front < 18:
        return 86 if controlled_bend else 60
    if front < 28:
        return 100 if controlled_bend else 82
    if front < 42:
        return 118 if controlled_bend else 105

    base_speed = get_section_speed(plan["section"])
    if plan["phase"] == "APPROACH":
        phase_speed = max(HAIRPIN_SPEED, base_speed - 18)
    elif plan["phase"] == "APEX":
        phase_speed = max(HAIRPIN_SPEED, base_speed - 8)
    elif plan["phase"] == "EXIT":
        phase_speed = min(TARGET_SPEED, base_speed + 12)
    else:
        phase_speed = base_speed
    return min(phase_speed, get_predictive_front_speed_cap(state, previous_front))


def get_desired_track_pos(_state, _plan):
    """Keep the car on the centreline instead of chasing racing-line offsets."""
    return 0.0


def get_steering_limit(speed, track_position, stability_mode):
    if stability_mode:
        return 0.55
    if abs(track_position) > SAFE_TRACK_LIMIT:
        return 0.80
    if speed < 70:
        return 0.78
    if speed < 110:
        return 0.68
    if speed < 150:
        return 0.52
    return 0.38


def get_corner_throttle_cap(steer, track_position):
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

    if abs(track_position) > 0.55:
        cap = min(cap, 0.30)
    return cap


def calculate_steering(state, previous_steer=0.0, previous_front=None):
    plan = plan_next_curve(state, previous_front)
    speed = state["speedX"]
    slow_rotation = is_slow_tight_rotation_zone(state, plan)

    if is_stability_mode(state):
        lateral_damping = -state["speedY"] * LATERAL_STEER_DAMPING
        raw_steer = (
            state["angle"] * 8.0 / math.pi
            - state["trackPos"] * 0.9
            + lateral_damping
        )
        raw_steer = clamp(raw_steer, -0.55, 0.55)
    else:
        desired_pos = get_desired_track_pos(state, plan)
        angle_gain = 19.5 if slow_rotation else ANGLE_STEER_GAIN
        angle_correction = state["angle"] * angle_gain / math.pi
        abs_pos = abs(state["trackPos"])

        if abs_pos > HARD_TRACK_LIMIT:
            pos_gain = 1.25
        elif abs_pos > SAFE_TRACK_LIMIT:
            pos_gain = 0.95
        elif speed > 170:
            pos_gain = CENTERING_GAIN * 0.46
        elif speed > 120:
            pos_gain = CENTERING_GAIN * 0.62
        else:
            pos_gain = CENTERING_GAIN * 0.86

        if abs(desired_pos) > 0.05 and abs_pos < SAFE_TRACK_LIMIT:
            pos_gain *= 0.82
        if slow_rotation:
            pos_gain *= 0.68
        position_correction = (state["trackPos"] - desired_pos) * pos_gain

        sensor_steer = 0.0
        if abs_pos < 0.68:
            sensor_steer = (
                clamp(plan["best_angle"] / 19.0, -1.0, 1.0)
                * SENSOR_STEER_GAIN
            )
            if slow_rotation:
                sensor_steer = clamp(plan["best_angle"] / 19.0, -1.0, 1.0) * 0.19
            elif plan["section"] in ["FAST_BEND", "SHALLOW"] and speed > 95:
                sensor_steer *= 0.75

        lateral_damping = -state["speedY"] * LATERAL_STEER_DAMPING
        raw_steer = (
            angle_correction
            - position_correction
            + sensor_steer
            + lateral_damping
        )
        if slow_rotation:
            raw_steer += clamp(plan["best_angle"] / 19.0, -1.0, 1.0) * 0.16

        if state["trackPos"] > HARD_TRACK_LIMIT:
            raw_steer -= 0.85
        elif state["trackPos"] < -HARD_TRACK_LIMIT:
            raw_steer += 0.85
        elif state["trackPos"] > SAFE_TRACK_LIMIT:
            raw_steer -= 0.48
        elif state["trackPos"] < -SAFE_TRACK_LIMIT:
            raw_steer += 0.48
        steer_limit = get_steering_limit(speed, state["trackPos"], False)
        raw_steer = clamp(raw_steer, -steer_limit, steer_limit)

    if abs(raw_steer) < STEER_DEADZONE:
        raw_steer = 0.0
    smoothing = 0.78 if slow_rotation else STEER_SMOOTHING
    smoothed = smoothing * previous_steer + (1.0 - smoothing) * raw_steer

    if is_stability_mode(state):
        max_delta = 0.095
    elif abs(state["trackPos"]) > SAFE_TRACK_LIMIT:
        max_delta = 0.080
    elif slow_rotation:
        max_delta = 0.080
    elif plan["front"] < 35:
        max_delta = 0.070
    else:
        max_delta = MAX_STEER_CHANGE
    delta = clamp(smoothed - previous_steer, -max_delta, max_delta)
    steer_limit = get_steering_limit(
        speed,
        state["trackPos"],
        is_stability_mode(state),
    )
    return clamp(previous_steer + delta, -steer_limit, steer_limit)


def calculate_brake(state, _steer, previous_front=None):
    speed = state["speedX"]
    target = get_dynamic_target_speed(state, previous_front)
    plan = plan_next_curve(state, previous_front)
    if is_stability_mode(state):
        return 0.45 if abs(state["speedY"]) > HIGH_LATERAL_SPEED_LIMIT else 0.30

    emergency = get_emergency_front_brake(state)
    if emergency > 0:
        return emergency
    if abs(state["trackPos"]) > HARD_TRACK_LIMIT and speed > 30:
        return 1.00
    if abs(state["trackPos"]) > SAFE_TRACK_LIMIT and speed > 50:
        return 0.80
    if get_high_speed_entry_risk(state, previous_front):
        if speed > target + 40:
            return 0.70
        if speed > target + 25:
            return 0.48
        if speed > target + 12:
            return 0.24
    if plan["phase"] == "APPROACH" and speed > target + 9:
        return 0.48
    if speed > target + 42:
        return 1.00
    if speed > target + 28:
        return 0.68
    if speed > target + 15:
        return 0.32
    return 0.0


def calculate_throttle(state, brake, _steer, previous_front=None):
    speed = state["speedX"]
    target = get_dynamic_target_speed(state, previous_front)
    plan = plan_next_curve(state, previous_front)
    if brake > 0 or is_stability_mode(state) or abs(state["speedY"]) > 10.0:
        return 0.0
    if abs(state["speedY"]) > 6.0:
        return 0.12
    if get_high_speed_entry_risk(state, previous_front) and speed > target + 2:
        return 0.0
    if plan["front"] < 18:
        if is_controlled_low_speed_bend(state) and speed < target - 6:
            return 0.40 if speed < 55 else 0.26
        return 0.0
    if abs(state["trackPos"]) > SAFE_TRACK_LIMIT:
        return 0.02
    if speed < 30:
        return 0.20 if abs(state["angle"]) > 0.25 or abs(state["trackPos"]) > 0.45 else 0.45
    if speed < 80:
        return 0.25 if abs(state["angle"]) > 0.30 else 0.60

    if plan["phase"] == "APEX":
        if plan["section"] in ["TIGHT", "HAIRPIN"] and is_controlled_low_speed_bend(state):
            if speed < target - 14:
                return 0.34
            if speed < target - 5:
                return 0.20
            return 0.08
        if (
            plan["section"] in ["FAST_BEND", "SHALLOW"]
            and abs(state["angle"]) < 0.25
            and abs(state["speedY"]) < 4.5
        ):
            if plan["front"] > 70 and speed < target - 8:
                return 0.42
            return 0.34 if speed < target - 10 else 0.16
        if speed < target - 15 and abs(state["angle"]) < 0.35:
            return 0.30
        return 0.10

    if plan["phase"] == "EXIT":
        if abs(state["angle"]) > 0.30 or abs(state["speedY"]) > 6.4:
            return 0.14
        if speed < target - 28:
            return 0.92
        if speed < target - 10:
            return 0.56
        return 0.22
    if speed < target - 35:
        return 1.00
    if speed < target - 15:
        return 0.65
    if speed < target - 5:
        return 0.30
    return 0.10


def smooth_accel(previous_accel, desired_accel, brake, state, plan):
    if brake > 0:
        return 0.0
    if plan["section"] in ["TIGHT", "HAIRPIN"] or is_stability_mode(state):
        max_up, max_down = 0.17, 0.26
    elif plan["phase"] in ["APEX", "EXIT"]:
        max_up, max_down = 0.18, 0.28
    else:
        max_up, max_down = 0.24, 0.34
    return previous_accel + clamp(
        desired_accel - previous_accel,
        -max_down,
        max_up,
    )


def apply_traction_control(state, accel):
    if not ENABLE_TRACTION_CONTROL:
        return accel

    rear_spin = state["wheelSpinVel"][2] + state["wheelSpinVel"][3]
    front_spin = state["wheelSpinVel"][0] + state["wheelSpinVel"][1]
    if rear_spin - front_spin > 3.2:
        accel -= 0.10
    return clamp(accel, 0.0, 1.0)


def shift_gears(state):
    gear = 1
    for index, threshold in enumerate(GEAR_SPEEDS):
        if state["speedX"] > threshold:
            gear = index + 1
    return int(clamp(gear, 1, 6))


def should_shutdown(state, stuck_counter):
    if state.get("damage", 0) > 500:
        return True, "damage limit exceeded"
    if abs(state.get("trackPos", 0)) > 1.15:
        return True, "out of bounds"
    if abs(state.get("angle", 0)) > 2.2:
        return True, "wrong direction"
    if stuck_counter > 180:
        return True, "car stuck"
    return False, ""


def print_drive_debug(state, action, step, previous_front=None):
    if not ENABLE_DEBUG_OUTPUT or step % 10 != 0:
        return

    plan = plan_next_curve(state, previous_front)
    print(
        f"step={step:05d} | "
        f"speed={state['speedX']:6.1f} | "
        f"target={get_dynamic_target_speed(state, previous_front):6.1f} | "
        f"section={plan['section']:9s} | "
        f"phase={plan['phase']:8s} | "
        f"front={plan['front']:6.1f} | "
        f"dFront={plan['front_delta']:6.2f} | "
        f"best={plan['best_i']:02d}:{plan['best_distance']:5.1f} | "
        f"anglePlan={plan['best_angle']:5.1f} | "
        f"trackPos={state['trackPos']:7.3f} | "
        f"desired={get_desired_track_pos(state, plan):6.2f} | "
        f"angle={state['angle']:7.3f} | "
        f"stab={is_stability_mode(state)} | "
        f"steer={action['steer']:6.3f} | "
        f"accel={action['accel']:5.2f} | "
        f"brake={action['brake']:5.2f}"
    )


class RuleBasedAgent:
    name = "Rule-Based Anti-Spin Agent"
    agent_type = "rule_based"
    version = "1.9"
    seed = None
    uses_full_control = True
    max_steps = 100000

    def __init__(self, telemetry_path=DEFAULT_TELEMETRY_PATH):
        self.telemetry_path = Path(telemetry_path)
        self._telemetry_file = None
        self._telemetry_writer = None
        self.reset()

    @property
    def config(self):
        return {
            "target_speed": TARGET_SPEED,
            "traction_control": True,
            "steering_gain": ANGLE_STEER_GAIN,
            "centreline_target": 0.0,
            "speed_sensitive_steering": True,
            "telemetry_file": str(self.telemetry_path.relative_to(PROJECT_ROOT)),
        }

    def reset(self):
        self.close()
        self.previous_steer = 0.0
        self.previous_accel = 0.0
        self.previous_front = None
        self.stuck_counter = 0
        self.step = self.max_steps
        self.previous_gear = 1
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
            raise ValueError("RuleBasedAgent requires raw TORCS telemetry")

        state = telemetry
        self.stuck_counter = self.stuck_counter + 1 if state["speedX"] < 5 else 0
        shutdown, reason = should_shutdown(state, self.stuck_counter)

        if shutdown:
            print(f"\n[FAILSAFE] Shutdown triggered: {reason}")
            plan = plan_next_curve(state, self.previous_front)
            action = {
                "steer": self.previous_steer,
                "accel": 0.0,
                "brake": 1.0,
                "gear": self.previous_gear,
                "terminate": True,
                "termination_reason": reason,
            }
            self._write_telemetry(state, plan, action)
            return action

        plan = plan_next_curve(state, self.previous_front)
        steer = calculate_steering(state, self.previous_steer, self.previous_front)
        brake = calculate_brake(state, steer, self.previous_front)
        desired_accel = calculate_throttle(
            state,
            brake,
            steer,
            self.previous_front,
        )
        desired_accel = apply_traction_control(state, desired_accel)
        accel = smooth_accel(
            self.previous_accel,
            desired_accel,
            brake,
            state,
            plan,
        )
        accel = min(
            accel,
            get_corner_throttle_cap(steer, state["trackPos"]),
        )
        gear = shift_gears(state)
        action = {
            "steer": steer,
            "accel": accel,
            "brake": brake,
            "gear": gear,
            "terminate": False,
            "termination_reason": "",
        }

        self.previous_steer = steer
        self.previous_accel = accel
        self.previous_gear = gear
        print_drive_debug(state, action, self.step, self.previous_front)
        self._write_telemetry(state, plan, action)
        self.previous_front = state["track"][9]
        self.step -= 1
        return action

    def _write_telemetry(self, state, plan, action):
        if self._telemetry_writer is None or self.step % LOG_EVERY_N_STEPS != 0:
            return
        self._telemetry_writer.writerow(
            [
                self.step,
                state.get("speedX", 0),
                state.get("speedY", 0),
                state.get("speedZ", 0),
                state.get("angle", 0),
                state.get("trackPos", 0),
                state.get("damage", 0),
                state.get("rpm", 0),
                action["gear"],
                action["accel"],
                action["brake"],
                action["steer"],
                get_dynamic_target_speed(state, self.previous_front),
                plan["section"],
                plan["phase"],
                plan["front"],
                plan["front_delta"],
                plan["best_i"],
                plan["best_distance"],
                plan["best_angle"],
                plan["severity"],
                get_desired_track_pos(state, plan),
                get_emergency_front_brake(state),
                is_stability_mode(state),
                self.stuck_counter,
            ]
            + list(state["track"])
        )

    def close(self):
        if self._telemetry_file is not None:
            self._telemetry_file.close()
        self._telemetry_file = None
        self._telemetry_writer = None
