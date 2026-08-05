from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .project_discovery import AgentOption, TrackOption, compatible_tracks_for_agent
except ImportError:
    from project_discovery import AgentOption, TrackOption, compatible_tracks_for_agent


@dataclass(frozen=True)
class CodeSnippet:
    title: str
    source: str
    explanation: str
    code: str


@dataclass(frozen=True)
class FormulaNote:
    title: str
    formula: str
    explanation: str


@dataclass(frozen=True)
class AgentEducationProfile:
    title: str
    badge: str
    headline: str
    overview: tuple[str, ...]
    decision_steps: tuple[str, ...]
    algorithm_summary: tuple[str, ...]
    pseudocode: tuple[str, ...]
    formula_notes: tuple[FormulaNote, ...]
    code_snippets: tuple[CodeSnippet, ...]
    math_notes: tuple[str, ...]
    key_takeaways: tuple[str, ...]
    input_signals: tuple[str, ...]
    strengths: tuple[str, ...]
    failure_signs: tuple[str, ...]
    track_context: tuple[str, ...]
    metadata: tuple[tuple[str, str], ...]


def build_agent_education_profile(
    agent: AgentOption,
    tracks: list[TrackOption],
) -> AgentEducationProfile:
    template = _template_for(agent.agent_type)
    return AgentEducationProfile(
        title=agent.name,
        badge=template["badge"],
        headline=template["headline"],
        overview=tuple(template["overview"]),
        decision_steps=tuple(template["decision_steps"]),
        algorithm_summary=tuple(template["algorithm_summary"]),
        pseudocode=tuple(template["pseudocode"]),
        formula_notes=tuple(template["formula_notes"]),
        code_snippets=tuple(template["code_snippets"]),
        math_notes=tuple(template["math_notes"]),
        key_takeaways=tuple(template["key_takeaways"]),
        input_signals=tuple(template["input_signals"]),
        strengths=tuple(template["strengths"]),
        failure_signs=tuple(template["failure_signs"]),
        track_context=_track_context(agent, tracks),
        metadata=_metadata(agent),
    )


def _template_for(agent_type: str) -> dict[str, Any]:
    if agent_type == "map_aware":
        return {
            "badge": "Map-specific racing-line planner",
            "headline": (
                "This driver uses a precomputed racing line as a geometric plan, "
                "then turns that plan into steering, throttle, and braking decisions."
            ),
            "overview": (
                "It is not only reacting to the road immediately in front of the car.",
                (
                    "Before the run starts, the track is represented as a sequence of "
                    "centreline samples. The racing line shifts those samples left or "
                    "right across the road to create an intended route."
                ),
                (
                    "During the run, the agent uses lap distance to look up where the "
                    "car should be now, where it should be next, and how fast it should "
                    "aim to travel through that part of the circuit."
                ),
                (
                    "When it works well, the behaviour looks planned: the car sets up "
                    "corners early, cuts toward an apex, and exits with less panic "
                    "steering than a purely reactive controller."
                ),
            ),
            "decision_steps": (
                "Load the racing-line waypoints generated for the selected track.",
                "Read speed, yaw angle, side speed, track position, lap distance, and sensors.",
                "Interpolate the racing-line target at the current lap distance.",
                "Preview several lookahead points to estimate the next corner.",
                "Blend onto the line, guard unsafe targets, then calculate controls.",
            ),
            "algorithm_summary": (
                "The map-aware driver separates route planning from real-time control.",
                (
                    "The saved racing-line file stores target road position, target "
                    "speed, curvature, heading offset, speed change, and driving phase "
                    "around the lap."
                ),
                (
                    "At runtime, distance around the lap becomes the lookup key. The "
                    "agent interpolates between waypoints so the target changes smoothly "
                    "rather than jumping from point to point."
                ),
                (
                    "The controller is then a path-following problem: minimise the "
                    "difference between current road position and planned road position, "
                    "while respecting speed, stability, and sensor safety limits."
                ),
                (
                    "Live road sensors act as guardrails. They do not define the ideal "
                    "line, but they can slow the car or pull the target inward when the "
                    "ideal line is unsafe in the current state."
                ),
            ),
            "pseudocode": (
                "Load the racing-line JSON for this track.",
                "For each control tick, read speed, angle, side speed, trackPos, distFromStart, and road sensors.",
                "Convert the current lap distance into a racing-line waypoint lookup.",
                "Interpolate target road position, curvature, heading, speed, and phase.",
                "Choose a lookahead distance that grows with speed.",
                "Blend from the launch position onto the racing line using smoothstep.",
                "Clamp or guard the target if edge risk or road sensors show danger.",
                "Estimate a target speed from planned speed, curvature, lookahead, and live safety caps.",
                "Compute line error: current road position minus target road position.",
                "Steer using cross-track error, heading offset, curvature, and previous steer.",
                "Choose throttle or brake from the difference between current speed and target speed.",
                "Return full-control actions: steering, throttle, brake, and gear.",
            ),
            "formula_notes": (
                FormulaNote(
                    title="Normalised Track Position",
                    formula=(
                        r"$\mathrm{offset}_m = \mathrm{target\_track\_pos}"
                        r" \cdot \frac{\mathrm{track\_width}_m}{2}$"
                    ),
                    explanation=(
                        "TORCS represents road position as a normalised value. A value "
                        "of 0 is the centreline, -1 is the left edge, and +1 is the "
                        "right edge. The formula converts that normalised value into a "
                        "physical sideways offset in metres."
                    ),
                ),
                FormulaNote(
                    title="Racing Point Projection",
                    formula=(
                        r"$\mathbf{p}_{race,i} = \mathbf{c}_i"
                        r" + \mathbf{n}_i \cdot \mathrm{offset}_m$"
                    ),
                    explanation=(
                        "Each sampled centreline point has a perpendicular normal vector. "
                        "Moving along that normal places the racing line inside the actual "
                        "track instead of drawing a separate abstract curve."
                    ),
                ),
                FormulaNote(
                    title="Lap-Distance Interpolation",
                    formula=(
                        r"$\alpha = \frac{\mathrm{mod}(d - d_i, L)}{\mathrm{mod}(d_{next} - d_i, L)}$"
                        "\n"
                        r"$x(d) = x_i + \alpha(x_{next} - x_i)$"
                    ),
                    explanation=(
                        "The racing line is stored as waypoints, but the car can be at "
                        "any distance around the lap. Interpolation fills the gap between "
                        "two saved points, and modulo arithmetic wraps the lookup over "
                        "the start/finish line."
                    ),
                ),
                FormulaNote(
                    title="Speed-Dependent Lookahead",
                    formula=(
                        r"$lookahead_m = \mathrm{clamp}"
                        r"(24.0 + 0.16v_{km/h}, 24.0, 56.0)$"
                    ),
                    explanation=(
                        "The faster the car travels, the further ahead it should preview. "
                        "The clamp prevents the controller from looking too close at high "
                        "speed or too far ahead at low speed."
                    ),
                ),
                FormulaNote(
                    title="Smooth Merge Onto The Line",
                    formula=(
                        r"$t = \mathrm{clamp}\left(\frac{d_{travelled}}{d_{merge}}, 0, 1\right)$"
                        "\n"
                        r"$line\_blend = t^2(3 - 2t)$"
                    ),
                    explanation=(
                        "This smoothstep curve lets the car join the racing line gradually "
                        "after launch. It avoids snapping sideways toward the ideal line "
                        "before the car has enough speed and control authority."
                    ),
                ),
                FormulaNote(
                    title="Curvature-Limited Speed",
                    formula=(
                        r"$v_{m/s} \leq"
                        r" \sqrt{\frac{a_{lat}}{\max(|\kappa|, 10^{-6})}}$"
                    ),
                    explanation=(
                        "Sharper bends have larger curvature, so the maximum safe speed "
                        "falls. This is the same principle as cornering grip: for a fixed "
                        "lateral acceleration limit, tighter turns require lower speed."
                    ),
                ),
                FormulaNote(
                    title="Line Error",
                    formula=r"$e_{line} = track\_pos - target\_position$",
                    explanation=(
                        "A positive or negative error tells the controller which side of "
                        "the desired racing line the car is currently on. Steering and "
                        "speed limits are then adjusted from that error."
                    ),
                ),
            ),
            "code_snippets": (
                CodeSnippet(
                    title="Racing-line lookup",
                    source="src/racing_line/optimizer.py - RacingLine.lookup",
                    explanation=(
                        "The line is indexed by distance around the lap. The modulo "
                        "operation lets the lookup wrap cleanly over the start/finish line."
                    ),
                    code=(
                        "def lookup(self, distance):\n"
                        "    distance = float(distance) % self.track_length\n"
                        "    right_index = bisect.bisect_right(self._distances, distance) % len(self.waypoints)\n"
                        "    left_index = (right_index - 1) % len(self.waypoints)\n"
                        "    left = self.waypoints[left_index]\n"
                        "    right = self.waypoints[right_index]\n"
                        "    span = (right[\"distance\"] - left[\"distance\"]) % self.track_length\n"
                        "    travelled = (distance - left[\"distance\"]) % self.track_length\n"
                        "    fraction = travelled / span if span else 0.0\n"
                        "    return interpolated | {\"phase\": left[\"phase\"]}"
                    ),
                ),
                CodeSnippet(
                    title="Previewing the route ahead",
                    source="src/agents/map_aware_agent.py - MapAwareAgent.act",
                    explanation=(
                        "The controller does not only chase the point underneath the car. "
                        "It samples points ahead so steering and speed prepare for the next section."
                    ),
                    code=(
                        "lookahead_distance = clamp(24.0 + speed * 0.16, 24.0, 56.0)\n"
                        "target = self.racing_line.lookup(distance)\n"
                        "lookahead = self.racing_line.lookup(distance + lookahead_distance)\n"
                        "far_lookahead = self.racing_line.lookup(distance + lookahead_distance * 1.85)\n"
                        "very_far_lookahead = self.racing_line.lookup(distance + lookahead_distance * 3.25)\n"
                        "line_complexity = estimate_line_complexity(target, lookahead, far_lookahead)"
                    ),
                ),
                CodeSnippet(
                    title="Turning the plan into controls",
                    source="src/agents/map_aware_agent.py - MapAwareAgent.act",
                    explanation=(
                        "The agent softens the line error, calculates steering from the path, "
                        "then asks the speed controller for throttle and brake."
                    ),
                    code=(
                        "target_speed = calculate_dynamic_target_speed(...)\n"
                        "steering_target_position = track_position - effective_line_error\n"
                        "steer = calculate_path_tracking_steer(\n"
                        "    speed=speed,\n"
                        "    track_position=track_position,\n"
                        "    target_position=steering_target_position,\n"
                        "    heading_offset=desired_heading,\n"
                        "    curvature=target_curvature,\n"
                        ")\n"
                        "accel, brake, adjusted_target_speed = calculate_aggressive_speed_control(...)"
                    ),
                ),
            ),
            "math_notes": (
                "TORCS track position is normalised: 0 is the centre, -1 is the left edge, and +1 is the right edge.",
                (
                    "The optimizer samples the centreline and shifts each sample along the "
                    "track normal: racing point = centre point + normal * offset."
                ),
                (
                    "target_track_pos is the normal offset expressed as road position: "
                    "offset / (track_width / 2)."
                ),
                (
                    "Curvature is estimated from neighbouring racing-line points. Higher "
                    "curvature means the safe target speed should drop."
                ),
                (
                    "The speed profile is bounded by maximum speed, minimum speed, "
                    "lateral acceleration, acceleration, and braking limits."
                ),
                (
                    "Runtime lookup interpolates between waypoints using lap distance "
                    "modulo track length, so the plan behaves like a continuous loop."
                ),
            ),
            "key_takeaways": (
                "This is a planned driver: it has a target route before the race starts.",
                "The racing line is track-specific, so missing or mismatched line files matter.",
                "Sensors do not replace the map; they keep the map from becoming unsafe.",
                "Telemetry should show smoother preparation before corners when the line is working well.",
            ),
            "input_signals": (
                "Racing-line file for the selected track.",
                "Distance around the lap and current lap time.",
                "Speed, sideways speed, and car angle.",
                "Track position: centre, left edge, or right edge.",
                "Road sensors showing how much clear road is ahead.",
            ),
            "strengths": (
                "Best suited to tracks that already have a generated racing line.",
                "Usually more purposeful than a purely reactive driver on its known maps.",
                "Easier to explain because there is a visible target route.",
            ),
            "failure_signs": (
                "The selected track has no matching racing-line file.",
                "The car drifts far away from the line and the recovery becomes too late.",
                "Very sharp corners can expose poor speed control.",
                "A slow or stuck launch can make the planned line irrelevant.",
            ),
        }

    if agent_type == "rule_based":
        return {
            "badge": "Reactive rule-based controller",
            "headline": (
                "This driver uses transparent hand-written rules to turn live sensor "
                "readings into steering, speed, braking, and recovery decisions."
            ),
            "overview": (
                "It does not need a precomputed map of the track or a saved racing line.",
                (
                    "Instead, it treats each control step as a local decision problem: "
                    "what road can the sensors see, how fast is the car moving, and is "
                    "the car stable enough to keep attacking?"
                ),
                (
                    "The design is useful for explanation because every behaviour can be "
                    "linked back to a threshold, score, or rule. Its weakness is that it "
                    "cannot truly plan for parts of the circuit beyond the sensor view."
                ),
            ),
            "decision_steps": (
                "Read road sensors, speed, side speed, angle, track position, and wheel spin.",
                "Score the front-facing sensors to find the most useful visible road.",
                "Classify the visible section as straight, bend, tight corner, or hairpin.",
                "Choose speed and steering from the section, edge risk, and stability state.",
                "Apply gear, braking, and anti-spin rules before returning the action.",
            ),
            "algorithm_summary": (
                "The rule-based driver is reactive and interpretable.",
                (
                    "Its road model comes from 19 range sensors. The centre sensor looks "
                    "straight ahead, while neighbouring sensors provide angled views left "
                    "and right."
                ),
                (
                    "The controller scores those sensors, estimates how severe the next "
                    "section looks, and maps that section to a target speed."
                ),
                (
                    "Stability rules are deliberately separate from racing rules. If the "
                    "car has a high angle, high side speed, or is drifting toward an edge, "
                    "saving the car takes priority over going fast."
                ),
                (
                    "This makes the algorithm easy to audit, but it also gives it a lower "
                    "ceiling than a map-aware agent on a track where planning matters."
                ),
            ),
            "pseudocode": (
                "Read speed, side speed, angle, trackPos, wheel spin, and 19 road sensors.",
                "Find which front-facing sensor sees the most useful open road.",
                "Classify the visible section as straight, fast bend, shallow, medium, tight, or hairpin.",
                "Pick a target speed from that section type.",
                "Steer from car angle, road position, and best sensor direction.",
                "Brake or reduce throttle if the car is too fast for the visible road.",
                "Enter stability mode if angle, sideways speed, or edge risk is high.",
                "Apply gear and traction rules.",
                "Return the next full-control action.",
            ),
            "formula_notes": (
                FormulaNote(
                    title="Sensor Score",
                    formula=r"$score_i = track_i(1 - |i - 9|p)$",
                    explanation=(
                        "The middle sensor is index 9. This score favours open road but "
                        "penalises sensors that point too far away from the car's current "
                        "heading, reducing the chance of chasing an extreme edge."
                    ),
                ),
                FormulaNote(
                    title="Visible-Corner Severity",
                    formula=(
                        r"$severity = 0.40a$"
                        "\n"
                        r"$\quad + 0.45c$"
                        "\n"
                        r"$\quad + 0.15d$"
                    ),
                    explanation=(
                        "The rule-based agent compresses several sensor cues into one "
                        "severity value. Higher severity pushes the section label toward "
                        "TIGHT or HAIRPIN and lowers the target speed."
                    ),
                ),
                FormulaNote(
                    title="Closing Factor",
                    formula=(
                        r"$c = \max\left(0,"
                        r" \frac{145 - front\_sensor}{145}\right)$"
                    ),
                    explanation=(
                        "As the clear road in front gets shorter, this term rises. It is "
                        "a simple way to make the agent brake earlier when the visible "
                        "track is closing quickly."
                    ),
                ),
                FormulaNote(
                    title="Stability Mode",
                    formula=(
                        r"$u = |\theta| > 0.72$"
                        "\n"
                        r"$\mathrm{or}\ |v_y| > 20$"
                        "\n"
                        r"$\mathrm{or}\ s_{sideways} = \mathrm{high}$"
                    ),
                    explanation=(
                        "This is not a probability model. It is a set of safety thresholds "
                        "that tell the controller when recovery should override normal "
                        "cornering behaviour."
                    ),
                ),
            ),
            "code_snippets": (
                CodeSnippet(
                    title="Choosing the most useful road sensor",
                    source="src/agents/rule_based_agent.py - get_best_sensor",
                    explanation=(
                        "The middle sensor points straight ahead. Nearby sensors are scored "
                        "so the driver can aim toward open road without chasing extreme edges."
                    ),
                    code=(
                        "def get_best_sensor(state):\n"
                        "    track = state[\"track\"]\n"
                        "    front = track[9]\n"
                        "    candidate_indices = range(4, 15) if front < 65 else range(6, 13)\n"
                        "    best_index = 9\n"
                        "    best_score = front * 1.12\n"
                        "    for index in candidate_indices:\n"
                        "        offset = abs(index - 9)\n"
                        "        centre_penalty = 1.0 - offset * (0.035 if front < 65 else 0.10)\n"
                        "        score = track[index] * centre_penalty\n"
                        "        if score > best_score:\n"
                        "            best_score = score\n"
                        "            best_index = index\n"
                        "    return best_index"
                    ),
                ),
                CodeSnippet(
                    title="Classifying the road ahead",
                    source="src/agents/rule_based_agent.py - classify_section",
                    explanation=(
                        "The driver turns sensor readings into a human-readable road section. "
                        "That section then controls the target speed."
                    ),
                    code=(
                        "def classify_section(front, planned_angle_abs, best_distance):\n"
                        "    closing = max(0.0, (145.0 - front) / 145.0)\n"
                        "    angle_factor = planned_angle_abs / 19.0\n"
                        "    distance_factor = max(0.0, (125.0 - best_distance) / 125.0)\n"
                        "    severity = 0.40 * angle_factor + 0.45 * closing + 0.15 * distance_factor\n"
                        "    if front > 155 and planned_angle_abs <= 7:\n"
                        "        section = \"STRAIGHT\"\n"
                        "    elif severity < 0.46:\n"
                        "        section = \"SHALLOW\"\n"
                        "    elif severity < 0.88:\n"
                        "        section = \"TIGHT\"\n"
                        "    else:\n"
                        "        section = \"HAIRPIN\"\n"
                        "    return section, severity"
                    ),
                ),
            ),
            "math_notes": (
                "The rule-based controller is mostly thresholds and weighted scores, not learned weights.",
                "The front road sensor is track[9]; sensors around it represent increasingly angled views left and right.",
                (
                    "Section severity combines angle demand, closing distance, and best visible "
                    "distance: severity = 0.40 * angle + 0.45 * closing + 0.15 * distance."
                ),
                (
                    "Target speed is selected from section labels, so a HAIRPIN produces a "
                    "much lower speed target than a STRAIGHT."
                ),
                (
                    "Stability checks use absolute angle and sideways speed to decide when "
                    "saving the car matters more than attacking the road."
                ),
            ),
            "key_takeaways": (
                "This driver is easy to explain because it is built from visible rules.",
                "It can run on more tracks because it does not need a saved map.",
                "Its weakness is anticipation: it reacts to what the sensors can see now.",
                "Telemetry should show clear cause-and-effect between sensor danger and braking.",
            ),
            "input_signals": (
                "Road sensors showing left, centre, and right space.",
                "Speed and sideways movement.",
                "Track position and angle relative to the road.",
                "Wheel spin information for stability checks.",
                "Damage and off-track state.",
            ),
            "strengths": (
                "Can run on more tracks because it does not need a racing line.",
                "Good for explaining simple cause-and-effect driving logic.",
                "Useful baseline for showing why map knowledge can help.",
            ),
            "failure_signs": (
                "Late braking when the visible road changes quickly.",
                "Over-correcting after running wide.",
                "Oscillating steering when several rules compete.",
                "Lower ceiling than a track-specific planner on a known map.",
            ),
        }

    if agent_type in {"dyna_q_learning", "dyna_q_finalised", "dyna_q"}:
        is_finalised = agent_type == "dyna_q_finalised"
        return {
            "badge": (
                "Finalised model-based reinforcement learner"
                if is_finalised
                else "Live model-based reinforcement learner"
            ),
            "headline": (
                "This driver learns a racing policy from reward. Dyna-Q updates "
                "from real driving experience, then replays learned transitions "
                "internally so useful behaviour spreads faster through the Q-table."
            ),
            "overview": (
                "It does not load a precomputed racing line or copy another driver.",
                (
                    "The agent converts TORCS telemetry into compact state bins: "
                    "lap section, speed, road position, heading, side speed, front "
                    "sensor danger, and the best visible road direction."
                ),
                (
                    "It chooses from a small set of steering, throttle, coast, and "
                    "brake actions. The action space is deliberately discrete so a "
                    "classical reinforcement-learning method can learn from scratch."
                ),
                (
                    "Learning mode explores and updates the Q-table while driving. "
                    "Finalised mode loads a saved policy, switches exploration off, "
                    "and always chooses the best-known action."
                ),
            ),
            "decision_steps": (
                "Read speed, angle, side speed, track position, lap distance, and road sensors.",
                "Encode those signals into a compact discrete state.",
                "Choose an action using epsilon-greedy exploration in learning mode.",
                "Turn the selected action into smoothed steering, throttle, brake, and gear controls.",
                "On the next telemetry frame, calculate reward from progress, speed, stability, and safety.",
                "Update the Q-table from the real transition.",
                "Replay stored transitions internally for extra Dyna-Q planning updates.",
            ),
            "algorithm_summary": (
                "Dyna-Q combines direct reinforcement learning with lightweight planning.",
                (
                    "A normal Q-learning update learns from the transition that just "
                    "happened: state, action, reward, and next state."
                ),
                (
                    "Dyna-Q also stores that transition in a simple learned model. "
                    "After the real update, it samples old model entries and performs "
                    "additional imagined updates."
                ),
                (
                    "Those planning updates are why Dyna-Q is a good fit for a time "
                    "constrained TORCS project: every real driving step can produce "
                    "several learning updates."
                ),
                (
                    "The finalised agent uses the same Q-values but disables learning "
                    "and exploration, making behaviour repeatable for demonstration runs."
                ),
            ),
            "pseudocode": (
                "Start with an empty Q-table and an empty transition model.",
                "Read the current TORCS telemetry and convert it into a state.",
                "Select an action: explore sometimes, otherwise choose the highest Q-value.",
                "Send the selected steering/throttle/brake command to TORCS.",
                "Read the next telemetry frame.",
                "Reward progress and stable speed; penalise damage, off-track, sliding, and poor heading.",
                "Apply the Q-learning update to Q(state, action).",
                "Store the transition in the model.",
                "Sample stored transitions and replay several planning updates.",
                "Save the learned policy so finalised mode can run greedily later.",
            ),
            "formula_notes": (
                FormulaNote(
                    title="Q-Learning Update",
                    formula=(
                        r"$Q(s,a) \leftarrow Q(s,a) + \alpha"
                        r"[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$"
                    ),
                    explanation=(
                        "The action value moves toward the reward plus the discounted "
                        "value of the best next action. This is the real driving update."
                    ),
                ),
                FormulaNote(
                    title="Dyna-Q Model Replay",
                    formula=r"$model(s,a) = (r, s')$",
                    explanation=(
                        "After a real transition is observed, the agent stores what "
                        "happened. Planning steps sample those stored transitions and "
                        "apply the same Q update without needing another TORCS frame."
                    ),
                ),
                FormulaNote(
                    title="Epsilon-Greedy Choice",
                    formula=(
                        r"$a = random\ action \quad \mathrm{with\ probability}\ \epsilon$"
                        "\n"
                        r"$a = \arg\max_a Q(s,a) \quad \mathrm{otherwise}$"
                    ),
                    explanation=(
                        "Learning mode keeps some exploration so the agent can discover "
                        "better behaviour. Finalised mode sets epsilon to zero."
                    ),
                ),
                FormulaNote(
                    title="Progress Reward",
                    formula=r"$reward \propto \Delta distance + speed \cdot \cos(|angle|)$",
                    explanation=(
                        "Progress is valuable, but only when the car is reasonably "
                        "aligned with the road. Sliding, damage, and off-track states "
                        "reduce the reward."
                    ),
                ),
            ),
            "code_snippets": (
                CodeSnippet(
                    title="Real update plus planning replay",
                    source="src/agents/dyna_q_agent.py - DynaQBaseAgent.learn_from_transition",
                    explanation=(
                        "One real transition updates the Q-table, then the saved model "
                        "is sampled for extra replay updates."
                    ),
                    code=(
                        "def learn_from_transition(self, state, action, reward, next_state, terminal=False):\n"
                        "    self._q_update(state, action, reward, next_state, terminal)\n"
                        "    self.model[self._transition_key(state, action)] = DynaQTransition(\n"
                        "        next_state=next_state,\n"
                        "        reward=reward,\n"
                        "        terminal=terminal,\n"
                        "    )\n"
                        "    for _ in range(self.planning_steps):\n"
                        "        key = self._random.choice(model_keys)\n"
                        "        planned_state, planned_action = self._parse_transition_key(key)\n"
                        "        transition = self.model[key]\n"
                        "        self._q_update(planned_state, planned_action, transition.reward, transition.next_state, transition.terminal)"
                    ),
                ),
                CodeSnippet(
                    title="Learning vs finalised mode",
                    source="src/agents/dyna_q_agent.py - DynaQLearningAgent / DynaQFinalisedAgent",
                    explanation=(
                        "The learning agent saves Q-values and keeps exploration on. "
                        "The finalised agent loads a saved policy and turns learning off."
                    ),
                    code=(
                        "class DynaQLearningAgent(DynaQBaseAgent):\n"
                        "    learning_enabled = True\n"
                        "    save_policy = True\n\n"
                        "class DynaQFinalisedAgent(DynaQBaseAgent):\n"
                        "    learning_enabled = False\n"
                        "    epsilon = 0.0\n"
                        "    planning_steps = 0"
                    ),
                ),
            ),
            "math_notes": (
                "Dyna-Q is still classical reinforcement learning; it does not use a neural network.",
                "The Q-table is sparse: only state-action pairs the agent has seen need stored values.",
                "The learned model stores observed transitions, not a hand-written racing line.",
                "Planning updates make training more sample-efficient because one TORCS step can produce many Q-value updates.",
                "Finalised mode is the same policy with exploration and learning disabled.",
            ),
            "key_takeaways": (
                "The agent learns from reward rather than from a precomputed racing line.",
                "Discrete actions keep the problem small enough for a classical RL method.",
                "Planning replay is the reason Dyna-Q should improve faster than a plain one-step learner.",
                "A strong finalised policy still needs training time before it can race cleanly.",
            ),
            "input_signals": (
                "Distance around the lap.",
                "Speed, sideways speed, and car angle.",
                "Track position: centre, left edge, or right edge.",
                "Road sensors showing open space ahead.",
                "Damage and off-track signals for penalty rewards.",
            ),
            "strengths": (
                "Learns from scratch without a racing-line file.",
                "More sample-efficient than a plain Q-learning-only loop.",
                "Easy to explain because Q-values, rewards, and replay updates are visible.",
                "Finalised mode can run a saved policy with repeatable behaviour.",
            ),
            "failure_signs": (
                "The policy file is missing when finalised mode is selected.",
                "Too much exploration can cause unstable early runs.",
                "Sparse rewards or too many state bins can slow learning.",
                "It may need multiple training runs before it approaches engineered agents.",
            ),
        }

    if agent_type == "random":
        return {
            "badge": "Baseline random driver",
            "headline": (
                "This driver is an experimental baseline: it samples random actions "
                "so planned or rule-based agents can be compared against a weak control."
            ),
            "overview": (
                "It is not trying to learn the track, interpret sensors, or follow a strategy.",
                (
                    "Its value is methodological rather than competitive. It gives the "
                    "project a lower-bound comparison for judging whether the other "
                    "agents are genuinely adding intelligence."
                ),
                (
                    "In telemetry, random behaviour usually appears as noisy steering, "
                    "inconsistent throttle, poor repeatability, and early off-track moments."
                ),
            ),
            "decision_steps": (
                "Receive the current observation from the simulator.",
                "Sample a random steering and throttle action.",
                "Send that action to TORCS without planning ahead.",
                "Record the result as a baseline behaviour.",
                "Use the run as a low-skill comparison point.",
            ),
            "algorithm_summary": (
                "The random driver is intentionally simple and deliberately unskilled.",
                (
                    "It samples steering and throttle from fixed ranges and sends those "
                    "values to TORCS without understanding the track."
                ),
                (
                    "Because a seed can be reused, its randomness can still be made "
                    "repeatable enough for a fair baseline comparison."
                ),
                (
                    "For a dissertation, this agent is useful because it demonstrates "
                    "that better performance is not automatic: the other controllers "
                    "should clearly outperform this non-reasoning baseline."
                ),
            ),
            "pseudocode": (
                "Create a seeded random-number generator.",
                "For each simulator step, sample steering between -0.4 and 0.4.",
                "Sample throttle between 0.3 and 0.7.",
                "Return those two values as the action.",
                "Reset by reusing the same seed so the baseline can be repeated.",
            ),
            "formula_notes": (
                FormulaNote(
                    title="Uniform Steering Sample",
                    formula=r"$steering \sim \mathcal{U}(-0.4, 0.4)$",
                    explanation=(
                        "Every steering value inside this interval is equally likely. "
                        "The sample is bounded so the car does not constantly request "
                        "full steering lock, but it is still not a planned decision."
                    ),
                ),
                FormulaNote(
                    title="Uniform Throttle Sample",
                    formula=r"$throttle \sim \mathcal{U}(0.3, 0.7)$",
                    explanation=(
                        "Throttle is also sampled uniformly from a fixed interval. The "
                        "agent does not check whether it is on a straight, entering a "
                        "corner, or leaving the track."
                    ),
                ),
                FormulaNote(
                    title="Repeatable Randomness",
                    formula=r"$random\_sequence = \mathrm{Random}(seed)$",
                    explanation=(
                        "A fixed seed produces the same sequence again, which helps make "
                        "baseline comparisons reproducible even though the behaviour is random."
                    ),
                ),
            ),
            "code_snippets": (
                CodeSnippet(
                    title="Random baseline action",
                    source="src/agents/random_agent.py - RandomAgent.act",
                    explanation=(
                        "The ranges deliberately avoid full steering lock and full throttle, "
                        "but the agent still has no idea where the road is."
                    ),
                    code=(
                        "def act(self, _observation, _telemetry=None):\n"
                        "    steering = self._random.uniform(-0.4, 0.4)\n"
                        "    throttle = self._random.uniform(0.3, 0.7)\n"
                        "    return [steering, throttle]"
                    ),
                ),
            ),
            "math_notes": (
                "This baseline uses uniform random sampling rather than a driving model.",
                "Every value inside the allowed steering range has equal probability.",
                "The seed makes a random run repeatable, which is useful for comparisons.",
                "There is no target speed, racing line, reward model, or recovery equation.",
            ),
            "key_takeaways": (
                "This is a sanity-check baseline, not a serious racing strategy.",
                "A useful agent should beat it clearly and consistently.",
                "Noisy telemetry is expected because actions are not tied to road context.",
            ),
            "input_signals": (
                "Simulator observation is received, but not meaningfully interpreted.",
                "No racing-line file.",
                "No long-term memory of the lap.",
                "No deliberate recovery strategy.",
                "No track-specific setup.",
            ),
            "strengths": (
                "Very simple baseline for experiments.",
                "Useful for proving that trained or engineered agents add value.",
                "Easy to identify in comparison charts because behaviour is noisy.",
            ),
            "failure_signs": (
                "Frequent off-track exits.",
                "Sudden steering changes with no clear reason.",
                "Throttle and braking do not match corner demands.",
                "Poor repeatability between runs.",
            ),
        }

    return {
        "badge": "Discovered project agent",
        "headline": "This driver was discovered from the project configuration.",
        "overview": (
            "No dedicated education profile exists yet for this agent type.",
            "The dashboard can still show its metadata, compatible tracks, and runs.",
        ),
        "decision_steps": (
            "Load the agent class.",
            "Receive simulator observations.",
            "Return steering, throttle, brake, or gear commands.",
            "Record telemetry and run results.",
            "Compare the outcome with other saved runs.",
        ),
        "algorithm_summary": (
            "This discovered agent does not yet have a custom explanation profile.",
            "The dashboard can still describe its metadata and inspect its telemetry.",
        ),
        "pseudocode": (
            "Load the agent.",
            "Read simulator observations.",
            "Return an action.",
            "Record telemetry.",
            "Compare the saved result.",
        ),
        "formula_notes": (),
        "code_snippets": (),
        "math_notes": (
            "No dedicated mathematics notes are available for this agent type yet.",
        ),
        "key_takeaways": (
            "Add a profile entry to explain this agent in the same format as the others.",
        ),
        "input_signals": (
            "Simulator telemetry.",
            "Agent-specific configuration.",
        ),
        "strengths": (
            "Can be inspected through saved telemetry and comparisons.",
        ),
        "failure_signs": (
            "Unexpected off-track, stuck, or crashed outcomes.",
        ),
    }


def _metadata(agent: AgentOption) -> tuple[tuple[str, str], ...]:
    return (
        ("Agent type", agent.agent_type),
        ("Version", agent.version),
        (
            "Control style",
            (
                "Full control: steering, throttle, brake, and gear"
                if agent.uses_full_control
                else "Gym action output"
            ),
        ),
        (
            "Track dependency",
            (
                "Needs a racing-line file"
                if agent.requires_racing_line
                else "No racing-line file required"
            ),
        ),
        ("Target laps", _format_optional_int(agent.target_laps)),
        ("Maximum steps", _format_optional_int(agent.max_steps)),
    )


def _track_context(
    agent: AgentOption,
    tracks: list[TrackOption],
) -> tuple[str, ...]:
    compatible_tracks = compatible_tracks_for_agent(agent, tracks)
    if not tracks:
        return ("No TORCS tracks were discovered.",)

    if agent.requires_racing_line:
        if not compatible_tracks:
            return (
                "This agent needs racing-line files, but none were discovered.",
                "Generate a racing line before using it on a new map.",
            )
        return (
            (
                "Racing-line tracks: "
                f"{_format_track_list(compatible_tracks)}."
            ),
            (
                f"{len(compatible_tracks)} of {len(tracks)} discovered tracks "
                "currently have matching racing-line data."
            ),
            "New tracks need their own racing-line JSON before this agent is a fair fit.",
        )

    return (
        f"Compatible tracks: {_format_track_list(compatible_tracks)}.",
        (
            f"This agent can be selected for {len(compatible_tracks)} of "
            f"{len(tracks)} discovered tracks."
        ),
        "Performance can still vary by track shape, even without a map dependency.",
    )


def _format_track_list(tracks: list[TrackOption]) -> str:
    if not tracks:
        return "none"

    labels = [track.label for track in tracks[:6]]
    remaining = len(tracks) - len(labels)
    if remaining > 0:
        labels.append(f"{remaining} more")
    return ", ".join(labels)


def _format_optional_int(value: int | None) -> str:
    return "--" if value is None else f"{value:,}"
