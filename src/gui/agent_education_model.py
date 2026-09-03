from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .novice_education import build_novice_agent_guide
    from .project_discovery import AgentOption, TrackOption, compatible_tracks_for_agent
except ImportError:
    from novice_education import build_novice_agent_guide
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
    quick_facts: tuple[tuple[str, str], ...]
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
    language: str = "en",
) -> AgentEducationProfile:
    template = _template_for(agent.agent_type)
    if language == "cy":
        template = _welsh_template(agent.agent_type, template)
    return AgentEducationProfile(
        title=agent.name,
        badge=template["badge"],
        headline=template["headline"],
        quick_facts=_quick_facts(agent.agent_type, language),
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
        track_context=_track_context(agent, tracks, language),
        metadata=_metadata(agent, language),
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
                (
                    "The live dashboard exposes the current state, chosen action, "
                    "reward, TD error, epsilon, Q-table size, and planning replay "
                    "count so the learning process is visible while TORCS runs."
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
                "Show the current reward, TD error, Q-table size, and replay count in the live dashboard.",
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

    if agent_type == "td3_scratch":
        return {
            "badge": "Reward-only neural reinforcement learner",
            "headline": (
                "This driver is a from-scratch TD3 continuous-control agent: a neural "
                "policy directly controls steering and longitudinal intent from TORCS telemetry."
            ),
            "overview": (
                "It does not load Agent 3, copy a teacher, use behaviour cloning, or imitate a racing line.",
                (
                    "The policy sees raw driving signals such as speed, heading, road sensors, "
                    "track position, wheel-spin balance, damage, lap time, and distance phase."
                ),
                (
                    "TD3 is deterministic at deployment, while training adds exploration noise "
                    "and learns from a replay buffer of reward-labelled transitions."
                ),
                (
                    "The curriculum makes the problem tractable by starting with launch and "
                    "stability, then moving to first-corner survival, sector progress, and full-lap attempts."
                ),
            ),
            "decision_steps": (
                "Read speed, side speed, angle, track position, distance, damage, wheel spin, and road sensors.",
                "Normalise those raw values into the TD3 observation vector.",
                "Ask the actor network for continuous steering and longitudinal controls.",
                "Decode its versioned action contract into TORCS controls and apply automatic gear shifting.",
                "During training, score the next telemetry frame with progress, stability, safety, and milestone rewards.",
                "Store transitions in replay memory and update twin critics plus the delayed actor policy.",
                "Save best-distance checkpoints separately from best-reward checkpoints.",
            ),
            "algorithm_summary": (
                "TD3 stands for Twin Delayed Deep Deterministic Policy Gradient.",
                (
                    "The actor is a neural controller. It maps the current observation directly "
                    "to a continuous action instead of selecting from a hand-written action list."
                ),
                (
                    "Two critic networks estimate action value. Taking the smaller critic target "
                    "helps reduce the value overestimation that can destabilise deterministic actor-critic learning."
                ),
                (
                    "The actor is updated less frequently than the critics. This delayed update "
                    "lets value estimates settle before the policy follows them."
                ),
                (
                    "The reward is engineered around forward progress and survival, not around "
                    "matching a saved route. Curriculum stages change the episode objective while keeping learning from scratch."
                ),
            ),
            "pseudocode": (
                "Start with random actor and critic network weights.",
                "Reset TORCS into the current curriculum stage.",
                "Observe raw telemetry and build the normalised feature vector.",
                "Actor outputs continuous controls; exploration noise is added during training.",
                "Send controls to TORCS with automatic gear selection.",
                "Reward forward progress, clean alignment, new furthest distance, and stage completion.",
                "Penalise off-track, late braking, sliding, reversing, stuck behaviour, and damage.",
                "Store observation, action, reward, next observation, and done flag in replay memory.",
                "Update twin critics from replay; periodically update the actor and target networks.",
            ),
            "formula_notes": (
                FormulaNote(
                    title="Deterministic Actor",
                    formula=r"$a_t = \mu_\theta(s_t)$",
                    explanation=(
                        "The actor network maps the current telemetry state directly to a "
                        "continuous control vector. At runtime no random action sampling is needed."
                    ),
                ),
                FormulaNote(
                    title="Twin Critic Target",
                    formula=(
                        r"$y = r + \gamma \min_i Q_{\phi_i'}"
                        r"(s', \mu_{\theta'}(s') + \epsilon)$"
                    ),
                    explanation=(
                        "TD3 uses two target critics and keeps the smaller value estimate. "
                        "The clipped target noise smooths the critic around nearby actions."
                    ),
                ),
                FormulaNote(
                    title="Scratch Progress Reward",
                    formula=r"$R \approx w_p\Delta d + w_m\Delta d_{best} - P_{risk}$",
                    explanation=(
                        "The reward emphasises forward progress and each new furthest distance. "
                        "Risk penalties come from off-track sensors, angle, slide, damage, and late braking."
                    ),
                ),
            ),
            "code_snippets": (
                CodeSnippet(
                    title="Raw telemetry observation",
                    source="src/agents/td3_agent.py - build_td3_observation",
                    explanation=(
                        "The observation is built from TORCS telemetry only. There is no racing-line lookup "
                        "and no teacher action in the feature vector."
                    ),
                    code=(
                        "features = [\n"
                        "    normalise_signed(speedX, 240.0),\n"
                        "    normalise_signed(angle, 1.0),\n"
                        "    normalise_signed(trackPos, 1.4),\n"
                        "    normalise_sensor(front_sensor),\n"
                        "    normalise_sensor(min_track_sensor),\n"
                        "    distance_phase,\n"
                        "    previous_steer,\n"
                        "    previous_accel,\n"
                        "    previous_brake,\n"
                        "]"
                    ),
                ),
                CodeSnippet(
                    title="Direct continuous controls",
                    source="src/agents/td3_agent.py - decode_td3_action",
                    explanation=(
                        "TD3 owns steering and the throttle-versus-brake decision. The only engineered "
                        "control left outside the learning problem is automatic gear selection. Legacy "
                        "champions retain their original three-head decoder during continuation."
                    ),
                    code=(
                        "if legacy_contract:\n"
                        "    controls = decode_legacy_td3_action(raw)\n"
                        "else:\n"
                        "    controls = decode_td3_action(raw)\n"
                        "gear = shift_gears(speed_kmh)"
                    ),
                ),
                CodeSnippet(
                    title="Curriculum reward",
                    source="scripts/train_td3_agent.py - calculate_td3_reward",
                    explanation=(
                        "The reward does not compare against a racing line. It scores progress, "
                        "new distance milestones, stability, late braking risk, and failures."
                    ),
                    code=(
                        "reward = progress * stage.progress_weight\n"
                        "reward += new_distance * stage.milestone_weight\n"
                        "reward -= off_track_penalty\n"
                        "reward -= late_braking_penalty\n"
                        "if stage_success:\n"
                        "    reward += stage.success_reward"
                    ),
                ),
            ),
            "math_notes": (
                "The actor and critics start from random neural-network weights.",
                "Replay buffer samples break the tight correlation between consecutive TORCS frames.",
                "Best-distance checkpointing matters because shaped reward can improve before full laps appear.",
                "Distance phase is map position information, not a racing-line target or imitation signal.",
                "Automatic gear shifting narrows the learning problem to steering and longitudinal control.",
            ),
            "key_takeaways": (
                "This is the clean neural DRL showcase agent.",
                "It learns from reward feedback rather than copied actions.",
                "Curriculum learning makes from-scratch racing less brittle.",
                "Training logs, reward curves, and best-distance checkpoints are dissertation evidence.",
            ),
            "input_signals": (
                "Raw TORCS speed, angle, side speed, and track position.",
                "Nineteen road range sensors.",
                "Damage, lap time, distance around the lap, and wheel spin.",
                "Previous control output for smooth policy context.",
                "No racing-line file and no teacher action.",
            ),
            "strengths": (
                "Direct neural continuous control.",
                "No dependence on a precomputed racing line.",
                "Replay-buffer learning is well suited to long-horizon control.",
                "Best-distance checkpointing captures partial progress during long training.",
            ),
            "failure_signs": (
                "Early training may produce stuck launches or immediate off-track exits.",
                "Too little exploration noise can prevent discovering useful throttle and steering combinations.",
                "Too much reward for speed can cause late braking and corner entry crashes.",
                "It may need overnight or multi-day training before it becomes demonstrably competent.",
            ),
        }

    if agent_type == "n_step_td3":
        return _n_step_td3_template(sensor_only=False)

    if agent_type == "sensor_n_step_td3":
        return _n_step_td3_template(sensor_only=True)

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


_WELSH_FORMULA_GUIDES = {
    "Normalised Track Position": (
        "Safle Trac wedi'i Normaleiddio",
        "Mae 0 yn ganol y ffordd, -1 yn ymyl chwith a +1 yn ymyl dde. Mae'r hafaliad yn trosi'r raddfa hon yn bellter ochr mewn metrau.",
    ),
    "Racing Point Projection": (
        "Taflunio Pwynt Rasio",
        "Mae'r pwynt yn dechrau ar ganol y trac ac yn symud ar draws lled y ffordd i greu'r llwybr rasio bwriedig.",
    ),
    "Lap-Distance Interpolation": (
        "Rhyngosod Pellter Lap",
        "Mae rhyngosod yn llenwi'r bwlch rhwng dau bwynt llwybr fel bod y targed yn newid yn llyfn, gan gynnwys dros y llinell gychwyn.",
    ),
    "Speed-Dependent Lookahead": (
        "Edrych Ymlaen yn ôl Cyflymder",
        "Wrth i'r car fynd yn gyflymach, mae'n edrych ymhellach ymlaen. Mae'r terfynau'n atal golwg sy'n rhy fyr neu'n rhy bell.",
    ),
    "Smooth Merge Onto The Line": (
        "Ymuno'n Llyfn â'r Llinell",
        "Mae'r gromlin hon yn symud y car tuag at y llwybr fesul tipyn ar ôl lansio, yn hytrach na neidio i'r ochr.",
    ),
    "Curvature-Limited Speed": (
        "Cyflymder wedi'i Gyfyngu gan Grymedd",
        "Mae cromlin fwy yn golygu cornel fwy tynn. Felly mae'r cyflymder diogel yn gostwng wrth i'r tro fynd yn fwy tynn.",
    ),
    "Line Error": (
        "Gwall y Llinell",
        "Mae'r gwahaniaeth rhwng safle'r car a'r targed yn dweud i'r rheolydd pa ochr sydd angen cywiriad.",
    ),
    "Sensor Score": (
        "Sgôr Synhwyrydd",
        "Mae'r sgôr yn cyfuno faint o ffordd agored sydd i'w gweld ac ongl y synhwyrydd er mwyn ffafrio cyfeiriad defnyddiol.",
    ),
    "Visible-Corner Severity": (
        "Pa Mor Lem yw'r Gornel Weladwy",
        "Mae'r gymhareb rhwng y ffordd syth ymlaen a'r ochr fwy agored yn amcangyfrif pa mor ofalus y dylai'r car fod.",
    ),
    "Closing Factor": (
        "Ffactor Cau",
        "Mae darlleniad blaen sy'n crebachu yn dangos bod y car yn cyrraedd rhwystr neu gornel, felly dylai baratoi'n gynt.",
    ),
    "Stability Mode": (
        "Mesur Sefydlogrwydd",
        "Mae symudiad i'r ochr, llithro olwynion ac ongl y car yn ffurfio mesur syml o ba mor ansefydlog yw'r car.",
    ),
    "Q-Learning Update": (
        "Diweddariad Q-Learning",
        "Mae'r hen sgôr yn symud tuag at y wobr newydd ynghyd â'r canlyniad gorau a ddisgwylir o'r sefyllfa nesaf.",
    ),
    "Dyna-Q Model Replay": (
        "Ailchwarae Model Dyna-Q",
        "Ar ôl dysgu o gam go iawn, mae'r asiant yn ail-ymarfer profiadau o'i gof er mwyn defnyddio data'n fwy effeithlon.",
    ),
    "Epsilon-Greedy Choice": (
        "Dewis Epsilon-Greedy",
        "Fel arfer dewisir y weithred â'r sgôr uchaf, ond weithiau caiff dewis arall ei brofi er mwyn archwilio.",
    ),
    "Progress Reward": (
        "Gwobr Cynnydd",
        "Mae'r wobr yn ffafrio symud ymlaen, cyflymder defnyddiol a sefydlogrwydd, ac yn cosbi methiant.",
    ),
    "Deterministic Actor": (
        "Actor Penderfynol",
        "Mae'r actor yn troi cyflwr y car yn rheolyddion parhaus. Wrth werthuso, mae'r un mewnbwn yn cynhyrchu'r un dewis.",
    ),
    "Twin Critic Target": (
        "Targed Dau Feirniad",
        "Mae TD3 yn defnyddio'r amcangyfrif lleiaf o ddau feirniad er mwyn lleihau hyder gormodol mewn gweithred fregus.",
    ),
    "Scratch Progress Reward": (
        "Gwobr Cynnydd o'r Dechrau",
        "Mae cynnydd ymlaen a chwblhau lap yn dda; mae gyrru oddi ar y trac neu fethu yn wael. Nid yw'r hafaliad yn darparu gweithred i'w chopïo.",
    ),
    "Uniform Steering Sample": (
        "Sampl Llywio Unffurf",
        "Mae pob gwerth llywio o fewn y terfynau yn cael yr un siawns. Nid oes penderfyniad deallus yma.",
    ),
    "Uniform Throttle Sample": (
        "Sampl Cyflymu Unffurf",
        "Dewisir cyflymu ar hap o ystod sefydlog, heb ddefnyddio'r ffordd neu gyflwr y car.",
    ),
    "Repeatable Randomness": (
        "Hapusrwydd Ailadroddadwy",
        "Mae hedyn sefydlog yn ail-greu'r un dilyniant hap, sy'n gwneud cymariaethau prawf yn decach.",
    ),
    "History Observation": (
        "Arsylwad â Hanes",
        "Mae tri ffrâm telemetreg a thri gweithred flaenorol yn rhoi cipolwg byr o symudiad diweddar i'r actor.",
    ),
    "Three-Step Critic Target": (
        "Targed Beirniad Tri Cham",
        "Mae'r targed yn defnyddio tair gwobr go iawn cyn gofyn i'r rhwydwaith amcangyfrif beth sy'n digwydd wedyn.",
    ),
    "Target Policy Smoothing": (
        "Llyfnhau'r Polisi Targed",
        "Mae sŵn bach wedi'i glipio yn ystod hyfforddi'r beirniad yn ffafrio gweithredoedd sy'n parhau'n dda ar ôl newid bach.",
    ),
    "Sensor Stability Reward": (
        "Gwobr Sefydlogrwydd Synwyryddion",
        "Mae cyflymder ymlaen yn werthfawr pan fo'r car wedi'i alinio â'r ffordd. Mae cosbau bach am ymyl y trac a llywio sydyn yn helpu sefydlogrwydd heb bennu llwybr.",
    ),
    "Racing-Line Velocity Reward": (
        "Gwobr Cyflymder y Llinell Rasio",
        "Mae'r wobr yn ffafrio symud ymlaen yn agos at y llwybr parod. Mae'n sgorio canlyniad gweithred niwral; nid yw'n rhoi gweithred athro.",
    ),
    "Delayed Actor Objective": (
        "Amcan Actor wedi'i Oedi",
        "Mae'r actor yn newid yn llai aml na'r beirniaid ac yn dysgu dewis rheolyddion y mae'r beirniad cyntaf yn rhagweld fydd yn ddefnyddiol.",
    ),
}


_WELSH_CODE_GUIDES = {
    "Racing-line lookup": "Mae'r cod yn dod o hyd i'r ddau bwynt llwybr o amgylch y car ac yn rhyngosod rhyngddynt.",
    "Previewing the route ahead": "Mae'r rheolydd yn samplu sawl pwynt o'i flaen er mwyn paratoi llywio a chyflymder cyn y gornel.",
    "Turning the plan into controls": "Mae'r cod yn trosi'r targed llwybr a chyflymder yn llywio, cyflymu a brecio.",
    "Choosing the most useful road sensor": "Mae'r cod yn graddio'r synwyryddion ac yn dewis y cyfeiriad sy'n cynnig y ffordd fwyaf defnyddiol.",
    "Classifying the road ahead": "Mae'r rheolau'n dosbarthu'r ffordd weladwy er mwyn dewis ymateb addas.",
    "Real update plus planning replay": "Mae un profiad go iawn yn diweddaru'r tabl, ac yna defnyddir profiadau o'r cof ar gyfer ymarfer ychwanegol.",
    "Learning vs finalised mode": "Mae'r modd dysgu yn archwilio ac yn newid y tabl; mae'r modd terfynol yn defnyddio'r dewis gorau sydd wedi'i gadw.",
    "Raw telemetry observation": "Mae'r cod yn graddio telemetreg TORCS i mewnbynnau sefydlog ar gyfer y rhwydwaith.",
    "Direct continuous controls": "Mae allbynnau'r actor yn dod yn llywio a bwriad pedal parhaus, gyda dewis gêr awtomatig.",
    "Curriculum reward": "Mae'r cod yn cyfrifo gwobr o gynnydd a diogelwch yn ystod hyfforddiant hanesyddol Agent 6.",
    "Random baseline action": "Mae'r llinell sylfaen yn samplu rheolyddion ar hap er mwyn darparu cymhariaeth isaf syml.",
    "Telemetry and history state": "Mae'r contract yn cyfuno telemetreg bresennol, hanes byr a gweithredoedd blaenorol mewn un mewnbwn.",
    "Twin-critic learning target": "Mae'r cod yn ychwanegu sŵn bach at y targed ac yn defnyddio'r gwerth lleiaf o'r ddau feirniad.",
    "Delayed neural policy update": "Mae'r actor yn newid ar gamau oedi er mwyn dilyn gweithredoedd â gwerth hirdymor uwch.",
}

_WELSH_CODE_TITLES = {
    "Racing-line lookup": "Chwilio'r llinell rasio",
    "Previewing the route ahead": "Rhagweld y llwybr o'ch blaen",
    "Turning the plan into controls": "Troi'r cynllun yn rheolyddion",
    "Choosing the most useful road sensor": "Dewis y synhwyrydd ffordd mwyaf defnyddiol",
    "Classifying the road ahead": "Dosbarthu'r ffordd o'ch blaen",
    "Real update plus planning replay": "Diweddariad go iawn ac ailchwarae cynllunio",
    "Learning vs finalised mode": "Modd dysgu a modd terfynol",
    "Raw telemetry observation": "Arsylwad telemetreg crai",
    "Direct continuous controls": "Rheolyddion parhaus uniongyrchol",
    "Curriculum reward": "Gwobr y cwricwlwm",
    "Random baseline action": "Gweithred llinell sylfaen ar hap",
    "Telemetry and history state": "Cyflwr telemetreg a hanes",
    "Twin-critic learning target": "Targed dysgu'r ddau feirniad",
    "Delayed neural policy update": "Diweddariad polisi niwral wedi'i oedi",
}


def _welsh_template(agent_type: str, template: dict[str, Any]) -> dict[str, Any]:
    guide = build_novice_agent_guide(agent_type, language="cy")
    localised = dict(template)
    localised.update(
        {
            "badge": guide.badge,
            "headline": guide.headline,
            "overview": guide.driving_story,
            "decision_steps": guide.decision_steps,
            "algorithm_summary": (*guide.learning_story, *guide.key_takeaways),
            "pseudocode": guide.decision_steps,
            "formula_notes": tuple(
                _welsh_formula_note(note) for note in template["formula_notes"]
            ),
            "code_snippets": tuple(
                _welsh_code_snippet(snippet) for snippet in template["code_snippets"]
            ),
            "math_notes": (*guide.learning_story, *guide.key_takeaways),
            "key_takeaways": guide.key_takeaways,
            "input_signals": guide.input_signals,
            "strengths": guide.strengths,
            "failure_signs": guide.failure_signs,
        }
    )
    return localised


def _welsh_formula_note(note: FormulaNote) -> FormulaNote:
    title, explanation = _WELSH_FORMULA_GUIDES.get(
        note.title,
        (
            note.title,
            "Mae'r hafaliad hwn yn crynhoi un rhan o resymeg yr asiant. Mae'r symbolau a'r gwerthoedd yn aros yr un fath ym mhob iaith.",
        ),
    )
    return FormulaNote(title=title, formula=note.formula, explanation=explanation)


def _welsh_code_snippet(snippet: CodeSnippet) -> CodeSnippet:
    explanation = _WELSH_CODE_GUIDES.get(
        snippet.title,
        "Mae'r darn hwn yn dangos sut mae'r syniad yn cael ei droi'n gamau Python yn y prosiect.",
    )
    return CodeSnippet(
        title=_WELSH_CODE_TITLES.get(snippet.title, snippet.title),
        source=snippet.source,
        explanation=explanation,
        code=snippet.code,
    )


def _quick_facts(
    agent_type: str,
    language: str = "en",
) -> tuple[tuple[str, str], ...]:
    facts = {
        "map_aware": (
            ("Approach", "Planned route + live sensors"),
            ("Learns from", "Does not train"),
            ("Policy", "Rules and a route plan"),
            ("Racing line", "Required"),
        ),
        "rule_based": (
            ("Approach", "Live sensor rules"),
            ("Learns from", "Does not train"),
            ("Policy", "A hand-written rulebook"),
            ("Racing line", "Not used"),
        ),
        "dyna_q_learning": (
            ("Approach", "Learns action scores"),
            ("Learns from", "Rewards + memory practice"),
            ("Policy", "An action score table"),
            ("Racing line", "Not used"),
        ),
        "dyna_q_finalised": (
            ("Approach", "Uses learned action scores"),
            ("Learns from", "Saved earlier training"),
            ("Policy", "The best saved table choice"),
            ("Racing line", "Not used"),
        ),
        "dyna_q": (
            ("Approach", "Learns action scores"),
            ("Learns from", "Rewards + memory practice"),
            ("Policy", "An action score table"),
            ("Racing line", "Not used"),
        ),
        "td3_scratch": (
            ("Approach", "Neural continuous control"),
            ("Learns from", "Rewards + replay memory"),
            ("Policy", "A trained neural network"),
            ("Racing line", "Not used"),
        ),
        "n_step_td3": (
            ("Approach", "Neural control + route preview"),
            ("Learns from", "Rewards + replay memory"),
            ("Policy", "A trained neural network"),
            ("Racing line", "Route preview only"),
        ),
        "sensor_n_step_td3": (
            ("Approach", "Neural control from sensors"),
            ("Learns from", "Rewards + its own good laps"),
            ("Policy", "A trained neural network"),
            ("Racing line", "Not used"),
        ),
        "random": (
            ("Approach", "Random comparison"),
            ("Learns from", "Does not train"),
            ("Policy", "Random control values"),
            ("Racing line", "Not used"),
        ),
    }
    selected = facts.get(
        agent_type,
        (
            ("Approach", "Project agent"),
            ("Learns from", "Agent specific"),
            ("Policy", "See guide"),
            ("Racing line", "See metadata"),
        ),
    )
    if language != "cy":
        return selected

    translations = {
        "Planned route + live sensors": "Llwybr wedi'i gynllunio + synwyryddion byw",
        "Does not train": "Nid yw'n hyfforddi",
        "Rules and a route plan": "Rheolau a chynllun llwybr",
        "Required": "Angenrheidiol",
        "Live sensor rules": "Rheolau synwyryddion byw",
        "A hand-written rulebook": "Llyfr rheolau wedi'i ysgrifennu â llaw",
        "Not used": "Heb ei ddefnyddio",
        "Learns action scores": "Yn dysgu sgoriau gweithredoedd",
        "Rewards + memory practice": "Gwobrau + ymarfer o'r cof",
        "An action score table": "Tabl sgoriau gweithredoedd",
        "Uses learned action scores": "Yn defnyddio sgoriau wedi'u dysgu",
        "Saved earlier training": "Hyfforddiant cynharach wedi'i gadw",
        "The best saved table choice": "Y dewis gorau yn y tabl wedi'i gadw",
        "Neural continuous control": "Rheolaeth niwral barhaus",
        "Rewards + replay memory": "Gwobrau + cof ailchwarae",
        "A trained neural network": "Rhwydwaith niwral wedi'i hyfforddi",
        "Neural control + route preview": "Rheolaeth niwral + rhagolwg llwybr",
        "Route preview only": "Rhagolwg llwybr yn unig",
        "Neural control from sensors": "Rheolaeth niwral o synwyryddion",
        "Rewards + its own good laps": "Gwobrau + ei lapiau da ei hun",
        "Random comparison": "Cymhariaeth ar hap",
        "Random control values": "Gwerthoedd rheoli ar hap",
        "Project agent": "Asiant y prosiect",
        "Agent specific": "Yn benodol i'r asiant",
        "See guide": "Gweler y canllaw",
        "See metadata": "Gweler y manylion technegol",
    }
    return tuple((key, translations.get(value, value)) for key, value in selected)


def _n_step_td3_template(*, sensor_only: bool) -> dict[str, Any]:
    if sensor_only:
        badge = "Sensor-only N-step TD3 policy"
        headline = (
            "Agent 8 learns continuous steering and acceleration from TORCS sensors "
            "without a racing-line file or an external teacher."
        )
        overview = (
            "It receives vehicle motion, road-range sensors, wheel spin, and recent driving history.",
            (
                "The observation deliberately contains no target route. Racing-line feature "
                "slots are zeroed so the network contract stays compatible with Agent 7."
            ),
            (
                "Most training is reward-driven TD3. Later stability experiments can retain "
                "a small sample of Agent 8's own successful laps; this is self-generated "
                "experience, not an external demonstration."
            ),
            (
                "At deployment, exploration noise is disabled and the neural actor produces "
                "the controls directly. Automatic shifting keeps the learning task focused "
                "on steering, throttle, and braking."
            ),
        )
        decision_steps = (
            "Read vehicle motion, track position, wheel spin, and all 19 road sensors.",
            "Normalise 45 sensor features and append recent observations and actions.",
            "Pass the 141-value history state through the deterministic actor network.",
            "Decode signed steering and longitudinal intent into steering, throttle, and brake.",
            "Apply automatic gear selection and send the controls to TORCS.",
        )
        reward_title = "Sensor Stability Reward"
        reward_formula = (
            r"$r = \frac{v_x}{250}\cos(\theta)"
            r" - \beta p^2 - \alpha(\Delta steer)^2"
            r" + B_{lap} - P_{failure}$"
        )
        reward_explanation = (
            "Forward velocity is useful only when aligned with the road. Small centring and "
            "steering-rate terms improve stability without prescribing a racing line, while "
            "lap completion and physical failure have clear terminal consequences."
        )
        input_signals = (
            "Longitudinal, sideways, and vertical speed plus estimated acceleration.",
            "Heading angle, track position, RPM, and four wheel-spin values.",
            "All 19 TORCS road-range sensors.",
            "Three recent observation frames and three previous two-value actions.",
            "No racing-line target, curvature map, or teacher action.",
        )
        strengths = (
            "Learns its own route from local sensors and long-term reward.",
            "The 3-step return carries useful outcomes back through time faster than one-step TD3.",
            "Twin critics and target smoothing reduce optimistic estimates of fragile actions.",
            "Saved evaluation checkpoints separate reliable policies from occasional fast laps.",
        )
        failure_signs = (
            "Repeated failure at one corner suggests a narrow policy rather than missing map data.",
            "Steering saturation or oscillation can trade reliability for a rare fast lap.",
            "A high single-run pace with poor completion rate is not a reliable champion.",
            "Self-generated success replay can overfit if it overwhelms ordinary failure and recovery data.",
        )
        key_takeaways = (
            "Agent 8 is the sensor-only neural comparison against engineered and map-aware drivers.",
            "It does not receive a racing line, target speed, or teacher control.",
            "Its 141 inputs include short-term memory, allowing the actor to infer motion trends.",
            "Any optional imitation term uses only laps discovered by Agent 8 itself and is recorded in checkpoint metadata.",
        )
        agent_note = (
            "Agent 8 preserves the same observation width as Agent 7, but its racing-line "
            "difference and lookahead-curvature slots are fixed at zero."
        )
    else:
        badge = "Racing-line-informed N-step TD3 policy"
        headline = (
            "Agent 7 learns continuous control with N-step TD3 while using racing-line "
            "geometry as context, never as a copied teacher action."
        )
        overview = (
            "It combines live TORCS telemetry with the target-line offset and upcoming curvature.",
            (
                "The racing line tells the network where the planned route lies and how the "
                "road bends ahead. It does not supply steering, throttle, brake, or target-speed actions."
            ),
            (
                "Three recent observations and three recent actions give the policy short-term "
                "memory of acceleration, steering changes, and developing slides."
            ),
            (
                "During training, replayed transitions update two value networks and then a "
                "delayed actor. During evaluation, the actor runs deterministically without exploration noise."
            ),
        )
        decision_steps = (
            "Read vehicle motion, track position, wheel spin, and all 19 road sensors.",
            "Look up racing-line offset and curvature ahead at the current lap distance.",
            "Normalise 45 features and append recent observations and actions into 141 values.",
            "Pass that state through the actor to produce signed steering and longitudinal intent.",
            "Decode the action, select the gear automatically, and send controls to TORCS.",
        )
        reward_title = "Racing-Line Velocity Reward"
        reward_formula = (
            r"$r = \frac{v_x}{250}"
            r"[\cos(\theta)-|\sin(\theta)|-|p-p_{line}|]"
            r" - \alpha(\Delta steer)^2$"
        )
        reward_explanation = (
            "The reward favours speed that is aligned with the road and close to the saved "
            "line. It scores the result of the neural action; it does not tell the actor which action to copy."
        )
        input_signals = (
            "Longitudinal, sideways, and vertical speed plus estimated acceleration.",
            "Heading angle, track position, RPM, and four wheel-spin values.",
            "All 19 TORCS road-range sensors.",
            "Racing-line position difference and lookahead curvature samples.",
            "Three recent observation frames and three previous two-value actions.",
        )
        strengths = (
            "Preview curvature gives the neural policy advance warning of corners.",
            "The 3-step return carries useful outcomes back through time faster than one-step TD3.",
            "Twin critics and target smoothing reduce optimistic estimates of fragile actions.",
            "The network still discovers steering, acceleration, and braking from reward feedback.",
        )
        failure_signs = (
            "A missing or mismatched racing-line file invalidates the observation context.",
            "Strong line adherence can favour consistency over a faster route discovered elsewhere.",
            "Steering saturation or oscillation can waste speed even when laps complete.",
            "A fast isolated lap is not evidence of robust performance across repeated evaluations.",
        )
        key_takeaways = (
            "Agent 7 is neural DRL with privileged track geometry, not behaviour cloning.",
            "The racing line contributes observations and reward context, but never control labels.",
            "Its 141 inputs include short-term history as well as current telemetry.",
            "Passive evaluation saves strong policies without rolling weights back into training.",
        )
        agent_note = (
            "Agent 7 adds racing-line position error and lookahead curvature to the same "
            "vehicle and road signals used by the sensor-only variant."
        )

    return {
        "badge": badge,
        "headline": headline,
        "overview": overview,
        "decision_steps": decision_steps,
        "algorithm_summary": (
            "N-step TD3 is a deterministic actor-critic algorithm for continuous control.",
            (
                "The actor maps the 141-value state directly to two continuous outputs: "
                "steering and signed longitudinal intent."
            ),
            (
                "The replay buffer stores experience from both successful and failed driving. "
                "A sampled transition uses up to three rewards before bootstrapping from the target critics."
            ),
            (
                "Two critics estimate long-term return. The smaller target estimate is used, "
                "which limits the overestimation that can destabilise deterministic policies."
            ),
            (
                "Clipped noise is added only to target actions during critic learning. The actor "
                "updates less often, then slowly updates its target network."
            ),
            agent_note,
        ),
        "pseudocode": (
            "Reset TORCS and initialise the three-frame observation/action history.",
            "Read and normalise the current telemetry into 45 base features.",
            "Append three base observations and three prior actions to form 141 inputs.",
            "Ask the actor for signed steering and longitudinal intent; add exploration noise only in training.",
            "Send decoded controls to TORCS and observe reward, next state, and termination.",
            "Accumulate a discounted 3-step return and store the transition in replay memory.",
            "Sample replay and update both critics toward the smaller smoothed target value.",
            "On delayed update steps, improve the actor using critic Q1's gradient.",
            "Soft-update target networks and periodically save passive evaluation checkpoints.",
        ),
        "formula_notes": (
            FormulaNote(
                title="History Observation",
                formula=r"$s_t=[o_{t-2},o_{t-1},o_t,a_{t-3},a_{t-2},a_{t-1}]\in\mathbb{R}^{141}$",
                explanation=(
                    "Three 45-feature telemetry frames and three previous two-value actions "
                    "give the feed-forward actor a compact view of recent motion."
                ),
            ),
            FormulaNote(
                title="Three-Step Critic Target",
                formula=(
                    r"$y_t=\sum_{k=0}^{2}\gamma^k r_{t+k}"
                    r"+\gamma^3(1-d)\min_{i\in\{1,2\}}Q'_{i}(s_{t+3},\tilde a)$"
                ),
                explanation=(
                    "The target includes three observed rewards before asking the target critics "
                    "to estimate what happens later. Terminal transitions do not bootstrap."
                ),
            ),
            FormulaNote(
                title="Target Policy Smoothing",
                formula=r"$\tilde a=\mathrm{clip}(\mu'(s')+\mathrm{clip}(\epsilon,-c,c),-1,1)$",
                explanation=(
                    "Clipped Gaussian noise makes the critic value actions that remain useful "
                    "under small neighbouring control changes, rather than narrow action spikes."
                ),
            ),
            FormulaNote(
                title=reward_title,
                formula=reward_formula,
                explanation=reward_explanation,
            ),
            FormulaNote(
                title="Delayed Actor Objective",
                formula=r"$L_{actor}=-\mathbb{E}[Q_1(s,\mu(s))]$",
                explanation=(
                    "The actor follows the first critic's gradient only on delayed update steps. "
                    "Selected Agent 8 stability experiments may add a decaying loss on actions "
                    "from its own completed laps; checkpoint metadata records that choice."
                ),
            ),
        ),
        "code_snippets": (
            CodeSnippet(
                title="Telemetry and history state",
                source="src/n_step_td3/contracts.py - build_base_observation / HistoryEncoder",
                explanation=(
                    "The shared contract creates 45 base inputs and combines three frames with "
                    "three previous actions. Sensor-only mode zeroes all racing-line values."
                ),
                code=(
                    "base = build_base_observation(\n"
                    "    telemetry, previous_telemetry=previous,\n"
                    "    racing_line=racing_line,\n"
                    "    include_racing_line_features=use_racing_line,\n"
                    ")\n"
                    "state = history.observe(base)  # shape: (141,)"
                ),
            ),
            CodeSnippet(
                title="Twin-critic learning target",
                source="src/n_step_td3/learner.py - NstepTd3Learner._gradient_step",
                explanation=(
                    "Target smoothing and the smaller critic value implement the two central TD3 protections."
                ),
                code=(
                    "target_noise = normal_noise(actions.shape).mul(policy_noise)\n"
                    "target_noise = target_noise.clamp(-noise_clip, noise_clip)\n"
                    "next_actions = (actor_target(next_obs) + target_noise).clamp(-1, 1)\n"
                    "target_q1, target_q2 = critic_target(next_obs, next_actions)\n"
                    "target_q = returns + gamma_n * torch.minimum(target_q1, target_q2)"
                ),
            ),
            CodeSnippet(
                title="Delayed neural policy update",
                source="src/n_step_td3/learner.py - NstepTd3Learner._gradient_step",
                explanation=(
                    "The actor is updated less frequently than the critics and is trained to "
                    "select controls with a high predicted long-term return."
                ),
                code=(
                    "if optimizer_steps % policy_delay == 0:\n"
                    "    predicted_actions = actor(observations)\n"
                    "    actor_loss = -critic.first(observations, predicted_actions).mean()\n"
                    "    actor_loss.backward()\n"
                    "    actor_optimizer.step()"
                ),
            ),
        ),
        "math_notes": (
            "The actor is the neural driver; the critics are training-time judges of long-term return.",
            "The 3-step target balances immediate evidence with a learned estimate of later outcomes.",
            "Taking the smaller critic target reduces optimistic value errors.",
            "Target policy noise regularises critic learning; evaluation itself remains deterministic.",
            "Automatic gears are an engineered convenience, while steering and pedal intent remain neural outputs.",
        ),
        "key_takeaways": key_takeaways,
        "input_signals": input_signals,
        "strengths": strengths,
        "failure_signs": failure_signs,
    }


def _metadata(
    agent: AgentOption,
    language: str = "en",
) -> tuple[tuple[str, str], ...]:
    if language == "cy":
        return (
            ("Math yr asiant", agent.agent_type),
            ("Fersiwn", agent.version),
            (
                "Dull rheoli",
                (
                    "Rheolaeth lawn: llywio, cyflymu, brecio a gêr"
                    if agent.uses_full_control
                    else "Allbwn gweithred Gym"
                ),
            ),
            (
                "Dibyniaeth ar drac",
                (
                    "Mae angen ffeil llinell rasio"
                    if agent.requires_racing_line
                    else "Nid oes angen ffeil llinell rasio"
                ),
            ),
            ("Lapiau targed", _format_optional_int(agent.target_laps)),
            ("Uchafswm camau", _format_optional_int(agent.max_steps)),
        )

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
    language: str = "en",
) -> tuple[str, ...]:
    compatible_tracks = compatible_tracks_for_agent(agent, tracks)
    if not tracks:
        return (
            "Ni ddarganfuwyd unrhyw draciau TORCS."
            if language == "cy"
            else "No TORCS tracks were discovered.",
        )

    if agent.requires_racing_line:
        if not compatible_tracks:
            if language == "cy":
                return (
                    "Mae angen ffeiliau llinell rasio ar yr asiant hwn, ond ni ddarganfuwyd yr un.",
                    "Cynhyrchwch linell rasio cyn ei ddefnyddio ar fap newydd.",
                )
            return (
                "This agent needs racing-line files, but none were discovered.",
                "Generate a racing line before using it on a new map.",
            )
        if language == "cy":
            return (
                f"Traciau â llinell rasio: {_format_track_list(compatible_tracks, language)}.",
                (
                    f"Mae gan {len(compatible_tracks)} o'r {len(tracks)} trac a "
                    "ddarganfuwyd ddata llinell rasio gyfatebol ar hyn o bryd."
                ),
                "Mae angen JSON llinell rasio ei hun ar drac newydd cyn bod yr asiant hwn yn ddewis teg.",
            )
        return (
            (
                "Racing-line tracks: "
                f"{_format_track_list(compatible_tracks, language)}."
            ),
            (
                f"{len(compatible_tracks)} of {len(tracks)} discovered tracks "
                "currently have matching racing-line data."
            ),
            "New tracks need their own racing-line JSON before this agent is a fair fit.",
        )

    if language == "cy":
        return (
            f"Traciau cydnaws: {_format_track_list(compatible_tracks, language)}.",
            (
                f"Gellir dewis yr asiant hwn ar gyfer {len(compatible_tracks)} "
                f"o'r {len(tracks)} trac a ddarganfuwyd."
            ),
            "Gall perfformiad amrywio yn ôl siâp y trac, hyd yn oed heb ddibynnu ar fap.",
        )

    return (
        f"Compatible tracks: {_format_track_list(compatible_tracks, language)}.",
        (
            f"This agent can be selected for {len(compatible_tracks)} of "
            f"{len(tracks)} discovered tracks."
        ),
        "Performance can still vary by track shape, even without a map dependency.",
    )


def _format_track_list(
    tracks: list[TrackOption],
    language: str = "en",
) -> str:
    if not tracks:
        return "dim" if language == "cy" else "none"

    labels = [track.label for track in tracks[:6]]
    remaining = len(tracks) - len(labels)
    if remaining > 0:
        labels.append(
            f"{remaining} arall" if language == "cy" else f"{remaining} more"
        )
    return ", ".join(labels)


def _format_optional_int(value: int | None) -> str:
    return "--" if value is None else f"{value:,}"
