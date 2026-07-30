from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .project_discovery import AgentOption, TrackOption, compatible_tracks_for_agent
except ImportError:
    from project_discovery import AgentOption, TrackOption, compatible_tracks_for_agent


@dataclass(frozen=True)
class AgentEducationProfile:
    title: str
    badge: str
    headline: str
    overview: tuple[str, ...]
    decision_steps: tuple[str, ...]
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
                "This driver uses a saved mathematical racing line, then adjusts "
                "the car to stay near that line while managing speed."
            ),
            "overview": (
                "It is not just reacting to the road immediately in front of the car.",
                (
                    "The racing line gives it a target route through the track, so "
                    "its steering and speed choices are tied to where the car should "
                    "be next."
                ),
                (
                    "When it works well, it looks deliberate: it aims for a line, "
                    "sets up corners earlier, and tries to avoid late panic steering."
                ),
            ),
            "decision_steps": (
                "Load the racing-line points for the selected track.",
                "Read the car's speed, angle, road position, and road sensors.",
                "Find the nearby target point on the racing line.",
                "Steer back toward the target while keeping the car stable.",
                "Use throttle or brake depending on the corner and available road.",
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
                "This driver follows hand-written rules that react to sensors, "
                "wheel behaviour, and road position."
            ),
            "overview": (
                "It does not need a precomputed map of the track.",
                (
                    "The controller watches the road sensors and applies a set of "
                    "if-this-then-that decisions for steering, throttle, braking, "
                    "gear changes, and anti-spin recovery."
                ),
                (
                    "It is useful as a comparison because the decision process is "
                    "clear, but it can be late to prepare for corners it cannot see yet."
                ),
            ),
            "decision_steps": (
                "Read road sensors, speed, track position, and wheel behaviour.",
                "Choose a safe steering direction from the visible road space.",
                "Reduce throttle or brake when the car looks unstable.",
                "Apply gear and anti-spin rules to keep control.",
                "Repeat the same rule checks at every control step.",
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

    if agent_type == "random":
        return {
            "badge": "Baseline random driver",
            "headline": (
                "This driver is a control baseline: it makes random driving choices "
                "so better agents have something simple to beat."
            ),
            "overview": (
                "It is not trying to learn the track or follow a strategy.",
                (
                    "Its value is experimental rather than competitive: if another "
                    "agent cannot beat this baseline, that is a warning sign."
                ),
                (
                    "In telemetry, random behaviour usually appears as unstable "
                    "steering, inconsistent throttle, and early off-track moments."
                ),
            ),
            "decision_steps": (
                "Receive the current observation from the simulator.",
                "Sample a random steering and throttle action.",
                "Send that action to TORCS without planning ahead.",
                "Record the result as a baseline behaviour.",
                "Use the run as a low-skill comparison point.",
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
