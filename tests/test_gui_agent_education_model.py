from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.agent_education_model import build_agent_education_profile  # noqa: E402
from gui.project_discovery import AgentOption, TrackOption  # noqa: E402


class GuiAgentEducationModelTests(unittest.TestCase):
    def test_map_aware_profile_explains_racing_line_dependency(self):
        profile = build_agent_education_profile(
            _agent("Map-Aware Racing-Line Agent", "map_aware", requires_line=True),
            _tracks(),
        )

        self.assertEqual(profile.badge, "Map-specific racing-line planner")
        self.assertIn(
            "Racing-line file for the selected track.",
            profile.input_signals,
        )
        self.assertIn(
            "The map-aware driver separates route planning",
            profile.algorithm_summary[0],
        )
        self.assertTrue(any("target_track_pos" in note for note in profile.math_notes))
        self.assertTrue(
            any("Racing Point Projection" == note.title for note in profile.formula_notes)
        )
        self.assertTrue(any(r"\sqrt" in note.formula for note in profile.formula_notes))
        self.assertTrue(any("smoothstep" in note.explanation for note in profile.formula_notes))
        self.assertTrue(
            any("RacingLine.lookup" in snippet.source for snippet in profile.code_snippets)
        )
        self.assertIn(
            "Track dependency",
            [label for label, _value in profile.metadata],
        )
        self.assertIn("Corkscrew", profile.track_context[0])
        self.assertIn("New tracks need their own racing-line JSON", profile.track_context[2])

    def test_rule_based_profile_is_not_map_dependent(self):
        profile = build_agent_education_profile(
            _agent("Rule-Based Anti-Spin Agent", "rule_based", requires_line=False),
            _tracks(),
        )

        metadata = dict(profile.metadata)
        self.assertEqual(metadata["Track dependency"], "No racing-line file required")
        self.assertIn("Reactive rule-based controller", profile.badge)
        self.assertIn("Compatible tracks", profile.track_context[0])
        self.assertTrue(
            any("Visible-Corner Severity" == note.title for note in profile.formula_notes)
        )
        self.assertTrue(any("severity" in note for note in profile.math_notes))
        self.assertTrue(
            any("get_best_sensor" in snippet.source for snippet in profile.code_snippets)
        )

    def test_random_profile_is_marked_as_baseline(self):
        profile = build_agent_education_profile(
            _agent("Random Agent", "random", requires_line=False),
            _tracks(),
        )

        self.assertEqual(profile.badge, "Baseline random driver")
        self.assertIn("low-skill comparison point", profile.decision_steps[-1])
        self.assertIn("uniform random sampling", profile.math_notes[0])
        self.assertTrue(
            any("Uniform Steering Sample" == note.title for note in profile.formula_notes)
        )
        self.assertTrue(any(r"\mathcal{U}" in note.formula for note in profile.formula_notes))
        self.assertEqual(len(profile.code_snippets), 1)

    def test_dyna_q_profile_explains_model_replay_and_finalised_mode(self):
        profile = build_agent_education_profile(
            _agent("Dyna-Q Learning Agent", "dyna_q_learning", requires_line=False),
            _tracks(),
        )

        self.assertIn("reinforcement learner", profile.badge)
        self.assertTrue(
            any("planning updates" in note.casefold() for note in profile.math_notes)
        )
        self.assertTrue(any("Dyna-Q Model Replay" == note.title for note in profile.formula_notes))
        self.assertTrue(any(r"\max" in note.formula for note in profile.formula_notes))
        self.assertTrue(
            any("learn_from_transition" in snippet.source for snippet in profile.code_snippets)
        )
        self.assertIn("Compatible tracks", profile.track_context[0])
        self.assertTrue(
            any("live dashboard" in note.casefold() for note in profile.overview)
        )

    def test_td3_profile_explains_reward_only_neural_control(self):
        profile = build_agent_education_profile(
            _agent("TD3 Scratch Racer", "td3_scratch", requires_line=False),
            _tracks(),
        )

        self.assertEqual(profile.badge, "Reward-only neural reinforcement learner")
        self.assertIn("from-scratch TD3", profile.headline)
        self.assertTrue(any("Twin Critic Target" == note.title for note in profile.formula_notes))
        self.assertTrue(any("build_td3_observation" in snippet.source for snippet in profile.code_snippets))
        self.assertIn("No racing-line file required", dict(profile.metadata)["Track dependency"])


def _agent(name: str, agent_type: str, *, requires_line: bool) -> AgentOption:
    return AgentOption(
        name=name,
        agent_type=agent_type,
        version="test",
        class_path=f"agents.{agent_type}",
        uses_full_control=agent_type != "random",
        requires_racing_line=requires_line,
        max_steps=150000,
        target_laps=1,
    )


def _tracks() -> list[TrackOption]:
    return [
        TrackOption(
            track_id="corkscrew",
            display_name="Corkscrew",
            category="road",
            path=Path("torcs/tracks/road/corkscrew/corkscrew.xml"),
            racing_line_path=Path("data/racing_lines/corkscrew.json"),
        ),
        TrackOption(
            track_id="aalborg",
            display_name="Aalborg",
            category="road",
            path=Path("torcs/tracks/road/aalborg/aalborg.xml"),
            racing_line_path=None,
        ),
    ]


if __name__ == "__main__":
    unittest.main()
