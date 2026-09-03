from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.novice_education import build_novice_agent_guide  # noqa: E402


class NoviceEducationTests(unittest.TestCase):
    def test_every_discovered_agent_has_a_five_step_plain_guide(self) -> None:
        agent_types = (
            "map_aware",
            "rule_based",
            "dyna_q_learning",
            "dyna_q_finalised",
            "td3_scratch",
            "n_step_td3",
            "sensor_n_step_td3",
            "random",
        )

        for agent_type in agent_types:
            with self.subTest(agent_type=agent_type):
                guide = build_novice_agent_guide(agent_type)
                self.assertEqual(len(guide.decision_steps), 5)
                self.assertTrue(guide.headline)
                self.assertTrue(guide.driving_story)
                self.assertTrue(guide.learning_story)
                self.assertTrue(guide.input_signals)
                self.assertTrue(guide.strengths)
                self.assertTrue(guide.failure_signs)

    def test_neural_guides_explain_actor_and_critics_without_claiming_a_teacher(self) -> None:
        guide = build_novice_agent_guide("sensor_n_step_td3")
        learning = " ".join(guide.learning_story)

        self.assertIn("neural driver", learning)
        self.assertIn("independent coaches", learning)
        self.assertIn("no prepared route", " ".join(guide.input_signals).lower())

    def test_engineered_and_table_agents_state_how_they_improve(self) -> None:
        map_aware = build_novice_agent_guide("map_aware")
        dyna_q = build_novice_agent_guide("dyna_q_learning")

        self.assertIn("does not train", " ".join(map_aware.learning_story).lower())
        self.assertIn("scorebook", " ".join(dyna_q.learning_story).lower())

    def test_every_agent_has_a_complete_welsh_novice_guide(self) -> None:
        agent_types = (
            "map_aware",
            "rule_based",
            "dyna_q_learning",
            "dyna_q_finalised",
            "n_step_td3",
            "sensor_n_step_td3",
            "random",
        )

        for agent_type in agent_types:
            with self.subTest(agent_type=agent_type):
                english = build_novice_agent_guide(agent_type)
                welsh = build_novice_agent_guide(agent_type, language="cy")
                self.assertEqual(len(welsh.decision_steps), 5)
                self.assertNotEqual(welsh.headline, english.headline)
                self.assertTrue(all(welsh.driving_story))
                self.assertTrue(all(welsh.learning_story))
                self.assertTrue(all(welsh.input_signals))


if __name__ == "__main__":
    unittest.main()
