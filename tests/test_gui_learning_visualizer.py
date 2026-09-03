from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.learning_visualizer import LearningVisualizerPanel  # noqa: E402
from gui.learning_visualizer_model import (  # noqa: E402
    build_learning_visualization_profile,
    inspect_td3_checkpoint,
)


class LearningVisualizerModelTests(unittest.TestCase):
    def test_td3_profile_covers_the_complete_actor_critic_update(self) -> None:
        profile = build_learning_visualization_profile("sensor_n_step_td3")

        self.assertEqual(profile.kind, "td3")
        self.assertEqual(len(profile.steps), 7)
        self.assertEqual(profile.steps[0].active_group, "replay")
        self.assertEqual(profile.steps[-1].active_group, "update")
        self.assertTrue(any("Twin critics" in step.title for step in profile.steps))
        self.assertTrue(any("min(Q1', Q2')" in step.equation for step in profile.steps))

    def test_dyna_q_and_engineered_agents_get_truthful_visual_profiles(self) -> None:
        dyna = build_learning_visualization_profile("dyna_q_learning")
        map_aware = build_learning_visualization_profile("map_aware")

        self.assertEqual(dyna.kind, "dyna_q")
        self.assertTrue(any(step.active_group == "q_update" for step in dyna.steps))
        self.assertEqual(map_aware.kind, "static")
        self.assertIn("nothing is trained", map_aware.status)
        self.assertEqual(map_aware.steps[-1].active_group, "no_update")

    def test_reads_native_actor_checkpoint_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(
                {
                    "model_family": "test_n_step_td3",
                    "environment_steps": 12_000,
                    "actor_updates": 450,
                    "actor": {
                        "network.0.weight": torch.tensor(
                            [[0.1, -0.2], [0.3, -0.4]],
                            dtype=torch.float32,
                        ),
                        "network.0.bias": torch.tensor([0.0, 0.1]),
                        "network.2.weight": torch.tensor([[0.5, -0.6]]),
                    },
                },
                path,
            )
            before = path.read_bytes()

            statistics = inspect_td3_checkpoint(path)

            self.assertEqual(statistics.model_family, "test_n_step_td3")
            self.assertEqual(statistics.parameter_count, 8)
            self.assertEqual(statistics.tensor_count, 3)
            self.assertEqual(statistics.environment_steps, 12_000)
            self.assertEqual(statistics.actor_updates, 450)
            self.assertEqual(len(statistics.layers), 2)
            self.assertEqual(path.read_bytes(), before)

    def test_reads_stable_baselines_actor_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent6.zip"
            policy = io.BytesIO()
            torch.save(
                {
                    "actor.mu.0.weight": torch.ones((3, 2)),
                    "actor.mu.0.bias": torch.zeros(3),
                    "actor_target.mu.0.weight": torch.full((3, 2), 9.0),
                    "critic.qf0.0.weight": torch.full((3, 2), 5.0),
                },
                policy,
            )
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("policy.pth", policy.getvalue())

            statistics = inspect_td3_checkpoint(path)

            self.assertEqual(statistics.model_family, "stable_baselines3_td3")
            self.assertEqual(statistics.parameter_count, 9)
            self.assertEqual(statistics.tensor_count, 2)
            self.assertEqual(len(statistics.layers), 1)
            self.assertAlmostEqual(statistics.mean_absolute_weight, 1.0)


class LearningVisualizerPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_manual_steps_update_the_explanation_without_starting_timer(self) -> None:
        panel = LearningVisualizerPanel()
        panel.set_agent("dyna_q_learning", "Dyna-Q Learning Agent")
        first_title = panel.step_label.text()

        panel.next_step()

        self.assertNotEqual(panel.step_label.text(), first_title)
        self.assertIn("Step 2 of", panel.step_label.text())
        self.assertFalse(panel.timer.isActive())
        panel.close()

    def test_reduced_motion_advances_without_starting_animation(self) -> None:
        panel = LearningVisualizerPanel()
        panel.set_agent("sensor_n_step_td3", "Agent 8")
        panel.set_reduce_motion(True)

        panel.toggle_playback()

        self.assertEqual(panel.step_index, 1)
        self.assertFalse(panel.timer.isActive())
        self.assertIn("next learning step", panel.play_button.toolTip().lower())
        self.assertEqual(
            panel.play_button.accessibleName(),
            "Show next learning step",
        )
        panel.close()


if __name__ == "__main__":
    unittest.main()
