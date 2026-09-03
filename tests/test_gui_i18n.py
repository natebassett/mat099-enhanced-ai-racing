from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

from gui.i18n import tr, translate_widget_tree  # noqa: E402


class GuiTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_known_text_translates_and_technical_text_is_preserved(self) -> None:
        self.assertEqual(tr("Settings", "cy"), "Gosodiadau")
        self.assertEqual(tr("Formulas", "cy"), "Fformiwlâu")
        self.assertEqual(tr("Step 3", "cy"), "Cam 3")
        self.assertEqual(tr("TD3 Q1 loss", "cy"), "TD3 Q1 loss")

    def test_widget_tree_can_switch_back_to_english(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        translated = QLabel("Settings")
        technical = QLabel("sensor_n_step_td3 v0.1")
        layout.addWidget(translated)
        layout.addWidget(technical)

        translate_widget_tree(root, "cy")
        self.assertEqual(translated.text(), "Gosodiadau")
        self.assertEqual(technical.text(), "sensor_n_step_td3 v0.1")

        translate_widget_tree(root, "en")
        self.assertEqual(translated.text(), "Settings")
        self.assertEqual(technical.text(), "sensor_n_step_td3 v0.1")
        root.close()


if __name__ == "__main__":
    unittest.main()
