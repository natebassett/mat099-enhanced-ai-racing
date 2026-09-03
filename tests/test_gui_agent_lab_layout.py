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

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter, QTabWidget  # noqa: E402

from gui.main_window import AgentAlgorithmDialog, MainWindow  # noqa: E402


class AgentLabLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.tabs.setCurrentIndex(self.window.agents_tab_index)

    def tearDown(self) -> None:
        self.window.close()

    def test_agent_lab_uses_summary_and_three_progressive_detail_tabs(self) -> None:
        page = self.window.tabs.currentWidget()
        splitter = page.findChild(QSplitter, "agentLabSplitter")
        tabs = page.findChild(QTabWidget, "agentLabTabs")

        self.assertIsNotNone(splitter)
        self.assertEqual(splitter.count(), 2)
        self.assertFalse(splitter.childrenCollapsible())
        self.assertGreaterEqual(splitter.widget(0).minimumWidth(), 280)
        self.assertGreaterEqual(splitter.widget(1).minimumWidth(), 620)
        self.assertIsNotNone(tabs)
        self.assertEqual(
            [tabs.tabText(index) for index in range(tabs.count())],
            [
                "Driving",
                "Learning",
                "Inside the Brain",
                "Strengths & Limits",
            ],
        )

        self.assertEqual(
            self.window.agent_algorithm_button.text(),
            "See equations and code",
        )
        self.assertIn(
            "Imagine",
            self.window.agent_education_headline_label.text(),
        )

    def test_agent_lab_pages_scroll_vertically_without_horizontal_overflow(self) -> None:
        page = self.window.tabs.currentWidget()
        for object_name in (
            "agentDrivingScroll",
            "agentLearningScroll",
            "learningVisualizerScroll",
            "agentLimitsScroll",
        ):
            scroll = page.findChild(QScrollArea, object_name)
            self.assertIsNotNone(scroll)
            self.assertTrue(scroll.widgetResizable())
            self.assertEqual(
                scroll.horizontalScrollBarPolicy(),
                Qt.ScrollBarAlwaysOff,
            )

    def test_agent9_demo_is_not_present_in_agent_lab(self) -> None:
        agent_types = {
            self.window.agent_education_combo.itemData(index, Qt.UserRole).agent_type
            for index in range(self.window.agent_education_combo.count())
        }

        self.assertIn("n_step_td3", agent_types)
        self.assertIn("sensor_n_step_td3", agent_types)
        self.assertNotIn("agent8_recorded_elite_lap", agent_types)

    def test_algorithm_guide_tabs_use_the_themed_subwindow_navigation(self) -> None:
        dialog = AgentAlgorithmDialog(self.window.agent_education_profile, self.window)
        tabs = dialog.findChild(QTabWidget, "algorithmGuideTabs")

        self.assertIsNotNone(tabs)
        self.assertGreaterEqual(tabs.count(), 3)
        dialog.close()

    def test_welsh_setting_refreshes_agent_lab_and_algorithm_guide(self) -> None:
        self.window.app_settings = self.window.app_settings.__class__(language="cy")
        self.window._apply_language("cy")

        self.assertEqual(self.window.agent_lab_tabs.tabText(0), "Gyrru")
        self.assertIn("Mae", self.window.agent_education_headline_label.text())
        self.assertIn("Traciau", self.window.agent_education_tracks_box.toPlainText())

        dialog = AgentAlgorithmDialog(
            self.window.agent_education_profile,
            self.window,
            language="cy",
        )
        tabs = dialog.findChild(QTabWidget, "algorithmGuideTabs")
        self.assertEqual(tabs.tabText(0), "Trosolwg")
        self.assertIn("Canllaw Algorithm", dialog.windowTitle())
        dialog.close()

        self.window._apply_language("en")
        self.assertEqual(self.window.agent_lab_tabs.tabText(0), "Driving")
        self.assertNotIn("Mae", self.window.agent_education_headline_label.text())


if __name__ == "__main__":
    unittest.main()
