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

from gui.main_window import MainWindow  # noqa: E402


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
            ["How It Drives", "How It Learns", "Use & Limits"],
        )

    def test_agent_lab_pages_scroll_vertically_without_horizontal_overflow(self) -> None:
        page = self.window.tabs.currentWidget()
        for object_name in (
            "agentDrivingScroll",
            "agentLearningScroll",
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


if __name__ == "__main__":
    unittest.main()
