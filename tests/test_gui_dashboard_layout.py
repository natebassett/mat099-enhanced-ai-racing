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


class DashboardLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_dashboard_allocates_space_to_controls_telemetry_and_insight(self) -> None:
        dashboard = self.window.tabs.currentWidget()
        splitter = dashboard.findChild(QSplitter, "dashboardSplitter")

        self.assertIsNotNone(splitter)
        self.assertEqual(splitter.count(), 3)
        self.assertGreaterEqual(splitter.widget(0).minimumWidth(), 240)
        self.assertGreaterEqual(splitter.widget(1).minimumWidth(), 500)
        self.assertGreaterEqual(splitter.widget(2).minimumWidth(), 285)
        self.assertFalse(splitter.childrenCollapsible())

    def test_live_charts_are_grouped_and_insight_panel_scrolls(self) -> None:
        dashboard = self.window.tabs.currentWidget()
        chart_tabs = dashboard.findChild(QTabWidget, "contentTabs")
        insight_scroll = dashboard.findChild(QScrollArea, "dashboardInsightScroll")

        self.assertIsNotNone(chart_tabs)
        self.assertEqual(chart_tabs.count(), 3)
        self.assertEqual(
            [chart_tabs.tabText(index) for index in range(chart_tabs.count())],
            ["Speed", "Steering", "Throttle / Brake"],
        )
        self.assertIsNotNone(insight_scroll)
        self.assertTrue(insight_scroll.widgetResizable())
        self.assertEqual(
            insight_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarAlwaysOff,
        )


if __name__ == "__main__":
    unittest.main()
