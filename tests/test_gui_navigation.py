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

from PySide6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

from gui.navigation import PrimaryNavigation  # noqa: E402


class PrimaryNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_pages_keep_tab_compatible_selection_contract(self) -> None:
        navigation = PrimaryNavigation()
        dashboard = QWidget()
        history = QWidget()

        self.assertEqual(navigation.addTab(dashboard, "Live Telemetry"), 0)
        self.assertEqual(navigation.addTab(history, "Runs"), 1)
        self.assertEqual(navigation.count(), 2)
        self.assertEqual(navigation.currentIndex(), 0)
        self.assertIs(navigation.currentWidget(), dashboard)

        navigation.setCurrentIndex(1)

        self.assertEqual(navigation.currentIndex(), 1)
        self.assertIs(navigation.currentWidget(), history)
        self.assertEqual(navigation.indexOf(history), 1)
        self.assertEqual(navigation.tabText(1), "Runs")

    def test_navigation_items_are_accessible_and_keyboard_reachable(self) -> None:
        navigation = PrimaryNavigation()
        navigation.addTab(QWidget(), "Dashboard", tool_tip="Open dashboard")
        navigation.addTab(QWidget(), "Agents", tool_tip="Explore agents")

        buttons = navigation.findChildren(QPushButton)

        self.assertEqual([button.text() for button in buttons], ["Dashboard", "Agents"])
        self.assertEqual(
            [button.accessibleName() for button in buttons],
            ["Dashboard", "Agents"],
        )
        self.assertEqual(
            [button.shortcut().toString() for button in buttons],
            ["Alt+1", "Alt+2"],
        )


if __name__ == "__main__":
    unittest.main()
