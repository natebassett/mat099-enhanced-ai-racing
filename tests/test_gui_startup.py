from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.results_view import ResultsView  # noqa: E402
from gui.startup_view import StartupView  # noqa: E402
from gui.theme import DARK_THEME, chart_palette_for_preferences  # noqa: E402


class GuiStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_startup_view_clamps_progress_and_exposes_status(self) -> None:
        startup = StartupView()

        startup.set_progress(130, "Ready")

        self.assertEqual(startup.progress.value(), 100)
        self.assertEqual(startup.status_label.text(), "Ready")
        self.assertTrue(startup.progress.isTextVisible() is False)
        startup.close()

    def test_results_can_wait_until_the_page_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            view = ResultsView(Path(directory), load_immediately=False)

            self.assertFalse(view.is_loaded)
            self.assertIn("when you open", view.summary_label.text())

            view.set_visual_theme(
                DARK_THEME,
                chart_palette_for_preferences("standard", dark=True),
            )
            self.assertFalse(view.is_loaded)

            view.reload()
            self.assertTrue(view.is_loaded)
            view.close()


if __name__ == "__main__":
    unittest.main()
