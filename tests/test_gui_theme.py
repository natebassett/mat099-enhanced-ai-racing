from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.theme import DARK_THEME, LIGHT_THEME, build_stylesheet  # noqa: E402


class GuiThemeTests(unittest.TestCase):
    def test_light_theme_defines_shared_navigation_and_surface_styles(self):
        stylesheet = build_stylesheet(LIGHT_THEME)

        self.assertIn("QFrame#primaryNavigationHeader", stylesheet)
        self.assertIn('QPushButton[navigationItem="true"]', stylesheet)
        self.assertIn('QFrame[metricCard="true"]', stylesheet)
        self.assertIn('QFrame[pipelineStep="true"]', stylesheet)
        self.assertIn("QScrollArea > QWidget > QWidget", stylesheet)
        self.assertIn("QComboBox QAbstractItemView", stylesheet)
        self.assertIn("selection-background-color", stylesheet)
        self.assertIn(LIGHT_THEME.accent, stylesheet)
        self.assertIn(LIGHT_THEME.canvas, stylesheet)

    def test_dark_theme_uses_dark_palette_without_changing_component_contracts(self):
        stylesheet = build_stylesheet(DARK_THEME)

        self.assertIn(DARK_THEME.canvas, stylesheet)
        self.assertIn(DARK_THEME.surface, stylesheet)
        self.assertIn("QPushButton[primary=\"true\"]", stylesheet)


if __name__ == "__main__":
    unittest.main()
