from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.theme import (  # noqa: E402
    ACCESSIBLE_DARK_THEME,
    DARK_THEME,
    HIGH_CONTRAST_LIGHT_THEME,
    LIGHT_THEME,
    build_stylesheet,
    chart_palette_for_preferences,
    palette_for_preferences,
)


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

    def test_preferences_select_accessible_and_high_contrast_palettes(self):
        self.assertIs(
            palette_for_preferences("dark", "accessible"),
            ACCESSIBLE_DARK_THEME,
        )
        self.assertIs(
            palette_for_preferences("light", "high_contrast"),
            HIGH_CONTRAST_LIGHT_THEME,
        )

    def test_accessible_charts_use_distinct_colours_and_redundant_series_styles(self):
        palette = chart_palette_for_preferences("accessible", dark=False)

        self.assertNotEqual(palette.throttle, palette.brake)
        self.assertNotEqual(palette.comparison_a, palette.comparison_b)
        self.assertNotEqual(palette.agent6, palette.agent7)
        self.assertNotEqual(palette.agent7, palette.agent8)


if __name__ == "__main__":
    unittest.main()
