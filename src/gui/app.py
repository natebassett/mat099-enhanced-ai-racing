from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from main_window import MainWindow
    from settings_model import SettingsStore
    from theme import apply_application_theme, palette_for_preferences
else:
    from .main_window import MainWindow
    from .settings_model import SettingsStore
    from .theme import apply_application_theme, palette_for_preferences


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced AI Racing")
    app.setOrganizationName("MAT099")
    project_root = Path(__file__).resolve().parents[2]
    settings = SettingsStore(
        project_root / "data" / "generated" / "gui_settings.json"
    ).load()
    apply_application_theme(
        app,
        palette_for_preferences(
            settings.appearance_mode,
            settings.colour_mode,
            app,
        ),
    )

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
