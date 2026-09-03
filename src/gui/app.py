from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from i18n import tr
    from settings_model import SettingsStore
    from startup_view import StartupView
    from theme import apply_application_theme, palette_for_preferences
else:
    from .i18n import tr
    from .settings_model import SettingsStore
    from .startup_view import StartupView
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

    startup = StartupView()
    language = settings.language
    startup.set_progress(8, tr("Preparing the interface...", language))
    startup.show_centered()
    app.processEvents()

    startup.set_progress(20, tr("Loading application components...", language))
    app.processEvents()
    MainWindow = _load_main_window_class()

    window = MainWindow(
        progress_callback=lambda value, message: _update_startup(
            app,
            startup,
            value,
            tr(message, language),
        )
    )
    startup.set_progress(100, tr("Ready", language))
    window.show()
    app.processEvents()
    startup.close()

    return app.exec()


def _load_main_window_class():
    if __package__ in {None, ""}:
        from main_window import MainWindow
    else:
        from .main_window import MainWindow
    return MainWindow


def _update_startup(
    application: QApplication,
    startup: StartupView,
    value: int,
    message: str,
) -> None:
    startup.set_progress(value, message)
    application.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
