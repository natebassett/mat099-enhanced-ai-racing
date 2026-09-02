from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from main_window import MainWindow
    from theme import apply_application_theme
else:
    from .main_window import MainWindow
    from .theme import apply_application_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced AI Racing")
    app.setOrganizationName("MAT099")
    apply_application_theme(app)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
