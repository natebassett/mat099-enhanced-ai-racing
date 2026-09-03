from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class StartupView(QWidget):
    """Small responsive launch window shown while the main UI is assembled."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen)
        self.setObjectName("startupView")
        self.setFixedSize(440, 190)
        self.setAccessibleName("Enhanced AI Racing is loading")

        panel = QFrame()
        panel.setObjectName("startupPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 22, 24, 22)
        panel_layout.setSpacing(5)

        title = QLabel("Enhanced AI Racing")
        title.setObjectName("startupTitle")
        context = QLabel("TELEMETRY & LEARNING LAB")
        context.setObjectName("startupContext")
        self.status_label = QLabel("Preparing the application...")
        self.status_label.setObjectName("startupStatus")
        self.progress = QProgressBar()
        self.progress.setObjectName("startupProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)

        panel_layout.addWidget(title)
        panel_layout.addWidget(context)
        panel_layout.addStretch(1)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.progress)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel)

    def show_centered(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.move(available.center() - self.rect().center())
        self.show()

    def set_progress(self, value: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, int(value))))
        self.status_label.setText(message)
        self.status_label.setAccessibleDescription(message)

