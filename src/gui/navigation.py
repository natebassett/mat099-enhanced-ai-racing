from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class PrimaryNavigation(QWidget):
    """Full-width application navigation with a stacked page area."""

    currentChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("primaryNavigation")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("primaryNavigationHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 0, 18, 0)
        header_layout.setSpacing(4)

        product_lockup = QWidget()
        product_lockup.setObjectName("productLockup")
        product_layout = QVBoxLayout(product_lockup)
        product_layout.setContentsMargins(0, 4, 0, 4)
        product_layout.setSpacing(0)

        product_name = QLabel("Enhanced AI Racing")
        product_name.setObjectName("productName")
        product_name.setAccessibleName("Enhanced AI Racing")
        product_context = QLabel("TELEMETRY & LEARNING LAB")
        product_context.setObjectName("productContext")
        product_layout.addWidget(product_name)
        product_layout.addWidget(product_context)
        header_layout.addWidget(product_lockup)

        divider = QFrame()
        divider.setObjectName("navigationDivider")
        divider.setFixedSize(1, 24)
        header_layout.addSpacing(12)
        header_layout.addWidget(divider)
        header_layout.addSpacing(12)

        self._button_layout = QHBoxLayout()
        self._button_layout.setContentsMargins(0, 0, 0, 0)
        self._button_layout.setSpacing(2)
        header_layout.addLayout(self._button_layout)
        header_layout.addStretch(1)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        self._labels: list[str] = []

        self._pages = QStackedWidget()
        self._pages.setObjectName("primaryPageStack")
        self._pages.currentChanged.connect(self._on_current_changed)

        root_layout.addWidget(header)
        root_layout.addWidget(self._pages, 1)

    def addTab(
        self,
        page: QWidget,
        label: str,
        tool_tip: str = "",
    ) -> int:
        index = self._pages.addWidget(page)
        button = QPushButton(label)
        button.setProperty("navigationItem", True)
        button.setCheckable(True)
        button.setAccessibleName(label)
        button.setAccessibleDescription(tool_tip)
        button.setToolTip(tool_tip)
        button.setShortcut(QKeySequence(f"Alt+{index + 1}"))

        button.clicked.connect(
            lambda checked=False, page_index=index: self.setCurrentIndex(page_index)
        )
        self._button_group.addButton(button, index)
        self._button_layout.addWidget(button)
        self._buttons.append(button)
        self._labels.append(label)

        if index == 0:
            button.setChecked(True)
        return index

    def count(self) -> int:
        return self._pages.count()

    def currentIndex(self) -> int:
        return self._pages.currentIndex()

    def currentWidget(self) -> QWidget | None:
        return self._pages.currentWidget()

    def indexOf(self, widget: QWidget) -> int:
        return self._pages.indexOf(widget)

    def tabText(self, index: int) -> str:
        if 0 <= index < len(self._labels):
            return self._labels[index]
        return ""

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < self._pages.count():
            self._pages.setCurrentIndex(index)

    def _on_current_changed(self, index: int) -> None:
        button = self._button_group.button(index)
        if button is not None:
            button.setChecked(True)
        self.currentChanged.emit(index)
