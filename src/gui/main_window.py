from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class MetricSpec:
    label: str
    value: str
    unit: str = ""


class MainWindow(QMainWindow):
    """Stage 1 GUI shell with placeholder controls and telemetry surfaces."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Enhanced AI Racing Telemetry Dashboard")
        self.resize(1280, 760)

        self.agent_combo = QComboBox()
        self.track_combo = QComboBox()
        self.car_combo = QComboBox()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.status_label = QLabel("Idle")
        self.explanation_box = QTextEdit()

        self._configure_controls()
        self._build_ui()
        self._connect_signals()

    def _configure_controls(self) -> None:
        self.agent_combo.addItems(
            [
                "Map-Aware Racing-Line Agent",
                "Rule-Based Anti-Spin Agent",
                "Random Agent",
            ]
        )
        self.track_combo.addItems(
            [
                "g-track-3",
                "corkscrew",
                "e-track-1",
                "e-track-2",
                "forza",
            ]
        )
        self.car_combo.addItems(
            [
                "car1-ow1",
                "car1-trb1",
                "car2-trb1",
                "155-DTM",
                "acura-nsx-sz",
            ]
        )
        self.stop_button.setEnabled(False)
        self.status_label.setObjectName("statusLabel")

        self.explanation_box.setReadOnly(True)
        self.explanation_box.setText(
            "Race explanation will appear here once live telemetry is connected."
        )

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        tabs.addTab(self._build_run_history_tab(), "Run History")

        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Stage 1 GUI shell loaded")

    def _build_dashboard_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_telemetry_panel())
        splitter.addWidget(self._build_explanation_panel())
        splitter.setSizes([270, 690, 320])

        layout.addWidget(splitter)
        return page

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Race Control")
        title.setFont(_section_font())
        layout.addWidget(title)

        layout.addWidget(self._combo_group("Agent", self.agent_combo))
        layout.addWidget(self._combo_group("Track", self.track_combo))
        layout.addWidget(self._combo_group("Car", self.car_combo))

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_group)

        layout.addStretch(1)
        return panel

    def _build_telemetry_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Live Telemetry")
        title.setFont(_section_font())
        layout.addWidget(title)

        metrics = [
            MetricSpec("Speed", "--", "km/h"),
            MetricSpec("Gear", "--"),
            MetricSpec("Throttle", "--", "%"),
            MetricSpec("Brake", "--", "%"),
            MetricSpec("Steer", "--"),
            MetricSpec("Track Pos", "--"),
            MetricSpec("Lap Time", "--", "s"),
            MetricSpec("Damage", "--"),
        ]
        metric_grid = QGridLayout()
        for index, metric in enumerate(metrics):
            metric_grid.addWidget(
                self._metric_card(metric),
                index // 4,
                index % 4,
            )
        layout.addLayout(metric_grid)

        chart_row = QHBoxLayout()
        chart_row.addWidget(self._placeholder_plot("Speed"))
        chart_row.addWidget(self._placeholder_plot("Steering"))
        layout.addLayout(chart_row)

        pedal_plot = self._placeholder_plot("Throttle / Brake")
        layout.addWidget(pedal_plot)

        return panel

    def _build_explanation_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("What The Car Is Doing")
        title.setFont(_section_font())
        layout.addWidget(title)
        layout.addWidget(self.explanation_box)

        return panel

    def _build_run_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Run History")
        title.setFont(_section_font())
        layout.addWidget(title)

        table = QTableWidget(0, 7)
        table.setHorizontalHeaderLabels(
            [
                "Run",
                "Agent",
                "Track",
                "Best Lap",
                "Avg Speed",
                "Off Track",
                "Result",
            ]
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setMinimumHeight(420)

        placeholder_rows = [
            ["--", "Map-Aware Racing-Line Agent", "g-track-3", "--", "--", "--", "Waiting for saved runs"],
            ["--", "Rule-Based Anti-Spin Agent", "corkscrew", "--", "--", "--", "Stage 2 will load real data"],
        ]
        table.setRowCount(len(placeholder_rows))
        for row_index, row in enumerate(placeholder_rows):
            for column_index, value in enumerate(row):
                table.setItem(row_index, column_index, QTableWidgetItem(value))

        layout.addWidget(table)
        return page

    def _combo_group(self, label: str, combo: QComboBox) -> QGroupBox:
        group = QGroupBox(label)
        layout = QVBoxLayout(group)
        layout.addWidget(combo)
        return group

    def _metric_card(self, metric: MetricSpec) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setMinimumHeight(82)

        layout = QVBoxLayout(card)
        label = QLabel(metric.label)
        label.setObjectName("metricLabel")

        value = QLabel(metric.value if not metric.unit else f"{metric.value} {metric.unit}")
        value.setObjectName("metricValue")
        value.setAlignment(Qt.AlignBottom | Qt.AlignLeft)

        layout.addWidget(label)
        layout.addWidget(value)
        return card

    def _placeholder_plot(self, title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setTitle(title)
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setMinimumHeight(190)
        plot.plot([0, 1, 2, 3, 4], [0, 0, 0, 0, 0], pen=pg.mkPen("#4f6f8f", width=2))
        return plot

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._handle_start_clicked)
        self.stop_button.clicked.connect(self._handle_stop_clicked)

    def _handle_start_clicked(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(
            f"Ready to run {self.agent_combo.currentText()} on "
            f"{self.track_combo.currentText()} with {self.car_combo.currentText()}."
        )
        self.statusBar().showMessage("Stage 1 placeholder start clicked")
        self.explanation_box.setText(
            "Stage 1 only confirms the GUI flow. TORCS launch and live telemetry "
            "will be wired in later stages."
        )

    def _handle_stop_clicked(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Idle")
        self.statusBar().showMessage("Stage 1 placeholder stop clicked")
        self.explanation_box.setText(
            "Race explanation will appear here once live telemetry is connected."
        )


def _section_font() -> QFont:
    font = QFont()
    font.setPointSize(13)
    font.setBold(True)
    return font
