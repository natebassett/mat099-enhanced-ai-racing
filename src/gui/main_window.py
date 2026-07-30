from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, QThread, QTimer
from PySide6.QtGui import QFont, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from .agent_education_model import (
        AgentEducationProfile,
        build_agent_education_profile,
    )
    from .comparison_model import (
        compare_runs,
        comparison_finish_time,
        moment_comparison_reasons,
        moment_panel_text,
        nearest_snapshot_index,
        run_combo_label,
        run_title,
        snapshot_times,
    )
    from .project_discovery import (
        AgentOption,
        CarOption,
        ProjectOptions,
        RunSummary,
        TrackOption,
        compatible_tracks_for_agent,
        discover_run_history,
        load_project_options,
    )
    from .race_worker import RaceWorker
    from .racing_line_model import (
        RacingLineProfile,
        RacingLineVisualProfile,
        build_racing_line_visual_profile,
        load_racing_line_profile,
        racing_line_point_text,
        racing_line_summary_lines,
    )
    from .racing_line_view import RacingLineGraphWidget, RacingLineWidget
    from .telemetry_model import (
        ExplanationFrame,
        TelemetryHistory,
        TelemetrySnapshot,
        build_explanation_frame,
        format_explanation_entry,
        format_value,
        summarize_run_explanation,
    )
    from .track_view import RoadSensorWidget, TrackPositionWidget
except ImportError:
    from agent_education_model import (
        AgentEducationProfile,
        build_agent_education_profile,
    )
    from comparison_model import (
        compare_runs,
        comparison_finish_time,
        moment_comparison_reasons,
        moment_panel_text,
        nearest_snapshot_index,
        run_combo_label,
        run_title,
        snapshot_times,
    )
    from project_discovery import (
        AgentOption,
        CarOption,
        ProjectOptions,
        RunSummary,
        TrackOption,
        compatible_tracks_for_agent,
        discover_run_history,
        load_project_options,
    )
    from race_worker import RaceWorker
    from racing_line_model import (
        RacingLineProfile,
        RacingLineVisualProfile,
        build_racing_line_visual_profile,
        load_racing_line_profile,
        racing_line_point_text,
        racing_line_summary_lines,
    )
    from racing_line_view import RacingLineGraphWidget, RacingLineWidget
    from telemetry_model import (
        ExplanationFrame,
        TelemetryHistory,
        TelemetrySnapshot,
        build_explanation_frame,
        format_explanation_entry,
        format_value,
        summarize_run_explanation,
    )
    from track_view import RoadSensorWidget, TrackPositionWidget


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    value: str
    unit: str = ""


class MainWindow(QMainWindow):
    """Desktop GUI shell with real project option discovery."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Enhanced AI Racing Telemetry Dashboard")
        self.resize(1280, 760)

        self.project_options: ProjectOptions = load_project_options()

        self.agent_combo = QComboBox()
        self.track_combo = QComboBox()
        self.car_combo = QComboBox()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.status_label = QLabel("Idle")
        self.explanation_status_label = QLabel("Idle")
        self.explanation_state_label = QLabel("Telemetry idle.")
        self.explanation_reason_label = QLabel("- No live sample yet.")
        self.explanation_box = QTextEdit()
        self.run_history_table = QTableWidget(0, 8)
        self.run_history_source_label = QLabel()
        self.tabs: QTabWidget | None = None
        self.review_tab_index = 0
        self.compare_tab_index = 0
        self.agents_tab_index = 0
        self.review_title_label = QLabel("No run selected")
        self.review_status_label = QLabel("Idle")
        self.review_summary_box = QTextEdit()
        self.review_play_button = QPushButton("Play")
        self.review_slider = QSlider(Qt.Horizontal)
        self.review_time_label = QLabel("Select a run from Run History.")
        self.review_metric_value_labels: dict[str, QLabel] = {}
        self.review_track_position_widget = TrackPositionWidget()
        self.review_road_sensor_widget = RoadSensorWidget()
        self.review_moment_status_label = QLabel("Idle")
        self.review_moment_state_label = QLabel("Select a run to review.")
        self.review_moment_reason_label = QLabel("- No replay sample selected.")
        self.compare_track_combo = QComboBox()
        self.compare_run_a_combo = QComboBox()
        self.compare_run_b_combo = QComboBox()
        self.compare_button = QPushButton("Compare")
        self.compare_summary_label = QLabel("No comparison loaded")
        self.compare_summary_box = QTextEdit()
        self.compare_play_button = QPushButton("Play")
        self.compare_slider = QSlider(Qt.Horizontal)
        self.compare_time_label = QLabel("Select two saved runs.")
        self.compare_delta_label = QLabel("- No comparison loaded.")
        self.compare_a_title_label = QLabel("Run A")
        self.compare_b_title_label = QLabel("Run B")
        self.compare_a_status_label = QLabel("Idle")
        self.compare_b_status_label = QLabel("Idle")
        self.compare_a_state_label = QLabel("No run selected.")
        self.compare_b_state_label = QLabel("No run selected.")
        self.compare_a_values_label = QLabel("-")
        self.compare_b_values_label = QLabel("-")
        self.agent_education_combo = QComboBox()
        self.agent_education_title_label = QLabel("Agent Guide")
        self.agent_education_badge_label = QLabel("Idle")
        self.agent_education_headline_label = QLabel("Select an agent.")
        self.agent_algorithm_button = QPushButton("Algorithm Guide")
        self.agent_education_metadata_box = QTextEdit()
        self.agent_education_overview_box = QTextEdit()
        self.agent_education_signals_box = QTextEdit()
        self.agent_education_strengths_box = QTextEdit()
        self.agent_education_failure_box = QTextEdit()
        self.agent_education_tracks_box = QTextEdit()
        self.agent_pipeline_labels: list[QLabel] = []
        self.racing_line_track_combo = QComboBox()
        self.racing_line_play_button = QPushButton("Play")
        self.racing_line_reset_button = QPushButton("Reset")
        self.racing_line_map_button = QPushButton("Open Map")
        self.racing_line_progress_label = QLabel("No racing line loaded")
        self.racing_line_summary_box = QTextEdit()
        self.racing_line_point_box = QTextEdit()
        self.racing_line_graph_widget = RacingLineGraphWidget()
        self.racing_line_visualizer_group: QGroupBox | None = None
        self.track_position_widget = TrackPositionWidget()
        self.road_sensor_widget = RoadSensorWidget()
        self.race_thread: QThread | None = None
        self.race_worker: RaceWorker | None = None
        self.review_timer = QTimer(self)
        self.compare_timer = QTimer(self)
        self.racing_line_timer = QTimer(self)
        self.review_snapshots: list[TelemetrySnapshot] = []
        self.review_run: dict[str, Any] | None = None
        self.review_current_index = 0
        self.compare_runs: list[dict[str, Any]] = []
        self.compare_run_a: dict[str, Any] | None = None
        self.compare_run_b: dict[str, Any] | None = None
        self.compare_snapshots_a: list[TelemetrySnapshot] = []
        self.compare_snapshots_b: list[TelemetrySnapshot] = []
        self.compare_times_a: list[float] = []
        self.compare_times_b: list[float] = []
        self.compare_current_time = 0.0
        self.racing_line_profile: RacingLineProfile | None = None
        self.racing_line_visual_profile: RacingLineVisualProfile | None = None
        self.racing_line_map_dialog: RacingLineMapDialog | None = None
        self.agent_education_profile: AgentEducationProfile | None = None
        self.agent_algorithm_dialog: AgentAlgorithmDialog | None = None
        self.racing_line_distance = 0.0
        self.telemetry_history = TelemetryHistory()
        self.metric_value_labels: dict[str, QLabel] = {}
        self.chart_window_seconds = 90.0
        self.speed_plot: pg.PlotWidget | None = None
        self.steer_plot: pg.PlotWidget | None = None
        self.pedal_plot: pg.PlotWidget | None = None
        self.review_speed_plot: pg.PlotWidget | None = None
        self.review_steer_plot: pg.PlotWidget | None = None
        self.review_pedal_plot: pg.PlotWidget | None = None
        self.review_speed_curve: Any = None
        self.review_steer_curve: Any = None
        self.review_throttle_curve: Any = None
        self.review_brake_curve: Any = None
        self.review_speed_marker: Any = None
        self.review_steer_marker: Any = None
        self.review_pedal_marker: Any = None
        self.compare_speed_plot: pg.PlotWidget | None = None
        self.compare_speed_curve_a: Any = None
        self.compare_speed_curve_b: Any = None
        self.compare_speed_marker: Any = None
        self.speed_curve: Any = None
        self.steer_curve: Any = None
        self.throttle_curve: Any = None
        self.brake_curve: Any = None
        self.explanation_interval_seconds = 10.0
        self.explanation_interval_steps = 500
        self._last_explanation_step: int | None = None
        self._last_explanation_lap_time: float | None = None
        self._last_explanation_event_key: str | None = None

        self._configure_controls()
        self._build_ui()
        self._refresh_compare_options()
        self._update_agent_education()
        self._connect_signals()
        self._update_selection_details()

    def _configure_controls(self) -> None:
        self._populate_combo(self.agent_combo, self.project_options.agents)
        self._populate_combo(self.agent_education_combo, self.project_options.agents)
        self._populate_racing_line_tracks()
        self._refresh_track_options(preferred_track_id="g-track-3")
        self._populate_combo(self.car_combo, self.project_options.cars)
        self._select_combo_item(self.car_combo, "car_id", "car1-ow1")

        self.stop_button.setEnabled(False)
        self.status_label.setObjectName("statusLabel")

        self.explanation_status_label.setObjectName("explanationStatus")
        self.explanation_status_label.setAlignment(Qt.AlignCenter)
        self.explanation_status_label.setMinimumHeight(28)
        self.explanation_state_label.setWordWrap(True)
        self.explanation_reason_label.setWordWrap(True)
        self.explanation_reason_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._set_explanation_current(
            "Idle",
            "Telemetry idle.",
            ["No live sample yet."],
        )
        self.explanation_box.setReadOnly(True)
        self.explanation_box.document().setMaximumBlockCount(240)
        self.explanation_box.setText("Race log idle.")

        self.review_title_label.setFont(_section_font())
        self.review_status_label.setAlignment(Qt.AlignCenter)
        self.review_status_label.setMinimumHeight(28)
        self.review_summary_box.setReadOnly(True)
        self.review_summary_box.setMaximumHeight(112)
        self.review_summary_box.setText("Select a saved run from Run History.")
        self.review_slider.setEnabled(False)
        self.review_play_button.setEnabled(False)
        self.review_moment_state_label.setWordWrap(True)
        self.review_moment_reason_label.setWordWrap(True)
        self.review_moment_reason_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._set_review_status("Idle")
        self._set_review_moment(
            "Idle",
            "Select a run to review.",
            ["No replay sample selected."],
        )
        self.review_timer.setInterval(250)

        self.compare_summary_label.setFont(_section_font())
        self.compare_summary_box.setReadOnly(True)
        self.compare_summary_box.setMaximumHeight(128)
        self.compare_summary_box.setText("Awaiting same-track saved runs.")
        self.compare_slider.setEnabled(False)
        self.compare_play_button.setEnabled(False)
        self.compare_button.setEnabled(False)
        self.compare_delta_label.setWordWrap(True)
        for label in (
            self.compare_a_title_label,
            self.compare_b_title_label,
            self.compare_a_state_label,
            self.compare_b_state_label,
            self.compare_a_values_label,
            self.compare_b_values_label,
        ):
            label.setWordWrap(True)
        for label in (self.compare_a_status_label, self.compare_b_status_label):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(28)
            label.setStyleSheet(_severity_style("Idle"))
        self.compare_timer.setInterval(250)

        self.agent_education_title_label.setFont(_section_font())
        self.agent_education_badge_label.setAlignment(Qt.AlignCenter)
        self.agent_education_badge_label.setMinimumHeight(28)
        self.agent_education_badge_label.setStyleSheet(_agent_badge_style())
        self.agent_education_headline_label.setWordWrap(True)
        self.agent_education_headline_label.setMinimumHeight(54)
        self.agent_algorithm_button.setEnabled(False)
        for box in (
            self.agent_education_metadata_box,
            self.agent_education_overview_box,
            self.agent_education_signals_box,
            self.agent_education_strengths_box,
            self.agent_education_failure_box,
            self.agent_education_tracks_box,
            self.racing_line_summary_box,
            self.racing_line_point_box,
        ):
            box.setReadOnly(True)
            box.setMinimumHeight(104)
        self.racing_line_summary_box.setMaximumHeight(124)
        self.racing_line_point_box.setMaximumHeight(124)
        self.racing_line_progress_label.setWordWrap(True)
        self.racing_line_play_button.setEnabled(False)
        self.racing_line_reset_button.setEnabled(False)
        self.racing_line_map_button.setEnabled(False)
        self.racing_line_timer.setInterval(60)

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tabs.addTab(self._build_run_history_tab(), "Run History")
        self.review_tab_index = self.tabs.addTab(self._build_review_tab(), "Review")
        self.compare_tab_index = self.tabs.addTab(self._build_compare_tab(), "Compare")
        self.agents_tab_index = self.tabs.addTab(self._build_agents_tab(), "Agents")

        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage(self._discovery_summary())

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

        title = QLabel("Live Driver Dashboard")
        title.setFont(_section_font())
        layout.addWidget(title)

        metrics = [
            MetricSpec("speed", "Speed", "--", "km/h"),
            MetricSpec("gear", "Gear", "--"),
            MetricSpec("throttle", "Throttle", "--", "%"),
            MetricSpec("brake", "Brake", "--", "%"),
            MetricSpec("steer", "Steer", "--"),
            MetricSpec("track_pos", "Track Pos", "--"),
            MetricSpec("lap_time", "Lap Time", "--", "s"),
            MetricSpec("damage", "Damage", "--"),
        ]
        metric_grid = QGridLayout()
        for index, metric in enumerate(metrics):
            metric_grid.addWidget(
                self._metric_card(metric),
                index // 4,
                index % 4,
            )
        layout.addLayout(metric_grid)

        road_row = QHBoxLayout()
        road_row.addWidget(self._visual_group("Road Position", self.track_position_widget))
        road_row.addWidget(self._visual_group("Road Ahead", self.road_sensor_widget))
        layout.addLayout(road_row)

        chart_row = QHBoxLayout()
        self.speed_plot = self._build_plot("Speed", "km/h")
        self.speed_plot.setYRange(0.0, 220.0)
        self.speed_curve = self.speed_plot.plot(
            [],
            [],
            pen=pg.mkPen("#2f80ed", width=2),
        )
        chart_row.addWidget(self.speed_plot)

        self.steer_plot = self._build_plot("Steering", "steer")
        self.steer_plot.setYRange(-1.0, 1.0)
        self.steer_curve = self.steer_plot.plot(
            [],
            [],
            pen=pg.mkPen("#8e5cf7", width=2),
        )
        chart_row.addWidget(self.steer_plot)
        layout.addLayout(chart_row)

        self.pedal_plot = self._build_plot("Throttle / Brake", "%")
        self.pedal_plot.setYRange(0.0, 100.0)
        self.throttle_curve = self.pedal_plot.plot(
            [],
            [],
            pen=pg.mkPen("#27ae60", width=2),
        )
        self.brake_curve = self.pedal_plot.plot(
            [],
            [],
            pen=pg.mkPen("#c0392b", width=2),
        )
        layout.addWidget(
            self._chart_legend(
                [
                    ("Throttle", "#27ae60"),
                    ("Brake", "#c0392b"),
                ]
            )
        )
        layout.addWidget(self.pedal_plot)

        return panel

    def _build_explanation_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("What The Car Is Doing")
        title.setFont(_section_font())
        layout.addWidget(title)

        current_group = QGroupBox("Current State")
        current_layout = QVBoxLayout(current_group)
        current_layout.addWidget(self.explanation_status_label)
        current_layout.addWidget(self.explanation_state_label)
        layout.addWidget(current_group)

        reason_group = QGroupBox("Why")
        reason_layout = QVBoxLayout(reason_group)
        reason_layout.addWidget(self.explanation_reason_label)
        layout.addWidget(reason_group)

        log_group = QGroupBox("Event Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.explanation_box)
        layout.addWidget(log_group, 1)

        return panel

    def _build_run_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Run History")
        title.setFont(_section_font())
        layout.addWidget(title)

        self.run_history_source_label.setText(
            f"Source: {self.project_options.run_history_source}"
        )
        layout.addWidget(self.run_history_source_label)

        self.run_history_table.setHorizontalHeaderLabels(
            [
                "Run",
                "Started",
                "Agent",
                "Track",
                "Best Lap",
                "Avg Speed",
                "Off Track",
                "Result",
            ]
        )
        self.run_history_table.horizontalHeader().setStretchLastSection(True)
        self.run_history_table.setAlternatingRowColors(True)
        self.run_history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.run_history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.run_history_table.setMinimumHeight(420)
        self._populate_run_history_table(self.project_options.runs)

        layout.addWidget(self.run_history_table)
        return page

    def _build_review_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_review_replay_panel())
        splitter.addWidget(self._build_review_moment_panel())
        splitter.setSizes([820, 360])
        layout.addWidget(splitter)

        return page

    def _build_review_replay_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self.review_title_label)

        summary_group = QGroupBox("Run Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.review_status_label)
        summary_layout.addWidget(self.review_summary_box)
        layout.addWidget(summary_group)

        controls_group = QGroupBox("Replay Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.addWidget(self.review_play_button)
        controls_layout.addWidget(self.review_slider, 1)
        controls_layout.addWidget(self.review_time_label)
        layout.addWidget(controls_group)

        metrics = [
            MetricSpec("speed", "Speed", "--", "km/h"),
            MetricSpec("gear", "Gear", "--"),
            MetricSpec("throttle", "Throttle", "--", "%"),
            MetricSpec("brake", "Brake", "--", "%"),
            MetricSpec("steer", "Steer", "--"),
            MetricSpec("track_pos", "Track Pos", "--"),
            MetricSpec("lap_time", "Lap Time", "--", "s"),
            MetricSpec("damage", "Damage", "--"),
        ]
        metric_grid = QGridLayout()
        for index, metric in enumerate(metrics):
            metric_grid.addWidget(
                self._review_metric_card(metric),
                index // 4,
                index % 4,
            )
        layout.addLayout(metric_grid)

        road_row = QHBoxLayout()
        road_row.addWidget(
            self._visual_group("Replay Road Position", self.review_track_position_widget)
        )
        road_row.addWidget(
            self._visual_group("Replay Road Ahead", self.review_road_sensor_widget)
        )
        layout.addLayout(road_row)

        chart_row = QHBoxLayout()
        self.review_speed_plot = self._build_plot("Speed", "km/h")
        self.review_speed_plot.setYRange(0.0, 220.0)
        self.review_speed_curve = self.review_speed_plot.plot(
            [],
            [],
            pen=pg.mkPen("#2f80ed", width=2),
        )
        self.review_speed_marker = _add_review_marker(self.review_speed_plot)
        chart_row.addWidget(self.review_speed_plot)

        self.review_steer_plot = self._build_plot("Steering", "steer")
        self.review_steer_plot.setYRange(-1.0, 1.0)
        self.review_steer_curve = self.review_steer_plot.plot(
            [],
            [],
            pen=pg.mkPen("#8e5cf7", width=2),
        )
        self.review_steer_marker = _add_review_marker(self.review_steer_plot)
        chart_row.addWidget(self.review_steer_plot)
        layout.addLayout(chart_row)

        self.review_pedal_plot = self._build_plot("Throttle / Brake", "%")
        self.review_pedal_plot.setYRange(0.0, 100.0)
        self.review_throttle_curve = self.review_pedal_plot.plot(
            [],
            [],
            pen=pg.mkPen("#27ae60", width=2),
        )
        self.review_brake_curve = self.review_pedal_plot.plot(
            [],
            [],
            pen=pg.mkPen("#c0392b", width=2),
        )
        self.review_pedal_marker = _add_review_marker(self.review_pedal_plot)
        layout.addWidget(
            self._chart_legend(
                [
                    ("Throttle", "#27ae60"),
                    ("Brake", "#c0392b"),
                ]
            )
        )
        layout.addWidget(self.review_pedal_plot)

        return panel

    def _build_review_moment_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("At This Moment")
        title.setFont(_section_font())
        layout.addWidget(title)

        current_group = QGroupBox("Replay State")
        current_layout = QVBoxLayout(current_group)
        current_layout.addWidget(self.review_moment_status_label)
        current_layout.addWidget(self.review_moment_state_label)
        layout.addWidget(current_group)

        reason_group = QGroupBox("Why")
        reason_layout = QVBoxLayout(reason_group)
        reason_layout.addWidget(self.review_moment_reason_label)
        layout.addWidget(reason_group)

        layout.addStretch(1)
        return panel

    def _build_compare_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_compare_replay_panel())
        splitter.addWidget(self._build_compare_moment_panel())
        splitter.setSizes([850, 360])
        layout.addWidget(splitter)

        return page

    def _build_compare_replay_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Compare Saved Runs")
        title.setFont(_section_font())
        layout.addWidget(title)

        controls_group = QGroupBox("Selection")
        controls_layout = QGridLayout(controls_group)
        controls_layout.addWidget(QLabel("Track"), 0, 0)
        controls_layout.addWidget(QLabel("Run A"), 0, 1)
        controls_layout.addWidget(QLabel("Run B"), 0, 2)
        controls_layout.addWidget(self.compare_track_combo, 1, 0)
        controls_layout.addWidget(self.compare_run_a_combo, 1, 1)
        controls_layout.addWidget(self.compare_run_b_combo, 1, 2)
        controls_layout.addWidget(self.compare_button, 1, 3)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 2)
        controls_layout.setColumnStretch(2, 2)
        layout.addWidget(controls_group)

        summary_group = QGroupBox("Comparison Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.addWidget(self.compare_summary_label)
        summary_layout.addWidget(self.compare_summary_box)
        layout.addWidget(summary_group)

        replay_group = QGroupBox("Synced Replay")
        replay_layout = QHBoxLayout(replay_group)
        replay_layout.addWidget(self.compare_play_button)
        replay_layout.addWidget(self.compare_slider, 1)
        replay_layout.addWidget(self.compare_time_label)
        layout.addWidget(replay_group)

        self.compare_speed_plot = self._build_plot("Speed Comparison", "km/h")
        self.compare_speed_plot.setMinimumHeight(280)
        self.compare_speed_plot.setYRange(0.0, 220.0)
        self.compare_speed_curve_a = self.compare_speed_plot.plot(
            [],
            [],
            pen=pg.mkPen("#2f80ed", width=2),
        )
        self.compare_speed_curve_b = self.compare_speed_plot.plot(
            [],
            [],
            pen=pg.mkPen("#d97706", width=2),
        )
        self.compare_speed_marker = _add_review_marker(self.compare_speed_plot)

        layout.addWidget(
            self._chart_legend(
                [
                    ("Run A", "#2f80ed"),
                    ("Run B", "#d97706"),
                ]
            )
        )
        layout.addWidget(self.compare_speed_plot)
        layout.addStretch(1)

        return panel

    def _build_compare_moment_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Same Moment")
        title.setFont(_section_font())
        layout.addWidget(title)

        difference_group = QGroupBox("Difference")
        difference_layout = QVBoxLayout(difference_group)
        difference_layout.addWidget(self.compare_delta_label)
        layout.addWidget(difference_group)

        layout.addWidget(
            self._build_compare_side_group(
                "Run A",
                self.compare_a_title_label,
                self.compare_a_status_label,
                self.compare_a_state_label,
                self.compare_a_values_label,
            )
        )
        layout.addWidget(
            self._build_compare_side_group(
                "Run B",
                self.compare_b_title_label,
                self.compare_b_status_label,
                self.compare_b_state_label,
                self.compare_b_values_label,
            )
        )

        layout.addStretch(1)
        return panel

    def _build_compare_side_group(
        self,
        title: str,
        run_label: QLabel,
        status_label: QLabel,
        state_label: QLabel,
        values_label: QLabel,
    ) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        run_label.setFont(_section_font())
        layout.addWidget(run_label)
        layout.addWidget(status_label)
        layout.addWidget(state_label)
        layout.addWidget(values_label)
        return group

    def _build_agents_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_agent_education_left_panel())
        splitter.addWidget(self._build_agent_education_detail_panel())
        splitter.setSizes([330, 850])
        layout.addWidget(splitter)

        return page

    def _build_agent_education_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        title = QLabel("Agents")
        title.setFont(_section_font())
        layout.addWidget(title)
        layout.addWidget(self._combo_group("Driver", self.agent_education_combo))

        profile_group = QGroupBox("Profile")
        profile_layout = QVBoxLayout(profile_group)
        profile_layout.addWidget(self.agent_education_title_label)
        profile_layout.addWidget(self.agent_education_badge_label)
        profile_layout.addWidget(self.agent_education_headline_label)
        profile_layout.addWidget(self.agent_algorithm_button)
        layout.addWidget(profile_group)

        layout.addWidget(
            self._text_group("Project Metadata", self.agent_education_metadata_box)
        )
        layout.addStretch(1)
        return panel

    def _build_agent_education_detail_panel(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(
            self._text_group("What This Driver Is Doing", self.agent_education_overview_box)
        )

        decision_group = QGroupBox("Decision Flow")
        decision_layout = QVBoxLayout(decision_group)
        decision_layout.addWidget(self._build_agent_pipeline())
        layout.addWidget(decision_group)

        layout.addWidget(self._build_racing_line_visualizer_panel())

        first_row = QHBoxLayout()
        first_row.addWidget(
            self._text_group("What It Watches", self.agent_education_signals_box)
        )
        first_row.addWidget(
            self._text_group("Where It Helps", self.agent_education_strengths_box)
        )
        layout.addLayout(first_row)

        second_row = QHBoxLayout()
        second_row.addWidget(
            self._text_group("Failure Signs", self.agent_education_failure_box)
        )
        second_row.addWidget(
            self._text_group("Track Fit", self.agent_education_tracks_box)
        )
        layout.addLayout(second_row)
        layout.addStretch(1)

        scroll_area.setWidget(content)
        return scroll_area

    def _build_agent_pipeline(self) -> QWidget:
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(8)
        self.agent_pipeline_labels.clear()

        for index in range(5):
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setMinimumHeight(88)

            card_layout = QVBoxLayout(card)
            step_label = QLabel(f"Step {index + 1}")
            step_label.setObjectName("metricLabel")
            body_label = QLabel("--")
            body_label.setWordWrap(True)
            body_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self.agent_pipeline_labels.append(body_label)

            card_layout.addWidget(step_label)
            card_layout.addWidget(body_label)
            layout.addWidget(card, index // 3, index % 3)

        return widget

    def _build_racing_line_visualizer_panel(self) -> QGroupBox:
        group = QGroupBox("Racing-Line Visualizer")
        self.racing_line_visualizer_group = group
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Track"))
        controls.addWidget(self.racing_line_track_combo, 1)
        controls.addWidget(self.racing_line_play_button)
        controls.addWidget(self.racing_line_reset_button)
        controls.addWidget(self.racing_line_map_button)
        controls.addWidget(self.racing_line_progress_label)
        layout.addLayout(controls)

        layout.addWidget(self.racing_line_graph_widget)
        layout.addWidget(
            self._chart_legend(
                [
                    ("Brake", "#c0392b"),
                    ("Accelerate", "#27ae60"),
                    ("Full throttle", "#1f7a4d"),
                    ("Turn", "#d97706"),
                    ("Settle", "#2f80ed"),
                ]
            )
        )

        details = QHBoxLayout()
        details.addWidget(self._text_group("Line Summary", self.racing_line_summary_box))
        details.addWidget(self._text_group("Current Target", self.racing_line_point_box))
        layout.addLayout(details)
        return group

    def _combo_group(self, label: str, combo: QComboBox) -> QGroupBox:
        group = QGroupBox(label)
        layout = QVBoxLayout(group)
        layout.addWidget(combo)
        return group

    def _text_group(self, label: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(label)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _visual_group(self, label: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(label)
        layout = QVBoxLayout(group)
        layout.addWidget(widget)
        return group

    def _chart_legend(self, items: list[tuple[str, str]]) -> QWidget:
        legend = QWidget()
        layout = QHBoxLayout(legend)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addStretch(1)

        for label, color in items:
            swatch = QLabel()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #202020;"
            )
            text = QLabel(label)
            layout.addWidget(swatch)
            layout.addWidget(text)

        return legend

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
        self.metric_value_labels[metric.key] = value

        layout.addWidget(label)
        layout.addWidget(value)
        return card

    def _review_metric_card(self, metric: MetricSpec) -> QFrame:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setMinimumHeight(72)

        layout = QVBoxLayout(card)
        label = QLabel(metric.label)
        label.setObjectName("metricLabel")

        value = QLabel(metric.value if not metric.unit else f"{metric.value} {metric.unit}")
        value.setObjectName("metricValue")
        value.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        self.review_metric_value_labels[metric.key] = value

        layout.addWidget(label)
        layout.addWidget(value)
        return card

    def _build_plot(self, title: str, left_axis_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setTitle(title)
        plot.setBackground("w")
        plot.setLabel("left", left_axis_label)
        plot.setLabel("bottom", "lap time", units="s")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.setMinimumHeight(170)
        return plot

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._handle_start_clicked)
        self.stop_button.clicked.connect(self._handle_stop_clicked)
        self.agent_combo.currentIndexChanged.connect(self._handle_agent_changed)
        self.track_combo.currentIndexChanged.connect(self._update_selection_details)
        self.car_combo.currentIndexChanged.connect(self._update_selection_details)
        self.agent_education_combo.currentIndexChanged.connect(
            self._update_agent_education
        )
        self.agent_algorithm_button.clicked.connect(self._open_agent_algorithm_dialog)
        self.run_history_table.cellClicked.connect(self._handle_run_history_clicked)
        self.review_slider.valueChanged.connect(self._handle_review_slider_changed)
        self.review_play_button.clicked.connect(self._handle_review_play_clicked)
        self.review_timer.timeout.connect(self._advance_review_replay)
        self.compare_track_combo.currentIndexChanged.connect(
            self._handle_compare_track_changed
        )
        self.compare_run_a_combo.currentIndexChanged.connect(
            self._update_compare_button_state
        )
        self.compare_run_b_combo.currentIndexChanged.connect(
            self._update_compare_button_state
        )
        self.compare_button.clicked.connect(self._load_selected_comparison)
        self.compare_slider.valueChanged.connect(self._handle_compare_slider_changed)
        self.compare_play_button.clicked.connect(self._handle_compare_play_clicked)
        self.compare_timer.timeout.connect(self._advance_compare_replay)
        self.racing_line_track_combo.currentIndexChanged.connect(
            self._load_selected_racing_line
        )
        self.racing_line_play_button.clicked.connect(self._handle_racing_line_play_clicked)
        self.racing_line_reset_button.clicked.connect(self._reset_racing_line_animation)
        self.racing_line_map_button.clicked.connect(self._open_racing_line_map_dialog)
        self.racing_line_timer.timeout.connect(self._advance_racing_line_animation)

    def _handle_start_clicked(self) -> None:
        if self.race_thread is not None:
            return

        agent = self.selected_agent()
        track = self.selected_track()
        car = self.selected_car()
        if agent is None or track is None or car is None:
            self.status_label.setText("No compatible race setup selected.")
            return

        self._set_race_controls_enabled(False)
        self._reset_live_telemetry()
        self.status_label.setText("Starting race...")
        self.statusBar().showMessage("Starting TORCS worker thread")
        setup_message = f"Starting {agent.name} on {track.track_id} with {car.car_id}."
        self._set_explanation_current(
            "Normal",
            setup_message,
            ["Waiting for the first live telemetry sample."],
        )
        self.explanation_box.setText(f"[setup] {setup_message}")

        self.race_thread = QThread(self)
        self.race_worker = RaceWorker(agent, track, car)
        self.race_worker.moveToThread(self.race_thread)

        self.race_thread.started.connect(self.race_worker.run)
        self.race_worker.status_changed.connect(self._handle_worker_status)
        self.race_worker.explanation_changed.connect(self._handle_worker_explanation)
        self.race_worker.telemetry_received.connect(self._handle_telemetry_received)
        self.race_worker.run_saved.connect(self._handle_run_saved)
        self.race_worker.race_finished.connect(self._handle_race_finished)
        self.race_worker.race_failed.connect(self._handle_race_failed)
        self.race_worker.completed.connect(self.race_thread.quit)
        self.race_worker.completed.connect(self.race_worker.deleteLater)
        self.race_thread.finished.connect(self.race_thread.deleteLater)
        self.race_thread.finished.connect(self._clear_race_thread)
        self.race_thread.start()

    def _handle_stop_clicked(self) -> None:
        if self.race_worker is None:
            self._reset_race_controls()
            return

        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping race...")
        self.statusBar().showMessage("Stop requested")
        self.race_worker.request_stop()

    def selected_agent(self) -> AgentOption | None:
        return self.agent_combo.currentData(Qt.UserRole)

    def selected_track(self) -> TrackOption | None:
        return self.track_combo.currentData(Qt.UserRole)

    def selected_car(self) -> CarOption | None:
        return self.car_combo.currentData(Qt.UserRole)

    def selected_education_agent(self) -> AgentOption | None:
        return self.agent_education_combo.currentData(Qt.UserRole)

    def selected_racing_line_track(self) -> TrackOption | None:
        return self.racing_line_track_combo.currentData(Qt.UserRole)

    def _populate_combo(self, combo: QComboBox, options: list[object]) -> None:
        combo.clear()
        for option in options:
            combo.addItem(option.label, option)

    def _populate_racing_line_tracks(self) -> None:
        self.racing_line_track_combo.clear()
        for track in self.project_options.tracks:
            if track.racing_line_path is not None:
                self.racing_line_track_combo.addItem(track.label, track)

        self._select_combo_item(
            self.racing_line_track_combo,
            "track_id",
            "g-track-3",
        )

    def _select_combo_item(self, combo: QComboBox, attribute: str, value: str) -> bool:
        for index in range(combo.count()):
            option = combo.itemData(index, Qt.UserRole)
            if getattr(option, attribute, None) == value:
                combo.setCurrentIndex(index)
                return True
        return False

    def _handle_agent_changed(self) -> None:
        previous_track = self.selected_track()
        self._refresh_track_options(
            preferred_track_id=previous_track.track_id if previous_track else None
        )
        self._update_selection_details()

    def _refresh_track_options(self, preferred_track_id: str | None = None) -> None:
        agent = self.selected_agent()
        tracks = self.project_options.tracks
        if agent is not None:
            tracks = compatible_tracks_for_agent(agent, tracks)

        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for track in tracks:
            self.track_combo.addItem(_track_label(track), track)

        selected = False
        if preferred_track_id:
            selected = self._select_combo_item(
                self.track_combo,
                "track_id",
                preferred_track_id,
            )
        if not selected and agent is not None and agent.requires_racing_line:
            selected = self._select_combo_item(
                self.track_combo,
                "track_id",
                "g-track-3",
            )
        if not selected and self.track_combo.count() > 0:
            self.track_combo.setCurrentIndex(0)

        self.track_combo.blockSignals(False)
        self.start_button.setEnabled(self.track_combo.count() > 0)

    def _update_agent_education(self) -> None:
        if self.agent_algorithm_dialog is not None:
            self.agent_algorithm_dialog.close()

        agent = self.selected_education_agent()
        if agent is None:
            self.agent_education_profile = None
            self.agent_education_title_label.setText("No agent discovered")
            self.agent_education_badge_label.setText("Idle")
            self.agent_education_headline_label.setText(
                "No project agent metadata is available."
            )
            self.agent_algorithm_button.setEnabled(False)
            self._set_text_box(self.agent_education_metadata_box, ())
            self._set_text_box(self.agent_education_overview_box, ())
            self._set_text_box(self.agent_education_signals_box, ())
            self._set_text_box(self.agent_education_strengths_box, ())
            self._set_text_box(self.agent_education_failure_box, ())
            self._set_text_box(self.agent_education_tracks_box, ())
            for label in self.agent_pipeline_labels:
                label.setText("--")
            self._sync_racing_line_visualizer(None)
            return

        profile = build_agent_education_profile(
            agent,
            self.project_options.tracks,
        )
        self.agent_education_profile = profile
        self.agent_algorithm_button.setEnabled(True)
        self.agent_education_title_label.setText(profile.title)
        self.agent_education_badge_label.setText(profile.badge)
        self.agent_education_headline_label.setText(profile.headline)
        self.agent_education_metadata_box.setPlainText(
            _format_key_value_lines(profile.metadata)
        )
        self._set_text_box(self.agent_education_overview_box, profile.overview)
        self._set_text_box(self.agent_education_signals_box, profile.input_signals)
        self._set_text_box(self.agent_education_strengths_box, profile.strengths)
        self._set_text_box(self.agent_education_failure_box, profile.failure_signs)
        self._set_text_box(self.agent_education_tracks_box, profile.track_context)

        for index, label in enumerate(self.agent_pipeline_labels):
            if index < len(profile.decision_steps):
                label.setText(profile.decision_steps[index])
            else:
                label.setText("--")
        self._sync_racing_line_visualizer(agent)

    def _set_text_box(self, box: QTextEdit, lines: tuple[str, ...]) -> None:
        box.setPlainText(_format_bullet_lines(lines))

    def _open_agent_algorithm_dialog(self) -> None:
        if self.agent_education_profile is None:
            return
        if self.agent_algorithm_dialog is not None:
            self.agent_algorithm_dialog.raise_()
            self.agent_algorithm_dialog.activateWindow()
            return

        dialog = AgentAlgorithmDialog(self.agent_education_profile, self)
        self.agent_algorithm_dialog = dialog
        dialog.finished.connect(
            lambda _result, guide_dialog=dialog: self._handle_agent_algorithm_closed(
                guide_dialog
            )
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _handle_agent_algorithm_closed(
        self,
        dialog: "AgentAlgorithmDialog",
    ) -> None:
        if dialog is self.agent_algorithm_dialog:
            self.agent_algorithm_dialog = None
            dialog.deleteLater()

    def _sync_racing_line_visualizer(self, agent: AgentOption | None) -> None:
        if agent is None or not agent.requires_racing_line:
            self._set_racing_line_unavailable(
                "This driver does not use a saved racing-line file."
            )
            if self.racing_line_visualizer_group is not None:
                self.racing_line_visualizer_group.setVisible(False)
            return

        if self.racing_line_visualizer_group is not None:
            self.racing_line_visualizer_group.setVisible(True)
        self.racing_line_track_combo.setEnabled(True)
        self._load_selected_racing_line()

    def _load_selected_racing_line(self, *_args: object) -> None:
        agent = self.selected_education_agent()
        if agent is not None and not agent.requires_racing_line:
            self._sync_racing_line_visualizer(agent)
            return

        track = self.selected_racing_line_track()
        if track is None or track.racing_line_path is None:
            self._set_racing_line_unavailable("No racing-line track discovered.")
            return

        try:
            profile = load_racing_line_profile(track.racing_line_path)
            visual_profile = build_racing_line_visual_profile(profile, track.path)
        except Exception as exc:
            self._set_racing_line_unavailable(f"Could not load racing line: {exc}")
            return

        self.racing_line_profile = profile
        self.racing_line_visual_profile = visual_profile
        self.racing_line_distance = 0.0
        self.racing_line_graph_widget.set_profile(profile)
        self.racing_line_summary_box.setPlainText(
            _format_bullet_lines(racing_line_summary_lines(profile))
        )
        self.racing_line_play_button.setEnabled(True)
        self.racing_line_reset_button.setEnabled(True)
        self.racing_line_map_button.setEnabled(True)
        self.racing_line_timer.stop()
        self.racing_line_play_button.setText("Play")
        self._set_racing_line_distance(0.0)

    def _set_racing_line_unavailable(self, message: str) -> None:
        if self.racing_line_map_dialog is not None:
            self.racing_line_map_dialog.close()
        self.racing_line_profile = None
        self.racing_line_visual_profile = None
        self.racing_line_distance = 0.0
        self.racing_line_timer.stop()
        self.racing_line_play_button.setText("Play")
        self.racing_line_play_button.setEnabled(False)
        self.racing_line_reset_button.setEnabled(False)
        self.racing_line_map_button.setEnabled(False)
        self.racing_line_graph_widget.set_profile(None)
        self.racing_line_summary_box.setPlainText(message)
        self.racing_line_point_box.setPlainText("No racing-line target.")
        self.racing_line_progress_label.setText("No racing line loaded")

    def _handle_racing_line_play_clicked(self) -> None:
        if self.racing_line_profile is None:
            return
        if self.racing_line_timer.isActive():
            self.racing_line_timer.stop()
            self.racing_line_play_button.setText("Play")
            return

        if self.racing_line_distance >= self.racing_line_profile.track_length_m - 0.1:
            self._set_racing_line_distance(0.0)
        self.racing_line_play_button.setText("Pause")
        self.racing_line_timer.start()

    def _reset_racing_line_animation(self) -> None:
        self.racing_line_timer.stop()
        self.racing_line_play_button.setText("Play")
        self._set_racing_line_distance(0.0)

    def _advance_racing_line_animation(self) -> None:
        if self.racing_line_profile is None:
            self.racing_line_timer.stop()
            self.racing_line_play_button.setText("Play")
            return

        step = max(6.0, self.racing_line_profile.track_length_m / 420.0)
        next_distance = self.racing_line_distance + step
        if next_distance >= self.racing_line_profile.track_length_m:
            self._set_racing_line_distance(self.racing_line_profile.track_length_m)
            self.racing_line_timer.stop()
            self.racing_line_play_button.setText("Play")
            return

        self._set_racing_line_distance(next_distance)

    def _set_racing_line_distance(self, distance: float) -> None:
        if self.racing_line_profile is None:
            return

        self.racing_line_distance = max(
            0.0,
            min(distance, self.racing_line_profile.track_length_m),
        )
        self.racing_line_graph_widget.set_distance(self.racing_line_distance)
        point = self.racing_line_graph_widget.current_point()
        self.racing_line_point_box.setPlainText(racing_line_point_text(point))
        self.racing_line_progress_label.setText(
            f"{self.racing_line_distance:.0f}m / "
            f"{self.racing_line_profile.track_length_m:.0f}m"
        )

    def _open_racing_line_map_dialog(self) -> None:
        if self.racing_line_profile is None or self.racing_line_visual_profile is None:
            return
        if self.racing_line_map_dialog is not None:
            self.racing_line_map_dialog.raise_()
            self.racing_line_map_dialog.activateWindow()
            return

        dialog = RacingLineMapDialog(
            self.racing_line_profile,
            self.racing_line_visual_profile,
            self.racing_line_distance,
            self,
        )
        self.racing_line_map_dialog = dialog
        dialog.finished.connect(
            lambda _result, map_dialog=dialog: self._handle_racing_line_map_closed(
                map_dialog
            )
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _handle_racing_line_map_closed(
        self,
        dialog: "RacingLineMapDialog",
    ) -> None:
        if dialog is self.racing_line_map_dialog:
            self._set_racing_line_distance(dialog.current_distance())
            self.racing_line_map_dialog = None
            dialog.deleteLater()

    def _populate_run_history_table(self, runs: list[RunSummary]) -> None:
        self.run_history_table.setRowCount(max(1, len(runs)))
        if not runs:
            values = ["--", "--", "--", "--", "--", "--", "--", "No saved runs found"]
            for column_index, value in enumerate(values):
                self.run_history_table.setItem(
                    0,
                    column_index,
                    QTableWidgetItem(value),
                )
            return

        for row_index, run in enumerate(runs):
            values = [
                str(run.run_id),
                _short_datetime(run.started_at),
                run.agent_name,
                run.track,
                _format_seconds(run.best_lap_time_seconds),
                _format_speed(run.avg_speed),
                "--" if run.off_track_count is None else str(run.off_track_count),
                run.termination_reason,
            ]
            for column_index, value in enumerate(values):
                self.run_history_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )

        self.run_history_table.resizeColumnsToContents()

    def _handle_run_history_clicked(self, row: int, _column: int) -> None:
        run_id_item = self.run_history_table.item(row, 0)
        if run_id_item is None:
            return

        try:
            run_id = int(run_id_item.text())
        except ValueError:
            return

        self._load_review_run(run_id)
        if self.tabs is not None:
            self.tabs.setCurrentIndex(self.review_tab_index)

    def _load_review_run(self, run_id: int) -> None:
        from storage import RaceRepository

        repository = RaceRepository()
        run = repository.get_run(run_id)
        if run is None:
            QMessageBox.warning(self, "Run Not Found", f"Run #{run_id} was not found.")
            return

        samples = repository.list_run_telemetry(run_id)
        snapshots = [TelemetrySnapshot.from_sample(sample) for sample in samples]
        best_run = repository.best_run_for_track(
            run["track"],
            agent_type=run.get("agent_type"),
        )

        self.review_run = run
        self.review_snapshots = snapshots
        self.review_current_index = 0
        self.review_timer.stop()
        self.review_play_button.setText("Play")
        self._set_review_status(_completion_status(run.get("termination_reason")))
        self.review_title_label.setText(
            f"Run #{run_id} - {run.get('agent_name', 'Unknown Agent')} "
            f"on {run.get('track', 'unknown track')}"
        )

        summary = summarize_run_explanation(snapshots, run, run_id=run_id)
        comparison = _review_comparison_text(run, best_run)
        self.review_summary_box.setText(f"{summary}\n\n{comparison}")

        self._update_review_charts()
        self._configure_review_slider()
        if snapshots:
            self._set_review_current_index(0)
        else:
            self._clear_review_snapshot()

        self.statusBar().showMessage(f"Loaded run #{run_id} for review")

    def _configure_review_slider(self) -> None:
        self.review_slider.blockSignals(True)
        self.review_slider.setMinimum(0)
        self.review_slider.setMaximum(max(0, len(self.review_snapshots) - 1))
        self.review_slider.setValue(0)
        self.review_slider.setEnabled(bool(self.review_snapshots))
        self.review_slider.blockSignals(False)
        self.review_play_button.setEnabled(len(self.review_snapshots) > 1)

    def _clear_review_snapshot(self) -> None:
        self._reset_review_metrics()
        self.review_track_position_widget.set_snapshot(None)
        self.review_road_sensor_widget.set_snapshot(None)
        self._set_review_moment(
            "Idle",
            "No saved telemetry samples for this run.",
            ["New runs will include lightweight sampled telemetry."],
        )
        self.review_time_label.setText("No replay samples")

    def _update_review_charts(self) -> None:
        history = TelemetryHistory(max_points=max(1, len(self.review_snapshots)))
        for snapshot in self.review_snapshots:
            history.append(snapshot)
        x_values = history.chart_times()

        if self.review_speed_curve is not None:
            self.review_speed_curve.setData(x_values, history.values("speed_kmh"))
            self._apply_review_x_range(self.review_speed_plot, x_values)
        if self.review_steer_curve is not None:
            self.review_steer_curve.setData(x_values, history.values("steer"))
            self._apply_review_x_range(self.review_steer_plot, x_values)
        if self.review_throttle_curve is not None:
            self.review_throttle_curve.setData(
                x_values,
                history.values("throttle_pct"),
            )
        if self.review_brake_curve is not None:
            self.review_brake_curve.setData(x_values, history.values("brake_pct"))
            self._apply_review_x_range(self.review_pedal_plot, x_values)

    def _set_review_current_index(self, index: int) -> None:
        if not self.review_snapshots:
            self._clear_review_snapshot()
            return

        self.review_current_index = max(0, min(index, len(self.review_snapshots) - 1))
        snapshot = self.review_snapshots[self.review_current_index]
        self._update_review_metric_values(snapshot)
        self.review_track_position_widget.set_snapshot(snapshot)
        self.review_road_sensor_widget.set_snapshot(snapshot)

        frame = build_explanation_frame(snapshot)
        self._set_review_moment(frame.severity, frame.headline, frame.reasons)
        replay_time = _snapshot_replay_time(snapshot, self.review_current_index)
        finish_time = _review_finish_time(self.review_snapshots)
        self.review_time_label.setText(
            f"{replay_time:.1f}s / {finish_time:.1f}s"
        )
        self._set_review_marker_position(replay_time)

        if self.review_slider.value() != self.review_current_index:
            self.review_slider.blockSignals(True)
            self.review_slider.setValue(self.review_current_index)
            self.review_slider.blockSignals(False)

    def _handle_review_slider_changed(self, value: int) -> None:
        self._set_review_current_index(value)

    def _handle_review_play_clicked(self) -> None:
        if not self.review_snapshots:
            return
        if self.review_timer.isActive():
            self.review_timer.stop()
            self.review_play_button.setText("Play")
            return

        self.review_play_button.setText("Pause")
        self.review_timer.start()

    def _advance_review_replay(self) -> None:
        if not self.review_snapshots:
            self.review_timer.stop()
            self.review_play_button.setText("Play")
            return

        next_index = self.review_current_index + self._review_playback_step_size()
        if next_index >= len(self.review_snapshots):
            self._set_review_current_index(len(self.review_snapshots) - 1)
            self.review_timer.stop()
            self.review_play_button.setText("Play")
            return

        self._set_review_current_index(next_index)

    def _review_playback_step_size(self) -> int:
        return max(1, len(self.review_snapshots) // 240)

    def _update_review_metric_values(self, snapshot: TelemetrySnapshot) -> None:
        self._set_review_metric("speed", format_value(snapshot.speed_kmh, ".1f"))
        self._set_review_metric(
            "gear",
            "--" if snapshot.gear is None else str(snapshot.gear),
        )
        self._set_review_metric("throttle", format_value(snapshot.throttle_pct, ".0f"))
        self._set_review_metric("brake", format_value(snapshot.brake_pct, ".0f"))
        self._set_review_metric("steer", format_value(snapshot.steer, ".3f"))
        self._set_review_metric("track_pos", format_value(snapshot.track_pos, ".3f"))
        self._set_review_metric("lap_time", format_value(snapshot.cur_lap_time, ".2f"))
        self._set_review_metric("damage", format_value(snapshot.damage, ".0f"))

    def _set_review_metric(self, key: str, value: str) -> None:
        label = self.review_metric_value_labels.get(key)
        if label is None:
            return
        unit = _metric_unit(key)
        label.setText(value if not unit else f"{value} {unit}")

    def _reset_review_metrics(self) -> None:
        for key, label in self.review_metric_value_labels.items():
            unit = _metric_unit(key)
            label.setText("--" if not unit else f"-- {unit}")

    def _set_review_marker_position(self, replay_time: float) -> None:
        for marker in (
            self.review_speed_marker,
            self.review_steer_marker,
            self.review_pedal_marker,
        ):
            if marker is not None:
                marker.setValue(replay_time)

    def _apply_review_x_range(
        self,
        plot: pg.PlotWidget | None,
        x_values: list[float],
    ) -> None:
        if plot is None or not x_values:
            return
        plot.setXRange(0.0, max(1.0, x_values[-1]), padding=0.02)

    def _refresh_compare_options(self, preferred_track: str | None = None) -> None:
        from storage import RaceRepository

        current_track = preferred_track or self.compare_track_combo.currentData(
            Qt.UserRole
        )
        try:
            self.compare_runs = RaceRepository().list_runs(limit=500)
        except Exception as exc:
            self.compare_runs = []
            self._clear_compare_selectors()
            self._clear_compare_view(f"Could not load saved runs: {exc}")
            return

        tracks = sorted(
            {
                str(run.get("track"))
                for run in self.compare_runs
                if run.get("track")
            }
        )

        self.compare_track_combo.blockSignals(True)
        self.compare_track_combo.clear()
        for track in tracks:
            run_count = len(self._compare_runs_for_track(track))
            self.compare_track_combo.addItem(f"{track} ({run_count} runs)", track)

        selected = False
        if current_track:
            for index in range(self.compare_track_combo.count()):
                if self.compare_track_combo.itemData(index, Qt.UserRole) == current_track:
                    self.compare_track_combo.setCurrentIndex(index)
                    selected = True
                    break
        if not selected and self.compare_track_combo.count() > 0:
            self.compare_track_combo.setCurrentIndex(0)
        self.compare_track_combo.blockSignals(False)

        self._refresh_compare_run_combos()
        if not tracks:
            self._clear_compare_view("No saved runs are available yet.")

    def _clear_compare_selectors(self) -> None:
        for combo in (
            self.compare_track_combo,
            self.compare_run_a_combo,
            self.compare_run_b_combo,
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        self.compare_button.setEnabled(False)

    def _handle_compare_track_changed(self) -> None:
        self._refresh_compare_run_combos()
        track = self.compare_track_combo.currentData(Qt.UserRole)
        if track:
            self._clear_compare_view(f"Awaiting a {track} comparison.")
        else:
            self._clear_compare_view("No saved runs are available yet.")

    def _refresh_compare_run_combos(self) -> None:
        track = self.compare_track_combo.currentData(Qt.UserRole)
        runs = self._compare_runs_for_track(str(track)) if track else []

        for combo in (self.compare_run_a_combo, self.compare_run_b_combo):
            combo.blockSignals(True)
            combo.clear()
            for run in runs:
                combo.addItem(run_combo_label(run), run)
            combo.blockSignals(False)

        if len(runs) > 1:
            self.compare_run_a_combo.setCurrentIndex(0)
            self.compare_run_b_combo.setCurrentIndex(1)

        self._update_compare_button_state()
        if len(runs) < 2 and track:
            self._clear_compare_view(
                f"{track} needs at least two saved runs before comparison."
            )
        elif runs and self.compare_run_a is None and self.compare_run_b is None:
            self.compare_summary_label.setText("Ready to compare")
            self.compare_summary_box.setText("Selected runs are ready.")

    def _compare_runs_for_track(self, track: str) -> list[dict[str, Any]]:
        return [
            run
            for run in self.compare_runs
            if str(run.get("track")) == track
        ]

    def _update_compare_button_state(self) -> None:
        run_a = self._selected_compare_run(self.compare_run_a_combo)
        run_b = self._selected_compare_run(self.compare_run_b_combo)
        self.compare_button.setEnabled(
            run_a is not None
            and run_b is not None
            and run_a.get("id") != run_b.get("id")
            and run_a.get("track") == run_b.get("track")
        )

    def _selected_compare_run(self, combo: QComboBox) -> dict[str, Any] | None:
        run = combo.currentData(Qt.UserRole)
        return run if isinstance(run, dict) else None

    def _load_selected_comparison(self) -> None:
        from storage import RaceRepository

        selected_a = self._selected_compare_run(self.compare_run_a_combo)
        selected_b = self._selected_compare_run(self.compare_run_b_combo)
        if selected_a is None or selected_b is None:
            self._clear_compare_view("No saved run pair selected.")
            return
        if selected_a.get("id") == selected_b.get("id"):
            self._clear_compare_view("Run A and Run B are the same saved run.")
            return
        if selected_a.get("track") != selected_b.get("track"):
            self._clear_compare_view("Run A and Run B must be from the same track.")
            return

        repository = RaceRepository()
        run_a = repository.get_run(int(selected_a["id"]))
        run_b = repository.get_run(int(selected_b["id"]))
        if run_a is None or run_b is None:
            QMessageBox.warning(self, "Run Not Found", "One selected run was not found.")
            return

        samples_a = repository.list_run_telemetry(int(run_a["id"]))
        samples_b = repository.list_run_telemetry(int(run_b["id"]))
        self.compare_run_a = run_a
        self.compare_run_b = run_b
        self.compare_snapshots_a = [
            TelemetrySnapshot.from_sample(sample) for sample in samples_a
        ]
        self.compare_snapshots_b = [
            TelemetrySnapshot.from_sample(sample) for sample in samples_b
        ]
        self.compare_times_a = snapshot_times(self.compare_snapshots_a)
        self.compare_times_b = snapshot_times(self.compare_snapshots_b)
        self.compare_current_time = 0.0
        self.compare_timer.stop()
        self.compare_play_button.setText("Play")

        summary = compare_runs(
            run_a,
            run_b,
            self.compare_snapshots_a,
            self.compare_snapshots_b,
        )
        self.compare_summary_label.setText(summary.headline)
        self.compare_summary_box.setText("\n".join(summary.details))
        self.compare_a_title_label.setText(run_title(run_a, "Run A"))
        self.compare_b_title_label.setText(run_title(run_b, "Run B"))

        self._update_compare_chart()
        self._configure_compare_slider(summary.finish_time)
        self._set_compare_current_time(0.0)
        self.statusBar().showMessage(
            f"Comparing run #{run_a['id']} and run #{run_b['id']}"
        )

    def _clear_compare_view(self, message: str) -> None:
        self.compare_timer.stop()
        self.compare_play_button.setText("Play")
        self.compare_play_button.setEnabled(False)
        self.compare_slider.blockSignals(True)
        self.compare_slider.setMinimum(0)
        self.compare_slider.setMaximum(0)
        self.compare_slider.setValue(0)
        self.compare_slider.setEnabled(False)
        self.compare_slider.blockSignals(False)
        self.compare_time_label.setText("No comparison loaded")
        self.compare_summary_label.setText("No comparison loaded")
        self.compare_summary_box.setText(message)
        self.compare_delta_label.setText("- No comparison loaded.")
        self.compare_current_time = 0.0
        self.compare_run_a = None
        self.compare_run_b = None
        self.compare_snapshots_a = []
        self.compare_snapshots_b = []
        self.compare_times_a = []
        self.compare_times_b = []
        self.compare_a_title_label.setText("Run A")
        self.compare_b_title_label.setText("Run B")
        self._set_compare_side(
            self.compare_a_status_label,
            self.compare_a_state_label,
            self.compare_a_values_label,
            "Idle",
            "No run selected.",
            "-",
        )
        self._set_compare_side(
            self.compare_b_status_label,
            self.compare_b_state_label,
            self.compare_b_values_label,
            "Idle",
            "No run selected.",
            "-",
        )
        self._update_compare_chart()

    def _configure_compare_slider(self, finish_time: float) -> None:
        max_tick = max(0, int(round(finish_time * 10.0)))
        has_comparable_samples = bool(self.compare_snapshots_a and self.compare_snapshots_b)

        self.compare_slider.blockSignals(True)
        self.compare_slider.setMinimum(0)
        self.compare_slider.setMaximum(max_tick)
        self.compare_slider.setValue(0)
        self.compare_slider.setEnabled(has_comparable_samples and max_tick > 0)
        self.compare_slider.blockSignals(False)
        self.compare_play_button.setEnabled(has_comparable_samples and max_tick > 0)

    def _update_compare_chart(self) -> None:
        if self.compare_speed_curve_a is not None:
            self.compare_speed_curve_a.setData(
                self.compare_times_a,
                _snapshot_values(self.compare_snapshots_a, "speed_kmh"),
            )
        if self.compare_speed_curve_b is not None:
            self.compare_speed_curve_b.setData(
                self.compare_times_b,
                _snapshot_values(self.compare_snapshots_b, "speed_kmh"),
            )
        if self.compare_speed_marker is not None:
            self.compare_speed_marker.setValue(self.compare_current_time)
        self._apply_compare_x_range()

    def _apply_compare_x_range(self) -> None:
        if self.compare_speed_plot is None:
            return
        finish_time = self._compare_finish_time()
        self.compare_speed_plot.setXRange(0.0, max(1.0, finish_time), padding=0.02)

    def _set_compare_current_time(self, target_time: float) -> None:
        if not self.compare_snapshots_a and not self.compare_snapshots_b:
            self.compare_time_label.setText("No replay samples")
            return

        finish_time = self._compare_finish_time()
        self.compare_current_time = max(0.0, min(target_time, finish_time))
        snapshot_a = self._compare_snapshot_at_time(
            self.compare_snapshots_a,
            self.compare_times_a,
            self.compare_current_time,
        )
        snapshot_b = self._compare_snapshot_at_time(
            self.compare_snapshots_b,
            self.compare_times_b,
            self.compare_current_time,
        )

        self._set_compare_side_from_snapshot(
            self.compare_a_status_label,
            self.compare_a_state_label,
            self.compare_a_values_label,
            snapshot_a,
        )
        self._set_compare_side_from_snapshot(
            self.compare_b_status_label,
            self.compare_b_state_label,
            self.compare_b_values_label,
            snapshot_b,
        )
        self.compare_delta_label.setText(
            _format_reason_lines(moment_comparison_reasons(snapshot_a, snapshot_b))
        )
        self.compare_time_label.setText(
            f"{self.compare_current_time:.1f}s / {finish_time:.1f}s"
        )
        if self.compare_speed_marker is not None:
            self.compare_speed_marker.setValue(self.compare_current_time)

        slider_value = int(round(self.compare_current_time * 10.0))
        slider_value = max(
            self.compare_slider.minimum(),
            min(slider_value, self.compare_slider.maximum()),
        )
        if self.compare_slider.value() != slider_value:
            self.compare_slider.blockSignals(True)
            self.compare_slider.setValue(slider_value)
            self.compare_slider.blockSignals(False)

    def _compare_snapshot_at_time(
        self,
        snapshots: list[TelemetrySnapshot],
        times: list[float],
        replay_time: float,
    ) -> TelemetrySnapshot | None:
        if not snapshots or not times:
            return None
        return snapshots[nearest_snapshot_index(times, replay_time)]

    def _set_compare_side_from_snapshot(
        self,
        status_label: QLabel,
        state_label: QLabel,
        values_label: QLabel,
        snapshot: TelemetrySnapshot | None,
    ) -> None:
        severity, headline, values = moment_panel_text(snapshot)
        self._set_compare_side(
            status_label,
            state_label,
            values_label,
            severity,
            headline,
            values,
        )

    def _set_compare_side(
        self,
        status_label: QLabel,
        state_label: QLabel,
        values_label: QLabel,
        severity: str,
        headline: str,
        values: str,
    ) -> None:
        status_label.setText(severity)
        status_label.setStyleSheet(_severity_style(severity))
        state_label.setText(headline)
        values_label.setText(values)

    def _handle_compare_slider_changed(self, value: int) -> None:
        self._set_compare_current_time(value / 10.0)

    def _handle_compare_play_clicked(self) -> None:
        if not (self.compare_snapshots_a and self.compare_snapshots_b):
            return
        if self.compare_timer.isActive():
            self.compare_timer.stop()
            self.compare_play_button.setText("Play")
            return

        self.compare_play_button.setText("Pause")
        self.compare_timer.start()

    def _advance_compare_replay(self) -> None:
        if not (self.compare_snapshots_a and self.compare_snapshots_b):
            self.compare_timer.stop()
            self.compare_play_button.setText("Play")
            return

        finish_time = self._compare_finish_time()
        next_time = self.compare_current_time + self._compare_playback_step_seconds()
        if next_time >= finish_time:
            self._set_compare_current_time(finish_time)
            self.compare_timer.stop()
            self.compare_play_button.setText("Play")
            return

        self._set_compare_current_time(next_time)

    def _compare_playback_step_seconds(self) -> float:
        return max(0.1, self._compare_finish_time() / 240.0)

    def _compare_finish_time(self) -> float:
        return comparison_finish_time(
            self.compare_snapshots_a,
            self.compare_snapshots_b,
        )

    def _update_selection_details(self) -> None:
        agent = self.selected_agent()
        track = self.selected_track()
        car = self.selected_car()
        if agent is None or track is None or car is None:
            return

        self.statusBar().showMessage(
            f"{self._discovery_summary()} Selected: {agent.agent_type} / "
            f"{track.track_id} / {car.car_id}. "
            f"{self.track_combo.count()} compatible tracks shown."
        )
        self.agent_combo.setToolTip(
            f"{agent.class_path}\n"
            f"Full control: {'yes' if agent.uses_full_control else 'no'}\n"
            f"Requires racing line: {'yes' if agent.requires_racing_line else 'no'}\n"
            f"Target laps: {_format_optional(agent.target_laps)}"
        )
        self.track_combo.setToolTip(
            f"{track.track_id}\n"
            f"Category: {track.category}\n"
            f"Racing line: {_format_path(track.racing_line_path)}\n"
            f"{track.path}"
        )
        self.car_combo.setToolTip(
            f"{car.car_id}\nCategory: {car.category}\n{car.path}"
        )

    def _handle_worker_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def _handle_telemetry_received(self, snapshot: object) -> None:
        if not isinstance(snapshot, TelemetrySnapshot):
            return

        self.telemetry_history.append(snapshot)
        self._update_metric_values(snapshot)
        self.track_position_widget.set_snapshot(snapshot)
        self.road_sensor_widget.set_snapshot(snapshot)
        self._update_telemetry_charts()
        explanation_frame = build_explanation_frame(snapshot)
        self._set_explanation_current(
            explanation_frame.severity,
            explanation_frame.headline,
            explanation_frame.reasons,
        )
        if self._should_append_explanation(snapshot, explanation_frame):
            self._append_explanation(
                format_explanation_entry(snapshot, explanation_frame)
            )

    def _handle_run_saved(self, run_id: int) -> None:
        self._reload_run_history()
        self.statusBar().showMessage(f"Saved race run #{run_id}")

    def _handle_race_finished(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._set_explanation_current(
                "Normal",
                "Race finished.",
                ["No structured run summary was returned."],
            )
            self._append_explanation("Race finished.")
            return

        results = payload.get("results", {}) or {}
        run_id = payload.get("run_id")
        termination = results.get("termination_reason", "finished")
        summary = summarize_run_explanation(
            self.telemetry_history.snapshots(),
            results,
            run_id=run_id,
        )
        self._set_explanation_current(
            _completion_status(termination),
            (
                "Race complete."
                if termination == "target_laps_completed"
                else f"Race ended by {termination}."
            ),
            summary.splitlines(),
        )
        self._append_explanation(summary)

    def _handle_race_failed(self, traceback_text: str) -> None:
        summary = (
            traceback_text.strip().splitlines()[-1]
            if traceback_text
            else "Unknown error"
        )
        self.status_label.setText("Race failed")
        self.statusBar().showMessage(summary)
        self._set_explanation_current(
            "At Risk",
            "Race failed.",
            [summary],
        )
        self.explanation_box.setText(traceback_text)
        QMessageBox.critical(self, "Race Failed", summary)

    def _clear_race_thread(self) -> None:
        self.race_thread = None
        self.race_worker = None
        self._reset_race_controls()

    def _reset_live_telemetry(self) -> None:
        self.telemetry_history.clear()
        self._last_explanation_step = None
        self._last_explanation_lap_time = None
        self._last_explanation_event_key = None
        for key, label in self.metric_value_labels.items():
            unit = _metric_unit(key)
            label.setText("--" if not unit else f"-- {unit}")
        self.track_position_widget.set_snapshot(None)
        self.road_sensor_widget.set_snapshot(None)
        self._update_telemetry_charts()
        self._set_explanation_current(
            "Idle",
            "Telemetry idle.",
            ["No live sample yet."],
        )
        self.explanation_box.setText("Race log idle.")

    def _update_metric_values(self, snapshot: TelemetrySnapshot) -> None:
        self._set_metric("speed", format_value(snapshot.speed_kmh, ".1f"))
        self._set_metric("gear", "--" if snapshot.gear is None else str(snapshot.gear))
        self._set_metric("throttle", format_value(snapshot.throttle_pct, ".0f"))
        self._set_metric("brake", format_value(snapshot.brake_pct, ".0f"))
        self._set_metric("steer", format_value(snapshot.steer, ".3f"))
        self._set_metric("track_pos", format_value(snapshot.track_pos, ".3f"))
        self._set_metric("lap_time", format_value(snapshot.cur_lap_time, ".2f"))
        self._set_metric("damage", format_value(snapshot.damage, ".0f"))

    def _set_metric(self, key: str, value: str) -> None:
        label = self.metric_value_labels.get(key)
        if label is None:
            return
        unit = _metric_unit(key)
        label.setText(value if not unit else f"{value} {unit}")

    def _update_telemetry_charts(self) -> None:
        x_values, speed_values = self.telemetry_history.windowed_series(
            "speed_kmh",
            self.chart_window_seconds,
        )
        if self.speed_curve is not None:
            self.speed_curve.setData(x_values, speed_values)
            self._apply_chart_x_range(self.speed_plot, x_values)

        x_values, steer_values = self.telemetry_history.windowed_series(
            "steer",
            self.chart_window_seconds,
        )
        if self.steer_curve is not None:
            self.steer_curve.setData(x_values, steer_values)
            self._apply_chart_x_range(self.steer_plot, x_values)

        x_values, throttle_values = self.telemetry_history.windowed_series(
            "throttle_pct",
            self.chart_window_seconds,
        )
        if self.throttle_curve is not None:
            self.throttle_curve.setData(x_values, throttle_values)

        x_values, brake_values = self.telemetry_history.windowed_series(
            "brake_pct",
            self.chart_window_seconds,
        )
        if self.brake_curve is not None:
            self.brake_curve.setData(x_values, brake_values)
            self._apply_chart_x_range(self.pedal_plot, x_values)

    def _apply_chart_x_range(
        self,
        plot: pg.PlotWidget | None,
        x_values: list[float],
    ) -> None:
        if plot is None or not x_values:
            return
        right = max(self.chart_window_seconds, x_values[-1])
        left = max(0.0, right - self.chart_window_seconds)
        plot.setXRange(left, right, padding=0.0)

    def _handle_worker_explanation(self, message: str) -> None:
        if len(self.telemetry_history) == 0:
            self._set_explanation_current(
                "Normal",
                message,
                ["Waiting for the first live telemetry sample."],
            )
            self.explanation_box.setText(message)
            return
        self._append_explanation(message)

    def _should_append_explanation(
        self,
        snapshot: TelemetrySnapshot,
        explanation_frame: ExplanationFrame,
    ) -> bool:
        event_key = explanation_frame.event_key
        first_entry = self._last_explanation_step is None
        previous_event_key = self._last_explanation_event_key
        important_change = (
            previous_event_key is not None
            and event_key != previous_event_key
            and (
                event_key in {
                    "off_track",
                    "damage",
                    "stuck",
                    "very_limited_road",
                    "track_edge",
                    "heavy_braking",
                }
                or previous_event_key
                in {
                    "off_track",
                    "damage",
                    "stuck",
                    "very_limited_road",
                    "track_edge",
                    "heavy_braking",
                }
            )
        )
        periodic_update = self._explanation_interval_elapsed(snapshot)

        if first_entry or important_change or periodic_update:
            self._last_explanation_step = snapshot.step
            self._last_explanation_lap_time = snapshot.cur_lap_time
            self._last_explanation_event_key = event_key
            return True
        return False

    def _explanation_interval_elapsed(self, snapshot: TelemetrySnapshot) -> bool:
        if self._last_explanation_step is None:
            return True
        if (
            snapshot.cur_lap_time is not None
            and self._last_explanation_lap_time is not None
        ):
            return (
                snapshot.cur_lap_time - self._last_explanation_lap_time
                >= self.explanation_interval_seconds
            )
        return snapshot.step - self._last_explanation_step >= self.explanation_interval_steps

    def _set_explanation_current(
        self,
        severity: str,
        headline: str,
        reasons: tuple[str, ...] | list[str],
    ) -> None:
        self.explanation_status_label.setText(severity)
        self.explanation_status_label.setStyleSheet(_severity_style(severity))
        self.explanation_state_label.setText(headline)
        self.explanation_reason_label.setText(_format_reason_lines(reasons))

    def _set_review_status(self, severity: str) -> None:
        self.review_status_label.setText(severity)
        self.review_status_label.setStyleSheet(_severity_style(severity))

    def _set_review_moment(
        self,
        severity: str,
        headline: str,
        reasons: tuple[str, ...] | list[str],
    ) -> None:
        self.review_moment_status_label.setText(severity)
        self.review_moment_status_label.setStyleSheet(_severity_style(severity))
        self.review_moment_state_label.setText(headline)
        self.review_moment_reason_label.setText(_format_reason_lines(reasons))

    def _append_explanation(self, message: str) -> None:
        if self.explanation_box.toPlainText().strip():
            self.explanation_box.append("")
        self.explanation_box.append(message)
        self._scroll_explanation_to_latest()
        QTimer.singleShot(0, self._scroll_explanation_to_latest)

    def _scroll_explanation_to_latest(self) -> None:
        self.explanation_box.moveCursor(QTextCursor.MoveOperation.End)
        self.explanation_box.ensureCursorVisible()
        scroll_bar = self.explanation_box.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _set_race_controls_enabled(self, enabled: bool) -> None:
        self.agent_combo.setEnabled(enabled)
        self.track_combo.setEnabled(enabled)
        self.car_combo.setEnabled(enabled)
        self.start_button.setEnabled(enabled and self.track_combo.count() > 0)
        self.stop_button.setEnabled(not enabled)

    def _reset_race_controls(self) -> None:
        self._set_race_controls_enabled(True)
        if self.status_label.text() in {"Stopping race...", "Starting race..."}:
            self.status_label.setText("Idle")
        self._update_selection_details()

    def _reload_run_history(self) -> None:
        runs, source = discover_run_history()
        self.project_options = ProjectOptions(
            agents=self.project_options.agents,
            tracks=self.project_options.tracks,
            cars=self.project_options.cars,
            runs=runs,
            run_history_source=source,
        )
        self.run_history_source_label.setText(f"Source: {source}")
        self._populate_run_history_table(runs)
        self._refresh_compare_options()

    def _discovery_summary(self) -> str:
        return (
            f"Discovered {len(self.project_options.agents)} agents, "
            f"{len(self.project_options.tracks)} tracks, "
            f"{len(self.project_options.cars)} cars, "
            f"{len(self.project_options.runs)} saved runs."
        )


class AgentAlgorithmDialog(QDialog):
    def __init__(
        self,
        profile: AgentEducationProfile,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Algorithm Guide - {profile.title}")
        self.resize(960, 700)

        layout = QVBoxLayout(self)
        title = QLabel(profile.title)
        title.setFont(_section_font())
        subtitle = QLabel(profile.badge)
        subtitle.setAlignment(Qt.AlignLeft)
        subtitle.setStyleSheet(_agent_badge_style())
        subtitle.setMinimumHeight(28)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        tabs = QTabWidget()
        tabs.addTab(self._build_overview_tab(profile), "Overview")
        tabs.addTab(self._build_pseudocode_tab(profile), "Algorithm")
        if profile.formula_notes:
            tabs.addTab(self._build_formula_tab(profile), "Formulas")
        if profile.code_snippets:
            tabs.addTab(self._build_code_tab(profile), "Code")
        tabs.addTab(self._build_math_tab(profile), "Interpretation")
        layout.addWidget(tabs, 1)

        controls = QHBoxLayout()
        controls.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        controls.addWidget(close_button)
        layout.addLayout(controls)

    def _build_overview_tab(self, profile: AgentEducationProfile) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(
            _standalone_text_group(
                "How The Algorithm Thinks",
                _guide_text_box(_format_bullet_lines(profile.algorithm_summary)),
            )
        )
        layout.addWidget(
            _standalone_text_group(
                "What Users Should Notice",
                _guide_text_box(_format_bullet_lines(profile.key_takeaways)),
            )
        )
        layout.addStretch(1)
        return _scrollable_dialog_page(content)

    def _build_pseudocode_tab(self, profile: AgentEducationProfile) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(
            _standalone_text_group(
                "Readable Algorithm",
                _guide_text_box(
                    _format_numbered_lines(profile.pseudocode),
                    monospace=True,
                    minimum_height=360,
                ),
            )
        )
        layout.addStretch(1)
        return _scrollable_dialog_page(content)

    def _build_formula_tab(self, profile: AgentEducationProfile) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)

        for note in profile.formula_notes:
            group = QGroupBox(note.title)
            group_layout = QVBoxLayout(group)

            explanation = QLabel(note.explanation)
            explanation.setWordWrap(True)

            group_layout.addWidget(_formula_display_widget(note.formula))
            group_layout.addWidget(explanation)
            layout.addWidget(group)

        layout.addStretch(1)
        return _scrollable_dialog_page(content)

    def _build_code_tab(self, profile: AgentEducationProfile) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)

        for snippet in profile.code_snippets:
            group = QGroupBox(snippet.title)
            group_layout = QVBoxLayout(group)

            source = QLabel(snippet.source)
            source.setObjectName("metricLabel")
            source.setWordWrap(True)
            explanation = QLabel(snippet.explanation)
            explanation.setWordWrap(True)

            line_count = max(4, snippet.code.count("\n") + 1)
            code_box = _guide_text_box(
                snippet.code,
                monospace=True,
                minimum_height=min(260, 34 + line_count * 18),
            )
            code_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

            group_layout.addWidget(source)
            group_layout.addWidget(explanation)
            group_layout.addWidget(code_box)
            layout.addWidget(group)

        layout.addStretch(1)
        return _scrollable_dialog_page(content)

    def _build_math_tab(self, profile: AgentEducationProfile) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(
            _standalone_text_group(
                "Maths / Logic In Human Terms",
                _guide_text_box(
                    _format_bullet_lines(profile.math_notes),
                    minimum_height=420,
                ),
            )
        )
        layout.addStretch(1)
        return _scrollable_dialog_page(content)


class RacingLineMapDialog(QDialog):
    def __init__(
        self,
        profile: RacingLineProfile,
        visual_profile: RacingLineVisualProfile,
        initial_distance: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Racing Line Map - {profile.track_name}")
        self.resize(1120, 760)

        self.profile = profile
        self.distance = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(60)

        self.map_widget = RacingLineWidget()
        self.map_widget.setMinimumHeight(500)
        self.map_widget.set_profile(visual_profile)
        self.map_widget.set_follow_enabled(True)
        self.play_button = QPushButton("Play")
        self.reset_button = QPushButton("Reset")
        self.view_button = QPushButton("Zoom Out")
        self.progress_label = QLabel("0m")
        self.progress_label.setMinimumWidth(120)
        self.summary_box = QTextEdit()
        self.point_box = QTextEdit()
        for box in (self.summary_box, self.point_box):
            box.setReadOnly(True)
            box.setMinimumHeight(118)
        self.summary_box.setPlainText(
            _format_bullet_lines(racing_line_summary_lines(profile))
        )

        layout = QVBoxLayout(self)
        title = QLabel("Actual Track Racing Line")
        title.setFont(_section_font())
        layout.addWidget(title)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.reset_button)
        controls.addWidget(self.view_button)
        controls.addWidget(self.progress_label)
        controls.addStretch(1)
        layout.addLayout(controls)

        layout.addWidget(self.map_widget)
        layout.addWidget(_phase_legend_widget())

        details = QHBoxLayout()
        details.addWidget(_standalone_text_group("Line Summary", self.summary_box))
        details.addWidget(_standalone_text_group("Current Target", self.point_box))
        layout.addLayout(details)

        self.play_button.clicked.connect(self._handle_play_clicked)
        self.reset_button.clicked.connect(self._reset_animation)
        self.view_button.clicked.connect(self._toggle_view_mode)
        self.timer.timeout.connect(self._advance_animation)
        self._set_distance(initial_distance)

    def current_distance(self) -> float:
        return self.distance

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowDeactivate:
            self.close()
        return super().event(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        super().closeEvent(event)

    def _handle_play_clicked(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
            return

        if self.distance >= self.profile.track_length_m - 0.1:
            self._set_distance(0.0)
        self.play_button.setText("Pause")
        self.timer.start()

    def _reset_animation(self) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self._set_distance(0.0)

    def _toggle_view_mode(self) -> None:
        follow_enabled = not self.map_widget.follow_enabled()
        self.map_widget.set_follow_enabled(follow_enabled)
        self.view_button.setText("Zoom Out" if follow_enabled else "Follow")

    def _advance_animation(self) -> None:
        step = max(6.0, self.profile.track_length_m / 420.0)
        next_distance = self.distance + step
        if next_distance >= self.profile.track_length_m:
            self._set_distance(self.profile.track_length_m)
            self.timer.stop()
            self.play_button.setText("Play")
            return

        self._set_distance(next_distance)

    def _set_distance(self, distance: float) -> None:
        self.distance = max(0.0, min(distance, self.profile.track_length_m))
        self.map_widget.set_distance(self.distance)
        self.point_box.setPlainText(racing_line_point_text(self.map_widget.current_point()))
        self.progress_label.setText(
            f"{self.distance:.0f}m / {self.profile.track_length_m:.0f}m"
        )


def _scrollable_dialog_page(content: QWidget) -> QScrollArea:
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(content)
    return scroll_area


def _guide_text_box(
    text: str,
    *,
    monospace: bool = False,
    minimum_height: int = 180,
) -> QTextEdit:
    box = QTextEdit()
    box.setReadOnly(True)
    box.setPlainText(text)
    box.setMinimumHeight(minimum_height)
    if monospace:
        box.setFont(_monospace_font())
    return box


def _formula_display_widget(formula: str) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    pixmap = _render_formula_pixmap(formula)
    if pixmap is not None and not pixmap.isNull():
        formula_label = QLabel()
        formula_label.setPixmap(pixmap)
        formula_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        formula_label.setMinimumHeight(pixmap.height() + 12)
        formula_label.setStyleSheet(
            "background-color: #f8f8f6; "
            "border: 1px solid #555555; "
            "padding: 8px;"
        )
        layout.addWidget(formula_label)
        return container

    formula_box = _guide_text_box(
        formula,
        monospace=True,
        minimum_height=min(150, 44 + formula.count("\n") * 20),
    )
    formula_box.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
    layout.addWidget(formula_box)
    return container


def _render_formula_pixmap(formula: str) -> QPixmap | None:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception:
        return None

    try:
        line_count = max(1, formula.count("\n") + 1)
        figure = Figure(
            figsize=(8.4, max(0.78, 0.48 * line_count)),
            dpi=145,
            facecolor="#f8f8f6",
        )
        canvas = FigureCanvasAgg(figure)
        figure.text(
            0.02,
            0.52,
            formula,
            ha="left",
            va="center",
            color="#111111",
            fontsize=17,
            math_fontfamily="dejavuserif",
        )

        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            pad_inches=0.18,
            facecolor=figure.get_facecolor(),
        )
        pixmap = QPixmap()
        if not pixmap.loadFromData(buffer.getvalue(), "PNG"):
            return None
        return pixmap
    except Exception:
        return None


def _format_numbered_lines(lines: tuple[str, ...]) -> str:
    if not lines:
        return "1. --"
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


def _monospace_font() -> QFont:
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(9)
    return font


def _add_review_marker(plot: pg.PlotWidget) -> Any:
    marker = pg.InfiniteLine(
        pos=0.0,
        angle=90,
        movable=False,
        pen=pg.mkPen("#202020", width=1),
    )
    plot.addItem(marker)
    return marker


def _snapshot_values(
    snapshots: list[TelemetrySnapshot],
    attribute: str,
) -> list[float]:
    values: list[float] = []
    for snapshot in snapshots:
        value = getattr(snapshot, attribute)
        values.append(float(value) if value is not None else float("nan"))
    return values


def _review_comparison_text(
    run: dict[str, Any],
    best_run: dict[str, Any] | None,
) -> str:
    selected_lap = run.get("best_lap_time_seconds")
    if not isinstance(selected_lap, (int, float)):
        return "Comparison: this run has no completed lap time yet."
    if best_run is None:
        return "Comparison: no completed baseline run found for this track."

    best_lap = best_run.get("best_lap_time_seconds")
    if not isinstance(best_lap, (int, float)):
        return "Comparison: no completed baseline run found for this track."

    delta = selected_lap - best_lap
    if run.get("id") == best_run.get("id"):
        return (
            f"Comparison: this is the best saved comparable lap for "
            f"{run.get('track', 'this track')} at {selected_lap:.3f}s."
        )

    return (
        f"Comparison: best comparable is run #{best_run.get('id')} at "
        f"{best_lap:.3f}s. This run is {_format_delta(delta)}."
    )


def _format_delta(delta: float) -> str:
    if abs(delta) < 0.001:
        return "equal to the baseline"
    if delta > 0:
        return f"+{delta:.3f}s slower"
    return f"{abs(delta):.3f}s faster"


def _snapshot_replay_time(snapshot: TelemetrySnapshot, fallback_index: int) -> float:
    if snapshot.cur_lap_time is not None and snapshot.cur_lap_time >= 0.0:
        return snapshot.cur_lap_time
    return float(fallback_index)


def _review_finish_time(snapshots: list[TelemetrySnapshot]) -> float:
    if not snapshots:
        return 0.0
    return max(
        _snapshot_replay_time(snapshot, index)
        for index, snapshot in enumerate(snapshots)
    )


def _section_font() -> QFont:
    font = QFont()
    font.setPointSize(13)
    font.setBold(True)
    return font


def _standalone_text_group(label: str, widget: QWidget) -> QGroupBox:
    group = QGroupBox(label)
    layout = QVBoxLayout(group)
    layout.addWidget(widget)
    return group


def _phase_legend_widget() -> QWidget:
    legend = QWidget()
    layout = QHBoxLayout(legend)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addStretch(1)

    for label, color in (
        ("Brake", "#c0392b"),
        ("Accelerate", "#27ae60"),
        ("Full throttle", "#1f7a4d"),
        ("Turn", "#d97706"),
        ("Settle", "#2f80ed"),
    ):
        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid #202020;"
        )
        text = QLabel(label)
        layout.addWidget(swatch)
        layout.addWidget(text)

    return legend


def _format_bullet_lines(lines: tuple[str, ...]) -> str:
    if not lines:
        return "--"
    return "\n".join(f"- {line}" for line in lines)


def _format_key_value_lines(items: tuple[tuple[str, str], ...]) -> str:
    if not items:
        return "--"
    return "\n".join(f"{label}: {value}" for label, value in items)


def _format_optional(value: object) -> str:
    return "--" if value is None else str(value)


def _metric_unit(key: str) -> str:
    return {
        "speed": "km/h",
        "throttle": "%",
        "brake": "%",
        "lap_time": "s",
    }.get(key, "")


def _completion_status(termination: object) -> str:
    if termination in {"stuck", "launch_stuck"}:
        return "Stuck"
    if termination in {"crashed", "off_track", "torcs_closed"}:
        return "At Risk"
    if termination == "stopped":
        return "Caution"
    return "Normal"


def _format_reason_lines(reasons: tuple[str, ...] | list[str]) -> str:
    if not reasons:
        return "- No active reason."
    return "\n".join(f"- {reason}" for reason in reasons)


def _severity_style(severity: str) -> str:
    palette = {
        "Idle": ("#3b3b3b", "#d0d0d0"),
        "Normal": ("#1f7a4d", "#ffffff"),
        "Correcting": ("#265d97", "#ffffff"),
        "Caution": ("#8a5a00", "#ffffff"),
        "At Risk": ("#9f1f18", "#ffffff"),
        "Stuck": ("#6f1d1b", "#ffffff"),
    }
    background, foreground = palette.get(severity, palette["Idle"])
    return (
        f"background-color: {background}; "
        f"color: {foreground}; "
        "border-radius: 4px; "
        "font-weight: 700; "
        "padding: 4px 8px;"
    )


def _agent_badge_style() -> str:
    return (
        "background-color: #234b63; "
        "color: #ffffff; "
        "border-radius: 4px; "
        "font-weight: 700; "
        "padding: 4px 8px;"
    )


def _format_seconds(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}s"


def _format_speed(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 50.0:.1f} km/h"


def _format_path(value: object) -> str:
    return "not generated" if value is None else str(value)


def _track_label(track: TrackOption) -> str:
    suffix = " - raceline ready" if track.has_racing_line else ""
    return f"{track.label}{suffix}"


def _short_datetime(value: str) -> str:
    if not value:
        return "--"
    return value.replace("T", " ").replace("+00:00", " UTC")[:19]
