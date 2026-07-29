from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
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
    from .telemetry_model import (
        TelemetryHistory,
        TelemetrySnapshot,
        explanation_event_key,
        format_explanation_entry,
        format_value,
    )
except ImportError:
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
    from telemetry_model import (
        TelemetryHistory,
        TelemetrySnapshot,
        explanation_event_key,
        format_explanation_entry,
        format_value,
    )


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
        self.explanation_box = QTextEdit()
        self.run_history_table = QTableWidget(0, 8)
        self.run_history_source_label = QLabel()
        self.race_thread: QThread | None = None
        self.race_worker: RaceWorker | None = None
        self.telemetry_history = TelemetryHistory()
        self.metric_value_labels: dict[str, QLabel] = {}
        self.speed_curve: Any = None
        self.steer_curve: Any = None
        self.throttle_curve: Any = None
        self.brake_curve: Any = None
        self.explanation_interval_steps = 200
        self._last_explanation_step: int | None = None
        self._last_explanation_event_key: str | None = None

        self._configure_controls()
        self._build_ui()
        self._connect_signals()
        self._update_selection_details()

    def _configure_controls(self) -> None:
        self._populate_combo(self.agent_combo, self.project_options.agents)
        self._refresh_track_options(preferred_track_id="g-track-3")
        self._populate_combo(self.car_combo, self.project_options.cars)
        self._select_combo_item(self.car_combo, "car_id", "car1-ow1")

        self.stop_button.setEnabled(False)
        self.status_label.setObjectName("statusLabel")

        self.explanation_box.setReadOnly(True)
        self.explanation_box.document().setMaximumBlockCount(240)
        self.explanation_box.setText(
            "Project discovery is connected. Live race explanations will appear "
            "here once telemetry is connected."
        )

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        tabs.addTab(self._build_run_history_tab(), "Run History")

        self.setCentralWidget(tabs)
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

        title = QLabel("Live Telemetry")
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

        chart_row = QHBoxLayout()
        speed_plot = self._build_plot("Speed", "km/h")
        self.speed_curve = speed_plot.plot(
            [],
            [],
            pen=pg.mkPen("#2f80ed", width=2),
        )
        chart_row.addWidget(speed_plot)

        steering_plot = self._build_plot("Steering", "steer")
        steering_plot.setYRange(-1.0, 1.0)
        self.steer_curve = steering_plot.plot(
            [],
            [],
            pen=pg.mkPen("#8e5cf7", width=2),
        )
        chart_row.addWidget(steering_plot)
        layout.addLayout(chart_row)

        pedal_plot = self._build_plot("Throttle / Brake", "%")
        pedal_plot.setYRange(0.0, 100.0)
        self.throttle_curve = pedal_plot.plot(
            [],
            [],
            pen=pg.mkPen("#27ae60", width=2),
            name="Throttle",
        )
        self.brake_curve = pedal_plot.plot(
            [],
            [],
            pen=pg.mkPen("#c0392b", width=2),
            name="Brake",
        )
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
        self.metric_value_labels[metric.key] = value

        layout.addWidget(label)
        layout.addWidget(value)
        return card

    def _build_plot(self, title: str, left_axis_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setTitle(title)
        plot.setBackground("w")
        plot.setLabel("left", left_axis_label)
        plot.setLabel("bottom", "step")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setMinimumHeight(190)
        return plot

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._handle_start_clicked)
        self.stop_button.clicked.connect(self._handle_stop_clicked)
        self.agent_combo.currentIndexChanged.connect(self._handle_agent_changed)
        self.track_combo.currentIndexChanged.connect(self._update_selection_details)
        self.car_combo.currentIndexChanged.connect(self._update_selection_details)

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
        self.explanation_box.setText(
            f"Starting {agent.name} on {track.track_id} with {car.car_id}."
        )

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

    def _populate_combo(self, combo: QComboBox, options: list[object]) -> None:
        combo.clear()
        for option in options:
            combo.addItem(option.label, option)

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
        self._update_telemetry_charts()
        if self._should_append_explanation(snapshot):
            self._append_explanation(format_explanation_entry(snapshot))

    def _handle_run_saved(self, run_id: int) -> None:
        self._reload_run_history()
        self.statusBar().showMessage(f"Saved race run #{run_id}")

    def _handle_race_finished(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self.explanation_box.setText("Race finished.")
            return

        results = payload.get("results", {}) or {}
        run_id = payload.get("run_id")
        termination = results.get("termination_reason", "finished")
        laps = results.get("laps_completed", 0)
        best_lap = results.get("best_lap_time_seconds")
        saved_text = f" Saved as run #{run_id}." if run_id is not None else ""
        lap_text = (
            f" Best lap: {best_lap:.3f}s."
            if isinstance(best_lap, (int, float))
            else ""
        )
        self._append_explanation(
            f"Race ended by {termination}. Laps completed: {laps}.{lap_text}{saved_text}"
        )

    def _handle_race_failed(self, traceback_text: str) -> None:
        summary = (
            traceback_text.strip().splitlines()[-1]
            if traceback_text
            else "Unknown error"
        )
        self.status_label.setText("Race failed")
        self.statusBar().showMessage(summary)
        self.explanation_box.setText(traceback_text)
        QMessageBox.critical(self, "Race Failed", summary)

    def _clear_race_thread(self) -> None:
        self.race_thread = None
        self.race_worker = None
        self._reset_race_controls()

    def _reset_live_telemetry(self) -> None:
        self.telemetry_history.clear()
        self._last_explanation_step = None
        self._last_explanation_event_key = None
        for key, label in self.metric_value_labels.items():
            unit = _metric_unit(key)
            label.setText("--" if not unit else f"-- {unit}")
        self._update_telemetry_charts()

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
        steps = self.telemetry_history.steps()
        if self.speed_curve is not None:
            self.speed_curve.setData(steps, self.telemetry_history.values("speed_kmh"))
        if self.steer_curve is not None:
            self.steer_curve.setData(steps, self.telemetry_history.values("steer"))
        if self.throttle_curve is not None:
            self.throttle_curve.setData(
                steps,
                self.telemetry_history.values("throttle_pct"),
            )
        if self.brake_curve is not None:
            self.brake_curve.setData(steps, self.telemetry_history.values("brake_pct"))

    def _handle_worker_explanation(self, message: str) -> None:
        if len(self.telemetry_history) == 0:
            self.explanation_box.setText(message)
            return
        self._append_explanation(message)

    def _should_append_explanation(self, snapshot: TelemetrySnapshot) -> bool:
        event_key = explanation_event_key(snapshot)
        first_entry = self._last_explanation_step is None
        step_delta = (
            self.explanation_interval_steps
            if self._last_explanation_step is None
            else snapshot.step - self._last_explanation_step
        )
        important_change = (
            self._last_explanation_event_key is not None
            and event_key != self._last_explanation_event_key
            and event_key
            in {
                "off_track",
                "damage",
                "very_limited_road",
                "track_edge",
                "braking",
            }
        )

        if (
            first_entry
            or important_change
            or step_delta >= self.explanation_interval_steps
        ):
            self._last_explanation_step = snapshot.step
            self._last_explanation_event_key = event_key
            return True
        return False

    def _append_explanation(self, message: str) -> None:
        scroll_bar = self.explanation_box.verticalScrollBar()
        was_at_bottom = scroll_bar.value() >= scroll_bar.maximum() - 4

        if self.explanation_box.toPlainText().strip():
            self.explanation_box.append("")
        self.explanation_box.append(message)

        if was_at_bottom:
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

    def _discovery_summary(self) -> str:
        return (
            f"Discovered {len(self.project_options.agents)} agents, "
            f"{len(self.project_options.tracks)} tracks, "
            f"{len(self.project_options.cars)} cars, "
            f"{len(self.project_options.runs)} saved runs."
        )


def _section_font() -> QFont:
    font = QFont()
    font.setPointSize(13)
    font.setBold(True)
    return font


def _format_optional(value: object) -> str:
    return "--" if value is None else str(value)


def _metric_unit(key: str) -> str:
    return {
        "speed": "km/h",
        "throttle": "%",
        "brake": "%",
        "lap_time": "s",
    }.get(key, "")


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
