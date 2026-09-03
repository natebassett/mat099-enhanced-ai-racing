from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from .results_model import (
        EvaluationBatch,
        RaceAgentSummary,
        ResultsDataset,
        TrainingRun,
        featured_evaluation,
        filter_evaluations,
        load_results_dataset,
        representative_evaluations,
    )
    from .theme import (
        ChartPalette,
        LIGHT_THEME,
        ThemePalette,
        chart_palette_for_preferences,
    )
except ImportError:
    from results_model import (
        EvaluationBatch,
        RaceAgentSummary,
        ResultsDataset,
        TrainingRun,
        featured_evaluation,
        filter_evaluations,
        load_results_dataset,
        representative_evaluations,
    )
    from theme import (
        ChartPalette,
        LIGHT_THEME,
        ThemePalette,
        chart_palette_for_preferences,
    )


PACE_TARGET_SECONDS = 90.0


class ResultsView(QWidget):
    """Read-only analytics workspace backed by saved project evidence."""

    def __init__(self, project_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.theme_palette = LIGHT_THEME
        self.chart_palette = chart_palette_for_preferences(
            "standard",
            dark=False,
        )
        self.dataset = ResultsDataset((), (), (), (), ())
        self._evaluation_rows: list[EvaluationBatch] = []
        self._training_rows: list[TrainingRun] = []
        self.content_tabs = QTabWidget()

        self.overview_title_label = QLabel("No evaluated neural racer found")
        self.overview_status_label = QLabel("Awaiting evidence")
        self.overview_summary_label = QLabel()
        self.overview_detail_label = QLabel()
        self.overview_metrics = self._make_metric_labels(
            ("completion", "typical_lap", "fastest_lap", "trials")
        )
        self.overview_agent_labels: dict[str, QLabel] = {}
        self.overview_reliability_bars: dict[str, QProgressBar] = {}
        self.overview_pace_labels: dict[str, QLabel] = {}
        self.technical_evidence_page: QWidget | None = None

        self.evaluation_track_combo = QComboBox()
        self.evaluation_agent_combo = QComboBox()
        self.evaluation_batch_combo = QComboBox()
        self.evaluation_table = QTableWidget(0, 8)
        self.evaluation_source_label = QLabel("No evaluation selected.")
        self.evaluation_contract_label = QLabel()
        self.evaluation_metrics = self._make_metric_labels(
            ("trials", "completion", "median_lap", "fastest_lap", "minimum_progress")
        )
        self.pace_reliability_plot = self._make_plot(
            "Median completed lap (s)", "Completion rate (%)"
        )
        self.pace_reliability_plot.addLegend(offset=(8, 8))
        self.trial_distance_plot = self._make_plot("Trial", "Distance (m)")

        self.training_track_combo = QComboBox()
        self.training_agent_combo = QComboBox()
        self.training_run_combo = QComboBox()
        self.training_table = QTableWidget(0, 6)
        self.training_story_label = QLabel()
        self.training_source_label = QLabel("No training run selected.")
        self.training_contract_label = QLabel()
        self.training_metrics = self._make_metric_labels(
            ("episodes", "interactions", "completed", "fastest_lap", "farthest")
        )
        self.training_distance_plot = self._make_plot(
            "Training steps", "Distance (m)"
        )
        self.training_distance_plot.addLegend(offset=(8, 8))
        self.training_lap_plot = self._make_plot(
            "Training steps", "Completed lap (s)"
        )

        self.race_track_combo = QComboBox()
        self.race_table = QTableWidget(0, 7)
        self.race_source_label = QLabel("No saved race results found.")
        self.race_metrics = self._make_metric_labels(
            ("agents", "runs", "completed", "best_lap")
        )
        self.race_pace_plot = self._make_plot("Best lap (s)", "Agent")
        self.race_reliability_plot = self._make_plot("Completion rate (%)", "Agent")
        self.race_pace_plot.getAxis("left").setWidth(112)
        self.race_reliability_plot.getAxis("left").setWidth(112)

        self.refresh_button = QPushButton("Refresh evidence")
        self.summary_label = QLabel()
        self.warning_label = QLabel()

        self._build_ui()
        self._connect_signals()
        self.reload()

    def set_visual_theme(
        self,
        palette: ThemePalette,
        chart_palette: ChartPalette,
    ) -> None:
        self.theme_palette = palette
        self.chart_palette = chart_palette
        for plot in (
            self.pace_reliability_plot,
            self.trial_distance_plot,
            self.training_distance_plot,
            self.training_lap_plot,
            self.race_pace_plot,
            self.race_reliability_plot,
        ):
            plot.setBackground(palette.surface)
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setTextPen(palette.muted)
                axis.setPen(palette.border_strong)
        self.reload()

    def reload(self) -> None:
        evaluation_id = self.evaluation_batch_combo.currentData()
        training_id = self.training_run_combo.currentData()
        self.dataset = load_results_dataset(self.project_root)
        self._populate_filters()
        self._refresh_overview()
        self._refresh_evaluation_options(preferred_id=evaluation_id)
        self._refresh_training_options(preferred_id=training_id)
        self._refresh_race_results()
        self.summary_label.setText(
            f"Evidence loaded: {len(self.dataset.evaluation_batches):,} evaluations  |  "
            f"{len(self.dataset.training_runs):,} training sessions  |  "
            f"{len(self.dataset.run_history):,} races"
        )
        self.warning_label.setVisible(bool(self.dataset.warnings))
        self.warning_label.setText("  |  ".join(self.dataset.warnings[:3]))

    def take_technical_evidence_page(self) -> QWidget | None:
        page = self.technical_evidence_page
        if page is None:
            return None
        self.technical_evidence_page = None
        page.setParent(None)
        page.show()
        return page

    def _refresh_overview(self) -> None:
        track_batches = [
            batch
            for batch in self.dataset.evaluation_batches
            if batch.track == "g-track-3"
        ]
        batches = track_batches or list(self.dataset.evaluation_batches)
        featured = featured_evaluation(batches)
        representatives = {
            batch.agent_family: batch
            for batch in representative_evaluations(batches)
        }

        if featured is None:
            self.overview_title_label.setText("No evaluated neural racer found")
            self.overview_status_label.setText("Awaiting evidence")
            self.overview_summary_label.setText(
                "Run a repeated evaluation to populate this overview."
            )
            self.overview_detail_label.clear()
            self._set_metric_values(self.overview_metrics, {})
        else:
            completed = featured.completed_laps
            trials = featured.trials
            typical = featured.median_lap_time_seconds
            self.overview_title_label.setText(
                _friendly_agent_name(featured.agent_family)
            )
            self.overview_status_label.setText(_result_status(featured))
            self.overview_summary_label.setText(
                f"Completed {completed} of {trials} evaluation trials. "
                f"A typical successful lap took {_seconds(typical)}."
            )
            if typical is not None:
                difference = PACE_TARGET_SECONDS - typical
                pace_message = (
                    f"{abs(difference):.3f} seconds below the 90-second target"
                    if difference >= 0.0
                    else f"{abs(difference):.3f} seconds above the 90-second target"
                )
            else:
                pace_message = "No completed-lap time was recorded"
            self.overview_detail_label.setText(
                f"{pace_message}. Result saved {featured.timestamp} on "
                f"{featured.track}."
            )
            self._set_metric_values(
                self.overview_metrics,
                {
                    "completion": f"{completed} / {trials}",
                    "typical_lap": _seconds(typical),
                    "fastest_lap": _seconds(featured.fastest_lap_time_seconds),
                    "trials": str(trials),
                },
            )

        for family in ("agent6", "agent7", "agent8"):
            batch = representatives.get(family)
            bar = self.overview_reliability_bars[family]
            pace_label = self.overview_pace_labels[family]
            if batch is None:
                bar.setValue(0)
                bar.setFormat("No evaluation")
                pace_label.setText("--")
                continue
            percentage = round(batch.completion_rate * 100.0)
            bar.setValue(percentage)
            bar.setFormat(
                f"{batch.completed_laps}/{batch.trials} laps ({percentage}%)"
            )
            if batch.median_lap_time_seconds is not None:
                pace_label.setText(_seconds(batch.median_lap_time_seconds))
            else:
                pace_label.setText(
                    f"{_metres(batch.median_distance_m)} typical progress"
                )

    def _build_ui(self) -> None:
        self.setObjectName("pageCanvas")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        title = QLabel("Results & Evidence")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Inspect pace, reliability, and learning progress from saved experiment data."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack, 1)
        self.refresh_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.refresh_button.setToolTip("Reload saved evaluations, training logs, and races")
        header.addWidget(self.refresh_button)
        root.addLayout(header)

        self.content_tabs.setObjectName("contentTabs")
        self.content_tabs.addTab(self._build_overview_tab(), "Overview")
        self.content_tabs.addTab(self._build_training_tab(), "Learning Journey")
        self.content_tabs.addTab(self._build_race_tab(), "Agent Comparison")
        root.addWidget(self.content_tabs, 1)

        # Preserved for the later Settings > Research Evidence workspace.
        self.technical_evidence_page = self._build_evaluation_tab()
        self.technical_evidence_page.setParent(self)
        self.technical_evidence_page.hide()

        footer = QHBoxLayout()
        self.summary_label.setObjectName("mutedText")
        self.warning_label.setObjectName("resultsWarning")
        self.warning_label.setWordWrap(True)
        footer.addWidget(self.summary_label)
        footer.addStretch(1)
        footer.addWidget(self.warning_label)
        root.addLayout(footer)

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        feature = QFrame()
        feature.setProperty("surface", True)
        feature_layout = QGridLayout(feature)
        feature_layout.setContentsMargins(16, 13, 16, 13)
        feature_layout.setHorizontalSpacing(16)
        feature_layout.setVerticalSpacing(5)

        eyebrow = QLabel("FASTEST RELIABLE NEURAL RACER")
        eyebrow.setObjectName("metricLabel")
        self.overview_title_label.setObjectName("overviewResultTitle")
        self.overview_title_label.setWordWrap(True)
        self.overview_status_label.setObjectName("learningModeLabel")
        self.overview_status_label.setAlignment(Qt.AlignCenter)
        self.overview_status_label.setMinimumWidth(170)
        self.overview_summary_label.setWordWrap(True)
        self.overview_detail_label.setObjectName("mutedText")
        self.overview_detail_label.setWordWrap(True)

        feature_layout.addWidget(eyebrow, 0, 0)
        feature_layout.addWidget(self.overview_status_label, 0, 1, 2, 1)
        feature_layout.addWidget(self.overview_title_label, 1, 0)
        feature_layout.addWidget(self.overview_summary_label, 2, 0, 1, 2)
        feature_layout.addWidget(self.overview_detail_label, 3, 0, 1, 2)
        feature_layout.setColumnStretch(0, 1)
        layout.addWidget(feature)

        layout.addWidget(
            self._metric_strip(
                (
                    ("Laps Completed", self.overview_metrics["completion"]),
                    ("Typical Lap", self.overview_metrics["typical_lap"]),
                    ("Fastest Lap", self.overview_metrics["fastest_lap"]),
                    ("Evaluation Trials", self.overview_metrics["trials"]),
                )
            )
        )

        comparison = QFrame()
        comparison.setProperty("surface", True)
        comparison.setMinimumHeight(220)
        comparison.setMaximumHeight(250)
        comparison_layout = QGridLayout(comparison)
        comparison_layout.setContentsMargins(16, 12, 16, 14)
        comparison_layout.setHorizontalSpacing(14)
        comparison_layout.setVerticalSpacing(9)

        comparison_title = QLabel("Neural Agent Results")
        comparison_title.setFont(_section_font())
        comparison_subtitle = QLabel(
            "Results use at least 10 evaluation attempts when they are available."
        )
        comparison_subtitle.setObjectName("mutedText")
        comparison_subtitle.setWordWrap(True)
        comparison_layout.addWidget(comparison_title, 0, 0, 1, 3)
        comparison_layout.addWidget(comparison_subtitle, 1, 0, 1, 3)

        agent_heading = QLabel("Agent")
        reliability_heading = QLabel("Laps completed")
        pace_heading = QLabel("Typical lap")
        for heading in (agent_heading, reliability_heading, pace_heading):
            heading.setObjectName("metricLabel")
        comparison_layout.addWidget(agent_heading, 2, 0)
        comparison_layout.addWidget(reliability_heading, 2, 1)
        comparison_layout.addWidget(pace_heading, 2, 2)

        for row, family in enumerate(("agent6", "agent7", "agent8"), start=3):
            agent_label = QLabel(_friendly_agent_name(family))
            agent_label.setObjectName("overviewAgentName")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("No completed laps")
            bar.setMinimumHeight(26)
            pace_label = QLabel("--")
            pace_label.setObjectName("factValue")
            pace_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pace_label.setMinimumWidth(190)
            self.overview_agent_labels[family] = agent_label
            self.overview_reliability_bars[family] = bar
            self.overview_pace_labels[family] = pace_label
            comparison_layout.addWidget(agent_label, row, 0)
            comparison_layout.addWidget(bar, row, 1)
            comparison_layout.addWidget(pace_label, row, 2)
        comparison_layout.setColumnStretch(1, 1)
        layout.addWidget(comparison)

        note = QLabel(
            "A fast lap shows capability. Repeated lap completion shows reliability. "
            "Both are kept visible so one exceptional run does not hide failures."
        )
        note.setObjectName("resultsInsight")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _build_evaluation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        filters = QGridLayout()
        filters.setHorizontalSpacing(8)
        filters.addWidget(self._field("Track", self.evaluation_track_combo), 0, 0)
        filters.addWidget(self._field("Agent", self.evaluation_agent_combo), 0, 1)
        filters.addWidget(self._field("Evaluation batch", self.evaluation_batch_combo), 0, 2)
        filters.setColumnStretch(0, 1)
        filters.setColumnStretch(1, 1)
        filters.setColumnStretch(2, 3)
        layout.addLayout(filters)

        layout.addWidget(
            self._metric_strip(
                (
                    ("Trials", self.evaluation_metrics["trials"]),
                    ("Completion", self.evaluation_metrics["completion"]),
                    ("Median Lap", self.evaluation_metrics["median_lap"]),
                    ("Fastest Lap", self.evaluation_metrics["fastest_lap"]),
                    ("Minimum Progress", self.evaluation_metrics["minimum_progress"]),
                )
            )
        )

        plots = QSplitter(Qt.Horizontal)
        plots.setChildrenCollapsible(False)
        plots.addWidget(
            self._plot_panel(
                "Pace vs Reliability",
                "Every point is one saved evaluation batch; upper-left is better.",
                self.pace_reliability_plot,
            )
        )
        plots.addWidget(
            self._plot_panel(
                "Selected Trial Distance",
                "Green trials completed a lap; red trials terminated early.",
                self.trial_distance_plot,
            )
        )
        plots.setSizes([520, 520])

        evidence = QWidget()
        evidence_layout = QVBoxLayout(evidence)
        evidence_layout.setContentsMargins(0, 0, 0, 0)
        evidence_layout.setSpacing(5)
        self._configure_table(
            self.evaluation_table,
            (
                "Agent",
                "Policy",
                "Date",
                "Trials",
                "Completed",
                "Median Lap",
                "Min Progress",
                "Source",
            ),
        )
        self.evaluation_table.setMinimumHeight(150)
        evidence_layout.addWidget(self.evaluation_table)
        self._prepare_evidence_label(self.evaluation_source_label)
        self._prepare_evidence_label(self.evaluation_contract_label)
        evidence_layout.addWidget(self.evaluation_source_label)
        evidence_layout.addWidget(self.evaluation_contract_label)

        vertical = QSplitter(Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        vertical.addWidget(plots)
        vertical.addWidget(evidence)
        vertical.setSizes([330, 230])
        layout.addWidget(vertical, 1)
        return page

    def _build_training_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        filters.addWidget(self._field("Driver", self.training_agent_combo), 1)
        filters.addWidget(
            self._field("Training session", self.training_run_combo),
            3,
        )
        layout.addLayout(filters)

        layout.addWidget(
            self._metric_strip(
                (
                    ("Episodes", self.training_metrics["episodes"]),
                    ("Completed Laps", self.training_metrics["completed"]),
                    ("Fastest Lap", self.training_metrics["fastest_lap"]),
                    ("Farthest", self.training_metrics["farthest"]),
                )
            )
        )

        self.training_story_label.setObjectName("resultsInsight")
        self.training_story_label.setWordWrap(True)
        layout.addWidget(self.training_story_label)

        plots = QSplitter(Qt.Horizontal)
        plots.setChildrenCollapsible(False)
        plots.addWidget(
            self._plot_panel(
                "How Far the Agent Travelled",
                "Each episode and the furthest distance reached so far.",
                self.training_distance_plot,
            )
        )
        plots.addWidget(
            self._plot_panel(
                "Lap Time Improvement",
                "Completed laps compared with the 90-second goal.",
                self.training_lap_plot,
            )
        )
        plots.setSizes([520, 520])
        layout.addWidget(plots, 1)
        return page

    def _build_race_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(self._field("Track", self.race_track_combo), 1)
        filter_row.addStretch(3)
        layout.addLayout(filter_row)
        layout.addWidget(
            self._metric_strip(
                (
                    ("Agents", self.race_metrics["agents"]),
                    ("Saved Runs", self.race_metrics["runs"]),
                    ("Completed Runs", self.race_metrics["completed"]),
                    ("Best Lap", self.race_metrics["best_lap"]),
                )
            )
        )

        plots = QSplitter(Qt.Horizontal)
        plots.setChildrenCollapsible(False)
        plots.addWidget(
            self._plot_panel(
                "Best Saved Lap",
                "Fastest completed GUI run for each agent; lower is better.",
                self.race_pace_plot,
            )
        )
        plots.addWidget(
            self._plot_panel(
                "Observed Completion Rate",
                "Completed GUI runs divided by all saved runs for the selected track.",
                self.race_reliability_plot,
            )
        )

        evidence = QWidget()
        evidence_layout = QVBoxLayout(evidence)
        evidence_layout.setContentsMargins(0, 0, 0, 0)
        evidence_layout.setSpacing(5)
        self._configure_table(
            self.race_table,
            (
                "Agent",
                "Runs",
                "Completed",
                "Completion",
                "Best Lap",
                "Median Lap",
                "Median Speed",
            ),
        )
        self.race_table.setMinimumHeight(150)
        evidence_layout.addWidget(self.race_table)
        self._prepare_evidence_label(self.race_source_label)
        evidence_layout.addWidget(self.race_source_label)

        vertical = QSplitter(Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        vertical.addWidget(plots)
        vertical.addWidget(evidence)
        vertical.setSizes([330, 230])
        layout.addWidget(vertical, 1)
        return page

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self.reload)
        self.evaluation_track_combo.currentIndexChanged.connect(
            self._refresh_evaluation_options
        )
        self.evaluation_agent_combo.currentIndexChanged.connect(
            self._refresh_evaluation_options
        )
        self.evaluation_batch_combo.currentIndexChanged.connect(
            self._refresh_selected_evaluation
        )
        self.evaluation_table.itemSelectionChanged.connect(
            self._select_evaluation_from_table
        )
        self.training_track_combo.currentIndexChanged.connect(
            self._refresh_training_options
        )
        self.training_agent_combo.currentIndexChanged.connect(
            self._refresh_training_options
        )
        self.training_run_combo.currentIndexChanged.connect(
            self._refresh_selected_training
        )
        self.race_track_combo.currentIndexChanged.connect(self._refresh_race_results)

    def _populate_filters(self) -> None:
        tracks = self.dataset.tracks
        evaluation_agents = sorted(
            {
                batch.agent_family: batch.agent_label
                for batch in self.dataset.evaluation_batches
            }.items()
        )
        training_agents = sorted(
            {
                run.agent_family: run.agent_label
                for run in self.dataset.training_runs
            }.items()
        )
        all_tracks = (("All tracks", "All tracks"),) + tuple(
            (track, track) for track in tracks
        )
        self._replace_combo(self.evaluation_track_combo, all_tracks)
        self._replace_combo(self.training_track_combo, all_tracks)
        race_tracks = sorted({item.track for item in self.dataset.race_summaries})
        self._replace_combo(
            self.race_track_combo,
            tuple((track, track) for track in race_tracks),
            preferred="g-track-3",
        )
        self._replace_combo(
            self.evaluation_agent_combo,
            (("All agents", "All agents"),)
            + tuple((label, key) for key, label in evaluation_agents),
        )
        self._replace_combo(
            self.training_agent_combo,
            (("All agents", "All agents"),)
            + tuple((label, key) for key, label in training_agents),
        )

    def _refresh_evaluation_options(self, *_: Any, preferred_id: Any = None) -> None:
        track = str(self.evaluation_track_combo.currentData() or "All tracks")
        agent = str(self.evaluation_agent_combo.currentData() or "All agents")
        self._evaluation_rows = filter_evaluations(
            self.dataset.evaluation_batches,
            track=track,
            agent_family=agent,
        )
        current_id = preferred_id or self.evaluation_batch_combo.currentData()
        self._replace_combo(
            self.evaluation_batch_combo,
            tuple((batch.label, batch.batch_id) for batch in self._evaluation_rows),
            preferred=current_id,
        )
        self._populate_evaluation_table()
        self._draw_pace_reliability()
        self._refresh_selected_evaluation()

    def _refresh_selected_evaluation(self, *_: Any) -> None:
        selected_id = self.evaluation_batch_combo.currentData()
        batch = next(
            (item for item in self._evaluation_rows if item.batch_id == selected_id),
            None,
        )
        self.trial_distance_plot.clear()
        if batch is None:
            self._set_metric_values(self.evaluation_metrics, {})
            self.evaluation_source_label.setText("No evaluation selected.")
            self.evaluation_contract_label.clear()
            return

        self._set_metric_values(
            self.evaluation_metrics,
            {
                "trials": f"{batch.trials}",
                "completion": f"{batch.completion_rate * 100:.0f}%",
                "median_lap": _seconds(batch.median_lap_time_seconds),
                "fastest_lap": _seconds(batch.fastest_lap_time_seconds),
                "minimum_progress": _metres(batch.minimum_distance_m),
            },
        )
        self.evaluation_source_label.setText(f"Source: {batch.source_path}")
        self.evaluation_source_label.setToolTip(str(batch.source_path))
        self.evaluation_contract_label.setText(
            f"Policy: {batch.policy_path}  |  Observation: {batch.observation_version}  |  "
            f"Action: {batch.action_version}  |  Reward: {batch.reward_version}"
        )
        self.evaluation_contract_label.setToolTip(self.evaluation_contract_label.text())

        x_values = list(range(1, batch.trials + 1))
        completed_x = [x for x, episode in zip(x_values, batch.episodes) if episode.completed]
        completed_y = [episode.distance_m for episode in batch.episodes if episode.completed]
        failed_x = [x for x, episode in zip(x_values, batch.episodes) if not episode.completed]
        failed_y = [episode.distance_m for episode in batch.episodes if not episode.completed]
        if completed_x:
            self.trial_distance_plot.plot(
                completed_x,
                completed_y,
                pen=None,
                symbol="o",
                symbolSize=9,
                symbolBrush=self.chart_palette.complete,
                symbolPen=self.chart_palette.complete,
            )
            finish_distance = _median(completed_y)
            self.trial_distance_plot.addItem(
                pg.InfiniteLine(
                    pos=finish_distance,
                    angle=0,
                    pen=pg.mkPen(
                        self.chart_palette.complete,
                        width=1,
                        style=Qt.DashLine,
                    ),
                    label="lap",
                    labelOpts={
                        "color": self.chart_palette.complete,
                        "position": 0.92,
                    },
                )
            )
        if failed_x:
            self.trial_distance_plot.plot(
                failed_x,
                failed_y,
                pen=None,
                symbol="x",
                symbolSize=10,
                symbolPen=pg.mkPen(self.chart_palette.failure, width=2),
            )
        self.trial_distance_plot.enableAutoRange()

    def _draw_pace_reliability(self) -> None:
        self.pace_reliability_plot.clear()
        for family in ("agent6", "agent7", "agent8", "other"):
            batches = [
                batch
                for batch in self._evaluation_rows
                if batch.agent_family == family
                and batch.median_lap_time_seconds is not None
            ]
            if not batches:
                continue
            self.pace_reliability_plot.plot(
                [batch.median_lap_time_seconds for batch in batches],
                [batch.completion_rate * 100.0 for batch in batches],
                pen=None,
                symbol={
                    "agent6": "t",
                    "agent7": "s",
                    "agent8": "o",
                    "other": "d",
                }[family],
                symbolSize=[max(7, min(14, 6 + math.sqrt(batch.trials))) for batch in batches],
                symbolBrush=self._agent_colour(family),
                symbolPen=pg.mkPen(
                    self.chart_palette.marker_outline,
                    width=1,
                ),
                name=batches[0].agent_label,
            )
        self.pace_reliability_plot.addItem(
            pg.InfiniteLine(
                pos=PACE_TARGET_SECONDS,
                angle=90,
                pen=pg.mkPen(
                    self.chart_palette.target,
                    width=1,
                    style=Qt.DashLine,
                ),
                label="90 s",
                labelOpts={
                    "color": self.chart_palette.target,
                    "position": 0.88,
                },
            )
        )
        self.pace_reliability_plot.setYRange(0.0, 105.0, padding=0.02)

    def _populate_evaluation_table(self) -> None:
        table = self.evaluation_table
        table.setRowCount(len(self._evaluation_rows))
        for row, batch in enumerate(self._evaluation_rows):
            values = (
                batch.agent_label,
                batch.policy_label,
                batch.timestamp,
                str(batch.trials),
                f"{batch.completed_laps}/{batch.trials} ({batch.completion_rate * 100:.0f}%)",
                _seconds(batch.median_lap_time_seconds),
                _metres(batch.minimum_distance_m),
                batch.source_path.name,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, batch.batch_id)
                if column == 7:
                    item.setToolTip(str(batch.source_path))
                table.setItem(row, column, item)
        table.resizeRowsToContents()

    def _select_evaluation_from_table(self) -> None:
        row = self.evaluation_table.currentRow()
        item = self.evaluation_table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        batch_id = item.data(Qt.UserRole)
        index = self.evaluation_batch_combo.findData(batch_id)
        if index >= 0:
            self.evaluation_batch_combo.setCurrentIndex(index)

    def _refresh_training_options(self, *_: Any, preferred_id: Any = None) -> None:
        track = str(self.training_track_combo.currentData() or "All tracks")
        agent = str(self.training_agent_combo.currentData() or "All agents")
        self._training_rows = [
            run
            for run in self.dataset.training_runs
            if (track == "All tracks" or run.track == track)
            and (agent == "All agents" or run.agent_family == agent)
        ]
        current_id = preferred_id or self.training_run_combo.currentData()
        self._replace_combo(
            self.training_run_combo,
            tuple((run.label, run.run_id) for run in self._training_rows),
            preferred=current_id,
        )
        self._refresh_selected_training()

    def _refresh_selected_training(self, *_: Any) -> None:
        selected_id = self.training_run_combo.currentData()
        run = next((item for item in self._training_rows if item.run_id == selected_id), None)
        self.training_distance_plot.clear()
        self.training_lap_plot.clear()
        self.training_table.setRowCount(0)
        if run is None:
            self._set_metric_values(self.training_metrics, {})
            self.training_story_label.setText("No training session selected.")
            self.training_source_label.setText("No training run selected.")
            self.training_contract_label.clear()
            return

        self._set_metric_values(
            self.training_metrics,
            {
                "episodes": f"{len(run.episodes):,}",
                "interactions": f"{run.total_interactions:,}",
                "completed": f"{run.completed_laps:,}",
                "fastest_lap": _seconds(run.fastest_lap_time_seconds),
                "farthest": _metres(run.farthest_distance_m),
            },
        )
        if run.completed_laps:
            self.training_story_label.setText(
                f"This session recorded {run.completed_laps} completed "
                f"{'lap' if run.completed_laps == 1 else 'laps'} across "
                f"{len(run.episodes)} training episodes. Its fastest completed lap "
                f"was {_seconds(run.fastest_lap_time_seconds)}, while shorter episodes "
                "remain visible as part of the learning process."
            )
        else:
            self.training_story_label.setText(
                f"This session did not complete a lap. Its furthest episode reached "
                f"{_metres(run.farthest_distance_m)} across {len(run.episodes)} "
                "recorded training episodes."
            )
        self.training_source_label.setText(f"Source: {run.source_path}")
        self.training_source_label.setToolTip(str(run.source_path))
        self.training_contract_label.setText(
            f"Observation: {run.observation_version}  |  Action: {run.action_version}  |  "
            f"Reward: {run.reward_version}"
        )

        episodes = list(run.episodes)
        x_values = [episode.interactions for episode in episodes]
        distances = [episode.distance_m for episode in episodes]
        running_best: list[float] = []
        best = 0.0
        for distance in distances:
            best = max(best, distance)
            running_best.append(best)
        self.training_distance_plot.plot(
            x_values,
            distances,
            pen=pg.mkPen(self.chart_palette.neutral, width=1),
            symbol="o" if len(episodes) < 80 else None,
            symbolSize=4,
            symbolBrush=self.chart_palette.neutral,
            name="Episode distance",
        )
        self.training_distance_plot.plot(
            x_values,
            running_best,
            pen=pg.mkPen(self.theme_palette.accent, width=2),
            name="Running best",
        )
        completed = [
            episode
            for episode in run.episodes
            if episode.completed and episode.lap_time_seconds is not None
        ]
        if completed:
            self.training_lap_plot.plot(
                [episode.interactions for episode in completed],
                [episode.lap_time_seconds for episode in completed],
                pen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=self.chart_palette.complete,
                symbolPen=self.chart_palette.complete,
            )
        self.training_lap_plot.addItem(
            pg.InfiniteLine(
                pos=PACE_TARGET_SECONDS,
                angle=0,
                pen=pg.mkPen(
                    self.chart_palette.target,
                    width=1,
                    style=Qt.DashLine,
                ),
                label="90 s",
                labelOpts={
                    "color": self.chart_palette.target,
                    "position": 0.92,
                },
            )
        )
        self.training_distance_plot.enableAutoRange()
        self.training_lap_plot.enableAutoRange()
        self._populate_training_table(run)

    def _populate_training_table(self, run: TrainingRun) -> None:
        rows = list(reversed(run.episodes[-150:]))
        self.training_table.setRowCount(len(rows))
        for row, episode in enumerate(rows):
            values = (
                str(episode.episode),
                episode.mode.replace("_", " ").title(),
                f"{episode.interactions:,}",
                f"{episode.distance_m:,.1f} m",
                _seconds(episode.lap_time_seconds),
                episode.termination_reason.replace("_", " ").title(),
            )
            for column, value in enumerate(values):
                self.training_table.setItem(row, column, QTableWidgetItem(value))
        self.training_table.resizeRowsToContents()

    def _refresh_race_results(self, *_: Any) -> None:
        track = str(self.race_track_combo.currentData() or "")
        rows = [summary for summary in self.dataset.race_summaries if summary.track == track]
        total_runs = sum(summary.runs for summary in rows)
        completed_runs = sum(summary.completed_runs for summary in rows)
        best_laps = [
            summary.best_lap_time_seconds
            for summary in rows
            if summary.best_lap_time_seconds is not None
        ]
        self._set_metric_values(
            self.race_metrics,
            {
                "agents": str(len(rows)),
                "runs": f"{total_runs:,}",
                "completed": f"{completed_runs:,}",
                "best_lap": _seconds(min(best_laps) if best_laps else None),
            },
        )
        self._draw_race_bars(rows)
        self._populate_race_table(rows)
        source = rows[0].source_path if rows else "No saved race results found"
        self.race_source_label.setText(f"Source: {source}")
        self.race_source_label.setToolTip(source)

    def _draw_race_bars(self, rows: list[RaceAgentSummary]) -> None:
        self.race_pace_plot.clear()
        self.race_reliability_plot.clear()
        positions = list(range(len(rows)))
        labels = [
            (position, _short_agent_name(row.agent_name))
            for position, row in zip(positions, rows)
        ]
        self.race_pace_plot.getAxis("left").setTicks([labels])
        self.race_reliability_plot.getAxis("left").setTicks([labels])
        pace_positions = [
            position
            for position, row in zip(positions, rows)
            if row.best_lap_time_seconds is not None
        ]
        pace_values = [
            row.best_lap_time_seconds
            for row in rows
            if row.best_lap_time_seconds is not None
        ]
        if pace_values:
            self.race_pace_plot.addItem(
                pg.BarGraphItem(
                    x0=0.0,
                    y=pace_positions,
                    height=0.62,
                    width=pace_values,
                    brush=QColor(self.theme_palette.accent),
                    pen=pg.mkPen(self.theme_palette.accent),
                )
            )
        if rows:
            self.race_reliability_plot.addItem(
                pg.BarGraphItem(
                    x0=0.0,
                    y=positions,
                    height=0.62,
                    width=[row.completion_rate * 100.0 for row in rows],
                    brush=QColor(self.chart_palette.comparison_a),
                    pen=pg.mkPen(self.chart_palette.comparison_a),
                )
            )
        self.race_pace_plot.enableAutoRange()
        self.race_reliability_plot.setXRange(0.0, 105.0, padding=0.02)
        self.race_pace_plot.setYRange(-0.75, max(0.75, len(rows) - 0.25), padding=0.02)
        self.race_reliability_plot.setYRange(
            -0.75,
            max(0.75, len(rows) - 0.25),
            padding=0.02,
        )

    def _populate_race_table(self, rows: list[RaceAgentSummary]) -> None:
        self.race_table.setRowCount(len(rows))
        for row_index, summary in enumerate(rows):
            values = (
                summary.agent_name,
                str(summary.runs),
                str(summary.completed_runs),
                f"{summary.completion_rate * 100:.0f}%",
                _seconds(summary.best_lap_time_seconds),
                _seconds(summary.median_lap_time_seconds),
                f"{summary.median_average_speed_kmh:.1f} km/h"
                if summary.median_average_speed_kmh is not None
                else "--",
            )
            for column, value in enumerate(values):
                self.race_table.setItem(row_index, column, QTableWidgetItem(value))
        self.race_table.resizeRowsToContents()

    def _make_plot(self, x_label: str, y_label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setBackground(self.theme_palette.surface)
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.setMinimumHeight(210)
        plot.setMenuEnabled(False)
        plot.getPlotItem().setClipToView(True)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setTextPen(self.theme_palette.muted)
            axis.setPen(self.theme_palette.border_strong)
        return plot

    def _agent_colour(self, family: str) -> str:
        return {
            "agent6": self.chart_palette.agent6,
            "agent7": self.chart_palette.agent7,
            "agent8": self.chart_palette.agent8,
            "other": self.chart_palette.neutral,
        }.get(family, self.chart_palette.neutral)

    @staticmethod
    def _plot_panel(title_text: str, subtitle_text: str, plot: pg.PlotWidget) -> QWidget:
        panel = QFrame()
        panel.setProperty("surface", True)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 9, 10, 8)
        layout.setSpacing(2)
        title = QLabel(title_text)
        title.setFont(_section_font())
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("mutedText")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(plot, 1)
        return panel

    @staticmethod
    def _field(label_text: str, control: QWidget) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(label_text)
        label.setObjectName("metricLabel")
        layout.addWidget(label)
        layout.addWidget(control)
        return wrapper

    @staticmethod
    def _make_metric_labels(keys: Iterable[str]) -> dict[str, QLabel]:
        labels: dict[str, QLabel] = {}
        for key in keys:
            label = QLabel("--")
            label.setObjectName("metricValue")
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            labels[key] = label
        return labels

    @staticmethod
    def _metric_strip(items: Sequence[tuple[str, QLabel]]) -> QWidget:
        strip = QWidget()
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for title_text, value_label in items:
            card = QFrame()
            card.setProperty("metricCard", True)
            card.setMinimumHeight(66)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 7, 10, 7)
            card_layout.setSpacing(1)
            title = QLabel(title_text)
            title.setObjectName("metricLabel")
            card_layout.addWidget(title)
            card_layout.addWidget(value_label)
            layout.addWidget(card, 1)
        return strip

    @staticmethod
    def _configure_table(table: QTableWidget, headers: Sequence[str]) -> None:
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _prepare_evidence_label(label: QLabel) -> None:
        label.setObjectName("checkpointEvidence")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(False)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        label.setMinimumHeight(28)

    @staticmethod
    def _replace_combo(
        combo: QComboBox,
        items: Sequence[tuple[str, Any]],
        *,
        preferred: Any = None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for label, value in items:
            combo.addItem(label, value)
        if preferred is not None:
            index = combo.findData(preferred)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @staticmethod
    def _set_metric_values(labels: dict[str, QLabel], values: dict[str, str]) -> None:
        for key, label in labels.items():
            label.setText(values.get(key, "--"))


def _seconds(value: float | None) -> str:
    return f"{value:.3f} s" if value is not None else "--"


def _metres(value: float | None) -> str:
    return f"{value:,.0f} m" if value is not None else "--"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _short_agent_name(name: str) -> str:
    replacements = (
        ("Sensor-Only N-Step TD3 Racer", "Agent 8"),
        ("N-Step TD3 Racer", "Agent 7"),
        ("TD3 Scratch Racer", "Agent 6"),
        ("Map-Aware Racing-Line Agent", "Map-aware"),
        ("Rule-Based Anti-Spin Agent", "Rule-based"),
        ("Dyna-Q Finalised Agent", "Dyna-Q final"),
        ("Dyna-Q Learning Agent", "Dyna-Q learning"),
        ("Random Agent", "Random"),
    )
    for full_name, short_name in replacements:
        if full_name.casefold() in name.casefold():
            return short_name
    return name[:18]


def _friendly_agent_name(family: str) -> str:
    return {
        "agent6": "Agent 6 - Scratch DRL Racer",
        "agent7": "Agent 7 - Racing-line DRL Racer",
        "agent8": "Agent 8 - Sensor-only DRL Racer",
    }.get(family, family.replace("_", " ").title())


def _result_status(batch: EvaluationBatch) -> str:
    if batch.completion_rate >= 0.9 and (
        batch.median_lap_time_seconds or math.inf
    ) < PACE_TARGET_SECONDS:
        return "Fast and reliable"
    if batch.completion_rate >= 0.8:
        return "Reliable"
    if batch.completed_laps:
        return "Promising, still variable"
    return "No completed lap yet"


def _section_font() -> QFont:
    font = QFont()
    font.setPointSize(11)
    font.setBold(True)
    return font
