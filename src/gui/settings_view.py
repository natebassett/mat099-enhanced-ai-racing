from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from .i18n import tr, translate_widget_tree
    from .settings_model import GuiSettings, START_PAGES, SettingsStore
except ImportError:
    from i18n import tr, translate_widget_tree
    from settings_model import GuiSettings, START_PAGES, SettingsStore


SOURCE_GROUPS = (
    (
        "Reinforcement Learning",
        (
            (
                "TD3 paper",
                "https://arxiv.org/abs/1802.09477",
                "Introduces twin critics, delayed actor updates, and target "
                "policy smoothing.",
            ),
            (
                "OpenAI Spinning Up: TD3",
                "https://spinningup.openai.com/en/latest/algorithms/td3.html",
                "A practical explanation of the TD3 learning loop and design choices.",
            ),
            (
                "torcsRL",
                "https://github.com/raphaelsenn/torcsRL",
                "Reference TORCS reinforcement-learning project reviewed for Agent 7.",
            ),
        ),
    ),
    (
        "Evaluation Practice",
        (
            (
                "Deep RL at the Edge of the Statistical Precipice",
                "https://arxiv.org/abs/2108.13264",
                "Motivates repeated trials and careful reporting of variable results.",
            ),
            (
                "Deep Reinforcement Learning That Matters",
                "https://arxiv.org/abs/1709.06560",
                "Examines reproducibility and sensitivity in deep RL experiments.",
            ),
        ),
    ),
    (
        "Learning Visualisation",
        (
            (
                "The Building Blocks of Interpretability",
                "https://distill.pub/2018/building-blocks/",
                "Shows how complex neural computation can be explained in "
                "visual stages.",
            ),
            (
                "CNN Explainer",
                "https://arxiv.org/abs/2004.15004",
                "Demonstrates progressive disclosure for novice-facing "
                "neural-network education.",
            ),
        ),
    ),
    (
        "Accessibility and Language",
        (
            (
                "WCAG: Use of Colour",
                "https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html",
                "Requires visible labels, shapes, or patterns when colour carries meaning.",
            ),
            (
                "WCAG: Contrast Minimum",
                "https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html",
                "Provides the contrast basis for standard and high-contrast themes.",
            ),
            (
                "Qt Translation Framework",
                "https://doc.qt.io/qtforpython-6/PySide6/QtCore/QTranslator.html",
                "Documents Qt's runtime support for translated application text.",
            ),
        ),
    ),
)


class SettingsView(QWidget):
    settings_saved = Signal(object)
    clear_cache_requested = Signal()
    reset_run_history_requested = Signal()

    def __init__(
        self,
        *,
        agents: Iterable[Any],
        tracks: Iterable[Any],
        settings: GuiSettings,
        store: SettingsStore,
        project_root: Path,
        research_evidence_page: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.project_root = project_root
        self.current_settings = settings
        self.current_language = settings.language
        self._storage_summary: tuple[int, int, int] | None = None
        self.research_evidence_page = research_evidence_page

        self.agent_combo = QComboBox()
        self.track_combo = QComboBox()
        self.appearance_combo = QComboBox()
        self.colour_mode_combo = QComboBox()
        self.language_combo = QComboBox()
        self.chart_window_slider = QSlider(Qt.Horizontal)
        self.chart_window_value_label = QLabel()
        self.reduce_motion_checkbox = QCheckBox("Reduce animated movement")
        self.td3_advisory_checkbox = QCheckBox(
            "Show the reliability note before starting a TD3 race"
        )
        self.start_page_group = QButtonGroup(self)
        self.start_page_buttons: dict[str, QPushButton] = {}
        self.save_button = QPushButton("Save settings")
        self.reset_button = QPushButton("Restore defaults")
        self.save_status_label = QLabel()
        self.cache_summary_label = QLabel("Checking temporary cache...")
        self.run_history_summary_label = QLabel("Checking saved runs...")
        self.clear_cache_button = QPushButton("Clear temporary cache")
        self.reset_history_button = QPushButton("Reset run history")
        self.tabs = QTabWidget()

        self._populate_options(agents, tracks)
        self._configure_controls()
        self._build_ui()
        self._connect_signals()
        self.set_settings(settings)

    def set_settings(self, settings: GuiSettings) -> None:
        self.current_settings = settings
        self._select_combo_data(self.agent_combo, settings.default_agent_type)
        self._select_combo_data(self.track_combo, settings.default_track_id)
        self.chart_window_slider.setValue(settings.chart_window_seconds)
        self.reduce_motion_checkbox.setChecked(settings.reduce_motion)
        self.td3_advisory_checkbox.setChecked(settings.show_td3_advisory)
        self._select_combo_data(self.appearance_combo, settings.appearance_mode)
        self._select_combo_data(self.colour_mode_combo, settings.colour_mode)
        self._select_combo_data(self.language_combo, settings.language)
        button = self.start_page_buttons.get(settings.start_page)
        if button is not None:
            button.setChecked(True)
        self.set_language(settings.language)
        self._update_chart_window_label(settings.chart_window_seconds)

    def set_language(self, language: str) -> None:
        self.current_language = "cy" if language == "cy" else "en"
        translate_widget_tree(self, self.current_language)
        self._translate_combo_items(
            self.appearance_combo,
            {"system": "Follow Windows", "light": "Light", "dark": "Dark"},
        )
        self._translate_combo_items(
            self.colour_mode_combo,
            {
                "standard": "Standard",
                "accessible": "Colour-accessible",
                "high_contrast": "High contrast",
            },
        )
        self._translate_combo_items(
            self.language_combo,
            {"en": "English", "cy": "Cymraeg"},
        )
        for page_name, button in self.start_page_buttons.items():
            button.setText(tr(page_name, self.current_language))
        self._update_chart_window_label(self.chart_window_slider.value())
        if self._storage_summary is not None:
            run_count, cache_bytes, cache_directories = self._storage_summary
            self.refresh_storage_summary(
                run_count=run_count,
                cache_bytes=cache_bytes,
                cache_directories=cache_directories,
            )

    def selected_settings(self) -> GuiSettings:
        button = self.start_page_group.checkedButton()
        start_page = (
            str(button.property("pageName"))
            if button is not None
            else "Live Telemetry"
        )
        return GuiSettings(
            default_agent_type=str(self.agent_combo.currentData() or ""),
            default_track_id=str(self.track_combo.currentData() or "g-track-3"),
            start_page=start_page,
            chart_window_seconds=self.chart_window_slider.value(),
            reduce_motion=self.reduce_motion_checkbox.isChecked(),
            appearance_mode=str(self.appearance_combo.currentData() or "light"),
            colour_mode=str(self.colour_mode_combo.currentData() or "standard"),
            language=str(self.language_combo.currentData() or "en"),
            show_td3_advisory=self.td3_advisory_checkbox.isChecked(),
        )

    def refresh_storage_summary(
        self,
        *,
        run_count: int,
        cache_bytes: int,
        cache_directories: int,
    ) -> None:
        self._storage_summary = (run_count, cache_bytes, cache_directories)
        if self.current_language == "cy":
            self.run_history_summary_label.setText(
                f"{run_count:,} {'ras wedi\'i chadw' if run_count == 1 else 'ras wedi\'u cadw'}"
            )
            self.cache_summary_label.setText(
                f"{_format_bytes(cache_bytes)} mewn {cache_directories:,} "
                f"{'ffolder dros dro' if cache_directories == 1 else 'ffolder dros dro'}"
            )
        else:
            self.run_history_summary_label.setText(
                f"{run_count:,} saved {'run' if run_count == 1 else 'runs'}"
            )
            self.cache_summary_label.setText(
                f"{_format_bytes(cache_bytes)} across "
                f"{cache_directories:,} temporary "
                f"{'folder' if cache_directories == 1 else 'folders'}"
            )

    def set_maintenance_enabled(self, enabled: bool) -> None:
        self.clear_cache_button.setEnabled(enabled)
        self.reset_history_button.setEnabled(enabled)

    def save(self) -> None:
        settings = self.selected_settings()
        try:
            self.store.save(settings)
        except OSError as error:
            prefix = (
                "Ni ellid cadw'r gosodiadau"
                if self.current_language == "cy"
                else "Settings could not be saved"
            )
            self.save_status_label.setText(f"{prefix}: {error}")
            self.save_status_label.setProperty("saveState", "error")
            self.save_status_label.style().unpolish(self.save_status_label)
            self.save_status_label.style().polish(self.save_status_label)
            return
        self.current_settings = settings
        self.set_language(settings.language)
        self.save_status_label.setText(tr("Settings saved", self.current_language))
        self.save_status_label.setProperty("saveState", "saved")
        self.save_status_label.style().unpolish(self.save_status_label)
        self.save_status_label.style().polish(self.save_status_label)
        self.settings_saved.emit(settings)

    def restore_defaults(self) -> None:
        self.set_settings(GuiSettings())
        self.save()

    def _populate_options(
        self,
        agents: Iterable[Any],
        tracks: Iterable[Any],
    ) -> None:
        self.agent_combo.addItem("Keep the current driver", "")
        for agent in agents:
            self.agent_combo.addItem(str(agent.name), str(agent.agent_type))
        for track in tracks:
            self.track_combo.addItem(str(track.label), str(track.track_id))

        self.appearance_combo.addItem("Follow Windows", "system")
        self.appearance_combo.addItem("Light", "light")
        self.appearance_combo.addItem("Dark", "dark")
        self.colour_mode_combo.addItem("Standard", "standard")
        self.colour_mode_combo.addItem("Colour-accessible", "accessible")
        self.colour_mode_combo.addItem("High contrast", "high_contrast")
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Cymraeg", "cy")

    def _configure_controls(self) -> None:
        for combo in (
            self.agent_combo,
            self.track_combo,
            self.appearance_combo,
            self.colour_mode_combo,
            self.language_combo,
        ):
            combo.setMinimumContentsLength(24)
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )

        self.chart_window_slider.setRange(30, 180)
        self.chart_window_slider.setSingleStep(15)
        self.chart_window_slider.setPageStep(30)
        self.chart_window_slider.setTickInterval(30)
        self.chart_window_slider.setTickPosition(QSlider.TicksBelow)
        self.chart_window_value_label.setObjectName("factValue")
        self.chart_window_value_label.setMinimumWidth(92)
        self.chart_window_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.start_page_group.setExclusive(True)
        for index, page_name in enumerate(START_PAGES):
            button = QPushButton(page_name)
            button.setCheckable(True)
            button.setProperty("segmented", True)
            button.setProperty(
                "segmentPosition",
                "first"
                if index == 0
                else "last"
                if index == len(START_PAGES) - 1
                else "middle",
            )
            button.setProperty("pageName", page_name)
            button.setToolTip(f"Open {page_name} when the application starts")
            self.start_page_group.addButton(button)
            self.start_page_buttons[page_name] = button

        self.save_button.setProperty("primary", True)
        self.save_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.reset_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton)
        )
        self.clear_cache_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )
        self.reset_history_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )
        self.reset_history_button.setProperty("danger", True)
        self.cache_summary_label.setObjectName("factValue")
        self.run_history_summary_label.setObjectName("factValue")
        self.save_status_label.setObjectName("settingsSaveStatus")

    def _build_ui(self) -> None:
        self.setObjectName("pageCanvas")
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Choose how the application opens and how much live information it shows."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs.setObjectName("contentTabs")
        self.tabs.addTab(self._build_preferences_tab(), "Preferences")
        self.tabs.addTab(self._build_research_tab(), "Research Evidence")
        self.tabs.addTab(self._build_sources_tab(), "Sources & Methods")
        self.tabs.addTab(self._build_storage_tab(), "Data & Storage")
        root.addWidget(self.tabs, 1)

    def _build_preferences_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()
        page.setObjectName("scrollCanvas")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        defaults = QGroupBox("Race Defaults")
        defaults_layout = QGridLayout(defaults)
        defaults_layout.setHorizontalSpacing(12)
        defaults_layout.setVerticalSpacing(7)
        defaults_layout.addWidget(self._field_label("Driver"), 0, 0)
        defaults_layout.addWidget(self.agent_combo, 1, 0)
        defaults_layout.addWidget(self._field_label("Track"), 0, 1)
        defaults_layout.addWidget(self.track_combo, 1, 1)
        defaults_layout.setColumnStretch(0, 1)
        defaults_layout.setColumnStretch(1, 1)
        layout.addWidget(defaults)

        opening = QGroupBox("Opening Screen")
        opening_layout = QHBoxLayout(opening)
        opening_layout.setSpacing(0)
        for page_name in START_PAGES:
            opening_layout.addWidget(self.start_page_buttons[page_name], 1)
        layout.addWidget(opening)

        appearance = QGroupBox("Appearance & Language")
        appearance_layout = QGridLayout(appearance)
        appearance_layout.setHorizontalSpacing(12)
        appearance_layout.setVerticalSpacing(7)
        appearance_layout.addWidget(self._field_label("Theme"), 0, 0)
        appearance_layout.addWidget(self._field_label("Colour presentation"), 0, 1)
        appearance_layout.addWidget(self._field_label("Language"), 0, 2)
        appearance_layout.addWidget(self.appearance_combo, 1, 0)
        appearance_layout.addWidget(self.colour_mode_combo, 1, 1)
        appearance_layout.addWidget(self.language_combo, 1, 2)
        for column in range(3):
            appearance_layout.setColumnStretch(column, 1)
        layout.addWidget(appearance)

        display = QGroupBox("Display")
        display_layout = QGridLayout(display)
        display_layout.setHorizontalSpacing(12)
        display_layout.setVerticalSpacing(8)
        display_layout.addWidget(self._field_label("Live chart history"), 0, 0)
        display_layout.addWidget(self.chart_window_value_label, 0, 1)
        display_layout.addWidget(self.chart_window_slider, 1, 0, 1, 2)
        display_layout.addWidget(self.reduce_motion_checkbox, 2, 0, 1, 2)
        layout.addWidget(display)

        advisories = QGroupBox("Helpful Notices")
        advisories_layout = QVBoxLayout(advisories)
        advisories_layout.addWidget(self.td3_advisory_checkbox)
        advisory_detail = QLabel(
            "The note explains the measured completion rate before a learned "
            "driver starts. It does not change how the agent drives."
        )
        advisory_detail.setObjectName("mutedText")
        advisory_detail.setWordWrap(True)
        advisories_layout.addWidget(advisory_detail)
        layout.addWidget(advisories)

        actions = QHBoxLayout()
        actions.addWidget(self.save_status_label)
        actions.addStretch(1)
        actions.addWidget(self.reset_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        layout.addStretch(1)
        scroll.setWidget(page)
        return scroll

    def _build_storage_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()
        page.setObjectName("scrollCanvas")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        cache = QGroupBox("Temporary Cache")
        cache_layout = QGridLayout(cache)
        cache_detail = QLabel(
            "Removes generated Python bytecode and refreshes cached checkpoint "
            "summaries. Models, race runs, and research evidence stay safe."
        )
        cache_detail.setObjectName("mutedText")
        cache_detail.setWordWrap(True)
        cache_layout.addWidget(self.cache_summary_label, 0, 0)
        cache_layout.addWidget(cache_detail, 1, 0)
        cache_layout.addWidget(self.clear_cache_button, 0, 1, 2, 1)
        cache_layout.setColumnStretch(0, 1)
        layout.addWidget(cache)

        history = QGroupBox("Run History")
        history_layout = QGridLayout(history)
        history_detail = QLabel(
            "Deletes older GUI race runs while preserving one useful replay "
            "for each agent. Completed laps are preferred so Review and Compare "
            "remain usable."
        )
        history_detail.setObjectName("mutedText")
        history_detail.setWordWrap(True)
        history_layout.addWidget(self.run_history_summary_label, 0, 0)
        history_layout.addWidget(history_detail, 1, 0)
        history_layout.addWidget(self.reset_history_button, 0, 1, 2, 1)
        history_layout.setColumnStretch(0, 1)
        layout.addWidget(history)

        protected = QGroupBox("Always Protected")
        protected_layout = QVBoxLayout(protected)
        protected_text = QLabel(
            "Training checkpoints, replay buffers, evaluation JSON/CSV files, "
            "training logs, recorded laps, racing lines, and project source code "
            "are never removed by these controls."
        )
        protected_text.setObjectName("resultsInsight")
        protected_text.setWordWrap(True)
        protected_layout.addWidget(protected_text)
        layout.addWidget(protected)
        layout.addStretch(1)
        scroll.setWidget(page)
        return scroll

    def _build_research_tab(self) -> QWidget:
        if self.research_evidence_page is None:
            placeholder = QLabel("No saved evaluation evidence was found.")
            placeholder.setAlignment(Qt.AlignCenter)
            return placeholder
        self.research_evidence_page.setParent(None)
        self.research_evidence_page.show()
        return self.research_evidence_page

    def _build_sources_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        page = QWidget()
        page.setObjectName("scrollCanvas")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        context = QLabel(
            "These sources informed the learning algorithms, evaluation method, "
            "and educational visualisations used in this project."
        )
        context.setObjectName("resultsInsight")
        context.setWordWrap(True)
        layout.addWidget(context)

        for group_title, sources in SOURCE_GROUPS:
            group = QGroupBox(group_title)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(0)
            for index, source in enumerate(sources):
                group_layout.addWidget(self._source_row(*source))
                if index < len(sources) - 1:
                    divider = QFrame()
                    divider.setFrameShape(QFrame.HLine)
                    divider.setObjectName("settingsDivider")
                    group_layout.addWidget(divider)
            layout.addWidget(group)

        local = QGroupBox("Project Documentation")
        local_layout = QVBoxLayout(local)
        local_layout.addWidget(
            self._local_document_row(
                "Results workspace methodology",
                self.project_root / "docs" / "gui_results_workspace.md",
                "How completion, lap pace, trial counts, and evidence "
                "provenance are reported.",
            )
        )
        local_layout.addWidget(
            self._local_document_row(
                "Learning visualiser methodology",
                self.project_root / "docs" / "gui_learning_visualizer.md",
                "What the educational neural-network animation represents "
                "and where its evidence ends.",
            )
        )
        local_layout.addWidget(
            self._local_document_row(
                "Settings and sources guide",
                self.project_root / "docs" / "gui_settings_and_sources.md",
                "Which preferences apply immediately and where detailed "
                "evidence is kept.",
            )
        )
        layout.addWidget(local)
        layout.addStretch(1)
        scroll.setWidget(page)
        return scroll

    def _connect_signals(self) -> None:
        self.save_button.clicked.connect(self.save)
        self.reset_button.clicked.connect(self.restore_defaults)
        self.chart_window_slider.valueChanged.connect(
            self._update_chart_window_label
        )
        self.clear_cache_button.clicked.connect(self.clear_cache_requested)
        self.reset_history_button.clicked.connect(
            self.reset_run_history_requested
        )

    def _update_chart_window_label(self, seconds: int) -> None:
        unit = "eiliad" if self.current_language == "cy" else "seconds"
        self.chart_window_value_label.setText(f"{seconds} {unit}")

    def _translate_combo_items(
        self,
        combo: QComboBox,
        source_labels: dict[str, str],
    ) -> None:
        for index in range(combo.count()):
            key = str(combo.itemData(index) or "")
            source = source_labels.get(key)
            if source is not None:
                combo.setItemText(index, tr(source, self.current_language))

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("metricLabel")
        return label

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _source_row(title: str, url: str, description: str) -> QWidget:
        row = QWidget()
        layout = QGridLayout(row)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setHorizontalSpacing(14)
        link = QLabel(
            f'<a href="{escape(url, quote=True)}">{escape(title)}</a>'
        )
        link.setObjectName("sourceLink")
        link.setOpenExternalLinks(True)
        link.setToolTip(url)
        detail = QLabel(description)
        detail.setObjectName("mutedText")
        detail.setWordWrap(True)
        layout.addWidget(link, 0, 0)
        layout.addWidget(detail, 0, 1)
        layout.setColumnStretch(1, 1)
        return row

    @staticmethod
    def _local_document_row(title: str, path: Path, description: str) -> QWidget:
        resolved_path = path.resolve()
        row = SettingsView._source_row(
            title,
            resolved_path.as_uri(),
            description,
        )
        row.setToolTip(str(resolved_path))
        return row


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return "0 B"
