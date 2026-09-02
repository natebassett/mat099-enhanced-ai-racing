from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

try:
    from .learning_visualizer_model import (
        CheckpointStatistics,
        LearningVisualizationProfile,
        build_learning_visualization_profile,
        default_checkpoint_for_agent,
        inspect_td3_checkpoint,
    )
except ImportError:
    from learning_visualizer_model import (
        CheckpointStatistics,
        LearningVisualizationProfile,
        build_learning_visualization_profile,
        default_checkpoint_for_agent,
        inspect_td3_checkpoint,
    )


ACCENT = QColor("#087F73")
POSITIVE = QColor("#18864B")
NEGATIVE = QColor("#C56A12")
VALUE_BLUE = QColor("#2F6F9F")
DANGER = QColor("#B42318")
TEXT = QColor("#182026")
MUTED = QColor("#5D6973")
BORDER = QColor("#DCE2E6")
SURFACE = QColor("#FFFFFF")
SURFACE_ALT = QColor("#F8FAFB")


class LearningVisualizerPanel(QWidget):
    """Read-only educational animation for the selected driving agent."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(690)
        self.profile = build_learning_visualization_profile(None)
        self.step_index = 0
        self.phase = 0.0
        self._speed = 1.0
        self._statistics_cache: dict[Path, CheckpointStatistics] = {}

        self.title_label = QLabel(self.profile.title)
        self.title_label.setObjectName("pageTitle")
        self.status_label = QLabel(self.profile.status)
        self.status_label.setObjectName("learningModeLabel")
        self.status_label.setWordWrap(True)
        self.step_label = QLabel()
        self.step_label.setObjectName("learningStepLabel")
        self.step_label.setWordWrap(True)
        self.explanation_label = QLabel()
        self.explanation_label.setWordWrap(True)
        self.equation_label = QLabel()
        self.equation_label.setObjectName("learningEquation")
        self.equation_label.setWordWrap(True)
        self.checkpoint_label = QLabel()
        self.checkpoint_label.setObjectName("checkpointEvidence")
        self.checkpoint_label.setWordWrap(True)

        self.diagram = LearningDiagramWidget()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(7)

        self.previous_button = QPushButton()
        self.previous_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self.previous_button.setToolTip("Previous learning step")
        self.previous_button.setAccessibleName("Previous learning step")
        self.play_button = QPushButton()
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_button.setToolTip("Play learning animation")
        self.play_button.setAccessibleName("Play learning animation")
        self.next_button = QPushButton()
        self.next_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        )
        self.next_button.setToolTip("Next learning step")
        self.next_button.setAccessibleName("Next learning step")
        self.reset_button = QPushButton()
        self.reset_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.reset_button.setToolTip("Reset learning animation")
        self.reset_button.setAccessibleName("Reset learning animation")
        for button in (
            self.previous_button,
            self.play_button,
            self.next_button,
            self.reset_button,
        ):
            button.setFixedWidth(38)

        self.speed_combo = QComboBox()
        for label, value in (("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)):
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setMaximumWidth(82)
        self.speed_combo.setToolTip("Animation speed")

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._advance_animation)
        self.previous_button.clicked.connect(self.previous_step)
        self.play_button.clicked.connect(self.toggle_playback)
        self.next_button.clicked.connect(self.next_step)
        self.reset_button.clicked.connect(self.reset)
        self.speed_combo.currentIndexChanged.connect(self._update_speed)

        self._build_ui()
        self._show_current_step()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        heading.addWidget(self.title_label, 1)
        heading.addWidget(self.status_label)
        layout.addLayout(heading)
        layout.addWidget(self.checkpoint_label)
        layout.addWidget(self.diagram, 1)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.reset_button)
        controls.addSpacing(8)
        speed_label = QLabel("Speed")
        speed_label.setObjectName("metricLabel")
        controls.addWidget(speed_label)
        controls.addWidget(self.speed_combo)
        controls.addStretch(1)
        controls.addWidget(self.step_label)
        layout.addLayout(controls)
        layout.addWidget(self.progress)

        explanation = QFrame()
        explanation.setProperty("learningExplanation", True)
        explanation_layout = QVBoxLayout(explanation)
        explanation_layout.setContentsMargins(12, 9, 12, 9)
        explanation_layout.addWidget(self.explanation_label)
        explanation_layout.addWidget(self.equation_label)
        layout.addWidget(explanation)

    def set_agent(self, agent_type: str | None, agent_name: str = "") -> None:
        self.pause()
        self.profile = build_learning_visualization_profile(agent_type)
        self.title_label.setText(self.profile.title)
        self.status_label.setText(self.profile.status)
        self.diagram.set_profile(self.profile)
        self.diagram.set_checkpoint_statistics(None)
        self.checkpoint_label.setText(self._checkpoint_message(agent_type, agent_name))
        self.step_index = 0
        self.phase = 0.0
        self._show_current_step()

    def _checkpoint_message(self, agent_type: str | None, agent_name: str) -> str:
        checkpoint = default_checkpoint_for_agent(agent_type)
        if checkpoint is None:
            if self.profile.kind == "dyna_q":
                return "Learning state: Q-table values, not neural parameters"
            return "Learning state: no trainable neural parameters"

        try:
            statistics = self._statistics_cache.get(checkpoint)
            if statistics is None:
                statistics = inspect_td3_checkpoint(checkpoint)
                self._statistics_cache[checkpoint] = statistics
            self.diagram.set_checkpoint_statistics(statistics)
            steps = (
                ""
                if statistics.environment_steps is None
                else f" | {statistics.environment_steps:,} environment steps"
            )
            updates = (
                ""
                if statistics.actor_updates is None
                else f" | {statistics.actor_updates:,} actor updates"
            )
            return (
                f"Actual saved actor: {checkpoint.name} | "
                f"{statistics.parameter_count:,} parameters | "
                f"mean |weight| {statistics.mean_absolute_weight:.4f}"
                f"{steps}{updates}"
            )
        except Exception as exc:
            label = agent_name or "Selected agent"
            return f"{label} checkpoint measurements unavailable: {exc}"

    def toggle_playback(self) -> None:
        if self.timer.isActive():
            self.pause()
            return
        if self.step_index >= len(self.profile.steps) - 1 and self.phase >= 1.0:
            self.reset()
        self.timer.start()
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        )
        self.play_button.setToolTip("Pause learning animation")

    def pause(self) -> None:
        self.timer.stop()
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_button.setToolTip("Play learning animation")

    def reset(self) -> None:
        self.pause()
        self.step_index = 0
        self.phase = 0.0
        self._show_current_step()

    def previous_step(self) -> None:
        self.pause()
        self.step_index = max(0, self.step_index - 1)
        self.phase = 0.0
        self._show_current_step()

    def next_step(self) -> None:
        self.pause()
        self.step_index = min(len(self.profile.steps) - 1, self.step_index + 1)
        self.phase = 0.0
        self._show_current_step()

    def _update_speed(self) -> None:
        self._speed = float(self.speed_combo.currentData() or 1.0)

    def _advance_animation(self) -> None:
        self.phase = min(1.0, self.phase + 0.035 * self._speed)
        self.diagram.set_animation_state(self.step_index, self.phase)
        if self.phase < 1.0:
            return
        if self.step_index >= len(self.profile.steps) - 1:
            self.pause()
            return
        self.step_index += 1
        self.phase = 0.0
        self._show_current_step()

    def _show_current_step(self) -> None:
        step = self.profile.steps[self.step_index]
        total = len(self.profile.steps)
        self.step_label.setText(f"Step {self.step_index + 1} of {total}: {step.title}")
        self.explanation_label.setText(step.explanation)
        self.equation_label.setText(step.equation)
        self.progress.setRange(0, max(1, total - 1))
        self.progress.setValue(self.step_index)
        self.previous_button.setEnabled(self.step_index > 0)
        self.next_button.setEnabled(self.step_index < total - 1)
        self.diagram.set_animation_state(self.step_index, self.phase)


class LearningDiagramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.profile = build_learning_visualization_profile(None)
        self.statistics: CheckpointStatistics | None = None
        self.step_index = 0
        self.phase = 0.0
        self.setMinimumHeight(400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Animated agent learning diagram")

    def set_profile(self, profile: LearningVisualizationProfile) -> None:
        self.profile = profile
        self.step_index = 0
        self.phase = 0.0
        self.update()

    def set_checkpoint_statistics(
        self,
        statistics: CheckpointStatistics | None,
    ) -> None:
        self.statistics = statistics
        self.update()

    def set_animation_state(self, step_index: int, phase: float) -> None:
        self.step_index = max(0, min(step_index, len(self.profile.steps) - 1))
        self.phase = max(0.0, min(float(phase), 1.0))
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), SURFACE_ALT)
        painter.setPen(QPen(BORDER, 1.0))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        if self.profile.kind == "td3":
            self._draw_td3(painter)
        elif self.profile.kind == "dyna_q":
            self._draw_dyna_q(painter)
        else:
            self._draw_static(painter)

    def _draw_td3(self, painter: QPainter) -> None:
        width = float(self.width())
        active = self.profile.steps[self.step_index].active_group
        network_top = 48.0
        network_bottom = min(245.0, float(self.height()) * 0.60)
        layer_x = (116.0, width * 0.38, width * 0.64, width - 128.0)
        labels = ("Sensor groups", "Actor layer 1", "Actor layer 2", "Controls")
        node_counts = (5, 4, 4, 2)
        nodes = [
            _vertical_points(x, network_top + 28.0, network_bottom, count)
            for x, count in zip(layer_x, node_counts)
        ]
        strengths = self._layer_strengths(3)
        actor_active = active in {"actor_forward", "backward", "update"}

        _draw_section_label(painter, 24.0, 16.0, "ACTOR POLICY")
        for layer_index in range(3):
            colour = QColor(ACCENT if actor_active else MUTED)
            alpha = 155 if actor_active else 62
            colour.setAlpha(alpha)
            line_width = 0.7 + strengths[layer_index] * 1.8
            for source_index, start in enumerate(nodes[layer_index]):
                for target_index, end in enumerate(nodes[layer_index + 1]):
                    edge_colour = QColor(colour)
                    if actor_active and (source_index + target_index + layer_index) % 3 == 0:
                        edge_colour = QColor(NEGATIVE)
                        edge_colour.setAlpha(alpha)
                    painter.setPen(QPen(edge_colour, line_width))
                    painter.drawLine(start, end)

        for layer_index, layer_nodes in enumerate(nodes):
            for node_index, point in enumerate(layer_nodes):
                is_active = actor_active and (
                    layer_index <= max(0, min(3, int(self.phase * 4.0)))
                    or active in {"backward", "update"}
                )
                fill = ACCENT if is_active else SURFACE
                outline = ACCENT if is_active else BORDER
                radius = 8.0 if layer_index in {0, 3} else 7.0
                painter.setBrush(fill)
                painter.setPen(QPen(outline, 1.4))
                painter.drawEllipse(point, radius, radius)
                if layer_index == 0:
                    _draw_right_aligned_text(
                        painter,
                        QRectF(8, point.y() - 10, 92, 20),
                        self.profile.input_labels[node_index],
                    )
                elif layer_index == 3:
                    _draw_left_aligned_text(
                        painter,
                        QRectF(point.x() + 13, point.y() - 10, 112, 20),
                        self.profile.output_labels[node_index],
                    )

        for x, label in zip(layer_x, labels):
            _draw_centered_text(painter, QRectF(x - 60, network_top - 19, 120, 16), label, muted=True)

        if active == "actor_forward":
            self._draw_network_pulse(painter, nodes, self.phase, forward=True)
        elif active == "backward":
            self._draw_network_pulse(painter, nodes, self.phase, forward=False)
        elif active == "update":
            self._draw_weight_updates(painter, nodes)

        lower_top = network_bottom + 36.0
        lower_height = max(58.0, float(self.height()) - lower_top - 22.0)
        replay_rect = QRectF(24, lower_top, width * 0.19, lower_height)
        critic_height = (lower_height - 8.0) / 2.0
        q1_rect = QRectF(width * 0.31, lower_top, width * 0.22, critic_height)
        q2_rect = QRectF(
            width * 0.31,
            lower_top + critic_height + 8.0,
            width * 0.22,
            critic_height,
        )
        target_rect = QRectF(width * 0.65, lower_top, width * 0.31 - 24, lower_height)
        _draw_process_box(painter, replay_rect, "Replay", "states + rewards", active == "replay", ACCENT)
        _draw_process_box(painter, q1_rect, "Critic Q1", "value estimate", active in {"critics", "loss", "backward"}, VALUE_BLUE)
        _draw_process_box(painter, q2_rect, "Critic Q2", "independent estimate", active in {"critics", "loss", "backward"}, VALUE_BLUE)
        _draw_process_box(painter, target_rect, "TD target", "reward + smaller Q'", active == "target", POSITIVE)
        _draw_box_arrow(painter, replay_rect, q1_rect, active == "replay")
        _draw_box_arrow(painter, replay_rect, q2_rect, active == "replay")
        _draw_box_arrow(painter, q1_rect, target_rect, active in {"critics", "target", "loss"})
        _draw_box_arrow(painter, q2_rect, target_rect, active in {"critics", "target", "loss"})

    def _draw_network_pulse(
        self,
        painter: QPainter,
        nodes: list[list[QPointF]],
        phase: float,
        *,
        forward: bool,
    ) -> None:
        segment_float = min(2.999, phase * 3.0)
        segment = int(segment_float)
        local_phase = segment_float - segment
        source_layer = segment if forward else 3 - segment
        target_layer = source_layer + 1 if forward else source_layer - 1
        start = nodes[source_layer][0]
        end = nodes[target_layer][0]
        point = QPointF(
            start.x() + (end.x() - start.x()) * local_phase,
            start.y() + (end.y() - start.y()) * local_phase,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(POSITIVE if forward else NEGATIVE)
        painter.drawEllipse(point, 5.0, 5.0)

    def _draw_weight_updates(
        self,
        painter: QPainter,
        nodes: list[list[QPointF]],
    ) -> None:
        pulse = 2.0 + self.phase * 4.0
        for layer in nodes[1:3]:
            for point in layer:
                colour = POSITIVE if int(point.y()) % 2 == 0 else NEGATIVE
                colour = QColor(colour)
                colour.setAlpha(max(35, int(190 * (1.0 - self.phase))))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(colour, 1.6))
                painter.drawEllipse(point, 8.0 + pulse, 8.0 + pulse)

    def _layer_strengths(self, count: int) -> list[float]:
        if self.statistics is None or not self.statistics.layers:
            return [0.45, 0.58, 0.38][:count]
        values = [layer.mean_absolute_weight for layer in self.statistics.layers[:count]]
        values.extend([values[-1]] * (count - len(values)))
        maximum = max(values) or 1.0
        return [max(0.18, value / maximum) for value in values]

    def _draw_dyna_q(self, painter: QPainter) -> None:
        active = self.profile.steps[self.step_index].active_group
        width = float(self.width())
        margin = 24.0
        gap = 12.0
        box_width = (width - margin * 2 - gap * 2) / 3.0
        box_height = 76.0
        boxes = (
            (QRectF(margin, 42, box_width, box_height), "State bins", "compact telemetry", "state"),
            (QRectF(margin + box_width + gap, 42, box_width, box_height), "Action choice", "explore or exploit", "choice"),
            (QRectF(margin + (box_width + gap) * 2, 42, box_width, box_height), "Reward", "progress and safety", "reward"),
            (QRectF(margin + (box_width + gap) * 2, 138, box_width, box_height), "TD error", "new target - old value", "td_error"),
            (QRectF(margin + box_width + gap, 138, box_width, box_height), "Q-table", "one cell changes", "q_update"),
            (QRectF(margin, 138, box_width, box_height), "Model replay", "remembered transition", "planning"),
        )
        _draw_section_label(painter, margin, 16, "VALUE-LEARNING LOOP")
        for rect, title, subtitle, group in boxes:
            _draw_process_box(painter, rect, title, subtitle, active == group, ACCENT)
        for start_index, end_index in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
            _draw_box_arrow(
                painter,
                boxes[start_index][0],
                boxes[end_index][0],
                active in {boxes[start_index][3], boxes[end_index][3]},
            )

        table_top = 244.0
        table_rect = QRectF(margin, table_top, width - margin * 2, max(92.0, self.height() - table_top - 22.0))
        painter.setPen(QPen(BORDER, 1.0))
        painter.setBrush(SURFACE)
        painter.drawRoundedRect(table_rect, 5, 5)
        _draw_text(painter, QRectF(table_rect.left() + 12, table_rect.top() + 8, 180, 20), "Example Q-table")
        rows, columns = 3, 5
        grid = table_rect.adjusted(12, 34, -12, -12)
        cell_width = grid.width() / columns
        cell_height = grid.height() / rows
        for row in range(rows):
            for column in range(columns):
                cell = QRectF(
                    grid.left() + column * cell_width,
                    grid.top() + row * cell_height,
                    cell_width,
                    cell_height,
                )
                selected = row == 1 and column == 2
                fill = QColor("#CDEDE8") if selected and active in {"q_update", "planning"} else SURFACE_ALT
                painter.setBrush(fill)
                painter.setPen(QPen(BORDER, 1.0))
                painter.drawRect(cell)
                value = "+0.14" if selected and active in {"q_update", "planning"} else f"{(row - column) * 0.03:+.2f}"
                _draw_centered_text(painter, cell, value, muted=not selected)

    def _draw_static(self, painter: QPainter) -> None:
        active = self.profile.steps[self.step_index].active_group
        width = float(self.width())
        margin = 34.0
        gap = 24.0
        box_width = (width - margin * 2 - gap * 2) / 3.0
        y = 92.0
        height = 112.0
        middle_title = self.profile.steps[1].equation
        boxes = (
            (QRectF(margin, y, box_width, height), "Observation", "TORCS telemetry", "observe"),
            (QRectF(margin + box_width + gap, y, box_width, height), "Decision", middle_title, "decide"),
            (QRectF(margin + (box_width + gap) * 2, y, box_width, height), "Action", "driving controls", "act"),
        )
        _draw_section_label(painter, margin, 34, "DETERMINISTIC CONTROL PATH")
        for rect, title, subtitle, group in boxes:
            _draw_process_box(painter, rect, title, subtitle, active == group, ACCENT)
        _draw_box_arrow(painter, boxes[0][0], boxes[1][0], active in {"observe", "decide"})
        _draw_box_arrow(painter, boxes[1][0], boxes[2][0], active in {"decide", "act"})

        no_update = QRectF(margin, 250, width - margin * 2, 82)
        _draw_process_box(
            painter,
            no_update,
            "No learning update",
            "The controller's parameters remain unchanged after the action.",
            active == "no_update",
            MUTED,
        )


def _vertical_points(x: float, top: float, bottom: float, count: int) -> list[QPointF]:
    if count <= 1:
        return [QPointF(x, (top + bottom) / 2.0)]
    spacing = (bottom - top) / float(count - 1)
    return [QPointF(x, top + spacing * index) for index in range(count)]


def _draw_process_box(
    painter: QPainter,
    rect: QRectF,
    title: str,
    subtitle: str,
    active: bool,
    accent: QColor,
) -> None:
    fill = QColor(accent)
    fill.setAlpha(28 if active else 0)
    painter.setBrush(fill if active else SURFACE)
    painter.setPen(QPen(accent if active else BORDER, 2.0 if active else 1.0))
    painter.drawRoundedRect(rect, 5, 5)
    _draw_centered_text(
        painter,
        QRectF(rect.left() + 6, rect.top() + 10, rect.width() - 12, 22),
        title,
        bold=True,
    )
    _draw_centered_text(
        painter,
        QRectF(rect.left() + 7, rect.top() + 34, rect.width() - 14, rect.height() - 40),
        subtitle,
        muted=True,
        wrap=True,
    )


def _draw_arrow(
    painter: QPainter,
    start: QPointF,
    end: QPointF,
    active: bool,
) -> None:
    colour = ACCENT if active else BORDER
    painter.setPen(QPen(colour, 2.0 if active else 1.2))
    painter.drawLine(start, end)
    direction = QPointF(end.x() - start.x(), end.y() - start.y())
    length = max(1.0, (direction.x() ** 2 + direction.y() ** 2) ** 0.5)
    unit = QPointF(direction.x() / length, direction.y() / length)
    normal = QPointF(-unit.y(), unit.x())
    tip = end
    base = QPointF(tip.x() - unit.x() * 7.0, tip.y() - unit.y() * 7.0)
    arrow = QPolygonF(
        [
            tip,
            QPointF(base.x() + normal.x() * 3.5, base.y() + normal.y() * 3.5),
            QPointF(base.x() - normal.x() * 3.5, base.y() - normal.y() * 3.5),
        ]
    )
    painter.setBrush(colour)
    painter.setPen(Qt.NoPen)
    painter.drawPolygon(arrow)


def _draw_box_arrow(
    painter: QPainter,
    source: QRectF,
    target: QRectF,
    active: bool,
) -> None:
    if target.left() >= source.right():
        start = QPointF(source.right(), source.center().y())
        end = QPointF(target.left(), target.center().y())
    elif target.top() >= source.bottom():
        start = QPointF(source.center().x(), source.bottom())
        end = QPointF(target.center().x(), target.top())
    elif target.right() <= source.left():
        start = QPointF(source.left(), source.center().y())
        end = QPointF(target.right(), target.center().y())
    else:
        start = QPointF(source.center().x(), source.top())
        end = QPointF(target.center().x(), target.bottom())
    _draw_arrow(painter, start, end, active)


def _draw_section_label(painter: QPainter, x: float, y: float, text: str) -> None:
    painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
    painter.setPen(MUTED)
    painter.drawText(QPointF(x, y + 9), text)


def _draw_text(
    painter: QPainter,
    rect: QRectF,
    text: str,
    *,
    muted: bool = False,
    bold: bool = False,
    alignment: Qt.AlignmentFlag = Qt.AlignLeft | Qt.AlignVCenter,
    wrap: bool = False,
) -> None:
    painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold if bold else QFont.Weight.Normal))
    painter.setPen(MUTED if muted else TEXT)
    flags = alignment
    if wrap:
        flags |= Qt.TextWordWrap
    painter.drawText(rect, flags, text)


def _draw_centered_text(
    painter: QPainter,
    rect: QRectF,
    text: str,
    *,
    muted: bool = False,
    bold: bool = False,
    wrap: bool = False,
) -> None:
    _draw_text(
        painter,
        rect,
        text,
        muted=muted,
        bold=bold,
        alignment=Qt.AlignCenter,
        wrap=wrap,
    )


def _draw_right_aligned_text(painter: QPainter, rect: QRectF, text: str) -> None:
    _draw_text(
        painter,
        rect,
        text,
        muted=True,
        alignment=Qt.AlignRight | Qt.AlignVCenter,
    )


def _draw_left_aligned_text(painter: QPainter, rect: QRectF, text: str) -> None:
    _draw_text(
        painter,
        rect,
        text,
        muted=True,
        alignment=Qt.AlignLeft | Qt.AlignVCenter,
    )
