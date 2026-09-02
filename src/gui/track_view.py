from __future__ import annotations

from statistics import mean

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

try:
    from .telemetry_model import TelemetrySnapshot
except ImportError:
    from telemetry_model import TelemetrySnapshot


class TrackPositionWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._track_position: float | None = None
        self.setMinimumHeight(82)

    def set_snapshot(self, snapshot: TelemetrySnapshot | None) -> None:
        self._track_position = None if snapshot is None else snapshot.track_pos
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(14, 12, -14, -12)
        road = QRectF(rect.left(), rect.center().y() - 12, rect.width(), 24)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#353535"))
        painter.drawRoundedRect(road, 4, 4)

        painter.setPen(QPen(QColor("#f2f2f2"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            QPointF(road.center().x(), road.top() - 7),
            QPointF(road.center().x(), road.bottom() + 7),
        )

        painter.setPen(QPen(QColor("#b14b4b"), 2))
        painter.drawLine(QPointF(road.left(), road.top()), QPointF(road.left(), road.bottom()))
        painter.drawLine(
            QPointF(road.right(), road.top()),
            QPointF(road.right(), road.bottom()),
        )

        track_position = _clamp(self._track_position or 0.0, -1.0, 1.0)
        marker_x = road.left() + ((track_position + 1.0) / 2.0) * road.width()
        marker_color = _position_color(track_position)
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setBrush(marker_color)
        painter.drawEllipse(QPointF(marker_x, road.center().y()), 8, 8)

        painter.setFont(_small_font())
        painter.setPen(QColor("#5D6973"))
        painter.drawText(rect.left(), rect.top(), "Left edge")
        painter.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), 16),
            Qt.AlignmentFlag.AlignHCenter,
            "Centre",
        )
        painter.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), 16),
            Qt.AlignmentFlag.AlignRight,
            "Right edge",
        )
        painter.drawText(
            QRectF(rect.left(), road.bottom() + 8, rect.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            _position_label(self._track_position),
        )


class RoadSensorWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._track_sensors: tuple[float, ...] = ()
        self.setMinimumHeight(106)

    def set_snapshot(self, snapshot: TelemetrySnapshot | None) -> None:
        self._track_sensors = () if snapshot is None else snapshot.track_sensors
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(14, 10, -14, -12)
        zones = sensor_zones(self._track_sensors)
        if not zones:
            painter.setPen(QColor("#5D6973"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Road ahead unavailable")
            return

        gap = 8
        bar_width = (rect.width() - gap * (len(zones) - 1)) / len(zones)
        max_height = max(22.0, rect.height() - 34)
        baseline = rect.bottom() - 18

        painter.setFont(_small_font())
        for index, (label, distance) in enumerate(zones):
            left = rect.left() + index * (bar_width + gap)
            height = max(4.0, min(1.0, distance / 100.0) * max_height)
            bar = QRectF(left, baseline - height, bar_width, height)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_sensor_color(distance))
            painter.drawRoundedRect(bar, 3, 3)

            painter.setPen(QColor("#5D6973"))
            painter.drawText(
                QRectF(left, baseline + 2, bar_width, 16),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )
            painter.drawText(
                QRectF(left, rect.top(), bar_width, 16),
                Qt.AlignmentFlag.AlignCenter,
                f"{distance:.0f}m",
            )


def sensor_zones(track_sensors: tuple[float, ...]) -> list[tuple[str, float]]:
    if len(track_sensors) < 19:
        return []
    return [
        ("Left", _zone_distance(track_sensors[0:4])),
        ("L-Front", _zone_distance(track_sensors[4:8])),
        ("Front", _zone_distance(track_sensors[8:11])),
        ("R-Front", _zone_distance(track_sensors[11:15])),
        ("Right", _zone_distance(track_sensors[15:19])),
    ]


def _zone_distance(values: tuple[float, ...]) -> float:
    return min(mean(values), 200.0)


def _position_label(track_position: float | None) -> str:
    if track_position is None:
        return "Waiting for position"
    side = "right" if track_position > 0 else "left"
    edge = abs(track_position)
    if edge > 0.90:
        return f"Near {side} edge"
    if edge > 0.62:
        return f"Running wide {side}"
    if edge < 0.18:
        return "Centred"
    return f"Slightly {side}"


def _position_color(track_position: float) -> QColor:
    edge = abs(track_position)
    if edge > 0.90:
        return QColor("#d6453d")
    if edge > 0.62:
        return QColor("#d99b24")
    return QColor("#2ba66a")


def _sensor_color(distance: float) -> QColor:
    if distance < 8.0:
        return QColor("#d6453d")
    if distance < 24.0:
        return QColor("#d99b24")
    return QColor("#2ba66a")


def _small_font() -> QFont:
    font = QFont()
    font.setPointSize(8)
    return font


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
