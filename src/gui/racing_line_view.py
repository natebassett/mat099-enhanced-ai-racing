from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

try:
    from .racing_line_model import (
        Coordinate,
        RacingLinePoint,
        RacingLineProfile,
        RacingLineVisualPoint,
        RacingLineVisualProfile,
        phase_family,
        point_at_distance,
        visual_point_at_distance,
    )
except ImportError:
    from racing_line_model import (
        Coordinate,
        RacingLinePoint,
        RacingLineProfile,
        RacingLineVisualPoint,
        RacingLineVisualProfile,
        phase_family,
        point_at_distance,
        visual_point_at_distance,
    )


class RacingLineGraphWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._profile: RacingLineProfile | None = None
        self._distance = 0.0
        self.setMinimumHeight(230)

    def set_profile(self, profile: RacingLineProfile | None) -> None:
        self._profile = profile
        self._distance = 0.0
        self.update()

    def set_distance(self, distance: float) -> None:
        if self._profile is None:
            self._distance = 0.0
        else:
            track_length = max(self._profile.track_length_m, 1.0)
            self._distance = max(0.0, min(distance, track_length))
        self.update()

    def current_point(self) -> RacingLinePoint | None:
        return point_at_distance(self._profile, self._distance)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(14, 12, -14, -12)
        painter.fillRect(rect, QColor("#f7f7f7"))

        if self._profile is None or not self._profile.points:
            painter.setPen(QColor("#303030"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Racing line unavailable")
            return

        graph = QRectF(rect.left() + 58, rect.top() + 18, rect.width() - 70, rect.height() - 48)
        self._draw_axes(painter, rect, graph)
        self._draw_full_line(painter, graph)
        self._draw_traced_line(painter, graph)
        self._draw_marker(painter, graph)

    def _draw_axes(self, painter: QPainter, rect: QRectF, graph: QRectF) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawRect(graph)

        painter.setPen(QPen(QColor("#b14b4b"), 2))
        painter.drawLine(QPointF(graph.left(), graph.top()), QPointF(graph.right(), graph.top()))
        painter.drawLine(
            QPointF(graph.left(), graph.bottom()),
            QPointF(graph.right(), graph.bottom()),
        )

        painter.setPen(QPen(QColor("#8c8c8c"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            QPointF(graph.left(), graph.center().y()),
            QPointF(graph.right(), graph.center().y()),
        )

        painter.setFont(_small_font())
        painter.setPen(QColor("#505050"))
        painter.drawText(
            QRectF(rect.left(), graph.top() - 8, 54, 18),
            Qt.AlignmentFlag.AlignRight,
            "Left edge",
        )
        painter.drawText(
            QRectF(rect.left(), graph.center().y() - 9, 54, 18),
            Qt.AlignmentFlag.AlignRight,
            "Centre",
        )
        painter.drawText(
            QRectF(rect.left(), graph.bottom() - 10, 54, 18),
            Qt.AlignmentFlag.AlignRight,
            "Right edge",
        )
        painter.drawText(
            QRectF(graph.left(), graph.bottom() + 10, graph.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            "target road position across the lap",
        )

    def _draw_full_line(self, painter: QPainter, graph: QRectF) -> None:
        points = self._visible_points()
        if len(points) < 2:
            return

        painter.setPen(QPen(QColor("#9a9a9a"), 1))
        previous = self._point_to_screen(points[0], graph)
        for point in points[1:]:
            current = self._point_to_screen(point, graph)
            painter.drawLine(previous, current)
            previous = current

    def _draw_traced_line(self, painter: QPainter, graph: QRectF) -> None:
        points = self._visible_points()
        if len(points) < 2:
            return

        for left, right in zip(points, points[1:]):
            if left.distance > self._distance:
                break
            end = right
            if right.distance > self._distance:
                end = _interpolate_graph_point(left, right, self._distance)

            painter.setPen(QPen(_phase_color(left.phase), 3))
            painter.drawLine(
                self._point_to_screen(left, graph),
                self._point_to_screen(end, graph),
            )
            if right.distance > self._distance:
                break

    def _draw_marker(self, painter: QPainter, graph: QRectF) -> None:
        point = self.current_point()
        if point is None:
            return

        marker = self._point_to_screen(point, graph)
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(marker, 7, 7)
        painter.setBrush(_phase_color(point.phase))
        painter.drawEllipse(marker, 4, 4)

    def _point_to_screen(self, point: RacingLinePoint, graph: QRectF) -> QPointF:
        assert self._profile is not None
        x_fraction = point.distance / max(self._profile.track_length_m, 1.0)
        y_fraction = (point.target_track_pos + 1.0) / 2.0
        return QPointF(
            graph.left() + x_fraction * graph.width(),
            graph.top() + y_fraction * graph.height(),
        )

    def _visible_points(self) -> list[RacingLinePoint]:
        assert self._profile is not None
        points = list(self._profile.points)
        step = max(1, len(points) // 900)
        visible = points[::step]
        if visible[-1] != points[-1]:
            visible.append(points[-1])
        return visible


class RacingLineWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._profile: RacingLineVisualProfile | None = None
        self._distance = 0.0
        self._follow_enabled = True
        self._follow_window_m = 170.0
        self.setMinimumHeight(320)

    def set_profile(self, profile: RacingLineVisualProfile | None) -> None:
        self._profile = profile
        self._distance = 0.0
        self.update()

    def set_distance(self, distance: float) -> None:
        if self._profile is None:
            self._distance = 0.0
        else:
            track_length = max(self._profile.racing_line.track_length_m, 1.0)
            self._distance = max(0.0, min(distance, track_length))
        self.update()

    def current_distance(self) -> float:
        return self._distance

    def current_point(self) -> RacingLinePoint | None:
        point = visual_point_at_distance(self._profile, self._distance)
        return None if point is None else point.source

    def set_follow_enabled(self, enabled: bool) -> None:
        self._follow_enabled = enabled
        self.update()

    def follow_enabled(self) -> bool:
        return self._follow_enabled

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(14, 12, -14, -12)
        painter.fillRect(rect, QColor("#f7f7f7"))

        if self._profile is None or not self._profile.line_points:
            painter.setPen(QColor("#303030"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Racing line unavailable")
            return

        mapper = _screen_mapper(self._view_bounds(), rect.adjusted(8, 8, -8, -24))
        self._draw_road(painter, mapper)
        self._draw_full_line(painter, mapper)
        self._draw_traced_line(painter, mapper)
        self._draw_marker(painter, mapper)
        self._draw_footer(painter, rect)

    def _draw_road(self, painter: QPainter, mapper) -> None:
        assert self._profile is not None
        road_polygon = QPolygonF(
            [
                *[mapper(point) for point in self._profile.left_edge],
                *[mapper(point) for point in reversed(self._profile.right_edge)],
            ]
        )
        painter.setPen(QPen(QColor("#202020"), 1))
        painter.setBrush(QColor("#404040"))
        painter.drawPolygon(road_polygon)

        self._draw_polyline(
            painter,
            self._profile.left_edge,
            mapper,
            QPen(QColor("#b14b4b"), 2),
            close=True,
        )
        self._draw_polyline(
            painter,
            self._profile.right_edge,
            mapper,
            QPen(QColor("#b14b4b"), 2),
            close=True,
        )
        self._draw_polyline(
            painter,
            self._profile.centerline,
            mapper,
            QPen(QColor("#f0f0f0"), 2, Qt.PenStyle.DashLine),
            close=True,
        )
        if self._profile.left_edge and self._profile.right_edge:
            painter.setPen(QPen(QColor("#ffffff"), 3))
            painter.drawLine(
                mapper(self._profile.left_edge[0]),
                mapper(self._profile.right_edge[0]),
            )
            painter.setPen(QPen(QColor("#111111"), 1))
            painter.drawLine(
                mapper(self._profile.left_edge[0]),
                mapper(self._profile.right_edge[0]),
            )

    def _draw_full_line(self, painter: QPainter, mapper) -> None:
        assert self._profile is not None
        coordinates = [(point.x, point.y) for point in self._profile.line_points]
        self._draw_polyline(
            painter,
            coordinates,
            mapper,
            QPen(QColor("#a8a8a8"), 2),
            close=True,
        )

    def _draw_traced_line(self, painter: QPainter, mapper) -> None:
        assert self._profile is not None
        points = self._profile.line_points
        if len(points) < 2:
            return

        for left, right in zip(points, points[1:]):
            if left.source.distance > self._distance:
                break
            end = right
            if right.source.distance > self._distance:
                end = _interpolate_visual_point(left, right, self._distance)

            painter.setPen(QPen(_phase_color(left.source.phase), 5))
            painter.drawLine(
                mapper((left.x, left.y)),
                mapper((end.x, end.y)),
            )
            if right.source.distance > self._distance:
                break

    def _draw_marker(self, painter: QPainter, mapper) -> None:
        point = visual_point_at_distance(self._profile, self._distance)
        if point is None:
            return

        marker = mapper((point.x, point.y))
        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(marker, 10, 10)
        painter.setBrush(_phase_color(point.source.phase))
        painter.drawEllipse(marker, 6, 6)

    def _draw_footer(self, painter: QPainter, rect: QRectF) -> None:
        assert self._profile is not None
        painter.setFont(_small_font())
        painter.setPen(QColor("#303030"))
        painter.drawText(
            QRectF(rect.left(), rect.bottom() - 18, rect.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            self._footer_text(),
        )

    def _draw_polyline(
        self,
        painter: QPainter,
        coordinates: tuple[Coordinate, ...] | list[Coordinate],
        mapper,
        pen: QPen,
        *,
        close: bool = False,
    ) -> None:
        if len(coordinates) < 2:
            return

        painter.setPen(pen)
        step = max(1, len(coordinates) // 1200)
        visible = list(coordinates[::step])
        if visible[-1] != coordinates[-1]:
            visible.append(coordinates[-1])
        for left, right in zip(visible, visible[1:]):
            painter.drawLine(mapper(left), mapper(right))
        if close:
            painter.drawLine(mapper(visible[-1]), mapper(visible[0]))

    def _view_bounds(self) -> tuple[float, float, float, float]:
        assert self._profile is not None
        point = visual_point_at_distance(self._profile, self._distance)
        if not self._follow_enabled or point is None:
            return self._profile.bounds

        half_span = self._follow_window_m / 2.0
        return (
            point.x - half_span,
            point.y - half_span,
            point.x + half_span,
            point.y + half_span,
        )

    def _footer_text(self) -> str:
        assert self._profile is not None
        mode = "follow view" if self._follow_enabled else "full track"
        return f"{self._profile.racing_line.track_name} actual TORCS track geometry - {mode}"


def _screen_mapper(bounds: tuple[float, float, float, float], rect: QRectF):
    min_x, min_y, max_x, max_y = bounds
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(rect.width() / span_x, rect.height() / span_y)
    fitted_width = span_x * scale
    fitted_height = span_y * scale
    left = rect.left() + (rect.width() - fitted_width) / 2.0
    top = rect.top() + (rect.height() - fitted_height) / 2.0

    def mapper(point: Coordinate) -> QPointF:
        x, y = point
        return QPointF(
            left + (x - min_x) * scale,
            top + fitted_height - (y - min_y) * scale,
        )

    return mapper


def _interpolate_visual_point(
    left: RacingLineVisualPoint,
    right: RacingLineVisualPoint,
    distance: float,
) -> RacingLineVisualPoint:
    span = max(0.001, right.source.distance - left.source.distance)
    fraction = max(0.0, min(1.0, (distance - left.source.distance) / span))
    source = RacingLinePoint(
        distance=distance,
        target_track_pos=_interpolate(
            left.source.target_track_pos,
            right.source.target_track_pos,
            fraction,
        ),
        target_speed_kmh=_interpolate(
            left.source.target_speed_kmh,
            right.source.target_speed_kmh,
            fraction,
        ),
        curvature=_interpolate(left.source.curvature, right.source.curvature, fraction),
        phase=left.source.phase,
        reference_section=left.source.reference_section,
    )
    return RacingLineVisualPoint(
        x=_interpolate(left.x, right.x, fraction),
        y=_interpolate(left.y, right.y, fraction),
        source=source,
    )


def _interpolate_graph_point(
    left: RacingLinePoint,
    right: RacingLinePoint,
    distance: float,
) -> RacingLinePoint:
    span = max(0.001, right.distance - left.distance)
    fraction = max(0.0, min(1.0, (distance - left.distance) / span))
    return RacingLinePoint(
        distance=distance,
        target_track_pos=_interpolate(
            left.target_track_pos,
            right.target_track_pos,
            fraction,
        ),
        target_speed_kmh=_interpolate(
            left.target_speed_kmh,
            right.target_speed_kmh,
            fraction,
        ),
        curvature=_interpolate(left.curvature, right.curvature, fraction),
        phase=left.phase,
        reference_section=left.reference_section,
    )


def _interpolate(left: float, right: float, fraction: float) -> float:
    return left + (right - left) * fraction


def _phase_color(phase: str) -> QColor:
    palette = {
        "Brake": QColor("#c0392b"),
        "Accelerate": QColor("#27ae60"),
        "Full throttle": QColor("#1f7a4d"),
        "Turn": QColor("#d97706"),
        "Settle": QColor("#2f80ed"),
    }
    return palette.get(phase_family(phase), palette["Settle"])


def _small_font() -> QFont:
    font = QFont()
    font.setPointSize(8)
    return font
