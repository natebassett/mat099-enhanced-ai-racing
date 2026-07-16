import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackSegment:
    name: str
    segment_type: str
    length: float
    turn_radians: float
    grade_percent: float


class TorcsTrackMap:
    """Track geometry extracted from a TORCS track-definition XML file."""

    def __init__(self, name, width, segments, source_path):
        self.name = name
        self.width = width
        self.segments = segments
        self.source_path = Path(source_path)

    @classmethod
    def from_xml(cls, path):
        path = Path(path)
        xml_text = (
            path.read_text(encoding="utf-8")
            .replace("&default-surfaces;", "")
            .replace("&default-objects;", "")
        )
        root = ET.fromstring(xml_text)
        main_track = next(
            section
            for section in root.findall("section")
            if section.get("name") == "Main Track"
        )
        width = _attribute_number(main_track, "width")
        segment_section = next(
            section
            for section in main_track.findall("section")
            if section.get("name") == "Track Segments"
        )

        segments = []
        for section in segment_section.findall("section"):
            attributes = _direct_attributes(section)
            segment_type = attributes["type"]
            if segment_type == "str":
                length = float(attributes["lg"])
                turn_radians = 0.0
            else:
                turn_sign = 1.0 if segment_type == "lft" else -1.0
                arc = math.radians(float(attributes["arc"]))
                start_radius = float(attributes["radius"])
                end_radius = float(attributes.get("end radius", start_radius))
                # TORCS transition segments describe a radius range. The mean
                # radius preserves the declared total heading while closely
                # matching the generated track geometry.
                length = arc * (start_radius + end_radius) / 2.0
                turn_radians = turn_sign * arc

            segments.append(
                TrackSegment(
                    name=section.get("name"),
                    segment_type=segment_type,
                    length=length,
                    turn_radians=turn_radians,
                    grade_percent=float(attributes.get("grade", 0.0)),
                )
            )

        return cls(root.get("name"), width, segments, path)

    @property
    def geometry_length(self):
        return sum(segment.length for segment in self.segments)

    def sample_centerline(self, spacing=3.0, measured_length=None):
        """Return a closed, dense centreline with segment interpolation data."""
        points = []
        x = y = heading = raw_distance = 0.0

        for segment_index, segment in enumerate(self.segments):
            sample_count = max(1, round(segment.length / spacing))
            step = segment.length / sample_count
            curvature = segment.turn_radians / segment.length

            for sample_index in range(sample_count):
                fraction = sample_index / sample_count
                points.append(
                    {
                        "raw_distance": raw_distance,
                        "x": x,
                        "y": y,
                        "heading": heading,
                        "curvature": curvature,
                        "segment_index": segment_index,
                        "segment_fraction": fraction,
                    }
                )
                midpoint_heading = heading + curvature * step / 2.0
                x += math.cos(midpoint_heading) * step
                y += math.sin(midpoint_heading) * step
                heading += curvature * step
                raw_distance += step

        # Track-editor transition approximations leave a small closure drift.
        # Distribute it smoothly over the lap instead of introducing a seam.
        total_raw_distance = raw_distance
        distance_scale = (measured_length or total_raw_distance) / total_raw_distance
        for point in points:
            lap_fraction = point["raw_distance"] / total_raw_distance
            point["x"] -= x * lap_fraction
            point["y"] -= y * lap_fraction
            point["distance"] = point["raw_distance"] * distance_scale

        _calculate_closed_geometry(points, measured_length or total_raw_distance)
        return points


def _direct_attributes(section):
    return {
        child.get("name"): child.get("val")
        for child in section
        if child.tag in {"attnum", "attstr"}
    }


def _attribute_number(section, name):
    for child in section:
        if child.tag == "attnum" and child.get("name") == name:
            return float(child.get("val"))
    raise ValueError(f"Missing numeric track attribute: {name}")


def _calculate_closed_geometry(points, track_length):
    count = len(points)
    for index, point in enumerate(points):
        previous = points[(index - 1) % count]
        following = points[(index + 1) % count]
        tangent_x = following["x"] - previous["x"]
        tangent_y = following["y"] - previous["y"]
        tangent_length = math.hypot(tangent_x, tangent_y) or 1.0
        point["normal_x"] = -tangent_y / tangent_length
        point["normal_y"] = tangent_x / tangent_length

    for index, point in enumerate(points):
        previous = points[(index - 1) % count]
        following = points[(index + 1) % count]
        heading_before = math.atan2(
            point["y"] - previous["y"],
            point["x"] - previous["x"],
        )
        heading_after = math.atan2(
            following["y"] - point["y"],
            following["x"] - point["x"],
        )
        heading_change = math.atan2(
            math.sin(heading_after - heading_before),
            math.cos(heading_after - heading_before),
        )
        distance_before = _cyclic_distance(
            previous["distance"],
            point["distance"],
            track_length,
        )
        distance_after = _cyclic_distance(
            point["distance"],
            following["distance"],
            track_length,
        )
        point["curvature"] = heading_change / max(
            (distance_before + distance_after) / 2.0,
            0.1,
        )


def _cyclic_distance(start, end, track_length):
    distance = end - start
    return distance if distance >= 0 else distance + track_length
