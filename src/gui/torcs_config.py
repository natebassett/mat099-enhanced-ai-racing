from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TorcsRaceSetup:
    track_id: str
    track_category: str
    car_id: str


class TorcsRuntimeConfig:
    """Temporarily apply GUI race selections to TORCS XML config files."""

    def __init__(
        self,
        setup: TorcsRaceSetup,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.setup = setup
        self.project_root = Path(project_root)
        self.practice_path = (
            self.project_root / "torcs" / "config" / "raceman" / "practice.xml"
        )
        self.scr_server_path = (
            self.project_root / "torcs" / "drivers" / "scr_server" / "scr_server.xml"
        )
        self._original_files: dict[Path, str] = {}

    def __enter__(self) -> "TorcsRuntimeConfig":
        self._original_files = {
            self.practice_path: self.practice_path.read_text(encoding="utf-8"),
            self.scr_server_path: self.scr_server_path.read_text(encoding="utf-8"),
        }
        apply_runtime_config(self.practice_path, self.scr_server_path, self.setup)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        for path, original_text in self._original_files.items():
            path.write_text(original_text, encoding="utf-8")


def apply_runtime_config(
    practice_path: Path,
    scr_server_path: Path,
    setup: TorcsRaceSetup,
) -> None:
    practice_root = ET.parse(practice_path).getroot()
    scr_server_root = ET.parse(scr_server_path).getroot()

    _set_practice_track(practice_root, setup.track_id, setup.track_category)
    _set_scr_server_car(scr_server_root, setup.car_id)

    _write_xml(practice_path, practice_root)
    _write_xml(scr_server_path, scr_server_root)


def _set_practice_track(root: ET.Element, track_id: str, category: str) -> None:
    tracks_section = _find_required_section(root, "Tracks")
    first_track = _find_required_section(tracks_section, "1")
    _set_attribute(first_track, "attstr", "name", track_id)
    _set_attribute(first_track, "attstr", "category", category)


def _set_scr_server_car(root: ET.Element, car_id: str) -> None:
    robots_section = _find_required_section(root, "Robots")
    index_section = _find_required_section(robots_section, "index")
    first_driver = _find_required_section(index_section, "0")
    _set_attribute(first_driver, "attstr", "car name", car_id)


def _find_required_section(parent: ET.Element, name: str) -> ET.Element:
    section = next(
        (
            child
            for child in parent.findall("section")
            if child.get("name") == name
        ),
        None,
    )
    if section is None:
        raise ValueError(f"Missing TORCS config section: {name}")
    return section


def _set_attribute(
    section: ET.Element,
    tag_name: str,
    attribute_name: str,
    value: str,
) -> None:
    attribute = next(
        (
            child
            for child in section.findall(tag_name)
            if child.get("name") == attribute_name
        ),
        None,
    )
    if attribute is None:
        attribute = ET.SubElement(section, tag_name, {"name": attribute_name})
    attribute.set("val", value)


def _write_xml(path: Path, root: ET.Element) -> None:
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
