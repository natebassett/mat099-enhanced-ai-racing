from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from project_paths import PROJECT_ROOT

BACKUP_SUFFIX = ".codex-runtime.bak"


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
        self._config_paths = [self.practice_path, self.scr_server_path]

    def __enter__(self) -> "TorcsRuntimeConfig":
        restore_stale_runtime_backups(self.project_root)
        self._create_runtime_backups()

        try:
            apply_runtime_config(self.practice_path, self.scr_server_path, self.setup)
        except Exception:
            self._restore_runtime_backups()
            raise

        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._restore_runtime_backups()

    def _create_runtime_backups(self) -> None:
        for path in self._config_paths:
            _backup_path(path).write_text(
                path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    def _restore_runtime_backups(self) -> None:
        for path in self._config_paths:
            backup_path = _backup_path(path)
            if not backup_path.exists():
                continue
            path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink()


def restore_stale_runtime_backups(project_root: Path = PROJECT_ROOT) -> list[Path]:
    restored_paths = []
    for path in _runtime_config_paths(Path(project_root)):
        backup_path = _backup_path(path)
        if not backup_path.exists():
            continue
        path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
        backup_path.unlink()
        restored_paths.append(path)
    return restored_paths


def _runtime_config_paths(project_root: Path) -> list[Path]:
    return [
        project_root / "torcs" / "config" / "raceman" / "practice.xml",
        project_root / "torcs" / "drivers" / "scr_server" / "scr_server.xml",
    ]


def _backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{BACKUP_SUFFIX}")


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
