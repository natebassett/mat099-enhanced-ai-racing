from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.torcs_config import (  # noqa: E402
    BACKUP_SUFFIX,
    TorcsRaceSetup,
    TorcsRuntimeConfig,
    apply_runtime_config,
    restore_stale_runtime_backups,
)


class TorcsConfigTests(unittest.TestCase):
    def test_applies_track_and_car_to_torcs_config_files(self):
        with tempfile.TemporaryDirectory() as directory:
            practice_path, scr_server_path = self._copy_config_files(Path(directory))

            apply_runtime_config(
                practice_path,
                scr_server_path,
                TorcsRaceSetup(
                    track_id="corkscrew",
                    track_category="road",
                    car_id="155-DTM",
                ),
            )

            practice_root = ET.parse(practice_path).getroot()
            scr_root = ET.parse(scr_server_path).getroot()

            track_section = self._find_section(
                self._find_section(practice_root, "Tracks"),
                "1",
            )
            driver_section = self._find_section(
                self._find_section(
                    self._find_section(scr_root, "Robots"),
                    "index",
                ),
                "0",
            )

            self.assertEqual(
                self._attribute_value(track_section, "attstr", "name"),
                "corkscrew",
            )
            self.assertEqual(
                self._attribute_value(track_section, "attstr", "category"),
                "road",
            )
            self.assertEqual(
                self._attribute_value(driver_section, "attstr", "car name"),
                "155-DTM",
            )

    def test_runtime_config_restores_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            practice_path, scr_server_path = self._copy_config_files(project_root)
            original_practice = practice_path.read_text(encoding="utf-8")
            original_scr_server = scr_server_path.read_text(encoding="utf-8")

            with TorcsRuntimeConfig(
                TorcsRaceSetup(
                    track_id="corkscrew",
                    track_category="road",
                    car_id="155-DTM",
                ),
                project_root=project_root,
            ):
                self.assertNotEqual(
                    practice_path.read_text(encoding="utf-8"),
                    original_practice,
                )
                self.assertNotEqual(
                    scr_server_path.read_text(encoding="utf-8"),
                    original_scr_server,
                )
                self.assertTrue(self._backup_path(practice_path).exists())
                self.assertTrue(self._backup_path(scr_server_path).exists())

            self.assertEqual(practice_path.read_text(encoding="utf-8"), original_practice)
            self.assertEqual(
                scr_server_path.read_text(encoding="utf-8"),
                original_scr_server,
            )
            self.assertFalse(self._backup_path(practice_path).exists())
            self.assertFalse(self._backup_path(scr_server_path).exists())

    def test_stale_runtime_backups_are_restored_before_a_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            practice_path, scr_server_path = self._copy_config_files(project_root)
            original_practice = practice_path.read_text(encoding="utf-8")
            original_scr_server = scr_server_path.read_text(encoding="utf-8")

            self._backup_path(practice_path).write_text(
                original_practice,
                encoding="utf-8",
            )
            self._backup_path(scr_server_path).write_text(
                original_scr_server,
                encoding="utf-8",
            )
            practice_path.write_text("<broken />", encoding="utf-8")
            scr_server_path.write_text("<broken />", encoding="utf-8")

            restored_paths = restore_stale_runtime_backups(project_root)

            self.assertEqual(set(restored_paths), {practice_path, scr_server_path})
            self.assertEqual(practice_path.read_text(encoding="utf-8"), original_practice)
            self.assertEqual(
                scr_server_path.read_text(encoding="utf-8"),
                original_scr_server,
            )
            self.assertFalse(self._backup_path(practice_path).exists())
            self.assertFalse(self._backup_path(scr_server_path).exists())

    def _copy_config_files(self, project_root: Path) -> tuple[Path, Path]:
        practice_path = project_root / "torcs" / "config" / "raceman" / "practice.xml"
        scr_server_path = (
            project_root / "torcs" / "drivers" / "scr_server" / "scr_server.xml"
        )
        practice_path.parent.mkdir(parents=True, exist_ok=True)
        scr_server_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            PROJECT_ROOT / "torcs" / "config" / "raceman" / "practice.xml",
            practice_path,
        )
        shutil.copyfile(
            PROJECT_ROOT / "torcs" / "drivers" / "scr_server" / "scr_server.xml",
            scr_server_path,
        )
        return practice_path, scr_server_path

    def _backup_path(self, path: Path) -> Path:
        return path.with_name(f"{path.name}{BACKUP_SUFFIX}")

    def _find_section(self, parent: ET.Element, name: str) -> ET.Element:
        return next(
            child for child in parent.findall("section") if child.get("name") == name
        )

    def _attribute_value(
        self,
        section: ET.Element,
        tag_name: str,
        attribute_name: str,
    ) -> str | None:
        return next(
            child.get("val")
            for child in section.findall(tag_name)
            if child.get("name") == attribute_name
        )


if __name__ == "__main__":
    unittest.main()
