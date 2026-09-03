from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import project_paths  # noqa: E402


class ProjectPathTests(unittest.TestCase):
    def test_source_mode_resolves_repository_root(self) -> None:
        self.assertEqual(project_paths.PROJECT_ROOT, PROJECT_ROOT)

    def test_frozen_mode_resolves_pyinstaller_data_root(self) -> None:
        with TemporaryDirectory() as directory:
            packaged_root = Path(directory).resolve()
            with patch.object(sys, "_MEIPASS", str(packaged_root), create=True):
                self.assertEqual(project_paths._application_root(), packaged_root)


if __name__ == "__main__":
    unittest.main()
