from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gui.application_maintenance import (  # noqa: E402
    clear_application_cache,
    inspect_application_cache,
)


class ApplicationMaintenanceTests(unittest.TestCase):
    def test_cache_cleanup_is_limited_to_python_cache_under_source_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_cache = root / "src" / "gui" / "__pycache__"
            script_cache = root / "scripts" / "__pycache__"
            protected_data = root / "data" / "generated" / "__pycache__"
            protected_model = root / "models" / "champion.pt"
            protected_env = root / "torcs-env" / "Lib" / "__pycache__"
            for path in (source_cache, script_cache, protected_data, protected_env):
                path.mkdir(parents=True)
                (path / "cache.pyc").write_bytes(b"cache")
            protected_model.parent.mkdir(parents=True)
            protected_model.write_bytes(b"model")

            summary = inspect_application_cache(root)
            removed = clear_application_cache(root)

            self.assertEqual(summary.directories, 2)
            self.assertEqual(summary.bytes_used, 10)
            self.assertEqual(removed, summary)
            self.assertFalse(source_cache.exists())
            self.assertFalse(script_cache.exists())
            self.assertTrue(protected_data.exists())
            self.assertTrue(protected_env.exists())
            self.assertEqual(protected_model.read_bytes(), b"model")


if __name__ == "__main__":
    unittest.main()
