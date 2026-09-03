from __future__ import annotations

import sys
from pathlib import Path


def _application_root() -> Path:
    packaged_root = getattr(sys, "_MEIPASS", None)
    if packaged_root:
        return Path(packaged_root).resolve()
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _application_root()
