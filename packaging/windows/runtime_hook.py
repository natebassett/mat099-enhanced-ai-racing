from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def _configure_packaged_dll_search_path() -> None:
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    bundle_root = Path(sys._MEIPASS)
    candidates = (
        bundle_root / "PySide6",
        bundle_root / "shiboken6",
        bundle_root / "torch" / "lib",
        bundle_root,
    )
    dll_directories = [path for path in candidates if path.is_dir()]
    if not dll_directories:
        return

    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        [*(str(path) for path in dll_directories), existing_path]
    )

    if hasattr(os, "add_dll_directory"):
        for path in dll_directories:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


_configure_packaged_dll_search_path()
