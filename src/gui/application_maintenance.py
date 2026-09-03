from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheSummary:
    directories: int
    bytes_used: int


def inspect_application_cache(project_root: Path) -> CacheSummary:
    paths = tuple(_cache_directories(project_root))
    return CacheSummary(
        directories=len(paths),
        bytes_used=sum(_directory_size(path) for path in paths),
    )


def clear_application_cache(project_root: Path) -> CacheSummary:
    paths = tuple(_cache_directories(project_root))
    bytes_used = sum(_directory_size(path) for path in paths)
    removed = 0
    for path in paths:
        if not path.exists():
            continue
        shutil.rmtree(path)
        removed += 1
    return CacheSummary(directories=removed, bytes_used=bytes_used)


def _cache_directories(project_root: Path):
    project_root = project_root.resolve()
    for relative_root in ("src", "scripts"):
        search_root = (project_root / relative_root).resolve()
        if not search_root.is_dir():
            continue
        for path in search_root.rglob("__pycache__"):
            resolved = path.resolve()
            if path.is_symlink() or not path.is_dir():
                continue
            try:
                resolved.relative_to(search_root)
            except ValueError:
                continue
            yield resolved


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total
