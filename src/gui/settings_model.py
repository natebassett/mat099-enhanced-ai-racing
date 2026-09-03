from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SETTINGS_VERSION = 2
START_PAGES = ("Live Telemetry", "Results", "Agent Lab")
APPEARANCE_MODES = ("system", "light", "dark")
COLOUR_MODES = ("standard", "accessible", "high_contrast")
LANGUAGES = ("en", "cy")


@dataclass(frozen=True)
class GuiSettings:
    default_agent_type: str = ""
    default_track_id: str = "g-track-3"
    start_page: str = "Live Telemetry"
    chart_window_seconds: int = 90
    reduce_motion: bool = False
    appearance_mode: str = "light"
    colour_mode: str = "standard"
    language: str = "en"
    show_td3_advisory: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> GuiSettings:
        start_page = str(values.get("start_page") or cls.start_page)
        if start_page not in START_PAGES:
            start_page = cls.start_page

        chart_window = _bounded_int(
            values.get("chart_window_seconds"),
            default=cls.chart_window_seconds,
            minimum=30,
            maximum=180,
        )
        return cls(
            default_agent_type=str(values.get("default_agent_type") or ""),
            default_track_id=str(
                values.get("default_track_id") or cls.default_track_id
            ),
            start_page=start_page,
            chart_window_seconds=chart_window,
            reduce_motion=_as_bool(values.get("reduce_motion"), default=False),
            appearance_mode=_choice(
                values.get("appearance_mode"),
                choices=APPEARANCE_MODES,
                default=cls.appearance_mode,
            ),
            colour_mode=_choice(
                values.get("colour_mode"),
                choices=COLOUR_MODES,
                default=cls.colour_mode,
            ),
            language=_choice(
                values.get("language"),
                choices=LANGUAGES,
                default=cls.language,
            ),
            show_td3_advisory=_as_bool(
                values.get("show_td3_advisory"),
                default=cls.show_td3_advisory,
            ),
        )


class SettingsStore:
    """Small JSON preference store kept separate from experiment artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> GuiSettings:
        if not self.path.is_file():
            return GuiSettings()
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return GuiSettings()
        if not isinstance(document, dict):
            return GuiSettings()
        return GuiSettings.from_mapping(document)

    def save(self, settings: GuiSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"version": SETTINGS_VERSION, **asdict(settings)}
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _choice(value: Any, *, choices: tuple[str, ...], default: str) -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in choices else default
