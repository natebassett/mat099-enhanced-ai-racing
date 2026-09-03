from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from gui.settings_model import GuiSettings, SettingsStore  # noqa: E402
from gui.settings_view import SettingsView  # noqa: E402


class SettingsModelTests(unittest.TestCase):
    def test_store_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui_settings.json"
            store = SettingsStore(path)
            expected = GuiSettings(
                default_agent_type="sensor_n_step_td3",
                default_track_id="g-track-3",
                start_page="Results",
                chart_window_seconds=120,
                reduce_motion=True,
                appearance_mode="dark",
                colour_mode="accessible",
                language="cy",
                show_td3_advisory=False,
            )

            store.save(expected)

            self.assertEqual(store.load(), expected)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["version"], 2)

    def test_invalid_values_fall_back_to_safe_preferences(self) -> None:
        settings = GuiSettings.from_mapping(
            {
                "start_page": "Unknown",
                "chart_window_seconds": 999,
                "reduce_motion": "yes",
                "appearance_mode": "sepia",
                "colour_mode": "red-green",
                "language": "fr",
                "show_td3_advisory": "no",
            }
        )

        self.assertEqual(settings.start_page, "Live Telemetry")
        self.assertEqual(settings.chart_window_seconds, 180)
        self.assertTrue(settings.reduce_motion)
        self.assertEqual(settings.appearance_mode, "light")
        self.assertEqual(settings.colour_mode, "standard")
        self.assertEqual(settings.language, "en")
        self.assertFalse(settings.show_td3_advisory)


class SettingsViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_preferences_save_and_research_evidence_is_nested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui_settings.json"
            store = SettingsStore(path)
            research_page = QWidget()
            settings = GuiSettings()
            view = SettingsView(
                agents=(
                    SimpleNamespace(name="Agent 8", agent_type="sensor_n_step_td3"),
                ),
                tracks=(
                    SimpleNamespace(label="CG Track 3", track_id="g-track-3"),
                ),
                settings=settings,
                store=store,
                project_root=Path(directory),
                research_evidence_page=research_page,
            )
            emitted: list[GuiSettings] = []
            view.settings_saved.connect(emitted.append)

            view.agent_combo.setCurrentIndex(1)
            view.chart_window_slider.setValue(120)
            view.reduce_motion_checkbox.setChecked(True)
            view.appearance_combo.setCurrentIndex(
                view.appearance_combo.findData("dark")
            )
            view.colour_mode_combo.setCurrentIndex(
                view.colour_mode_combo.findData("accessible")
            )
            view.language_combo.setCurrentIndex(view.language_combo.findData("cy"))
            view.td3_advisory_checkbox.setChecked(False)
            view.start_page_buttons["Results"].setChecked(True)
            view.save()

            self.assertEqual(view.tabs.count(), 4)
            self.assertIs(view.tabs.widget(1), research_page)
            self.assertEqual(store.load().start_page, "Results")
            self.assertEqual(store.load().chart_window_seconds, 120)
            self.assertEqual(store.load().default_agent_type, "sensor_n_step_td3")
            self.assertTrue(store.load().reduce_motion)
            self.assertEqual(store.load().appearance_mode, "dark")
            self.assertEqual(store.load().colour_mode, "accessible")
            self.assertEqual(store.load().language, "cy")
            self.assertFalse(store.load().show_td3_advisory)
            self.assertEqual(emitted, [store.load()])
            self.assertEqual(view.save_status_label.text(), "Cadwyd y gosodiadau")
            view.close()

    def test_language_switch_keeps_canonical_setting_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            view = SettingsView(
                agents=(),
                tracks=(SimpleNamespace(label="CG Track 3", track_id="g-track-3"),),
                settings=GuiSettings(language="cy"),
                store=SettingsStore(Path(directory) / "gui_settings.json"),
                project_root=Path(directory),
            )

            self.assertEqual(view.tabs.tabText(0), "Dewisiadau")
            self.assertEqual(view.start_page_buttons["Results"].text(), "Canlyniadau")
            self.assertEqual(view.language_combo.currentData(), "cy")
            self.assertEqual(view.selected_settings().start_page, "Live Telemetry")

            view.set_language("en")

            self.assertEqual(view.tabs.tabText(0), "Preferences")
            self.assertEqual(view.start_page_buttons["Results"].text(), "Results")
            self.assertEqual(view.language_combo.itemData(1), "cy")
            view.close()

    def test_storage_actions_are_explicit_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            view = SettingsView(
                agents=(),
                tracks=(),
                settings=GuiSettings(),
                store=SettingsStore(Path(directory) / "gui_settings.json"),
                project_root=Path(directory),
            )
            emitted: list[str] = []
            view.clear_cache_requested.connect(lambda: emitted.append("cache"))
            view.reset_run_history_requested.connect(
                lambda: emitted.append("history")
            )
            view.refresh_storage_summary(
                run_count=12,
                cache_bytes=2_048,
                cache_directories=3,
            )

            view.clear_cache_button.click()
            view.reset_history_button.click()

            self.assertEqual(emitted, ["cache", "history"])
            self.assertIn("12 saved runs", view.run_history_summary_label.text())
            self.assertIn("2.0 KB", view.cache_summary_label.text())
            view.close()

    def test_main_window_exposes_settings_in_primary_navigation(self) -> None:
        from gui.main_window import MainWindow, _td3_advisory_text

        window = MainWindow()

        self.assertEqual(window.tabs.tabText(window.settings_tab_index), "Settings")
        window.tabs.setCurrentIndex(window.settings_tab_index)
        self.assertIs(window.settings_view, window.tabs.currentWidget())
        self.assertIs(
            window.settings_view.tabs.widget(1),
            window.settings_view.research_evidence_page,
        )
        self.assertIn("19 of 20", _td3_advisory_text("sensor_n_step_td3"))
        self.assertIn("19 o 20", _td3_advisory_text("sensor_n_step_td3", "cy"))
        window.close()

    def test_local_methodology_documents_are_openable_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            docs = project_root / "docs"
            docs.mkdir()
            guide = docs / "gui_results_workspace.md"
            guide.write_text("# Guide\n", encoding="utf-8")

            row = SettingsView._local_document_row(
                "Results methodology",
                guide,
                "How results are reported.",
            )
            labels = row.findChildren(QWidget)
            link_labels = [
                label
                for label in labels
                if getattr(label, "objectName", lambda: "")() == "sourceLink"
            ]

            self.assertEqual(len(link_labels), 1)
            self.assertIn(guide.resolve().as_uri(), link_labels[0].text())
            self.assertEqual(row.toolTip(), str(guide.resolve()))


if __name__ == "__main__":
    unittest.main()
