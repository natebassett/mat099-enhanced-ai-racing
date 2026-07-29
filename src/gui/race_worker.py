from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

try:
    from .project_discovery import AgentOption, CarOption, TrackOption
    from .torcs_config import TorcsRaceSetup, TorcsRuntimeConfig
except ImportError:
    from project_discovery import AgentOption, CarOption, TrackOption
    from torcs_config import TorcsRaceSetup, TorcsRuntimeConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


class RaceWorker(QObject):
    status_changed = Signal(str)
    explanation_changed = Signal(str)
    run_saved = Signal(int)
    race_finished = Signal(object)
    race_failed = Signal(str)
    completed = Signal()

    def __init__(
        self,
        agent_option: AgentOption,
        track_option: TrackOption,
        car_option: CarOption,
    ) -> None:
        super().__init__()
        self.agent_option = agent_option
        self.track_option = track_option
        self.car_option = car_option
        self._stop_requested = False
        self._runner: Any = None
        self._agent: Any = None

    @Slot()
    def run(self) -> None:
        from runner.torcs_runner import TorcsRunner
        from storage import RaceRepository

        repository = RaceRepository()
        results: dict[str, Any] | None = None
        run_id: int | None = None

        try:
            self.status_changed.emit("Configuring TORCS race setup...")
            self._agent = self._create_agent()
            agent_id = repository.register_agent(
                name=self._agent.name,
                agent_type=self._agent.agent_type,
                version=self._agent.version,
                config=self._agent.config,
            )

            setup = TorcsRaceSetup(
                track_id=self.track_option.track_id,
                track_category=self.track_option.category,
                car_id=self.car_option.car_id,
            )
            with TorcsRuntimeConfig(setup):
                self._runner = TorcsRunner()
                self.status_changed.emit("Launching TORCS...")
                self.explanation_changed.emit(
                    "TORCS is starting. The dashboard will stay responsive while "
                    "the simulator opens in its own window."
                )
                self._runner.launch()
                if self._stop_requested:
                    raise RaceStoppedError("Race stopped before connecting to TORCS")

                self.status_changed.emit("Connecting to TORCS...")
                self._runner.connect()

                self.status_changed.emit(
                    f"Running {self.agent_option.name} on {self.track_option.track_id}..."
                )
                self._runner.load_track(self.track_option.track_id)
                results = self._runner.run(
                    self._agent,
                    stop_requested=self._should_stop,
                )

            self.status_changed.emit("Saving race results...")
            run_id = repository.record_run(
                agent_id=agent_id,
                track=self.track_option.track_id,
                seed=getattr(self._agent, "seed", None),
                results=results,
            )
            repository.record_run_telemetry(
                run_id,
                results.get("telemetry_samples", []),
            )
            self.run_saved.emit(run_id)
            self.race_finished.emit({"run_id": run_id, "results": results})
            self.status_changed.emit(f"Race finished. Saved as run #{run_id}.")
        except RaceStoppedError as error:
            self.status_changed.emit(str(error))
            self.explanation_changed.emit(str(error))
            self.race_finished.emit(
                {
                    "run_id": run_id,
                    "results": results or {"termination_reason": "stopped"},
                }
            )
        except Exception:
            if self._stop_requested:
                self.status_changed.emit("Race stopped.")
                self.explanation_changed.emit("Race stopped before completion.")
                self.race_finished.emit(
                    {
                        "run_id": run_id,
                        "results": results or {"termination_reason": "stopped"},
                    }
                )
            else:
                self.race_failed.emit(traceback.format_exc())
        finally:
            if self._runner is not None:
                self._runner.shutdown()
            if self._agent is not None and results is None:
                close_agent = getattr(self._agent, "close", None)
                if close_agent is not None:
                    close_agent()
            self.completed.emit()

    @Slot()
    def request_stop(self) -> None:
        self._stop_requested = True
        self.status_changed.emit("Stopping race...")
        if self._runner is not None:
            self._runner.shutdown()

    def _should_stop(self) -> bool:
        return self._stop_requested

    def _create_agent(self) -> Any:
        agent_class = _load_class(self.agent_option.class_path)
        if self.agent_option.requires_racing_line:
            if self.track_option.racing_line_path is None:
                raise ValueError(
                    f"{self.agent_option.name} requires a saved racing line for "
                    f"{self.track_option.track_id}"
                )
            return agent_class(racing_line_path=self.track_option.racing_line_path)
        return agent_class()


class RaceStoppedError(RuntimeError):
    pass


def _load_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
