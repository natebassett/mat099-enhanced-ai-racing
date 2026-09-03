from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

SMOKE_TEST_ARGUMENT = "--packaging-smoke-test"
SMOKE_TEST_LOG = Path(tempfile.gettempdir()) / "enhanced-ai-racing-smoke-test.log"


def run_packaging_smoke_test() -> int:
    try:
        from PySide6.QtWidgets import QApplication

        from agents.n_step_td3_agent import NstepTd3Agent
        from agents.sensor_n_step_td3_agent import SensorNstepTd3Agent
        from gui.app import main as _gui_main
        from gui.project_discovery import load_project_options
        from project_paths import PROJECT_ROOT

        qt_application = QApplication.instance() or QApplication([])
        if not callable(_gui_main):
            raise RuntimeError("Packaged GUI entry point is unavailable")

        torcs_executable = PROJECT_ROOT / "torcs" / "wtorcs.exe"
        if not torcs_executable.is_file():
            raise FileNotFoundError(f"Packaged TORCS is missing: {torcs_executable}")

        options = load_project_options(PROJECT_ROOT)
        if len(options.agents) != 7 or not options.tracks or not options.cars:
            raise RuntimeError("Packaged driver, track, or car discovery is incomplete")

        agents = (
            NstepTd3Agent(require_policy=True),
            SensorNstepTd3Agent(require_policy=True),
        )
        if not all(agent.policy_loaded for agent in agents):
            raise RuntimeError("One or more packaged neural policies did not load")

        qt_application.quit()
    except Exception:
        SMOKE_TEST_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        return 1

    SMOKE_TEST_LOG.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    if SMOKE_TEST_ARGUMENT in sys.argv:
        raise SystemExit(run_packaging_smoke_test())

    from gui.app import main

    raise SystemExit(main())
