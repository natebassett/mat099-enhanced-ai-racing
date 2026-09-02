from __future__ import annotations

import sys
from pathlib import Path

from evaluate_n_step_td3_agent import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "agent8_sensor_n_step_td3_v3_clean_observation"
)


if __name__ == "__main__":
    main(
        [
            "--observation-noise-std",
            "0",
            "--output-dir",
            str(OUTPUT_DIR),
            *sys.argv[1:],
        ]
    )
