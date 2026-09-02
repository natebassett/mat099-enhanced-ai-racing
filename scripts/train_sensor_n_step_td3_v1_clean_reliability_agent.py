from __future__ import annotations

import sys
from pathlib import Path

from train_n_step_td3_agent import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACE_SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "agent8_sensor_n_step_td3_v1_stability"
    / "champion_82_294s_5of20_external.pt"
)


def has_resume_argument(arguments: list[str]) -> bool:
    return any(
        argument in {"--resume", "--resume-best", "--resume-pace"}
        or argument.startswith("--resume=")
        for argument in arguments
    )


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    if not has_resume_argument(arguments):
        arguments = ["--resume", str(PACE_SOURCE_CHECKPOINT), *arguments]
    main(
        [
            "--sensor-only",
            "--sensor-v1-clean-reliability-continuation",
            "--observation-noise-std",
            "0",
            *arguments,
        ]
    )
