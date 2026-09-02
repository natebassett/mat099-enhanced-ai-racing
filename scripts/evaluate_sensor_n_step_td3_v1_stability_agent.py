from __future__ import annotations

import sys
from pathlib import Path

from evaluate_n_step_td3_agent import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_stability"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "agent8_sensor_n_step_td3_v1_stability"
DEFAULT_MODEL_CANDIDATES = (
    MODEL_DIR / "best_evaluation.pt",
    MODEL_DIR / "best_pace.pt",
    MODEL_DIR / "best_distance.pt",
    MODEL_DIR / "latest.pt",
)


def default_model_path() -> Path:
    return next(
        (path for path in DEFAULT_MODEL_CANDIDATES if path.is_file()),
        DEFAULT_MODEL_CANDIDATES[0],
    )


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    has_policy_path = any(
        argument == "--policy-path" or argument.startswith("--policy-path=")
        for argument in arguments
    )
    has_output_dir = any(
        argument == "--output-dir" or argument.startswith("--output-dir=")
        for argument in arguments
    )
    if not has_policy_path:
        arguments = ["--policy-path", str(default_model_path()), *arguments]
    if not has_output_dir:
        arguments = ["--output-dir", str(OUTPUT_DIR), *arguments]
    main(arguments)
