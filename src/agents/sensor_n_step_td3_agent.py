from __future__ import annotations

from pathlib import Path

from agents.n_step_td3_agent import NstepTd3Agent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3"
RELIABILITY_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_clean_reliability"
)
ROBUST_PACE_MODEL_DIR = (
    PROJECT_ROOT / "models" / "agent8_sensor_n_step_td3_v1_robust_pace"
)
SELF_IMITATION_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "agent8_sensor_n_step_td3_self_imitation_stability"
)
DEFAULT_MODEL_CANDIDATES = (
    SELF_IMITATION_MODEL_DIR / "champion_83_038s_36of40_clean.pt",
    ROBUST_PACE_MODEL_DIR / "best_evaluation.pt",
    RELIABILITY_MODEL_DIR / "best_evaluation.pt",
    MODEL_DIR / "best_evaluation.pt",
    MODEL_DIR / "best_pace.pt",
    MODEL_DIR / "best_distance.pt",
    MODEL_DIR / "latest.pt",
)


class SensorNstepTd3Agent(NstepTd3Agent):
    name = "Sensor-Only N-Step TD3 Racer"
    agent_type = "sensor_n_step_td3"
    version = "0.1"
    requires_racing_line = False
    default_model_candidates = DEFAULT_MODEL_CANDIDATES
