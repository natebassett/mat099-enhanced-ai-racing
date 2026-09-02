"""Shared N-step TD3 core for the line-aware and sensor-only racers."""

from .contracts import (
    ACTION_SIZE,
    BASE_OBSERVATION_SIZE,
    HISTORY_ACTIONS,
    HISTORY_OBSERVATIONS,
    OBSERVATION_SIZE,
    HistoryEncoder,
    build_base_observation,
    decode_action,
)
from .learner import NstepTd3Config, NstepTd3Learner
from .racing_line import RacingLineFeatureMap

__all__ = [
    "ACTION_SIZE",
    "BASE_OBSERVATION_SIZE",
    "HISTORY_ACTIONS",
    "HISTORY_OBSERVATIONS",
    "OBSERVATION_SIZE",
    "HistoryEncoder",
    "NstepTd3Config",
    "NstepTd3Learner",
    "RacingLineFeatureMap",
    "build_base_observation",
    "decode_action",
]
