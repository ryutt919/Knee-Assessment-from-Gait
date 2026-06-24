"""HA-referenced gait normality scoring."""

from .components import (
    CHANNELS,
    SIDES,
    SPEEDS,
    WaveformDataset,
    aggregate_cycle_waveforms,
    biological_identity,
    load_cycle_waveforms,
)
from .scorer import GaitNormalityScorer, RecoveryScorer
from .validation import run_scoring_pipeline

__all__ = [
    "CHANNELS",
    "SIDES",
    "SPEEDS",
    "WaveformDataset",
    "aggregate_cycle_waveforms",
    "biological_identity",
    "load_cycle_waveforms",
    "GaitNormalityScorer",
    "RecoveryScorer",
    "run_scoring_pipeline",
]
