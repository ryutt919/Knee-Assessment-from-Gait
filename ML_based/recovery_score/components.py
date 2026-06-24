"""Waveform aggregation and distance components for gait normality scoring."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPEEDS = ("slow", "normal", "fast")
SIDES = ("injured", "contralateral")
CHANNELS = (
    "hip_adduction",
    "hip_int_rotation",
    "hip_flexion",
    "knee_adduction",
    "knee_int_rotation",
    "knee_flexion",
    "ankle_adduction",
    "ankle_int_rotation",
    "ankle_dorsiflexion",
)
N_TIMEPOINTS = 101
EPSILON = 1e-12


@dataclass(frozen=True)
class WaveformDataset:
    """One trial-balanced waveform tensor per subject/session."""

    metadata: pd.DataFrame
    matrices: np.ndarray  # (sessions, speeds, sides, channels, 101)
    cycle_counts: np.ndarray  # (sessions, speeds, sides)
    trial_counts: np.ndarray  # (sessions, speeds, sides)


def biological_identity(subject_id: str) -> str:
    """Collapse ACLD/ACLR longitudinal sessions into one patient identity."""
    match = re.fullmatch(r"(ACLD|ACLR|HA)(.+)", str(subject_id))
    if match is None:
        return f"UNKNOWN::{subject_id}"
    cohort, suffix = match.groups()
    return f"ACL::{suffix}" if cohort in {"ACLD", "ACLR"} else f"HA::{suffix}"


def waveform_columns() -> list[str]:
    return [f"{channel}_{point:03d}" for channel in CHANNELS for point in range(N_TIMEPOINTS)]


def load_cycle_waveforms(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    columns = [
        "subject_id",
        "group",
        "speed",
        "trial_id",
        "side_basis",
        "cycle_idx",
        *waveform_columns(),
    ]
    frame = pd.read_parquet(path, columns=columns)
    validate_cycle_frame(frame)
    return frame


def validate_cycle_frame(frame: pd.DataFrame) -> None:
    required = {
        "subject_id",
        "group",
        "speed",
        "trial_id",
        "side_basis",
        *waveform_columns(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"cycle waveform columns missing: {missing[:10]}")
    if frame.empty:
        raise ValueError("cycle waveform frame is empty")
    invalid_speeds = sorted(set(frame["speed"].dropna().astype(str)) - set(SPEEDS))
    invalid_sides = sorted(set(frame["side_basis"].dropna().astype(str)) - set(SIDES))
    if invalid_speeds or invalid_sides:
        raise ValueError(f"unexpected speed/side values: speeds={invalid_speeds}, sides={invalid_sides}")


def aggregate_cycle_waveforms(frame: pd.DataFrame) -> WaveformDataset:
    """Average cycles within trial, then trials within each subject condition."""
    validate_cycle_frame(frame)
    wave_cols = waveform_columns()
    session_cols = ["subject_id", "group"]
    condition_cols = [*session_cols, "speed", "side_basis"]
    trial_cols = [*condition_cols, "trial_id"]

    subject_group_counts = frame.groupby("subject_id", observed=True)["group"].nunique()
    if (subject_group_counts != 1).any():
        bad = subject_group_counts[subject_group_counts != 1].index.tolist()
        raise ValueError(f"subject_id maps to multiple session groups: {bad}")

    trial_means = frame.groupby(trial_cols, observed=True, sort=False)[wave_cols].mean()
    condition_means = trial_means.groupby(condition_cols, observed=True, sort=False).mean()
    cycle_counts = frame.groupby(condition_cols, observed=True, sort=False).size()
    trial_counts = frame.groupby(condition_cols, observed=True, sort=False)["trial_id"].nunique()

    metadata = (
        frame[session_cols]
        .drop_duplicates()
        .assign(identity_id=lambda x: x["subject_id"].map(biological_identity))
        .sort_values(["group", "subject_id"])
        .reset_index(drop=True)
    )
    matrices = np.empty(
        (len(metadata), len(SPEEDS), len(SIDES), len(CHANNELS), N_TIMEPOINTS),
        dtype=np.float64,
    )
    cycles = np.zeros((len(metadata), len(SPEEDS), len(SIDES)), dtype=np.int32)
    trials = np.zeros_like(cycles)

    for row_idx, row in metadata.iterrows():
        for speed_idx, speed in enumerate(SPEEDS):
            for side_idx, side in enumerate(SIDES):
                key = (row["subject_id"], row["group"], speed, side)
                if key not in condition_means.index:
                    raise ValueError(f"missing required subject condition: {key}")
                matrices[row_idx, speed_idx, side_idx] = condition_means.loc[key].to_numpy(
                    dtype=np.float64
                ).reshape(len(CHANNELS), N_TIMEPOINTS)
                cycles[row_idx, speed_idx, side_idx] = int(cycle_counts.loc[key])
                trials[row_idx, speed_idx, side_idx] = int(trial_counts.loc[key])

    if not np.isfinite(matrices).all():
        raise ValueError("aggregated waveform tensor contains non-finite values")
    return WaveformDataset(metadata=metadata, matrices=matrices, cycle_counts=cycles, trial_counts=trials)


def gvs_matrix(matrix: np.ndarray, reference_mean: np.ndarray) -> np.ndarray:
    """Gait Variable Score for speed x side x channel blocks."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 4:
        matrix = matrix[np.newaxis, ...]
    expected = (len(SPEEDS), len(SIDES), len(CHANNELS), N_TIMEPOINTS)
    if matrix.shape[1:] != expected or reference_mean.shape != expected:
        raise ValueError(f"unexpected waveform shape: matrix={matrix.shape}, reference={reference_mean.shape}")
    return np.sqrt(np.mean(np.square(matrix - reference_mean), axis=-1))


def raw_distances(matrix: np.ndarray, reference_mean: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return raw overall/subdomain distances and the 54-element GVS tensor."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 4:
        matrix = matrix[np.newaxis, ...]
    gvs = gvs_matrix(matrix, reference_mean)
    joint_indices = {
        "hip": np.arange(0, 3),
        "knee": np.arange(3, 6),
        "ankle": np.arange(6, 9),
    }
    distances: dict[str, np.ndarray] = {
        "overall": np.sqrt(np.mean(np.square(gvs), axis=(1, 2, 3))),
    }
    for joint, indices in joint_indices.items():
        distances[joint] = np.sqrt(np.mean(np.square(gvs[:, :, :, indices]), axis=(1, 2, 3)))
    for speed_idx, speed in enumerate(SPEEDS):
        distances[speed] = np.sqrt(np.mean(np.square(gvs[:, speed_idx]), axis=(1, 2)))

    subject_bilateral = matrix.mean(axis=2)
    reference_bilateral = reference_mean.mean(axis=1)
    distances["bilateral"] = np.sqrt(
        np.mean(np.square(subject_bilateral - reference_bilateral), axis=(1, 2, 3))
    )
    subject_asymmetry = matrix[:, :, 0] - matrix[:, :, 1]
    reference_asymmetry = reference_mean[:, 0] - reference_mean[:, 1]
    distances["asymmetry"] = np.sqrt(
        np.mean(np.square(subject_asymmetry - reference_asymmetry), axis=(1, 2, 3))
    )
    return distances, gvs


def top_waveform_regions(
    matrix: np.ndarray,
    reference_mean: np.ndarray,
    limit: int = 5,
    half_width: int = 5,
) -> str:
    """Describe the largest waveform deviations without inferential claims."""
    if matrix.ndim != 4:
        raise ValueError("top_waveform_regions expects one session matrix")
    diff = matrix - reference_mean
    gvs = np.sqrt(np.mean(np.square(diff), axis=-1))
    ranked = np.argsort(gvs.ravel())[::-1][:limit]
    rows: list[dict[str, float | int | str]] = []
    for flat_idx in ranked:
        speed_idx, side_idx, channel_idx = np.unravel_index(flat_idx, gvs.shape)
        curve = diff[speed_idx, side_idx, channel_idx]
        peak = int(np.argmax(np.abs(curve)))
        rows.append(
            {
                "speed": SPEEDS[speed_idx],
                "side": SIDES[side_idx],
                "channel": CHANNELS[channel_idx],
                "gvs_degrees": round(float(gvs[speed_idx, side_idx, channel_idx]), 5),
                "peak_cycle_pct": peak,
                "region_start_pct": max(0, peak - half_width),
                "region_end_pct": min(100, peak + half_width),
                "signed_peak_difference_degrees": round(float(curve[peak]), 5),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def hierarchical_bootstrap_matrix(session_frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample trials, then cycles within sampled trials, for one session."""
    validate_cycle_frame(session_frame)
    wave_cols = waveform_columns()
    output = np.empty((len(SPEEDS), len(SIDES), len(CHANNELS), N_TIMEPOINTS), dtype=np.float64)
    for speed_idx, speed in enumerate(SPEEDS):
        for side_idx, side in enumerate(SIDES):
            condition = session_frame[
                session_frame["speed"].eq(speed) & session_frame["side_basis"].eq(side)
            ]
            trial_arrays = [
                trial[wave_cols].to_numpy(dtype=np.float64).reshape(-1, len(CHANNELS), N_TIMEPOINTS)
                for _, trial in condition.groupby("trial_id", observed=True, sort=False)
            ]
            if not trial_arrays:
                raise ValueError(f"bootstrap condition missing: speed={speed}, side={side}")
            sampled_trial_indices = rng.integers(0, len(trial_arrays), size=len(trial_arrays))
            sampled_trial_means = []
            for trial_idx in sampled_trial_indices:
                trial = trial_arrays[int(trial_idx)]
                cycle_indices = rng.integers(0, len(trial), size=len(trial))
                sampled_trial_means.append(trial[cycle_indices].mean(axis=0))
            output[speed_idx, side_idx] = np.mean(sampled_trial_means, axis=0)
    return output
