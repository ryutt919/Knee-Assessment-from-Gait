"""Waveform-first gait analysis for adult HA/ACLD/ACLR cohorts.

This script implements a standalone analysis pipeline directly from
``data/processed/raw_merged.parquet`` and produces the following outputs:

- ``subject_speed_waveforms.csv``
- ``spm_results.csv``
- ``feature_table.csv``
- ``lmm_results.csv``
- ``feature_ranking.csv``
- ``validation_summary.csv``
- ``sensitivity_comparison.csv``

The pipeline is intentionally separated from the existing peak-based
preprocessing scripts so we can keep the legacy workflow intact while adding
the new waveform-based analysis.
"""

from __future__ import annotations

import logging
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import spm1d
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*Non-sphericity corrections for one-way ANOVA are currently approximate.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated in version 1.8.*",
    category=FutureWarning,
)


BASE_DIR = Path(__file__).resolve().parents[2]
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PATH_RAW = PROCESSED_DIR / "raw_merged.parquet"
PATH_ID = DATA_DIR / "ID.csv"
OUT_DIR = PROCESSED_DIR / "waveform_based"
PATH_OUT_WAVEFORMS = OUT_DIR / "subject_speed_waveforms.csv"
PATH_OUT_SPM = OUT_DIR / "spm_results.csv"
PATH_OUT_FEATURES = OUT_DIR / "feature_table.csv"
PATH_OUT_LMM = OUT_DIR / "lmm_results.csv"
PATH_OUT_RANKING = OUT_DIR / "feature_ranking.csv"
PATH_OUT_VALIDATION = OUT_DIR / "validation_summary.csv"
PATH_OUT_SENS = OUT_DIR / "sensitivity_comparison.csv"

SUBJECT_ID_ALIASES = {
    "ACLR38": "ACLR36",
}

ADULT_RAW_GROUPS = {
    "Healthy adults": "HA",
    "ACLD": "ACLD",
    "ACLR": "ACLR",
}

HEEL_CONTACT_COLS = {
    "Right": "footContacts_2",
    "Left": "footContacts_0",
}
MAX_TIME_GAP_MS = 100.0

GAIT_PHASES = {
    "loading_response": (0, 10),
    "mid_stance": (10, 30),
    "terminal_stance": (30, 50),
    "swing": (60, 100),
}

POINT_COLS = [f"point_{i:03d}" for i in range(101)]
PRIMARY_SPEEDS = ["slow", "normal", "fast"]
PRIMARY_GROUPS = ["HA", "ACLD", "ACLR"]
PAIRWISE_COMPARISONS = [("HA", "ACLD"), ("HA", "ACLR"), ("ACLD", "ACLR")]
PRIMARY_ALPHA = 0.05
HEALTHY_PSEUDO_SEED = 42


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    joint_label: str
    axis_index: int
    axis_label: str
    anatomical_label: str
    side: str
    column: str


def parse_mvnx_labels(mvnx_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Extract segment/joint/sensor labels from the MVNX header."""
    segment_labels: list[str] = []
    joint_labels: list[str] = []
    sensor_labels: list[str] = []

    with mvnx_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "<frame " in line:
                break
            seg = re.search(r'<segment\s[^>]*label="([^"]+)"', line)
            if seg:
                segment_labels.append(seg.group(1))
            jt = re.search(r'<joint\s[^>]*label="([^"]+)"', line)
            if jt:
                joint_labels.append(jt.group(1))
            sens = re.search(r'<sensor\s[^>]*label="([^"]+)"', line)
            if sens:
                sensor_labels.append(sens.group(1))
    return segment_labels, joint_labels, sensor_labels


def find_sample_files() -> tuple[Path, Path]:
    """Pick one adult normal-speed mvnx/xlsx pair for label validation."""
    mvnx_candidates = sorted(DATA_DIR.glob("Healthy adults/*/Gait/Normal/*.mvnx"))
    xlsx_candidates = sorted(DATA_DIR.glob("Healthy adults/*/Gait/Normal/*.xlsx"))
    if not mvnx_candidates or not xlsx_candidates:
        raise FileNotFoundError("샘플 MVNX/XLSX 파일을 찾을 수 없습니다.")
    return mvnx_candidates[0], xlsx_candidates[0]


def build_joint_feature_specs(sample_mvnx: Path) -> dict[str, dict[str, FeatureSpec]]:
    """Build correct hip/knee/ankle column mapping from MVNX joint labels."""
    _, joint_labels, _ = parse_mvnx_labels(sample_mvnx)
    joint_index = {label: idx for idx, label in enumerate(joint_labels)}

    base_defs = {
        "hip": ("Hip", "adduction", "int_rotation", "flexion"),
        "knee": ("Knee", "adduction", "int_rotation", "flexion"),
        "ankle": ("Ankle", "adduction", "int_rotation", "dorsiflexion"),
    }
    label_map = {
        "adduction": "Abduction/Adduction",
        "int_rotation": "Internal/External Rotation",
        "flexion": "Flexion/Extension",
        "dorsiflexion": "Dorsiflexion/Plantarflexion",
    }

    specs: dict[str, dict[str, FeatureSpec]] = {}
    for side in ("Right", "Left"):
        side_specs: dict[str, FeatureSpec] = {}
        for joint_key, (joint_suffix, axis0, axis1, axis2) in base_defs.items():
            raw_label = f"j{side}{joint_suffix}"
            if raw_label not in joint_index:
                raise KeyError(f"MVNX joint label 누락: {raw_label}")
            base_col = joint_index[raw_label] * 3
            ordered = [axis0, axis1, axis2]
            for axis_idx, axis_name in enumerate(ordered):
                feature_name = f"{joint_key}_{axis_name}"
                side_specs[feature_name] = FeatureSpec(
                    feature_name=feature_name,
                    joint_label=raw_label,
                    axis_index=axis_idx,
                    axis_label=axis_name,
                    anatomical_label=f"{side} {joint_suffix} {label_map[axis_name]}",
                    side=side,
                    column=f"jointAngle_{base_col + axis_idx}",
                )
        specs[side] = side_specs
    return specs


def load_metadata() -> pd.DataFrame:
    """Load adult metadata and assign pseudo injured side for healthy adults."""

    def to_person_id(subject_id: str) -> str:
        if subject_id.startswith("ACL"):
            match = re.search(r"(\d+)$", subject_id)
            if match:
                return f"ACL_{match.group(1)}"
        return subject_id

    meta = pd.read_csv(PATH_ID).copy()
    meta["subject_id"] = meta["ID"].replace(SUBJECT_ID_ALIASES)
    meta["sex"] = meta["Sex"].astype(str).str.strip().str.lower().replace({"female": "Female", "male": "Male", "f": "Female", "m": "Male"})
    meta["age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["weight"] = pd.to_numeric(meta["Weight"], errors="coerce")
    meta["height"] = pd.to_numeric(meta["Height"], errors="coerce")
    meta["injured_leg"] = meta["Injured leg"].replace({"nan": np.nan})

    adults = meta[meta["Group"].isin([1, 3, 4])].copy()
    adults["group"] = adults["Group"].map({1: "HA", 3: "ACLD", 4: "ACLR"})
    adults["person_id"] = adults["subject_id"].apply(to_person_id)

    acld = adults[adults["group"] == "ACLD"].copy()
    right_ratio = float((acld["injured_leg"] == "Right").mean())

    healthy = adults[adults["group"] == "HA"].copy().sort_values("subject_id").reset_index(drop=True)
    n_right = int(round(len(healthy) * right_ratio))
    rng = np.random.default_rng(HEALTHY_PSEUDO_SEED)
    perm = rng.permutation(len(healthy))
    right_idx = set(perm[:n_right].tolist())
    healthy["injured_leg"] = ["Right" if idx in right_idx else "Left" for idx in range(len(healthy))]
    healthy["pseudo_injured_leg"] = True

    injured = adults[adults["group"] != "HA"].copy()
    injured["pseudo_injured_leg"] = False

    keep_cols = [
        "subject_id",
        "group",
        "person_id",
        "sex",
        "age",
        "weight",
        "height",
        "injured_leg",
        "pseudo_injured_leg",
    ]
    combined = pd.concat([healthy[keep_cols], injured[keep_cols]], ignore_index=True)
    combined = combined.drop_duplicates("subject_id").sort_values(["group", "subject_id"]).reset_index(drop=True)
    return combined


def load_adult_raw(metadata: pd.DataFrame, joint_specs: dict[str, dict[str, FeatureSpec]]) -> pd.DataFrame:
    """Load adult raw parquet rows with the columns required for waveform analysis."""
    subject_ids = metadata["subject_id"].tolist()
    joint_cols = sorted({spec.column for side_specs in joint_specs.values() for spec in side_specs.values()})
    columns = ["subject_id", "group", "speed", "time_ms", "file_name"] + joint_cols + list(HEEL_CONTACT_COLS.values())

    dataset = ds.dataset(PATH_RAW, format="parquet")
    filter_expr = pc.field("subject_id").isin(subject_ids)
    df = dataset.to_table(columns=columns, filter=filter_expr).to_pandas()

    if df.empty:
        raise ValueError("원시 parquet에서 대상 adult cohort 데이터를 읽지 못했습니다.")

    df["subject_id"] = df["subject_id"].replace(SUBJECT_ID_ALIASES)
    df["group"] = df["group"].replace(ADULT_RAW_GROUPS)
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")

    df = df[df["group"].isin(PRIMARY_GROUPS)].copy()
    df = df.merge(metadata, on=["subject_id", "group"], how="inner", suffixes=("", "_meta"))
    df = df.sort_values(["subject_id", "speed", "file_name", "time_ms"]).reset_index(drop=True)
    return df


def detect_heel_strikes(contact_signal: np.ndarray, min_gap: int = 30) -> list[int]:
    """Detect heel strike events from heel-contact rising edges."""
    signal = np.asarray(contact_signal, dtype=int)
    if signal.size < 2:
        return []

    rising = np.where((signal[1:] == 1) & (signal[:-1] == 0))[0] + 1
    kept: list[int] = []
    for idx in rising.tolist():
        if not kept or idx - kept[-1] >= min_gap:
            kept.append(int(idx))
    return kept


def split_trial_on_time_gaps(trial_df: pd.DataFrame, max_gap_ms: float = MAX_TIME_GAP_MS) -> list[pd.DataFrame]:
    """Split one trial into contiguous chunks using the time axis."""
    ordered = trial_df.sort_values("time_ms").reset_index(drop=True)
    if ordered.empty:
        return []

    time_ms = pd.to_numeric(ordered["time_ms"], errors="coerce")
    dt = time_ms.diff()
    split_points = np.where((dt > max_gap_ms) | dt.isna())[0].tolist()

    chunks: list[pd.DataFrame] = []
    start = 0
    for stop in split_points[1:] + [len(ordered)]:
        chunk = ordered.iloc[start:stop].reset_index(drop=True)
        if not chunk.empty:
            chunks.append(chunk)
        start = stop
    return chunks


def build_stride_segments(trial_df: pd.DataFrame, leg: str) -> list[tuple[int, int]]:
    """Split a trial into heel-strike-to-heel-strike strides for one leg."""
    heel_col = HEEL_CONTACT_COLS[leg]
    heel_signal = pd.to_numeric(trial_df[heel_col], errors="coerce").fillna(0).astype(int).to_numpy()
    heel_strikes = detect_heel_strikes(heel_signal, min_gap=30)

    segments: list[tuple[int, int]] = []
    if len(heel_strikes) < 2:
        return segments

    time_ms = pd.to_numeric(trial_df["time_ms"], errors="coerce").to_numpy()
    for start, stop in zip(heel_strikes[:-1], heel_strikes[1:]):
        frame_len = stop - start
        if frame_len < 30 or frame_len > 250:
            continue
        duration_ms = float(time_ms[stop] - time_ms[start]) if not np.isnan(time_ms[start]) and not np.isnan(time_ms[stop]) else np.nan
        if not np.isnan(duration_ms) and not (300.0 <= duration_ms <= 2500.0):
            continue
        segments.append((int(start), int(stop)))
    return segments


def select_stride_segments(segments: list[tuple[int, int]], variant: str) -> list[tuple[int, int]]:
    """Return the segment list for the primary or sensitivity analysis."""
    if variant == "full":
        return segments
    if variant == "midtrial":
        if len(segments) > 2:
            return segments[1:-1]
        return segments
    raise ValueError(f"알 수 없는 analysis_variant: {variant}")


def interpolate_stride(signal: np.ndarray, start: int, stop: int, n_points: int = 101) -> np.ndarray:
    """Interpolate one stride to a fixed number of points."""
    segment = np.asarray(signal[start : stop + 1], dtype=float)
    if len(segment) < 2:
        raise ValueError("stride segment 길이가 너무 짧습니다.")
    x_old = np.linspace(0.0, 1.0, len(segment))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, segment)


def waveform_rows_for_variant(
    raw_df: pd.DataFrame,
    joint_specs: dict[str, dict[str, FeatureSpec]],
    variant: str,
) -> pd.DataFrame:
    """Aggregate stride-normalized waveforms to subject-speed-side means."""
    rows: list[dict[str, object]] = []

    for (subject_id, group, speed, person_id, injured_leg, pseudo_injured_leg), subject_speed_df in raw_df.groupby(
        ["subject_id", "group", "speed", "person_id", "injured_leg", "pseudo_injured_leg"], sort=False
    ):
        contra_leg = "Left" if injured_leg == "Right" else "Right"
        trial_count = int(subject_speed_df["file_name"].nunique())

        for side_basis, actual_leg in (("injured", injured_leg), ("contralateral", contra_leg)):
            feature_waves: dict[str, list[np.ndarray]] = {feature_name: [] for feature_name in joint_specs[actual_leg]}
            stride_count = 0

            for _, trial_df in subject_speed_df.groupby("file_name", sort=False):
                for chunk_df in split_trial_on_time_gaps(trial_df, max_gap_ms=MAX_TIME_GAP_MS):
                    base_segments = build_stride_segments(chunk_df, actual_leg)
                    segments = select_stride_segments(base_segments, variant)
                    if not segments:
                        continue

                    for start, stop in segments:
                        stride_count += 1
                        for feature_name, spec in joint_specs[actual_leg].items():
                            signal = pd.to_numeric(chunk_df[spec.column], errors="coerce").to_numpy(dtype=float)
                            try:
                                feature_waves[feature_name].append(interpolate_stride(signal, start, stop))
                            except ValueError:
                                continue

            if stride_count == 0:
                continue

            for feature_name, waves in feature_waves.items():
                if not waves:
                    continue
                mean_wave = np.nanmean(np.vstack(waves), axis=0)
                row = {
                    "analysis_variant": variant,
                    "subject_id": subject_id,
                    "person_id": person_id,
                    "group": group,
                    "speed": speed,
                    "side_basis": side_basis,
                    "actual_leg": actual_leg,
                    "injured_leg": injured_leg,
                    "pseudo_injured_leg": bool(pseudo_injured_leg),
                    "feature": feature_name,
                    "trial_count": trial_count,
                    "stride_count": stride_count,
                }
                row.update({POINT_COLS[i]: float(mean_wave[i]) for i in range(len(POINT_COLS))})
                rows.append(row)

    return pd.DataFrame(rows)


def cluster_endpoints_to_pct(endpoints: tuple[float, float]) -> tuple[int, int]:
    """Convert spm1d cluster endpoints to integer gait-cycle percentages."""
    start = max(0, int(math.floor(float(endpoints[0]))))
    end = min(100, int(math.ceil(float(endpoints[1]))))
    return start, end


def compute_eta_squared(values: np.ndarray, groups: Iterable[str]) -> float:
    """Compute eta squared on cluster-mean values."""
    ser = pd.Series(values, dtype=float)
    grp = pd.Series(list(groups), dtype=str)
    df = pd.DataFrame({"value": ser, "group": grp}).dropna()
    if df["group"].nunique() < 2 or len(df) < 3:
        return np.nan

    grand_mean = df["value"].mean()
    ss_between = sum(len(g["value"]) * (g["value"].mean() - grand_mean) ** 2 for _, g in df.groupby("group"))
    ss_total = ((df["value"] - grand_mean) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def compute_hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Hedges' g for two groups."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa = np.nanvar(a, ddof=1)
    sb = np.nanvar(b, ddof=1)
    pooled = (((len(a) - 1) * sa) + ((len(b) - 1) * sb)) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return np.nan
    d = (np.nanmean(a) - np.nanmean(b)) / np.sqrt(pooled)
    correction = 1.0 - (3.0 / (4.0 * (len(a) + len(b)) - 9.0))
    return float(d * correction)


def extract_region_means(waveframe: pd.DataFrame, start_pct: int, end_pct: int) -> np.ndarray:
    """Extract cluster-mean waveform values for one DataFrame slice."""
    cols = [POINT_COLS[i] for i in range(start_pct, end_pct + 1)]
    return waveframe[cols].mean(axis=1).to_numpy(dtype=float)


def run_primary_spm(waveforms_df: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    """Run SPM ANOVA and post-hoc tests for one waveform table."""
    rows: list[dict[str, object]] = []

    for speed in PRIMARY_SPEEDS:
        for side_basis in ("injured", "contralateral"):
            for feature in sorted(waveforms_df["feature"].unique()):
                subset = waveforms_df[
                    (waveforms_df["speed"] == speed)
                    & (waveforms_df["side_basis"] == side_basis)
                    & (waveforms_df["feature"] == feature)
                    & (waveforms_df["group"].isin(PRIMARY_GROUPS))
                ].copy()

                if subset["group"].nunique() < 3:
                    continue

                group_counts = subset.groupby("group")["subject_id"].nunique().reindex(PRIMARY_GROUPS).fillna(0).astype(int)
                if (group_counts < 2).any():
                    continue

                Y = subset[POINT_COLS].to_numpy(dtype=float)
                group_codes = {name: idx for idx, name in enumerate(PRIMARY_GROUPS)}
                A = subset["group"].map(group_codes).to_numpy(dtype=int)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    anova = spm1d.stats.anova1(Y, A, equal_var=False)
                    anova_inf = anova.inference(alpha=PRIMARY_ALPHA)
                omnibus_p = float(anova_inf.p_set)

                if anova_inf.h0reject and anova_inf.clusters:
                    for cluster_index, cluster in enumerate(anova_inf.clusters):
                        start_pct, end_pct = cluster_endpoints_to_pct(cluster.endpoints)
                        region_values = extract_region_means(subset, start_pct, end_pct)
                        eta_sq = compute_eta_squared(region_values, subset["group"])
                        group_means = subset.assign(region_mean=region_values).groupby("group")["region_mean"].mean()
                        rows.append(
                            {
                                "analysis_variant": analysis_set,
                                "comparison_scope": "primary",
                                "test_type": "omnibus",
                                "speed": speed,
                                "side_basis": side_basis,
                                "feature": feature,
                                "group_a": "HA",
                                "group_b": "ACLD/ACLR",
                                "cluster_index": cluster_index,
                                "cluster_p": float(cluster.P),
                                "test_p_raw": omnibus_p,
                                "start_pct": start_pct,
                                "end_pct": end_pct,
                                "extent": float(cluster.extent),
                                "stat_max": float(np.max(np.abs(anova_inf.z[start_pct : end_pct + 1]))),
                                "effect_size": eta_sq,
                                "effect_metric": "eta_sq_cluster_mean",
                                "mean_diff": np.nan,
                                "ha_mean": float(group_means.get("HA", np.nan)),
                                "acld_mean": float(group_means.get("ACLD", np.nan)),
                                "aclr_mean": float(group_means.get("ACLR", np.nan)),
                                "n_HA": int(group_counts["HA"]),
                                "n_ACLD": int(group_counts["ACLD"]),
                                "n_ACLR": int(group_counts["ACLR"]),
                            }
                        )
                else:
                    rows.append(
                        {
                            "analysis_variant": analysis_set,
                            "comparison_scope": "primary",
                            "test_type": "omnibus",
                            "speed": speed,
                            "side_basis": side_basis,
                            "feature": feature,
                            "group_a": "HA",
                            "group_b": "ACLD/ACLR",
                            "cluster_index": np.nan,
                            "cluster_p": np.nan,
                            "test_p_raw": omnibus_p,
                            "start_pct": np.nan,
                            "end_pct": np.nan,
                            "extent": np.nan,
                            "stat_max": np.nan,
                            "effect_size": np.nan,
                            "effect_metric": "eta_sq_cluster_mean",
                            "mean_diff": np.nan,
                            "ha_mean": np.nan,
                            "acld_mean": np.nan,
                            "aclr_mean": np.nan,
                            "n_HA": int(group_counts["HA"]),
                            "n_ACLD": int(group_counts["ACLD"]),
                            "n_ACLR": int(group_counts["ACLR"]),
                        }
                    )

                if not anova_inf.h0reject:
                    continue

                for group_a, group_b in PAIRWISE_COMPARISONS:
                    pair = subset[subset["group"].isin([group_a, group_b])].copy()
                    YA = pair[pair["group"] == group_a][POINT_COLS].to_numpy(dtype=float)
                    YB = pair[pair["group"] == group_b][POINT_COLS].to_numpy(dtype=float)
                    if len(YA) < 2 or len(YB) < 2:
                        continue

                    ttest = spm1d.stats.ttest2(YA, YB, equal_var=False)
                    t_inf = ttest.inference(alpha=PRIMARY_ALPHA, two_tailed=True)
                    pair_p = float(t_inf.p_set)

                    if t_inf.h0reject and t_inf.clusters:
                        for cluster_index, cluster in enumerate(t_inf.clusters):
                            start_pct, end_pct = cluster_endpoints_to_pct(cluster.endpoints)
                            region_a = extract_region_means(pair[pair["group"] == group_a], start_pct, end_pct)
                            region_b = extract_region_means(pair[pair["group"] == group_b], start_pct, end_pct)
                            mean_diff = float(np.nanmean(region_a) - np.nanmean(region_b))
                            effect_size = compute_hedges_g(region_a, region_b)
                            rows.append(
                                {
                                    "analysis_variant": analysis_set,
                                    "comparison_scope": "primary",
                                    "test_type": "posthoc",
                                    "speed": speed,
                                    "side_basis": side_basis,
                                    "feature": feature,
                                    "group_a": group_a,
                                    "group_b": group_b,
                                    "cluster_index": cluster_index,
                                    "cluster_p": float(cluster.P),
                                    "test_p_raw": pair_p,
                                    "start_pct": start_pct,
                                    "end_pct": end_pct,
                                    "extent": float(cluster.extent),
                                    "stat_max": float(np.max(np.abs(t_inf.z[start_pct : end_pct + 1]))),
                                    "effect_size": effect_size,
                                    "effect_metric": "hedges_g_cluster_mean",
                                    "mean_diff": mean_diff,
                                    "ha_mean": np.nan,
                                    "acld_mean": np.nan,
                                    "aclr_mean": np.nan,
                                    "n_HA": int(group_counts["HA"]),
                                    "n_ACLD": int(group_counts["ACLD"]),
                                    "n_ACLR": int(group_counts["ACLR"]),
                                }
                            )
                    else:
                        rows.append(
                            {
                                "analysis_variant": analysis_set,
                                "comparison_scope": "primary",
                                "test_type": "posthoc",
                                "speed": speed,
                                "side_basis": side_basis,
                                "feature": feature,
                                "group_a": group_a,
                                "group_b": group_b,
                                "cluster_index": np.nan,
                                "cluster_p": np.nan,
                                "test_p_raw": pair_p,
                                "start_pct": np.nan,
                                "end_pct": np.nan,
                                "extent": np.nan,
                                "stat_max": np.nan,
                                "effect_size": np.nan,
                                "effect_metric": "hedges_g_cluster_mean",
                                "mean_diff": np.nan,
                                "ha_mean": np.nan,
                                "acld_mean": np.nan,
                                "aclr_mean": np.nan,
                                "n_HA": int(group_counts["HA"]),
                                "n_ACLD": int(group_counts["ACLD"]),
                                "n_ACLR": int(group_counts["ACLR"]),
                            }
                        )

    result = pd.DataFrame(rows)
    return result


def run_paired_spm(waveforms_df: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    """Run paired ACLD vs ACLR SPM on matched ACL participants."""
    rows: list[dict[str, object]] = []
    acl_wave = waveforms_df[waveforms_df["group"].isin(["ACLD", "ACLR"])].copy()

    for speed in PRIMARY_SPEEDS:
        for side_basis in ("injured", "contralateral"):
            for feature in sorted(acl_wave["feature"].unique()):
                subset = acl_wave[(acl_wave["speed"] == speed) & (acl_wave["side_basis"] == side_basis) & (acl_wave["feature"] == feature)].copy()
                if subset.empty:
                    continue

                person_counts = subset.groupby(["person_id", "group"]).size().unstack(fill_value=0)
                matched_ids = person_counts[(person_counts.get("ACLD", 0) > 0) & (person_counts.get("ACLR", 0) > 0)].index.tolist()
                if len(matched_ids) < 3:
                    continue

                YA = (
                    subset[(subset["person_id"].isin(matched_ids)) & (subset["group"] == "ACLD")]
                    .sort_values("person_id")[POINT_COLS]
                    .to_numpy(dtype=float)
                )
                YB = (
                    subset[(subset["person_id"].isin(matched_ids)) & (subset["group"] == "ACLR")]
                    .sort_values("person_id")[POINT_COLS]
                    .to_numpy(dtype=float)
                )
                if len(YA) != len(YB) or len(YA) < 3:
                    continue

                ttest = spm1d.stats.ttest_paired(YA, YB)
                t_inf = ttest.inference(alpha=PRIMARY_ALPHA, two_tailed=True)
                paired_p = float(t_inf.p_set)

                if t_inf.h0reject and t_inf.clusters:
                    for cluster_index, cluster in enumerate(t_inf.clusters):
                        start_pct, end_pct = cluster_endpoints_to_pct(cluster.endpoints)
                        region_a = YA[:, start_pct : end_pct + 1].mean(axis=1)
                        region_b = YB[:, start_pct : end_pct + 1].mean(axis=1)
                        diff = region_b - region_a
                        rows.append(
                            {
                                "analysis_variant": analysis_set,
                                "comparison_scope": "paired_acl",
                                "test_type": "paired_posthoc",
                                "speed": speed,
                                "side_basis": side_basis,
                                "feature": feature,
                                "group_a": "ACLD",
                                "group_b": "ACLR",
                                "cluster_index": cluster_index,
                                "cluster_p": float(cluster.P),
                                "test_p_raw": paired_p,
                                "start_pct": start_pct,
                                "end_pct": end_pct,
                                "extent": float(cluster.extent),
                                "stat_max": float(np.max(np.abs(t_inf.z[start_pct : end_pct + 1]))),
                                "effect_size": float(np.nanmean(diff) / np.nanstd(diff, ddof=1)) if np.nanstd(diff, ddof=1) > 0 else np.nan,
                                "effect_metric": "paired_dz_cluster_mean",
                                "mean_diff": float(np.nanmean(region_b) - np.nanmean(region_a)),
                                "ha_mean": np.nan,
                                "acld_mean": float(np.nanmean(region_a)),
                                "aclr_mean": float(np.nanmean(region_b)),
                                "n_HA": 0,
                                "n_ACLD": int(len(YA)),
                                "n_ACLR": int(len(YB)),
                            }
                        )
                else:
                    rows.append(
                        {
                            "analysis_variant": analysis_set,
                            "comparison_scope": "paired_acl",
                            "test_type": "paired_posthoc",
                            "speed": speed,
                            "side_basis": side_basis,
                            "feature": feature,
                            "group_a": "ACLD",
                            "group_b": "ACLR",
                            "cluster_index": np.nan,
                            "cluster_p": np.nan,
                            "test_p_raw": paired_p,
                            "start_pct": np.nan,
                            "end_pct": np.nan,
                            "extent": np.nan,
                            "stat_max": np.nan,
                            "effect_size": np.nan,
                            "effect_metric": "paired_dz_cluster_mean",
                            "mean_diff": np.nan,
                            "ha_mean": np.nan,
                            "acld_mean": np.nan,
                            "aclr_mean": np.nan,
                            "n_HA": 0,
                            "n_ACLD": int(len(YA)),
                            "n_ACLR": int(len(YB)),
                        }
                    )

    return pd.DataFrame(rows)


def apply_fdr_to_spm(spm_results: pd.DataFrame) -> pd.DataFrame:
    """Apply FDR separately for omnibus, posthoc, and paired tests."""
    result = spm_results.copy()

    test_keys = [
        "analysis_variant",
        "comparison_scope",
        "test_type",
        "speed",
        "side_basis",
        "feature",
        "group_a",
        "group_b",
    ]

    correction_tables: list[pd.DataFrame] = []
    for key, group in result.groupby(["analysis_variant", "comparison_scope", "test_type"], sort=False):
        tests = group[test_keys + ["test_p_raw"]].drop_duplicates(subset=test_keys).copy()
        raw = tests["test_p_raw"].fillna(1.0).to_numpy(dtype=float)
        reject, p_fdr, _, _ = multipletests(raw, method="fdr_bh")
        tests["test_p_fdr"] = p_fdr
        tests["significant_fdr"] = reject
        correction_tables.append(tests[test_keys + ["test_p_fdr", "significant_fdr"]])

    corrected = (
        pd.concat(correction_tables, ignore_index=True) if correction_tables else pd.DataFrame(columns=test_keys + ["test_p_fdr", "significant_fdr"])
    )
    result = result.merge(corrected, on=test_keys, how="left")
    return result


def region_masks_from_spm(spm_results: pd.DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    """Build speed/side/feature union masks from significant primary SPM clusters."""
    masks: dict[tuple[str, str, str], np.ndarray] = {}
    sig = spm_results[
        (spm_results["analysis_variant"] == "full")
        & (spm_results["comparison_scope"] == "primary")
        & (spm_results["significant_fdr"])
        & spm_results["start_pct"].notna()
    ]

    for (speed, side_basis, feature), group in sig.groupby(["speed", "side_basis", "feature"]):
        mask = np.zeros(101, dtype=bool)
        for _, row in group.iterrows():
            start = int(row["start_pct"])
            end = int(row["end_pct"])
            mask[start : end + 1] = True
        masks[(speed, side_basis, feature)] = mask
    return masks


def summarize_waveform(wave: np.ndarray) -> dict[str, float]:
    """Compute standard waveform summary metrics."""
    arr = np.asarray(wave, dtype=float)
    peak_max_idx = int(np.nanargmax(arr))
    peak_min_idx = int(np.nanargmin(arr))
    summary = {
        "peak_max": float(np.nanmax(arr)),
        "peak_min": float(np.nanmin(arr)),
        "rom": float(np.nanmax(arr) - np.nanmin(arr)),
        "peak_max_pct": float(peak_max_idx),
        "peak_min_pct": float(peak_min_idx),
        "waveform_mean": float(np.nanmean(arr)),
    }
    for phase_name, (start, end) in GAIT_PHASES.items():
        summary[f"{phase_name}_mean"] = float(np.nanmean(arr[start : end + 1]))
    return summary


def build_feature_table(
    waveforms_df: pd.DataFrame,
    metadata: pd.DataFrame,
    spm_results: pd.DataFrame,
) -> pd.DataFrame:
    """Convert waveform rows into a wide subject-speed feature table."""
    primary_waves = waveforms_df[waveforms_df["analysis_variant"] == "full"].copy()
    region_masks = region_masks_from_spm(spm_results)

    base_rows: dict[tuple[str, str], dict[str, object]] = {}
    for (subject_id, speed), group in primary_waves.groupby(["subject_id", "speed"], sort=False):
        meta = metadata[metadata["subject_id"] == subject_id].iloc[0]
        row = {
            "subject_id": subject_id,
            "person_id": meta["person_id"],
            "group": meta["group"],
            "speed": speed,
            "sex": meta["sex"],
            "age": meta["age"],
            "weight": meta["weight"],
            "height": meta["height"],
            "injured_leg": meta["injured_leg"],
            "pseudo_injured_leg": bool(meta["pseudo_injured_leg"]),
        }

        for _, wf_row in group.iterrows():
            feature = wf_row["feature"]
            side = wf_row["side_basis"]
            wave = wf_row[POINT_COLS].to_numpy(dtype=float)
            summary = summarize_waveform(wave)
            for metric, value in summary.items():
                row[f"{feature}_{side}_{metric}"] = value

            mask = region_masks.get((speed, side, feature))
            if mask is not None and mask.any():
                row[f"{feature}_{side}_spm_region_mean"] = float(np.nanmean(wave[mask]))
            else:
                row[f"{feature}_{side}_spm_region_mean"] = np.nan

        base_rows[(subject_id, speed)] = row

    feature_df = pd.DataFrame(base_rows.values())

    metrics_for_diff = [
        "peak_max",
        "peak_min",
        "rom",
        "peak_max_pct",
        "peak_min_pct",
        "waveform_mean",
        "loading_response_mean",
        "mid_stance_mean",
        "terminal_stance_mean",
        "swing_mean",
        "spm_region_mean",
    ]
    metrics_for_lsi = [
        "peak_max",
        "peak_min",
        "rom",
        "waveform_mean",
        "loading_response_mean",
        "mid_stance_mean",
        "terminal_stance_mean",
        "swing_mean",
        "spm_region_mean",
    ]

    all_features = sorted({row["feature"] for _, row in primary_waves.iterrows()})
    derived_cols: dict[str, pd.Series] = {}
    for feature in all_features:
        for metric in metrics_for_diff:
            injured_col = f"{feature}_injured_{metric}"
            contra_col = f"{feature}_contralateral_{metric}"
            if injured_col in feature_df.columns and contra_col in feature_df.columns:
                derived_cols[f"{feature}_diff_{metric}"] = feature_df[injured_col] - feature_df[contra_col]
        for metric in metrics_for_lsi:
            injured_col = f"{feature}_injured_{metric}"
            contra_col = f"{feature}_contralateral_{metric}"
            if injured_col in feature_df.columns and contra_col in feature_df.columns:
                denom = feature_df[contra_col].replace(0, np.nan)
                derived_cols[f"{feature}_lsi_{metric}"] = 100.0 * (feature_df[injured_col] / denom)

    if derived_cols:
        feature_df = pd.concat([feature_df, pd.DataFrame(derived_cols, index=feature_df.index)], axis=1)

    return feature_df.sort_values(["group", "subject_id", "speed"]).reset_index(drop=True)


def candidate_feature_columns(feature_df: pd.DataFrame, spm_results: pd.DataFrame) -> list[str]:
    """Select candidate wide features from significant waveform findings."""
    sig = spm_results[(spm_results["analysis_variant"] == "full") & (spm_results["comparison_scope"] == "primary") & (spm_results["significant_fdr"])]
    base_features = sorted(sig["feature"].dropna().unique().tolist())
    if not base_features:
        base_features = sorted({col.split("_injured_")[0] for col in feature_df.columns if "_injured_" in col})

    candidates: list[str] = []
    for base in base_features:
        for col in feature_df.columns:
            if (
                col.startswith(f"{base}_injured_")
                or col.startswith(f"{base}_contralateral_")
                or col.startswith(f"{base}_diff_")
                or col.startswith(f"{base}_lsi_")
            ):
                candidates.append(col)
    return sorted(dict.fromkeys(candidates))


def fit_one_lmm(data: pd.DataFrame, feature_name: str, grouping_col: str = "person_id") -> tuple[dict[str, object], dict[str, object] | None]:
    """Fit one LMM with and without covariates as needed."""
    required = ["group", "speed", grouping_col, feature_name]
    sub = data[required + ["age", "sex"]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=["group", "speed", grouping_col, feature_name])

    if len(sub) < 12 or sub["group"].nunique() < 2:
        return {"feature": feature_name, "model_status": "insufficient_data"}, None

    sub["y"] = pd.to_numeric(sub[feature_name], errors="coerce")
    sub = sub.dropna(subset=["y"])
    if len(sub) < 12:
        return {"feature": feature_name, "model_status": "insufficient_data"}, None

    formulas = [
        ("with_covariates", "y ~ C(group) * C(speed) + age + C(sex)"),
        ("no_covariates", "y ~ C(group) * C(speed)"),
    ]

    fit_result = None
    fit_label = None
    for label, formula in formulas:
        tmp = sub.copy()
        needed = ["y", "group", "speed", grouping_col]
        if "age" in formula:
            needed += ["age", "sex"]
        tmp = tmp.dropna(subset=needed)
        if len(tmp) < 12 or tmp["group"].nunique() < 2 or tmp["speed"].nunique() < 2:
            continue
        try:
            model = mixedlm(formula, tmp, groups=tmp[grouping_col])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message="Random effects covariance is singular")
                fitted = model.fit(reml=False, method="lbfgs", disp=False)
            fit_result = (tmp, fitted)
            fit_label = label
            break
        except Exception as exc:
            fit_result = (tmp, exc)
            fit_label = f"{label}_failed"

    if fit_result is None:
        return {"feature": feature_name, "model_status": "fit_failed"}, None

    tmp, fitted = fit_result
    if isinstance(fitted, Exception):
        return {"feature": feature_name, "model_status": str(fit_label), "error": str(fitted)}, None

    pvalues = fitted.pvalues
    group_terms = [term for term in pvalues.index if "C(group)" in term and ":" not in term]
    interaction_terms = [term for term in pvalues.index if "C(group)" in term and "C(speed)" in term]

    result = {
        "feature": feature_name,
        "model_status": "ok",
        "model_variant": fit_label,
        "n_obs": int(len(tmp)),
        "n_persons": int(tmp[grouping_col].nunique()),
        "p_group_main": float(np.min([pvalues[t] for t in group_terms])) if group_terms else np.nan,
        "p_interaction": float(np.min([pvalues[t] for t in interaction_terms])) if interaction_terms else np.nan,
        "coef_group_terms": "; ".join(f"{term}={fitted.params[term]:.6f}" for term in group_terms),
        "coef_interaction_terms": "; ".join(f"{term}={fitted.params[term]:.6f}" for term in interaction_terms),
        "mean_HA": float(tmp[tmp["group"] == "HA"]["y"].mean()) if "HA" in tmp["group"].unique() else np.nan,
        "mean_ACLD": float(tmp[tmp["group"] == "ACLD"]["y"].mean()) if "ACLD" in tmp["group"].unique() else np.nan,
        "mean_ACLR": float(tmp[tmp["group"] == "ACLR"]["y"].mean()) if "ACLR" in tmp["group"].unique() else np.nan,
    }
    return result, {"data": tmp, "fit": fitted}


def run_lmm_suite(feature_df: pd.DataFrame, features: list[str], subset_name: str, groups: list[str]) -> pd.DataFrame:
    """Run LMMs for a list of wide features."""
    rows: list[dict[str, object]] = []
    sub_df = feature_df[feature_df["group"].isin(groups)].copy()
    for feature_name in features:
        result, _ = fit_one_lmm(sub_df, feature_name, grouping_col="person_id")
        result["analysis_subset"] = subset_name
        rows.append(result)

    out = pd.DataFrame(rows)
    for pcol in ("p_group_main", "p_interaction"):
        if pcol not in out.columns:
            continue
        valid = out[pcol].notna()
        fdr = np.full(len(out), np.nan)
        sig = np.zeros(len(out), dtype=bool)
        if valid.any():
            reject, p_fdr, _, _ = multipletests(out.loc[valid, pcol].to_numpy(dtype=float), method="fdr_bh")
            fdr[np.where(valid)[0]] = p_fdr
            sig[np.where(valid)[0]] = reject
        out[f"{pcol}_fdr"] = fdr
        out[f"{pcol}_sig_fdr"] = sig
    return out


def run_feature_ranking(feature_df: pd.DataFrame, lmm_results: pd.DataFrame, spm_results: pd.DataFrame) -> pd.DataFrame:
    """Rank explanatory features with grouped multinomial elastic-net."""
    candidate = (
        lmm_results[
            (lmm_results["analysis_subset"] == "primary")
            & (lmm_results["p_group_main_sig_fdr"].fillna(False) | lmm_results["p_interaction_sig_fdr"].fillna(False))
        ]["feature"]
        .dropna()
        .tolist()
    )

    if not candidate:
        candidate = candidate_feature_columns(feature_df, spm_results)[:20]

    model_df = feature_df[["subject_id", "person_id", "group"] + candidate].copy()
    model_df = model_df.dropna(subset=["group"])
    if len(candidate) == 0 or model_df["group"].nunique() < 3:
        return pd.DataFrame(columns=["feature", "importance", "rank", "best_params", "cv_macro_f1", "cv_accuracy"])

    X = model_df[candidate]
    y = model_df["group"].to_numpy()
    groups = model_df["person_id"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 3:
        return pd.DataFrame(columns=["feature", "importance", "rank", "best_params", "cv_macro_f1", "cv_accuracy"])

    grid = list(ParameterGrid({"clf__C": [0.05, 0.1, 0.5, 1.0, 2.0], "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}))
    cv = GroupKFold(n_splits=n_splits)
    best = None

    for params in grid:
        fold_f1: list[float] = []
        fold_acc: list[float] = []
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            max_iter=10000,
                            random_state=42,
                        ),
                    ),
                ]
            )
            pipe.set_params(**params)
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred = pipe.predict(X.iloc[test_idx])
            fold_f1.append(f1_score(y[test_idx], pred, average="macro"))
            fold_acc.append(accuracy_score(y[test_idx], pred))

        summary = {
            "params": params,
            "macro_f1": float(np.mean(fold_f1)),
            "accuracy": float(np.mean(fold_acc)),
        }
        if best is None or summary["macro_f1"] > best["macro_f1"]:
            best = summary

    final_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )
    final_pipe.set_params(**best["params"])
    final_pipe.fit(X, y)
    clf = final_pipe.named_steps["clf"]
    importance = np.abs(clf.coef_).mean(axis=0)

    ranking = (
        pd.DataFrame(
            {
                "feature": candidate,
                "importance": importance,
                "rank": pd.Series(importance).rank(ascending=False, method="dense").astype(int),
                "best_params": str(best["params"]),
                "cv_macro_f1": best["macro_f1"],
                "cv_accuracy": best["accuracy"],
            }
        )
        .sort_values(["rank", "feature"])
        .reset_index(drop=True)
    )
    return ranking


def compare_mapping_with_xlsx(sample_xlsx: Path, joint_specs: dict[str, dict[str, FeatureSpec]]) -> list[dict[str, object]]:
    """Validate that raw jointAngle ordering matches anatomical XLSX labels."""
    xls = pd.ExcelFile(sample_xlsx)
    cols = xls.parse("Joint Angles ZXY", nrows=1).columns.tolist()[1:]
    rows: list[dict[str, object]] = []
    for side in ("Right", "Left"):
        for feature_name, spec in joint_specs[side].items():
            raw_index = int(spec.column.split("_")[1])
            xlsx_label = cols[raw_index]
            rows.append(
                {
                    "check": "joint_mapping",
                    "item": f"{spec.column}:{feature_name}",
                    "expected": spec.anatomical_label,
                    "observed": xlsx_label,
                    "passed": xlsx_label == spec.anatomical_label,
                }
            )
    return rows


def build_validation_summary(
    waveforms_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    joint_specs: dict[str, dict[str, FeatureSpec]],
    sample_xlsx: Path,
) -> pd.DataFrame:
    """Generate validation checks requested in the plan."""
    rows = compare_mapping_with_xlsx(sample_xlsx, joint_specs)

    sagittal_features = ["hip_flexion", "knee_flexion", "ankle_dorsiflexion"]
    for feature in sagittal_features:
        rom_cols = [f"{feature}_{side}_rom" for side in ("injured", "contralateral") if f"{feature}_{side}_rom" in feature_df.columns]
        if not rom_cols:
            continue
        rom_df = feature_df[["group", "speed"] + rom_cols].copy()
        rom_df["rom_mean"] = rom_df[rom_cols].mean(axis=1)
        speed_means = rom_df.groupby("speed")["rom_mean"].mean().reindex(PRIMARY_SPEEDS)
        monotonic = bool(speed_means.is_monotonic_increasing)
        rows.append(
            {
                "check": "speed_rom_monotonic",
                "item": feature,
                "expected": "slow < normal < fast",
                "observed": " | ".join(f"{speed}:{speed_means[speed]:.4f}" for speed in PRIMARY_SPEEDS),
                "passed": monotonic,
            }
        )

    for speed in ("normal", "fast"):
        for feature in sagittal_features:
            col = f"{feature}_injured_rom"
            if col not in feature_df.columns:
                continue
            means = feature_df[feature_df["speed"] == speed].groupby("group")[col].mean()
            passed = bool(
                ("HA" in means.index)
                and (("ACLD" not in means.index) or means["HA"] >= means.get("ACLD", -np.inf))
                and (("ACLR" not in means.index) or means["HA"] >= means.get("ACLR", -np.inf))
            )
            rows.append(
                {
                    "check": "ha_vs_acl_sagittal_rom",
                    "item": f"{feature}:{speed}",
                    "expected": "HA >= ACLD/ACLR",
                    "observed": " | ".join(f"{grp}:{means.get(grp, np.nan):.4f}" for grp in PRIMARY_GROUPS),
                    "passed": passed,
                }
            )

    sign_cols = ["hip_flexion_injured_peak_max", "knee_flexion_injured_peak_max", "ankle_dorsiflexion_injured_peak_max"]
    for col in sign_cols:
        if col not in feature_df.columns:
            continue
        passed = bool(feature_df[col].mean() > 0)
        rows.append(
            {
                "check": "sign_convention_positive_primary_peak",
                "item": col,
                "expected": "positive mean peak",
                "observed": f"{feature_df[col].mean():.4f}",
                "passed": passed,
            }
        )

    return pd.DataFrame(rows)


def build_sensitivity_comparison(spm_results: pd.DataFrame) -> pd.DataFrame:
    """Compare full-trial and mid-trial omnibus decisions."""
    full = (
        spm_results[
            (spm_results["analysis_variant"] == "full") & (spm_results["comparison_scope"] == "primary") & (spm_results["test_type"] == "omnibus")
        ][["speed", "side_basis", "feature", "test_p_raw", "test_p_fdr", "significant_fdr"]]
        .drop_duplicates(subset=["speed", "side_basis", "feature"])
        .rename(
            columns={
                "test_p_raw": "full_p_raw",
                "test_p_fdr": "full_p_fdr",
                "significant_fdr": "full_sig_fdr",
            }
        )
    )
    mid = (
        spm_results[
            (spm_results["analysis_variant"] == "midtrial") & (spm_results["comparison_scope"] == "primary") & (spm_results["test_type"] == "omnibus")
        ][["speed", "side_basis", "feature", "test_p_raw", "test_p_fdr", "significant_fdr"]]
        .drop_duplicates(subset=["speed", "side_basis", "feature"])
        .rename(
            columns={
                "test_p_raw": "mid_p_raw",
                "test_p_fdr": "mid_p_fdr",
                "significant_fdr": "mid_sig_fdr",
            }
        )
    )

    merged = full.merge(mid, on=["speed", "side_basis", "feature"], how="outer")
    merged["same_fdr_decision"] = merged["full_sig_fdr"].fillna(False) == merged["mid_sig_fdr"].fillna(False)
    return merged.sort_values(["feature", "side_basis", "speed"]).reset_index(drop=True)


def save_outputs(
    waveforms_df: pd.DataFrame,
    spm_results: pd.DataFrame,
    feature_df: pd.DataFrame,
    lmm_results: pd.DataFrame,
    ranking_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
) -> None:
    """Persist analysis outputs."""
    output_paths = [
        PATH_OUT_WAVEFORMS,
        PATH_OUT_SPM,
        PATH_OUT_FEATURES,
        PATH_OUT_LMM,
        PATH_OUT_RANKING,
        PATH_OUT_VALIDATION,
        PATH_OUT_SENS,
    ]
    for parent in {path.parent for path in output_paths}:
        parent.mkdir(parents=True, exist_ok=True)

    waveforms_df.to_csv(PATH_OUT_WAVEFORMS, index=False, encoding="utf-8-sig")
    spm_results.to_csv(PATH_OUT_SPM, index=False, encoding="utf-8-sig")
    feature_df.to_csv(PATH_OUT_FEATURES, index=False, encoding="utf-8-sig")
    lmm_results.to_csv(PATH_OUT_LMM, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(PATH_OUT_RANKING, index=False, encoding="utf-8-sig")
    validation_df.to_csv(PATH_OUT_VALIDATION, index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(PATH_OUT_SENS, index=False, encoding="utf-8-sig")


def run_waveform_group_analysis() -> dict[str, pd.DataFrame]:
    """Run the full waveform-first analysis."""
    log.info("▶ waveform group analysis 시작")
    sample_mvnx, sample_xlsx = find_sample_files()
    metadata = load_metadata()
    joint_specs = build_joint_feature_specs(sample_mvnx)
    raw_df = load_adult_raw(metadata, joint_specs)

    wave_full = waveform_rows_for_variant(raw_df, joint_specs, variant="full")
    wave_mid = waveform_rows_for_variant(raw_df, joint_specs, variant="midtrial")
    waveforms_df = pd.concat([wave_full, wave_mid], ignore_index=True)

    primary_spm_full = run_primary_spm(wave_full, analysis_set="full")
    primary_spm_mid = run_primary_spm(wave_mid, analysis_set="midtrial")
    paired_spm = run_paired_spm(wave_full, analysis_set="full")
    spm_results = apply_fdr_to_spm(pd.concat([primary_spm_full, primary_spm_mid, paired_spm], ignore_index=True))

    feature_df = build_feature_table(waveforms_df, metadata, spm_results)
    primary_candidates = candidate_feature_columns(feature_df, spm_results)
    lmm_primary = run_lmm_suite(feature_df, primary_candidates, subset_name="primary", groups=PRIMARY_GROUPS)

    matched_acl_ids = feature_df[feature_df["group"].isin(["ACLD", "ACLR"])].groupby("person_id")["group"].nunique()
    matched_person_ids = matched_acl_ids[matched_acl_ids == 2].index.tolist()
    acl_paired_df = feature_df[(feature_df["group"].isin(["ACLD", "ACLR"])) & (feature_df["person_id"].isin(matched_person_ids))].copy()
    lmm_paired = run_lmm_suite(acl_paired_df, primary_candidates, subset_name="paired_acl", groups=["ACLD", "ACLR"])
    lmm_results = pd.concat([lmm_primary, lmm_paired], ignore_index=True)

    ranking_df = run_feature_ranking(feature_df, lmm_results, spm_results)
    validation_df = build_validation_summary(waveforms_df, feature_df, joint_specs, sample_xlsx)
    sensitivity_df = build_sensitivity_comparison(spm_results)

    save_outputs(
        waveforms_df=waveforms_df,
        spm_results=spm_results,
        feature_df=feature_df,
        lmm_results=lmm_results,
        ranking_df=ranking_df,
        validation_df=validation_df,
        sensitivity_df=sensitivity_df,
    )

    log.info(f"✅ waveform rows: {len(waveforms_df)} -> {PATH_OUT_WAVEFORMS}")
    log.info(f"✅ spm results: {len(spm_results)} -> {PATH_OUT_SPM}")
    log.info(f"✅ feature rows: {len(feature_df)} -> {PATH_OUT_FEATURES}")
    log.info(f"✅ lmm rows: {len(lmm_results)} -> {PATH_OUT_LMM}")
    log.info(f"✅ ranking rows: {len(ranking_df)} -> {PATH_OUT_RANKING}")
    log.info(f"✅ validation rows: {len(validation_df)} -> {PATH_OUT_VALIDATION}")
    log.info(f"✅ sensitivity rows: {len(sensitivity_df)} -> {PATH_OUT_SENS}")

    return {
        "waveforms": waveforms_df,
        "spm": spm_results,
        "features": feature_df,
        "lmm": lmm_results,
        "ranking": ranking_df,
        "validation": validation_df,
        "sensitivity": sensitivity_df,
    }


if __name__ == "__main__":
    run_waveform_group_analysis()
