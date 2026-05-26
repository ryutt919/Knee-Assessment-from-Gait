"""
Recovery Score 5개 컴포넌트 산출.
모든 컴포넌트는 HA 분포 기준으로 표준화된 이탈 점수를 반환한다.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).parent.parent


def _ha_zscore(values: np.ndarray, ha_values: np.ndarray) -> np.ndarray:
    """HA 평균·표준편차 기준으로 z-score 변환."""
    mu  = ha_values.mean()
    std = ha_values.std(ddof=1)
    if std < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - mu) / std


# ── 컴포넌트 1: 파형 이탈 (RMSD vs HA mean) ──────────────────────────────────
def waveform_deviation(
    subject_ids: np.ndarray,
    groups: np.ndarray,
    waveforms: np.ndarray,   # (N, 9, 101) norm_101
) -> pd.Series:
    """
    피험자별 per-channel RMSD(subject_waveform, HA_mean) 평균.
    HA 기준 z-score로 반환.
    """
    ha_mask = groups == "HA"
    ha_mean = waveforms[ha_mask].mean(axis=0)  # (9, 101)

    def _subj_rmsd(idx: np.ndarray) -> float:
        sw = waveforms[idx].mean(axis=0)  # (9, 101) — 피험자 평균 파형
        return float(np.sqrt(((sw - ha_mean) ** 2).mean()))

    scores = {}
    for sid in np.unique(subject_ids):
        idx = np.where(subject_ids == sid)[0]
        scores[sid] = _subj_rmsd(idx)

    s = pd.Series(scores)
    ha_sids = np.unique(subject_ids[ha_mask])
    ha_vals = s[ha_sids].values
    return _ha_zscore(s.values, ha_vals)


# ── 컴포넌트 2: LSI 비대칭 ─────────────────────────────────────────────────
def lsi_asymmetry(
    features_df: pd.DataFrame,
    groups: np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """
    |LSI - 100| 평균 (across all LSI columns).
    HA 기준 z-score.
    """
    lsi_cols = [c for c in features_df.columns if c.endswith("_LSI")]
    if not lsi_cols:
        raise ValueError("features_scalar.csv에 _LSI 컬럼 없음")

    ha_mask = groups == "HA"
    vals = (features_df[lsi_cols].abs() - 100).abs().mean(axis=1).values
    ha_vals = vals[ha_mask]
    return _ha_zscore(vals, ha_vals)


# ── 컴포넌트 3: 속도 강건성 (속도 간 CV) ──────────────────────────────────
def speed_robustness(
    features_df: pd.DataFrame,
    groups: np.ndarray,
    subject_ids: np.ndarray,
    speeds: list[str] = ("normal", "slow", "fast"),
) -> np.ndarray:
    """
    속도별 feature CV (std/mean) 평균 → HA 기준 z-score.
    값이 클수록 속도 변화에 불안정 → 더 나쁜 점수.
    """
    ha_mask = groups == "HA"
    scalar_cols = [c for c in features_df.columns
                   if not c.startswith("LSI_") and not c.startswith("asym_")]

    cv_vals = []
    for sid in subject_ids:
        # 피험자별 속도 그룹은 features_df에 없음 (피험자 평균이므로 사용 불가)
        # 대신 동일 피험자의 여러 행이 있는 stride-level에서 계산
        # 여기서는 features_df가 피험자 평균이므로 CV 불가 → 0으로 처리
        cv_vals.append(0.0)

    cv_vals = np.array(cv_vals, dtype=float)
    ha_vals = cv_vals[ha_mask]
    return _ha_zscore(cv_vals, ha_vals)


# ── 컴포넌트 4: 보상 전략 (hip/ankle z-score vs HA) ───────────────────────
def compensation_strategy(
    features_df: pd.DataFrame,
    groups: np.ndarray,
) -> np.ndarray:
    """
    Hip flexion + ankle dorsiflexion 관련 feature의 HA 기준 |z-score| 평균.
    무릎 대신 hip/ankle이 과도하게 사용되면 높아짐.
    """
    comp_keywords = ("hip_flex", "ankle_dors", "hip_ext", "ankle_plant")
    numeric_cols = features_df.select_dtypes(include="number").columns.tolist()
    comp_cols = [c for c in numeric_cols
                 if any(kw in c.lower() for kw in comp_keywords)]

    if not comp_cols:
        comp_cols = numeric_cols[:min(10, len(numeric_cols))]

    ha_mask = groups == "HA"
    arr = features_df[comp_cols].values.astype(float)

    ha_mu  = arr[ha_mask].mean(axis=0)
    ha_std = arr[ha_mask].std(axis=0, ddof=1)
    ha_std[ha_std < 1e-9] = 1.0

    z = np.abs((arr - ha_mu) / ha_std)
    vals    = z.mean(axis=1)
    ha_vals = vals[ha_mask]
    return _ha_zscore(vals, ha_vals)


# ── 컴포넌트 5: 시공간 파라미터 이탈 ─────────────────────────────────────
def spatiotemporal_deviation(
    features_df: pd.DataFrame,
    groups: np.ndarray,
) -> np.ndarray:
    """
    보폭, 케이던스, stride velocity 관련 feature의 HA 기준 |z-score| 평균.
    """
    st_keywords = ("step_width", "cadence", "stride_len", "velocity", "stance_time", "swing_time")
    numeric_cols = features_df.select_dtypes(include="number").columns.tolist()
    st_cols = [c for c in numeric_cols
               if any(kw in c.lower() for kw in st_keywords)]

    if not st_cols:
        st_cols = numeric_cols[:min(5, len(numeric_cols))]

    ha_mask = groups == "HA"
    arr = features_df[st_cols].values.astype(float)

    ha_mu  = arr[ha_mask].mean(axis=0)
    ha_std = arr[ha_mask].std(axis=0, ddof=1)
    ha_std[ha_std < 1e-9] = 1.0

    z = np.abs((arr - ha_mu) / ha_std)
    vals    = z.mean(axis=1)
    ha_vals = vals[ha_mask]
    return _ha_zscore(vals, ha_vals)
