"""
파형 로더
- load_waveform_norm101: waveforms_stride.parquet → (N, 9ch, 101)
- load_waveform_cycle101: cycle_waveforms_101.parquet → (N, 9ch, 101)
- load_waveform_raw    : stride_raw_waveforms.parquet → (N, 9ch, max_len) + mask

waveform_norm flag (config.features.waveform_norm):
  zscore      — 채널별 z-score (train fold 통계 기준). fit_scaler/transform_scaler 사용
  ha_centered — HA mean/std 기준 표준화 (Recovery Score용)
  none        — 원시 각도(°)

반환: X (ndarray float32), mask (ndarray bool | None), y, groups
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

JOINTS = [
    "hip_adduction", "hip_int_rotation", "hip_flexion",
    "knee_adduction", "knee_int_rotation", "knee_flexion",
    "ankle_adduction", "ankle_int_rotation", "ankle_dorsiflexion",
]
SPEED_ORDER = ["normal", "slow", "fast"]


def _speed_onehot(speed_series: pd.Series) -> np.ndarray:
    dummies = pd.get_dummies(speed_series, prefix="speed")
    for sp in SPEED_ORDER:
        col = f"speed_{sp}"
        if col not in dummies.columns:
            dummies[col] = 0
    return dummies[[f"speed_{sp}" for sp in SPEED_ORDER]].values.astype(float)


def _apply_target(df: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    if cfg.targets.mode == "binary":
        return df["group"].map(cfg.targets.binary_map).values.astype(int)
    label_map = {"ACLD": 0, "ACLR": 1, "HA": 2}
    return df["group"].map(label_map).values.astype(int)


def load_waveform_norm101(cfg: DictConfig):
    """
    101pt 정규화 파형 로드.
    Returns: X (N, 9, 101, dtype=float32), speed_onehot (N,3), mask=None, y, groups
    """
    path = PROCESSED / cfg.data.waveform_norm_101
    df = pd.read_parquet(path)

    seqs = []
    for joint in JOINTS:
        cols = [f"{joint}_{t:03d}" for t in range(101)]
        seqs.append(df[cols].values.astype(np.float32))  # (N, 101)

    X = np.stack(seqs, axis=1)  # (N, 9, 101)
    y = _apply_target(df, cfg)
    groups = df["subject_id"].values
    speed_oh = _speed_onehot(df["speed"]) if cfg.features.speed_as_feature else None
    return X, speed_oh, None, y, groups


def load_waveform_cycle101(cfg: DictConfig):
    """
    Cycle-level 101pt 정규화 파형 로드.
    Returns: X (N, 9, 101, dtype=float32), speed_onehot (N,3), mask=None, y, groups
    """
    path = PROCESSED / cfg.data.get("cycle_waveforms_101", "cycle_waveforms_101.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — features/extract_cycle_waveforms_101.py를 먼저 실행하세요."
        )
    df = pd.read_parquet(path)

    seqs = []
    for joint in JOINTS:
        cols = [f"{joint}_{t:03d}" for t in range(101)]
        seqs.append(df[cols].values.astype(np.float32))

    X = np.stack(seqs, axis=1)
    y = _apply_target(df, cfg)
    groups = df["subject_id"].values
    speed_oh = _speed_onehot(df["speed"]) if cfg.features.speed_as_feature else None
    return X, speed_oh, None, y, groups


def load_waveform_raw(cfg: DictConfig):
    """
    Raw padded 파형 로드 (stride_raw_waveforms.parquet 필요).
    Returns: X (N, 9, max_len, float32), mask (N, 9, max_len, bool), speed_oh, y, groups
    """
    path = PROCESSED / cfg.data.raw_cycles
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — features/extract_raw_cycles.py를 먼저 실행하세요."
        )
    df = pd.read_parquet(path)

    # max_len 추론
    sample_joint = JOINTS[0]
    wf_cols = [c for c in df.columns if c.startswith(f"{sample_joint}_") and not c.startswith("mask_")]
    max_len = len(wf_cols)

    seqs, masks = [], []
    for joint in JOINTS:
        wf_c = [f"{joint}_{t:04d}" for t in range(max_len)]
        mk_c = [f"mask_{joint}_{t:04d}" for t in range(max_len)]
        seqs.append(df[wf_c].values.astype(np.float32))
        masks.append(df[mk_c].values.astype(bool))

    X    = np.stack(seqs, axis=1)   # (N, 9, max_len)
    mask = np.stack(masks, axis=1)  # (N, 9, max_len)
    y = _apply_target(df, cfg)
    groups = df["subject_id"].values
    speed_oh = _speed_onehot(df["speed"]) if cfg.features.speed_as_feature else None
    return X, speed_oh, mask, y, groups


def load_waveform(cfg: DictConfig):
    """config.features.waveform_type에 따라 분기."""
    if cfg.features.waveform_type == "norm_101":
        return load_waveform_norm101(cfg)
    elif cfg.features.waveform_type == "cycle_norm_101":
        return load_waveform_cycle101(cfg)
    elif cfg.features.waveform_type == "raw_padded":
        return load_waveform_raw(cfg)
    else:
        raise ValueError(f"알 수 없는 waveform_type: {cfg.features.waveform_type}")


class WaveformScaler:
    """채널별 waveform 표준화 (누수 방지: fit은 train fold에서만)."""

    def __init__(self, mode: str = "zscore"):
        self.mode = mode   # zscore | ha_centered | none
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, ha_mask: np.ndarray | None = None):
        """X: (N, 9, T)"""
        if self.mode == "none":
            return self
        if self.mode == "ha_centered":
            assert ha_mask is not None, "ha_centered는 ha_mask 필요 (y==0 mask)"
            src = X[ha_mask]
        else:
            src = X
        # 채널별(axis=1) 통계 → (1, 9, 1) broadcast 가능
        self.mean_ = src.mean(axis=(0, 2), keepdims=True).mean(axis=2, keepdims=True)
        std = src.std(axis=(0, 2), keepdims=True).mean(axis=2, keepdims=True)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "none" or self.mean_ is None:
            return X
        return (X - self.mean_) / self.std_
