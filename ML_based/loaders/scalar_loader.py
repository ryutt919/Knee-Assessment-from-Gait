"""
스칼라 피처 로더
- load_subject_scalar: features_scalar.csv (피험자 평균, 237행)
- load_stride_scalar : stride_level_peaks.parquet (stride 단위, ~14400행 trim 후)

Stride Trim: (subject_id, speed, trial_id) 그룹별 앞뒤 stride_trim개 제거
Speed one-hot: normal/slow/fast → [1,0,0], [0,1,0], [0,0,1]
η²-guided selection: feature_ranking.csv top-k feature만 사용
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

ROOT = Path(__file__).parent.parent.parent
ML   = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"

SPEED_ORDER = ["normal", "slow", "fast"]


def _speed_onehot(speed_series: pd.Series) -> pd.DataFrame:
    dummies = pd.get_dummies(speed_series, prefix="speed")
    for sp in SPEED_ORDER:
        col = f"speed_{sp}"
        if col not in dummies.columns:
            dummies[col] = 0
    return dummies[[f"speed_{sp}" for sp in SPEED_ORDER]].astype(float)


def _apply_target(df: pd.DataFrame, cfg: DictConfig) -> np.ndarray:
    if cfg.targets.mode == "binary":
        return df["group"].map(cfg.targets.binary_map).values.astype(int)
    else:
        label_map = {"ACLD": 0, "ACLR": 1, "HA": 2}
        return df["group"].map(label_map).values.astype(int)


def _eta2_feature_names(cfg: DictConfig) -> list[str] | None:
    sel = cfg.features.feature_select
    if sel == "all" or sel == "none":
        return None
    if not sel.startswith("eta2_top"):
        return None
    k = int(sel.replace("eta2_top", ""))
    ranking_path = PROCESSED / cfg.data.feature_ranking
    ranking = pd.read_csv(ranking_path)
    top_features = ranking.nlargest(k, "eta2")["feature"].tolist()
    return top_features


def load_subject_scalar(cfg: DictConfig):
    """
    Returns: X (ndarray), y (ndarray), groups (ndarray of subject_id strings)
    """
    path = PROCESSED / cfg.data.scalar_subject
    df = pd.read_csv(path)

    meta_cols = ["subject_id", "group", "speed", "injured_leg"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    top_feats = _eta2_feature_names(cfg)
    if top_feats:
        feat_cols = [c for c in feat_cols if c in top_feats]

    X_feats = df[feat_cols].values.astype(float)

    if cfg.features.speed_as_feature:
        speed_oh = _speed_onehot(df["speed"]).values
        X = np.hstack([X_feats, speed_oh])
    else:
        X = X_feats

    y = _apply_target(df, cfg)
    groups = df["subject_id"].values
    return X, y, groups


def _trim_strides(df: pd.DataFrame, stride_trim: int) -> pd.DataFrame:
    """trial(=file_name 또는 trial_id) 기준으로 앞뒤 stride_trim개 제거."""
    if stride_trim == 0:
        return df

    trial_col = "trial_id" if "trial_id" in df.columns else None

    if trial_col is None:
        # trial_id 없을 때: stride_idx만으로 trim (subject/speed 그룹 내)
        def trim_group(g):
            sorted_g = g.sort_values("stride_idx")
            n = len(sorted_g)
            if n <= stride_trim * 2:
                return sorted_g.iloc[0:0]
            return sorted_g.iloc[stride_trim: n - stride_trim]

        return df.groupby(["subject_id", "speed", "side"], group_keys=False).apply(trim_group)

    def trim_group(g):
        sorted_g = g.sort_values("stride_idx")
        n = len(sorted_g)
        if n <= stride_trim * 2:
            return sorted_g.iloc[0:0]
        return sorted_g.iloc[stride_trim: n - stride_trim]

    return df.groupby(["subject_id", "speed", "side", trial_col], group_keys=False).apply(
        trim_group
    )


def load_stride_scalar(cfg: DictConfig):
    """
    Returns: X (ndarray), y (ndarray), groups (ndarray of subject_id strings)
    """
    path = PROCESSED / cfg.data.scalar_stride
    df = pd.read_parquet(path)

    df = _trim_strides(df, cfg.features.stride_trim)
    print(f"[scalar_loader] stride trim 후: {len(df)}행")

    meta_cols = {"subject_id", "group", "speed", "side", "stride_idx", "trial_id"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    top_feats = _eta2_feature_names(cfg)
    if top_feats:
        base_top_feats = set()
        for f in top_feats:
            # _contralateral 포함: subject-level에서 파생된 모든 suffix를 strip
            for suffix in ["_LSI", "_injured", "_uninjured", "_asym", "_contralateral"]:
                if f.endswith(suffix):
                    f = f[:-len(suffix)]
                    break
            base_top_feats.add(f)
        feat_cols_set = set(feat_cols)
        selected = set()
        for base in base_top_feats:
            if base in feat_cols_set:
                selected.add(base)
            else:
                # _asym 등으로 strip 시 base가 컬럼에 없을 경우 prefix 매칭
                # e.g. 'knee_flexion' → 'knee_flexion_peak', 'knee_flexion_ROM', ...
                for c in feat_cols:
                    if c.startswith(base + "_"):
                        selected.add(c)
        feat_cols = [c for c in feat_cols if c in selected]
        print(f"[scalar_loader] eta2 feature select: {len(feat_cols)}개 컬럼 선택됨")

    X_feats = df[feat_cols].values.astype(float)

    if cfg.features.speed_as_feature:
        speed_oh = _speed_onehot(df["speed"]).values
        X = np.hstack([X_feats, speed_oh])
    else:
        X = X_feats

    y = _apply_target(df, cfg)
    groups = df["subject_id"].values
    return X, y, groups
