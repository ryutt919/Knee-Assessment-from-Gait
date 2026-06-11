"""
스칼라 피처 로더
- load_subject_scalar: features_scalar.csv (피험자 평균, 237행)
- load_stride_scalar : stride_level_peaks.parquet (stride 단위, ~14400행 trim 후)

Stride Trim: (subject_id, speed, trial_id) 그룹별 앞뒤 stride_trim개 제거
Speed one-hot: normal/slow/fast → [1,0,0], [0,1,0], [0,0,1]
η²-guided selection:
  - eta2_topK  : feature_ranking.csv top-K feature만 사용 (η² 내림차순 정렬)
  - eta2_sorted: 전체 feature를 η² 내림차순으로 정렬 (n_features HPO용)
  - all / none : 선택 없이 전체
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

_SUBJ_SUFFIXES = ["_LSI", "_injured", "_uninjured", "_asym", "_contralateral"]


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


def _strip_subj_suffix(feat: str) -> str:
    """subject-level feature명에서 side/metric suffix를 제거해 base name 반환."""
    for s in _SUBJ_SUFFIXES:
        if feat.endswith(s):
            return feat[: -len(s)]
    return feat


def _build_base_eta2(ranking_path: Path) -> dict[str, float]:
    """feature_ranking.csv → {base_name: max_eta2} 딕셔너리."""
    ranking = pd.read_csv(ranking_path)
    eta2_col = next((c for c in ranking.columns if "eta" in c.lower()), "eta2")
    feat_col = next((c for c in ranking.columns if "feature" in c.lower()), "feature")
    ranking = ranking[[feat_col, eta2_col]].rename(columns={feat_col: "feature", eta2_col: "eta2"})
    # speed별 여러 행 → max
    per_feat = ranking.groupby("feature")["eta2"].max()
    base_eta2: dict[str, float] = {}
    for feat_name, eta2_val in per_feat.items():
        base = _strip_subj_suffix(feat_name)
        base_eta2[base] = max(base_eta2.get(base, 0.0), float(eta2_val))
    return base_eta2


def _stride_col_eta2(col: str, base_eta2: dict[str, float]) -> float:
    """stride-level 컬럼명을 base_eta2 딕셔너리로 η² 점수 조회."""
    if col in base_eta2:
        return base_eta2[col]
    for base, val in base_eta2.items():
        if col.startswith(base + "_") or col == base:
            return val
    return 0.0


def _eta2_sort_feat_cols(feat_cols: list[str], ranking_path: Path) -> list[str]:
    """stride-level feat_cols를 η² 내림차순으로 정렬."""
    base_eta2 = _build_base_eta2(ranking_path)
    return sorted(feat_cols, key=lambda c: _stride_col_eta2(c, base_eta2), reverse=True)


def _eta2_feature_names(cfg: DictConfig) -> list[str] | None:
    """
    feature_select 설정에 따른 필터 feature 목록 반환.
    - all / none / eta2_sorted → None (필터 없음)
    - eta2_topK → top-K feature 이름 목록
    """
    sel = cfg.features.feature_select
    if sel in ("all", "none", "eta2_sorted"):
        return None
    if not sel.startswith("eta2_top"):
        return None
    k = int(sel.replace("eta2_top", ""))
    ranking_path = PROCESSED / cfg.data.feature_ranking
    ranking = pd.read_csv(ranking_path)
    eta2_col = next((c for c in ranking.columns if "eta" in c.lower()), "eta2")
    feat_col = next((c for c in ranking.columns if "feature" in c.lower()), "feature")
    ranking = ranking[[feat_col, eta2_col]].rename(columns={feat_col: "feature", eta2_col: "eta2"})
    ranking = ranking.groupby("feature")["eta2"].max().reset_index()
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
    Returns: X (ndarray), y (ndarray), groups (ndarray), feature_names (list[str])
    feature_names: X의 각 컬럼 이름 (η² 정렬 모드 시 η² 내림차순)
    """
    path = PROCESSED / cfg.data.scalar_stride
    df = pd.read_parquet(path)

    df = _trim_strides(df, cfg.features.stride_trim)
    print(f"[scalar_loader] stride trim 후: {len(df)}행")

    meta_cols = {"subject_id", "group", "speed", "side", "stride_idx", "trial_id"}
    feat_cols = [c for c in df.columns if c not in meta_cols]

    sel = cfg.features.feature_select
    ranking_path = PROCESSED / cfg.data.feature_ranking

    top_feats = _eta2_feature_names(cfg)
    if top_feats:
        base_top_feats = set()
        for f in top_feats:
            base_top_feats.add(_strip_subj_suffix(f))
        feat_cols_set = set(feat_cols)
        selected = set()
        for base in base_top_feats:
            if base in feat_cols_set:
                selected.add(base)
            else:
                for c in feat_cols:
                    if c.startswith(base + "_"):
                        selected.add(c)
        feat_cols = [c for c in feat_cols if c in selected]
        print(f"[scalar_loader] eta2 feature select: {len(feat_cols)}개 컬럼 선택됨")

    # η² 정렬: eta2_sorted 또는 eta2_topK 모드에서 적용
    if sel == "eta2_sorted" or sel.startswith("eta2_top"):
        feat_cols = _eta2_sort_feat_cols(feat_cols, ranking_path)
        print(f"[scalar_loader] η² 내림차순 정렬 완료 (상위 5): {feat_cols[:5]}")

    X_feats = df[feat_cols].values.astype(float)

    if cfg.features.speed_as_feature:
        speed_oh = _speed_onehot(df["speed"]).values
        X = np.hstack([X_feats, speed_oh])
        feature_names = list(feat_cols) + [f"speed_{sp}" for sp in SPEED_ORDER]
    else:
        X = X_feats
        feature_names = list(feat_cols)

    y = _apply_target(df, cfg)
    groups = df["subject_id"].values
    return X, y, groups, feature_names
