"""
04_shap_analysis.py — SHAP 기반 Impairment Score 기여 피처 해석 및 시각화

입력:
  - data/processed/mahalanobis_features.parquet  (Waveform 특징)
  - results/oof_results.parquet                  (OOF Impairment Score 결과)
  - results/01_optuna_best_params.json           (최적 파라미터)

출력:
  - results/03_shap_interpretation/summary_plot.png
  - results/03_shap_interpretation/{subject_id}_waterfall.png  (ACL 피험자별)

방법:
  1. 최적 파라미터로 전체 데이터에 대해 파이프라인 재학습 (HA 전체 사용)
  2. 파이프라인을 함수 f(x) = Impairment Score(x) 로 래핑
  3. XGBoost Proxy 모델을 Impairment Score에 대해 훈련 (TreeSHAP 활용)
  4. SHAP Summary Plot + 개별 피험자 Waterfall Plot 저장
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # 화면 표시 없이 파일 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

ROOT    = Path(__file__).resolve().parent.parent.parent
SANDBOX = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR    = SANDBOX / "results"
SHAP_DIR       = RESULTS_DIR / "03_shap_interpretation"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

SCALER_MAP = {
    "zscore":  StandardScaler,
    "robust":  RobustScaler,
    "minmax":  MinMaxScaler,
}


# ─── 파이프라인 함수 재정의 (02와 동일한 로직, 전체 HA 사용) ─────────────────

def train_pipeline(
    df_ha: pd.DataFrame,
    wave_cols: list[str],
    scaling: str = "zscore",
    pca_k_method: str = "kaiser",
    pca_variance_ratio: float = 0.90,
    pca_fixed_k: int = 10,
    mcd_support_fraction: float = 0.75,
):
    """
    정상인(HA) 전체 데이터로 스케일러 + PCA + MCD 학습.

    Returns: (scaler, pca_k, mcd, mu_ha, sigma_ha)
    """
    from sklearn.decomposition import PCA

    ScalerCls = SCALER_MAP.get(scaling, StandardScaler)
    X_ha = df_ha[wave_cols].values.astype(np.float64)
    X_ha = np.nan_to_num(X_ha, nan=0.0)

    # 스케일러 적합
    scaler = ScalerCls()
    X_ha_s = scaler.fit_transform(X_ha)

    # PCA 적합
    pca_full = PCA()
    pca_full.fit(X_ha_s)
    ev = pca_full.explained_variance_

    if pca_k_method == "kaiser":
        k = max(int(np.sum(ev >= 1.0)), 2)
    elif pca_k_method == "variance_ratio":
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = max(int(np.searchsorted(cumvar, pca_variance_ratio) + 1), 2)
    else:
        k = min(pca_fixed_k, len(ev))
    k = min(k, X_ha_s.shape[0] - 1, X_ha_s.shape[1])
    k = max(k, 2)

    pca_k = PCA(n_components=k)
    pca_k.fit(X_ha_s)
    X_ha_pc = pca_k.transform(X_ha_s)

    # MCD 적합
    n_ha = len(X_ha_pc)
    min_support = max(int(mcd_support_fraction * n_ha), k + 1)
    actual_fraction = min_support / n_ha if n_ha > 0 else mcd_support_fraction
    mcd = MinCovDet(support_fraction=actual_fraction, random_state=42)
    mcd.fit(X_ha_pc)

    # HA 분포 통계
    dm_ha = _mahal(X_ha_pc, mcd)
    mu_ha    = float(np.mean(dm_ha))
    sigma_ha = float(np.std(dm_ha))

    return scaler, pca_k, mcd, mu_ha, sigma_ha


def _mahal(X_pc: np.ndarray, mcd) -> np.ndarray:
    delta = X_pc - mcd.location_
    Vi    = np.linalg.pinv(mcd.covariance_)
    return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, Vi, delta)))


def score_samples(X: np.ndarray, scaler, pca_k, mcd, mu_ha: float, sigma_ha: float) -> np.ndarray:
    """임의 샘플의 Impairment Score 계산."""
    X_s  = scaler.transform(np.nan_to_num(X, nan=0.0))
    X_pc = pca_k.transform(X_s)
    dm   = _mahal(X_pc, mcd)
    imp  = np.maximum(0.0, (dm - mu_ha) / (sigma_ha + 1e-12))
    return imp


def get_meta_cols(df: pd.DataFrame) -> list[str]:
    meta = {"subject_id", "group", "speed", "trial_id", "actual_leg",
            "side_basis", "stride_idx", "cycle_len", "label",
            "mahal_dist", "impairment_score"}
    return [c for c in df.columns if c in meta]


def get_wave_cols(df: pd.DataFrame) -> list[str]:
    meta = {"subject_id", "group", "speed", "trial_id", "actual_leg",
            "side_basis", "stride_idx", "cycle_len", "label",
            "mahal_dist", "impairment_score"}
    return [c for c in df.columns if c not in meta]


# ─── Proxy 모델 학습 및 SHAP ─────────────────────────────────────────────────

def run_shap_analysis(
    df_feat: pd.DataFrame,
    best_params: dict,
    oof_df: pd.DataFrame,
    n_shap_bg: int = 100,
    n_shap_subjects: int = 10,
) -> None:
    """
    XGBoost Proxy → TreeSHAP 분석 및 시각화 저장.

    Args:
        df_feat      : 전체 Waveform 특징 DataFrame
        best_params  : Optuna 최적 파라미터 (또는 기본값)
        oof_df       : OOF 결과 (impairment_score 컬럼 포함)
        n_shap_bg    : SHAP 배경 샘플 수 (KernelSHAP용, Proxy 학습 시 무시)
        n_shap_subjects: ACL 피험자 중 Waterfall Plot 저장할 수
    """
    try:
        import shap
        import xgboost as xgb
    except ImportError:
        print("[04] ❌ shap 또는 xgboost 미설치. pip install shap xgboost")
        return

    wave_cols = get_wave_cols(df_feat)
    print(f"[04] Waveform 컬럼 수: {len(wave_cols)}")

    # 1. 전체 HA 학습으로 파이프라인 재훈련
    df_ha = df_feat[df_feat["group"] == "HA"].copy()
    print(f"[04] HA 전체 학습: n={len(df_ha)}")

    scaler, pca_k, mcd, mu_ha, sigma_ha = train_pipeline(
        df_ha,
        wave_cols,
        scaling=best_params.get("scaling", "zscore"),
        pca_k_method=best_params.get("pca_k_method", "kaiser"),
        pca_variance_ratio=best_params.get("pca_variance_ratio", 0.90),
        pca_fixed_k=best_params.get("pca_fixed_k", 10),
        mcd_support_fraction=best_params.get("mcd_support_fraction", 0.75),
    )
    print(f"[04] PCA k={pca_k.n_components}, HA mu={mu_ha:.3f}, sigma={sigma_ha:.3f}")

    # 2. 전체 데이터 Impairment Score 계산
    X_all = df_feat[wave_cols].values.astype(np.float64)
    imp_all = score_samples(X_all, scaler, pca_k, mcd, mu_ha, sigma_ha)
    df_feat = df_feat.copy()
    df_feat["impairment_score"] = imp_all

    # 3. Proxy XGBoost 학습 (Waveform → Impairment Score 회귀)
    #    : 파이프라인의 비선형 동작을 XGBoost로 근사하여 TreeSHAP 적용
    print("[04] XGBoost Proxy 모델 학습 중...")
    X_np = np.nan_to_num(df_feat[wave_cols].values.astype(np.float64), nan=0.0)
    y_np = df_feat["impairment_score"].values.astype(np.float64)

    proxy = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    proxy.fit(X_np, y_np, eval_set=[(X_np, y_np)], verbose=False)

    # Proxy 정확도
    y_pred = proxy.predict(X_np)
    corr   = float(np.corrcoef(y_np, y_pred)[0, 1])
    print(f"[04] Proxy 학습 R (상관계수): {corr:.4f}")

    # 4. TreeSHAP 계산
    print("[04] SHAP 값 계산 중...")
    explainer = shap.TreeExplainer(proxy)
    shap_values = explainer(X_np, check_additivity=False)  # (n_samples, n_features)

    # 채널 이름 추출 (채널명_001~101 → 채널명)
    channel_names = [c.rsplit("_", 1)[0] for c in wave_cols]

    # ── 5-a. Summary Plot (Feature 수 많으므로 채널 단위로 Aggregation) ────
    n_wave = len(wave_cols)
    unique_channels = []
    seen = set()
    for cn in channel_names:
        if cn not in seen:
            unique_channels.append(cn)
            seen.add(cn)

    # 채널별 SHAP 절대값 합산
    shap_per_channel = np.zeros((len(df_feat), len(unique_channels)))
    for ci, ch in enumerate(unique_channels):
        idxs = [i for i, c in enumerate(channel_names) if c == ch]
        shap_per_channel[:, ci] = np.abs(shap_values.values[:, idxs]).sum(axis=1)

    # Summary bar plot (채널 단위)
    mean_shap = shap_per_channel.mean(axis=0)
    order     = np.argsort(mean_shap)[::-1][:30]  # top-30 채널
    top_channels = [unique_channels[i] for i in order]
    top_vals     = mean_shap[order]

    # 채널별 색상 (타입 구분)
    def _color(ch: str) -> str:
        if ch.startswith("sensorFreeAcc"):  return "#1f77b4"
        if ch.startswith("sensorOrientation"): return "#ff7f0e"
        if ch.startswith("sensorMagnetic"):  return "#2ca02c"
        return "#d62728"  # joint angle

    colors = [_color(c) for c in top_channels]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_channels)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_channels)))
    ax.set_yticklabels(top_channels[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP| (Proxy XGBoost)", fontsize=11)
    ax.set_title("SHAP Summary — Impairment Score 기여 채널 Top-30", fontsize=13)

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Free Acceleration (IMU)"),
        Patch(facecolor="#ff7f0e", label="Orientation (IMU)"),
        Patch(facecolor="#2ca02c", label="Magnetic Field (IMU)"),
        Patch(facecolor="#d62728", label="Joint Angle"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    out_summary = SHAP_DIR / "summary_plot.png"
    fig.savefig(out_summary, dpi=150)
    plt.close(fig)
    print(f"[04] Summary Plot 저장: {out_summary}")

    # ── 5-b. 개별 피험자 Waterfall Plot ─────────────────────────────────────
    acl_subjects = (
        df_feat[df_feat["group"].isin(["ACLD", "ACLR"])]
        .groupby("subject_id")["impairment_score"]
        .mean()
        .sort_values(ascending=False)
        .head(n_shap_subjects)
        .index.tolist()
    )

    for subj_id in acl_subjects:
        subj_mask = df_feat["subject_id"] == subj_id
        subj_idx  = np.where(subj_mask.values)[0]
        if len(subj_idx) == 0:
            continue

        # 해당 피험자의 평균 SHAP (채널 단위)
        subj_shap = shap_values.values[subj_idx]       # (n_strides, n_features)
        subj_shap_mean = subj_shap.mean(axis=0)        # (n_features,)

        # 채널별 집계
        ch_shap = np.zeros(len(unique_channels))
        for ci, ch in enumerate(unique_channels):
            idxs = [i for i, c in enumerate(channel_names) if c == ch]
            ch_shap[ci] = subj_shap_mean[idxs].sum()

        top_order = np.argsort(np.abs(ch_shap))[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 7))
        vals = ch_shap[top_order]
        chns = [unique_channels[i] for i in top_order]
        cols_wf = ["#d62728" if v > 0 else "#1f77b4" for v in vals]

        ax.barh(range(len(chns)), vals[::-1], color=cols_wf[::-1])
        ax.set_yticks(range(len(chns)))
        ax.set_yticklabels(chns[::-1], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (Proxy → Impairment Score)", fontsize=10)

        grp = df_feat.loc[subj_mask, "group"].iloc[0]
        avg_imp = df_feat.loc[subj_mask, "impairment_score"].mean()
        ax.set_title(
            f"SHAP Waterfall — {subj_id} ({grp})\n"
            f"평균 Impairment Score = {avg_imp:.3f}",
            fontsize=11,
        )
        plt.tight_layout()
        out_wf = SHAP_DIR / f"{subj_id}_waterfall.png"
        fig.savefig(out_wf, dpi=150)
        plt.close(fig)
        print(f"[04] Waterfall Plot 저장: {out_wf}")

    print(f"[04] ✅ SHAP 분석 완료. 결과 디렉토리: {SHAP_DIR}")


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(test_mode: bool = False) -> None:
    # 특징 데이터 로드
    feat_path = DATA_PROCESSED / ("mahalanobis_features_test.parquet" if test_mode else "mahalanobis_features.parquet")
    if not feat_path.exists():
        print(f"[04] ❌ 특징 파일 없음: {feat_path}")
        sys.exit(1)

    # OOF 결과 로드
    oof_path = RESULTS_DIR / ("oof_results_test.parquet" if test_mode else "oof_results.parquet")
    if not oof_path.exists():
        print(f"[04] ⚠️  OOF 결과 없음: {oof_path}  (02_mahalanobis_pipeline.py를 먼저 실행)")
        oof_df = None
    else:
        oof_df = pd.read_parquet(oof_path)

    # 최적 파라미터 로드
    params_path = RESULTS_DIR / "01_optuna_best_params.json"
    if params_path.exists():
        best_params = json.loads(params_path.read_text())["best_params"]
        print(f"[04] Optuna 최적 파라미터 사용: {best_params}")
    else:
        best_params = {"scaling": "zscore", "pca_k_method": "kaiser"}
        print("[04] ⚠️  Optuna 결과 없음, 기본 파라미터 사용:", best_params)

    df_feat = pd.read_parquet(feat_path)
    print(f"[04] 특징 DataFrame: shape={df_feat.shape}")

    n_subjects = min(5, df_feat["subject_id"].nunique()) if test_mode else 10

    run_shap_analysis(
        df_feat=df_feat,
        best_params=best_params,
        oof_df=oof_df,
        n_shap_bg=50 if test_mode else 100,
        n_shap_subjects=n_subjects,
    )


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
