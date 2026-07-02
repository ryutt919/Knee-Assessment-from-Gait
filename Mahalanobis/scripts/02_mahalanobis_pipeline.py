"""
02_mahalanobis_pipeline.py — PCA + Robust MCD 마할라노비스 거리 및 Impairment Score 계산

입력: data/processed/mahalanobis_features.parquet (or test variant)
출력: results/02_evaluation_report.md  (평가 리포트)
      results/oof_results.parquet      (OOF 예측 결과, SHAP 입력용)

파이프라인 (GroupKFold 교차검증):
  1. 피험자(subject_id) 기준 GroupKFold(n_splits=5)
  2. 각 fold:
     a. 스케일러(StandardScaler) → HA train 데이터로 fit
     b. PCA → HA train PC scores로 fit  (Kaiser criterion: eigenvalue >= 1)
     c. Robust MCD (support_fraction=0.75) → HA train PC scores로 fit
     d. Validation HA + ACL → PC 투영 → 마할라노비스 거리 계산
     e. Impairment Score = max(0, (D_M - mu_HA) / sigma_HA)
  3. OOF 전체 AUC-ROC (HA=0, ACL=1) 평가
  4. 그룹별 Impairment Score 분포 통계
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

ROOT    = Path(__file__).resolve().parent.parent.parent
SANDBOX = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR    = SANDBOX / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

SCALER_MAP = {
    "zscore":  StandardScaler,
    "robust":  RobustScaler,
    "minmax":  MinMaxScaler,
}


def get_waveform_cols(df: pd.DataFrame) -> list[str]:
    """특징 컬럼(메타 제외) 목록 반환."""
    meta = {"subject_id", "group", "speed", "trial_id", "actual_leg",
            "side_basis", "stride_idx", "cycle_len"}
    return [c for c in df.columns if c not in meta]


def select_pca_k(pca: PCA, method: str, variance_ratio: float = 0.90, fixed_k: int = 10,
                 n_ha_samples: int = 9999) -> int:
    """Kaiser criterion / variance ratio / fixed 중 하나로 PC 수 결정.
    
    MCD 안정성을 위해 최대 PC 수는 min(n_ha // 5, 100)으로 제한.
    """
    ev = pca.explained_variance_
    # 안전 상한: MCD는 n >> p 조건 필요 (최소 5배 여유)
    hard_max = min(n_ha_samples // 5, 100)
    hard_max = max(hard_max, 2)

    if method == "kaiser":
        k = int(np.sum(ev >= 1.0))
        return max(min(k, hard_max), 2)
    elif method == "variance_ratio":
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cumvar, variance_ratio) + 1)
        return max(min(k, len(ev), hard_max), 2)
    else:  # fixed
        return max(min(fixed_k, len(ev), hard_max), 2)


def compute_mahalanobis(X_query: np.ndarray, mcd: MinCovDet) -> np.ndarray:
    """MCD 기반 마할라노비스 거리 계산."""
    delta = X_query - mcd.location_
    Vi = np.linalg.pinv(mcd.covariance_)
    dists = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, Vi, delta)))
    return dists


def impairment_score(dists: np.ndarray, mu_ha: float, sigma_ha: float) -> np.ndarray:
    """정상군 기준 Z-score 형태 Impairment Score (0 이상)."""
    if sigma_ha < 1e-12:
        return np.zeros_like(dists)
    return np.maximum(0.0, (dists - mu_ha) / sigma_ha)


# ─── 단일 설정으로 전체 CV 실행 ───────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    scaling: str = "zscore",
    pca_k_method: str = "kaiser",
    pca_variance_ratio: float = 0.90,
    pca_fixed_k: int = 10,
    speed_filter: str = "all",
    distance_metric: str = "mahalanobis",
    n_splits: int = 5,
    mcd_support_fraction: float = 0.75,
    verbose: bool = True,
) -> dict:
    """
    GroupKFold CV를 수행하고 OOF 결과 및 주요 메트릭을 반환.

    Returns:
        dict with keys:
          oof_df     : OOF 결과 DataFrame
          auc_roc    : overall OOF AUC-ROC (HA=0, ACL=1)
          group_stats: 그룹별 Impairment Score 통계
          fold_aucs  : fold별 AUC
          best_params: 사용된 파라미터
    """
    wave_cols = get_waveform_cols(df)
    if len(wave_cols) == 0:
        raise ValueError("Waveform 컬럼이 없습니다.")

    # 속도 필터
    if speed_filter != "all":
        df = df[df["speed"] == speed_filter].copy()
    if len(df) == 0:
        raise ValueError(f"speed_filter='{speed_filter}' 적용 후 데이터 없음.")

    # 레이블 (HA=0, ACL=1)
    df = df.copy()
    df["label"] = (df["group"] != "HA").astype(int)

    subjects = df["subject_id"].values
    X = df[wave_cols].values.astype(np.float64)
    y = df["label"].values
    groups_arr = subjects

    gkf = GroupKFold(n_splits=n_splits)
    ScalerCls = SCALER_MAP.get(scaling, StandardScaler)

    oof_indices: list[int] = []
    oof_dm: list[float] = []
    oof_imp: list[float] = []
    fold_aucs: list[float] = []

    total_splits = gkf.get_n_splits(X, y, groups_arr)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups_arr)):
        X_train_all, X_val = X[train_idx], X[val_idx]
        y_train_all = y[train_idx]
        groups_train = groups_arr[train_idx]

        # HA train 데이터만 사용하여 스케일러/PCA/MCD 적합
        ha_mask = y_train_all == 0
        X_ha = X_train_all[ha_mask]

        if len(X_ha) < 10:
            if verbose:
                print(f"  Fold {fold_i+1}: HA 학습 샘플 부족 ({len(X_ha)}), 스킵")
            continue

        # 1. 스케일러 (HA train fit)
        scaler = ScalerCls()
        scaler.fit(X_ha)
        X_ha_scaled  = scaler.transform(X_ha)
        X_val_scaled = scaler.transform(X_val)

        # NaN 처리
        X_ha_scaled  = np.nan_to_num(X_ha_scaled,  nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

        # 2. PCA (HA train fit)
        pca_k_s = PCA()
        pca_k_s.fit(X_ha_scaled)
        k = select_pca_k(pca_k_s, pca_k_method, pca_variance_ratio, pca_fixed_k,
                          n_ha_samples=len(X_ha))
        k = min(k, X_ha_scaled.shape[0] - 1, X_ha_scaled.shape[1])
        k = max(k, 2)

        pca_k = PCA(n_components=k)
        pca_k.fit(X_ha_scaled)
        X_ha_pc  = pca_k.transform(X_ha_scaled)
        X_val_pc = pca_k.transform(X_val_scaled)

        # 3. Robust MCD (HA PC scores fit)
        n_ha_pc = len(X_ha_pc)
        # MCD는 최소 샘플 수 조건: support_fraction * n >= p + 1
        min_support = max(int(mcd_support_fraction * n_ha_pc), k + 1)
        actual_fraction = min_support / n_ha_pc if n_ha_pc > 0 else mcd_support_fraction

        try:
            mcd = MinCovDet(support_fraction=actual_fraction, random_state=42)
            mcd.fit(X_ha_pc)
        except Exception as e:
            if verbose:
                print(f"  Fold {fold_i+1}: MCD 실패 ({e}), 일반 공분산 사용")
            from sklearn.covariance import EmpiricalCovariance
            ec = EmpiricalCovariance().fit(X_ha_pc)
            # MinCovDet 인터페이스 모사
            class FallbackMCD:
                location_ = ec.location_
                covariance_ = ec.covariance_
            mcd = FallbackMCD()

        # 4. 마할라노비스 거리 계산
        dm_val = compute_mahalanobis(X_val_pc, mcd)

        if distance_metric == "squared_mahalanobis":
            dm_val = dm_val ** 2

        # HA train 샘플의 거리도 계산 (Impairment Score 정규화용)
        dm_ha_train = compute_mahalanobis(X_ha_pc, mcd)
        if distance_metric == "squared_mahalanobis":
            dm_ha_train = dm_ha_train ** 2
        mu_ha = float(np.mean(dm_ha_train))
        sigma_ha = float(np.std(dm_ha_train))

        # 5. Impairment Score
        imp_val = impairment_score(dm_val, mu_ha, sigma_ha)

        # 6. 이 fold의 AUC (HA=0 ACL=1, 거리가 클수록 ACL)
        y_val = y[val_idx]
        if len(np.unique(y_val)) < 2:
            fold_auc = float("nan")
        else:
            fold_auc = roc_auc_score(y_val, dm_val)
        fold_aucs.append(fold_auc)

        oof_indices.extend(val_idx.tolist())
        oof_dm.extend(dm_val.tolist())
        oof_imp.extend(imp_val.tolist())

        if verbose:
            print(f"  Fold {fold_i+1}/{total_splits}: k_pc={k}, n_HA={len(X_ha)}, "
                  f"AUC={fold_auc:.4f}, mu_HA_dist={mu_ha:.2f}, sigma={sigma_ha:.2f}")

    # OOF 결과 조합
    oof_order = np.argsort(oof_indices)
    oof_df = df.iloc[[oof_indices[i] for i in oof_order]].copy().reset_index(drop=True)
    oof_df["mahal_dist"]       = [oof_dm[i] for i in oof_order]
    oof_df["impairment_score"] = [oof_imp[i] for i in oof_order]

    # 전체 OOF AUC
    valid_mask = ~oof_df["mahal_dist"].isna()
    if oof_df[valid_mask]["label"].nunique() >= 2:
        overall_auc = roc_auc_score(
            oof_df[valid_mask]["label"], oof_df[valid_mask]["mahal_dist"]
        )
    else:
        overall_auc = float("nan")

    # 그룹별 통계
    group_stats = (
        oof_df.groupby("group")[["mahal_dist", "impairment_score"]]
        .agg(["mean", "std", "median"])
        .round(4)
        .to_dict()
    )

    return {
        "oof_df":    oof_df,
        "auc_roc":   overall_auc,
        "group_stats": group_stats,
        "fold_aucs": fold_aucs,
        "best_params": dict(
            scaling=scaling,
            pca_k_method=pca_k_method,
            pca_variance_ratio=pca_variance_ratio,
            pca_fixed_k=pca_fixed_k,
            speed_filter=speed_filter,
            distance_metric=distance_metric,
            mcd_support_fraction=mcd_support_fraction,
        ),
    }


# ─── 리포트 저장 ──────────────────────────────────────────────────────────────

def save_report(result: dict, out_path: Path) -> None:
    lines = ["# Mahalanobis Impairment Score — 교차 검증 리포트\n"]
    lines.append(f"## 파라미터\n```json\n{json.dumps(result['best_params'], indent=2, ensure_ascii=False)}\n```\n")
    lines.append(f"## 전체 OOF AUC-ROC\n`{result['auc_roc']:.4f}`\n")
    lines.append("## Fold별 AUC\n")
    for i, a in enumerate(result["fold_aucs"]):
        lines.append(f"- Fold {i+1}: `{a:.4f}`\n")
    lines.append("\n## 그룹별 Impairment Score 통계\n")
    oof_df = result["oof_df"]
    for grp, sub in oof_df.groupby("group"):
        dm_mean = sub["mahal_dist"].mean()
        dm_std  = sub["mahal_dist"].std()
        imp_mean = sub["impairment_score"].mean()
        imp_std  = sub["impairment_score"].std()
        lines.append(
            f"- **{grp}**: D_M mean={dm_mean:.3f} ± {dm_std:.3f}, "
            f"Impairment={imp_mean:.3f} ± {imp_std:.3f}\n"
        )
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[02] 리포트 저장: {out_path}")


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main(test_mode: bool = False) -> dict:
    src = DATA_PROCESSED / ("mahalanobis_features_test.parquet" if test_mode else "mahalanobis_features.parquet")
    if not src.exists():
        print(f"[02] ❌ 입력 파일 없음: {src}")
        print("[02] 먼저 01_data_preprocessing.py를 실행하세요.")
        sys.exit(1)

    print(f"[02] 로드: {src}")
    df = pd.read_parquet(src)
    print(f"[02] shape={df.shape}, groups={df['group'].value_counts().to_dict()}")

    n_subjects = df["subject_id"].nunique()
    n_splits   = min(5, n_subjects // 3) if n_subjects >= 3 else 2
    n_splits   = max(n_splits, 2)
    print(f"[02] n_subjects={n_subjects}, GroupKFold n_splits={n_splits}")

    result = run_pipeline(
        df,
        scaling="zscore",
        pca_k_method="kaiser",
        speed_filter="all",
        distance_metric="mahalanobis",
        n_splits=n_splits,
        mcd_support_fraction=0.75,
        verbose=True,
    )

    print(f"\n[02] ✅ 전체 OOF AUC-ROC: {result['auc_roc']:.4f}")

    # OOF 결과 저장 (SHAP 분석용)
    oof_out = RESULTS_DIR / ("oof_results_test.parquet" if test_mode else "oof_results.parquet")
    result["oof_df"].to_parquet(oof_out, index=False)
    print(f"[02] OOF 결과 저장: {oof_out}")

    # 리포트 저장
    save_report(result, RESULTS_DIR / "02_evaluation_report.md")

    return result


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
