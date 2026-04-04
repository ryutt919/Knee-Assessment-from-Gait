"""
02_statistics.py
ACL 보행 생체역학 통계 분석 파이프라인

분석 흐름:
1. Shapiro-Wilk (정규성) + Levene's (등분산성) 사전 검정
2. LMM / Two-way Mixed ANOVA (Group × Speed, Subject_ID = Random Effect)
3. Tukey HSD / Dunn's Test (Bonferroni 보정) 사후 검정
4. Partial η² / Epsilon-squared 효과 크기 산출
5. Z-score 정규화 후 Point-Biserial 상관계수 산출
6. 결과 CSV 저장
"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
from itertools import combinations

from scipy import stats
from scipy.stats import shapiro, levene, kruskal, pointbiserialr
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 로깅 설정
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# 경로 설정
# ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDS_DIR     = os.path.join(BASE_DIR, "mds")
PATH_DATA   = os.path.join(MDS_DIR, "analysis_data.csv")
PATH_STATS  = os.path.join(MDS_DIR, "stats_result.csv")
PATH_CORR   = os.path.join(MDS_DIR, "correlation_result.csv")

ALPHA = 0.05

# ───────────────────────────────────────────────
# 분석 대상 피처 목록
# ───────────────────────────────────────────────
KINEMATIC_PEAKS = [
    "hip_flexion_injured",    "hip_adduction_injured",
    "knee_flexion_injured",   "knee_adduction_injured",  "knee_int_rotation_injured",
    "ankle_dorsiflexion_injured",
    "hip_flexion_contralateral", "knee_flexion_contralateral",
    "ankle_dorsiflexion_contralateral",
]
LSI_FEATURES = [
    "hip_flexion_LSI", "knee_flexion_LSI", "ankle_dorsiflexion_LSI"
]
SPATIOTEMPORAL = [
    "gait_speed_mps", "cadence_spm", "stride_length_mean_m",
    "step_width_mean_m_orth", "double_support_pct",
    "single_support_L_pct", "single_support_R_pct"
]
ALL_FEATURES = KINEMATIC_PEAKS + LSI_FEATURES + SPATIOTEMPORAL


# ───────────────────────────────────────────────
# 효과 크기 계산 함수
# ───────────────────────────────────────────────
def partial_eta_squared(df: pd.DataFrame, feature: str) -> float:
    """
    Partial η² = SS_between / SS_total
    그룹 설명 분산 / 전체 분산의 비율
    """
    grand_mean = df[feature].mean()
    groups     = [g[feature].values for _, g in df.groupby("group")]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return round(ss_between / ss_total, 4) if ss_total > 0 else np.nan


def epsilon_squared(h_stat: float, n: int, k: int = 3) -> float:
    """
    Kruskal-Wallis ε² = (H - k + 1) / (n - k)
    비모수 버전 효과 크기
    """
    return round((h_stat - k + 1) / (n - k), 4) if (n - k) > 0 else np.nan


# ───────────────────────────────────────────────
# Step 1: 사전 검정
# ───────────────────────────────────────────────
def run_preliminary_tests(df: pd.DataFrame, feature: str) -> dict:
    """
    Shapiro-Wilk 정규성 검정 (그룹별 모두 통과 시 True)
    Levene's Test 등분산성 검정
    """
    groups = [g[feature].dropna().values for _, g in df.groupby("group")]

    # Shapiro-Wilk: 모든 그룹이 정규성 충족해야 is_normal=True
    is_normal = all(
        (shapiro(g)[1] > ALPHA if len(g) >= 3 else False)
        for g in groups
    )

    # Levene's Test
    _, p_lev = levene(*groups)
    is_equal_var = (p_lev > ALPHA)

    return {"is_normal": is_normal, "is_equal_var": is_equal_var}


# ───────────────────────────────────────────────
# Step 2 & 3: 피처별 주 통계 분석
# ───────────────────────────────────────────────
def run_post_hoc_tukey(df: pd.DataFrame, feature: str) -> dict:
    """Tukey HSD 사후 검정 결과를 딕셔너리로 반환"""
    result = {}
    try:
        tukey = pairwise_tukeyhsd(df[feature], df["group"])
        t_df  = pd.DataFrame(
            data    = tukey._results_table.data[1:],
            columns = tukey._results_table.data[0]
        )
        for _, r in t_df.iterrows():
            key = f"tukey_p_{r['group1']}_vs_{r['group2']}"
            result[key] = round(float(r["p-adj"]), 4)
    except Exception as e:
        result["tukey_error"] = str(e)
    return result


def run_post_hoc_dunn(df: pd.DataFrame, feature: str) -> dict:
    """
    Dunn's Test (Bonferroni 보정) — scikit_posthocs 없이 수동 구현
    Mann-Whitney U + Bonferroni 보정으로 대체
    """
    result = {}
    groups = df["group"].unique().tolist()
    n_pairs = len(list(combinations(groups, 2)))

    for g1, g2 in combinations(groups, 2):
        a = df[df["group"] == g1][feature].dropna().values
        b = df[df["group"] == g2][feature].dropna().values
        if len(a) < 2 or len(b) < 2:
            result[f"dunn_p_{g1}_vs_{g2}"] = np.nan
            continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Bonferroni 보정: p × 검정 쌍 수
        p_adj = min(p * n_pairs, 1.0)
        result[f"dunn_p_{g1}_vs_{g2}"] = round(p_adj, 4)
    return result


def analyze_feature(df: pd.DataFrame, feature: str) -> dict:
    """
    단일 피처에 대해 전체 통계 파이프라인 수행:
    사전 검정 → 모델 선택 → 주효과 p값 → 사후 검정 → 효과 크기
    """
    result = {"feature": feature}
    sub_df = df[["subject_id", "group", "speed", feature]].dropna()

    if len(sub_df) < 9:
        result["note"] = "샘플 부족 (<9)"
        return result

    pre = run_preliminary_tests(sub_df, feature)
    result["shapiro_normal"]   = pre["is_normal"]
    result["levene_equal_var"] = pre["is_equal_var"]

    if pre["is_normal"] and pre["is_equal_var"]:
        # ── Linear Mixed Model (Two-way Mixed ANOVA 근사) ──
        result["model"] = "LMM"
        try:
            # 안전한 컬럼명 처리 (특수문자 제거)
            clean_feat = feature.replace("-", "_")
            sub_df2 = sub_df.rename(columns={feature: clean_feat})

            formula = f"{clean_feat} ~ C(group) * C(speed)"
            model   = mixedlm(formula, sub_df2, groups=sub_df2["subject_id"])
            fitted  = model.fit(reml=True, method="lbfgs", disp=False)

            # Group 주효과 p값 (ACLR 계수의 p값을 대표값으로 사용)
            group_keys = [k for k in fitted.pvalues.index if "group" in k.lower() and "speed" not in k.lower()]
            if group_keys:
                result["p_group_main"] = round(float(np.min([fitted.pvalues[k] for k in group_keys])), 4)
            else:
                result["p_group_main"] = np.nan

            # 상호작용 효과 p값
            int_keys = [k for k in fitted.pvalues.index if "group" in k.lower() and "speed" in k.lower()]
            if int_keys:
                result["p_interaction"] = round(float(np.min([fitted.pvalues[k] for k in int_keys])), 4)
            else:
                result["p_interaction"] = np.nan

            result["effect_metric"] = "partial_eta_sq"
            result["effect_size"]   = partial_eta_squared(sub_df, feature)

        except Exception as e:
            result["lmm_error"] = str(e)
            result["model"] = "LMM_FAILED"

        # Tukey HSD 사후 검정 (그룹 간)
        result.update(run_post_hoc_tukey(sub_df, feature))

    else:
        # ── Kruskal-Wallis (비모수) ──
        result["model"] = "Kruskal-Wallis"
        groups_data = [g[feature].dropna().values for _, g in sub_df.groupby("group")]
        h_stat, p_kw = kruskal(*groups_data)

        result["p_group_main"]  = round(p_kw, 4)
        result["effect_metric"] = "epsilon_sq"
        result["effect_size"]   = epsilon_squared(h_stat, len(sub_df))

        # Dunn's Test (Bonferroni) 사후 검정
        result.update(run_post_hoc_dunn(sub_df, feature))

    return result


# ───────────────────────────────────────────────
# Step 4: Z-score 정규화 → Point-Biserial 상관
# ───────────────────────────────────────────────
def run_correlation_analysis(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    1. 피험자별 평균값으로 집계 (speed 조건 → 피험자 레벨로 축소)
    2. Z-score 표준화 (단위 스케일 편향 제거)
    3. 그룹 더미 변수(is_ACLD, is_ACLR, is_Healthy) 생성
    4. Point-Biserial 상관계수 산출 + p값 포함
    """
    log.info("상관관계 분석: Z-score 표준화 후 Point-Biserial 산출...")

    feat_cols = [f for f in features if f in df.columns]

    # 피험자별 평균 집계
    agg = df.groupby("subject_id")[feat_cols + ["group"]].agg(
        {**{f: "mean" for f in feat_cols}, "group": "first"}
    ).reset_index()

    # Z-score 표준화 (컬럼별 독립 정규화)
    scaler = StandardScaler()
    agg[feat_cols] = scaler.fit_transform(
        agg[feat_cols].fillna(agg[feat_cols].mean())
    )

    # 그룹 더미 변수
    for grp in ["ACLD", "ACLR", "Healthy"]:
        agg[f"is_{grp}"] = (agg["group"] == grp).astype(int)

    dummy_cols = ["is_ACLD", "is_ACLR", "is_Healthy"]
    records = []

    for feat in feat_cols:
        rec = {"feature": feat}
        for dummy in dummy_cols:
            r, p = pointbiserialr(agg[dummy], agg[feat])
            rec[f"r_{dummy}"]  = round(r, 4)
            rec[f"p_{dummy}"]  = round(p, 4)
            # 유의성 표기
            rec[f"sig_{dummy}"] = (
                "***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else "ns"
            )
        records.append(rec)

    corr_df = pd.DataFrame(records)
    log.info(f"  상관관계 행렬 완료: {corr_df.shape}")
    return corr_df


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    df = pd.read_csv(PATH_DATA)
    log.info(f"분석 데이터 로드: {df.shape}")
    log.info(f"그룹 분포:\n{df.groupby(['group', 'speed']).size().to_string()}")

    # ── 피처별 통계 분석 ──
    log.info("\n=== 통계 분석 시작 ===")
    stat_records = []
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            log.warning(f"  컬럼 없음: {feat}")
            continue
        log.info(f"  분석 중: {feat}")
        stat_records.append(analyze_feature(df, feat))

    stats_df = pd.DataFrame(stat_records)
    stats_df.to_csv(PATH_STATS, index=False, encoding="utf-8-sig")
    log.info(f"\n✅ 통계 결과 저장 → {PATH_STATS}")

    # 유의미한 피처 요약 출력
    sig_feats = stats_df[stats_df["p_group_main"] < ALPHA][["feature", "model", "p_group_main", "effect_size", "effect_metric"]]
    log.info(f"\n그룹 간 유의미한 피처 (p < {ALPHA}):\n{sig_feats.to_string(index=False)}")

    # ── 상관관계 분석 ──
    log.info("\n=== 상관관계 분석 시작 ===")
    corr_df = run_correlation_analysis(df, ALL_FEATURES)
    corr_df.to_csv(PATH_CORR, index=False, encoding="utf-8-sig")
    log.info(f"✅ 상관관계 결과 저장 → {PATH_CORR}")


if __name__ == "__main__":
    main()
