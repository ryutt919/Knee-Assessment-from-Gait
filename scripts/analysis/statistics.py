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

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.path.exists(os.path.join(BASE_DIR, "mds")):
    BASE_DIR = os.getcwd()
MDS_DIR     = os.path.join(BASE_DIR, "mds")
PATH_DATA   = os.path.join(MDS_DIR, "analysis_data.csv")
PATH_STATS  = os.path.join(MDS_DIR, "stats_result.csv")
PATH_CORR   = os.path.join(MDS_DIR, "correlation_result.csv")

ALPHA = 0.05

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

def partial_eta_squared(df: pd.DataFrame, feature: str) -> float:
    grand_mean = df[feature].mean()
    groups     = [g[feature].values for _, g in df.groupby("group")]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return round(ss_between / ss_total, 4) if ss_total > 0 else np.nan

def epsilon_squared(h_stat: float, n: int, k: int = 3) -> float:
    return round((h_stat - k + 1) / (n - k), 4) if (n - k) > 0 else np.nan

def run_preliminary_tests(df: pd.DataFrame, feature: str) -> dict:
    groups = [g[feature].dropna().values for _, g in df.groupby("group")]
    is_normal = all((shapiro(g)[1] > ALPHA if len(g) >= 3 else False) for g in groups)
    _, p_lev = levene(*groups)
    return {"is_normal": is_normal, "is_equal_var": (p_lev > ALPHA)}

def run_post_hoc_tukey(df: pd.DataFrame, feature: str) -> dict:
    result = {}
    try:
        tukey = pairwise_tukeyhsd(df[feature], df["group"])
        t_df  = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        for _, r in t_df.iterrows():
            result[f"tukey_p_{r['group1']}_vs_{r['group2']}"] = round(float(r["p-adj"]), 4)
    except Exception as e:
        result["tukey_error"] = str(e)
    return result

def run_post_hoc_dunn(df: pd.DataFrame, feature: str) -> dict:
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
        result[f"dunn_p_{g1}_vs_{g2}"] = round(min(p * n_pairs, 1.0), 4)
    return result

def analyze_feature(df: pd.DataFrame, feature: str) -> dict:
    result = {"feature": feature}
    sub_df = df[["subject_id", "group", "speed", feature]].dropna()
    if len(sub_df) < 9:
        result["note"] = "샘플 부족 (<9)"
        return result

    pre = run_preliminary_tests(sub_df, feature)
    result.update({"shapiro_normal": pre["is_normal"], "levene_equal_var": pre["is_equal_var"]})

    if pre["is_normal"] and pre["is_equal_var"]:
        result["model"] = "LMM"
        try:
            clean_feat = feature.replace("-", "_")
            sub_df2 = sub_df.rename(columns={feature: clean_feat})
            model = mixedlm(f"{clean_feat} ~ C(group) * C(speed)", sub_df2, groups=sub_df2["subject_id"])
            fitted = model.fit(reml=True, method="lbfgs", disp=False)

            group_keys = [k for k in fitted.pvalues.index if "group" in k.lower() and "speed" not in k.lower()]
            result["p_group_main"] = round(float(np.min([fitted.pvalues[k] for k in group_keys])), 4) if group_keys else np.nan

            int_keys = [k for k in fitted.pvalues.index if "group" in k.lower() and "speed" in k.lower()]
            result["p_interaction"] = round(float(np.min([fitted.pvalues[k] for k in int_keys])), 4) if int_keys else np.nan

            result.update({"effect_metric": "partial_eta_sq", "effect_size": partial_eta_squared(sub_df, feature)})
        except Exception as e:
            result.update({"lmm_error": str(e), "model": "LMM_FAILED"})

        result.update(run_post_hoc_tukey(sub_df, feature))
    else:
        result["model"] = "Kruskal-Wallis"
        groups_data = [g[feature].dropna().values for _, g in sub_df.groupby("group")]
        h_stat, p_kw = kruskal(*groups_data)
        result.update({"p_group_main": round(p_kw, 4), "effect_metric": "epsilon_sq", "effect_size": epsilon_squared(h_stat, len(sub_df))})
        result.update(run_post_hoc_dunn(sub_df, feature))
    return result

def run_correlation_analysis(df: pd.DataFrame, features: list) -> pd.DataFrame:
    feat_cols = [f for f in features if f in df.columns]
    agg = df.groupby("subject_id")[feat_cols + ["group"]].agg({**{f: "mean" for f in feat_cols}, "group": "first"}).reset_index()
    
    scaler = StandardScaler()
    agg[feat_cols] = scaler.fit_transform(agg[feat_cols].fillna(agg[feat_cols].mean()))
    
    for grp in ["ACLD", "ACLR", "Healthy"]:
        agg[f"is_{grp}"] = (agg["group"] == grp).astype(int)

    records = []
    for feat in feat_cols:
        rec = {"feature": feat}
        for dummy in ["is_ACLD", "is_ACLR", "is_Healthy"]:
            r, p = pointbiserialr(agg[dummy], agg[feat])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            rec.update({f"r_{dummy}": round(r, 4), f"p_{dummy}": round(p, 4), f"sig_{dummy}": sig})
        records.append(rec)
    return pd.DataFrame(records)

def run_statistics():
    log.info("▶ 통계 분석 파이프라인 시작 (분리된 모듈)")
    if not os.path.exists(PATH_DATA):
        log.error(f"데이터 파일 누락: {PATH_DATA}")
        return

    df = pd.read_csv(PATH_DATA)
    stat_records = [analyze_feature(df, feat) for feat in ALL_FEATURES if feat in df.columns]
    stats_df = pd.DataFrame(stat_records)
    stats_df.to_csv(PATH_STATS, index=False, encoding="utf-8-sig")
    log.info(f"✅ 통계 분석 완료. 결과 저장됨: {PATH_STATS}")

    corr_df = run_correlation_analysis(df, ALL_FEATURES)
    corr_df.to_csv(PATH_CORR, index=False, encoding="utf-8-sig")
    log.info(f"✅ 상관관계 분석 완료. 결과 저장됨: {PATH_CORR}")
    
    return stats_df, corr_df
