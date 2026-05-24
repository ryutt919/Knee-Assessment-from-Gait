"""
step017: 전체 LMM + Kruskal-Wallis fallback
- 모든 scalar feature에 LMM 적용
- Shapiro-Wilk p<0.05 → Kruskal + Dunn 사용
- 출력: data/processed/stats_results.csv

컬럼: feature, comparison, speed, estimate, SE, p, p_adj, cohens_d, eta2
"""
import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"
OUT = ROOT / "data" / "processed" / "stats_results.csv"

try:
    import statsmodels.formula.api as smf
    from scipy import stats
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[017] 경고: statsmodels 미설치")

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def partial_eta2(groups_data: list) -> float:
    overall_mean = np.mean([v for g in groups_data for v in g])
    ss_between = sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups_data)
    ss_within = sum(np.sum((np.array(g) - np.mean(g)) ** 2) for g in groups_data)
    if ss_between + ss_within == 0:
        return np.nan
    return float(ss_between / (ss_between + ss_within))


def test_normality(values: np.ndarray) -> bool:
    if len(values) < 8:
        return True
    try:
        _, p = stats.shapiro(values)
        return bool(p >= 0.05)
    except Exception:
        return True


def run_stats_for_feature(df: pd.DataFrame, feat: str, speed: str) -> list:
    sub = df[df["speed"] == speed][["subject_id", "group", feat]].dropna()
    if len(sub) < 10:
        return []

    groups_data = {g: sub[sub["group"] == g][feat].values for g in ["ACLD", "ACLR", "HA"]}
    groups_data = {g: v for g, v in groups_data.items() if len(v) >= 3}

    if len(groups_data) < 2:
        return []

    is_normal = all(test_normality(v) for v in groups_data.values())
    rows = []

    if is_normal and HAS_STATSMODELS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    f"Q('{feat}') ~ C(group)",
                    data=sub.rename(columns={feat: feat}),
                    groups=sub["subject_id"],
                ).fit(reml=True)

            eta2_val = np.nan
            group_vals = list(groups_data.values())
            if len(group_vals) >= 2:
                eta2_val = partial_eta2(group_vals)

            for param, pval in model.pvalues.items():
                if "group" not in param.lower():
                    continue
                rows.append({
                    "feature": feat, "speed": speed,
                    "comparison": param, "method": "LMM",
                    "estimate": float(model.params.get(param, np.nan)),
                    "SE": float(model.bse.get(param, np.nan)),
                    "p": float(pval),
                    "p_adj": min(float(pval) * len(model.pvalues), 1.0),
                    "eta2": eta2_val, "cohens_d": np.nan,
                })
        except Exception as e:
            is_normal = False

    if not is_normal or not rows:
        group_vals = list(groups_data.values())
        try:
            _, p_kw = stats.kruskal(*group_vals)
        except Exception:
            p_kw = np.nan

        group_names = list(groups_data.keys())
        pairs = [(g1, g2) for i, g1 in enumerate(group_names) for g2 in group_names[i+1:]]

        for g1, g2 in pairs:
            v1, v2 = groups_data.get(g1, np.array([])), groups_data.get(g2, np.array([]))
            if len(v1) < 2 or len(v2) < 2:
                continue
            try:
                _, p_mann = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                p_adj = min(p_mann * len(pairs), 1.0)
            except Exception:
                p_mann, p_adj = np.nan, np.nan
            d = cohens_d(v1, v2)
            rows.append({
                "feature": feat, "speed": speed,
                "comparison": f"{g1}_vs_{g2}", "method": "Kruskal",
                "estimate": float(np.mean(v1) - np.mean(v2)),
                "SE": np.nan, "p": p_kw, "p_adj": p_adj,
                "eta2": partial_eta2([v1, v2]),
                "cohens_d": d,
            })

    return rows


def run():
    df = pd.read_csv(FEATURES)

    scalar_cols = [c for c in df.columns
                   if any(s in c for s in ["_K1_", "_K2_", "_ROM_", "_IC_angle_", "_K1_timing_"])
                   and c not in ["subject_id", "group", "speed", "injured_leg"]]

    print(f"[017] 분석 feature 수: {len(scalar_cols)}")

    all_rows = []
    for i, feat in enumerate(scalar_cols):
        for speed in ["normal", "slow", "fast"]:
            rows = run_stats_for_feature(df, feat, speed)
            all_rows.extend(rows)
        if (i + 1) % 20 == 0:
            print(f"[017] 진행: {i+1}/{len(scalar_cols)}")

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT, index=False)
    print(f"[017] ✅ stats_results.csv 저장: {OUT} ({len(df_out)}행)")


if __name__ == "__main__":
    run()
