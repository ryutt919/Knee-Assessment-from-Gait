"""
step018: 전체 feature에 대해 eta² + Cohen's d 계산
- one-way ANOVA → partial eta²
- pairwise Cohen's d: ACLD vs ACLR, ACLD vs HA, ACLR vs HA

출력: data/processed/feature_ranking.csv
컬럼: feature, eta2, d_ACLD_ACLR, d_ACLD_HA, d_ACLR_HA, p_adj, speed
"""
import numpy as np
import pandas as pd
from scipy import stats
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"
OUT = ROOT / "data" / "processed" / "feature_ranking.csv"

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("[018] pingouin 미설치 — numpy fallback 사용")


def cohens_d_np(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    s_pool = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if s_pool == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / s_pool)


def partial_eta2_np(groups: list) -> float:
    overall = np.mean([v for g in groups for v in g])
    ss_b = sum(len(g) * (np.mean(g) - overall) ** 2 for g in groups)
    ss_w = sum(np.sum((np.array(g) - np.mean(g)) ** 2) for g in groups)
    denom = ss_b + ss_w
    return float(ss_b / denom) if denom > 0 else np.nan


def run():
    df = pd.read_csv(FEATURES)

    feat_cols = [c for c in df.columns
                 if any(s in c for s in ["_K1_", "_K2_", "_ROM_", "_IC_angle_", "_K1_timing_", "_asym"])
                 and c not in ["subject_id", "group", "speed", "injured_leg"]]

    print(f"[018] feature 수: {len(feat_cols)}")
    GROUPS = ["ACLD", "ACLR", "HA"]
    PAIRS = [("ACLD", "ACLR"), ("ACLD", "HA"), ("ACLR", "HA")]

    rows = []
    for feat in feat_cols:
        for speed in ["normal", "slow", "fast"]:
            sub = df[df["speed"] == speed][["group", feat]].dropna()
            g_data = {g: sub[sub["group"] == g][feat].values for g in GROUPS}
            valid = {g: v for g, v in g_data.items() if len(v) >= 3}

            if len(valid) < 2:
                continue

            all_vals = [v for v in valid.values()]
            eta2_val = partial_eta2_np(all_vals)

            try:
                _, p_anova = stats.f_oneway(*all_vals)
                p_adj = min(float(p_anova) * len(feat_cols), 1.0)
            except Exception:
                p_anova, p_adj = np.nan, np.nan

            if HAS_PINGOUIN:
                try:
                    aov = pg.anova(data=sub, dv=feat, between="group")
                    eta2_val = float(aov.loc[0, "np2"]) if "np2" in aov.columns else eta2_val
                except Exception:
                    pass

            row = {"feature": feat, "speed": speed, "eta2": eta2_val, "p_anova": float(p_anova) if not np.isnan(p_anova) else np.nan, "p_adj": p_adj}

            for g1, g2 in PAIRS:
                v1 = valid.get(g1, np.array([]))
                v2 = valid.get(g2, np.array([]))
                if HAS_PINGOUIN and len(v1) >= 2 and len(v2) >= 2:
                    try:
                        d = pg.compute_effsize(v1, v2, eftype="cohen")
                        row[f"d_{g1}_{g2}"] = float(d)
                    except Exception:
                        row[f"d_{g1}_{g2}"] = cohens_d_np(v1, v2)
                else:
                    row[f"d_{g1}_{g2}"] = cohens_d_np(v1, v2)

            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("eta2", ascending=False).reset_index(drop=True)
    df_out.to_csv(OUT, index=False)
    print(f"[018] ✅ feature_ranking.csv 저장: {OUT} ({len(df_out)}행)")
    print(f"[018] Top-5 features by eta²:")
    print(df_out[["feature", "speed", "eta2"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run()
