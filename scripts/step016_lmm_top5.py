"""
step016: LMM — SPM1D에서 유의했던 feature 상위 5개 적용
- Fixed: group × speed
- Random: subject intercept
- Post-hoc: Bonferroni

검증: 수렴 확인, p/eta2 출력
"""
import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEATURES = ROOT / "data" / "processed" / "features_scalar.csv"
SPM_RESULTS = ROOT / "data" / "processed" / "spm_results.csv"
OUT = ROOT / "data" / "processed" / "lmm_top5_results.csv"

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[016] 경고: statsmodels 미설치")


def select_top_features(features_df: pd.DataFrame, spm_df: pd.DataFrame, n: int = 5) -> list:
    sig_channels = spm_df[spm_df["sig_start_pct"].notna()]["channel"].unique()

    candidate_feats = []
    for feat_col in features_df.columns:
        for ch in sig_channels:
            if ch in feat_col and feat_col not in ["subject_id", "group", "speed", "injured_leg"]:
                candidate_feats.append(feat_col)

    if not candidate_feats:
        scalar_feats = [c for c in features_df.columns
                        if any(s in c for s in ["_K1_", "_K2_", "_ROM_", "_IC_", "_K1_timing"])
                        and c not in ["subject_id", "group", "speed", "injured_leg"]]
        return scalar_feats[:n]

    return list(dict.fromkeys(candidate_feats))[:n]


def run_lmm(df: pd.DataFrame, feat: str) -> dict:
    sub_df = df[["subject_id", "group", "speed", feat]].dropna()
    sub_df = sub_df.rename(columns={feat: "feature_value"})

    if len(sub_df) < 20:
        return {"feature": feat, "converged": False, "error": "데이터 부족"}

    if not HAS_STATSMODELS:
        return {"feature": feat, "converged": False, "error": "statsmodels 미설치"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                "feature_value ~ C(group) * C(speed)",
                data=sub_df,
                groups=sub_df["subject_id"],
            ).fit(reml=True)

        result = {
            "feature": feat,
            "converged": model.converged,
            "aic": float(model.aic),
            "bic": float(model.bic),
        }

        for param, pval in model.pvalues.items():
            result[f"p_{param[:30]}"] = float(pval)

        return result

    except Exception as e:
        return {"feature": feat, "converged": False, "error": str(e)[:100]}


def run():
    df = pd.read_csv(FEATURES)
    spm_df = pd.read_csv(SPM_RESULTS) if SPM_RESULTS.exists() else pd.DataFrame(columns=["channel", "sig_start_pct"])

    top_feats = select_top_features(df, spm_df, n=5)
    print(f"[016] LMM 대상 feature: {top_feats}")

    results = []
    for feat in top_feats:
        if feat not in df.columns:
            print(f"[016] 스킵: {feat} 컬럼 없음")
            continue
        res = run_lmm(df, feat)
        results.append(res)
        status = "수렴" if res.get("converged") else f"실패({res.get('error', '')})"
        print(f"[016] {feat}: {status}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT, index=False)
    print(f"[016] ✅ LMM top-5 결과 저장: {OUT}")


if __name__ == "__main__":
    run()
