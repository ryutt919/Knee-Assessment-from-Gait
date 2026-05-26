"""
SHAP 분석 + feature_ranking.csv (η²) 순위 비교.
- TreeExplainer: RF, GBT, XGBoost, LightGBM
- LinearExplainer: LogReg, LinearSVC
- DeepExplainer: CNN1D (PyTorch)
- KernelExplainer: SVM-RBF (느림 — 100 background samples)
출력: beeswarm plot, η² vs SHAP Spearman 상관, 교집합 리스트
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ML = Path(__file__).parent.parent

_TREE_MODELS    = ("rf", "gbt", "xgboost", "lightgbm")
_LINEAR_MODELS  = ("logreg", "linearsvc")
_KERNEL_MODELS  = ("svm_rbf", "fpca")
_DEEP_MODELS    = ("cnn1d", "transformer")


def _get_explainer(model_name: str, model, X_background: np.ndarray):
    import shap
    if model_name in _TREE_MODELS:
        raw = getattr(model, "_clf", model)
        return shap.TreeExplainer(raw)
    elif model_name in _LINEAR_MODELS:
        raw = getattr(model, "_clf", model)
        return shap.LinearExplainer(raw, X_background)
    elif model_name in _KERNEL_MODELS:
        raw = getattr(model, "_clf", model)
        bg = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(raw.predict_proba, bg)
    elif model_name in _DEEP_MODELS:
        import torch
        net = model.net
        bg_t = torch.tensor(X_background[:50].astype(np.float32))
        return shap.DeepExplainer(net, bg_t)
    else:
        bg = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model.predict_proba, bg)


def compute_shap(
    model_name: str,
    model,
    X_test: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """
    SHAP 값 계산. 반환: (n_samples, n_features) — multi-class는 평균 abs.
    """
    import shap

    explainer = _get_explainer(model_name, model, X_background)

    if model_name in _DEEP_MODELS:
        import torch
        X_t = torch.tensor(X_test.astype(np.float32))
        raw = explainer.shap_values(X_t)
    else:
        raw = explainer.shap_values(X_test)

    # 이진 분류: raw가 list[2] → 클래스 1만 사용
    # 다중 분류: list[k] → 절댓값 평균
    if isinstance(raw, list):
        if len(raw) == 2:
            shap_vals = raw[1]
        else:
            shap_vals = np.mean([np.abs(v) for v in raw], axis=0)
    else:
        shap_vals = raw

    return shap_vals


def shap_importance(shap_vals: np.ndarray,
                    feature_names: list[str] | None = None) -> pd.Series:
    """mean(|SHAP|) per feature."""
    imp = np.abs(shap_vals).mean(axis=0)
    if feature_names:
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)
    return pd.Series(imp).sort_values(ascending=False)


def compare_with_eta2(
    shap_imp: pd.Series,
    feature_ranking_path: str | Path,
    top_k: int = 10,
) -> dict:
    """
    SHAP top-k와 η² top-k 비교.
    Returns: {spearman_rho, p_value, intersection_features, shap_rank, eta2_rank}
    """
    from scipy.stats import spearmanr

    fr = pd.read_csv(feature_ranking_path)
    # feature_ranking.csv 컬럼 확인
    feat_col = next((c for c in fr.columns if "feature" in c.lower()), fr.columns[0])
    eta2_col = next((c for c in fr.columns if "eta" in c.lower() or "η" in c), fr.columns[1])

    fr = fr[[feat_col, eta2_col]].dropna()
    fr.columns = ["feature", "eta2"]
    fr = fr.sort_values("eta2", ascending=False).reset_index(drop=True)

    # 공통 feature
    common = list(set(shap_imp.index) & set(fr["feature"]))
    if len(common) < 5:
        return {"error": f"공통 feature {len(common)}개 — 비교 불가"}

    shap_rank = shap_imp[common].rank(ascending=False)
    eta2_rank = fr.set_index("feature").loc[common, "eta2"].rank(ascending=False)

    rho, pval = spearmanr(shap_rank, eta2_rank)

    shap_top  = set(shap_imp.head(top_k).index)
    eta2_top  = set(fr.head(top_k)["feature"])
    intersection = sorted(shap_top & eta2_top)

    return {
        "spearman_rho":        float(rho),
        "p_value":             float(pval),
        "intersection_count":  len(intersection),
        "intersection_features": intersection,
        "shap_top_k":          list(shap_imp.head(top_k).index),
        "eta2_top_k":          list(fr.head(top_k)["feature"]),
    }


def save_beeswarm(
    shap_vals: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    save_path: str | Path = "",
    top_k: int = 20,
) -> None:
    """SHAP beeswarm plot을 PNG로 저장."""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = shap.Explanation(
        values=shap_vals,
        data=X_test,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(exp, max_display=top_k, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def run_shap_analysis(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    feature_ranking_path: str | Path | None = None,
    save_dir: str | Path = "",
) -> dict:
    """
    전체 SHAP 분석 파이프라인.
    Returns: {shap_importance, comparison, beeswarm_path}
    """
    shap_vals = compute_shap(model_name, model, X_test, X_train, feature_names)
    imp       = shap_importance(shap_vals, feature_names)

    result: dict = {"shap_importance": imp}

    if feature_ranking_path and Path(feature_ranking_path).exists():
        result["comparison"] = compare_with_eta2(imp, feature_ranking_path)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        bee_path = save_dir / f"shap_beeswarm_{model_name}.png"
        save_beeswarm(shap_vals, X_test, feature_names, bee_path)
        result["beeswarm_path"] = str(bee_path)

    return result
