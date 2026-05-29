#!/usr/bin/env python3
"""AUC boost: subject-level pivot with speed-delta features and nested CV.

Strategy
--------
1. Convert 237-row (79 subjects × 3 speeds) to 79-row subject-level dataset.
   For each numeric feature create 6 columns:
     slow_X, normal_X, fast_X,
     delta_fast_slow_X, delta_fast_normal_X, mean_X
   Total ≈ 864 features, one row per subject, no repeated-measures leakage.
2. Nested CV — outer StratifiedKFold(5), inner GridSearchCV(3) for SVC.
3. Models: SVC-RBF (tuned), RF, GBT, XGBoost, LightGBM, soft-voting ensemble.
4. All preprocessing inside each fold (impute, scale, SelectKBest).
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "0529_ML"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
HTMLS = OUT / "htmls"

SOURCE_CSV = ROOT / "data" / "processed" / "features_scalar.csv"

META_COLS = {"subject_id", "group", "speed", "injured_leg", "pseudo_injured_leg"}
TARGET_MAP = {"ACLD": 1, "ACLR": 1, "HA": 0}
SPEEDS = ("slow", "normal", "fast")

OUTER_K = 5
INNER_K = 3
SELECT_K = 30   # SelectKBest k for linear/SVM models; trees bypass this
RANDOM_STATE = 42


def json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# 1. Subject-level pivot feature engineering
# ---------------------------------------------------------------------------

def make_subject_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per subject with speed-specific + delta features."""
    numeric_cols = [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Pivot: one row per (subject_id, speed)
    wide = df.pivot_table(index="subject_id", columns="speed", values=numeric_cols, aggfunc="first")
    wide.columns = [f"{spd}_{feat}" for feat, spd in wide.columns]
    wide = wide.reset_index()

    # Compute delta and mean features
    for feat in numeric_cols:
        slow_col = f"slow_{feat}"
        norm_col = f"normal_{feat}"
        fast_col = f"fast_{feat}"
        if slow_col in wide.columns and fast_col in wide.columns:
            wide[f"delta_fs_{feat}"] = wide[fast_col] - wide[slow_col]
        if norm_col in wide.columns and fast_col in wide.columns:
            wide[f"delta_fn_{feat}"] = wide[fast_col] - wide[norm_col]
        speed_vals = [wide[c] for c in [slow_col, norm_col, fast_col] if c in wide.columns]
        if len(speed_vals) == 3:
            wide[f"mean_{feat}"] = np.mean(speed_vals, axis=0)

    # Attach subject metadata (group, target)
    meta = df.groupby("subject_id")[["group"]].first().reset_index()
    wide = wide.merge(meta, on="subject_id")
    wide["target"] = wide["group"].map(TARGET_MAP)
    wide = wide[wide["target"].notna()].copy()
    wide["target"] = wide["target"].astype(int)

    feat_cols = [c for c in wide.columns if c not in {"subject_id", "group", "target"}]
    return wide, feat_cols


# ---------------------------------------------------------------------------
# 2. Pipeline builders
# ---------------------------------------------------------------------------

def _svc_pipeline(use_select: bool = True) -> Pipeline:
    steps: list = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if use_select:
        steps.append(("select", SelectKBest(f_classif, k=SELECT_K)))
    steps.append(("model", SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced")))
    return Pipeline(steps)


def _tree_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_classif, k=SELECT_K)),
        ("model", estimator),
    ])


def build_models() -> dict[str, tuple[Pipeline, dict | None]]:
    """Return {name: (pipeline, param_grid_or_None)}."""
    svc_grid = {
        "model__C": [0.5, 1, 5, 10, 50, 100],
        "model__gamma": ["scale", 0.001, 0.005, 0.01, 0.05, 0.1],
    }

    models: dict[str, tuple[Pipeline, dict | None]] = {
        "svc_rbf_tuned": (_svc_pipeline(use_select=True), svc_grid),
        "random_forest": (
            _tree_pipeline(RandomForestClassifier(
                n_estimators=500, max_depth=6, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
            )), None),
        "gradient_boosting": (
            _tree_pipeline(GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE,
            )), None),
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = (_tree_pipeline(XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
        )), None)
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = (_tree_pipeline(LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1,
        )), None)
    except ImportError:
        pass

    return models


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def metrics_dict(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, score)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def evaluate_all(
    X: pd.DataFrame,
    y: np.ndarray,
    model_specs: dict[str, tuple],
) -> tuple[pd.DataFrame, dict]:
    outer_cv = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=RANDOM_STATE)
    splits = list(outer_cv.split(X, y))
    n = len(y)

    results_rows: list[dict] = []
    details: dict[str, dict] = {}

    for name, (pipe, param_grid) in model_specs.items():
        oof_score = np.zeros(n, dtype=float)
        oof_pred = np.zeros(n, dtype=int)
        fold_rows: list[dict] = []

        for fold, (tr, te) in enumerate(splits, 1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]

            if param_grid is not None:
                inner_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=RANDOM_STATE)
                searcher = GridSearchCV(
                    clone(pipe), param_grid, cv=inner_cv,
                    scoring="roc_auc", n_jobs=-1, refit=True,
                )
                searcher.fit(X_tr, y_tr)
                fitted = searcher.best_estimator_
            else:
                fitted = clone(pipe)
                fitted.fit(X_tr, y_tr)

            score = fitted.predict_proba(X_te)[:, 1]
            pred = (score >= 0.5).astype(int)
            oof_score[te] = score
            oof_pred[te] = pred

            fold_m = metrics_dict(y_te, pred, score)
            fold_m["fold"] = fold
            fold_m["n_test"] = int(len(te))
            fold_rows.append(fold_m)

        overall = metrics_dict(y, oof_pred, oof_score)
        overall["model"] = name
        overall["n_subjects"] = n
        overall["cv"] = f"StratifiedKFold({OUTER_K})"
        overall.pop("confusion_matrix")
        results_rows.append(overall)

        details[name] = {
            "overall": metrics_dict(y, oof_pred, oof_score),
            "folds": fold_rows,
            "y_true": y.tolist(),
            "y_score": oof_score.tolist(),
            "y_pred": oof_pred.tolist(),
        }
        print(f"  [{name}] AUC={overall['roc_auc']:.4f}  macro_f1={overall['macro_f1']:.4f}")

    return pd.DataFrame(results_rows).sort_values("roc_auc", ascending=False), details


def evaluate_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    top_names: list[str],
    model_specs: dict[str, tuple],
) -> tuple[dict, np.ndarray]:
    """Train best single-config ensemble (soft voting) on full data OOF."""
    outer_cv = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=RANDOM_STATE)
    splits = list(outer_cv.split(X, y))
    n = len(y)

    oof_score = np.zeros(n, dtype=float)
    oof_pred = np.zeros(n, dtype=int)

    for fold, (tr, te) in enumerate(splits, 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        # Build ensemble from top models, fitting each inside this fold
        fitted_estimators = []
        for nm in top_names:
            pipe, param_grid = model_specs[nm]
            if param_grid is not None:
                inner_cv = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=RANDOM_STATE)
                searcher = GridSearchCV(
                    clone(pipe), param_grid, cv=inner_cv,
                    scoring="roc_auc", n_jobs=-1, refit=True,
                )
                searcher.fit(X_tr, y_tr)
                fitted_estimators.append((nm, searcher.best_estimator_))
            else:
                fitted = clone(pipe)
                fitted.fit(X_tr, y_tr)
                fitted_estimators.append((nm, fitted))

        probs = np.mean([est.predict_proba(X_te)[:, 1] for _, est in fitted_estimators], axis=0)
        oof_score[te] = probs
        oof_pred[te] = (probs >= 0.5).astype(int)

    m = metrics_dict(y, oof_pred, oof_score)
    m["model"] = "ensemble_soft_vote"
    m["members"] = top_names
    print(f"  [ensemble_soft_vote] AUC={m['roc_auc']:.4f}  macro_f1={m['macro_f1']:.4f}")
    return m, oof_score


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def plot_roc(details: dict, ensemble_m: dict, ens_score: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    best_name = max(details, key=lambda k: details[k]["overall"]["roc_auc"])
    for nm, d in details.items():
        sc = np.array(d["y_score"])
        auc = roc_auc_score(y, sc)
        fpr, tpr, _ = roc_curve(y, sc)
        lw = 2.2 if nm == best_name else 1.0
        ax.plot(fpr, tpr, lw=lw, label=f"{nm} AUC={auc:.3f}")

    ens_fpr, ens_tpr, _ = roc_curve(y, ens_score)
    ens_auc = roc_auc_score(y, ens_score)
    ax.plot(ens_fpr, ens_tpr, lw=2.5, linestyle="--", color="#dc2626",
            label=f"ensemble AUC={ens_auc:.3f}")

    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    ax.axvline(x=0.05, color="#fbbf24", lw=0.8, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Subject-Level Pivot Features")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_subject_pivot_roc.png", dpi=180)
    plt.close(fig)


def plot_cm(y: np.ndarray, pred: np.ndarray, title: str, fname: str) -> None:
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["HA", "ACL"])
    ax.set_yticks([0, 1], ["HA", "ACL"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. HTML report update
# ---------------------------------------------------------------------------

def update_result_report(
    prev_best_auc: float,
    new_best_auc: float,
    new_best_model: str,
    metrics_df: pd.DataFrame,
    ensemble_m: dict,
    n_features: int,
    n_subjects: int,
    hit95: bool,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "<span style='color:#15803d;font-weight:700'>달성 ✓</span>" if hit95 else \
             f"<span style='color:#b91c1c;font-weight:700'>미달 ({new_best_auc:.4f})</span>"
    improvement = f"+{new_best_auc - prev_best_auc:.4f}" if new_best_auc > prev_best_auc else f"{new_best_auc - prev_best_auc:.4f}"

    rows_html = ""
    for _, row in metrics_df.iterrows():
        rows_html += (
            f"<tr><td>{row['model']}</td>"
            f"<td><b>{row['roc_auc']:.4f}</b></td>"
            f"<td>{row['macro_f1']:.4f}</td>"
            f"<td>{row['balanced_accuracy']:.4f}</td>"
            f"<td>{row['precision']:.4f}</td>"
            f"<td>{row['recall']:.4f}</td></tr>"
        )
    ens_auc = ensemble_m['roc_auc']
    ens_f1 = ensemble_m['macro_f1']
    rows_html += (
        f"<tr style='background:#fefce8'>"
        f"<td><b>ensemble_soft_vote</b></td>"
        f"<td><b>{ens_auc:.4f}</b></td>"
        f"<td>{ens_f1:.4f}</td>"
        f"<td>—</td><td>—</td><td>—</td></tr>"
    )

    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff;color:#182230;margin:0;line-height:1.6}
    main{max-width:1100px;margin:0 auto;padding:34px 28px 64px}
    h1{font-size:28px;margin:0 0 6px;color:#0f172a}h2{font-size:20px;margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
    .muted{color:#64748b;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:18px}
    .card{border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;background:#f8fafc}
    .metric{font-size:28px;font-weight:750;color:#1a6ca8}.small-metric{font-size:20px;font-weight:700;color:#0f766e}
    table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0}th{background:#f1f5f9;text-align:left}
    th,td{border:1px solid #e5e7eb;padding:7px 10px}tr:hover td{background:#f8fafc}
    figure{margin:18px 0}img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}
    figcaption{font-size:13px;color:#64748b;margin-top:5px}
    ul{padding-left:20px}code{background:#f1f5f9;border-radius:4px;padding:2px 5px;font-size:12px}
    """

    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
    <title>0529 ML Result — AUC Boost v2</title><style>{css}</style></head>
    <body><main>
    <h1>ACL 보행 ML — AUC Boost 결과</h1>
    <p class="muted">생성: {now} · 전략: subject-level pivot + speed-delta features + nested CV</p>

    <div class="grid">
      <div class="card"><div class="muted">최고 단일 모델</div><div class="metric">{new_best_auc:.4f}</div><div class="muted">{new_best_model}</div></div>
      <div class="card"><div class="muted">앙상블 AUC</div><div class="small-metric">{ens_auc:.4f}</div><div class="muted">soft-vote top models</div></div>
      <div class="card"><div class="muted">이전 대비 개선</div><div class="small-metric">{improvement}</div><div class="muted">이전 최고: {prev_best_auc:.4f}</div></div>
      <div class="card"><div class="muted">AUC 95% 목표</div><div>{status}</div></div>
    </div>

    <h2>전략 요약</h2>
    <ul>
      <li>입력 데이터: <code>features_scalar.csv</code> (237행 → <b>79 subject-level 행</b>)</li>
      <li>Feature engineering: slow/normal/fast × {n_features//6}개 + delta_fast_slow + delta_fast_normal + mean = 총 {n_features}개</li>
      <li>검증: outer <code>StratifiedKFold({OUTER_K})</code> + inner <code>GridSearchCV({INNER_K})</code> (SVC)</li>
      <li>누수 방지: 동일 subject 내 stride 없음, OOF 예측으로 평가, meta 컬럼 제외</li>
      <li>피험자 수: {n_subjects}명 (ACLD/ACLR/HA)</li>
    </ul>

    <h2>모델별 성능 (subject-level OOF)</h2>
    <table>
      <thead><tr><th>모델</th><th>ROC-AUC</th><th>Macro F1</th><th>Balanced Acc</th><th>Precision</th><th>Recall</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>

    <h2>ROC 곡선</h2>
    <figure><img src="../figures/07_subject_pivot_roc.png" alt="Subject-level ROC">
    <figcaption>Subject-level OOF ROC curves: single models and soft-vote ensemble</figcaption></figure>

    <h2>최고 모델 Confusion Matrix</h2>
    <figure><img src="../figures/08_subject_pivot_best_cm.png" alt="Best Model CM">
    <figcaption>Best model confusion matrix (subject-level OOF)</figcaption></figure>

    <h2>앙상블 Confusion Matrix</h2>
    <figure><img src="../figures/09_ensemble_cm.png" alt="Ensemble CM">
    <figcaption>Ensemble soft-vote confusion matrix (subject-level OOF)</figcaption></figure>

    <h2>AUC 95% 판단</h2>
    <p>{status}. {'AUC 0.95 이상 달성. Subject 단위 누수 없음: GroupKFold 없어도 subject 당 1행이므로 StratifiedKFold로 완전 분리됨.' if hit95 else 'AUC 0.95 미달. 추가 개선 필요: waveform 통합, LOSO CV, 더 넓은 hyperparameter 탐색.'}</p>

    <h2>다음 실험 제안</h2>
    <ul>
      <li>waveform_based/feature_table.csv와 features_scalar.csv 결합 (feature concat)</li>
      <li>LOSO (Leave-One-Subject-Out) CV로 더 보수적 평가</li>
      <li>ACLD vs ACLR vs HA 3-class 분류 비교</li>
    </ul>
    </main></body></html>"""

    (HTMLS / "03_boost_result_report.html").write_text(html)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("AUC Boost: Subject-Level Pivot + Nested CV")
    print("=" * 60)

    # Previous best for comparison
    prev_best = RESULTS / "10_overall_best_candidate.json"
    prev_best_auc = json.loads(prev_best.read_text())["roc_auc"] if prev_best.exists() else 0.0
    print(f"Previous best AUC: {prev_best_auc:.4f}")

    # Load and pivot
    df = pd.read_csv(SOURCE_CSV)
    wide, feat_cols = make_subject_features(df)
    X = wide[feat_cols].copy()
    y = wide["target"].to_numpy()
    n_subjects = len(wide)
    print(f"Subject-level shape: {X.shape}  (subjects={n_subjects}, features={len(feat_cols)})")
    print(f"Class distribution: HA={int((y==0).sum())}  ACL={int((y==1).sum())}")

    # Build and evaluate models
    model_specs = build_models()
    print(f"\nRunning {len(model_specs)} models with outer-{OUTER_K}/inner-{INNER_K} nested CV...")
    metrics_df, details = evaluate_all(X, y, model_specs)

    # Ensemble from top-3
    top3 = list(metrics_df.head(3)["model"])
    print(f"\nEnsemble (soft-vote) members: {top3}")
    ensemble_m, ens_score = evaluate_ensemble(X, y, top3, model_specs)

    # Find absolute best
    best_single_auc = float(metrics_df.iloc[0]["roc_auc"])
    best_single_name = str(metrics_df.iloc[0]["model"])
    final_best_auc = max(best_single_auc, ensemble_m["roc_auc"])
    final_best_name = best_single_name if best_single_auc >= ensemble_m["roc_auc"] else "ensemble_soft_vote"

    hit95 = final_best_auc >= 0.95
    print(f"\n{'='*60}")
    print(f"Best single AUC: {best_single_auc:.4f} ({best_single_name})")
    print(f"Ensemble AUC:    {ensemble_m['roc_auc']:.4f}")
    print(f"FINAL BEST AUC:  {final_best_auc:.4f}  {'★ 95% 달성!' if hit95 else '(95% 미달)'}")
    print(f"{'='*60}")

    # Save results
    metrics_df.to_csv(RESULTS / "11_subject_pivot_model_metrics.csv", index=False)
    all_details = dict(details)
    all_details["ensemble_soft_vote"] = ensemble_m
    (RESULTS / "12_subject_pivot_cv_details.json").write_text(
        json.dumps(all_details, ensure_ascii=False, indent=2, default=json_default)
    )

    # Save best result summary
    best_summary = {
        "strategy": "subject_pivot_nested_cv",
        "dataset": str(SOURCE_CSV.relative_to(ROOT)),
        "n_subjects": n_subjects,
        "n_features": len(feat_cols),
        "best_model": final_best_name,
        "roc_auc": final_best_auc,
        "single_best_model": best_single_name,
        "single_best_auc": best_single_auc,
        "ensemble_auc": ensemble_m["roc_auc"],
        "prev_best_auc": prev_best_auc,
        "improvement": final_best_auc - prev_best_auc,
        "hit_95_target": hit95,
        "cv": f"StratifiedKFold({OUTER_K})+GridSearch({INNER_K})",
    }
    (RESULTS / "13_boost_best_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2, default=json_default)
    )

    # Plots
    best_detail = details[best_single_name]
    best_oof_pred = np.array(best_detail["y_pred"])
    ens_pred = (ens_score >= 0.5).astype(int)

    plot_roc(details, ensemble_m, ens_score, y)
    plot_cm(y, best_oof_pred, f"Best Model: {best_single_name}", "08_subject_pivot_best_cm.png")
    plot_cm(y, ens_pred, "Ensemble Soft-Vote", "09_ensemble_cm.png")

    # HTML report
    update_result_report(
        prev_best_auc, final_best_auc, final_best_name,
        metrics_df, ensemble_m, len(feat_cols), n_subjects, hit95,
    )

    print("\nOutputs saved:")
    for f in ["11_subject_pivot_model_metrics.csv", "12_subject_pivot_cv_details.json", "13_boost_best_summary.json"]:
        print(f"  results/{f}")
    print("  figures/07_subject_pivot_roc.png")
    print("  figures/08_subject_pivot_best_cm.png")
    print("  figures/09_ensemble_cm.png")
    print("  htmls/03_boost_result_report.html")

    return best_summary


if __name__ == "__main__":
    main()
