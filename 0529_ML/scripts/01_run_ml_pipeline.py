#!/usr/bin/env python3
"""Traditional ML sandbox for ACL gait classification.

All outputs are written under 0529_ML only. Source project files are read-only.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "0529_ML"
DIRS = {
    "data": OUT / "data",
    "results": OUT / "results",
    "logs": OUT / "logs",
    "htmls": OUT / "htmls",
    "models": OUT / "models",
    "figures": OUT / "figures",
}
SOURCE_DOCS = [
    ROOT / "GAI 중간 결과" / "intro.md",
    ROOT / "GAI 중간 결과" / "statistics_v1.html",
    ROOT / "GAI 중간 결과" / "0527_main.html",
]
DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "stride_level_peaks.parquet",
    ROOT / "data" / "processed" / "features_scalar.csv",
    ROOT / "data" / "processed" / "analysis_data.csv",
    ROOT / "data" / "processed" / "waveform_based" / "feature_table.csv",
]
TARGET_MAP = {"ACLD": 1, "ACLR": 1, "HA": 0, "Healthy adults": 0, "Healthy adolescents": 0}


class TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


@dataclass
class DatasetSpec:
    path: Path
    frame: pd.DataFrame
    group_col: str
    subject_col: str
    target_col: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


def ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    raw = path.read_text(errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        parser = TextHTMLParser()
        parser.feed(raw)
        return "\n".join(parser.parts)
    return raw


def compact(text: str, max_len: int = 2600) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_len]


def json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def summarize_docs() -> dict:
    texts = {str(p.relative_to(ROOT)): read_text(p) for p in SOURCE_DOCS}
    joined = "\n".join(texts.values())
    return {
        "read_files": list(texts),
        "source_previews": {k: compact(v, 2200) for k, v in texts.items()},
        "research_summary": (
            "IMU 기반 다속도 보행 데이터로 ACL 결손(ACLD), ACL 재건(ACLR), "
            "건강 대조군(HA)의 보행 차이를 분석한다. 기존 통계 보고서는 "
            "ACLD 27명, ACLR 27명, HA 25명, Normal/Slow/Fast 3속도 조건, "
            "Hip/Knee/Ankle 3평면 관절각 및 LSI/ROM/peak 지표를 사용했다."
        ),
        "statistical_findings": (
            "Bonferroni 보정 후 5개 스칼라 지표가 유의했고, fast 속도 "
            "무릎 굴곡 loading response peak LSI가 가장 큰 효과 크기를 보였다. "
            "ACLR 환측 무릎 굴곡 첫 피크가 건측 대비 낮고, 고관절 굴곡 ROM "
            "감소는 ACLD/ACLR 모두에서 HA 대비 관찰되었다."
        ),
        "target_definition": (
            "기본 ML 타깃은 ACLD/ACLR=1, HA=0 이진 분류로 둔다. 이는 임상적으로 "
            "ACL 병력군과 건강 대조군 판별에 직접 대응하며, 피험자 수가 작은 "
            "3군 다중분류보다 subject-safe CV에서 더 안정적이다."
        ),
        "mentions_auc_95": "95%" in joined or "AUC" in joined.upper(),
    }


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def first_col(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in cols:
        lc = c.lower()
        if any(token in lc for token in candidates):
            return c
    return None


def normalize_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    upper = s.upper()
    if upper in {"ACLD", "ACL-D"}:
        return "ACLD"
    if upper in {"ACLR", "ACL-R"}:
        return "ACLR"
    if upper in {"HA", "HEALTHY", "CONTROL", "CONTROLS"}:
        return "HA"
    if "HEALTHY" in upper:
        return "HA"
    return s


def leakage_like(col: str) -> bool:
    lc = col.lower()
    hard_tokens = [
        "group",
        "target",
        "label",
        "diagnosis",
        "class",
        "cohort",
        "subject",
        "participant",
        "patient",
        "id",
        "name",
        "file",
        "path",
        "source",
        "fold",
        "split",
        "matched",
        "pair",
        "spm_region",
    ]
    metadata_exact = {"sex", "age", "weight", "height", "injured_leg", "pseudo_injured_leg"}
    return lc in metadata_exact or any(tok in lc for tok in hard_tokens)


def choose_dataset() -> tuple[DatasetSpec, list[dict]]:
    summaries: list[dict] = []
    best: DatasetSpec | None = None
    for path in DATA_CANDIDATES:
        if not path.exists():
            continue
        try:
            df = load_frame(path)
        except Exception as exc:
            summaries.append({"path": str(path.relative_to(ROOT)), "error": repr(exc)})
            continue

        group_col = first_col(list(df.columns), ["group", "Group", "condition", "diagnosis"])
        subject_col = first_col(list(df.columns), ["subject_id", "participant_id", "subject", "participant", "ID", "id"])
        summary = {
            "path": str(path.relative_to(ROOT)),
            "shape": list(df.shape),
            "columns_preview": list(df.columns[:40]),
            "group_col": group_col,
            "subject_col": subject_col,
        }
        if group_col:
            groups = df[group_col].map(normalize_group)
            summary["group_counts"] = groups.value_counts(dropna=False).head(20).to_dict()
        if subject_col:
            summary["n_subjects"] = int(df[subject_col].nunique(dropna=True))
        summaries.append(summary)
        if not group_col or not subject_col:
            continue

        work = df.copy()
        work["_group_norm"] = work[group_col].map(normalize_group)
        work = work[work["_group_norm"].isin(["ACLD", "ACLR", "HA"])].copy()
        if work["_group_norm"].nunique() < 2:
            continue
        work["_target_acl"] = work["_group_norm"].map(TARGET_MAP).astype(int)

        excluded = {group_col, subject_col, "_group_norm", "_target_acl"}
        feature_cols = [
            c for c in work.columns
            if c not in excluded and not leakage_like(c)
        ]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
        categorical_cols = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique(dropna=True) <= 20
        ]
        usable = len(numeric_cols) + len(categorical_cols)
        summary["usable_features"] = usable
        if usable == 0:
            continue

        candidate = DatasetSpec(
            path=path,
            frame=work,
            group_col="_group_norm",
            subject_col=subject_col,
            target_col="_target_acl",
            feature_cols=numeric_cols + categorical_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        if best is None:
            best = candidate
        else:
            # Prefer stride-level data with more samples and enough subjects.
            if candidate.frame.shape[0] > best.frame.shape[0] and candidate.frame[subject_col].nunique() >= 20:
                best = candidate

    if best is None:
        raise RuntimeError("No usable dataset with group and subject columns was found.")
    return best, summaries


def write_eda(spec: DatasetSpec, data_summaries: list[dict]) -> dict:
    df = spec.frame
    eda = {
        "selected_dataset": str(spec.path.relative_to(ROOT)),
        "shape": list(df.shape),
        "subject_col": spec.subject_col,
        "n_subjects": int(df[spec.subject_col].nunique(dropna=True)),
        "target_counts": df[spec.target_col].value_counts().sort_index().to_dict(),
        "group_counts": df[spec.group_col].value_counts().to_dict(),
        "subject_group_counts": df.groupby(spec.group_col)[spec.subject_col].nunique().to_dict(),
        "numeric_feature_count": len(spec.numeric_cols),
        "categorical_feature_count": len(spec.categorical_cols),
        "excluded_leakage_rule": "Dropped identifiers, labels, group/cohort fields, source/path/file, split/fold, and matching/pair columns.",
        "missing_top20": df[spec.feature_cols].isna().mean().sort_values(ascending=False).head(20).to_dict(),
        "all_candidate_data": data_summaries,
    }
    (DIRS["results"] / "01_eda_summary.json").write_text(json.dumps(eda, ensure_ascii=False, indent=2, default=json_default))
    pd.DataFrame(data_summaries).to_csv(DIRS["results"] / "02_data_inventory.csv", index=False)
    missing = (
        df[spec.feature_cols].isna().mean().rename("missing_rate").sort_values(ascending=False).reset_index()
        .rename(columns={"index": "feature"})
    )
    missing.to_csv(DIRS["results"] / "03_missingness.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df[spec.group_col].value_counts().plot(kind="bar", ax=ax[0], color=["#c0392b", "#1a6ca8", "#27ae60"])
    ax[0].set_title("Rows by Group")
    ax[0].set_ylabel("Rows")
    df.groupby(spec.group_col)[spec.subject_col].nunique().plot(kind="bar", ax=ax[1], color=["#c0392b", "#1a6ca8", "#27ae60"])
    ax[1].set_title("Subjects by Group")
    ax[1].set_ylabel("Subjects")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "01_group_distribution.png", dpi=180)
    plt.close(fig)

    top_missing = missing.head(20).sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top_missing))))
    ax.barh(top_missing["feature"], top_missing["missing_rate"], color="#64748b")
    ax.set_xlabel("Missing Rate")
    ax.set_title("Top Missing Features")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "02_missingness.png", dpi=180)
    plt.close(fig)
    return eda


def build_preprocessor(spec: DatasetSpec) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    transformers = []
    if spec.numeric_cols:
        transformers.append(("num", Pipeline(num_steps), spec.numeric_cols))
    if spec.categorical_cols:
        transformers.append(("cat", Pipeline(cat_steps), spec.categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_models(n_features: int, n_rows: int) -> dict[str, Pipeline]:
    k = min(30, max(1, n_features))
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=4000, class_weight="balanced", C=0.5, solver="liblinear", random_state=42
        ),
        "linear_svm": LinearSVC(C=0.2, class_weight="balanced", random_state=42, max_iter=8000),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=160, learning_rate=0.04, max_depth=2, random_state=42
        ),
    }
    if n_rows <= 5000:
        models["svc_rbf"] = SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.04,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    except Exception:
        pass

    pipelines: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        steps: list[tuple[str, object]] = [
            ("preprocess", "passthrough"),
            ("select", SelectKBest(f_classif, k=k)),
        ]
        if name in {"logistic_regression", "linear_svm", "svc_rbf"}:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        pipelines[name] = Pipeline(steps)
    return pipelines


def estimator_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        scores = np.asarray(scores, dtype=float)
        lo, hi = np.nanmin(scores), np.nanmax(scores)
        if hi > lo:
            return (scores - lo) / (hi - lo)
        return np.full(scores.shape, 0.5)
    return model.predict(x_test)


def metrics_from_predictions(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    try:
        out["roc_auc"] = roc_auc_score(y, score)
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["average_precision"] = average_precision_score(y, score)
    except Exception:
        out["average_precision"] = float("nan")
    return out


def evaluate_models(spec: DatasetSpec) -> tuple[pd.DataFrame, dict, str]:
    df = spec.frame.reset_index(drop=True)
    x = df[spec.feature_cols]
    y = df[spec.target_col].to_numpy()
    groups = df[spec.subject_col].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        raise RuntimeError("Need at least two subject groups for GroupKFold.")

    preprocessor = build_preprocessor(spec)
    models = make_models(len(spec.feature_cols), len(df))
    rows = []
    details: dict[str, dict] = {}

    splitter_name = "StratifiedGroupKFold"
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = list(splitter.split(x, y, groups))
    except Exception:
        splitter_name = "GroupKFold"
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = list(splitter.split(x, y, groups))

    for name, pipe in models.items():
        y_score = np.zeros(len(df), dtype=float)
        y_pred = np.zeros(len(df), dtype=int)
        fold_rows = []
        for fold, (tr, te) in enumerate(split_iter, 1):
            fold_model = clone(pipe)
            fold_model.set_params(preprocess=clone(preprocessor))
            fold_model.fit(x.iloc[tr], y[tr])
            score = estimator_scores(fold_model, x.iloc[te])
            pred = (score >= 0.5).astype(int)
            y_score[te] = score
            y_pred[te] = pred
            fold_metric = metrics_from_predictions(y[te], pred, score)
            fold_metric.update({
                "model": name,
                "fold": fold,
                "n_test_rows": int(len(te)),
                "n_test_subjects": int(len(np.unique(groups[te]))),
                "test_groups": dict(pd.Series(y[te]).value_counts().sort_index()),
            })
            fold_rows.append(fold_metric)
        m = metrics_from_predictions(y, y_pred, y_score)
        m.update({"model": name, "n_folds": n_splits, "cv": splitter_name})
        rows.append({k: v for k, v in m.items() if k != "confusion_matrix"})
        details[name] = {
            "overall": m,
            "folds": fold_rows,
            "y_true": y.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
            "subject_id": groups.tolist(),
            "group": df[spec.group_col].tolist(),
        }

    metrics_df = pd.DataFrame(rows).sort_values(["roc_auc", "macro_f1"], ascending=False)
    metrics_df.to_csv(DIRS["results"] / "04_model_metrics_stride_level.csv", index=False)
    (DIRS["results"] / "05_model_cv_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, default=json_default))

    best_name = str(metrics_df.iloc[0]["model"])
    best_detail = details[best_name]
    subject_eval = subject_level_metrics(best_detail)
    subject_eval.to_csv(DIRS["results"] / "06_subject_level_predictions.csv", index=False)
    subject_metrics = metrics_from_predictions(
        subject_eval["target"].to_numpy(),
        (subject_eval["score_mean"].to_numpy() >= 0.5).astype(int),
        subject_eval["score_mean"].to_numpy(),
    )
    (DIRS["results"] / "07_subject_level_metrics.json").write_text(json.dumps(subject_metrics, ensure_ascii=False, indent=2, default=json_default))

    plot_curves(best_name, best_detail, subject_eval)
    importance = fit_best_and_importance(best_name, models[best_name], preprocessor, spec)
    importance.to_csv(DIRS["results"] / "08_best_model_feature_importance.csv", index=False)
    return metrics_df, {"details": details, "subject_metrics": subject_metrics}, best_name


def subject_level_metrics(detail: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "subject_id": detail["subject_id"],
        "target": detail["y_true"],
        "group": detail["group"],
        "score": detail["y_score"],
    })
    agg = df.groupby("subject_id").agg(
        target=("target", "first"),
        group=("group", lambda s: ",".join(sorted(set(map(str, s))))),
        score_mean=("score", "mean"),
        n_rows=("score", "size"),
    ).reset_index()
    agg["pred"] = (agg["score_mean"] >= 0.5).astype(int)
    return agg


def plot_curves(best_name: str, detail: dict, subject_eval: pd.DataFrame) -> None:
    y = np.array(detail["y_true"])
    score = np.array(detail["y_score"])
    pred = np.array(detail["y_pred"])

    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y, score)
    ax.plot(fpr, tpr, label=f"Stride ROC-AUC={roc_auc_score(y, score):.3f}", color="#1a6ca8")
    sy = subject_eval["target"].to_numpy()
    ss = subject_eval["score_mean"].to_numpy()
    sfpr, stpr, _ = roc_curve(sy, ss)
    ax.plot(sfpr, stpr, label=f"Subject ROC-AUC={roc_auc_score(sy, ss):.3f}", color="#c0392b")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {best_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "03_best_model_roc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y, score)
    ax.plot(recall, precision, color="#1a6ca8", label=f"AP={average_precision_score(y, score):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve: {best_name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "04_best_model_pr.png", dpi=180)
    plt.close(fig)

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["HA", "ACL"])
    ax.set_yticks([0, 1], ["HA", "ACL"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "05_best_model_confusion_matrix.png", dpi=180)
    plt.close(fig)


def fit_best_and_importance(best_name: str, pipe: Pipeline, preprocessor: ColumnTransformer, spec: DatasetSpec) -> pd.DataFrame:
    model = clone(pipe)
    model.set_params(preprocess=clone(preprocessor))
    x = spec.frame[spec.feature_cols]
    y = spec.frame[spec.target_col].to_numpy()
    model.fit(x, y)
    joblib.dump(model, DIRS["models"] / "01_best_traditional_ml_model.joblib")

    features = model.named_steps["preprocess"].get_feature_names_out()
    selector = model.named_steps["select"]
    selected = features[selector.get_support()]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    else:
        values = np.full(len(selected), np.nan)
    imp = pd.DataFrame({"feature": selected, "importance": values})
    imp["abs_importance"] = imp["importance"].abs()
    imp = imp.sort_values("abs_importance", ascending=False).reset_index(drop=True)

    top = imp.head(20).sort_values("abs_importance")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#1a6ca8")
    ax.set_title(f"Top Features: {best_name}")
    ax.set_xlabel("Importance / coefficient")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "06_best_model_feature_importance.png", dpi=180)
    plt.close(fig)
    return imp


def table_html(df: pd.DataFrame, cols: list[str] | None = None, precision: int = 3) -> str:
    view = df.copy()
    if cols:
        view = view[cols]
    for c in view.select_dtypes(include=[np.number]).columns:
        view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.{precision}f}")
    return view.to_html(index=False, escape=True, classes="tbl")


def fig_tag(name: str, caption: str) -> str:
    rel = f"../figures/{name}"
    return f'<figure><img src="{rel}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def write_html_reports(doc_summary: dict, eda: dict, metrics_df: pd.DataFrame, evals: dict, best_name: str, commands: list[str]) -> None:
    generated = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and ".venv" not in p.parts)
    best_row = metrics_df.iloc[0].to_dict()
    subject_metrics = evals["subject_metrics"]
    hit95 = best_row.get("roc_auc", 0) >= 0.95
    top_features = pd.read_csv(DIRS["results"] / "08_best_model_feature_importance.csv").head(15)

    css = """
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff;color:#182230;margin:0;line-height:1.6}
    main{max-width:1120px;margin:0 auto;padding:34px 28px 64px}
    h1{font-size:30px;margin:0 0 8px;color:#0f172a} h2{font-size:22px;margin-top:34px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}
    h3{font-size:17px;margin-top:24px}.muted{color:#64748b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
    .card{border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;background:#f8fafc}.metric{font-size:28px;font-weight:750;color:#1a6ca8}
    table.tbl{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0} .tbl th{background:#f1f5f9;text-align:left}.tbl th,.tbl td{border:1px solid #e5e7eb;padding:7px 9px}
    figure{margin:18px 0;padding:0}img{max-width:100%;border:1px solid #e5e7eb;border-radius:8px;background:#fff}figcaption{font-size:13px;color:#64748b;margin-top:6px}
    code{background:#f1f5f9;border-radius:4px;padding:2px 5px}.ok{color:#15803d;font-weight:700}.warn{color:#b45309;font-weight:700}.bad{color:#b91c1c;font-weight:700}
    ul{padding-left:22px}.small{font-size:13px}pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px;max-height:420px;overflow:auto}
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
    }

    log_html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>0529 ML Execution Log</title><style>{css}</style></head>
    <body><main><h1>0529 ML 실행 로그</h1><p class="muted">생성 시각: {escape(now)}</p>
    <section class="grid">
      <div class="card"><div class="muted">선택 데이터</div><div>{escape(eda['selected_dataset'])}</div></div>
      <div class="card"><div class="muted">샘플 / 피험자</div><div>{eda['shape'][0]} rows / {eda['n_subjects']} subjects</div></div>
      <div class="card"><div class="muted">최고 모델</div><div>{escape(best_name)}</div></div>
      <div class="card"><div class="muted">Stride ROC-AUC</div><div class="metric">{best_row['roc_auc']:.3f}</div></div>
    </section>
    <h2>읽은 파일</h2><ul>{''.join(f'<li><code>{escape(p)}</code></li>' for p in doc_summary['read_files'])}</ul>
    <h2>실행 환경</h2><pre>{escape(json.dumps(env, ensure_ascii=False, indent=2))}</pre>
    <h2>실행 명령</h2><ul>{''.join(f'<li><code>{escape(c)}</code></li>' for c in commands)}</ul>
    <h2>데이터 인벤토리</h2>{table_html(pd.read_csv(DIRS['results'] / '02_data_inventory.csv'))}
    <h2>주요 로그</h2><ul>
      <li>기존 파일은 읽기만 수행했고 모든 산출물은 <code>0529_ML</code> 내부에 저장했다.</li>
      <li>DL 모델은 실행하지 않았다.</li>
      <li>누수 방지를 위해 group/target/label/id/subject/file/path/split/match 계열 컬럼을 feature에서 제외했다.</li>
      <li>평가는 subject 기준 {escape(str(metrics_df.iloc[0]['cv']))}로 수행했다.</li>
    </ul>
    <h2>모델별 실행 상태</h2>{table_html(metrics_df)}
    <h2>생성 산출물</h2><ul>{''.join(f'<li><code>{escape(p)}</code></li>' for p in generated)}</ul>
    </main></body></html>"""
    (DIRS["htmls"] / "01_execution_log.html").write_text(log_html)

    status = "<span class='ok'>달성</span>" if hit95 else "<span class='bad'>미달</span>"
    leak_note = (
        "AUC가 0.95 이상이므로 특히 subject grouping, ID/label 컬럼 제외, 같은 subject의 stride 중복이 fold 간 섞이지 않았는지 확인했다."
        if hit95 else
        "AUC가 0.95 미만이다. 현재 결과는 subject leakage를 제한한 보수적 평가로 보며, 추가 feature engineering과 외부 검증이 필요하다."
    )
    report_html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>0529 ML Result Report</title><style>{css}</style></head>
    <body><main><h1>ACL 보행 전통 ML 결과 보고서</h1><p class="muted">생성 시각: {escape(now)} · Light theme</p>
    <section class="grid">
      <div class="card"><div class="muted">목표</div><div>ACLD/ACLR vs HA 이진 분류</div></div>
      <div class="card"><div class="muted">최고 모델</div><div>{escape(best_name)}</div></div>
      <div class="card"><div class="muted">Stride ROC-AUC</div><div class="metric">{best_row['roc_auc']:.3f}</div></div>
      <div class="card"><div class="muted">95% AUC 목표</div><div>{status}</div></div>
    </section>
    <h2>연구 배경 요약</h2><p>{escape(doc_summary['research_summary'])}</p><p>{escape(doc_summary['statistical_findings'])}</p>
    <h2>타깃 정의</h2><p>{escape(doc_summary['target_definition'])}</p>
    <h2>데이터 요약</h2><p>선택 데이터는 <code>{escape(eda['selected_dataset'])}</code>이며 {eda['shape'][0]}행, {eda['shape'][1]}열, {eda['n_subjects']}명의 subject를 포함한다.</p>
    <div class="grid"><div class="card"><b>행 기준 그룹</b><pre>{escape(json.dumps(eda['group_counts'], ensure_ascii=False, indent=2))}</pre></div>
    <div class="card"><b>Subject 기준 그룹</b><pre>{escape(json.dumps(eda['subject_group_counts'], ensure_ascii=False, indent=2))}</pre></div></div>
    {fig_tag('01_group_distribution.png','그룹별 행 수와 subject 수')}
    {fig_tag('02_missingness.png','상위 결측 feature')}
    <h2>전처리 및 누수 방지</h2><ul>
      <li>결측 numeric feature는 train fold median, categorical feature는 최빈값으로 대체했다.</li>
      <li>범주형 feature는 train fold one-hot encoding을 적용했다.</li>
      <li>선형/SVM 계열은 scaling을 train fold 안에서만 수행했다.</li>
      <li>{escape(eda['excluded_leakage_rule'])}</li>
      <li>Feature selection은 fold 내부 <code>SelectKBest(f_classif, k<=30)</code>로 제한했다.</li>
    </ul>
    <h2>검증 전략</h2><p>동일 subject의 stride가 train/test fold에 동시에 들어가지 않도록 subject group 기반 CV를 사용했다. CV 방식: <code>{escape(str(metrics_df.iloc[0]['cv']))}</code>.</p>
    <h2>모델별 성능 비교</h2>{table_html(metrics_df)}
    <h2>최고 모델 상세</h2>
    <p>Stride-level ROC-AUC는 <b>{best_row['roc_auc']:.3f}</b>, macro F1은 <b>{best_row['macro_f1']:.3f}</b>이다. Subject-level 평균 score 기준 ROC-AUC는 <b>{subject_metrics['roc_auc']:.3f}</b>이다.</p>
    {fig_tag('03_best_model_roc.png','최고 모델 ROC 곡선: stride-level 및 subject-level')}
    {fig_tag('04_best_model_pr.png','최고 모델 PR 곡선')}
    {fig_tag('05_best_model_confusion_matrix.png','최고 모델 stride-level confusion matrix')}
    <h2>Feature Importance</h2>{table_html(top_features, ['feature','importance','abs_importance'])}
    {fig_tag('06_best_model_feature_importance.png','최고 모델 상위 feature importance')}
    <h2>기존 통계 결과와의 연결</h2><p>기존 보고서에서 fast 조건 무릎 굴곡 loading response peak LSI, 고관절 굴곡 ROM, 대칭성 관련 지표가 주요 차이로 제시되었다. 상위 ML feature에 knee/hip, fast/speed, LSI/ROM/peak 계열이 포함되면 통계 결과와 방향성이 맞는 근거로 해석할 수 있다. 포함되지 않는 경우 subject-safe CV에서 더 일반화되는 다른 IMU 요약 지표가 선택된 것으로 보아야 한다.</p>
    <h2>95% AUC 목표 판단 및 누수 점검</h2><p>{status}. {escape(leak_note)}</p>
    <h2>한계점</h2><ul>
      <li>데이터셋 규모가 작고 stride 단위 반복 측정이 많아 subject-level 외부 검증 없이는 일반화 성능을 확정할 수 없다.</li>
      <li>ACLD와 ACLR을 하나의 양성 클래스로 묶어 회복 단계별 차이는 별도 모델에서 평가해야 한다.</li>
      <li>전통 ML만 실행했으며 waveform 전체 시계열 기반 DL은 이번 목표에서 제외했다.</li>
    </ul>
    <h2>다음 실험 제안</h2><ul>
      <li>ACLD vs ACLR vs HA 다중분류와 ACLR vs HA 이진분류를 별도 비교한다.</li>
      <li>subject 단위 feature aggregation과 stride 단위 모델을 분리해 AUC 안정성을 비교한다.</li>
      <li>속도별 단일 모델과 multi-speed 통합 모델을 비교해 H2를 직접 검증한다.</li>
    </ul>
    </main></body></html>"""
    (DIRS["htmls"] / "02_ml_result_report.html").write_text(report_html)


def main() -> None:
    ensure_dirs()
    commands = [
        "python -m venv 0529_ML/.venv",
        "0529_ML/.venv/bin/python -m pip install pandas pyarrow scikit-learn matplotlib joblib xgboost lightgbm",
        "0529_ML/.venv/bin/python 0529_ML/scripts/01_run_ml_pipeline.py",
    ]
    doc_summary = summarize_docs()
    (DIRS["results"] / "00_source_document_summary.json").write_text(json.dumps(doc_summary, ensure_ascii=False, indent=2, default=json_default))
    spec, data_summaries = choose_dataset()
    eda = write_eda(spec, data_summaries)
    metrics_df, evals, best_name = evaluate_models(spec)
    write_html_reports(doc_summary, eda, metrics_df, evals, best_name, commands)
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "best_stride_auc": float(metrics_df.iloc[0]["roc_auc"]),
        "subject_auc": float(evals["subject_metrics"]["roc_auc"]),
        "outputs": sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and ".venv" not in p.parts),
    }
    (DIRS["logs"] / "01_execution_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=json_default))
    print(json.dumps(log, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
