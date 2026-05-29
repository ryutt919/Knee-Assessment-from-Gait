#!/usr/bin/env python3
"""Compare all usable candidate datasets with the same leakage rules."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "0529_ML"
MOD_PATH = OUT / "scripts" / "01_run_ml_pipeline.py"
spec = importlib.util.spec_from_file_location("mlpipe", MOD_PATH)
mlpipe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["mlpipe"] = mlpipe
spec.loader.exec_module(mlpipe)


def usable_specs() -> list:
    specs = []
    for path in mlpipe.DATA_CANDIDATES:
        if not path.exists():
            continue
        df = mlpipe.load_frame(path)
        group_col = mlpipe.first_col(list(df.columns), ["group", "Group", "condition", "diagnosis"])
        subject_col = mlpipe.first_col(list(df.columns), ["subject_id", "participant_id", "subject", "participant", "ID", "id"])
        if not group_col or not subject_col:
            continue
        work = df.copy()
        work["_group_norm"] = work[group_col].map(mlpipe.normalize_group)
        work = work[work["_group_norm"].isin(["ACLD", "ACLR", "HA"])].copy()
        if work["_group_norm"].nunique() < 2:
            continue
        work["_target_acl"] = work["_group_norm"].map(mlpipe.TARGET_MAP).astype(int)
        excluded = {group_col, subject_col, "_group_norm", "_target_acl"}
        feature_cols = [c for c in work.columns if c not in excluded and not mlpipe.leakage_like(c)]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
        categorical_cols = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique(dropna=True) <= 20
        ]
        if numeric_cols or categorical_cols:
            specs.append(mlpipe.DatasetSpec(
                path=path,
                frame=work,
                group_col="_group_norm",
                subject_col=subject_col,
                target_col="_target_acl",
                feature_cols=numeric_cols + categorical_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
            ))
    return specs


def evaluate_spec(ds) -> list[dict]:
    df = ds.frame.reset_index(drop=True)
    x = df[ds.feature_cols]
    y = df[ds.target_col].to_numpy()
    groups = df[ds.subject_col].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(x, y, groups))
        cv_name = "StratifiedGroupKFold"
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(x, y, groups))
        cv_name = "GroupKFold"

    preprocessor = mlpipe.build_preprocessor(ds)
    models = mlpipe.make_models(len(ds.feature_cols), len(df))
    rows = []
    for name, pipe in models.items():
        y_score = np.zeros(len(df), dtype=float)
        y_pred = np.zeros(len(df), dtype=int)
        for tr, te in splits:
            fold_model = clone(pipe)
            fold_model.set_params(preprocess=clone(preprocessor))
            fold_model.fit(x.iloc[tr], y[tr])
            score = mlpipe.estimator_scores(fold_model, x.iloc[te])
            y_score[te] = score
            y_pred[te] = (score >= 0.5).astype(int)
        m = mlpipe.metrics_from_predictions(y, y_pred, y_score)
        m.update({
            "dataset": str(ds.path.relative_to(ROOT)),
            "model": name,
            "rows": len(df),
            "subjects": int(df[ds.subject_col].nunique()),
            "features": len(ds.feature_cols),
            "numeric_features": len(ds.numeric_cols),
            "categorical_features": len(ds.categorical_cols),
            "cv": cv_name,
        })
        rows.append({k: v for k, v in m.items() if k != "confusion_matrix"})
    return rows


def main() -> None:
    rows = []
    for ds in usable_specs():
        rows.extend(evaluate_spec(ds))
    metrics = pd.DataFrame(rows).sort_values(["roc_auc", "macro_f1"], ascending=False)
    out_csv = OUT / "results" / "09_candidate_dataset_model_metrics.csv"
    metrics.to_csv(out_csv, index=False)
    best = metrics.iloc[0].to_dict()
    (OUT / "results" / "10_overall_best_candidate.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2, default=mlpipe.json_default)
    )

    report = OUT / "htmls" / "02_ml_result_report.html"
    html = report.read_text()
    section = f"""
    <h2>후보 데이터셋 추가 비교</h2>
    <p>동일한 누수 제거 규칙과 subject group CV로 모든 사용 가능 후보 테이블을 비교했다. 최고 후보는
    <code>{best['dataset']}</code> / <code>{best['model']}</code>이며 ROC-AUC는 <b>{best['roc_auc']:.3f}</b>이다.</p>
    {metrics.head(20).to_html(index=False, escape=True, classes='tbl')}
    """
    html = html.replace("</main></body></html>", section + "</main></body></html>")
    report.write_text(html)
    print(json.dumps({
        "best_dataset": best["dataset"],
        "best_model": best["model"],
        "best_auc": float(best["roc_auc"]),
        "metrics_csv": str(out_csv.relative_to(OUT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
